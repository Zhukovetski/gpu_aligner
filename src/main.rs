use bio::io::fasta;
use clap::Parser;
use flate2::read::MultiGzDecoder;
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::fs::File;
use std::io::BufRead;
use std::io::{BufReader, Read};
use std::sync::Arc;
use std::time::Duration;
use tokio::io::AsyncWriteExt;
use tokio::sync::mpsc;
use tokio::time::Instant;
use wgpu::util::DeviceExt;

const SEQ_LEN: usize = 100;
const PACKED_LEN: usize = 7;
const MATCH_SCORE: i32 = 3;

const READ_LEN: usize = 100;
const REF_LEN: usize = 128;
const READ_WORDS: usize = 7;
const REF_WORDS: usize = 8;

#[derive(Parser, Debug, Clone)]
#[command(author = "Твое Имя", version = "1.0", about, long_about = None)]
pub struct Args {
    #[arg(short = 'r', long)]
    pub reference: String,
    #[arg(short = 'i', long)]
    pub reads: String,
    #[arg(short = 'o', long, default_value = "alignment.sam")]
    pub output: String,
    #[arg(short = 'b', long, default_value_t = 100_000)]
    pub batch_size: usize,
    #[arg(short = 'l', long, default_value_t = 0)]
    pub limit: usize,
    #[arg(short = 'p', long)]
    pub progress: bool,
    #[arg(short = 'k', long, default_value_t = 15)]
    pub kmer_size: usize,
    #[arg(short = 'f', long, default_value_t = 50)]
    pub max_freq: u32,
}

struct GpuBatch {
    id: usize,
    packed_a: Vec<u32>,
    packed_b: Vec<u32>,
    valid_reads: usize,
}

struct ReadMeta {
    name: Vec<u8>,
    pos: u32,
    seq: Vec<u8>,
    qual: Vec<u8>,
    is_revcomp: bool,
}

struct MetaBatch {
    reads: Vec<ReadMeta>,
}

struct ScoreBatch {
    scores_and_coords: Vec<i32>,
    directions: Vec<u32>,
    gpu_meta: Vec<u32>,
}

struct RawRead {
    name: Vec<u8>,
    seq: Vec<u8>,
    qual: Vec<u8>,
}

struct ReadChunk {
    reads: Vec<RawRead>,
}

fn trim_newline(s: &[u8]) -> &[u8] {
    let mut len = s.len();
    while len > 0 && (s[len - 1] == b'\n' || s[len - 1] == b'\r') {
        len -= 1;
    }
    &s[..len]
}

fn trim_newline_inplace(s: &mut Vec<u8>) {
    let mut len = s.len();
    while len > 0 && (s[len - 1] == b'\n' || s[len - 1] == b'\r') {
        len -= 1;
    }
    s.truncate(len); // Меняет длину вектора без перевыделения памяти!
}
fn open_flexible_reader(path: &str) -> BufReader<Box<dyn Read + Send>> {
    let file = File::open(path).unwrap_or_else(|_| panic!("❌ Не удалось открыть файл: {}", path));
    if path.ends_with(".gz") {
        BufReader::new(Box::new(MultiGzDecoder::new(file)))
    } else {
        BufReader::new(Box::new(file))
    }
}
fn reverse_complement(seq: &[u8]) -> Vec<u8> {
    seq.iter()
        .rev()
        .map(|&b| match b.to_ascii_uppercase() {
            b'A' => b'T',
            b'C' => b'G',
            b'G' => b'C',
            b'T' => b'A',
            _ => b'N',
        })
        .collect()
}
fn reverse_complement_1(seq: &mut [u8]) {
    if seq.is_empty() {
        return;
    }
    let mut left = 0;
    let mut right = seq.len() - 1;

    while left <= right {
        // Достаем символы слева и справа, переводя в верхний регистр
        let a = seq[left].to_ascii_uppercase();
        let b = seq[right].to_ascii_uppercase();

        // Меняем их местами и одновременно трансформируем
        seq[left] = match b {
            b'A' => b'T',
            b'C' => b'G',
            b'G' => b'C',
            b'T' => b'A',
            _ => b'N',
        };

        seq[right] = match a {
            b'A' => b'T',
            b'C' => b'G',
            b'G' => b'C',
            b'T' => b'A',
            _ => b'N',
        };

        left += 1;
        right = right.saturating_sub(1)
    }
}

fn pack_dna(seq: &[u8]) -> Vec<u32> {
    let mut packed = Vec::with_capacity(PACKED_LEN);
    let mut current_u32 = 0u32;
    for (i, &b) in seq.iter().enumerate() {
        let val = match b.to_ascii_uppercase() {
            b'A' => 0u32,
            b'C' => 1u32,
            b'G' => 2u32,
            b'T' => 3u32,
            _ => 0u32,
        };
        current_u32 |= val << ((i % 16) * 2);
        if i % 16 == 15 || i == seq.len() - 1 {
            packed.push(current_u32);
            current_u32 = 0;
        }
    }
    packed
}

fn pack_kmer(seq: &[u8], k: usize) -> Option<u64> {
    if seq.len() < k {
        return None;
    }
    let mut kmer = 0u64;
    for &b in &seq[0..k] {
        // Soft-masking: если буква строчная (повтор) или 'N', мы возвращаем None!
        let val = match b {
            b'A' => 0,
            b'C' => 1,
            b'G' => 2,
            b'T' => 3,
            _ => return None, // Строчные a,c,g,t тоже попадут сюда и отфильтруются!
        };
        kmer = (kmer << 2) | val;
    }
    Some(kmer)
}

//Статическая таблица подстановки для ASCII-символов ДНК.
//Индексы соответствуют кодам ASCII. Поддерживает как UPPER, так и lower case.
const DNA_MAP: [u32; 256] = {
    let mut map = [0u32; 256];
    map[b'A' as usize] = 0;
    map[b'a' as usize] = 0;
    map[b'C' as usize] = 1;
    map[b'c' as usize] = 1;
    map[b'G' as usize] = 2;
    map[b'g' as usize] = 2;
    map[b'T' as usize] = 3;
    map[b't' as usize] = 3;
    map
};

// Таблица для проверки валидности символов в pack_15mer (чтобы отсекать символы вроде 'N' или мусор)
const IS_VALID_DNA: [bool; 256] = {
    let mut map = [false; 256];
    map[b'A' as usize] = true;
    map[b'a' as usize] = true;
    map[b'C' as usize] = true;
    map[b'c' as usize] = true;
    map[b'G' as usize] = true;
    map[b'g' as usize] = true;
    map[b'T' as usize] = true;
    map[b't' as usize] = true;
    map
};

// Оптимизированная упаковка ДНК напрямую в целевой буфер батча (БЕЗ аллокаций кучи!)
fn pack_dna_inplace(seq: &[u8], out_buf: &mut [u32]) {
    let mut current_u32 = 0u32;
    let mut out_idx = 0;

    for (i, &b) in seq.iter().enumerate() {
        // Мгновенное получение битового значения из таблицы без match и без to_ascii_uppercase
        let val = unsafe { *DNA_MAP.get_unchecked(b as usize) };

        current_u32 |= val << ((i % 16) * 2);

        if i % 16 == 15 || i == seq.len() - 1 {
            if out_idx < out_buf.len() {
                out_buf[out_idx] = current_u32;
                out_idx += 1;
            }
            current_u32 = 0;
        }
    }
}

// 1. Убираем async, чтобы функция мгновенно возвращала управление рантайму
fn reader_task(
    reads_path: String,
    batch_size: usize,
    chunk_tx: mpsc::Sender<ReadChunk>,
) -> tokio::task::JoinHandle<()> {
    // spawn_blocking запускается в фоне, мы НЕ пишем .await в конце функции
    tokio::task::spawn_blocking(move || {
        println!("📦 [Reader] Начало стриминга чтений из {}...", reads_path);

        // Открываем файл. Обратите внимание на тип Box<dyn BufRead + Send>
        let file =
            File::open(&reads_path).unwrap_or_else(|_| panic!("❌ Файл не найден: {}", reads_path));

        let mut fastq_reader: Box<dyn BufRead + Send> = if reads_path.ends_with(".gz") {
            // Оборачиваем MultiGzDecoder в BufReader, чтобы получить типаж BufRead
            Box::new(BufReader::new(MultiGzDecoder::new(file)))
        } else {
            Box::new(BufReader::new(file))
        };

        let mut chunk = Vec::with_capacity(batch_size);

        // Повторно используемые буферы для строк
        let mut line_name = Vec::with_capacity(128);
        let mut line_seq = Vec::with_capacity(512);
        let mut line_plus = Vec::with_capacity(32);
        let mut line_qual = Vec::with_capacity(512);

        loop {
            line_name.clear();
            line_seq.clear();
            line_plus.clear();
            line_qual.clear();

            if fastq_reader.read_until(b'\n', &mut line_name).unwrap() == 0 {
                break; // EOF
            }
            fastq_reader.read_until(b'\n', &mut line_seq).unwrap();
            fastq_reader.read_until(b'\n', &mut line_plus).unwrap();
            fastq_reader.read_until(b'\n', &mut line_qual).unwrap();

            // Обрезаем \n и \r прямо внутри существующих векторов (без аллокаций!)
            trim_newline_inplace(&mut line_name);
            trim_newline_inplace(&mut line_seq);
            trim_newline_inplace(&mut line_qual);

            // Обработка заголовка FASTQ без копирования памяти
            let name = if line_name.starts_with(b"@") {
                line_name[1..].to_vec()
            } else {
                line_name.clone()
            };

            // Копируем очищенные байты
            chunk.push(RawRead {
                name,
                seq: line_seq.clone(),
                qual: line_qual.clone(),
            });

            if chunk.len() == batch_size {
                // ВАЖНО: Заменяем заполненный chunk новым вектором с выделенной памятью.
                // Это предотвращает сброс capacity до нуля, который делал std::mem::take.
                let full_chunk = std::mem::replace(&mut chunk, Vec::with_capacity(batch_size));

                if chunk_tx
                    .blocking_send(ReadChunk { reads: full_chunk })
                    .is_err()
                {
                    break; // Канал закрыт, воркеры вышли
                }
            }
        }

        if !chunk.is_empty() {
            let _ = chunk_tx.blocking_send(ReadChunk { reads: chunk });
        }
        println!("📦 [Reader] Чтение файла завершено.");
    })
}

async fn producer_task(
    args: Args,
    mut chunk_rx: mpsc::Receiver<ReadChunk>,
    gpu_tx: mpsc::Sender<GpuBatch>,
    meta_tx: mpsc::Sender<MetaBatch>,
) {
    tokio::task::spawn_blocking(move || {
        let mut batch_id = 0;

        while let Some(chunk) = chunk_rx.blocking_recv() {
            let mut temp_packed_reads = Vec::with_capacity(chunk.reads.len());
            let mut temp_meta = Vec::with_capacity(chunk.reads.len());
            let mut count = 0;

            for raw_read in chunk.reads {
                let seq = &raw_read.seq;
                let qual = &raw_read.qual;
                let name = &raw_read.name;

                if seq.len() == SEQ_LEN && !seq.contains(&b'N') {
                    // ПРОСТО ПАКУЕМ И ОТПРАВЛЯЕМ! Никаких поисков!
                    temp_packed_reads.push(pack_dna(seq));

                    temp_meta.push(ReadMeta {
                        name: name.clone(),
                        pos: 0, // Позицию теперь найдет GPU
                        seq: seq.clone(),
                        qual: qual.clone(),
                        is_revcomp: false, // GPU сама скажет нам, если перевернет
                    });
                    count += 1;
                }
            }

            if count > 0 {
                let mut soa_packed_reads = vec![0u32; count * PACKED_LEN];
                for (i, read_packed) in temp_packed_reads.iter().enumerate() {
                    for w in 0..PACKED_LEN {
                        soa_packed_reads[w * count + i] = read_packed[w];
                    }
                }

                // Обрати внимание: мы больше не шлем packed_a (куски генома), шлем только риды!
                if gpu_tx
                    .blocking_send(GpuBatch {
                        id: batch_id,
                        packed_a: vec![],
                        packed_b: soa_packed_reads,
                        valid_reads: count,
                    })
                    .is_err()
                {
                    break;
                }
                if meta_tx
                    .blocking_send(MetaBatch { reads: temp_meta })
                    .is_err()
                {
                    break;
                }
                batch_id += 1;

                if args.limit > 0 && batch_id >= args.limit {
                    return;
                }
            }
        }
    })
    .await
    .unwrap();
}

async fn gpu_actor(
    args: Args,
    index_keys: Arc<Vec<u32>>,
    index_values: Arc<Vec<u32>>,
    packed_genome: Arc<Vec<u32>>,
    mut rx: mpsc::Receiver<GpuBatch>,
    score_tx: mpsc::Sender<ScoreBatch>,
) {
    let instance = wgpu::Instance::default();

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance, // Требуем NVIDIA дискретку
            force_fallback_adapter: false,
            compatible_surface: None,
            apply_limit_buckets: false, // <-- ДОБАВЛЯЕМ ЭТО ПОЛЕ (выключаем ограничение лимитов)
        })
        .await
        .unwrap_or_else(|_| {
            panic!("❌ Не удалось найти дискретную видеокарту!");
        });

    // Логируем, какую карту реально выбрал рантайм
    let info = adapter.get_info();
    println!("🚀 Используемый GPU: [{:?}] {}", info.backend, info.name);
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            required_limits: wgpu::Limits {
                max_storage_buffer_binding_size: 512 * 1024 * 1024,
                max_buffer_size: 512 * 1024 * 1024,
                ..adapter.limits()
            },
            ..Default::default()
        })
        .await
        .unwrap();

    let shader = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));
    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    // 1. СТАТИЧНЫЕ БУФЕРЫ (Индекс и Геном - заливаем один раз!)
    let buffer_keys = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&index_keys),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let buffer_values = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&index_values),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let buffer_genome = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&packed_genome),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let buffer_reads_size = (args.batch_size * PACKED_LEN * 4) as wgpu::BufferAddress;
    let scores_size = (args.batch_size * 3 * 4) as wgpu::BufferAddress;
    let dirs_size = (args.batch_size * 100 * PACKED_LEN * 4) as wgpu::BufferAddress;
    let meta_size = (args.batch_size * 2 * 4) as wgpu::BufferAddress; // [pos, is_revcomp]

    // Обрати внимание: мы убрали buffer_a (он больше не нужен, геном теперь в buffer_genome)
    let buffer_reads = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: buffer_reads_size,
        mapped_at_creation: false,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let buffer_scores = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: scores_size,
        mapped_at_creation: false,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let buffer_dirs = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: dirs_size,
        mapped_at_creation: false,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let buffer_meta = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: meta_size,
        mapped_at_creation: false,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });

    let staging_scores = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: scores_size,
        mapped_at_creation: false,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
    });
    let staging_dirs = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: dirs_size,
        mapped_at_creation: false,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
    });
    let staging_meta = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: meta_size,
        mapped_at_creation: false,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
    });

    let bind_group_layout = compute_pipeline.get_bind_group_layout(0);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer_keys.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buffer_values.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: buffer_genome.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: buffer_reads.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: buffer_scores.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: buffer_dirs.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: buffer_meta.as_entire_binding(),
            },
        ],
    });

    while let Some(batch) = rx.recv().await {
        queue.write_buffer(&buffer_reads, 0, bytemuck::cast_slice(&batch.packed_b));

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cpass.set_pipeline(&compute_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (batch.valid_reads as f32 / 64.0).ceil() as u32;
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }

        let copy_scores_size = (batch.valid_reads * 3 * 4) as wgpu::BufferAddress;
        let copy_dirs_size = (batch.valid_reads * 100 * PACKED_LEN * 4) as wgpu::BufferAddress;
        let copy_meta_size = (batch.valid_reads * 2 * 4) as wgpu::BufferAddress;

        encoder.copy_buffer_to_buffer(&buffer_scores, 0, &staging_scores, 0, copy_scores_size);
        encoder.copy_buffer_to_buffer(&buffer_dirs, 0, &staging_dirs, 0, copy_dirs_size);
        encoder.copy_buffer_to_buffer(&buffer_meta, 0, &staging_meta, 0, copy_meta_size);
        queue.submit(Some(encoder.finish()));

        let slice_scores = staging_scores.slice(..copy_scores_size);
        let slice_dirs = staging_dirs.slice(..copy_dirs_size);
        let slice_meta = staging_meta.slice(..copy_meta_size);

        let (tx1, rx1) = tokio::sync::oneshot::channel();
        let (tx2, rx2) = tokio::sync::oneshot::channel();
        let (tx3, rx3) = tokio::sync::oneshot::channel();

        slice_scores.map_async(wgpu::MapMode::Read, move |v| {
            tx1.send(v).unwrap();
        });
        slice_dirs.map_async(wgpu::MapMode::Read, move |v| {
            tx2.send(v).unwrap();
        });
        slice_meta.map_async(wgpu::MapMode::Read, move |v| {
            tx3.send(v).unwrap();
        });

        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();

        if rx1.await.is_ok() && rx2.await.is_ok() && rx3.await.is_ok() {
            let data_scores = slice_scores.get_mapped_range().unwrap();
            let data_dirs = slice_dirs.get_mapped_range().unwrap();
            let data_meta = slice_meta.get_mapped_range().unwrap();
            if score_tx
                .send(ScoreBatch {
                    scores_and_coords: bytemuck::cast_slice(&data_scores).to_vec(),
                    directions: bytemuck::cast_slice(&data_dirs).to_vec(),
                    gpu_meta: bytemuck::cast_slice(&data_meta).to_vec(), // Добавь это поле в ScoreBatch!
                })
                .await
                .is_err()
            {
                break;
            }
        }
        staging_scores.unmap();
        staging_dirs.unmap();
        staging_meta.unmap();
    }
}

async fn writer_task(
    output_path: String,
    genome: Arc<Vec<u8>>,
    mut meta_rx: mpsc::Receiver<MetaBatch>,
    mut score_rx: mpsc::Receiver<ScoreBatch>,
    pb: ProgressBar,
) {
    let file = tokio::fs::File::create(&output_path).await.unwrap();
    let mut writer = tokio::io::BufWriter::new(file);

    writer
        .write_all(b"@HD\tVN:1.6\tSO:unsorted\n")
        .await
        .unwrap();
    writer
        .write_all(format!("@SQ\tSN:reference\tLN:{}\n", genome.len()).as_bytes())
        .await
        .unwrap();
    let mut total_written = 0;
    while let (Some(meta_batch), Some(score_batch)) = (meta_rx.recv().await, score_rx.recv().await)
    {
        let batch_size = meta_batch.reads.len();

        let sam_lines: Vec<String> = tokio::task::spawn_blocking(move || {
            meta_batch
                .reads
                .into_par_iter()
                .enumerate()
                .map(|(read_idx, meta)| {
                    let max_score = score_batch.scores_and_coords[read_idx * 3];
                    println!("{:?}", max_score);

                    // ЧИТАЕМ ОТВЕТ ОТ GPU!
                    let gpu_pos = score_batch.gpu_meta[read_idx * 2];
                    let gpu_revcomp = score_batch.gpu_meta[read_idx * 2 + 1] == 1;

                    // Переворачиваем строку и качество, если GPU сделала Reverse Complement
                    let mut final_seq = meta.seq.clone();
                    let mut final_qual = meta.qual.clone();
                    if gpu_revcomp {
                        final_seq = reverse_complement(&final_seq);
                        final_qual.reverse();
                    }

                    let meta_name = unsafe { std::str::from_utf8_unchecked(&meta.name) };
                    let meta_seq = unsafe { std::str::from_utf8_unchecked(&final_seq) };
                    let meta_qual = unsafe { std::str::from_utf8_unchecked(&final_qual) };
                    let flag = if gpu_revcomp { 16 } else { 0 };

                    if gpu_pos == 0xFFFFFFFF || max_score < 200 {
                        return format!(
                            "{}\t4\t*\t0\t0\t*\t*\t0\t0\t{}\t{}\n",
                            meta_name, meta_seq, meta_qual
                        );
                    }

                    let max_i = score_batch.scores_and_coords[read_idx * 3 + 1] as usize;
                    let max_j = score_batch.scores_and_coords[read_idx * 3 + 2] as usize;

                    // O(N) Быстрый Traceback по матрице GPU!
                    let mut i = max_i;
                    let mut j = max_j;
                    let mut align_start_i = i;
                    let mut clip_start = 0;
                    let mut cigar_ops = Vec::new();

                    loop {
                        let w = j / 16;
                        let bit_offset = (j % 16) * 2;
                        let dir_idx = (i * PACKED_LEN + w) * batch_size + read_idx;
                        let dir = (score_batch.directions[dir_idx] >> bit_offset) & 3;

                        if dir == 0 {
                            clip_start = j + 1;
                            break;
                        }
                        align_start_i = i;

                        if dir == 1 {
                            cigar_ops.push('M');
                            if i == 0 || j == 0 {
                                clip_start = j;
                                break;
                            }
                            i -= 1;
                            j -= 1;
                        } else if dir == 2 {
                            cigar_ops.push('D');
                            if i == 0 {
                                clip_start = j + 1;
                                break;
                            }
                            i -= 1;
                        } else if dir == 3 {
                            cigar_ops.push('I');
                            if j == 0 {
                                clip_start = 0;
                                break;
                            }
                            j -= 1;
                        }
                    }

                    cigar_ops.reverse();
                    let mut final_cigar = String::new();
                    if clip_start > 0 {
                        final_cigar.push_str(&format!("{}S", clip_start));
                    }

                    let mut count = 0;
                    let mut last_op = None;
                    for &op in &cigar_ops {
                        match last_op {
                            Some(c) if c == op => count += 1,
                            Some(c) => {
                                final_cigar.push_str(&format!("{}{}", count, c));
                                last_op = Some(op);
                                count = 1;
                            }
                            None => {
                                last_op = Some(op);
                                count = 1;
                            }
                        }
                    }
                    if let Some(c) = last_op {
                        final_cigar.push_str(&format!("{}{}", count, c));
                    }

                    let clip_end = SEQ_LEN - 1 - max_j;
                    if clip_end > 0 {
                        final_cigar.push_str(&format!("{}S", clip_end));
                    }

                    let exact_pos = gpu_pos + align_start_i as u32 + 1;
                    format!(
                        "{}\t{}\treference\t{}\t60\t{}\t*\t0\t0\t{}\t{}\tAS:i:{}\n",
                        meta_name, flag, exact_pos, final_cigar, meta_seq, meta_qual, max_score
                    )
                })
                .collect()
        })
        .await
        .unwrap();

        for line in sam_lines {
            writer.write_all(line.as_bytes()).await.unwrap();
            total_written += 1;
        }

        pb.inc(batch_size as u64);
    }
    writer.flush().await.unwrap();

    if pb.is_hidden() {
        println!(
            "📝 [Writer] Файл записан. Выровнено ридов: {}",
            total_written
        );
    }
}

#[tokio::main]
async fn main() {
    let index_start = Instant::now();
    let args = Args::parse();

    println!("📦 Загрузка генома из {}...", args.reference);
    let fasta_reader = fasta::Reader::new(open_flexible_reader(&args.reference));
    let mut genome = Vec::new();
    for record in fasta_reader.records() {
        genome.extend_from_slice(record.unwrap().seq());
    }

    let index_start = Instant::now();

    println!("🧠 Строим индекс (k={})...", args.kmer_size);
    // Храним позицию, либо маркер "Слишком часто" (например, u32::MAX)
    let mut index: FxHashMap<u64, u32> = FxHashMap::default();

    for i in 0..=(genome.len() - args.kmer_size) {
        if let Some(kmer) = pack_kmer(&genome[i..i + args.kmer_size], args.kmer_size) {
            index
                .entry(kmer)
                .and_modify(|pos| *pos = u32::MAX) // Если встретили 2+ раз, помечаем как "мусорный повтор"
                .or_insert(i as u32);
        }
    }

    // Удаляем все ключи, которые мы пометили как мусор (u32::MAX)
    index.retain(|_, &mut pos| pos != u32::MAX);
    println!("✅ Уникальных надежных якорей: {}", index.len());

    let mut kmers_with_pos: Vec<(u32, u32)> = Vec::with_capacity(genome.len());

    for i in 0..=(genome.len() - SEQ_LEN) {
        if let Some(kmer) = pack_kmer(&genome[i..i + 15]) {
            kmers_with_pos.push((kmer, i as u32));
        }
    }

    // 2. Сортируем массив по ключу (kmer) - это необходимо для бинарного поиска!
    // sort_unstable работает в разы быстрее обычного sort
    kmers_with_pos.sort_unstable_by_key(|&(k, _)| k);

    // 3. Убираем дубликаты! Если 15-мер встречается в геноме 10 раз,
    // для MVP мы берем только первое вхождение (чтобы бинарный поиск не сошел с ума)
    kmers_with_pos.dedup_by_key(|&mut (k, _)| k);

    // 4. Разделяем на два параллельных массива (SoA - Structure of Arrays)
    let mut index_keys = Vec::with_capacity(kmers_with_pos.len());
    let mut index_values = Vec::with_capacity(kmers_with_pos.len());
    for (k, v) in kmers_with_pos {
        index_keys.push(k);
        index_values.push(v);
    }

    // 5. Упаковываем ВЕСЬ геном в u32 (чтобы GPU сама брала нужные куски)
    let packed_genome = pack_dna(&genome);

    println!(
        "✅ Индекс построен за {:#?}. Уникальных якорей: {}",
        index_start.elapsed(),
        index_keys.len()
    );
    println!(
        "📊 Размер генома в памяти GPU: {:.2} MB",
        (packed_genome.len() * 4) as f64 / 1_048_576.0
    );

    // Оборачиваем их в Arc для передачи Акторам
    let index_keys_arc = Arc::new(index_keys);
    let index_values_arc = Arc::new(index_values);
    let packed_genome_arc = Arc::new(packed_genome);
    let genome_arc = Arc::new(genome);

    // НАШИ КАНАЛЫ
    let (chunk_tx, chunk_rx) = mpsc::channel(3); // От Reader к Producer
    let (gpu_tx, gpu_rx) = mpsc::channel(3); // От Producer к GPU
    let (meta_tx, meta_rx) = mpsc::channel(3); // От Producer к Writer
    let (score_tx, score_rx) = mpsc::channel(3); // От GPU к Writer

    let pb = if args.progress {
        let pb = ProgressBar::new_spinner();
        pb.set_style(
            ProgressStyle::default_spinner()
                // Шаблон: Спиннер | Время | Сообщение | Количество ридов | Скорость
                .template("{spinner:.green} [{elapsed_precise}] {msg} {pos} ридов ({per_sec})")
                .unwrap(),
        );
        pb.enable_steady_tick(Duration::from_millis(100)); // Обновлять каждые 100 мс
        pb.set_message("Выравнивание...");
        pb
    } else {
        ProgressBar::hidden() // Если флаг не передан, бар будет невидимым (без оверхеда)
    };

    // СПАВНИМ 4 АКТОРОВ!
    let reader_handle = reader_task(args.reads.clone(), args.batch_size, chunk_tx);
    let producer_handle = tokio::spawn(producer_task(args.clone(), chunk_rx, gpu_tx, meta_tx));
    let gpu_handle = tokio::spawn(gpu_actor(
        args.clone(),
        index_keys_arc,
        index_values_arc,
        packed_genome_arc,
        gpu_rx,
        score_tx,
    ));
    let writer_handle = tokio::spawn(writer_task(
        args.output.clone(),
        genome_arc.clone(),
        meta_rx,
        score_rx,
        pb.clone(),
    ));

    //Ждем, пока все 4 завершат работу
    let _ = tokio::join!(producer_handle, gpu_handle, writer_handle);

    if args.progress {
        pb.finish_with_message("✅ Готово!");
    }
    println!("Общее время работы: {:?}", index_start.elapsed());
}
