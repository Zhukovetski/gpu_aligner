use bio::io::fasta;
use clap::Parser;
use flate2::read::MultiGzDecoder;
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::fmt::Write;
use std::fs::File;
use std::io::BufRead;
use std::io::{BufReader, Read};
use std::mem::take;
use std::sync::Arc;
use std::time::Duration;
use tokio::io::AsyncWriteExt;
use tokio::sync::mpsc;
use tokio::time::Instant;
use wgpu::PipelineCompilationOptions;

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
    packed_a: Vec<u32>, // Референсы (кусочки генома по 128 букв)
    packed_b: Vec<u32>, // Риды (по 100 букв)
    valid_reads: usize,
}

struct ReadMeta {
    name: Vec<u8>,
    pos: u32, // Истинная позиция в геноме
    seq: Vec<u8>,
    qual: Vec<u8>,
    is_revcomp: bool,
}

struct MetaBatch {
    reads: Vec<ReadMeta>,
}

struct RawRead {
    name: Vec<u8>,
    seq: Vec<u8>,
    qual: Vec<u8>,
}

struct ReadChunk {
    reads: Vec<RawRead>,
}

struct ScoreBatch {
    scores_and_coords: Vec<i32>, // Только 2 числа на рид: [score, align_start_j]
    cigar_bin: Vec<u32>,         // 16 бинарных чисел CIGAR
}

fn trim_newline_inplace(s: &mut Vec<u8>) {
    let mut len = s.len();
    while len > 0 && (s[len - 1] == b'\n' || s[len - 1] == b'\r') {
        len -= 1;
    }
    s.truncate(len); // Меняет длину вектора без перевыделения памяти!
}
fn open_flexible_reader(path: &str) -> BufReader<Box<dyn Read + Send>> {
    let file = File::open(path).unwrap_or_else(|_| panic!("❌ Не удалось открыть файл: {path}"));

    if std::path::Path::new(path)
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("gz"))
    {
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

fn pack_dna(seq: &[u8], expected_words: usize) -> Vec<u32> {
    let mut packed = vec![0u32; expected_words]; // Сразу выделяем точный размер!
    for (i, &b) in seq.iter().enumerate() {
        let val = match b.to_ascii_uppercase() {
            b'A' => 0,
            b'C' => 1,
            b'G' => 2,
            b'T' => 3,
            _ => 0,
        };
        packed[i / 16] |= val << ((i % 16) * 2);
    }
    packed
}

fn pack_kmer(seq: &[u8], k: usize) -> Option<u32> {
    if seq.len() < k {
        return None;
    }
    let mut kmer = 0u32;
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

// Вспомогательная структура для передачи готовой пары
struct MappedPair {
    meta: ReadMeta,
    packed_ref: Vec<u32>,
    packed_read: Vec<u32>,
}

// Эта функция берет ОДИН рид и пытается найти ему место в геноме
fn try_map_read(
    seq: &[u8],
    qual: &[u8],
    name: &[u8],
    is_revcomp: bool,
    genome: &[u8],
    index: &FxHashMap<u32, u32>,
    kmer_size: usize,
) -> Option<MappedPair> {
    // Ищем в начале, середине и конце
    let seed_offsets = [0, 42, READ_LEN.saturating_sub(kmer_size)];

    for &offset in &seed_offsets {
        // 1. Пакуем 15-мер
        let kmer = pack_kmer(&seq[offset..offset + kmer_size], kmer_size)?;

        // 2. Ищем в индексе
        let hit_pos = *index.get(&kmer)?;

        // 3. Вычисляем начало референса с запасом 14 баз влево
        let ref_start = hit_pos.saturating_sub(offset as u32).saturating_sub(14) as usize;

        // 4. Проверяем, что не вылезли за пределы генома
        if ref_start + REF_LEN <= genome.len() {
            let ref_snippet = &genome[ref_start..ref_start + REF_LEN];

            return Some(MappedPair {
                meta: ReadMeta {
                    name: name.to_vec(),
                    pos: ref_start as u32,
                    seq: seq.to_vec(),
                    qual: qual.to_vec(),
                    is_revcomp,
                },
                packed_ref: pack_dna(ref_snippet, REF_WORDS),
                packed_read: pack_dna(seq, READ_WORDS),
            });
        }
    }
    None // Ни один сид не подошел
}

// 1. Убираем async, чтобы функция мгновенно возвращала управление рантайму
fn reader_task(
    reads_path: String,
    batch_size: usize,
    chunk_tx: mpsc::Sender<ReadChunk>,
) -> tokio::task::JoinHandle<()> {
    // spawn_blocking запускается в фоне, мы НЕ пишем .await в конце функции
    tokio::task::spawn_blocking(move || {
        println!("📦 [Reader] Начало стриминга чтений из {reads_path}...");

        // Открываем файл. Обратите внимание на тип Box<dyn BufRead + Send>
        let file =
            File::open(&reads_path).unwrap_or_else(|_| panic!("❌ Файл не найден: {reads_path}"));

        let mut fastq_reader: Box<dyn BufRead + Send> = if std::path::Path::new(&reads_path)
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("gz"))
        {
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
                // Читает Строку 1 (@Name)
                break; // EOF
            }
            fastq_reader.read_until(b'\n', &mut line_seq).unwrap(); // Читает Строку 2 (DNA)
            fastq_reader.read_until(b'\n', &mut line_plus).unwrap(); // Читает Строку 3 (+)
            fastq_reader.read_until(b'\n', &mut line_qual).unwrap(); // Читает Строку 4 (Qual)

            // Обрезаем \n и \r прямо внутри существующих векторов (без аллокаций!)
            trim_newline_inplace(&mut line_name);
            trim_newline_inplace(&mut line_seq);
            trim_newline_inplace(&mut line_qual);

            // Обработка заголовка FASTQ без копирования памяти
            let mut name = take(&mut line_name);
            if name.starts_with(b"@") {
                // drain(0..1) удаляет первый байт '@', сдвигая остальные влево.
                // Это происходит прямо внутри вектора, без новой аллокации кучи!
                name.drain(0..1);
            }

            // Копируем очищенные байты
            chunk.push(RawRead {
                name,
                seq: take(&mut line_seq),
                qual: take(&mut line_qual),
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

fn producer_task(
    args: Args,
    genome: Arc<Vec<u8>>,
    mut chunk_rx: mpsc::Receiver<ReadChunk>,
    gpu_tx: mpsc::Sender<GpuBatch>,
    meta_tx: mpsc::Sender<MetaBatch>,
) -> tokio::task::JoinHandle<()> {
    tokio::task::spawn_blocking(move || {
        println!(
            "🧠 [Producer] Строим k-mer индекс (k={})...",
            args.kmer_size
        );
        let mut index: FxHashMap<u32, u32> = FxHashMap::default();

        for i in 0..=(genome.len() - args.kmer_size) {
            if let Some(kmer) = pack_kmer(&genome[i..i + args.kmer_size], args.kmer_size) {
                // index
                //     .entry(kmer)
                //     .or_insert_with(|| u32::try_from(i).unwrap());
                index
                    .entry(kmer)
                    .and_modify(|pos| *pos = u32::MAX) // Помечаем дубликаты
                    .or_insert_with(|| u32::try_from(i).unwrap());
            }
        }

        // ВАЖНО ДЛЯ ТЕСТА: Если мы тестируем на 128 буквах, закомментируй следующую строку!
        // Иначе она удалит тестовые риды из 'A' как дубликаты.
        index.retain(|_, &mut pos| pos != u32::MAX);

        println!("✅ [Producer] Индекс построен. Якорей: {}", index.len());

        let mut batch_id = 0;

        while let Some(chunk) = chunk_rx.blocking_recv() {
            let mut temp_packed_a = Vec::with_capacity(chunk.reads.len());
            let mut temp_packed_b = Vec::with_capacity(chunk.reads.len());
            let mut temp_meta = Vec::with_capacity(chunk.reads.len());
            let mut count = 0;

            for raw_read in chunk.reads {
                let seq = &raw_read.seq;
                let qual = &raw_read.qual;
                let name = &raw_read.name;

                if seq.len() == READ_LEN && !seq.contains(&b'N') {
                    // 1. Пробуем прямую цепь
                    let mapped =
                        try_map_read(seq, qual, name, false, &genome, &index, args.kmer_size);

                    // 2. Если не вышло, пробуем обратную цепь
                    let mapped = mapped.or_else(|| {
                        let rev_seq = reverse_complement(seq);
                        let mut rev_qual = qual.clone();
                        rev_qual.reverse();
                        try_map_read(
                            &rev_seq,
                            &rev_qual,
                            name,
                            true,
                            &genome,
                            &index,
                            args.kmer_size,
                        )
                    });

                    // 3. Если всё успешно, складываем в батч
                    if let Some(m) = mapped {
                        temp_packed_a.push(m.packed_ref);
                        temp_packed_b.push(m.packed_read);
                        temp_meta.push(m.meta);
                        count += 1;
                    }
                }
            }

            // Отправляем батч
            if count > 0 {
                let mut soa_packed_a = vec![0u32; args.batch_size * REF_WORDS];
                let mut soa_packed_b = vec![0u32; args.batch_size * READ_WORDS];
                for (i, (read_a, read_b)) in
                    temp_packed_a.iter().zip(temp_packed_b.iter()).enumerate()
                {
                    for w in 0..REF_WORDS {
                        soa_packed_a[w * args.batch_size + i] = read_a[w];
                    }
                    for w in 0..READ_WORDS {
                        soa_packed_b[w * args.batch_size + i] = read_b[w];
                    }
                }

                if gpu_tx
                    .blocking_send(GpuBatch {
                        id: batch_id,
                        packed_a: soa_packed_a,
                        packed_b: soa_packed_b,
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
}

async fn gpu_actor(
    args: Args,
    mut rx: mpsc::Receiver<GpuBatch>,
    score_tx: mpsc::Sender<ScoreBatch>,
) {
    let instance = wgpu::Instance::default();

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
            apply_limit_buckets: false,
        })
        .await
        .unwrap_or_else(|_| panic!("❌ Не удалось найти дискретную видеокарту!"));

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
        compilation_options: PipelineCompilationOptions::default(),
        cache: None,
    });

    let buffer_a_size = (args.batch_size * REF_WORDS * 4) as wgpu::BufferAddress;
    let buffer_b_size = (args.batch_size * READ_WORDS * 4) as wgpu::BufferAddress;
    let scores_size = (args.batch_size * 2 * 4) as wgpu::BufferAddress;
    let dirs_size = (args.batch_size * 100 * REF_WORDS * 4) as wgpu::BufferAddress;
    let cigar_size = (args.batch_size * 16 * 4) as wgpu::BufferAddress;

    let buffer_a = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: buffer_a_size,
        mapped_at_creation: false,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    let buffer_b = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: buffer_b_size,
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
        usage: wgpu::BufferUsages::STORAGE,
    });

    let buffer_cigar = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: cigar_size,
        mapped_at_creation: false,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });

    let staging_scores = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: scores_size,
        mapped_at_creation: false,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
    });

    let staging_cigar = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: cigar_size,
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
                resource: buffer_a.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buffer_b.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: buffer_scores.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: buffer_dirs.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: buffer_cigar.as_entire_binding(),
            },
        ],
    });

    while let Some(batch) = rx.recv().await {
        queue.write_buffer(&buffer_a, 0, bytemuck::cast_slice(&batch.packed_a));
        queue.write_buffer(&buffer_b, 0, bytemuck::cast_slice(&batch.packed_b));

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });

            cpass.set_pipeline(&compute_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups((batch.valid_reads as f32 / 64.0).ceil() as u32, 1, 1);
        }

        let copy_scores_size = (batch.valid_reads * 2 * 4) as wgpu::BufferAddress;
        let copy_cigar_size = (batch.valid_reads * 16 * 4) as wgpu::BufferAddress;

        encoder.copy_buffer_to_buffer(&buffer_scores, 0, &staging_scores, 0, copy_scores_size);
        encoder.copy_buffer_to_buffer(&buffer_cigar, 0, &staging_cigar, 0, copy_cigar_size);

        queue.submit(Some(encoder.finish()));

        let (tx1, rx1) = tokio::sync::oneshot::channel();
        let (tx2, rx2) = tokio::sync::oneshot::channel();

        staging_scores
            .slice(..copy_scores_size)
            .map_async(wgpu::MapMode::Read, move |v| {
                tx1.send(v).unwrap();
            });

        staging_cigar
            .slice(..copy_cigar_size)
            .map_async(wgpu::MapMode::Read, move |v| {
                tx2.send(v).unwrap();
            });

        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();

        if rx1.await.is_ok() && rx2.await.is_ok() {
            let data_score = staging_scores
                .slice(..copy_scores_size)
                .get_mapped_range()
                .unwrap();

            let data_cigar = staging_cigar
                .slice(..copy_cigar_size)
                .get_mapped_range()
                .unwrap();

            if score_tx
                .send(ScoreBatch {
                    scores_and_coords: bytemuck::cast_slice(&data_score).to_vec(),
                    cigar_bin: bytemuck::cast_slice(&data_cigar).to_vec(),
                })
                .await
                .is_err()
            {
                break;
            }
        }

        // Обязательно освобождаем буферы перед следующим батчем
        staging_scores.unmap();
        staging_cigar.unmap();
    }
}

async fn writer_task(
    output_path: String,
    len_genome: usize,
    batch_size: usize,
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
        .write_all(format!("@SQ\tSN:reference\tLN:{len_genome}\n").as_bytes())
        .await
        .unwrap();

    let mut total_written = 0;

    while let (Some(meta_batch), Some(score_batch)) = (meta_rx.recv().await, score_rx.recv().await)
    {
        let sam_lines: Vec<String> = tokio::task::spawn_blocking(move || {
            meta_batch
                .reads
                .into_par_iter()
                .enumerate()
                .map(|(read_idx, meta)| {
                    // Читаем скор и смещение (умножение на 2!)
                    let max_score = score_batch.scores_and_coords[read_idx * 2];
                    let align_start_j =
                        score_batch.scores_and_coords[read_idx * 2 + 1].cast_unsigned();

                    let meta_name = unsafe { std::str::from_utf8_unchecked(&meta.name) };
                    let meta_seq = unsafe { std::str::from_utf8_unchecked(&meta.seq) };
                    let meta_qual = unsafe { std::str::from_utf8_unchecked(&meta.qual) };

                    let flag = if meta.is_revcomp { 16 } else { 0 };

                    // Мусор фильтруем
                    if max_score < 200 {
                        return format!(
                            "{meta_name}\t4\t*\t0\t0\t*\t*\t0\t0\t{meta_seq}\t{meta_qual}\n"
                        );
                    }

                    // Идеальные совпадения
                    if max_score == (i32::try_from(READ_LEN).unwrap() * MATCH_SCORE) {
                        return format!(
                            "{}\t{}\treference\t{}\t60\t{}M\t*\t0\t0\t{}\t{}\tAS:i:{}\n",
                            meta_name,
                            flag,
                            meta.pos + 15,
                            READ_LEN,
                            meta_seq,
                            meta_qual,
                            max_score
                        );
                    }

                    let mut final_cigar = String::new();
                    let mut cur_op = 'X';
                    let mut cur_count = 0;

                    for k in (0..16).rev() {
                        let val = score_batch.cigar_bin[read_idx * 16 + k];
                        if val != 0xFFFF_FFFF {
                            let count = val >> 2;
                            let op = match val & 3 {
                                0 => 'M',
                                1 => 'I',
                                2 => 'D',
                                _ => 'S',
                            };

                            if op == cur_op {
                                cur_count += count; // Склеиваем одинаковые операции!
                            } else {
                                if cur_op != 'X' {
                                    let _ = write!(final_cigar, "{cur_count}{cur_op}");
                                }
                                cur_op = op;
                                cur_count = count;
                            }
                        }
                    }
                    if cur_op != 'X' {
                        let _ = write!(final_cigar, "{cur_count}{cur_op}");
                    }

                    let exact_pos = meta.pos + align_start_j + 1;
                    format!(
                        "{meta_name}\t{flag}\treference\t{exact_pos}\t60\t{final_cigar}\
                        \t*\t0\t0\t{meta_seq}\t{meta_qual}\tAS:i:{max_score}\n",
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
        println!("📝 [Writer] Файл записан. Выровнено ридов: {total_written}");
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
    let genome_arc = Arc::new(genome);
    let genome_len = genome_arc.len();

    let (chunk_tx, chunk_rx) = mpsc::channel(3); // От Reader к Producer
    let (gpu_tx, gpu_rx) = mpsc::channel(3); // От Producer к GPU
    let (meta_tx, meta_rx) = mpsc::channel(3); // От Producer к Writer
    let (score_tx, score_rx) = mpsc::channel(3); // От GPU к Writer

    let pb = if args.progress {
        let pb = ProgressBar::new_spinner();
        let style = ProgressStyle::default_spinner()
            .template("{spinner:.green} [{elapsed_precise}] {msg} {pos} ридов ({per_sec})")
            .unwrap();
        pb.set_style(style);
        pb.set_message("Выравнивание...");
        pb.enable_steady_tick(Duration::from_millis(100));
        pb
    } else {
        ProgressBar::hidden()
    };

    reader_task(args.reads.clone(), args.batch_size, chunk_tx);
    let producer_handle = producer_task(args.clone(), genome_arc, chunk_rx, gpu_tx, meta_tx);
    let gpu_handle = tokio::spawn(gpu_actor(args.clone(), gpu_rx, score_tx));
    let writer_handle = tokio::spawn(writer_task(
        args.output.clone(),
        genome_len,
        args.batch_size,
        meta_rx,
        score_rx,
        pb.clone(),
    ));

    let (prod_res, gpu_res, writer_res) = tokio::join!(producer_handle, gpu_handle, writer_handle);

    prod_res.unwrap();
    gpu_res.unwrap();
    writer_res.unwrap();

    if args.progress {
        pb.finish_with_message("✅ Готово!");
    }
    println!("Общее время работы: {:?}", index_start.elapsed());
}
