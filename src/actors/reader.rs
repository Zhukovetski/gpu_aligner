use std::{
    fs::File,
    io::{BufRead, BufReader, Read},
    mem::take,
};

use bio::io::fastq;
use bio::io::fastq::FastqRead;
use flate2::read::MultiGzDecoder;
use tokio::sync::mpsc;

pub struct RawPair {
    pub name: String,
    pub seq1: Vec<u8>,
    pub qual1: Vec<u8>,
    pub seq2: Vec<u8>,
    pub qual2: Vec<u8>,
}

pub struct PairChunk {
    pub pairs: Vec<RawPair>,
}

pub fn open_flexible_reader(path: &str) -> BufReader<Box<dyn Read + Send>> {
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

fn trim_newline_inplace(s: &mut Vec<u8>) {
    let mut len = s.len();
    while len > 0 && (s[len - 1] == b'\n' || s[len - 1] == b'\r') {
        len -= 1;
    }
    s.truncate(len); // Меняет длину вектора без перевыделения памяти!
}

fn pair_reader_task(
    path1: String,
    path2: String,
    batch_size: usize,
    chunk_tx: mpsc::Sender<PairChunk>,
) -> tokio::task::JoinHandle<()> {
    tokio::task::spawn_blocking(move || {
        let mut r1 = fastq::Reader::new(open_flexible_reader(&path1));
        let mut r2 = fastq::Reader::new(open_flexible_reader(&path2));

        let mut rec1 = fastq::Record::default();
        let mut rec2 = fastq::Record::default();

        let mut current_pairs = Vec::with_capacity(batch_size);

        loop {
            r1.read(&mut rec1).expect("Error reading FASTQ file 1");
            r2.read(&mut rec2).expect("Error reading FASTQ file 2");

            // Если оба файла закончились
            if rec1.is_empty() || rec2.is_empty() {
                break;
            }

            current_pairs.push(RawPair {
                name: rec1.id().to_string(),
                seq1: rec1.seq().to_vec(),
                qual1: rec1.qual().to_vec(),
                seq2: rec2.seq().to_vec(),
                qual2: rec2.qual().to_vec(),
            });

            if current_pairs.len() >= batch_size {
                let chunk = PairChunk {
                    pairs: std::mem::replace(&mut current_pairs, Vec::with_capacity(batch_size)),
                };
                if chunk_tx.blocking_send(chunk).is_err() {
                    return;
                }
            }
        }
        if !current_pairs.is_empty() {
            let chunk = PairChunk {
                pairs: current_pairs,
            };
            let _ = chunk_tx.blocking_send(chunk);
        }
    })
}

pub struct RawRead {
    pub name: Vec<u8>,
    pub seq: Vec<u8>,
    pub qual: Vec<u8>,
}

pub struct ReadChunk {
    pub reads: Vec<RawRead>,
}

// 1. Убираем async, чтобы функция мгновенно возвращала управление рантайму
pub fn reader_task(
    reads_path: String,
    batch_size: usize,
    chunk_tx: mpsc::Sender<ReadChunk>,
) -> tokio::task::JoinHandle<()> {
    // spawn_blocking запускается в фоне, мы НЕ пишем .await в конце функции
    tokio::task::spawn_blocking(move || {
        println!("📦 [Reader] Начало стриминга чтений из {reads_path}...");

        let mut fastq_reader = open_flexible_reader(&reads_path);

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
