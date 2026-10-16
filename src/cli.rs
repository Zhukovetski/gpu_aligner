use clap::Parser;

#[derive(Parser, Debug, Clone)]
#[command(author = "Твое Имя", version = "1.0", about, long_about = None)]
pub struct Args {
    #[arg(short = 'r', long)]
    pub reference: String,

    /// Первый файл ридов (или единственный для Single-End)
    #[arg(short = 'i', short_alias = '1', long = "reads")]
    pub reads: String,

    /// Второй файл ридов для Paired-End (опционально)
    #[arg(short = '2', long = "reads2")]
    pub reads2: Option<String>,

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
