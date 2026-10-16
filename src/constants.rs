pub const MATCH_SCORE: i32 = 3;
pub const READ_LEN: usize = 100;
pub const REF_LEN: usize = 128;
pub const READ_WORDS: usize = 7;
pub const REF_WORDS: usize = 8;
pub const REF_WORDS_4BIT: usize = 16; // 128 нуклеотидов / 8 клеток на u32
pub const N_MASK_WORDS: usize = 4; // ceil(128 / 32) = 4 слова для масок 'N'
pub const SCORES_PER_READ: usize = 4; // max_score, max_i, max_j, actual_rows
pub const CIGAR_CAPACITY: usize = 32; // 32 операции вместо 16
pub const PADDING_LEN: usize = 256;

