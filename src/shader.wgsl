@group(0) @binding(0) var<storage, read> seqs_a: array<u32>; // Референсы (8 u32 на рид)
@group(0) @binding(1) var<storage, read> seqs_b: array<u32>; // Риды (7 u32 на рид)
@group(0) @binding(2) var<storage, read_write> scores_out: array<i32>; // [score, max_i, max_j]
@group(0) @binding(3) var<storage, read_write> directions: array<u32>; // 100 * 8 u32 на рид

const READ_LEN: usize = 100;
const REF_LEN: usize = 128;
const READ_WORDS: usize = 7;
const REF_WORDS: usize = 8;

const MATCH_SCORE: i32 = 3;
const MISMATCH_PENALTY: i32 = -3;
const GAP_PENALTY: i32 = -2;

@compute
@workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let batch_size = arrayLength(&scores_out) / 3u;
    if idx >= batch_size { return; }

    // 1. ЗАГРУЖАЕМ В РЕГИСТРЫ (Zero Spilling!)
    var read_packed: array<u32, 7>;
    for (var w = 0u; w < READ_WORDS; w = w + 1u) {
        read_packed[w] = seqs_b[w * batch_size + idx];
    }

    var ref_packed: array<u32, 8>;
    for (var w = 0u; w < REF_WORDS; w = w + 1u) {
        ref_packed[w] = seqs_a[w * batch_size + idx];
    }

    // 2. ДИНАМИЧЕСКОЕ ПРОГРАММИРОВАНИЕ
    var rows: array<array<i32, 129>, 2>;
    var max_score: i32 = 0;
    var max_i: u32 = 0u;
    var max_j: u32 = 0u;

    for (var i = 0u; i < READ_LEN; i = i + 1u) {
        let curr_row_idx = i & 1u;
        let prev_row_idx = 1u - curr_row_idx;

        // Декодируем букву рида на лету
        let char_read = (read_packed[i >> 4u] >> ((i & 15u) << 1u)) & 3u;

        rows[curr_row_idx][0] = 0;
        var left: i32 = 0;

        var trace_row: array<u32, 8>; // Буфер для направлений (8 * 32 = 256 бит > 128 клеток * 2)

        for (var j = 0u; j < REF_LEN; j = j + 1u) {
            // Декодируем букву референса на лету
            let char_ref = (ref_packed[j >> 4u] >> ((j & 15u) << 1u)) & 3u;

            let match_val = select(MISMATCH_PENALTY, MATCH_SCORE, char_read == char_ref);

            let diag = rows[prev_row_idx][j] + match_val;
            let up = rows[prev_row_idx][j + 1] + GAP_PENALTY;
            let left_val = left + GAP_PENALTY;

            let current = max(0, max(diag, max(up, left_val)));
            rows[curr_row_idx][j + 1] = current;
            left = current;

            if current > max_score {
                max_score = current; max_i = i; max_j = j;
            }

            var dir = 0u;
            if current > 0 {
                if current == diag { dir = 1u; } // 1 = Match
                else if current == up { dir = 2u; } // 2 = Insertion (Read consumed, Ref wait)
                else if current == left_val { dir = 3u; } // 3 = Deletion (Ref consumed, Read wait)
            }
            trace_row[j >> 4u] |= (dir << ((j & 15u) << 1u));
        }

        // Записываем строку Traceback в глобальную память
        for (var w = 0u; w < REF_WORDS; w = w + 1u) {
            directions[(i * REF_WORDS + w) * batch_size + idx] = trace_row[w];
        }
    }

    scores_out[idx * 3u + 0u] = max_score;
    scores_out[idx * 3u + 1u] = i32(max_i);
    scores_out[idx * 3u + 2u] = i32(max_j);
}