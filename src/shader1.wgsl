@group(0) @binding(0) var<storage, read> seqs_a: array<u32>;
@group(0) @binding(1) var<storage, read> seqs_b: array<u32>;
@group(0) @binding(2) var<storage, read_write> scores_out: array<i32>;
@group(0) @binding(3) var<storage, read_write> directions: array<u32>; // НОВЫЙ БУФЕР

const SEQ_LEN: u32 = 100u;
const PACKED_LEN: u32 = 7u; 

const MATCH_SCORE: i32 = 3;
const MISMATCH_PENALTY: i32 = -3;
const GAP_PENALTY: i32 = -2;

@compute
@workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    // В scores_out теперь 3 числа на рид: [score, max_i, max_j]
    let batch_size = arrayLength(&scores_out) / 3u;
    if idx >= batch_size { return; }

    var local_seq_b: array<u32, 112>;
    for (var w = 0u; w < PACKED_LEN; w = w + 1u) {
        let packed_val = seqs_b[w * batch_size + idx];
        for (var b = 0u; b < 16u; b = b + 1u) {
            local_seq_b[(w << 4u) + b] = (packed_val >> (b << 1u)) & 3u;
        }
    }

    var rows: array<array<i32, 101>, 2>;
    var max_score: i32 = 0;
    var max_i: u32 = 0u;
    var max_j: u32 = 0u;

    for (var i = 0u; i < SEQ_LEN; i = i + 1u) {
        let curr_row_idx = i & 1u;
        let prev_row_idx = 1u - curr_row_idx;

        let w_idx = i >> 4u;
        let shift_a = (i & 15u) << 1u;
        let char_a = (seqs_a[w_idx * batch_size + idx] >> shift_a) & 3u;

        rows[curr_row_idx][0] = 0;
        var left: i32 = 0;

        // В этой строке мы будем накапливать 2-битные направления
        var trace_row: array<u32, 7>;

        for (var j = 0u; j < SEQ_LEN; j = j + 1u) {
            let char_b = local_seq_b[j];
            let match_val = select(MISMATCH_PENALTY, MATCH_SCORE, char_a == char_b);

            let diag = rows[prev_row_idx][j] + match_val;
            let up = rows[prev_row_idx][j + 1] + GAP_PENALTY;
            let left_val = left + GAP_PENALTY;

            let current = max(0, max(diag, max(up, left_val)));
            rows[curr_row_idx][j + 1] = current;
            left = current;

            // Сохраняем не только скор, но и координаты!
            if current > max_score {
                max_score = current;
                max_i = i;
                max_j = j;
            }

            // Вычисляем направление для Traceback (0=Stop, 1=Diag, 2=Up, 3=Left)
            var dir = 0u;
            if current > 0 {
                if current == diag { dir = 1u; }
                else if current == up { dir = 2u; }
                else if current == left_val { dir = 3u; }
            }

            // Упаковываем 2 бита в наш u32
            trace_row[j >> 4u] |= (dir << ((j & 15u) << 1u));
        }

        // Записываем строку направлений в VRAM (SoA укладка для скорости!)
        for (var w = 0u; w < PACKED_LEN; w = w + 1u) {
            directions[(i * PACKED_LEN + w) * batch_size + idx] = trace_row[w];
        }
    }

    // Возвращаем результат
    scores_out[idx * 3u + 0u] = max_score;
    scores_out[idx * 3u + 1u] = i32(max_i);
    scores_out[idx * 3u + 2u] = i32(max_j);
}