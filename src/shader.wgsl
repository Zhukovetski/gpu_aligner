@group(0) @binding(0)
var<storage, read> seqs_a: array<u32>;
@group(0) @binding(1)
var<storage, read> seqs_b: array<u32>;
@group(0) @binding(2)
var<storage, read_write> scores_out: array<i32>;
@group(0) @binding(3)
var<storage, read_write> directions: array<u32>;
@group(0) @binding(4)
var<storage, read_write> cigar_out: array<u32>;

const READ_LEN: u32 = 100u;
const REF_LEN: u32 = 128u;
const READ_WORDS: u32 = 7u;
const REF_WORDS: u32 = 8u;

const MATCH_SCORE: i32 = 3;
const MISMATCH_PENALTY: i32 = -3;
const GAP_PENALTY: i32 = -2;

// Достать нуклеотид (A, C, G, T) из упакованного u32 массива по индексу
fn get_nucleotide(packed_array: array<u32, 8>, index: u32) -> u32 {
    let word_idx = index >> 4u; // index / 16
    let bit_offset = (index & 15u) << 1u; // (index % 16) * 2
    return (packed_array[word_idx] >> bit_offset) & 3u;
}

// Перегрузка для ридов (так как размер массива отличается, в WGSL они жестко привязаны к размеру)
fn get_read_nucleotide(packed_array: array<u32, 7>, index: u32) -> u32 {
    let word_idx = index >> 4u;
    let bit_offset = (index & 15u) << 1u;
    return (packed_array[word_idx] >> bit_offset) & 3u;
}

// Закодировать направление в 2 бита и впечь в маску строки
fn pack_direction(current_mask: u32, index: u32, direction: u32) -> u32 {
    let bit_offset = (index & 15u) << 1u;
    return current_mask | (direction << bit_offset);
}

// Извлечь направление из глобальной памяти
fn get_saved_direction(idx: u32, batch_size: u32, i: u32, j: u32) -> u32 {
    let w = j >> 4u;
    let bit_offset = (j & 15u) << 1u;
    let dir_val = directions[(i * REF_WORDS + w) * batch_size + idx];
    return (dir_val >> bit_offset) & 3u;
}

// Упаковать операцию CIGAR (длина + тип) в одно число u32
fn pack_cigar_op(count: u32, op: u32) -> u32 {
    return (count << 2u) | op;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let batch_size = arrayLength(&scores_out) / 2u;
    if idx >= batch_size {
        return;
    }

    // ШАГ 1: ЗАГРУЗКА ДАННЫХ ИЗ ГЛОБАЛЬНОЙ ПАМЯТИ В РЕГИСТРЫ
    var read_packed: array<u32, 7>;
    for(var w = 0u; w < READ_WORDS; w = w + 1u) {
        read_packed[w] = seqs_b[w * batch_size + idx];
    }

    var ref_packed: array<u32, 8>;
    for(var w = 0u; w < REF_WORDS; w = w + 1u) {
        ref_packed[w] = seqs_a[w * batch_size + idx];
    }

    // Инициализация двух рабочих строк матрицы скоринга
    var rows: array<array<i32, 129>, 2>;
    for(var k = 0u; k < 129u; k = k + 1u) {
        rows[0][k] = 0;
        rows[1][k] = 0;
    }

    var max_score: i32 = 0;
    var max_i: u32 = 0u;
    var max_j: u32 = 0u;

    // ШАГ 2: РАСЧЕТ МАТРИЦЫ И СОХРАНЕНИЕ НАПРАВЛЕНИЙ
    for(var i = 0u; i < READ_LEN; i = i + 1u) {
        let curr_row_idx = i & 1u;
        let prev_row_idx = 1u - curr_row_idx;
        let char_read = get_read_nucleotide(read_packed, i);

        rows[curr_row_idx][0] = 0;
        var left: i32 = 0;

        var trace_row: array<u32, 8>;
        for(var k = 0u; k < 8u; k = k + 1u) {
            trace_row[k] = 0u;
        }

        for(var j = 0u; j < REF_LEN; j = j + 1u) {
            let char_ref = get_nucleotide(ref_packed, j);
            let match_val = select(MISMATCH_PENALTY, MATCH_SCORE, char_read == char_ref);

            let diag = rows[prev_row_idx][j] + match_val;
            let up = rows[prev_row_idx][j + 1] + GAP_PENALTY;
            let left_val = left + GAP_PENALTY;

            let current = max(0, max(diag, max(up, left_val)));
            rows[curr_row_idx][j + 1] = current;
            left = current;

            // 1. Обновление max_score, max_i, max_j
            let is_new_max = current > max_score;
            max_score = select(max_score, current, is_new_max);
            max_i = select(max_i, i, is_new_max);
            max_j = select(max_j, j, is_new_max);

            // 2. Определение направления dir
            // Вычисляем маски для каждого условия
            let is_positive = current > 0;
            let is_diag = current == diag;
            let is_up = current == up;
            let is_left = current == left_val;

            // Собираем dir последовательно с ног на голову (или через вложенные select)
            var dir = 0u;
            dir = select(dir, 3u, is_positive && is_left);
            dir = select(dir, 2u, is_positive && is_up);
            dir = select(dir, 1u, is_positive && is_diag);

            trace_row[j >> 4u] = pack_direction(trace_row[j >> 4u], j, dir);
        }

        // Сброс посчитанной строки направлений в глобальную память GPU
        for(var w = 0u; w < REF_WORDS; w = w + 1u) {
            directions[(i * REF_WORDS + w) * batch_size + idx] = trace_row[w];
        }
    }

    // ШАГ 3: ОБРАТНЫЙ ПУТЬ И СБОРКА CIGAR-СТРОКИ

    var cigar: array<u32, 16>;
    for(var k = 0u; k < 16u; k = k + 1u) {
        cigar[k] = 0xFFFFFFFFu;
    }
    var cigar_idx = 0u;

    var cur_op = 0xFFu;
    var cur_count = 0u;

    var i = max_i;
    var j = max_j;
    var align_start_j = j;

    // Мягкое клиппирование хвоста рида, если выравнивание завершилось раньше конца
    let clip_end = READ_LEN - 1u - max_i;
    if clip_end > 0u {
        cur_op = 3u; // Операция 'S' (Soft clip)
        cur_count = clip_end;
    }

    loop {
        let dir = get_saved_direction(idx, batch_size, i, j);

        // Стоп: дошли до начала локального выравнивания (скор упал до 0)
        if dir == 0u {
            let clip_start = i + 1u;
            if clip_start > 0u {
                if cur_op == 3u {
                    cur_count += clip_start;
                } else {
                    if cur_op != 0xFFu {
                        cigar[cigar_idx] = pack_cigar_op(cur_count, cur_op);
                        cigar_idx += 1u;
                    }
                    cur_op = 3u;
                    cur_count = clip_start;
                }
            }
            break;
        }

        align_start_j = j;
        var step_op = 0u;

        if dir == 1u {
            // Диагональ (М)
            step_op = 0u;
            if i == 0u || j == 0u {
                if cur_op == step_op {
                    cur_count += 1u;
                } else {
                    if cur_op != 0xFFu {
                        cigar[cigar_idx] = pack_cigar_op(cur_count, cur_op);
                        cigar_idx += 1u;
                    }
                    cur_op = step_op;
                    cur_count = 1u;
                }
                if i > 0u {
                    if cur_op != 0xFFu {
                        cigar[cigar_idx] = pack_cigar_op(cur_count, cur_op);
                        cigar_idx += 1u;
                    }
                    cur_op = 3u;
                    cur_count = i;
                }
                break;
            }
            i -= 1u;
            j -= 1u;
        } else if dir == 2u {
            // Сверху (I)
            step_op = 1u;
            if i == 0u {
                if cur_op == step_op {
                    cur_count += 1u;
                } else {
                    if cur_op != 0xFFu {
                        cigar[cigar_idx] = pack_cigar_op(cur_count, cur_op);
                        cigar_idx += 1u;
                    }
                    cur_op = step_op;
                    cur_count = 1u;
                }
                break;
            }
            i -= 1u;
        } else if dir == 3u {
            // Слева (D)
            step_op = 2u;
            if j == 0u {
                if cur_op == step_op {
                    cur_count += 1u;
                } else {
                    if cur_op != 0xFFu {
                        cigar[cigar_idx] = pack_cigar_op(cur_count, cur_op);
                        cigar_idx += 1u;
                    }
                    cur_op = step_op;
                    cur_count = 1u;
                }
                if i > 0u {
                    if cur_op != 0xFFu {
                        cigar[cigar_idx] = pack_cigar_op(cur_count, cur_op);
                        cigar_idx += 1u;
                    }
                    cur_op = 3u;
                    cur_count = i + 1u;
                }
                break;
            }
            j -= 1u;
        }

        // Группировка одинаковых операций (Run-Length Encoding)
        if cur_op == step_op {
            cur_count += 1u;
        } else {
            if cur_op != 0xFFu {
                cigar[cigar_idx] = pack_cigar_op(cur_count, cur_op);
                cigar_idx += 1u;
            }
            cur_op = step_op;
            cur_count = 1u;
        }
    }

    if cur_op != 0xFFu {
        cigar[cigar_idx] = pack_cigar_op(cur_count, cur_op);
    }

    // ШАГ 4: ЗАПИСЬ ФИНАЛЬНЫХ РЕЗУЛЬТАТОВ ДЛЯ CPU
    scores_out[idx * 2u + 0u] = max_score;
    scores_out[idx * 2u + 1u] = i32(align_start_j);

    for(var k = 0u; k < 16u; k = k + 1u) {
        // Умножаем индекс рида на 16 и прибавляем смещение
        cigar_out[idx * 16u + k] = cigar[k];
    }
}
