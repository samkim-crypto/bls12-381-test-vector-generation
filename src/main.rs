use blst::{blst_bendian_from_fp, blst_fp12, blst_lendian_from_fp};
use blstrs::{G1Affine, G1Projective, G2Affine, G2Projective, Gt, Scalar};
use group::{Curve, Group, ff::Field, prime::PrimeCurveAffine};
use hex_literal::hex;
use rand::rngs::OsRng;
use std::collections::BTreeMap;

#[derive(Clone, Copy)]
enum Endianness {
    BE,
    LE,
}

// Helper functions matching the solana-bls12-381-core implementation
fn reverse_48_byte_chunks(bytes: &mut [u8]) {
    for chunk in bytes.chunks_mut(48) {
        chunk.reverse();
    }
}

// Swaps c0 and c1 for G2 elements (Fq2). Operates on 96-byte chunks.
fn swap_g2_c0_c1(bytes: &mut [u8]) {
    for fq2_chunk in bytes.chunks_exact_mut(96) {
        // In BE (blstrs output), this chunk is [c1, c0]. We swap them for LE structure.
        let (c1, c0) = fq2_chunk.split_at_mut(48);
        c0.swap_with_slice(c1);
    }
}

// Struct to hold test case data
#[derive(Debug, Clone)]
struct TestCase {
    op_type: String, // e.g., "ADDITION", "VALIDATION" (used for filename)
    name: String,
    input_be: String,
    output_be: String, // Can be hex bytes or a boolean string
    input_le: String,
    output_le: String, // Can be hex bytes or a boolean string
}

// --- Transformation helpers (G1/G2 to BE/LE) ---
fn g1_to_be(p: &G1Affine) -> [u8; 96] {
    p.to_uncompressed()
}
fn g1_to_le(p: &G1Affine) -> [u8; 96] {
    let mut b = p.to_uncompressed();
    reverse_48_byte_chunks(&mut b);
    b
}
fn g1_compress_to_be(p: &G1Affine) -> [u8; 48] {
    p.to_compressed()
}
fn g1_compress_to_le(p: &G1Affine) -> [u8; 48] {
    let mut b = p.to_compressed();
    reverse_48_byte_chunks(&mut b); // Handles byte order and flag position
    b
}

fn g2_to_be(p: &G2Affine) -> [u8; 192] {
    p.to_uncompressed()
}
fn g2_to_le(p: &G2Affine) -> [u8; 192] {
    let mut b = p.to_uncompressed();
    swap_g2_c0_c1(&mut b);
    reverse_48_byte_chunks(&mut b);
    b
}
fn g2_compress_to_be(p: &G2Affine) -> [u8; 96] {
    p.to_compressed()
}
fn g2_compress_to_le(p: &G2Affine) -> [u8; 96] {
    let mut b = p.to_compressed();
    swap_g2_c0_c1(&mut b);
    reverse_48_byte_chunks(&mut b);
    b
}

// --- Test Case Generators (Addition/Subtraction) ---
fn generate_g1_op(op: &str, name: &str, p1: &G1Projective, p2: &G1Projective) -> TestCase {
    let op_type = if op == "ADD" {
        "ADDITION"
    } else {
        "SUBTRACTION"
    }
    .to_string();
    let result = if op == "ADD" { p1 + p2 } else { p1 - p2 };
    let (p1_a, p2_a, res_a) = (
        G1Affine::from(p1),
        G1Affine::from(p2),
        G1Affine::from(result),
    );

    TestCase {
        op_type,
        name: format!("G1_{}_{}", op, name),
        input_be: hex::encode([g1_to_be(&p1_a), g1_to_be(&p2_a)].concat()),
        output_be: hex::encode(g1_to_be(&res_a).to_vec()),
        input_le: hex::encode([g1_to_le(&p1_a), g1_to_le(&p2_a)].concat()),
        output_le: hex::encode(g1_to_le(&res_a).to_vec()),
    }
}

fn generate_g2_op(op: &str, name: &str, p1: &G2Projective, p2: &G2Projective) -> TestCase {
    let op_type = if op == "ADD" {
        "ADDITION"
    } else {
        "SUBTRACTION"
    }
    .to_string();
    let result = if op == "ADD" { p1 + p2 } else { p1 - p2 };
    let (p1_a, p2_a, res_a) = (
        G2Affine::from(p1),
        G2Affine::from(p2),
        G2Affine::from(result),
    );

    TestCase {
        op_type,
        name: format!("G2_{}_{}", op, name),
        input_be: hex::encode([g2_to_be(&p1_a), g2_to_be(&p2_a)].concat()),
        output_be: hex::encode(g2_to_be(&res_a).to_vec()),
        input_le: hex::encode([g2_to_le(&p1_a), g2_to_le(&p2_a)].concat()),
        output_le: hex::encode(g2_to_le(&res_a).to_vec()),
    }
}

// --- Test Case Generators (Multiplication) ---
fn generate_g1_mul(name: &str, p: &G1Projective, s: &Scalar) -> TestCase {
    let result = p * s;
    let (p_a, res_a) = (G1Affine::from(p), G1Affine::from(result));

    let s_le = s.to_bytes_le();
    let s_be = s.to_bytes_be();

    TestCase {
        op_type: "MULTIPLICATION".to_string(),
        name: format!("G1_MUL_{}", name),
        input_be: hex::encode([g1_to_be(&p_a).as_ref(), s_be.as_ref()].concat()),
        output_be: hex::encode(g1_to_be(&res_a).to_vec()),
        input_le: hex::encode([g1_to_le(&p_a).as_ref(), s_le.as_ref()].concat()),
        output_le: hex::encode(g1_to_le(&res_a).to_vec()),
    }
}

fn generate_g2_mul(name: &str, p: &G2Projective, s: &Scalar) -> TestCase {
    let result = p * s;
    let (p_a, res_a) = (G2Affine::from(p), G2Affine::from(result));

    let s_le = s.to_bytes_le();
    let s_be = s.to_bytes_be();

    TestCase {
        op_type: "MULTIPLICATION".to_string(),
        name: format!("G2_MUL_{}", name),
        input_be: hex::encode([g2_to_be(&p_a).as_ref(), s_be.as_ref()].concat()),
        output_be: hex::encode(g2_to_be(&res_a).to_vec()),
        input_le: hex::encode([g2_to_le(&p_a).as_ref(), s_le.as_ref()].concat()),
        output_le: hex::encode(g2_to_le(&res_a).to_vec()),
    }
}

// --- Test Case Generators (Decompression) ---
fn generate_g1_decompress(name: &str, p: &G1Projective) -> TestCase {
    let p_a = G1Affine::from(p);
    TestCase {
        op_type: "DECOMPRESSION".to_string(),
        name: format!("G1_DECOMPRESS_{}", name),
        input_be: hex::encode(g1_compress_to_be(&p_a).to_vec()),
        output_be: hex::encode(g1_to_be(&p_a).to_vec()),
        input_le: hex::encode(g1_compress_to_le(&p_a).to_vec()),
        output_le: hex::encode(g1_to_le(&p_a).to_vec()),
    }
}

fn generate_g2_decompress(name: &str, p: &G2Projective) -> TestCase {
    let p_a = G2Affine::from(p);
    TestCase {
        op_type: "DECOMPRESSION".to_string(),
        name: format!("G2_DECOMPRESS_{}", name),
        input_be: hex::encode(g2_compress_to_be(&p_a).to_vec()),
        output_be: hex::encode(g2_to_be(&p_a).to_vec()),
        input_le: hex::encode(g2_compress_to_le(&p_a).to_vec()),
        output_le: hex::encode(g2_to_le(&p_a).to_vec()),
    }
}

// Generate a compressed point that is valid field element but NOT on the curve
fn generate_g1_decompress_invalid_curve(name: &str, p: &G1Projective) -> TestCase {
    let p_a = G1Affine::from(p);
    let mut compressed_be = g1_compress_to_be(&p_a);

    // Corrupt the last byte (least significant in BE).
    // This changes x -> x'. It is extremely unlikely x' is also on the curve.
    compressed_be[47] = compressed_be[47].wrapping_add(1);

    // Prepare LE version
    let mut compressed_le = compressed_be;
    reverse_48_byte_chunks(&mut compressed_le);

    TestCase {
        op_type: "DECOMPRESSION".to_string(),
        name: format!("G1_DECOMPRESS_{}_INVALID_CURVE", name),
        input_be: hex::encode(compressed_be),
        output_be: "INVALID".to_string(),
        input_le: hex::encode(compressed_le),
        output_le: "INVALID".to_string(),
    }
}

// Generate a compressed point where x >= P (Modulus)
fn generate_g1_decompress_invalid_field() -> TestCase {
    // P = 0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab
    let modulus_p_be: [u8; 48] = hex!(
        "1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab"
    );

    let mut invalid_be = modulus_p_be;
    // Set Compressed flag (bit 7 of byte 0).
    // 0x1a is 0001 1010. 0x80 | 0x1a = 0x9a (1001 1010).
    // This creates a "Compressed Point" claim where the value is exactly P.
    invalid_be[0] |= 0x80;

    let mut invalid_le = invalid_be;
    reverse_48_byte_chunks(&mut invalid_le);

    TestCase {
        op_type: "DECOMPRESSION".to_string(),
        name: "G1_DECOMPRESS_FIELD_TOO_LARGE_INVALID".to_string(),
        input_be: hex::encode(invalid_be),
        output_be: "INVALID".to_string(),
        input_le: hex::encode(invalid_le),
        output_le: "INVALID".to_string(),
    }
}

// Same for G2
fn generate_g2_decompress_invalid_curve(name: &str, p: &G2Projective) -> TestCase {
    let p_a = G2Affine::from(p);
    let mut compressed_be = g2_compress_to_be(&p_a);

    // Corrupt the last byte of the whole array.
    compressed_be[95] = compressed_be[95].wrapping_add(1);

    let mut compressed_le = compressed_be;
    swap_g2_c0_c1(&mut compressed_le);
    reverse_48_byte_chunks(&mut compressed_le);

    TestCase {
        op_type: "DECOMPRESSION".to_string(),
        name: format!("G2_DECOMPRESS_{}_INVALID_CURVE", name),
        input_be: hex::encode(compressed_be),
        output_be: "INVALID".to_string(),
        input_le: hex::encode(compressed_le),
        output_le: "INVALID".to_string(),
    }
}

fn generate_g2_decompress_invalid_field() -> TestCase {
    // Create an invalid G2 where c0 >= P.
    // G2 Compressed BE structure: [c1 (with flags), c0]
    let modulus_p_be: [u8; 48] = hex!(
        "1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab"
    );

    let mut invalid_be = [0u8; 96];

    // Set c1 to 0 (valid) but set Compressed flag
    invalid_be[0] |= 0x80;

    // Set c0 to P (invalid)
    invalid_be[48..96].copy_from_slice(&modulus_p_be);

    let mut invalid_le = invalid_be;
    swap_g2_c0_c1(&mut invalid_le);
    reverse_48_byte_chunks(&mut invalid_le);

    TestCase {
        op_type: "DECOMPRESSION".to_string(),
        name: "G2_DECOMPRESS_FIELD_TOO_LARGE_INVALID".to_string(),
        input_be: hex::encode(invalid_be),
        output_be: "INVALID".to_string(),
        input_le: hex::encode(invalid_le),
        output_le: "INVALID".to_string(),
    }
}

// --- Test Case Generators (Validation) ---

// Helper for validation (takes raw BE bytes, calculates LE bytes)
fn generate_g1_validation_raw(name: &str, bytes_be: &[u8; 96], expected_valid: bool) -> TestCase {
    let mut bytes_le = *bytes_be;
    reverse_48_byte_chunks(&mut bytes_le);
    let expected_str = expected_valid.to_string();

    TestCase {
        op_type: "VALIDATION".to_string(),
        name: format!("G1_VALIDATE_{}", name),
        input_be: hex::encode(bytes_be),
        output_be: expected_str.clone(),
        input_le: hex::encode(bytes_le),
        output_le: expected_str,
    }
}

fn generate_g2_validation_raw(name: &str, bytes_be: &[u8; 192], expected_valid: bool) -> TestCase {
    let mut bytes_le = *bytes_be;
    // Convert BE (c1, c0) bytes to LE structure (c0, c1) bytes.
    swap_g2_c0_c1(&mut bytes_le);
    reverse_48_byte_chunks(&mut bytes_le);
    let expected_str = expected_valid.to_string();

    TestCase {
        op_type: "VALIDATION".to_string(),
        name: format!("G2_VALIDATE_{}", name),
        input_be: hex::encode(bytes_be),
        output_be: expected_str.clone(),
        input_le: hex::encode(bytes_le),
        output_le: expected_str,
    }
}

// Generate a G2 point where one coordinate component is >= P
fn generate_g2_validation_invalid_field() -> TestCase {
    // P = 0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab
    let modulus_p_be: [u8; 48] = hex!(
        "1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab"
    );

    // G2 Uncompressed is 192 bytes: (x_c1, x_c0, y_c1, y_c0) in BE (blstrs standard)
    // We will set x_c0 to P, which is invalid.
    let mut invalid_be = [0u8; 192];

    // Place P at offset 48 (x_c0)
    invalid_be[48..96].copy_from_slice(&modulus_p_be);

    // Set other bytes to valid values (e.g., from generator) to ensure only the field check fails
    let gen_be = g2_to_be(&G2Affine::generator());
    invalid_be[0..48].copy_from_slice(&gen_be[0..48]); // x_c1
    invalid_be[96..192].copy_from_slice(&gen_be[96..192]); // y

    // Prepare LE version
    let mut invalid_le = invalid_be;
    // Apply G2 LE transformation: Swap c0/c1 chunks, then reverse each chunk.
    swap_g2_c0_c1(&mut invalid_le);
    reverse_48_byte_chunks(&mut invalid_le);

    TestCase {
        op_type: "VALIDATION".to_string(),
        name: "G2_VALIDATE_FIELD_X_EQ_P_INVALID".to_string(),
        input_be: hex::encode(invalid_be),
        output_be: "false".to_string(), // Invalid = false
        input_le: hex::encode(invalid_le),
        output_le: "false".to_string(),
    }
}

// Replicating the logic from Agave to ensure test vectors match the spec
fn serialize_gt(gt: Gt, endianness: Endianness) -> Vec<u8> {
    let val: blst_fp12 = unsafe { std::mem::transmute(gt) };
    let mut out = Vec::with_capacity(576);

    // Collect all 12 coefficients in canonical memory order (Ascending Degree)
    //    Level 1 (Fp12): c0, c1
    //    Level 2 (Fp6):  c0, c1, c2
    //    Level 3 (Fp2):  c0, c1
    //    Order: [c0, c1, c2... c11]
    let mut coeffs = Vec::with_capacity(12);
    for fp6 in val.fp6.iter() {
        for fp2 in fp6.fp2.iter() {
            coeffs.push(fp2.fp[0]); // c0
            coeffs.push(fp2.fp[1]); // c1
        }
    }

    // Apply Endianness Logic
    let iter: Box<dyn Iterator<Item = _>> = match endianness {
        Endianness::LE => {
            // LE: Keep canonical order (Lowest Degree First: c0 -> c11)
            Box::new(coeffs.into_iter())
        }
        Endianness::BE => {
            // Instead of just swapping c0/c1 locally, we reverse the entire list.
            // This effectively puts the Highest Degree Coefficient first (c11 -> c0).
            // Matches: "swap the whole 48*12 bytes" logic.
            Box::new(coeffs.into_iter().rev())
        }
    };

    let mut ptr = out.as_mut_ptr();
    unsafe {
        for fp in iter {
            match endianness {
                // BE: Serialize individual element as Big Endian
                Endianness::BE => blst_bendian_from_fp(ptr, &fp),
                // LE: Serialize individual element as Little Endian
                Endianness::LE => blst_lendian_from_fp(ptr, &fp),
            }
            ptr = ptr.add(48);
        }
        out.set_len(576);
    }
    out
}

fn generate_pairing(name: &str, num_pairs: usize) -> TestCase {
    let mut rng = OsRng;
    let mut g1_vec = Vec::new();
    let mut g2_vec = Vec::new();

    // We start with Identity. In blstrs, Gt is additive.
    let mut result = Gt::identity();

    for _ in 0..num_pairs {
        let p = G1Affine::from(G1Projective::random(&mut rng));
        let q = G2Affine::from(G2Projective::random(&mut rng));

        // Accumulate result: res = res + pairing(p, q)
        // Note: blstrs implements pairing as additive in the Gt group trait
        result = result + blstrs::pairing(&p, &q);

        g1_vec.push(p);
        g2_vec.push(q);
    }

    // Construct Input Blobs
    // BE: [G1_1, G1_2...][G2_1, G2_2...]
    // LE: [G1_1, G1_2...][G2_1, G2_2...]
    let mut input_be = Vec::new();
    let mut input_le = Vec::new();

    // Append all G1s
    for p in &g1_vec {
        input_be.extend_from_slice(&g1_to_be(p));
        input_le.extend_from_slice(&g1_to_le(p));
    }
    // Append all G2s
    for q in &g2_vec {
        input_be.extend_from_slice(&g2_to_be(q));
        input_le.extend_from_slice(&g2_to_le(q));
    }

    TestCase {
        op_type: "PAIRING".to_string(),
        name: format!("PAIRING_{}", name),
        input_be: hex::encode(input_be),
        output_be: hex::encode(serialize_gt(result, Endianness::BE)),
        input_le: hex::encode(input_le),
        output_le: hex::encode(serialize_gt(result, Endianness::LE)),
    }
}

fn generate_pairing_bilinearity() -> TestCase {
    let mut rng = OsRng;

    // e(a*P, Q) == e(P, a*Q)
    // We will generate inputs for 2 pairs: [a*P, P] and [Q, -a*Q]
    // The sum should be identity: e(aP, Q) * e(P, -aQ) = e(P, Q)^a * e(P, Q)^-a = 1

    let p = G1Projective::random(&mut rng);
    let q = G2Projective::random(&mut rng);
    let s = Scalar::random(&mut rng);

    let p_times_s = G1Affine::from(p * s);
    let q_affine = G2Affine::from(q);

    let p_affine = G1Affine::from(p);
    let q_times_neg_s = G2Affine::from(q * (-s));

    let g1_vec = vec![p_times_s, p_affine];
    let g2_vec = vec![q_affine, q_times_neg_s];

    // Result should be identity
    let result = Gt::identity();

    let mut input_be = Vec::new();
    let mut input_le = Vec::new();

    for p in &g1_vec {
        input_be.extend_from_slice(&g1_to_be(p));
        input_le.extend_from_slice(&g1_to_le(p));
    }
    for q in &g2_vec {
        input_be.extend_from_slice(&g2_to_be(q));
        input_le.extend_from_slice(&g2_to_le(q));
    }

    TestCase {
        op_type: "PAIRING".to_string(),
        name: "PAIRING_BILINEARITY_IDENTITY".to_string(),
        input_be: hex::encode(input_be),
        output_be: hex::encode(serialize_gt(result, Endianness::BE)),
        input_le: hex::encode(input_le),
        output_le: hex::encode(serialize_gt(result, Endianness::LE)),
    }
}

fn main() {
    let mut rng = OsRng;
    // Use BTreeMap to keep output sorted alphabetically by operation type (filename)
    let mut test_cases: BTreeMap<String, Vec<TestCase>> = BTreeMap::new();

    // Setup necessary points and scalars
    let g1_gen = G1Projective::generator();
    let g2_gen = G2Projective::generator();
    let g1_inf = G1Projective::identity();
    let g2_inf = G2Projective::identity();
    let p1_rand1 = G1Projective::random(&mut rng);
    let p1_rand2 = G1Projective::random(&mut rng);
    let p2_rand1 = G2Projective::random(&mut rng);
    let p2_rand2 = G2Projective::random(&mut rng);
    let s_rand1 = Scalar::random(&mut rng);
    let s_zero = Scalar::ZERO;
    let s_one = Scalar::ONE;
    let s_minus_one = -Scalar::ONE;
    let s_worst_case = -Scalar::ONE;

    // Macro to easily add cases to the map
    macro_rules! add_case {
        ($case:expr) => {
            test_cases
                .entry($case.op_type.clone())
                .or_default()
                .push($case);
        };
    }

    // --- Generate all cases ---

    // // Addition/Subtraction
    // add_case!(generate_g1_op("ADD", "RANDOM", &p1_rand1, &p1_rand2));
    // add_case!(generate_g1_op("ADD", "WORST_CASE", &p1_rand1, &p1_rand2));
    // add_case!(generate_g1_op("ADD", "DOUBLING", &p1_rand1, &p1_rand1));
    // add_case!(generate_g1_op("ADD", "P_PLUS_INF", &p1_rand1, &g1_inf));
    // add_case!(generate_g1_op("ADD", "INF_PLUS_INF", &g1_inf, &g1_inf));
    //
    // add_case!(generate_g1_op("SUB", "RANDOM", &p1_rand1, &p1_rand2));
    // add_case!(generate_g1_op("SUB", "WORST_CASE", &p1_rand1, &p1_rand2));
    // add_case!(generate_g1_op("SUB", "P_MINUS_P", &p1_rand1, &p1_rand1));
    // add_case!(generate_g1_op("SUB", "INF_MINUS_P", &g1_inf, &p1_rand1));
    // add_case!(generate_g1_op("SUB", "P_MINUS_INF", &p1_rand1, &g1_inf));
    //
    // add_case!(generate_g2_op("ADD", "RANDOM", &p2_rand1, &p2_rand2));
    // add_case!(generate_g2_op("ADD", "WORST_CASE", &p2_rand1, &p2_rand2));
    // add_case!(generate_g2_op("ADD", "DOUBLING", &p2_rand1, &p2_rand1));
    // add_case!(generate_g2_op("ADD", "P_PLUS_INF", &p2_rand1, &g2_inf));
    // add_case!(generate_g2_op("ADD", "INF_PLUS_INF", &g2_inf, &g2_inf));
    //
    // add_case!(generate_g2_op("SUB", "RANDOM", &p2_rand1, &p2_rand2));
    // add_case!(generate_g2_op("SUB", "WORST_CASE", &p2_rand1, &p2_rand2));
    // add_case!(generate_g2_op("SUB", "P_MINUS_P", &p2_rand1, &p2_rand1));
    // add_case!(generate_g2_op("SUB", "INF_MINUS_P", &g2_inf, &p2_rand1));
    // add_case!(generate_g2_op("SUB", "P_MINUS_INF", &p2_rand1, &g2_inf));
    //
    // // Multiplication
    // add_case!(generate_g1_mul("RANDOM", &p1_rand1, &s_rand1));
    // add_case!(generate_g1_mul("WORST_CASE", &p1_rand1, &s_worst_case));
    // add_case!(generate_g1_mul("SCALAR_ZERO", &p1_rand1, &s_zero));
    // add_case!(generate_g1_mul("SCALAR_ONE", &p1_rand1, &s_one));
    // add_case!(generate_g1_mul("SCALAR_MINUS_ONE", &p1_rand1, &s_minus_one)); // New Case
    // add_case!(generate_g1_mul("POINT_INFINITY", &g1_inf, &s_rand1));
    //
    // add_case!(generate_g2_mul("RANDOM", &p2_rand1, &s_rand1));
    // add_case!(generate_g2_mul("WORST_CASE", &p2_rand1, &s_worst_case));
    // add_case!(generate_g2_mul("SCALAR_ZERO", &p2_rand1, &s_zero));
    // add_case!(generate_g2_mul("SCALAR_ONE", &p2_rand1, &s_one));
    // add_case!(generate_g2_mul("SCALAR_MINUS_ONE", &p2_rand1, &s_minus_one)); // New Case
    // add_case!(generate_g2_mul("POINT_INFINITY", &g2_inf, &s_rand1));
    //
    // // Decompression
    // add_case!(generate_g1_decompress("RANDOM", &p1_rand1));
    // add_case!(generate_g1_decompress("WORST_CASE", &p1_rand1));
    // add_case!(generate_g1_decompress("INFINITY", &g1_inf));
    // add_case!(generate_g1_decompress("GENERATOR", &g1_gen));
    // add_case!(generate_g1_decompress_invalid_curve("RANDOM", &p1_rand1));
    // add_case!(generate_g1_decompress_invalid_field());
    //
    // add_case!(generate_g2_decompress("RANDOM", &p2_rand1));
    // add_case!(generate_g2_decompress("WORST_CASE", &p2_rand1));
    // add_case!(generate_g2_decompress("INFINITY", &g2_inf));
    // add_case!(generate_g2_decompress("GENERATOR", &g2_gen));
    // add_case!(generate_g2_decompress_invalid_curve("RANDOM", &p2_rand1));
    // add_case!(generate_g2_decompress_invalid_field());
    //
    // // --- Validation ---
    //
    // add_case!(generate_g1_validation_raw(
    //     "RANDOM_VALID",
    //     &g1_to_be(&G1Affine::from(p1_rand1)),
    //     true
    // ));
    // add_case!(generate_g1_validation_raw(
    //     "WORST_CASE",
    //     &g1_to_be(&G1Affine::from(p1_rand1)),
    //     true
    // ));
    // add_case!(generate_g1_validation_raw(
    //     "INFINITY_VALID",
    //     &g1_to_be(&G1Affine::identity()),
    //     true
    // ));
    // add_case!(generate_g1_validation_raw(
    //     "GENERATOR_VALID",
    //     &g1_to_be(&G1Affine::generator()),
    //     true
    // ));
    //
    // add_case!(generate_g2_validation_raw(
    //     "RANDOM_VALID",
    //     &g2_to_be(&G2Affine::from(p2_rand1)),
    //     true
    // ));
    // add_case!(generate_g2_validation_raw(
    //     "WORST_CASE",
    //     &g2_to_be(&G2Affine::from(p2_rand1)),
    //     true
    // ));
    // add_case!(generate_g2_validation_raw(
    //     "INFINITY_VALID",
    //     &g2_to_be(&G2Affine::identity()),
    //     true
    // ));
    //
    // // 2. Invalid Cases
    // // G1 Not on Curve
    // let mut invalid_bytes_g1 = G1Affine::from(p1_rand1).to_uncompressed();
    // invalid_bytes_g1[95] = invalid_bytes_g1[95].wrapping_add(5);
    // if G1Affine::from_uncompressed(&invalid_bytes_g1)
    //     .is_none()
    //     .into()
    // {
    //     add_case!(generate_g1_validation_raw(
    //         "NOT_ON_CURVE_INVALID",
    //         &invalid_bytes_g1,
    //         false
    //     ));
    // }
    //
    // // G1 Field Invalid
    // let modulus_p_bytes_be: [u8; 48] = hex!(
    //     "1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab"
    // );
    // let mut invalid_field_g1 = [0u8; 96];
    // invalid_field_g1[0..48].copy_from_slice(&modulus_p_bytes_be);
    // invalid_field_g1[48..96].copy_from_slice(&G1Affine::generator().to_uncompressed()[48..96]);
    // add_case!(generate_g1_validation_raw(
    //     "FIELD_X_EQ_P_INVALID",
    //     &invalid_field_g1,
    //     false
    // ));
    //
    // // G2 Not on Curve
    // let mut invalid_bytes_g2 = G2Affine::from(p2_rand1).to_uncompressed();
    // invalid_bytes_g2[191] = invalid_bytes_g2[191].wrapping_add(5);
    // if G2Affine::from_uncompressed(&invalid_bytes_g2)
    //     .is_none()
    //     .into()
    // {
    //     add_case!(generate_g2_validation_raw(
    //         "NOT_ON_CURVE_INVALID",
    //         &invalid_bytes_g2,
    //         false
    //     ));
    // }
    //
    // // G2 Field Invalid (NEW)
    // add_case!(generate_g2_validation_invalid_field());

    // --- Pairing ---
    // 1. Identity (0 pairs) - Input is empty, Output is GT::Identity
    // We handle the empty input manually in the test case struct because loops wont run
    let id_gt = Gt::identity();
    add_case!(TestCase {
        op_type: "PAIRING".to_string(),
        name: "PAIRING_IDENTITY".to_string(),
        input_be: "".to_string(),
        output_be: hex::encode(serialize_gt(id_gt, Endianness::BE)),
        input_le: "".to_string(),
        output_le: hex::encode(serialize_gt(id_gt, Endianness::LE)),
    });

    // 2. Standard Cases
    add_case!(generate_pairing("ONE_PAIR", 1));
    add_case!(generate_pairing("WORST_CASE", 1));
    add_case!(generate_pairing("TWO_PAIRS", 2));
    add_case!(generate_pairing("THREE_PAIRS", 3));

    // 3. Bilinearity / Cancellation Check
    add_case!(generate_pairing_bilinearity());

    // --- Output Results Grouped by Operation Type (Filename) ---

    println!("// Generated Test Constants for solana-bls12-381-core");

    for (op_type, cases) in test_cases {
        let filename = op_type.to_lowercase();
        println!("\n// ========================================");
        println!("// Vectors for src/test_vectors/{}.rs", filename);
        println!("// ========================================\n");

        for case in cases {
            println!("\n// Test Case: {}", case.name);
            println!(
                "pub const INPUT_BE_{}: &[u8] = &hex!(\"{}\");",
                case.name, case.input_be
            );
            println!(
                "pub const INPUT_LE_{}: &[u8] = &hex!(\"{}\");",
                case.name, case.input_le
            );

            // Handle boolean output for validation, hex otherwise
            if op_type == "VALIDATION" {
                // For validation, the output string is the boolean expectation
                // Ensure BE and LE expectations match before printing
                assert_eq!(case.output_be, case.output_le);
                println!(
                    "pub const EXPECTED_{}: bool = {};",
                    case.name, case.output_be
                );
            } else if case.output_be == "INVALID" {
            } else {
                println!(
                    "pub const OUTPUT_BE_{}: &[u8] = &hex!(\"{}\");",
                    case.name, case.output_be
                );
                println!(
                    "pub const OUTPUT_LE_{}: &[u8] = &hex!(\"{}\");",
                    case.name, case.output_le
                );
            }
        }
    }
}
