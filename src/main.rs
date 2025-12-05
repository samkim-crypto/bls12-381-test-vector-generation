use blstrs::{G1Affine, G1Projective, G2Affine, G2Projective, Scalar};
use group::{Curve, Group, ff::Field, prime::PrimeCurveAffine};
use hex_literal::hex;
use rand::rngs::OsRng;
use std::collections::BTreeMap;

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
    let s_bytes = s.to_bytes_le(); // Scalars are LE for syscall input

    TestCase {
        op_type: "MULTIPLICATION".to_string(),
        name: format!("G1_MUL_{}", name),
        input_be: hex::encode([s_bytes.as_ref(), &g1_to_be(&p_a)].concat()),
        output_be: hex::encode(g1_to_be(&res_a).to_vec()),
        input_le: hex::encode([s_bytes.as_ref(), &g1_to_le(&p_a)].concat()),
        output_le: hex::encode(g1_to_le(&res_a).to_vec()),
    }
}

fn generate_g2_mul(name: &str, p: &G2Projective, s: &Scalar) -> TestCase {
    let result = p * s;
    let (p_a, res_a) = (G2Affine::from(p), G2Affine::from(result));
    let s_bytes = s.to_bytes_le();

    TestCase {
        op_type: "MULTIPLICATION".to_string(),
        name: format!("G2_MUL_{}", name),
        input_be: hex::encode([s_bytes.as_ref(), &g2_to_be(&p_a)].concat()),
        output_be: hex::encode(g2_to_be(&res_a).to_vec()),
        input_le: hex::encode([s_bytes.as_ref(), &g2_to_le(&p_a)].concat()),
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
    let s_one = Scalar::ZERO;

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

    // Addition/Subtraction
    add_case!(generate_g1_op("ADD", "RANDOM", &p1_rand1, &p1_rand2));
    add_case!(generate_g1_op("ADD", "DOUBLING", &p1_rand1, &p1_rand1));
    add_case!(generate_g1_op("ADD", "P_PLUS_INF", &p1_rand1, &g1_inf));
    add_case!(generate_g1_op("ADD", "INF_PLUS_INF", &g1_inf, &g1_inf));

    add_case!(generate_g1_op("SUB", "RANDOM", &p1_rand1, &p1_rand2));
    add_case!(generate_g1_op("SUB", "P_MINUS_P", &p1_rand1, &p1_rand1));
    add_case!(generate_g1_op("SUB", "INF_MINUS_P", &g1_inf, &p1_rand1));
    add_case!(generate_g1_op("SUB", "P_MINUS_INF", &p1_rand1, &g1_inf));

    add_case!(generate_g2_op("ADD", "RANDOM", &p2_rand1, &p2_rand2));
    add_case!(generate_g2_op("ADD", "DOUBLING", &p2_rand1, &p2_rand1));
    add_case!(generate_g2_op("ADD", "P_PLUS_INF", &p2_rand1, &g2_inf));
    add_case!(generate_g2_op("ADD", "INF_PLUS_INF", &g2_inf, &g2_inf));

    add_case!(generate_g2_op("SUB", "RANDOM", &p2_rand1, &p2_rand2));
    add_case!(generate_g2_op("SUB", "P_MINUS_P", &p2_rand1, &p2_rand1));
    add_case!(generate_g2_op("SUB", "INF_MINUS_P", &g2_inf, &p2_rand1));
    add_case!(generate_g2_op("SUB", "P_MINUS_INF", &p2_rand1, &g2_inf));

    // Multiplication
    add_case!(generate_g1_mul("RANDOM", &p1_rand1, &s_rand1));
    add_case!(generate_g1_mul("SCALAR_ZERO", &p1_rand1, &s_zero));
    add_case!(generate_g1_mul("SCALAR_ONE", &p1_rand1, &s_one));
    add_case!(generate_g1_mul("POINT_INFINITY", &g1_inf, &s_rand1));

    add_case!(generate_g2_mul("RANDOM", &p2_rand1, &s_rand1));
    add_case!(generate_g2_mul("SCALAR_ZERO", &p2_rand1, &s_zero));
    add_case!(generate_g2_mul("SCALAR_ONE", &p2_rand1, &s_one));
    add_case!(generate_g2_mul("POINT_INFINITY", &g2_inf, &s_rand1));

    // Decompression
    add_case!(generate_g1_decompress("RANDOM", &p1_rand1));
    add_case!(generate_g1_decompress("INFINITY", &g1_inf));
    add_case!(generate_g1_decompress("GENERATOR", &g1_gen));
    add_case!(generate_g2_decompress("RANDOM", &p2_rand1));
    add_case!(generate_g2_decompress("INFINITY", &g2_inf));
    add_case!(generate_g2_decompress("GENERATOR", &g2_gen));

    // --- Validation ---

    // 1. Valid Points
    add_case!(generate_g1_validation_raw(
        "RANDOM_VALID",
        &g1_to_be(&G1Affine::from(p1_rand1)),
        true
    ));
    add_case!(generate_g1_validation_raw(
        "INFINITY_VALID",
        &g1_to_be(&G1Affine::identity()),
        true
    ));
    add_case!(generate_g1_validation_raw(
        "GENERATOR_VALID",
        &g1_to_be(&G1Affine::generator()),
        true
    ));
    add_case!(generate_g2_validation_raw(
        "RANDOM_VALID",
        &g2_to_be(&G2Affine::from(p2_rand1)),
        true
    ));
    add_case!(generate_g2_validation_raw(
        "INFINITY_VALID",
        &g2_to_be(&G2Affine::identity()),
        true
    ));

    // 2. Invalid: Not on curve (G1)
    // Take a valid point and slightly modify the Y coordinate.
    let mut invalid_bytes_g1 = G1Affine::from(p1_rand1).to_uncompressed();
    invalid_bytes_g1[95] = invalid_bytes_g1[95].wrapping_add(5); // Modify LSB of Y coord (BE)
    // Ensure the modification resulted in an invalid point (required check for blstrs)
    if G1Affine::from_uncompressed(&invalid_bytes_g1)
        .is_none()
        .into()
    {
        add_case!(generate_g1_validation_raw(
            "NOT_ON_CURVE_INVALID",
            &invalid_bytes_g1,
            false
        ));
    }

    // 3. Invalid: Field Element >= P
    // P = 0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab
    let modulus_p_bytes_be: [u8; 48] = hex!(
        "1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab"
    );
    let mut invalid_field_g1 = [0u8; 96];
    invalid_field_g1[0..48].copy_from_slice(&modulus_p_bytes_be); // X = P
    // Y coordinate doesn't matter if X is invalid, but set it anyway
    invalid_field_g1[48..96].copy_from_slice(&G1Affine::generator().to_uncompressed()[48..96]);
    // blstrs should reject this because X is not a valid field element
    if G1Affine::from_uncompressed(&invalid_field_g1)
        .is_none()
        .into()
    {
        add_case!(generate_g1_validation_raw(
            "FIELD_X_EQ_P_INVALID",
            &invalid_field_g1,
            false
        ));
    }

    // 4. Invalid: Not on curve (G2)
    let mut invalid_bytes_g2 = G2Affine::from(p2_rand1).to_uncompressed();
    invalid_bytes_g2[191] = invalid_bytes_g2[191].wrapping_add(5); // Modify LSB of Y_c0 (BE)
    if G2Affine::from_uncompressed(&invalid_bytes_g2)
        .is_none()
        .into()
    {
        add_case!(generate_g2_validation_raw(
            "NOT_ON_CURVE_INVALID",
            &invalid_bytes_g2,
            false
        ));
    }

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
