# GPU DNA Aligner (WIP / Research Prototype)

> ⚠️ **Status: Early Experimental Prototype.**  
> This project is in active early development. Many features are experimental, partially implemented, or subject to redesign.

An experimental project exploring how much dynamic programming throughput can be squeezed out of consumer and legacy hardware (CPU SIMD + portable GPU compute) for DNA sequence alignment.

The goal is to test whether low-end or older hardware (like 10-year-old laptops or integrated GPUs) can handle heavy genomic workloads (Smith-Waterman) using Rust and modern vendor-agnostic compute APIs (WebGPU/WGSL).

---

## Early MVP Benchmark

A preliminary proof-of-concept run to validate whether the approach is viable on weak hardware:

* **Hardware:** Intel HD Graphics 5600 / NVIDIA GTX 960M (Mobile, 2015)
* **Dataset:** *E. coli* (~4.6M bp) vs. 21.4M reads (Illumina FASTQ, 5.7 GB)
* **MVP Wall Time:** `~300 s` (rough end-to-end pipeline run)
* **Throughput:** `~71,000` alignments/sec

*Note: These are early sanity-check numbers from a minimal prototype pipeline; edge cases and proper error handling are still being worked out.*

---

## Planned Architecture & Core Ideas

The goal is to build an asynchronous streaming engine that maximizes hardware saturation:

* **2-Bit Sequence Packing:** Pack bases (`A, C, G, T`) into 2-bit representations to reduce PCIe memory bus transfers.
* **Heterogeneous Load Partitioning:** 
  - Exact matches filtered on CPU via SIMD K-mer hashing (O(1)).
  - Complex dynamic programming alignment offloaded to the GPU ($O(N^2)$).
* **Branchless WGSL Kernel:** Structure-of-Arrays (SoA) layout with branchless selection (`select`, `max`) to minimize warp divergence on weak GPUs.
* **Bounded-Memory Streaming:** Multi-stage pipeline using Tokio channels to process arbitrarily large FASTQ files with minimal RAM footprint.

---

## Current Status & Roadmap

- [x] Basic end-to-end MVP (FASTQ stream → GPU compute kernel)
- [x] 2-bit nucleotide encoding prototype
- [ ] Robust error handling and edge-case testing
- [ ] SIMD K-mer pre-filtering on CPU (`fearless_simd`)
- [ ] Optimized branchless WGSL traceback kernel
- [ ] Proper SAM/BAM file emitter
- [ ] Native CUDA backend option via `cuda-oxide`
- [ ] Detailed profiling and PCIe bottleneck analysis

## Stack

* **Language:** Rust
* **GPU Compute:** `wgpu` (WGSL)
* **Concurrency:** `tokio`, `rayon`
* **SIMD (Planned):** `fearless_simd`

## License
MIT
