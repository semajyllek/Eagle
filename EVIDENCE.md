# Batched LocateAnything — GPU evidence

Branch: `batched-vectorized-decode` (commit `a3fd74e`)
Hardware: A100-80GB, bf16. Generated 2026-06-17 22:28.

## Detection: batched speedup + parity

B=1 baseline is `worker.predict` (the original sequential path); batched is
`worker.predict_batch` on distinct real images — the actual shipped API.

| batch_size | dataset | samples | B=1 (s) | batched (s) | speedup | parity | median coord diff (px) |
|---|---|---|---|---|---|---|---|
| 8 | COCO | 50 | 73.8 | 26.7 | 2.77x | 45/50 | 0.0 |
| 8 | LVIS | 50 | 68.6 | 26.6 | 2.58x | 44/50 | 0.0 |
| 16 | COCO | 50 | 74.0 | 19.3 | 3.83x | 45/50 | 0.0 |
| 16 | LVIS | 50 | 69.4 | 20.6 | 3.36x | 45/50 | 0.0 |

### Throughput sweep: sequential vs batched

N distinct images, sequential = N x `predict_batch([img],[q])`, batched = one `predict_batch(imgs, qs)`.

| batch_size | speedup |
|---|---|
| 1 | 1.12x |
| 2 | 1.28x |
| 4 | 1.50x |
| 8 | 1.65x |
| 16 | 1.74x |

## One image, many queries (RefCOCOg)

| batch_size | dataset | B=1 (s) | batched (s) | speedup |
|---|---|---|---|---|
| 8 | RefCOCOg_val | 127.7 | 64.5 | 1.98x |
| 16 | RefCOCOg_val | 128.2 | 64.5 | 1.99x |

## Decode-loop bookkeeping profile

Per-step timing at B=16/32. Step1 (input assembly) and Step5 (cache compaction)
are vectorized tensor ops with zero host-device syncs; Step3 is the model forward.

| batch_size | mode | total (s) | step1_assemble (s) | step3_forward (s) | step5_compact (s) |
|---|---|---|---|---|---|
| 16 | hybrid | 8.743 | 0.0091 | 1.3046 | 0.0551 |
| 16 | fast | 8.557 | 0.0060 | 1.1471 | 0.0368 |
| 32 | hybrid | 17.082 | 0.0110 | 2.2381 | 0.0991 |
| 32 | fast | 16.844 | 0.0074 | 2.0908 | 0.0663 |
| 16 | hybrid | 8.780 | 0.0092 | 1.3306 | 0.0550 |
| 16 | fast | 8.567 | 0.0065 | 1.1608 | 0.0368 |
| 32 | hybrid | 17.104 | 0.0117 | 2.2399 | 0.0992 |
| 32 | fast | 16.863 | 0.0082 | 2.0897 | 0.0664 |

## torch.compile feasibility probe

One steady-state decode step, eager vs `torch.compile(mode="reduce-overhead")`.
The vectorized mask dispatch eliminates per-row `.item()` host syncs, removing
`torch._dynamo` graph breaks and enabling efficient CUDA graph capture.

| batch_size | A | W | Ckv | eager (s) | compiled (s) | speedup |
|---|---|---|---|---|---|---|
| 16 | 16 | 9 | 1170 | 0.0522 | 0.0242 | 2.154x |
| 32 | 32 | 9 | 1170 | 0.0669 | 0.0412 | 1.622x |
| 16 | 16 | 9 | 1170 | 0.0529 | 0.0244 | 2.164x |
| 32 | 32 | 9 | 1170 | 0.0669 | 0.0415 | 1.614x |

### Real decode-loop torch.compile A/B

Full `predict_batch` with the language model `torch.compile`d under genuine
ragged shapes (Ckv grows every step). Inputs are real decode-bound eval images
so wall_x reflects the decode-dominated regime where compile pays off.

| bs | steps | config | eager_wall (s) | compiled_wall (s) | wall_x | fwd_x | recompiles | parity (struct) |
|---|---|---|---|---|---|---|---|---|
| 8 | 63 | dec | 4.687 | 2.880 | 1.627x | 2.087x | 0 | 7/8 |
| 8 | 63 | dec+vit | 4.687 | 2.257 | 2.076x | 2.698x | 0 | 7/8 |
| 16 | 55 | dec | 5.286 | 3.844 | 1.375x | 1.833x | 0 | 15/16 |
| 16 | 55 | dec+vit | 5.286 | 4.146 | 1.275x | 1.481x | 0 | 14/16 |

### End-to-end speedup ladder (stacked_eval)

The real production speedup: 16 eval images through the multi-batch shipped path
at eval-length tokens (2048), vs the pristine original model at B=1.

| bs | pristine B=1 (s) | batched eager (s) | batch_x | config | compiled (s) | compile_x | total_x |
|---|---|---|---|---|---|---|---|
| 8 | 25.22 | 7.656 | 3.294x | dec | 4.580 | 1.671x | 5.506x |
| 8 | 25.22 | 7.656 | 3.294x | dec+vit | 3.997 | 1.915x | 6.310x |
| 16 | 25.29 | 5.376 | 4.703x | dec | 3.794 | 1.417x | 6.665x |
| 16 | 25.29 | 5.376 | 4.703x | dec+vit | 4.170 | 1.289x | 6.064x |

### Detection quality under torch.compile (compile_ap)

100 real eval samples through eager vs compiled, scored with the repo metric.
mAP(compiled) vs mAP(eager) — the definitive gate for whether compile costs real detections.

| dataset | bs | config | AP (eager) | AP (compiled) | AP50 (eager) | AP50 (compiled) | parity (exact) | parity (struct) |
|---|---|---|---|---|---|---|---|---|
| COCO | 8 | dec | 0.7694 | 0.7727 | 0.8635 | 0.8695 | 30/100 | 90/100 |
| COCO | 8 | dec+vit | 0.7694 | 0.7718 | 0.8635 | 0.8692 | 29/100 | 85/100 |
| LVIS | 8 | dec | 0.8780 | 0.8871 | 0.9546 | 0.9547 | 45/100 | 87/100 |
| LVIS | 8 | dec+vit | 0.8780 | 0.8820 | 0.9546 | 0.9544 | 34/100 | 85/100 |
| COCO | 16 | dec | 0.7732 | 0.7714 | 0.8705 | 0.8623 | 32/100 | 92/100 |
| COCO | 16 | dec+vit | 0.7732 | 0.7835 | 0.8705 | 0.8769 | 31/100 | 86/100 |
| LVIS | 16 | dec | 0.8752 | 0.8849 | 0.9540 | 0.9511 | 48/100 | 86/100 |
| LVIS | 16 | dec+vit | 0.8752 | 0.8810 | 0.9540 | 0.9515 | 29/100 | 78/100 |

