# STOKE — Parallel Tempering Fork

A research fork of [STOKE](https://github.com/StanfordPL/stoke), the stochastic superoptimizer for x86-64 assembly. This fork replaces STOKE's simulated annealing (SA) search with **parallel tempering** (PT), a Markov chain Monte Carlo technique that runs multiple SA chains at different temperatures and periodically swaps configurations between them.

This work was done as part of undergraduate research at the George Mason University Experimental Geometry Lab, exploring whether PT's ability to escape local minima improves optimizer convergence on compiler optimization benchmarks.

---

## Background

### What STOKE does

STOKE treats compiler optimization as a stochastic search problem. Given a target x86-64 function, it proposes random mutations (opcode swaps, operand changes, instruction insertions/deletions) and accepts or rejects them using a cost function that combines:

- **Correctness cost**: how often the rewrite produces wrong outputs on test inputs
- **Performance cost**: estimated latency relative to the target

The search accepts worse solutions with a probability governed by the Metropolis–Hastings criterion, allowing it to climb out of local minima. Over millions of iterations, it can find rewrites that are both correct and faster than compiler-generated code — including tricks like replacing a bit-counting loop with a single `popcnt` instruction.

### The local minima problem

SA with a fixed temperature has a known failure mode: it either runs too hot (accepts bad moves indiscriminately, can't converge) or too cold (gets trapped in a local minimum and stops exploring). STOKE's cycle-restart strategy partially compensates by periodically resetting from scratch, but restarts discard progress.

### Parallel Tempering

PT runs `N` independent SA chains simultaneously, each at a different temperature (parameterized by `beta`, the inverse temperature). Periodically, neighboring replicas attempt to **exchange configurations** based on an acceptance probability derived from the energy difference:

```
P(swap) = exp(min(0, (β_i - β_{i+1}) * (cost_{i+1} - cost_i)))
```

The hot replicas (low beta) explore broadly; the cold replicas (high beta) refine. When a hot replica finds a promising basin, the exchange mechanism propagates it to the colder chains for refinement. This is a standard technique in computational chemistry and Bayesian inference, but has not (to our knowledge) been applied to superoptimizer search.

**Hypothesis**: PT should reduce the number of restarts needed and find lower-cost rewrites within the same iteration budget by maintaining exploration pressure across the temperature ladder.

> **Known limitation**: STOKE's search does not handle functions containing loops. This is a pre-existing constraint in the original optimizer, not a limitation of this fork.

---

## What this fork implements

### Core changes

| File | Change |
|------|--------|
| `src/search/search.cc` | Added `run_parallel_tempering()` — the PT main loop with per-replica SA steps, replica exchange, stats tracking, and callbacks |
| `src/search/search.h` | Declared `run_parallel_tempering()` |
| `tools/apps/stoke_search.cc` | Added `--parallel_tempering`, `--num_replicas`, `--pt_beta_min`, `--pt_beta_max` CLI flags; dispatch to PT path |

### New CLI flags

```
--parallel_tempering          Use PT instead of SA
--num_replicas <int>          Number of temperature levels (default: 8)
--pt_beta_min <double>        Lowest beta / highest temperature (default: 0.01)
--pt_beta_max <double>        Highest beta / lowest temperature (default: 1.0)
```

The beta schedule is geometric: `beta_i = beta_min * (beta_max / beta_min)^(i / (N-1))`, evenly spacing replicas on a log scale.

---

## Building

Requires Docker (tested with Docker 25). The build uses the official `stanfordpl/stoke` image as a base.

```bash
docker build --platform linux/amd64 -f Dockerfile.pt -t stoke_pt .
```

This compiles `stoke_search`, `stoke_extract`, `stoke_testcase`, and `stoke_replace` from source with the PT changes applied.

> **Note**: The build targets `linux/amd64`. On Apple Silicon, Docker runs it under emulation — the build works, but the runtime sandbox (which executes x86-64 candidate programs) may segfault partway through a run due to qemu limitations. On native x86-64 Linux, the run completes cleanly.

---

## Running the demo

The tutorial example optimizes a naive `popcnt` implementation (counting set bits in a 64-bit integer via a loop) and attempts to find a shorter, faster equivalent.

### Setup

```bash
docker run --rm --platform linux/amd64 -it stoke_pt bash

# Inside the container:
cd /home/stoke/stoke/examples/tutorial
g++ -std=c++11 -fno-inline -O3 main.cc -o a.out
stoke_extract --config extract.conf
mkdir -p tcs
stoke_testcase --target bins/_Z6popcntm.s -o tcs/_Z6popcntm --max_testcases 1024
```

This produces `bins/_Z6popcntm.s` (the disassembled target) and `tcs/_Z6popcntm` (1024 test inputs).

### Simulated Annealing (baseline)

```bash
stoke_search \
  --out result_sa.s \
  --target bins/_Z6popcntm.s \
  --def_in '{ %rdi %rax }' --live_out '{ %rax }' \
  --testcases tcs/_Z6popcntm \
  --training_set '{ 0 ... 7 }' --test_set '{ 8 ... 1023 }' \
  --distance hamming --misalign_penalty 1 --reduction sum --sig_penalty 9999 \
  --global_swap_mass 1 --instruction_mass 1 --local_swap_mass 1 \
  --opcode_mass 1 --operand_mass 1 --rotate_mass 1 --opcode_width_mass 1 \
  --initial_instruction_number 5 --statistics_interval 100000 \
  --timeout_iterations 1000000 --cycle_timeout 200000 \
  --strategy hold_out --cost 'correctness + latency' --postprocessing simple \
  --cpu_flags '{ popcnt }' --beta 1.0
```

### Parallel Tempering

```bash
stoke_search \
  --out result_pt.s \
  --target bins/_Z6popcntm.s \
  --def_in '{ %rdi %rax }' --live_out '{ %rax }' \
  --testcases tcs/_Z6popcntm \
  --training_set '{ 0 ... 7 }' --test_set '{ 8 ... 1023 }' \
  --distance hamming --misalign_penalty 1 --reduction sum --sig_penalty 9999 \
  --global_swap_mass 1 --instruction_mass 1 --local_swap_mass 1 \
  --opcode_mass 1 --operand_mass 1 --rotate_mass 1 --opcode_width_mass 1 \
  --initial_instruction_number 5 --statistics_interval 100000 \
  --timeout_iterations 1000000 --cycle_timeout 1000000 \
  --strategy hold_out --cost 'correctness + latency' --postprocessing simple \
  --cpu_flags '{ popcnt }' \
  --parallel_tempering --num_replicas 4 --pt_beta_min 0.05 --pt_beta_max 1.0
```

### Sample output (PT, 300k iterations)

```
Running parallel tempering (4 replicas, beta: 0.05 → 1, timeout: 300000 iterations):

Progress Update:
Lowest Cost Discovered (233)     Lowest Known Correct Cost (8)
._Z6popcntm:                     ._Z6popcntm:
cbtw                             testq %rdi, %rdi
retq                             je .L_40062f
                                 xorl %eax, %eax
                                 .L_400620:
                                 movl %edi, %edx
                                 andl $0x1, %edx
                                 addl %edx, %eax
                                 shrq $0x1, %rdi
                                 jne .L_400620
                                 cltq
                                 retq
                                 .L_40062f:
                                 xorl %eax, %eax
                                 retq

[... 18 progress updates ...]

Progress Update:
Lowest Cost Discovered (21)      Lowest Known Correct Cost (8)
```

After 300k iterations the best-found rewrite has cost 21 (the locked-in correct rewrite sits at cost 8, the original). In a full run (16M iterations), STOKE typically discovers that `popcnt %rdi, %rax` — a single hardware instruction — is both correct and optimal.

---

## Current state

| Component | Status |
|-----------|--------|
| `run_parallel_tempering()` core loop | Working |
| Replica exchange (Metropolis criterion) | Working |
| Per-move statistics aggregated across replicas | Working |
| Progress and statistics callbacks | Working |
| CLI flags (`--parallel_tempering`, `--num_replicas`, etc.) | Working |
| Beta schedule (geometric) | Working |
| Cycle-restart strategy for PT | Not implemented — PT runs as a single long pass |
| Formal benchmarking vs. SA | Not done — requires native x86 hardware |

---

## What's left / future work

1. **Benchmark**: Run controlled comparisons of PT vs. SA on the STOKE benchmark suite (requires native x86). The core question is whether PT's replica exchange reduces restarts needed to find a verified rewrite.

2. **Adaptive temperature scheduling**: The current geometric schedule is fixed. Online methods that tune betas based on observed swap rates would be more principled.

3. **Parallel execution**: Right now the N replicas run sequentially in a single thread. Running them in parallel threads with synchronized swap steps is the natural next step and would make PT genuinely faster per wall-clock second.

4. **Loop support**: A fundamental limitation of the underlying STOKE framework — not specific to this fork.

---

## References

- [STOKE: Stochastic Superoptimization](https://cs.stanford.edu/people/eschkufz/docs/asplos_13.pdf) — Schkufza, Sharma, Aiken (ASPLOS 2013)
- [Parallel Tempering](https://en.wikipedia.org/wiki/Parallel_tempering) — Geyer (1991), Hukushima & Nemoto (1996)
- [Original STOKE repository](https://github.com/StanfordPL/stoke)
