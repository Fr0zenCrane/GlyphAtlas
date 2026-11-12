# Training and checkpoints

For the published source-exclusive corpus, use `configs/train_glyphatlas_1p5m.json` (or `train_glyphatlas_1p5m_random_atlas.json`). These preserve the corresponding dev numerical recipe and require the exact full published dataset, including repository, commit, catalog hash and per-source counts. See [DATA.md](DATA.md).

The training implementation is the dev full-transformer runner, not LoRA training. Default inputs are fixed 512 × 512. The batch consists of 128 unique parent images across ranks and accumulation steps; per-rank microbatch is one. World size must be at least eight and divide 128.

| Setting | Default |
| --- | --- |
| Initialization | Fresh official FLUX.1-Fill-dev; Atlas initialized from the native projection |
| Atlas height | Fixed 64; random-height config is an alternative |
| Trainable parameters | 11,902,391,360 backbone + 593,536 Atlas |
| Optimizer | AdamW, betas 0.9/0.999, epsilon 1e-8, weight decay 0.01 |
| LR | 2e-5 to 2e-6, cosine schedule, 500-update warmup |
| Precision | FP32 master parameters/Adam states and gradient reduction; BF16 compute/VAE |
| Objective | Flow/mask weights 1; PP-OCRv3 feature MSE coefficient 0.25 × 2t; ODM and CTC zero |
| Timestep | Native sampling, no training shift |
| Gradient clipping | Global norm 0.5 after accumulation and replica averaging |
| Default save interval | 2,500 optimizer updates, plus final checkpoint |

These are the inherited dev defaults, not a claim that every paper experiment uses this exact configuration. Changing the guarded objective or global batch requires an explicit implementation change and numerical validation.

The upstream constructor still loads ODM during initialization, then removes it before training. The training asset downloader therefore retains its pinned weight file for compatibility even though ODM contributes no loss.

## Multi-node launch

Use a shared checkout, dataset, model directory and output path. Run the same command on each node with its own zero-based rank:

```bash
export ATLAS_NNODES=4 ATLAS_NPROC=8
export ATLAS_NODE_RANK=0              # Set 0, 1, 2 or 3 on the respective node.
export MASTER_ADDR=training-node-0 MASTER_PORT=29500
bash tools/launch.sh --config configs/c4_node_shard_bs128.json \
  --dataset data/custom --output runs/multinode --steps 12500
```

`auto` uses node-local FSDP when there are multiple nodes, with corresponding FP32 gradient shards averaged across nodes once per global update. It retains `/accumulation` and replica averaging before clipping. Code/config/data/device agreement is checked across ranks. `tools/slurm_train.example.sh` is a template requiring your cluster resources; it defaults to zero-update validation.

`--validate-only` performs a real accumulated backward pass with zero optimizer updates. `--validation-local-size` emulates logical node groups on one physical host; it cannot establish inter-node performance or network correctness.

## Exact resume

```bash
bash tools/launch.sh --config configs/c4_bs128.json \
  --dataset data/custom --output runs/train-resumed --steps 12500 \
  --resume runs/train/checkpoint-002500
```

Resume requires a full checkpoint produced by this release, with the same code, data identity, config, topology and total LR horizon. Always use a new output directory. Do not change `--steps` when resuming. To run a bounded segment without changing the LR horizon, use `--stop-after ABSOLUTE_STEP`.

```text
checkpoint-002500/
  transformer/                 BF16 inference export
  distributed/                 Full model state
  optimizer-shard*.pt          Node-sharded optimizer state, where applicable
  rng-rank*.pt
  aux-rank*.pt                 Atlas/cache state per rank
  training_state.json
  resume_aux_complete.json
```

A BF16 inference export alone cannot resume AdamW. Private dev checkpoints have a different runtime-code hash and are not advertised as exact-resume compatible with this reorganized tree, even though complete transformer exports remain inference-compatible. Full checkpoints are large; the runner requires 250 GB free before each save and does not automatically prune them.

## Resolution schedules

Use `configs/c4_random_resolution_bs128.json` to sample Atlas heights 48/64/80 with the inherited allocator. Fixed-64 remains the default.

`configs/legacy/continuous_resolution_schedule.json` preserves the dev continuous fixed-to-random resolution schedule for regression tests and historical reproducibility. It is not the recommended default or a separate mandatory training stage. Internal `stage` identifiers are retained only for checkpoint compatibility. Its pinned 32-H200 topology and horizon are historical constraints; do not use that config for a different deployment.
