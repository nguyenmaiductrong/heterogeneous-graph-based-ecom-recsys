# Config và log Training
## Model v2
### Config
data:
  data_dir: /content/data/
  node_counts:
    brand: 1919
    category: 14
    product: 29892
    user: 203063
  struct_dir: /content/data/node_mappings

model:
  dropout: 0.3
  embed_dim: 128
  n_layers: 2
  rank: 32
  use_grad_checkpoint: false
  n_intents: 32

sampler:
  hop1_budget: 8
  hop2_budget: 4
  hop1_sample_replace: true

# Total loss:
#   L = L_BPR + lambda_cl*L_CL + lambda_conv*L_conv + lambda_mono*L_mono + lambda_wd*||theta||^2
# Per-behavior BPR weights: w_b = clip((N_purchase / N_b) ** alpha, w_min, 1.0)
loss:
  lambda_cl: 0.1 # contrastive (HierarchicalMBCL)
  lambda_conv: 0.0 # funnel prior s_view < s_cart < s_purchase (0 = off)
  lambda_mono: 0.0 # monotonic decay prior lam_view >= lam_cart >= lam_purchase (0 = off)
  funnel_margin: 0.1
  alpha: 0.5 # exponent in (N_p / N_b) ** alpha; alpha in [0.25, 0.5]
  w_min: 0.05 # floor for view/cart weight

hierarchy_cl:
  enabled: true
  tau: 0.1
  hard_k: 32
  min_pair_overlap: 4
  pair_weights: null # null = auto progressive (view -> cart -> purchase)

training:
  amp: true
  use_bf16: true
  batch_size: 8192
  device: cuda
  epochs: 30
  eval_batch_size: 8192
  eval_every: 1
  eval_subsample: 20000 # 0 = full eval each cycle
  eval_seed: 42
  l2_lambda: 1.0e-05 # lambda_wd
  lr: 1.0e-03
  min_lr: 1.0e-06
  warmup_epochs: 2
  max_grad_norm: 1.0
  cl_every_k: 1 # 1 = run CL each step; raise to 5 for ~5x cheaper CL
  max_view_triplets: 3000000
  num_neg: 16
  num_workers: 8
  patience: 8
  save_dir: checkpoints-v2
  weight_decay: 1.0e-02
  pin_memory: true
  persistent_workers: true
  prefetch_factor: 4

evaluation:
  full_ranking: true
  primary_metric: "NDCG@20"
  ks: [10, 20, 50]
  metrics: ["HR@10", "HR@20", "HR@50", "NDCG@10", "NDCG@20", "NDCG@50"]

wandb:
  artifact_name: bpatmp-checkpoint-v2
  enabled: true
  entity: nguyenmaiductrong37-h-c-vi-n-c-ng-ngh-b-u-ch-nh-vi-n-th-ng
  project: bpatmp-recsys
  run_name: bpatmp-v2
  save_every: 1

# A100 optimizations
a100:
  allow_tf32: true
  cudnn_benchmark: true
  use_fused_adamw: true
  compile_model: false
  empty_cache_freq: 0
### Log
2026-05-08 16:06:08,094 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 0 -> vv6 confirmed (state=COMMITTED, 368.2 MB). Safe to close Colab.
2026-05-08 23:06:08
2026-05-08 16:06:08,138 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v2/best.pt
2026-05-08 23:06:08
2026-05-08 16:06:08,138 - src.training.trainer - INFO - Epoch 000 | train/loss=2.6332 | train/cl_loss=7.6744 | train/skipped_batches=0.0000 | train/lr=0.0005 | HR@10=0.0043 | NDCG@10=0.0014 | HR@20=0.0046 | NDCG@20=0.0014 | HR@50=0.0060 | NDCG@50=0.0015  <- best
2026-05-08 23:06:08
epochs:   3%|▎         | 1/30 [07:26<1:55:45, 239.51s/it, loss=2.4220, NDCG_20=0.0085, best_primary=0.0085]2026-05-08 16:09:36,447 - src.training.checkpoint_manager - INFO - Saved local: checkpoints-v2/epoch_001.pt (368.2 MB)
2026-05-08 23:06:08
2026-05-08 16:09:37,785 - src.training.checkpoint_manager - INFO - Artifact enqueued: bpatmp-checkpoint-v2 epoch-001
2026-05-08 23:10:03
2026-05-08 16:10:03,839 - src.training.checkpoint_manager - INFO - Size OK: 368.23 MB (diff 0.00%)
2026-05-08 23:10:03
2026-05-08 16:10:03,839 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 1 -> vv7 confirmed (state=COMMITTED, 368.2 MB). Safe to close Colab.
2026-05-08 23:10:03
2026-05-08 16:10:03,883 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v2/epoch_000.pt
2026-05-08 23:10:03
2026-05-08 16:10:03,928 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v2/best.pt
2026-05-08 23:10:03
2026-05-08 16:10:03,928 - src.training.trainer - INFO - Epoch 001 | train/loss=2.4220 | train/cl_loss=7.6745 | train/skipped_batches=0.0000 | train/lr=0.0010 | HR@10=0.0196 | NDCG@10=0.0064 | HR@20=0.0309 | NDCG@20=0.0085 | HR@50=0.0493 | NDCG@50=0.0116  <- best
2026-05-08 23:10:03
epochs:   7%|▋         | 2/30 [11:23<1:50:45, 237.32s/it, loss=2.2567, NDCG_20=0.0455, best_primary=0.0455]2026-05-08 16:13:33,942 - src.training.checkpoint_manager - INFO - Saved local: checkpoints-v2/epoch_002.pt (368.2 MB)
2026-05-08 23:10:03
2026-05-08 16:13:35,260 - src.training.checkpoint_manager - INFO - Artifact enqueued: bpatmp-checkpoint-v2 epoch-002
2026-05-08 23:13:59
2026-05-08 16:13:59,908 - src.training.checkpoint_manager - INFO - Size OK: 368.23 MB (diff 0.00%)
2026-05-08 23:13:59
2026-05-08 16:13:59,909 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 2 -> vv8 confirmed (state=COMMITTED, 368.2 MB). Safe to close Colab.
2026-05-08 23:13:59
2026-05-08 16:13:59,953 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v2/epoch_001.pt
2026-05-08 23:14:00
2026-05-08 16:13:59,998 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v2/best.pt
2026-05-08 23:14:00
2026-05-08 16:13:59,998 - src.training.trainer - INFO - Epoch 002 | train/loss=2.2567 | train/cl_loss=7.6782 | train/skipped_batches=0.0000 | train/lr=0.0010 | HR@10=0.0958 | NDCG@10=0.0394 | HR@20=0.1253 | NDCG@20=0.0455 | HR@50=0.1681 | NDCG@50=0.0527  <- best
2026-05-08 23:14:00
epochs:  10%|█         | 3/30 [15:20<1:46:32, 236.75s/it, loss=2.1241, NDCG_20=0.0379, best_primary=0.0455]2026-05-08 16:17:30,476 - src.training.checkpoint_manager - INFO - Saved local: checkpoints-v2/epoch_003.pt (368.2 MB)
2026-05-08 23:14:00
2026-05-08 16:17:32,226 - src.training.checkpoint_manager - INFO - Artifact enqueued: bpatmp-checkpoint-v2 epoch-003
2026-05-08 23:17:59
2026-05-08 16:17:59,074 - src.training.checkpoint_manager - INFO - Size OK: 368.23 MB (diff 0.00%)
2026-05-08 23:17:59
2026-05-08 16:17:59,074 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 3 -> vv9 confirmed (state=COMMITTED, 368.2 MB). Safe to close Colab.
2026-05-08 23:17:59
2026-05-08 16:17:59,118 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v2/epoch_002.pt
2026-05-08 23:17:59
2026-05-08 16:17:59,119 - src.training.trainer - INFO - Epoch 003 | train/loss=2.1241 | train/cl_loss=7.6758 | train/skipped_batches=0.0000 | train/lr=0.0010 | HR@10=0.0812 | NDCG@10=0.0325 | HR@20=0.1102 | NDCG@20=0.0379 | HR@50=0.1459 | NDCG@50=0.0440
2026-05-08 23:17:59
epochs:  13%|█▎        | 4/30 [19:20<1:42:59, 237.69s/it, loss=2.0567, NDCG_20=0.0488, best_primary=0.0488]2026-05-08 16:21:30,254 - src.training.checkpoint_manager - INFO - Saved local: checkpoints-v2/epoch_004.pt (368.2 MB)
2026-05-08 23:17:59
2026-05-08 16:21:32,371 - src.training.checkpoint_manager - INFO - Artifact enqueued: bpatmp-checkpoint-v2 epoch-004
2026-05-08 23:21:59
2026-05-08 16:21:59,614 - src.training.checkpoint_manager - INFO - Size OK: 368.23 MB (diff 0.00%)
2026-05-08 23:21:59
2026-05-08 16:21:59,614 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 4 -> vv10 confirmed (state=COMMITTED, 368.2 MB). Safe to close Colab.
2026-05-08 23:21:59
2026-05-08 16:21:59,659 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v2/epoch_003.pt
2026-05-08 23:21:59
2026-05-08 16:21:59,704 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v2/best.pt
2026-05-08 23:21:59
2026-05-08 16:21:59,704 - src.training.trainer - INFO - Epoch 004 | train/loss=2.0567 | train/cl_loss=7.6782 | train/skipped_batches=0.0000 | train/lr=0.0010 | HR@10=0.1042 | NDCG@10=0.0408 | HR@20=0.1443 | NDCG@20=0.0488 | HR@50=0.2026 | NDCG@50=0.0591  <- best
2026-05-08 23:21:59
epochs:  17%|█▋        | 5/30 [23:22<1:39:28, 238.73s/it, loss=2.0126, NDCG_20=0.0528, best_primary=0.0528]2026-05-08 16:25:32,897 - src.training.checkpoint_manager - INFO - Saved local: checkpoints-v2/epoch_005.pt (368.2 MB)
2026-05-08 23:21:59
2026-05-08 16:25:34,144 - src.training.checkpoint_manager - INFO - Artifact enqueued: bpatmp-checkpoint-v2 epoch-005
2026-05-08 23:26:01
2026-05-08 16:26:01,848 - src.training.checkpoint_manager - INFO - Size OK: 368.23 MB (diff 0.00%)
2026-05-08 23:26:01
2026-05-08 16:26:01,849 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 5 -> vv11 confirmed (state=COMMITTED, 368.2 MB). Safe to close Colab.
2026-05-08 23:26:01
2026-05-08 16:26:01,894 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v2/epoch_004.pt
2026-05-08 23:26:01
2026-05-08 16:26:01,943 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v2/best.pt
2026-05-08 23:26:01
2026-05-08 16:26:01,943 - src.training.trainer - INFO - Epoch 005 | train/loss=2.0126 | train/cl_loss=7.6769 | train/skipped_batches=0.0000 | train/lr=0.0010 | HR@10=0.1140 | NDCG@10=0.0445 | HR@20=0.1570 | NDCG@20=0.0528 | HR@50=0.2316 | NDCG@50=0.0657  <- best
2026-05-08 23:26:01
epochs:  20%|██        | 6/30 [27:23<1:35:58, 239.92s/it, loss=1.9764, NDCG_20=0.0490, best_primary=0.0528]2026-05-08 16:29:33,919 - src.training.checkpoint_manager - INFO - Saved local: checkpoints-v2/epoch_006.pt (368.2 MB)
2026-05-08 23:26:01
2026-05-08 16:29:35,204 - src.training.checkpoint_manager - INFO - Artifact enqueued: bpatmp-checkpoint-v2 epoch-006
2026-05-08 23:30:03
2026-05-08 16:30:03,264 - src.training.checkpoint_manager - INFO - Size OK: 368.23 MB (diff 0.00%)
2026-05-08 23:30:03
2026-05-08 16:30:03,264 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 6 -> vv12 confirmed (state=COMMITTED, 368.2 MB). Safe to close Colab.
2026-05-08 23:30:03
2026-05-08 16:30:03,308 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v2/epoch_005.pt
2026-05-08 23:30:03
2026-05-08 16:30:03,309 - src.training.trainer - INFO - Epoch 006 | train/loss=1.9764 | train/cl_loss=7.6753 | train/skipped_batches=0.0000 | train/lr=0.0009 | HR@10=0.1048 | NDCG@10=0.0414 | HR@20=0.1431 | NDCG@20=0.0490 | HR@50=0.2117 | NDCG@50=0.0609
2026-05-08 23:30:03
epochs:  23%|██▎       | 7/30 [31:27<1:32:09, 240.40s/it, loss=1.9480, NDCG_20=0.0583, best_primary=0.0583]2026-05-08 16:33:37,623 - src.training.checkpoint_manager - INFO - Saved local: checkpoints-v2/epoch_007.pt (368.2 MB)
2026-05-08 23:30:03
2026-05-08 16:33:38,828 - src.training.checkpoint_manager - INFO - Artifact enqueued: bpatmp-checkpoint-v2 epoch-007
2026-05-08 23:34:03
2026-05-08 16:34:03,138 - src.training.checkpoint_manager - INFO - Size OK: 368.23 MB (diff 0.00%)
2026-05-08 23:34:03
2026-05-08 16:34:03,138 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 7 -> vv13 confirmed (state=COMMITTED, 368.2 MB). Safe to close Colab.
2026-05-08 23:34:03
2026-05-08 16:34:03,185 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v2/epoch_006.pt
2026-05-08 23:34:03
2026-05-08 16:34:03,231 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v2/best.pt
2026-05-08 23:34:03
2026-05-08 16:34:03,232 - src.training.trainer - INFO - Epoch 007 | train/loss=1.9480 | train/cl_loss=7.6780 | train/skipped_batches=0.0000 | train/lr=0.0009 | HR@10=0.1243 | NDCG@10=0.0504 | HR@20=0.1633 | NDCG@20=0.0583 | HR@50=0.2326 | NDCG@50=0.0706  <- best
2026-05-08 23:34:03
epochs:  27%|██▋       | 8/30 [35:24<1:28:05, 240.24s/it, loss=1.9230, NDCG_20=0.0639, best_primary=0.0639]2026-05-08 16:37:34,683 - src.training.checkpoint_manager - INFO - Saved local: checkpoints-v2/epoch_008.pt (368.2 MB)
2026-05-08 23:34:03
2026-05-08 16:37:36,644 - src.training.checkpoint_manager - INFO - Artifact enqueued: bpatmp-checkpoint-v2 epoch-008
2026-05-08 23:38:02
2026-05-08 16:38:02,022 - src.training.checkpoint_manager - INFO - Size OK: 368.23 MB (diff 0.00%)
2026-05-08 23:38:02
2026-05-08 16:38:02,022 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 8 -> vv14 confirmed (state=COMMITTED, 368.2 MB). Safe to close Colab.
2026-05-08 23:38:02
2026-05-08 16:38:02,066 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v2/epoch_007.pt
2026-05-08 23:38:02
2026-05-08 16:38:02,112 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v2/best.pt
2026-05-08 23:38:02
2026-05-08 16:38:02,112 - src.training.trainer - INFO - Epoch 008 | train/loss=1.9230 | train/cl_loss=7.6765 | train/skipped_batches=0.0000 | train/lr=0.0009 | HR@10=0.1396 | NDCG@10=0.0538 | HR@20=0.1881 | NDCG@20=0.0639 | HR@50=0.2642 | NDCG@50=0.0781  <- best
2026-05-08 23:38:02
epochs:  30%|███       | 9/30 [39:24<1:23:56, 239.82s/it, loss=1.9014, NDCG_20=0.0521, best_primary=0.0639]2026-05-08 16:41:34,095 - src.training.checkpoint_manager - INFO - Saved local: checkpoints-v2/epoch_009.pt (368.2 MB)
2026-05-08 23:38:02
2026-05-08 16:41:35,307 - src.training.checkpoint_manager - INFO - Artifact enqueued: bpatmp-checkpoint-v2 epoch-009
2026-05-08 23:41:59
2026-05-08 16:41:59,484 - src.training.checkpoint_manager - INFO - Size OK: 368.23 MB (diff 0.00%)
2026-05-08 23:41:59
2026-05-08 16:41:59,485 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 9 -> vv15 confirmed (state=COMMITTED, 368.2 MB). Safe to close Colab.
2026-05-08 23:41:59
2026-05-08 16:41:59,529 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v2/epoch_008.pt
2026-05-08 23:41:59
2026-05-08 16:41:59,530 - src.training.trainer - INFO - Epoch 009 | train/loss=1.9014 | train/cl_loss=7.6752 | train/skipped_batches=0.0000 | train/lr=0.0008 | HR@10=0.1132 | NDCG@10=0.0451 | HR@20=0.1484 | NDCG@20=0.0521 | HR@50=0.2122 | NDCG@50=0.0631
2026-05-08 23:41:59
epochs:  33%|███▎      | 10/30 [43:24<1:19:41, 239.08s/it, loss=1.8811, NDCG_20=0.0527, best_primary=0.0639]2026-05-08 16:45:34,942 - src.training.checkpoint_manager - INFO - Saved local: checkpoints-v2/epoch_010.pt (368.2 MB)
2026-05-08 23:41:59
2026-05-08 16:45:36,176 - src.training.checkpoint_manager - INFO - Artifact enqueued: bpatmp-checkpoint-v2 epoch-010
2026-05-08 23:46:02
2026-05-08 16:46:02,099 - src.training.checkpoint_manager - INFO - Size OK: 368.23 MB (diff 0.00%)
2026-05-08 23:46:02
2026-05-08 16:46:02,100 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 10 -> vv16 confirmed (state=COMMITTED, 368.2 MB). Safe to close Colab.
2026-05-08 23:46:02
2026-05-08 16:46:02,147 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v2/epoch_009.pt
2026-05-08 23:46:02
2026-05-08 16:46:02,147 - src.training.trainer - INFO - Epoch 010 | train/loss=1.8811 | train/cl_loss=7.6784 | train/skipped_batches=0.0000 | train/lr=0.0008 | HR@10=0.1146 | NDCG@10=0.0456 | HR@20=0.1502 | NDCG@20=0.0527 | HR@50=0.2170 | NDCG@50=0.0644
2026-05-08 23:46:02
epochs:  37%|███▋      | 11/30 [47:26<1:16:03, 240.16s/it, loss=1.8624, NDCG_20=0.0625, best_primary=0.0639]2026-05-08 16:49:36,908 - src.training.checkpoint_manager - INFO - Saved local: checkpoints-v2/epoch_011.pt (368.2 MB)
2026-05-08 23:46:02
2026-05-08 16:49:38,433 - src.training.checkpoint_manager - INFO - Artifact enqueued: bpatmp-checkpoint-v2 epoch-011
2026-05-08 23:50:06
2026-05-08 16:50:06,217 - src.training.checkpoint_manager - INFO - Size OK: 368.23 MB (diff 0.00%)
2026-05-08 23:50:06
2026-05-08 16:50:06,217 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 11 -> vv17 confirmed (state=COMMITTED, 368.2 MB). Safe to close Colab.
2026-05-08 23:50:06
2026-05-08 16:50:06,267 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v2/epoch_010.pt
2026-05-08 23:50:06
2026-05-08 16:50:06,267 - src.training.trainer - INFO - Epoch 011 | train/loss=1.8624 | train/cl_loss=7.6764 | train/skipped_batches=0.0000 | train/lr=0.0007 | HR@10=0.1364 | NDCG@10=0.0528 | HR@20=0.1840 | NDCG@20=0.0625 | HR@50=0.2589 | NDCG@50=0.0761
2026-05-08 23:50:06
epochs:  40%|████      | 12/30 [51:29<1:12:24, 241.37s/it, loss=1.8452, NDCG_20=0.0492, best_primary=0.0639]2026-05-08 16:53:39,224 - src.training.checkpoint_manager - INFO - Saved local: checkpoints-v2/epoch_012.pt (368.2 MB)
2026-05-08 23:50:06
2026-05-08 16:53:40,538 - src.training.checkpoint_manager - INFO - Artifact enqueued: bpatmp-checkpoint-v2 epoch-012
2026-05-08 23:54:07
2026-05-08 16:54:07,340 - src.training.checkpoint_manager - INFO - Size OK: 368.23 MB (diff 0.00%)
2026-05-08 23:54:07
2026-05-08 16:54:07,341 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 12 -> vv18 confirmed (state=COMMITTED, 368.2 MB). Safe to close Colab.
2026-05-08 23:54:07
2026-05-08 16:54:07,387 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v2/epoch_011.pt
2026-05-08 23:54:07
2026-05-08 16:54:07,387 - src.training.trainer - INFO - Epoch 012 | train/loss=1.8452 | train/cl_loss=7.6774 | train/skipped_batches=0.0000 | train/lr=0.0007 | HR@10=0.1106 | NDCG@10=0.0419 | HR@20=0.1481 | NDCG@20=0.0492 | HR@50=0.2099 | NDCG@50=0.0596
2026-05-08 23:54:07
epochs:  43%|████▎     | 13/30 [55:30<1:08:21, 241.29s/it, loss=1.8276, NDCG_20=0.0580, best_primary=0.0639]2026-05-08 16:57:40,713 - src.training.checkpoint_manager - INFO - Saved local: checkpoints-v2/epoch_013.pt (368.2 MB)
2026-05-08 23:54:07
2026-05-08 16:57:42,010 - src.training.checkpoint_manager - INFO - Artifact enqueued: bpatmp-checkpoint-v2 epoch-013
2026-05-08 23:58:07
2026-05-08 16:58:07,067 - src.training.checkpoint_manager - INFO - Size OK: 368.23 MB (diff 0.00%)
2026-05-08 23:58:07
2026-05-08 16:58:07,067 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 13 -> vv19 confirmed (state=COMMITTED, 368.2 MB). Safe to close Colab.
2026-05-08 23:58:07
2026-05-08 16:58:07,114 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v2/epoch_012.pt
2026-05-08 23:58:07
2026-05-08 16:58:07,115 - src.training.trainer - INFO - Epoch 013 | train/loss=1.8276 | train/cl_loss=7.6763 | train/skipped_batches=0.0000 | train/lr=0.0006 | HR@10=0.1275 | NDCG@10=0.0503 | HR@20=0.1667 | NDCG@20=0.0580 | HR@50=0.2293 | NDCG@50=0.0688
2026-05-08 23:58:07
epochs:  47%|████▋     | 14/30 [59:30<1:04:13, 240.82s/it, loss=1.8111, NDCG_20=0.0558, best_primary=0.0639]2026-05-08 17:01:40,410 - src.training.checkpoint_manager - INFO - Saved local: checkpoints-v2/epoch_014.pt (368.2 MB)
2026-05-08 23:58:07
2026-05-08 17:01:41,715 - src.training.checkpoint_manager - INFO - Artifact enqueued: bpatmp-checkpoint-v2 epoch-014
2026-05-09 00:02:07
2026-05-08 17:02:07,660 - src.training.checkpoint_manager - INFO - Size OK: 368.23 MB (diff 0.00%)
2026-05-09 00:02:07
2026-05-08 17:02:07,660 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 14 -> vv20 confirmed (state=COMMITTED, 368.2 MB). Safe to close Colab.
2026-05-09 00:02:07
2026-05-08 17:02:07,707 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v2/epoch_013.pt
2026-05-09 00:02:07
2026-05-08 17:02:07,708 - src.training.trainer - INFO - Epoch 014 | train/loss=1.8111 | train/cl_loss=7.6764 | train/skipped_batches=0.0000 | train/lr=0.0006 | HR@10=0.1224 | NDCG@10=0.0475 | HR@20=0.1661 | NDCG@20=0.0558 | HR@50=0.2319 | NDCG@50=0.0679
2026-05-09 00:02:07
epochs:  50%|█████     | 15/30 [1:03:31<1:00:11, 240.75s/it, loss=1.7947, NDCG_20=0.0476, best_primary=0.0639]2026-05-08 17:05:41,479 - src.training.checkpoint_manager - INFO - Saved local: checkpoints-v2/epoch_015.pt (368.2 MB)
2026-05-09 00:02:07
2026-05-08 17:05:42,801 - src.training.checkpoint_manager - INFO - Artifact enqueued: bpatmp-checkpoint-v2 epoch-015
2026-05-09 00:06:09
2026-05-08 17:06:09,401 - src.training.checkpoint_manager - INFO - Size OK: 368.23 MB (diff 0.00%)
2026-05-09 00:06:09
2026-05-08 17:06:09,401 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 15 -> vv21 confirmed (state=COMMITTED, 368.2 MB). Safe to close Colab.
2026-05-09 00:06:09
2026-05-08 17:06:09,448 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v2/epoch_014.pt
2026-05-09 00:06:09
2026-05-08 17:06:09,448 - src.training.trainer - INFO - Epoch 015 | train/loss=1.7947 | train/cl_loss=7.6786 | train/skipped_batches=0.0000 | train/lr=0.0005 | HR@10=0.1037 | NDCG@10=0.0400 | HR@20=0.1437 | NDCG@20=0.0476 | HR@50=0.2053 | NDCG@50=0.0580
2026-05-09 00:06:09
epochs:  53%|█████▎    | 16/30 [1:07:33<56:14, 241.05s/it, loss=1.7782, NDCG_20=0.0513, best_primary=0.0639]2026-05-08 17:09:43,722 - src.training.checkpoint_manager - INFO - Saved local: checkpoints-v2/epoch_016.pt (368.2 MB)
2026-05-09 00:06:09
2026-05-08 17:09:45,080 - src.training.checkpoint_manager - INFO - Artifact enqueued: bpatmp-checkpoint-v2 epoch-016
2026-05-09 00:10:11
2026-05-08 17:10:11,214 - src.training.checkpoint_manager - INFO - Size OK: 368.23 MB (diff 0.00%)
2026-05-09 00:10:11
2026-05-08 17:10:11,214 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 16 -> vv22 confirmed (state=COMMITTED, 368.2 MB). Safe to close Colab.
2026-05-09 00:10:11
2026-05-08 17:10:11,261 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v2/epoch_015.pt
2026-05-09 00:10:11
2026-05-08 17:10:11,262 - src.training.trainer - INFO - Epoch 016 | train/loss=1.7782 | train/cl_loss=7.6754 | train/skipped_batches=0.0000 | train/lr=0.0004 | HR@10=0.1148 | NDCG@10=0.0438 | HR@20=0.1522 | NDCG@20=0.0513 | HR@50=0.2114 | NDCG@50=0.0617
2026-05-09 00:10:11
2026-05-08 17:10:11,262 - src.training.trainer - INFO - Early stopping at epoch 16. Best NDCG@20=0.0639

## Model v3
### Config
data:
  data_dir: /content/data/
  node_counts:
    brand: 1919
    category: 14
    product: 29892
    user: 203063
  struct_dir: /content/data/node_mappings

model:
  dropout: 0.2
  embed_dim: 256
  n_layers: 3
  rank: 64
  use_grad_checkpoint: false
  n_intents: 64

sampler:
  hop1_budget: 16
  hop2_budget: 8
  hop1_sample_replace: true

# Total loss:
#   L = L_BPR + lambda_cl*L_CL + lambda_conv*L_conv + lambda_mono*L_mono + lambda_wd*||theta||^2
# Per-behavior BPR weights: w_b = clip((N_purchase / N_b) ** alpha, w_min, 1.0)
loss:
  lambda_cl: 0.15 # contrastive (HierarchicalMBCL)
  lambda_conv: 0.1 # funnel prior s_view < s_cart < s_purchase (0 = off)
  lambda_mono: 0.05 # monotonic decay prior lam_view >= lam_cart >= lam_purchase (0 = off)
  funnel_margin: 0.1
  alpha: 0.5 # exponent in (N_p / N_b) ** alpha; alpha in [0.25, 0.5]
  w_min: 0.05 # floor for view/cart weight

hierarchy_cl:
  enabled: true
  tau: 0.1
  hard_k: 64
  min_pair_overlap: 4
  pair_weights: null # null = auto progressive (view -> cart -> purchase)

training:
  amp: true
  use_bf16: true
  batch_size: 8192
  device: cuda
  epochs: 40
  eval_batch_size: 8192
  eval_every: 1
  eval_subsample: 20000 # 0 = full eval each cycle
  eval_seed: 42
  l2_lambda: 1.0e-05 # lambda_wd
  lr: 8.0e-04
  min_lr: 1.0e-06
  warmup_epochs: 3
  max_grad_norm: 1.0
  cl_every_k: 2 # raise to 2 since deeper model; keep CL quality high without per-step overhead
  max_view_triplets: 3000000
  num_neg: 32
  num_workers: 8
  patience: 10
  save_dir: checkpoints-v3
  weight_decay: 1.0e-02
  pin_memory: true
  persistent_workers: true
  prefetch_factor: 4

evaluation:
  full_ranking: true
  primary_metric: "NDCG@20"
  ks: [10, 20, 50]
  metrics: ["HR@10", "HR@20", "HR@50", "NDCG@10", "NDCG@20", "NDCG@50"]

wandb:
  artifact_name: bpatmp-checkpoint-v3
  enabled: true
  entity: nguyenmaiductrong37-h-c-vi-n-c-ng-ngh-b-u-ch-nh-vi-n-th-ng
  project: bpatmp-recsys
  run_name: bpatmp-v3
  save_every: 1

# A100 optimizations
a100:
  allow_tf32: true
  cudnn_benchmark: true
  use_fused_adamw: true
  compile_model: false
  empty_cache_freq: 0

### Log
2026-05-09 08:40:50
2026-05-09 01:40:50,808 - src.training.checkpoint_manager - INFO - No checkpoint found on W&B — starting from epoch 0.
2026-05-09 08:40:50
epochs:   0%|          | 0/40 [09:53<?, ?it/s, loss=2.5282, NDCG_20=0.0003, best_primary=0.0003]2026-05-09 01:50:50,172 - src.training.checkpoint_manager - INFO - Saved local: checkpoints-v3/epoch_000.pt (764.8 MB)
2026-05-09 08:40:50
2026-05-09 01:50:52,120 - src.training.checkpoint_manager - INFO - Artifact enqueued: bpatmp-checkpoint-v3 epoch-000
2026-05-09 08:51:02
2026-05-09 01:51:02,691 - src.training.checkpoint_manager - INFO - Size OK: 764.85 MB (diff 0.00%)
2026-05-09 08:51:02
2026-05-09 01:51:02,691 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 0 -> vv0 confirmed (state=COMMITTED, 764.8 MB). Safe to close Colab.
2026-05-09 08:51:02
2026-05-09 01:51:02,777 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v3/best.pt
2026-05-09 08:51:02
2026-05-09 01:51:02,778 - src.training.trainer - INFO - Epoch 000 | train/loss=2.5282 | train/cl_loss=4.3591 | train/skipped_batches=0.0000 | train/lr=0.0003 | HR@10=0.0005 | NDCG@10=0.0002 | HR@20=0.0008 | NDCG@20=0.0003 | HR@50=0.0015 | NDCG@50=0.0003  <- best
2026-05-09 08:51:02
epochs:   2%|▎         | 1/40 [20:02<6:37:46, 611.97s/it, loss=2.3828, NDCG_20=0.0211, best_primary=0.0211]2026-05-09 02:00:56,453 - src.training.checkpoint_manager - INFO - Saved local: checkpoints-v3/epoch_001.pt (764.8 MB)
2026-05-09 08:51:02
2026-05-09 02:00:58,537 - src.training.checkpoint_manager - INFO - Artifact enqueued: bpatmp-checkpoint-v3 epoch-001
2026-05-09 09:01:09
2026-05-09 02:01:09,286 - src.training.checkpoint_manager - INFO - Size OK: 764.85 MB (diff 0.00%)
2026-05-09 09:01:09
2026-05-09 02:01:09,286 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 1 -> vv1 confirmed (state=COMMITTED, 764.8 MB). Safe to close Colab.
2026-05-09 09:01:09
2026-05-09 02:01:09,374 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v3/epoch_000.pt
2026-05-09 09:01:09
2026-05-09 02:01:09,461 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v3/best.pt
2026-05-09 09:01:09
2026-05-09 02:01:09,462 - src.training.trainer - INFO - Epoch 001 | train/loss=2.3828 | train/cl_loss=4.3608 | train/skipped_batches=0.0000 | train/lr=0.0005 | HR@10=0.0600 | NDCG@10=0.0146 | HR@20=0.0905 | NDCG@20=0.0211 | HR@50=0.1240 | NDCG@50=0.0272  <- best
2026-05-09 09:01:09
epochs:   5%|▌         | 2/40 [30:11<6:25:36, 608.86s/it, loss=2.2285, NDCG_20=0.0765, best_primary=0.0765]2026-05-09 02:11:05,140 - src.training.checkpoint_manager - INFO - Saved local: checkpoints-v3/epoch_002.pt (764.8 MB)
2026-05-09 09:01:09
2026-05-09 02:11:10,036 - src.training.checkpoint_manager - INFO - Artifact enqueued: bpatmp-checkpoint-v3 epoch-002
2026-05-09 09:11:20
2026-05-09 02:11:20,828 - src.training.checkpoint_manager - INFO - Size OK: 764.85 MB (diff 0.00%)
2026-05-09 09:11:20
2026-05-09 02:11:20,829 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 2 -> vv2 confirmed (state=COMMITTED, 764.8 MB). Safe to close Colab.
2026-05-09 09:11:20
2026-05-09 02:11:20,918 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v3/epoch_001.pt
2026-05-09 09:11:21
2026-05-09 02:11:21,010 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v3/best.pt
2026-05-09 09:11:21
2026-05-09 02:11:21,010 - src.training.trainer - INFO - Epoch 002 | train/loss=2.2285 | train/cl_loss=4.3545 | train/skipped_batches=0.0000 | train/lr=0.0008 | HR@10=0.1629 | NDCG@10=0.0643 | HR@20=0.2175 | NDCG@20=0.0765 | HR@50=0.2893 | NDCG@50=0.0917  <- best
2026-05-09 09:11:21
epochs:   8%|▊         | 3/40 [40:23<6:16:13, 610.09s/it, loss=2.0897, NDCG_20=0.0768, best_primary=0.0768]2026-05-09 02:21:16,806 - src.training.checkpoint_manager - INFO - Saved local: checkpoints-v3/epoch_003.pt (764.8 MB)
2026-05-09 09:11:21
2026-05-09 02:21:18,947 - src.training.checkpoint_manager - INFO - Artifact enqueued: bpatmp-checkpoint-v3 epoch-003
2026-05-09 09:21:30
2026-05-09 02:21:30,430 - src.training.checkpoint_manager - INFO - Size OK: 764.85 MB (diff 0.00%)
2026-05-09 09:21:30
2026-05-09 02:21:30,430 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 3 -> vv3 confirmed (state=COMMITTED, 764.8 MB). Safe to close Colab.
2026-05-09 09:21:30
2026-05-09 02:21:30,522 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v3/epoch_002.pt
2026-05-09 09:21:30
2026-05-09 02:21:30,615 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v3/best.pt
2026-05-09 09:21:30
2026-05-09 02:21:30,615 - src.training.trainer - INFO - Epoch 003 | train/loss=2.0897 | train/cl_loss=4.3560 | train/skipped_batches=0.0000 | train/lr=0.0008 | HR@10=0.1691 | NDCG@10=0.0646 | HR@20=0.2250 | NDCG@20=0.0768 | HR@50=0.3083 | NDCG@50=0.0933  <- best
2026-05-09 09:21:30
epochs:  10%|█         | 4/40 [50:30<6:05:56, 609.90s/it, loss=1.9930, NDCG_20=0.0677, best_primary=0.0768]2026-05-09 02:31:23,984 - src.training.checkpoint_manager - INFO - Saved local: checkpoints-v3/epoch_004.pt (764.8 MB)
2026-05-09 09:21:30
2026-05-09 02:31:25,939 - src.training.checkpoint_manager - INFO - Artifact enqueued: bpatmp-checkpoint-v3 epoch-004
2026-05-09 09:31:37
2026-05-09 02:31:37,184 - src.training.checkpoint_manager - INFO - Size OK: 764.85 MB (diff 0.00%)
2026-05-09 09:31:37
2026-05-09 02:31:37,184 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 4 -> vv4 confirmed (state=COMMITTED, 764.8 MB). Safe to close Colab.
2026-05-09 09:31:37
2026-05-09 02:31:37,271 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v3/epoch_003.pt
2026-05-09 09:31:37
2026-05-09 02:31:37,272 - src.training.trainer - INFO - Epoch 004 | train/loss=1.9930 | train/cl_loss=4.3551 | train/skipped_batches=0.0000 | train/lr=0.0008 | HR@10=0.1479 | NDCG@10=0.0590 | HR@20=0.1908 | NDCG@20=0.0677 | HR@50=0.2489 | NDCG@50=0.0784
2026-05-09 09:31:37
epochs:  12%|█▎        | 5/40 [1:00:38<5:55:05, 608.73s/it, loss=1.9257, NDCG_20=0.0625, best_primary=0.0768]2026-05-09 02:41:31,851 - src.training.checkpoint_manager - INFO - Saved local: checkpoints-v3/epoch_005.pt (764.8 MB)
2026-05-09 09:31:37
2026-05-09 02:41:33,789 - src.training.checkpoint_manager - INFO - Artifact enqueued: bpatmp-checkpoint-v3 epoch-005
2026-05-09 09:41:45
2026-05-09 02:41:45,932 - src.training.checkpoint_manager - INFO - Size OK: 764.85 MB (diff 0.00%)
2026-05-09 09:41:45
2026-05-09 02:41:45,933 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 5 -> vv5 confirmed (state=COMMITTED, 764.8 MB). Safe to close Colab.
2026-05-09 09:41:46
2026-05-09 02:41:46,026 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v3/epoch_004.pt
2026-05-09 09:41:46
2026-05-09 02:41:46,026 - src.training.trainer - INFO - Epoch 005 | train/loss=1.9257 | train/cl_loss=4.3507 | train/skipped_batches=0.0000 | train/lr=0.0008 | HR@10=0.1371 | NDCG@10=0.0507 | HR@20=0.1943 | NDCG@20=0.0625 | HR@50=0.2933 | NDCG@50=0.0804
2026-05-09 09:41:46
epochs:  15%|█▌        | 6/40 [1:10:43<5:44:57, 608.74s/it, loss=1.8753, NDCG_20=0.0594, best_primary=0.0768]2026-05-09 02:51:37,306 - src.training.checkpoint_manager - INFO - Saved local: checkpoints-v3/epoch_006.pt (764.8 MB)
2026-05-09 09:41:46
2026-05-09 02:51:39,238 - src.training.checkpoint_manager - INFO - Artifact enqueued: bpatmp-checkpoint-v3 epoch-006
2026-05-09 09:51:50
2026-05-09 02:51:50,846 - src.training.checkpoint_manager - INFO - Size OK: 764.85 MB (diff 0.00%)
2026-05-09 09:51:50
2026-05-09 02:51:50,846 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 6 -> vv6 confirmed (state=COMMITTED, 764.8 MB). Safe to close Colab.
2026-05-09 09:51:50
2026-05-09 02:51:50,937 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v3/epoch_005.pt
2026-05-09 09:51:50
2026-05-09 02:51:50,937 - src.training.trainer - INFO - Epoch 006 | train/loss=1.8753 | train/cl_loss=4.3500 | train/skipped_batches=0.0000 | train/lr=0.0008 | HR@10=0.1312 | NDCG@10=0.0497 | HR@20=0.1809 | NDCG@20=0.0594 | HR@50=0.2566 | NDCG@50=0.0727
2026-05-09 09:51:50
epochs:  18%|█▊        | 7/40 [1:20:45<5:34:07, 607.49s/it, loss=1.8457, NDCG_20=0.0604, best_primary=0.0768]2026-05-09 03:01:38,939 - src.training.checkpoint_manager - INFO - Saved local: checkpoints-v3/epoch_007.pt (764.8 MB)
2026-05-09 09:51:50
2026-05-09 03:01:40,876 - src.training.checkpoint_manager - INFO - Artifact enqueued: bpatmp-checkpoint-v3 epoch-007
2026-05-09 10:01:51
2026-05-09 03:01:51,745 - src.training.checkpoint_manager - INFO - Size OK: 764.85 MB (diff 0.00%)
2026-05-09 10:01:51
2026-05-09 03:01:51,746 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 7 -> vv7 confirmed (state=COMMITTED, 764.8 MB). Safe to close Colab.
2026-05-09 10:01:51
2026-05-09 03:01:51,837 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v3/epoch_006.pt
2026-05-09 10:01:51
2026-05-09 03:01:51,837 - src.training.trainer - INFO - Epoch 007 | train/loss=1.8457 | train/cl_loss=4.3620 | train/skipped_batches=0.0000 | train/lr=0.0008 | HR@10=0.1315 | NDCG@10=0.0503 | HR@20=0.1831 | NDCG@20=0.0604 | HR@50=0.2638 | NDCG@50=0.0745
2026-05-09 10:01:51
epochs:  20%|██        | 8/40 [1:30:51<5:22:52, 605.39s/it, loss=1.8220, NDCG_20=0.0624, best_primary=0.0768]2026-05-09 03:11:45,211 - src.training.checkpoint_manager - INFO - Saved local: checkpoints-v3/epoch_008.pt (764.8 MB)
2026-05-09 10:01:51
2026-05-09 03:11:47,306 - src.training.checkpoint_manager - INFO - Artifact enqueued: bpatmp-checkpoint-v3 epoch-008
2026-05-09 10:11:58
2026-05-09 03:11:58,072 - src.training.checkpoint_manager - INFO - Size OK: 764.85 MB (diff 0.00%)
2026-05-09 10:11:58
2026-05-09 03:11:58,072 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 8 -> vv8 confirmed (state=COMMITTED, 764.8 MB). Safe to close Colab.
2026-05-09 10:11:58
2026-05-09 03:11:58,164 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v3/epoch_007.pt
2026-05-09 10:11:58
2026-05-09 03:11:58,164 - src.training.trainer - INFO - Epoch 008 | train/loss=1.8220 | train/cl_loss=4.3591 | train/skipped_batches=0.0000 | train/lr=0.0007 | HR@10=0.1399 | NDCG@10=0.0541 | HR@20=0.1845 | NDCG@20=0.0624 | HR@50=0.2698 | NDCG@50=0.0767
2026-05-09 10:11:58
epochs:  22%|██▎       | 9/40 [1:40:56<5:12:56, 605.68s/it, loss=1.8017, NDCG_20=0.0639, best_primary=0.0768]2026-05-09 03:21:49,925 - src.training.checkpoint_manager - INFO - Saved local: checkpoints-v3/epoch_009.pt (764.8 MB)
2026-05-09 10:11:58
2026-05-09 03:21:51,996 - src.training.checkpoint_manager - INFO - Artifact enqueued: bpatmp-checkpoint-v3 epoch-009
2026-05-09 10:22:02
2026-05-09 03:22:02,272 - src.training.checkpoint_manager - INFO - Size OK: 764.85 MB (diff 0.00%)
2026-05-09 10:22:02
2026-05-09 03:22:02,273 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 9 -> vv9 confirmed (state=COMMITTED, 764.8 MB). Safe to close Colab.
2026-05-09 10:22:02
2026-05-09 03:22:02,365 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v3/epoch_008.pt
2026-05-09 10:22:02
2026-05-09 03:22:02,366 - src.training.trainer - INFO - Epoch 009 | train/loss=1.8017 | train/cl_loss=4.3519 | train/skipped_batches=0.0000 | train/lr=0.0007 | HR@10=0.1419 | NDCG@10=0.0557 | HR@20=0.1847 | NDCG@20=0.0639 | HR@50=0.2620 | NDCG@50=0.0770
2026-05-09 10:22:02
epochs:  25%|██▌       | 10/40 [1:51:00<5:02:36, 605.23s/it, loss=1.7838, NDCG_20=0.0562, best_primary=0.0768]2026-05-09 03:31:53,919 - src.training.checkpoint_manager - INFO - Saved local: checkpoints-v3/epoch_010.pt (764.8 MB)
2026-05-09 10:22:02
2026-05-09 03:31:55,923 - src.training.checkpoint_manager - INFO - Artifact enqueued: bpatmp-checkpoint-v3 epoch-010
2026-05-09 10:32:07
2026-05-09 03:32:07,841 - src.training.checkpoint_manager - INFO - Size OK: 764.85 MB (diff 0.00%)
2026-05-09 10:32:07
2026-05-09 03:32:07,841 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 10 -> vv10 confirmed (state=COMMITTED, 764.8 MB). Safe to close Colab.
2026-05-09 10:32:07
2026-05-09 03:32:07,935 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v3/epoch_009.pt
2026-05-09 10:32:07
2026-05-09 03:32:07,935 - src.training.trainer - INFO - Epoch 010 | train/loss=1.7838 | train/cl_loss=4.3585 | train/skipped_batches=0.0000 | train/lr=0.0007 | HR@10=0.1259 | NDCG@10=0.0477 | HR@20=0.1720 | NDCG@20=0.0562 | HR@50=0.2566 | NDCG@50=0.0708
2026-05-09 10:32:07
epochs:  28%|██▊       | 11/40 [2:01:10<4:52:34, 605.33s/it, loss=1.7631, NDCG_20=0.0518, best_primary=0.0768]2026-05-09 03:42:03,844 - src.training.checkpoint_manager - INFO - Saved local: checkpoints-v3/epoch_011.pt (764.8 MB)
2026-05-09 10:32:07
2026-05-09 03:42:07,746 - src.training.checkpoint_manager - INFO - Artifact enqueued: bpatmp-checkpoint-v3 epoch-011
2026-05-09 10:42:19
2026-05-09 03:42:19,400 - src.training.checkpoint_manager - INFO - Size OK: 764.85 MB (diff 0.00%)
2026-05-09 10:42:19
2026-05-09 03:42:19,400 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 11 -> vv11 confirmed (state=COMMITTED, 764.8 MB). Safe to close Colab.
2026-05-09 10:42:19
2026-05-09 03:42:19,494 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v3/epoch_010.pt
2026-05-09 10:42:19
2026-05-09 03:42:19,494 - src.training.trainer - INFO - Epoch 011 | train/loss=1.7631 | train/cl_loss=4.3516 | train/skipped_batches=0.0000 | train/lr=0.0007 | HR@10=0.1168 | NDCG@10=0.0470 | HR@20=0.1434 | NDCG@20=0.0518 | HR@50=0.1953 | NDCG@50=0.0601
2026-05-09 10:42:19
epochs:  30%|███       | 12/40 [2:11:19<4:43:22, 607.23s/it, loss=1.7485, NDCG_20=0.0524, best_primary=0.0768]2026-05-09 03:52:12,744 - src.training.checkpoint_manager - INFO - Saved local: checkpoints-v3/epoch_012.pt (764.8 MB)
2026-05-09 10:42:19
2026-05-09 03:52:18,048 - src.training.checkpoint_manager - INFO - Artifact enqueued: bpatmp-checkpoint-v3 epoch-012
2026-05-09 10:52:29
2026-05-09 03:52:29,702 - src.training.checkpoint_manager - INFO - Size OK: 764.85 MB (diff 0.00%)
2026-05-09 10:52:29
2026-05-09 03:52:29,702 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 12 -> vv12 confirmed (state=COMMITTED, 764.8 MB). Safe to close Colab.
2026-05-09 10:52:29
2026-05-09 03:52:29,798 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v3/epoch_011.pt
2026-05-09 10:52:29
2026-05-09 03:52:29,798 - src.training.trainer - INFO - Epoch 012 | train/loss=1.7485 | train/cl_loss=4.3540 | train/skipped_batches=0.0000 | train/lr=0.0007 | HR@10=0.1177 | NDCG@10=0.0466 | HR@20=0.1507 | NDCG@20=0.0524 | HR@50=0.2258 | NDCG@50=0.0647
2026-05-09 10:52:29
epochs:  32%|███▎      | 13/40 [2:21:27<4:33:40, 608.16s/it, loss=1.7342, NDCG_20=0.0639, best_primary=0.0768]2026-05-09 04:02:21,036 - src.training.checkpoint_manager - INFO - Saved local: checkpoints-v3/epoch_013.pt (764.8 MB)
2026-05-09 10:52:29
2026-05-09 04:02:23,166 - src.training.checkpoint_manager - INFO - Artifact enqueued: bpatmp-checkpoint-v3 epoch-013
2026-05-09 11:02:34
2026-05-09 04:02:34,059 - src.training.checkpoint_manager - INFO - Size OK: 764.85 MB (diff 0.00%)
2026-05-09 11:02:34
2026-05-09 04:02:34,059 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 13 -> vv13 confirmed (state=COMMITTED, 764.8 MB). Safe to close Colab.
2026-05-09 11:02:34
2026-05-09 04:02:34,153 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v3/epoch_012.pt
2026-05-09 11:02:34
2026-05-09 04:02:34,153 - src.training.trainer - INFO - Epoch 013 | train/loss=1.7342 | train/cl_loss=4.3578 | train/skipped_batches=0.0000 | train/lr=0.0006 | HR@10=0.1411 | NDCG@10=0.0555 | HR@20=0.1846 | NDCG@20=0.0639 | HR@50=0.2698 | NDCG@50=0.0785
2026-05-09 11:02:34
2026-05-09 04:02:34,153 - src.training.trainer - INFO - Early stopping at epoch 13. Best NDCG@20=0.0768
2026-05-09 11:02:34
epochs:  32%|███▎      | 13/40 [2:21:43<4:54:20, 654.10s/it, loss=1.7342, NDCG_20=0.0639, best_primary=0.0768]

## Model v4
### Config
data:
  data_dir: /content/data/
  node_counts:
    brand: 1919
    category: 14
    product: 29892
    user: 203063
  struct_dir: /content/data/node_mappings

model:
  dropout: 0.2
  embed_dim: 256
  n_layers: 3
  rank: 64
  use_grad_checkpoint: false
  n_intents: 128 

sampler:
  hop1_budget: 20 
  hop2_budget: 12 
  hop1_sample_replace: true

# Total loss:
#   L = L_BPR + lambda_cl*L_CL + lambda_conv*L_conv + lambda_mono*L_mono + lambda_wd*||theta||^2
# Per-behavior BPR weights: w_b = clip((N_purchase / N_b) ** alpha, w_min, 1.0)
loss:
  lambda_cl: 0.2 
  lambda_conv: 0.1 
  lambda_mono: 0.05
  funnel_margin: 0.1
  alpha: 0.5 
  w_min: 0.05 

hierarchy_cl:
  enabled: true
  tau: 0.1
  hard_k: 128 # doubled: harder negatives in CL → stronger representation learning
  min_pair_overlap: 4
  pair_weights: null # null = auto progressive (view -> cart -> purchase)

training:
  amp: true
  use_bf16: true
  batch_size: 8192
  device: cuda
  epochs: 60 # extended: patience=15 needs room; v3 stopped at 13 with potential remaining
  eval_batch_size: 8192
  eval_every: 1
  eval_subsample: 20000 # 0 = full eval each cycle
  eval_seed: 42
  l2_lambda: 1.0e-05 # lambda_wd
  lr: 1.0e-03 # up from 0.0008: v3 peaked early then plateaued; higher lr gives more exploration momentum
  min_lr: 1.0e-06
  warmup_epochs: 5 # extended warmup to stabilize higher lr before full training
  max_grad_norm: 1.0
  cl_every_k: 1 # back to 1: CL actively learning (cl_loss 4.35 in v3), don't skip steps
  max_view_triplets: 5000000 # increased: 80GB allows larger triplet pool for view behavior
  num_neg: 64 # doubled: A100-80GB has headroom; harder BPR negatives → sharper ranking
  num_workers: 8
  patience: 15 # up from 10: v3 stopped too early (best epoch 3, stopped epoch 13)
  save_dir: checkpoints-v4
  weight_decay: 1.0e-02
  pin_memory: true
  persistent_workers: true
  prefetch_factor: 4

evaluation:
  full_ranking: true
  primary_metric: "NDCG@20"
  ks: [10, 20, 50]
  metrics: ["HR@10", "HR@20", "HR@50", "NDCG@10", "NDCG@20", "NDCG@50"]

wandb:
  artifact_name: bpatmp-checkpoint-v4
  enabled: true
  entity: nguyenmaiductrong37-h-c-vi-n-c-ng-ngh-b-u-ch-nh-vi-n-th-ng
  project: bpatmp-recsys
  run_name: bpatmp-v4
  save_every: 1

# A100-80GB optimizations
a100:
  allow_tf32: true
  cudnn_benchmark: true
  use_fused_adamw: true
  compile_model: false
  empty_cache_freq: 0

### Log
2026-05-09 14:08:23
2026-05-09 07:08:23,751 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 0 -> vv0 confirmed (state=COMMITTED, 765.2 MB). Safe to close Colab.
2026-05-09 14:08:23
2026-05-09 07:08:23,842 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v4/best.pt
2026-05-09 14:08:23
2026-05-09 07:08:23,843 - src.training.trainer - INFO - Epoch 000 | train/loss=3.4787 | train/cl_loss=8.7294 | train/skipped_batches=0.0000 | train/lr=0.0002 | HR@10=0.0001 | NDCG@10=0.0000 | HR@20=0.0004 | NDCG@20=0.0001 | HR@50=0.0011 | NDCG@50=0.0001  <- best
2026-05-09 14:08:23
epochs:   2%|▏         | 1/60 [34:32<17:24:08, 1061.84s/it, loss=3.3383, NDCG_20=0.0010, best_primary=0.0010]2026-05-09 07:25:17,406 - src.training.checkpoint_manager - INFO - Saved local: checkpoints-v4/epoch_001.pt (765.2 MB)
2026-05-09 14:08:23
2026-05-09 07:25:20,085 - src.training.checkpoint_manager - INFO - Artifact enqueued: bpatmp-checkpoint-v4 epoch-001
2026-05-09 14:26:06
2026-05-09 07:26:06,766 - src.training.checkpoint_manager - INFO - Size OK: 765.24 MB (diff 0.00%)
2026-05-09 14:26:06
2026-05-09 07:26:06,767 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 1 -> vv1 confirmed (state=COMMITTED, 765.2 MB). Safe to close Colab.
2026-05-09 14:26:06
2026-05-09 07:26:06,857 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v4/epoch_000.pt
2026-05-09 14:26:06
2026-05-09 07:26:06,949 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v4/best.pt
2026-05-09 14:26:06
2026-05-09 07:26:06,949 - src.training.trainer - INFO - Epoch 001 | train/loss=3.3383 | train/cl_loss=8.7320 | train/skipped_batches=0.0000 | train/lr=0.0004 | HR@10=0.0001 | NDCG@10=0.0000 | HR@20=0.0083 | NDCG@20=0.0010 | HR@50=0.1242 | NDCG@50=0.0185  <- best
2026-05-09 14:26:06
epochs:   3%|▎         | 2/60 [52:17<17:07:09, 1062.59s/it, loss=3.1698, NDCG_20=0.0330, best_primary=0.0330]2026-05-09 07:43:02,591 - src.training.checkpoint_manager - INFO - Saved local: checkpoints-v4/epoch_002.pt (765.2 MB)
2026-05-09 14:26:06
2026-05-09 07:43:05,400 - src.training.checkpoint_manager - INFO - Artifact enqueued: bpatmp-checkpoint-v4 epoch-002
2026-05-09 14:43:55
2026-05-09 07:43:55,621 - src.training.checkpoint_manager - INFO - Size OK: 765.24 MB (diff 0.00%)
2026-05-09 14:43:55
2026-05-09 07:43:55,622 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 2 -> vv2 confirmed (state=COMMITTED, 765.2 MB). Safe to close Colab.
2026-05-09 14:43:55
2026-05-09 07:43:55,711 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v4/epoch_001.pt
2026-05-09 14:43:55
2026-05-09 07:43:55,804 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v4/best.pt
2026-05-09 14:43:55
2026-05-09 07:43:55,804 - src.training.trainer - INFO - Epoch 002 | train/loss=3.1698 | train/cl_loss=8.7220 | train/skipped_batches=0.0000 | train/lr=0.0006 | HR@10=0.0761 | NDCG@10=0.0224 | HR@20=0.1315 | NDCG@20=0.0330 | HR@50=0.2027 | NDCG@50=0.0465  <- best
2026-05-09 14:43:55
epochs:   5%|▌         | 3/60 [1:10:06<16:52:10, 1065.45s/it, loss=3.0657, NDCG_20=0.0585, best_primary=0.0585]2026-05-09 08:00:51,409 - src.training.checkpoint_manager - INFO - Saved local: checkpoints-v4/epoch_003.pt (765.2 MB)
2026-05-09 14:43:55
2026-05-09 08:00:54,232 - src.training.checkpoint_manager - INFO - Artifact enqueued: bpatmp-checkpoint-v4 epoch-003
2026-05-09 15:01:46
2026-05-09 08:01:46,335 - src.training.checkpoint_manager - INFO - Size OK: 765.24 MB (diff 0.00%)
2026-05-09 15:01:46
2026-05-09 08:01:46,336 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 3 -> vv3 confirmed (state=COMMITTED, 765.2 MB). Safe to close Colab.
2026-05-09 15:01:46
2026-05-09 08:01:46,427 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v4/epoch_002.pt
2026-05-09 15:01:46
2026-05-09 08:01:46,525 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v4/best.pt
2026-05-09 15:01:46
2026-05-09 08:01:46,526 - src.training.trainer - INFO - Epoch 003 | train/loss=3.0657 | train/cl_loss=8.7261 | train/skipped_batches=0.0000 | train/lr=0.0008 | HR@10=0.1359 | NDCG@10=0.0497 | HR@20=0.1790 | NDCG@20=0.0585 | HR@50=0.2420 | NDCG@50=0.0698  <- best
2026-05-09 15:01:46
epochs:   7%|▋         | 4/60 [1:27:53<16:36:21, 1067.53s/it, loss=2.9942, NDCG_20=0.0551, best_primary=0.0585]2026-05-09 08:18:38,737 - src.training.checkpoint_manager - INFO - Saved local: checkpoints-v4/epoch_004.pt (765.2 MB)
2026-05-09 15:01:46
2026-05-09 08:18:41,464 - src.training.checkpoint_manager - INFO - Artifact enqueued: bpatmp-checkpoint-v4 epoch-004
2026-05-09 15:19:26
2026-05-09 08:19:26,422 - src.training.checkpoint_manager - INFO - Size OK: 765.24 MB (diff 0.00%)
2026-05-09 15:19:26
2026-05-09 08:19:26,422 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 4 -> vv4 confirmed (state=COMMITTED, 765.2 MB). Safe to close Colab.
2026-05-09 15:19:26
2026-05-09 08:19:26,516 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v4/epoch_003.pt
2026-05-09 15:19:26
2026-05-09 08:19:26,516 - src.training.trainer - INFO - Epoch 004 | train/loss=2.9942 | train/cl_loss=8.7263 | train/skipped_batches=0.0000 | train/lr=0.0010 | HR@10=0.1222 | NDCG@10=0.0465 | HR@20=0.1661 | NDCG@20=0.0551 | HR@50=0.2296 | NDCG@50=0.0662
2026-05-09 15:19:26
epochs:   8%|▊         | 5/60 [1:45:34<16:16:04, 1064.81s/it, loss=2.9544, NDCG_20=0.0681, best_primary=0.0681]2026-05-09 08:36:19,164 - src.training.checkpoint_manager - INFO - Saved local: checkpoints-v4/epoch_005.pt (765.2 MB)
2026-05-09 15:19:26
2026-05-09 08:36:22,064 - src.training.checkpoint_manager - INFO - Artifact enqueued: bpatmp-checkpoint-v4 epoch-005
2026-05-09 15:37:05
2026-05-09 08:37:05,672 - src.training.checkpoint_manager - INFO - Size OK: 765.24 MB (diff 0.00%)
2026-05-09 15:37:05
2026-05-09 08:37:05,673 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 5 -> vv5 confirmed (state=COMMITTED, 765.2 MB). Safe to close Colab.
2026-05-09 15:37:05
2026-05-09 08:37:05,764 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v4/best.pt
2026-05-09 15:37:05
2026-05-09 08:37:05,859 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v4/epoch_004.pt
2026-05-09 15:37:05
2026-05-09 08:37:05,859 - src.training.trainer - INFO - Epoch 005 | train/loss=2.9544 | train/cl_loss=8.7235 | train/skipped_batches=0.0000 | train/lr=0.0010 | HR@10=0.1537 | NDCG@10=0.0573 | HR@20=0.2069 | NDCG@20=0.0681 | HR@50=0.2893 | NDCG@50=0.0840  <- best
2026-05-09 15:37:05
epochs:  10%|█         | 6/60 [2:03:15<15:56:39, 1062.95s/it, loss=2.9125, NDCG_20=0.0649, best_primary=0.0681]2026-05-09 08:54:00,382 - src.training.checkpoint_manager - INFO - Saved local: checkpoints-v4/epoch_006.pt (765.2 MB)
2026-05-09 15:37:05
2026-05-09 08:54:03,626 - src.training.checkpoint_manager - INFO - Artifact enqueued: bpatmp-checkpoint-v4 epoch-006
2026-05-09 15:54:47
2026-05-09 08:54:47,602 - src.training.checkpoint_manager - INFO - Size OK: 765.24 MB (diff 0.00%)
2026-05-09 15:54:47
2026-05-09 08:54:47,602 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 6 -> vv6 confirmed (state=COMMITTED, 765.2 MB). Safe to close Colab.
2026-05-09 15:54:47
2026-05-09 08:54:47,692 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v4/epoch_005.pt
2026-05-09 15:54:47
2026-05-09 08:54:47,692 - src.training.trainer - INFO - Epoch 006 | train/loss=2.9125 | train/cl_loss=8.7342 | train/skipped_batches=0.0000 | train/lr=0.0010 | HR@10=0.1562 | NDCG@10=0.0535 | HR@20=0.2104 | NDCG@20=0.0649 | HR@50=0.2839 | NDCG@50=0.0793
2026-05-09 15:54:47
epochs:  12%|█▏        | 7/60 [2:20:57<15:38:37, 1062.59s/it, loss=2.8780, NDCG_20=0.0445, best_primary=0.0681]2026-05-09 09:11:42,236 - src.training.checkpoint_manager - INFO - Saved local: checkpoints-v4/epoch_007.pt (765.2 MB)
2026-05-09 15:54:47
2026-05-09 09:11:44,776 - src.training.checkpoint_manager - INFO - Artifact enqueued: bpatmp-checkpoint-v4 epoch-007
2026-05-09 16:12:37
2026-05-09 09:12:37,831 - src.training.checkpoint_manager - INFO - Size OK: 765.24 MB (diff 0.00%)
2026-05-09 16:12:37
2026-05-09 09:12:37,831 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 7 -> vv7 confirmed (state=COMMITTED, 765.2 MB). Safe to close Colab.
2026-05-09 16:12:37
2026-05-09 09:12:37,928 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v4/epoch_006.pt
2026-05-09 16:12:37
2026-05-09 09:12:37,929 - src.training.trainer - INFO - Epoch 007 | train/loss=2.8780 | train/cl_loss=8.7256 | train/skipped_batches=0.0000 | train/lr=0.0010 | HR@10=0.1012 | NDCG@10=0.0404 | HR@20=0.1234 | NDCG@20=0.0445 | HR@50=0.1639 | NDCG@50=0.0507
2026-05-09 16:12:37
epochs:  13%|█▎        | 8/60 [2:38:48<15:23:01, 1065.02s/it, loss=2.8544, NDCG_20=0.0575, best_primary=0.0681]2026-05-09 09:29:32,934 - src.training.checkpoint_manager - INFO - Saved local: checkpoints-v4/epoch_008.pt (765.2 MB)
2026-05-09 16:12:37
2026-05-09 09:29:35,442 - src.training.checkpoint_manager - INFO - Artifact enqueued: bpatmp-checkpoint-v4 epoch-008
2026-05-09 16:30:20
2026-05-09 09:30:20,542 - src.training.checkpoint_manager - INFO - Size OK: 765.24 MB (diff 0.00%)
2026-05-09 16:30:20
2026-05-09 09:30:20,543 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 8 -> vv8 confirmed (state=COMMITTED, 765.2 MB). Safe to close Colab.
2026-05-09 16:30:20
2026-05-09 09:30:20,635 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v4/epoch_007.pt
2026-05-09 16:30:20
2026-05-09 09:30:20,636 - src.training.trainer - INFO - Epoch 008 | train/loss=2.8544 | train/cl_loss=8.7274 | train/skipped_batches=0.0000 | train/lr=0.0010 | HR@10=0.1231 | NDCG@10=0.0494 | HR@20=0.1636 | NDCG@20=0.0575 | HR@50=0.2360 | NDCG@50=0.0698
2026-05-09 16:30:20
epochs:  15%|█▌        | 9/60 [2:56:30<15:04:39, 1064.30s/it, loss=2.8378, NDCG_20=0.0378, best_primary=0.0681]2026-05-09 09:47:15,097 - src.training.checkpoint_manager - INFO - Saved local: checkpoints-v4/epoch_009.pt (765.2 MB)
2026-05-09 16:30:20
2026-05-09 09:47:17,712 - src.training.checkpoint_manager - INFO - Artifact enqueued: bpatmp-checkpoint-v4 epoch-009
2026-05-09 16:48:02
2026-05-09 09:48:02,203 - src.training.checkpoint_manager - INFO - Size OK: 765.24 MB (diff 0.00%)
2026-05-09 16:48:02
2026-05-09 09:48:02,203 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 9 -> vv9 confirmed (state=COMMITTED, 765.2 MB). Safe to close Colab.
2026-05-09 16:48:02
2026-05-09 09:48:02,298 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v4/epoch_008.pt
2026-05-09 16:48:02
2026-05-09 09:48:02,299 - src.training.trainer - INFO - Epoch 009 | train/loss=2.8378 | train/cl_loss=8.7342 | train/skipped_batches=0.0000 | train/lr=0.0010 | HR@10=0.0804 | NDCG@10=0.0330 | HR@20=0.1074 | NDCG@20=0.0378 | HR@50=0.1587 | NDCG@50=0.0456
2026-05-09 16:48:02
epochs:  17%|█▋        | 10/60 [3:14:11<14:46:14, 1063.48s/it, loss=2.8225, NDCG_20=0.0270, best_primary=0.0681]2026-05-09 10:04:56,605 - src.training.checkpoint_manager - INFO - Saved local: checkpoints-v4/epoch_010.pt (765.2 MB)
2026-05-09 16:48:02
2026-05-09 10:04:59,130 - src.training.checkpoint_manager - INFO - Artifact enqueued: bpatmp-checkpoint-v4 epoch-010
2026-05-09 17:05:49
2026-05-09 10:05:49,414 - src.training.checkpoint_manager - INFO - Size OK: 765.24 MB (diff 0.00%)
2026-05-09 17:05:49
2026-05-09 10:05:49,415 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 10 -> vv10 confirmed (state=COMMITTED, 765.2 MB). Safe to close Colab.
2026-05-09 17:05:49
2026-05-09 10:05:49,508 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v4/epoch_009.pt
2026-05-09 17:05:49
2026-05-09 10:05:49,508 - src.training.trainer - INFO - Epoch 010 | train/loss=2.8225 | train/cl_loss=8.7267 | train/skipped_batches=0.0000 | train/lr=0.0010 | HR@10=0.0624 | NDCG@10=0.0237 | HR@20=0.0818 | NDCG@20=0.0270 | HR@50=0.1259 | NDCG@50=0.0334
2026-05-09 17:05:49
epochs:  18%|█▊        | 11/60 [3:22:28<15:01:57, 1104.44s/it, loss=2.8225, NDCG_20=0.0270, best_primary=0.0681]

## Model v5
### Config
data:
  data_dir: /content/data/
  node_counts:
    brand: 1919
    category: 14
    product: 29892
    user: 203063
  struct_dir: /content/data/node_mappings

model:
  dropout: 0.2
  embed_dim: 256
  n_layers: 3
  rank: 64
  use_grad_checkpoint: false
  n_intents: 96

sampler:
  hop1_budget: 20
  hop2_budget: 12
  hop1_sample_replace: true

# Total loss:
#   L = L_BPR + lambda_cl*L_CL + lambda_conv*L_conv + lambda_mono*L_mono + lambda_wd*||theta||^2
# Per-behavior BPR weights: w_b = clip((N_purchase / N_b) ** alpha, w_min, 1.0)
loss:
  lambda_cl: 0.15
  lambda_conv: 0.1
  lambda_mono: 0.05
  funnel_margin: 0.1
  alpha: 0.5
  w_min: 0.05

hierarchy_cl:
  enabled: true
  tau: 0.07
  hard_k: 64
  min_pair_overlap: 4
  pair_weights: null # null = auto progressive (view -> cart -> purchase)

training:
  amp: true
  use_bf16: true
  batch_size: 8192
  device: cuda
  epochs: 60
  eval_batch_size: 8192
  eval_every: 1
  eval_subsample: 60000
  eval_seed: 42
  l2_lambda: 1.0e-05 # lambda_wd
  lr: 1.0e-03
  min_lr: 1.0e-06
  warmup_epochs: 4
  max_grad_norm: 1.0
  cl_every_k: 2
  max_view_triplets: 5000000
  num_neg: 48
  num_workers: 8
  patience: 15
  save_dir: checkpoints-v5
  weight_decay: 1.0e-02
  pin_memory: true
  persistent_workers: true
  prefetch_factor: 4

evaluation:
  full_ranking: true
  primary_metric: "NDCG@20"
  ks: [10, 20, 50]
  metrics: ["HR@10", "HR@20", "HR@50", "NDCG@10", "NDCG@20", "NDCG@50"]

wandb:
  artifact_name: bpatmp-checkpoint-v5
  enabled: true
  entity: nguyenmaiductrong37-h-c-vi-n-c-ng-ngh-b-u-ch-nh-vi-n-th-ng
  project: bpatmp-recsys
  run_name: bpatmp-v5
  save_every: 1

# A100-80GB optimizations
a100:
  allow_tf32: true
  cudnn_benchmark: true
  use_fused_adamw: true
  compile_model: false
  empty_cache_freq: 0

### Log
2026-05-09 17:54:41
2026-05-09 10:54:41,815 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 0 -> vv0 confirmed (state=COMMITTED, 765.0 MB). Safe to close Colab.
2026-05-09 17:54:41
2026-05-09 10:54:41,904 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v5/best.pt
2026-05-09 17:54:41
2026-05-09 10:54:41,905 - src.training.trainer - INFO - Epoch 000 | train/loss=2.3838 | train/cl_loss=4.3461 | train/skipped_batches=0.0000 | train/lr=0.0003 | HR@10=0.0002 | NDCG@10=0.0001 | HR@20=0.0004 | NDCG@20=0.0001 | HR@50=0.0010 | NDCG@50=0.0001  <- best
2026-05-09 17:54:41
epochs:   2%|▏         | 1/60 [34:13<17:06:01, 1043.41s/it, loss=2.2112, NDCG_20=0.0285, best_primary=0.0285]2026-05-09 11:11:35,066 - src.training.checkpoint_manager - INFO - Saved local: checkpoints-v5/epoch_001.pt (765.0 MB)
2026-05-09 17:54:41
2026-05-09 11:11:37,879 - src.training.checkpoint_manager - INFO - Artifact enqueued: bpatmp-checkpoint-v5 epoch-001
2026-05-09 18:12:06
2026-05-09 11:12:06,457 - src.training.checkpoint_manager - INFO - Size OK: 765.04 MB (diff 0.00%)
2026-05-09 18:12:06
2026-05-09 11:12:06,457 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 1 -> vv1 confirmed (state=COMMITTED, 765.0 MB). Safe to close Colab.
2026-05-09 18:12:06
2026-05-09 11:12:06,547 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v5/epoch_000.pt
2026-05-09 18:12:06
2026-05-09 11:12:06,640 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v5/best.pt
2026-05-09 18:12:06
2026-05-09 11:12:06,641 - src.training.trainer - INFO - Epoch 001 | train/loss=2.2112 | train/cl_loss=4.3509 | train/skipped_batches=0.0000 | train/lr=0.0005 | HR@10=0.0668 | NDCG@10=0.0243 | HR@20=0.0856 | NDCG@20=0.0285 | HR@50=0.1154 | NDCG@50=0.0344  <- best
2026-05-09 18:12:06
epochs:   3%|▎         | 2/60 [51:40<16:49:23, 1044.19s/it, loss=2.0730, NDCG_20=0.0690, best_primary=0.0690]2026-05-09 11:29:02,033 - src.training.checkpoint_manager - INFO - Saved local: checkpoints-v5/epoch_002.pt (765.0 MB)
2026-05-09 18:12:06
2026-05-09 11:29:04,665 - src.training.checkpoint_manager - INFO - Artifact enqueued: bpatmp-checkpoint-v5 epoch-002
2026-05-09 18:29:32
2026-05-09 11:29:32,165 - src.training.checkpoint_manager - INFO - Size OK: 765.04 MB (diff 0.00%)
2026-05-09 18:29:32
2026-05-09 11:29:32,165 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 2 -> vv2 confirmed (state=COMMITTED, 765.0 MB). Safe to close Colab.
2026-05-09 18:29:32
2026-05-09 11:29:32,253 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v5/epoch_001.pt
2026-05-09 18:29:32
2026-05-09 11:29:32,345 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v5/best.pt
2026-05-09 18:29:32
2026-05-09 11:29:32,345 - src.training.trainer - INFO - Epoch 002 | train/loss=2.0730 | train/cl_loss=4.3444 | train/skipped_batches=0.0000 | train/lr=0.0008 | HR@10=0.1496 | NDCG@10=0.0573 | HR@20=0.2035 | NDCG@20=0.0690 | HR@50=0.2752 | NDCG@50=0.0834  <- best
2026-05-09 18:29:32
epochs:   5%|▌         | 3/60 [1:09:01<16:32:38, 1044.88s/it, loss=1.9691, NDCG_20=0.0661, best_primary=0.0690]2026-05-09 11:46:22,463 - src.training.checkpoint_manager - INFO - Saved local: checkpoints-v5/epoch_003.pt (765.0 MB)
2026-05-09 18:29:32
2026-05-09 11:46:25,457 - src.training.checkpoint_manager - INFO - Artifact enqueued: bpatmp-checkpoint-v5 epoch-003
2026-05-09 18:46:51
2026-05-09 11:46:51,942 - src.training.checkpoint_manager - INFO - Size OK: 765.04 MB (diff 0.00%)
2026-05-09 18:46:51
2026-05-09 11:46:51,943 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 3 -> vv3 confirmed (state=COMMITTED, 765.0 MB). Safe to close Colab.
2026-05-09 18:46:52
2026-05-09 11:46:52,035 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v5/epoch_002.pt
2026-05-09 18:46:52
2026-05-09 11:46:52,036 - src.training.trainer - INFO - Epoch 003 | train/loss=1.9691 | train/cl_loss=4.3422 | train/skipped_batches=0.0000 | train/lr=0.0010 | HR@10=0.1424 | NDCG@10=0.0558 | HR@20=0.1913 | NDCG@20=0.0661 | HR@50=0.2683 | NDCG@50=0.0809
2026-05-09 18:46:52
epochs:   7%|▋         | 4/60 [1:26:21<16:13:18, 1042.83s/it, loss=1.9100, NDCG_20=0.0662, best_primary=0.0690]2026-05-09 12:03:42,493 - src.training.checkpoint_manager - INFO - Saved local: checkpoints-v5/epoch_004.pt (765.0 MB)
2026-05-09 18:46:52
2026-05-09 12:03:44,869 - src.training.checkpoint_manager - INFO - Artifact enqueued: bpatmp-checkpoint-v5 epoch-004
2026-05-09 19:04:12
2026-05-09 12:04:12,229 - src.training.checkpoint_manager - INFO - Size OK: 765.04 MB (diff 0.00%)
2026-05-09 19:04:12
2026-05-09 12:04:12,230 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 4 -> vv4 confirmed (state=COMMITTED, 765.0 MB). Safe to close Colab.
2026-05-09 19:04:12
2026-05-09 12:04:12,325 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v5/epoch_003.pt
2026-05-09 19:04:12
2026-05-09 12:04:12,326 - src.training.trainer - INFO - Epoch 004 | train/loss=1.9100 | train/cl_loss=4.3453 | train/skipped_batches=0.0000 | train/lr=0.0010 | HR@10=0.1448 | NDCG@10=0.0558 | HR@20=0.1948 | NDCG@20=0.0662 | HR@50=0.2672 | NDCG@50=0.0801
2026-05-09 19:04:12
epochs:   8%|▊         | 5/60 [1:43:42<15:55:05, 1041.92s/it, loss=1.8561, NDCG_20=0.0617, best_primary=0.0690]2026-05-09 12:21:03,750 - src.training.checkpoint_manager - INFO - Saved local: checkpoints-v5/epoch_005.pt (765.0 MB)
2026-05-09 19:04:12
2026-05-09 12:21:06,185 - src.training.checkpoint_manager - INFO - Artifact enqueued: bpatmp-checkpoint-v5 epoch-005
2026-05-09 19:21:32
2026-05-09 12:21:32,934 - src.training.checkpoint_manager - INFO - Size OK: 765.04 MB (diff 0.00%)
2026-05-09 19:21:32
2026-05-09 12:21:32,934 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 5 -> vv5 confirmed (state=COMMITTED, 765.0 MB). Safe to close Colab.
2026-05-09 19:21:33
2026-05-09 12:21:33,024 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v5/epoch_004.pt
2026-05-09 19:21:33
2026-05-09 12:21:33,024 - src.training.trainer - INFO - Epoch 005 | train/loss=1.8561 | train/cl_loss=4.3442 | train/skipped_batches=0.0000 | train/lr=0.0010 | HR@10=0.1298 | NDCG@10=0.0523 | HR@20=0.1766 | NDCG@20=0.0617 | HR@50=0.2495 | NDCG@50=0.0749
2026-05-09 19:21:33
epochs:  10%|█         | 6/60 [2:01:04<15:37:21, 1041.50s/it, loss=1.8211, NDCG_20=0.0612, best_primary=0.0690]2026-05-09 12:38:25,814 - src.training.checkpoint_manager - INFO - Saved local: checkpoints-v5/epoch_006.pt (765.0 MB)
2026-05-09 19:21:33
2026-05-09 12:38:31,193 - src.training.checkpoint_manager - INFO - Artifact enqueued: bpatmp-checkpoint-v5 epoch-006
2026-05-09 19:38:57
2026-05-09 12:38:57,436 - src.training.checkpoint_manager - INFO - Size OK: 765.04 MB (diff 0.00%)
2026-05-09 19:38:57
2026-05-09 12:38:57,436 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 6 -> vv6 confirmed (state=COMMITTED, 765.0 MB). Safe to close Colab.
2026-05-09 19:38:57
2026-05-09 12:38:57,530 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v5/epoch_005.pt
2026-05-09 19:38:57
2026-05-09 12:38:57,531 - src.training.trainer - INFO - Epoch 006 | train/loss=1.8211 | train/cl_loss=4.3410 | train/skipped_batches=0.0000 | train/lr=0.0010 | HR@10=0.1398 | NDCG@10=0.0520 | HR@20=0.1855 | NDCG@20=0.0612 | HR@50=0.2519 | NDCG@50=0.0731
2026-05-09 19:38:57
epochs:  12%|█▏        | 7/60 [2:11:29<16:35:31, 1127.01s/it, loss=1.8211, NDCG_20=0.0612, best_primary=0.0690]

## Model v6
data:
  data_dir: /content/data/
  node_counts:
    brand: 1919
    category: 14
    product: 29892
    user: 203063
  struct_dir: /content/data/node_mappings

model:
  dropout: 0.2
  embed_dim: 256
  n_layers: 3
  rank: 64
  use_grad_checkpoint: false
  n_intents: 64

sampler:
  hop1_budget: 16
  hop2_budget: 8
  hop1_sample_replace: true

# Total loss:
#   L = L_BPR + lambda_cl*L_CL + lambda_conv*L_conv + lambda_mono*L_mono + lambda_wd*||theta||^2
# Per-behavior BPR weights: w_b = clip((N_purchase / N_b) ** alpha, w_min, 1.0)
loss:
  lambda_cl: 0.15 # contrastive (HierarchicalMBCL)
  lambda_conv: 0.1 # funnel prior s_view < s_cart < s_purchase (0 = off)
  lambda_mono: 0.05 # monotonic decay prior lam_view >= lam_cart >= lam_purchase (0 = off)
  funnel_margin: 0.1
  alpha: 0.5 # exponent in (N_p / N_b) ** alpha; alpha in [0.25, 0.5]
  w_min: 0.05 # floor for view/cart weight

hierarchy_cl:
  enabled: true
  tau: 0.1
  hard_k: 64
  min_pair_overlap: 4
  pair_weights: null # null = auto progressive (view -> cart -> purchase)

training:
  amp: true
  use_bf16: true
  batch_size: 8192
  device: cuda
  epochs: 40
  eval_batch_size: 8192
  eval_every: 1
  eval_subsample: 20000 # 0 = full eval each cycle
  eval_seed: 42
  l2_lambda: 1.0e-05 # lambda_wd
  lr: 8.0e-04
  min_lr: 1.0e-06
  warmup_epochs: 3
  max_grad_norm: 1.0
  cl_every_k: 2 # raise to 2 since deeper model; keep CL quality high without per-step overhead
  max_view_triplets: 3000000
  num_neg: 32
  num_workers: 8
  patience: 10
  save_dir: checkpoints-v6
  weight_decay: 1.0e-02
  pin_memory: true
  persistent_workers: true
  prefetch_factor: 4

evaluation:
  full_ranking: true
  primary_metric: "NDCG@20"
  ks: [10, 20, 50]
  metrics: ["HR@10", "HR@20", "HR@50", "NDCG@10", "NDCG@20", "NDCG@50"]

wandb:
  artifact_name: bpatmp-checkpoint-v6
  enabled: true
  entity: nguyenmaiductrong37-h-c-vi-n-c-ng-ngh-b-u-ch-nh-vi-n-th-ng
  project: bpatmp-recsys
  run_name: bpatmp-v6
  save_every: 1

# A100 optimizations
a100:
  allow_tf32: true
  cudnn_benchmark: true
  use_fused_adamw: true
  compile_model: false
  empty_cache_freq: 0

### Log
2026-05-09 21:35:04
2026-05-09 14:35:04,843 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 0 -> vv3 confirmed (state=COMMITTED, 764.8 MB). Safe to close Colab.
2026-05-09 21:35:04
2026-05-09 14:35:04,932 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v6/best.pt
2026-05-09 21:35:04
2026-05-09 14:35:04,932 - src.training.trainer - INFO - Epoch 000 | train/loss=2.5273 | train/cl_loss=4.3530 | train/skipped_batches=0.0000 | train/lr=0.0003 | HR@10=0.0008 | NDCG@10=0.0002 | HR@20=0.0011 | NDCG@20=0.0003 | HR@50=0.0017 | NDCG@50=0.0003  <- best
2026-05-09 21:35:04
epochs:   2%|▎         | 1/40 [19:56<6:39:49, 615.12s/it, loss=2.3781, NDCG_20=0.0014, best_primary=0.0014]2026-05-09 14:44:48,843 - src.training.checkpoint_manager - INFO - Saved local: checkpoints-v6/epoch_001.pt (764.8 MB)
2026-05-09 21:35:04
2026-05-09 14:44:51,263 - src.training.checkpoint_manager - INFO - Artifact enqueued: bpatmp-checkpoint-v6 epoch-001
2026-05-09 21:45:17
2026-05-09 14:45:17,766 - src.training.checkpoint_manager - INFO - Size OK: 764.85 MB (diff 0.00%)
2026-05-09 21:45:17
2026-05-09 14:45:17,767 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 1 -> vv4 confirmed (state=COMMITTED, 764.8 MB). Safe to close Colab.
2026-05-09 21:45:17
2026-05-09 14:45:17,857 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v6/epoch_000.pt
2026-05-09 21:45:17
2026-05-09 14:45:17,950 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v6/best.pt
2026-05-09 21:45:17
2026-05-09 14:45:17,950 - src.training.trainer - INFO - Epoch 001 | train/loss=2.3781 | train/cl_loss=4.3576 | train/skipped_batches=0.0000 | train/lr=0.0005 | HR@10=0.0037 | NDCG@10=0.0012 | HR@20=0.0056 | NDCG@20=0.0014 | HR@50=0.0183 | NDCG@50=0.0029  <- best
2026-05-09 21:45:17
epochs:   5%|▌         | 2/40 [30:10<6:28:47, 613.88s/it, loss=2.2218, NDCG_20=0.0381, best_primary=0.0381]2026-05-09 14:55:03,522 - src.training.checkpoint_manager - INFO - Saved local: checkpoints-v6/epoch_002.pt (764.8 MB)
2026-05-09 21:45:17
2026-05-09 14:55:05,932 - src.training.checkpoint_manager - INFO - Artifact enqueued: bpatmp-checkpoint-v6 epoch-002
2026-05-09 21:55:32
2026-05-09 14:55:32,533 - src.training.checkpoint_manager - INFO - Size OK: 764.85 MB (diff 0.00%)
2026-05-09 21:55:32
2026-05-09 14:55:32,533 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 2 -> vv5 confirmed (state=COMMITTED, 764.8 MB). Safe to close Colab.
2026-05-09 21:55:32
2026-05-09 14:55:32,622 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v6/epoch_001.pt
2026-05-09 21:55:32
2026-05-09 14:55:32,715 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v6/best.pt
2026-05-09 21:55:32
2026-05-09 14:55:32,715 - src.training.trainer - INFO - Epoch 002 | train/loss=2.2218 | train/cl_loss=4.3570 | train/skipped_batches=0.0000 | train/lr=0.0008 | HR@10=0.0905 | NDCG@10=0.0302 | HR@20=0.1290 | NDCG@20=0.0381 | HR@50=0.1817 | NDCG@50=0.0477  <- best
2026-05-09 21:55:32
epochs:   8%|▊         | 3/40 [40:24<6:18:48, 614.28s/it, loss=2.0874, NDCG_20=0.0567, best_primary=0.0567]2026-05-09 15:05:17,077 - src.training.checkpoint_manager - INFO - Saved local: checkpoints-v6/epoch_003.pt (764.8 MB)
2026-05-09 21:55:32
2026-05-09 15:05:19,660 - src.training.checkpoint_manager - INFO - Artifact enqueued: bpatmp-checkpoint-v6 epoch-003
2026-05-09 22:05:45
2026-05-09 15:05:45,991 - src.training.checkpoint_manager - INFO - Size OK: 764.85 MB (diff 0.00%)
2026-05-09 22:05:45
2026-05-09 15:05:45,991 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 3 -> vv6 confirmed (state=COMMITTED, 764.8 MB). Safe to close Colab.
2026-05-09 22:05:46
2026-05-09 15:05:46,082 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v6/epoch_002.pt
2026-05-09 22:05:46
2026-05-09 15:05:46,177 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v6/best.pt
2026-05-09 22:05:46
2026-05-09 15:05:46,177 - src.training.trainer - INFO - Epoch 003 | train/loss=2.0874 | train/cl_loss=4.3553 | train/skipped_batches=0.0000 | train/lr=0.0008 | HR@10=0.1221 | NDCG@10=0.0470 | HR@20=0.1696 | NDCG@20=0.0567 | HR@50=0.2429 | NDCG@50=0.0703  <- best
2026-05-09 22:05:46
epochs:  10%|█         | 4/40 [50:38<6:08:22, 613.96s/it, loss=1.9862, NDCG_20=0.0640, best_primary=0.0640]2026-05-09 15:15:30,703 - src.training.checkpoint_manager - INFO - Saved local: checkpoints-v6/epoch_004.pt (764.8 MB)
2026-05-09 22:05:46
2026-05-09 15:15:34,944 - src.training.checkpoint_manager - INFO - Artifact enqueued: bpatmp-checkpoint-v6 epoch-004
2026-05-09 22:16:02
2026-05-09 15:16:02,131 - src.training.checkpoint_manager - INFO - Size OK: 764.85 MB (diff 0.00%)
2026-05-09 22:16:02
2026-05-09 15:16:02,131 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 4 -> vv7 confirmed (state=COMMITTED, 764.8 MB). Safe to close Colab.
2026-05-09 22:16:02
2026-05-09 15:16:02,221 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v6/epoch_003.pt
2026-05-09 22:16:02
2026-05-09 15:16:02,315 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v6/best.pt
2026-05-09 22:16:02
2026-05-09 15:16:02,316 - src.training.trainer - INFO - Epoch 004 | train/loss=1.9862 | train/cl_loss=4.3600 | train/skipped_batches=0.0000 | train/lr=0.0008 | HR@10=0.1359 | NDCG@10=0.0537 | HR@20=0.1853 | NDCG@20=0.0640 | HR@50=0.2576 | NDCG@50=0.0773  <- best
2026-05-09 22:16:02
epochs:  12%|█▎        | 5/40 [1:00:53<5:58:36, 614.75s/it, loss=1.9162, NDCG_20=0.0666, best_primary=0.0666]2026-05-09 15:25:46,591 - src.training.checkpoint_manager - INFO - Saved local: checkpoints-v6/epoch_005.pt (764.8 MB)
2026-05-09 22:16:02
2026-05-09 15:25:49,241 - src.training.checkpoint_manager - INFO - Artifact enqueued: bpatmp-checkpoint-v6 epoch-005
2026-05-09 22:26:16
2026-05-09 15:26:16,727 - src.training.checkpoint_manager - INFO - Size OK: 764.85 MB (diff 0.00%)
2026-05-09 22:26:16
2026-05-09 15:26:16,727 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 5 -> vv8 confirmed (state=COMMITTED, 764.8 MB). Safe to close Colab.
2026-05-09 22:26:16
2026-05-09 15:26:16,820 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v6/best.pt
2026-05-09 22:26:16
2026-05-09 15:26:16,911 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v6/epoch_004.pt
2026-05-09 22:26:16
2026-05-09 15:26:16,911 - src.training.trainer - INFO - Epoch 005 | train/loss=1.9162 | train/cl_loss=4.3538 | train/skipped_batches=0.0000 | train/lr=0.0008 | HR@10=0.1429 | NDCG@10=0.0564 | HR@20=0.1925 | NDCG@20=0.0666 | HR@50=0.2698 | NDCG@50=0.0806  <- best
2026-05-09 22:26:16
epochs:  15%|█▌        | 6/40 [1:11:06<5:48:19, 614.69s/it, loss=1.8731, NDCG_20=0.0662, best_primary=0.0666]2026-05-09 15:35:58,870 - src.training.checkpoint_manager - INFO - Saved local: checkpoints-v6/epoch_006.pt (764.8 MB)
2026-05-09 22:26:16
2026-05-09 15:36:02,924 - src.training.checkpoint_manager - INFO - Artifact enqueued: bpatmp-checkpoint-v6 epoch-006
2026-05-09 22:36:31
2026-05-09 15:36:31,212 - src.training.checkpoint_manager - INFO - Size OK: 764.85 MB (diff 0.00%)
2026-05-09 22:36:31
2026-05-09 15:36:31,212 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 6 -> vv9 confirmed (state=COMMITTED, 764.8 MB). Safe to close Colab.
2026-05-09 22:36:31
2026-05-09 15:36:31,303 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v6/epoch_005.pt
2026-05-09 22:36:31
2026-05-09 15:36:31,304 - src.training.trainer - INFO - Epoch 006 | train/loss=1.8731 | train/cl_loss=4.3553 | train/skipped_batches=0.0000 | train/lr=0.0008 | HR@10=0.1462 | NDCG@10=0.0563 | HR@20=0.1947 | NDCG@20=0.0662 | HR@50=0.2702 | NDCG@50=0.0803
2026-05-09 22:36:31
epochs:  18%|█▊        | 7/40 [1:13:29<5:46:25, 629.87s/it, loss=1.8731, NDCG_20=0.0662, best_primary=0.0666]

## Model v7
2026-05-09 21:35:04
2026-05-09 14:35:04,843 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 0 -> vv3 confirmed (state=COMMITTED, 764.8 MB). Safe to close Colab.
2026-05-09 21:35:04
2026-05-09 14:35:04,932 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v6/best.pt
2026-05-09 21:35:04
2026-05-09 14:35:04,932 - src.training.trainer - INFO - Epoch 000 | train/loss=2.5273 | train/cl_loss=4.3530 | train/skipped_batches=0.0000 | train/lr=0.0003 | HR@10=0.0008 | NDCG@10=0.0002 | HR@20=0.0011 | NDCG@20=0.0003 | HR@50=0.0017 | NDCG@50=0.0003  <- best
2026-05-09 21:35:04
epochs:   2%|▎         | 1/40 [19:56<6:39:49, 615.12s/it, loss=2.3781, NDCG_20=0.0014, best_primary=0.0014]2026-05-09 14:44:48,843 - src.training.checkpoint_manager - INFO - Saved local: checkpoints-v6/epoch_001.pt (764.8 MB)
2026-05-09 21:35:04
2026-05-09 14:44:51,263 - src.training.checkpoint_manager - INFO - Artifact enqueued: bpatmp-checkpoint-v6 epoch-001
2026-05-09 21:45:17
2026-05-09 14:45:17,766 - src.training.checkpoint_manager - INFO - Size OK: 764.85 MB (diff 0.00%)
2026-05-09 21:45:17
2026-05-09 14:45:17,767 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 1 -> vv4 confirmed (state=COMMITTED, 764.8 MB). Safe to close Colab.
2026-05-09 21:45:17
2026-05-09 14:45:17,857 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v6/epoch_000.pt
2026-05-09 21:45:17
2026-05-09 14:45:17,950 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v6/best.pt
2026-05-09 21:45:17
2026-05-09 14:45:17,950 - src.training.trainer - INFO - Epoch 001 | train/loss=2.3781 | train/cl_loss=4.3576 | train/skipped_batches=0.0000 | train/lr=0.0005 | HR@10=0.0037 | NDCG@10=0.0012 | HR@20=0.0056 | NDCG@20=0.0014 | HR@50=0.0183 | NDCG@50=0.0029  <- best
2026-05-09 21:45:17
epochs:   5%|▌         | 2/40 [30:10<6:28:47, 613.88s/it, loss=2.2218, NDCG_20=0.0381, best_primary=0.0381]2026-05-09 14:55:03,522 - src.training.checkpoint_manager - INFO - Saved local: checkpoints-v6/epoch_002.pt (764.8 MB)
2026-05-09 21:45:17
2026-05-09 14:55:05,932 - src.training.checkpoint_manager - INFO - Artifact enqueued: bpatmp-checkpoint-v6 epoch-002
2026-05-09 21:55:32
2026-05-09 14:55:32,533 - src.training.checkpoint_manager - INFO - Size OK: 764.85 MB (diff 0.00%)
2026-05-09 21:55:32
2026-05-09 14:55:32,533 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 2 -> vv5 confirmed (state=COMMITTED, 764.8 MB). Safe to close Colab.
2026-05-09 21:55:32
2026-05-09 14:55:32,622 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v6/epoch_001.pt
2026-05-09 21:55:32
2026-05-09 14:55:32,715 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v6/best.pt
2026-05-09 21:55:32
2026-05-09 14:55:32,715 - src.training.trainer - INFO - Epoch 002 | train/loss=2.2218 | train/cl_loss=4.3570 | train/skipped_batches=0.0000 | train/lr=0.0008 | HR@10=0.0905 | NDCG@10=0.0302 | HR@20=0.1290 | NDCG@20=0.0381 | HR@50=0.1817 | NDCG@50=0.0477  <- best
2026-05-09 21:55:32
epochs:   8%|▊         | 3/40 [40:24<6:18:48, 614.28s/it, loss=2.0874, NDCG_20=0.0567, best_primary=0.0567]2026-05-09 15:05:17,077 - src.training.checkpoint_manager - INFO - Saved local: checkpoints-v6/epoch_003.pt (764.8 MB)
2026-05-09 21:55:32
2026-05-09 15:05:19,660 - src.training.checkpoint_manager - INFO - Artifact enqueued: bpatmp-checkpoint-v6 epoch-003
2026-05-09 22:05:45
2026-05-09 15:05:45,991 - src.training.checkpoint_manager - INFO - Size OK: 764.85 MB (diff 0.00%)
2026-05-09 22:05:45
2026-05-09 15:05:45,991 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 3 -> vv6 confirmed (state=COMMITTED, 764.8 MB). Safe to close Colab.
2026-05-09 22:05:46
2026-05-09 15:05:46,082 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v6/epoch_002.pt
2026-05-09 22:05:46
2026-05-09 15:05:46,177 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v6/best.pt
2026-05-09 22:05:46
2026-05-09 15:05:46,177 - src.training.trainer - INFO - Epoch 003 | train/loss=2.0874 | train/cl_loss=4.3553 | train/skipped_batches=0.0000 | train/lr=0.0008 | HR@10=0.1221 | NDCG@10=0.0470 | HR@20=0.1696 | NDCG@20=0.0567 | HR@50=0.2429 | NDCG@50=0.0703  <- best
2026-05-09 22:05:46
epochs:  10%|█         | 4/40 [50:38<6:08:22, 613.96s/it, loss=1.9862, NDCG_20=0.0640, best_primary=0.0640]2026-05-09 15:15:30,703 - src.training.checkpoint_manager - INFO - Saved local: checkpoints-v6/epoch_004.pt (764.8 MB)
2026-05-09 22:05:46
2026-05-09 15:15:34,944 - src.training.checkpoint_manager - INFO - Artifact enqueued: bpatmp-checkpoint-v6 epoch-004
2026-05-09 22:16:02
2026-05-09 15:16:02,131 - src.training.checkpoint_manager - INFO - Size OK: 764.85 MB (diff 0.00%)
2026-05-09 22:16:02
2026-05-09 15:16:02,131 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 4 -> vv7 confirmed (state=COMMITTED, 764.8 MB). Safe to close Colab.
2026-05-09 22:16:02
2026-05-09 15:16:02,221 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v6/epoch_003.pt
2026-05-09 22:16:02
2026-05-09 15:16:02,315 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v6/best.pt
2026-05-09 22:16:02
2026-05-09 15:16:02,316 - src.training.trainer - INFO - Epoch 004 | train/loss=1.9862 | train/cl_loss=4.3600 | train/skipped_batches=0.0000 | train/lr=0.0008 | HR@10=0.1359 | NDCG@10=0.0537 | HR@20=0.1853 | NDCG@20=0.0640 | HR@50=0.2576 | NDCG@50=0.0773  <- best
2026-05-09 22:16:02
epochs:  12%|█▎        | 5/40 [1:00:53<5:58:36, 614.75s/it, loss=1.9162, NDCG_20=0.0666, best_primary=0.0666]2026-05-09 15:25:46,591 - src.training.checkpoint_manager - INFO - Saved local: checkpoints-v6/epoch_005.pt (764.8 MB)
2026-05-09 22:16:02
2026-05-09 15:25:49,241 - src.training.checkpoint_manager - INFO - Artifact enqueued: bpatmp-checkpoint-v6 epoch-005
2026-05-09 22:26:16
2026-05-09 15:26:16,727 - src.training.checkpoint_manager - INFO - Size OK: 764.85 MB (diff 0.00%)
2026-05-09 22:26:16
2026-05-09 15:26:16,727 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 5 -> vv8 confirmed (state=COMMITTED, 764.8 MB). Safe to close Colab.
2026-05-09 22:26:16
2026-05-09 15:26:16,820 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v6/best.pt
2026-05-09 22:26:16
2026-05-09 15:26:16,911 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v6/epoch_004.pt
2026-05-09 22:26:16
2026-05-09 15:26:16,911 - src.training.trainer - INFO - Epoch 005 | train/loss=1.9162 | train/cl_loss=4.3538 | train/skipped_batches=0.0000 | train/lr=0.0008 | HR@10=0.1429 | NDCG@10=0.0564 | HR@20=0.1925 | NDCG@20=0.0666 | HR@50=0.2698 | NDCG@50=0.0806  <- best
2026-05-09 22:26:16
epochs:  15%|█▌        | 6/40 [1:11:06<5:48:19, 614.69s/it, loss=1.8731, NDCG_20=0.0662, best_primary=0.0666]2026-05-09 15:35:58,870 - src.training.checkpoint_manager - INFO - Saved local: checkpoints-v6/epoch_006.pt (764.8 MB)
2026-05-09 22:26:16
2026-05-09 15:36:02,924 - src.training.checkpoint_manager - INFO - Artifact enqueued: bpatmp-checkpoint-v6 epoch-006
2026-05-09 22:36:31
2026-05-09 15:36:31,212 - src.training.checkpoint_manager - INFO - Size OK: 764.85 MB (diff 0.00%)
2026-05-09 22:36:31
2026-05-09 15:36:31,212 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 6 -> vv9 confirmed (state=COMMITTED, 764.8 MB). Safe to close Colab.
2026-05-09 22:36:31
2026-05-09 15:36:31,303 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v6/epoch_005.pt
2026-05-09 22:36:31
2026-05-09 15:36:31,304 - src.training.trainer - INFO - Epoch 006 | train/loss=1.8731 | train/cl_loss=4.3553 | train/skipped_batches=0.0000 | train/lr=0.0008 | HR@10=0.1462 | NDCG@10=0.0563 | HR@20=0.1947 | NDCG@20=0.0662 | HR@50=0.2702 | NDCG@50=0.0803
2026-05-09 22:36:31
epochs:  18%|█▊        | 7/40 [1:13:29<5:46:25, 629.87s/it, loss=1.8731, NDCG_20=0.0662, best_primary=0.0666]

### Log
2026-05-10 04:56:35,825 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 0 -> vv25 confirmed (state=COMMITTED, 765.2 MB). Safe to close Colab.
2026-05-10 11:56:35
2026-05-10 04:56:35,915 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v7/best.pt
2026-05-10 11:56:35
2026-05-10 04:56:35,916 - src.training.trainer - INFO - Epoch 000 | train/loss=1.8630 | train/cl_loss=0.0000 | train/skipped_batches=0.0000 | train/lr=0.0003 | HR@10=0.0041 | NDCG@10=0.0009 | HR@20=0.0088 | NDCG@20=0.0016 | HR@50=0.0636 | NDCG@50=0.0086  <- best | DIAG λ: L0_lambda_cart=0.694 L0_lambda_purchase=0.694 L0_lambda_struct=0.693 L0_lambda_view=0.694 L1_lambda_cart=0.693 L1_lambda_purchase=0.694 | DIAG z: baw0_zbeta_norm_cart=7.984 baw0_zbeta_norm_purchase=7.973 baw0_zbeta_norm_struct=7.988 baw0_zbeta_norm_view=7.984 baw1_zbeta_norm_cart=7.982 baw1_zbeta_norm_purchase=7.979
2026-05-10 11:56:35
epochs:   5%|▌         | 1/20 [20:25<3:22:15, 638.69s/it, loss=1.5910, NDCG_20=0.0066, best_primary=0.0066]2026-05-10 05:06:25,768 - src.training.checkpoint_manager - INFO - Saved local: checkpoints-v7/epoch_001.pt (765.2 MB)
2026-05-10 11:56:35
2026-05-10 05:06:28,320 - src.training.checkpoint_manager - INFO - Artifact enqueued: bpatmp-checkpoint-v7 epoch-001
2026-05-10 12:07:14
2026-05-10 05:07:14,684 - src.training.checkpoint_manager - INFO - Size OK: 765.21 MB (diff 0.00%)
2026-05-10 12:07:14
2026-05-10 05:07:14,684 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 1 -> vv26 confirmed (state=COMMITTED, 765.2 MB). Safe to close Colab.
2026-05-10 12:07:14
2026-05-10 05:07:14,771 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v7/epoch_000.pt
2026-05-10 12:07:14
2026-05-10 05:07:14,861 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v7/best.pt
2026-05-10 12:07:14
2026-05-10 05:07:14,861 - src.training.trainer - INFO - Epoch 001 | train/loss=1.5910 | train/cl_loss=0.0000 | train/skipped_batches=0.0000 | train/lr=0.0005 | HR@10=0.0169 | NDCG@10=0.0040 | HR@20=0.0339 | NDCG@20=0.0066 | HR@50=0.0735 | NDCG@50=0.0125  <- best | DIAG λ: L0_lambda_cart=0.698 L0_lambda_purchase=0.691 L0_lambda_struct=0.693 L0_lambda_view=0.691 L1_lambda_cart=0.693 L1_lambda_purchase=0.693 | DIAG z: baw0_zbeta_norm_cart=7.688 baw0_zbeta_norm_purchase=7.750 baw0_zbeta_norm_struct=7.953 baw0_zbeta_norm_view=7.804 baw1_zbeta_norm_cart=7.941 baw1_zbeta_norm_purchase=8.023
2026-05-10 12:07:14
epochs:  10%|█         | 2/20 [31:02<3:11:39, 638.84s/it, loss=1.4040, NDCG_20=0.0023, best_primary=0.0066]2026-05-10 05:17:02,821 - src.training.checkpoint_manager - INFO - Saved local: checkpoints-v7/epoch_002.pt (765.2 MB)
2026-05-10 12:07:14
2026-05-10 05:17:05,648 - src.training.checkpoint_manager - INFO - Artifact enqueued: bpatmp-checkpoint-v7 epoch-002
2026-05-10 12:17:50
2026-05-10 05:17:50,135 - src.training.checkpoint_manager - INFO - Size OK: 765.21 MB (diff 0.00%)
2026-05-10 12:17:50
2026-05-10 05:17:50,136 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 2 -> vv27 confirmed (state=COMMITTED, 765.2 MB). Safe to close Colab.
2026-05-10 12:17:50
2026-05-10 05:17:50,224 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v7/epoch_001.pt
2026-05-10 12:17:50
2026-05-10 05:17:50,224 - src.training.trainer - INFO - Epoch 002 | train/loss=1.4040 | train/cl_loss=0.0000 | train/skipped_batches=0.0000 | train/lr=0.0008 | HR@10=0.0063 | NDCG@10=0.0016 | HR@20=0.0120 | NDCG@20=0.0023 | HR@50=0.0311 | NDCG@50=0.0044 | DIAG λ: L0_lambda_cart=0.696 L0_lambda_purchase=0.688 L0_lambda_struct=0.693 L0_lambda_view=0.692 L1_lambda_cart=0.691 L1_lambda_purchase=0.690 | DIAG z: baw0_zbeta_norm_cart=7.446 baw0_zbeta_norm_purchase=7.492 baw0_zbeta_norm_struct=7.895 baw0_zbeta_norm_view=7.662 baw1_zbeta_norm_cart=8.047 baw1_zbeta_norm_purchase=8.044
2026-05-10 12:17:50
epochs:  15%|█▌        | 3/20 [41:36<3:00:33, 637.25s/it, loss=1.2531, NDCG_20=0.0037, best_primary=0.0066]2026-05-10 05:27:36,823 - src.training.checkpoint_manager - INFO - Saved local: checkpoints-v7/epoch_003.pt (765.2 MB)
2026-05-10 12:17:50
2026-05-10 05:27:39,295 - src.training.checkpoint_manager - INFO - Artifact enqueued: bpatmp-checkpoint-v7 epoch-003
2026-05-10 12:28:27
2026-05-10 05:28:27,045 - src.training.checkpoint_manager - INFO - Size OK: 765.21 MB (diff 0.00%)
2026-05-10 12:28:27
2026-05-10 05:28:27,045 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 3 -> vv28 confirmed (state=COMMITTED, 765.2 MB). Safe to close Colab.
2026-05-10 12:28:27
2026-05-10 05:28:27,137 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v7/epoch_002.pt
2026-05-10 12:28:27
2026-05-10 05:28:27,137 - src.training.trainer - INFO - Epoch 003 | train/loss=1.2531 | train/cl_loss=0.0000 | train/skipped_batches=0.0000 | train/lr=0.0008 | HR@10=0.0096 | NDCG@10=0.0025 | HR@20=0.0179 | NDCG@20=0.0037 | HR@50=0.0436 | NDCG@50=0.0067 | DIAG λ: L0_lambda_cart=0.697 L0_lambda_purchase=0.697 L0_lambda_struct=0.693 L0_lambda_view=0.684 L1_lambda_cart=0.692 L1_lambda_purchase=0.694 | DIAG z: baw0_zbeta_norm_cart=7.301 baw0_zbeta_norm_purchase=7.301 baw0_zbeta_norm_struct=7.826 baw0_zbeta_norm_view=7.475 baw1_zbeta_norm_cart=8.069 baw1_zbeta_norm_purchase=8.071
2026-05-10 12:28:27
epochs:  20%|██        | 4/20 [52:13<2:49:53, 637.12s/it, loss=1.1416, NDCG_20=0.0068, best_primary=0.0068]2026-05-10 05:38:14,798 - src.training.checkpoint_manager - INFO - Saved local: checkpoints-v7/epoch_004.pt (765.2 MB)
2026-05-10 12:28:27
2026-05-10 05:38:17,229 - src.training.checkpoint_manager - INFO - Artifact enqueued: bpatmp-checkpoint-v7 epoch-004
2026-05-10 12:39:04
2026-05-10 05:39:04,547 - src.training.checkpoint_manager - INFO - Size OK: 765.21 MB (diff 0.00%)
2026-05-10 12:39:04
2026-05-10 05:39:04,548 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 4 -> vv29 confirmed (state=COMMITTED, 765.2 MB). Safe to close Colab.
2026-05-10 12:39:04
2026-05-10 05:39:04,638 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v7/epoch_003.pt
2026-05-10 12:39:04
2026-05-10 05:39:04,729 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v7/best.pt
2026-05-10 12:39:04
2026-05-10 05:39:04,730 - src.training.trainer - INFO - Epoch 004 | train/loss=1.1416 | train/cl_loss=0.0000 | train/skipped_batches=0.0000 | train/lr=0.0008 | HR@10=0.0164 | NDCG@10=0.0049 | HR@20=0.0290 | NDCG@20=0.0068 | HR@50=0.0630 | NDCG@50=0.0112  <- best | DIAG λ: L0_lambda_cart=0.696 L0_lambda_purchase=0.699 L0_lambda_struct=0.693 L0_lambda_view=0.683 L1_lambda_cart=0.692 L1_lambda_purchase=0.698 | DIAG z: baw0_zbeta_norm_cart=7.114 baw0_zbeta_norm_purchase=7.164 baw0_zbeta_norm_struct=7.760 baw0_zbeta_norm_view=7.365 baw1_zbeta_norm_cart=8.165 baw1_zbeta_norm_purchase=8.061
2026-05-10 12:39:04
epochs:  25%|██▌       | 5/20 [1:02:53<2:39:19, 637.29s/it, loss=1.0764, NDCG_20=0.0092, best_primary=0.0092]2026-05-10 05:48:53,576 - src.training.checkpoint_manager - INFO - Saved local: checkpoints-v7/epoch_005.pt (765.2 MB)
2026-05-10 12:39:04
2026-05-10 05:48:56,039 - src.training.checkpoint_manager - INFO - Artifact enqueued: bpatmp-checkpoint-v7 epoch-005
2026-05-10 12:49:40
2026-05-10 05:49:40,284 - src.training.checkpoint_manager - INFO - Size OK: 765.21 MB (diff 0.00%)
2026-05-10 12:49:40
2026-05-10 05:49:40,285 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 5 -> vv30 confirmed (state=COMMITTED, 765.2 MB). Safe to close Colab.
2026-05-10 12:49:40
2026-05-10 05:49:40,378 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v7/best.pt
2026-05-10 12:49:40
2026-05-10 05:49:40,468 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v7/epoch_004.pt
2026-05-10 12:49:40
2026-05-10 05:49:40,469 - src.training.trainer - INFO - Epoch 005 | train/loss=1.0764 | train/cl_loss=0.0000 | train/skipped_batches=0.0000 | train/lr=0.0007 | HR@10=0.0214 | NDCG@10=0.0063 | HR@20=0.0396 | NDCG@20=0.0092 | HR@50=0.0842 | NDCG@50=0.0151  <- best | DIAG λ: L0_lambda_cart=0.690 L0_lambda_purchase=0.699 L0_lambda_struct=0.693 L0_lambda_view=0.691 L1_lambda_cart=0.687 L1_lambda_purchase=0.687 | DIAG z: baw0_zbeta_norm_cart=7.031 baw0_zbeta_norm_purchase=7.075 baw0_zbeta_norm_struct=7.695 baw0_zbeta_norm_view=7.253 baw1_zbeta_norm_cart=8.129 baw1_zbeta_norm_purchase=7.975
2026-05-10 12:49:40
epochs:  30%|███       | 6/20 [1:13:27<2:28:34, 636.76s/it, loss=1.0321, NDCG_20=0.0054, best_primary=0.0092]2026-05-10 05:59:27,602 - src.training.checkpoint_manager - INFO - Saved local: checkpoints-v7/epoch_006.pt (765.2 MB)
2026-05-10 12:49:40
2026-05-10 05:59:30,055 - src.training.checkpoint_manager - INFO - Artifact enqueued: bpatmp-checkpoint-v7 epoch-006
2026-05-10 13:00:21
2026-05-10 06:00:21,349 - src.training.checkpoint_manager - INFO - Size OK: 765.21 MB (diff 0.00%)
2026-05-10 13:00:21
2026-05-10 06:00:21,349 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 6 -> vv31 confirmed (state=COMMITTED, 765.2 MB). Safe to close Colab.
2026-05-10 13:00:21
2026-05-10 06:00:21,437 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v7/epoch_005.pt
2026-05-10 13:00:21
2026-05-10 06:00:21,437 - src.training.trainer - INFO - Epoch 006 | train/loss=1.0321 | train/cl_loss=0.0000 | train/skipped_batches=0.0000 | train/lr=0.0007 | HR@10=0.0138 | NDCG@10=0.0039 | HR@20=0.0243 | NDCG@20=0.0054 | HR@50=0.0500 | NDCG@50=0.0086 | DIAG λ: L0_lambda_cart=0.686 L0_lambda_purchase=0.697 L0_lambda_struct=0.693 L0_lambda_view=0.691 L1_lambda_cart=0.681 L1_lambda_purchase=0.681 | DIAG z: baw0_zbeta_norm_cart=6.890 baw0_zbeta_norm_purchase=6.841 baw0_zbeta_norm_struct=7.634 baw0_zbeta_norm_view=7.082 baw1_zbeta_norm_cart=8.111 baw1_zbeta_norm_purchase=7.832
2026-05-10 13:00:21
epochs:  35%|███▌      | 7/20 [1:24:09<2:18:15, 638.14s/it, loss=0.9932, NDCG_20=0.0075, best_primary=0.0092]2026-05-10 06:10:09,187 - src.training.checkpoint_manager - INFO - Saved local: checkpoints-v7/epoch_007.pt (765.2 MB)
2026-05-10 13:00:21
2026-05-10 06:10:11,667 - src.training.checkpoint_manager - INFO - Artifact enqueued: bpatmp-checkpoint-v7 epoch-007
2026-05-10 13:10:56
2026-05-10 06:10:56,387 - src.training.checkpoint_manager - INFO - Size OK: 765.21 MB (diff 0.00%)
2026-05-10 13:10:56
2026-05-10 06:10:56,387 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 7 -> vv32 confirmed (state=COMMITTED, 765.2 MB). Safe to close Colab.
2026-05-10 13:10:56
2026-05-10 06:10:56,480 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v7/epoch_006.pt
2026-05-10 13:10:56
2026-05-10 06:10:56,480 - src.training.trainer - INFO - Epoch 007 | train/loss=0.9932 | train/cl_loss=0.0000 | train/skipped_batches=0.0000 | train/lr=0.0006 | HR@10=0.0186 | NDCG@10=0.0054 | HR@20=0.0314 | NDCG@20=0.0075 | HR@50=0.0583 | NDCG@50=0.0108 | DIAG λ: L0_lambda_cart=0.685 L0_lambda_purchase=0.692 L0_lambda_struct=0.693 L0_lambda_view=0.695 L1_lambda_cart=0.681 L1_lambda_purchase=0.680 | DIAG z: baw0_zbeta_norm_cart=6.835 baw0_zbeta_norm_purchase=6.776 baw0_zbeta_norm_struct=7.578 baw0_zbeta_norm_view=7.028 baw1_zbeta_norm_cart=8.056 baw1_zbeta_norm_purchase=7.787
2026-05-10 13:10:56
epochs:  40%|████      | 8/20 [1:34:40<2:07:25, 637.15s/it, loss=0.9598, NDCG_20=0.0129, best_primary=0.0129]2026-05-10 06:20:40,952 - src.training.checkpoint_manager - INFO - Saved local: checkpoints-v7/epoch_008.pt (765.2 MB)

## Model v8 (hiện tại)
## Config
data:
  data_dir: /content/data/
  node_counts:
    brand: 1919
    category: 14
    product: 29892
    user: 203063
  struct_dir: /content/data/node_mappings

model:
  dropout: 0.2
  embed_dim: 256
  n_layers: 3
  rank: 64
  use_grad_checkpoint: false
  n_intents: 64

sampler:
  hop1_budget: 16
  hop2_budget: 8
  hop1_sample_replace: true

loss:
  lambda_cl: 0.15
  lambda_conv: 0.1
  lambda_mono: 0.05
  funnel_margin: 0.1
  alpha: 0.5
  w_min: 0.05

hierarchy_cl:
  enabled: true
  tau: 0.1
  hard_k: 64
  min_pair_overlap: 4
  pair_weights: null

training:
  amp: true
  use_bf16: true
  batch_size: 8192
  device: cuda
  epochs: 25
  eval_batch_size: 8192
  eval_every: 1
  eval_subsample: 60000
  eval_seed: 42
  l2_lambda: 1.0e-05
  lr: 8.0e-04
  min_lr: 1.0e-06
  warmup_epochs: 3
  max_grad_norm: 5.0
  cl_every_k: 2
  max_view_triplets: 3000000
  num_neg: 32
  num_workers: 8
  patience: 12
  save_dir: checkpoints-v8
  weight_decay: 1.0e-02
  pin_memory: true
  persistent_workers: true
  prefetch_factor: 4

evaluation:
  full_ranking: true
  primary_metric: "NDCG@20"
  ks: [10, 20, 50]
  metrics: ["HR@10", "HR@20", "HR@50", "NDCG@10", "NDCG@20", "NDCG@50"]

wandb:
  artifact_name: bpatmp-checkpoint-v8
  enabled: true
  entity: nguyenmaiductrong37-h-c-vi-n-c-ng-ngh-b-u-ch-nh-vi-n-th-ng
  project: bpatmp-recsys
  run_name: bpatmp-v8
  save_every: 1

a100:
  allow_tf32: true
  cudnn_benchmark: true
  use_fused_adamw: true
  compile_model: false
  empty_cache_freq: 0
### Log
2026-05-10 06:44:30,341 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 0 -> vv0 confirmed (state=COMMITTED, 765.2 MB). Safe to close Colab.
2026-05-10 13:44:30
2026-05-10 06:44:30,428 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v8/best.pt
2026-05-10 13:44:30
2026-05-10 06:44:30,428 - src.training.trainer - INFO - Epoch 000 | train/loss=2.5238 | train/cl_loss=4.3597 | train/skipped_batches=0.0000 | train/lr=0.0003 | HR@10=0.0024 | NDCG@10=0.0008 | HR@20=0.0034 | NDCG@20=0.0010 | HR@50=0.0053 | NDCG@50=0.0012  <- best | DIAG λ: L0_lambda_cart=0.693 L0_lambda_purchase=0.694 L0_lambda_struct=0.693 L0_lambda_view=0.694 L1_lambda_cart=0.693 L1_lambda_purchase=0.693 | DIAG z: baw0_zbeta_norm_cart=7.999 baw0_zbeta_norm_purchase=7.998 baw0_zbeta_norm_struct=8.000 baw0_zbeta_norm_view=8.000 baw1_zbeta_norm_cart=8.000 baw1_zbeta_norm_purchase=8.003
2026-05-10 13:44:30
epochs:   4%|▍         | 1/25 [21:37<4:31:19, 678.33s/it, loss=2.2902, NDCG_20=0.0162, best_primary=0.0162]2026-05-10 06:54:58,284 - src.training.checkpoint_manager - INFO - Saved local: checkpoints-v8/epoch_001.pt (765.2 MB)
2026-05-10 13:44:30
2026-05-10 06:55:01,279 - src.training.checkpoint_manager - INFO - Artifact enqueued: bpatmp-checkpoint-v8 epoch-001
2026-05-10 13:55:48
2026-05-10 06:55:48,464 - src.training.checkpoint_manager - INFO - Size OK: 765.21 MB (diff 0.00%)
2026-05-10 13:55:48
2026-05-10 06:55:48,465 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 1 -> vv1 confirmed (state=COMMITTED, 765.2 MB). Safe to close Colab.
2026-05-10 13:55:48
2026-05-10 06:55:48,552 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v8/epoch_000.pt
2026-05-10 13:55:48
2026-05-10 06:55:48,640 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v8/best.pt
2026-05-10 13:55:48
2026-05-10 06:55:48,640 - src.training.trainer - INFO - Epoch 001 | train/loss=2.2902 | train/cl_loss=4.3582 | train/skipped_batches=0.0000 | train/lr=0.0005 | HR@10=0.0380 | NDCG@10=0.0114 | HR@20=0.0674 | NDCG@20=0.0162 | HR@50=0.1247 | NDCG@50=0.0247  <- best | DIAG λ: L0_lambda_cart=0.698 L0_lambda_purchase=0.688 L0_lambda_struct=0.693 L0_lambda_view=0.692 L1_lambda_cart=0.693 L1_lambda_purchase=0.690 | DIAG z: baw0_zbeta_norm_cart=7.962 baw0_zbeta_norm_purchase=7.871 baw0_zbeta_norm_struct=8.000 baw0_zbeta_norm_view=7.918 baw1_zbeta_norm_cart=8.010 baw1_zbeta_norm_purchase=8.104
2026-05-10 13:55:48
epochs:   8%|▊         | 2/25 [32:54<4:19:59, 678.26s/it, loss=2.0867, NDCG_20=0.0074, best_primary=0.0162]2026-05-10 07:06:09,269 - src.training.checkpoint_manager - INFO - Saved local: checkpoints-v8/epoch_002.pt (765.2 MB)
2026-05-10 13:55:48
2026-05-10 07:06:11,490 - src.training.checkpoint_manager - INFO - Artifact enqueued: bpatmp-checkpoint-v8 epoch-002
2026-05-10 14:06:56
2026-05-10 07:06:56,041 - src.training.checkpoint_manager - INFO - Size OK: 765.21 MB (diff 0.00%)
2026-05-10 14:06:56
2026-05-10 07:06:56,042 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 2 -> vv2 confirmed (state=COMMITTED, 765.2 MB). Safe to close Colab.
2026-05-10 14:06:56
2026-05-10 07:06:56,131 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v8/epoch_001.pt
2026-05-10 14:06:56
2026-05-10 07:06:56,132 - src.training.trainer - INFO - Epoch 002 | train/loss=2.0867 | train/cl_loss=4.3614 | train/skipped_batches=0.0000 | train/lr=0.0008 | HR@10=0.0200 | NDCG@10=0.0062 | HR@20=0.0288 | NDCG@20=0.0074 | HR@50=0.0497 | NDCG@50=0.0099 | DIAG λ: L0_lambda_cart=0.698 L0_lambda_purchase=0.686 L0_lambda_struct=0.693 L0_lambda_view=0.698 L1_lambda_cart=0.700 L1_lambda_purchase=0.684 | DIAG z: baw0_zbeta_norm_cart=8.052 baw0_zbeta_norm_purchase=7.568 baw0_zbeta_norm_struct=8.000 baw0_zbeta_norm_view=7.748 baw1_zbeta_norm_cart=8.105 baw1_zbeta_norm_purchase=8.017
2026-05-10 14:06:56
epochs:  12%|█▏        | 3/25 [43:56<4:06:53, 673.34s/it, loss=1.9271, NDCG_20=0.0019, best_primary=0.0162]2026-05-10 07:17:10,945 - src.training.checkpoint_manager - INFO - Saved local: checkpoints-v8/epoch_003.pt (765.2 MB)
2026-05-10 14:06:56
2026-05-10 07:17:13,435 - src.training.checkpoint_manager - INFO - Artifact enqueued: bpatmp-checkpoint-v8 epoch-003
2026-05-10 14:18:01
2026-05-10 07:18:01,474 - src.training.checkpoint_manager - INFO - Size OK: 765.21 MB (diff 0.00%)
2026-05-10 14:18:01
2026-05-10 07:18:01,475 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 3 -> vv3 confirmed (state=COMMITTED, 765.2 MB). Safe to close Colab.
2026-05-10 14:18:01
2026-05-10 07:18:01,565 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v8/epoch_002.pt
2026-05-10 14:18:01
2026-05-10 07:18:01,566 - src.training.trainer - INFO - Epoch 003 | train/loss=1.9271 | train/cl_loss=4.3507 | train/skipped_batches=0.0000 | train/lr=0.0008 | HR@10=0.0051 | NDCG@10=0.0013 | HR@20=0.0098 | NDCG@20=0.0019 | HR@50=0.0234 | NDCG@50=0.0034 | DIAG λ: L0_lambda_cart=0.694 L0_lambda_purchase=0.690 L0_lambda_struct=0.693 L0_lambda_view=0.704 L1_lambda_cart=0.700 L1_lambda_purchase=0.688 | DIAG z: baw0_zbeta_norm_cart=8.033 baw0_zbeta_norm_purchase=7.431 baw0_zbeta_norm_struct=8.000 baw0_zbeta_norm_view=7.580 baw1_zbeta_norm_cart=8.244 baw1_zbeta_norm_purchase=8.123
2026-05-10 14:18:01
epochs:  16%|█▌        | 4/25 [54:59<3:54:34, 670.22s/it, loss=1.8175, NDCG_20=0.0058, best_primary=0.0162]2026-05-10 07:28:14,296 - src.training.checkpoint_manager - INFO - Saved local: checkpoints-v8/epoch_004.pt (765.2 MB)
2026-05-10 14:18:01
2026-05-10 07:28:16,485 - src.training.checkpoint_manager - INFO - Artifact enqueued: bpatmp-checkpoint-v8 epoch-004
2026-05-10 14:29:03
2026-05-10 07:29:03,970 - src.training.checkpoint_manager - INFO - Size OK: 765.21 MB (diff 0.00%)
2026-05-10 14:29:03
2026-05-10 07:29:03,970 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 4 -> vv4 confirmed (state=COMMITTED, 765.2 MB). Safe to close Colab.
2026-05-10 14:29:04
2026-05-10 07:29:04,063 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v8/epoch_003.pt
2026-05-10 14:29:04
2026-05-10 07:29:04,064 - src.training.trainer - INFO - Epoch 004 | train/loss=1.8175 | train/cl_loss=4.3541 | train/skipped_batches=0.0000 | train/lr=0.0008 | HR@10=0.0139 | NDCG@10=0.0048 | HR@20=0.0212 | NDCG@20=0.0058 | HR@50=0.0416 | NDCG@50=0.0081 | DIAG λ: L0_lambda_cart=0.694 L0_lambda_purchase=0.689 L0_lambda_struct=0.693 L0_lambda_view=0.709 L1_lambda_cart=0.700 L1_lambda_purchase=0.689 | DIAG z: baw0_zbeta_norm_cart=7.961 baw0_zbeta_norm_purchase=7.407 baw0_zbeta_norm_struct=8.000 baw0_zbeta_norm_view=7.535 baw1_zbeta_norm_cart=8.266 baw1_zbeta_norm_purchase=8.136
2026-05-10 14:29:04
epochs:  20%|██        | 5/25 [55:51<3:42:28, 667.44s/it, loss=1.8175, NDCG_20=0.0058, best_primary=0.0162]
2026-05-10 14:29:04
train:  69%|██████▉   | 788/1135 [06:51<03:07,  1.85it/s, cl=0.0000, loss=1.1114]
2026-05-10 14:40:08
2026-05-10 07:40:08,350 - src.training.checkpoint_manager - INFO - Size OK: 765.21 MB (diff 0.00%)
2026-05-10 14:40:08
2026-05-10 07:40:08,350 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 5 -> vv5 confirmed (state=COMMITTED, 765.2 MB). Safe to close Colab.
2026-05-10 14:40:08
2026-05-10 07:40:08,440 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v8/epoch_004.pt
2026-05-10 14:40:08
2026-05-10 07:40:08,441 - src.training.trainer - INFO - Epoch 005 | train/loss=1.7536 | train/cl_loss=4.3591 | train/skipped_batches=0.0000 | train/lr=0.0008 | HR@10=0.0125 | NDCG@10=0.0036 | HR@20=0.0249 | NDCG@20=0.0054 | HR@50=0.0527 | NDCG@50=0.0085 | DIAG λ: L0_lambda_cart=0.697 L0_lambda_purchase=0.675 L0_lambda_struct=0.693 L0_lambda_view=0.726 L1_lambda_cart=0.701 L1_lambda_purchase=0.688 | DIAG z: baw0_zbeta_norm_cart=7.963 baw0_zbeta_norm_purchase=7.345 baw0_zbeta_norm_struct=8.000 baw0_zbeta_norm_view=7.421 baw1_zbeta_norm_cart=8.342 baw1_zbeta_norm_purchase=8.138
2026-05-10 14:40:08
epochs:  24%|██▍       | 6/25 [1:06:56<3:31:01, 666.40s/it, loss=1.7536, NDCG_20=0.0054, best_primary=0.0162]

## Model v9
### Config
data:
  data_dir: /content/data/
  node_counts:
    brand: 1919
    category: 14
    product: 29892
    user: 203063
  struct_dir: /content/data/node_mappings

model:
  dropout: 0.2
  embed_dim: 256
  n_layers: 3
  rank: 64
  use_grad_checkpoint: false
  n_intents: 64

sampler:
  hop1_budget: 16
  hop2_budget: 8
  hop1_sample_replace: true

loss:
  lambda_cl: 0.05      # ha tu 0.15: CL~4.3 * 0.15 = 0.65 ap dao BPR~2.0
  lambda_conv: 0.0     # tat funnel prior ban dau, bat lai sau ep ~5 khi BPR on dinh
  lambda_mono: 0.0     # tat mono prior ban dau (raw_lambda chua break symmetry)
  funnel_margin: 0.1
  alpha: 0.5
  w_min: 0.05

hierarchy_cl:
  enabled: true
  tau: 0.1
  hard_k: 64
  min_pair_overlap: 4
  pair_weights: null

training:
  amp: true
  use_bf16: true
  batch_size: 8192
  device: cuda
  epochs: 25
  eval_batch_size: 8192
  eval_every: 1
  eval_subsample: 60000
  eval_seed: 42
  l2_lambda: 1.0e-05
  lr: 8.0e-04
  min_lr: 1.0e-06
  warmup_epochs: 5     # tang tu 3 -> 5: cho BPR on dinh truoc khi prior bat
  max_grad_norm: 1.0   # ha tu 5.0: 5.0 gay phan ky o v8 (NDCG sup tu ep.2)
  cl_every_k: 2
  max_view_triplets: 3000000
  num_neg: 32
  num_workers: 8
  patience: 12
  save_dir: checkpoints-v9
  weight_decay: 1.0e-02
  pin_memory: true
  persistent_workers: true
  prefetch_factor: 4

evaluation:
  full_ranking: true
  primary_metric: "NDCG@20"
  ks: [10, 20, 50]
  metrics: ["HR@10", "HR@20", "HR@50", "NDCG@10", "NDCG@20", "NDCG@50"]

wandb:
  artifact_name: bpatmp-checkpoint-v9
  enabled: true
  entity: nguyenmaiductrong37-h-c-vi-n-c-ng-ngh-b-u-ch-nh-vi-n-th-ng
  project: bpatmp-recsys
  run_name: bpatmp-v9
  save_every: 1

a100:
  allow_tf32: true
  cudnn_benchmark: true
  use_fused_adamw: true
  compile_model: false
  empty_cache_freq: 0

### Log

2026-05-10 08:48:37,262 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 0 -> vv0 confirmed (state=COMMITTED, 764.7 MB). Safe to close Colab.
2026-05-10 15:48:37
2026-05-10 08:48:37,348 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v9/best.pt
2026-05-10 15:48:37
2026-05-10 08:48:37,349 - src.training.trainer - INFO - Epoch 000 | train/loss=2.0902 | train/cl_loss=4.3580 | train/skipped_batches=0.0000 | train/lr=0.0002 | HR@10=0.0007 | NDCG@10=0.0002 | HR@20=0.0008 | NDCG@20=0.0002 | HR@50=0.0016 | NDCG@50=0.0002  <- best | DIAG λ: L0_lambda_cart=0.692 L0_lambda_purchase=0.693 L0_lambda_struct=0.693 L0_lambda_view=0.694 L1_lambda_cart=0.693 L1_lambda_purchase=0.693 | DIAG z: baw0_zbeta_norm_cart=8.001 baw0_zbeta_norm_purchase=7.999 baw0_zbeta_norm_struct=8.000 baw0_zbeta_norm_view=8.000 baw1_zbeta_norm_cart=8.001 baw1_zbeta_norm_purchase=7.997
2026-05-10 15:48:37
epochs:   4%|▍         | 1/25 [21:27<4:28:43, 671.82s/it, loss=1.9225, NDCG_20=0.0171, best_primary=0.0171]2026-05-10 08:58:56,133 - src.training.checkpoint_manager - INFO - Saved local: checkpoints-v9/epoch_001.pt (764.7 MB)
2026-05-10 15:48:37
2026-05-10 08:59:01,039 - src.training.checkpoint_manager - INFO - Artifact enqueued: bpatmp-checkpoint-v9 epoch-001
2026-05-10 15:59:51
2026-05-10 08:59:51,307 - src.training.checkpoint_manager - INFO - Size OK: 764.68 MB (diff 0.00%)
2026-05-10 15:59:51
2026-05-10 08:59:51,308 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 1 -> vv1 confirmed (state=COMMITTED, 764.7 MB). Safe to close Colab.
2026-05-10 15:59:51
2026-05-10 08:59:51,396 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v9/epoch_000.pt
2026-05-10 15:59:51
2026-05-10 08:59:51,483 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v9/best.pt
2026-05-10 15:59:51
2026-05-10 08:59:51,483 - src.training.trainer - INFO - Epoch 001 | train/loss=1.9225 | train/cl_loss=4.3508 | train/skipped_batches=0.0000 | train/lr=0.0003 | HR@10=0.0426 | NDCG@10=0.0142 | HR@20=0.0570 | NDCG@20=0.0171 | HR@50=0.0981 | NDCG@50=0.0235  <- best | DIAG λ: L0_lambda_cart=0.693 L0_lambda_purchase=0.692 L0_lambda_struct=0.693 L0_lambda_view=0.696 L1_lambda_cart=0.691 L1_lambda_purchase=0.691 | DIAG z: baw0_zbeta_norm_cart=7.889 baw0_zbeta_norm_purchase=7.912 baw0_zbeta_norm_struct=8.000 baw0_zbeta_norm_view=7.933 baw1_zbeta_norm_cart=7.993 baw1_zbeta_norm_purchase=8.005
2026-05-10 15:59:51
epochs:   8%|▊         | 2/25 [32:40<4:18:03, 673.18s/it, loss=1.7237, NDCG_20=0.0088, best_primary=0.0171]2026-05-10 09:10:08,950 - src.training.checkpoint_manager - INFO - Saved local: checkpoints-v9/epoch_002.pt (764.7 MB)
2026-05-10 15:59:51
2026-05-10 09:10:11,084 - src.training.checkpoint_manager - INFO - Artifact enqueued: bpatmp-checkpoint-v9 epoch-002
2026-05-10 16:11:00
2026-05-10 09:11:00,789 - src.training.checkpoint_manager - INFO - Size OK: 764.68 MB (diff 0.00%)
2026-05-10 16:11:00
2026-05-10 09:11:00,790 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 2 -> vv2 confirmed (state=COMMITTED, 764.7 MB). Safe to close Colab.
2026-05-10 16:11:00
2026-05-10 09:11:00,878 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v9/epoch_001.pt
2026-05-10 16:11:00
2026-05-10 09:11:00,879 - src.training.trainer - INFO - Epoch 002 | train/loss=1.7237 | train/cl_loss=4.3547 | train/skipped_batches=0.0000 | train/lr=0.0005 | HR@10=0.0170 | NDCG@10=0.0073 | HR@20=0.0254 | NDCG@20=0.0088 | HR@50=0.0426 | NDCG@50=0.0112 | DIAG λ: L0_lambda_cart=0.692 L0_lambda_purchase=0.684 L0_lambda_struct=0.693 L0_lambda_view=0.704 L1_lambda_cart=0.691 L1_lambda_purchase=0.685 | DIAG z: baw0_zbeta_norm_cart=7.682 baw0_zbeta_norm_purchase=7.829 baw0_zbeta_norm_struct=8.000 baw0_zbeta_norm_view=7.864 baw1_zbeta_norm_cart=8.027 baw1_zbeta_norm_purchase=8.094
2026-05-10 16:11:00
epochs:  12%|█▏        | 3/25 [43:48<4:06:11, 671.45s/it, loss=1.5771, NDCG_20=0.0667, best_primary=0.0667]2026-05-10 09:21:16,956 - src.training.checkpoint_manager - INFO - Saved local: checkpoints-v9/epoch_003.pt (764.7 MB)
2026-05-10 16:11:00
2026-05-10 09:21:19,298 - src.training.checkpoint_manager - INFO - Artifact enqueued: bpatmp-checkpoint-v9 epoch-003
2026-05-10 16:22:03
2026-05-10 09:22:03,858 - src.training.checkpoint_manager - INFO - Size OK: 764.68 MB (diff 0.00%)
2026-05-10 16:22:03
2026-05-10 09:22:03,859 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 3 -> vv3 confirmed (state=COMMITTED, 764.7 MB). Safe to close Colab.
2026-05-10 16:22:03
2026-05-10 09:22:03,949 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v9/epoch_002.pt
2026-05-10 16:22:04
2026-05-10 09:22:04,041 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v9/best.pt
2026-05-10 16:22:04
2026-05-10 09:22:04,041 - src.training.trainer - INFO - Epoch 003 | train/loss=1.5771 | train/cl_loss=4.3598 | train/skipped_batches=0.0000 | train/lr=0.0006 | HR@10=0.1436 | NDCG@10=0.0540 | HR@20=0.2056 | NDCG@20=0.0667 | HR@50=0.2915 | NDCG@50=0.0832  <- best | DIAG λ: L0_lambda_cart=0.705 L0_lambda_purchase=0.683 L0_lambda_struct=0.693 L0_lambda_view=0.699 L1_lambda_cart=0.690 L1_lambda_purchase=0.683 | DIAG z: baw0_zbeta_norm_cart=7.757 baw0_zbeta_norm_purchase=7.670 baw0_zbeta_norm_struct=8.000 baw0_zbeta_norm_view=7.743 baw1_zbeta_norm_cart=8.078 baw1_zbeta_norm_purchase=8.200
2026-05-10 16:22:04
epochs:  16%|█▌        | 4/25 [54:56<3:53:51, 668.18s/it, loss=1.4615, NDCG_20=0.0617, best_primary=0.0667]2026-05-10 09:32:24,325 - src.training.checkpoint_manager - INFO - Saved local: checkpoints-v9/epoch_004.pt (764.7 MB)
2026-05-10 16:22:04
2026-05-10 09:32:27,145 - src.training.checkpoint_manager - INFO - Artifact enqueued: bpatmp-checkpoint-v9 epoch-004
2026-05-10 16:33:12
2026-05-10 09:33:12,414 - src.training.checkpoint_manager - INFO - Size OK: 764.68 MB (diff 0.00%)
2026-05-10 16:33:12
2026-05-10 09:33:12,415 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 4 -> vv4 confirmed (state=COMMITTED, 764.7 MB). Safe to close Colab.
2026-05-10 16:33:12
2026-05-10 09:33:12,503 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v9/epoch_003.pt
2026-05-10 16:33:12
2026-05-10 09:33:12,504 - src.training.trainer - INFO - Epoch 004 | train/loss=1.4615 | train/cl_loss=4.3545 | train/skipped_batches=0.0000 | train/lr=0.0008 | HR@10=0.1362 | NDCG@10=0.0518 | HR@20=0.1870 | NDCG@20=0.0617 | HR@50=0.2703 | NDCG@50=0.0769 | DIAG λ: L0_lambda_cart=0.717 L0_lambda_purchase=0.683 L0_lambda_struct=0.693 L0_lambda_view=0.700 L1_lambda_cart=0.686 L1_lambda_purchase=0.686 | DIAG z: baw0_zbeta_norm_cart=7.976 baw0_zbeta_norm_purchase=7.433 baw0_zbeta_norm_struct=8.000 baw0_zbeta_norm_view=7.843 baw1_zbeta_norm_cart=8.098 baw1_zbeta_norm_purchase=8.141
2026-05-10 16:33:12
epochs:  20%|██        | 5/25 [1:06:01<3:42:45, 668.28s/it, loss=1.3788, NDCG_20=0.0631, best_primary=0.0667]2026-05-10 09:43:30,132 - src.training.checkpoint_manager - INFO - Saved local: checkpoints-v9/epoch_005.pt (764.7 MB)
2026-05-10 16:33:12
2026-05-10 09:43:32,279 - src.training.checkpoint_manager - INFO - Artifact enqueued: bpatmp-checkpoint-v9 epoch-005
2026-05-10 16:44:20
2026-05-10 09:44:20,348 - src.training.checkpoint_manager - INFO - Size OK: 764.68 MB (diff 0.00%)
2026-05-10 16:44:20
2026-05-10 09:44:20,349 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 5 -> vv5 confirmed (state=COMMITTED, 764.7 MB). Safe to close Colab.
2026-05-10 16:44:20
2026-05-10 09:44:20,443 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v9/epoch_004.pt
2026-05-10 16:44:20
2026-05-10 09:44:20,443 - src.training.trainer - INFO - Epoch 005 | train/loss=1.3788 | train/cl_loss=4.3469 | train/skipped_batches=0.0000 | train/lr=0.0008 | HR@10=0.1397 | NDCG@10=0.0529 | HR@20=0.1922 | NDCG@20=0.0631 | HR@50=0.2815 | NDCG@50=0.0789 | DIAG λ: L0_lambda_cart=0.723 L0_lambda_purchase=0.684 L0_lambda_struct=0.693 L0_lambda_view=0.697 L1_lambda_cart=0.682 L1_lambda_purchase=0.684 | DIAG z: baw0_zbeta_norm_cart=7.920 baw0_zbeta_norm_purchase=7.261 baw0_zbeta_norm_struct=8.000 baw0_zbeta_norm_view=7.744 baw1_zbeta_norm_cart=8.004 baw1_zbeta_norm_purchase=8.134
2026-05-10 16:44:20
epochs:  24%|██▍       | 6/25 [1:17:09<3:31:35, 668.17s/it, loss=1.3158, NDCG_20=0.0679, best_primary=0.0679]2026-05-10 09:54:38,498 - src.training.checkpoint_manager - INFO - Saved local: checkpoints-v9/epoch_006.pt (764.7 MB)
2026-05-10 16:44:20
2026-05-10 09:54:40,642 - src.training.checkpoint_manager - INFO - Artifact enqueued: bpatmp-checkpoint-v9 epoch-006
2026-05-10 16:55:24
2026-05-10 09:55:24,508 - src.training.checkpoint_manager - INFO - Size OK: 764.68 MB (diff 0.00%)
2026-05-10 16:55:24
2026-05-10 09:55:24,509 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 6 -> vv6 confirmed (state=COMMITTED, 764.7 MB). Safe to close Colab.
2026-05-10 16:55:24
2026-05-10 09:55:24,600 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v9/epoch_005.pt
2026-05-10 16:55:24
2026-05-10 09:55:24,694 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v9/best.pt
2026-05-10 16:55:24
2026-05-10 09:55:24,695 - src.training.trainer - INFO - Epoch 006 | train/loss=1.3158 | train/cl_loss=4.3543 | train/skipped_batches=0.0000 | train/lr=0.0008 | HR@10=0.1517 | NDCG@10=0.0568 | HR@20=0.2073 | NDCG@20=0.0679 | HR@50=0.2969 | NDCG@50=0.0843  <- best | DIAG λ: L0_lambda_cart=0.727 L0_lambda_purchase=0.683 L0_lambda_struct=0.693 L0_lambda_view=0.709 L1_lambda_cart=0.683 L1_lambda_purchase=0.686 | DIAG z: baw0_zbeta_norm_cart=7.846 baw0_zbeta_norm_purchase=7.172 baw0_zbeta_norm_struct=8.000 baw0_zbeta_norm_view=7.734 baw1_zbeta_norm_cart=8.091 baw1_zbeta_norm_purchase=8.140
2026-05-10 16:55:24
epochs:  28%|██▊       | 7/25 [1:28:10<3:20:03, 666.89s/it, loss=1.2702, NDCG_20=0.0601, best_primary=0.0679]2026-05-10 10:05:38,595 - src.training.checkpoint_manager - INFO - Saved local: checkpoints-v9/epoch_007.pt (764.7 MB)
2026-05-10 16:55:24
2026-05-10 10:05:40,726 - src.training.checkpoint_manager - INFO - Artifact enqueued: bpatmp-checkpoint-v9 epoch-007
2026-05-10 17:06:27
2026-05-10 10:06:27,355 - src.training.checkpoint_manager - INFO - Size OK: 764.68 MB (diff 0.00%)
2026-05-10 17:06:27
2026-05-10 10:06:27,355 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 7 -> vv7 confirmed (state=COMMITTED, 764.7 MB). Safe to close Colab.
2026-05-10 17:06:27
2026-05-10 10:06:27,444 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v9/epoch_006.pt
2026-05-10 17:06:27
2026-05-10 10:06:27,444 - src.training.trainer - INFO - Epoch 007 | train/loss=1.2702 | train/cl_loss=4.3515 | train/skipped_batches=0.0000 | train/lr=0.0008 | HR@10=0.1322 | NDCG@10=0.0509 | HR@20=0.1797 | NDCG@20=0.0601 | HR@50=0.2567 | NDCG@50=0.0737 | DIAG λ: L0_lambda_cart=0.686 L0_lambda_purchase=0.723 L0_lambda_struct=0.693 L0_lambda_view=0.727 L1_lambda_cart=0.681 L1_lambda_purchase=0.675 | DIAG z: baw0_zbeta_norm_cart=7.626 baw0_zbeta_norm_purchase=7.122 baw0_zbeta_norm_struct=8.000 baw0_zbeta_norm_view=7.695 baw1_zbeta_norm_cart=8.211 baw1_zbeta_norm_purchase=8.163
2026-05-10 17:06:27
epochs:  32%|███▏      | 8/25 [1:29:01<3:08:34, 665.57s/it, loss=1.2702, NDCG_20=0.0601, best_primary=0.0679]
2026-05-10 17:06:27
train:  12%|█▏        | 136/1135 [01:11<08:58,  1.86it/s, cl=0.0000, loss=1.0098]
2026-05-10 17:17:32
2026-05-10 10:17:32,755 - src.training.checkpoint_manager - INFO - Size OK: 764.68 MB (diff 0.00%)
2026-05-10 17:17:32
2026-05-10 10:17:32,755 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 8 -> vv8 confirmed (state=COMMITTED, 764.7 MB). Safe to close Colab.
2026-05-10 17:17:32
2026-05-10 10:17:32,851 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v9/epoch_007.pt
2026-05-10 17:17:32
2026-05-10 10:17:32,851 - src.training.trainer - INFO - Epoch 008 | train/loss=1.2328 | train/cl_loss=4.3556 | train/skipped_batches=0.0000 | train/lr=0.0007 | HR@10=0.1036 | NDCG@10=0.0390 | HR@20=0.1494 | NDCG@20=0.0477 | HR@50=0.2185 | NDCG@50=0.0597 | DIAG λ: L0_lambda_cart=0.687 L0_lambda_purchase=0.721 L0_lambda_struct=0.693 L0_lambda_view=0.728 L1_lambda_cart=0.687 L1_lambda_purchase=0.680 | DIAG z: baw0_zbeta_norm_cart=7.513 baw0_zbeta_norm_purchase=7.191 baw0_zbeta_norm_struct=8.000 baw0_zbeta_norm_view=7.692 baw1_zbeta_norm_cart=8.253 baw1_zbeta_norm_purchase=8.380
2026-05-10 17:17:32
epochs:  36%|███▌      | 9/25 [1:40:07<2:57:28, 665.52s/it, loss=1.2328, NDCG_20=0.0477, best_primary=0.0679]

## Model v10
### Config
data:
  data_dir: /content/data/
  node_counts:
    brand: 1919
    category: 14
    product: 29892
    user: 203063
  struct_dir: /content/data/node_mappings

model:
  dropout: 0.2
  embed_dim: 256
  n_layers: 3
  rank: 64
  use_grad_checkpoint: false
  n_intents: 64

sampler:
  hop1_budget: 16
  hop2_budget: 8
  hop1_sample_replace: true

loss:
  lambda_cl: 0.10      # compromise giua v3 (0.15) va v10 (0.05). v10 chung minh 0.05 qua yeu.
  lambda_conv: 0.05    # bat lai funnel prior nhung yeu hon v3 (0.1)
  lambda_mono: 0.025   # bat lai mono prior. raw_lambda gio init bat doi xung roi -> mono co cho neo.
  funnel_margin: 0.1
  alpha: 0.5
  w_min: 0.05

hierarchy_cl:
  enabled: true
  tau: 0.1
  hard_k: 64
  min_pair_overlap: 4
  pair_weights: null

training:
  amp: true
  use_bf16: true
  batch_size: 8192
  device: cuda
  epochs: 25
  eval_batch_size: 8192
  eval_every: 1
  eval_subsample: 60000
  eval_seed: 42
  l2_lambda: 1.0e-05
  lr: 8.0e-04
  min_lr: 1.0e-06
  warmup_epochs: 5     # tang tu 3 -> 5: cho BPR on dinh truoc khi prior bat
  max_grad_norm: 1.0   # ha tu 5.0: 5.0 gay phan ky o v8 (NDCG sup tu ep.2)
  cl_every_k: 2
  max_view_triplets: 3000000
  num_neg: 32
  num_workers: 8
  patience: 12
  save_dir: checkpoints-v10
  weight_decay: 1.0e-02
  pin_memory: true
  persistent_workers: true
  prefetch_factor: 4

evaluation:
  full_ranking: true
  primary_metric: "NDCG@20"
  ks: [10, 20, 50]
  metrics: ["HR@10", "HR@20", "HR@50", "NDCG@10", "NDCG@20", "NDCG@50"]

wandb:
  artifact_name: bpatmp-checkpoint-v10
  enabled: true
  entity: nguyenmaiductrong37-h-c-vi-n-c-ng-ngh-b-u-ch-nh-vi-n-th-ng
  project: bpatmp-recsys
  run_name: bpatmp-v10
  save_every: 1

a100:
  allow_tf32: true
  cudnn_benchmark: true
  use_fused_adamw: true
  compile_model: false
  empty_cache_freq: 0
### Log
2026-05-10 17:40:42
2026-05-10 10:40:42,374 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 0 -> vv0 confirmed (state=COMMITTED, 765.2 MB). Safe to close Colab.
2026-05-10 17:40:42
2026-05-10 10:40:42,461 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v10/best.pt
2026-05-10 17:40:42
2026-05-10 10:40:42,462 - src.training.trainer - INFO - Epoch 000 | train/loss=2.3093 | train/cl_loss=4.3623 | train/skipped_batches=0.0000 | train/lr=0.0002 | HR@10=0.0035 | NDCG@10=0.0009 | HR@20=0.0044 | NDCG@20=0.0011 | HR@50=0.0061 | NDCG@50=0.0013  <- best | DIAG λ: L0_lambda_cart=0.974 L0_lambda_purchase=0.693 L0_lambda_struct=0.693 L0_lambda_view=1.313 L1_lambda_cart=0.973 L1_lambda_purchase=0.693 | DIAG z: baw0_zbeta_norm_cart=8.000 baw0_zbeta_norm_purchase=8.000 baw0_zbeta_norm_struct=8.000 baw0_zbeta_norm_view=7.999 baw1_zbeta_norm_cart=8.000 baw1_zbeta_norm_purchase=7.996
2026-05-10 17:40:42
epochs:   4%|▍         | 1/25 [21:22<4:29:06, 672.76s/it, loss=2.1397, NDCG_20=0.0166, best_primary=0.0166]2026-05-10 10:50:58,952 - src.training.checkpoint_manager - INFO - Saved local: checkpoints-v10/epoch_001.pt (765.2 MB)
2026-05-10 17:40:42
2026-05-10 10:51:01,462 - src.training.checkpoint_manager - INFO - Artifact enqueued: bpatmp-checkpoint-v10 epoch-001
2026-05-10 17:51:51
2026-05-10 10:51:51,772 - src.training.checkpoint_manager - INFO - Size OK: 765.21 MB (diff 0.00%)
2026-05-10 17:51:51
2026-05-10 10:51:51,772 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 1 -> vv1 confirmed (state=COMMITTED, 765.2 MB). Safe to close Colab.
2026-05-10 17:51:51
2026-05-10 10:51:51,861 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v10/epoch_000.pt
2026-05-10 17:51:51
2026-05-10 10:51:51,949 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v10/best.pt
2026-05-10 17:51:51
2026-05-10 10:51:51,949 - src.training.trainer - INFO - Epoch 001 | train/loss=2.1397 | train/cl_loss=4.3551 | train/skipped_batches=0.0000 | train/lr=0.0003 | HR@10=0.0415 | NDCG@10=0.0149 | HR@20=0.0500 | NDCG@20=0.0166 | HR@50=0.0597 | NDCG@50=0.0180  <- best | DIAG λ: L0_lambda_cart=0.978 L0_lambda_purchase=0.685 L0_lambda_struct=0.693 L0_lambda_view=1.324 L1_lambda_cart=0.975 L1_lambda_purchase=0.688 | DIAG z: baw0_zbeta_norm_cart=7.929 baw0_zbeta_norm_purchase=7.895 baw0_zbeta_norm_struct=8.000 baw0_zbeta_norm_view=7.947 baw1_zbeta_norm_cart=8.002 baw1_zbeta_norm_purchase=7.992
2026-05-10 17:51:51
epochs:   8%|▊         | 2/25 [32:35<4:17:09, 670.83s/it, loss=1.9244, NDCG_20=0.0430, best_primary=0.0430]2026-05-10 11:02:12,723 - src.training.checkpoint_manager - INFO - Saved local: checkpoints-v10/epoch_002.pt (765.2 MB)
2026-05-10 17:51:51
2026-05-10 11:02:15,145 - src.training.checkpoint_manager - INFO - Artifact enqueued: bpatmp-checkpoint-v10 epoch-002
2026-05-10 18:03:03
2026-05-10 11:03:03,531 - src.training.checkpoint_manager - INFO - Size OK: 765.21 MB (diff 0.00%)
2026-05-10 18:03:03
2026-05-10 11:03:03,531 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 2 -> vv2 confirmed (state=COMMITTED, 765.2 MB). Safe to close Colab.
2026-05-10 18:03:03
2026-05-10 11:03:03,620 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v10/epoch_001.pt
2026-05-10 18:03:03
2026-05-10 11:03:03,714 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v10/best.pt
2026-05-10 18:03:03
2026-05-10 11:03:03,714 - src.training.trainer - INFO - Epoch 002 | train/loss=1.9244 | train/cl_loss=4.3610 | train/skipped_batches=0.0000 | train/lr=0.0005 | HR@10=0.0958 | NDCG@10=0.0344 | HR@20=0.1372 | NDCG@20=0.0430 | HR@50=0.1956 | NDCG@50=0.0535  <- best | DIAG λ: L0_lambda_cart=0.977 L0_lambda_purchase=0.677 L0_lambda_struct=0.693 L0_lambda_view=1.334 L1_lambda_cart=0.972 L1_lambda_purchase=0.690 | DIAG z: baw0_zbeta_norm_cart=7.767 baw0_zbeta_norm_purchase=7.778 baw0_zbeta_norm_struct=8.000 baw0_zbeta_norm_view=7.869 baw1_zbeta_norm_cart=8.096 baw1_zbeta_norm_purchase=8.130
2026-05-10 18:03:03
epochs:  12%|█▏        | 3/25 [43:44<4:06:07, 671.26s/it, loss=1.7797, NDCG_20=0.0519, best_primary=0.0519]2026-05-10 11:13:18,762 - src.training.checkpoint_manager - INFO - Saved local: checkpoints-v10/epoch_003.pt (765.2 MB)
2026-05-10 18:03:03
2026-05-10 11:13:21,000 - src.training.checkpoint_manager - INFO - Artifact enqueued: bpatmp-checkpoint-v10 epoch-003
2026-05-10 18:14:11
2026-05-10 11:14:11,811 - src.training.checkpoint_manager - INFO - Size OK: 765.21 MB (diff 0.00%)
2026-05-10 18:14:11
2026-05-10 11:14:11,812 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 3 -> vv3 confirmed (state=COMMITTED, 765.2 MB). Safe to close Colab.
2026-05-10 18:14:11
2026-05-10 11:14:11,899 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v10/epoch_002.pt
2026-05-10 18:14:11
2026-05-10 11:14:11,994 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v10/best.pt
2026-05-10 18:14:11
2026-05-10 11:14:11,994 - src.training.trainer - INFO - Epoch 003 | train/loss=1.7797 | train/cl_loss=4.3538 | train/skipped_batches=0.0000 | train/lr=0.0006 | HR@10=0.1172 | NDCG@10=0.0425 | HR@20=0.1643 | NDCG@20=0.0519 | HR@50=0.2324 | NDCG@50=0.0645  <- best | DIAG λ: L0_lambda_cart=0.986 L0_lambda_purchase=0.664 L0_lambda_struct=0.693 L0_lambda_view=1.342 L1_lambda_cart=0.977 L1_lambda_purchase=0.684 | DIAG z: baw0_zbeta_norm_cart=7.385 baw0_zbeta_norm_purchase=7.497 baw0_zbeta_norm_struct=8.000 baw0_zbeta_norm_view=7.750 baw1_zbeta_norm_cart=8.186 baw1_zbeta_norm_purchase=8.236
2026-05-10 18:14:11
epochs:  16%|█▌        | 4/25 [54:55<3:54:31, 670.08s/it, loss=1.6681, NDCG_20=0.0581, best_primary=0.0581]2026-05-10 11:24:31,715 - src.training.checkpoint_manager - INFO - Saved local: checkpoints-v10/epoch_004.pt (765.2 MB)
2026-05-10 18:14:11
2026-05-10 11:24:34,040 - src.training.checkpoint_manager - INFO - Artifact enqueued: bpatmp-checkpoint-v10 epoch-004
2026-05-10 18:25:20
2026-05-10 11:25:20,209 - src.training.checkpoint_manager - INFO - Size OK: 765.21 MB (diff 0.00%)
2026-05-10 18:25:20
2026-05-10 11:25:20,209 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 4 -> vv4 confirmed (state=COMMITTED, 765.2 MB). Safe to close Colab.
2026-05-10 18:25:20
2026-05-10 11:25:20,300 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v10/epoch_003.pt
2026-05-10 18:25:20
2026-05-10 11:25:20,395 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v10/best.pt
2026-05-10 18:25:20
2026-05-10 11:25:20,396 - src.training.trainer - INFO - Epoch 004 | train/loss=1.6681 | train/cl_loss=4.3533 | train/skipped_batches=0.0000 | train/lr=0.0008 | HR@10=0.1316 | NDCG@10=0.0476 | HR@20=0.1859 | NDCG@20=0.0581 | HR@50=0.2727 | NDCG@50=0.0736  <- best | DIAG λ: L0_lambda_cart=0.985 L0_lambda_purchase=0.658 L0_lambda_struct=0.693 L0_lambda_view=1.357 L1_lambda_cart=0.978 L1_lambda_purchase=0.684 | DIAG z: baw0_zbeta_norm_cart=7.216 baw0_zbeta_norm_purchase=7.440 baw0_zbeta_norm_struct=8.000 baw0_zbeta_norm_view=7.743 baw1_zbeta_norm_cart=8.245 baw1_zbeta_norm_purchase=8.287
2026-05-10 18:25:20
epochs:  20%|██        | 5/25 [1:06:02<3:43:09, 669.48s/it, loss=1.5940, NDCG_20=0.0665, best_primary=0.0665]2026-05-10 11:35:35,292 - src.training.checkpoint_manager - INFO - Saved local: checkpoints-v10/epoch_005.pt (765.2 MB)
2026-05-10 18:25:20
2026-05-10 11:35:37,406 - src.training.checkpoint_manager - INFO - Artifact enqueued: bpatmp-checkpoint-v10 epoch-005
2026-05-10 18:36:24
2026-05-10 11:36:24,531 - src.training.checkpoint_manager - INFO - Size OK: 765.21 MB (diff 0.00%)
2026-05-10 18:36:24
2026-05-10 11:36:24,531 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 5 -> vv5 confirmed (state=COMMITTED, 765.2 MB). Safe to close Colab.
2026-05-10 18:36:24
2026-05-10 11:36:24,627 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v10/best.pt
2026-05-10 18:36:24
2026-05-10 11:36:24,714 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v10/epoch_004.pt
2026-05-10 18:36:24
2026-05-10 11:36:24,715 - src.training.trainer - INFO - Epoch 005 | train/loss=1.5940 | train/cl_loss=4.3591 | train/skipped_batches=0.0000 | train/lr=0.0008 | HR@10=0.1510 | NDCG@10=0.0534 | HR@20=0.2146 | NDCG@20=0.0665 | HR@50=0.3062 | NDCG@50=0.0835  <- best | DIAG λ: L0_lambda_cart=0.985 L0_lambda_purchase=0.661 L0_lambda_struct=0.693 L0_lambda_view=1.350 L1_lambda_cart=0.977 L1_lambda_purchase=0.682 | DIAG z: baw0_zbeta_norm_cart=7.069 baw0_zbeta_norm_purchase=7.361 baw0_zbeta_norm_struct=8.000 baw0_zbeta_norm_view=7.694 baw1_zbeta_norm_cart=8.338 baw1_zbeta_norm_purchase=8.217
2026-05-10 18:36:24
epochs:  24%|██▍       | 6/25 [1:17:07<3:31:26, 667.72s/it, loss=1.5353, NDCG_20=0.0606, best_primary=0.0665]2026-05-10 11:46:40,014 - src.training.checkpoint_manager - INFO - Saved local: checkpoints-v10/epoch_006.pt (765.2 MB)
2026-05-10 18:36:24
2026-05-10 11:46:42,135 - src.training.checkpoint_manager - INFO - Artifact enqueued: bpatmp-checkpoint-v10 epoch-006
2026-05-10 18:47:37
2026-05-10 11:47:37,770 - src.training.checkpoint_manager - INFO - Size OK: 765.21 MB (diff 0.00%)
2026-05-10 18:47:37
2026-05-10 11:47:37,771 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 6 -> vv6 confirmed (state=COMMITTED, 765.2 MB). Safe to close Colab.
2026-05-10 18:47:37
2026-05-10 11:47:37,861 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v10/epoch_005.pt
2026-05-10 18:47:37
2026-05-10 11:47:37,861 - src.training.trainer - INFO - Epoch 006 | train/loss=1.5353 | train/cl_loss=4.3527 | train/skipped_batches=0.0000 | train/lr=0.0008 | HR@10=0.1347 | NDCG@10=0.0512 | HR@20=0.1836 | NDCG@20=0.0606 | HR@50=0.2644 | NDCG@50=0.0749 | DIAG λ: L0_lambda_cart=0.983 L0_lambda_purchase=0.651 L0_lambda_struct=0.693 L0_lambda_view=1.372 L1_lambda_cart=0.976 L1_lambda_purchase=0.675 | DIAG z: baw0_zbeta_norm_cart=6.916 baw0_zbeta_norm_purchase=7.341 baw0_zbeta_norm_struct=8.000 baw0_zbeta_norm_view=7.663 baw1_zbeta_norm_cart=8.374 baw1_zbeta_norm_purchase=8.159
2026-05-10 18:47:37
epochs:  28%|██▊       | 7/25 [1:28:16<3:20:50, 669.50s/it, loss=1.4964, NDCG_20=0.0645, best_primary=0.0665]2026-05-10 11:57:49,007 - src.training.checkpoint_manager - INFO - Saved local: checkpoints-v10/epoch_007.pt (765.2 MB)
2026-05-10 18:47:37
2026-05-10 11:57:51,177 - src.training.checkpoint_manager - INFO - Artifact enqueued: bpatmp-checkpoint-v10 epoch-007
2026-05-10 18:58:42
2026-05-10 11:58:42,670 - src.training.checkpoint_manager - INFO - Size OK: 765.21 MB (diff 0.00%)
2026-05-10 18:58:42
2026-05-10 11:58:42,671 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 7 -> vv7 confirmed (state=COMMITTED, 765.2 MB). Safe to close Colab.
2026-05-10 18:58:42
2026-05-10 11:58:42,775 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v10/epoch_006.pt
2026-05-10 18:58:42
2026-05-10 11:58:42,775 - src.training.trainer - INFO - Epoch 007 | train/loss=1.4964 | train/cl_loss=4.3585 | train/skipped_batches=0.0000 | train/lr=0.0008 | HR@10=0.1448 | NDCG@10=0.0547 | HR@20=0.1943 | NDCG@20=0.0645 | HR@50=0.2840 | NDCG@50=0.0803 | DIAG λ: L0_lambda_cart=0.983 L0_lambda_purchase=0.650 L0_lambda_struct=0.693 L0_lambda_view=1.373 L1_lambda_cart=0.978 L1_lambda_purchase=0.674 | DIAG z: baw0_zbeta_norm_cart=6.836 baw0_zbeta_norm_purchase=7.310 baw0_zbeta_norm_struct=8.000 baw0_zbeta_norm_view=7.655 baw1_zbeta_norm_cart=8.437 baw1_zbeta_norm_purchase=8.092
2026-05-10 18:58:42
epochs:  32%|███▏      | 8/25 [1:39:22<3:09:16, 668.04s/it, loss=1.4663, NDCG_20=0.0511, best_primary=0.0665]2026-05-10 12:08:55,053 - src.training.checkpoint_manager - INFO - Saved local: checkpoints-v10/epoch_008.pt (765.2 MB)
2026-05-10 18:58:42
2026-05-10 12:08:57,232 - src.training.checkpoint_manager - INFO - Artifact enqueued: bpatmp-checkpoint-v10 epoch-008
2026-05-10 19:09:44
2026-05-10 12:09:44,984 - src.training.checkpoint_manager - INFO - Size OK: 765.21 MB (diff 0.00%)
2026-05-10 19:09:44
2026-05-10 12:09:44,984 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 8 -> vv8 confirmed (state=COMMITTED, 765.2 MB). Safe to close Colab.
2026-05-10 19:09:45
2026-05-10 12:09:45,076 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v10/epoch_007.pt
2026-05-10 19:09:45
2026-05-10 12:09:45,076 - src.training.trainer - INFO - Epoch 008 | train/loss=1.4663 | train/cl_loss=4.3608 | train/skipped_batches=0.0000 | train/lr=0.0007 | HR@10=0.1146 | NDCG@10=0.0453 | HR@20=0.1451 | NDCG@20=0.0511 | HR@50=0.1960 | NDCG@50=0.0596 | DIAG λ: L0_lambda_cart=0.990 L0_lambda_purchase=0.675 L0_lambda_struct=0.693 L0_lambda_view=1.331 L1_lambda_cart=0.974 L1_lambda_purchase=0.657 | DIAG z: baw0_zbeta_norm_cart=6.669 baw0_zbeta_norm_purchase=6.902 baw0_zbeta_norm_struct=8.000 baw0_zbeta_norm_view=7.561 baw1_zbeta_norm_cart=8.503 baw1_zbeta_norm_purchase=8.211
2026-05-10 19:09:45
epochs:  36%|███▌      | 9/25 [1:50:28<2:57:39, 666.24s/it, loss=1.4382, NDCG_20=0.0543, best_primary=0.0665]2026-05-10 12:20:01,260 - src.training.checkpoint_manager - INFO - Saved local: checkpoints-v10/epoch_009.pt (765.2 MB)
2026-05-10 19:09:45
2026-05-10 12:20:03,434 - src.training.checkpoint_manager - INFO - Artifact enqueued: bpatmp-checkpoint-v10 epoch-009
2026-05-10 19:20:51
2026-05-10 12:20:51,997 - src.training.checkpoint_manager - INFO - Size OK: 765.21 MB (diff 0.00%)
2026-05-10 19:20:51
2026-05-10 12:20:51,997 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 9 -> vv9 confirmed (state=COMMITTED, 765.2 MB). Safe to close Colab.
2026-05-10 19:20:52
2026-05-10 12:20:52,093 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v10/epoch_008.pt
2026-05-10 19:20:52
2026-05-10 12:20:52,093 - src.training.trainer - INFO - Epoch 009 | train/loss=1.4382 | train/cl_loss=4.3547 | train/skipped_batches=0.0000 | train/lr=0.0007 | HR@10=0.1208 | NDCG@10=0.0467 | HR@20=0.1602 | NDCG@20=0.0543 | HR@50=0.2227 | NDCG@50=0.0651 | DIAG λ: L0_lambda_cart=0.991 L0_lambda_purchase=0.674 L0_lambda_struct=0.693 L0_lambda_view=1.331 L1_lambda_cart=0.972 L1_lambda_purchase=0.664 | DIAG z: baw0_zbeta_norm_cart=6.671 baw0_zbeta_norm_purchase=6.976 baw0_zbeta_norm_struct=8.000 baw0_zbeta_norm_view=7.574 baw1_zbeta_norm_cart=8.503 baw1_zbeta_norm_purchase=8.279
2026-05-10 19:20:52
epochs:  40%|████      | 10/25 [1:51:22<2:46:37, 666.48s/it, loss=1.4382, NDCG_20=0.0543, best_primary=0.0665]
2026-05-10 19:20:52
train:  13%|█▎        | 150/1135 [01:20<08:30,  1.93it/s, cl=0.0000, loss=0.9554]

## Model v12
### Config
data:
  data_dir: /content/data/
  node_counts:
    brand: 1919
    category: 14
    product: 29892
    user: 203063
  struct_dir: /content/data/node_mappings

model:
  dropout: 0.2
  embed_dim: 256
  n_layers: 3
  rank: 64
  use_grad_checkpoint: true
  n_intents: 64

sampler:
  hop1_budget: 16
  hop2_budget: 8
  hop1_sample_replace: true

loss:
  lambda_cl: 0.15
  lambda_conv: 0.1
  lambda_mono: 0.05
  funnel_margin: 0.1
  alpha: 0.5
  w_min: 0.05

hierarchy_cl:
  enabled: true
  tau: 0.1
  hard_k: 64
  min_pair_overlap: 4
  pair_weights: null

training:
  amp: true
  use_bf16: true
  batch_size: 8192
  device: cuda
  epochs: 25
  eval_batch_size: 8192
  eval_every: 1
  eval_subsample: 60000
  eval_seed: 42
  l2_lambda: 1.0e-05
  lr: 1.0e-04
  min_lr: 1.0e-06
  warmup_epochs: 5
  max_grad_norm: 1.0
  cl_every_k: 2
  max_view_triplets: 3000000
  num_neg: 32
  num_workers: 8
  patience: 12
  save_dir: checkpoints-v12
  weight_decay: 1.0e-02
  pin_memory: true
  persistent_workers: true
  prefetch_factor: 4

evaluation:
  full_ranking: true
  primary_metric: "NDCG@20"
  ks: [10, 20, 50]
  metrics: ["HR@10", "HR@20", "HR@50", "NDCG@10", "NDCG@20", "NDCG@50"]

wandb:
  artifact_name: bpatmp-checkpoint-v12
  enabled: true
  entity: nguyenmaiductrong37-h-c-vi-n-c-ng-ngh-b-u-ch-nh-vi-n-th-ng
  project: bpatmp-recsys
  run_name: bpatmp-v12
  save_every: 1

a100:
  allow_tf32: true
  cudnn_benchmark: true
  use_fused_adamw: true
  compile_model: false
  empty_cache_freq: 0
### Log
2026-05-11 00:13:16
2026-05-10 17:13:16,204 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 0 -> vv6 confirmed (state=COMMITTED, 771.9 MB). Safe to close Colab.
2026-05-11 00:13:16
2026-05-10 17:13:16,293 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v12/best.pt
2026-05-11 00:13:16
2026-05-10 17:13:16,294 - src.training.trainer - INFO - Epoch 000 | train/loss=2.5348 | train/cl_loss=4.3541 | train/skipped_batches=0.0000 | train/lr=0.0000 | HR@10=0.0025 | NDCG@10=0.0007 | HR@20=0.0043 | NDCG@20=0.0010 | HR@50=0.0090 | NDCG@50=0.0016  <- best | DIAG λ: L0_lambda_cart=0.974 L0_lambda_purchase=0.693 L0_lambda_struct=0.693 L0_lambda_view=1.314 L1_lambda_cart=0.974 L1_lambda_purchase=0.693 | DIAG z: baw0_zbeta_norm_cart=8.000 baw0_zbeta_norm_purchase=8.000 baw0_zbeta_norm_struct=8.000 baw0_zbeta_norm_view=8.000 baw1_zbeta_norm_cart=8.000 baw1_zbeta_norm_purchase=8.001
2026-05-11 00:13:16
epochs:   4%|▍         | 1/25 [32:18<6:38:23, 995.99s/it, loss=2.3263, NDCG_20=0.0110, best_primary=0.0110]2026-05-10 17:29:07,018 - src.training.checkpoint_manager - INFO - Saved local: checkpoints-v12/epoch_001.pt (771.9 MB)
2026-05-11 00:13:16
2026-05-10 17:29:10,917 - src.training.checkpoint_manager - INFO - Artifact enqueued: bpatmp-checkpoint-v12 epoch-001
2026-05-11 00:29:57
2026-05-10 17:29:57,169 - src.training.checkpoint_manager - INFO - Size OK: 771.94 MB (diff 0.00%)
2026-05-11 00:29:57
2026-05-10 17:29:57,170 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 1 -> vv7 confirmed (state=COMMITTED, 771.9 MB). Safe to close Colab.
2026-05-11 00:29:57
2026-05-10 17:29:57,260 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v12/epoch_000.pt
2026-05-11 00:29:57
2026-05-10 17:29:57,347 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v12/best.pt
2026-05-11 00:29:57
2026-05-10 17:29:57,348 - src.training.trainer - INFO - Epoch 001 | train/loss=2.3263 | train/cl_loss=4.3612 | train/skipped_batches=0.0000 | train/lr=0.0000 | HR@10=0.0269 | NDCG@10=0.0066 | HR@20=0.0487 | NDCG@20=0.0110 | HR@50=0.0741 | NDCG@50=0.0159  <- best | DIAG λ: L0_lambda_cart=0.974 L0_lambda_purchase=0.693 L0_lambda_struct=0.693 L0_lambda_view=1.313 L1_lambda_cart=0.974 L1_lambda_purchase=0.692 | DIAG z: baw0_zbeta_norm_cart=7.998 baw0_zbeta_norm_purchase=7.998 baw0_zbeta_norm_struct=8.000 baw0_zbeta_norm_view=7.998 baw1_zbeta_norm_cart=7.998 baw1_zbeta_norm_purchase=7.999
2026-05-11 00:29:57
epochs:   8%|▊         | 2/25 [48:57<6:22:56, 998.97s/it, loss=2.1273, NDCG_20=0.0105, best_primary=0.0110]2026-05-10 17:45:41,069 - src.training.checkpoint_manager - INFO - Saved local: checkpoints-v12/epoch_002.pt (771.9 MB)
2026-05-11 00:29:57
2026-05-10 17:45:43,385 - src.training.checkpoint_manager - INFO - Artifact enqueued: bpatmp-checkpoint-v12 epoch-002
2026-05-11 00:46:32
2026-05-10 17:46:32,108 - src.training.checkpoint_manager - INFO - Size OK: 771.94 MB (diff 0.00%)
2026-05-11 00:46:32
2026-05-10 17:46:32,108 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 2 -> vv8 confirmed (state=COMMITTED, 771.9 MB). Safe to close Colab.
2026-05-11 00:46:32
2026-05-10 17:46:32,199 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v12/epoch_001.pt
2026-05-11 00:46:32
2026-05-10 17:46:32,199 - src.training.trainer - INFO - Epoch 002 | train/loss=2.1273 | train/cl_loss=4.3602 | train/skipped_batches=0.0000 | train/lr=0.0001 | HR@10=0.0240 | NDCG@10=0.0070 | HR@20=0.0412 | NDCG@20=0.0105 | HR@50=0.0646 | NDCG@50=0.0149 | DIAG λ: L0_lambda_cart=0.974 L0_lambda_purchase=0.693 L0_lambda_struct=0.693 L0_lambda_view=1.313 L1_lambda_cart=0.973 L1_lambda_purchase=0.692 | DIAG z: baw0_zbeta_norm_cart=7.993 baw0_zbeta_norm_purchase=7.993 baw0_zbeta_norm_struct=8.000 baw0_zbeta_norm_view=7.994 baw1_zbeta_norm_cart=7.999 baw1_zbeta_norm_purchase=8.000
2026-05-11 00:46:32
epochs:  12%|█▏        | 3/25 [1:05:30<6:05:35, 997.09s/it, loss=2.0545, NDCG_20=0.0064, best_primary=0.0110]2026-05-10 18:02:13,651 - src.training.checkpoint_manager - INFO - Saved local: checkpoints-v12/epoch_003.pt (771.9 MB)
2026-05-11 00:46:32
2026-05-10 18:02:16,155 - src.training.checkpoint_manager - INFO - Artifact enqueued: bpatmp-checkpoint-v12 epoch-003
2026-05-11 01:03:08
2026-05-10 18:03:08,107 - src.training.checkpoint_manager - INFO - Size OK: 771.94 MB (diff 0.00%)
2026-05-11 01:03:08
2026-05-10 18:03:08,107 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 3 -> vv9 confirmed (state=COMMITTED, 771.9 MB). Safe to close Colab.
2026-05-11 01:03:08
2026-05-10 18:03:08,197 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v12/epoch_002.pt
2026-05-11 01:03:08
2026-05-10 18:03:08,197 - src.training.trainer - INFO - Epoch 003 | train/loss=2.0545 | train/cl_loss=4.3560 | train/skipped_batches=0.0000 | train/lr=0.0001 | HR@10=0.0143 | NDCG@10=0.0040 | HR@20=0.0282 | NDCG@20=0.0064 | HR@50=0.0558 | NDCG@50=0.0108 | DIAG λ: L0_lambda_cart=0.974 L0_lambda_purchase=0.692 L0_lambda_struct=0.693 L0_lambda_view=1.312 L1_lambda_cart=0.973 L1_lambda_purchase=0.692 | DIAG z: baw0_zbeta_norm_cart=7.988 baw0_zbeta_norm_purchase=7.989 baw0_zbeta_norm_struct=8.000 baw0_zbeta_norm_view=7.991 baw1_zbeta_norm_cart=8.000 baw1_zbeta_norm_purchase=8.001
2026-05-11 01:03:08
epochs:  16%|█▌        | 4/25 [1:22:08<5:48:49, 996.66s/it, loss=1.9938, NDCG_20=0.0108, best_primary=0.0110]2026-05-10 18:18:51,953 - src.training.checkpoint_manager - INFO - Saved local: checkpoints-v12/epoch_004.pt (771.9 MB)
2026-05-11 01:03:08
2026-05-10 18:18:54,227 - src.training.checkpoint_manager - INFO - Artifact enqueued: bpatmp-checkpoint-v12 epoch-004
2026-05-11 01:19:43
2026-05-10 18:19:43,200 - src.training.checkpoint_manager - INFO - Size OK: 771.94 MB (diff 0.00%)
2026-05-11 01:19:43
2026-05-10 18:19:43,201 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 4 -> vv10 confirmed (state=COMMITTED, 771.9 MB). Safe to close Colab.
2026-05-11 01:19:43
2026-05-10 18:19:43,297 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v12/epoch_003.pt
2026-05-11 01:19:43
2026-05-10 18:19:43,297 - src.training.trainer - INFO - Epoch 004 | train/loss=1.9938 | train/cl_loss=4.3598 | train/skipped_batches=0.0000 | train/lr=0.0001 | HR@10=0.0246 | NDCG@10=0.0066 | HR@20=0.0470 | NDCG@20=0.0108 | HR@50=0.0800 | NDCG@50=0.0165 | DIAG λ: L0_lambda_cart=0.975 L0_lambda_purchase=0.692 L0_lambda_struct=0.693 L0_lambda_view=1.312 L1_lambda_cart=0.972 L1_lambda_purchase=0.693 | DIAG z: baw0_zbeta_norm_cart=7.971 baw0_zbeta_norm_purchase=7.979 baw0_zbeta_norm_struct=8.000 baw0_zbeta_norm_view=7.985 baw1_zbeta_norm_cart=8.000 baw1_zbeta_norm_purchase=8.007
2026-05-11 01:19:43
epochs:  20%|██        | 5/25 [1:38:43<5:32:01, 996.10s/it, loss=1.8903, NDCG_20=0.0091, best_primary=0.0110]2026-05-10 18:35:26,759 - src.training.checkpoint_manager - INFO - Saved local: checkpoints-v12/epoch_005.pt (771.9 MB)
2026-05-11 01:19:43
2026-05-10 18:35:30,450 - src.training.checkpoint_manager - INFO - Artifact enqueued: bpatmp-checkpoint-v12 epoch-005
2026-05-11 01:36:15
2026-05-10 18:36:15,242 - src.training.checkpoint_manager - INFO - Size OK: 771.94 MB (diff 0.00%)
2026-05-11 01:36:15
2026-05-10 18:36:15,242 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 5 -> vv11 confirmed (state=COMMITTED, 771.9 MB). Safe to close Colab.
2026-05-11 01:36:15
2026-05-10 18:36:15,334 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v12/epoch_004.pt
2026-05-11 01:36:15
2026-05-10 18:36:15,334 - src.training.trainer - INFO - Epoch 005 | train/loss=1.8903 | train/cl_loss=4.3586 | train/skipped_batches=0.0000 | train/lr=0.0001 | HR@10=0.0261 | NDCG@10=0.0073 | HR@20=0.0364 | NDCG@20=0.0091 | HR@50=0.0573 | NDCG@50=0.0121 | DIAG λ: L0_lambda_cart=0.974 L0_lambda_purchase=0.692 L0_lambda_struct=0.693 L0_lambda_view=1.313 L1_lambda_cart=0.971 L1_lambda_purchase=0.694 | DIAG z: baw0_zbeta_norm_cart=7.944 baw0_zbeta_norm_purchase=7.965 baw0_zbeta_norm_struct=8.000 baw0_zbeta_norm_view=7.986 baw1_zbeta_norm_cart=8.006 baw1_zbeta_norm_purchase=8.014
2026-05-11 01:36:15
epochs:  24%|██▍       | 6/25 [1:55:17<5:14:59, 994.72s/it, loss=1.7630, NDCG_20=0.0051, best_primary=0.0110]2026-05-10 18:52:00,323 - src.training.checkpoint_manager - INFO - Saved local: checkpoints-v12/epoch_006.pt (771.9 MB)
2026-05-11 01:36:15
2026-05-10 18:52:02,463 - src.training.checkpoint_manager - INFO - Artifact enqueued: bpatmp-checkpoint-v12 epoch-006
2026-05-11 01:52:48
2026-05-10 18:52:48,844 - src.training.checkpoint_manager - INFO - Size OK: 771.94 MB (diff 0.00%)
2026-05-11 01:52:48
2026-05-10 18:52:48,844 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 6 -> vv12 confirmed (state=COMMITTED, 771.9 MB). Safe to close Colab.
2026-05-11 01:52:48
2026-05-10 18:52:48,935 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v12/epoch_005.pt
2026-05-11 01:52:48
2026-05-10 18:52:48,936 - src.training.trainer - INFO - Epoch 006 | train/loss=1.7630 | train/cl_loss=4.3541 | train/skipped_batches=0.0000 | train/lr=0.0001 | HR@10=0.0157 | NDCG@10=0.0034 | HR@20=0.0263 | NDCG@20=0.0051 | HR@50=0.0464 | NDCG@50=0.0078 | DIAG λ: L0_lambda_cart=0.975 L0_lambda_purchase=0.691 L0_lambda_struct=0.693 L0_lambda_view=1.312 L1_lambda_cart=0.971 L1_lambda_purchase=0.694 | DIAG z: baw0_zbeta_norm_cart=7.911 baw0_zbeta_norm_purchase=7.949 baw0_zbeta_norm_struct=8.000 baw0_zbeta_norm_view=7.979 baw1_zbeta_norm_cart=8.003 baw1_zbeta_norm_purchase=8.025
2026-05-11 01:52:48
epochs:  28%|██▊       | 7/25 [2:11:52<4:58:18, 994.35s/it, loss=1.5990, NDCG_20=0.0038, best_primary=0.0110]2026-05-10 19:08:35,735 - src.training.checkpoint_manager - INFO - Saved local: checkpoints-v12/epoch_007.pt (771.9 MB)
2026-05-11 01:52:48
2026-05-10 19:08:37,911 - src.training.checkpoint_manager - INFO - Artifact enqueued: bpatmp-checkpoint-v12 epoch-007
2026-05-11 02:09:29
2026-05-10 19:09:29,600 - src.training.checkpoint_manager - INFO - Size OK: 771.94 MB (diff 0.00%)
2026-05-11 02:09:29
2026-05-10 19:09:29,600 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 7 -> vv13 confirmed (state=COMMITTED, 771.9 MB). Safe to close Colab.
2026-05-11 02:09:29
2026-05-10 19:09:29,691 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v12/epoch_006.pt
2026-05-11 02:09:29
2026-05-10 19:09:29,692 - src.training.trainer - INFO - Epoch 007 | train/loss=1.5990 | train/cl_loss=4.3586 | train/skipped_batches=0.0000 | train/lr=0.0001 | HR@10=0.0111 | NDCG@10=0.0026 | HR@20=0.0193 | NDCG@20=0.0038 | HR@50=0.0390 | NDCG@50=0.0062 | DIAG λ: L0_lambda_cart=0.976 L0_lambda_purchase=0.691 L0_lambda_struct=0.693 L0_lambda_view=1.310 L1_lambda_cart=0.971 L1_lambda_purchase=0.694 | DIAG z: baw0_zbeta_norm_cart=7.882 baw0_zbeta_norm_purchase=7.927 baw0_zbeta_norm_struct=8.000 baw0_zbeta_norm_view=7.966 baw1_zbeta_norm_cart=8.002 baw1_zbeta_norm_purchase=8.031
2026-05-11 02:09:29
epochs:  32%|███▏      | 8/25 [2:28:30<4:42:18, 996.39s/it, loss=1.4503, NDCG_20=0.0047, best_primary=0.0110]2026-05-10 19:25:13,619 - src.training.checkpoint_manager - INFO - Saved local: checkpoints-v12/epoch_008.pt (771.9 MB)
2026-05-11 02:09:29
2026-05-10 19:25:15,925 - src.training.checkpoint_manager - INFO - Artifact enqueued: bpatmp-checkpoint-v12 epoch-008
2026-05-11 02:26:01
2026-05-10 19:26:01,608 - src.training.checkpoint_manager - INFO - Size OK: 771.94 MB (diff 0.00%)
2026-05-11 02:26:01
2026-05-10 19:26:01,608 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 8 -> vv14 confirmed (state=COMMITTED, 771.9 MB). Safe to close Colab.
2026-05-11 02:26:01
2026-05-10 19:26:01,702 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v12/epoch_007.pt
2026-05-11 02:26:01
2026-05-10 19:26:01,702 - src.training.trainer - INFO - Epoch 008 | train/loss=1.4503 | train/cl_loss=4.3555 | train/skipped_batches=0.0000 | train/lr=0.0001 | HR@10=0.0147 | NDCG@10=0.0038 | HR@20=0.0220 | NDCG@20=0.0047 | HR@50=0.0356 | NDCG@50=0.0063 | DIAG λ: L0_lambda_cart=0.978 L0_lambda_purchase=0.690 L0_lambda_struct=0.693 L0_lambda_view=1.311 L1_lambda_cart=0.970 L1_lambda_purchase=0.694 | DIAG z: baw0_zbeta_norm_cart=7.855 baw0_zbeta_norm_purchase=7.912 baw0_zbeta_norm_struct=8.000 baw0_zbeta_norm_view=7.959 baw1_zbeta_norm_cart=7.997 baw1_zbeta_norm_purchase=8.034
2026-05-11 02:26:01
epochs:  36%|███▌      | 9/25 [2:45:00<4:25:20, 995.02s/it, loss=1.3320, NDCG_20=0.0036, best_primary=0.0110]2026-05-10 19:41:43,506 - src.training.checkpoint_manager - INFO - Saved local: checkpoints-v12/epoch_009.pt (771.9 MB)
2026-05-11 02:26:01
2026-05-10 19:41:45,695 - src.training.checkpoint_manager - INFO - Artifact enqueued: bpatmp-checkpoint-v12 epoch-009
2026-05-11 02:42:35
2026-05-10 19:42:35,792 - src.training.checkpoint_manager - INFO - Size OK: 771.94 MB (diff 0.00%)
2026-05-11 02:42:35
2026-05-10 19:42:35,793 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 9 -> vv15 confirmed (state=COMMITTED, 771.9 MB). Safe to close Colab.
2026-05-11 02:42:35
2026-05-10 19:42:35,884 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v12/epoch_008.pt
2026-05-11 02:42:35
2026-05-10 19:42:35,884 - src.training.trainer - INFO - Epoch 009 | train/loss=1.3320 | train/cl_loss=4.3548 | train/skipped_batches=0.0000 | train/lr=0.0001 | HR@10=0.0100 | NDCG@10=0.0023 | HR@20=0.0188 | NDCG@20=0.0036 | HR@50=0.0327 | NDCG@50=0.0052 | DIAG λ: L0_lambda_cart=0.977 L0_lambda_purchase=0.690 L0_lambda_struct=0.693 L0_lambda_view=1.313 L1_lambda_cart=0.970 L1_lambda_purchase=0.694 | DIAG z: baw0_zbeta_norm_cart=7.834 baw0_zbeta_norm_purchase=7.909 baw0_zbeta_norm_struct=8.000 baw0_zbeta_norm_view=7.957 baw1_zbeta_norm_cart=7.996 baw1_zbeta_norm_purchase=8.039
2026-05-11 02:42:35
epochs:  40%|████      | 10/25 [3:01:35<4:08:41, 994.76s/it, loss=1.2325, NDCG_20=0.0028, best_primary=0.0110]2026-05-10 19:58:18,205 - src.training.checkpoint_manager - INFO - Saved local: checkpoints-v12/epoch_010.pt (771.9 MB)
2026-05-11 02:42:35
2026-05-10 19:58:21,706 - src.training.checkpoint_manager - INFO - Artifact enqueued: bpatmp-checkpoint-v12 epoch-010
2026-05-11 02:59:08
2026-05-10 19:59:08,412 - src.training.checkpoint_manager - INFO - Size OK: 771.94 MB (diff 0.00%)
2026-05-11 02:59:08
2026-05-10 19:59:08,412 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 10 -> vv16 confirmed (state=COMMITTED, 771.9 MB). Safe to close Colab.
2026-05-11 02:59:08
2026-05-10 19:59:08,510 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v12/epoch_009.pt
2026-05-11 02:59:08
2026-05-10 19:59:08,510 - src.training.trainer - INFO - Epoch 010 | train/loss=1.2325 | train/cl_loss=4.3529 | train/skipped_batches=0.0000 | train/lr=0.0001 | HR@10=0.0088 | NDCG@10=0.0020 | HR@20=0.0148 | NDCG@20=0.0028 | HR@50=0.0255 | NDCG@50=0.0041 | DIAG λ: L0_lambda_cart=0.977 L0_lambda_purchase=0.689 L0_lambda_struct=0.693 L0_lambda_view=1.314 L1_lambda_cart=0.970 L1_lambda_purchase=0.694 | DIAG z: baw0_zbeta_norm_cart=7.823 baw0_zbeta_norm_purchase=7.912 baw0_zbeta_norm_struct=8.000 baw0_zbeta_norm_view=7.955 baw1_zbeta_norm_cart=7.995 baw1_zbeta_norm_purchase=8.042
2026-05-11 02:59:08
epochs:  44%|████▍     | 11/25 [3:18:08<3:51:57, 994.11s/it, loss=1.1442, NDCG_20=0.0030, best_primary=0.0110]2026-05-10 20:14:51,538 - src.training.checkpoint_manager - INFO - Saved local: checkpoints-v12/epoch_011.pt (771.9 MB)
2026-05-11 02:59:08
2026-05-10 20:14:53,797 - src.training.checkpoint_manager - INFO - Artifact enqueued: bpatmp-checkpoint-v12 epoch-011
2026-05-11 03:15:39
2026-05-10 20:15:39,573 - src.training.checkpoint_manager - INFO - Size OK: 771.94 MB (diff 0.00%)
2026-05-11 03:15:39
2026-05-10 20:15:39,573 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 11 -> vv17 confirmed (state=COMMITTED, 771.9 MB). Safe to close Colab.
2026-05-11 03:15:39
2026-05-10 20:15:39,668 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v12/epoch_010.pt
2026-05-11 03:15:39
2026-05-10 20:15:39,668 - src.training.trainer - INFO - Epoch 011 | train/loss=1.1442 | train/cl_loss=4.3542 | train/skipped_batches=0.0000 | train/lr=0.0001 | HR@10=0.0095 | NDCG@10=0.0023 | HR@20=0.0156 | NDCG@20=0.0030 | HR@50=0.0253 | NDCG@50=0.0041 | DIAG λ: L0_lambda_cart=0.978 L0_lambda_purchase=0.688 L0_lambda_struct=0.693 L0_lambda_view=1.313 L1_lambda_cart=0.970 L1_lambda_purchase=0.693 | DIAG z: baw0_zbeta_norm_cart=7.805 baw0_zbeta_norm_purchase=7.915 baw0_zbeta_norm_struct=8.000 baw0_zbeta_norm_view=7.948 baw1_zbeta_norm_cart=7.992 baw1_zbeta_norm_purchase=8.043
2026-05-11 03:15:39
epochs:  48%|████▊     | 12/25 [3:34:36<3:35:11, 993.21s/it, loss=1.0775, NDCG_20=0.0030, best_primary=0.0110]2026-05-10 20:31:19,800 - src.training.checkpoint_manager - INFO - Saved local: checkpoints-v12/epoch_012.pt (771.9 MB)
2026-05-11 03:15:39
2026-05-10 20:31:22,137 - src.training.checkpoint_manager - INFO - Artifact enqueued: bpatmp-checkpoint-v12 epoch-012
2026-05-11 03:32:06
2026-05-10 20:32:06,376 - src.training.checkpoint_manager - INFO - Size OK: 771.94 MB (diff 0.00%)
2026-05-11 03:32:06
2026-05-10 20:32:06,377 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 12 -> vv18 confirmed (state=COMMITTED, 771.9 MB). Safe to close Colab.
2026-05-11 03:32:06
2026-05-10 20:32:06,472 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v12/epoch_011.pt
2026-05-11 03:32:06
2026-05-10 20:32:06,472 - src.training.trainer - INFO - Epoch 012 | train/loss=1.0775 | train/cl_loss=4.3578 | train/skipped_batches=0.0000 | train/lr=0.0001 | HR@10=0.0097 | NDCG@10=0.0024 | HR@20=0.0147 | NDCG@20=0.0030 | HR@50=0.0241 | NDCG@50=0.0041 | DIAG λ: L0_lambda_cart=0.978 L0_lambda_purchase=0.687 L0_lambda_struct=0.693 L0_lambda_view=1.313 L1_lambda_cart=0.970 L1_lambda_purchase=0.693 | DIAG z: baw0_zbeta_norm_cart=7.800 baw0_zbeta_norm_purchase=7.916 baw0_zbeta_norm_struct=8.000 baw0_zbeta_norm_view=7.948 baw1_zbeta_norm_cart=7.993 baw1_zbeta_norm_purchase=8.049
2026-05-11 03:32:06
epochs:  52%|█████▏    | 13/25 [3:51:05<3:18:15, 991.27s/it, loss=1.0266, NDCG_20=0.0026, best_primary=0.0110]2026-05-10 20:47:48,539 - src.training.checkpoint_manager - INFO - Saved local: checkpoints-v12/epoch_013.pt (771.9 MB)
2026-05-11 03:32:06
2026-05-10 20:47:50,743 - src.training.checkpoint_manager - INFO - Artifact enqueued: bpatmp-checkpoint-v12 epoch-013
2026-05-11 03:48:36
2026-05-10 20:48:36,166 - src.training.checkpoint_manager - INFO - Size OK: 771.94 MB (diff 0.00%)
2026-05-11 03:48:36
2026-05-10 20:48:36,167 - src.training.checkpoint_manager - INFO - [OK] Checkpoint epoch 13 -> vv19 confirmed (state=COMMITTED, 771.9 MB). Safe to close Colab.
2026-05-11 03:48:36
2026-05-10 20:48:36,264 - src.training.checkpoint_manager - INFO - Removed old checkpoint: checkpoints-v12/epoch_012.pt
2026-05-11 03:48:36
2026-05-10 20:48:36,265 - src.training.trainer - INFO - Epoch 013 | train/loss=1.0266 | train/cl_loss=4.3551 | train/skipped_batches=0.0000 | train/lr=0.0001 | HR@10=0.0086 | NDCG@10=0.0020 | HR@20=0.0134 | NDCG@20=0.0026 | HR@50=0.0235 | NDCG@50=0.0037 | DIAG λ: L0_lambda_cart=0.979 L0_lambda_purchase=0.687 L0_lambda_struct=0.693 L0_lambda_view=1.314 L1_lambda_cart=0.970 L1_lambda_purchase=0.693 | DIAG z: baw0_zbeta_norm_cart=7.796 baw0_zbeta_norm_purchase=7.921 baw0_zbeta_norm_struct=8.000 baw0_zbeta_norm_view=7.949 baw1_zbeta_norm_cart=7.996 baw1_zbeta_norm_purchase=8.052
2026-05-11 03:48:36
2026-05-10 20:48:36,265 - src.training.trainer - INFO - Early stopping at epoch 13. Best NDCG@20=0.0110