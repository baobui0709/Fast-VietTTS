# HF Relay Training Plan — Colab T4

## Mục tiêu

Thiết kế workflow train dần trên nhiều phiên Colab T4 hợp lệ:

```text
Phiên A train 2h → upload checkpoint lên Hugging Face
Phiên B tải checkpoint mới nhất → train tiếp 2h → upload checkpoint
Phiên C tải checkpoint mới nhất → train tiếp
```

## Nguyên tắc

Không chạy nhiều phiên cùng ghi vào một checkpoint. Training phải theo kiểu relay / baton handoff.

## Hugging Face repos đề xuất

### Dataset repo

```text
username/fast-viettts-token-dataset
```

Chứa:

```text
metadata.csv
manifest_clean.jsonl
tokens/
  shard_000/
  shard_001/
token_manifest.jsonl
dataset_stats.json
```

### Model repo

```text
username/fast-viettts-student
```

Chứa:

```text
checkpoints/latest.pt
checkpoints/step_001000.pt
train_state.json
train_log.jsonl
eval_samples/
```

## Dataset format đầu vào

Metadata dạng:

```text
file001.wav|nội dung audio
file002.wav|nội dung audio
```

Khuyến nghị đóng gói audio thành zip/tar shards trước khi upload lên Hugging Face nếu có nhiều file nhỏ.

## Daily relay workflow

Mỗi phiên Colab:

1. Cài môi trường training tối giản.
2. Login Hugging Face.
3. Download `latest.pt` và `train_state.json`.
4. Train trong thời gian cho phép.
5. Save checkpoint local.
6. Upload `latest.pt`, checkpoint mốc, `train_state.json`, `train_log.jsonl`.
7. Tắt phiên.

## Lock file

Nên dùng lock để tránh hai phiên ghi checkpoint cùng lúc:

```json
{
  "locked": true,
  "worker": "colab_A",
  "started_at": "2026-05-09T08:00:00",
  "base_step": 1200
}
```

Nếu lock quá hạn, coi là stale.

## Giai đoạn trước training

Trước khi train student:

1. Audit metadata.
2. Tạo manifest clean.
3. Tokenize audio / export target tokens theo shard.
4. Upload token shards lên HF.
5. Chỉ khi token dataset ổn mới train.

## Kỳ vọng tốc độ

Student nhỏ trên T4 có thể hướng tới:

```text
RTF 0.8x–1.0x trong giai đoạn đầu
RTF 0.5x–0.8x nếu student đủ tốt
```

Điều này tương đương khoảng 1.5x–2.5x nhanh hơn bản hiện tại, không phải 5x.
