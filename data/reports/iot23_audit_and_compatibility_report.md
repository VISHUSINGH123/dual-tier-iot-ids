# IoT-23 Dataset Audit, Schema Compatibility & Offline Evaluation Report

## 1. Dataset Source & Verification Summary
- **Archive Path**: `M:\iot_23_datasets_small.tar.gz`
- **Extracted Root**: `M:\projects\proj2(3 credits)\data\raw\iot_23_datasets_small`
- **Total Scenarios Audited**: `23` (`20` Malware captures + `3` Honeypot captures)
- **Total Uncompressed Size**: `44,880.67 MB` (`44.88 GB`)
- **Total Connection Records**: `325,309,946`
- **Total Parse Errors**: `0`

### Delimiter & Format Findings
- All 23 files declare `#separator \x09`, `#empty_field (empty)`, `#unset_field -`, and 23 columns in `#fields`.
- **Mixed Delimiter Quirk**: In all 23 files, columns 1–20 (`ts` through `resp_ip_bytes`) are separated by tabs (`\t`), whereas columns 21–23 (`tunnel_parents   label   detailed-label`) are separated by **3 spaces (`   `)** inside the final tab token.
- **Case-Sensitive `label` Conventions**:
  - Lowercase `'benign'`: `1,956` rows (used exclusively in the 3 honeypot captures: `CTU-Honeypot-Capture-4-1`, `CTU-Honeypot-Capture-5-1`, `Somfy-01`).
  - Capitalized `'Benign'`: `30,858,735` rows (used in the 20 `CTU-IoT-Malware-Capture-*` files; always paired with `detailed-label = '-'`).
  - Capitalized `'Malicious'`: `294,449,255` rows.

## 2. Per-File Audit Table (All 23 Captures)

| # | Scenario | Size (MB) | Rows | Parse Errors | `benign` | `Benign` | `Malicious` | Top `detailed-label` (Malicious) |
|---:|:---|---:|---:|---:|---:|---:|---:|:---|
| 1 | `CTU-Honeypot-Capture-4-1` | 0.06 | 452 | 0 | 452 | 0 | 0 | - |
| 2 | `CTU-Honeypot-Capture-5-1` | 0.17 | 1,374 | 0 | 1,374 | 0 | 0 | - |
| 3 | `Somfy-01` | 0.02 | 130 | 0 | 130 | 0 | 0 | - |
| 4 | `CTU-IoT-Malware-Capture-1-1` | 141.44 | 1,008,748 | 0 | 0 | 469,275 | 539,473 | PartOfAHorizontalPortScan: 539,465, C&C: 8 |
| 5 | `CTU-IoT-Malware-Capture-17-1` | 7,762.31 | 54,659,855 | 0 | 0 | 31,438 | 54,628,417 | PartOfAHorizontalPortScan: 27,311,187, Okiru: 13,655,215, DDoS: 13,655,172 |
| 6 | `CTU-IoT-Malware-Capture-20-1` | 0.40 | 3,209 | 0 | 0 | 3,193 | 16 | C&C-Torii: 16 |
| 7 | `CTU-IoT-Malware-Capture-21-1` | 0.41 | 3,286 | 0 | 0 | 3,272 | 14 | C&C-Torii: 14 |
| 8 | `CTU-IoT-Malware-Capture-3-1` | 23.25 | 156,103 | 0 | 0 | 4,536 | 151,567 | PartOfAHorizontalPortScan: 145,597, Attack: 5,962, C&C: 8 |
| 9 | `CTU-IoT-Malware-Capture-33-1` | 7,503.43 | 54,454,591 | 0 | 0 | 1,380,791 | 53,073,800 | PartOfAHorizontalPortScan: 39,459,055, Okiru: 13,609,467, C&C-HeartBeat: 5,278 |
| 10 | `CTU-IoT-Malware-Capture-34-1` | 2.88 | 23,145 | 0 | 0 | 1,923 | 21,222 | DDoS: 14,394, C&C: 6,706, PartOfAHorizontalPortScan: 122 |
| 11 | `CTU-IoT-Malware-Capture-35-1` | 1,278.09 | 10,447,787 | 0 | 0 | 8,262,389 | 2,185,398 | DDoS: 2,185,302, C&C: 81, C&C-FileDownload: 12 |
| 12 | `CTU-IoT-Malware-Capture-36-1` | 1,705.03 | 13,645,098 | 0 | 0 | 2,663 | 13,642,435 | Okiru: 13,626,744, C&C-HeartBeat: 15,688, Okiru-Attack: 3 |
| 13 | `CTU-IoT-Malware-Capture-39-1` | 10,381.09 | 73,568,981 | 0 | 0 | 7,337 | 73,561,644 | PartOfAHorizontalPortScan: 73,559,437, C&C: 1,530, Attack: 677 |
| 14 | `CTU-IoT-Malware-Capture-42-1` | 0.56 | 4,426 | 0 | 0 | 4,420 | 6 | FileDownload: 3, C&C-FileDownload: 3 |
| 15 | `CTU-IoT-Malware-Capture-43-1` | 8,929.50 | 67,321,809 | 0 | 0 | 20,574,934 | 46,746,875 | PartOfAHorizontalPortScan: 37,911,674, Okiru: 8,765,885, DDoS: 65,803 |
| 16 | `CTU-IoT-Malware-Capture-44-1` | 0.03 | 237 | 0 | 0 | 211 | 26 | C&C: 14, C&C-FileDownload: 11, DDoS: 1 |
| 17 | `CTU-IoT-Malware-Capture-48-1` | 507.11 | 3,394,338 | 0 | 0 | 3,734 | 3,390,604 | PartOfAHorizontalPortScan: 3,386,119, Attack: 2,752, C&C-PartOfAHorizontalPortScan: 888 |
| 18 | `CTU-IoT-Malware-Capture-49-1` | 814.09 | 5,410,561 | 0 | 0 | 3,665 | 5,406,896 | PartOfAHorizontalPortScan: 5,404,959, C&C: 1,922, FileDownload: 14 |
| 19 | `CTU-IoT-Malware-Capture-52-1` | 2,928.09 | 19,781,378 | 0 | 0 | 1,794 | 19,779,584 | PartOfAHorizontalPortScan: 19,779,564, C&C-FileDownload: 12, C&C: 6 |
| 20 | `CTU-IoT-Malware-Capture-60-1` | 445.90 | 3,581,028 | 0 | 0 | 2,476 | 3,578,552 | DDoS: 3,578,457, C&C-HeartBeat: 95 |
| 21 | `CTU-IoT-Malware-Capture-7-1` | 1,508.01 | 11,454,714 | 0 | 0 | 75,955 | 11,378,759 | Okiru: 11,333,397, DDoS: 39,584, C&C-HeartBeat: 5,778 |
| 22 | `CTU-IoT-Malware-Capture-8-1` | 1.36 | 10,403 | 0 | 0 | 2,181 | 8,222 | C&C: 8,222 |
| 23 | `CTU-IoT-Malware-Capture-9-1` | 947.44 | 6,378,293 | 0 | 0 | 22,548 | 6,355,745 | PartOfAHorizontalPortScan: 6,355,745 |

## 3. Global `(label, detailed-label)` Distribution

| `label` | `detailed-label` | Exact Flow Count | Share (%) |
|:---|:---|---:|---:|
| `Malicious` | `PartOfAHorizontalPortScan` | 213,852,924 | 65.7382% |
| `Malicious` | `Okiru` | 60,990,708 | 18.7485% |
| `Benign` | `-` | 30,858,735 | 9.4859% |
| `Malicious` | `DDoS` | 19,538,713 | 6.0062% |
| `Malicious` | `C&C-HeartBeat` | 33,673 | 0.0104% |
| `Malicious` | `C&C` | 21,995 | 0.0068% |
| `Malicious` | `Attack` | 9,398 | 0.0029% |
| `benign` | `-` | 1,956 | 0.0006% |
| `Malicious` | `C&C-PartOfAHorizontalPortScan` | 888 | 0.0003% |
| `Malicious` | `C&C-HeartBeat-Attack` | 834 | 0.0003% |
| `Malicious` | `C&C-FileDownload` | 53 | 0.0000% |
| `Malicious` | `C&C-Torii` | 30 | 0.0000% |
| `Malicious` | `FileDownload` | 18 | 0.0000% |
| `Malicious` | `C&C-HeartBeat-FileDownload` | 11 | 0.0000% |
| `Malicious` | `PartOfAHorizontalPortScan-Attack` | 5 | 0.0000% |
| `Malicious` | `Okiru-Attack` | 3 | 0.0000% |
| `Malicious` | `C&C-Mirai` | 2 | 0.0000% |

## 4. Schema Compatibility Analysis (Task 2)

### A. Primary 14-Feature APAFC/ONNX Pipeline (`iot_ids.features.FEATURE_NAMES`)
- **Directly Derivable (6 features)**: `duration_ms`, `packet_count`, `byte_count`, `mean_packet_bytes`, `src_port`, `dst_port`.
- **Approximable (2 features)**: `protocol_id` (via `service`/`id.resp_p` port mapping) and `tcp_flags` (reconstructed from Zeek `history` and `conn_state`).
- **Unavailable in Zeek `conn.log.labeled` (6 features)**: `mqtt_qos`, `mqtt_payload_bytes`, `coap_code`, `http_status`, `sixlowpan_dispatch`, `payload_entropy` (require L7 application headers or raw packet payloads).

### B. Exploratory 61-Feature Edge-IIoTset Pipeline (`src.data_loader.ALL_61_FEATURES`)
- **Incompatible**: 56 of 61 features are packet-level deep inspection fields (`mqtt.*`, `coap.*`, `http.*`, `sixlowpan.*`, `dns.*`, `mbtcp.*`, `arp.*`) absent from Zeek connection summaries.

## 5. Zero-Shot Offline Evaluation of Existing Untouched `edge_ids.onnx` on IoT-23 Sample (Task 3)
- **Evaluated Sample Size**: `10,819` real IoT-23 flows (up to 500 per capture across all 23 files).
- **Binary Confusion Matrix (`[Benign, Malicious]`)**: `[[288, 4629], [320, 5582]]`
- **Binary Macro-F1**: `0.3986`
- **Binary False-Positive Rate (FPR)**: `0.9414`
- **Projected Multiclass Macro-F1**: `0.0963`
