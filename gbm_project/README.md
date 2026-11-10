# GBM longitudinal framework (3D CNN-Swin  ->  GAT  ->  time-aware Transformer  ->  gated fusion)

## IntelliJ setup
1. Install the **Python plugin** (Settings > Plugins) or use PyCharm.
2. File > Open > select this folder. File > Project Structure > SDKs > add a **Python 3.10/3.11 virtualenv**.
3. Open a terminal inside IntelliJ:  `pip install -r requirements.txt`
   (install the CUDA build of torch from pytorch.org if you have an NVIDIA GPU).
4. Edit `configs/config.yaml` (dataset path, clinical CSV path and its column names).
5. For each step create a Run Configuration: **Module name** = `src.train.xxx`, **Parameters** = `--config configs/config.yaml`,
   **Working directory** = project root.

## Run order
| Step | Command (module) | What it does |
|------|------------------|--------------|
| 1 | `src.data.scan_dataset` | scans UCSD-PTGBM folders -> `outputs/scan_table.csv` (CHECK this file) |
| 2 | `src.train.train_seg` | trains CNN-Swin segmentation (Dice, IoU, HD95 for ET/TC/WT) |
| 3 | `src.train.extract_features` | runs the model on every visit, caches ET/NEC/ED/WT graph nodes |
| 4 | `src.train.train_longitudinal` | GAT + temporal Transformer + fusion, progression + survival |

Patient-level train/val/test splits are saved in `outputs/splits.json` and reused by steps 2 and 4.
