# Hyperspectral Real-World Super-Resolution
Guide for training real-world HSI models and inference

## Setup
- Place `data` folder in directory with the following structure:
  - Train image pairs
    - HR images: `data/processed/full_hsi_val/train/hr`
    - LR images: `data/processed/full_hsi_val/train/lr`
  - Validation image pairs
    - HR images: `data/processed/full_hsi_val/val/hr`
    - LR images: `data/processed/full_hsi_val/val/lr`
  - Test image pairs
    - HR images: `data/processed/full_hsi_val/test/hr`
    - LR images: `data/processed/full_hsi_val/test/lr`
- Place `results` folder in directory

## Training
### Create dataset
- The bicubic images used in training can be created using `build_bicubic.py`.
- If using new data, generate meta info using the following command (replace capitalized words as necessary): `python3 Real-ESRGAN/scripts/generate_meta_info_pairdata.py --input PATH/hr PATH/bi_240 --root PATH PATH --meta_info PATH/meta_info_NAME.txt`

### Start training
- Configure training parameters in `Real-ESRGAN/options/FILENAME.yml`.
- Run `python3 Real-ESRGAN/realesrgan/train.py -opt Real-ESRGAN/options/OPTIONS_FILE --auto_resume`

## Inference
- Place trained model in `Real-ESRGAN/weights/MODEL_NAME.pth`
- Modify `Real-ESRGAN/inference_realesrgan.py`
  - Copy and adapt `elif args.model_name == 'MODEL_NAME':` and the next two lines to reflect the new model
- Place images from `data/processed/full_hsi_val/test/lr` in `Real-ESRGAN/inputs`
- Run `python3 Real-ESRGAN/inference_realesrgan.py -n MODEL_NAME -i Real-ESRGAN/inputs`
- Copy generated upscaled images from `results/c_sr/` to `results/models/MODEL_NAME/sr/`
- Run `evaluate_models.ipynb` and change `model_name = ...` appropriately

