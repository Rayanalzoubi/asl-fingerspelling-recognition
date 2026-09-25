# ASL Fingerspelling Recognition

A Flask web app that recognizes American Sign Language (ASL) fingerspelling letters from an uploaded image or video.

The app finds the hand with [MediaPipe](https://developers.google.com/mediapipe), crops it, and classifies it with a fine-tuned EfficientNet model into one of 29 classes: the letters **A–Z** plus **del**, **nothing** and **space**.

## How it works

1. **Hand detection:** MediaPipe Hands locates the hand, and the image is cropped around its landmarks with a 20 px margin ([mediapipe_handler.py](mediapipe_handler.py)).
2. **Preprocessing:** the crop is converted to grayscale, resized to 225×225, and scaled to `[0, 1]`.
3. **Classification:** the EfficientNet model predicts the letter and a confidence score.
4. **Videos:** every 5th frame is classified, and the most frequent prediction is returned.

## Project structure

| File | Purpose |
| --- | --- |
| `app.py` | Flask web app (upload page and prediction) |
| `mediapipe_handler.py` | `HandExtractor`: detects and crops the hand |
| `templates/index.html` | Upload page (Arabic UI) |
| `extract_hands_from_videos.py` | Extracts hand crops from a video to build fine-tuning data |
| `build_model_and_load.py` | Rebuilds the EfficientNetB1 architecture, loads the weights and saves an `.h5` copy |
| `check_model_shape.py` | Prints the model's input and output shapes |

## Setup

Requires **Python 3.10** (TensorFlow 2.19 and MediaPipe 0.10.21 do not support Python 3.13).

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### Download the model

The trained model is too large for the repository. Download `asl_efficientnet_best_finetuned.keras` from the [Releases](../../releases) page and place it in the project root.

## Usage

```bash
python app.py
```

Open http://127.0.0.1:5000, upload an image (`.jpg`, `.jpeg`, `.png`) or a video (`.mp4`, `.avi`, `.mov`), and click the button to analyze it.

Set `FLASK_DEBUG=1` to run in debug mode.

### Extracting training images from a video

Set `video_path` in `extract_hands_from_videos.py`, then run:

```bash
python extract_hands_from_videos.py
```

The hand crops are saved to `asl_single_finetune/`.

## Training

The model was trained and fine-tuned in Google Colab. The training notebook is not included in this repository.

## License

[MIT](LICENSE)
