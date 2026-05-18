
# Face Recognition Photo Sorter

Oraganise photos into seperate folders based on face.

Automatically organize your photos by face using deep learning.  
This tool scans a folder of images, detects faces, matches them against a known face database, and copies each photo into a subfolder named after the recognized person.

When an unknown face is encountered, the script shows a preview and asks for a name – the face is then added to the memory for future runs.

---

## Features

- **GPU‑accelerated** face detection & recognition (InsightFace)
- **Interactive learning** – name new faces on the fly
- **Persistent memory** – recognized faces are saved to a pickle file
- **Automatic sorting** – photos are copied to `sorted_photos/<person>/`
- **Original cleanup** – processed images are deleted from the input folder
- **Cosine similarity** matching with configurable threshold

---

## Requirements

- Python 3.8+
- A **GPU** with CUDA support is **strongly recommended** (the script uses `CUDAExecutionProvider` by default)
- For CPU‑only execution, you must modify the `providers` list (see [Troubleshooting](#troubleshooting))

### Python Packages

```bash
pip install opencv-python numpy insightface