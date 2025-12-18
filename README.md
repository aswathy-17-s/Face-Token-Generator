# 🧑‍🦱 Face Token Generator for Ration Shop

A **computer vision–based system** that generates **tokens for ration shop beneficiaries using face recognition**. The project helps automate beneficiary identification, reduce fraud, and ensure fair distribution of ration items by verifying a person’s face before issuing a token.

---

##  Problem Statement

Traditional ration shop systems often face issues such as:

* Duplicate or fake beneficiaries
* Identity fraud using ration cards
* Manual verification errors

This project solves the problem by using **Face Recognition and Object Detection** to uniquely identify individuals and generate tokens only for verified faces.

---

##  Features

* Face detection and recognition
* Token generation for verified faces
* Separate handling of known and unknown faces
* Video input support for real-time testing
* Prevents duplicate token generation
* Uses deep learning models for accuracy

---

## Tech Stack

* **Programming Language**: Python
* **Computer Vision**: OpenCV, Dlib
* **Deep Learning Models**:

  * YOLOv8 (`yolov8n.pt`)
  * MobileNet SSD (`deploy.prototxt`, `mobilenet_iter_73000.caffemodel`)
* **Frameworks/Libraries**:

  * NumPy
  * imutils

---

##  Project Structure

```
FACE-TOKEN-GENERATOR/
│── Known_Faces/              # Images of registered beneficiaries
│── Unknown_Faces/            # Images of unrecognized faces
│── Tokens/                   # Generated tokens
│── app.py                    # Application logic
│── main.py                   # Main execution file
│── test.py                   # Testing scripts
│── test1.py
│── test2.py
│── deploy.prototxt           # MobileNet SSD config
│── mobilenet_iter_73000.caffemodel
│── yolov8n.pt                # YOLOv8 model
│── 11.mp4                    # Sample test video
│── 12.mp4                    # Sample test video
│── README.md
```

---

##  How to Run the Project

###  Prerequisites

* Python 3.9+
* Webcam or video file

###  Install Dependencies

```bash
pip install opencv-python dlib numpy imutils ultralytics
```

*(If `dlib` installation fails, use the provided `.whl` file)*

---

###  Run the Application

```bash
python main.py
```

or

```bash
python app.py
```

---

##  Working Explanation

1. System captures face from camera or video
2. Face is detected using deep learning models
3. Face is matched with images in `Known_Faces`
4. If matched:

   * Token is generated and stored
5. If not matched:

   * Face is saved in `Unknown_Faces`

---

## Important Notes

* Model files (`.pt`, `.caffemodel`) are included for demonstration purposes
* In real deployment, models should be managed securely
* Project is intended for **academic and prototype use**

---

##  Future Enhancements

* Database integration (MongoDB / MySQL)
* Aadhaar-based verification
* Web or mobile interface
* Cloud deployment
* Multi-camera support

---

##  Author

**Aswathy S**
GitHub: [https://github.com/aswathy-17-s](https://github.com/aswathy-17-s)

---

## ⭐ Support

If you find this project useful, please ⭐ the repository on GitHub!
