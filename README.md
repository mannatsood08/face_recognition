# Face Recognition Attendance System

## Project Overview
This project is a face recognition-based attendance system using OpenCV, Streamlit, and K-Nearest Neighbors (KNN) for classification. It captures faces, stores them, and recognizes them in real time to mark attendance.

## Features
- **Face Detection & Recognition**: Uses OpenCV's Haar Cascade to detect faces and KNN for classification.
- **Automated Attendance Logging**: Recognized faces are logged into CSV files with timestamps.
- **User-Friendly Interface**: Displays attendance records using Streamlit.
- **Voice Feedback**: Uses text-to-speech to confirm attendance.

## Project Structure
```
Face_Recognition/
│── data/
│   ├── haarcascade_frontalface_default.xml
│   ├── names.pkl
│   ├── face_data.pkl
│── Attendance/
│── backgroundd.png
│── add_face.py
│── app.py
│── test.py
│── README.md
```

## Installation & Setup
### Prerequisites
Ensure you have the following installed:
- Python 3.x
- OpenCV
- NumPy
- Pandas
- Streamlit
- Scikit-learn
- PyWin32 (for text-to-speech on Windows)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/Face_Recognition.git
   cd Face_Recognition
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
### 1. Adding a New Face
Run the following command to add a new face to the dataset:
```bash
python add_face.py
```
- Enter your name when prompted.
- The script captures 100 face images and stores them.

### 2. Running the Attendance System
Run the face recognition system with:
```bash
python test.py
```
- It captures real-time video and detects faces.
- Press 'o' to mark attendance and save it in a CSV file.
- Press 'q' to exit.

### 3. Viewing Attendance Records
Use the Streamlit app to display attendance records:
```bash
streamlit run app.py
```
- The latest attendance CSV file is loaded and displayed.

## Troubleshooting
- Ensure `haarcascade_frontalface_default.xml` is in the `data/` folder.
- If the camera does not open, check webcam permissions.
- If `test.py` does not recognize faces, ensure faces were added using `add_face.py`.

## Future Improvements
- Improve face recognition accuracy with deep learning models.
- Develop a web-based interface for easier access.
- Store attendance records in a database instead of CSV files.

## License
This project is open-source and available under the MIT License.

## Contributors
- Mannat Sood (soodmannat16@gmail.com / mannatsood08)
![Screenshot 2025-02-05 134240](https://github.com/user-attachments/assets/11f6295c-78cd-4699-a286-a3a464eb8aa6)

