# Automated Attendance Management System Using Face Recognition

This project implements an **automated attendance system** leveraging facial recognition technology. It provides a streamlined approach to attendance tracking, reducing manual errors and administrative overhead.

---

## Features

### Admin Capabilities
- Register new employees  
- Add employee photos to the training dataset  
- Train the facial recognition model  
- View and filter attendance reports by date or employee  

### Employee Capabilities
- Login to the system  
- Mark attendance via facial recognition (time-in and time-out)  
- View personal attendance records  

---

## Technologies Used
- **Face Detection**: face_recognition    
- **Backend**: Python and FastAPI  
- **Frontend**: React  
- **Database**: SQL  

---

## Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Aniruddha103/Automated-Attendance-Management-System-Using-Face-Recognition.git
cd Automated-Attendance-Management-System-Using-Face-Recognition

python -m venv venv
source venv/bin/activate    

pip install -r requirements.txt

cd attendance-frontend 
npm start 
uvicorn main:app --port 8000
