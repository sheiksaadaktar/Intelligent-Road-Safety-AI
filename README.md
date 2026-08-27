# Intelligent Road Safety AI

An AI-based road safety monitoring system that uses computer vision to detect and track vehicles, analyze their movement, calculate distances between road users, and identify potentially dangerous situations.

## 📌 Project Overview

Road accidents can occur due to unsafe distances, sudden movements, high relative speeds, and interactions between multiple vehicles.

The goal of this project is to develop an **Intelligent Road Safety AI system** that can analyze traffic video footage and automatically identify potentially risky interactions between vehicles.

The system currently uses **YOLO object detection and tracking** to identify vehicles and maintain their identities across video frames. It then performs movement and distance analysis to estimate whether vehicles are approaching each other and assigns a risk level.

The project is being developed incrementally, with each stage tested and committed to GitHub.

---

## 🎯 Objectives

The main objectives of the project are:

- Detect vehicles in road video footage.
- Track individual vehicles across consecutive frames.
- Assign a unique ID to each detected vehicle.
- Calculate vehicle movement and velocity.
- Calculate distances between detected vehicles.
- Determine whether two vehicles are approaching each other.
- Assign a risk level based on distance and relative movement.
- Visualize detected vehicles and risk levels directly on the video.
- Build a foundation for a real-time intelligent road safety monitoring system.

---

## 🧠 System Workflow

The current system follows this general pipeline:

```text
Road Video
    ↓
YOLO Object Detection
    ↓
Multi-Object Tracking
    ↓
Vehicle Identification
    ↓
Movement / Velocity Analysis
    ↓
Pairwise Distance Calculation
    ↓
Approaching Vehicle Detection
    ↓
Risk Score Calculation
    ↓
Risk Classification
    ↓
Visualized Output Video


🚗 Vehicle Detection and Tracking

The project uses the Ultralytics YOLO framework for object detection and tracking.

Detected objects are assigned tracking IDs so that the system can follow individual vehicles across frames.

For example:

ID 6  → bus
ID 18 → car
ID 25 → car
ID 28 → car
ID 17 → car

The tracking IDs allow the system to analyze how each vehicle moves over time.


📊 Movement Analysis

For each tracked object, the system calculates movement between consecutive frames.

The movement is calculated using the Euclidean distance between the current and previous center positions of the object.

Conceptually:

movement = √((x₂ - x₁)² + (y₂ - y₁)²)

The system then uses the video FPS to estimate velocity in pixels per second.

A short moving average is also used to smooth the movement measurements and reduce frame-to-frame noise.

Example output:

Frame: 599 | ID: 18 | car | Velocity: 554.78 px/sec
Frame: 599 | ID: 17 | car | Velocity: 317.41 px/sec
📏 Distance Analysis

The system calculates the distance between every pair of tracked vehicles.

For two vehicles, the distance between their center points is calculated using:

distance = √((x₁ - x₂)² + (y₁ - y₂)²)

Example:

ID 18 (car) <-> ID 17 (car) = 123.12 px

This allows the system to monitor vehicle interactions rather than analyzing each vehicle independently.

⚠️ Approaching Vehicle Detection

The system compares the current distance between two vehicles with their previous distance.

If:

current distance < previous distance

the vehicles are considered to be approaching each other.

Example:

Previous distance: 147 px
Current distance: 123 px

Approaching: YES

This information is used together with distance and closing speed to estimate potential risk.

🚨 Risk Analysis

The project includes a risk analysis component that evaluates interactions between vehicles.

The system considers factors such as:

Distance between vehicles
Whether vehicles are approaching
Closing speed
Relative movement

A risk score is generated and converted into a risk category.

The current categories include:

LOW
MEDIUM
HIGH
CRITICAL

Example:

Frame: 599
HIGH/CRITICAL RISK
ID 18 (car) <-> ID 17 (car)
Distance: 123.12 px
Closing Speed: 1150.52 px/s
Score: 80

The risk system is currently a research/development implementation and will be refined as the project progresses.

🎥 Risk Visualization

The project also generates a processed video showing the detected vehicles and their risk information.

The visualization helps us inspect how well the system is performing directly on the original road footage.

The generated output currently includes information such as:

Vehicle tracking IDs
Object classes
Vehicle positions
Distance information
Risk levels
Risk interactions

Example output file:

output/risk_analysis.mp4


📂 Project Structure
Intelligent-Road-Safety-AI/
│
├── data/
│   └── road.jpg
│
├── src/
│   ├── movement_analysis.py
│   ├── multi_object_test.py
│   ├── object_counter.py
│   ├── risk_analysis.py
│   ├── risk_visualization.py
│   ├── test_opencv.py
│   ├── tracking_data.py
│   ├── tracking_test.py
│   ├── video_test.py
│   ├── yolo_test.py
│   └── yolo_video_test.py
│
├── requirements.txt
├── README.md
└── .gitignore


🛠️ Technologies Used

The project currently uses:

Python
OpenCV
YOLO
Ultralytics
NumPy
Computer Vision
Multi-Object Tracking
Git
GitHub


💻 Installation
1. Clone the repository
git clone https://github.com/sheiksaadaktar/Intelligent-Road-Safety-AI.git
2. Enter the project directory
cd Intelligent-Road-Safety-AI
3. Create a virtual environment

Windows:

python -m venv venv
4. Activate the virtual environment

Windows PowerShell:

venv\Scripts\Activate.ps1
5. Install dependencies
pip install -r requirements.txt


▶️ Running the Project
Movement Analysis

Run:

python src/movement_analysis.py

This analyzes vehicle movement and estimates velocity.

Risk Analysis

Run:

python src/risk_analysis.py

This analyzes vehicle interactions, distances, approaching behavior, and risk levels.

Risk Visualization

Run:

python src/risk_visualization.py

The processed video will be generated in:

output/risk_analysis.mp4

On Windows, the output video can be opened using:

start output\risk_analysis.mp4


📈 Example Analysis

The system can produce output similar to:

Frame: 598 | Distance |
ID 18 (car) <-> ID 17 (car) =
142.32 px

Approaching: True
Risk: CRITICAL
Score: 75

This indicates that the two tracked vehicles are relatively close and their distance/closing behavior has triggered a high-risk assessment.


🔬 Current Development Stage

The project is currently in the prototype and algorithm-development stage.

The following components have already been implemented:

 YOLO object detection
 Video-based vehicle detection
 Multi-object tracking
 Vehicle tracking IDs
 Movement measurement
 Velocity estimation
 Pairwise vehicle distance calculation
 Approaching vehicle detection
 Basic risk classification
 Risk scoring
 Risk visualization
 Git version control
 GitHub repository


🚧 Future Development

Planned improvements include:

1. Improved Risk Estimation

The current risk system primarily uses image-space measurements.

Future versions will improve the risk model by considering:

Relative velocity
Direction of movement
Time-to-collision (TTC)
Vehicle trajectories
Object size
Road perspective
Lane information
2. Time-to-Collision (TTC)

A major future improvement is implementing Time-to-Collision.

Instead of considering only distance, the system will estimate how long it would take for two approaching vehicles to reach the same position if their current motion continues.

Conceptually:

TTC = Distance / Closing Speed

This can provide a more meaningful measurement of collision risk.

3. Lane Detection

Future versions may identify road lanes and determine whether vehicles are:

Staying within their lane
Changing lanes
Drifting between lanes
Moving into another vehicle's path
4. Collision Detection

The system will eventually be improved to distinguish between:

Normal interaction
        ↓
Potential conflict
        ↓
High-risk interaction
        ↓
Possible collision
5. Real-Time Monitoring

The long-term goal is to support real-time road safety monitoring using:

CCTV cameras
Traffic cameras
Dash cameras
Other video sources
6. Dashboard

A future version may include a monitoring dashboard displaying:

Vehicles Detected
Active Tracks
Low Risk Events
Medium Risk Events
High Risk Events
Critical Events

along with live video and alerts.


🧪 Testing

The project is being tested using recorded road traffic video.

Testing focuses on:

Detection accuracy
Tracking stability
Vehicle ID consistency
Movement estimation
Distance calculation
Risk classification
Visualization quality

Testing will continue as new features are added.


📌 Important Note About Measurements

The current distance and velocity values are measured in pixels, not physical units such as meters or kilometers per hour.

For example:

Distance: 142 px
Velocity: 350 px/sec

These values depend on the camera position, video resolution, perspective, and scene geometry.

Future development will investigate camera calibration and perspective transformation so that physical-world measurements can be estimated more accurately.



🔄 Development and Version Control

Git and GitHub are being used to maintain the development history of the project.

Major changes are committed separately so that the development process can be tracked.

Example commit history:

Initial project setup
        ↓
Add vehicle distance and risk analysis
        ↓
Add risk analysis visualization
        ↓
Future improvements...

This makes it easier to:

Track changes
Revert problematic changes
Collaborate with teammates
Review development progress
Maintain different versions of the system

👥 Team

This project is being developed as a collaborative road safety AI project.

Team members:

Mohammed Taif
Sheik Saad Aktar
Adil Shahan

Additional contributors will be added as the project develops.

📜 License

This project is currently intended for educational and research purposes.

A formal open-source license may be added in a future version.


🚀 Project Vision

The ultimate goal of the Intelligent Road Safety AI project is to create an intelligent computer-vision system capable of continuously monitoring road environments, understanding interactions between road users, identifying dangerous situations, and providing early warnings before accidents occur.

The project will progressively move from basic object detection toward a more comprehensive understanding of:

Vehicles
   ↓
Movement
   ↓
Interaction
   ↓
Trajectory
   ↓
Collision Risk
   ↓
Early Warning

The aim is to use AI and computer vision to make road environments safer through proactive risk detection.