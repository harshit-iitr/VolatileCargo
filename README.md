# PS2: The Volatile Cargo
**Stark Industries Hackathon Submission**

## 👤 Student Details
* **Name:** Harshit Agrawal
* **Enrolment Number:** 8962424738
* **Branch:** Data Science & Artificial Intelligence
* **Year:** 1st Year

---

## Final Result
* **Method:** Reinforcement Learning (PPO)
* **Best Score:** **57** (Overall Score)

My final submission uses a **Proximal Policy Optimization (PPO)** agent trained to perform trajectory optimization on the provided road profiles. The agent successfully learned to compensate for the 20ms actuator delay by analyzing the delay buffer history.

### Model Performance

![RL Response](output/Screenshot%202025-12-29%20200740.png)


(Blue line represents movement of sprung mass, green represents actual track)

---


## 🚀 How to Run

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt


2. To run the simulation and create submission_rl.csv:
```bash

python main.py
   
