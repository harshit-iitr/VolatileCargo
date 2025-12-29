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

*(Blue line represents movement of sprung mass, green represents actual track)*
---

## 🧪 Approaches & Experiments
To solve the challenge of High-Frequency Jerk with Latency, I experimented with three distinct control strategies:

### 1. Deep Reinforcement Learning (PPO)
* **Approach:** Trained a PPO (Proximal Policy Optimisation) agent using `stable-baselines3`.
* **Innovation:** Augmented the state space to include the **Delay Buffer History**. This allowed the policy network to learn the causal relationship between a command sent at $t$ and the force applied at $t+20ms$.
* **Tuning:** Utilized a **Cubic Reward Function** ($Reward \propto -|Jerk|^3$). This non-linear penalty forced the agent to prioritize minimizing large "spikes" (like the Profile 4 speed breaker) over small vibrations.

### 2. Hybrid Heuristic Controller (Tried)
* **Approach:** A rule-based system combining Skyhook theory with a threshold-based "Pothole Detector."
* **Observation:** While effective on wave profiles, the heuristic trigger was too slow to react to the sharp speed breaker due to the 4-step delay. The RL agent outperformed this by learning to anticipate the terrain.

---

## 🚀 How to Run

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt


2. To run the simulation and create submission_rl.csv:
```bash

python main.py
   
