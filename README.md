# ME5413 Final Project - Jackal Autonomous Navigation

This workspace contains simulation, mapping, and navigation packages developed for the ME5413 Final Project, involving a Jackal robot navigating through a Gazebo environment using SLAM, autonomous planning, and obstacle avoidance.

---

## 📁 Workspace Structure

```
ME5413_Final_Project/
├── src/
│   ├── jackal_nav/                  # Custom navigation package
│   │   ├── launch/                  # Navigation launch files (move_base, amcl, auto navigation)
│   │   ├── config/                  # Costmap and planner parameter YAMLs
│   │   │   ├── costmap_common_params.yaml
│   │   │   ├── local_costmap_params.yaml
│   │   │   ├── global_costmap_params.yaml
│   │   │   ├── base_local_planner_params.yaml
│   │   │   ├── teb_local_planner_params.yaml
│   │   │   └── global_planner_params.yaml
│   │   ├── maps/                    # Pre-saved map files (e.g., map1.yaml + map1.pgm)
│   │   ├── scripts/                 # Python scripts (e.g. auto_goal_publisher.py)
│   ├── me5413_world/               # Project simulation environment
│   │   ├── launch/                  # World and simulation launch files
│   │   ├── rviz/                    # RViz configuration files
│   │   ├── worlds/                  # Gazebo world files
│   │   ├── config/                  # Goal and map parameters
│   ├── jackal_description/         # Jackal robot URDF and model
│   └── other_dependencies/...      # (e.g. teleop_twist_keyboard, gmapping)
```

---

## 🚀 Launch Instructions

### 1. Build the workspace

```bash
cd ~/ME5413_Final_Project
catkin_make
source devel/setup.bash
```

### 2. Launch navigation

This will launch the following core components:

- `map_server`: loads the static map (from `maps/map1.yaml`)
- `amcl`: adaptive Monte Carlo localization for pose estimation
- `move_base`: handles global planning and local obstacle avoidance
- `rviz`: visualization interface with navigation config

```bash
roslaunch jackal_nav navigation.launch
```

### 3. Optional: Auto goal publishing

```bash
rosrun jackal_nav auto_goal_publisher.py
```

---

## ⚙️ Key Features

- SLAM with GMapping / Cartographer
- Autonomous Navigation with `move_base`
- Global planner: `NavfnROS` / `GlobalPlanner`
- Local planner: `TrajectoryPlannerROS` / `teb_local_planner`
- Adaptive Monte Carlo Localization (AMCL)
- Obstacle inflation, layered costmaps, dynamic reconfiguration
- RViz visualization and manual goal tools

---

## 📌 Troubleshooting

- **Localization errors:** Use RViz's 2D Pose Estimate to initialize position.
- **Navigation fails:** Check TF, costmap parameters, and inflation radius.
- **Robot doesn’t move:** Make sure `/cmd_vel` is being published by move\_base.

---

## ✍️ Author & Contributions

- Navigation and state machine: [Your Name]
- Simulation and environment setup: [Teammate A]
- Mapping and localization: [Teammate B]

---

## 📚 References

- Jackal Simulation & Navigation Tutorials
- ROS Navigation Stack
- Clearpath Robotics documentation
- ME5413 course materials

---

Feel free to fork, extend, or contribute!

> "Autonomous robots begin with great maps and smarter plans."

