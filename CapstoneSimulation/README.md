This version documents **how to run Gazebo (Ignition/Fortress+)** properly with ROS 2 Jazzy, how to spawn models, and includes quick troubleshooting notes.

---

```markdown
# 🧠 Gazebo (Ignition) Setup & Usage Guide — ROS 2 Jazzy

This guide explains how to install, launch, and use the new Gazebo (formerly Ignition Gazebo) simulator with ROS 2 Jazzy.  
It also includes commands for spawning models directly using the `gz` CLI.

---

## 🚀 1. Verify Installation

After installing ROS 2 Jazzy, the Gazebo tools are bundled under:
```
```bash
/opt/ros/jazzy/opt/gz_sim_vendor/
```


To confirm `gz` is available:
```bash
which gz
# Expected output:
# /usr/bin/gz
```

To check Gazebo version:

```bash
gz --version
```

---

## 🧩 2. Launch Gazebo (Empty World)

Start an empty world:

```bash
gz sim empty.sdf
```

If the GUI opens successfully, you’re good to go.
Keep this window open when running the next commands.

---

## 📦 3. Spawn Models via CLI

### ✅ Box Example

Spawns a 1 × 1 × 1 m box at position (0, 0, 0.5):

```bash
gz service -s /world/empty/create \
  --reqtype gz.msgs.EntityFactory \
  --reptype gz.msgs.Boolean \
  --req 'sdf: "<sdf version=\"1.6\"><model name=\"box\"><pose>0 0 0.5 0 0 0</pose><link name=\"link\"><collision name=\"collision\"><geometry><box><size>1 1 1</size></box></geometry></collision><visual name=\"visual\"><geometry><box><size>1 1 1</size></box></geometry></visual></link></model></sdf>"'
```

You should see:

```
data: true
```

and the box should appear in your world.

---

### ⚪ Sphere Example

```bash
gz service -s /world/empty/create \
  --reqtype gz.msgs.EntityFactory \
  --reptype gz.msgs.Boolean \
  --req 'sdf: "<sdf version=\"1.6\"><model name=\"sphere\"><pose>1 0 0.5 0 0 0</pose><link name=\"link\"><collision name=\"collision\"><geometry><sphere><radius>0.5</radius></sphere></geometry></collision><visual name=\"visual\"><geometry><sphere><radius>0.5</radius></sphere></geometry></visual></link></model></sdf>"'
```

---

### 🟣 Cylinder Example

```bash
gz service -s /world/empty/create \
  --reqtype gz.msgs.EntityFactory \
  --reptype gz.msgs.Boolean \
  --req 'sdf: "<sdf version=\"1.6\"><model name=\"cylinder\"><pose>2 0 0.5 0 0 0</pose><link name=\"link\"><collision name=\"collision\"><geometry><cylinder><radius>0.3</radius><length>1.0</length></cylinder></geometry></collision><visual name=\"visual\"><geometry><cylinder><radius>0.3</radius><length>1.0</length></cylinder></geometry></visual></link></model></sdf>"'
```

---

## 🧱 4. Spawning Models from Files (Optional)

If you have an SDF or URDF file (e.g., `my_robot.sdf`), use:

```bash
gz service -s /world/empty/create \
  --reqtype gz.msgs.EntityFactory \
  --reptype gz.msgs.Boolean \
  --req 'sdf_filename: "/absolute/path/to/my_robot.sdf"'
```

You can also provide a pose:

```bash
--req 'sdf_filename: "/absolute/path/to/my_robot.sdf", pose: { position: {x: 0, y: 0, z: 1} }'
```

---

## ⚙️ 5. Useful Commands

| Command         | Description                 |
| --------------- | --------------------------- |
| `gz topic -l`   | List all active topics      |
| `gz service -l` | List all active services    |
| `gz stats`      | Show simulation stats       |
| `gz world info` | Display world info          |
| `gz log info`   | Check if logging is enabled |

---

## 🧰 6. Troubleshooting

| Problem                       | Fix                                                                  |
| ----------------------------- | -------------------------------------------------------------------- |
| GUI doesn’t open              | Check `gz sim --render-engine ogre2 empty.sdf`                       |
| Command says `invalid option` | You’re using old syntax (use `gz service` instead of `gz sim spawn`) |
| `data: false`                 | World not running or wrong service name                              |
| ROS can’t find Gazebo         | Source your setup: `source /opt/ros/jazzy/setup.bash`                |

---

## 📘 References

* Gazebo Docs: [https://gazebosim.org](https://gazebosim.org)
* ROS 2 Jazzy Docs: [https://docs.ros.org/en/jazzy](https://docs.ros.org/en/jazzy)

---

### ✅ Quick Recap

Run these 3 commands to test Gazebo quickly:

```bash
source /opt/ros/jazzy/setup.bash
gz sim empty.sdf
gz service -s /world/empty/create --reqtype gz.msgs.EntityFactory --reptype gz.msgs.Boolean --req 'sdf: "<sdf version=\"1.6\"><model name=\"box\"><pose>0 0 0.5 0 0 0</pose><link name=\"link\"><collision name=\"collision\"><geometry><box><size>1 1 1</size></box></geometry></collision><visual name=\"visual\"><geometry><box><size>1 1 1</size></box></geometry></visual></link></model></sdf>"'
```

---

> **Author:** Pranay Katyal
> **Tested on:** Ubuntu 24.04 LTS + ROS 2 Jazzy + Gazebo Harmonic (Ignition)
