# SETUP:
```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot

conda create -y -n lerobot python=3.12
conda activate lerobot

conda install -y -c conda-forge ffmpeg
pip install -e .

python -c "import lerobot; print('LeRobot installed')"

# Install related libraries
pip install -e ".[feetech]"
pip install -e ".[hardware]"
pip install -e ".[aloha]"
pip install -e ".[dataset]"
pip install -e ".[viz]"

```

# ROBOT BRING-UP:

## 1. Find the USB Ports:
- Use **_lerobot-find-port_** to find the ports of the leader and follower arm. 
- Run **_lerobot-find-port_** with both arms connected. Remove follower arm, press enter and we’ll get the port for the follower. Same for the leader arm.
1.  Follower Arm USB: **"/dev/ttyACM1"**
2. Leader Arm USB: **"/dev/ttyACM0"**
## 2. Find Camera Ports:
Use **_lerobot-find-cameras_** to record the images from the cameras corresponding to the ports.
1. External Cam: OpenCV Camera @ /dev/video10
2. Follower Cam: OpenCV Camera @ /dev/video4

## 3. Calibrate:
- Saves the configurations at _/home/user/.cache/huggingface/lerobot/calibration/robots/so_follower/my_so101_follower.json_ and _/home/user/.cache/huggingface/lerobot/calibration/teleoperators/so_leader/my_so101_leader.json_.
### Follower Arm:
```python
lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/ttyACM1 --robot.id=my_so101_follower
```
### Leader Arm:
```python 
lerobot-calibrate --teleop.type=so101_leader --teleop.port=/dev/ttyACM0 --teleop.id=my_so101_leader
```

## 4. Teleoperate
- To make sure that there is no lag between the two arms and the calibration is done correctly.
```python
lerobot-teleoperate --robot.type=so101_follower --robot.port=/dev/ttyACM1 --robot.id=my_so101_follower --teleop.type=so101_leader --teleop.port=/dev/ttyACM0 --teleop.id=my_so101_leader
```

# DATASETS:
## 1. Local Folders:
- To save the recorded demos and trained models/policies locally.

```bash
--repo-id local/so101   --root /home/harsh-mulodhia/so101/lerobot/data/pp-fixed
```

## 2. RECORD:
- To save demonstrations on the setup for training.

```bash 
lerobot-record --robot.type=so101_follower --robot.port=/dev/ttyACM1 --robot.id=my_so101_follower --teleop.type=so101_leader --teleop.port=/dev/ttyACM0 --teleop.id=my_so101_leader --robot.cameras="{ext: {type: opencv, index_or_path: 8, width: 640, height: 480, fps: 30}, arm: {type: opencv, index_or_path: 10, width: 640, height: 480, fps: 30}}" --dataset.repo_id=local/so101 --dataset.root=/home/harsh-mulodhia/so101/lerobot/data/pp-fixed --dataset.num_episodes=15 --dataset.single_task="pick and place the blue ball in the box" --display_data=true --dataset.push_to_hub=false --resume=false
```

- Add extra cams in robot.cameras dict as :
  - front: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}


### 3. Replay an episode:
```bash
lerobot-replay --robot.type=so101_follower --robot.port=/dev/ttyACM1 --robot.id=my_so101_follower --dataset.repo_id=local/so101 --dataset.root=/home/harsh-mulodhia/so101/lerobot/data/pp1 --dataset.episode=0
```

## 4. Visualize Dataset:
- Run the command below and open the [URL](http://localhost:9095/?url=rerun%2Bhttp%3A%2F%2Flocalhost%3A9879%2Fproxy).
```bash
lerobot-dataset-viz --repo-id local/so101 --root /home/harsh-mulodhia/so101/lerobot/data/pp-fixed --mode distant --episode-index 0 --web-port 9095 --grpc-port 9879
```
- Or use the official **LeRobot Viewer** VS Code Extension.

## 5. Merge Datasets:
```bash
lerobot-edit-dataset --operation.type merge --operation.repo_ids "['local/pp1', 'local/pick-object']" --operation.roots "['/mnt/data/PhysicalAI/Harsh/SO101/lerobot/data/pp1', '/mnt/data/PhysicalAI/Harsh/SO101/lerobot/data/pick-object']" --new_repo_id local/multitask-v1 --new_root /mnt/data/PhysicalAI/Harsh/SO101/lerobot/data/multitask-v1
```

# Training the Robot arm:

## 1. ACT:
```bash
lerobot-train --dataset.repo_id=local/so101 --dataset.root=/mnt/data/PhysicalAI/Harsh/SO101/lerobot/data/pp-fixed --policy.type=act --output_dir=outputs/train/act_so101_pp-fixed --job_name=act_so101_pp-fixed --policy.device=cuda --wandb.enable=false --policy.push_to_hub=false --steps=60000 --save_checkpoint=true --save_freq=5000
```

## 2. SmolVLA
- SmolVLA base model expects cameras named camera1 and camera2.
```bash
lerobot-train --policy.path=lerobot/smolvla_base --dataset.repo_id=local/so101 --dataset.root=/mnt/data/PhysicalAI/Harsh/SO101/lerobot/data/pp-fixed --output_dir=outputs/train/smolvla_so101_pp-fixed --job_name=smolvla_so101_pp-fixed1 --policy.device=cuda --policy.use_amp=true --policy.push_to_hub=false --wandb.enable=false --batch_size=8 --steps=60000 --eval.n_episodes=0 --rename_map='{"observation.images.arm": "observation.images.camera1", "observation.images.ext": "observation.images.camera2"}'
```

## 3. Gr00t
```bash
nohup lerobot-train --policy.type=groot --policy.pretrained_path=nvidia/GR00T-N1.7-3B --dataset.repo_id=local/so101 --dataset.root=/mnt/data/PhysicalAI/Harsh/SO101/lerobot/data/pp-fixed --output_dir=outputs/train/groot_so101_pp-fixed --job_name=groot_so101_pp-fixed1 --policy.device=cuda --policy.use_amp=true --policy.push_to_hub=false --wandb.enable=false --batch_size=8 --steps=100000 --eval.n_episodes=0 --rename_map='{"observation.images.arm": "observation.images.camera1", "observation.images.ext": "observation.images.camera2"}'  > logs/train_groot100k.log 2>&1 &

```


# INFERENCE:

## EVAL OR TRAINED REPLAY:
### 1. Using server for inference
- Run on the server container (training_roboarm):
```bash
python3 -m lerobot.async_inference.policy_server --host=0.0.0.0 --port=8080
```
- Run on the local machine:
### 1. ACT
```bash 
python -m lerobot.async_inference.robot_client --robot.type=so101_follower --robot.port=/dev/ttyACM1 --robot.id=my_so101_follower --robot.cameras="{arm: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, ext: {type: opencv, index_or_path: 12, width: 640, height: 480, fps: 30}}" --task="pick and place the blue ball in the box" --server_address=192.168.17.177:8080 --policy_type=act --pretrained_name_or_path=/mnt/data/PhysicalAI/Harsh/SO101/lerobot/outputs/train/act_so101_pp-fixed1/checkpoints/last/pretrained_model --policy_device=cuda --debug_visualize_queue_size=False --actions_per_chunk=50
```
### 2. SmolVLA
```bash
python -m lerobot.async_inference.robot_client --robot.type=so101_follower --robot.port=/dev/ttyACM1 --robot.id=my_so101_follower --robot.cameras="{arm: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, ext: {type: opencv, index_or_path: 12, width: 640, height: 480, fps: 30}}" --task="pick and place the blue ball in the box" --server_address=192.168.17.177:8080 --policy_type=act --pretrained_name_or_path=/mnt/data/PhysicalAI/Harsh/SO101/lerobot/outputs/train/smolvla_so101_pp-fixed/checkpoints/last/pretrained_model -- policy_device=cuda --debug_visualize_queue_size=False --actions_per_chunk=50
```

### 3. Gr00t
```bash 


```

# 2. For same system inference
- For a single-machine local execution loop. It expects the policy, the GPU, and the physical robot to all be running on the same computer
```bash
lerobot-rollout --robot.type=so101_follower --robot.port=/dev/ttyACM1 --robot.id=my_so101_follower --policy.path=outputs/train/act_so101_pp1/checkpoints/last/pretrained_model --policy.device=cpu
```

```bash
lerobot-eval --policy.path=outputs/train/act_so101_pp1/checkpoints/last/pretrained_model --env.type=aloha --eval.device=cpu --eval.n_episodes=10
```

### Train on combined dataset:
```bash
lerobot-train --dataset.repo_id=local/multitask-v1 --dataset.root=/mnt/data/PhysicalAI/Harsh/SO101/lerobot/data/multitask-v1 --policy.type=act --output_dir=outputs/train/act_so101_multitask --job_name=act_so101_multitask --policy.device=cuda --wandb.enable=false --policy.push_to_hub=false --steps=50000 --save_checkpoint=true --save_freq=5000
```

# Extras
## smolvla

---

# Jetson
- Connect with **Jetson Orin NX** using the **Seeed Studio reServer Industrial** device using **SSH**. -Y flag for display.
```bash
ssh -Y groot@192.168.100.123
cd /home/groot/Harsh/so101
```

```bash
docker start lerobot_dev
docker exec -it lerobot_dev /bin/bash

source /opt/venv/bin/activate

pip install --isolated --no-cache-dir --index-url https://pypi.org/simple -e .
pip install --isolated --no-cache-dir --index-url https://pypi.org/simple -e ".[feetech]"
pip install --isolated --no-cache-dir --index-url https://pypi.org/simple -e ".[hardware]"
pip install --isolated --no-cache-dir --index-url https://pypi.org/simple -e ".[dataset]"
pip install --isolated --no-cache-dir --index-url https://pypi.org/simple -e ".[viz]"
pip install --isolated --no-cache-dir --index-url https://pypi.org/simple -e ".[dev]"
pip install --isolated --no-cache-dir --index-url https://pypi.org/simple --upgrade transformers
```
