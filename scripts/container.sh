#!/usr/bin/env bash
# Copy only this project into /tmp in the running lecture container.
set -euo pipefail
cd "$(dirname "$0")/.."
container=${PICKUP_CONTAINER:-ros2-docker}
remote=/tmp/physical_ai_pickup
# Exclude upstream checkout and generated data; use the existing pinned training runtime.
docker exec "$container" mkdir -p "$remote"
tar --exclude=.git --exclude=lerobot --exclude=.pixi --exclude=data --exclude=checkpoints --exclude=__pycache__ -cf - . | docker exec -i "$container" tar -xf - -C "$remote"
docker exec -it -e WANDB_MODE="${WANDB_MODE:-offline}" -u ubuntu "$container" bash -lc '
set -e
source /opt/ros/humble/setup.bash
source /home/ubuntu/airobot_ws/install/setup.bash
cd /tmp/physical_ai_pickup
mkdir -p /home/ubuntu/pickup_experiment/data /home/ubuntu/pickup_experiment/checkpoints
[[ -e data ]] || ln -s /home/ubuntu/pickup_experiment/data data
[[ -e checkpoints ]] || ln -s /home/ubuntu/pickup_experiment/checkpoints checkpoints
if [[ ! -x /tmp/physical_ai_pickup/object_pose || native/object_pose.cpp -nt /tmp/physical_ai_pickup/object_pose ]]; then
  g++ -std=c++17 native/object_pose.cpp -o /tmp/physical_ai_pickup/object_pose $(pkg-config --cflags --libs ignition-transport11 ignition-msgs8)
fi
export PYTHONPATH="/tmp/physical_ai_pickup:${PYTHONPATH:-}"
exec /home/ubuntu/robots/pai_training/.pixi/envs/default/bin/python "$@"
' bash "$@"
