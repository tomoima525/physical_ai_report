// Preserve model names: this lecture's Pose_V -> TFMessage bridge drops them.
#include <ignition/transport/Node.hh>
#include <ignition/msgs/pose_v.pb.h>
#include <iostream>
#include <iomanip>
#include <string>
std::string object_name;
void poses(const ignition::msgs::Pose_V &msg) {
  for (const auto &pose : msg.pose()) {
    if (pose.name() == object_name) {
      const auto &p = pose.position();
      std::cout << std::setprecision(17) << p.x() << " " << p.y() << " " << p.z() << std::endl;
    }
  }
}
int main(int argc, char **argv) {
  if (argc != 3) return 2;
  object_name = argv[2];
  ignition::transport::Node node;
  if (!node.Subscribe(std::string("/world/") + argv[1] + "/pose/info", poses)) return 1;
  ignition::transport::waitForShutdown();
}
