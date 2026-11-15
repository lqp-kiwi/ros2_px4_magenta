# Dùng chung source cho PC mô phỏng và Orin

Hai máy dùng chung source trong workspace này. `worlds/` và `models/` dành
cho Gazebo trên PC; Orin không cần chúng để chạy driver và node ROS.
`src/px4_control` đã là thư mục source độc lập, không còn trỏ sang
`/home/kiwi/ros2_px4_gz_ws`. Workspace gz gốc được giữ nguyên.

## Phiên bản

| Thành phần | Phiên bản |
| --- | --- |
| PC hiện tại | Ubuntu 22.04 / ROS 2 Humble / x86_64 |
| Orin Ubuntu 24.04 | ROS 2 Jazzy / aarch64; cần build và kiểm tra trên board |
| PX4 messages | `release/1.14`, commit `ffb6e80` |
| OrbbecSDK_ROS2 | commit `8e7cad2`, SDK 2.9.3, có thư viện arm64 |
| Micro XRCE-DDS Agent | v2.4.2, commit `57d0862` |

Ubuntu 24.04 dùng [ROS 2 Jazzy](https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html).
Mỗi terminal chỉ source một bản ROS. `px4_msgs` phải khớp firmware PX4;
giữ bản 1.14 hiện tại, không thay bằng nhánh `main` từ workspace gz.

Agent 2.4.2 được giữ từ workspace gz. Trên Humble, Agent dùng thư viện DDS
của ROS. Trên Jazzy, script build riêng các dependency Fast-CDR 1 và
Fast-DDS 2.12 để tránh liên kết với Fast-CDR 2 của Jazzy; lần build đầu cần
Internet. Bảng [tương thích của PX4](https://docs.px4.io/main/en/middleware/uxrce_dds)
khuyến nghị Agent 2.4.3 khi dùng bộ DDS có sẵn của Jazzy. Cách build 2.4.2
với dependency riêng ở đây chưa được xác nhận trên Orin. Không dùng
Agent 3.x với PX4 1.14.

## Chuyển source lần đầu

Trên PC, chạy từ workspace này:

```bash
bash tools/export_orin.sh /tmp/ros2_px4_orin.tar.gz
scp /tmp/ros2_px4_orin.tar.gz USER@ORIN_IP:~/
```

Trên Orin, giải nén vào workspace mới để giữ cấu hình đang dùng:

```bash
mkdir -p ~/ros2_px4_orin_ws
tar -xzf ~/ros2_px4_orin.tar.gz -C ~/ros2_px4_orin_ws
cd ~/ros2_px4_orin_ws
```

Gói chứa source driver và SDK arm64, bỏ qua Git metadata, tài nguyên Gazebo,
`build/`, `install/`, `log/` và dữ liệu bay. Không cần copy thêm từ workspace gz.

## Cài dependency và build trên Orin

Cài ROS 2 Jazzy theo hướng dẫn chính thức, sau đó mở terminal mới:

```bash
source /opt/ros/jazzy/setup.bash
sudo apt update
sudo apt install build-essential cmake git python3-colcon-common-extensions \
  python3-vcstool python3-rosdep python3-pip python3-numpy python3-opencv \
  python3-matplotlib python3-lxml ffmpeg \
  libasio-dev libtinyxml2-dev libspdlog-dev libeigen3-dev libyaml-cpp-dev \
  libdw-dev libssl-dev libgflags-dev libgoogle-glog-dev nlohmann-json3-dev \
  ros-jazzy-cv-bridge ros-jazzy-camera-calibration-parsers \
  ros-jazzy-image-transport-plugins ros-jazzy-compressed-image-transport \
  ros-jazzy-xacro
```

Nếu rosdep chưa khởi tạo, chạy `sudo rosdep init` một lần. Sau đó:

```bash
rosdep update
rosdep install --from-paths src --ignore-src --rosdistro jazzy -r -y \
  --skip-keys python3-pymavlink-pip
python3 -m pip install --target "$HOME/.local/share/ros2_px4_python" 'pymavlink>=2.4,<3'
export PYTHONPATH="$HOME/.local/share/ros2_px4_python${PYTHONPATH:+:$PYTHONPATH}"
bash tools/build_workspace.sh --with-agent
source install/jazzy-aarch64/setup.bash
```

Script chỉ build; không cài apt package hay chạy camera, Gazebo hoặc mission.
`pymavlink` được dùng trong mission GPS hiện tại. Lệnh pip cài vào thư mục
riêng để dùng được với Python hệ thống của ROS trên Ubuntu 24.04; thêm dòng
`export PYTHONPATH=...` ở trên vào terminal chạy mission khi cần.
Các package được build tuần tự. Với board ít RAM, giảm số tác vụ biên dịch
bằng `--jobs` theo `--help`.

Mỗi bản ROS và kiến trúc CPU dùng thư mục `build/`, `install/`, `log/` riêng.
Trên Orin, các package ROS được cài tại `install/jazzy-aarch64/`; Agent ở
`install/agent-jazzy-aarch64/bin/MicroXRCEAgent`. Không cần đưa thư viện DDS
riêng của Agent vào môi trường ROS.

Trên PC Humble, dùng các package `ros-humble-*` tương ứng và cài thêm
`ros-humble-fastcdr ros-humble-fastrtps` để build Agent với DDS hệ thống.

## Quyền USB và kiểm tra camera

Trên Orin, cài udev rule một lần rồi rút và cắm lại camera:

```bash
sudo cp src/OrbbecSDK_ROS2/orbbec_camera/scripts/99-obsensor-libusb.rules \
  /etc/udev/rules.d/
sudo udevadm control --reload-rules
sudo udevadm trigger
```

Sau khi build và source `install/jazzy-aarch64/setup.bash` trên Orin:

```bash
ros2 run orbbec_camera list_devices_node
ros2 run depth_camera camera --no-cloud
```

Mở terminal khác, source cùng workspace rồi chạy
`ros2 run depth_camera depth_grid`. Grid mặc định giới hạn 5 m; camera hiện
tại lọc khoảng 0,5–10 m. Thêm `--ros-args -p max_range:=10.0` nếu cần grid
đến 10 m. Với Gemini 336L, giữ software RGB alignment và không thêm tham số
`enable_depth_auto_exposure_priority`.

SIYI mặc định dùng IP `192.168.144.25`; có thể đổi `ip` hoặc `rtsp_url`
qua ROS parameter. Chưa kiểm tra camera, kết nối PX4 hay bay trên board.

## Đồng bộ các lần sau

Git đồng bộ các package tự viết, cấu hình, tools và tài nguyên mô phỏng.
Ba repo ngoài được ghim phiên bản trong `drivers.repos`, không nhúng source
vào Git cha. Trên clone mới, chạy:

```bash
vcs import --skip-existing . < drivers.repos
```

Lệnh bổ sung repo còn thiếu và giữ nguyên checkout, cấu hình đã có.
Với repo Git có cùng URL, lệnh vẫn fetch metadata nhưng không đổi checkout.
Khi muốn đổi dependency, cập nhật commit trong `drivers.repos` có chủ đích.

Sau khi commit và push source này, hai máy có thể pull cùng branch. Orin
không cần cài Gazebo để build các package trong `src`. Gói export chỉ là
snapshot để chuyển source, không tự đồng bộ hai chiều. Hiện chưa push lên GitHub.

## Mô phỏng trên PC

Giữ phiên bản PX4 và Gazebo đang dùng. Thêm `models/` của workspace này vào
`GZ_SIM_RESOURCE_PATH` cùng các model của PX4. `PX4_GZ_WORLD` phải khớp tên
world. Một số world cần tài nguyên Fuel và cấu hình plugin trong PX4
`server.config`; build ROS không cài các thành phần này.

## Giới hạn hiện có

Một số console entry point cũ của `px4_control` trỏ tới module bị thiếu hoặc
sai tên. Lần đồng bộ này giữ nguyên code mission, không chạy lệnh arm/offboard.
Kiểm tra tên module trước khi chọn mission.
