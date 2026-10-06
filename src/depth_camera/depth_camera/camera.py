"""Start the Orbbec camera, publishing only RGB image, depth image and point cloud.
Only sets up the camera. Anything that USES the data goes in its own file (depth_grid.py, ...).

Usage:
  ros2 run depth_camera camera                     # RGB + depth + point cloud
  ros2 run depth_camera camera --no-cloud          # RGB + depth only (lighter on the Jetson)
  ros2 run depth_camera camera --rgb-cloud         # point cloud colored by RGB
  ros2 run depth_camera camera -p depth_fps:=15    # override any param

Topics:
  /camera/color/image_raw           RGB image
  /camera/depth/image_raw           depth image
  /camera/depth/points              point cloud      (frame: camera_depth_optical_frame)
  /camera/depth_registered/points   RGB point cloud  (frame: camera_color_optical_frame)
"""
import os
import sys

# ---------------- Camera config (edit here) ----------------
# Gemini 336L supported sizes:
#   color: 424x240 480x270 640x360 640x400 640x480 848x480 1280x720 1280x800
#          fps 5/10/15/30 (60 up to 848x480, and MJPG), format MJPG/RGB/BGR/YUYV...
#   depth: 424x240 480x270 640x360 640x400 640x480 848x480 1280x720 1280x800
#          fps 5/15/30 (60 up to 848x480), format Y16
PARAMS = {
    'camera_name': 'camera',

    # RGB stream
    'enable_color': True,
    'color_width': 640,
    'color_height': 360,
    'color_fps': 30,           # 15 = even lighter
    'color_format': 'RGB',     # raw RGB: no JPEG decoding on the CPU

    # Depth stream
    'enable_depth': True,
    'depth_width': 640,
    'depth_height': 360,
    'depth_fps': 30,           # 15 = even lighter
    'depth_format': 'Y16',

    # Point cloud
    'enable_point_cloud': True,
    'point_cloud_decimation_filter_factor': 2,  # 1 = all points, 4 = fewer

    # Depth range: drop everything nearer than min / farther than max (mm).
    # Applies to both depth image and point cloud. Set at startup only.
    'enable_threshold_filter': False,
    'threshold_filter_min': 100,
    'threshold_filter_max': 5000,

    # Depth filters
    'enable_noise_removal_filter': False,
    'enable_spatial_filter': False,     # smooth surfaces
    'enable_temporal_filter': False,    # smooth over time
    'enable_hole_filling_filter': False,

    # Publish raw images only (no /compressed, /theora, ... topics)
    'color.image_raw.enable_pub_plugins': '[image_transport/raw]',
    'depth.image_raw.enable_pub_plugins': '[image_transport/raw]',

    # Do NOT add 'enable_depth_auto_exposure_priority': this camera's firmware
    # rejects it and the driver then fails to start.
}

# Extra params for --rgb-cloud (depth aligned to color, then colored)
RGB_CLOUD = {
    'enable_point_cloud': False,
    'enable_colored_point_cloud': True,
    'depth_registration': True,
    'align_mode': 'SW',  # 'HW' is not supported by this camera
    'depth.image_unaligned.enable_pub_plugins': '[image_transport/raw]',
}

# Extra params for --no-cloud (only RGB + depth images)
NO_CLOUD = {
    'enable_point_cloud': False,
}
# -----------------------------------------------------------


def main():
    args = sys.argv[1:]
    if '--rgb-cloud' in args:
        args.remove('--rgb-cloud')
        PARAMS.update(RGB_CLOUD)
    if '--no-cloud' in args:
        args.remove('--no-cloud')
        PARAMS.update(NO_CLOUD)

    cmd = ['ros2', 'run', 'orbbec_camera', 'orbbec_camera_node', '--ros-args',
           '-r', '__ns:=/camera', '-r', '__node:=camera']
    for name, value in PARAMS.items():
        if isinstance(value, bool):
            value = str(value).lower()
        cmd += ['-p', f'{name}:={value}']
    os.execvp('ros2', cmd + args)  # replace this process with the driver


if __name__ == '__main__':
    main()