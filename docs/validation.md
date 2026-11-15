# Integration validation — 2026-11-15

Validated on Ubuntu 22.04.5, x86_64, ROS 2 Humble.

## Build

```bash
bash tools/build_workspace.sh --ros-distro humble --with-agent --jobs 6
```

Passed: all eight ROS packages (`px4_msgs`, `px4_control`, `depth_camera`,
`orbbec_camera_msgs`, `orbbec_camera`, `orbbec_description`, `siyi_interfaces`,
`siyi_camera`) and the separate Micro XRCE-DDS Agent 2.4.2 build.

ROS outputs are in `build/humble-x86_64` and `install/humble-x86_64`.
The Agent is in `install/agent-humble-x86_64`. The build uses separate
directories from the old workspace cache.

`ros2 pkg executables` lists all three depth-camera executables, including the
restored `depth_grid`, and all four SIYI executables. The control package prefix
resolves inside this workspace's new install directory.

## Tests

```bash
source /opt/ros/humble/setup.bash
source install/humble-x86_64/setup.bash
colcon --log-base log/humble-x86_64 test \
  --build-base build/humble-x86_64 --install-base install/humble-x86_64 \
  --packages-select depth_camera px4_control siyi_camera \
  --return-code-on-test-failure
colcon test-result --test-result-base build/humble-x86_64 --verbose
```

Result: 9 tests, 0 errors, **6 failures**, 3 skipped.
Each Python package fails its existing `flake8` and `pep257` checks; the
copyright checks are skipped. The lint failures concern preserved source and
source imported from the gz workspace. No Python implementation was reformatted
or changed during integration. These checks do not verify flight/perception.

The two new Bash scripts pass `bash -n` and ShellCheck. Invalid build arguments
and attempts to overwrite an existing export archive are rejected.

## Preservation and portability

- SHA-256 checks confirm all original control and depth-camera files except
  the deliberately updated dependency manifests retain their original bytes.
- The materialized control directory and copied drivers have independent files;
  the original gz workspace is unchanged.
- Third-party Git revisions match `drivers.repos`; those checkouts remain clean.
- The Orbbec SDK includes a real AArch64 library. Package XML and Python source
  parse successfully.
- The export contains source and the ARM64 SDK, with no external symlinks,
  Git metadata, workspace build outputs, flight logs, or Gazebo assets.

## Pending

Ubuntu 24.04/Jazzy/Orin compilation and hardware operation have not been tested.
No camera, simulator, PX4 connection, or flight mission was started.
The user requested synchronization later: no transfer to Orin or GitHub push
was performed, and changes remain uncommitted in the local working tree.
