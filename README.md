# PX4 workspace: simulation and Orin

This workspace shares one `src/` between the simulation PC and the Orin companion computer.
Use ROS 2 Humble on Ubuntu 22.04, or ROS 2 Jazzy on Ubuntu 24.04. Build separately
on each machine; do not copy `build/`, `install/`, or `log/` across architectures
or ROS distributions.
The Jazzy/Orin path still needs a build and hardware validation on the board.

- Shared: PX4 control, PX4 1.14 messages, Orbbec depth camera, SIYI camera,
  and Micro XRCE-DDS Agent.
- Simulation PC: `worlds/` and `models/`, plus separately installed PX4/Gazebo.
- Orin: physical cameras and the connection to the flight controller.

See [Orin setup and source synchronization](docs/orin.md). From the workspace
root, `bash tools/build_workspace.sh --help` explains the build options and
`bash tools/export_orin.sh` exports portable sources without simulation assets.
See [local validation results](docs/validation.md) for the checks already run.

Third-party revisions are recorded in `drivers.repos`. On a fresh Git clone,
populate missing dependencies with `vcs import --skip-existing . < drivers.repos`.
The legacy `.gitmodules` file does not register active Git submodules here.
