# navigation_simulator_2d

2D robot navigation simulator with a simple LiDAR model

## How to install

1. Download

```
git clone git@github.com:kohonda/navigation_simulator_2d.git
```

2. Create venv

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install

```bash
cd navigation_simulator_2d
pip3 install -e .
```

## How to run example

```bash
cd test
python3 test_simulator.py
```

<div align="center">
<img src="https://user-images.githubusercontent.com/50091520/202944565-7eaa24d4-c7b6-4ebb-88a5-9032e64f2729.gif">
</div>

- Green/Red circle: Robot (When red, the robot is in collision with obstacles)
- Black: known static object, given as prepared map
- Blue: unknown static/dynamic objects
- Red: LiDAR scan

## How to make new map

```bash
cd script
python3 map_creator.py [config-yaml]
```

Example config is [here](script/example.yaml)