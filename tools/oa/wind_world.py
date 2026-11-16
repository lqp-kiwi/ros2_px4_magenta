#!/usr/bin/env python3
"""Windy copy of a Gazebo world for the headless SITL stack (tools/oa/sitl.sh up WORLD --wind E,N).

Gazebo Sim 8 loads <wind><linear_velocity> of the world into its wind entity, and the
MulticopterMotorModel system of every rotor adds rotor drag against the air moving at that
velocity. PX4's own windy.sdf world works the same way; the drone model is not changed.
The x500 feels the wind only through this rotor drag, about 0.25 N per m/s of air speed in a
hover, so a simulated wind pushes less than a real one of the same speed.

Gusts (--gust G > 0): the WindEffects system varies the wind speed by +-G (a fraction of it) with
period GUST_PERIOD. It goes into a copy of PX4's server.config, not into the world: any <plugin>
in the world makes Gazebo ignore server.config (see the comment in worlds/*.sdf).
The world copy keeps its world name, so PX4 and the flight scoring see the same world.

Usage:
    wind_world.py WORLD.sdf OUT.sdf --wind E,N [--gust G] [--server-config IN OUT]
    E, N: velocity of the air in m/s towards east (world x) and towards north (world y).
"""
import argparse
import sys
import xml.etree.ElementTree as ET

GUST_PERIOD = 8.0   # period of the gust sine (s)
RISE_TIME = 5.0     # WindEffects time constant: the wind builds up from 0 when Gazebo starts (s)


def parse_wind(text):
    try:
        east, north = (float(v) for v in text.split(','))
    except ValueError:
        raise argparse.ArgumentTypeError(f'wants E,N in m/s, e.g. 0,5 (got {text!r})')
    return east, north


def write_windy_world(src, dst, east, north):
    """Write a copy of the world with <wind> set (an existing one is replaced)."""
    tree = ET.parse(src)
    world = tree.getroot().find('world')
    if world is None:
        sys.exit(f'wind_world.py: no <world> in {src}')
    for old in world.findall('wind'):
        world.remove(old)
    wind = ET.SubElement(world, 'wind')
    ET.SubElement(wind, 'linear_velocity').text = f'{east:g} {north:g} 0'
    tree.write(dst, encoding='utf-8', xml_declaration=True)


def write_server_config(src, dst, gust):
    """Write a copy of PX4's server.config, plus the WindEffects system when there are gusts."""
    tree = ET.parse(src)
    plugins = tree.getroot().find('plugins')
    if plugins is None:
        sys.exit(f'wind_world.py: no <plugins> in {src}')
    if gust > 0:
        plugin = ET.SubElement(plugins, 'plugin', {
            'entity_name': '*', 'entity_type': 'world',
            'filename': 'gz-sim-wind-effects-system', 'name': 'gz::sim::systems::WindEffects'})
        horizontal = ET.SubElement(plugin, 'horizontal')
        magnitude = ET.SubElement(horizontal, 'magnitude')
        ET.SubElement(magnitude, 'time_for_rise').text = f'{RISE_TIME:g}'
        sine = ET.SubElement(magnitude, 'sin')
        ET.SubElement(sine, 'amplitude_percent').text = f'{gust:g}'
        ET.SubElement(sine, 'period').text = f'{GUST_PERIOD:g}'
        direction = ET.SubElement(horizontal, 'direction')     # constant direction
        ET.SubElement(direction, 'time_for_rise').text = f'{RISE_TIME:g}'
    tree.write(dst, encoding='utf-8')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('world', help='world .sdf to copy')
    ap.add_argument('out', help='windy copy to write')
    ap.add_argument('--wind', type=parse_wind, required=True, metavar='E,N',
                    help='air velocity towards east and north (m/s)')
    ap.add_argument('--gust', type=float, default=0.0,
                    help='wind speed variation as a fraction, 0 = constant (default 0)')
    ap.add_argument('--server-config', nargs=2, metavar=('IN', 'OUT'),
                    help="PX4's server.config and the copy to write")
    args = ap.parse_args()
    if not 0.0 <= args.gust < 1.0:
        ap.error('--gust must be in [0, 1)')
    if args.gust > 0 and not args.server_config:
        ap.error('--gust needs --server-config')
    write_windy_world(args.world, args.out, *args.wind)
    if args.server_config:
        write_server_config(*args.server_config, args.gust)


if __name__ == '__main__':
    main()
