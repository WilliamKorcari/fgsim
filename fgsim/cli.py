"""Console script for fgsim."""
import argparse
import sys


"""Console script for fgsim."""
parser = argparse.ArgumentParser()

subparsers = parser.add_subparsers(help="Available Commands", dest="command")

train_parser = subparsers.add_parser("train")

geo_parser = subparsers.add_parser("geo")

args = parser.parse_args()
