"""Console script for fgsim."""
import argparse

parser = argparse.ArgumentParser()

subparsers = parser.add_subparsers(help="Available Commands", dest="command")

train_parser = subparsers.add_parser("train")
parser.add_argument("-t", "--tag", default="default", required=False)
geo_parser = subparsers.add_parser("geo")

args = parser.parse_args()
