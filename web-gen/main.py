import argparse
parser = argparse.ArgumentParser()
parser.add_argument('image', type=str)
args = parser.parse_args()

print('Product:', args.image)