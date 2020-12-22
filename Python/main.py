#!/usr/bin/env python
#-*- coding:utf-8 -*-
# datetime:2020/12/8 15:51


import argparse
import BiasedSVM as bs

parser = argparse.ArgumentParser(description="The Biased-SVM method")
parser.add_argument("--feature", "-f", help="feature_code", required=True)
parser.add_argument("--c1", "-c1", help="", required=True)
parser.add_argument("--beta", "-beta", help="", required=True)
parser.add_argument("--input", "-i", help="input file", required=True)
parser.add_argument("--cross_validation", "-cv", help="K value of cross validation", default="-1")
args = parser.parse_args()


if __name__ == '__main__':
    try:
        bs.run(args.feature, args.c1, args.beta, args.input, args.cross_validation)
    except Exception as e:
        print(e)