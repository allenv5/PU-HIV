#!/usr/bin/env python
#-*- coding:utf-8 -*-
# datetime:2020/12/8 15:51


import argparse
import BiasedSVM as bs

parser = argparse.ArgumentParser(description="The Biased-SVM method")
parser.add_argument("--code", "-c", help="feature_code", required=True)
parser.add_argument("--train", "-train", help="train data", default="train")
parser.add_argument("--test", "-test", help="test data", default="test")
args = parser.parse_args()


if __name__ == '__main__':
    try:
        bs.run(args.code, args.train, args.test)
    except Exception as e:
        print(e)