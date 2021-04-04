import sys
import numpy as np
import random

mt1 = [line.strip('\n').lower() for line in open(sys.argv[1])]
mt2 = [line.strip('\n').lower() for line in open(sys.argv[2])]
ref = [line.strip('\n').lower() for line in open(sys.argv[3])]

for i in range(len(mt1)):
  if '<unk>' not in mt2[i] and '<unk>' not in mt1[i]:
    print(mt1[i]+'\t'+mt2[i]+'\t'+ref[i])
    
