import numpy as np
import re


word = 1
s = [1,2,3,4,1,2,3,1,1,1]
w = [m.start() for m in re.finditer(word, s)]
print(w)

