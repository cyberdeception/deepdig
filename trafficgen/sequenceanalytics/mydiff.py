

from difflib import SequenceMatcher
text1 = open("outfile1150").read()
text2 = open("outfile1160").read()
m = SequenceMatcher(None, text1, text2)
print m.ratio()
