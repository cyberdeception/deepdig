from   selenium import webdriver
from LoginWordPressData import LoginWordPressData
from RegisterUserWordpressData import RegisterUserWPressData
from CreatePostWordpressData import CreatePostWordPressData
from CreateProduct import RegisterProductWPressData
import random 
import os
import time

def getTitle(filename,titleLen):
    fo = open(filename, "rw+")
    fo.seek(0, os.SEEK_END)
    size = fo.tell()
    fo.seek(0,os.SEEK_SET)
    offset = random.randint(1, size-titleLen)
    mytitleLen = random.randint(1, titleLen)
    fo.seek(offset,os.SEEK_SET)
    text = fo.readline()
    text = text.replace("'", "")
    text = text.replace("\"", "")
    text = unicode(text, errors='ignore')
    return text[:-1]

    
    



def getText(filename,textLen):
    fo = open(filename, "rw+")
    fo.seek(0, os.SEEK_END)
    size = fo.tell()
    fo.seek(0,os.SEEK_SET)
    offset = random.randint(1, size-textLen)
    mytitleLen = random.randint(1, textLen)
    fo.seek(offset,os.SEEK_SET)
    text = fo.readline()
    text = text.replace("'", "")
    text = text.replace("\"", "")
    text = unicode(text, errors='ignore')
    return text[:-1]


def getUserData(filename):
    fo = open(filename, "rw+")
    count =0
    userData = ""
    num_lines = sum(1 for linect in fo)
    offset = random.randint(1, num_lines-2)
    print offset
    fo.seek(0,os.SEEK_SET)
    for line in fo:
        count = count+1
        if count >= offset:
            print line
            userData = line
            break;
    fo.close
    return userData[:-1].split("\t")

browser = webdriver.Firefox()

loginData = {"username":"admin","password":"pass123","url":"54.218.47.176","app":"wordpress"}
test_login = LoginWordPressData("login",loginData,browser)

test_login.login()
txtfilename = "./bbc.txt"
fakedata = "./fakeproducts.csv"

for i in range(1,2):
    productData = getUserData(fakedata)
    print productData
    productData ={"url":"54.218.47.176","app":"wordpress","prodname":productData[0],"prodprice":productData[1]}
    regProd = RegisterProductWPressData("regprod",productData,browser)
    regProd.registerproduct()
    time.sleep(4)










