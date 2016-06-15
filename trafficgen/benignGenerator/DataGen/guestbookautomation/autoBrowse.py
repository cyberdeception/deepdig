from   selenium import webdriver

from FillGuestBook import FillGuestBook
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
    return userData[:-1].split(",")

browser = webdriver.Firefox()

txtfilename = "./bbc.txt"
fakedata = "./userdata.txt"


for i in range(1,2):
    userData = getUserData(fakedata)
    print userData
    mytitle = getTitle(txtfilename,40)
    myText = getText(txtfilename,200)
    email = userData[1]+userData[2]+"@gmail.com"
    guestbookData = {"realname": mytitle,"email": email ,"url":"10.176.147.83","comment":myText,\
	           "username":userData[1],"city":userData[4],"state":userData[5] }
    reg = FillGuestBook("addguest",guestbookData,browser)
    reg.addGuest()
    browser.get("https://10.176.147.83/guestbook.html")



reg.closeBrowser()




    


 


