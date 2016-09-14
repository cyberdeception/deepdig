from   selenium import webdriver
from   selenium.common.exceptions import TimeoutException
from   selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.support.ui import WebDriverWait

from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
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

loginData = {"username":"admin","password":"pass123","url":"104.154.117.255","app":"wordpress"}
test_login = LoginWordPressData("login",loginData,browser)

test_login.login()
txtfilename = "./bbc.txt"
fakedata = "./fakeproducts.csv"


for k in range(1,2):
	for i in range(1,3):
	    productData = getUserData(fakedata)
	    url = "104.154.117.255"
	    app = "wordpress"
	    product_page = "https://" + url + "/" + app + "/product/" + productData[0].lower()
            #product_page = "https://" + url + "/" + app + "/product/" +"icecream-vanilla"
	    browser.get( product_page )
	    time.sleep(2)
            try:
	         submit = browser.find_element_by_class_name('single_add_to_cart_button')
	         submit.click()
	         time.sleep(2)
	         browser.get("https://104.154.117.255/wordpress/cart/")
	         browser.get("https://104.154.117.255/wordpress/checkout/")
	         submit2 = WebDriverWait(browser, 4).until(EC.presence_of_element_located((By.ID, "place_order")))
                 print "Page is ready!"
		 time.sleep(5)
		 browser.execute_script("document.getElementsByName('woocommerce_checkout_place_order')[0].click()")
                 time.sleep(7)
            except NoSuchElementException:
                 print "continue"
                 time.sleep(4)
		    





