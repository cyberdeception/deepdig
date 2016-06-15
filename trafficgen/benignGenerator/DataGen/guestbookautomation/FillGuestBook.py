import unittest
from   selenium import webdriver
from   selenium.common.exceptions import TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
import time


from BaseWordpressData import BaseWordPressData
class FillGuestBook(BaseWordPressData):

    def __init__(self,type, dataDict,browser):
        BaseWordPressData.__init__(self, type,browser)
        self.dataDict = dataDict

    def addGuest(self):
        email = self.dataDict['email']
        realname = self.dataDict['realname']
        #http://54.191.135.35/addguest.html
        url = self.dataDict['url']
        comment = self.dataDict['comment']
        city = self.dataDict['city']
        state = self.dataDict['state']
        add_page = "https://" + url + "/addguest.html"

        self.browser.get( add_page )

        # Fetch username, password input boxes and submit button
        # This time I'm now testing if the elements were found.
        # See the previous exmaples to see how to do that.
        usernameTB = self.browser.find_element_by_name( "username" )
        realnameTB = self.browser.find_element_by_name( "realname" )
        urlTB = self.browser.find_element_by_name( "url" )
        commentTB = self.browser.find_element_by_name( "comments" )
        cityTB = self.browser.find_element_by_name( "city" )
        stateTB = self.browser.find_element_by_name( "state" )
        submit   = self.browser.find_element_by_xpath("//input[@type='submit']")

        # Input text in username and password inputboxes
        usernameTB.send_keys( email )
        realnameTB.send_keys(realname)
        urlTB.send_keys(url)
        commentTB.send_keys(comment)
        cityTB.send_keys(city)
        stateTB.send_keys(state)
        # Click on the submit button
        submit.click()

        # Create wait obj with a 5 sec timeout, and default 0.5 poll frequency
        wait = WebDriverWait( self.browser, 5 )

        time.sleep(5)





