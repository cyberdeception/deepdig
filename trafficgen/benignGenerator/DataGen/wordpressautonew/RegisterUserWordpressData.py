import unittest
from   selenium import webdriver
from   selenium.common.exceptions import TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
import time


from BaseWordpressData import BaseWordPressData
class RegisterUserWPressData(BaseWordPressData):

    def __init__(self,type, dataDict,browser):
        BaseWordPressData.__init__(self, type,browser)
        self.dataDict = dataDict

    def register(self):
        username = self.dataDict['username']
        email = self.dataDict['email']
        firstname = self.dataDict['firstname']
        lastname = self.dataDict['lastname']
        #http://localhost/wordpress/wp-admin/user-new.php
        url = self.dataDict['url']
        app = self.dataDict['app']
        register_page = "https://" + url + "/" + app + "/wp-admin/user-new.php"

        self.browser.get( register_page )

        # Fetch username, password input boxes and submit button
        # This time I'm now testing if the elements were found.
        # See the previous exmaples to see how to do that.
        usernameTB = self.browser.find_element_by_id( "user_login" )
        emailTB = self.browser.find_element_by_id( "email" )
        firstnameTB = self.browser.find_element_by_id( "first_name" )
        lastnameTB = self.browser.find_element_by_id( "last_name" )
        submit   = self.browser.find_element_by_id( "createusersub"   )

        # Input text in username and password inputboxes
        usernameTB.send_keys( username )
        emailTB.send_keys(email)
        firstnameTB.send_keys(firstname)
        lastnameTB.send_keys(lastname)
        # Click on the submit button
        submit.click()

        # Create wait obj with a 5 sec timeout, and default 0.5 poll frequency
        wait = WebDriverWait( self.browser, 5 )

        time.sleep(5)





