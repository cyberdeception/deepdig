import unittest
from   selenium import webdriver
from   selenium.common.exceptions import TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
import time


from BaseWordpressData import BaseWordPressData
class LoginWordPressData(BaseWordPressData):

    def __init__(self,type, dataDict,browser):
        BaseWordPressData.__init__(self, type,browser)
        self.dataDict = dataDict


    def login(self):
        username = self.dataDict['username']
        password = self.dataDict['password']
        url = self.dataDict['url']
        app = self.dataDict['app']
        login_page = "https://" + url + "/" + app + "/wp-login.php"


        self.browser.get( login_page )

        # Fetch username, password input boxes and submit button
        # This time I'm now testing if the elements were found.
        # See the previous exmaples to see how to do that.
        usernameTB = self.browser.find_element_by_id( "user_login" )
        passwordTB = self.browser.find_element_by_id( "user_pass" )
        submit   = self.browser.find_element_by_id( "wp-submit"   )

        # Input text in username and password inputboxes
        usernameTB.send_keys( username)
        passwordTB.send_keys( password )

        # Click on the submit button
        submit.click()

        # Create wait obj with a 5 sec timeout, and default 0.5 poll frequency
        wait = WebDriverWait( self.browser, 5 )



        time.sleep(5)





