import unittest
from   selenium import webdriver
from   selenium.common.exceptions import TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
import time
from BaseWordpressData import BaseWordPressData
class CreatePostWordPressData(BaseWordPressData):

    def __init__(self,type, dataDict,browser):
        BaseWordPressData.__init__(self, type,browser)
        self.dataDict = dataDict
#http://localhost/wordpress/wp-admin/post-new.php
    def sendPost(self):
        title = self.dataDict['title']
        body = self.dataDict['body']

        #http://localhost/wordpress/wp-admin/user-new.php
        url = self.dataDict['url']
        app = self.dataDict['app']
        register_page = "https://" + url + "/" + app + "/wp-admin/post-new.php"

        self.browser.get( register_page )

        # Fetch username, password input boxes and submit button
        # This time I'm now testing if the elements were found.
        # See the previous exmaples to see how to do that.
        titleTB = self.browser.find_element_by_id( "title" )
        submit   = self.browser.find_element_by_id( "publish"   )

        # Input text in username and password inputboxes
        titleTB.send_keys( title )
        #nput to tinymice works great
        self.browser.execute_script("tinyMCE.activeEditor.setContent('%s')" %body)
        # Click on the submit button
        submit.click()

        # Create wait obj with a 5 sec timeout, and default 0.5 poll frequency
        wait = WebDriverWait( self.browser, 5 )

        time.sleep(5)





