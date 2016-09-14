import unittest
from   selenium import webdriver
from   selenium.common.exceptions import TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import time
from BaseWordpressData import BaseWordPressData
class CreateSocialPost(BaseWordPressData):

    def __init__(self,type, dataDict,browser):
        BaseWordPressData.__init__(self, type,browser)
        self.dataDict = dataDict

    def sendPost(self):
        title = self.dataDict['post']
       
        #http://104.154.117.255/wordpress/members/admin/
        url = self.dataDict['url']
        app = self.dataDict['app']
        register_page = "https://" + url + "/" + app + "/members/admin/"

        self.browser.get( register_page )

        # Fetch username, password input boxes and submit button
        # This time I'm now testing if the elements were found.
        # See the previous exmaples to see how to do that.
        titleTB = self.browser.find_element_by_id( "whats-new" )
        #submit   = self.browser.find_element_by_id( "aw-whats-new-submit"   )
        submit  = WebDriverWait(self.browser, 4).until(EC.presence_of_element_located((By.ID, "aw-whats-new-submit")))

        # Input text in username and password inputboxes
        titleTB.send_keys( title )
        self.browser.execute_script("document.getElementById('aw-whats-new-submit').click()")
        # Create wait obj with a 5 sec timeout, and default 0.5 poll frequency
        wait = WebDriverWait( self.browser, 5 )

        time.sleep(5)





