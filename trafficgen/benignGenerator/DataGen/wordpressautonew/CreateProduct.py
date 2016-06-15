import unittest
from   selenium import webdriver
from   selenium.common.exceptions import TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
import time


from BaseWordpressData import BaseWordPressData
class RegisterProductWPressData(BaseWordPressData):

    def __init__(self,type, dataDict,browser):
        BaseWordPressData.__init__(self, type,browser)
        self.dataDict = dataDict

    def registerproduct(self):
        prodname = self.dataDict['prodname']
        prodprice = self.dataDict['prodprice']
        
        url = self.dataDict['url']
        app = self.dataDict['app']
        register_page = "https://" + url + "/" + app + "/wp-admin/post-new.php?post_type=product"

        self.browser.get( register_page )

        # Fetch username, password input boxes and submit button
        # This time I'm now testing if the elements were found.
        # See the previous exmaples to see how to do that.
        productTB = self.browser.find_element_by_id( "title" )
        priceTB = self.browser.find_element_by_id( "_regular_price" )
        
        submit   = self.browser.find_element_by_id( "publish"   )

        # Input text in username and password inputboxes
        productTB.send_keys( prodname )
        priceTB.send_keys(prodprice)
        
        # Click on the submit button
        submit.click()

        # Create wait obj with a 5 sec timeout, and default 0.5 poll frequency
        wait = WebDriverWait( self.browser, 5 )

        time.sleep(5)





