
from   selenium import webdriver
from   selenium.common.exceptions import TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
import time
class BaseWordPressData:

    def __init__(self, type,browser):
        self.type = type
        self.browser = browser


    def logout(self):
        self.browser.get("http://localhost/wordpress/wp-login.php?action=logout")


    def closeBrowser(self):
        self.browser.close()

