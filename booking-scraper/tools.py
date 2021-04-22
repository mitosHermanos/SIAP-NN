from webdriver_manager.chrome import ChromeDriverManager
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import random
from time import sleep


def to_soup(data, parser="lxml"):
    return BeautifulSoup(data, parser)


def create_driver(base_url, headless=False):
    options = Options()
    options.add_argument("--disable-notifications")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-software-rasterizer")

    # disable logging
    options.add_argument("--disable-logging")
    options.add_experimental_option("excludeSwitches", ["enable-logging"])

    if headless:
        options.add_argument("--headless")

    driver_manager = ChromeDriverManager(log_level=30,
                                         cache_valid_range=10)
    driver = webdriver.Chrome(options=options, executable_path=driver_manager.install())

    driver.maximize_window()

    # log("Opening browser: " + base_url)
    driver.get(base_url)
    # random_sleep()
    return driver


def set_value_to_field(driver, selector, selector_type, value):
    element = find_single_element(driver, selector, selector_type)
    element.send_keys(value)


def find_single_element(driver, selector, selector_type):
    if selector_type == "css_selector":
        element = driver.find_element_by_css_selector(selector)
    elif selector_type == "class_name":
        element = driver.find_element_by_class_name(selector)
    elif selector_type == "xpath":
        element = driver.find_element_by_xpath(selector)
    elif selector_type == "id":
        element = driver.find_element_by_id(selector)
    elif selector_type == "name":
        element = driver.find_element_by_name(selector)
    else:
        element = driver.find_element_by_css_selector(selector)
    return element


def random_sleep(min_sleep_sec=1,
                 max_sleep_sec=2, log_sleeping=True):
    min_sleep_sec = min_sleep_sec * 1 * 1.5
    max_sleep_sec = max_sleep_sec * 1 * 1.5
    sleep_sec = random.uniform(min_sleep_sec, max_sleep_sec)
    sleep(sleep_sec)
