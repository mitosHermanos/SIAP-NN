from selenium.webdriver.support import expected_conditions as EC


from selenium.webdriver.common.by import By
import pandas as pd
from selenium.webdriver.support.wait import WebDriverWait

from tools import create_driver, set_value_to_field, find_single_element, random_sleep, to_soup

BASE_URL = "https://www.booking.com/"
ID = "id"
CLASS_NAME = "class_name"
XPATH = "xpath"
search_id = "ss"
DRIVER = create_driver(BASE_URL)
hotel_name = "park plaza london"
hotel_name_2 = "park plaza Wrocław"
hotel_name_3 = "park plaza riverbank"
hotel_name_4 = "park plaza park royal"
hotel_name_5 = "park plaza victoria"
hotel_name_6 = "park plaza victoria london"
hotel_name_7 = "park plaza country hall london"
hotel_name_8 = "park plaza vondelpark"
hotel_name_9 = "park plaza leeds"

hotel_name_10 = "Radisson Blu Edwardian Grafton Hotel, London" #3 2168 done
hotel_name_11 = "Radisson Blu Plaza Hotel Sydney" #2 2436 done
hotel_name_12a = "Radisson Hotel & Suites Sydney" #2
hotel_name_12 = "Radisson Blu Hotel, Edinburgh City Centre" #2 4018 done
hotel_name_13 = "Radisson Blu Hotel, Dubai Waterfront" #3 2800 done
hotel_name_14 = " Radisson Blu Hotel, Glasgow" #2 3194 done
hotel_name_15 = "Radisson Blu Edwardian Grafton Hotel, London" #3 2168
hotel_name_16 = "Radisson Blu Manchester Airport" #2 8833 done
hotel_name_17 = "Radisson Blu Hotel, Liverpool" #2 5016 done


g_country = []
i_room = []
n_s = []
t_t = []
p_review = []
n_review = []
grade_list = []
title_list = []
date_of_stay = []
nights_s = []


def load_page():
    set_value_to_field(DRIVER, search_id, ID, hotel_name_15)
    random_sleep()
    find_single_element(DRIVER, "//div[@class='xp__button']//button[@type='submit']", XPATH).click()
    random_sleep()
    # href = find_single_element(DRIVER, "//div[@id='hotel_180016']//a[@class=' sr_item_photo_link "
    #                                    "sr_hotel_preview_track  ']", XPATH).getAttribute("href")
    # href = DRIVER.find_element(by=By.XPATH, value="//div[@id='hotel_180016']//a[@class=' sr_item_photo_link "
    #                                               "sr_hotel_preview_track  ']").get_attribute("href")

    href = DRIVER.find_element(by=By.XPATH, value='// *[ @ id = "hotellist_inner"] /'
                                                  ' div[1] / div[2] / div[1] / div[1] / div[1] / h3 / a').get_attribute("href")
    # soap = to_soup(DRIVER.page_source)
    # review_list = soap.find("ul", class_="review_list")
    # print(review_list)
    DRIVER.get(href)
    soap = to_soup(DRIVER.page_source)
    find_single_element(DRIVER, "show_reviews_tab", ID).click()
    find_single_element(DRIVER, "//div[@id='review_lang_filter']//button[@class='bui-button bui-button--secondary']",
                        XPATH).click()
    random_sleep()
    find_single_element(DRIVER, "//div[@id='review_lang_filter']//div[@class='bui-dropdown__content']"
                                "//div[@class='bui-dropdown-menu']//ul//li[3]//button",
                        XPATH).click()


def get_reviews():
    soap = to_soup(DRIVER.page_source)
    num_of_pages = soap.find("div", class_="bui-pagination") \
        .find("div", class_="bui-pagination__list").findAll("div", class_="bui-pagination__item")
    exact_num = num_of_pages[7].findAll("span")[0]
    exact_int = int(exact_num.text.strip())
    print(exact_int)
    i = 1

    while i < exact_int:
        soap = to_soup(DRIVER.page_source)
        random_sleep(2, 3)
        review_list = soap.find("div", id="review_list_page_container").find("ul", class_="review_list")

        list_item = review_list.findAll("li", class_="review_list_new_item_block")
        for li in list_item:
            left_info = li.find("div", class_="bui-grid__column-3 c-review-block__left")
            right_review = li.find("div", class_="bui-grid__column-9 c-review-block__right")

            guest = left_info.find("div", class_="c-review-block__row c-review-block__guest")

            # COUNTRY OF GUEST
            try:
                guest_country = guest.find("div", class_="bui-avatar-block__text") \
                    .find("span", class_="bui-avatar-block__subtitle").text.strip()
            except:
                guest_country = ''
            print('1', guest_country)
            g_country.append(guest_country)
            # ROOM INFO
            try:
                room_info = left_info.find("div", class_="bui-list__body").text.strip()
            except:
                room_info = ''
            if room_info.__contains__("\xa0\n\n"):
                room_info = ''
            print('2', room_info)
            i_room.append(room_info)
            stay_info = left_info.find("ul",
                                       class_="bui-list bui-list--text bui-list--icon bui_font_caption c-review-block__row c-review-block__stay-date")
            # NIGHTS STAYED AND DATE OF VISIT
            try:
                nights_stayed_date = stay_info.find("div", class_="bui-list__body").text.strip()
            except:
                nights_stayed_date = ''
            parsed = nights_stayed_date.replace("\xa0\n\n", "").strip()
            split_list = parsed.split("·")
            nights_stayed = split_list[0]
            date = split_list[1]
            print('3', nights_stayed_date)
            nights_s.append(nights_stayed)
            date_of_stay.append(date)
            # n_s.append(nights_stayed_date)
            travel_type_info = left_info.find("ul",
                                              class_="bui-list bui-list--text bui-list--icon bui_font_caption review-panel-wide__traveller_type c-review-block__row")
            # TRAVEL TYPE
            try:
                travel_type = travel_type_info.find("div", class_="bui-list__body").text.strip()
            except:
                travel_type = ''
            print('4', travel_type)
            t_t.append(travel_type)
            # REVIEWS
            review_divs = right_review.findAll("div", class_="c-review-block__row")
            try:
                poz_review = review_divs[1].find("div", class_="c-review__row").text.strip()
                if poz_review.__contains__("Disliked"):
                    poz_review = ''
                split_list = poz_review.split("·")
                poz_review = split_list[1]
            except:
                poz_review = ''
            try:
                neg_review = review_divs[1].find("div", class_="c-review__row lalala").text.strip()
                if neg_review.__contains__("Liked"):
                    neg_review = ''
                split_list = neg_review.split("·")
                neg_review = split_list[1]
            except:
                neg_review = ''
            print('5', poz_review)
            p_review.append(poz_review)
            print('6', neg_review)
            n_review.append(neg_review)

            grade_and_title = review_divs[0].find("div", class_="bui-grid")
            try:
                title = grade_and_title.find("h3", class_="c-review-block__title c-review__title--ltr").text.strip()
            except:
                title = ''
            try:
                grade = grade_and_title.find("div", class_="bui-review-score__badge").text.strip()
            except:
                grade = ''
            print('7', grade, title)
            grade_list.append(grade)
            title_list.append(title)
            print(i, "<-********************************************************************************************")
        # find_single_element(DRIVER, "pagenext", CLASS_NAME).click()
        # random_sleep()
        # find_single_element(DRIVER, "//div[@class='c-pagination']"
        #                             "//div[@class='bui-pagination']"
        #                             "//div[@class='bui-pagination__nav']"
        #                             "//div[@class='bui-pagination__list page_link']"
        #                             "//div[@class='bui-pagination__item bui-pagination__next-arrow']"
        #                             "//a[@class='pagenext']", XPATH).click()

        try:
            button = WebDriverWait(DRIVER, 10).until(EC.element_to_be_clickable((By.XPATH, "//div[@class='c-pagination']"
                                    "//div[@class='bui-pagination']"
                                    "//div[@class='bui-pagination__nav']"
                                    "//div[@class='bui-pagination__list page_link']"
                                    "//div[@class='bui-pagination__item bui-pagination__next-arrow']"
                                    "//a[@class='pagenext']")))
            button.click()
        except:
            export_to_csv()


        # soap.get(next_page.get_attribute("href"))
        # next_page.click()
        i = i + 1


def export_to_csv():
    name_dict = {
        'Guest_country': g_country,
        'Room_info': i_room,
        'Nights_stayed': nights_s,
        'Date of stay': date_of_stay,
        'Travel_type': t_t,
        'Positive_reviews': p_review,
        'Negative_reviews': n_review,
        'Grade': grade_list,
        'Title': title_list
    }
    df = pd.DataFrame(name_dict)
    df.to_csv('booking.csv')


if __name__ == '__main__':
    load_page()
    random_sleep()
    get_reviews()
    export_to_csv()
