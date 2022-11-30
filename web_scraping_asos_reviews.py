import pandas as pd
import requests
from bs4 import BeautifulSoup
from lxml import etree

"""
@author: Sriya Madapusi Vasudevan
"""
def getRequestObject(url):
    """
    Method to get the response from the url
    :param url: url
    :return: the response from url
    """
    try:
        resp = requests.get(url)
        print(f"Response is OK, code: {resp.status_code}, page: {url}")
        assert resp.status_code == 200
        return resp
    except AssertionError:
        print(f"Response is not OK, code: {resp.status_code}, page: {url}")


def getNumberReviews(soup):
    """
    Get the current number of reviews available on the website
    :param soup: Beautiful Soup object
    :return: none
    """
    span_list = soup.findAll(
        lambda tag: tag.name == 'div' and tag.has_attr('id') and tag['id'] == "business-unit-title")
    span_text = span_list[0].text.replace("\xa0", "")
    text_list = span_text.split(" ")
    num_reviews = text_list[1].replace(",", "")
    print(f"Total number of reviews: {num_reviews}")


def extractReviews(soup, company_name, date_published, rating_value, review_body):
    """
    Extract all needed review objects such as date published, rating value and
    review
    :param soup: BeautifulSoup object
    :param company_name: company name
    :param date_published: date review is published
    :param rating_value: rating value out of 5
    :param review_body: text from the review
    :return: company name, date published, rating value and review body
    """
    dom = etree.HTML(str(soup))
    review_item_list = dom.xpath('//*[contains(@class, "reviewCard")]/article/section')
    print(f"length of reviews per page: {len(review_item_list)}")

    for element in review_item_list:
        try:
            rating = element.findall('div')[0].get('data-service-review-rating')
            reviewDate = element.findall('div')[0].findall('div')[1].find('time').get('datetime')
            reviewText = element.findall('div')[1].findtext('p')
            if reviewText is None:
                reviewText = element.findall('div')[1].find('h2').findtext('a')
        except AttributeError:
            continue

        company_name.append('ASOS')
        date_published.append(reviewDate)
        rating_value.append(rating)
        review_body.append(reviewText)

    return company_name, date_published, rating_value, review_body


initial_url = "https://www.trustpilot.com/review/www.asos.com"
response = getRequestObject(initial_url)
soup = BeautifulSoup(response.content, "html.parser")
getNumberReviews(soup)

companyName = []
datePublished = []
ratingValue = []
reviewBody = []

# for the first page
extractReviews(soup, companyName, datePublished, ratingValue, reviewBody)

for i in range(2, 35):
    url = "https://www.trustpilot.com/review/www.asos.com?page=" + str(i)
    response = getRequestObject(url)
    soup = BeautifulSoup(response.content, "html.parser")
    extractReviews(soup, companyName, datePublished, ratingValue, reviewBody)

reviews_df = pd.DataFrame((list(zip(companyName, datePublished, ratingValue, reviewBody))),
                          columns=['companyName', 'datePublished', 'ratingValue', 'reviewBody'])

print(f"Total collected reviews: {reviews_df.shape[0]}")

reviews_df.to_csv("ASOS_reviews_sriya.csv")
