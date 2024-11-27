import re
from pymongo import MongoClient
import requests
from bs4 import BeautifulSoup

# MongoDB connection
client = MongoClient('mongodb://localhost:27017/')
db = client['news_database']  # Database name
collection = db['articles']  # Collection name

class Article:
    def __init__(self, title, content, href_content, date, article_body):
        self.title = title
        self.content = content
        self.href_content = href_content
        self.date = date
        self.article_body = article_body
        self.bodyLength = len(article_body.split())
        self.positive = 0
        self.negative = 0
        self.neutral = 0

# Define a function to remove special characters from a string
def remove_special_characters(text):
    pattern = r'[^a-zA-Z0-9\s]'
    cleaned_text = re.sub(pattern, '', text)
    return cleaned_text

articles = list()

# URLs of the webpages to scrape
urls = ["https://www.moneycontrol.com/news/tags/companies.html", "https://www.moneycontrol.com/news/business.html"]

# Send a GET request to each URL
for url in urls:
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the HTML content of the page
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find all the list items with class "clearfix"
        news_items = soup.find_all('li', class_='clearfix')

        # Loop through the news items and extract the required information
        for item in news_items:
            if item.find(class_='isPremiumCrown'):
                continue
            title = item.find('h2').text
            content = item.find('p').text
            href_content = item.find('a')['href']
            date = item.find('span').text

            # For site content
            response = requests.get(href_content)
            html_doc = response.text
            temp = BeautifulSoup(html_doc, 'html.parser')

            target_script = temp.find_all('script', type='application/ld+json')
            target = (target_script[2])
            # Extract the content from the script tag
            script_content = target.string.strip()
            articleBody = ((script_content[(script_content.find('articleBody') + 13):(script_content.find('"author"')):]))

            # Clean the article body
            new_body = remove_special_characters(articleBody)

            # Print the information for each news item
            print("Content:", content)
            print("Title:", title)
            print("Anchor Tag (href):", href_content)
            print("Date:", date)
            print("Site Content:", new_body)
            print(len(new_body.split()))
            print("-" * 150)

            # Store data in list
            article = Article(title, content, href_content, date, new_body)
            articles.append(article)

            # Prepare data for MongoDB
            article_data = {
                'title': title,
                'content': content,
                'href_content': href_content,
                'date': date,
                'article_body': new_body,
                'bodyLength': len(new_body.split()),
                'positive': 0,
                'negative': 0,
                'neutral': 0
            }

            # Insert data into MongoDB
            collection.insert_one(article_data)
    else:
        print("Failed to retrieve the webpage. Status code:", response.status_code)

print("Data scraping and insertion completed.")
