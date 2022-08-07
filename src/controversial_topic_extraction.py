import requests
from bs4 import BeautifulSoup

subject = 'Wikipedia:List of controversial issues'
url = "https://en.wikipedia.org/w/api.php?action=parse&format=json&page=Wikipedia:List_of_controversial_issues&section=1&prop=text"

response = requests.get(url)
data = response.json()

type(data["parse"]["text"]["*"])
soup = BeautifulSoup(data["parse"]["text"]["*"], 'html.parser')

topics = []

for link in soup.find_all('a'):
    topics.append(link.get('title'))

with open("./data/controversial_topics.txt", "w") as f:
    for i in topics[18:]:
        f.write(str(i))
        f.write("\n")