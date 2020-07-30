import requests
from bs4 import BeautifulSoup
import re,csv

url="http://car.bitauto.com/xuanchegongju/?mid=8"
headers={'user-agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.131 Safari/537.36'}
html=requests.get(url,headers=headers,timeout=10)
content = html.text
soup = BeautifulSoup(content, 'html.parser', from_encoding='utf-8')
page_num = soup.find_all(attrs={"data-pages":True})[0]["data-pages"]

list = []
for i in range(1,int(page_num)+1):
    url_1 = url+"&page="+str(i)
    soup2 = BeautifulSoup(requests.get(url_1,headers=headers,timeout=10).text, 'html.parser', from_encoding='utf-8')
    car_result = soup2.find_all(class_="search-result-list-item")
    for item in car_result:
        dict = {}
        dict["name"]= item.find(class_="cx-name text-hover").string
        price = item.find(class_="cx-price").string
        try:
            dict["low_price"]=re.match("(\d+\.\d+)\-(\d+\.\d+)万",price).group(1)+"万"
            dict["up_price"]=re.match("(\d+\.\d+)\-(\d+\.\d+)万",price).group(2)+"万"
        except:
            dict["low_price"]=price
            dict["up_price"]=price
        dict["pic"]= item.find(class_="img")["src"]
        list.append(dict)

f=open('list.csv','w',newline='')
writer = csv.writer(f)
writer.writerow(['name','low_price','up_price','pic'])
for item in list:
    writer.writerow([item['name'],item['low_price'],item['up_price'],item['pic']])
f.close()
