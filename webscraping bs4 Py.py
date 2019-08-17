# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 21:25:12 2019

@author: ADMIN
"""
import bs4
import requests
from requests import get
from bs4 import BeautifulSoup as soup

#multiple pages for elements 1 to 1000
for p in range(1,502,100): 
 source=requests.get('https://www.imdb.com/search/title/?title_type=feature&release_date=2016-01-01,2016-12-31&count=100&start={}'.format(p)).text
 source

 page_soup=soup(source,"html.parser")
 page_soup.h1
 page_soup.p
 type(page_soup)
#creating a placeholder of elements
 container=page_soup.find_all('div',class_='lister-item-content')
 print(type(container))
 print(len(container))

#first element of container
 container[0].div
 title=container[0].h3.a.text
 title
 year=container[0].h3.find("span", class_="lister-item-year text-muted unbold").text 
 year
 rating= container[0].strong.text
 rating
#All elements of container
 for container in container:
    title= container.h3.a.text
    year= container.h3.find("span", class_="lister-item-year text-muted unbold").text
    rating= container.strong.text
    
    print("Name:"+title)
    print("year:"+year)
    print("rating:"+rating)




    