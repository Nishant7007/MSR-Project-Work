import time
import datetime
import os
import csv
from os import path
import re

from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import StaleElementReferenceException

from selenium import webdriver
import pandas as pd
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException

chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--window-size=1420,1080')
chrome_options.add_argument('--headless')
chrome_options.add_argument('--disable-gpu')

from datetime import timedelta, date

def daterange(date1, date2):
        for n in range(int ((date2 - date1).days)+1):
                yield date1 + timedelta(n)

def givedates(start_dt,end_dt):
    return [dt.strftime("%Y-%m-%d") for dt in daterange(start_dt, end_dt) ]

def monthtodays(month,year):
    if month in ["November","April","June","September"]:
        return 30
    elif month in ["February"]:
        if year%4 ==0:
            return 29
        else:
            return 28
    else:
        return 31

months1 = ["January","February","March","April","May","June","July","August","September","October","November","December"]

def month_name_to_number(number):
    if number == "January":
        return 1
    elif number == "February":
        return 2
    elif number == "March":
        return 3
    elif number == "April":
        return 4
    elif number == "May":
        return 5
    elif number == "June":
        return 6
    elif number == "July":
        return 7
    elif number == "August":
        return 8
    elif number == "September":
        return 9
    elif number == "October":
        return 10
    elif number == "November":
        return 11
    elif number == "December":
        return 12

def extractRetailData(centres, months, years):
    browser = webdriver.Chrome(chrome_options=chrome_options)
    url = 'http://nhb.gov.in/OnlineClient/MonthlyPriceAndArrivalReport.aspx'
    dict = {}
    for i in range(len(centres)):
        centre = centres[i]
        year = years[i]
        month = months[i]
        myfile = '../Data/Processed/Retail/'+str(centre)+'_'+str(year)+'_'+str(month)+'.csv'
        print(myfile)
        if(path.exists(myfile)):
                print('file Exists, skipping')
                dict[centre] = [1,month,year];continue
        if(centre in ['MUMBAI', 'Bengaluru' ,'DELHI']):
            commodity = 'ONION'
        else:
            commodity = 'POTATO'
        print(centre, year, month,commodity)
        try:
            filedata = []
            browser.get(url)
            browser.implicitly_wait(600)
            browser.find_element_by_xpath("//*[@id=\"ctl00_ContentPlaceHolder1_ddlyear\"]/option[contains(text(),\""+str(year)+"\")]").click()
            browser.find_element_by_xpath("//*[@id=\"ctl00_ContentPlaceHolder1_ddlmonth\"]/option[contains(text(),\""+month+"\")]").click()
            browser.implicitly_wait(600)
            browser.find_element_by_xpath("//*[@id=\"ctl00_ContentPlaceHolder1_drpCategoryName\"]/option[contains(text(),\""+"VEGETABLES"+"\")]").click()
            browser.implicitly_wait(600)
            browser.find_element_by_xpath("//*[@id=\"ctl00_ContentPlaceHolder1_drpCropName\"]/option[contains(text(),\""+commodity+"\")]").click()
            browser.implicitly_wait(600)
            browser.find_element_by_xpath("//*[@id=\"ctl00_ContentPlaceHolder1_ddlvariety\"]/option[contains(text(),\""+commodity+"\")]").click()
            browser.implicitly_wait(600)
            browser.find_element_by_xpath("//*[@id=\"ctl00_ContentPlaceHolder1_LsboxCenterList\"]/option[contains(text(),\""+centre+"\")]").click()
            browser.implicitly_wait(600)
            browser.find_element_by_xpath("//*[@id=\"ctl00_ContentPlaceHolder1_btnSearch\"]").click()
            print('data downloading')
            table = browser.find_element_by_xpath("//*[@id=\"ctl00_ContentPlaceHolder1_GridViewmonthlypriceandarrivalreport\"]")
            dict[centre] = [1,month,year]
        except NoSuchElementException:
            print('No Such Element Found')
            dict[centre] = [0,month,year]
            continue
        rows = table.find_elements_by_tag_name("tr")
        monthnum = month_name_to_number(month)
        start_dt = date(year,monthnum,1)
        end_dt = date(year, monthnum, monthtodays(month,year))
        dates = givedates(start_dt,end_dt)
        cells = rows[1].find_elements_by_xpath(".//*[local-name(.)='td']")
        temp = [cell.text for cell in cells]
        for i in range(0,len(dates)):
            temp1 = temp[i+4].split()
            if temp1 != []:
                mystr=dates[i]+',' + centre+ ',' +str(round(int(temp1[3]) / 100, 2))+'\n'
            else:
                mystr=dates[i]+ ',' + centre + ',0'
            #print('mystr')
            #print('mystr');print(mystr)
            new_str = re.sub('[\"\n]','',mystr)
            #print('newstr');print(new_str)
            filedata.append(new_str)
        filedata.sort()
        #print(filedata)
        (pd.DataFrame(filedata)).to_csv(myfile)
    return dict

