#this is same as download.py but this is for potato

import time
import datetime
import os
import csv
from selenium import webdriver
#from bs4 import BeautifulSoup
import pandas as pd
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

months1 = ["January","February","March","April","May","June","July","August","September","October","November","December"]
months2 = ["July","August","September","October","November", "December"]
months3 = ["September","October","November","December"]
months4 = ["December"]

'''
start_year = 2014
end_year = 2017
mandi_file = pd.read_csv('mandis.csv')
mandicode = mandi_file['mandicode']
mandiname = mandi_file['mandiname']
mandistate = mandi_file['statecode']
mandi_map = {}
mandi_state_map={}
i=0
for row in mandiname:
	mandi_map[row] = mandicode[i]
	mandi_state_map[row] = mandistate[i]
	i = i+1
'''

centernames = ["Madhya Pradesh"]

def extractdata():

	#path_to_chromedriver = '/usr/local/Cellar/chromedriver/2.35'
	#browser = webdriver.Chrome()
	#url = 'http://agmarknet.gov.in/PriceAndArrivals/DatewiseCommodityReport.aspx'
	#browser.get(url)
	#print("url has been opened")
	#myfile= open('data/wholesaleData/mynewdata.csv','w')
	for center in centernames:
		start_year = 2006
		end_year = 2009
		for year in range(start_year,end_year+1):
			months = months1
			# if(year == 2017):
			# 	months = months2
			# elif(year == 2016):
			# 	months = months3

			for month in months:
				print(month + "started")
				browser = webdriver.Chrome('/home/nishant/research/chromedriver')
				url = 'http://agmarknet.gov.in/PriceAndArrivals/DatewiseCommodityReport.aspx'
				browser.get(url)
				print (year,month)
				browser.implicitly_wait(6)
				browser.find_element_by_xpath("//*[@id=\"cphBody_cboYear\"]/option[contains(text(),\""+str(year)+"\")]").click()
				browser.find_element_by_xpath("//*[@id=\"cphBody_cboMonth\"]/option[contains(text(),\""+month+"\")]").click()
				browser.implicitly_wait(10)
				browser.find_element_by_xpath("//*[@id=\"cphBody_cboState\"]/option[contains(text(),\""+center+"\")]").click()
				browser.implicitly_wait(10)
				browser.find_element_by_xpath("//*[@id=\"cphBody_cboCommodity\"]/option[contains(text(),\""+"Onion"+"\")]").click()
				browser.implicitly_wait(10)
				browser.find_element_by_xpath("//*[@id=\"cphBody_btnSubmit\"]").click()
				table = browser.find_element_by_xpath("//*[@id=\"cphBody_gridRecords\"]")
				rows = table.find_elements_by_tag_name("tr")
				st = ''
				count=0
				for row in rows:
					cells = row.find_elements_by_xpath(".//*[local-name(.)='th' or local-name(.)='td']")
					#print(cells)
					for cell in cells:
						st += cell.text+','
						print(cell.text)
					st+='\n'
					#print(st)
				myfile= open('/home/nishant/research/Low_Price_Anomaly_Detection/Data/wholesale/mp2/mynewdata_'+str(year)+'_'+str(month)+'.csv','a')
				myfile.write(st)
				myfile.close()
				print(month+"completed")
				browser.close()
				#browser.find_element_by_xpath("//*[@id=\"LinkButton1\"]").click()

if __name__ == '__main__':
	extractdata()