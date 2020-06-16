from selenium import webdriver
from folder_creater import commodity_list_1, commodity_list_2, commodity_list_3, commodity_list_4, save_list_4, centres
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import StaleElementReferenceException

chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--window-size=1420,1080')
chrome_options.add_argument('--headless') 
chrome_options.add_argument('--disable-gpu')

def crawl_data(centres, month):
	i = 0

	dict ={}
	for centre in centres:
		dict[centre]=0
	for centre in centres:
		start_year = 2020
		end_year = 2020
		for year in range(start_year,end_year+1):
			print(centre)
			try:
				driver = webdriver.Chrome(chrome_options=chrome_options)
				driver.get('http://agmarknet.gov.in/PriceAndArrivals/DatewiseCommodityReport.aspx#')
				#print('0')

				driver.find_element_by_xpath("//*[@id=\"cphBody_cboYear\"]/option[contains(text(),\""+str(year)+"\")]").click()
				driver.implicitly_wait(6)
				#print('1')

				driver.find_element_by_xpath("//*[@id=\"cphBody_cboMonth\"]/option[contains(text(),\""+str(month)+"\")]").click()
				driver.implicitly_wait(10)
				#print('2')


				driver.find_element_by_xpath("//*[@id=\"cphBody_cboState\"]/option[contains(text(),\""+str(centre)+"\")]").click()
				driver.implicitly_wait(10)
				#print('3')

				driver.find_element_by_xpath("//*[@id=\"cphBody_cboCommodity\"]/option[contains(text(),\""+str('Onion')+"\")]").click()
				driver.implicitly_wait(10)
				print('downloading data')

				driver.find_element_by_xpath("//*[@id=\"cphBody_btnSubmit\"]").click()
				table = driver.find_element_by_xpath("//*[@id=\"cphBody_gridRecords\"]")
				rows = table.find_elements_by_tag_name("tr")
				st = ''
				count=0
				for row in rows:
					cells = row.find_elements_by_xpath(".//*[local-name(.)='th' or local-name(.)='td']")
					#print(cells)
					for cell in cells:
						st += cell.text+','
						#print(cell.text)
					st+='\n'

				myfile= open('../Data/Original/Wholesale/'+str(centre)+'/mynewdata_'+str(year)+'_'+str(month)+'.csv')
				myfile.write(st)
				myfile.close()
				#print(month+"completed")
				driver.close()
				i+=1
				dict[center] = 1

			except (NoSuchElementException,StaleElementReferenceException) as e:
				#print("NR")
				driver.close()
				continue
	return i,dict