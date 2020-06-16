
from os import path
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import StaleElementReferenceException

chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--window-size=1420,1080')
chrome_options.add_argument('--headless')
chrome_options.add_argument('--disable-gpu')

def extractWholesaleData(centres, months, years):
    dict ={}
    for i in range(len(centres)):
        centre = centres[i]
        month = months[i]
        year = years[i]
        fileName = '../Data/Original/WholesaleRaw/'+str(centre)+'/mynewdata_'+str(year)+'_'+str(month)+'.csv'
        if(path.exists(fileName)):
                print('file Exists, skipping')
                dict[centre] = [1,month,year]
                continue
        if(centre in ['Maharashtra', 'Karnataka' ,'NCT of Delhi']):
            commodity = 'Onion'
        else:
            commodity = 'Potato'
        print(centre, year, month,commodity)

        for i in range(3):
            print(i,'th time')
            try:
                print('1',end=' ')
                driver = webdriver.Chrome(chrome_options=chrome_options)

                print('2',end=' ')
                driver.get('http://agmarknet.gov.in/PriceAndArrivals/DatewiseCommodityReport.aspx#')

                print('3',end=' ')
                driver.find_element_by_xpath("//*[@id=\"cphBody_cboYear\"]/option[contains(text(),\""+str(year)+"\")]").click()
                driver.implicitly_wait(606)

                for k in range(10):
                    try:
                        print(k,end=' ')
                        driver.find_element_by_xpath("//*[@id=\"cphBody_cboMonth\"]/option[contains(text(),\""+str(month)+"\")]").click()
                        driver.implicitly_wait(20)
                        break
                    except (StaleElementReferenceException) as x:
                        k+=1

                print('6',end=' ')
                driver.implicitly_wait(600)

                print('7',end=' ')
                driver.find_element_by_xpath("//*[@id=\"cphBody_cboState\"]/option[contains(text(),\""+str(centre)+"\")]").click()
                driver.implicitly_wait(600)
                print('9',end=' ')
                driver.find_element_by_xpath("//*[@id=\"cphBody_cboCommodity\"]/option[contains(text(),\""+str(commodity)+"\")]").click()

                print('10',end=' ')
                driver.implicitly_wait(600)
                print('downloading data')

                driver.find_element_by_xpath("//*[@id=\"cphBody_btnSubmit\"]").click()
                table = driver.find_element_by_xpath("//*[@id=\"cphBody_gridRecords\"]")
                rows = table.find_elements_by_tag_name("tr")
                st = ''
                for row in rows:
                    cells = row.find_elements_by_xpath(".//*[local-name(.)='th' or local-name(.)='td']")
                    #print(cells)
                    for cell in cells:
                        st += cell.text+','
                        #print(cell.text)
                    st+='\n'

                myfile= open('../Data/Original/WholesaleRaw/'+str(centre)+'/mynewdata_'+str(year)+'_'+str(month)+'.csv',"w")
                myfile.write(st)
                myfile.close()
                print(month+" completed")
                driver.close()
                dict[centre] = [1,month,year]
                break
            except (NoSuchElementException,StaleElementReferenceException) as e:
                i+=1
                print("NR")
                driver.close()
                dict[centre] = [0,month,year]
                continue
    return dict
