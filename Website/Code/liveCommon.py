months = ["January","February","March","April","May","June","July","August","September","October","November","December"]

years = [2020, 2021, 2022]


commodity_list =["Onion" , "Potato"]

potatoMandis = ["Uttar Pradesh", "West Bengal"]
OnionMandis = ["Maharashtra", "NCT of Delhi", "Karnataka"]

states = ["Uttar Pradesh", "West Bengal", "Maharashtra", "NCT of Delhi", "Karnataka"]

potatoCentres = ["Lucknow", "Kolkata"]
OnionCentes = ["Mumbai", "Delhi", "Bangalore"] 

mandiCodes = { "Maharashtra" : ["Lasalgaon" , 146],
				"Karnataka": ["Bangalore" , 109],
				"West Bengal" : ["Kalyani" , 578],
				"NCT of Delhi" : ["Azadpur" , 164],
				"Uttar Pradesh": ["Bijnaur" , 284] 
			}

months_dict = {"Maharashtra" : ["January", 2020],
				"Karnataka" : ["January", 2020],
				"West Bengal" :["January", 2020],
				"NCT of Delhi" :["January", 2020] ,
				"Uttar Pradesh" : ["January", 2020]
			}
	


def wholeDataCrawling(myDicts):
	centres = []
	months = []
	years = []

	for k in myDicts['monthsDict'].keys():
		centres.append(k)
		months.append(myDicts['monthsDict'][k][0])
		years.append(myDicts['monthsDict'][k][1])

	new_dict = {}
	dict = crawl_data(centres, months, years)
	for k in dict.keys():
		if(dict[k][0] == 1):
			indexValue = monthList.index(dict[k][1])
			if((indexValue + 1) == len(monthList)):
				m = monthList[0]
				y = dict[k][2] + 1
			else:
				m = monthList[indexValue+1]
				y = dict[k][2]
		else:
			m = dict[k][1]
			y = dict[k][2]
		new_dict[centres] = [m.y]
	myDicts['monthsDict'] = new_dict
	return myDicts


def retailDataCrawling(myDicts):
	centres = []
	months = []
	years = []

	for k in myDicts['monthsDictCentres'].keys():
		centres.append(k)
		months.append(myDicts['monthsDictCentres'][k][0])
		years.append(myDicts['monthsDictCentres'][k][1])

	new_dict = {}
	dict = extractData(centres, months, years)
	for k in dict.keys():
		if(dict[k][0] == 1):
			indexValue = monthList.index(dict[k][1])
			if((indexValue + 1) == len(monthList)):
				m = monthList[0]
				y = dict[k][2] + 1
			else:
				m = monthList[indexValue+1]
				y = dict[k][2]
		else:
			m = dict[k][1]
			y = dict[k][2]
		new_dict[centres] = [m.y]
	myDicts['monthsDictCentres'] = new_dict
	return myDicts



