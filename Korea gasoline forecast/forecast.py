import pickle
import csv
import numpy as np
import matplotlib.pyplot as plt

option = "Korea"
####################################
# making index for first and then start on
####################################

month_list=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

def make_index(year):
	#put the starting year that you want to make index
	if type(year) != int:
		print("Error: Check input. Year should be in integer format.")
		return
	a=open("pickle set\\index_to_p.p",'wb')
	b=open("pickle set\\period_to_i.p",'wb')

	itop = {}
	ptoi = {}

	index=0
	for i in range(year,2019):
		for month in range(12):
			period = month_list[month] + " " + str(i)
			index+=1
			ptoi[period] = index
			itop[index] = period

	pickle.dump(itop,a)
	pickle.dump(ptoi,b)
	a.close()
	b.close()
	return

#we need itop, ptoi after this point so implement the make index.
make_index(1900)
f_index = open("pickle set\\index_to_p.p",'rb')
itop = pickle.load(f_index)
f_index.close()

f_period = open("pickle set\\period_to_i.p",'rb')
ptoi = pickle.load(f_period)
f_period.close()


############################################
#	Defining result class 		#
############################################

class forecast_result():
	def __init__(self):
		self.name = ""
		self.info = ""
		self.step = 0	#zero means we want to forecast 1~24 steps by one model
		self.time_window = 0
		self.original = {}
		self.fitted = {}
		self.forecast = {}
		self.training_num = 0
		self.testing_num = 0
		self.total_num = 0
		self.option = ""
	


	def information(self):
		print("--------<information>----------")
		print("Model name: " + self.name)
		print("Information: " + self.info)

		if self.step==0:
			print("Forecasting step: 1 to 24")
		else:
			print("Forecasting step: " + str(self.step))

		print("Number of training/testing data :" + str(self.training_num) + ", " + str(self.testing_num))
		fit_list = list(self.fitted.keys())
		forecast_list = list(self.forecast.keys())
		fit_start = min(fit_list)
		fit_end = max(fit_list)
		forecast_start = min(forecast_list)
		forecast_end = max(forecast_list)
		print("Fitting: "+itop[fit_start] + " to " + itop[fit_end])
		print("Forecasting: "+itop[forecast_start] + " to " +itop[forecast_end])
		print("-------------------------------")
		return



	def plot(self):
		total_list = list(self.original.keys())
		fitted_list = list(self.fitted.keys())
		forecast_list = list(self.forecast.keys())
		total_list.sort()
		fitted_list.sort()
		forecast_list.sort()

		y_total = []
		y_fitted = []
		y_forecast = []
		for i in total_list:
			y_total.append(self.original[i])
		for i in fitted_list:
			y_fitted.append(self.fitted[i])
		for i in forecast_list:
			y_forecast.append(self.forecast[i])

		plt.plot(total_list,y_total,'b',label="Real data values")
		plt.plot(fitted_list,y_fitted,'g',label="Fitted values")
		plt.plot(forecast_list,y_forecast,'r',label="Forecast values")
		plt.xlabel("time[month index]")
		plt.ylabel("real price[won/L]")
		plt.title(self.name)
		plt.legend(loc='upper left')
		plt.show()

		return


	def mspe(self,option="forecast",print_option="not quiet"):
		error = []
		if option=="forecast":
			error_index = list(self.forecast.keys())
			for i in error_index:
				error.append(self.forecast[i]-self.original[i])
		elif option=="fitted":
			error_index = list(self.fitted.keys())
			for i in error_index:
				error.append(self.fitted[i]-self.original[i])
		else:
			raise ValueError("Check your option. 'forecast' and 'fitted' options are available. Default is 'forecast'.")
			return

		sum_error = 0
		for e in error:
			sum_error+=np.power(e,2)

		if print_option == "not quiet":
			print("MSPE of the model you selected ("+str(option)+"): "+str(sum_error))
		return sum_error


	def mspe_ratio(self,print_option = "not quiet"):
		error = self.mspe(print_option="queit")

		model_scale = model_nochange(self.step,testnum=self.testing_num,time_window=self.time_window)
		error_scale = model_scale.mspe(print_option="quiet")
		error_rate = error/error_scale

		if print_option!= "quiet":
			print("Error / No-change error = "+str(error_rate))
		return error_rate


	def success_ratio(self, print_option="not quiet"):
		if self.step == 0:
			print("You actually don't need to count success ratio in this case...")
			return
		
		if self.name == "Nochange":
			print("Can't evaluate success rate for no-change model.")
			return

		h = self.step
		count_correct = 0
		count_total = 0
		for i in self.forecast.keys():
			sign_real = self.original[i]-self.original[i-h]
			sign_forecast = self.forecast[i]-self.original[i-h]
			check = sign_real * sign_forecast
			count_total+=1
			#########should modity this part - 기준을 정확하게 0으로 둘 게 아니라, hypothesis를 이용. no change는 항상 0으로 가정하기 때문.
			if check>=0:
				count_correct+=1

		answer = count_correct / count_total
		if print_option != "quiet":
			print("Success ratio of the model you selected: "+str(count_correct)+" out of "+str(count_total)+" = "+str(answer))
		return answer


##########################################
#	additional function	
#		change forecast window
#		cut from the latest data
##########################################
def change_window(original,cutting_num):
	cutdict = {}
	keylist = list(original.keys())
	keylist.sort()
	cutkey = keylist[:-cutting_num]
	for k in cutkey:
		cutdict[k] = original[k]
	return cutdict


############################################
#	preparing global indexes			####
#		pickle files are basically included		
#		but implement prepare in case of 	
#	 	revising the original data 		
############################################

def prepare_all():
	prepare_exchange()
	prepare_k_cpi()
	prepare_k_gasoline()
	prepare_k_import()
	return


def prepare_exchange():
	f_exchange_text = open("data set\\Exchange rate.csv",'r')
	r2 = csv.reader(f_exchange_text)

	ex_dict = {}
	for info in r2:
		raw_name=info[0].split("-")
		month_num = int(raw_name[1])
		new_name = month_list[month_num-1]+" "+raw_name[0]
		try:
			ex_dict[ptoi[new_name]]=float(info[1])
		except:
			pass

	f_ex_save = open("pickle set\\exchange_rate.p",'wb')
	pickle.dump(ex_dict,f_ex_save)
	f_ex_save.close()
	return


def prepare_k_cpi():
	#korea cpi in won
	f_kcpiwon_text=open("data set\\Korea cpi.csv",'r')
	r1=csv.reader(f_kcpiwon_text)

	raw_data = []
	kcpi = {}

	for line in r1:
		raw_data.append(line)

	period_cpi = raw_data[0]
	value_cpi = raw_data[1]

	for i in range(len(period_cpi)):
		p=period_cpi[i]
		p_list = p.split("-")
		if p_list[1][0]=="0" or p_list[1][0]=="1":
			p_name = p_list[0] + " 20" + p_list[1]
		else:
			p_name = p_list[0] + " 19" + p_list[1]
		
		#scale을 world cpi와 맞춰주기 위해 100 나눔
		try:
			kcpi[ptoi[p_name]] = float(value_cpi[i])/100
		except:
			pass

	f_kcpiwon_save = open("pickle set\\kcpi_dict.p",'wb')
	pickle.dump(kcpi,f_kcpiwon_save)
	f_kcpiwon_save.close()


	####make pi dict####
	eir=[] #Expected value of inflation rate
	time_key = list(kcpi.keys())
	time_key.sort()
	max_time = time_key[-1]
	min_time = time_key[0]
	for h in range(1,25):
		dict_eir={}
		inflation = {}
		inflation[min_time+h]=(kcpi[min_time+h]-kcpi[min_time])/h
		dict_eir[min_time+h]=inflation[min_time+h]
		for t in range(min_time+h+1,max_time):
			inflation[t]=(kcpi[t]-kcpi[t-h])/h
			dict_eir[t]=((t-min_time-h+1)*dict_eir[t-1]+inflation[t])/(t-min_time-h+2)
		eir.append(dict_eir)
		f_eirsave = open("pickle set\\k_eir_dict.p",'wb')
	pickle.dump(eir,f_eirsave)
	f_eirsave.close()

	return


def prepare_k_gasoline():
	#read gasoline csv file
	f_gasoline_text = open("data set\\Korea gasoline nominal.csv",'r')
	r=csv.reader(f_gasoline_text)

	nominal_dict = {}
	for info in r:
		raw_name = info[0].split("-")
		if raw_name[1][0] == "9" or raw_name[1][0] == "8":
			name = raw_name[0]+" 19"+raw_name[1]
		elif raw_name[1][0] == "0" or raw_name[1][0]=="1":
			name = raw_name[0]+" 20"+raw_name[1]
		nominal_dict[ptoi[name]]=float(info[1])

	#deflate
	f_cpi = open("pickle set\\kcpi_dict.p",'rb')
	cpi_dict = pickle.load(f_cpi)
	
	real_price = {}
	log_price = {}
	for key in nominal_dict.keys():
		try:
			value = nominal_dict[key]/ cpi_dict[key]
			real_price[key]=value
			log_price[key]=np.log(value)
		except:
			pass

	f_real_save = open("pickle set\\k_gasoline_dict.p",'wb')
	pickle.dump(real_price,f_real_save)
	f_real_save.close()

	f_log_save = open("pickle set\\logkorea_dict.p",'wb')
	pickle.dump(log_price,f_log_save)
	f_log_save.close()

	return log_price


def prepare_k_import():
	f_kcpi=open("pickle set\\kcpi_dict.p",'rb')
	kcpi_dict = pickle.load(f_kcpi)


	f_exchange = open("pickle set\\exchange_rate.p",'rb')
	ex_dict = pickle.load(f_exchange)


	f_import = open("data set\\Korea import prices.csv",'r')
	r=csv.reader(f_import)

	imp_dict={}
		
	for line in r:
		period_list = line[0].split("-")
		try:
			year = int(period_list[0])
			year = period_list[0]
			month = period_list[1]
		except:
			year = int(period_list[1])
			year = period_list[1]
			month = period_list[0]


		if year[0]=="0" or year[0]=="1":
			index = ptoi[month+" 20"+year]
		else:
			index = ptoi[month+" 19"+year]

		try:
			imp_dict[index] = (((float(line[1])*ex_dict[index])/float(kcpi_dict[index])))/158.9873
		except:
			pass
		

	f_import_save = open("pickle set\\k_import_dict.p",'wb')
	pickle.dump(imp_dict,f_import_save)
	f_import_save.close()

	return


############################################
#	main part - making model 	#
############################################

#import global files first
f_korea = open("pickle set\\k_gasoline_dict.p",'rb')
kgasoline_dict = pickle.load(f_korea)
f_korea.close()

f_kcpi = open("pickle set\\kcpi_dict.p",'rb')
kcpi_dict=pickle.load(f_kcpi)
f_kcpi.close()

f_import = open("pickle set\\k_import_dict.p",'rb')
imp_dict=pickle.load(f_import)



def prepare_VAR():
	def prepare_prod():	
		#production change ==> prod_dict
		f_prod_text=open("data set\\Global oil production.csv",'r')
		r_prod=csv.reader(f_prod_text)
		a=[]
		for line in r_prod:
			a.append(line)
		
		date=a[0]
		production=a[1]
		prod_dict = {}
		for i in range(len(date)-1):
			try:
				prod_dict[ptoi[date[i]]] = float(production[i])/float(production[i-1])-1
			except:
				pass

		return prod_dict

	def prepare_invt():
		#US crude oil csv to dict	
		f_usc = open("data set\\US crude oil inventories.csv",'r')
		r = csv.reader(f_usc)
		info_list = []
		for line in r:
			info_list.append(line)

		check_dict = {}
		us_crude_dict = {}

		for info in info_list:
			p = info[0].split("-")

			if p[2][0]=="8" or p[2][0]=="9":
				period = p[1]+ " 19"+p[2]
			else:
				period = p[1]+ " 20"+p[2]

			index = ptoi[period]
			if index in check_dict.keys():
				n = check_dict[index]
				check_dict[index] = n+1
				us_crude_dict[index] = us_crude_dict[index]*n/(n+1) + float(info[1])/(n+1)
			else:
				check_dict[index]=1
				us_crude_dict[index] = float(info[1])

		#US petroleum csv to dict
		f_usp = open("data set\\US petroleum inventories.csv",'r')

		#OECD petroleum csv to dict

		#inventories to it's changes dict
		invt_dict = {}

		####will change this part(after finding the data)
		invt_dict = us_crude_dict


		#inventories change ==> invt_dict
		invt_change = {}
		for i in invt_dict.keys():
			try:
				invt_change[i] = invt_dict[i]-invt_dict[i-1]
			except:
				pass

		return invt_dict


	def prepare_kinvt():
		#Korea crude oil inventories change
		f_text = open('data set\\Korea crude oil inventories.csv','r')
		r_info = csv.reader(f_text)
		list_invt=[]
		for info in r_info:
			list_invt.append(info)

		kinvt_dict={}
		for line in list_invt:
			period_list = line[0].split("-")
			if period_list[1][0]=="0" or period_list[1][0]=="1":
				year = "20"+period_list[1]
			else:
				year = "19"+period_list[1]
			index = ptoi[period_list[0]+" "+year]

			value = line[1].replace(",","")
			try:
				kinvt_dict[index] = float(value)
			except:
				pass

		kinvt_diff = {}
		for key in kinvt_dict.keys():
			try:
				kinvt_diff[key]=kinvt_dict[key]-kinvt_dict[key-1]
			except:
				pass

		return kinvt_diff


	def prepare_gdp():
	#gdp per capita - global real activity ==> gdp_dict
	#make population dict first
		f_po_text = open("data set\\Korea population.csv",'r')
		r_po = csv.reader(f_po_text)
		po_info=[]
		for info in r_po:
			po_info.append(info)

		period_year = po_info[0]
		value_year = po_info[1]
		period_month = po_info[2]
		value_month = po_info[3]

		po_dict = {}
		for i in range(len(period_year)):
			p=period_year[i]
			for m in month_list:
				p_name = m+" "+p
				try:
					po_dict[ptoi[p_name]]=float(value_year[i].replace(",",""))
				except:
					pass

		for i in range(len(period_month)):
			p_list=period_month[i].split("-")
			if p_list[1][0]=="0" or p_list[1][0]=="1":
				p_name = p_list[0]+" 20"+p_list[1]
			else:
				p_name = p_list[0]+" 19"+p_list[1]

			try:
				po_dict[ptoi[p_name]]=float(value_month[i].replace(",",""))
			except:
				pass
	#make gdp per capita real price dictionary
		f_gdp_text = open("data set\\Korea gdp nominal.csv",'r')
		r_gdp = csv.reader(f_gdp_text)
		gdp_info=[]
		for info in r_gdp:
			gdp_info.append(info)

		period_gdp = gdp_info[0]
		value_gdp = gdp_info[1]
		gdp_dict = {}

		for i in range(len(value_gdp)):
			v=value_gdp[i]
			if "," in v:
				new_v=v.replace(",","")
				value_gdp[i]=new_v
		
		gdp_dict={}
		for i in range(len(period_gdp)):
			p = period_gdp[i]
			v = value_gdp[i]
			#v 정리
			if "," in v:
				v=v.replace(",","")
			#p 정리 
			p_list = p.split(" ")
			if len(p)==8:
				year = p_list[0]
				m = p_list[1]
				if m=="1/4":
					period = ["Jan "+year, "Feb "+year, "Mar "+year]
				elif m=="1/2":
					period = ["Apr "+year, "May "+year, "Jun "+year]
				else:
					period = ["Jul "+year, "Aug "+year, "Sep "+year]

			else:
				year = str(int(p_list[0])-1)
				period = ["Oct "+year, "Nov "+year, "Dec "+year]

			#deflate and divide by population
			for p_name in period:
				try:
					index = ptoi[p_name]
					gdp_dict[index] = (float(v)/po_dict[index]*kcpi_dict[index])*(10**9)
				except:
					pass

		gdp_rate = {}
		for key in gdp_dict.keys():
			try:
				g2=gdp_dict[key]
				g1=gdp_dict[key-1]
				rate = (g2-g1)/g1
				gdp_rate[key]=rate
			except:
				pass

		return gdp_rate


	def prepare_imp():
	#Import price ==> imp_dict
	##prepare global에서 import prepare 한 뒤에 시작.

		imp_rate = {}
		for key in imp_dict.keys():
			try:
				i2=imp_dict[key+1]
				i1=imp_dict[key]
				rate = (i2-i1)/i1
				imp_rate[key]=rate
			except:
				pass
		return imp_rate

	##################################3
	#integrating all components
	total_dict={}
	total_dict["prod"]=prepare_prod()
	total_dict["gdp"]=prepare_gdp()
	total_dict["invt"]=prepare_invt()
	total_dict["kinvt"]=prepare_kinvt()
	total_dict["imp"]=prepare_imp()

	f_save = open("pickle set\\model1_VARvector.p",'wb')
	pickle.dump(total_dict,f_save)
	f_save.close()

	return

def model_VAR(step,testnum=50,time_window=0,model_option=1,prepare=True):
	if prepare == True:
		prepare_VAR()

	original = kgasoline_dict
	f_logkorea = open("pickle set\\logkorea_dict.p",'rb')
	log_original = pickle.load(f_logkorea)
	f_logkorea.close()

	if time_window != 0:
		original = change_window(original,time_window)
		log_original = change_window(log_original,time_window)

	import statsmodels.api as sm
	import pandas
	
	f_data = open("pickle set\\model1_VARvector.p",'rb')
	vec_data = pickle.load(f_data)

	prod_dict = vec_data["prod"]
	gdp_dict = vec_data["gdp"]
	invt_dict = vec_data["invt"]
	kinvt_dict = vec_data["kinvt"]
	imp_dict = vec_data["imp"]

	#put oil data according to your option, depends on model number(korea)
	total_vector={}
	if model_option == 1:
		for key in log_original.keys():
			try:
				vector=[float(log_original[key]),float(prod_dict[key]),float(gdp_dict[key]),float(invt_dict[key])]
				total_vector[key] = vector
			except:
				pass
	elif model_option==2:
		for key in log_original.keys():
			try:
				vector=[float(log_original[key]),float(prod_dict[key]),float(gdp_dict[key]),float(kinvt_dict[key])]
				total_vector[key] = vector
			except:
				pass
	elif model_option==3:
		for key in log_original.keys():
			try:
				vector=[float(log_original[key]),float(prod_dict[key]),float(gdp_dict[key]),float(kinvt_dict[key]),float(invt_dict[key])]
				total_vector[key] = vector
			except:
				pass
	elif model_option == 4:
		for key in log_original.keys():
			try:
				vector=[float(log_original[key]),float(prod_dict[key]),float(gdp_dict[key]),float(invt_dict[key]),float(imp_dict[key])]
				total_vector[key] = vector
			except:
				pass
	else:
		print("Check your option!!!")
		return

	#split into test and training part
	keylist = list(total_vector.keys())
	keylist.sort()
	training_vector={}
	for k in keylist[:-testnum]:
		training_vector[k] = total_vector[k]

	xraw=pandas.DataFrame(training_vector)
	x=np.asarray(xraw).transpose()

	model = sm.tsa.VAR(x)
	var_result = model.fit(12)


	total_list=[]
	for k in keylist:
		total_list.append(total_vector[k])

	start_point = min(keylist)
	fitted = {}
	for i in keylist[12:-testnum-step]:
		fitted[i+step] = np.exp(var_result.forecast(total_list[:i-start_point],step)[-1][0])

	forecast = {}
	for i in keylist[-testnum-step:-step]:
		forecast[i+step] = np.exp(var_result.forecast(total_list[:i-start_point],step)[-1][0])


	result=forecast_result()

	result.name = "VAR model"
	result.info = "Option: "+option+", model number is "+str(model_option)
	result.step = step
	result.time_window=time_window
	result.option = option

	result.original = original
	result.fitted = fitted
	result.forecast = forecast

	result.training_num = len(list(fitted.keys()))
	result.testing_num = testnum
	result.total_num = len(list(original.keys()))

	f_model = open("result set\\VAR_1.p",'wb')
	pickle.dump(result,f_model)
	f_model.close()

	return result


def prepare_ind():
	f_urate_text = open('data set\\Korea utilization rate.csv','r')
	r_urate = csv.reader(f_urate_text)
	a=[]
	for info in r_urate:
		a.append(info)

	period_list = a[0]
	value_list = a[1]

	value_dict={}
	for i in range(len(period_list)):
		p_list=period_list[i]
		year = p_list[:4]
		month = p_list[-2:]
		month_index = int(month)
		month_str = month_list[month_index-1]
		p_name=month_str+" "+year

		try:
			index = ptoi[p_name]
			value_dict[index] = float(value_list[i])
		except:
			pass
	ind_index = value_dict


	pi_total = []
	for step in range(1,25):
		pi = {}
		for i in ind_index.keys():
			try:
				pi[i] = (ind_index[i]/ind_index[i-step])-1
			except:
				pass
		pi_total.append(pi)

	f_pi = open("pickle set\\model2_korea_urate.p",'wb')
	pickle.dump(pi_total,f_pi)
	f_pi.close()

	return

def model_industrial(step,testnum=50,time_window=0,prepare=True,model_option=1):
	if prepare:
		prepare_ind()

	f_pi = open("pickle set\\model2_korea_urate.p",'rb')
	pi_total = pickle.load(f_pi)
	pi_dict = pi_total[step-1]

	f_eir = open("pickle set\\k_eir_dict.p",'rb')
	eir_total = pickle.load(f_eir)
	eir_dict = eir_total[step-1]

	original = kgasoline_dict

	if time_window != 0:
		original = change_window(original,time_window)

	original_key = list(original.keys())
	original_key.sort()
	data_key=[]
	for key in original_key:
		if key in pi_dict.keys() and key in eir_dict.keys():
			data_key.append(key)


	if model_option==1:
		beta = 1

	else:	#if option == "coefficient"

		def error(x):
			b=np.float(x)
			error_list = []
			for num in range(len(data_key)-step-testnum):
				key = data_key[num]
				#error_term = original[key+step]-original[key]*(1+b*pi_dict[key]-eir_dict[key])
				error_term = original[key+step]/original[key]-1-b*pi_dict[key]
				error_list.append(error_term)
			return np.array(error_list)

		from scipy.optimize import least_squares
		beta = np.array([1])
		ans = least_squares(error,beta)
		beta = np.float(ans.x)
		print("estimated beta is: ",beta)


	#lag가 step만큼 생기기 때문에...
	fitted = {}
	for i in data_key[:-testnum-step]:
		fitted[i+step] = original[i]*(1+beta*pi_dict[i]-eir_dict[i])

	forecast = {}
	for i in data_key[-testnum-step:-step]:
		forecast[i+step] = original[i]*(1+beta*pi_dict[i]-eir_dict[i])


	result = forecast_result()
	result.name = "Industrial demand model"
	result.info = "Option: "+option
	result.step = step
	result.time_window = time_window
	result.option = option
	result.original = original
	result.fitted = fitted
	result.forecast = forecast
	result.training_num = len(list(fitted.keys()))
	result.testing_num = len(list(forecast.keys()))
	result.total_num = len(list(original.keys()))


	f_model = open("result set\\Ind_material_1.p",'wb')
	pickle.dump(result,f_model)
	f_model.close()

	return result


def prepare_gasoline():
	prepare_k_gasoline()
	return

def model_gasoline(step,testnum=50,time_window=0,model_option=1,prepare=True):
	if prepare:
		prepare_gasoline()

	original = kgasoline_dict

	if time_window != 0:
		original = change_window(original,time_window)

	f_eir = open("pickle set\\k_eir_dict.p",'rb')
	eir_total = pickle.load(f_eir)
	eir_dict = eir_total[step-1]
	
	#for convenience, change name	
	gasoline = original
	crude = imp_dict

	###########estimate beta###############
	keylist = []
	for key in list(crude.keys()):
		try:
			gasoline[key]
			keylist.append(key)
		except:
			pass

	keylist.sort()
	gas_list=[]
	crude_list=[]
	for k in keylist:
		gas_list.append(float(gasoline[k]))
		crude_list.append(float(crude[k]))


	def error1(x):
		b=np.float(x)
		error_list = []
		crude_copy = crude_list.copy()
		gas_copy = gas_list.copy()
		for num in range(len(keylist)-step-testnum):
			c0 = crude_copy[0]
			g0 = gas_copy[0]
			error_term = np.log(gas_copy[step]/g0)-b*(np.log(c0/g0))
			crude_copy.remove(c0)
			gas_copy.remove(g0)
			error_list.append(error_term)
		return np.array(error_list)

	def error2(x):
		b=np.float(x)
		error_list = []
		crude_copy = crude_list.copy()
		gas_copy = gas_list.copy()
		for num in range(len(keylist)-step-testnum):
			c0 = crude_copy[0]
			g0 = gas_copy[0]
			error_term = np.log(gas_copy[step]/c0)-b*(np.log(g0/c0))
			crude_copy.remove(c0)
			gas_copy.remove(g0)
			error_list.append(error_term)
		return np.array(error_list)		


	from scipy.optimize import least_squares
	from math import exp
	beta = np.empty([1])
	#ans = least_squares(error,beta,verbose=2)
	if model_option == 1:
		ans = least_squares(error1,beta)
	else:
		ans = least_squares(error2,beta)
	beta = np.float(ans.x)
	############################################


	if model_option == 1:
		fitted = {}
		for i in keylist[:-testnum-step]:
			try:
				fitted[i+step] = original[i]*(exp(beta*(np.log(float(crude[i])/float(gasoline[i])))-eir_dict[i]))
			except:
				pass

		forecast = {}
		for i in keylist[-testnum-step:-step]:
			try:
				forecast[i+step] = original[i]*(exp(beta*(np.log(float(crude[i])/float(gasoline[i])))-eir_dict[i]))
			except:
				pass

	else:
		fitted = {}
		for i in keylist[:-testnum-step]:
			try:
				fitted[i+step] = crude[i]*(exp(beta*(np.log(float(gasoline[i])/float(crude[i])))-eir_dict[i]))
			except:
				pass

		forecast = {}
		for i in keylist[-testnum-step:-step]:
			try:
				forecast[i+step] = crude[i]*(exp(beta*(np.log(float(gasoline[i])/float(crude[i])))-eir_dict[i]))
			except:
				pass


	result = forecast_result()
	result.name = "Gasoline spread model"
	result.info = "Option: "+option+", model option:"+str(model_option)+"\n 		estimated beta = "+str(beta)
	result.step = step
	result.time_window=time_window
	result.option = option
	result.original = original
	result.fitted = fitted
	result.forecast = forecast
	result.training_num = len(list(fitted.keys()))
	result.testing_num = len(list(forecast.keys()))
	result.total_num = len(list(original.keys()))


	f_model = open("result set\\Gasoline spread_1.p",'wb')
	pickle.dump(result,f_model)
	f_model.close()

	return result


def prepare_products():
	f_mprice = open("data set\\Korea products prices.csv",'r')
	r_price = csv.reader(f_mprice)
	info = []
	for line in r_price:
		info.append(line)

	price_dict = {}
	for data in info:
		period_raw = data[0]
		p_list = period_raw.split("-")
		if len(p_list)>1:
			if p_list[1][0] == "0" or p_list[1][0] == "1":
				year = "20"+p_list[1]
			else:
				year = "19"+p_list[1]
			period = p_list[0] + " " + year
			index = ptoi[period]

			price_vec = []
			for i in range(3):
				num = float(data[i+1].replace(",",""))
				price_vec.append(num)
			price_dict[index]=price_vec

	#make pi dictionary; change ratio of the average prices of refined products
	#will be used for VAR
	pi_total = []
	for step in range(1,25):
		pi = {}
		for i in price_dict.keys():
			try:
				v=price_dict[i]
				v0=price_dict[i-step]
				pi_v=[]
				for d in range(3):
					pi_v.append(v[d]/v0[d]-1)
				pi[i] = pi_v
			except:
				pass
		pi_total.append(pi)

	#make avg dictionary; average value of the prices at that time
	avg_dict = {}
	for key in price_dict.keys():
		v = price_dict[key]
		avg_dict[key] = (v[0]+v[1]+v[2])/3

	#avg changing rate dictionary
	avg_rate = []
	for h in range(24):
		rate_dict={}
		for key in avg_dict.keys():
			try:
				rate_dict[key]=avg_dict[key]/avg_dict[key-h-1]-1
			except:
				pass
		avg_rate.append(rate_dict)


	f_save = open("pickle set\\model4_products prices.p",'wb')
	pickle.dump(price_dict,f_save)
	f_save.close()

	pi_save = open("pickle set\\model4_prices rate.p",'wb')
	pickle.dump(pi_total,pi_save)
	pi_save.close()

	avg_save = open("pickle set\\model4_average price.p",'wb')
	pickle.dump(avg_dict,avg_save)
	avg_save.close()

	return

def model_products(step, testnum=50, model_option=1, time_window=0, prepare=True):
	if bool(prepare):
		prepare_products()
	#print("vector label: 실내등유, 자동차용경유(1%), 선박용경유(0.003%)")
	f_vector = open("pickle set\\model4_products prices.p",'rb')
	v_data = pickle.load(f_vector)

	f_eir = open("pickle set\\k_eir_dict.p",'rb')
	eir_total = pickle.load(f_eir)
	eir_dict = eir_total[step-1]

	f_pi = open("pickle set\\model4_prices rate.p",'rb')
	pi_total = pickle.load(f_pi)
	pi_dict = pi_total[step-1]

	f_avg = open("pickle set\\model4_average price.p",'rb')
	avg_dict = pickle.load(f_avg)


	original = kgasoline_dict

	if time_window != 0:
		original = change_window(original,time_window)

	data_key=[]
	for key in original.keys():
		if bool(key in v_data.keys()) and bool(key in pi_dict.keys()):
			data_key.append(key)
	data_key.sort()

	def error1(x):
		b=np.float(x)
		error_list = []

		for num in range(len(data_key)-step-testnum):
			key = data_key[num]
			g0 = kgasoline_dict[key]
			gh = kgasoline_dict[key+step]
			avg = avg_dict[key]
			error_term = np.log(gh/g0)-b*(np.log(avg/g0))
			error_list.append(error_term)
		return np.array(error_list)

	def error3(x):
		b=np.float(x)
		error_list = []

		for num in range(len(data_key)-step-testnum):
			key = data_key[num]
			error_term = kgasoline_dict[key+step]/kgasoline_dict[key]-1-b*(avg_dict[key+step]/avg_dict[key]-1)
			error_list.append(error_term)
		return np.array(error_list) 


	if model_option==1:
		from scipy.optimize import least_squares
		beta = np.array([1])
		ans = least_squares(error1,beta)
		beta = np.float(ans.x)
		print("estimated beta is: ",beta)

		fitted = {}
		for i in data_key[:-testnum-step]:
			fitted[i+step] = original[i]*np.exp(beta*np.log(avg_dict[i]/original[i])-eir_dict[i])

		forecast = {}
		for i in data_key[-testnum-step:-step]:
			forecast[i+step] = original[i]*np.exp(beta*np.log(avg_dict[i]/original[i])-eir_dict[i])

	elif model_option==2:
		fitted = {}
		for i in data_key[:-testnum-step]:
			fitted[i+step] = original[i]*(avg_dict[i]/avg_dict[i-step]-eir_dict[i])

		forecast = {}
		for i in data_key[-testnum-step:-step]:
			forecast[i+step] = original[i]*(avg_dict[i]/avg_dict[i-step]-eir_dict[i])
		pass


	elif model_option==3:
		from scipy.optimize import least_squares
		beta = np.array([1])		
		ans = least_squares(error3,beta)
		beta = np.float(ans.x)
		print("estimated beta is: ",beta)

		fitted = {}
		for i in data_key[:-testnum-step]:
			fitted[i+step] = original[i]*(1+beta*(avg_dict[i]/avg_dict[i-step]-1)-eir_dict[i])

		forecast = {}
		for i in data_key[-testnum-step:-step]:
			forecast[i+step] = original[i]*(1+beta*(avg_dict[i]/avg_dict[i-step]-1)-eir_dict[i])


	elif model_option==4:
		import statsmodels.api as sm
		import pandas
		f_log = open("pickle set\\logkorea_dict.p",'rb')
		log_price = pickle.load(f_log)
		f_log.close()
		if time_window != 0:
			log_price = change_window(log_price,time_window)

		vector={}
		for key in data_key:
			try:
				pi_vector = pi_dict[key]
				vector[key]=[log_price[key],pi_vector[0],pi_vector[1],pi_vector[2]]
			except:
				pass
		new_key = list(vector.keys())
		new_key.sort()
		training_vector={}
		for k in new_key[:-testnum]:
			training_vector[k] = vector[k]

		xraw=pandas.DataFrame(training_vector)
		x=np.asarray(xraw).transpose()

		model = sm.tsa.VAR(x)
		var_result = model.fit(12)

		total_list=[]
		for k in new_key:
			total_list.append(vector[k])

		start_point = min(new_key)
		fitted = {}
		for i in new_key[12:-testnum-step]:
			fitted[i+step] = np.exp(var_result.forecast(total_list[:i-start_point],step)[-1][0])

		forecast = {}
		for i in new_key[-testnum-step:-step]:
			forecast[i+step] = np.exp(var_result.forecast(total_list[:i-start_point],step)[-1][0])


	
	result = forecast_result()
	result.name = "Petroleum products spread model"
	result.info = "Option: "+option+", model option is "+str(model_option)
	result.step = step
	result.time_window=time_window
	result.option = option
	result.original = original
	result.fitted = fitted
	result.forecast = forecast
	result.training_num = len(list(fitted.keys()))
	result.testing_num = len(list(forecast.keys()))
	result.total_num = len(list(original.keys()))


	f_model = open("result set\\Product spread_1.p",'wb')
	pickle.dump(result,f_model)
	f_model.close()

	return result


def prepare_nochange():
	return

def model_nochange(step,testnum=50,time_window=0):
	original = kgasoline_dict

	if time_window != 0:
		original = change_window(original,time_window)

	data_key = list(original.keys())
	data_key.sort()

	fitted = {}
	for i in data_key[:-testnum-step]:
		fitted[i+step] = original[i]

	forecast = {}
	for i in data_key[-testnum-step:-step]:
		forecast[i+step] = original[i]


	result = forecast_result()
	result.name = "Nochange model"
	result.info = "Option: " + option
	result.step = step
	result.time_window=time_window
	result.option = option
	result.original = original
	result.fitted = fitted
	result.forecast = forecast
	result.training_num = len(list(fitted.keys()))
	result.testing_num = len(list(forecast.keys()))
	result.total_num = len(list(original.keys()))


	f_model = open("result set\\Nochange_1.p",'wb')
	pickle.dump(result,f_model)
	f_model.close()

	return result



######################################
# 	combination
######################################

def combination1(step,testnum=50,model_list=[1,2,3,4,5],model_option=[1,1,1,1],time_window=0,prepare=True):
	#simple average
	original = kgasoline_dict

	if time_window != 0:
		original = change_window(original,time_window)

	count = 0
	model_dict = {}

	if 1 in model_list:
		count += 1
		model_dict[1] = model_VAR(step=step,testnum=testnum,time_window=time_window,model_option=model_option[0],prepare=prepare)

	if 2 in model_list:
		count += 1
		model_dict[2] = model_industrial(step=step,testnum=testnum,time_window=time_window,model_option=model_option[1],prepare=prepare)

	if 3 in model_list:
		count += 1
		model_dict[3] = model_gasoline(step=step,testnum=testnum,time_window=time_window,model_option=model_option[2],prepare=prepare)

	if 4 in model_list:
		count += 1
		model_dict[4] = model_products(step=step,testnum=testnum,time_window=time_window,model_option=model_option[3],prepare=prepare)

	if 5 in model_list:
		count += 1
		model_dict[5] = model_nochange(step=step,testnum=testnum,time_window=time_window)


	keylist = original.keys()
	fitted = {}
	forecast = {}

	for key in keylist:
		sum_fit = 0
		sum_forecast = 0
		try:
			for i in model_list:
				model = model_dict[i]
				sum_fit += model.fitted[key]
			fitted[key] = sum_fit/count
		except:
			pass

		try:
			for i in model_list:
				model = model_dict[i]
				sum_forecast += model.forecast[key]
			forecast[key] = sum_forecast/count
		except:
			pass


	result = forecast_result()
	result.name = "Combination"
	result.info = "Average"
	result.step = step
	result.time_window=time_window
	result.option = option
	result.original = original
	result.fitted = fitted
	result.forecast = forecast
	result.training_num = len(list(fitted.keys()))
	result.testing_num = testnum
	result.total_num = len(list(original.keys()))

	f_save = open("result set\\Combination_1.p",'wb')
	pickle.dump(result,f_save)
	f_save.close()

	return result



##############################################
# 	Evaluating model
###############################################

def plot_ratio(model_num=0,model_option=1,combi_list=[1,2,3,4,5],combi_option=[1,1,1,1],testnum=50,time_window=0,prepare=False):
			# For individual model, change the model_num. For combination model, change the combi_list and let the model_num=0

	x=list(range(1,25))
	ys=[]
	ye=[]
	ye_rate=[]
	for step in range(24):
		if model_num==1:
			R = model_VAR(step=step+1,model_option=model_option, testnum=testnum, time_window=time_window, prepare=prepare)
		elif model_num==2:
			R = model_industrial(step=step+1,model_option=model_option, testnum=testnum, time_window=time_window, prepare=prepare)
		elif model_num==3:
			R = model_gasoline(step=step+1,model_option=model_option, testnum=testnum, time_window=time_window, prepare=prepare)
		elif model_num==4:
			R = model_products(step=step+1,model_option=model_option, testnum=testnum, time_window=time_window, prepare=prepare)
		elif model_num==0:
			R = combination1(step=step+1,model_list=combi_list,model_option=combi_option,testnum=testnum,time_window=time_window,prepare=prepare)
		else:
			print("Check your model number")
			return
		ys.append(R.success_ratio(print_option="quiet"))
		ye.append(R.mspe(print_option="quiet"))
		ye_rate.append(R.mspe_ratio(print_option="quiet"))


	import matplotlib.pyplot as plt

	plt.subplot(211)
	plt.plot(x,ys,'r',marker=".",label="success ratio")
	plt.hlines(0.5,x[0],x[-1],linestyles='dashdot')
	plt.ylabel('Success ratio')
	plt.legend(loc='upper left',fontsize='xx-small')


	ax1 = plt.subplot(212)
	ax1.plot(x,ye_rate, 'g', marker=".",label="mspe ratio")
	ax1.hlines(1,x[0],x[-1],linestyles='dashed')
	ax1.set_ylabel("MSPE ratio")

	ax2 = ax1.twinx()
	ax2.plot(x,ye, 'b', marker=".", label="mspe value")
	ax2.set_ylabel("MSPE")
	ax2.grid(False)

	ax1.set_xlabel("forecast horizon")
	ax1.legend(loc='upper left',fontsize='xx-small')
	ax2.legend(loc='upper center',fontsize='xx-small')
	plt.show()
	

	return


def check_model(model_num=0,model_option=1,combi_list=[1,2,3,4,5],combi_option=[1,1,1,1],testnum=50,time_window=0,prepare=False):
	for h in range(24):
		if model_num==0:
			R=combination1(step=h+1,model_list=combi_list,model_option=combi_option,testnum=testnum,time_window=time_window,prepare=prepare)
		
		elif model_num==1:
			R=model_VAR(step=h+1,testnum=testnum,time_window=time_window,model_option=model_option,prepare=prepare)
		elif model_num==2:
			R=model_industrial(step=h+1,testnum=testnum,time_window=time_window,model_option=model_option,prepare=prepare)
		elif model_num==3:
			R=model_gasoline(step=h+1,testnum=testnum,time_window=time_window,model_option=model_option,prepare=prepare)
		elif model_num==4:
			R=model_products(step=h+1,testnum=testnum,time_window=time_window,model_option=model_option,prepare=prepare)
		print("<step "+str(h+1)+">")
		print(str(R.success_ratio(print_option='quiet'))+"/"+str(R.mspe_ratio(print_option="quiet")))
		print("----------\n")
	return



#R=combination1(step=3,model_list=[1,2,3,5],time_window=0,prepare=False)
#R=model_VAR(step=3,testnum=50,time_window=0,model_option=1,prepare=False)
#R=model_industrial(step=3,testnum=50,time_window=0,model_option=1,prepare=False)
#R=model_gasoline(step=3,testnum=50,time_window=0,model_option=2,prepare=False)
#R=model_products(step=3,testnum=50,time_window=0,model_option=4,prepare=True)
#R=model_nochange(step=3,testnum=50,time_window=0,prepare=False)

#R.information()
#R.mspe_ratio()
#R.success_ratio()
#R.plot()

