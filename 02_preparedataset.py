#create a sample file

#import header
import csv
import random
from random import randint, uniform
import configparser
import os

#create customer master data file
#open a file 
file = open(os.path.join('Dataset','customermaster.dat'),'a')
fwriter = csv.writer(file,delimiter=',')

#read from config file
cfg = configparser.RawConfigParser()
cfg.read(os.path.join('config','configurationfile.ini'))

minsalarya = cfg.getint('mastercustomerdescription','minsalary')
maxsalarya = cfg.getint('mastercustomerdescription','maxsalary')
custidmin = cfg.getint('transactiondata','customeridmin')
custidmax = cfg.getint('transactiondata','customeridmax')
maxtransaction = cfg.getint('transactiondata','maxtransactioncount')
mintransamount = cfg.getint('transactiondata','mintransactionamount')
maxtransamount = cfg.getint('transactiondata','maxtransactionamount')
transstartdate = cfg.getint('transactiondata','transactionstartdate')
transenddate = cfg.getint('transactiondata','transactionenddate')

for i in range(custidmin,custidmax):
	customername = 'a'
	customersalary = randint(minsalarya,maxsalarya)
	customerid = i
	customerrecord = customername  + ' ' + str(customersalary) +  ' '+str(customerid)
	fwriter.writerow(customerrecord.split())

file.close()

#create transaction file
filetrans=open(os.path.join('Dataset','transactiondata.dat'),'a')
ftransactionwriter = csv.writer(filetrans,delimiter=',')

TRANSSTATUS=['DONE','COMPLETED']

for i in range(1,maxtransaction):	
	custid = randint(custidmin,custidmax-1)
	debitorcredit = int(round(random.uniform(0.0,0.99)))
	transamount = randint(mintransamount,maxtransamount)
	transdate=randint(transstartdate,transenddate)
	transstatus = int(round(random.uniform(0.0,0.99)))
	
	transrow = str(custid) + ' ' + str(debitorcredit) + ' ' + str(transamount) + ' ' + str(transdate) + ' '+ str(transstatus)
	ftransactionwriter.writerow(transrow.split())
	
filetrans.close()

################################
	
	
