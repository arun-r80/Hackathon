import configparser
import os
#file to write configuration values
#write configuration values for min and max salary
config = configparser.RawConfigParser()

#config values for master data
config.add_section('mastercustomerdescription')
config.set('mastercustomerdescription','minsalary','10000')
config.set('mastercustomerdescription','maxsalary','7500000')

#config values for transaction data
config.add_section('transactiondata')
config.set('transactiondata','transactionstartdate','42370')
config.set('transactiondata','transactionenddate','42825')
config.set('transactiondata','mintransactionamount','1')
config.set('transactiondata','maxtransactionamount','10000000')
config.set('transactiondata','customeridmin','1')
config.set('transactiondata','customeridmax','99')
config.set('transactiondata','maxtransactioncount','50000')


fconfig = open(os.path.join('config','configurationfile.ini'),'w')
config.write(fconfig)
fconfig.close()