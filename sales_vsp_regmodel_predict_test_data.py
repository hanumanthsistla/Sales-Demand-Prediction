#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 13:43:42 2020
Input Parameters for Model:

BILLING_DATE
SALES_OFFICE
REGION_CODE
MATERIAL
MILL_CODE
SECTION_CODE
MATERIAL_SIZE
MATERIAL_GRADE_CODE
CUSTOMER_CODE
DIVISION
PANDEMIC_IND

Predict : Aales volume (Tons)

@author: hduser
"""

import requests, json

# url = '[http://localhost:8000/makecalc/](http://localhost:8000/makecalc/)'

# url = 'http://localhost:8000/makecalc/' 

url = 'http://erp-adm-119540:8000/makecalc/' 


text = json.dumps(
    
    
    {"1":{
        
        "BILLING_DATE":                   "01/01/20",
        "SALES_OFFICE":                    9940,
        "REGION_CODE":                     9900,  
        "MATERIAL":                        615100716,
        "MILL_CODE":                       61,
        "SECTION_CODE":                    5,
        "MATERIAL_SIZE":                   20020,
        "MATERIAL_GRADE_CODE":             1007,
        "CUSTOMER_CODE":                   9007806,
        "DIVISION":                        31,
        "PANDEMIC_IND":                    1
        
    
     
    },
    
   "2":{
    
        
        
        "BILLING_DATE":                   "01/01/20",
        "SALES_OFFICE":                    9910,
        "REGION_CODE":                     9900,  
        "MATERIAL":                        161200012800301,
        "MILL_CODE":                       16,
        "SECTION_CODE":                    12,
        "MATERIAL_SIZE":                   12,
        "MATERIAL_GRADE_CODE":             8003,
        "CUSTOMER_CODE":                   9002301,
        "DIVISION":                        34,
        "PANDEMIC_IND":                    1
        
        

    },
    
    
    "3":{
    
        
        "BILLING_DATE":                   "28/05/20",
        "SALES_OFFICE":                    9530,
        "REGION_CODE":                     9500,  
        "MATERIAL":                        171610050130200,
        "MILL_CODE":                       17,
        "SECTION_CODE":                    16,
        "MATERIAL_SIZE":                   10050,
        "MATERIAL_GRADE_CODE":             1302,
        "CUSTOMER_CODE":                   9002152,
        "DIVISION":                        35,
        "PANDEMIC_IND":                    1
        
    },
     
     
    "4":{
    
        
        "BILLING_DATE":                   "18/09/20",
        "SALES_OFFICE":                    9720,
        "REGION_CODE":                     9700,  
        "MATERIAL":                        171407777400600,
        "MILL_CODE":                       17,
        "SECTION_CODE":                    14,
        "MATERIAL_SIZE":                   7777,
        "MATERIAL_GRADE_CODE":             4006,
        "CUSTOMER_CODE":                   9018548,
        "DIVISION":                        36,
        "PANDEMIC_IND":                    1
        
    },
    
     
    "5":{
    
        
        "BILLING_DATE":                   "05/10/20",
        "SALES_OFFICE":                    9160,
        "REGION_CODE":                     9100,  
        "MATERIAL":                        461800007530201,
        "MILL_CODE":                       46,
        "SECTION_CODE":                    18,
        "MATERIAL_SIZE":                   7,
        "MATERIAL_GRADE_CODE":             5301,
        "CUSTOMER_CODE":                   9004666,
        "DIVISION":                        32,
        "PANDEMIC_IND":                    1
        
    }
     
    })

headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
r = requests.post(url, data=text, headers=headers)
print('\nPredicted VSP Sales Volumes in Tons for the above given Data:')
print('\n---------------------------------------------------------------------')

print(r,r.text)

