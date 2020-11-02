import numpy as np

import pandas as pd

from datetime import date

def doTheCalculation(data):

	data['dayofyear']=(data['BILLING_DATE']-

    	data['BILLING_DATE'].apply(lambda x: date(x.year,1,1))

    	.astype('datetime64[ns]')).apply(lambda x: x.days)

	X = np.array(data[['SALES_OFFICE', 'REGION_CODE', 'MATERIAL', 'MILL_CODE', 'SECTION_CODE', 'MATERIAL_SIZE','MATERIAL_GRADE_CODE', 
                    'CUSTOMER_CODE','DIVISION', 'PANDEMIC_IND',
                    
                    
                    
                    ]])

	return X
