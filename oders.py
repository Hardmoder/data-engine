import pandas as pd
from efficient_apriori import apriori


data=pd.read_csv("订单表.csv",encoding='gbk')

transactions=[]
temp_index=0

oders_series=data.set_index('客户ID')['产品名称']
oders_series=oders_series.sort_index()  

for i, m in oders_series.items():
		if i != temp_index:
			temp_set = set()
			temp_index = i
			temp_set.add(m)
			transactions.append(temp_set)
		else:
			temp_set.add(m)

itemsets, rules = apriori(transactions, min_support=0.1,  min_confidence=0.5)
print('频繁项集：', itemsets)
print('关联规则：', rules)
