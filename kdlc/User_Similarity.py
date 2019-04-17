#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re
import numpy as np
import math
import copy
filename = 'train.txt' # txt文件和当前脚本在同一目录下，所以不用写具体路径
User = []
Poi = []
count = 0
with open(filename, 'r') as file_to_read:
    while True:
        lines = file_to_read.readline() # 整行读取数据
        if not lines:
            break
            pass
        #lines=lines.replace(r'^\s*/g','')

        #lines=lines.replace(r'\s*$/g','')#去除结尾空格

        #lines=lines.replace(r'\s{2,}/g'," ")#多个空格合并成一个

        user_tmp= re.split(r'\s+',lines)[0]# 将整行数据分割处理，如果分割符是空格，括号里就不用传入参数，如果是逗号， 则传入‘，'字符。
        poi_tmp= re.split(r'\s+',lines)[1]
        User.append(user_tmp)  # 添加新读取的数据
        Poi.append(poi_tmp)
        count += 1
        pass
     #User = np.array(User) # 将数据从list类型转换为array类型。


#将User，Poi数据改为纯数字ID
User = [(re.split(r'[a-zA-Z\_]+',i)[1]) for i in User]
Poi= [(re.split(r'[a-zA-Z\_]+',i)[1]) for i in Poi]
#============================================================================
#函数名称：GetPoiSet(user)
#函数返回：User的签到过的POI集合
#参数说明：User
#功能概要：返回一位用户的所有签到POI集合
#============================================================================
def GetPoiSet(user):
  GetPOIIndex = [i for i,x in enumerate(User) if x == user]
  GetPoi = [Poi[int(i)] for i in GetPOIIndex] 
  return GetPoi

USER = sorted(set(User),key = User.index) #使用list类的sort方法去除User中的重复用户
POI = sorted(set(Poi),key = Poi.index)
Users_Poi = []#用来存每一位用户的Poi列表
for i in USER:
  Users_Poi.append(GetPoiSet(i))
#print len(USER)

#print len(UsersPoi1)
Users_Poi_60 = []
for i in Users_Poi:
  if len(set(i)) > 60:
    Users_Poi_60.append(sorted(set(i),key = i.index))
#print Users_Poi_60[:10]#len(Users_Poi_60) = 422
POISelect = []
for i in range(10):
  POISelect = set(POISelect)|set(Users_Poi_60[i])
#print len(POISelect)

# USER=set(User)#2321个User
# POI=set(Poi)#5596个POI
UsersPoi1 = copy.deepcopy(Users_Poi_60)#深复制两个原始列表Users_Poi，然后错位
UsersPoi2 = copy.deepcopy(Users_Poi_60)
UsersPoi1.pop()
UsersPoi2.pop(0)


#Check_in_Matrix = np.zeros((2321,5596)) #这样生成初始签到矩阵会有问题
'''生成签到矩阵'''
# Check_in_Matrix = [([0] * 5596) for i in range(2321)]
# for (i,j) in zip(User,POI):
  # Check_in_Matrix[int(i)][int(j)] = 1

#============================================================================
#函数名称：cos(vector1,vector2)
#函数返回：余弦相似度
#参数说明：两个向量
#功能概要：返回两位用户签到向量的余弦相似度
#============================================================================
def Cos(vector1,vector2):  #求余弦相似度
    dot_product = 0.0;  
    normA = 0.0;  
    normB = 0.0;  
    for a,b in zip(vector1,vector2):  
        dot_product += a*b  
        normA += a**2  
        normB += b**2  
    if normA == 0.0 or normB==0.0:  
        return None  
    else:  
        return dot_product / ((normA*normB)**0.5) 

#============================================================================
#函数名称：calcuteSimilar(series1,series2)
#函数返回：两个用户原始签到数据list的相似度
#参数说明：两个签到list
#功能概要：返回两位用户签到数据的余弦相似度
#============================================================================
def CalcuteSimilar(series1,series2):  
    '''
    计算余弦相似度 
    :param data1: 数据集1 Series 
    :param data2: 数据集2 Series 
    :return: 相似度 
    '''
    unionLen = len(set(series1) & set(series2))  
    if unionLen == 0: return 0.0  
    product = len(series1) * len(series2)  
    similarity = unionLen / math.sqrt(product)  
    return similarity

def CalcutePr(usersPoi1,usersPoi2,targetPOI = '1'):
  userSimilarity_List = []
  for i in range(1):#len(usersPoi1)=2320
    sum1 = 0.0
    sum2 = 0.0
    for j in range(len(usersPoi2)):
        temp = CalcuteSimilar(usersPoi1[i],usersPoi2[j])
        if targetPOI in usersPoi2[j]:
            c = 1
        else:
            c = 0
        sum1 = sum1 + c * temp
        sum2 = sum2 + temp
        if sum2<0.000001:
          similarity = 0
        else:
          similarity = sum1/sum2
    #userSimilarity_List.append('%.4f' %similarity)
  return (similarity)#userSimilarity_List
#UserSimilarity_List = CalcutePr(UsersPoi1,UsersPoi2)#根据用户相似度推荐POIID = 1给所有用户的概率
Pr_Basedon_UserSimilarity = []
for i in POISelect:
  Pr_Basedon_UserSimilarity.append('%.4f' %CalcutePr(UsersPoi1,UsersPoi2,i))
#print Pr_Basedon_UserSimilarity
# dictionary = dict(zip(POI[:10], Pr_Basedon_UserSimilarity))
# fl=open('Pr_Basedon_UserSimilarity_POI1_10.txt', 'w')
# for i in Pr_Basedon_UserSimilarity:
    # fl.writelines(str(i))
    # fl.write("\n")
# fl.close()


#UserSimiliarity = ['%.4f' %Cos(Check_in_Matrix[0],Check_in_Matrix[i]) for i in range(2321)]
#USERSimiliarityTOP10 = sorted(set(UserSimilarity_List),reverse = True)[1:11] #使用list类的sort方法去除重复元素
#UserSimiliarityTOP10Index = [i for j in USERSimiliarityTOP10 for i, x in enumerate(UserSimilarity_List) if x == j][0:10]
#print USERSimiliarityTOP10
#print UserSimiliarityTOP10Index

   # pass
# fl=open('user.txt', 'w')
# for i in User:
    # fl.write(i)
    # fl.write("\n")
# fl.close()	
# f2=open('poi.txt', 'w')
# for i in POI:
    # f2.write(i)
    # f2.write("\n")
# f2.close()