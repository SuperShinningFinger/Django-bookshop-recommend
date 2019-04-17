#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re
import numpy as np
import math
import copy
import seaborn as sns
from math import radians, cos, sin, asin, sqrt  
import matplotlib  
import matplotlib.pyplot as plt  
from scipy import stats 
from matplotlib.ticker import MultipleLocator, FormatStrFormatter 
from kdlc.User_Similarity import Users_Poi,POISelect,Poi
from functools import reduce

filename = 'Foursquare_poi_position.txt' # txt文件和当前脚本在同一目录下，所以不用写具体路径
Poi_position=[]
User = []
Longitude = []
Latitude = []
with open(filename, 'r') as file_to_read:
	while True:
		lines = file_to_read.readline() # 整行读取数据
		if not lines:
			break
		poi_position_tmp = re.split(r'\s+',lines)[0:3]
		user_tmp= re.split(r'\s+',lines)[0]# 将整行数据分割处理，如果分割符是空格，括号里就不用传入参数，如果是逗号， 则传入‘，'字符。
		longitude_tmp= re.split(r'\s+',lines)[1]
		latitude_tmp= re.split(r'\s+',lines)[2]
		Poi_position.append(poi_position_tmp)
		User.append(int(user_tmp))  # 添加新读取的数据
		Longitude.append(float(longitude_tmp))
		Latitude.append(float(latitude_tmp))		
		pass
#print Poi_position[0:5],type(Poi_position),type(Poi_position[0][1]),type(Poi_position[0][2])
filename1 = 'Foursquare_poi_position.txt' #''PoiSet_User1083.txt' # txt文件和当前脚本在同一目录下，所以不用写具体路径
AppointedUser = []
with open(filename1, 'r') as file_to_read:
    while True:
        lines = file_to_read.readline()  # 整行读取数据
        if not lines:
            break

        #AppointedUser.append(int(re.split(r'\n', lines)[0]))
        AppointedUser.append(int(re.split('	', lines)[0]))
        pass
#len(AppointedUser)=342 		

AppointedUser = {}.fromkeys(AppointedUser).keys()#去除list中重复的元素，不改变原来的顺序
sorted(AppointedUser)
AppointedUser=list(AppointedUser)
#Appointeduser = copy.deepcopy(AppointedUser)
#print AppointedUser[0:20]#,Poi_position[0:10],type(AppointedUser)
AppointedPoi = []#存放根据AppointedUser指定的user的POI以及其地理信息

for i in Poi_position[:5596]:
  if not AppointedUser:
       break
  elif int(i[0]) != AppointedUser[-1]:
       AppointedUser.pop()
       AppointedPoi.append(i)
  else:
       continue
#print AppointedPoi[0:10]
POISelectTmp = []
POISELECT = []
for i in POISelect:
  POISelectTmp.append(int(i))
#print POISelectTmp[0]
POISelectTmp.sort(reverse = True)
for i in Poi_position[:5596]:
  if not POISelectTmp:
       break
  elif int(i[0]) == POISelectTmp[-1]:
       POISelectTmp.pop()
       POISELECT.append(i)
  else:
       continue
#print POISELECT[:10]

"""
Users_POI = []
for i in range(10):
  temp = Users_Poi[i]
  Users_POI.append(temp)
#print Users_POI[0]
UserPOITemp1 = []
USERPOI = []
for i in range(10):
  for j in Users_POI[i]:
	UserPOITemp1.append(int(j))
  USERPOI.append({}.fromkeys(UserPOITemp1).keys())
#In spite of you and me and the whole silly world going to pieces around us, I love you.

for i in range(10):
  USERPOI[i].sort(reverse = True)
#print USERPOI[0]
USERPOICopy = copy.deepcopy(USERPOI)
count = 0
AppointedPoiLocationSet = []
while count < len(USERPOICopy):
  AppointedPoiLocationSet.append([])
  for i in Poi_position[:5596]:
	if not USERPOICopy[count]:
		break
	elif int(i[0]) == int(USERPOICopy[count][-1]):
		USERPOICopy[count].pop()
		AppointedPoiLocationSet[count].append(i)
		
	else:
		continue
  count += 1
#print AppointedPoiLocationSet
#print USERPOI
"""

'''
Poi_positiontmp1=copy.deepcopy(AppointedPoi)#深复制两个原始数据AppointedPoi，然后错位相减
Poi_positiontmp2=copy.deepcopy(AppointedPoi)

Poi_positiontmp1.pop()
Poi_positiontmp2.pop(0)
Poi_positiontmp3 = []
'''
#============================================================================
#函数名称：haversine(lon1, lat1, lon2, lat2)
#函数返回：POI的距离
#参数说明：两个POI的经纬度，经度1，纬度1，经度2，纬度2
#功能概要：返回距离，单位KM
#============================================================================
def haversine(lon1, lat1, lon2, lat2): # 经度1，纬度1，经度2，纬度2 （十进制度数）  
    """ 
    Calculate the great circle distance between two points  
    on the earth (specified in decimal degrees) 
    """  
    # 将十进制度数转化为弧度  
    lon1, lat1, lon2, lat2 = map(radians, [float(lon1), float(lat1), float(lon2), float(lat2)])  
  
    # haversine公式  
    dlon = lon2 - lon1   
    dlat = lat2 - lat1   
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2  
    c = 2 * asin(sqrt(a))   
    r = 6371 # 地球平均半径，单位为公里  
    return c * r#单位是公里

'''
for i in Poi_positiontmp1:
 for j in Poi_positiontmp2:
  Poi_positiontmp3.append('%.4f' % haversine(j[1],j[2],i[1],i[2]))
Poi_positiontmp4 = []
for i in Poi_positiontmp3:
 Poi_positiontmp4.append(int(float(i)))
Poi_positiontmp4.sort()
#print Poi_positiontmp4[0:3000]#len(Poi_positiontmp4) == 16384
# Poi_positiontmp3_np = np.array(Poi_positiontmp3, dtype = float)
# Poi_positiontmp3_np.sort()
# print  Poi_positiontmp3_np[0:10]
Poi_position_NumCount=[]#存放各个距离长度的频数
Poi_positiontmp5 = sorted(set(Poi_positiontmp4),key = Poi_positiontmp4.index) #使用list类的sort方法实现list删除重复元素并保持原来的顺序
#print Poi_positiontmp5
#处理一下距离分布
for i in Poi_positiontmp5:
   j = Poi_positiontmp4.count(i)
   if Poi_positiontmp5.index(i) < 20:
     Poi_position_NumCount.append(j)
   elif Poi_positiontmp5.index(i) < 60:
      #Poi_position_NumCount.append(j/(0.034*Poi_positiontmp5.index(i)))
	  Poi_position_NumCount.append(j/(0.15*Poi_positiontmp5.index(i)))
   elif Poi_positiontmp5.index(i) < 150:
      #Poi_position_NumCount.append(j/(0.034*Poi_positiontmp5.index(i)))
	  Poi_position_NumCount.append(j/(0.13*Poi_positiontmp5.index(i)))
   elif Poi_positiontmp5.index(i) < 2000:
     Poi_position_NumCount.append(j/(0.00425*Poi_positiontmp5.index(i)*Poi_positiontmp5.index(i)))
   else:
      Poi_position_NumCount.append(j/(2e-6*Poi_positiontmp5.index(i)*Poi_positiontmp5.index(i)*Poi_positiontmp5.index(i)))
Poi_position_NumCountTmp=[]
Poi_position_NumCountTmp1 = []
Poi_position_NumCountTmp2 = []
Poi_positiontmp6 = []
Poi_positiontmp7 = []
Sum = sum(Poi_position_NumCount)
for i in Poi_position_NumCount:
   Poi_position_NumCountTmp2.append(i/float(Sum))
#print len(Poi_position_NumCountTmp2)=2688
for i in Poi_position_NumCountTmp2:
    Poi_position_NumCountTmp1.append(math.log(i,2))
#print Sumtmp
Poi_positiontmp5[0]=1
for i in Poi_positiontmp5:
   Poi_positiontmp6.append(math.log(i,2))
'''


#print Poi_positiontmp6
#print Poi_position_NumCount[:10],Poi_positiontmp4.count(0),len(Poi_position_NumCount),len(Poi_positiontmp5)

#sns.distplot(Poi_positiontmp6,Poi_position_NumCountTmp1, rug=True, hist=False)
#plt.hist(Poi_positiontmp5,bins=2688, color='steelblue', normed=True)
#sns.distplot(Poi_positiontmp4,kde=False, fit=stats.expon)
#2.散点图,只是用用scat函数来调用即可
# plt.scatter(Poi_positiontmp6,Poi_position_NumCountTmp2)
# plt.show()
# def get_CDF(numList,A):
    # print "total number of numList %d"%len(numList)
    # numArray = np.asarray(numList)
    # dx = .01
    # bins_array = np.arange(-0.5,1.5,dx)
    # B = np.asarray(A)
    # hist, bin_edges = np.histogram(numArray, bins=B, normed=False)
    # cdf = np.cumsum(hist)
    # cdf = cdf/float(hist.sum())
  
    # bins_list = bins_array[1:]
    # return (bins_list, cdf)

# get_CDF(Poi_position_NumCount,Poi_positiontmp5)
# print A,B
#---------------------------------------------------
# xmajorLocator   = MultipleLocator(1) #将x主刻度标签设置为20的倍数
# xmajorFormatter = FormatStrFormatter('%5.1f') #设置x轴标签文本的格式
# xminorLocator   = MultipleLocator(0.1) #将x轴次刻度标签设置为5的倍数


# ymajorLocator   = MultipleLocator(0.01) #将y轴主刻度标签设置为0.5的倍数
# ymajorFormatter = FormatStrFormatter('%1.2f') #设置y轴标签文本的格式
# yminorLocator   = MultipleLocator(0.002) #将此y轴次刻度标签设置为0.1的倍数

 

# ax = plt.subplot(111) #注意:一般都在ax中设置,不再plot中设置
# plt.plot(Poi_positiontmp6,Poi_position_NumCountTmp2)

# #设置主刻度标签的位置,标签文本的格式
# ax.xaxis.set_major_locator(xmajorLocator)
# ax.xaxis.set_major_formatter(xmajorFormatter)

# ax.yaxis.set_major_locator(ymajorLocator)
# ax.yaxis.set_major_formatter(ymajorFormatter)

# #显示次刻度标签的位置,没有标签文本
# ax.xaxis.set_minor_locator(xminorLocator)
# ax.yaxis.set_minor_locator(yminorLocator)

# ax.xaxis.grid(True, which='major') #x坐标轴的网格使用主刻度
# ax.yaxis.grid(True, which='minor') #y坐标轴的网格使用次刻度

# plt.show()


# ##log坐标轴,画出距离分布的散点图
# plt.figure(1)
# plt.scatter(Poi_positiontmp5,Poi_position_NumCountTmp2)
# plt.xscale('log')
# plt.yscale('log')
# plt.xlim(1e0, 1e4)
# plt.ylim(1e-7, 1e0)
# plt.title('log')
# plt.grid(True)
# ##线性坐标轴
# # # plt.figure(2)
# # # plt.plot(Poi_positiontmp5,Poi_position_NumCountTmp2)
# # # plt.yscale('linear')
# # # plt.title('linear')
# # # plt.grid(True)

# plt.savefig('Distance_distribute.pdf', dpi=1024)
# plt.show()
#============================================================================
#函数名称：linefit(x , y)
#函数返回：斜率，截距，准确率
#参数说明：x,y
#功能概要：最小二乘拟合
#============================================================================
def linefit(x , y):##线性最小二乘估计
    N = float(len(x))
    sx,sy,sxx,syy,sxy=0,0,0,0,0
    for i in range(0,int(N)):
        sx  += x[i]
        sy  += y[i]
        sxx += x[i]*x[i]
        syy += y[i]*y[i]
        sxy += x[i]*y[i]
    a = (sy*sx/N -sxy)/( sx*sx/N -sxx)
    b = (sy - a*sx)/N
    r = abs(sy*sx/N-sxy)/math.sqrt((sxx-sx*sx/N)*(syy-sy*sy/N))
    return a,b,r
#a,b,r=linefit(Poi_positiontmp6[:120],Poi_position_NumCountTmp1[:120])
#print u'拟合结果'+': y = %10.5f x + %10.5f , r=%10.5f' % (a,b,r) 


'''
#拟合结果：   y =   -1.41084 x +   -0.67804 , r=   0.88617(10为底)y =   -1.41084 x +   -2.25238 , r=   0.88617(2为底)
#与上面最小而成比较，结果是一致的，这二者都可以用，多项式拟合解决一切问题
'''
#########可以利用多项式拟合的函数，degree=1也可以拟合
# Linear_fitting = np.polyfit(Poi_positiontmp6[:120], Poi_position_NumCountTmp1[:120], 1)  #一次多项式拟合，相当于线性拟合
# Linear_fitting_polynomia = np.poly1d(Linear_fitting)
# print Linear_fitting            #Linear_fitting =        [-1.41083995 -0.67803519](10为底)[-1.41083995 -2.25238416](2为底)
# print Linear_fitting_polynomia  #Linear_fitting_polynomia =  -1.411 x - 0.678，-1.411 x - 2.252
#画出线性拟合函数的图像，做一下比较 
# plt.figure(2)
# yy = [a * x + b for x in Poi_positiontmp6]
# plt.plot(Poi_positiontmp6,yy,'r-')
# plt.title(u'linear_fit')
# plt.grid(True)
# plt.show()
#============================================================================
#函数名称：multiplication(x,y)
#函数返回：乘积
#参数说明：因子x,因子y
#功能概要：计算两个数的乘积
#============================================================================
def multiplication(x,y):
  return float(x)*float(y)
def max(x,y):
  x = float(x)
  y = float(y)
  if x > y:
    return x
  else:
    return y
def mean(x,y):
  return (float(x)+float(y))/2
#============================================================================
#函数名称：Probability(distance)
#函数返回：根据地理位置推荐概率top10
#参数说明：给定用户，用户签到过的POI的集合L，POI集合（给定Poi）,回归得到的指数分布参数a,b
#功能概要：计算地理位置推荐概率
#============================================================================
def Probability(L, poi_set,appointeduser = 1083,  a = 0.2099,b = -1.4108):
  A = []
  B = []
  C = []
  for  i in poi_set:
    A = []
    for j in L:
        x = ('%.4f' % haversine(j[1],j[2],i[1],i[2]))#计算距离
        X = float(x)
        if X > 0:
         y = a * math.pow(X,b)
        if y > 0 and y < 1:
         A.append('%.4f' %y)
	
    B.append(A)
	


  for i in B:

    C.append('%.4f' %reduce(mean,i))
  #print type(y),type(B[0][0])
  return C  
  pass

'''
def CalcutePr(user, L, targetPOI = [40.7197,-74.0025],  a = 0.4099,b = -1.4108):
  A = []
  B = []
  C = []
  for  i in poi_set:
	A = []
	for j in L:
		x = ('%.4f' % haversine(j[1],j[2],i[1],i[2]))#计算距离
		X = float(x)
		
		if X > 0:
		 y = a * math.pow(X,b)
		if y > 0 and y < 1:
		 A.append('%.4f' %y)
	
	B.append(A)
	
    
  for i in B:
	
	C.append('%.4f' %reduce(mean,i))
  #print type(y),type(B[0][0])
  return C  
  pass
  
'''

#GeographicalList = []#存储推荐给用户的所有POI的概率
# for i in range(10):
  # GeographicalList.append(Probability(AppointedPoiLocationSet[i],Poi_position[:5596]))

print(AppointedPoi)
GeographicalList = Probability(AppointedPoi,POISELECT)

GeographicalListTOP10 = sorted(GeographicalList)[0:10]
GeographicalTOP10Index = [i for j in GeographicalListTOP10 for i, x in enumerate(GeographicalList) if x == j][0:10]
print (GeographicalListTOP10,GeographicalTOP10Index)
