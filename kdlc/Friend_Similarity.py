#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re
import numpy as np
import copy
from kdlc.User_Similarity import POISelect
#============================================================================
#函数名称：Readtext(filename)
#函数返回：两个list包含文件的两列
#参数说明：文件名称
#功能概要：单行读取文件
#============================================================================
def Readtext(filename):
 filename = filename # txt文件和当前脚本在同一目录下，所以不用写具体路径
 User1 = []
 User2 = []
 count1 = 0
 with open(filename, 'r') as file_to_read:
    while True:
        lines = file_to_read.readline() # 整行读取数据
        if not lines:
            break
            pass
        #lines=lines.replace(r'^\s*/g','')

        #lines=lines.replace(r'\s*$/g','')#去除结尾空格

        #lines=lines.replace(r'\s{2,}/g'," ")#多个空格合并成一个

        user1_tmp= re.split(r'\s+',lines)[0]# 将整行数据分割处理，如果分割符是空格，括号里就不用传入参数，如果是逗号， 则传入‘，'字符。
        user2_tmp= re.split(r'\s+',lines)[1]
        User1.append(user1_tmp)  # 添加新读取的数据
        User2.append(user2_tmp)
        count1 += 1
        pass
 return  User1,User2
User1,User2=Readtext('FriendTies.txt')
Utemp1=User1
Utemp2=User2
User1=Utemp1+Utemp2
User2=Utemp2+Utemp1
# print User1[0:10]
# print User2[0:10]
'''可以直接生成好友关系矩阵，但是都取1好像接下去也不是那么好处理，所以还是用二维list
Friendship_Matrix = [([0] * 2322) for i in range(2322)]
for (i,j) in zip(User1,User2):
  Friendship_Matrix[int(i)][int(j)] = 1
''' 
User,Poi=Readtext('train.txt')
User = [(re.split(r'[a-zA-Z\_]+',i)[1]) for i in User]
Poi= [(re.split(r'[a-zA-Z\_]+',i)[1]) for i in Poi]
POI = sorted(set(Poi),key = Poi.index)
# print User[0:10]
# print Poi[0:10]

#============================================================================
#函数名称：GetFriendSet(user)
#函数返回：User的好友集合
#参数说明：User
#功能概要：返回一位用户的所有好友集合
#============================================================================
def GetFriendSet(user):
 GetFriendIndex = [i for i,x in enumerate(User1) if x == user]	
 if len(GetFriendIndex):
  GetFriend = [User2[int(i)] for i in GetFriendIndex] 
 else:
  GetFriend=[]
 return GetFriend


#下面计算每一位用户的好友集合，首先将User处理成不重复list并且排序不能变化,不然不能与Poi集合对应
#USER = []
#[USER.append(i) for i in User if not i in USER]#使用遍历的方法祛除重复元素并保持原来的顺序
USER = sorted(set(User),key = User.index) #使用list类的sort方法
#print USER[0]
Friend_List=[]
for i in USER:
 Friend_List.append(GetFriendSet(i))
#print Friend_List[0:10]

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
Poi_List = []
for i in USER:
 Poi_List.append(GetPoiSet(i))
# print Poi_List[0]
# fl=open('PoiSet_User1083.txt', 'w')#将Poi_List[0]的POI数据存起来作地理位置图
# for i in Poi_List[0]:
    # fl.write(i)
    # fl.write("\n")
# fl.close()

#============================================================================
#函数名称：FriendSimilarity(user1F=[],user2F=[],user1L=[],user2L=[],y=0.9)
#函数返回：User的签到过的POI集合
#参数说明：User1的好友集合，User2的好友集合，User1的POI集合，User2的POI集合
#功能概要：返回好友相似度
#============================================================================
def FriendSimilarity(user1F=[],user2F=[],user1L=[],user2L=[],y=0.9):#返回好友相似度
 if (user1F == [] and user2F ==[]) or  (user1L == [] and user2L ==[]):
  return 0
 else:
  coefficient_F = float(len(set(user1F)&set(user2F)))/float(len(set(user1F)|set(user2F)))
  coefficient_L= float(len(set(user1L)&set(user2L)))/float(len(set(user1L)|set(user2L)))
 return coefficient_F * y + coefficient_L * (1.0-y)
#print [FriendSimilarity(Friend_List[0],Friend_List[1],Poi_List[0],Poi_List[1])]

Friend_List1=copy.deepcopy(Friend_List)#深复制两个原始列表Friend_List，然后错位
Friend_List2=copy.deepcopy(Friend_List)
Friend_List1.pop()
Friend_List2.pop(0)
Poi_List1=copy.deepcopy(Poi_List)#深复制两个原始列表Poi_List，然后错位
Poi_List2=copy.deepcopy(Poi_List)
Poi_List1.pop()
Poi_List2.pop(0)
#print len(Friend_List1),len(Poi_List1),len(Poi_List)
def CalcutePr(friend_List1,friend_List2,poi_List1,poi_List2,targetPOI = '1'): 
  friendSimilarity_List = []
  for i in range(1):#len(Friend_List1)=2320
    sum1 = 0.0
    sum2 = 0.0
    for j in range(len(Friend_List2)):#事实上只需要计算好友集合里的就可以
        temp = FriendSimilarity(friend_List1[i],friend_List2[j],poi_List1[i],poi_List2[j])
        if targetPOI in poi_List2[j]:
            c = 1
        else:
            c = 0
        sum1 = sum1 + c * temp
        sum2 = sum2 + temp
        if sum2<0.000001:
          similarity = 0
        else:
          similarity = sum1/sum2
	#friendSimilarity_List.append('%.4f' %similarity)
  return similarity#friendSimilarity_List

#FriendSimilarity_List = CalcutePr(Friend_List1,Friend_List2,Poi_List1,Poi_List2)
Pr_Basedon_FriendSimilarity = []
for i in POISelect:
  Pr_Basedon_FriendSimilarity.append('%.4f' %CalcutePr(Friend_List1,Friend_List2,Poi_List1,Poi_List2,i))
#print  Pr_Basedon_FriendSimilarity[:10]
# dictionary = dict(zip(POI[:10], Pr_Basedon_FriendSimilarity))
# fl=open('Pr_Basedon_UserSimilarity_POI1_10.txt', 'w')
# for i in Pr_Basedon_UserSimilarity:
    # fl.writelines(str(i))
    # fl.write("\n")
# fl.close()
#FriendSimilarityTOP10 = sorted(FriendSimilarity_List, cmp=None, key=None, reverse=True)[0:10]
#FriendSimilarityTOP10Index = [i for j in FriendSimilarityTOP10 for i, x in enumerate(FriendSimilarity_List) if x == j][0:10]
#print FriendSimilarityTOP10,FriendSimilarityTOP10Index








