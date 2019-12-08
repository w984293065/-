
import numpy as np
import config as cf
import time
import copy


# 生成超级顾客

def superCustomer(i, point3):
    # 用来存放超级顾客信息
    smallSup = []
    bigSup = []

    # 根据该划分值求三级网点普通和大件的量
    for j in range(cf.numOfPoint3):
        # 求普通件的量
        sumnum = 0
        for t in range(i + 1):
            sumnum = sumnum + point3[j].P[t][0]
        point3[j].small = sumnum
        # 求大件的量
        sumnum = 0
        for t in range(5 - i):
            sumnum = sumnum + point3[j].P[5 - t][0]
        point3[j].big = sumnum
        # print("i=%d small=%d big=%d"%(j,point3[j].small,point3[j].big))
    # 生成普通件超级顾客，用V2进行运输
    pointleft = np.zeros(cf.numOfPoint3)  # 点被使用与否的标志，0表示未划分，1表示划分
    car = cf.car_loading[1][2 * i]
    ##开始模拟生成
    while (isAllUsed(pointleft) == 0):
        oneSuperCustomer = []  # 一个超级顾客成员
        wholeExpressNum = 0  # 一个超级顾客总运输量
        wholeTime = 0  # 一个超级顾客花费总时间
        lastPoint = -1  # 上一个待加入的点
        while (car > wholeExpressNum and wholeTime < 360):
            # 第一个点，顺序选择未加入的点加入
            if wholeTime == 0:
                nowPoint = -1  # 待加入的点
                for j in range(cf.numOfPoint3):
                    if pointleft[j] == 0:
                        nowPoint = j
                        stayTime = np.random.randint(15, 16)  # 每个点花费的时间
                        oneSuperCustomer.append(nowPoint)
                        pointleft[nowPoint] = 1
                        wholeExpressNum = wholeExpressNum + point3[nowPoint].small
                        wholeTime = wholeTime + stayTime
                        lastPoint = nowPoint
                        break
            # 除第一个点按照距离最近的点加入
            else:
                distance = 1000  # 定义一个超长距离
                nowPoint = -1  # 定义待加入的点
                for j in range(cf.numOfPoint3):
                    if cf.Point3Distance(lastPoint, j) < distance and lastPoint != j and pointleft[j] == 0:
                        distance = cf.Point3Distance(lastPoint, j)
                        nowPoint = j
                # 判断能不能加入这个点
                stayTime = np.random.randint(15, 16)  # 点花费的时间
                if wholeTime + stayTime > 360 or wholeExpressNum + point3[nowPoint].small > car or nowPoint == -1:
                    break
                else:
                    oneSuperCustomer.append(nowPoint)
                    pointleft[nowPoint] = 1
                    wholeExpressNum = wholeExpressNum + point3[nowPoint].small
                    wholeTime = wholeTime + stayTime
                    lastPoint = nowPoint
        smallSup.append(oneSuperCustomer)

    # 生成大件超级顾客，用V3进行运输
    pointleft = np.zeros(cf.numOfPoint3)  # 点被使用与否的标志，0表示未划分，1表示划分
    car = cf.car_loading[2][2 * i + 1]
    ##开始模拟生成
    while (isAllUsed(pointleft) == 0):
        oneSuperCustomer = []  # 一个超级顾客成员
        wholeExpressNum = 0  # 一个超级顾客总运输量
        wholeTime = 0  # 一个超级顾客花费总时间
        lastPoint = -1  # 上一个待加入的点
        while (car > wholeExpressNum and wholeTime < 360):
            # 第一个点，顺序选择未加入的点加入
            if wholeTime == 0:
                nowPoint = -1  # 待加入的点
                for j in range(cf.numOfPoint3):
                    if pointleft[j] == 0:
                        nowPoint = j
                        stayTime = np.random.randint(25, 26)  # 每个点花费的时间
                        oneSuperCustomer.append(nowPoint)
                        pointleft[nowPoint] = 1
                        wholeExpressNum = wholeExpressNum + point3[nowPoint].big
                        wholeTime = wholeTime + stayTime
                        lastPoint = nowPoint
                        break
            # 除第一个点按照距离最近的点加入
            else:
                distance = 1000  # 定义一个超长距离
                nowPoint = -1  # 定义待加入的点
                for j in range(cf.numOfPoint3):
                    if cf.Point3Distance(lastPoint, j) < distance and lastPoint != j and pointleft[j] == 0:
                        distance = cf.Point3Distance(lastPoint, j)
                        nowPoint = j
                # 判断能不能加入这个点
                stayTime = np.random.randint(25, 26)  # 点花费的时间
                if wholeTime + stayTime > 360 or wholeExpressNum + point3[nowPoint].big > car or nowPoint == -1:
                    break
                else:
                    oneSuperCustomer.append(nowPoint)
                    pointleft[nowPoint] = 1
                    wholeExpressNum = wholeExpressNum + point3[nowPoint].big
                    wholeTime = wholeTime + stayTime
                    lastPoint = nowPoint
        bigSup.append(oneSuperCustomer)

    # 返回
    return smallSup, bigSup

def smallSup_size(smallSup,point3):
    smallSup_sizes=[]
    for l in range(len(smallSup)):
        w = 0
        for m in range(len(smallSup[l])):
            w += point3[smallSup[l][m]].small
        smallSup_sizes.append(w)
    return smallSup_sizes

def bigSup_size(bigSup,point3):
    bigSup_sizes=[]
    for l in range(len(bigSup)):
        w = 0
        for m in range(len(bigSup[l])):
            w += point3[bigSup[l][m]].big
        bigSup_sizes.append(w)
    return bigSup_sizes

def smallSup_cycle_distance(smallSup):
    list5 = []  # 运来存放每个超级顾客内部的最长距离
    for j in range(len(smallSup)):
        list1 = []
        for l in range(len(smallSup[j]) - 1):
            for m in range(l, len(smallSup[j])):
                list1.append(cf.Point3Distance(smallSup[j][l], smallSup[j][m]))
        list1.sort()
        list1.reverse()
        if len(list1) == 0:  # 由于超级顾客在形成中产生了空超级顾客，所以有此判断；若不会形成空超级顾客，则可删除
            list5.append(0)
        else:
            list5.append(list1[0])
    cycle_distance = []
    for j in range(len(list5)):
        cycle_distance.append(0.71*((list5[j] ** 2 / 2 * len(smallSup[j])) ** (1 / 2)))
    return cycle_distance

def bigSup_cycle_distance(bigSup):
    list5 = []  # 运来存放每个超级顾客内部的最长距离
    for j in range(len(bigSup)):
        list1 = []
        for l in range(len(bigSup[j]) - 1):
            for m in range(l, len(bigSup[j])):
                list1.append(cf.Point3Distance(bigSup[j][l], bigSup[j][m]))
        list1.sort()
        list1.reverse()
        if len(list1) == 0:  # 由于超级顾客在形成中产生了空超级顾客，所以有此判断；若不会形成空超级顾客，则可删除
            list5.append(0)
        else:
            list5.append(list1[0])
    cycle_distance = []
    for j in range(len(list5)):
        cycle_distance.append(0.71*((list5[j] ** 2 / 2 * len(bigSup[j])) ** (1 / 2)))
    return cycle_distance

#判断point3是否全部被划分
def isAllUsed(Array):
    for i in range(len(Array)):
        if Array[i]==0:
            return 0
    return 1

#  cyk:生成各超级顾客到各中转中心的距离（平均距离）
def get_distance(smallSup,bigSup,point3):
    smallSupDistance=np.zeros((len(smallSup),int(cf.numOfPoint2)))
    for i in range(len(smallSup)):
        for j in range(cf.numOfPoint2):
            for k in range(len(smallSup[i])):
                smallSupDistance[i][j]+=cf.Point32Distance(smallSup[i][k],j)*(1/len(smallSup[i]))
    bigSupDistance=np.zeros((len(bigSup),int(cf.numOfPoint2)))
    for i in range(len(bigSup)):
        for j in range(cf.numOfPoint2):
            for k in range(len(bigSup[i])):
                bigSupDistance[i][j]+=cf.Point32Distance(bigSup[i][k],j)*(1/len(bigSup[i]))
    return smallSupDistance,bigSupDistance

def stock_table(Sup):
    table=[]
    [table.append([]) for i in range(cf.numOfPoint2)]
    for j in range(len(table)):
        [table[j].append(0) for k in range(len(Sup))]
    return table

def get_original_small_table_ub(i,Sup,Supdistance,table,Supsizes):  #  i表示临界值
    rest_size=[cf.point_2_load_size[i*2] for j in range(cf.numOfPoint2)]
    for j in range(len(Sup)):
        list1 = []
        [list1.append([float(Supdistance[j][k]),k]) for k in range(cf.numOfPoint2)]
        list1.sort()
        t=0
        while t<1000:
            if Supsizes[j]<=rest_size[list1[t][1]]:
                table[list1[t][1]][j]=1
                rest_size[list1[t][1]]-=Supsizes[j]
                break
            else:
                t+=1
    return table

def get_original_big_table_ub(i,Sup,Supdistance,table,Supsizes):  #  i表示临界值
    rest_size=[cf.point_2_load_size[i*2+1] for j in range(cf.numOfPoint2)]
    for j in range(len(Sup)):
        list1 = []
        [list1.append([float(Supdistance[j][k]),k]) for k in range(cf.numOfPoint2)]
        list1.sort()
        t=0
        while t<1000:
            if Supsizes[j]<=rest_size[list1[t][1]]:
                table[list1[t][1]][j]=1
                rest_size[list1[t][1]]-=Supsizes[j]
                break
            else:
                t+=1120
    return table

def open2small(i,t,smallSup,smallSupDistance,table,lamda,smallSup_sizes,small_cycle_distance):  #  cyk:t表示临界值为第t个，从0开始数
    value_list=[]
    for j in range(len(smallSup)):
        value_list.append((cf.cv2*(2*smallSupDistance[j][i]+small_cycle_distance[j])-lamda[j])*(-1))

    weight_list=[]
    weight_list=smallSup_sizes[:]
    for j in range(len(weight_list)):
        if weight_list[j]==0:
            weight_list[j]=0.000001
    capacity=int(cf.point_2_load_size[t*2])
    list1=[]
    for j in range(len(value_list)):
        list1.append(value_list[j]/weight_list[j])
    list2=[k for k in range(len(value_list))]
    list3=list(zip(list1,list2))
    list3.sort()
    list3.reverse()
    sizes=0
    for j in range(len(list3)):
        if sizes+weight_list[list3[j][1]]<=capacity:
            sizes+=weight_list[list3[j][1]]
            table[i][list3[j][1]]=1
        else:
            break
    c=0
    for j in range(len(smallSup)):
        c+=(cf.cv2*(2*smallSupDistance[j][i]+small_cycle_distance[j])-lamda[j])*table[i][j]
    if (c+cf.fs+2*10*cf.point_12_distance[0][i])>=0:
        for j in range(len(smallSup)):
            table[i][j]=0
    for j in range(len(smallSup)):
        if smallSupDistance[j][i]>cf.length:
            table[i][j]=0

    return table


def open2big(i,t,bigSup,bigSupDistance,table,lamda,bigSup_sizes,big_cycle_distance):  #  cyk:k表示临界值为第k个，从0开始数
    value_list=[]
    for j in range(len(bigSup)):
        value_list.append((5*(2*bigSupDistance[j][i]+big_cycle_distance[j])-lamda[j])*(-1))
    weight_list=[]
    weight_list=bigSup_sizes[:]
    for j in range(len(weight_list)):
        if weight_list[j]==0:
            weight_list[j]=0.0000001
    capacity=int(cf.point_2_load_size[t*2+1])
    list1=[]
    for j in range(len(value_list)):
        list1.append(value_list[j] / weight_list[j])
    list2 = [k for k in range(len(value_list))]
    list3 = list(zip(list1, list2))
    list3.sort()
    list3.reverse()
    sizes = 0
    for j in range(len(list3)):
        if sizes + weight_list[list3[j][1]] <= capacity:
            sizes += weight_list[list3[j][1]]
            table[i][list3[j][1]] = 1
        else:
            break
    c = 0
    for j in range(len(bigSup)):
        c += (5 * (2*bigSupDistance[j][i] + big_cycle_distance[j]) - lamda[j]) * table[i][j]
    if (c + cf.fL + 2 * 10 * cf.point_12_distance[0][i]) >= 0:
        for j in range(len(bigSup)):
            table[i][j] = 0
    for j in range(len(bigSup)):
        if bigSupDistance[j][i] > cf.length:
            table[i][j] = 0

    return table


'''def get_i_dont_know(table,smallSup,smallSup_sizes,lamda):
    idc = []  # 用于存放每一个标准超级顾客的idc的值的列表
    x_location = []  # 用于存放开放的备选点的序号
    x_verse_location = []  # 用于存放不开放的备选点序号
    for j in range(cf.numOfPoint2):
        if any(table[j][k] != 0 for k in range(len(smallSup))):
            x_location.append(j)
    for j in range(cf.numOfPoint2):
        if all(table[j][k] == 0 for k in range(len(smallSup))):
            x_verse_location.append(j)
    for j in range(len(smallSup)):
        idc_value = 0
        for l in range(cf.numOfPoint2):
            idc_value += table[l][j]
        idc.append(idc_value)
    for j in range(len(smallSup)):
        if idc[j] == 0:
            value1=[]
            weight=smallSup_sizes[j]
            if weight==0:
                weight=0.000001
            for k in range(cf.numOfPoint2):
                value1.append([(cf.cv2 * (2 * smallSupDistance[j][k] + small_cycle_distance[j]) - lamda[j]) * (-1)/weight,k])
            value1.sort()
            value1.reverse()
            print(value1)
            table[value1[0][1]][j]=1
    return table'''





def get_small_table_ub(t,smallSup,table_lb,smallSupDistance,smallSup_sizes):  #  t表示临界值
    table=copy.deepcopy(table_lb)
    idc=[]  #  用于存放每一个标准超级顾客的idc的值的列表
    x_location=[]  #  用于存放开放的备选点的序号
    x_verse_location=[]  #用于存放不开放的备选点序号
    for j in range(cf.numOfPoint2):
        if any(table[j][k]!=0 for k in range(len(smallSup))):
            x_location.append(j)
    for j in range(cf.numOfPoint2):
        if all(table[j][k]==0 for k in range(len(smallSup))):
            x_verse_location.append(j)
    for j in range(len(smallSup)):
        idc_value=0
        for l in range(cf.numOfPoint2):
            idc_value+=table[l][j]
        idc.append(idc_value)
    for j in range(len(smallSup)):
        if idc[j]>1:
            list1 = []
            for k in range(cf.numOfPoint2):
                if table[k][j] == 1:
                    list1.append([smallSupDistance[j][k], k])
            list1.sort()
            for k in range(1, len(list1)):
                table[list1[k][1]][j] = 0
            x_location=[]
            x_verse_location=[]
            for m in range(cf.numOfPoint2):
                if any(table[m][k] != 0 for k in range(len(smallSup))):
                    x_location.append(m)
            for m in range(cf.numOfPoint2):
                if all(table[m][k] == 0 for k in range(len(smallSup))):
                    x_verse_location.append(m)
    for j in range(len(smallSup)):
        if idc[j] == 1:
            continue
        elif idc[j] == 0:
            list1 = []  # 用来放在已经开放的且能够容纳该超级顾客的备选点的距离和编号
            list2 = []  # 用来存放所有的未开放的备选点到该超级顾客距离和编号
            for k in x_location:
                allsize = 0  # 已经开放的二级点k的包含的包裹数量
                for l in range(len(smallSup_sizes)):
                    allsize += table[k][l] * smallSup_sizes[l]
                if cf.point_2_load_size[t * 2] - allsize >= smallSup_sizes[j] and smallSupDistance[j][k] < cf.length:
                    list1.append([smallSupDistance[j][k], k])
                    
            if len(list1) != 0:
                list1.sort()
                table[list1[0][1]][j] = 1
            else:
                for k in x_verse_location:
                    list2.append([smallSupDistance[j][k], k])
                list2.sort()
                table[list2[0][1]][j] = 1
                x_location.append(list2[0][1])
                x_verse_location.remove(list2[0][1])

    '''for j in range(len(smallSup)):
        if idc[j]==1:
            continue
        elif idc[j]==0:
            list1=[]  #  用来放在已经开放的且能够容纳该超级顾客的备选点的距离和编号
            list2=[]  #  用来存放所有的未开放的备选点到该超级顾客距离和编号
            for k in x_location:
                allsize=0  #  已经开放的二级点k的包含的包裹数量
                for l in range(len(smallSup_sizes)):
                    allsize+=table[k][l]*smallSup_sizes[l]
                if cf.point_2_load_size[t*2]-allsize>smallSup_sizes[j] and smallSupDistance[j][k]<cf.length:
                    list1.append([smallSupDistance[j][k],k])
            if len(list1)!=0:
                list1.sort()
                table[list1[0][1]][j]=1
            else:
                for k in x_verse_location:
                    list2.append([smallSupDistance[j][k],k])
                list2.sort()
                print(list1)
                print(list2)
                table[list2[0][1]][j]=1
                x_location.append(list2[0][1])
                x_verse_location.remove(list2[0][1])
        else:
            list1=[]
            list2=[]
            for k in range(cf.numOfPoint2):
                if table[k][j]==1:
                    list1.append([smallSupDistance[j][k],k])
            list1.sort()
            for k in range(1,len(list1)):
                table[list1[k][1]][j]=0
            for j in range(cf.numOfPoint2):
                if any(table[j][k] != 0 for k in range(len(smallSup))):
                    x_location.append(j)
            for j in range(cf.numOfPoint2):
                if all(table[j][k] == 0 for k in range(len(smallSup))):
                    x_verse_location.append(j)'''
    table_ub=copy.deepcopy(table)
    return table_ub

def get_big_table_ub(t,bigSup,table_lb,bigSupDistance,bigSup_sizes):  #  t表示临界值
    table=copy.deepcopy(table_lb)
    idc=[]  #  用于存放每一个标准超级顾客的idc的值的列表
    x_location=[]  #  用于存放开放的备选点的序号
    x_verse_location=[]  #用于存放不开放的备选点序号
    for j in range(cf.numOfPoint2):
        if any(table[j][k]==1 for k in range(len(bigSup))):
            x_location.append(j)
    for j in range(cf.numOfPoint2):
        if all(table[j][k]==0 for k in range(len(bigSup))):
            x_verse_location.append(j)
    for j in range(len(bigSup)):
        idc_value=0
        for l in range(cf.numOfPoint2):
            idc_value+=table[l][j]
        idc.append(idc_value)
    for j in range(len(bigSup)):
        if idc[j]>1:
            list1 = []
            for k in range(cf.numOfPoint2):
                if table[k][j] == 1:
                    list1.append([bigSupDistance[j][k], k])
            list1.sort()
            for k in range(1, len(list1)):
                table[list1[k][1]][j] = 0
            x_location=[]
            x_verse_location=[]
            for j in range(cf.numOfPoint2):
                if any(table[j][k] == 1 for k in range(len(bigSup))):
                    x_location.append(j)
            for j in range(cf.numOfPoint2):
                if all(table[j][k] == 0 for k in range(len(bigSup))):
                    x_verse_location.append(j)
    for j in range(len(bigSup)):
        if idc[j] == 1:
            continue
        elif idc[j] == 0:
            list1 = []  # 用来放在已经开放的且能够容纳该超级顾客的备选点的距离和编号
            list2 = []  # 用来存放所有的未开放的备选点到该超级顾客距离和编号
            for k in x_location:
                allsize = 0
                for l in range(len(bigSup_sizes)):
                    allsize += table[k][l] * bigSup_sizes[l]
                if cf.point_2_load_size[t * 2 + 1] - allsize > bigSup_sizes[j] and bigSupDistance[j][k] < cf.length:
                    list1.append([bigSupDistance[j][k], k])
            if len(list1) != 0:
                list1.sort()
                table[list1[0][1]][j] = 1
            else:
                for k in x_verse_location:
                    list2.append([bigSupDistance[j][k], k])
                list2.sort()
                table[list2[0][1]][j] = 1
                x_location.append(list2[0][1])
                x_verse_location.remove(list2[0][1])
    '''for j in range(len(bigSup)):
        if idc[j]==1:
            continue
        elif idc[j]==0:
            list1=[]  #  用来放在已经开放的且能够容纳该超级顾客的备选点的距离和编号
            list2=[]  #  用来存放所有的未开放的备选点到该超级顾客距离和编号
            for k in x_location:
                allsize=0
                for l in range(len(bigSup_sizes)):
                    allsize+=table[k][l]*bigSup_sizes[l]
                if cf.point_2_load_size[t*2+1]-allsize>bigSup_sizes[j] and bigSupDistance[j][k]<cf.length:
                    list1.append([bigSupDistance[j][k],k])
            if len(list1)!=0:
                list1.sort()
                table[list1[0][1]][j]=1
            else:
                for k in x_verse_location:
                    list2.append([bigSupDistance[j][k],k])
                list2.sort()
                table[list2[0][1]][j]=1
                x_location.append(list2[0][1])
                x_verse_location.remove(list2[0][1])
        else:
            list1=[]
            list2=[]
            for k in range(cf.numOfPoint2):
                if table[k][j]==1:
                    list1.append([bigSupDistance[j][k],k])
            list1.sort()
            for k in range(1,len(list1)):
                table[list1[k][1]][j]=0
            for j in range(cf.numOfPoint2):
                if any(table[j][k] == 1 for k in range(len(bigSup))):
                    x_location.append(j)
            for j in range(cf.numOfPoint2):
                if all(table[j][k] == 0 for k in range(len(bigSup))):
                    x_verse_location.append(j)'''

    table_ub=copy.deepcopy(table)
    return table_ub

def get_small(i,table,smallSup,smallSupDistance,point3,smallSup_sizes,cycle_distance):
    totalmoney=0
    totalopen=0
    m1=0
    m2=0
    m3=0
    m4=0
    m5=0
    allsizesize=0
    x_location=[]
    for j in range(cf.numOfPoint2):  #  可以尝试简化
        if any(table[j][k]!=0 for k in range(len(smallSup))):
            x_location.append(j)
    x_location2=x_location
    for j in x_location2:
        if all(table[j][k]*smallSup_sizes[k]==0 for k in range(len(smallSup))):
            x_location.remove(j)
    totalopen=len(x_location)
    totalmoney=totalopen*cf.fs
    m1+=totalopen*cf.fs  #  第一部分建设成本

    for j in x_location:
        totalmoney += 2 * cf.point_12_distance[0][j]*10
        m2 += 2 * cf.point_12_distance[0][j]*10  # 一级到二级的运输成本
    final_size=0
    for j in range(cf.numOfPoint3):
        for k in range(i+1):
            for l in range(2):
                final_size+=point3[j].P[k][l]
    totalmoney+=final_size*cf.moneyofProcessingExpress[i][1]
    m3=final_size*cf.moneyofProcessingExpress[i][1]  #  二级点的处理成本
    tt = 0  # 记录空超级顾客的数量
    distance = 0
    for j in range(len(cycle_distance)):
        if cycle_distance[j] == 0:
            tt += 1
    for j in x_location:
        for k in range(len(smallSup)):
            if table[j][k] == 1 and cycle_distance[k] != 0:
                distance += 2 * smallSupDistance[k][j]
    totaldistance = distance + sum(cycle_distance)
    totalmoney += cf.cv2 * totaldistance
    m4 += cf.cv2 * totaldistance

    '''for j in x_location:
        for k in range(len(smallSup)):
            list1 = []  # 用来存放已经开放的中转中心所服务的三级点之间的距离和编号
            if table[j][k] == 1:
                for l in range(len(smallSup[k])):
                    for m in range(l+1,len(smallSup[k])):
                        list1.append([cf.Point32Distance(l,m),l,m])
                list1.sort()
                list1.reverse()
                if len(list1)==0:
                    totaldistance=((((0/(2**(1/2)))**2)*len(smallSup[k]))**(1/2))*0.71+2*smallSupDistance[k][j]
                else:
                    totaldistance=((((list1[0][0]/(2**(1/2)))**2)*len(smallSup[k]))**(1/2))*0.71+2*smallSupDistance[k][j]
                totalmoney+=1*totaldistance
                m4+=1*totaldistance  #  二级点到超级顾客和超级顾客内部的运输成本'''
    final_size=0
    for j in range(cf.numOfPoint3):
        for k in range(i+1):
            for l in range(2):
                final_size+=point3[j].P[k][l]
    
    totalmoney+=final_size*cf.moneyofProcessingExpress[i][0]
    m5+=final_size*cf.moneyofProcessingExpress[i][0]
    return totalmoney, m1,m2+m4,m3+m5

def get_big(i,table,bigSup,bigSupDistance,point3,bigSup_sizes,cycle_distance):
    totalmoney=0
    totalopen=0
    m1=0
    m2=0
    m3=0
    m4=0
    m5=0
    allsizesize=0
    x_location=[]
    for j in range(cf.numOfPoint2):
        if any(table[j][k]==1 for k in range(len(bigSup))):
            x_location.append(j)
    x_location2=x_location[:]
    for j in x_location2:
        if all(table[j][k]*bigSup_sizes[k]==0 for k in range(len(bigSup))):
            x_location.remove(j)
    totalopen=len(x_location)
    totalmoney=totalopen*cf.fL
    m1+=totalopen*cf.fL
    for j in x_location:
        totalmoney += 2 * cf.point_12_distance[0][j]*10
        m2 += 2 * cf.point_12_distance[0][j]*10  # 一级到二级的运输成本

    final_size=0
    for j in range(cf.numOfPoint3):
        for k in range(5-i):
            for l in range(2):
                final_size+=point3[j].P[5-k][l]
    totalmoney += final_size * cf.moneyofProcessingExpress[i][3]
    m3 += final_size * cf.moneyofProcessingExpress[i][3]  # 二级点的处理成本
    tt = 0  # 记录空超级顾客的数量
    distance = 0
    for j in range(len(cycle_distance)):
        if cycle_distance[j] == 0:
            tt += 1
    for j in x_location:
        for k in range(len(bigSup)):
            if table[j][k] == 1 and cycle_distance[k] != 0:
                distance += 2 * bigSupDistance[k][j]
    totaldistance = distance + sum(cycle_distance)
    totalmoney += 5 * totaldistance
    m4 += 5 * totaldistance

    '''for j in x_location:
            for k in range(len(bigSup)):
                if table[j][k] == 1:
                    if list5[k]==0:      #    由于超级顾客在形成中产生了空超级顾客，所以有此判断；若不会形成空超级顾客，则可删除
                        totaldistance=0
                    else:
                        totaldistance = ((list5[k] ** 2) / 2) * len(bigSup[k]) ** (1 / 2) + 2 * 0.71 * bigSupDistance[k][j]
                    print('totaldistance=',totaldistance)
                    totalmoney += 5 * totaldistance
                    m4 += 5 * totaldistance  # 二级点到超级顾客和超级顾客内部的运输成本'''
    '''for j in x_location:
        for k in range(len(bigSup)):
            list1 = []  # 用来存放已经开放的中转中心所服务的三级点之间的距离和编号
            if table[j][k] == 1:
                for l in range(len(bigSup[k])):
                    for m in range(l+1,len(bigSup[k])):
                        list1.append([cf.Point32Distance(l,m),l,m])
                list1.sort()
                list1.reverse()
                if len(list1)==0:
                    totaldistance=((((0/(2**(1/2)))**2)*len(bigSup[k]))**(1/2))*0.71+2*bigSupDistance[k][j]
                else:
                    totaldistance=((((list1[0][0]/(2**(1/2)))**2)*len(bigSup[k]))**(1/2))*0.71+2*bigSupDistance[k][j]
                totalmoney+=5*totaldistance
                m4+=5*totaldistance  #  二级点到超级顾客和超级顾客内部的运输成本'''
    final_size=0
    for j in range(cf.numOfPoint3):
        for k in range(5-i):
            for l in range(2):
                final_size+=point3[j].P[5-k][l]
    totalmoney+=final_size*cf.moneyofProcessingExpress[i][2]
    m5+=final_size*cf.moneyofProcessingExpress[i][2]  #  三级点的处理成本
    return totalmoney, m1,m2+m4,m3+m5



def get_lamda(lamda,beta,lb,ub,table_lb,Sup):
    lamda_last=lamda[:]
    d=[]
    for j in range(len(Sup)):
        t=0
        for k in range(cf.numOfPoint2):
            t+=table_lb[k][j]
        d.append(1-t)
    d_length=0
    for j in range(len(d)):
        d_length+=d[j]**2
    if d_length==0:
        theta=0
    else:
        theta=beta*abs((ub-lb))/(d_length**(1/2))
    for j in range(len(lamda)):
        lamda[j]=lamda_last[j]+theta*d[j]
    return lamda,d


def judge(d,time1):
    if all(d[j]==0 for j in range(len(d))) or time1 > 60:# 张会芳：这个位置的数字（time1>后面）就是指迭代的次数
        return 1
    else:
        return 0

def get_small_original_lamda(smallSup,smallSupDistance):
    small_lamda=[]
    for i in range(len(smallSup)):
        w=0
        list1=[]
        for j in range(cf.numOfPoint2):
            w=2*1*smallSupDistance[i][j]
            list1.append(w)
        small_lamda.append(min(list1))
    return small_lamda

def get_big_original_lamda(bigSup,bigSupDistance):
    big_lamda=[]
    for i in range(len(bigSup)):
        w=0
        list1=[]
        for j in range(cf.numOfPoint2):
            w=2*1*bigSupDistance[i][j]
            list1.append(w)
        big_lamda.append(min(list1))
    return big_lamda

def final_cost(small_ub,small_ub_cost1,small_ub_cost2,small_ub_cost3,small_x_location,big_ub,big_ub_cost1,big_ub_cost2,big_ub_cost3,big_x_location):
    for j in small_x_location:
        for k in big_x_location:
            if j==k:
                small_ub_cost2-=cf.point_12_distance[0][j]*10*2
                small_ub-=cf.point_12_distance[0][j]*10*2
    return small_ub+big_ub,small_ub_cost1+big_ub_cost1,small_ub_cost2+big_ub_cost2,small_ub_cost3+big_ub_cost3





if __name__=='__main__':
    start=time.time()
    point3 = cf.Point3LoadData('3-point-expressNum-new.txt')
    with open('result.txt', 'w') as f:
        for i in range(5):
            print("正在生成%dkg的相关信息---" % (cf.expressWeight[i]))
            smallSup, bigSup = superCustomer(i, point3)
            smallSup_sizes=smallSup_size(smallSup, point3)
            bigSup_sizes=bigSup_size(bigSup, point3)
            small_cycle_distance=smallSup_cycle_distance(smallSup)
            big_cycle_distance=bigSup_cycle_distance(bigSup)
            smallSupDistance,bigSupDistance=get_distance(smallSup,bigSup,point3)
            small_lamda = [min(
                cf.cv2*(smallSupDistance[j][k] + small_cycle_distance[j]) for k in range(cf.numOfPoint2)) for j in
                range(len(smallSup))]
            big_lamda = [min(5 * (bigSupDistance[j][k] + big_cycle_distance[j]) for k in range(cf.numOfPoint2)) for j in
                         range(len(bigSup))]
            table=stock_table(smallSup)
            small_table_ub_original=get_original_small_table_ub(i,smallSup,smallSupDistance,table,smallSup_sizes)
            table=stock_table(bigSup)
            big_table_ub_original=get_original_big_table_ub(i,bigSup,bigSupDistance,table,bigSup_sizes)
            small_ub, small_ub_cost1, small_ub_cost2, small_ub_cost3 = get_small(i, small_table_ub_original, smallSup,
                                                                                 smallSupDistance, point3,
                                                                                 smallSup_sizes,small_cycle_distance)
            big_ub, big_ub_cost1, big_ub_cost2, big_ub_cost3 = get_big(i, big_table_ub_original, bigSup, bigSupDistance, point3,
                                                                       bigSup_sizes,big_cycle_distance)
            small_lamda=[]

            '''small_lamda=[(small_ub_cost1+small_ub_cost2)/len(smallSup) for j in range(len(smallSup))]
            big_lamda=[(big_ub_cost1+big_ub_cost2)/len(bigSup) for j in range(len(bigSup))]'''
            d=[5]

            '''table=stock_table(smallSup)
            for j in range(cf.numOfPoint2):
                small_table=open2small(j,i,smallSup,smallSupDistance,table,small_lamda,smallSup_sizes)
            small_table_lb=copy.deepcopy(small_table)
            small_table_ub = get_small_table_ub(i, smallSup, small_table_lb, smallSupDistance,smallSup_sizes)
            table = stock_table(bigSup)
            for j in range(cf.numOfPoint2):
                big_table=open2big(j,i,bigSup,bigSupDistance,table,big_lamda,bigSup_sizes)
            big_table_lb=copy.deepcopy(big_table)
            big_table_ub=get_big_table_ub(i,bigSup,big_table_lb,bigSupDistance,bigSup_sizes)
            small_lb,small_lb_cost1,small_lb_cost2,small_lb_cost3=get_small(i,small_table_lb,smallSup,smallSupDistance,point3,smallSup_sizes)
            small_ub,small_ub_cost1,small_ub_cost2,small_ub_cost3=get_small(i,small_table_ub,smallSup,smallSupDistance,point3,smallSup_sizes)
            big_lb,big_lb_cost1,big_lb_cost2,big_lb_cost3=get_big(i,big_table_lb,bigSup,bigSupDistance,point3,bigSup_sizes)
            big_ub, big_ub_cost1, big_ub_cost2, big_ub_cost3 = get_big(i, big_table_ub, bigSup, bigSupDistance, point3,bigSup_sizes)'''
            '''for j in range(cf.numOfPoint2):
                jj = 0
                for k in range(len(smallSup)):
                    jj += small_table_lb[j][k]
                print('lb=',  (j, jj))
            print('diyici ub-lb=',small_ub-small_lb)
            print('diyici lamda=',small_lamda)
            for j in range(cf.numOfPoint2):
                jj = 0
                for k in range(len(bigSup)):
                    jj += big_table_lb[j][k]
                print('lb=',  (j, jj))
            print('diyici ub-lb=',big_ub-big_lb)
            print('diyici lamda=',big_lamda)'''
            time1=1
            beta=1.5
            original_beta=1.5
            ub_minus_lb_collector=[]
            list123 = []  # 用来存放每一次迭代的ub的解
            small_lamda = [min(
                cf.cv2 * (smallSupDistance[j][k] + small_cycle_distance[j]) for k in range(cf.numOfPoint2)) for j in
                range(len(smallSup))]

            '''small_lamda,d=get_lamda(small_lamda,beta,small_lb,small_ub,small_table_lb,smallSup)'''
            while judge(d,time1)==0:

                if len(ub_minus_lb_collector)>1:
                    if ub_minus_lb_collector[len(ub_minus_lb_collector)-1]==ub_minus_lb_collector[len(ub_minus_lb_collector)-2]:
                        beta/=2
                    '''if abs(ub_minus_lb_collector[len(ub_minus_lb_collector)-1]-ub_minus_lb_collector[len(ub_minus_lb_collector)-2])<500:
                        beta=original_beta'''
                table = stock_table(smallSup)
                for j in range(cf.numOfPoint2):
                    small_table = open2small(j, i, smallSup, smallSupDistance, table, small_lamda,smallSup_sizes,small_cycle_distance)

                small_table_lb = copy.deepcopy(small_table)
                small_table_ub = get_small_table_ub(i, smallSup, small_table_lb, smallSupDistance,smallSup_sizes)
                small_lb, small_lb_cost1, small_lb_cost2, small_lb_cost3 = get_small(i, small_table_lb, smallSup,
                                                                                     smallSupDistance, point3,smallSup_sizes,small_cycle_distance)

                small_ub, small_ub_cost1, small_ub_cost2, small_ub_cost3 = get_small(i, small_table_ub, smallSup,
                                                                                     smallSupDistance, point3,smallSup_sizes,small_cycle_distance)
                ub_minus_lb_collector.append(abs(small_ub - small_lb))
                time1+=1
            print(small_ub, small_ub_cost1, small_ub_cost2, small_ub_cost3)

            time1=1
            beta=1.5
            original_beta=beta
            d=[5]
            ub_minus_lb_collector = []
            while judge(d,time1)==0:
                if len(ub_minus_lb_collector)>1:
                    if ub_minus_lb_collector[len(ub_minus_lb_collector)-1] == ub_minus_lb_collector[len(ub_minus_lb_collector) - 2]:
                        beta/=2
                table = stock_table(bigSup)
                for j in range(cf.numOfPoint2):
                    big_table = open2big(j, i, bigSup, bigSupDistance, table, big_lamda,bigSup_sizes,big_cycle_distance)
                big_table_lb = copy.deepcopy(big_table)
                big_lb, big_lb_cost1, big_lb_cost2, big_lb_cost3 = get_big(i, big_table_lb, bigSup, bigSupDistance,point3,bigSup_sizes,big_cycle_distance)
                big_table_ub = get_big_table_ub(i, bigSup, big_table_lb, bigSupDistance, bigSup_sizes)
                big_ub, big_ub_cost1, big_ub_cost2, big_ub_cost3 = get_big(i, big_table_ub, bigSup, bigSupDistance,point3,bigSup_sizes,big_cycle_distance)
                ub_minus_lb_collector.append(abs(big_ub - big_lb))


                big_lamda, d = get_lamda(big_lamda, beta, big_lb_cost1, big_ub_cost1, big_table_lb, bigSup)

                time1+=1
            print(big_ub, big_ub_cost1, big_ub_cost2, big_ub_cost3)

            small_x_location=[]
            big_x_location=[]
            mix=0
            for j in range(cf.numOfPoint2):
                if any(small_table_ub[j][k] != 0 for k in range(len(smallSup))):
                    small_x_location.append(j)
            small_x_location1=small_x_location[:]
            for j in small_x_location1:
                if all(small_table_ub[j][k]*smallSup_sizes[k]==0 for k in range(len(smallSup))):
                    small_x_location.remove(j)
            for j in range(cf.numOfPoint2):
                if any(big_table_ub[j][k] != 0 for k in range(len(bigSup))):
                    big_x_location.append(j)
            big_x_location1=big_x_location[:]
            for j in big_x_location1:
                if all(big_table_ub[j][k]*bigSup_sizes[k]==0 for k in range(len(bigSup))):
                    big_x_location.remove(j)
            list_mixSup=[]
            for j in range(len(small_x_location)):
                for k in range(len(big_x_location)):
                    if small_x_location[j]==big_x_location[k]:
                        list_mixSup.append(small_x_location[j])
                        mix+=1
            small=len(small_x_location)-mix
            big=len(big_x_location)-mix
            '''for j in small_x_location:
                list123=[]
                total123=0
                for k in range(len(smallSup)):
                    if small_table_ub[j][k]==1:
                        list123.append(k)
                        total123+=smallSup_sizes[k]
                print(j,':',list123,total123)'''
            

            print('v1的总体平均装载率为：',(sum(smallSup_sizes)+sum(bigSup_sizes))/(cf.car_loading[0][i*2]*(small+mix)+cf.car_loading[0][i*2+1]*(big+mix)))
            mix_sum_sizes=0
            small_mix_sum_sizes=0
            big_mix_sum_sizes=0
            for j in list_mixSup:
                for k in range(len(smallSup)):
                    if small_table_ub[j][k]==1:
                        mix_sum_sizes+=smallSup_sizes[k]
                        small_mix_sum_sizes+=smallSup_sizes[k]
                for k in range(len(bigSup)):
                    if big_table_ub[j][k]==1:
                        mix_sum_sizes+=bigSup_sizes[k]
                        big_mix_sum_sizes+=bigSup_sizes[k]
            print('v1的标准平均装载率为：',(sum(smallSup_sizes)-small_mix_sum_sizes)/(cf.car_loading[0][i*2]*small))
            print('v1的大件平均装载率为：',(sum(bigSup_sizes)-big_mix_sum_sizes)/(cf.car_loading[0][i*2+1]*big))
            print('v1的混合平均装载率为：',mix_sum_sizes/((cf.car_loading[0][i*2]+cf.car_loading[0][i*2+1])*mix))
            print('二到三级的总体平均装载率为：',(sum(smallSup_sizes)+sum(bigSup_sizes))/((cf.car_loading[1][i*2]*len(smallSup)+(cf.car_loading[2][i*2+1]*len(bigSup)))))
            ww=0  #用来记录空超级顾客的数量
            for j in range(len(smallSup)):
                if smallSup_sizes[j]==0:
                    ww+=1
            
            print('二级到三级的标准平均装载率为：',sum(smallSup_sizes)/(cf.car_loading[1][i*2]*(len(smallSup)-ww)))
            yy=0
            for j in range(len(bigSup)):
                if bigSup_sizes[j]==0:
                    yy+=1
            print('二级到三级的标准平均装载率为：',sum(bigSup_sizes)/(cf.car_loading[2][i*2+1]*(len(bigSup)-yy)))
            print('small_x_location=',small_x_location)
            print('big_x_location=',big_x_location)
            print('标准中转中心的个数为：',small)
            print('大件中转中心的个数为:',big)
            print('混合中转中心的个数为:',mix)
            total_cost,total_cost1,total_cost2,total_cost3=final_cost(small_ub,small_ub_cost1,small_ub_cost2,small_ub_cost3,small_x_location,big_ub,big_ub_cost1,big_ub_cost2,big_ub_cost3,big_x_location)
            print('总成本=',total_cost,'建设成本=',total_cost1,'运输成本=',total_cost2,'处理成本=',total_cost3)
            
    end=time.time()
    print('time=',end-start)
   
