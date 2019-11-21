import math
import numpy as np
import cmath
import random
from scipy.optimize import curve_fit


##########0常用参数#############################################################

#已知参数：
#膜对两种材料Ni（1号）和Ti（2号）的：
#相干散射长度b（单位：fm），
#吸收截面σ（单位：barn=1e-24cm^2），
#质量密度ρ（单位：g/cm^3），
#原子质量A（单位：g/mol）
b1 = 10.3
b2 = -3.438
sigma1 = 4.49
sigma2 = 6.09
rho1 = 8.908
rho2 = 4.506
A1 = 58.6934
A2 = 47.867

#中间参数计算：
#计算Ni,Ti两种材料的散射长度密度ρb（单位：1e14m^-2）
#已知基底材料Si的散射长度密度ρbs（单位：1e14m^-2）
rhob1 = b1*rho1*6.022/A1
rhob2 = b2*rho2*6.022/A2
rhobs = 2.08

#计算Ni的全反射临界波矢kc（单位：1e7m^-1=0.01nm^-1=1e-3Å-1）
kc = 2*math.sqrt(math.pi*rhob1)


############1反射率计算函数#####################################################

#----------1.1基于矩阵方法的反射率计算函数--------------------------------------#

#Function1.1.1
#理想条件（无粗糙度）下的反射率计算函数，采用矩阵方法
#输入参数为：
#垂直方向的波矢分量k0（单位：0.01nm^-1），
#膜对序列d=[[1,d11,d12]……[n,dn1.dn2]]（膜层厚度单位：nm）
#输出：多层膜对入射波矢垂直分量为k0的中子的反射率
def Ref(k0,d=[]):
    M = np.ones([2,2],dtype =complex)
    M2 = np.ones([2,2],dtype =complex)
    M1 = np.ones([2,2],dtype =complex)
    k1 = cmath.sqrt(k0**2-4*math.pi*rhob1)
    k2 = cmath.sqrt(k0**2-4*math.pi*rhob2)
    ks = cmath.sqrt(k0**2-4*math.pi*rhobs)
    M[0][0] = (k2+ks)/(2*k2)
    M[0][1] = (k2-ks)/(2*k2)
    M[1][0] = (k2-ks)/(2*k2)
    M[1][1] = (k2+ks)/(2*k2)
    #对矩阵元素进行归一化，防止膜层数目过多时矩阵元素数值过大溢出
    M = M/np.sum(M**2)
    j = 0 
    #每个膜对逐次计算反射率，从最底层迭代上去，psi=ikd为计算公式中e的复指数
    while j< len(d):
        psi2 = complex(0,k2*d[len(d)-1-j][2]*0.01)
        M2[0][0] = (k1+k2)*cmath.exp(-psi2)/(2*k1)
        M2[0][1] = (k1-k2)*cmath.exp(psi2)/(2*k1)
        M2[1][0] = (k1-k2)*cmath.exp(-psi2)/(2*k1)
        M2[1][1] = (k1+k2)*cmath.exp(psi2)/(2*k1)
        M = np.dot(M2,M)
        psi1 = complex(0,k1*d[len(d)-1-j][1]*0.01)
        M1[0][0] = (k1+k2)*cmath.exp(-psi1)/(2*k2)
        M1[0][1] = (k2-k1)*cmath.exp(psi1)/(2*k2)
        M1[1][0] = (k2-k1)*cmath.exp(-psi1)/(2*k2)
        M1[1][1] = (k1+k2)*cmath.exp(psi1)/(2*k2)
        M = np.dot(M1,M)
        #对矩阵元素进行归一化，防止膜层数目过多时矩阵元素数值过大溢出
        M = M/np.sum(M**2)
        j = j+1
    r = M[1][0]/M[0][0]
    return abs(r)**2

#Function1.1.2
#考虑材料吸收中子条件下的反射率计算函数，采用矩阵方法
#输入参数为：
#垂直方向的波矢分量k0（单位：0.01nm^-1），
#入射中子波长lambda0（单位：nm），
#膜对序列d=[[1,d11,d12]……[n,dn1.dn2]]（膜层厚度单位：nm）
#输出：多层膜对入射波矢垂直分量为k0的中子的反射率
def RefAbs(k0,lambda0,d=[]):
    #计算两种材料的复数散射长度密度
    rhoba1 = rho1*6.022/A1*complex(b1,-sigma1/(2*lambda0)*10**(-4))
    rhoba2 = rho2*6.022/A2*complex(b2,-sigma2/(2*lambda0)*10**(-4))
    M = np.ones([2,2],dtype =complex)
    M2 = np.ones([2,2],dtype =complex)
    M1 = np.ones([2,2],dtype =complex)
    k1 = cmath.sqrt(k0**2-4*math.pi*rhoba1)
    k2 = cmath.sqrt(k0**2-4*math.pi*rhoba2)
    ks = cmath.sqrt(k0**2-4*math.pi*rhobs)
    M[0][0] = (k2+ks)/(2*k2)
    M[0][1] = (k2-ks)/(2*k2)
    M[1][0] = (k2-ks)/(2*k2)
    M[1][1] = (k2+ks)/(2*k2)
    j = 0 
    #每个膜对逐次计算反射率，从最底层迭代上去，psi=ikd为计算公式中e的复指数
    while j< len(d):
        psi2 = complex(0,k2*d[len(d)-1-j][2]*0.01)
        M2[0][0] = (k1+k2)*cmath.exp(-psi2)/(2*k1)
        M2[0][1] = (k1-k2)*cmath.exp(psi2)/(2*k1)
        M2[1][0] = (k1-k2)*cmath.exp(-psi2)/(2*k1)
        M2[1][1] = (k1+k2)*cmath.exp(psi2)/(2*k1)
        M = np.dot(M2,M)
        psi1 = complex(0,k1*d[len(d)-1-j][1]*0.01)
        M1[0][0] = (k1+k2)*cmath.exp(-psi1)/(2*k2)
        M1[0][1] = (k2-k1)*cmath.exp(psi1)/(2*k2)
        M1[1][0] = (k2-k1)*cmath.exp(-psi1)/(2*k2)
        M1[1][1] = (k1+k2)*cmath.exp(psi1)/(2*k2)
        M = np.dot(M1,M)
        j = j+1
    r = M[1][0]/M[0][0]
    return abs(r)**2

#Function1.1.3  
#粗糙度为常数值，且认为波矢经过膜层时不变的条件下的反射率计算函数，采用矩阵方法
#此函数不推荐使用  
#输入参数为：
#垂直方向的波矢分量k0（单位：0.01nm^-1），
#均方根界面粗糙度sigmaDW（单位：nm），
#膜对序列d=[[1,d11,d12]……[n,dn1.dn2]]（膜层厚度单位：nm）
#输出：多层膜对入射波矢垂直分量为k0的中子的反射率
def RefRou(k0,sigmaDW,d=[]):
    M = np.ones([2,2],dtype =complex)
    M2 = np.ones([2,2],dtype =complex)
    M1 = np.ones([2,2],dtype =complex)
    k1 = cmath.sqrt(k0**2-4*math.pi*rhob1)
    k2 = cmath.sqrt(k0**2-4*math.pi*rhob2)
    ks = cmath.sqrt(k0**2-4*math.pi*rhobs)
    M[0][0] = (k2+ks)/(2*k2)
    M[0][1] = (k2-ks)/(2*k2)
    M[1][0] = (k2-ks)/(2*k2)
    M[1][1] = (k2+ks)/(2*k2)
    #对矩阵元素进行归一化，防止膜层数目过多时矩阵元素数值过大溢出
    M = M/np.sum(M**2)
    j = 0
    #每个膜对逐次计算反射率，从最底层迭代上去，psi=ikd为计算公式中e的复指数
    while j < len(d):
        psi2 = complex(0,k2*d[len(d)-1-j][2]*0.01)
        M2[0][0] = (k1+k2)*cmath.exp(-psi2)/(2*k1)
        M2[0][1] = (k1-k2)*cmath.exp(psi2)/(2*k1)
        M2[1][0] = (k1-k2)*cmath.exp(-psi2)/(2*k1)
        M2[1][1] = (k1+k2)*cmath.exp(psi2)/(2*k1)
        M = np.dot(M2,M)
        psi1 = complex(0,k1*d[len(d)-1-j][1]*0.01)
        M1[0][0] = (k1+k2)*cmath.exp(-psi1)/(2*k2)
        M1[0][1] = (k2-k1)*cmath.exp(psi1)/(2*k2)
        M1[1][0] = (k2-k1)*cmath.exp(-psi1)/(2*k2)
        M1[1][1] = (k1+k2)*cmath.exp(psi1)/(2*k2)
        M = np.dot(M1,M)
        #对矩阵元素进行归一化，防止膜层数目过多时矩阵元素数值过大溢出
        M = M/np.sum(M**2)
        j = j+1
    r = M[1][0]/M[0][0]
    #用Debye-Waller因子修正反射率
    R = abs(r)**2*math.exp(-(2*sigmaDW*k0*0.01)**2)
    return R

#Function1.1.4
#粗糙度为常数值条件下的反射率计算函数，采用矩阵方法
#输入参数为：
#垂直方向的波矢分量k0（单位：0.01nm^-1），
#均方根界面粗糙度sigmaDW（单位：nm），
#膜对序列d=[[1,d11,d12]……[n,dn1.dn2]]（膜层厚度单位：nm）
#输出：多层膜对入射波矢垂直分量为k0的中子的反射率
def RefRouCon(k0,sigmaDW,d=[]):
    M = np.ones([2,2],dtype =complex)
    M2 = np.ones([2,2],dtype =complex)
    M1 = np.ones([2,2],dtype =complex)
    k1 = cmath.sqrt(k0**2-4*math.pi*rhob1)
    k2 = cmath.sqrt(k0**2-4*math.pi*rhob2)
    ks = cmath.sqrt(k0**2-4*math.pi*rhobs)
    M[0][0] = (k2+ks)/(2*k2)*cmath.exp((sigmaDW*0.01)**2*k1*k2)
    M[0][1] = (k2-ks)/(2*k2)*cmath.exp(-(sigmaDW*0.01)**2*k1*k2)
    M[1][0] = (k2-ks)/(2*k2)*cmath.exp(-(sigmaDW*0.01)**2*k1*k2)
    M[1][1] = (k2+ks)/(2*k2)*cmath.exp((sigmaDW*0.01)**2*k1*k2)
    #对矩阵元素进行归一化，防止膜层数目过多时矩阵元素数值过大溢出
    M = M/np.sum(M**2)
    j = 0
    #每个膜对逐次计算反射率，从最底层迭代上去，psi=ikd为计算公式中e的复指数
    while j < len(d):
        psi2 = complex(0,k2*d[len(d)-1-j][2]*0.01)
        M2[0][0] = (k1+k2)*cmath.exp(-psi2)/(2*k1)*cmath.exp((sigmaDW*0.01)**2*k1*k2)
        M2[0][1] = (k1-k2)*cmath.exp(psi2)/(2*k1)*cmath.exp(-(sigmaDW*0.01)**2*k1*k2)
        M2[1][0] = (k1-k2)*cmath.exp(-psi2)/(2*k1)*cmath.exp(-(sigmaDW*0.01)**2*k1*k2)
        M2[1][1] = (k1+k2)*cmath.exp(psi2)/(2*k1)*cmath.exp((sigmaDW*0.01)**2*k1*k2)
        M = np.dot(M2,M)
        psi1 = complex(0,k1*d[len(d)-1-j][1]*0.01)
        M1[0][0] = (k1+k2)*cmath.exp(-psi1)/(2*k2)*cmath.exp((sigmaDW*0.01)**2*k1*k2)
        M1[0][1] = (k2-k1)*cmath.exp(psi1)/(2*k2)*cmath.exp(-(sigmaDW*0.01)**2*k1*k2)
        M1[1][0] = (k2-k1)*cmath.exp(-psi1)/(2*k2)*cmath.exp(-(sigmaDW*0.01)**2*k1*k2)
        M1[1][1] = (k1+k2)*cmath.exp(psi1)/(2*k2)*cmath.exp((sigmaDW*0.01)**2*k1*k2)
        M = np.dot(M1,M)
        #对矩阵元素进行归一化，防止膜层数目过多时矩阵元素数值过大溢出
        M = M/np.sum(M**2)
        j = j+1
    r = M[1][0]/M[0][0]
    return abs(r)**2

#Function1.1.5
#粗糙度逐层增长条件下的反射率计算函数,采用矩阵方法
#输入参数为：
#垂直方向的波矢分量k0（单位：0.01nm^-1），
#基底均方根粗糙度sigma0（单位：nm），
#粗糙度增长率h（单位：nm）,
#膜对序列d=[[1,d11,d12]……[n,dn1.dn2]]（膜层厚度单位：nm）
#输出：多层膜对入射波矢垂直分量为k0的中子的反射率
def RefRouGro(k0,sigma0,h,d=[]):
    M = np.ones([2,2],dtype =complex)
    M2 = np.ones([2,2],dtype =complex)
    M1 = np.ones([2,2],dtype =complex)
    k1 = cmath.sqrt(k0**2-4*math.pi*rhob1)
    k2 = cmath.sqrt(k0**2-4*math.pi*rhob2)
    ks = cmath.sqrt(k0**2-4*math.pi*rhobs)
    D = 0
    sigma = sigma0
    M[0][0] = (k2+ks)/(2*k2)*cmath.exp((sigma*0.01)**2*k1*k2)
    M[0][1] = (k2-ks)/(2*k2)*cmath.exp(-(sigma*0.01)**2*k1*k2)
    M[1][0] = (k2-ks)/(2*k2)*cmath.exp(-(sigma*0.01)**2*k1*k2)
    M[1][1] = (k2+ks)/(2*k2)*cmath.exp((sigma*0.01)**2*k1*k2)
    #对矩阵元素进行归一化，防止膜层数目过多时矩阵元素数值过大溢出
    M = M/np.sum(M**2)
    j = 0
    #每个膜对逐次计算反射率，从最底层迭代上去，psi=ikd为计算公式中e的复指数
    while j < len(d):
        #计算前面的膜层的累计厚度
        D = D+d[len(d)-1-j][2]
        sigma = math.sqrt(sigma0**2+h*D)
        psi2 = complex(0,k2*d[len(d)-1-j][2]*0.01)
        M2[0][0] = (k1+k2)*cmath.exp(-psi2)/(2*k1)*cmath.exp((sigma*0.01)**2*k1*k2)
        M2[0][1] = (k1-k2)*cmath.exp(psi2)/(2*k1)*cmath.exp(-(sigma*0.01)**2*k1*k2)
        M2[1][0] = (k1-k2)*cmath.exp(-psi2)/(2*k1)*cmath.exp(-(sigma*0.01)**2*k1*k2)
        M2[1][1] = (k1+k2)*cmath.exp(psi2)/(2*k1)*cmath.exp((sigma*0.01)**2*k1*k2)
        M = np.dot(M2,M)
        D = D+d[len(d)-1-j][1]
        sigma = math.sqrt(sigma0**2+h*D)
        psi1 = complex(0,k1*d[len(d)-1-j][1]*0.01)
        M1[0][0] = (k1+k2)*cmath.exp(-psi1)/(2*k2)*cmath.exp((sigma*0.01)**2*k1*k2)
        M1[0][1] = (k2-k1)*cmath.exp(psi1)/(2*k2)*cmath.exp(-(sigma*0.01)**2*k1*k2)
        M1[1][0] = (k2-k1)*cmath.exp(-psi1)/(2*k2)*cmath.exp(-(sigma*0.01)**2*k1*k2)
        M1[1][1] = (k1+k2)*cmath.exp(psi1)/(2*k2)*cmath.exp((sigma*0.01)**2*k1*k2)
        M = np.dot(M1,M)
        #对矩阵元素进行归一化，防止膜层数目过多时矩阵元素数值过大溢出
        M = M/np.sum(M**2)
        j = j+1
    r = M[1][0]/M[0][0]
    return abs(r)**2

#Function1.1.6
#粗糙度逐层增长且增长率随机的条件下的反射率计算函数,采用矩阵方法
#输入参数为：
#垂直方向的波矢分量k0（单位：0.01nm^-1），
#基底均方根粗糙度sigma0（单位：nm），
#粗糙度增长率的随机变化范围hmin,hmax（单位：nm）,
#膜对序列d=[[1,d11,d12]……[n,dn1.dn2]]（膜层厚度单位：nm）
#输出：多层膜对入射波矢垂直分量为k0的中子的反射率
def RefRouRan(k0,sigma0,hmin,hmax,d=[]):
    M = np.ones([2,2],dtype =complex)
    M2 = np.ones([2,2],dtype =complex)
    M1 = np.ones([2,2],dtype =complex)
    k1 = cmath.sqrt(k0**2-4*math.pi*rhob1)
    k2 = cmath.sqrt(k0**2-4*math.pi*rhob2)
    ks = cmath.sqrt(k0**2-4*math.pi*rhobs)
    D = 0
    sigma = sigma0
    M[0][0] = (k2+ks)/(2*k2)*cmath.exp((sigma*0.01)**2*k1*k2)
    M[0][1] = (k2-ks)/(2*k2)*cmath.exp(-(sigma*0.01)**2*k1*k2)
    M[1][0] = (k2-ks)/(2*k2)*cmath.exp(-(sigma*0.01)**2*k1*k2)
    M[1][1] = (k2+ks)/(2*k2)*cmath.exp((sigma*0.01)**2*k1*k2)
    #对矩阵元素进行归一化，防止膜层数目过多时矩阵元素数值过大溢出
    M = M/np.sum(M**2)
    j = 0
    #每个膜对逐次计算反射率，从最底层迭代上去，psi=ikd为计算公式中e的复指数
    while j < len(d):
        #计算前面的膜层的累计厚度
        D = D+d[len(d)-1-j][2]
        h = random.uniform(hmin,hmax)
        if sigma0**2+h*D > 0:
            sigma = math.sqrt(sigma0**2+h*D)
        else:
            sigma = 0
        psi2 = complex(0,k2*d[len(d)-1-j][2]*0.01)
        M2[0][0] = (k1+k2)*cmath.exp(-psi2)/(2*k1)*cmath.exp((sigma*0.01)**2*k1*k2)
        M2[0][1] = (k1-k2)*cmath.exp(psi2)/(2*k1)*cmath.exp(-(sigma*0.01)**2*k1*k2)
        M2[1][0] = (k1-k2)*cmath.exp(-psi2)/(2*k1)*cmath.exp(-(sigma*0.01)**2*k1*k2)
        M2[1][1] = (k1+k2)*cmath.exp(psi2)/(2*k1)*cmath.exp((sigma*0.01)**2*k1*k2)
        M = np.dot(M2,M)
        D = D+d[len(d)-1-j][1]
        h = random.uniform(hmin,hmax)
        if sigma0**2+h*D > 0:
            sigma = math.sqrt(sigma0**2+h*D)
        else:
            sigma = 0
        psi1 = complex(0,k1*d[len(d)-1-j][1]*0.01)
        M1[0][0] = (k1+k2)*cmath.exp(-psi1)/(2*k2)*cmath.exp((sigma*0.01)**2*k1*k2)
        M1[0][1] = (k2-k1)*cmath.exp(psi1)/(2*k2)*cmath.exp(-(sigma*0.01)**2*k1*k2)
        M1[1][0] = (k2-k1)*cmath.exp(-psi1)/(2*k2)*cmath.exp(-(sigma*0.01)**2*k1*k2)
        M1[1][1] = (k1+k2)*cmath.exp(psi1)/(2*k2)*cmath.exp((sigma*0.01)**2*k1*k2)
        M = np.dot(M1,M)
        #对矩阵元素进行归一化，防止膜层数目过多时矩阵元素数值过大溢出
        M = M/np.sum(M**2)
        j = j+1
    r = M[1][0]/M[0][0]
    return abs(r)**2

#Function1.1.7
#考虑材料吸收中子,且粗糙度为常数值条件下的反射率计算函数,采用矩阵方法
#输入参数为：
#垂直方向的波矢分量k0（单位：0.01nm^-1），
#入射中子波长lambda0（单位：nm），
#均方根界面粗糙度sigmaDW（单位：nm），
#膜对序列d=[[1,d11,d12]……[n,dn1.dn2]]（膜层厚度单位：nm）
#输出：多层膜对入射波矢垂直分量为k0的中子的反射率
def RefAbsRouCon(k0,lambda0,sigmaDW,d=[]):
    #计算两种材料的复数散射长度密度
    rhoba1 = rho1*6.022/A1*complex(b1,-sigma1/(2*lambda0)*10**(-4))
    rhoba2 = rho2*6.022/A2*complex(b2,-sigma2/(2*lambda0)*10**(-4))
    M = np.ones([2,2],dtype =complex)
    M2 = np.ones([2,2],dtype =complex)
    M1 = np.ones([2,2],dtype =complex)
    k1 = cmath.sqrt(k0**2-4*math.pi*rhoba1)
    k2 = cmath.sqrt(k0**2-4*math.pi*rhoba2)
    ks = cmath.sqrt(k0**2-4*math.pi*rhobs)
    M[0][0] = (k2+ks)/(2*k2)*cmath.exp((sigmaDW*0.01)**2*k1*k2)
    M[0][1] = (k2-ks)/(2*k2)*cmath.exp(-(sigmaDW*0.01)**2*k1*k2)
    M[1][0] = (k2-ks)/(2*k2)*cmath.exp(-(sigmaDW*0.01)**2*k1*k2)
    M[1][1] = (k2+ks)/(2*k2)*cmath.exp((sigmaDW*0.01)**2*k1*k2)
    #对矩阵元素进行归一化，防止膜层数目过多时矩阵元素数值过大溢出
    M = M/np.sum(M**2)
    j = 0
    #每个膜对逐次计算反射率，从最底层迭代上去，psi=ikd为计算公式中e的复指数
    while j < len(d):
        psi2 = complex(0,k2*d[len(d)-1-j][2]*0.01)
        M2[0][0] = (k1+k2)*cmath.exp(-psi2)/(2*k1)*cmath.exp((sigmaDW*0.01)**2*k1*k2)
        M2[0][1] = (k1-k2)*cmath.exp(psi2)/(2*k1)*cmath.exp(-(sigmaDW*0.01)**2*k1*k2)
        M2[1][0] = (k1-k2)*cmath.exp(-psi2)/(2*k1)*cmath.exp(-(sigmaDW*0.01)**2*k1*k2)
        M2[1][1] = (k1+k2)*cmath.exp(psi2)/(2*k1)*cmath.exp((sigmaDW*0.01)**2*k1*k2)
        M = np.dot(M2,M)
        psi1 = complex(0,k1*d[len(d)-1-j][1]*0.01)
        M1[0][0] = (k1+k2)*cmath.exp(-psi1)/(2*k2)*cmath.exp((sigmaDW*0.01)**2*k1*k2)
        M1[0][1] = (k2-k1)*cmath.exp(psi1)/(2*k2)*cmath.exp(-(sigmaDW*0.01)**2*k1*k2)
        M1[1][0] = (k2-k1)*cmath.exp(-psi1)/(2*k2)*cmath.exp(-(sigmaDW*0.01)**2*k1*k2)
        M1[1][1] = (k1+k2)*cmath.exp(psi1)/(2*k2)*cmath.exp((sigmaDW*0.01)**2*k1*k2)
        M = np.dot(M1,M)
        #对矩阵元素进行归一化，防止膜层数目过多时矩阵元素数值过大溢出
        M = M/np.sum(M**2)
        j = j+1
    r = M[1][0]/M[0][0]
    return abs(r)**2

#Function1.1.8
#考虑材料吸收中子,且粗糙度逐层增长条件下的反射率计算函数,采用矩阵方法
#输入参数为：
#垂直方向的波矢分量k0（单位：0.01nm^-1），
#基底均方根粗糙度sigma0（单位：nm），
#粗糙度增长率h（单位：nm）,
#膜对序列d=[[1,d11,d12]……[n,dn1.dn2]]（膜层厚度单位：nm）
#输出：多层膜对入射波矢垂直分量为k0的中子的反射率
def RefAbsRouGro(k0,lambda0,sigma0,h,d=[]):
    #计算两种材料的复数散射长度密度
    rhoba1 = rho1*6.022/A1*complex(b1,-sigma1/(2*lambda0)*10**(-4))
    rhoba2 = rho2*6.022/A2*complex(b2,-sigma2/(2*lambda0)*10**(-4))
    M = np.ones([2,2],dtype =complex)
    M2 = np.ones([2,2],dtype =complex)
    M1 = np.ones([2,2],dtype =complex)
    k1 = cmath.sqrt(k0**2-4*math.pi*rhoba1)
    k2 = cmath.sqrt(k0**2-4*math.pi*rhoba2)
    ks = cmath.sqrt(k0**2-4*math.pi*rhobs)
    D = 0
    sigma = sigma0
    M[0][0] = (k2+ks)/(2*k2)*cmath.exp((sigma*0.01)**2*k1*k2)
    M[0][1] = (k2-ks)/(2*k2)*cmath.exp(-(sigma*0.01)**2*k1*k2)
    M[1][0] = (k2-ks)/(2*k2)*cmath.exp(-(sigma*0.01)**2*k1*k2)
    M[1][1] = (k2+ks)/(2*k2)*cmath.exp((sigma*0.01)**2*k1*k2)
    #对矩阵元素进行归一化，防止膜层数目过多时矩阵元素数值过大溢出
    M = M/np.sum(M**2)
    j = 0
    #每个膜对逐次计算反射率，从最底层迭代上去，psi=ikd为计算公式中e的复指数
    while j < len(d):
        #计算前面的膜层的累计厚度
        D = D+d[len(d)-1-j][2]
        sigma = math.sqrt(sigma0**2+h*D)
        psi2 = complex(0,k2*d[len(d)-1-j][2]*0.01)
        M2[0][0] = (k1+k2)*cmath.exp(-psi2)/(2*k1)*cmath.exp((sigma*0.01)**2*k1*k2)
        M2[0][1] = (k1-k2)*cmath.exp(psi2)/(2*k1)*cmath.exp(-(sigma*0.01)**2*k1*k2)
        M2[1][0] = (k1-k2)*cmath.exp(-psi2)/(2*k1)*cmath.exp(-(sigma*0.01)**2*k1*k2)
        M2[1][1] = (k1+k2)*cmath.exp(psi2)/(2*k1)*cmath.exp((sigma*0.01)**2*k1*k2)
        M = np.dot(M2,M)
        D = D+d[len(d)-1-j][2]
        sigma = math.sqrt(sigma0**2+h*D)
        psi1 = complex(0,k1*d[len(d)-1-j][1]*0.01)
        M1[0][0] = (k1+k2)*cmath.exp(-psi1)/(2*k2)*cmath.exp((sigma*0.01)**2*k1*k2)
        M1[0][1] = (k2-k1)*cmath.exp(psi1)/(2*k2)*cmath.exp(-(sigma*0.01)**2*k1*k2)
        M1[1][0] = (k2-k1)*cmath.exp(-psi1)/(2*k2)*cmath.exp(-(sigma*0.01)**2*k1*k2)
        M1[1][1] = (k1+k2)*cmath.exp(psi1)/(2*k2)*cmath.exp((sigma*0.01)**2*k1*k2)
        M = np.dot(M1,M)
        #对矩阵元素进行归一化，防止膜层数目过多时矩阵元素数值过大溢出
        M = M/np.sum(M**2)
        j = j+1
    r = M[1][0]/M[0][0]
    return abs(r)**2

#----------1.2基于量子力学逐层迭代方法的反射率计算函数---------------------------#

#Function1.2.1
#理想条件（无粗糙度）下的反射率计算函数,采用量子力学迭代计算
#输入参数为：
#垂直方向的波矢分量k0（单位：0.01nm^-1），
#膜对序列d=[[1,d11,d12]……[n,dn1.dn2]]（膜层厚度单位：nm）
#输出：多层膜对入射波矢垂直分量为k0的中子的反射率
def Ref2(k0,d=[]):
    k1 = cmath.sqrt(k0**2-4*math.pi*rhob1)
    k2 = cmath.sqrt(k0**2-4*math.pi*rhob2)
    ks = cmath.sqrt(k0**2-4*math.pi*rhobs)
    r = (k2-ks)/(k2+ks)
    k = 0
    #每个膜对逐次计算反射率，从最底层迭代上去，psi=2ikd为计算公式中e的复指数
    while k < len(d):
        R = (k1-k2)/(k1+k2)
        psi = complex(0,2*k2*d[len(d)-1-k][2]*0.01)
        r = (R+r*cmath.exp(psi))/(1+R*r*cmath.exp(psi))
        R = (k2-k1)/(k1+k2)
        psi = complex(0,2*k1*d[len(d)-1-k][1]*0.01)
        r = (R+r*cmath.exp(psi))/(1+R*r*cmath.exp(psi))
        k = k+1
    return abs(r)**2

#Function1.2.2
#考虑材料吸收中子条件下的反射率计算函数,采用量子力学迭代计算
#输入参数为：
#垂直方向的波矢分量k0（单位：0.01nm^-1），
#入射中子波长lambda0（单位：nm），
#膜对序列d=[[1,d11,d12]……[n,dn1.dn2]]（膜层厚度单位：nm）
#输出：多层膜对入射波矢垂直分量为k0的中子的反射率
def RefAbs2(k0,lambda0,d=[]):
    #计算两种材料的复数散射长度密度
    rhoba1 = rho1*6.022/A1*complex(b1,-sigma1/(2*lambda0)*10**(-4))
    rhoba2 = rho2*6.022/A2*complex(b2,-sigma2/(2*lambda0)*10**(-4))
    k1 = cmath.sqrt(k0**2-4*math.pi*rhoba1)
    k2 = cmath.sqrt(k0**2-4*math.pi*rhoba2)
    ks = cmath.sqrt(k0**2-4*math.pi*rhobs)
    r = (k2-ks)/(k2+ks)
    j = 0
    #每个膜对逐次计算反射率，从最底层迭代上去，psi=2ikd为计算公式中e的复指数
    while j < len(d):
        R = (k1-k2)/(k1+k2)
        psi = complex(0,2*k2*d[len(d)-1-j][2]*0.01)
        r = (R+r*cmath.exp(psi))/(1+R*r*cmath.exp(psi))
        R = (k2-k1)/(k1+k2)
        psi = complex(0,2*k1*d[len(d)-1-j][1]*0.01)
        r = (R+r*cmath.exp(psi))/(1+R*r*cmath.exp(psi))
        j = j+1
    return abs(r)**2

#Function1.2.3
#粗糙度为常数值条件下的反射率计算函数，采用量子力学迭代计算
#输入参数为：
#垂直方向的波矢分量k0（单位：0.01nm^-1），
#均方根界面粗糙度sigmaDW（单位：nm），
#膜对序列d=[[1,d11,d12]……[n,dn1.dn2]]（膜层厚度单位：nm）
#输出：多层膜对入射波矢垂直分量为k0的中子的反射率
def RefRouCon2(k0,sigmaDW,d=[]):
    k1 = cmath.sqrt(k0**2-4*math.pi*rhob1)
    k2 = cmath.sqrt(k0**2-4*math.pi*rhob2)
    ks = cmath.sqrt(k0**2-4*math.pi*rhobs)
    r = (k2-ks)/(k2+ks)*cmath.exp(-(2*sigmaDW*0.01)**2*k1*k2/2)
    j = 0
    #每个膜对逐次计算反射率，从最底层迭代上去，psi=2ikd为计算公式中e的复指数
    while j < len(d):
        R = (k1-k2)/(k1+k2)*cmath.exp(-(2*sigmaDW*0.01)**2*k1*k2/2)
        psi = complex(0,2*k2*d[len(d)-1-j][2]*0.01)
        r = (R+r*cmath.exp(psi))/(1+R*r*cmath.exp(psi))
        R = (k2-k1)/(k1+k2)*cmath.exp(-(2*sigmaDW*0.01)**2*k1*k2/2)
        psi = complex(0,2*k1*d[len(d)-1-j][1]*0.01)
        r = (R+r*cmath.exp(psi))/(1+R*r*cmath.exp(psi))
        j = j+1
    return abs(r)**2

#Function1.2.4
#粗糙度逐层增长条件下的反射率计算函数,采用量子力学迭代计算
#输入参数为：
#垂直方向的波矢分量k0（单位：0.01nm^-1），
#基底均方根粗糙度sigma0（单位：nm），
#粗糙度增长率h（单位：nm）,
#膜对序列d=[[1,d11,d12]……[n,dn1.dn2]]（膜层厚度单位：nm）
#输出：多层膜对入射波矢垂直分量为k0的中子的反射率
def RefRouGro2(k0,sigma0,h,d=[]):
    k1 = cmath.sqrt(k0**2-4*math.pi*rhob1)
    k2 = cmath.sqrt(k0**2-4*math.pi*rhob2)
    ks = cmath.sqrt(k0**2-4*math.pi*rhobs)
    D = 0
    sigma = sigma0
    r = (k2-ks)/(k2+ks)*cmath.exp(-(2*sigma*0.01)**2*k1*k2/2)
    j = 0
    #每个膜对逐次计算反射率，从最底层迭代上去，psi=2ikd为计算公式中e的复指数
    while j < len(d):
        #计算前面的膜层的累计厚度
        D = D+d[len(d)-1-j][2]
        sigma = math.sqrt(sigma0**2+h*D)
        R = (k1-k2)/(k1+k2)*cmath.exp(-(2*sigma*0.01)**2*k1*k2/2)
        psi = complex(0,2*k2*d[len(d)-1-j][2]*0.01)
        r = (R+r*cmath.exp(psi))/(1+R*r*cmath.exp(psi))
        D = D+d[len(d)-1-j][1]
        sigma = math.sqrt(sigma0**2+h*D)
        R = (k2-k1)/(k1+k2)*cmath.exp(-(2*sigma*0.01)**2*k1*k2/2)
        psi = complex(0,2*k1*d[len(d)-1-j][1]*0.01)
        r = (R+r*cmath.exp(psi))/(1+R*r*cmath.exp(psi))
        j = j+1
    return abs(r)**2

#Function1.2.5
#粗糙度逐层增长且增长率随机的条件下的反射率计算函数,采用量子力学迭代计算
#输入参数为：
#垂直方向的波矢分量k0（单位：0.01nm^-1），
#基底均方根粗糙度sigma0（单位：nm），
#粗糙度增长率的随机变化范围hmin,hmax（单位：nm）,
#膜对序列d=[[1,d11,d12]……[n,dn1.dn2]]（膜层厚度单位：nm）
#输出：多层膜对入射波矢垂直分量为k0的中子的反射率
def RefRouRan2(k0,sigma0,hmin,hmax,d=[]):
    k1 = cmath.sqrt(k0**2-4*math.pi*rhob1)
    k2 = cmath.sqrt(k0**2-4*math.pi*rhob2)
    ks = cmath.sqrt(k0**2-4*math.pi*rhobs)
    D = 0
    sigma = sigma0
    r = (k2-ks)/(k2+ks)*cmath.exp(-(2*sigma*0.01)**2*k1*k2/2)
    j = 0
    #每个膜对逐次计算反射率，从最底层迭代上去，psi=2ikd为计算公式中e的复指数
    while j < len(d):
        #计算前面的膜层的累计厚度
        D = D+d[len(d)-1-j][2]
        h = random.uniform(hmin,hmax)
        if sigma0**2+h*D > 0:
            sigma = math.sqrt(sigma0**2+h*D)
        else:
            sigma = 0
        #sigma = math.sqrt(abs(sigma0**2+random.uniform(hmin,hmax)*D))
        R = (k1-k2)/(k1+k2)*cmath.exp(-(2*sigma*0.01)**2*k1*k2/2)
        psi = complex(0,2*k2*d[len(d)-1-j][2]*0.01)
        r = (R+r*cmath.exp(psi))/(1+R*r*cmath.exp(psi))
        D = D+d[len(d)-1-j][1]
        h = random.uniform(hmin,hmax)
        if sigma0**2+h*D > 0:
            sigma = math.sqrt(sigma0**2+h*D)
        else:
            sigma = 0
        #sigma = math.sqrt(abs(sigma0**2+random.uniform(hmin,hmax)*D))
        R = (k2-k1)/(k1+k2)*cmath.exp(-(2*sigma*0.01)**2*k1*k2/2)
        psi = complex(0,2*k1*d[len(d)-1-j][1]*0.01)
        r = (R+r*cmath.exp(psi))/(1+R*r*cmath.exp(psi))
        j = j+1
    return abs(r)**2

#Function1.2.6
#考虑材料吸收中子,且粗糙度为常数值条件下的反射率计算函数,采用量子力学迭代计算
#输入参数为：
#垂直方向的波矢分量k0（单位：0.01nm^-1），
#入射中子波长lambda0（单位：nm），
#均方根界面粗糙度sigmaDW（单位：nm），
#膜对序列d=[[1,d11,d12]……[n,dn1.dn2]]（膜层厚度单位：nm）
#输出：多层膜对入射波矢垂直分量为k0的中子的反射率
def RefAbsRouCon2(k0,lambda0,sigmaDW,d=[]):
    #计算两种材料的复数散射长度密度
    rhoba1 = rho1*6.022/A1*complex(b1,-sigma1/(2*lambda0)*10**(-4))
    rhoba2 = rho2*6.022/A2*complex(b2,-sigma2/(2*lambda0)*10**(-4))
    k1 = cmath.sqrt(k0**2-4*math.pi*rhoba1)
    k2 = cmath.sqrt(k0**2-4*math.pi*rhoba2)
    ks = cmath.sqrt(k0**2-4*math.pi*rhobs)
    r = (k2-ks)/(k2+ks)*cmath.exp(-(2*sigmaDW*0.01)**2*k1*k2/2)
    j = 0
    #每个膜对逐次计算反射率，从最底层迭代上去，psi=2ikd为计算公式中e的复指数
    while j < len(d):
        R = (k1-k2)/(k1+k2)*cmath.exp(-(2*sigmaDW*0.01)**2*k1*k2/2)
        psi = complex(0,2*k2*d[len(d)-1-j][2]*0.01)
        r = (R+r*cmath.exp(psi))/(1+R*r*cmath.exp(psi))
        R = (k2-k1)/(k1+k2)*cmath.exp(-(2*sigmaDW*0.01)**2*k1*k2/2)
        psi = complex(0,2*k1*d[len(d)-1-j][1]*0.01)
        r = (R+r*cmath.exp(psi))/(1+R*r*cmath.exp(psi))
        j = j+1
    return abs(r)**2

#Function1.2.7
#考虑材料吸收中子,且粗糙度逐层增长条件下的反射率计算函数,采用量子力学迭代计算
#输入参数为：
#垂直方向的波矢分量k0（单位：0.01nm^-1），
#基底均方根粗糙度sigma0（单位：nm），
#粗糙度增长率h（单位：nm）,
#膜对序列d=[[1,d11,d12]……[n,dn1.dn2]]（膜层厚度单位：nm）
#输出：多层膜对入射波矢垂直分量为k0的中子的反射率
def RefAbsRouGro2(k0,lambda0,sigma0,h,d=[]):
    #计算两种材料的复数散射长度密度
    rhoba1 = rho1*6.022/A1*complex(b1,-sigma1/(2*lambda0)*10**(-4))
    rhoba2 = rho2*6.022/A2*complex(b2,-sigma2/(2*lambda0)*10**(-4))
    k1 = cmath.sqrt(k0**2-4*math.pi*rhoba1)
    k2 = cmath.sqrt(k0**2-4*math.pi*rhoba2)
    ks = cmath.sqrt(k0**2-4*math.pi*rhobs)
    D = 0
    sigma = sigma0
    r = (k2-ks)/(k2+ks)*cmath.exp(-(2*sigma*0.01)**2*k1*k2/2)
    j = 0
    #每个膜对逐次计算反射率，从最底层迭代上去，psi=2ikd为计算公式中e的复指数
    while j < len(d):
        #计算前面的膜层的累计厚度
        D = D+d[len(d)-1-j][2]
        sigma = math.sqrt(sigma0**2+h*D)
        R = (k1-k2)/(k1+k2)*cmath.exp(-(2*sigma*0.01)**2*k1*k2/2)
        psi = complex(0,2*k2*d[len(d)-1-j][2]*0.01)
        r = (R+r*cmath.exp(psi))/(1+R*r*cmath.exp(psi))
        D = D+d[len(d)-1-j][1]
        sigma = math.sqrt(sigma0**2+h*D)
        R = (k2-k1)/(k1+k2)*cmath.exp(-(2*sigma*0.01)**2*k1*k2/2)
        psi = complex(0,2*k1*d[len(d)-1-j][1]*0.01)
        r = (R+r*cmath.exp(psi))/(1+R*r*cmath.exp(psi))
        j = j+1
    return abs(r)**2


##########2平均反射率计算函数###################################################
    
#Function2.1
#计算反射率曲线在1~m范围内的平均反射率
#输入参数为：
#反射率曲线的m值，
#一定范围的波矢垂直分量x（单位：0.01nm^-1），
#与该波矢垂直分量对应的反射率y
#输出：反射率曲线在1~m范围内的平均反射率
def AveR(m,x=[],y=[]):
    A = 0
    i = 0
    #面积法（直接除以(m-1)是不准确的，因为两头的点并没有刚好落在1和m上）
    #while x[i]/kc <= m:
    #   if x[i]/kc >= 1:
    #        A = A+(x[i+1]-x[i])/kc*y[i]
    #    i = i+1    
    #A = A/(m-1)
    #直接对所有点的反射率求算数平均
    j = 0
    while x[i]/kc <= m:
        if x[i]/kc >= 1:
            A = A+y[i]
            j = j+1
        i = i+1    
    if j != 0:
        A = A/j
    return A

#Function2.2
#计算反射率曲线在0~m范围内的平均反射率
#输入参数为：
#反射率曲线的m值，
#一定范围的波矢垂直分量x（单位：0.01nm^-1），
#与该波矢垂直分量对应的反射率y
#输出：反射率曲线在0~m范围内的平均反射率
def AveRF(m,x=[],y=[]):
    A = 0
    i = 0
    j = 0
    while x[i]/kc <= m:
        if x[i]/kc >= 0:
            A = A+y[i]
            j = j+1
        i = i+1    
    if j != 0:
        A = A/j
    return A


##########3超镜算法函数#########################################################
    
#Function3.1
#SM算法
#输入参数：
#m值,
#beta值（影响期望达到的反射率）
#输出：膜对序列d=[[1,d11,d12]……[n,dn1.dn2]]（膜层厚度单位：nm）
def SM(m,beta):
    #计算两种材料的全反射临界波矢（Ti的为平方值）
    qc1 = 2*math.sqrt(math.pi*rhob1)
    qc22 = 4*math.pi*rhob2
    Q12 = qc1/math.sqrt(qc1**2-qc22)
    b = beta*Q12**4+2*beta*Q12**2-1
    #迭代计算过程
    #计算所需膜对数
    N = math.ceil(beta*((m**2*Q12**2+1)**2-1)-b)
    #表面加一层厚厚的Ni
    d = [[1,70,0]]
    #迭代起点
    #前五层单独处理
    D0 = math.pi/(1.005*qc1)*100
    d.append([2,D0/2,D0/2])
    d.append([3,D0/2,D0/2])
    d.append([4,D0/2,D0/2])
    d.append([5,D0/2,D0/2])
    d.append([6,D0/2,D0/2])
    #依次迭代
    #从n=4开始也是为了利用上n=4,5的厚度，避免前五层与第六层厚度差异太大，出现反射率下掉
    n = 4
    while n < N:
        D = Q12*math.pi/(qc1*((1+beta**(-1)*(n+b))**(1/2)-1)**(1/2))*100
        d.append([n+3,D/2,D/2])
        n = n+1
    return d

#Function3.2
#RSD算法
#输入参数：
#设计反射率RL，
#m值
#输出：膜对序列d=[[1,d11,d12]……[n,dn1.dn2]]（膜层厚度单位：nm）
def RSD(m,RL):
    #计算特征波长
    lambdaNi = math.sqrt(math.pi/rhob1)*100
    lambdaTi2 = math.pi/rhob2*10000
    lambdaSM = lambdaNi/m
    #Ni的全反射临界波矢
    kNi = 2*math.pi/lambdaNi*100
    #迭代计算过程
    #迭代起点	
    k = m*kNi
    D = lambdaSM/2
    d1 = D/(2*math.sqrt(1-(2*D/lambdaNi)**2))
    d2 = D/(2*math.sqrt(1-(2*D**2/lambdaTi2)))
    N = 1
    while BraggHeightandWidth(N,d1,d2,k)[0] <= RL:
        N = N+1
    dk = BraggHeightandWidth(N,d1,d2,k)[1]
    M = [[1,d1,d2]]
    #依次迭代
    j = 1
    while 0 < D < lambdaNi/2:
        k = k-dk/(1.2*N)
        D = lambdaNi/(2*(k/kNi))
        if 0 < D < lambdaNi/2:
            d1 = D/(2*math.sqrt(1-(2*D/lambdaNi)**2))
            d2 = D/(2*math.sqrt(1-(2*D**2/lambdaTi2)))
            N = 1
            while BraggHeightandWidth(N,d1,d2,k)[0] <= RL:
                N = N+1
            dk = BraggHeightandWidth(N,d1,d2,k)[1]
            j = j+1
            M.append([j,d1,d2])
    #由于最表面的Ni层厚度太薄，因此额外加一层60nm厚的Ni层。		
    M.append([j+1,60,0])
    #为配合反射率计算函数，将膜对顺序反过来，从最表面开始
    d = np.ones([len(M),3])
    i = 0
    while i < len(M):
        d[i] = [i+1,M[len(M)-i-1][1],M[len(M)-i-1][2]]
        i = i+1
    return d

#Function3.3
#GRB算法
def GRB(m,R):
    #输入参数：设计反射率R，m值    
    alpha =4/(math.atanh(math.sqrt(R)))**2
    #中间参数计算
    #特征波长
    lambdac = math.sqrt(math.pi/(rhob1-rhob2))*100
    lambdaNi = math.sqrt(math.pi/rhob1)*100
    lambdaTi2 = math.pi/rhob2*10000
    lambdaSM = lambdaNi/m
    #计算过程：
    #迭代起点
    #n为膜层序数,迭代到最后即为膜层总数
    n = 1
    D = lambdaNi/2-0.2
    N = math.ceil(math.sqrt(1/(alpha*(D/lambdac)**4)))
    #N = math.ceil(math.atanh(math.sqrt(R))/(2*(D/lambdac)**2))
    #以上两种N的表达式是等价的
    d1 = D/(2*math.sqrt(1-(2*D/lambdaNi)**2))
    d2 = D/(2*math.sqrt(1-(2*D**2/lambdaTi2)))
    #用一个矩阵存储这些数据
    d = [[n,d1,d2]]
    #依次迭代
    while D > lambdaSM/2:
        n = n+1
        D = D*(1-N**(-2))
        N = math.ceil(math.sqrt(1/(alpha*(D/lambdac)**4)))
        #N = math.ceil(math.atanh(math.sqrt(R))/(2*(D/lambdac)**2))
        #以上两种N的表达式是等价的
        d1 = D/(2*math.sqrt(1-(2*D/lambdaNi)**2))
        d2 = D/(2*math.sqrt(1-(2*D**2/lambdaTi2)))
        d.append([n,d1,d2])
    return d

#Function3.4
#EAYAO算法
def EAYAO(N):
    #输入参数：膜对数目N   
    #迭代计算过程
    d = np.zeros([N,3])
    #迭代起点
    d1 = 50
    d2 = 20
    d[0] = [1,d1,d2]
    n = 1
    #依次递推
    while n < N:
        d1 = d2-2*(20/d2)**(-4.5)
        d2 = d1-2*(20/d1)**(-4.5)
        d[n] = [n+1,d1,d2]
        n = n+1
    return d

#Function3.5
#Improved Mezei算法
def IM(N):
    #输入参数：膜对数目N
    v = N
    #入射中子波长为0.2nm
    lambda0 = 0.2
    #中间参数计算
    #Ni的布拉格衍射厚度
    dc = lambda0/(2*math.asin(lambda0*math.sqrt(rhob1/math.pi)*0.01))
    #迭代计算过程
    #迭代起点	
    #xi0 = 2-math.sqrt(2)
    xi0 = 0.69
    #依次迭代
    n = 1
    xi = np.ones(v)
    xi = xi+10
    #牛顿迭代法求解
    while n <= v:
        x = xi[n-1]
        F = x**4-2*(n**(1/4))*xi0*(x**3)+x**2-2*(n**(1/4))*xi0*x+(n**(1/2))*xi0**2
        dF = 4*(x**3)-6*(n**(1/4))*xi0*(x**2)+2*x-2*(n**(1/4))*xi0
        if F == 0:
            xi[n-1] = x
            n = n+1
        else:
            while abs((F/dF)/x) > 10**(-8):
                x = x-F/dF
                F = x**4-2*(n**(1/4))*xi0*(x**3)+x**2-2*(n**(1/4))*xi0*x+(n**(1/2))*xi0**2
                dF = 4*(x**3)-6*(n**(1/4))*xi0*(x**2)+2*x-2*(n**(1/4))*xi0
            else:
                xi[n-1] = x
                n = n+1
    #生成膜对序列
    dd = 0.5
    j = 1
    d = np.ones([v+1,3])
    #最表面加一层厚厚的Ni
    d[0] = [1,60,0]
    for x in xi:
        d1 = dc/x
        d2 = (d1**(-2)+dc**(-2))**(-1/2)
        d[j] = [j+1,d1-dd,d2-dd]
        j = j+1
    return d

#Function3.6
#IC算法
def IC(m,R):
    #输入参数：设计反射率R，m值
    #Ni的全反射临界波矢
    kc = 2*math.sqrt(math.pi*rhob1)
    #迭代计算过程
    #迭代起点
    k = 1.18*kc
    #计算两种材料中的波矢
    k1 = math.sqrt(k**2-4*math.pi*rhob1)
    k2 = math.sqrt(k**2-4*math.pi*rhob2)
    #计算两种材料界面的反射系数
    r21 = (k2-k1)/(k2+k1)
    #计算两层膜的厚度
    d1 = math.pi/(2*k1)*100
    d2 = math.pi/(2*k2)*100
    #计算所需膜对数量
    zeta = (1-R)/2
    N = math.ceil(abs(math.log(zeta))/(2*math.sqrt(2)*r21))
    #计算布拉格峰宽度
    dk = math.sqrt(2)*k*r21/math.pi
    #记录膜对数量和膜层厚度数据
    M = [[N,d1,d2]]
    #依次迭代
    while k <= m*kc:
        k = k+2*dk/1.2
        k1 = math.sqrt(k**2-4*math.pi*rhob1)
        k2 = math.sqrt(k**2-4*math.pi*rhob2)
        r21 = (k2-k1)/(k2+k1)
        d1 = math.pi/(2*k1)*100
        d2 = math.pi/(2*k2)*100
        N = math.ceil(abs(math.log(zeta))/(2*math.sqrt(2)*r21))
        dk = math.sqrt(2)*k*r21/math.pi
        M.append([N,d1,d2])
    #最后一项其实k已经超过m*kc了，但是还是被记录了下来，应该删掉
    M = np.delete(M,len(M)-1,axis=0) 
    #计算总膜对数
    v = 0
    u = 0
    while u < len(M):
        v = v+M[u][0]
        u = u+1  
    #修补k=1.28kc处的反射率下降
    k = 1.28*kc
    k1 = math.sqrt(k**2-4*math.pi*rhob1)
    k2 = math.sqrt(k**2-4*math.pi*rhob2)
    d1 = math.pi/(2*k1)*100
    d2 = math.pi/(2*k2)*100
    #原文在膜对最表面和最底层各加上一层Ni
    #在k=1.28kc处补上五个相同膜对
    d = [[1,10/kc*100,0],
         [2,d1,d2],
         [3,d1,d2],
         [4,d1,d2],
         [5,d1,d2],
         [6,d1,d2]]
    #生成膜对序列
    i = 0
    j = 6
    while i < len(M):
        n = 0
        while n < M[i][0]:
            j = j+1
            d1 = M[i][1]
            d2 = M[i][2]
            d.append([j,d1,d2])
            n = n+1
        i = i+1
    #原文在膜对最表面和最底层各加上一层Ni
    d.append([j+1,10/kc*100,0])
    v = v+1+5+1
    return d

#Function3.7
#Masalovich算法
def Mas(m,R):
    #输入参数：设计反射率，m值
    #迭代计算过程
    #迭代起点
    #计算ρ，为Ti和Ni的散射长度密度之比
    rho = rhob2/rhob1
    #计算透射系数
    C = math.sqrt((m*m-1)/(m*m-rho))
    #计算Ni的全反射临界波矢kc
    kc = math.sqrt(4*math.pi*rhob1)
    #计算膜对厚度（单位：nm）
    D = math.pi*(1/math.sqrt(m*m-1)+1/math.sqrt(m*m-rho))/(2*kc)*100
    #计算两个单层分别的厚度
    a = D/(1+C)
    b = C*a
    #计算所需膜对数K
    #K = math.ceil(math.log(1-R)/math.log(C**2))
    K = 1
    #用一个矩阵记录m值，膜对数，膜层厚度
    M = [[m,K,a,b]]
    #依次迭代
    while m >= 1.05:
        #迭代计算透射率
        C = C+C*(1-C**2)*math.log(C**2)**2/((C-C**2-1)*math.log(1-R)**2)
        #计算膜对厚度
        D = D+D*math.log(C**2)**2/math.log(1-R)**2
        #计算两个单层的厚度
        a = D/(1+C)
        b = a*C
        #计算所需层数
        #K = math.ceil(math.log(1-R)/math.log(C**2))
        #计算超镜总膜对数
        K = K+1
        #计算峰位置对应的m值
        m = math.sqrt((C**2*rho-1)/(C**2-1))
        M.append([m,K,a,b])
    #生成膜对序列    
    d = np.zeros([K,3])
    j = 0
    while j < K:
        d[K-j-1] = [K-j,M[j][2],M[j][3]]
        j = j+1
    return d

#Function3.8
#ABC算法函数
def ABC(N):
    a = 48
    b = -0.16
    c = 0.3
    Gamma = 0.52
    d = np.zeros([N,3])
    n = 1
    while n <= N:
        D = a/(n+b)**c
        d[n-1] = [n,D*Gamma,D*(1-Gamma)]
        n = n+1
    return d


##########4超镜算法调参函数#####################################################
    
#Function4.1
#Improved Mezei算法调参函数
#待优化参数：方程初值xi0，膜对厚度变化dd
def IMtest(N,xi0,dd):
    #输入参数：膜对数目N
    v = N
    #入射中子波长为0.2nm
    lambda0 = 0.2
    #中间参数计算
    #Ni的布拉格衍射厚度
    dc = lambda0/(2*math.asin(lambda0*math.sqrt(rhob1/math.pi)*0.01))
    #迭代计算过程
    #迭代起点	
    #xi0 = 2-math.sqrt(2)
    #xi0 = 0.69
    #依次迭代
    n = 1
    xi = np.ones(v)
    xi = xi+10
    #牛顿迭代法求解
    while n <= v:
        x = xi[n-1]
        F = x**4-2*(n**(1/4))*xi0*(x**3)+x**2-2*(n**(1/4))*xi0*x+(n**(1/2))*xi0**2
        dF = 4*(x**3)-6*(n**(1/4))*xi0*(x**2)+2*x-2*(n**(1/4))*xi0
        if F == 0:
            xi[n-1] = x
            n = n+1
        else:
            while abs((F/dF)/x) > 10**(-8):
                x = x-F/dF
                F = x**4-2*(n**(1/4))*xi0*(x**3)+x**2-2*(n**(1/4))*xi0*x+(n**(1/2))*xi0**2
                dF = 4*(x**3)-6*(n**(1/4))*xi0*(x**2)+2*x-2*(n**(1/4))*xi0
            else:
                xi[n-1] = x
                n = n+1
    #生成膜对序列
    #dd = 0.5
    j = 1
    d = np.ones([v+1,3])
    #最表面加一层厚厚的Ni
    d[0] = [1,60,0]
    for x in xi:
        d1 = dc/x
        d2 = (d1**(-2)+dc**(-2))**(-1/2)
        d[j] = [j+1,d1-dd,d2-dd]
        j = j+1
    return d

#Function4.2
#SM算法调参函数
#待优化参数：R,膜对厚度比例Gamma
def SMtest(m,beta,R,Gamma):
    #输入参数：m值,beta值（与反射率有关）
    #中间参数计算
    #计算两种材料的全反射临界波矢（Ti的为平方值）
    qc1 = 2*math.sqrt(math.pi*rhob1)
    qc22 = 4*math.pi*rhob2
    Q12 = R**(1/4)*qc1/math.sqrt(qc1**2-qc22)
    b = beta*Q12**4+2*beta*Q12**2-1
    #迭代计算过程
    #计算所需膜对数
    N = math.ceil(beta*((m**2*Q12**2+1)**2-1)-b)
    #表面加一层厚厚的Ni
    d = [[1,70,0]]
    #迭代起点
    #前五层单独处理
    D0 = math.pi/(1.005*qc1)*100
    d.append([2,D0/2,D0/2])
    d.append([3,D0/2,D0/2])
    d.append([4,D0/2,D0/2])
    d.append([5,D0/2,D0/2])
    d.append([6,D0/2,D0/2])
    #依次迭代
    #从n=4开始也是为了利用上n=4,5的厚度，避免前五层与第六层厚度差异太大，出现反射率下掉
    n = 4
    while n < N:
        D = Q12*math.pi/(qc1*((1+beta**(-1)*(n+b))**(1/2)-1)**(1/2))*100
        d.append([n+3,D*Gamma,D*(1-Gamma)])
        n = n+1
    return d

#Function4.3
#ABC算法调参函数
def ABCtest(N,a,b,c,Gamma):
    d = np.zeros([N,3])
    n = 1
    while n <= N:
        D = a/(n+b)**c
        d[n-1] = [n,D*Gamma,D*(1-Gamma)]
        n = n+1
    return d

#Function4.4
#EAYAO算法调参函数
#待优化参数：厚度差上限delta,指数a
def EAYAOtest(N,delta,a):
    #输入参数：膜对数目N   
    #迭代计算过程
    d = np.zeros([N,3])
    #迭代起点
    d1 = 50
    d2 = 20
    d[0] = [1,d1,d2]
    n = 1
    #依次递推
    while n < N:
        d1 = d2-delta*(20/d2)**(-a)
        d2 = d1-delta*(20/d1)**(-a)
        d[n] = [n+1,d1,d2]
        n = n+1
    return d

#Function4.5
#IC算法调参函数
def ICtest(m,R,a,w):
    #输入参数：设计反射率R，m值
    #Ni的全反射临界波矢
    kc = 2*math.sqrt(math.pi*rhob1)
    #迭代计算过程
    #迭代起点
    k = a*kc
    #计算两种材料中的波矢
    k1 = math.sqrt(k**2-4*math.pi*rhob1)
    k2 = math.sqrt(k**2-4*math.pi*rhob2)
    #计算两种材料界面的反射系数
    r21 = (k2-k1)/(k2+k1)
    #计算两层膜的厚度
    d1 = math.pi/(2*k1)*100
    d2 = math.pi/(2*k2)*100
    #计算所需膜对数量
    zeta = (1-R)/2
    N = math.ceil(abs(math.log(zeta))/(2*math.sqrt(2)*r21))
    #计算布拉格峰宽度
    dk = math.sqrt(2)*k*r21/math.pi
    #记录膜对数量和膜层厚度数据
    M = [[N,d1,d2]]
    #依次迭代
    while k <= m*kc:
        k = k+2*dk/w
        k1 = math.sqrt(k**2-4*math.pi*rhob1)
        k2 = math.sqrt(k**2-4*math.pi*rhob2)
        r21 = (k2-k1)/(k2+k1)
        d1 = math.pi/(2*k1)*100
        d2 = math.pi/(2*k2)*100
        N = math.ceil(abs(math.log(zeta))/(2*math.sqrt(2)*r21))
        dk = math.sqrt(2)*k*r21/math.pi
        M.append([N,d1,d2])
    #最后一项其实k已经超过m*kc了，但是还是被记录了下来，应该删掉
    M = np.delete(M,len(M)-1,axis=0) 
    #计算总膜对数
    v = 0
    u = 0
    while u < len(M):
        v = v+M[u][0]
        u = u+1  
    #修补k=1.28kc处的反射率下降
    k = 1.28*kc
    k1 = math.sqrt(k**2-4*math.pi*rhob1)
    k2 = math.sqrt(k**2-4*math.pi*rhob2)
    d1 = math.pi/(2*k1)*100
    d2 = math.pi/(2*k2)*100
    #原文在膜对最表面和最底层各加上一层Ni
    #在k=1.28kc处补上五个相同膜对
    d = [[1,10/kc*100,0],
         [2,d1,d2],
         [3,d1,d2],
         [4,d1,d2],
         [5,d1,d2],
         [6,d1,d2]]
    #生成膜对序列
    i = 0
    j = 6
    while i < len(M):
        n = 0
        while n < M[i][0]:
            j = j+1
            d1 = M[i][1]
            d2 = M[i][2]
            d.append([j,d1,d2])
            n = n+1
        i = i+1
    #原文在膜对最表面和最底层各加上一层Ni
    d.append([j+1,10/kc*100,0])
    v = v+1+5+1
    return d

    
##########5计算n个膜对产生的布拉格峰的高度和宽度的函数############################
		
#Function5
def BraggHeightandWidth(N,d1,d2,k0):
    #生成一个由n个相同膜对构成的多层膜序列
    d = np.ones([N,3])
    i = 0
    while i < N:
        d[i] = [i+1,d1,d2]
        i = i+1
    #计算最大反射率
    Rmax = Ref2(k0,d)
    #寻找布拉格峰右侧的半高宽点
    jr = k0
    R = Ref2(jr,d)
    while R >= Rmax/2:
        jr = jr+k0/100
        R = Ref2(jr,d)
    #寻找布拉格峰左侧的半高宽点
    jl = k0
    R = Ref2(jl,d)
    while R >= Rmax/2:
        jl = jl-k0/100
        R = Ref2(jl,d)
    dk = jr-jl    
    return [Rmax,dk]
    
##########6拟合反射率曲线并给出拟合得到的m值的函数################################
    
#Function6.1
#本函数得到的m值为反射率下降到约为0.5的位置
def RefFit(d=[]):
    x0 = np.arange(0, 11*kc, 0.1)
    y = np.ones(len(x0))
    j = 0
    for x in x0:
        y[j] = Ref2(x,d)
        j = j+1
    #拟合函数，分段取值  
    def RefFunction(x,m,W,alpha):
        return np.piecewise(x, [x <= kc, x > kc], [lambda x: 1, lambda x: 0.5*(1-np.tanh((x-m*kc)/W))*(1-alpha*(x-kc))])
    #设定拟合参数的下界和上界
    param_bounds=([1,0,0],[np.inf,np.inf,np.inf])
    #输出拟合结果
    popt, pcov = curve_fit(RefFunction, x0, y, bounds=param_bounds, maxfev=500000)#bounds是拟合参数边界，maxfev是最大拟合次数，建议设大一些以保证得到结果
    return popt

#Function6.2
#本函数得到的m值为反射率下降沿起点的位置
def RefFit2(d=[]):
    x0 = np.arange(0, 11*kc, 0.1)
    y = np.ones(len(x0))
    j = 0
    for x in x0:
        y[j] = Ref2(x,d)
        j = j+1
    #拟合函数，分段取值  
    def RefFunction1(x,m,W,alpha):
        return np.piecewise(x, [x <= kc, x > kc], [lambda x: 1, lambda x: RefFunction2(x,m,W,alpha)])
    def RefFunction2(x,m,W,alpha):
        return np.piecewise(x, [x <= m*kc, x > m*kc], [lambda x: (1-alpha*(x-kc)), lambda x: (1-np.tanh((x-m*kc)/W))*(1-alpha*(x-kc))])
    #设定拟合参数的下界和上界
    param_bounds=([1,0,0],[np.inf,np.inf,np.inf])
    #输出拟合结果
    #popt是拟合参数[m,W,alpha]，pcov是协方差
    popt, pcov = curve_fit(RefFunction1, x0, y, bounds=param_bounds, maxfev=500000)#bounds是拟合参数边界，maxfev是最大拟合次数，建议设大一些以保证得到结果
    #计算拟合标准偏差
    #perr = np.sqrt(np.diag(pcov))
    #return [popt,perr]
    return popt

#Function6.3
#本函数适用于引入常数粗糙度的情形
#本函数得到的m值为反射率下降沿起点的位置
def RefFitRouCon(sigmaDW,d=[]):
    x0 = np.arange(0, 11*kc, 0.1)
    y = np.ones(len(x0))
    j = 0
    for x in x0:
        y[j] = RefRouCon(x,sigmaDW,d)
        j = j+1
    #拟合函数，分段取值  
    def RefFunction1(x,m,W,alpha,beta):
        return np.piecewise(x, [x <= kc, x > kc], [lambda x: 1, lambda x: RefFunction2(x,m,W,alpha,beta)])
    def RefFunction2(x,m,W,alpha,beta):
        return np.piecewise(x, [x <= m*kc, x > m*kc], [lambda x: 1+beta-beta*np.e**(alpha*(x-kc)), lambda x: (1-np.tanh((x-m*kc)/W))*(1+beta-beta*np.e**(alpha*(x-kc)))])
    #设定拟合参数的下界和上界
    param_bounds=([1,0,0,0],[np.inf,np.inf,np.inf,np.inf])
    #输出拟合结果
    #popt是拟合参数[m,W,alpha,beta]，pcov是协方差
    popt, pcov = curve_fit(RefFunction1, x0, y, bounds=param_bounds, maxfev=500000)#bounds是拟合参数边界，maxfev是最大拟合次数，建议设大一些以保证得到结果
    #计算拟合标准偏差
    #perr = np.sqrt(np.diag(pcov))
    #return [popt,perr]
    return popt

#Function6.4
#本函数适用于引入逐层制造的粗糙度的情形
#本函数得到的m值为反射率下降沿起点的位置
def RefFitRouGro(sigma0,h,d=[]):
    x0 = np.arange(0, 11*kc, 0.1)
    y = np.ones(len(x0))
    j = 0
    for x in x0:
        y[j] = RefRouGro(x,sigma0,h,d)
        j = j+1
    #拟合函数  ，分段取值  
    def RefFunction1(x,m,W,alpha1,alpha2,beta1,beta2,gamma):
        return np.piecewise(x, [x <= kc, x > kc], [lambda x: 1, lambda x: RefFunction2(x,m,W,alpha1,alpha2,beta1,beta2,gamma)])
    def RefFunction2(x,m,W,alpha1,alpha2,beta1,beta2,gamma):
        return np.piecewise(x, [x <= gamma*kc, x > gamma*kc], 
                            [lambda x: 1+beta1-beta1*np.e**(alpha1*(x-kc)), lambda x: RefFunction3(x,m,W,alpha1,alpha2,beta1,beta2,gamma)])
    def RefFunction3(x,m,W,alpha1,alpha2,beta1,beta2,gamma):
        return np.piecewise(x, [x <= m*kc, x > m*kc], 
                            [lambda x: 1+beta1-beta1*np.e**(alpha1*(gamma*kc-kc))-beta2+beta2*np.e**(alpha2*(x-gamma*kc)), 
                             lambda x: (1-np.tanh((x-m*kc)/W))*(1+beta1-beta1*np.e**(alpha1*(gamma*kc-kc))-beta2+beta2*np.e**(alpha2*(m*kc-gamma*kc)))])    
    #设定拟合参数的下界和上界
    param_bounds=([1,0,0,0,0,0,1],[np.inf,np.inf,np.inf,1,np.inf,np.inf,np.inf])
    #输出拟合结果
    popt, pcov = curve_fit(RefFunction1, x0, y, bounds=param_bounds, maxfev=500000)
    #bounds是拟合参数边界，maxfev是最大拟合次数，建议设大一些以保证得到结果
    #计算拟合标准偏差
    #perr = np.sqrt(np.diag(pcov))
    #return [popt,perr]
    return popt


##########7计算引入膜对制造误差后的平均反射率、拟合m值############################
    
#Function7.1
#计算单个算法生成的膜对序列引入膜对制造误差后的平均反射率、拟合m值
#输入参数为相对误差
def ThiErr(d,ddmin,ddmax):
    x0 = np.arange(0, 60, 0.1)
    m0 = RefFit2(d)[0]
    y0 = np.ones(len(x0))
    j = 0
    for x in x0:
        y0[j] = Ref(x,d)
        j = j+1
    A0 = AveRF(m0,x0,y0)
    dr = np.ones([len(d),3])
    dd = 0 #记录总偏差
    i = 0
    while i < len(d): #生成随机误差作用后的膜对序列
       a = random.uniform(ddmin, ddmax)
       b = random.uniform(ddmin, ddmax)
       dr[i] = [i+1,d[i][1]*(1+a/100),d[i][2]*(1+b/100)]
       dd = dd+(d[i][1]*a/100)**2+(d[i][2]*b/100)**2
       i = i+1
    dd = math.sqrt(dd/2/len(dr))
    m = RefFit2(dr)[0] #拟合反射率曲线，得到实际的m值
    y = np.ones(len(x0))
    j = 0
    for x in x0:
        y[j] = Ref(x,dr)
        j = j+1
    A = AveRF(m,x0,y) #计算平均反射率
    return [[dd,A-A0,m-m0],[A,A0,m,m0]]

#Function7.2
#计算单个算法生成的膜对序列引入膜对制造误差后的平均反射率、拟合m值
#输入参数为绝对误差，单位nm；随机数符合高斯分布
def ThiErrAbs(d,loc,scale):
    x0 = np.arange(0, 60, 0.1)
    m0 = RefFit2(d)[0]
    y0 = np.ones(len(x0))
    j = 0
    for x in x0:
        y0[j] = Ref2(x,d)
        j = j+1
    A0 = AveRF(m0,x0,y0)
    dr = np.ones([len(d),3])
    dd = 0 #记录总偏差
    i = 0
    while i < len(d): #生成随机误差作用后的膜对序列
       #a = random.uniform(ddmin, ddmax)
       a = np.random.normal(loc,scale)
       #b = random.uniform(ddmin, ddmax)
       b = np.random.normal(loc,scale)
       dr[i] = [i+1,d[i][1]+a,d[i][2]+b]
       dd = dd+a**2+b**2
       i = i+1
    dd = math.sqrt(dd/2/len(dr))
    m = RefFit2(dr)[0] #拟合反射率曲线，得到实际的m值
    y = np.ones(len(x0))
    j = 0
    for x in x0:
        y[j] = Ref2(x,dr)
        j = j+1
    A = AveRF(m,x0,y) #计算平均反射率
    return [[dd,A-A0,m-m0],[A,A0,m,m0]]


#8精度函数，由于标定膜层加工厚度
#Function8
def Accuracy(a,m):
    #a为厚度加工精度，m取值为最近的a的整数倍的数值
    n = math.floor(m/a) #向下取整
    b = m-a*n
    if b < a/2:
        m = a*n
    else:
        m = a*(n+1)
    return m  


#9经过粗糙度优化的算法实现函数
#Function9.1
#GRB算法
def GRBopt(m,R,sigmaDW):
    #输入参数：设计反射率R，m值    
    alpha =4/(math.atanh(math.sqrt(R)))**2
    #中间参数计算
    #特征波长
    lambdac = math.sqrt(math.pi/(rhob1-rhob2))*100
    lambdaNi = math.sqrt(math.pi/rhob1)*100
    lambdaTi2 = math.pi/rhob2*10000
    lambdaSM = lambdaNi/m
    #计算过程：
    #迭代起点
    #n为膜层序数,迭代到最后即为膜层总数
    n = 1
    D = lambdaNi/2-0.2
    N = math.ceil(math.sqrt(1/(alpha*(D/lambdac)**4)))
    #N = math.ceil(math.atanh(math.sqrt(R))/(2*(D/lambdac)**2))
    #以上两种N的表达式是等价的
    d1 = D/(2*math.sqrt(1-(2*D/lambdaNi)**2))
    d2 = D/(2*math.sqrt(1-(2*D**2/lambdaTi2)))
    #用一个矩阵存储这些数据
    d = [[n,d1,d2]]
    #依次迭代
    while D > lambdaSM/2:
        n = n+1
        D = D*(1-N**(-2)*math.exp(-(2*math.pi*sigmaDW/D)**2))
        N = math.ceil(math.sqrt(1/(alpha*(D/lambdac)**4)))
        #N = math.ceil(math.atanh(math.sqrt(R))/(2*(D/lambdac)**2))
        #以上两种N的表达式是等价的
        d1 = D/(2*math.sqrt(1-(2*D/lambdaNi)**2))
        d2 = D/(2*math.sqrt(1-(2*D**2/lambdaTi2)))
        d.append([n,d1,d2])
    return d


#10RSD算法升级版
#Function10
#输入设计反射率函数，可以实现形式各异的反射率曲线
def RSDplus(m,func):
    #输入参数：设计反射率函数func_R，m值
    #推荐设计反射率函数形式
    #def func_R(k):
    #    RL = f(k)
    #    return RL

    #中间参数计算
    #特征波长
    lambdaNi = math.sqrt(math.pi/rhob1)*100
    lambdaTi2 = math.pi/rhob2*10000
    lambdaSM = lambdaNi/m
    #Ni的全反射临界波矢
    kNi = 2*math.pi/lambdaNi*100
    #迭代计算过程
    #迭代起点	
    k = m*kNi
    D = lambdaSM/2
    d1 = D/(2*math.sqrt(1-(2*D/lambdaNi)**2))
    d2 = D/(2*math.sqrt(1-(2*D**2/lambdaTi2)))
    N = 1
    while BraggHeightandWidth(N,d1,d2,k)[0] <= func(k):
        N = N+1
    dk = BraggHeightandWidth(N,d1,d2,k)[1]
    M = [[1,d1,d2]]
    #自适应峰宽系数
    def a(k):
        a = 1.2+0.6*(k-kNi)/((m-1)*kNi)
        return a
    #依次迭代
    j = 1
    while 0 < D < lambdaNi/2:
        k = k-dk/(a(k)*N) 
        D = lambdaNi/(2*(k/kNi))
        if 0 < D < lambdaNi/2:
            d1 = D/(2*math.sqrt(1-(2*D/lambdaNi)**2))
            d2 = D/(2*math.sqrt(1-(2*D**2/lambdaTi2)))
            N = 1
            while BraggHeightandWidth(N,d1,d2,k)[0] <= func(k):
                N = N+1
            dk = BraggHeightandWidth(N,d1,d2,k)[1]
            j = j+1
            M.append([j,d1,d2])
    #由于最表面的Ni层厚度太薄，因此额外加一层60nm厚的Ni层。		
    M.append([j+1,60,0])
    #为配合反射率计算函数，将膜对顺序反过来，从最表面开始
    d = np.ones([len(M),3])
    i = 0
    while i < len(M):
        d[i] = [i+1,M[len(M)-i-1][1],M[len(M)-i-1][2]]
        i = i+1
    return d
