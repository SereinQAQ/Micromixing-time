import copy
import math
import time
import numpy as np
import pandas as pd
from numba import jit
from copy import copy
from Pitzer import Pitzer
@jit
def  jieli_ph(c_H2SO4):

    k2 =0.01023
    a=1
    b=-3*c_H2SO4-k2
    c=2*pow(c_H2SO4,2)
    x=(-b-pow(b*b-4*a*c,1/2))/2/a
    c_HSO4=x
    c_H=2*c_H2SO4-x
    c_SO4=c_H2SO4-x
    return c_HSO4,c_H,c_SO4
@jit
def v2(t,V1,tm):
    v2=math.exp(t/tm)
    return v2
#体积函数g（t）的导数
@jit
def dg_dt(t,tm):
    return math.exp(t/tm)/tm
@jit
def IS(t,n_H,n_HSO4,n_SO4,n_H2BO3,n_I,n_IO3,n_I3,n_K,n_Na,V1,tm):
    i=(n_H+n_HSO4+n_SO4*4+n_H2BO3+n_I+n_IO3+n_I3+n_K+n_Na)/2/v2(t,V1,tm)
    return i
@jit
def k4(I):
    i=I
    #Arian2021
    # k0 = 1.37e9
    # k4 = k0 * pow(10, -1.93 * pow(i, 1 / 2) / (1 + pow(i, 1 / 2)) + 0.4 * i)
    #Guichardan2021
    if i<=0.4:
        k4=pow(10,8.884-0.921*pow(i,1/2))
    else:
        k4=1.96e8
    return k4
@jit
def r2(t, n_HSO4, n_H, n_SO4,V1,tm,kr):
    return kr*1e11* n_HSO4 -1e11 * n_H * n_SO4 / v2(t,V1,tm)
@jit
def r3(t,n_H,n_H2BO3,V1,tm):
    return 1e11*n_H*n_H2BO3/v2(t,V1,tm)
#反应4生成I2
@jit
def r4(t,n_H,n_I,n_IO3,V1,tm,I):
    return k4(I)*pow(n_H,2)*pow(n_I,2)*n_IO3/pow(v2(t,V1,tm),4)
@jit
def r5(t,n_I,n_I2,n_I3,V1,tm):
    return 5.6e9*n_I*n_I2/v2(t,V1,tm)-7.5e6*n_I3
@jit
def d_n_HSO4_dt(t,n_HSO4,n_H,n_SO4,V1,tm,kr):
    return -r2(t,n_HSO4,n_H,n_SO4,V1,tm,kr)
@jit
def d_n_H_dt(t,n_HSO4,n_H,n_SO4,n_H2BO3,n_I,n_IO3,V1,tm,I,kr):
    return r2(t,n_HSO4,n_H,n_SO4,V1,tm,kr)-r3(t,n_H,n_H2BO3,V1,tm)-6*r4(t,n_H,n_I,n_IO3,V1,tm,I)
@jit
def d_n_H2BO3_dt(t,n_H,n_H2BO3,c_H2BO3,V1,tm):
    return dg_dt(t,tm)*c_H2BO3-r3(t,n_H,n_H2BO3,V1,tm)

#I变化的微分方程
@jit
def d_n_I_dt(t,n_I,n_H,n_IO3,n_I2,n_I3,c_I,V1,tm,I):
    return dg_dt(t,tm)*c_I-5*r4(t,n_H,n_I,n_IO3,V1,tm,I)-r5(t,n_I,n_I2,n_I3,V1,tm)

# IO3变化的微分方程
@jit
def d_n_IO3_dt(t,n_IO3,n_H,n_I,c_IO3,V1,tm,I):
    return dg_dt(t,tm)*c_IO3-r4(t,n_H,n_I,n_IO3,V1,tm,I)

# I2变化的微分方程
@jit
def  d_n_I2_dt(t,n_I2,n_H,n_I,n_IO3,n_I3,V1,tm,I):
    return 3*r4(t,n_H,n_I,n_IO3,V1,tm,I)-r5(t,n_I,n_I2,n_I3,V1,tm)
@jit
def  d_n_I3_dt(t,n_I3,n_I,n_I2,V1,tm):
    return r5(t,n_I,n_I2,n_I3,V1,tm)

# K变化的微分方程
@jit
def d_n_K_dt(t,c_K,V1,tm):
    return dg_dt(t,tm)*c_K
# Na变化的微分方程
@jit
def d_n_Na_dt(t,c_Na,V1,tm):
    return dg_dt(t,tm)*c_Na
@jit
def jisuan_Xs(n_I2,n_I3,c_H2SO4,c_IO3,c_H2BO3):
    return 2*(n_I2+n_I3)/c_H2SO4/2*(6*c_IO3+c_H2BO3)/6/c_IO3
@jit
def Kr(I,m_H, m_Na, m_HSO4, m_SO4, Z_H, Z_Na, Z_HSO4, Z_SO4):
    y = Pitzer(I, m_H, m_Na, m_HSO4, m_SO4, Z_H, Z_Na, Z_HSO4, Z_SO4)
    y_H=y[0]
    y_HSO4 = y[2]
    y_SO4 = y[3]

    return 0.01023/y_H/y_SO4*y_HSO4


@jit
def IM(c_H2SO4,c_H2BO3,c_I,c_IO3,R,tm,h):
    #定义离集体初始体积
    V2=1
    #定义环境体积
    V1=R
    #定义离集体初始物质的量
    n_HSO4=jieli_ph(c_H2SO4)[0]
    n_H=jieli_ph(c_H2SO4)[1]
    n_SO4 = jieli_ph(c_H2SO4)[2]
    n_H2BO3=0
    n_I = 0
    n_IO3 = 0
    n_I2=0
    n_I3=0
    n_K=0
    n_Na=0
    #定义溶液初始离子强度
    I=(n_HSO4+n_H+4*n_SO4)/2
    #定义环境初始浓度
    c_Na = c_H2BO3*R/(R+1)/2+c_H2BO3/2
    c_K = (c_I + c_IO3)*R/(R+1)/2+ (c_I + c_IO3)/2
    c_I_s = c_I*R/(R+1)/2+c_I/2
    c_IO3_s = c_IO3*R/(R+1)/2+c_IO3/2
    c_H2BO3_s = (c_H2BO3*R-c_H2SO4*2)/(R+1)/2+c_H2BO3/2
    #定义反应时间
    t=0
    i=0
    while True:

        #更新离集体体积
        V2=v2(t,V1,tm)

        #更新溶液离子强度
        I=IS(t,n_H,n_HSO4,n_SO4,n_H2BO3,n_I,n_IO3,n_I3,n_K,n_Na,V1,tm)
        #计算各物质在h时间步内浓度变化
        kr=Kr(I,n_H/V2,n_Na/V2,n_HSO4/V2,n_SO4/V2,1,1,-1,-2)

        #计算HSO4浓度变化
        k1=d_n_HSO4_dt(t, n_HSO4, n_H, n_SO4,V1,tm,kr)
        k2=d_n_HSO4_dt(t+h/2, n_HSO4+h/2*k1, n_H, n_SO4,V1,tm,kr)
        k3 = d_n_HSO4_dt(t + h / 2, n_HSO4 + h / 2 * k2, n_H, n_SO4,V1,tm,kr)
        k4 = d_n_HSO4_dt(t + h , n_HSO4 + h  * k3, n_H, n_SO4,V1,tm,kr)
        d_n_HSO4=1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4) * h
        #SO4的变化与HSO4相反
        d_n_SO4=-1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4) * h
        #计算H浓度变化
        k1 = d_n_H_dt(t,n_HSO4,n_H,n_SO4,n_H2BO3,n_I,n_IO3,V1,tm,I,kr)
        k2 = d_n_H_dt(t+h/2, n_HSO4, n_H+h/2*k1, n_SO4, n_H2BO3,n_I,n_IO3, V1, tm,I,kr)
        k3 = d_n_H_dt(t+h/2, n_HSO4, n_H+h/2*k2, n_SO4, n_H2BO3,n_I,n_IO3, V1, tm,I,kr)
        k4 = d_n_H_dt(t+h, n_HSO4, n_H+h*k3, n_SO4, n_H2BO3,n_I,n_IO3, V1, tm,I,kr)
        d_n_H=1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4) * h

        #计算H2BO3浓度变化
        k1 = d_n_H2BO3_dt(t,n_H,n_H2BO3,c_H2BO3_s,V1,tm)
        k2 = d_n_H2BO3_dt(t+h/2, n_H, n_H2BO3+h/2*k1, c_H2BO3_s, V1, tm)
        k3 = d_n_H2BO3_dt(t+h/2, n_H, n_H2BO3+h/2*k2, c_H2BO3_s, V1, tm)
        k4 = d_n_H2BO3_dt(t+h, n_H, n_H2BO3+h*k3, c_H2BO3_s, V1, tm)
        d_n_H2BO3 = 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4) * h
        #计算I浓度变化
        k1 = d_n_I_dt(t,n_I,n_H,n_IO3,n_I2,n_I3,c_I_s,V1,tm,I)
        k2 = d_n_I_dt(t+h/2, n_I+h/2*k1, n_H, n_IO3, n_I2, n_I3, c_I_s, V1, tm, I)
        k3 = d_n_I_dt(t+h/2, n_I+h/2*k2, n_H, n_IO3, n_I2, n_I3, c_I_s, V1, tm, I)
        k4 = d_n_I_dt(t+h, n_I+h*k3, n_H, n_IO3, n_I2, n_I3, c_I_s, V1, tm, I)
        d_n_I = 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4) * h
        #计算IO3浓度变化
        k1 = d_n_IO3_dt(t,n_IO3,n_H,n_I,c_IO3_s,V1,tm,I)
        k2 = d_n_IO3_dt(t+h/2, n_IO3+h/2*k2, n_H, n_I, c_IO3_s, V1, tm, I)
        k3 = d_n_IO3_dt(t+h/2, n_IO3+h/2*k3, n_H, n_I, c_IO3_s, V1, tm, I)
        k4 = d_n_IO3_dt(t+h, n_IO3+h*k3, n_H, n_I, c_IO3_s, V1, tm, I)
        d_n_IO3 = 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4) * h
        #计算I2浓度变化
        k1 = d_n_I2_dt(t,n_I2,n_H,n_I,n_IO3,n_I3,V1,tm,I)
        k2 = d_n_I2_dt(t+h/2, n_I2+h/2*k1, n_H, n_I, n_IO3, n_I3, V1, tm, I)
        k3 = d_n_I2_dt(t+h/2, n_I2+h/2*k2, n_H, n_I, n_IO3, n_I3, V1, tm, I)
        k4 = d_n_I2_dt(t+h, n_I2+h*k3, n_H, n_I, n_IO3, n_I3, V1, tm, I)
        d_n_I2 = 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4) * h
        #计算I3浓度变化
        k1 = d_n_I3_dt(t,n_I3,n_I,n_I2,V1,tm)
        k2 = d_n_I3_dt(t+h/2,n_I3+h/2*k1,n_I,n_I2,V1,tm)
        k3 = d_n_I3_dt(t+h/2,n_I3+h/2*k2,n_I,n_I2,V1,tm)
        k4 = d_n_I3_dt(t+h,n_I3+h*k3,n_I,n_I2,V1,tm)
        d_n_I3 = 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4) * h
        #计算K浓度变化
        k1 = d_n_K_dt(t,c_K,V1,tm)
        k2 = d_n_K_dt(t+h/2, c_K, V1, tm)
        k3 = d_n_K_dt(t+h/2, c_K, V1, tm)
        k4 = d_n_K_dt(t+h, c_K, V1, tm)
        d_n_K = 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4) * h
        #计算Na浓度变化
        k1 = d_n_Na_dt(t, c_Na, V1, tm)
        k2 = d_n_Na_dt(t + h / 2, c_Na, V1, tm)
        k3 = d_n_Na_dt(t + h / 2, c_Na, V1, tm)
        k4 = d_n_Na_dt(t + h, c_Na, V1, tm)
        d_n_Na = 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4) * h
        #更新各浓度
        n_HSO4+=d_n_HSO4
        n_H+=d_n_H
        n_SO4+=d_n_SO4
        n_H2BO3+=d_n_H2BO3
        n_I += d_n_I
        n_IO3 += d_n_IO3
        n_I2 += d_n_I2
        n_I3 += d_n_I3
        n_K += d_n_K
        n_Na += d_n_Na
        t=t+h
        # if i%10000==0:
        #     print(n_H/V2,jisuan_Xs(n_I2,n_I3,c_H2SO4,c_IO3,c_H2BO3))
        if n_H<1e-12:
            break

        if t>5*tm:
            break
    Xs=jisuan_Xs(n_I2,n_I3,c_H2SO4,c_IO3,c_H2BO3)

    return Xs

def run(c_H2SO4,c_H2BO3, c_I, c_IO3, R,Xs,h):
    tm=0.01*Xs
    while True:
        Xs_jisuan=IM(c_H2SO4,c_H2BO3,c_I,c_IO3,R,tm,h)
        print(tm,Xs_jisuan)

        tm=tm/Xs_jisuan*Xs

        if abs(Xs/Xs_jisuan-1)<1e-4:
            break
    return tm
if __name__ == '__main__':
    c_H2SO4 = 0.25/2
    c_H2BO3 = 0.0909
    c_I = 0.0117
    c_IO3 = 0.0023
    R = 10
    h = 1 / 1e11  # 时间步长
    Xs =0.003
    tm = run(c_H2SO4, c_H2BO3, c_I, c_IO3, R, Xs, h)
    print(tm)