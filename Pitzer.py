import  math

import pandas as pd
from numba import jit
import numpy as np
a1 = 2
A_fai = 0.391
b0_11 = 0.2065
b1_11 = 0.5556
b2_11=0
C_fai_11=0
seita_c1c1=0
b0_12 = 0.0298
b1_12 = 0
b2_12=0
C_fai_12=0.0438
b0_21 = 0.0454
b1_21 = 0.0398
b2_21=0
C_fai_21=0
b0_22 = 0.01958
b1_22 = 1.113
b2_22=0
C_fai_22=00.00497

seita_c1c1=0
seita_c2c2=0
seita_c1c2=0.036
seita_a1a1=0
seita_a2a2=0
seita_a1a2=0
da_fai_c1c1a1=0
da_fai_c1c1a2=0
da_fai_c1c2a1=-0.0129
da_fai_c1c2a2=0
da_fai_c2c2a1=0
da_fai_c2c2a2=0
da_fai_a1a1c1=0
da_fai_a1a1c2=0
da_fai_a2a2c1=0
da_fai_a2a2c2=0
da_fai_a1a2c1=0
da_fai_a1a2c2=-0.0094
laimuda_c1=0
laimuda_c2=0.1
laimuda_a1=-0.003
laimuda_a2=0.097
@jit
def g(x):
    return 2 * (1 - (1 + x) * math.exp(-x)) / pow(x, 2)
@jit
def dg_dx(x):
    return -2 * (1 - (1 + x + x * x / 2) * math.exp(-x)) / pow(x, 2)
@jit
def B_ca(I, b0_ca, b1_ca):
    b_ca = b0_ca + b1_ca * g(a1 * pow(I, 1 / 2))
    return b_ca
@jit
def B_pie_ca(I, b1_ca):
    b_fai_ca = b1_ca * dg_dx(a1 * pow(I, 1 / 2)) / I
    return b_fai_ca
@jit
def J(x):
    C1 = 4.581
    C2 = 0.7237
    C3 = 0.012
    C4 = 0.528
    return x * pow((4 + C1 * pow(x, -C2) * math.exp(-C3 * pow(x, C4))), -1)
@jit
def dJ_dx(x):
    C1 = 4.581
    C2 = 0.7237
    C3 = 0.012
    C4 = 0.528
    j1 = pow((4 + C1 * pow(x, -C2) * math.exp(-C3 * pow(x, C4))), -1)
    j2 = pow((4 + C1 * pow(x, -C2) * math.exp(-C3 * pow(x, C4))), -2)
    j3 = C1 * x * math.exp(-C3 * pow(x, C4))
    j4 = C2 * pow(x, -C2 - 1) + C3 * C4 * pow(x, C4 - 1) * pow(x, -C2)
    return j1 + j2 * j3 * j4
@jit
def fai_ij(I, Zi, Zj,seita_ij):
    xij = 6 * Zi * Zj * A_fai * pow(I, 1 / 2)
    xii = 6 * Zi * Zi * A_fai * pow(I, 1 / 2)
    xjj = 6 * Zj * Zj * A_fai * pow(I, 1 / 2)
    E_seita_ij = Zi * Zj / 4 / I * (J(xij) - J(xii) / 2 - J(xjj) / 2)
    return seita_ij + E_seita_ij
@jit
def fai_pie_ij(I, Zi, Zj):
    xij = 6 * Zi * Zj * A_fai * pow(I, 1 / 2)
    xii = 6 * Zi * Zi * A_fai * pow(I, 1 / 2)
    xjj = 6 * Zj * Zj * A_fai * pow(I, 1 / 2)
    E_seita_ij = Zi * Zj / 4 / I * (J(xij) - J(xii) / 2 - J(xjj) / 2)
    E_seita_pie_ij = -E_seita_ij / I + Zi * Zj / 8 / I / I * (
                xij * dJ_dx(xij) - xii * dJ_dx(xii) / 2 - xjj * dJ_dx(xjj) / 2)
    return E_seita_pie_ij
@jit
def C(C_fai_ij, Zi, Zj):
    return C_fai_ij / 2 / pow(abs(Zi * Zj), 1 / 2)
@jit
def F(I, m_c1, m_c2, m_a1, m_a2, Z_c1, Z_c2, Z_a1, Z_a2):
    B_pie_11 = B_pie_ca(I, b1_11)
    B_pie_12 = B_pie_ca(I, b1_12)
    B_pie_21 = B_pie_ca(I, b1_21)
    B_pie_22 = B_pie_ca(I, b1_22)

    fai_pie_cc = fai_pie_ij(I, Z_c1, Z_c2)
    fai_pie_aa = fai_pie_ij(I, Z_a1, Z_a2)
    fai_1 = m_c1 * m_a1 * B_pie_11 + m_c1 * m_a2 * B_pie_12 + m_c2 * m_a1 * B_pie_21 + m_c2 * m_a2 * B_pie_22
    fai_2 = m_c1 * m_c2 * fai_pie_cc
    fai_3 = m_a1 * m_a2 * fai_pie_aa
    f = -A_fai * (pow(I, 1 / 2) / (1 + 1.2 * pow(I, 1 / 2)) + 2*np.log(1+1.2*pow(I,1/2))/1.2)
    return f + fai_1 + fai_2 + fai_3
@jit
def ln_y_c1(I, m_c1, m_c2, m_a1, m_a2, Z_c1, Z_c2, Z_a1, Z_a2, Z):
    B_c1a1 = B_ca(I, b0_11, b1_11)
    B_c1a2 = B_ca(I, b0_12, b1_12)
    C_c1a1 = C(C_fai_11, Z_c1, Z_a1)
    C_c1a2 = C(C_fai_12, Z_c1, Z_a2)
    C_c2a1 = C(C_fai_21, Z_c2, Z_a1)
    C_c2a2 = C(C_fai_22, Z_c2, Z_a2)
    fai_c1c1=fai_ij(I,Z_c1,Z_c1,seita_c1c1)
    fai_c1c2 = fai_ij(I, Z_c1, Z_c2, seita_c1c2)
    y1 = pow(Z_c1, 2) * F(I, m_c1, m_c2, m_a1, m_a2, Z_c1, Z_c2, Z_a1, Z_a2)
    y2 = m_a1 * (2 * B_c1a1 + Z * C_c1a1) + m_a2 * (2 * B_c1a2 + Z * C_c1a2)
    y3 = m_c1 * (2 * fai_c1c1 + m_a1 * da_fai_c1c1a1 + m_a2 * da_fai_c1c1a2) + m_c2 * (
                2 * fai_c1c2 + m_a1 * da_fai_c1c2a1 + m_a2 * da_fai_c1c2a2)
    y4 = m_a1 * m_a2 * da_fai_a1a2c1
    y5 = Z_c1 * (m_c1 * m_a1 * C_c1a1 + m_c1 * m_a2 * C_c1a2 + m_c2 * m_a1 * C_c2a1 + m_c2 * m_a2 * C_c2a2)

    return pow(math.e, y1 + y2 + y3 + y4 + y5)
@jit
def ln_y_c2(I, m_c1, m_c2, m_a1, m_a2, Z_c1, Z_c2, Z_a1, Z_a2, Z):
    B_c2a1 = B_ca(I, b0_21, b1_21)
    B_c2a2 = B_ca(I, b0_22, b1_22)
    C_c1a1 = C(C_fai_11, Z_c1, Z_a1)
    C_c1a2 = C(C_fai_12, Z_c1, Z_a2)
    C_c2a1 = C(C_fai_21, Z_c2, Z_a1)
    C_c2a2 = C(C_fai_22, Z_c2, Z_a2)
    fai_c2c1=fai_ij(I,Z_c1,Z_c1,seita_c1c2)
    fai_c2c2 = fai_ij(I, Z_c1, Z_c2, seita_c2c2)
    y1 = pow(Z_c2, 2) * F(I, m_c1, m_c2, m_a1, m_a2, Z_c1, Z_c2, Z_a1, Z_a2)
    y2 = m_a1 * (2 * B_c2a1 + Z * C_c2a1) + m_a2 * (2 * B_c2a2 + Z * C_c2a2)
    y3 = m_c1 * (2 * fai_c2c1 + m_a1 * da_fai_c1c2a1 + m_a2 * da_fai_c1c2a2) + m_c2 * (
                2 * fai_c2c2 + m_a1 * da_fai_c2c2a1 + m_a2 * da_fai_c2c2a2)
    y4 = m_a1 * m_a2 * da_fai_a1a2c2
    y5 = Z_c2 * (m_c1 * m_a1 * C_c1a1 + m_c1 * m_a2 * C_c1a2 + m_c2 * m_a1 * C_c2a1 + m_c2 * m_a2 * C_c2a2)
    return pow(math.e, y1 + y2 + y3 + y4 + y5)
@jit
def ln_y_a1(I, m_c1, m_c2, m_a1, m_a2, Z_c1, Z_c2, Z_a1, Z_a2, Z):
    B_c1a1 = B_ca(I, b0_11, b1_11)
    B_c2a1 = B_ca(I, b0_21, b1_21)
    C_c1a1 = C(C_fai_11, Z_c1, Z_a1)
    C_c1a2 = C(C_fai_12, Z_c1, Z_a2)
    C_c2a1 = C(C_fai_21, Z_c2, Z_a1)
    C_c2a2 = C(C_fai_22, Z_c2, Z_a2)
    fai_a1a1=fai_ij(I,Z_c1,Z_c1,seita_a1a1)
    fai_a1a2 = fai_ij(I, Z_c1, Z_c2, seita_a1a2)
    y1 = pow(Z_a1, 2) * F(I, m_c1, m_c2, m_a1, m_a2, Z_c1, Z_c2, Z_a1, Z_a2)
    y2 = m_c1 * (2 * B_c1a1 + Z * C_c1a1) + m_c2 * (2 * B_c2a1 + Z * C_c2a1)
    y3 = m_a1 * (2 * fai_a1a1 + m_c1 * da_fai_a1a1c1 + m_c2 * da_fai_a1a1c2) + m_a2 * (
                2 * fai_a1a2 + m_c1 * da_fai_a1a2c1 + m_c2 * da_fai_a1a2c2)
    y4 = m_c1 * m_c2 * da_fai_c1c2a1
    y5 = Z_a1 * (m_c1 * m_a1 * C_c1a1 + m_c1 * m_a2 * C_c1a2 + m_c2 * m_a1 * C_c2a1 + m_c2 * m_a2 * C_c2a2)
    return pow(math.e, y1 + y2 + y3 + y4 + y5)
@jit
def ln_y_a2(I, m_c1, m_c2, m_a1, m_a2, Z_c1, Z_c2, Z_a1, Z_a2, Z):
    B_c1a2 = B_ca(I, b0_12, b1_12)
    B_c2a2 = B_ca(I, b0_22, b1_22)
    C_c1a1 = C(C_fai_11, Z_c1, Z_a1)
    C_c1a2 = C(C_fai_12, Z_c1, Z_a2)
    C_c2a1 = C(C_fai_21, Z_c2, Z_a1)
    C_c2a2 = C(C_fai_22, Z_c2, Z_a2)
    fai_a1a2=fai_ij(I,Z_c1,Z_c1,seita_a1a2)
    fai_a2a2 = fai_ij(I, Z_c1, Z_c2, seita_a2a2)
    y1 = pow(Z_a2, 2) * F(I, m_c1, m_c2, m_a1, m_a2, Z_c1, Z_c2, Z_a1, Z_a2)
    y2 = m_c1 * (2 * B_c1a2 + Z * C_c1a2) + m_c2 * (2 * B_c2a2 + Z * C_c2a2)
    y3 = m_a1 * (2 * fai_a1a2 + m_c1 * da_fai_a1a2c1 + m_c2 * da_fai_a1a2c2) + m_a2 * (
                2 * fai_a2a2 + m_c1 * da_fai_a2a2c1 + m_c2 * da_fai_a2a2c2)
    y4 = m_c1 * m_c2 * da_fai_c1c2a2
    y5 = Z_a2 * (m_c1 * m_a1 * C_c1a1 + m_c1 * m_a2 * C_c1a2 + m_c2 * m_a1 * C_c2a1 + m_c2 * m_a2 * C_c2a2)
    return pow(math.e, y1 + y2 + y3 + y4 + y5)


@jit
def Pitzer(I, m_c1, m_c2, m_a1, m_a2, Z_c1, Z_c2, Z_a1, Z_a2):
    Z = abs(Z_c1) * m_c1 + abs(Z_c2) * m_c2 + abs(Z_a1) * m_a1 + abs(Z_a2) * m_a2
    y_c1=ln_y_c1(I, m_c1, m_c2, m_a1, m_a2, Z_c1, Z_c2, Z_a1, Z_a2, Z)
    y_c2 = ln_y_c2(I, m_c1, m_c2, m_a1, m_a2, Z_c1, Z_c2, Z_a1, Z_a2, Z)
    y_a1 = ln_y_a1(I, m_c1, m_c2, m_a1, m_a2, Z_c1, Z_c2, Z_a1, Z_a2, Z)
    y_a2 = ln_y_a2(I, m_c1, m_c2, m_a1, m_a2, Z_c1, Z_c2, Z_a1, Z_a2, Z)

    return y_c1,y_c2,y_a1,y_a2

if __name__=='__main__':
    #硫酸钠比例
    fai_Na = 0.5
    fai_H = 1 - fai_Na
    i=0.1
    m_H = fai_H * i
    m_Na = fai_Na * i * 2
    m_HSO4 = fai_H * i
    m_SO4 = fai_Na * i
    Z_H = 1
    Z_Na = 1
    Z_HSO4 = -1
    Z_SO4 = -2
    I = (m_H * Z_H + m_Na * Z_Na + m_HSO4 + m_SO4 * pow(Z_SO4, 2)) / 2
    y = Pitzer(I, m_H, m_Na, m_HSO4, m_SO4, Z_H, Z_Na, Z_HSO4, Z_SO4)
    Ka= 0.01023 / y[0] / y[3] * y[2]
    print(Ka)
    # data=pd.DataFrame()
    # for fai in [0,0.25,0.5,0.75,1]:
    #     fai_Na=fai
    #     fai_H=1-fai_Na
    #     i=np.arange(1,1600,1)/1000
    #     m_H=fai_H*i
    #     m_Na=fai_Na*i*2
    #     m_HSO4=fai_H*i
    #     m_SO4=fai_Na*i
    #     Z_H=1
    #     Z_Na=1
    #     Z_HSO4=-1
    #     Z_SO4=-2
    #     I=(m_H*Z_H+m_Na*Z_Na+m_HSO4+m_SO4*pow(Z_SO4,2))/2
    #     Ka=np.zeros(len(i))
    #     for i in range(len(i)):
    #
    #         y=Pitzer(I[i],m_H[i],m_Na[i],m_HSO4[i],m_SO4[i],Z_H,Z_Na,Z_HSO4,Z_SO4)
    #         Ka[i]=0.01023/y[0]/y[3]*y[2]
    #
    #
    #     data['I%s' % fai] = I
    #     data['%s'%fai]=Ka
    # data.to_excel('Ka-I-fai.xlsx',index=False)
