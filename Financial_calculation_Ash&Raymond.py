import sympy
import numpy as np
from sympy import Symbol, solve,Eq

fv =[20]
price = [6.914]
def spot_rate(fv,price,terms):
    li = range(0,len(fv))
    for i in li:
        sr = (fv[i]/price[i])**(1/terms) -1
    print(sr)
    return
#spot_rate(fv,price,10)
#forward rate yearly
def forward_rate(sr1,i,srj,j):
    f_ij = ((1+srj)**j/(1+sr1)**i)**(1/(j-i)) -1
    print(f_ij)
    return
#forward_rate(0.084,5,0.1,6)

#forward rate monthly
def forward_rate_m(sr1,i,srj,j,m):
    f_ij = m*((1+srj/m)**j/(1+sr1/m)**i)**(1/(j-i)) -m
    print(f_ij)
    return
#forward_rate_m(0.069,2,0.09,4,12)


# Internal rate of return
def irr():
    li = list()
    while True:
        v = input("enter the cashflow,if end enter end")
        if v != "end":
            v = int(v)
            li.append(v)
        elif v == "end":
            break
    r = Symbol("r")
    c = 1/(1+r)
    pv = 0
    length = range(0,len(li))
    for i in length:
        pv = pv + li[i]*c**i
    result=solve(Eq(pv,0),r)
    print(result)
irr()


# solve for r
def solve_r (initial1,repeated1,initial2,repeated2,terms):
    r = Symbol("r")
    c = 1/(1+r)
    li = range(1,terms+1)
    e1 = initial1
    e2 = initial2
    for i in li:
        te1 = repeated1 * c**i
        e1 = e1 + te1
        te2 = repeated2 * c**i
        e2 = e2 + te2
    e = e1 - e2
    print(e)
    return

solve_r(-200,60,-150,46,5)




# Cut tree prblem
def cuttree(intial_value,ir,years):
    ir = float(ir)
    li = range(1,years)
    x1 = [1,years]
    for i in li:
        NPV = intial_value + (i+1)/(1+ir)**i
        print(NPV)
        y = np.array([1,NPV])
        plt.plot(x1, y)
    plt.show()
cuttree(-1, 0.1, 24)


#inflation
def r_ir(ir,inf):
    r0 = ((ir-inf)/(1+inf))
    print(r0)

r_ir(0.1,0.01)



# fixed coupon bonds price

def bond_price(coupon,ir,terms):
    li = range(1,terms+1)
    print(li)
    present_value = coupon * (1- (1/(1+ir))**terms) * terms
    print(present_value)
    return
bond_price(200,0.1,10)



# loan calculation
# !!! r/terms = ir
def loan(ir,terms,p):
    A = (p*ir*(1+ir)**terms)/((1+ir)**terms-1)
    print(A)
    return
loan(0.08/12,30*12,800000)

#Mortage balance
def mortage_balance(ir,terms,payment):
    P = (payment/ir)*(1-(1/(1+ir)**terms))
    print(P)
    return
mortage_balance(0.09,240,24000)




# fixed income security
def fixed_bond(ir,terms,fv):
    pv = (fv/ir)*(1-(1/(1+ir)**terms))
    print(pv)
    return
fixed_bond(0.01,60,22.24)



# Bond price and YTM
# c = coupon payment

def bond_price(ytm,c,terms,t_remain,fv):
    pv = fv/(1+(ytm/terms))**t_remain + c/ytm * (1-(1/(1+(ytm/terms))))
    print("price=",pv)
    return

bond_price(0.05,9,1,1,100)



# Macaulay Duration
# c = coupon rate /m ; y = yield per period/m; m = periods per uear; n = periods remains

def mac_duration(c,y,m,n):
    D = (1+y)/(m*y) - ((1+y+n*(c-y))/(m*c*((1+y)**n -1) + m*y))
    print(D)
    return
mac_duration(0.04,0.04,2,20)

# Modified Duration
def Modified_duration(D,ytm,m):
    D_m = D/(1+(ytm/m))
    print(D_m)
    return

Modified_duration(9.94,0.10,2)

# Portfolio Duration
# w_i = market value of bond_i/ market value of portfolio   d_i = duration of bond i  k=number of bonds

w_i=[0.416,0.440,0.144]
d_i=[3.861,8.047,9.168]
def prot_duration(w_i,d_i,k):
    li = range(0,k)
    pd = 0
    for i in li:
        pd_t = w_i[i]*d_i[i]
        pd = pd_t + pd
    print(pd)
    return
prot_duration(w_i,d_i,3)

# Quasi-Modified duration
# m times per yaer
def qua_modifed(pv,k,m,cflow,s):
    li = range(0,k)
    for i in li:
        Dq=(1/pv)*(k/m)*cflow[i]*(1+(s/m)**(k-1))
    print(Dq)
    return

#------------------CH6----------------------------
def ExVar_2asset(r1, r2, sd1, sd2, sd12, w1, w2):
    ex = w1*r1 + w2*r2
    var = w1**2*sd1**2 + 2*w1*w2*sd12 + w2**2*sd2**2
    sd = np.sqrt(var)
    print("EX:",ex,"Var:",var)
    return f'Ex = {ex}, Var = {var}, SD = {sd}'


ExVar_2asset(0.12, 0.15,0.2,0.18,0.01,0.25,0.75)


# Correlation Coefficient

def correlation_coefficient(sd1, sd2, sd12):
    print(sd12 / (sd1 * sd2))
    return

def covar(sd1, sd2, ro):
    re = ro*sd1*sd2
    print(re)
    return re

covar(0.15, 0.3, 0.25)

# Ep 6.9 Three uncorrelated assets

def weight_n_multipliers(r1, r2, r3, r, var):
    left = np.array([[1, 0, 0, -r1, -1],
                     [0, 1, 0, -r2, -1],
                     [0, 0, 1, -r3, -1],
                     [r1, r2, r3, 0, 0],
                     [1, 1, 1, 0, 0]])
    right = np.array([0, 0, 0, r, 1]).T
    re = np.linalg.solve(left, right)
    print(re)
    return re

weight_n_multipliers(0.1, 0.2, 0.3, 0.2, 1)

# Ep 6.10, 6.12 A securities portfolio

# Covariance V between each of two securities of secruity 1-5
V = np.array([[2.30, 0.93, 0.62, 0.74, -0.23],
              [0.93, 1.40, 0.22, 0.56, 0.26],
              [0.62, 0.22, 1.80, 0.78, -0.27],
              [0.74, 0.56, 0.78, 3.40, -0.56],
              [-0.23, 0.26, -0.27, -0.56, 2.6]])
r_bar = np.array([15.1, 12.5, 14.7, 9.02, 17.68]).T   # all in percentage

V = np.array([[213.160, 122.976, 112.067, 177.154],
              [122.976, 150.403, 160.497, 175.621],
              [112.067, 160.497, 189.387, 167.737],
              [177.154, 175.621, 167.737, 259.846]])
r_bar = np.array([13.98, 14.16, 17.44, 12.88]).T   # all in percentage

# Ex 6.7 Markowitz Fun

def rf_mini_var_2asset(V, r_bar, rf):
    v1 = np.linalg.solve(V, np.ones(len(V)))
    v2 = np.linalg.solve(V, r_bar)
    w1 = v1/sum(v1)
    w2 = v2/sum(v2)
    v  = v2 - rf*v1
    w  = v/sum(v)
    return print(f"v1 = {v}, w1 = {w1}\nv2 = {v2}, w2 = {w2}\nv  = {v}, w = {w}")


rf_mini_var_2asset(V, r_bar,2)            # rf = 10% and can be set as 0

V1 = np.array([[4, 3],
               [3, 9]])
r1 = np.array([10, 20])
rf_mini_var_2asset(V1, r1,5)

w = np.array([ 0.6372, -0.8962,  1.7218, -0.4628])


def port_var(V, w):
    var = 0
    for i in range(V.shape[1]):
        for j in range(V.shape[0]):
            var = var + w[i]*w[j]*V[i][j]
    return var

port_var(V, w)

V = np.array([[2, 1, 0], [1, 2, 1], [0, 1, 2]])
r_bar = np.array([0.4, 0.8, 0.8]).T
rf_mini_var_2asset(V, r_bar, 0.2)

# Minimum standard deviation, $\lambda = 0, \mu = 1$

def min_var(sd1, sd2, ro):
    sd12 = ro * sd1 * sd2
    re = (sd2**2-sd12) / (sd1**2+sd2**2-2*sd12)
    print(re)
    return re

min_var(0.15, 0.3, 0.25)

def optimal_weight(r1, r2, ro, rf):   # s1 = s2
    w1 = ((r1-rf)-(r2-rf)*ro) / (r1+r2-2*rf-(r1+r2-2*rf)*ro)
    w2 = 1 - w1
    return f'w1 = {w1}, w2 = {w2}'

optimal_weight(0.1,0.08,0.5,0.05)

# Ex 7.1 Capital market line

def cml_s(rb, rm, rf, sm):   # (r_bar, r_market, r_risk free, sd_market), return sd of the position
    re = (rb-rf)*sm / (rm-rf)
    return re

cml_s(0.39,0.23,0.07,0.32)

def cml_rb(rm, rf, sm, s):   # return the expected return (r_bar)
    re = rf + (rm-rf)*s/sm
    print(re)
    return

cml_rb(0.23, 0.07,0.32,0.64)

def allocate_a(n, rb, rm, rf):   # $n dollars, a in market portfolio, n-a in risk-free asset, return a
    # n*rb = a*rm + (n-a)*rf
    re = n*(rb-rf) / (rm-rf)
    print(re)
    return re

allocate_a(1000,0.39,0.23,0.07)

def expected_money(n, a, rm, rf):
    # $n total dollars, a in market portfolio, return expected money at the end of the year
    re = a*(1+rm) + (n-a)*(1+rf)
    print(re)
    return re

expected_money(1000,700,0.23,0.07)

# Mean-variance parameters

def num_parameters(n):
  re = 2*n+n*(n-1)/2
  print(re)
  return re

num_parameters(500)

# Rate of return

def rate_return(rf, rm, sd_m, cov_im):
    re = rf + cov_im/sd_m**2*(rm-rf)
    print(re)
    return re

rate_return(0.08,0.12,0.15,0.09)

def r_i(rf, var_a, var_b, cov_ab, rm):
    beta_a = 2*(var_a+cov_ab) / (var_a+var_b+2*cov_ab)
    beta_b = 2 - beta_a
    r_a = rf + beta_a*(rm-rf)
    r_b = rf + beta_b*(rm-rf)
    print("r_a =",r_a,"r_b",r_b)
    return f"r_a = {r_a}, r_b = {r_b}"

r_i(0.02,0.04,0.02,0.01,0.18)
