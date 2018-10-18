import matplotlib.pyplot as plt
import numpy as np

def draw_circle():
    angles=np.linspace(0,2*np.pi,1000)
    xs=np.cos(angles)
    ys=np.sin(angles)
    plt.plot(xs,ys,'r')
    plt.axes().set_aspect('equal','datalim')
    
def draw_square():
    size=1
    n_points=1000
    ones=np.ones((n_points,))
    v=np.linspace(size,size,n_points)
    color='h'
    plt.plot(ones,v,color)
    plt.plot(v,ones,color)
    plt.plot(ones,v,color)
    plt.plot(v,-ones,color)
    
    
def draw_backgroud():
    draw_square()
    draw_square()
    
#实验次数 
N_TRAINS=10**5
#在正方形里扔很多点
points=np.random.uniform(low=-1,high=1,size=(N_TRAINS,2))
#到原点距离小于1的点在圆内部
sq_radiuses=np.sum(points**2,axis=1)
in_circle=(sq_radiuses<1)
approx_pi=4*np.sum(in_circle)/N_TRAINS
draw_backgroud()
plt.scatter(points[:,0],points[:,1],c=in_circle,s=0.1)
print('近似的pi是',approx_pi)
