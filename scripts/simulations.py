import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage
from scipy import misc
from sklearn import (manifold, datasets, decomposition, ensemble, discriminant_analysis, random_projection)

def make_cube(dx,dy,dz):
    x_v = np.linspace(-1+dx,1+dx,9)
    y_v = np.linspace(-1+dy,1+dy,9)
    z_v = np.linspace(-1+dz,1+dz,9)
    coord = []
    for x in x_v:
        for y in y_v:
            for z in z_v:
                coord.append([x,y,z])
    return coord


def cube_simulation(dt, frames, sigma = -1):
    t = np.linspace(0,frames*dt - 1,frames)
    x = t*np.cos(t)
    y = t*np.sin(t)
    z = np.cos(t)

    video = []
    for idt in range(0,frames):
        video.append(make_cube(x[idt],y[idt],z[idt]))
    return video

def make_moving_square(frames, sigma = -1):
    square = np.ones((100,100))
    empty = np.zeros((200,100))
    square = np.concatenate((empty,square,empty), axis=0)
    video = []
    time = np.linspace(0,1,frames)
    for idt in time:
        left = np.zeros((500, int(np.ceil(idt*400))))
        right = np.zeros((500, int(np.floor((1-idt)*400))))
        img = np.concatenate((left, square, right), axis = 1)
        
        if sigma > 0:
            img += np.random.normal(0, sigma, (500, 500))
        
        video.append(img)
    return video

def make_rotating_square(frames,percent,theta, sigma = -1):
    square = np.ones((100,100))
    empty = np.zeros((200,100))
    square = np.concatenate((empty,square,empty), axis=0)
    idt = percent
    left = np.zeros((500, int(np.ceil(idt*400))))
    right = np.zeros((500, int(np.floor((1-idt)*400))))
    square = np.concatenate((left, square, right), axis = 1)
    
    video = []
    time = np.linspace(0,theta,frames)
    for angle in time:
        img = ndimage.interpolation.rotate(square, angle, reshape=False)
        
        if sigma > 0:
            img += np.random.normal(0, sigma, (500, 500))
        
        video.append(img)
    return video

def format_obervation_md_traj(t):
    size =  t.xyz.shape
    X = [f.reshape((1, -1))[0] for f in t.xyz]
    return X

    