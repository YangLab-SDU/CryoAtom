import time

import numpy as np
import random
class RandomCrop(object):
    def __init__(self,output_size:int,ispadding:bool=True):
        assert isinstance(output_size,int)
        self.output_size = (output_size,output_size,output_size)
        self.ispadding = ispadding
    def __call__(self,*x):
        y=[]
        d,h,w=x[0].shape
        od,oh,ow=self.output_size
        if self.ispadding:
            x=list(x)
            k1=max(od-d,0);pad1 = k1//2;pads1 = (pad1,pad1) if k1 % 2 == 0 else (pad1,pad1+1);
            k2 = max(oh-h, 0);pad2 = k2 // 2;pads2 = (pad2, pad2) if k2 % 2 == 0 else (pad2, pad2 + 1);
            k3 = max(ow-w, 0);pad3 = k3 // 2;pads3 = (pad3, pad3) if k3 % 2 == 0 else (pad3, pad3 + 1);
            for i in range(len(x)):
                x[i] = np.pad(x[i],(pads1,pads2,pads3),mode='constant')
        d, h, w = x[0].shape
        sd = random.randint(0,d-od)
        sh = random.randint(0,h-oh)
        sw = random.randint(0,w-ow)
        for ix in x:
            y.append(ix[sd:sd+od,sh:sh+oh,sw:sw+ow])
        y = tuple(y)
        return y

def image_segmentation(image):
    Z,Y,X = image.shape
    stride = 50
    kernel = 64
    pad_temp = (kernel-stride)//2
    segmentation=[]
    tz = int(np.ceil(Z/stride));ty = int(np.ceil(Y/stride));tx = int(np.ceil(X/stride));
    dz = (-Z)%stride;dy = (-Y)%stride;dx = (-X)%stride;
    step = (tz,ty,tx)
    mask = np.ones(image.shape)
    image_new = np.pad(image,((dz//2,dz//2),(dy//2,dy//2),(dx//2,dx//2)),mode='constant')
    mask = np.pad(mask, ((dz // 2, dz // 2), (dy // 2, dy // 2), (dx // 2, dx // 2)), mode='constant')
    image_new = np.pad(image_new,((pad_temp,pad_temp),(pad_temp,pad_temp),(pad_temp,pad_temp)),mode='constant')
    mask = np.pad(mask, ((pad_temp, pad_temp), (pad_temp, pad_temp), (pad_temp, pad_temp)), mode='constant')
    for ii in range(tz):
        for jj in range(ty):
            for kk in range(tx):
                segmentation.append((image_new[ii*stride:ii*stride+kernel,jj*stride:jj*stride+kernel,kk*stride:kk*stride+kernel]
                                    ,mask[ii*stride:ii*stride+kernel,jj*stride:jj*stride+kernel,kk*stride:kk*stride+kernel]))
    return segmentation,step
def image_reconstruction(segmentation,img_shape,step):
    tz,ty,tx = step
    Z,Y,X = img_shape
    stride = 50
    kernel = 64
    pad_temp = (kernel - stride) // 2
    ideal_shape = (tz*stride,ty*stride,tx*stride)
    CZ,CY,CX = np.array(ideal_shape)//2
    ideal_reconsturction = np.zeros(ideal_shape)
    img_number = 0
    for ii in range(tz):
        for jj in range(ty):
            for kk in range(tx):
                img_temp = segmentation[img_number]
                ideal_reconsturction[ii*stride:ii*stride+stride,jj*stride:jj*stride+stride,
                kk*stride:kk*stride+stride] = img_temp[pad_temp:pad_temp+stride,pad_temp:pad_temp+stride
                                              ,pad_temp:pad_temp+stride]
                img_number = img_number + 1
    reconsturction = ideal_reconsturction[CZ-Z//2:CZ+Z//2,CY-Y//2:CY+Y//2,CX-X//2:CX+X//2]
    return reconsturction
def map2atom(index,voxel_size,global_origin):
    ii,jj,kk=index
    index_new = np.array([kk,jj,ii])
    atom_coordinate = index_new * voxel_size + global_origin
    return atom_coordinate
def mass_center(mass,coordinate):
    new = np.dot(mass,coordinate)/np.sum(mass)
    return new
def range_intersection(i,m,arg):
    if m>0:
        range1 = np.array(range(i,i+m+1))
        if (i+1) <= arg:
            return range1[np.where(range1<=(arg-1))][-1]
        else:
            return (arg-1)
    elif m<0:
        range1 = np.array(range(i+m,i+1))
        if i >= 0:
            return range1[np.where(range1 >= 0)][0]
        else:
            return 0
    else:
        return i

def Find_trace(data,target,L,voxel_size,global_origin):
    atom=[]
    atom_in = 0
    data[:2,:,:] = 0
    data[-2:,:,] = 0
    data[:,:2,:] = 0
    data[:,-2:,:] = 0
    data[:,:,:2] = 0
    data[:,:,-2:] = 0
    for i in range(L):
        max_index = np.unravel_index(np.argmax(data),data.shape)
        ii,jj,kk = max_index
        atom_in = atom_in + target[ii,jj,kk]
        mass = np.array([data[ii,jj,kk],data[ii-1,jj,kk],data[ii+1,jj,kk],data[ii,jj-1,kk],data[ii,jj+1,kk],
        data[ii,jj,kk-1],data[ii,jj,kk+1]])
        coordinate = np.array([[ii+0.5,jj+0.5,kk+0.5],[ii-0.5,jj+0.5,kk+0.5],[ii+1.5,jj+0.5,kk+0.5],
                               [ii+0.5,jj-0.5,kk+0.5],[ii+0.5,jj+1.5,kk+0.5],[ii+0.5,jj+0.5,kk-0.5],
                               [ii+0.5,jj+0.5,kk+1.5]])
        new_i,new_j,new_k = mass_center(mass,coordinate)
        atom.append(map2atom((new_i,new_j,new_k),voxel_size=voxel_size,global_origin=global_origin))
        data[ii-1:ii+2,jj-1:jj+2,kk]=0
        data[ii-1:ii+2,jj,kk-1]=0
        data[ii,jj-1:jj+2,kk-1]=0
        data[ii - 1:ii + 2, jj, kk + 1] = 0
        data[ii, jj - 1:jj + 2, kk + 1] = 0
        data[ii,jj,kk+2]=0
        data[ii, jj, kk-2] = 0
        data[ii-2,jj,kk]=0
        data[ii + 2, jj, kk] = 0
        data[ii, jj-2, kk] = 0
        data[ii, jj + 2, kk] = 0
    atom = np.array(atom)
    precision = atom_in / L
    return atom,precision
def map_segmentation(image,windows_size:int=64,stride:int=50):
    Z,Y,X = image.shape
    blocks = []
    for ii in range(0,Z + stride,stride):
        for jj in range(0,Y + stride,stride):
            for kk in range(0,X + stride,stride):
                block = image[min(ii,Z-windows_size):min(ii+windows_size,Z),min(jj,Y-windows_size):min(jj+windows_size,Y)
                ,min(kk,X-windows_size):min(kk+windows_size,X)]
                blocks.append(block)
    return blocks
def map_reconstruction(blocks,image_shape,windows_size:int=64,stride:int=50):
    Z,Y,X = image_shape
    reconstruction = np.zeros(image_shape)
    counts = np.zeros(image_shape)
    image_number = 0
    for ii in range(0,Z + stride,stride):
        for jj in range(0,Y + stride,stride):
            for kk in range(0,X + stride,stride):
                reconstruction[min(ii+2,Z-windows_size+2):min(ii+windows_size-2,Z-2),min(jj+2,Y-windows_size+2):min(jj+windows_size-2,Y-2)
                ,min(kk+2,X-windows_size+2):min(kk+windows_size-2,X-2)] = reconstruction[min(ii+2,Z-windows_size+2):min(ii+windows_size-2,Z-2),min(jj+2,Y-windows_size+2):min(jj+windows_size-2,Y-2)
                ,min(kk+2,X-windows_size+2):min(kk+windows_size-2,X-2)] + blocks[image_number][2:-2,2:-2,2:-2]
                counts[min(ii+2,Z-windows_size+2):min(ii+windows_size-2,Z-2),min(jj+2,Y-windows_size+2):min(jj+windows_size-2,Y-2)
                ,min(kk+2,X-windows_size+2):min(kk+windows_size-2,X-2)] += 1
                image_number = image_number + 1
    reconstruction = reconstruction / (counts+1e-6)
    return reconstruction
def random_rotation90(*x):
    y=[]
    axis1,axis2 = np.random.choice([0,1,2],size=2,replace=False)
    k = random.randint(0,3)
    for ix in x:
        y.append(np.copy(np.rot90(np.copy(ix),k,(axis1,axis2))))
    y = tuple(y)
    return y
def make_mask(target):
    pad = 2
    data = np.pad(np.copy(target),((pad,pad),(pad,pad),(pad,pad)),mode='constant')
    mask_index = np.where(data==1)
    mask_pairs = zip(mask_index[0],mask_index[1],mask_index[2])
    for ii,jj,kk in mask_pairs:
        data[ii - 1:ii + 2, jj - 1:jj + 2, kk] = 1
        data[ii - 1:ii + 2, jj, kk - 1] = 1
        data[ii, jj - 1:jj + 2, kk - 1] = 1
        data[ii - 1:ii + 2, jj, kk + 1] = 1
        data[ii, jj - 1:jj + 2, kk + 1] = 1
        data[ii, jj, kk + 2] = 1
        data[ii, jj, kk - 2] = 1
        data[ii - 2, jj, kk] = 1
        data[ii + 2, jj, kk] = 1
        data[ii, jj - 2, kk] = 1
        data[ii, jj + 2, kk] = 1
    data = data[2:-2,2:-2,2:-2]
    return data

def test_segmentation(image):
    Z,Y,X = image.shape
    stride = 50
    kernel = 64
    blocks = []
    for ii in range(0, Z + stride, stride):
        for jj in range(0, Y + stride, stride):
            for kk in range(0, X + stride, stride):
                block = image[min(ii, Z - kernel):min(ii + kernel, Z),
                        min(jj, Y - kernel):min(jj + kernel, Y)
                , min(kk, X - kernel):min(kk + kernel, X)]
                blocks.append(block)
    return blocks
def test_reconstruction(blocks,img_shape):
    Z,Y,X = img_shape
    stride = 50
    kernel = 64
    pad_temp = (kernel-stride)//2
    reconstruction = np.zeros(img_shape)
    counts = np.zeros(img_shape)
    image_number = 0
    for ii in range(0, Z + stride, stride):
        for jj in range(0, Y + stride, stride):
            for kk in range(0, X + stride, stride):
                ii = ii + pad_temp
                jj = jj + pad_temp
                kk = kk + pad_temp
                IZ = Z - pad_temp
                IY = Y - pad_temp
                IX = X - pad_temp
                reconstruction[min(ii, IZ - stride):min(ii + stride, IZ),
                min(jj, IY - stride):min(jj + stride, IY)
                , min(kk, IX - stride):min(kk + stride, IX)] = reconstruction[min(ii, IZ - stride):min(
                    ii + stride, IZ), min(jj, IY - stride):min(jj + stride, IY)
                                                                         , min(kk, IX - stride):min(
                    kk + stride, IX)] + blocks[image_number][pad_temp:pad_temp+stride,pad_temp:pad_temp+stride,pad_temp:pad_temp+stride]
                counts[min(ii, IZ - stride):min(ii + stride, IZ),
                min(jj, IY - stride):min(jj + stride, IY)
                , min(kk, IX - stride):min(kk + stride, IX)] = counts[min(ii, IZ - stride):min(
                    ii + stride, IZ), min(jj, IY - stride):min(jj + stride, IY)
                                                                         , min(kk, IX - stride):min(
                    kk + stride, IX)] + 1
                image_number = image_number + 1
    counts = np.where(counts==0,1e-8,counts)
    reconstruction = reconstruction / counts
    return reconstruction
