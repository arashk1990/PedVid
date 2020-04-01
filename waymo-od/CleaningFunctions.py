import os
import tensorflow as tf
tf.enable_eager_execution()
import pandas as pd
import math
import numpy as np
import itertools
import collections
import re
from datetime import datetime

def scan(folder):   #get all file names      
    x=[]
    for path,dirs,files in os.walk(folder):
        for f in files:
            if f.startswith('segment'):
                x.append(os.path.join(path,f))
    return(x)

def getID(f):         #extract digits from files name as ID
    f = '_'.join(re.findall(r'\d+', f))       
    return f

def CountLLabels(frame):    #number of laser labels in a frame
    return len(frame.laser_labels)

def converttime(t):
    t = t/1000000
    t = datetime.fromtimestamp(t)
    return t


def Ped2df(SegID, tstamp, pedLoc,
           pedMeta, VehPos, tday, w):
    length = len(pedLoc)
    t, px, py, pz, \
    px_GF, py_GF, pz_GF, \
    vx_GF, vy_GF, vz_GF, \
    pw, pl, pht, \
    phd, psx, psy, \
    paccx, paccy = ([] for i in range(18))

    for i in range(length):
        t.append(tstamp[i])
        pw.append(pedLoc[i].width)
        pl.append(pedLoc[i].length)
        pht.append(pedLoc[i].height)
        phd.append(pedLoc[i].heading)
        psx.append(pedMeta[i].speed_x)
        psy.append(pedMeta[i].speed_y)
        paccx.append(pedMeta[i].accel_x)
        paccy.append(pedMeta[i].accel_y)

        # Ped position in Vehicle Frame
        PedPoseVF = np.array([pedLoc[i].center_x,
                              pedLoc[i].center_y,
                              pedLoc[i].center_z, 1])
        px.append(PedPoseVF[0])
        py.append(PedPoseVF[1])
        pz.append(PedPoseVF[2])
        # Ped position in Global Frame
        PedPoseGF = np.matmul(np.reshape(np.array(VehPos[i].transform), [4, 4]), PedPoseVF)
        px_GF.append(PedPoseGF[0])
        py_GF.append(PedPoseGF[1])
        pz_GF.append(PedPoseGF[2])
        # SDC position in Global Frame
        VehPoseGF = np.matmul(np.reshape(np.array(VehPos[i].transform), [4, 4]), [0, 0, 0, 1])
        vx_GF.append(VehPoseGF[0])
        vy_GF.append(VehPoseGF[1])
        vz_GF.append(VehPoseGF[2])

    data = np.array([t, px, py, pz,
                     px_GF, py_GF, pz_GF,
                     vx_GF, vy_GF, vz_GF,
                     pw, pl, pht, phd,
                     psx, psy, paccx, paccy]).T.tolist()

    columnsnames = ['Timestamp', 'x', 'y', 'z',
                    'x_GF', 'y_GF', 'pz_GF',
                    'x_GF', 'y_GF', 'z_GF',
                    'Width', 'Length', 'Height', 'Heading',
                    'Speed x', 'Speed y', 'accel x', 'accel y']
    pedDF = pd.DataFrame(data, columns=columnsnames)

    # transform = pedDF['vpose'].apply(pd.Series)
    # transform = transform.rename(columns= lambda x: 'AVtransform_' +str(x))
    # pedDF = pedDF.drop(columns= 'vpose')
    # pedDF = pd.concat([pedDF[:],transform[:]],axis=1)

    pedDF['Weather'] = w
    pedDF['Time of Day'] = tday
    pedDF['Segment ID'] = SegID
    return pedDF


def PedExtract(SegID,Frames):
    pedLoc = collections.defaultdict(list)  # pedestrians label box
    pedMeta = collections.defaultdict(list)  # pedestrians label metadata
    VehPos = collections.defaultdict(list)  # SDC transformation matrix
    tstamp = collections.defaultdict(list)
    tday = Frames[SegID][0].context.stats.time_of_day
    w = Frames[SegID][0].context.stats.weather

    for j in range(len(Frames[SegID])):
        t = Frames[SegID][j].timestamp_micros  # timestamp
        vpos = Frames[SegID][j].pose
        Nlabels = CountLLabels(Frames[SegID][j])  # number of laser labels

        for k in range(Nlabels):
            LidarLab = Frames[SegID][j].laser_labels[k]
            if LidarLab.type == 2:  # type 2: pedestrian

                PedID = LidarLab.id

                tstamp[PedID].append(t)
                pedLoc[PedID].append(LidarLab.box)

                pedMeta[PedID].append(LidarLab.metadata)

                VehPos[PedID].append(Frames[SegID][j].pose)

    Ped = {}
    for p in list(pedLoc.keys()):
        Ped[p] = Ped2df(SegID, tstamp[p], pedLoc[p],
                        pedMeta[p], VehPos[p], tday, w)

    return Ped

def MultiFrame(Frames):
    PedTraj = {}
    for i in range(len(Frames)):
        SegID = list(Frames.keys())[i]
        PedTraj.update(PedExtract(SegID,Frames))
    return PedTraj