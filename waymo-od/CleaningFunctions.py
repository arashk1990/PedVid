import os
import pandas as pd
import math
import numpy as np
import itertools
import collections
import re
from datetime import datetime


def scan(folder,keyword):   #get all file names      
    x=[]
    for path,dirs,files in os.walk(folder):
        for f in files:
            if f.startswith(keyword):
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


def Ped2df(SegID, tstamp, VehCount,
           PedCount,BikeCount, pedLoc,
           pedMeta, VehPos, tday, w):
    length = len(pedLoc)
    t,vCount,pCount,bCount, \
    px, py, pz, \
    px_GF, py_GF, pz_GF, \
    vx_GF, vy_GF, vz_GF, \
    pw, pl, pht, \
    phd, psx, psy, \
    paccx, paccy = ([] for i in range(21))

    for i in range(length):
        t.append(tstamp[i])
        
        vCount.append(VehCount[i])
        pCount.append(PedCount[i])
        bCount.append(BikeCount[i])
        
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
                     psx, psy, paccx,
                     paccy,vCount,pCount,
                     bCount]).T.tolist()

    columnsnames = ['Timestamp', 'x', 'y', 'z',
                    'px_GF', 'py_GF', 'pz_GF',
                    'x_GF_SDC', 'y_GF_SDC', 'z_GF_SDC',
                    'Width', 'Length', 'Height',
                    'Heading','Speed x', 'Speed y',
                    'accel x', 'accel y','Veh Count','Ped Count','Bike Count']
    pedDF = pd.DataFrame(data, columns=columnsnames)


    pedDF['Weather'] = w
    pedDF['Time of Day'] = tday
    pedDF['Segment ID'] = SegID
    return pedDF


def PedExtract(SegID,Frames,ObjectType):   #ObjectType: to be analysed, unknown:0, veh:1, ped:2, sign:3, bike:4 
    pedLoc = collections.defaultdict(list)  # pedestrians label box
    pedMeta = collections.defaultdict(list)  # pedestrians label metadata
    VehPos = collections.defaultdict(list)  # SDC transformation matrix
    tstamp = collections.defaultdict(list)
    VehCount = collections.defaultdict(list)
    PedCount = collections.defaultdict(list)
    BikeCount = collections.defaultdict(list)
    tday = Frames[SegID][0].context.stats.time_of_day
    w = Frames[SegID][0].context.stats.weather

    for j in range(len(Frames[SegID])):
        t = Frames[SegID][j].timestamp_micros  # timestamp
        # number of each object type in scene (ped,veh,bke)
        obj = Frames[SegID][j].context.stats.laser_object_counts
        if len(obj)>0:
            fun = lambda x: (x.type, x.count)
            objlist = list(map(fun,obj))
            objdict = dict((objlist[i][0],objlist[i][1]) for i in range(len(objlist)))
            objkeys = list(objdict.keys())

            if 1 in objkeys:
                vcount = objdict[1]
            else: vcount = 0

            if 2 in objkeys:
                pcount = objdict[2]
            else: pcount = 0

            if 4 in objkeys:
                bcount = objdict[4]
            else: bcount = 0
        else:
            vcount= pcount= bcount = 0

        vpos = Frames[SegID][j].pose
        Nlabels = CountLLabels(Frames[SegID][j])  # number of laser labels

        for k in range(Nlabels):
            LidarLab = Frames[SegID][j].laser_labels[k]
            if LidarLab.type == ObjectType:  # type 2: pedestrian

                PedID = LidarLab.id

                tstamp[PedID].append(t)
                
                VehCount[PedID].append(vcount)
                PedCount[PedID].append(pcount)
                BikeCount[PedID].append(bcount)
                
                pedLoc[PedID].append(LidarLab.box)

                pedMeta[PedID].append(LidarLab.metadata)

                VehPos[PedID].append(Frames[SegID][j].pose)

    Ped = {}
    for p in list(pedLoc.keys()):
        Ped[p] = Ped2df(SegID, tstamp[p],
                        VehCount[p], PedCount[p],BikeCount[p],
                        pedLoc[p], pedMeta[p], VehPos[p], tday, w)

    return Ped

def MultiFrame(Frames,ObjectType):
    PedTraj = {}
    for i in range(len(Frames)):
        SegID = list(Frames.keys())[i]
        PedTraj.update(PedExtract(SegID,Frames,ObjectType))
    return PedTraj