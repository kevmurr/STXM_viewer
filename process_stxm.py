import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
import scipy.ndimage.measurements as snm



mode="dpc" #"stxm" or "dpc"
import pyqtgraph as pg
import sys
#scanno=sys.argv[1]
scanno=59597
dim=(50,50)
R=(-1,1,-1,1)
#roi=None
roi=(100,130,80,110)
#vmax=700000
vmax=None
vmin=None
#vmin=800000
#scanno=str(sys.argv[1])
#dim=sys.argv[2].astype(np.int)
#R=sys.argv[3].astype(np.float)


def fill_edge_pixels(data, edge_ss=202, edge_fs=172):
    d = np.zeros((data.shape[0], data.shape[1] + 4, data.shape[2] + 4), dtype=data.dtype)
    nx1 = edge_ss
    ny1 = edge_fs
    for i in range(data.shape[0]):
        tt = data[i]
        nx, ny = np.shape(tt)
        t = np.zeros((nx + 4, ny + 4))

        t[:nx1, :ny1] = tt[:nx1, :ny1]
        t[:nx1, ny1 + 6:] = tt[:nx1, ny1 + 2:]
        t[nx1 + 6:, :ny1] = tt[nx1 + 2:, :ny1]
        t[nx1 + 6:, ny1 + 6:] = tt[nx1 + 2:, ny1 + 2:]

        a = np.rint(tt[nx1, :ny1] / 3).astype(data.dtype)
        t[nx1, :ny1] = a
        t[nx1 + 1, :ny1] = a
        t[nx1 + 2, :ny1] = a

        a = np.rint(tt[nx1 + 1, :ny1] / 3).astype(data.dtype)
        t[nx1 + 3, :ny1] = a
        t[nx1 + 4, :ny1] = a
        t[nx1 + 5, :ny1] = a

        a = np.rint(tt[nx1, ny1 + 2:] / 3).astype(data.dtype)
        t[nx1, ny1 + 6:] = a
        t[nx1 + 1, ny1 + 6:] = a
        t[nx1 + 2, ny1 + 6:] = a

        a = np.rint(tt[nx1 + 1, ny1 + 2:] / 3).astype(data.dtype)
        t[nx1 + 3, ny1 + 6:] = a
        t[nx1 + 4, ny1 + 6:] = a
        t[nx1 + 5, ny1 + 6:] = a

        a = np.rint(tt[:nx1, ny1] / 3).astype(data.dtype)
        t[:nx1, ny1] = a
        t[:nx1, ny1 + 1] = a
        t[:nx1, ny1 + 2] = a

        a = np.rint(tt[:nx1, ny1 + 1] / 3).astype(data.dtype)
        t[:nx1, ny1 + 3] = a
        t[:nx1, ny1 + 4] = a
        t[:nx1, ny1 + 5] = a

        a = np.rint(tt[nx1 + 2:, ny1] / 3).astype(data.dtype)
        t[nx1 + 6:, ny1] = a
        t[nx1 + 6:, ny1 + 1] = a
        t[nx1 + 6:, ny1 + 2] = a

        a = np.rint(tt[nx1 + 2:, ny1 + 1] / 3).astype(data.dtype)
        t[nx1 + 6:, ny1 + 3] = a
        t[nx1 + 6:, ny1 + 4] = a
        t[nx1 + 6:, ny1 + 5] = a

        t[nx1:nx1 + 3, ny1:ny1 + 3] = int(round(tt[nx1, ny1] / 9))
        t[nx1:nx1 + 3, ny1 + 3:ny1 + 6] = int(round(tt[nx1, ny1 + 1] / 9))
        t[nx1 + 3:nx1 + 6, ny1:ny1 + 3] = int(round(tt[nx1 + 1, ny1] / 9))
        t[nx1 + 3:nx1 + 6, ny1 + 3:ny1 + 6] = int(round(tt[nx1 + 1, ny1 + 1] / 9))

        d[i] = t

    return d


def load_plot_stxm(scanno, dim, R, roi=None ,transpose=False, path="/gpfs/cfel/cxi/scratch/user/murrayke/Raw_Data/Brookhaven/March_2019/",
                   savepath="/gpfs/cfel/cxi/scratch/user/murrayke/Processed_Data/Brookhaven/Mar_2019/STXM_images/", edge_ss=202, edge_fs=172, vmin=None,
                   vmax=None):
    # Dim:(ss,fs)
    # R: (ssmin,ssmax,fsmin,fsmax) [µm]

    # Make_STXM_array
    stxm = np.zeros(dim)

    # Load Data
    print("Loading Data")
    f = h5.File(path + "scan_" + str(scanno) + ".h5", "r")
    print("Loading finished")
    if roi==None:
        data = fill_edge_pixels(f["entry/instrument/detector/data"][()], edge_ss=202, edge_fs=172)
        roitxt=""
    else:
        data = fill_edge_pixels(f["entry/instrument/detector/data"][()], edge_ss=202, edge_fs=172)[:,roi[0]:roi[1],roi[2]:roi[3]]
        roitxt = "_roi_%s_%s_%s_%s"%(roi[0],roi[1],roi[2],roi[3])
    print(data.shape)
    # Get all ionchamber readings
    # ionchamber_readings=np.zeros((data.shape[0]))
    print("Processing and plotting")
    name_ionchamber = "sclr1_ch3"
    txtfnam = path + "scan_" + str(scanno) + ".txt"
    f2 = open(txtfnam, "r")
    f2_line_list = []
    for line in f2:
        f2_line_list.append(line)
    list_motornames = f2_line_list[0].split("\t")
    motorpositions = np.zeros((len(f2_line_list) - 1, len(list_motornames)))
    # Looking which entries are not numbers
    test_list_now = f2_line_list[1].split("\t")

    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    list_nonnumbers = []
    for i in range(0, len(test_list_now), 1):
        # print(test_list_now[i], is_number(test_list_now[i]), i)
        if is_number(test_list_now[i]) is False:
            list_nonnumbers.append(i)
    # print('list:', list_nonnumbers)
    #####################################
    for i in range(0, motorpositions.shape[0], 1):
        list_now = f2_line_list[i + 1].split("\t")
        # setting non numbers to zero
        for i2 in list_nonnumbers:
            list_now[i2] = 0
        motorpositions[i, :] = np.asarray(list_now)
    i_ion = None
    for i in range(0, len(list_motornames), 1):
        if list_motornames[i] == str(name_ionchamber):
            i_ion = i
    if i_ion is None:
        raise IOError("Coulndt find the ionchamber motorname")
    ionchamber_readings = motorpositions[:, i_ion]
    ionchamber_readings = ionchamber_readings.astype(np.float) / ionchamber_readings.max()

    # Now go through STXM
    if mode=="stxm":
        for i1 in range(0, dim[0]):
            for i2 in range(0, dim[1]):
                if (i1*dim[1]+i2)<(dim[0]*dim[1]):
                    stxm[i1, i2] = np.sum(data[i1 * dim[1] + i2, :, :]) / ionchamber_readings[i1 * dim[1] + i2]
                else:
                    stxm[i1,i2]=None
    elif mode=="dpc":
        dpc_map=stxm
        for i1 in range(0,dim[0]):
            for i2 in range(0,dim[1]):
                if (i1 * dim[1] + i2) < (dim[0] * dim[1]):
                    dpc_r=snm.center_of_mass(data[i1 * dim[1] + i2, :, :] / ionchamber_readings[i1 * dim[1] + i2])
                    dpc_map[i1,i2]=np.sqrt(dpc_r[0]**2+dpc_r[1]**2)
                else:
                    dpc_map[i1,i2]=None
        stxm=dpc_map
    else:
        IOError("Bad mode!")
    extent = np.roll(R, 2)
    print(np.amin(stxm),np.median(stxm),np.amax(stxm))
    if transpose == True:
        stxm = np.rot90(stxm)
        extent = np.roll(extent, 2)
    # Now save image
    f.close()
    if vmax == None and vmin == None:
        plt.imshow(stxm, extent=extent,cmap="gray")
    else:
        if vmax == None:
            plt.imshow(stxm, extent=extent, cmap="gray", vmin=vmin)
        if vmin == None:
            plt.imshow(stxm, extent=extent, cmap="gray", vmax=vmax)
        if vmin != None and vmax != None:
            plt.imshow(stxm, extent=extent, cmap="gray", vmin=vmin, vmax=vmax)
    plt.xlabel("Distance (µm)")
    plt.ylabel("Distance (µm)")
    if mode=="stxm":
        plt.savefig(savepath + "image_"+roitxt+"_" + str(scanno) + ".png", dpi=600)
    if mode=="dpc":
        plt.savefig(savepath + "dpc_" + roitxt + "_" + str(scanno) + ".png", dpi=600)
    
   
    #configarr=("scanno="+ str(scanno),"dim="+str(dim), "R="+str(R), "roi="+str(roi) ,"transpose="+str(transpose), "path="+str(path),"savepath="+str(savepath), "edge_ss="+str(edge_ss),"edge_fs="+str(edge_fs), "vmin=" +str(vmin),"vmax="+str(vmax))
    #write(savepath + "image_"+roitxt+"_" + str(scanno) + ".txt",configarr)
load_plot_stxm(scanno,dim,R,roi=roi,vmax=vmax,vmin=vmin)