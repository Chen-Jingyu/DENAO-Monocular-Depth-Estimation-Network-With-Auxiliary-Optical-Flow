import os
import re
import sys
import json
import time
import argparse 

import imageio
import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt

caffe_root = '../' 
sys.path.insert(0, caffe_root + 'python')
import caffe


def prepareonedata(data_path, testing_files):

    rgb_path = data_path + 'images/'
    depth_path = data_path + 'depths/'
    pose_path = data_path + 'poses/'

    ref_index = int(testing_files[0])
    nei_list = testing_files[1:]
    batchsize = 4
    prepared_batch = {"ref_img":np.zeros((batchsize,3,480,640)),"ref_depth":np.zeros((batchsize,1,480,640)),"neighbor_img":np.zeros((batchsize,3,480,640)),"ref_opt":np.zeros((batchsize,2,480,640)),
    "K":np.zeros((batchsize,4,1,1)),"T":np.zeros((batchsize,1,4,4)),"flag":1}
    for image_idx in range(batchsize):
        ref_depth_full = imageio.imread(os.path.join(depth_path, "{:04d}.exr".format(ref_index)))
        ref_img_full = imageio.imread(os.path.join(rgb_path, "{:04d}.png".format(ref_index)))
        with open(os.path.join(pose_path, "{:04d}.json".format(ref_index))) as f:
            r_info = json.load(f)
            r_c_x = r_info["c_x"]
            r_c_y = r_info["c_y"]
            r_f_x = r_info["f_x"]
            r_f_y = r_info["f_y"]
            r_extrinsic = np.array(r_info["extrinsic"])
            
        neighbor_img_full = imageio.imread(os.path.join(rgb_path, "{:04d}.png".format(int(nei_list[image_idx]))))
        with open(os.path.join(pose_path, "{:04d}.json".format(int(nei_list[image_idx])))) as f:
            n_info = json.load(f)
            n_c_x = n_info["c_x"]
            n_c_y = n_info["c_y"]
            n_f_x = n_info["f_x"]
            n_f_y = n_info["f_y"]
            n_extrinsic = np.array(n_info["extrinsic"])

        shared_data = {}
        shared_data["ref_img"] = ref_img_full
        shared_data["ref_depth"] = ref_depth_full
        shared_data["neighbor_img"] = neighbor_img_full
        shared_data["K"] = np.array([n_f_x, n_f_y, n_c_x, n_c_y])
        shared_data["T"] = np.matmul(n_extrinsic,la.inv(r_extrinsic))
        ref_img = shared_data['ref_img'].transpose(2,0,1)[np.newaxis,:]
        ref_depth = shared_data['ref_depth'][np.newaxis,np.newaxis,:]
        neighbor_img = shared_data['neighbor_img'].transpose(2,0,1)[np.newaxis,:]
        K = shared_data['K'][np.newaxis,:, np.newaxis, np.newaxis]
        T = shared_data['T'][np.newaxis,np.newaxis,:]

        prepared_batch["ref_img"][image_idx] = ref_img.astype(np.float32) / 255 - 0.5
        prepared_batch["ref_depth"][image_idx] = ref_depth
        prepared_batch["neighbor_img"][image_idx] = neighbor_img.astype(np.float32) / 255 - 0.5
        prepared_batch["K"][image_idx] = K
        prepared_batch["T"][image_idx] = T
        prepared_batch["img_name"] = "{:04d}.png".format(ref_index)
    return prepared_batch


def main(network, weights , data_path, save_path):

    net = caffe.Net(network,weights,caffe.TEST)

    testing_list_file = data_path + 'testing_list.txt'
    f = open(testing_list_file,"r")
    testing_list = f.readlines()
    f.close()

    testing_line = testing_list[0]
    testing_files = testing_line.split()

    batch = prepareonedata(data_path, testing_files)
    net.blobs['data_rgb/ref'].data[...] = batch["ref_img"]
    net.blobs['data_rgb/ref_depth_layer'].data[...] = batch["ref_img"][:1,:,:,:]
    net.blobs['data_rgb/nei'].data[...] = batch["neighbor_img"]
    net.blobs['data_K'].data[...] = batch["K"]
    net.blobs['data_T'].data[...] = batch["T"]
    net.forward()

    for i in range(len(testing_list)):

        print "index: ", i, batch['img_name']

        testing_line = testing_list[i]
        testing_files = testing_line.split()

        batch = prepareonedata(data_path, testing_files)
        net.blobs['data_rgb/ref'].data[...] = batch["ref_img"]
        net.blobs['data_rgb/ref_depth_layer'].data[...] = batch["ref_img"][:1,:,:,:]
        net.blobs['data_rgb/nei'].data[...] = batch["neighbor_img"]
        net.blobs['data_K'].data[...] = batch["K"]
        net.blobs['data_T'].data[...] = batch["T"]

        net.forward()

        depth = net.blobs['crop_invdepth_0/depth'].data[0,0,:,:]
        
        plt.figure()
        plt.imshow(depth, cmap=plt.get_cmap('viridis'), interpolation='nearest')
        plt.savefig('./image/'+batch['img_name'])


if __name__ == "__main__":

    parser = argparse.ArgumentParser() 

    parser.add_argument('-gpu', type=int, default=0, help='set gpu id')
    parser.add_argument('-network', type=str, default='./' , help='set network prototxt path')
    parser.add_argument('-weights', type=str, default='./', help='set network weight caffemodel path')
    parser.add_argument('-data_path', type=str, default='./', help='set input data path')
    parser.add_argument('-save_path', type=str, default='./', help='set output result path')
    args = parser.parse_args()

    caffe.set_mode_gpu()
    caffe.set_device(args.gpu)

    main(args.network, args.weights , args.data_path, args.save_path)
