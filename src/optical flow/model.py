import numpy as np
import pandas as pd
import cv2
import torch

class Model:
    
    NUM_WARM_UP_IMAGES = 12
    NUM_PREDICTION_TIMESTEPS = 4
    
    
    def __init__(self, batch_input):
        
        self.batch_input = batch_input
        
    def compute_flows(self, **kwargs):
        
        flows = []
        
        for image_i in range(self.NUM_WARM_UP_IMAGES):
            flow = cv2.calcOpticalFlowFarneback(
                prev=self.batch_input[image_i-1], next=self.batch_input[image_i], flow=None, **kwargs)
            flows.append(flow)
        return np.stack(flows).astype(np.float32)
    
    def weighted_average(self, flow):
        return np.average(flow, axis=0, weights=range(1, self.NUM_WARM_UP_IMAGES+1)).astype(np.float32)
        
    def remap_image(self, image, flow):

        height, width = flow.shape[:2]
        remap = -flow.copy()
        remap[..., 0] += np.arange(width)  # x map
        remap[..., 1] += np.arange(height)[:, np.newaxis]  # y map
        remapped_image = cv2.remap(src=image, map1=remap, map2=None, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        
        return remapped_image
    
    def generate(self):
        
        targets = []
        
        start_image = self.batch_input[-1]

        flows_default = self.compute_flows(pyr_scale=0.5, levels=200, winsize=13, 
        iterations=25, poly_n=5, poly_sigma=0.7, 
        flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

        flow_default = self.weighted_average(flows_default)
        
        for i in range(self.NUM_PREDICTION_TIMESTEPS):
            remapped_image = self.remap_image(start_image, flow_default * i)
            
            targets.append(remapped_image)
            
        return np.array(targets).astype('float32')


class Model2:
    
    NUM_WARM_UP_IMAGES = 16
    NUM_PREDICTION_TIMESTEPS = 4
    
    
    def __init__(self, batch_input):
        
        self.batch_input = batch_input
        
    def compute_flows(self, **kwargs):
        
        flows = []
        
        for image_i in range(self.NUM_WARM_UP_IMAGES):
            flow = cv2.calcOpticalFlowFarneback(
                prev=self.batch_input[image_i-1], next=self.batch_input[image_i], flow=None, **kwargs)
            flows.append(flow)
        return np.stack(flows).astype(np.float32)
    
    def weighted_average(self, flow):
        return np.average(flow, axis=0, weights=range(1, self.NUM_WARM_UP_IMAGES+1)).astype(np.float32)
        
    def remap_image(self, image, flow):

        height, width = flow.shape[:2]
        remap = -flow.copy()
        remap[..., 0] += np.arange(width)  # x map
        remap[..., 1] += np.arange(height)[:, np.newaxis]  # y map
        remapped_image = cv2.remap(src=image, map1=remap, map2=None, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        
        return remapped_image
    
    def generate(self):
        
        targets = []
        
        start_image = self.batch_input[-1]
        mx = np.amax(start_image[32:96, 32:96]) / 10
        mn = np.amin(start_image[32:96, 32:96]) / 10
        max_x,max_y = np.unravel_index(start_image[32:96, 32:96].argmax(), start_image[32:96, 32:96].shape)
        min_x,min_y = np.unravel_index(start_image[32:96, 32:96].argmin(), start_image[32:96, 32:96].shape)
        weights = (mx-mn) / 4

        flows_default = self.compute_flows(pyr_scale=0.5, levels=200, winsize=30, 
        iterations=25, poly_n=5, poly_sigma=1.1, 
        flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

        flow_default = self.weighted_average(flows_default)
        
        for i in range(self.NUM_PREDICTION_TIMESTEPS):
            remapped_image = self.remap_image(start_image, flow_default * i)
            out = (np.rint(remapped_image / (weights )) * (weights))
            out[min_x, min_y] -= mn
            out[max_x, max_y] += mx
            targets.append(remapped_image)
            
        return np.array(targets).astype('float32')


class Model3:
    
    NUM_WARM_UP_IMAGES = 20
    NUM_PREDICTION_TIMESTEPS = 4
    
    
    def __init__(self, batch_input):
        
        self.batch_input = batch_input
        
    def compute_flows(self, **kwargs):
        
        flows = []
        
        for image_i in range(self.NUM_WARM_UP_IMAGES):
            flow = cv2.calcOpticalFlowFarneback(
                prev=self.batch_input[image_i-1], next=self.batch_input[image_i], flow=None, **kwargs)
            flows.append(flow)
        return np.stack(flows).astype(np.float32)
    
    def weighted_average(self, flow):
        return np.average(flow, axis=0, weights=range(1, self.NUM_WARM_UP_IMAGES+1)).astype(np.float32)
        
    def remap_image(self, image, flow):

        height, width = flow.shape[:2]
        remap = -flow.copy()
        remap[..., 0] += np.arange(width)  # x map
        remap[..., 1] += np.arange(height)[:, np.newaxis]  # y map
        remapped_image = cv2.remap(src=image, map1=remap, map2=None, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        
        return remapped_image
    
    def generate(self):
        
        targets = []
        
        start_image = self.batch_input[-1]
        mx = np.amax(start_image[32:96, 32:96]) / 10
        mn = np.amin(start_image[32:96, 32:96]) / 10
        max_x,max_y = np.unravel_index(start_image[32:96, 32:96].argmax(), start_image[32:96, 32:96].shape)
        min_x,min_y = np.unravel_index(start_image[32:96, 32:96].argmin(), start_image[32:96, 32:96].shape)
        weights = (mx-mn) / 8

        flows_default = self.compute_flows(pyr_scale=0.5, levels=200, winsize=40, 
        iterations=25, poly_n=7, poly_sigma=1.1, 
        flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

        flow_default = self.weighted_average(flows_default)
        
        for i in range(self.NUM_PREDICTION_TIMESTEPS):
            remapped_image = self.remap_image(start_image, flow_default * i)
            out = (np.rint(remapped_image / (weights )) * (weights))
            out[min_x, min_y] -= mn
            out[max_x, max_y] += mx
            targets.append(remapped_image)
            
        return np.array(targets).astype('float32')

class Model4:
    
    NUM_WARM_UP_IMAGES = 24
    NUM_PREDICTION_TIMESTEPS = 4
    
    
    def __init__(self, batch_input):
        
        self.batch_input = batch_input
        
    def compute_flows(self, **kwargs):
        
        flows = []
        
        for image_i in range(self.NUM_WARM_UP_IMAGES):
            flow = cv2.calcOpticalFlowFarneback(
                prev=self.batch_input[image_i-1], next=self.batch_input[image_i], flow=None, **kwargs)
            flows.append(flow)
        return np.stack(flows).astype(np.float32)
    
    def weighted_average(self, flow):
        return np.average(flow, axis=0, weights=range(1, self.NUM_WARM_UP_IMAGES+1)).astype(np.float32)
        
    def remap_image(self, image, flow):

        height, width = flow.shape[:2]
        remap = -flow.copy()
        remap[..., 0] += np.arange(width)  # x map
        remap[..., 1] += np.arange(height)[:, np.newaxis]  # y map
        remapped_image = cv2.remap(src=image, map1=remap, map2=None, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        
        return remapped_image
    
    def generate(self):
        
        targets = []
        
        start_image = self.batch_input[-1]
        mx = np.amax(start_image[32:96, 32:96]) / 10
        mn = np.amin(start_image[32:96, 32:96]) / 10
        max_x,max_y = np.unravel_index(start_image[32:96, 32:96].argmax(), start_image[32:96, 32:96].shape)
        min_x,min_y = np.unravel_index(start_image[32:96, 32:96].argmin(), start_image[32:96, 32:96].shape)
        weights = (mx-mn) / 12

        flows_default = self.compute_flows(pyr_scale=0.5, levels=200, winsize=60, 
        iterations=25, poly_n=7, poly_sigma=1.5, 
        flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

        flow_default = self.weighted_average(flows_default)
        
        for i in range(self.NUM_PREDICTION_TIMESTEPS):
            remapped_image = self.remap_image(start_image, flow_default * i)
            out = (np.rint(remapped_image / (weights )) * (weights))
            out[min_x, min_y] -= mn
            out[max_x, max_y] += mx

            targets.append(remapped_image)
            
        return np.array(targets).astype('float32')

class Model5:
    
    NUM_WARM_UP_IMAGES = 28
    NUM_PREDICTION_TIMESTEPS = 4
    
    
    def __init__(self, batch_input):
        
        self.batch_input = batch_input
        
    def compute_flows(self, **kwargs):
        
        flows = []
        
        for image_i in range(self.NUM_WARM_UP_IMAGES):
            flow = cv2.calcOpticalFlowFarneback(
                prev=self.batch_input[image_i-1], next=self.batch_input[image_i], flow=None, **kwargs)
            flows.append(flow)
        return np.stack(flows).astype(np.float32)
    
    def weighted_average(self, flow):
        return np.average(flow, axis=0, weights=range(1, self.NUM_WARM_UP_IMAGES+1)).astype(np.float32)
        
    def remap_image(self, image, flow):

        height, width = flow.shape[:2]
        remap = -flow.copy()
        remap[..., 0] += np.arange(width)  # x map
        remap[..., 1] += np.arange(height)[:, np.newaxis]  # y map
        remapped_image = cv2.remap(src=image, map1=remap, map2=None, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        
        return remapped_image
    
    def generate(self):
        
        targets = []
        
        start_image = self.batch_input[-1]
        mx = np.amax(start_image[32:96, 32:96]) / 10
        mn = np.amin(start_image[32:96, 32:96]) / 10
        max_x,max_y = np.unravel_index(start_image[32:96, 32:96].argmax(), start_image[32:96, 32:96].shape)
        min_x,min_y = np.unravel_index(start_image[32:96, 32:96].argmin(), start_image[32:96, 32:96].shape)
        weights = (mx-mn) / 16

        flows_default = self.compute_flows(pyr_scale=0.5, levels=200, winsize=85, 
        iterations=25, poly_n=7, poly_sigma=1.5, 
        flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

        flow_default = self.weighted_average(flows_default)
        
        for i in range(self.NUM_PREDICTION_TIMESTEPS):
            remapped_image = self.remap_image(start_image, flow_default * i)
            out = (np.rint(remapped_image / (weights )) * (weights))
            out[min_x, min_y] -= mn
            out[max_x, max_y] += mx
            targets.append(remapped_image)
            
        return np.array(targets).astype('float32')


class Model6:
    
    NUM_WARM_UP_IMAGES = 32
    NUM_PREDICTION_TIMESTEPS = 4
    
    
    def __init__(self, batch_input):
        
        self.batch_input = batch_input
        
    def compute_flows(self, **kwargs):
        
        flows = []
        
        for image_i in range(self.NUM_WARM_UP_IMAGES):
            flow = cv2.calcOpticalFlowFarneback(
                prev=self.batch_input[image_i-1], next=self.batch_input[image_i], flow=None, **kwargs)
            flows.append(flow)
        return np.stack(flows).astype(np.float32)
    
    def weighted_average(self, flow):
        return np.average(flow, axis=0, weights=range(1, self.NUM_WARM_UP_IMAGES+1)).astype(np.float32)
        
    def remap_image(self, image, flow):

        height, width = flow.shape[:2]
        remap = -flow.copy()
        remap[..., 0] += np.arange(width)  # x map
        remap[..., 1] += np.arange(height)[:, np.newaxis]  # y map
        remapped_image = cv2.remap(src=image, map1=remap, map2=None, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        
        return remapped_image
    
    def generate(self):
        
        targets = []
        
        start_image = self.batch_input[-1]
        mx = np.amax(start_image[32:96, 32:96]) / 10
        mn = np.amin(start_image[32:96, 32:96]) / 10
        max_x,max_y = np.unravel_index(start_image[32:96, 32:96].argmax(), start_image[32:96, 32:96].shape)
        min_x,min_y = np.unravel_index(start_image[32:96, 32:96].argmin(), start_image[32:96, 32:96].shape)
        weights = (mx-mn) / 20
        

        flows_default = self.compute_flows(pyr_scale=0.5, levels=200, winsize=100, 
        iterations=25, poly_n=7, poly_sigma=1.5, 
        flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

        flow_default = self.weighted_average(flows_default)
        
        for i in range(self.NUM_PREDICTION_TIMESTEPS):
            remapped_image = self.remap_image(start_image, flow_default * i)
            out = (np.rint(remapped_image / (weights )) * (weights))
            out[min_x, min_y] -= mn
            out[max_x, max_y] += mx

            targets.append(remapped_image)
            
        return np.array(targets).astype('float32')
