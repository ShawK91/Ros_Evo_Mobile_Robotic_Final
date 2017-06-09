#!/usr/bin/env python
import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
#from sensor_msgs import LaserScan
from math import tanh
from random import randint
import math
import  cPickle
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import random
import numpy as np, sys, torch
from copy import deepcopy
import torch.nn.functional as F
from scipy.special import expit

class GRUMB(nn.Module):
	def __init__(self, input_size, memory_size, output_size, output_activation):
		super(GRUMB, self).__init__()

		self.input_size = input_size; self.memory_size = memory_size; self.output_size = output_size
		if output_activation == 'sigmoid': self.output_activation = F.sigmoid
		elif output_activation == 'tanh': self.output_activation = F.tanh
		else: self.output_activation = None
		#self.fast_net = Fast_GRUMB(input_size, memory_size, output_size, output_activation)

		#Input gate
		self.w_inpgate = Parameter(torch.rand(input_size, memory_size), requires_grad=1)
		self.w_rec_inpgate = Parameter(torch.rand(output_size, memory_size), requires_grad=1)
		self.w_mem_inpgate = Parameter(torch.rand(memory_size, memory_size), requires_grad=1)

		#Block Input
		self.w_inp = Parameter(torch.rand(input_size, memory_size), requires_grad=1)
		self.w_rec_inp = Parameter(torch.rand(output_size, memory_size), requires_grad=1)

		#Read Gate
		self.w_readgate = Parameter(torch.rand(input_size, memory_size), requires_grad=1)
		self.w_rec_readgate = Parameter(torch.rand(output_size, memory_size), requires_grad=1)
		self.w_mem_readgate = Parameter(torch.rand(memory_size, memory_size), requires_grad=1)

		#Write Gate
		self.w_writegate = Parameter(torch.rand(input_size, memory_size), requires_grad=1)
		self.w_rec_writegate = Parameter(torch.rand(output_size, memory_size), requires_grad=1)
		self.w_mem_writegate = Parameter(torch.rand(memory_size, memory_size), requires_grad=1)

		#Output weights
		self.w_hid_out = Parameter(torch.rand(memory_size, output_size), requires_grad=1)

		#Biases
		self.w_input_gate_bias = Parameter(torch.zeros(1, memory_size), requires_grad=1)
		self.w_block_input_bias = Parameter(torch.zeros(1, memory_size), requires_grad=1)
		self.w_readgate_bias = Parameter(torch.zeros(1, memory_size), requires_grad=1)
		self.w_writegate_bias = Parameter(torch.zeros(1, memory_size), requires_grad=1)

		# Adaptive components
		self.mem = Variable(torch.zeros(1, self.memory_size), requires_grad=1)
		self.out = Variable(torch.zeros(1, self.output_size), requires_grad=1)

	def reset(self):
		# Adaptive components
		self.mem = Variable(torch.zeros(1, self.memory_size), requires_grad=1)
		self.out = Variable(torch.zeros(1, self.output_size), requires_grad=1)

	# Some bias
	def graph_compute(self, input, rec_output, mem):
		# Compute hidden activation
		block_inp = F.sigmoid(input.mm(self.w_inp) + rec_output.mm(self.w_rec_inp) + self.w_block_input_bias)
		inp_gate = F.sigmoid(input.mm(self.w_inpgate) + mem.mm(self.w_mem_inpgate) + rec_output.mm(
		self.w_rec_inpgate) + self.w_input_gate_bias)
		inp_out = block_inp * inp_gate

		mem_out = F.sigmoid(input.mm(self.w_readgate) + rec_output.mm(self.w_rec_readgate) + mem.mm(self.w_mem_readgate) + self.w_readgate_bias) * mem

		hidden_act = mem_out + inp_out

		write_gate_out = F.sigmoid(input.mm(self.w_writegate) + mem.mm(self.w_mem_writegate) + rec_output.mm(self.w_rec_writegate) + self.w_writegate_bias)
		mem = mem + write_gate_out * F.tanh(hidden_act)

		output = hidden_act.mm(self.w_hid_out)
		if self.output_activation != None: output = self.output_activation(output)

		return output, mem


	def forward(self, input):
		x = Variable(torch.Tensor(input), requires_grad=True); x = x.unsqueeze(0)
		self.out, self.mem = self.graph_compute(x, self.out, self.mem)
		return self.out

	def predict(self, input):
		out = self.forward(input)
		output = out.data.numpy()
		return output

	def turn_grad_on(self):
		for param in self.parameters():
			param.requires_grad = True
			param.volatile = False

	def turn_grad_off(self):
		for param in self.parameters():
			param.requires_grad = False
			param.volatile = True

	def to_fast_net(self):
		keys = self.state_dict().keys()  # Get all keys
		params = self.state_dict()  # Self params
		fast_net_params = self.fast_net.param_dict  # Fast Net params
		for key in keys:
			fast_net_params[key][:] = params[key].numpy()

	def from_fast_net(self):
		keys = self.state_dict().keys() #Get all keys
		params = self.state_dict() #Self params
		fast_net_params = self.fast_net.param_dict #Fast Net params
		for key in keys:
			params[key][:] = torch.from_numpy(fast_net_params[key])


#Class with stopper controller
class GRUMB_controller:
	def __init__(self):
		rospy.init_node('stop_controller', anonymous=True) #Node
		self.pub = rospy.Publisher('/navigation_velocity_smoother/raw_cmd_vel',Twist, queue_size=10) #Publisher to command velocity
		self.sub = rospy.Subscriber("/scan", LaserScan, self.controller_callback) #Subscriber to laser data
		#Control Action defined and initialized 
		self.action = Twist() 
		self.action.linear.x = 0.0
		self.action.angular.z = 0.0
		self.action.linear.y = 0.0

		#Define max applicable to robot
		self.max_speed = 1.0
		self.grumb = GRUMB(32, 25, 3, None)



	def controller_callback(self, scan):
		scan_readings = scan.ranges
		scan_readings = np.array(scan_readings)
		scan_readings = scan_readings[::20]
		#Get parameters from the server
		input = list(scan_readings)
		net_out = self.grumb.forward(input).data.numpy()[0,:]
		
		print net_out
		self.action.linear.x = net_out[0]
		self.action.angular.z = net_out[1]
		self.action.linear.y = net_out[2]
		self.pub.publish(self.action) #Send control


if __name__ == '__main__':


  controller = GRUMB_controller()
  rospy.spin()









