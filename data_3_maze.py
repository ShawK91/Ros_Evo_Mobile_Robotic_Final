#!/usr/bin/env python
import rospy, numpy as np
from std_msgs.msg import String
from geometry_msgs.msg import Twist, PoseStamped, Pose
from sensor_msgs.msg import LaserScan
from move_base_msgs.msg import MoveBaseActionGoal
from actionlib_msgs.msg import GoalStatusArray
from nav_msgs.msg import Odometry
import sys
import cPickle
import time


tag = "\n".join(sys.argv)
tag = str(tag[-1])
#print tag
filename_x = 'x_' + tag + '.csv' 
filename_y = 'y_' + tag + '.csv' 
#print filename_x, filename_y

class Data_gatherer:
	def __init__(self):
		self.train_x = np.zeros((1,32))
		self.train_y = np.zeros((1,3))
		self.is_record = False
		self.is_sync = False
		self.is_pursuing = False
		self.base_odom = Pose()


		self.oracle_x = [[-1,-1,-1], [-1,-1,1], [-1,1,-1], [-1,1,1], [1,-1,-1], [1,-1,1], [1,1,-1], [1,1,1]]
		self.oracle_y = [[3.348, 0.955], [0.550, 1.051], [0.546, 8.904], [3.299, 8.929], [6.583, 9.0120], [9.087, 8.966], [9.316,0.897], [6.569,0.926]]
		self.oracle_y = self.format_goal(self.oracle_y)
		self.home_coordinates = [[5.0, 4.0]]
		self.home_goal = self.format_goal(self.home_coordinates)


	def update_command(self, cmd):
		if self.is_record:
			self.train_y = np.concatenate((self.train_y, np.array([[cmd.linear.x, cmd.linear.y, cmd.angular.z]])))
			self.is_sync = True

	def update_odom(self, odom):
		self.base_odom = odom.pose


	def update_scan(self, scan):
		if self.is_sync and self.is_record:
			scan_readings = scan.ranges
			scan_readings = np.array(scan_readings)
			scan_readings = scan_readings[::20]
			scan_readings = np.reshape(scan_readings, (1, len(scan_readings)))
			self.train_x = np.concatenate((self.train_x, scan_readings))
			self.is_sync = False

	def update_goal_status(self, status):
		l = status.status_list
		#print len(l)
		if len(l) > 0:
			if l[-1].status == 1:
				self.is_pursuing =  True
			elif l[-1].status == 3:
				self.is_pursuing = False


	def format_goal(self, vals):
		goals = []
		for val in vals:
			y = MoveBaseActionGoal()
			y.goal.target_pose.header.frame_id = 'map'
			y.goal.target_pose.pose.orientation.x = 0.0
			y.goal.target_pose.pose.orientation.y = 0.0
			y.goal.target_pose.pose.orientation.z = 0.0
			y.goal.target_pose.pose.orientation.w = 1.0
			y.goal.target_pose.pose.position.x = val[0]
			y.goal.target_pose.pose.position.y = val[1]
			#print y
			goals.append(y)
		return goals


	def run_trip(self, location):

		#Go to desired location
		if handle.is_pursuing == False:
			goal_pub.publish(self.oracle_y[location])
			self.is_record = True
			time.sleep(0.3)
			while handle.is_pursuing: None
			self.is_record = False


		#Return home
		if handle.is_pursuing == False:
			goal_pub.publish(self.home_goal[0])
			time.sleep(0.3)
			while handle.is_pursuing: None




	   


if __name__ == "__main__":
	handle = Data_gatherer()
	rospy.init_node('DataCollector', anonymous=True) #Node
	scan_sub = rospy.Subscriber('/scan', LaserScan, handle.update_scan)
	cmd_sub = rospy.Subscriber('/navigation_velocity_smoother/raw_cmd_vel', Twist, handle.update_command)
	goal_reached_sub = rospy.Subscriber('/move_base/status', GoalStatusArray, handle.update_goal_status)
	odom_sub = rospy.Subscriber('/base_pose_ground_truth', Odometry, handle.update_odom)


	goal_pub = rospy.Publisher('/move_base/goal',MoveBaseActionGoal, queue_size=1)
	goal_pub.publish(MoveBaseActionGoal()); time.sleep(0.3) #Dummy hack


	handle.run_trip(1)




	
	while not rospy.is_shutdown():
		rospy.spin()
		
	# print handle.train_x.shape, handle.train_y.shape
	# np.savetxt(filename_x, handle.train_x, delimiter = ',')
	# np.savetxt(filename_y, handle.train_y, delimiter = ',')

    





	#def reset_odom(self, odom_sub):
		#print 'RESET'
	 	#odom = Odometry()
		#odom.pose.pose.position.x = 5.0
		#odom.pose.pose.position.y = 4.0
		#odom.pose.pose.orientation.w = 1.0
		#odom_sub.publish(odom)

