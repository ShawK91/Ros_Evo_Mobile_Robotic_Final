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
import numpy as np, os
import mod_mdt as mod, sys
from random import randint
from torch.autograd import Variable
import torch.nn as nn, torch
from torch.nn import Parameter
import torch.nn.functional as F
import torch
import random
from operator import add


is_probe= False
class Tracker(): #Tracker
    def __init__(self, parameters):
        self.foldername = parameters.save_foldername + '/0000_CSV'
        self.fitnesses = []; self.avg_fitness = 0; self.tr_avg_fit = []
        self.hof_fitnesses = []; self.hof_avg_fitness = 0; self.hof_tr_avg_fit = []
        if not os.path.exists(self.foldername):
            os.makedirs(self.foldername)
        self.file_save = 'PyTorch_Multi_TMaze.csv'

    def add_fitness(self, fitness, generation):
        self.fitnesses.append(fitness)
        if len(self.fitnesses) > 100:
            self.fitnesses.pop(0)
        self.avg_fitness = sum(self.fitnesses)/len(self.fitnesses)
        if generation % 10 == 0: #Save to csv file
            filename = self.foldername + '/champ_train' + self.file_save
            self.tr_avg_fit.append(np.array([generation, self.avg_fitness]))
            np.savetxt(filename, np.array(self.tr_avg_fit), fmt='%.3f', delimiter=',')

    def add_hof_fitness(self, hof_fitness, generation):
        self.hof_fitnesses.append(hof_fitness)
        if len(self.hof_fitnesses) > 100:
            self.hof_fitnesses.pop(0)
        self.hof_avg_fitness = sum(self.hof_fitnesses)/len(self.hof_fitnesses)
        if generation % 10 == 0: #Save to csv file
            filename = self.foldername + '/champ_real' + self.file_save
            self.hof_tr_avg_fit.append(np.array([generation, self.hof_avg_fitness]))
            np.savetxt(filename, np.array(self.hof_tr_avg_fit), fmt='%.3f', delimiter=',')

    def save_csv(self, generation, filename):
        self.tr_avg_fit.append(np.array([generation, self.avg_fitness]))
        np.savetxt(filename, np.array(self.tr_avg_fit), fmt='%.3f', delimiter=',')

class SSNE_param:
    def __init__(self):
        self.num_input = 35
        self.num_hnodes = 25
        self.num_output = 3

        self.elite_fraction = 0.04
        self.crossover_prob = 0.05
        self.mutation_prob = 0.9
        self.extinction_prob = 0.004 #Probability of extinction event
        self.extinction_magnituide = 0.5 #Probabilty of extinction for each genome, given an extinction event
        self.weight_magnitude_limit = 10000000
        self.mut_distribution = 3 #1-Gaussian, 2-Laplace, 3-Uniform, ELSE-all 1s

class Parameters:
    def __init__(self):

        #SSNE stuff
        self.spread_exp = True
        self.population_size = 100
        self.load_seed = False
        self.ssne_param = SSNE_param()
        self.total_gens = 10000
        #Determine the nerual archiecture
        self.arch_type = 4 #1 LSTM
                           #2 GRUMB
                           #3 FF
                           # QUASI_GECCO

        self.timesteps = 10



        self.output_activation = 'tanh'
        if self.arch_type == 1: self.arch_type = 'LSTM'
        elif self.arch_type ==2: self.arch_type = 'GRUMB'
        elif self.arch_type == 3: self.arch_type = 'FF'
        elif self.arch_type == 4: self.arch_type = 'QUASI_GRUMB'
        else: sys.exit('Invalid choice of neural architecture')

        self.save_foldername = 'R_AutoNomous_TMaze/'

        # # BackProp
        # self.bprop_max_gens = 100
        # self.bprop_train_examples = 1000
        # self.skip_bprop = False
        # self.load_bprop_seed = False  # Loads a seed population from the save_foldername
        # # IF FALSE: Runs Backpropagation, saves it and uses that


        if not os.path.exists(self.save_foldername):
            os.makedirs(self.save_foldername)

        #Overrides
        if is_probe: self.load_seed = True #Overrides

class Agent_Pop:
    def __init__(self, parameters, i, is_static=False):
        self.parameters = parameters; self.ssne_param = self.parameters.ssne_param
        self.num_input = self.ssne_param.num_input; self.num_hidden = self.ssne_param.num_hnodes; self.num_output = self.ssne_param.num_output
        self.agent_id = i
        self.is_static = is_static

        #####CREATE POPULATION
        self.pop = []
        if is_static: #Static policy agent
            for i in range(self.parameters.population_size):
                self.pop.append(Static_policy(parameters))
        else:
            for i in range(self.parameters.population_size):
                if self.parameters.load_seed and i == 0: #Load seed if option
                    self.pop.append(self.load('champion_' + str(self.agent_id)))

                #Choose architecture
                if self.parameters.arch_type == "GRUMB":
                    self.pop.append(mod.PT_GRUMB(self.num_input, self.num_hidden, self.num_output, output_activation=self.parameters.output_activation))
                elif self.parameters.arch_type == "FF":
                    self.pop.append(mod.PT_FF(self.num_input, self.num_hidden, self.num_output, output_activation=self.parameters.output_activation))
                elif self.parameters.arch_type == "LSTM":
                    self.pop.append(mod.PT_LSTM(self.num_input, self.num_hidden, self.num_output, output_activation=self.parameters.output_activation))
                elif self.parameters.arch_type == "QUASI_GRUMB":
                    self.pop.append(mod.Quasi_GRUMB(self.num_input, self.num_hidden, self.num_output))
                else:
                    sys.exit('Invalid choice of architecture')
        self.champion_ind = None

        #Fitness evaluation list for the generation
        self.fitness_evals = [0.0] * self.parameters.population_size


    def reset(self):
        #Fitness evaluation list for the generation
        self.fitness_evals = [0.0] * self.parameters.population_size



    def load(self, filename):
        return torch.load(self.parameters.save_foldername + filename)

class Task_ROS_TMaze: #Autonomous Navigation T-Maze
    def __init__(self, parameters):
        self.parameters = parameters; self.ssne_param = self.parameters.ssne_param
        self.num_input = self.ssne_param.num_input; self.num_hidden = self.ssne_param.num_hnodes; self.num_output = self.ssne_param.num_output

        if self.parameters.arch_type == 'QUASI_GRUMB': self.ssne = mod.Quasi_GRUMB_SSNE(parameters)
        else: self.ssne = mod.Fast_SSNE(parameters) #nitialize SSNE engine

        #####Initiate the agent
        self.agent = Agent_Pop(parameters, 0)
        if is_probe: self.run_probe() #Run probe and end the program

        #ROS stuff
        self.oracle_x = [[-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1], [1, -1, -1], [1, -1, 1], [1, 1, -1],
                         [1, 1, 1]]
        self.oracle_y = [[3.348, 0.955], [0.550, 1.051], [0.546, 8.904], [3.299, 8.929], [6.583, 9.0120],
                         [9.087, 8.966], [9.316, 0.897], [6.569, 0.926]]
        self.oracle_goals = self.format_goal(self.oracle_y)
        self.home_coordinates = [[5.0, 4.0]]
        self.home_goal = self.format_goal(self.home_coordinates)

        rospy.init_node('Evolve_GRUMB', anonymous=True)  # Node
        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.update_scan)
        # self.cmd_sub = rospy.Subscriber('/navigation_velocity_smoother/raw_cmd_vel', Twist, self.update_command)
        self.goal_reached_sub = rospy.Subscriber('/move_base/status', GoalStatusArray, self.update_goal_status)
        self.odom_sub = rospy.Subscriber('/base_pose_ground_truth', Odometry, self.update_odom)

        self.goal_pub = rospy.Publisher('/move_base/goal', MoveBaseActionGoal, queue_size=1)
        self.goal_pub.publish(MoveBaseActionGoal()); time.sleep(0.3)  # Dummy hack
        self.pub = rospy.Publisher('/navigation_velocity_smoother/raw_cmd_vel', Twist,
                                   queue_size=10)  # Publisher to command velocity
        # Control Action defined and initialized
        self.action = Twist()
        self.action.linear.x = 0.0
        self.action.angular.z = 0.0
        self.action.linear.y = 0.0

        # Define max applicable to robot
        self.max_speed = 1.0

        self.is_pursuing = False
        self.scan_readings = None
        self.base_odom = Pose()

    def run_probe(self):
        team_ind = [0]*self.parameters.num_agents #Grab all the loaded champions

        set_train_x, set_train_y = self.get_training_maze(self.parameters.num_train_evals)
        fitnesses = [0] * self.parameters.num_agents
        for train_x, train_y in zip(set_train_x, set_train_y):  # For all maps in the training set
            fitnesses = map(add, fitnesses, self.compute_fitness(team_ind, train_x, train_y) / self.parameters.num_train_evals)  # Returns rewards for each member of the team for each individual training map
        #TODO FINISH PROBE

        sys.exit('PROBE COMPLETED')

    def save(self, individual, filename ):
        mod.pickle_object(individual, filename)
        #torch.save(individual, filename)
        #return individual.saver.save(individual.sess, self.save_foldername + filename)

    def load(self, filename):
        return mod.unpickle(filename)
        #return torch.load(filename)

    def predict(self, individual, input): #Runs the individual net and computes and output by feedforwarding
        return individual.predict(input)

    def run_bprop(self, model):
        if self.parameters.skip_bprop: return
        criterion = torch.nn.L1Loss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        train_x, train_y = self.test_sequence_data(self.parameters.bprop_train_examples, self.parameters.seq_len_train)

        for epoch in range(1, self.parameters.bprop_max_gens):
            epoch_loss = 0.0
            for example in range(len(train_x)):  # For all examples
                out = model.forward(train_x[example])

                # Compare with target and compute loss
                y = Variable(torch.Tensor(train_y[example])); y = y.unsqueeze(0)
                loss = criterion(out, y)
                optimizer.zero_grad()
                loss.backward(retain_variables=True)
                optimizer.step()
                epoch_loss += loss.data.numpy()[0]
            print 'Epoch: ', epoch, ' Loss: ', epoch_loss / len(train_x)



        self.save(model, self.save_foldername + 'bprop_simulator') #Save individual

    def compute_fitness(self, net):
        net.reset(); fitness = 0.0
        for goal_y, x, y in zip(self.oracle_goals, self.oracle_x, self.oracle_y): #All endpoints
            # Go HOME
            if self.is_pursuing == False:
                self.goal_pub.publish(self.home_goal[0])
                time.sleep(0.3)
                while self.is_pursuing: None

            # Let GRUMB controller go to goal
            for step in range(self.parameters.timesteps):
                net_inp = x + self.scan_readings
                net_out = net.forward(net_inp)

                print net_out
                self.action.linear.x = net_out[0][0]
                self.action.angular.z = net_out[1][0]
                self.action.linear.y = net_out[2][0]
                self.pub.publish(self.action)  # Send control
                time.sleep(0.1)
                print 'Action'

            #Compute closeness with goal
            #print self.base_odom
            fitness -= abs(self.base_odom.pose.position.x - y[0]) + abs(self.base_odom.pose.position.y - y[1])
            print fitness
        return fitness





    def update_command(self, cmd):
        if self.is_record:
            self.train_y = np.concatenate((self.train_y, np.array([[cmd.linear.x, cmd.linear.y, cmd.angular.z]])))
            self.is_sync = True

    def update_odom(self, odom):
        self.base_odom = odom.pose
        #print 'Update'

    def update_scan(self, scan):
        scan_readings = scan.ranges
        scan_readings = np.array(scan_readings)
        scan_readings = scan_readings[::20]
        self.scan_readings = list(scan_readings)



    def update_goal_status(self, status):
        l = status.status_list
        # print len(l)
        if len(l) > 0:
            if l[-1].status == 1:
                self.is_pursuing = True
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
            # print y
            goals.append(y)
        return goals





    def evolve(self, gen):
        tr_best_gen_fitness = 0.0

        #Reset Agent
        self.agent.reset()

        # MAIN LOOP
        for index, net in enumerate(self.agent.pop):  # For evaluation
            # SIMULATION AND TRACK REWARD
            fitness = self.compute_fitness(net)
            self.agent.fitness_evals[index] = fitness


        #Save population and HOF
        if gen % 100 == 0:

            ig_folder = self.parameters.save_foldername + '/Pop/'
            if not os.path.exists(ig_folder): os.makedirs(ig_folder)

            for individial_ind, individual in enumerate(self.agent.pop): #Save population
                self.save(individual, ig_folder + str(individial_ind))
            np.savetxt(self.parameters.save_foldername + '/gen_tag', np.array([gen + 1]), fmt='%.3f', delimiter=',')

        #SSNE Epoch: Selection and Mutation/Crossover step
        self.ssne.epoch(self.agent.pop, self.agent.fitness_evals)


        return tr_best_gen_fitness

    def get_training_maze(self, num_examples):
        set_x = []; set_y = []
        for _ in range(num_examples):

            # Distraction/Hallway parts
            train_x = []; train_y = [[],[],[]]
            for junction in range(self.parameters.depth):
                corridor_len = randint(self.parameters.corridor_bound[0], self.parameters.corridor_bound[1])
                for i in range(corridor_len-1, -1, -1):
                    train_x.append(i)
                for div in train_y:
                    div.append(1 if random.random()<0.5 else -1)

            #Make sure the target (goal location) for each division aren't the same
            if train_y[0] == train_y[1]: train_y[0][randint(0, self.parameters.depth -1)] *= -1
            if train_y[1] == train_y[2]: train_y[2][randint(0, self.parameters.depth - 1)] *= -1

            set_x.append(train_x); set_y.append(train_y)

        return set_x, set_y


if __name__ == "__main__":
    parameters = Parameters()  # Create the Parameters class
    tracker = Tracker(parameters)  # Initiate tracker
    print 'Multi-Agent TMaze Training ', parameters.arch_type

    # ROS stuff
    task = Task_ROS_TMaze(parameters)



    #EVOLVE
    for gen in range(1, parameters.total_gens):
        gen_best_fitnesses = task.evolve(gen)
        print 'Gen:', gen, ' Epoch_best:', '%.2f' % gen_best_fitnesses
        tracker.add_fitness(gen_best_fitnesses, gen)  # Add average global performance to tracker
        #tracker.add_hof_fitness(champ_real_fitness, gen)  # Add best global performance to tracker










