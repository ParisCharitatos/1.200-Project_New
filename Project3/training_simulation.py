import traci
import numpy as np
import random
import timeit
import torch
import torch.optim as optim
from torch.autograd import Variable

# phase codes based on environment.net.xml
PHASE_NS_GREEN = 0  # action 0 code 00
PHASE_NS_YELLOW = 1
PHASE_NSL_GREEN = 2  # action 1 code 01
PHASE_NSL_YELLOW = 3
PHASE_EW_GREEN = 4  # action 2 code 10
PHASE_EW_YELLOW = 5
PHASE_EWL_GREEN = 6  # action 3 code 11
PHASE_EWL_YELLOW = 7


class Simulation:
    def __init__(self, Model, ReplayBuffer, TrafficGen, sumo_cmd, gamma, max_steps, green_duration,
                 yellow_duration, num_states, num_actions, training_epochs, batch_size, learning_rate):
        self.sum_waiting_time = 0
        self.sum_queue_length = 0
        self.waiting_times = {}
        self.dqn = Model
        self.traffic_gen = TrafficGen
        self.gamma = gamma
        self.step = 0
        self.sumo_cmd = sumo_cmd
        self.max_steps = max_steps
        self.green_duration = green_duration
        self.yellow_duration = yellow_duration
        self.num_states = num_states
        self.num_actions = num_actions
        self.training_epochs = training_epochs
        self.replay_buffer = ReplayBuffer
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=learning_rate)
        self.batch_size = batch_size

    def run(self, episode, epsilon):
        """
        Runs an episode of simulation, then starts a training session
        """
        start_time = timeit.default_timer()

        # generate the route file for this simulation and set up sumo
        self.traffic_gen.generate_routefile(seed=episode)
        traci.start(self.sumo_cmd)

        print("\t [INFO] Start simulating the episode")

        # inits
        self.step = 0
        #old_total_wait = 0
        old_state = -1
        old_action = -1

        sum_reward = 0
        #sum_waiting = 0
        while self.step < self.max_steps:

            # get current state of the intersection
            current_state = self.get_state()

            # calculate reward of previous action: (change in cumulative waiting time between actions)
            # waiting time = seconds waited by a car since the spawned in the environment,
            #current_total_wait = self.collect_waiting_times()
            #reward = old_total_wait - current_total_wait

            # calculate reward depending on number of collisions and speed

            collisions_RL, speed_RL, lane_RL = self.collect_collisions_speeds
            
            RL = False

            for j in collisions_RL:

                if j == "RL":

                    RL = True
                    reward = -100
                
                if RL == False and speed_RL<=1:

                 reward = 1
                
                elif RL == False and speed_RL>=1:

                    reward = 2
            
                if lane_RL =="TL2_W3":

                    reward = 100
            

            # saving the data into the memory
            if self.step != 0:
                self.replay_buffer.push(old_state, old_action, reward, current_state)

            # choose the light phase to activate, based on the current state of the intersection
            action = self.choose_action(current_state, epsilon)

            # if the chosen phase is different from the last phase, activate the yellow phase
           # if self.step != 0 and old_action != action:
            #    self.set_yellow_phase(old_action)
             #   self.simulate(self.yellow_duration)

            # execute the phase selected before
           # self.set_green_phase(action)
            #self.simulate(self.green_duration)

            # saving variables for later & accumulate reward
            old_state = current_state
            old_action = action
            #old_total_wait = current_total_wait

            # saving only the meaningful reward to better see if the agent is behaving correctly
           # if reward < 0:
            #    sum_reward += reward
            #sum_waiting += current_total_wait

            if reward < 0:
                sum_reward += reward

        avg_reward = sum_reward / self.max_steps
        #avg_waiting = sum_waiting / self.max_steps
        traci.close()
        simulation_time = round(timeit.default_timer() - start_time, 1)

        print("\t [STAT] Average reward:", avg_reward,
              #"Average waiting time:", avg_waiting,
              "- Epsilon:", round(epsilon, 2))

        print("\t [INFO] Training the DQN")
        start_time = timeit.default_timer()
        # training the DQN

        sum_training_loss = 0
        for _ in range(self.training_epochs):
            sum_training_loss += self.compute_td_loss()
        avg_training_loss = sum_training_loss.item() / self.max_steps
        print("\t [STAT] Training Loss :", avg_training_loss)
        training_time = round(timeit.default_timer() - start_time, 1)

        return simulation_time, training_time, avg_reward, avg_training_loss

    def simulate(self, steps_todo):
        """
        Execute steps in sumo while gathering statistics
        """
        # do not do more steps than the maximum allowed number of steps
        if (self.step + steps_todo) >= self.max_steps:
            steps_todo = self.max_steps - self.step

        while steps_todo > 0:
            traci.simulationStep()  # simulate 1 step in sumo
            self.step += 1  # update the step counter
            steps_todo -= 1
            queue_length = self.get_queue_length()
            self.sum_queue_length += queue_length
            # 1 step while waiting in queue means 1 second waited
            # for each car, therefore queue_length == waited_seconds
            self.sum_waiting_time += queue_length

    def collect_collisions_speeds(self):

        lane_RL = traci.vehicle.getLaneID("RL")
        collisions_RL = traci.simulation.getCollidingVehiclesIDList()
        speed_RL = traci.vehicle.getSpeed("RL")

        return lane_RL, collisions_RL, speed_RL

    def collect_waiting_times(self):
        """
        Retrieve the waiting time of every car in the incoming roads and return the total waiting time
        """
        incoming_roads = ["E2TL", "N2TL", "W2TL", "S2TL"]
        car_list = traci.vehicle.getIDList()
        for car_id in car_list:
            wait_time = traci.vehicle.getAccumulatedWaitingTime(car_id)
            # get the road id where the car is located
            road_id = traci.vehicle.getRoadID(car_id)
            # consider only the waiting times of cars in incoming roads
            if road_id in incoming_roads:
                self.waiting_times[car_id] = wait_time
            else:
                # a car that was tracked has cleared the intersection
                if car_id in self.waiting_times:
                    del self.waiting_times[car_id]
        total_waiting_time = sum(self.waiting_times.values())
        return total_waiting_time

    def choose_action(self, state, epsilon):
        """
        According to epsilon-greedy policy, decide whether to perform exploration or exploitation
        """
        if random.random() < epsilon:
            # random action
            return random.randint(0, self.num_actions - 1)
        else:
            # the best action given the current state
            state = Variable(torch.FloatTensor(state).unsqueeze(0), requires_grad=False)
            q_value = self.dqn.forward(state)
            return q_value.max(1)[1].data[0]
    

    def set_yellow_phase(self, old_action):
        """
        Activate the correct yellow light combination in sumo
        """
        # obtain the yellow phase code, based on the old action (ref on environment.net.xml)
       # yellow_phase_code = old_action * 2 + 1
        #traci.trafficlight.setPhase("TL", yellow_phase_code)

    def set_green_phase(self, action_number):
        """
        Activate the correct green light combination in sumo
        """
       # if action_number == 0:
            #traci.trafficlight.setPhase("TL", PHASE_NS_GREEN)
       # elif action_number == 1:
           # traci.trafficlight.setPhase("TL", PHASE_NSL_GREEN)
        #elif action_number == 2:
            #traci.trafficlight.setPhase("TL", PHASE_EW_GREEN)
        #elif action_number == 3:
       #     traci.trafficlight.setPhase("TL", PHASE_EWL_GREEN)
    
    def set_acceleration(self, action_number):
        """
        Activate the correct green light combination in sumo
        """
        if action_number == 0:
            traci.vehicle.setDecel("RL", -5)
        elif action_number == 1:
            traci.vehicle.setAccel("RL", 0)
        elif action_number == 2:
            traci.vehicle.setAccel("RL", 5)
       

    def get_queue_length(self):
        """
        Retrieve the number of cars with speed = 0 in every incoming lane
        """
        halt_N = traci.edge.getLastStepHaltingNumber("N2TL")
        halt_S = traci.edge.getLastStepHaltingNumber("S2TL")
        halt_E = traci.edge.getLastStepHaltingNumber("E2TL")
        halt_W = traci.edge.getLastStepHaltingNumber("W2TL")
        queue_length = halt_N + halt_S + halt_E + halt_W
        return queue_length

    def get_state(self):
        """
        Retrieve the state of the intersection from sumo, in the form of cell occupancy
        """
        state = np.zeros(self.num_states)
        
        pos_RL = traci.vehicle.getPosition("RL")
        speed_RL = traci.vehicle.getSpeed("RL")
        pos_HDV = traci.vehicle.getPosition("HDV")
        speed_HDV = traci.vehicle.getSpeed("HDV")
        accel_HDV = traci.vehicle.getAcceleration("HDV")

        state[0,1] = pos_RL
        state[1,1] = speed_RL
        state[2,1] = pos_HDV
        state[3,1] = speed_HDV
        state[4,1] = accel_HDV
        
        
        return state     

    def compute_td_loss(self):

        """
        Compute the temporal difference loss as defined by the Q update
        """

        # sample a batch from the replay buffer
        state, action, reward, next_state = self.replay_buffer.sample(self.batch_size)

        # convert to pytorch variables
        state = Variable(torch.FloatTensor(np.float32(state)), requires_grad=False)
        next_state = Variable(torch.FloatTensor(np.float32(next_state)), requires_grad=False)
        action = Variable(torch.LongTensor(action), requires_grad=False)
        reward = Variable(torch.FloatTensor(reward), requires_grad=False)

        # obtain the q value for the current state by feeding the state to the DQN
        q_values = self.dqn.forward(state)
        # obtain the q value for the next state by feeding the next state to the DQN
        next_q_values = self.dqn.forward(next_state)

        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_values.max(1)[0]
        gamma = 0.75

        # TODO : Calculate the loss
        # set the loss variable to the right expression using the computed quantities above
        loss_function = torch.nn.MSELoss()
        loss = loss_function(reward + gamma*next_q_value, q_value)

        # next we do the gradient update based on the calculated loss
        self.optimizer.zero_grad()
        loss.backward() # PyTorch will automatically do the gradient update on the defined loss 
        self.optimizer.step()

        return loss

    def save_model(self, path, episode):
        """
        Save the DQN model
        """
        torch.save(self.dqn, path + "/" + str(episode))
