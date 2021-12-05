import numpy as np
import math


class TrafficGenerator:
    def __init__(self, max_steps, n_cars_generated):
        # how many cars per episode
        self._n_cars_generated = n_cars_generated
        self._max_steps = max_steps

    def generate_routefile(self, seed):
        """
        Generation of the route of every car for one episode
        """
        np.random.seed(seed)  # make tests reproducible

        # the generation of cars is distributed according to a weibull distribution
        timings = np.random.weibull(2, self._n_cars_generated)
        timings = np.sort(timings)

        # reshape the distribution to fit the interval 0:max_steps
        car_gen_steps = []
        min_old = math.floor(timings[1])
        max_old = math.ceil(timings[-1])
        min_new = 0
        max_new = self._max_steps
        for value in timings:
            car_gen_steps = np.append(car_gen_steps,
                                      ((max_new - min_new) / (max_old - min_old)) * (value - max_old) + max_new)

        # round every value to int -> effective steps when a car will be generated
        car_gen_steps = np.rint(car_gen_steps)

        # produce the file for cars generation, one car per line
        with open("intersection/episode_routes.rou.xml", "w") as routes:
            print("""<routes>
            <vType accel="1.0" decel="4.5" id="standard_car" length="5.0" minGap="2.5" maxSpeed="25" sigma="0.5" jmIgnoreFoeProb="1" jmIgnoreJunctionFoeProb="1" />

            <route id="N_W" edges="N2TL TL2W"/>
            <route id="N_S" edges="N2TL TL2S"/>
            <route id="S_W" edges="S2TL TL2W"/>
            <route id="S_W_RL" color="1,0,0" edges="S2TL TL2W"/>
            <route id="N_S_HDV" color="1,0,1" edges="N2TL TL2S"/>""", file=routes)

            print('    <vehicle id="HDV" type="standard_car" route="N_S_HDV" depart="1.0" '
                              'departLane="2" departSpeed="10" />', file=routes)
            
            print('    <vehicle id="RL" type="standard_car" route="S_W_RL" depart="1.0" '
                              'departLane="3" departSpeed="14" />', file=routes)
            
          


            for car_counter, step in enumerate(car_gen_steps):
                straight_or_turn = np.random.uniform()

                # choose direction: straight or turn - 75% of times the car goes straight
                if straight_or_turn < 0.75:
                    # choose a random source & destination
                    route_straight = np.random.randint(1, 3)
                    if route_straight == 2:
                        print('    <vehicle id="Ν_S_%i" type="standard_car" route="N_S" depart="%s" '
                              'departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    
                else:  # 25% of the time, the car turns
                    route_turn = np.random.randint(1, 3)  # choose random source source & destination
                    if route_turn == 1:
                        print('    <vehicle id="S_W_%i" type="standard_car" route="S_W" depart="%s" '
                              'departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 2:
                        print('    <vehicle id="N_W_%i" type="standard_car" route="N_W" depart="%s" '
                              'departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                  

            print("</routes>", file=routes)
