# IMPORTING MODULES
import os
import sys
import time
import traci
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from algorithm import ga


# SETTING SUMO ENVIRONMENT
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

sumoBinary = "C:\\Program Files (x86)\\Eclipse\\Sumo\\bin\\sumo-gui"
sumoCmd = [sumoBinary, "-c", "DHANMONDI_DATASETS/Dhanmondi.sumocfg"]
# sumoCmd = [sumoBinary, "-c", "MOTIJHEEL_DATASETS/Motijheel.sumocfg"]


# CONTROL STATE
def main(value1, value2, value3):
    """ FUNCTION TAKES ARGUEMENTS FROM THE CONTROL LOOP AND RUNS THE 
    CONTROL STATE IN THE TRAFFIC SIMULATOR SUMO """

    traci.start(sumoCmd)
    TLIDS = traci.trafficlight.getIDList()
    step = 0
    while step < 100:
        traci.simulationStep()
        for traffic_light_id in TLIDS:
            if value1 < value2 and value1 < value3:
                traci.trafficlight.setProgram(traffic_light_id, "program_01")
            elif value2 < value1 and value2 < value3:
                traci.trafficlight.setProgram(traffic_light_id, "program_02")
            else:
                traci.trafficlight.setProgram(traffic_light_id, "program_03")
        step += 1
    traci.close()


# CONTROL LOOP
for _ in range(0, 4):

    # RUNNING THE ALGORITHM
    output_result_01 = ga.run_ga(ga.problem, ga.params)
    output_result_02 = ga.run_ga(ga.problem, ga.params)
    output_result_03 = ga.run_ga(ga.problem, ga.params)

    plt.plot(output_result_01.best_cost)
    plt.xlim(0, ga.params.maxit)
    plt.xlabel("Iterations")
    plt.ylabel("Best Cost")
    plt.title("GENETIC ALGORITHM")
    plt.grid(True)
    plt.show()

    # plt.plot(output_result_02.best_cost)
    # plt.xlim(0, ga.params.maxit)
    # plt.xlabel("Iterations")
    # plt.ylabel("Best Cost")
    # plt.title("GENETIC ALGORITHM")
    # plt.grid(True)
    # plt.show()

    # EXTRACTING MINIMUN VALUE
    traffic_light_01 = min(output_result_01.best_cost)
    traffic_light_02 = min(output_result_02.best_cost)
    traffic_light_03 = min(output_result_03.best_cost)

    main(traffic_light_01, traffic_light_02, traffic_light_03)
