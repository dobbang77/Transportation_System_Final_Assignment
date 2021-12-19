from os import write
import numpy as np
import matplotlib.pyplot as plt
import math,copy, random

numNode = -1 # 100
numVehicle = -1 # 6 
capVehicle = -1 # 50000 차량의 최대 적재량
coord = [] #[i][0]=coord_X, [i][1]=coord_Y 노드의 좌표값
cost = [] #[i][j] i에서 j로 갈 때의 cost 100by100
demand = [] #[i]
print("########## read in .vrp file start ##########")
file = open("SLCL100-DV1M10-n100-k6.vrp")
lines = file.readlines()
for i in range(0,len(lines)):
    strings = lines[i].split()
    if (strings[0] == "DIMENSION:"):
        print(strings)
        numNode = int(strings[1])
    elif (strings[0] == "VEHICLE:"):
        print(strings)
        numVehicle = int(strings[1])
    elif (strings[0] == "CAPACITY:"):
        print(strings)
        capVehicle = int(strings[1])
    elif (strings[0] == "NODE_COORD_SECTION"):
        print(strings[0])
        i=i+1
        strings = lines[i].split()    
        while(strings[0] != "EDGE_WEIGHT_SECTION"):
            temp = [float(strings[1]), float(strings[2])]
            coord.append(temp)
            i=i+1
            strings = lines[i].split()
    elif (strings[0] == "EDGE_WEIGHT_SECTION"):
        print(strings[0])
        i=i+1
        strings = lines[i].split()
        while(strings[0] != "DEMAND_SECTION"):
            temp = []
            for j in range(0, len(strings)):
                temp.append(int(strings[j]))
            cost.append(temp)
            i=i+1
            strings = lines[i].split()
    elif (strings[0] == "DEMAND_SECTION"):
        print(strings[0])
        i=i+1
        strings = lines[i].split()
        while(strings[0] != "DEPOT_SECTION"):
            demand.append(int(strings[1]))
            i=i+1
            strings = lines[i].split()
    elif (strings[0] == "DEPOT_SECTION"):
        print(strings[0])
        i=i+1
        while(strings[0] != "EOF"):
            i=i+1
            strings = lines[i].split()
    elif (strings[0] == "EOF"):
        print(strings[0])
file.close()
print("########## read in .vrp file end ##########")

vehicle = [] #[[1,2,3,...],[6,7,9,...],[],[],[],[]] 각 차량이 방문하는 노드 & 경로
vehicleVolume = [] # [50000,40000,25000,60000,50000,55000] 각 차량별 화물의 적재량
vehicleCost = [] # [,,,,,,] 각 차량별 총비용
ofv = 0
print("########## initial solution for vrp start ##########")
for i in range(0, numVehicle):
    vehicle.append([])
    vehicleVolume.append(0)
    vehicleCost.append(0)

# Sweep Algorithm를 위한 각도 노드별 각도 계산
angles = [] # [노드 번호, 기준선과의 각도값]
#  시작점 노드 1 의 좌표
pointA_X = coord[5][0] 
pointA_Y = coord[5][1]
# 기준점 Depot 노드 0 의 좌표
pointB_X = coord[0][0]
pointB_Y = coord[0][1]
# 각도를 계산하기 위한 나머지 노드들의 좌표
for i in range(1, numNode):
    pointC_X = coord[i][0] 
    pointC_Y = coord[i][1] 
    atanA = math.atan2(pointA_Y-pointB_Y, pointA_X-pointB_X)
    atanC = math.atan2(pointC_Y-pointB_Y, pointC_X-pointB_X)
    diff = atanA - atanC 
    diff *= 180 / math.pi # 각도값으로 표현 
    if(diff < 0): 
        diff += 360.0 
    temp = []
    temp.append(i)
    temp.append(diff)
    angles.append(temp) # 앵글은 [노드 번호, 기준선과의 각도값]의 값을 가지게 된다.

# 각도의 오름차순으로 sorting하기 위한 이중 for문
for i in range(0, len(angles)-1):
    for j in range(i+1, len(angles)):
        angle_i = angles[i][1] # i 노드의 각도값
        angle_j = angles[j][1]
        if (angle_i > angle_j):
            temp = angles[i]
            angles[i] = angles[j]
            angles[j] = temp

# Sweep Algorithm
currVechicleIdx = 0 # 현재 차량
currVechicleVol = 0 # 현재 차량의 적재량
for i in range(0, len(angles)):
    if (currVechicleVol + demand[angles[i][0]] < capVehicle): 
        # 차량의 i 노드의 demand를 추가할 적재공간이 남아있다면 현재차량의 노드 추가
        vehicle[currVechicleIdx].append(angles[i][0])
        currVechicleVol = currVechicleVol + demand[angles[i][0]]
    else: # 아니라면 다음차랑으로 넘어감
        currVechicleIdx = currVechicleIdx + 1
        currVechicleVol = 0
        i = i - 1

# 차량에게 할당된 노드 # print(vehicle)

# Nearest Neighbor Algorithm
for i in range(0, numVehicle):
    # vehicle[i]의 element를 temp로 NN규칙을 통해 옮겨줌
    if (len(vehicle[i]) != 0): # 차량에 하나의 노드라도 배정되어 있는 경우에만 알고리즘 수향
        tempVehicle = []
        
        minValue = 10000000
        minCostNode = 0 # cost가 가장 가까운 노드
        for j in range(0,len(vehicle[i])): 
            if (cost[0][vehicle[i][j]] < minValue): # Depot으로 부터 cost가 가장 적은 노드 찾기 
                minValue = cost[0][vehicle[i][j]]
                minCostNode = vehicle[i][j]
        tempVehicle.append(minCostNode) 
        

        for k in range(0, len(vehicle[i])-1):
            minValue = 10000000
            minCostNode = 0
            for j in range(0,len(vehicle[i])): 
                if (vehicle[i][j] not in tempVehicle): # 방문했던 노드와 자기 자신은 방문하지 않음
                    if (cost[tempVehicle[len(tempVehicle)-1]][vehicle[i][j]] < minValue): 
                        # tempvehicle의 마지막 노드로 부터 계속 cost가 가장 적은 노드 찾기 
                        minValue = cost[tempVehicle[len(tempVehicle)-1]][vehicle[i][j]]
                        minCostNode = vehicle[i][j]
            tempVehicle.append(minCostNode) 
            
        vehicle[i] = tempVehicle        
"""
랜덤으로 차량에 노드를 추가하는 방법
for i in range(1, numNode):
    randomVehIdx = np.random.randint(0,numVehicle)
    vehicle[randomVehIdx].append(i)
"""

# evaluate iniital solution
ofv = 0
penaltyM = 10000000
for i in range(0, numVehicle):
    # calculate volume for each vehicle
    for j in range(0, len(vehicle[i])):
        vehicleVolume[i] += demand[vehicle[i][j]]

    if (vehicleVolume[i] > capVehicle): # 차량의 최대 적재량을 넘는 경우 ofv값에 패널티를 부여해서 넘지 않도록 설정함.   
        ofv = ofv + penaltyM
    # calculate cost for each vehicle
    # 0,[1,2,3],0
    # [0,1,2,3,0]
    
    if (len(vehicle[i]) == 0):
        vehicleCost[i] = 0
    else: # 비어있는 차량이 아닐때에만 cost계산 
        for j in range(0, len(vehicle[i])+1):
            if (j==0):
                vehicleCost[i] += cost[0][vehicle[i][j]]
            elif (j==len(vehicle[i])):
                vehicleCost[i] += cost[vehicle[i][j-1]][0]
            else:
                vehicleCost[i] += cost[vehicle[i][j-1]][vehicle[i][j]]
    # calculate ofv
    ofv += vehicleCost[i]

# print out    
for i in range(0, numVehicle):
    print("vehicle_"+str(i),vehicle[i])
print("vehicleVolume: ", vehicleVolume)
print("vehicleCost: ", vehicleCost)
print("ofv: ", ofv)
print("########## initial solution for vrp end ##########")

print("########## improve solution for vrp start ##########")
# Simulated Annealing을 사용한 ofv 개선 deepcopy를 통해 list 변경값 보존
curr_vehicle = copy.deepcopy(vehicle)
curr_vehicleVolume = copy.deepcopy(vehicleVolume)
curr_vehicleCost = copy.deepcopy(vehicleCost)
curr_ofv = copy.deepcopy(ofv)

prev_vehicle = copy.deepcopy(vehicle)
prev_vehicleVolume = copy.deepcopy(vehicleVolume)
prev_vehicleCost = copy.deepcopy(vehicleCost)
prev_ofv = copy.deepcopy(ofv)

best_vehicle = copy.deepcopy(vehicle)
best_vehicleVolume = copy.deepcopy(vehicleVolume)
best_vehicleCost = copy.deepcopy(vehicleCost)
best_ofv = copy.deepcopy(ofv)


temp_cost = copy.deepcopy(cost)
for i in range(0, len(cost)):
    sample = random.sample(temp_cost,1) # cost matrix의 10%를 랜덤 추출하여 평균을 구함
    temp = []
    temp.append(sample)



worse_prob = 0.5 
# 평균값 만큼 안좋아 질 때의 worse_prob의 확률로 accept
sample_cost = []
temp_cost = copy.deepcopy(cost)
for i in range(0, len(cost)//10):
    # cost matrix의 10%를 랜덤 추출하여 평균을 구함
    sample = random.sample(temp_cost,1) 
    if (sample not in temp or sample != 0): 
        # cost가 0이거나 이미 포함되어 있는 경우는 제외함
        sample_cost.append(sample)
average = np.mean(sample_cost)
initial_temperature = (-average) * (np.log(worse_prob))
curr_temperatrue = initial_temperature
stop_temperatrue = 1
curr_iteration = 0
stop_iteration = 1000
cooling = 0.9



while (curr_temperatrue > stop_temperatrue):
    print("current temperature : ", curr_temperatrue)
    while (stop_iteration > curr_iteration): # insert/swap 총 수행 횟수
        curr_iteration = curr_iteration + 1
        if(random.random() < 0.5): # 0.5 확률로는 swap 수행
            #swap을 위한 두 idx구하기
            idx1 = [] # node1 [vehIdx, nodeIdx] 차량번호, 노드 번호 
            temp = random.randint(0, numVehicle-1)
            while(len(curr_vehicle[temp]) == 0):
                temp = random.randint(0, numVehicle-1)
            idx1.append(temp)
            idx1.append(random.randint(0, len(curr_vehicle[idx1[0]])-1))
 
            idx2 = [] # node2 [vehIdx, nodeIdx] 
            temp = random.randint(0, numVehicle-1)
            while(len(curr_vehicle[temp]) == 0):
                temp = random.randint(0, numVehicle-1)
            idx2.append(temp)
            idx2.append(random.randint(0, len(curr_vehicle[idx2[0]])-1))

            if(idx1[0]==idx2[0] and idx1[1]==idx2[1]): # 차량이나 노드가 같은 경우는 swap을 실행하지 않음
                continue 
            
            #swap 실행
            node1 = curr_vehicle[idx1[0]][idx1[1]]
            node2 = curr_vehicle[idx2[0]][idx2[1]]
            curr_vehicle[idx1[0]][idx1[1]] = node2
            curr_vehicle[idx2[0]][idx2[1]] = node1

        else:
            #insert를 위한 두 idx 구하기
            idx1 = [] # node1 [vehIdx, nodeIdx] 차량번호, 노드 번호
            temp = random.randint(0, numVehicle-1)
            while(len(curr_vehicle[temp]) == 0):
                temp = random.randint(0, numVehicle-1)
            idx1.append(temp)
            idx1.append(random.randint(0, len(curr_vehicle[idx1[0]])-1))
    
            idx2 = [] # node1 insertion idx [vehIdx, nodeIdx]
            temp = random.randint(0, numVehicle-1)
            while(len(curr_vehicle[temp]) == 0 or temp == idx1[0]): 
                temp = random.randint(0, numVehicle-1)
            idx2.append(temp)
            idx2.append(random.randint(0, len(curr_vehicle[idx2[0]])))

            if(idx1[0]==idx2[0] and idx1[1]==idx2[1]):
                continue
            
            # insert 실행
            node1 = curr_vehicle[idx1[0]][idx1[1]]
            curr_vehicle[idx1[0]].remove(node1)
            curr_vehicle[idx2[0]].insert(idx2[1], node1)
        
    
        # 새로 구해진 ofv를 계산
        curr_ofv = 0
        for i in range(0, numVehicle):
            curr_vehicleVolume[i] = 0 # 초기화
            for j in range(0, len(curr_vehicle[i])):
                curr_vehicleVolume[i] += demand[curr_vehicle[i][j]]

            if (curr_vehicleVolume[i] > capVehicle):
                curr_ofv = curr_ofv + penaltyM
                
            curr_vehicleCost[i] = 0 # 초기화
            if (len(curr_vehicle[i]) == 0):
                curr_vehicleCost[i] = 0
            else: 
                for j in range(0, len(curr_vehicle[i])+1):
                    if (j==0):
                        curr_vehicleCost[i] += cost[0][curr_vehicle[i][j]]
                    elif (j==len(curr_vehicle[i])):
                        curr_vehicleCost[i] += cost[curr_vehicle[i][j-1]][0]
                    else:
                        curr_vehicleCost[i] += cost[curr_vehicle[i][j-1]][curr_vehicle[i][j]]

            curr_ofv += curr_vehicleCost[i]
        # print(curr_ofv)

        # 계산된 ofv값을 비교하여 더나은 ofv를 구하기 위한 과정
        diff_prev = prev_ofv - curr_ofv # 이전값과 현재값의 차이
        diff_best = best_ofv - curr_ofv # 최적값과 현재값의 차이
        if (diff_best > 0 ): 
            # 새로운 최적값이 나타났을 때
            print("new best solution! : ", curr_ofv)
            best_vehicle = copy.deepcopy(curr_vehicle)
            best_vehicleVolume = copy.deepcopy(curr_vehicleVolume)
            best_vehicleCost = copy.deepcopy(curr_vehicleCost)
            best_ofv = copy.deepcopy(curr_ofv)

            prev_vehicle = copy.deepcopy(curr_vehicle)
            prev_vehicleVolume = copy.deepcopy(curr_vehicleVolume)
            prev_vehicleCost = copy.deepcopy(curr_vehicleCost)
            prev_ofv = copy.deepcopy(curr_ofv)
        
        elif (diff_prev > 0):
            # 현재값이 이전값보다 개선되었을 때
            prev_vehicle = copy.deepcopy(curr_vehicle)
            prev_vehicleVolume = copy.deepcopy(curr_vehicleVolume)
            prev_vehicleCost = copy.deepcopy(curr_vehicleCost)
            prev_ofv = copy.deepcopy(curr_ofv)

        else:
            # 현재값이 이전값보다 악화되어도 확률에 따라 나빠진 해에 대해서도 가끔은 탐색을 수행
            prob = np.exp(diff_prev / curr_temperatrue) # 나빠진 정도가 낮을 때, 현재 온도가 높을때 해를 수용할 확률이 높아짐.
            if (random.random() < prob):
                # 지수분포로 구한 확률에 의해 현재값을 수용함
                prev_vehicle = copy.deepcopy(curr_vehicle)
                prev_vehicleVolume = copy.deepcopy(curr_vehicleVolume)
                prev_vehicleCost = copy.deepcopy(curr_vehicleCost)
                prev_ofv = copy.deepcopy(curr_ofv)

            else:
                # 확률에 포함되지 않는다면 값을 버림.
                curr_vehicle = copy.deepcopy(prev_vehicle)
                curr_vehicleVolume = copy.deepcopy(prev_vehicleVolume)
                curr_vehicleCost = copy.deepcopy(prev_vehicleCost)
                curr_ofv = copy.deepcopy(prev_ofv)


    curr_iteration = 0
    curr_temperatrue = curr_temperatrue * cooling # cooling facor를 통해 온도를 낮춘다.
            
    # best_ofv가 찾아지면 개선된 값을 대입해 다시 수행
    curr_vehicle = copy.deepcopy(best_vehicle)
    curr_vehicleVolume = copy.deepcopy(best_vehicleVolume)
    curr_vehicleCost = copy.deepcopy(best_vehicleCost)
    curr_ofv = copy.deepcopy(best_ofv)

    prev_vehicle = copy.deepcopy(best_vehicle)
    prev_vehicleVolume = copy.deepcopy(best_vehicleVolume)
    prev_vehicleCost = copy.deepcopy(best_vehicleCost)
    prev_ofv = copy.deepcopy(best_ofv)


print("########## improve solution for vrp end ##########")


print("########## plot solution start ##########")
colors = ['C0','C1','C2','C3','C4','C5','C6','C7','C8','C9','b','k','m','g','r']
# reserved
for i in range(0, numVehicle):
    color = colors[i%13]
    best_vehicle[i].insert(0,0)
    best_vehicle[i].append(0)
    for j in range(0, len(best_vehicle[i])-1):
        plt.plot(coord[best_vehicle[i][j]][0], coord[best_vehicle[i][j]][1],
                 (color+'o'), markersize=2)
        plt.text(coord[best_vehicle[i][j]][0], coord[best_vehicle[i][j]][1],
                 (int)(best_vehicle[i][j]), fontsize=8)
        plt.plot([coord[best_vehicle[i][j]][0], coord[best_vehicle[i][j+1]][0]],
                 [coord[best_vehicle[i][j]][1], coord[best_vehicle[i][j+1]][1]],
                 color, linewidth=0.5)
plt.show()
print("########## plot solution end ##########")




