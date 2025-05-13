import json
from fuzzywuzzy import fuzz
import numpy as np
import sys
import os
import json
import re
from datasets import load_dataset
import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, help='Input model name', required=False, default='gpt4o')
    parser.add_argument('--task', type=str, help='Input task name', required=False, default='filteredDataRouteOP')
    parser.add_argument('--numPlan', type=int, help='Input number of plans', required=False, default=1)
    parser.add_argument('--threshold', type=int, help='Input threshold for the macro', required=False, default=0.75)

    args = parser.parse_args()
    return args

def getID(name,address,category):
    #as long as there is a '-', then return -2
    #if there is an empty list, then return []
    #if the information doesn't match, return -1

    if name == "-" and address == "-":
        return -2

    #normal case
    idFromName = []
    idFromAddress = []

    address = address.split(",")[0]
    
    #restaurants
    if category == 'restaurants':
        for restaurant in restaurants:
            if restaurant['name'].lower() == name.lower():
                idFromName.append(restaurant['business_id'])
            if restaurant['address'].lower() == address.lower():
                idFromAddress.append(restaurant['business_id'])
        set1 = set(idFromName)
        set2 = set(idFromAddress)
        #if the extracted id from name and address make an agreement
        if(len(set1 & set2) == 1):
            return list(set1 & set2)[0]
        # if not, we have to use similarity score to determine the id
        else:
            name_sim_score = []
            address_sim_score = []

            for restaurant in restaurants:
                name_sim_score.append(fuzz.ratio(name.lower(), restaurant['name'].lower()))
                address_sim_score.append(fuzz.ratio(address.lower(), restaurant['address'].lower()))

            scores = np.array(name_sim_score) + np.array(address_sim_score)
            #if the score is high enough, then we claim the id
            if max(scores) >= 120:
                return restaurants[int(np.argmax(scores))]['business_id']
            #if the score is less than 60 for each, then we indicate that the business is out of the pool
            else:
                return -1
    #attractions 
    if category == 'attractions':
        for attraction in attractions:
            if attraction['name'].lower() == name.lower():
                idFromName.append(attraction['business_id'])
            if attraction['address'].lower() == address.lower():
                idFromAddress.append(attraction['business_id'])
        
        set1 = set(idFromName)
        set2 = set(idFromAddress)

        if(len(set1 & set2) == 1):
            return list(set1 & set2)[0]
        else:
            name_sim_score = []
            address_sim_score = []

            for attraction in attractions:
                name_sim_score.append(fuzz.ratio(name.lower(), attraction['name'].lower()))
                address_sim_score.append(fuzz.ratio(address.lower(), attraction['address'].lower()))

            if max(name_sim_score) == 100:
                return attractions[int(np.argmax(name_sim_score))]['business_id']

            scores = np.array(name_sim_score) + np.array(address_sim_score)
            if max(scores) >= 120:
                return attractions[int(np.argmax(scores))]['business_id']
            else:
                return -1
    #hotels
    if category == 'hotels':
        for hotel in hotels:
            if hotel['name'].lower() == name.lower():
                idFromName.append(hotel['business_id'])
            if hotel['address'].lower() == address.lower():
                idFromAddress.append(hotel['business_id'])
        set1 = set(idFromName)
        set2 = set(idFromAddress)
        if(len(set1 & set2) == 1):
            return list(set1 & set2)[0]
        else:
            name_sim_score = []
            address_sim_score = []

            for hotel in hotels:
                name_sim_score.append(fuzz.ratio(name.lower(), hotel['name'].lower()))
                address_sim_score.append(fuzz.ratio(address.lower(), hotel['address'].lower()))

            scores = np.array(name_sim_score) + np.array(address_sim_score)
            if max(scores) >= 120:
                return hotels[int(np.argmax(scores))]['business_id']
            else:
                return -1

def prepareEval(plan):
    plan_eval = []
    for days in plan['itinerary']:
        day = {}
        day['days'] = days['days']
        #print(days['breakfast']['name'])
        day['breakfast'] = getID(days['breakfast']['name'],days['breakfast']['address'],'restaurants')
        day['morning_attractions'] = [getID(attraction['name'],attraction['address'],'attractions') for attraction in days['morning_attractions']]
        day['lunch'] = getID(days['lunch']['name'],days['lunch']['address'],'restaurants')
        day['afternoon_attractions'] = [getID(attraction['name'],attraction['address'],'attractions') for attraction in days['afternoon_attractions']]
        day['dinner'] = getID(days['dinner']['name'],days['dinner']['address'],'restaurants')
        day['night_attractions'] = [getID(attraction['name'],attraction['address'],'attractions') for attraction in days['night_attractions']]
        day['accommodation'] = getID(days['accommodation']['name'],days['accommodation']['address'],'hotels')
        plan_eval.append(day)
    #print(plan_eval)
    return plan_eval

def evaluate_outSidePool(plan_eval):
    for day in plan_eval:
        for key,value in day.items():
            if isinstance(value, list):
                for id in value:
                    if id == -1:
                        return 1
            else:
                if value == -1:
                    return 1
    return 0

def evaluate_missingInfo(plan_eval):
    for day in plan_eval:
        for key,value in day.items():
            #night attraction can be skipped
            if key == 'night_attractions':
                continue

            if isinstance(value, list):

                if(len(value) == 0):
                        return 1
                else:
                    for val in value:
                        if val == -2:
                            return 1
            else:
                if value == -2:
                    return 1

    return 0

def evaluate_day(plan_eval,eval):
    day_numerator = 0
    day_denominator = 1

    if(len(plan_eval) == int(eval['day'][0][0])):
        day_numerator = 1
    return day_numerator, day_denominator

def evaluate_price(plan_eval,eval):
    price_numerator = 0
    price_denominator = 0

    price_map = {'cheap budget':['$','$$'],'moderate budget':['$','$$','$$$'],'expensive budget':['$$','$$$','$$$$']}
    price_limit = price_map[eval['price'][0]]

    #price - meals
    all_meals = []
    for day in plan_eval:
        all_meals.append(day['breakfast'])
        all_meals.append(day['lunch'])
        all_meals.append(day['dinner'])
    price_denominator += len(all_meals)

    for restaurant_id in all_meals:
        if restaurant_id != -1 and restaurant_id != -2:
            for restaurant in restaurants:
                if(restaurant['business_id'] == restaurant_id):
                    if(restaurant['price'] in price_limit):
                        price_numerator += 1
                    #else:
                    #    print(restaurant['business_id'],restaurant['name'],restaurant['address'])


    #price hotel
    for day in plan_eval:
        hotel_id = day['accommodation']
        if hotel_id != -1 and hotel_id != -2:
            price_denominator += 1
            for hotel in hotels:
                if(hotel['business_id'] == hotel_id):
                    if(hotel['price'] in price_limit):
                        price_numerator += 1
    


    #price - attractions
    all_attractions = []
    for day in plan_eval:
        for id in day['morning_attractions']:
            all_attractions.append(id)
        for id in day['afternoon_attractions']:
            all_attractions.append(id)
        for id in day['night_attractions']:
            all_attractions.append(id)
    price_denominator += len(all_attractions)

    for attraction_id in all_attractions:
        if attraction_id != -1 and attraction_id != -2:
            for attraction in attractions:
                if(attraction['business_id'] == attraction_id):
                    if(attraction['price'] in price_limit):
                        price_numerator += 1


    return price_numerator, price_denominator

def evaluate_attraction_orientation(plan_eval,eval):
    #attraction orientation
    orientation_numerator = 0
    orientation_denominator = 0

    oritentation_limit = eval['attraction'][0]
    oritentation_category = oritentation_limit
    oritentation_acceptable_list = ['medium ' + oritentation_limit, 'high ' + oritentation_limit]
    #print(oritentation_acceptable_list)
    all_attractions = []
    for day in plan_eval:
        for id in day['morning_attractions']:
            all_attractions.append(id)
        for id in day['afternoon_attractions']:
            all_attractions.append(id)
        for id in day['night_attractions']:
            all_attractions.append(id)
    orientation_denominator += len(all_attractions)
    for attraction_id in all_attractions:
        if attraction_id != -1 and attraction_id != -2:
            for attraction in attractions:
                if(attraction['business_id'] == attraction_id):
                    if attraction[oritentation_category] in oritentation_acceptable_list:
                        #print("attraction orientation is acceptable which is: ", attraction[oritentation_category])
                        orientation_numerator += 1
                        
    return orientation_numerator,orientation_denominator
                    #else:
                        #print("attraction orientation is not acceptable which is: ", attraction[oritentation_category])

def evaluate_cuisine(plan_eval,eval):
    #cuisine
    cuisine_numerator = 0
    cuisine_denominator = 0
    cuisine_satisfied = False

    cuisine_limit = [eval['cuisine'][0]]
    #print(cuisine_limit)
    if cuisine_limit == ['US']:
        cuisine_limit = ['American','American (New)','American (Traditional)']

    #at least we have one restaurant that match the cuisin, we will turn it to true
    all_meals = []
    for day in plan_eval:
        all_meals.append(day['breakfast'])
        all_meals.append(day['lunch'])
        all_meals.append(day['dinner'])
    cuisine_denominator += len(all_meals)
    #print(cuisine_denominator)

    for restaurant_id in all_meals:
        if restaurant_id != -1 and restaurant_id != -2:
            for restaurant in restaurants:
                if(restaurant['business_id'] == restaurant_id):
                    cuisine_provided = []
                    cuisine_provided.append(restaurant['cuisine_1'])
                    cuisine_provided.append(restaurant['cuisine_2'])
                    #print(restaurant_id)
                    #print((set(cuisine_provided))) 
                    if(len(list(set(cuisine_limit) & set(cuisine_provided))) > 0):
                        cuisine_numerator += 1
    #print(list(set(cuisine_limit)))
    
    return cuisine_numerator, cuisine_denominator

def evaluate_restaurants(plan_eval,eval):
    #restaurant 
    restaurants_numerator = 0
    restaurants_denominator = 0

    restaurants_limits = eval['restaurant']
    restaurants_category = [cat[5:] for cat in restaurants_limits]



    all_meals = []
    for day in plan_eval:
        all_meals.append(day['breakfast'])
        all_meals.append(day['lunch'])
        all_meals.append(day['dinner'])
    restaurants_denominator += len(all_meals)
    restaurants_denominator = restaurants_denominator * len(restaurants_category)

    for cat in restaurants_category:
        find_not_satisfication = False
        restaurants_acceptable_list = []
        restaurants_acceptable_list.append('good ' + cat)
        restaurants_acceptable_list.append('excellent ' + cat)

        for restaurant_id in all_meals: 
            if restaurant_id != -1 and restaurant_id != -2:
                for restaurant in restaurants:
                    if(restaurant['business_id'] == restaurant_id):
                        if(restaurant[cat] in restaurants_acceptable_list):
                            restaurants_numerator += 1
    return restaurants_numerator, restaurants_denominator

def evaluate_hotels(plan_eval,eval):
    #Hotel

    hotel_numerator = 0
    hotel_denominator = 0
    

    hotel_limit = eval['hotel']
    hotel_cat = [cat[5:] for cat in hotel_limit]

    all_hotels = []
    for day in plan_eval:
        hotel_id = day['accommodation']
        all_hotels.append(hotel_id)
    hotel_denominator += len(all_hotels)
    hotel_denominator *= len(hotel_cat)

    for cat in hotel_cat:
        find_not_satisfied = False
        hotel_acceptable_list = []
        hotel_acceptable_list.append('good ' + cat)
        hotel_acceptable_list.append('excellent ' + cat)

        for hotel_id in all_hotels:
            if hotel_id != -1 and hotel_id != -2:
                for hotel in hotels:
                    if(hotel['business_id'] == hotel_id):    
                        if(hotel[cat] in hotel_acceptable_list):
                            hotel_numerator += 1
    


    #note: only for hotel, we need to consider what if no recommendation, which means
    # all -2, we don't consider this in other categories since there is low chance that
    #llm didn't provide any reccommendation for food or attractions. 
    if(all(x == -2 for x in all_hotels)):
        hotel_numerator= 0 
        hotel_denominator = 0

    return hotel_numerator, hotel_denominator

def getFailure(failure_list):
    failure = [sum(x) for x in zip(*failure_list)]
    failure = [(x/len(failure_list)) for x in failure]
    return failure

def getMicro(preference_list):    
    micro = np.array([0,0])
    for record in preference_list:
        for cat in record:
            micro += np.array(cat)
    #print(micro)
    return micro[0]/micro[1]

def getMacro(preference_list):    
    numerator = []
    denomminator = len(preference_list)
    for record in preference_list:
        each =  [sum(x) for x in zip(*record)]
        numerator.append(1 if each[0] == each[1] else 0)
    #print(numerator, denomminator)
    return sum(numerator)/denomminator

def getNewMacro(preference_list, passrate):
    numerator = 0
    denominator = len(preference_list)

    for plan in preference_list:
        passedPlan = True
        for day in plan:
            if day[0] == day[1] == 0:
                continue
            if (day[0] / day[1]) < passrate:
                passedPlan = False
                break
        if passedPlan:
            numerator += 1
    #print(numerator, denominator)
    return numerator/denominator

def getFailureAndPreferenceList(model,task,numPlan):
    failure_list = []
    preference_list = []
    
    with open(f'Output/{model}/evals/{task}.jsonl', 'r') as f:
        plans = [json.loads(line) for line in f]

    with open('Dataset/evals.jsonl', 'r') as f:
        evals = [json.loads(line) for line in f]

    for i in range(numPlan):
        
        plan = plans[i]['plan']
        eval = evals[i]['eval_info']
        
        # Failure rate related
        # prepare a result list to return
        # outofpool, missinginfo,
        results = []
        # prepare the evaluation for each plan, search the business id
        plan_eval = prepareEval(plan)
        #print(plan_eval)

        outsidepool = evaluate_outSidePool(plan_eval)
        results.append(outsidepool)


        missingInfo = evaluate_missingInfo(plan_eval)
        results.append(missingInfo)
        #print(results)

        failure_list.append(results)

        # preference recall related
        
        results = []

        #day
        day_numerator, day_denominator = evaluate_day(plan_eval,eval)
        results.append([day_numerator, day_denominator])

        #price
        price_numerator, price_denominator = evaluate_price(plan_eval,eval)
        results.append([price_numerator, price_denominator])

        #attraction orientation
        attraction_numerator, attraction_denominator = evaluate_attraction_orientation(plan_eval,eval)
        results.append([attraction_numerator, attraction_denominator])
        
        #cuisine
        cuisine_numerator, cuisine_denominator = evaluate_cuisine(plan_eval,eval)
        results.append([cuisine_numerator, cuisine_denominator])

        #restaurants
        restaurants_numerator, restaurants_denominator = evaluate_restaurants(plan_eval,eval)
        results.append([restaurants_numerator, restaurants_denominator])

        #hotels
        hotels_numerator, hotels_denominator = evaluate_hotels(plan_eval,eval)
        results.append([hotels_numerator, hotels_denominator])

        preference_list.append(results)
        #print(preference_list)
    return failure_list, preference_list

def populateCordinates(plan_eval, data, data_hotel):
    cordinates = []
    for day in plan_eval:
        cordinate_one_day = []

        #if the hotel is invalid, we skip the day
        if(day['accommodation'] == -1 or day['accommodation'] == -2):
            continue

        if(day['accommodation'] != -1):
            cordinate_one_day.append(getCordinate_Hotel(day['accommodation'], data_hotel))
        
        for attraction in day['morning_attractions']:
            if(attraction != -1):
                cordinate_one_day.append(getCordinate(attraction,data))
        for attraction in day['afternoon_attractions']:
            if(attraction != -1):
                cordinate_one_day.append(getCordinate(attraction,data))
        for attraction in day['night_attractions']:
            if(attraction != -1):
                cordinate_one_day.append(getCordinate(attraction,data))
                
        cordinates.append(cordinate_one_day)
    return cordinates

def getCordinate(id,data):
    for attraction in data:
        if attraction['business_id'] == id:
            return (attraction['latitude'], attraction['longitude'])

def getCordinate_Hotel(id,data_hotel):
    for hotel in data_hotel:
        if hotel['business_id'] == id:
            return (hotel['latitude'], hotel['longitude'])
        
def getDistanceMatrix(cordinates):
    #print(cordinates)
    n = len(cordinates)
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            distance_matrix[i][j] = distance_matrix[j][i] = ((cordinates[i][0]*1000 - cordinates[j][0]*1000)**2 + (cordinates[i][1]*1000 - cordinates[j][1]*1000)**2)**0.5
    return distance_matrix

def populateShortestDistanceOneDay(cordinates):
    shortest_distance_list = []
    shortest_distance_info_lists = []
    for oneday in cordinates:
        distance_matrix = getDistanceMatrix(oneday)
        n = len(distance_matrix)
        info_lists = []
        optimized_distance = totalCost(1, 0, n, distance_matrix,info_lists)
        shortest_distance_list.append(optimized_distance)
        shortest_distance_info_lists.append(info_lists)
    return shortest_distance_list, shortest_distance_info_lists

def totalCost(mask, pos, n, cost, info_lists):
    distance_list = []
    i_list = []
    # Base case: if all cities are visited, return the
    # cost to return to the starting city (0)
    if mask == (1 << n) - 1:
        return cost[pos][0]

    ans = sys.maxsize   

    # Try visiting every city that has not been visited yet
    for i in range(n):
        if (mask & (1 << i)) == 0: 
            i_list.append(i)
            # If city i is not visited, visit it and 
            #  update the mask
            distance_list.append(cost[pos][i] +
                      totalCost(mask | (1 << i), i, n, cost, info_lists))
        

    info_list = [pos,i_list, distance_list]
    info_lists.append(info_list)
    
    ans = min(distance_list)
    return ans

def populatePlannedDistanceOneDay(cordinates):
    planned_distance_list = []
    for oneday in cordinates:
        distance_matrix = getDistanceMatrix(oneday)
        #print(distance_matrix)
        distance = 0
        for i in range(len(distance_matrix)):
            if i == len(distance_matrix) - 1:
                j = 0
            else:
                j = i + 1
            distance += distance_matrix[i][j]
        planned_distance_list.append(distance)
    return planned_distance_list

def getDistanceGapRatio(shortest_distances_by_day, planned_distances_by_day):
    distance_gap = 0
    total_distance = 0
    for optimized_distance, planned_distance in zip(shortest_distances_by_day, planned_distances_by_day):
        gap = []
        gap = np.sum(np.array(planned_distance) - np.array(optimized_distance))
        distance_gap += gap
        
        total = np.sum(np.array(planned_distance))
        total_distance += total
        
    return distance_gap / total_distance

def getOptimizedOrder(shortest_distance_info_lists):


    order_list = []
    for day in shortest_distance_info_lists:

        if len(day) == 0:
            order_list.append([[0],[0]])
            continue

        pos = 0
        n = len(day[-1][1]) + 1
        #get a list of 1 to n
        candidates = list(range(n-1))
        #add 1 to the values
        candidates = [x+1 for x in candidates]

        moves = []

        while len(candidates) > 0:
            #find the last one in the lnfo_list
            for i in range(len(day)):
                if day[i][0] == pos and day[i][1] == candidates:
                    #print(day[i][0],day[i][1])
                    next_move = day[i][1][np.argmin(day[i][2])]
                    #print(next_move)
                    pos = next_move
                    moves.append(next_move)
                    #take next move out of candidates
                    candidates.remove(next_move)

        moves_reversed = moves[::-1]
        optimized_route = [[0] + moves, [0] + moves_reversed]
        order_list.append(optimized_route)

    return order_list
        
def getPositionDeviationRatio(shortest_order_by_day):
    total_places = 0
    total_deviation = 0
    for plan in shortest_order_by_day:
        for day in plan:
            n = len(day[0])
            total_places += n
            output_route = list(range(n))
            gap_1 = sum([1 if x != y else 0 for x,y in zip(output_route,day[0])])
            gap_2 = sum([1 if x != y else 0 for x,y in zip(output_route,day[1])])
            total_deviation += min(gap_1, gap_2)
    return total_deviation / total_places

def daywiseTSP(model,task, numPlan):
    shortest_distances_by_day = []
    planned_distances_by_day = []
    shortest_order_by_day = []

    with open(f'Output/{model}/evals/{task}.jsonl', 'r') as f:
        plans = [json.loads(line) for line in f]
    for i in range(numPlan):
        if (i%20 == 0):
            print("Mode: day wise. We are at plan ", i)
        plan = plans[i]['plan'] 
        
        # Failure rate related
        # prepare a result list to return
        # outofpool, missinginfo,
        # prepare the evaluation for each plan, search the business id
        plan_eval = prepareEval(plan)
        #print(plan_eval)

        #get the cordinates
        cordinates = populateCordinates(plan_eval, attractions, hotels)
        #print(cordinates)
        #one day shortest distance
        shortest_distance_list_each_day, shortest_distance_info_lists = populateShortestDistanceOneDay(cordinates)
        #print(shortest_distance_info_lists)
        shortest_distances_by_day.append(shortest_distance_list_each_day)
        
        shortest_order_list_each_day = getOptimizedOrder(shortest_distance_info_lists)
        
        shortest_order_by_day.append(shortest_order_list_each_day)
        #shortest_order_by_day(info_list)
        
        #one day planned distance
        planned_distance_list_each_day = populatePlannedDistanceOneDay(cordinates)
        planned_distances_by_day.append(planned_distance_list_each_day)

        #plan wise (multi day) optimization calculation

    #get distance gap ratio
    distance_gap_ratio = getDistanceGapRatio(shortest_distances_by_day, planned_distances_by_day)

    #position deviation ratio
    position_deviation_ratio = getPositionDeviationRatio(shortest_order_by_day)
    
    return distance_gap_ratio, position_deviation_ratio

def getHotelIndex(day,cordinates):
    hotel_index = 0
    if day > 0:
        for j in range(day):
            hotel_index += len(cordinates[j])
    return hotel_index

def totalCost_multiday(mask, pos, day, cordinates, n, visited, cost, info_lists, memo):
    visit_requirement = len(cordinates[day])
    distance_list = []
    i_list = []

    hotel_index = getHotelIndex(day,cordinates)
    # Base case: if all cities are visited, return the
    # cost to return to the starting city (0)

    if mask == (1 << n) - 1:
        return cost[pos][hotel_index]
    
    if memo[pos][mask] != -1:
        return memo[pos][mask]

    if visit_requirement == visited:
        for i in range(n):
            if (mask & (1 << i)) == 0: 
                i_list.append(i)
                distance_list.append(cost[hotel_index][i] + totalCost_multiday(mask | (1 << i), i, day + 1, cordinates, n, 2, cost, info_lists,memo))
        
        info_list = [pos,i_list, distance_list]
        info_lists.append(info_list)
        
        return min(distance_list) + cost[pos][hotel_index] # change this to the old hotel position
    
    # Try visiting every city that has not been visited yet
    for i in range(n):
        if (mask & (1 << i)) == 0: 

            i_list.append(i)
            # If city i is not visited, visit it and 
             #  update the mask
            distance_list.append(cost[pos][i] +
                      totalCost_multiday(mask | (1 << i), i, day, cordinates, n, visited + 1, cost, info_lists,memo))
        

    info_list = [pos,i_list, distance_list]
    info_lists.append(info_list)
    
    memo[pos][mask] = min(distance_list)

    return min(distance_list)

def getDistanceMatrix_by_plan(cordinates):
    #print(cordinates)
    n = 0
    for day in cordinates:
        for place in day:
            n+=1
    flattened = []
    for day in cordinates:
        for location in day:
            flattened.append(location)
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            distance_matrix[i][j] = distance_matrix[j][i] = ((flattened[i][0]*1000 - flattened[j][0]*1000)**2 + (flattened[i][1]*1000 - flattened[j][1]*1000)**2)**0.5
    return distance_matrix

def getOptimizedDistance_by_plan(cordinates,distance_matrix):
    n = len(distance_matrix)
    info_lists = []
    #newMask will have all the hotels as 1 before the function.
    newMask = 1
    index_list = list(range(len(cordinates) - 1))
    index_list = index_list[::-1]
    newMask = 1
    for i in index_list:
        newMask = (newMask << (len(cordinates[i]))) + 1
    memo = [[-1] * (1 << n) for _ in range(n)]
    optimized_distance = totalCost_multiday(newMask,0,0,cordinates,n,1,distance_matrix, info_lists,memo)
    return optimized_distance, info_lists

def getOptimizedOrder_by_plan(info_lists):
    pos = 0
    lookfor = info_lists[-1][1].copy()
    moves = []
    while len(lookfor) > 0:
        for record in info_lists:
            if(record[0] == pos and record[1] == lookfor):
                nextmove = record[1][np.argmin(record[2])]
                pos = nextmove
                moves.append(nextmove)
                lookfor.remove(pos)
    return moves

def getPlannedDistance_by_plan(cordinates, distance_matrix):
    distance = 0
    for i in range(len(distance_matrix)):
        if i == len(distance_matrix) - 1:
            j = len(distance_matrix) - len(cordinates[-1])
        else:
            j = i + 1
        distance += distance_matrix[i][j]
    return distance

def getDistanceGapRatio_by_plan(optimized_distances_by_plan, planned_distances_by_plan):
    gaps = np.array([])
    for optimized,planned in zip(optimized_distances_by_plan,planned_distances_by_plan):
        gap = planned - optimized
        gaps = np.append(gaps,gap)
    total_gap = np.sum(gaps)
    total_planned = np.sum(planned_distances_by_plan)
    ratio = total_gap / total_planned
    return ratio

def getDistanceGapRatio_by_plan(optimized_distances_by_plan, planned_distances_by_plan):
    gaps = np.array([])
    for optimized,planned in zip(optimized_distances_by_plan,planned_distances_by_plan):
        gap = planned - optimized
        gaps = np.append(gaps,gap)
    total_gap = np.sum(gaps)
    total_planned = np.sum(planned_distances_by_plan)
    ratio = total_gap / total_planned
    return ratio

def getClusterJumpRatio(optimized_orders_by_plan, cordinates_list):
    gaps = 0
    totals = 0
    for order, cordinates in zip(optimized_orders_by_plan,cordinates_list):
        days = []
        for _ in cordinates:
            days.append(len(_))
        days = np.array(days) - 1 #
        #print(days)
        optimzied_cluster_count = len(days) #
        #print(optimzied_cluster_count)

        totalplaces = optimzied_cluster_count + len(order)
        #print(totalplaces)
        candidates = list(range(totalplaces))
        #print(candidates)
        
        cluster_list = [] #
        for cand in candidates:
            if cand in order:
                index = order.index(cand)
                clusterNumber = getcluster(index,days)
                cluster_list.append(clusterNumber)
        #print(cluster_list)

        cluster_set =  makeClusterSet(cluster_list,days)
        #print(cluster_set)

        cluster_visited_count = 0
        for cluster in cluster_set:
            cluster_visited_count += len(cluster)
        #print(cluster_visited_count)

        gaps += cluster_visited_count - optimzied_cluster_count
        totals += optimzied_cluster_count
    #print(gaps,totals)
    return gaps/totals

def getcluster(index,days):
    clusterNumber = 0
    for day in days:
        index = index - day
        if index < 0:
            return clusterNumber
        else:
            clusterNumber += 1

def makeClusterSet(cluster_list,days):
    cluster_sets = []
    for day in days:
        cluster = []
        n = day
        while n > 0:
            cluster.append(cluster_list[0])
            cluster_list.pop(0)
            n -= 1
        cluster_sets.append(set(cluster))
    return cluster_sets

def planwiseTSP(model, task, numPlan):
    optimized_distances_by_plan = []
    optimized_orders_by_plan = []
    planned_distances_by_plan = []
    cordinates_list = []

    ave_att_per_day = 0
    total_days = 0
    total_att = 0

    with open(f'Output/{model}/evals/{task}.jsonl', 'r') as f:
        plans = [json.loads(line) for line in f]

    for i in range(numPlan):
        
        if i%20 == 0:
            print("Mode: Plan wise. we are at plan: ", i)
        plan = plans[i]['plan'] 
        
        # Failure rate related
        # prepare a result list to return
        # outofpool, missinginfo,
        # prepare the evaluation for each plan, search the business id
        plan_eval = prepareEval(plan)
        #if(plan_eval)
        if plan_eval[0]['accommodation'] == -1 or plan_eval[0]['accommodation'] == -2:
            continue
        
        #get the cordinates
        cordinates = populateCordinates(plan_eval, attractions, hotels)
        cordinates_list.append(cordinates)
        total_days += len(cordinates)
        #print(len(cordinates))

        distance_matrix = getDistanceMatrix_by_plan(cordinates)
        
        total_att+=len(distance_matrix[0])
        ave_att_per_day = total_att / total_days
        #print(len(distance_matrix[0]))
        
        optimized_distance, info_lists = getOptimizedDistance_by_plan(cordinates,distance_matrix)
        
        optimized_distances_by_plan.append(optimized_distance)
        
        optimized_order = getOptimizedOrder_by_plan(info_lists)
        #print(optimized_order)

        optimized_orders_by_plan.append(optimized_order)
        

        planned_distance = getPlannedDistance_by_plan(cordinates, distance_matrix)
        planned_distances_by_plan.append(planned_distance)

    distance_gap_ratio_by_plan = getDistanceGapRatio_by_plan(optimized_distances_by_plan, planned_distances_by_plan)
    cluster_jump_ratio_by_plan = getClusterJumpRatio(optimized_orders_by_plan, cordinates_list)
    
    #return 0,0,ave_att_per_day
    return distance_gap_ratio_by_plan,cluster_jump_ratio_by_plan,ave_att_per_day

def paraacc(log,eval):
    #attractions
    att_deno = 2
    att_num = 2
    for step in reversed(log):
        if step['state'] == 'Successful':
            if "AttractionSearch" in step['action']:
                full = str(step['action'])
                pattern = r'\[.*\]'
                match = re.search(pattern, full)
                if(match):
                    preferences = match.group(0).lower()
                    #print(preferences)
                
                pattern2 = r'\[(.*),\s*\[(.*)\]\]'
                match = re.search(pattern2, preferences)
                if match:
                    budget = match.group(1).strip()
                    pref = match.group(2).strip()
                    #print(budget, pref)
                    pref_list = pref.split(', ')
                    #print(pref_list)
                if budget != eval['price'][0].lower():
                    att_num -= 1
                for p in pref_list:
                    if p not in eval['attraction']:
                        att_num -= 1

    #accommodation
    hotel_deno = 0
    hotel_num = 0
    for step in reversed(log):
        if step['state'] == 'Successful':
            if "AccommodationSearch" in step['action']:
                full = str(step['action'])
                pattern = r'\[.*\]'
                match = re.search(pattern, full)
                if(match):
                    preferences = match.group(0).lower()
                    #print(preferences)
                
                pattern2 = r'\[(.*),\s*\[(.*)\]\]'
                match = re.search(pattern2, preferences)
                if match:
                    budget = match.group(1).strip()
                    pref = match.group(2).strip()
                    #print(budget, pref)
                    pref_list = pref.split(', ')
                    #print(pref_list)
                    hotel_deno = 1 + len(pref_list)
                    hotel_num = hotel_deno
                    #print(hotel_deno)
                if budget != eval['price'][0].lower():
                    hotel_num -= 1
                for p in pref_list:
                    if p not in eval['hotel']:
                        hotel_num -= 1

    #restaurant
    res_deno = 0
    res_num = 0
    for step in reversed(log):
        if step['state'] == 'Successful':
            if "RestaurantSearch" in step['action']:
                full = str(step['action'])
                full = 'Action 3: RestaurantSearch[Moderate Budget, Vietnamese, [Good Flavor, Good Value]]'
                pattern = r'\[.*\]'
                match = re.search(pattern, full)
                if(match):
                    preferences = match.group(0).lower()
                    #print(preferences)
                
                pattern2 = r'\[(.*),(.*),\s*\[(.*)\]\]'
                match = re.search(pattern2, preferences)
                if match:
                    budget = match.group(1).strip()
                    cuisine = match.group(2).strip()
                    pref = match.group(3).strip()
                    #print(budget, cuisine, pref)
                    pref_list = pref.split(', ')
                    #print(pref_list)
                    res_deno = 2 + len(pref_list)
                    res_num = res_deno
                    #print(res_deno)
                if budget != eval['price'][0].lower():
                    res_num -= 1
                if cuisine != eval['cuisine'][0].lower():
                    res_num -= 1
                for p in pref_list:
                    if p not in eval['restaurant']:
                        res_num -= 1


    
    return res_num + att_num + hotel_num , res_deno + att_deno + hotel_deno

def getParameterACC(model,numPlan):
    allnum = 0
    alldeno = 0
    with open (f'Output/{model}/plans/toolUseLogs.jsonl', 'r') as f:
        logs = [json.loads(line.strip()) for line in f]
    
    with open ('Dataset/evals.jsonl', 'r') as f:
        evals = [json.loads(line.strip()) for line in f]
    
    for log,eval in zip(logs,evals):
        #print(eval['eval_info'])
        num,deno = paraacc(log['log'],eval['eval_info'])
        allnum += num
        alldeno += deno

    rate = allnum/alldeno
    return rate

def getValidRate(model,task,numPlan,passRate):
    
    with open(f'Output/{model}/evals/{task}.jsonl', 'r') as f:
        plans = [json.loads(line) for line in f]

    with open ('Dataset/evals.jsonl', 'r') as f:
        evals = [json.loads(line.strip()) for line in f]

    validated = 0

    for i in range(numPlan):
        plan = plans[i]['plan']
        eval = evals[i]['eval_info']
        
        # Failure rate related
        # prepare a result list to return
        # outofpool, missinginfo,
        results = []
        # prepare the evaluation for each plan, search the business id
        plan_eval = prepareEval(plan)


        #Failure Rate
        outsidepool = evaluate_outSidePool(plan_eval)
        results.append(outsidepool)


        missingInfo = evaluate_missingInfo(plan_eval)
        results.append(missingInfo)
        
        if results != [0,0]:
            continue
        
        #Preference match
        results = []

        #day
        day_numerator, day_denominator = evaluate_day(plan_eval,eval)
        results.append([day_numerator, day_denominator])

        #price
        price_numerator, price_denominator = evaluate_price(plan_eval,eval)
        results.append([price_numerator, price_denominator])

        #attraction orientation
        attraction_numerator, attraction_denominator = evaluate_attraction_orientation(plan_eval,eval)
        results.append([attraction_numerator, attraction_denominator])
        
        #cuisine
        cuisine_numerator, cuisine_denominator = evaluate_cuisine(plan_eval,eval)
        results.append([cuisine_numerator, cuisine_denominator])

        #restaurants
        restaurants_numerator, restaurants_denominator = evaluate_restaurants(plan_eval,eval)
        results.append([restaurants_numerator, restaurants_denominator])

        #hotels
        hotels_numerator, hotels_denominator = evaluate_hotels(plan_eval,eval)
        results.append([hotels_numerator, hotels_denominator])

        if(getOneMacro(results,passRate)):
            validated += 1
            


    return validated/numPlan

def getOneMacro(results,passRate):
    #print(results) #[[1, 1], [11, 16], [8, 8], [4, 6], [10, 12], [4, 4]]

    for cat in results:
        #print(cat)
        if cat[0] == cat[1] == 0:
            continue
        
        if cat[0] / cat[1] < passRate:
            return False

    return True

if __name__ == "__main__":
    args = parse_args()
    restaurants = load_dataset("EthanWTL81/ItinBench", "restaurant", split="test")
    hotels = load_dataset("EthanWTL81/ItinBench", "hotel", split="test")
    attractions = load_dataset("EthanWTL81/ItinBench", "attraction", split="test")

    #choose model and task
    model = args.model
    task = args.task
    print("==Model: ", model,"==")
    print("==Task: ", task,"==")
    
    #failure and preference
    numPlan = args.numPlan
    passRate = args.threshold
    failure_list, preference_list = getFailureAndPreferenceList(model,task,numPlan)
    #print(preference_list)
    failure = getFailure(failure_list)
    micro = getMicro(preference_list)
    macro = getNewMacro(preference_list, passRate)
    validRate = getValidRate(model,task,numPlan,passRate)
    print(" ==Failure: ", failure, " ==Micro: ", micro, " ==NewMacro: ", macro, "==Valid Plan rate: ", validRate)

    #if the task is tool use, we also print the para acc
    if task == 'toolUsePlans':
        acc = getParameterACC(model,numPlan)
        print('===== Tool Calling Parameter Acc: ', acc)

    #TSP
    distance_gap_ratio, position_deviation_ratio = daywiseTSP(model,task,numPlan)
    print("we are done with day wise eval")
    distance_ratio, cluster_ratio, att_per_day = planwiseTSP(model,task,numPlan)
    print("we are done with plan wise eval")

    #add a average attraction number per day, since lower number of att number results in better results, which is not fair
#    print("===Average Attractions per Day ",att_per_day,)
    
    print("===Average Attractions per Day ",att_per_day," ==DistanceGapRatio: ", distance_gap_ratio, " ==PositionDeviationRatio: ", position_deviation_ratio, " ==DistanceRatio: ", distance_ratio, " ==ClusterRatio: ", cluster_ratio, "==")
    print("========")
