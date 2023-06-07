import numpy as np
import math
import sympy as sp
from gurobipy import *
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
import scipy.optimize
import matplotlib.pyplot as plt
import warnings
import pandas as pd
# import networkx as nx
import timeit
from time import time
import numpy as np
import ctypes
import copy


def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3



# pre-process
def pre_process_intsol(flows_link, flows_path, y_sol, v_sol, path_set, dual_p, 
                       link, link_fixed, MODnode_ori,MODoper_node_ori, MODnode, MODoper, dummy_link, d):
    dual = dict()
    for i in link:
        dual[link[i]["start"],link[i]["end"]] = 0

    for i in link_fixed:
        dual[link[i]["start"],link[i]["end"]] = dual_p[i]

    y = y_sol
    v = v_sol

    h_sol_MOD = dict()
    for i in MODnode_ori:
        h_sol_MOD[i] = 0
    for i in range(len(MODnode)):
        if v[i] == 1:
            nodee = MODnode_ori[MODnode_ori.index(np.floor(MODnode[i]/10))]
            h_sol_MOD[nodee] = MODnode[i]%10        
    h_sol = dict()
    for i in MODoper:
        h_sol[i] = 0
    for i in MODoper:
        for j in MODoper_node_ori[i]:
            if h_sol_MOD[j] > 0:
                h_sol[i] = h_sol_MOD[j]
                break
    h = []
    for i in MODoper:
        h.append(h_sol[i])

    x_sol=dict()
    for i in range(len(link)):
        if i not in dummy_link:
            for s in range(len(d)):
                x_sol[(link[i]['start'],link[i]['end'],s)]=flows_link[s,i]

    x_sol_= dict()
    for i in range(len(link)):
        if i not in dummy_link:
            for s in range(len(d)):
                x_sol_[(link[i]['start'],link[i]['end'])]=sum(flows_link[s,i] for s in range(len(d)))
    out_of_sys = []
    for linkk in dummy_link:
#         print(sum(flows_link[ddd,linkk] for ddd in range(len(d))))
        out_of_sys.append(sum(flows_link[ddd,linkk] for ddd in range(len(d))))

    path_set_ = []
    flows_path_ = []
    count = 0
    for s in range(len(d)):
        path_set_od = []
        for pathh in path_set[s]:
            if flows_path[count]>0 and pathh[0] not in dummy_link:
                path_set_od.append(pathh)
                flows_path_.append(flows_path[count])
            count+=1
        path_set_.append(path_set_od)
    path_set = path_set_
    flows_path = flows_path_  

    lenn=0
    for iii in path_set:
        lenn+=len(iii)
    print("Number of Paths:", lenn)
    
    return path_set, flows_path, dual, y, v, h_sol_MOD, h_sol, h, x_sol, x_sol_, out_of_sys


###################################################################################################
###################################################################################################
###################################################################################################


###### cost allo prep
def cost_allo_prep(path_set, flows_path, dual, y, v, h_sol_MOD, h_sol, h, x_sol, x_sol_, out_of_sys,
                  link, dummy_link, d, link_Acce, MODoper, oper_list, t_coef, c_coef):

    tt={}
    cst={}
    for i in link:
        if i not in dummy_link:
            cst[link[i]['start'],link[i]['end']]=link[i]['operating cost']
            for s in range(len(d)):
                tt[link[i]['start'],link[i]['end'],s]=link[i]['travel cost']
       
    for i in link_Acce:    
        fs = h[MODoper.index(link_Acce[i]['operator'])]
        flow = sum(x_sol[link_Acce[i]['start'],link_Acce[i]['end'],s] for s in range(len(d)))
        for s in range(len(d)):
            if flow == 0:
                tt[link_Acce[i]['start'],link_Acce[i]['end'],s]=0     
            elif fs == 0:
                tt[link_Acce[i]['start'],link_Acce[i]['end'],s]=9999999999999
            else:
                tt[link_Acce[i]['start'],link_Acce[i]['end'],s]=t_coef[link_Acce[i]['operator']]*(fs)**(-2)*(flow)**1

    volume=dict()
    for i in link:
        if i not in dummy_link:
            volume[(link[i]['start'],link[i]['end'])]=sum(x_sol[link[i]['start'],link[i]['end'],s] for s in range(len(d)))
    volume

    users=dict()
    operated=dict()
    for i in link:
        if i not in dummy_link:
            operated[(link[i]['start'],link[i]['end'])]=link[i]['operator']
            io=0
            for s in range(len(d)):
                if x_sol[link[i]['start'],link[i]['end'],s]>0 and io==0:
                    users[i]=[s]
                    io=1
                elif x_sol[link[i]['start'],link[i]['end'],s]>0 and io==1:
                    users[i].append(s)
    # print("users = ", users)

    #Route taken from user s
    route=dict()
    for s in range(len(d)):
        io=0;
        for i in link:
            if i not in dummy_link:
                if i not in users:
                    users[i]={}
                if x_sol[link[i]['start'],link[i]['end'],s]>0 and io==0:
                    route[s]=[[link[i]['start'],link[i]['end']]]
                    io=1
                elif x_sol[link[i]['start'],link[i]['end'],s]>0 and io==1:
                    route[s].append([link[i]['start'],link[i]['end']])
    # print("route = ", route)

    operators=dict()
    for i in oper_list:
        operators[i]={}
    for i in users:
        if users[i]!={}:
            if operators[link[i]['operator']]=={}:
                operators[(link[i]['operator'])]=[i]
            else:
                operators[(link[i]['operator'])].append(i)
    del operators[0]
    # print("operators = ", operators)

    path_set_node = []
    for s in range(len(d)):
        path_set_node_od = []
        for pp in path_set[s]:
            path_node_od = []
            for ii in range(len(pp)):
                if ii == 0:
                    path_node_od.append(link[pp[ii]]['start'])
                    path_node_od.append(link[pp[ii]]['end'])
                else:
                    path_node_od.append(link[pp[ii]]['end'])
            path_set_node_od.append(path_node_od)
        path_set_node.append(path_set_node_od)

        
    # print("test_routes = ", path_set_node)

    import copy
    new_routes = copy.deepcopy(path_set_node)
    s = 0
    count = 0
    for ff in range(len(flows_path)):
        if count<len(path_set[s]):
            pass
        else:
            s+=1
            count = 0
        if flows_path[ff] == 0:
            new_routes[s].pop(count)
        count+=1
    # print("new_routes = ", new_routes)

    #operators in each user path
    oper=dict()
    user_paths=dict()
    path_flow=dict()
    belong=dict()
    count=-1
    lst=[]
    link_to_path={}
    operators_in_path={}
    operators_vol_path={}
    path_hash={}

    for s in range(len(d)):
        user_paths[s]={}
        for path in new_routes[s]:
            count+=1
            path_hash[count]=path
            path_flow[count]=flows_path[count]
            operators_in_path[count]={}              
            if user_paths[s]=={}:
                user_paths[s]=[count]
            else:
                user_paths[s].append(count)
            path_edges=list(zip(path,path[1:]))
            for edge in path_edges:
                    if (count,operated[edge]) not in lst:
                        if operators_in_path[count]=={}:
                            if operated[edge] != 0:
                                operators_in_path[count]=[operated[edge]]
                        else:
                            if operated[edge] != 0:
                                operators_in_path[count].append(operated[edge])
                        lst.append((count,operated[edge]))
                    if edge not in link_to_path:
                        link_to_path[edge]=[count]
                    else:
                        link_to_path[edge].append(count)
                        
    count=-1
    for s in range(len(d)):
        for path in new_routes[s]:
            count+=1 
            for oper in oper_list:
                if oper in operators_in_path[count]:
                    operators_vol_path[count,oper]=path_flow[count]
                else:
                    operators_vol_path[count,oper]=0
    # print('path_hash = ', path_hash)
    # print('path_flow = ', path_flow)
    
    return tt, cst, volume, users, operated, route, operators, path_set_node, new_routes, oper, user_paths, path_flow, belong, count, lst, link_to_path, operators_in_path, operators_vol_path, path_hash


###################################################################################################
###################################################################################################
###################################################################################################


def cost_allo(x_sol, path_set, d, out_of_sys, utility, MODoper, MODnode, MODnode_oper, t_coef, c_coef, oper_list, flows_link, operators, link, link_fixed, link_trans, link_MOD, link_Acce, link_Egre, dummy_link, link_from_to, 
         a_MODlink, a_fixedlink, a_translink, new_routes, V_coef, MODoper_Acce, orig, dest,
         user_paths, operators_in_path, nodes, dual, y, v, h, tt): 

    x = x_sol

    used_link_set = []
    for iii in path_set:
        for jjj in iii:
            used_link_set = used_link_set+jjj
    used_link_set = list(np.unique(used_link_set))

    m1 = Model()
    # Model variables
    c = m1.addVars(range(len(d)),lb = 0,vtype = GRB.CONTINUOUS, name = 'price')
    u = m1.addVars(range(len(d)),lb = 0,ub=utility, vtype = GRB.CONTINUOUS, name = 'u')
    m1.update()
    
    # (5a) profitability
    infra_cost = dict()
    for operr in MODoper:
        infra_cost[operr] = 0
    for nodee in MODnode_oper:
        operr = MODnode_oper[nodee]
        infra_cost[operr] = infra_cost[operr] + v[MODnode.index(nodee)]*V_coef[nodee]

    MODoper_C = dict()
    for i in MODoper:
        link_ = MODoper_Acce[i]
        demand = 0
        for ddd in range(len(d)):
            for l in link_:
                demand += x_sol[l[0],l[1],ddd]
        MODoper_C[i] = c_coef[i]*demand*(h[MODoper.index(i)]**2) + infra_cost[i]

    ttl_oper_cost = 0

    oper_list_ = oper_list.copy()  
    oper_list_.remove(0) 

    for oper in oper_list_:
        if oper not in MODoper: # fixed operator       
            ttl_oper_cost += sum(link[i]['operating cost'] for i in operators[oper])  
        else: # MOD operator
            ooo = intersection(operators[oper], list(a_MODlink))
            ttl_oper_cost += MODoper_C[oper]

    flow_s = list(np.array(d) - np.array(out_of_sys))

    m1.addConstr(quicksum(c[s]*flow_s[s] for s in range(len(d))) >= ttl_oper_cost, name = 'Profitability')
    m1.update()
#     m1.display()


    # (5b) feasibility (Requiring same payoff for user paths of the same O-D)
    count =- 1
    for s in range(len(d)):
        for path in new_routes[s]:
            count += 1
            rhs = utility[s]
            lhs = 0
            ttl_price = 0
            path_edges = list(zip(path,path[1:]))
            for edge in path_edges:
                rhs -= tt[edge[0],edge[1],s]     
            path_index = new_routes[s].index(path)
            m1.addConstr(u[s]+ c[s]== rhs, name = 'cost_allo_'+str(s)+'_'+str(path_index))    
    m1.update()
#     m1.display()

    # (5c) stability
    capacity = dict()
    cost = dict()

    #Users on link ij
    serves = dict()
    for i in link:
        if i not in dummy_link:
            io = 0
            for s in range(len(d)):
                if x[link[i]['start'],link[i]['end'],s]>0 and io == 0:
                    serves[(link[i]['start'],link[i]['end'])] = [s]
                    io=1
                elif x[link[i]['start'],link[i]['end'],s]>0 and io == 1:
                    serves[(link[i]['start'],link[i]['end'])].append(s)
    for i in link:
        if i not in dummy_link:
            capacity[(link[i]['start'],link[i]['end'])] = link[i]['capacity']
            cost[(link[i]['start'],link[i]['end'])] = link[i]['operating cost']
            if (link[i]['start'],link[i]['end']) not in serves:
                serves[(link[i]['start'],link[i]['end'])] = {} 

    C_ij = dict()
    for i in a_fixedlink:
        if  serves[(link_fixed[i]['start'],link_fixed[i]['end'])] == {}:
            C_ij[(link_fixed[i]['start'],link_fixed[i]['end'])] = link_fixed[i]['operating cost']
        else:
            C_ij[(link_fixed[i]['start'],link_fixed[i]['end'])] = 0
    for i in a_translink:
        C_ij[(link_trans[i]['start'],link_trans[i]['end'])] = 0

    for i in link_MOD:
        fs = h[MODoper.index(link_MOD[i]['operator'])]

        if fs > 0: # the operator is operating
            link_ = MODoper_Acce[link_MOD[i]['operator']]
            demand = 0
            for ddd in range(len(d)):
                for l in link_:
                    demand += x[l[0],l[1],ddd]
            C_ij[(link_MOD[i]['start'],link_MOD[i]['end'])] = c_coef[link_MOD[i]['operator']]*(fs**2)*((demand+1)**1) - c_coef[link_MOD[i]['operator']]*(fs**2)*((demand)**1)
            if v[MODnode.index(link_MOD[i]['start'])] == 0:
                C_ij[(link_MOD[i]['start'],link_MOD[i]['end'])] = C_ij[(link_MOD[i]['start'],link_MOD[i]['end'])]+V_coef[link_MOD[i]['start']]
            if v[MODnode.index(link_MOD[i]['end'])] == 0:
                C_ij[(link_MOD[i]['start'],link_MOD[i]['end'])] = C_ij[(link_MOD[i]['start'],link_MOD[i]['end'])]+V_coef[link_MOD[i]['end']]
        else: # the operator is not operating
            C_ij[(link_MOD[i]['start'],link_MOD[i]['end'])] = c_coef[link_MOD[i]['operator']]*(1**2)*(1**1)
            if v[MODnode.index(link_MOD[i]['start'])] == 0:
                C_ij[(link_MOD[i]['start'],link_MOD[i]['end'])] = C_ij[(link_MOD[i]['start'],link_MOD[i]['end'])]+V_coef[link_MOD[i]['start']]
            if v[MODnode.index(link_MOD[i]['end'])] == 0:
                C_ij[(link_MOD[i]['start'],link_MOD[i]['end'])] = C_ij[(link_MOD[i]['start'],link_MOD[i]['end'])]+V_coef[link_MOD[i]['end']]

    for i in link_Acce:
        C_ij[(link_Acce[i]['start'],link_Acce[i]['end'])] = 0
    for i in link_Egre:
        C_ij[(link_Egre[i]['start'],link_Egre[i]['end'])] = 0    

    T_ij = dict()
    for i in link_fixed:
        T_ij[(link_fixed[i]['start'],link_fixed[i]['end'])] = link_fixed[i]['travel cost']
    for i in link_trans:
        T_ij[(link_trans[i]['start'],link_trans[i]['end'])] = link_trans[i]['travel cost']
    for i in link_MOD:
        T_ij[(link_MOD[i]['start'],link_MOD[i]['end'])] = link_MOD[i]['travel cost']
    for i in link_Acce:
        fs = h[MODoper.index(link_Acce[i]['operator'])]
    #     print(link_Acce[i]['start'],link_Acce[i]['end'], fs)
        flow = sum(x[link_Acce[i]['start'],link_Acce[i]['end'],ddd] for ddd in range(len(d))) 
        if fs == 0:
            T_ij[(link_Acce[i]['start'],link_Acce[i]['end'])]  = 9999999999999
        else:
            T_ij[(link_Acce[i]['start'],link_Acce[i]['end'])] = t_coef[link_Acce[i]['operator']]*(fs)**(-2)*(flow+1)**1
    for i in link_Egre:
        T_ij[(link_Egre[i]['start'],link_Egre[i]['end'])] = 0

    ### revise link: remove closed MOD nodes, links, and transfer links
    MOD_node_closed = []
    for i in range(len(MODnode)):
        if v[i] == 0:
            MOD_node_closed.append(MODnode[i])
    link_used = copy.deepcopy(link)
    for i in link:
        start_nodee = link_used[i]['start']
        end_nodee = link_used[i]['end']
        if (start_nodee in MOD_node_closed) or (end_nodee in MOD_node_closed):
            del link_used[i]
        if i in dummy_link:
            del link_used[i]
            
            
    from itertools import combinations
    from MaaS_game_OD_subscribe import Dijkstra_cpp

    stability_path_set = dict()
    for s in range(len(d)):
        stability_path_set[s] = []

    for s in range(len(d)):
        stability_path_set_od = []

        oper_set = []
        for path_id in user_paths[s]:
            if operators_in_path[path_id] == {}:
                pass
            else:
                oper_set = oper_set + operators_in_path[path_id]
        oper_set = list(np.unique(oper_set))

        oper_comb = []
        for numm in range(1,len(oper_set)+1):
            comb_ = list(combinations(oper_set, numm))
            oper_comb = oper_comb+comb_

        # for i in list(oper_comb): 
        #     print (i) 

        oper_comb_fin = copy.deepcopy(oper_comb)
        for path_id in user_paths[s]:
            oper_path = operators_in_path[path_id]
            for iii in oper_comb:
                if all(operr not in iii for operr in oper_path):
                    if iii in oper_comb_fin:
                        oper_comb_fin.pop(oper_comb_fin.index(iii))

        for pi in oper_comb_fin:
            ## shortest path - such that no operator is in pi
            # graph matrix
            graph_omega = np.ones((len(nodes),len(nodes)))*(-1)
            for linkk in link:
                if linkk not in dummy_link:
                    if link[linkk]['operator'] not in pi:
                        start_id = nodes.index(link[linkk]['start'])
                        end_id = nodes.index(link[linkk]['end'])
                        graph_omega[start_id, end_id] = T_ij[link[linkk]['start'],link[linkk]['end']]+dual[link[linkk]['start'],link[linkk]['end']]+C_ij[link[linkk]['start'],link[linkk]['end']]
            # dijkstra
            O = int(orig[s])
            D = int(dest[s])
            O_id = nodes.index(O)
            D_id = nodes.index(D)
            path_node_ind_lst, dist = Dijkstra_cpp(nodes, O_id, D_id, graph_omega)
            path_node_ind = path_node_ind_lst[D_id]
            path_len = dist[D_id]
            path_node = []
            for node_ind in path_node_ind:
                nodee = nodes[node_ind]
                path_node.append(nodee)
            path_link = []
            for nodee_ind in range(len(path_node)-1):
                path_link.append(link_from_to[path_node[nodee_ind],path_node[nodee_ind+1]])

            r = path_link # link id's
    #         print(r)

            if r != []:
                stability_path_set_od.append(r)
                core = utility[s]
    #             UN=[]
    #             rhs=0
                for linkk in r:
                    core-=T_ij[link[linkk]['start'],link[linkk]['end']]
                    core-=dual[link[linkk]['start'],link[linkk]['end']]
                    core-=C_ij[link[linkk]['start'],link[linkk]['end']]
    #                 if linkk in used_link_set:
    #                     UN.append(c[linkk,link[linkk]['operator']])
    #             rhs+=sum(list(set(UN)))
                #adding stability constraints
            #     path_index = new_routes[s].index(j)
                m1.addConstr(c[s]+u[s]>=core, 
                             name = 'stability_'+str(r))

                m1.update()
        stability_path_set[s] = stability_path_set_od

    # m1.display()   

    obj = 88
    m1.update()

    m1.params.outputflag=0
    # m1.display()
    m1.optimize()
    #     print(m1.status)
    # print(u)
    # m1.computeIIS()
    # m1.write("model_infea.ilp")
    # a = read('model_infea.ilp')
    # a.display()

    if m1.status == 3 or m1.status == 4:
        m1.computeIIS()
        m1.write("model_infea.ilp")
        a = read('model_infea.ilp')
        a.display()
        infea_path = []
        for s in range(len(d)):
            for path in new_routes[s]:
                path_index = new_routes[s].index(path)
                if a.getConstrByName('cost_allo_'+str(s)+'_'+str(path_index)) == None:
                    pass
                else:
                    infea_path.append(path)
        unstab_path = []
        alter_path = []
        for s in range(len(d)):
            for path in new_routes[s]:
                path_index = new_routes[s].index(path)
                for k in range(len(path_set[s][path_index])):
                    if a.getConstrByName('stability_'+str(s)+'_'+str(path_index)+'_'+str(k)) == None:
                        pass
                    else:
                        unstab_path.append(path)
                        alter_path.append(path_set[s][path_index])
        print('Infeasible path:',infea_path)            
        print('Unstable path:',unstab_path)
        print('Alternative path:',alter_path)
        prices = []
        p = {}
        u_s = []
        u = [] 

        return 0, stability_path_set, [], [], [], [], [], []

    else:
        print('Cost Allocation feasible!')

        # ### User optimal
        obj = 0
        obj += quicksum(u[s]*(d[s]-out_of_sys[s]) for s in range(len(d))) 
        m1.setObjective(obj,GRB.MAXIMIZE)
        m1.update()
        m1.params.outputflag=0
        m1.optimize()
        x_ = m1.getAttr('X', m1.getVars())
        prices = x_[0:len(d)]
        p_user_opt=dict()
        for s in range(len(d)):
            p_user_opt[s]=prices[s]
        u_s_user_opt = x_[len(d) : len(d)+len(d)]
        u_user_opt = dict()
        for var in u:
            u_user_opt[var]=u[var].x

        # ### Operator optimal
        obj = 0
        obj += quicksum(c[s]*(d[s]-out_of_sys[s]) for s in range(len(d)))
        m1.setObjective(obj,GRB.MAXIMIZE)
        m1.update()
        m1.params.outputflag=0
        m1.optimize()
        x_ = m1.getAttr('X', m1.getVars())
        prices = x_[0:len(d)]
        p_oper_opt=dict()
        for s in range(len(d)):
            p_oper_opt[s]=prices[s]
        u_s_oper_opt = x_[len(d) : len(d)+len(d)]
        u_oper_opt = dict()
        for var in u:
            u_oper_opt[var]=u[var].x

        return 1, stability_path_set, p_user_opt, u_s_user_opt, u_user_opt, p_oper_opt, u_s_oper_opt, u_oper_opt

    

###################################################################################################
###################################################################################################
###################################################################################################


def subsidy_stablize(stability_path_set, x_sol, path_set, d, out_of_sys, utility, MODoper, MODnode, MODnode_oper, t_coef, c_coef, oper_list, flows_link, operators, link, link_fixed, link_trans, link_MOD, link_Acce, link_Egre, dummy_link, link_from_to, path_flow, 
             a_MODlink, a_fixedlink, a_translink, new_routes, V_coef, MODoper_Acce, orig, dest, 
             user_paths, operators_in_path, nodes, dual, y, v, h, tt):

    x = x_sol

    used_link_set = []
    for iii in path_set:
        for jjj in iii:
            used_link_set = used_link_set+jjj
    used_link_set = list(np.unique(used_link_set))
    lst = []
    for iii in used_link_set:
        aaa = (iii,link[iii]["operator"])
        lst.append(aaa)
    number_routes = dict()
    for i in range(len(d)):
        number_routes[i] = len(new_routes[i])
    lst_sub = []
    for i in number_routes:
        for iii in range(number_routes[i]):
            lst_sub.append((i,iii))
            
    m1 = Model()
    # Model variables
    c = m1.addVars(range(len(d)),lb = 0,vtype = GRB.CONTINUOUS, name = 'price')
    u = m1.addVars(range(len(d)),lb = 0,ub=utility, vtype = GRB.CONTINUOUS, name = 'u')
    S_path = m1.addVars(lst_sub, lb = 0, vtype = GRB.CONTINUOUS, name = 'subsidy_user')  
    m1.update()

    
    # (5a) profitability
    infra_cost = dict()
    for operr in MODoper:
        infra_cost[operr] = 0
    for nodee in MODnode_oper:
        operr = MODnode_oper[nodee]
        infra_cost[operr] = infra_cost[operr] + v[MODnode.index(nodee)]*V_coef[nodee]

    MODoper_C = dict()
    for i in MODoper:
        link_ = MODoper_Acce[i]
        demand = 0
        for ddd in range(len(d)):
            for l in link_:
                demand += x_sol[l[0],l[1],ddd]
        MODoper_C[i] = c_coef[i]*demand*(h[MODoper.index(i)]**2) + infra_cost[i]

    ttl_oper_cost = 0

    oper_list_ = oper_list.copy()  
    oper_list_.remove(0)

    for oper in oper_list_:
        if oper not in MODoper: # fixed operator       
            ttl_oper_cost += sum(link[i]['operating cost'] for i in operators[oper])  
        else: # MOD operator
            ooo = intersection(operators[oper], list(a_MODlink))
            ttl_oper_cost += MODoper_C[oper]

    flow_s = list(np.array(d) - np.array(out_of_sys))

    m1.addConstr(quicksum(c[s]*flow_s[s] for s in range(len(d))) >= ttl_oper_cost, name = 'Profitability')
    m1.update()

   
    # (5b) feasibility (Requiring same payoff for user paths of the same O-D)
    count =- 1
    for s in range(len(d)):
        for path in new_routes[s]:
            count += 1
            path_index = new_routes[s].index(path)
            rhs = utility[s] + S_path[s,path_index]
            lhs = 0
            ttl_price = 0
            path_edges = list(zip(path,path[1:]))
            for edge in path_edges:
                rhs -= tt[edge[0],edge[1],s]     
            path_index = new_routes[s].index(path)
            m1.addConstr(u[s]+ c[s]== rhs, name = 'cost_allo_'+str(s)+'_'+str(path_index))    
    m1.update()

    # (5c) stability
    capacity = dict()
    cost = dict()

    #Users on link ij
    serves = dict()
    for i in link:
        if i not in dummy_link:
            io = 0
            for s in range(len(d)):
                if x[link[i]['start'],link[i]['end'],s]>0 and io == 0:
                    serves[(link[i]['start'],link[i]['end'])] = [s]
                    io=1
                elif x[link[i]['start'],link[i]['end'],s]>0 and io == 1:
                    serves[(link[i]['start'],link[i]['end'])].append(s)
    for i in link:
        if i not in dummy_link:
            capacity[(link[i]['start'],link[i]['end'])] = link[i]['capacity']
            cost[(link[i]['start'],link[i]['end'])] = link[i]['operating cost']
            if (link[i]['start'],link[i]['end']) not in serves:
                serves[(link[i]['start'],link[i]['end'])] = {} 

    C_ij = dict()
    for i in a_fixedlink:
        if  serves[(link_fixed[i]['start'],link_fixed[i]['end'])] == {}:
            C_ij[(link_fixed[i]['start'],link_fixed[i]['end'])] = link_fixed[i]['operating cost']
        else:
            C_ij[(link_fixed[i]['start'],link_fixed[i]['end'])] = 0
    for i in a_translink:
        C_ij[(link_trans[i]['start'],link_trans[i]['end'])] = 0

    for i in link_MOD:
        fs = h[MODoper.index(link_MOD[i]['operator'])]

        if fs > 0: # the operator is operating
            link_ = MODoper_Acce[link_MOD[i]['operator']]
            demand = 0
            for ddd in range(len(d)):
                for l in link_:
                    demand += x[l[0],l[1],ddd]
            C_ij[(link_MOD[i]['start'],link_MOD[i]['end'])] = c_coef[link_MOD[i]['operator']]*(fs**2)*((demand+1)**1) - c_coef[link_MOD[i]['operator']]*(fs**2)*((demand)**1)
            if v[MODnode.index(link_MOD[i]['start'])] == 0:
                C_ij[(link_MOD[i]['start'],link_MOD[i]['end'])] = C_ij[(link_MOD[i]['start'],link_MOD[i]['end'])]+V_coef[link_MOD[i]['start']]
            if v[MODnode.index(link_MOD[i]['end'])] == 0:
                C_ij[(link_MOD[i]['start'],link_MOD[i]['end'])] = C_ij[(link_MOD[i]['start'],link_MOD[i]['end'])]+V_coef[link_MOD[i]['end']]
        else: # the operator is not operating
            C_ij[(link_MOD[i]['start'],link_MOD[i]['end'])] = c_coef[link_MOD[i]['operator']]*(1**2)*(1**1)
            if v[MODnode.index(link_MOD[i]['start'])] == 0:
                C_ij[(link_MOD[i]['start'],link_MOD[i]['end'])] = C_ij[(link_MOD[i]['start'],link_MOD[i]['end'])]+V_coef[link_MOD[i]['start']]
            if v[MODnode.index(link_MOD[i]['end'])] == 0:
                C_ij[(link_MOD[i]['start'],link_MOD[i]['end'])] = C_ij[(link_MOD[i]['start'],link_MOD[i]['end'])]+V_coef[link_MOD[i]['end']]

    for i in link_Acce:
        C_ij[(link_Acce[i]['start'],link_Acce[i]['end'])] = 0
    for i in link_Egre:
        C_ij[(link_Egre[i]['start'],link_Egre[i]['end'])] = 0    

    T_ij = dict()
    for i in link_fixed:
        T_ij[(link_fixed[i]['start'],link_fixed[i]['end'])] = link_fixed[i]['travel cost']
    for i in link_trans:
        T_ij[(link_trans[i]['start'],link_trans[i]['end'])] = link_trans[i]['travel cost']
    for i in link_MOD:
        T_ij[(link_MOD[i]['start'],link_MOD[i]['end'])] = link_MOD[i]['travel cost']
    for i in link_Acce:
        fs = h[MODoper.index(link_Acce[i]['operator'])]
    #     print(link_Acce[i]['start'],link_Acce[i]['end'], fs)
        flow = sum(x[link_Acce[i]['start'],link_Acce[i]['end'],ddd] for ddd in range(len(d))) 
        if fs == 0:
            T_ij[(link_Acce[i]['start'],link_Acce[i]['end'])]  = 9999999999999
        else:
            T_ij[(link_Acce[i]['start'],link_Acce[i]['end'])] = t_coef[link_Acce[i]['operator']]*(fs)**(-2)*(flow+1)**1
    for i in link_Egre:
        T_ij[(link_Egre[i]['start'],link_Egre[i]['end'])] = 0

    ### revise link: remove closed MOD nodes, links, and transfer links
    MOD_node_closed = []
    for i in range(len(MODnode)):
        if v[i] == 0:
            MOD_node_closed.append(MODnode[i])
    link_used = copy.deepcopy(link)
    for i in link:
        start_nodee = link_used[i]['start']
        end_nodee = link_used[i]['end']
        if (start_nodee in MOD_node_closed) or (end_nodee in MOD_node_closed):
            del link_used[i]
        if i in dummy_link:
            del link_used[i]

    for s in range(len(d)):
        for r in stability_path_set[s]:
            core = utility[s]
            for linkk in r:
                core-=T_ij[link[linkk]['start'],link[linkk]['end']]
                core-=dual[link[linkk]['start'],link[linkk]['end']]
                core-=C_ij[link[linkk]['start'],link[linkk]['end']]
    #             if linkk in used_link_set:
    #                 UN.append(c[linkk,link[linkk]['operator']])
    #         rhs+=sum(list(set(UN)))
            #adding stability constraints
        #     path_index = new_routes[s].index(j)
            m1.addConstr(c[s]+u[s]>=core, 
                         name = 'stability_'+str(r))
            m1.update()


    ################## Total Subsidies ##################
    obj = 0
    count = -1
    for s in range(len(d)):
        for path in new_routes[s]:
            count += 1
            path_index = new_routes[s].index(path)
            obj += path_flow[count]*S_path[s,path_index]

    m1.setObjective(obj,GRB.MINIMIZE)
    m1.update()
    m1.display()
    m1.params.outputflag=0
    m1.optimize()

    if m1.status == 2:        
        subsidy_feasi = 1        
        subsidy_obj = m1.getObjective()
        subsidy_obj = subsidy_obj.getValue()
        for ii in S_path:
            S_path[ii] = S_path[ii].x
    else:
        subsidy_feasi = 0
        subsidy_obj = []
        S_path = []

    return subsidy_feasi, subsidy_obj, S_path



###################################################################################################
###################################################################################################
###################################################################################################


def cost_allo_sub(stability_path_set, x_sol, path_set, d, out_of_sys, utility, MODoper, MODnode, MODnode_oper, t_coef, c_coef, oper_list, flows_link, operators, link, link_fixed, link_trans, link_MOD, link_Acce, link_Egre, dummy_link, link_from_to, path_flow, a_MODlink, a_fixedlink, a_translink, new_routes, V_coef, MODoper_Acce, orig, dest, user_paths, operators_in_path, nodes, dual, y, v, h, tt, S_path):

    x = x_sol

    used_link_set = []
    for iii in path_set:
        for jjj in iii:
            used_link_set = used_link_set+jjj
    used_link_set = list(np.unique(used_link_set))
    lst = []
    for iii in used_link_set:
        aaa = (iii,link[iii]["operator"])
        lst.append(aaa)

    m1 = Model()
    # Model variables
    c = m1.addVars(range(len(d)),lb = 0,vtype = GRB.CONTINUOUS, name = 'price')
    u = m1.addVars(range(len(d)),lb = 0,ub=utility, vtype = GRB.CONTINUOUS, name = 'u')    
    m1.update()
    
    # (5a) profitability
    infra_cost = dict()
    for operr in MODoper:
        infra_cost[operr] = 0
    for nodee in MODnode_oper:
        operr = MODnode_oper[nodee]
        infra_cost[operr] = infra_cost[operr] + v[MODnode.index(nodee)]*V_coef[nodee]

    MODoper_C = dict()
    for i in MODoper:
        link_ = MODoper_Acce[i]
        demand = 0
        for ddd in range(len(d)):
            for l in link_:
                demand += x_sol[l[0],l[1],ddd]
        MODoper_C[i] = c_coef[i]*demand*(h[MODoper.index(i)]**2) + infra_cost[i]

    ttl_oper_cost = 0

    oper_list_ = oper_list.copy()  
    oper_list_.remove(0)

    for oper in oper_list_:
        if oper not in MODoper: # fixed operator       
            ttl_oper_cost += sum(link[i]['operating cost'] for i in operators[oper])  
        else: # MOD operator
            ooo = intersection(operators[oper], list(a_MODlink))
            ttl_oper_cost += MODoper_C[oper]

    flow_s = list(np.array(d) - np.array(out_of_sys))

    m1.addConstr(quicksum(c[s]*flow_s[s] for s in range(len(d))) >= ttl_oper_cost, name = 'Profitability')
    m1.update()


    # (5b) feasibility (Requiring same payoff for user paths of the same O-D)
    count =- 1
    for s in range(len(d)):
        for path in new_routes[s]:
            count += 1
            path_index = new_routes[s].index(path)
            rhs = utility[s] + S_path[s,path_index]
            lhs = 0
            ttl_price = 0
            path_edges = list(zip(path,path[1:]))
            for edge in path_edges:
                rhs -= tt[edge[0],edge[1],s]     
            path_index = new_routes[s].index(path)
            m1.addConstr(u[s]+ c[s]== rhs, name = 'cost_allo_'+str(s)+'_'+str(path_index))    
    m1.update()

    # (5c) stability
    capacity = dict()
    cost = dict()

    #Users on link ij
    serves = dict()
    for i in link:
        if i not in dummy_link:
            io = 0
            for s in range(len(d)):
                if x[link[i]['start'],link[i]['end'],s]>0 and io == 0:
                    serves[(link[i]['start'],link[i]['end'])] = [s]
                    io=1
                elif x[link[i]['start'],link[i]['end'],s]>0 and io == 1:
                    serves[(link[i]['start'],link[i]['end'])].append(s)
    for i in link:
        if i not in dummy_link:
            capacity[(link[i]['start'],link[i]['end'])] = link[i]['capacity']
            cost[(link[i]['start'],link[i]['end'])] = link[i]['operating cost']
            if (link[i]['start'],link[i]['end']) not in serves:
                serves[(link[i]['start'],link[i]['end'])] = {} 

    C_ij = dict()
    for i in a_fixedlink:
        if  serves[(link_fixed[i]['start'],link_fixed[i]['end'])] == {}:
            C_ij[(link_fixed[i]['start'],link_fixed[i]['end'])] = link_fixed[i]['operating cost']
        else:
            C_ij[(link_fixed[i]['start'],link_fixed[i]['end'])] = 0
    for i in a_translink:
        C_ij[(link_trans[i]['start'],link_trans[i]['end'])] = 0

    for i in link_MOD:
        fs = h[MODoper.index(link_MOD[i]['operator'])]

        if fs > 0: # the operator is operating
            link_ = MODoper_Acce[link_MOD[i]['operator']]
            demand = 0
            for ddd in range(len(d)):
                for l in link_:
                    demand += x[l[0],l[1],ddd]
            C_ij[(link_MOD[i]['start'],link_MOD[i]['end'])] = c_coef[link_MOD[i]['operator']]*(fs**2)*((demand+1)**1) - c_coef[link_MOD[i]['operator']]*(fs**2)*((demand)**1)
            if v[MODnode.index(link_MOD[i]['start'])] == 0:
                C_ij[(link_MOD[i]['start'],link_MOD[i]['end'])] = C_ij[(link_MOD[i]['start'],link_MOD[i]['end'])]+V_coef[link_MOD[i]['start']]
            if v[MODnode.index(link_MOD[i]['end'])] == 0:
                C_ij[(link_MOD[i]['start'],link_MOD[i]['end'])] = C_ij[(link_MOD[i]['start'],link_MOD[i]['end'])]+V_coef[link_MOD[i]['end']]
        else: # the operator is not operating
            C_ij[(link_MOD[i]['start'],link_MOD[i]['end'])] = c_coef[link_MOD[i]['operator']]*(1**2)*(1**1)
            if v[MODnode.index(link_MOD[i]['start'])] == 0:
                C_ij[(link_MOD[i]['start'],link_MOD[i]['end'])] = C_ij[(link_MOD[i]['start'],link_MOD[i]['end'])]+V_coef[link_MOD[i]['start']]
            if v[MODnode.index(link_MOD[i]['end'])] == 0:
                C_ij[(link_MOD[i]['start'],link_MOD[i]['end'])] = C_ij[(link_MOD[i]['start'],link_MOD[i]['end'])]+V_coef[link_MOD[i]['end']]

    for i in link_Acce:
        C_ij[(link_Acce[i]['start'],link_Acce[i]['end'])] = 0
    for i in link_Egre:
        C_ij[(link_Egre[i]['start'],link_Egre[i]['end'])] = 0    

    T_ij = dict()
    for i in link_fixed:
        T_ij[(link_fixed[i]['start'],link_fixed[i]['end'])] = link_fixed[i]['travel cost']
    for i in link_trans:
        T_ij[(link_trans[i]['start'],link_trans[i]['end'])] = link_trans[i]['travel cost']
    for i in link_MOD:
        T_ij[(link_MOD[i]['start'],link_MOD[i]['end'])] = link_MOD[i]['travel cost']
    for i in link_Acce:
        fs = h[MODoper.index(link_Acce[i]['operator'])]
    #     print(link_Acce[i]['start'],link_Acce[i]['end'], fs)
        flow = sum(x[link_Acce[i]['start'],link_Acce[i]['end'],ddd] for ddd in range(len(d))) 
        if fs == 0:
            T_ij[(link_Acce[i]['start'],link_Acce[i]['end'])]  = 9999999999999
        else:
            T_ij[(link_Acce[i]['start'],link_Acce[i]['end'])] = t_coef[link_Acce[i]['operator']]*(fs)**(-2)*(flow+1)**1
    for i in link_Egre:
        T_ij[(link_Egre[i]['start'],link_Egre[i]['end'])] = 0

    ### revise link: remove closed MOD nodes, links, and transfer links
    MOD_node_closed = []
    for i in range(len(MODnode)):
        if v[i] == 0:
            MOD_node_closed.append(MODnode[i])
    link_used = copy.deepcopy(link)
    for i in link:
        start_nodee = link_used[i]['start']
        end_nodee = link_used[i]['end']
        if (start_nodee in MOD_node_closed) or (end_nodee in MOD_node_closed):
            del link_used[i]
        if i in dummy_link:
            del link_used[i]


    for s in range(len(d)):
        for r in stability_path_set[s]:
            core = utility[s]
            for linkk in r:
                core-=T_ij[link[linkk]['start'],link[linkk]['end']]
                core-=dual[link[linkk]['start'],link[linkk]['end']]
                core-=C_ij[link[linkk]['start'],link[linkk]['end']]
    #             if linkk in used_link_set:
    #                 UN.append(c[linkk,link[linkk]['operator']])
    #         rhs+=sum(list(set(UN)))
            #adding stability constraints
        #     path_index = new_routes[s].index(j)
            m1.addConstr(c[s]+u[s]>=core, 
                         name = 'stability_'+str(r))
            m1.update()

            
    # ### User optimal
    obj = 0
    obj += quicksum(u[s]*(d[s]-out_of_sys[s]) for s in range(len(d))) 
    m1.setObjective(obj,GRB.MAXIMIZE)
    m1.update()
    m1.params.outputflag=0
    m1.optimize()
    x_ = m1.getAttr('X', m1.getVars())
    prices = x_[0:len(d)]
    p_user_opt=dict()
    for s in range(len(d)):
        p_user_opt[s]=prices[s]
    u_s_user_opt = x_[len(d) : len(d)+len(d)]
    u_user_opt = dict()
    for var in u:
        u_user_opt[var]=u[var].x

        
    # ### Operator optimal
    obj = 0
    obj += quicksum(c[s]*(d[s]-out_of_sys[s]) for s in range(len(d)))
    m1.setObjective(obj,GRB.MAXIMIZE)
    m1.update()
    m1.params.outputflag=0
    m1.optimize()
    x_ = m1.getAttr('X', m1.getVars())
    prices = x_[0:len(d)]
    p_oper_opt=dict()
    for s in range(len(d)):
        p_oper_opt[s]=prices[s]
    u_s_oper_opt = x_[len(d) : len(d)+len(d)]
    u_oper_opt = dict()
    for var in u:
        u_oper_opt[var]=u[var].x

    return p_user_opt, u_s_user_opt, u_user_opt, p_oper_opt, u_s_oper_opt, u_oper_opt



###################################################################################################
###################################################################################################
###################################################################################################




