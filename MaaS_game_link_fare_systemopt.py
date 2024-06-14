import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gurobipy import *
import warnings
warnings.filterwarnings('ignore')
import copy
import sympy as sp
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
import scipy.optimize
import time
import timeit
import numpy as np
import ctypes
import copy
from Cost_allo_sub_link_systemopt import pre_process_intsol, cost_allo_prep, cost_allo, subsidy_stablize, cost_allo_sub



from ctypes import cdll
lib = cdll.LoadLibrary('./foo.so')

cpp_dijkstra = lib.cpp_dijkstra

# _doublepp = np.ctypeslib.ndpointer(dtype=np.uintp, ndim=1, flags='C') 
cpp_dijkstra.argtypes = [
    np.ctypeslib.ndpointer(dtype=ctypes.c_float, ndim=1),
    ctypes.c_int,
    np.ctypeslib.ndpointer(dtype=ctypes.c_float, ndim=1),
    np.ctypeslib.ndpointer(dtype=ctypes.c_int, ndim=1),
    ctypes.c_int
]


####################################################################################################
####################################################################################################
####################################################################################################



def b_and_b(PRINT, num_iter, tolerance, FW_tol, FW_conver_times, t_coef, c_coef, d, orig, dest, cap, b, oper_cost, V_coef, utility, link, link_fixed, link_trans, link_Acce, link_MOD, link_Egre, link_from_to, a_alllink, a_MODlink, a_fixedlink, a_translink, transin_oper, MOD_nodes, nodes, MODnode_oper, MODnode_ori, MODoper_node_ori, MODnode, MODoper, MODoper_Acce, outflow_link, inflow_link, dummy_link, oper_list):
   
    ttl_cost_best = np.inf
    node_count = 0

    ## branch & bound
    v_h_close = dict()
    for nodee in MOD_nodes:
        v_h_close[nodee] = []
    for nodee in MOD_nodes:
        h_nodee = nodee%10
        oper_nodee = MODnode_oper[nodee]
        for nodeee in MOD_nodes:
            h_nodeee = nodeee%10
            oper_nodeee = MODnode_oper[nodeee]
            if h_nodeee != h_nodee and oper_nodee == oper_nodeee:
                v_h_close[nodee].append(nodeee)   

    ## solve the root node
    y_fixed_1 = []
    y_fixed_0 = []
    v_fixed_1 = []
    v_fixed_0 = []
    node_count +=1
    print("Root node solving......(node", node_count,")")
    start_t = time.time()
    link_b,link_fixed_b,link_Acce_b,link_MOD_b,link_Egre_b,transin_oper_b,MOD_nodes_b,nodes_b,outflow_link_b,inflow_link_b,a_alllink_b,a_fixedlink_b,a_translink_b,a_MODlink_b,a_transinlink_b,a_transoutlink_b,link_node_b,N_b,link_from_to_b = network_branch(y_fixed_0, v_fixed_0, d, link, link_fixed, link_trans, link_Acce, link_MOD, link_Egre, transin_oper, MOD_nodes, nodes, outflow_link, inflow_link, dummy_link)
    flows_link, flows_path, y_sol, v_sol, path_set, dual_p = single_branch(PRINT,y_fixed_1, y_fixed_0, v_fixed_1, v_fixed_0, num_iter, tolerance,FW_tol,FW_conver_times,t_coef, c_coef,a_alllink_b,a_fixedlink_b,a_translink_b,a_MODlink_b,a_transinlink_b,a_transoutlink_b,cap,b,oper_cost,link_b,link, link_fixed_b,link_fixed, link_trans,link_MOD_b,link_Acce_b,link_Egre_b,dummy_link,orig,dest,d,nodes_b,MOD_nodes_b,MOD_nodes, transin_oper_b,outflow_link_b,inflow_link_b,V_coef, utility,N_b,link_from_to_b,link_node_b)
    obj_yv_value = obj_yv(t_coef, c_coef, flows_link, y_sol, v_sol, d, a_alllink_b, link, link_fixed, link_trans, link_MOD, dummy_link, MOD_nodes, transin_oper, V_coef, utility)
    end_t = time.time()
    print("Root node solved. Obj =",obj_yv_value, "Time = ", np.round((end_t-start_t)/60), "min", (end_t-start_t)%60, "sec") 

    # save the node in the queue
    flows_link_queue = [flows_link]
    flows_path_queue = [flows_path]
    y_sol_queue = [y_sol]
    v_sol_queue = [v_sol]
    obj_yv_queue = [obj_yv_value]
    y_bran_1_queue = [[]]
    y_bran_0_queue = [[]]
    v_bran_1_queue = [[]]
    v_bran_0_queue = [[]]


    for ite in range(9999999999999):
        # select the first node in queue to branch from
        y_sol = y_sol_queue[0]
        v_sol = v_sol_queue[0]
        print("Node selected: ")
        print("y:", y_sol)
        print("v:", v_sol)

        # select y/v to branch (for one node, fixed)
        y_selected = 0
        v_selected = 0
        for links in link_fixed:
            link_ind = a_fixedlink.index(links)
            if y_sol[link_ind] %1 != 0: # NOT integer
                branch = links
                y_selected = 1
                print('Fixed link selected to branch:', branch, (y_sol[link_ind]))
                break
        if y_selected == 0:
            for nodee in list(MOD_nodes):
                node_ind = list(MOD_nodes).index(nodee)
                if v_sol[node_ind] %1 != 0:  # NOT integer
                    branch = nodee
                    v_selected = 1
                    print('MOD node selected to branch:', branch, (v_sol[node_ind]))
                    break

        # new branch sets    
        if y_selected == 1:#branch from y
            ## branch node 1 (1)
            y_fixed_1 = list(np.sort(np.unique(y_bran_1_queue[0]+[branch])))
            y_fixed_0 = y_bran_0_queue[0]
            v_fixed_1 = v_bran_1_queue[0]
            v_fixed_0 = v_bran_0_queue[0]
            print('----- Branch 1 (y - 1) -----')
            print('y_1:',y_fixed_1)
            print('y_0:',y_fixed_0)
            print('v_1:',v_fixed_1)
            print('v_0:',v_fixed_0)

            # check if it's in the queue
            solve_branch = 1
            if y_fixed_1 in y_bran_1_queue:
                queue_ind = y_bran_1_queue.index(y_fixed_1)
                if y_bran_0_queue[queue_ind] == y_fixed_0 and v_bran_1_queue[queue_ind] == v_fixed_1 and v_bran_0_queue[queue_ind] == v_fixed_0:
                    print("Branch already in queue!")
                    solve_branch = 0

            if solve_branch == 1:
                # solve the node
                node_count +=1
                print("Branch solving...... (node", node_count,")")
                start_t = time.time()
                link_b,link_fixed_b,link_Acce_b,link_MOD_b,link_Egre_b,transin_oper_b,MOD_nodes_b,nodes_b,outflow_link_b,inflow_link_b,a_alllink_b,a_fixedlink_b,a_translink_b,a_MODlink_b,a_transinlink_b,a_transoutlink_b,link_node_b,N_b,link_from_to_b= network_branch(y_fixed_0, v_fixed_0, d, link, link_fixed, link_trans, link_Acce, link_MOD, link_Egre, transin_oper, MOD_nodes, nodes, outflow_link, inflow_link, dummy_link)
                flows_link, flows_path, y_sol, v_sol, path_set, dual_p = single_branch(PRINT,y_fixed_1, y_fixed_0, v_fixed_1, v_fixed_0, num_iter, tolerance,FW_tol,FW_conver_times,t_coef, c_coef,a_alllink_b,a_fixedlink_b,a_translink_b,a_MODlink_b,a_transinlink_b,a_transoutlink_b,cap,b,oper_cost,link_b,link, link_fixed_b,link_fixed, link_trans,link_MOD_b,link_Acce_b,link_Egre_b,dummy_link,orig,dest,d,nodes_b,MOD_nodes_b,MOD_nodes, transin_oper_b,outflow_link_b,inflow_link_b,V_coef, utility,N_b,link_from_to_b,link_node_b)
                obj_yv_value = obj_yv(t_coef, c_coef, flows_link, y_sol, v_sol, d, a_alllink_b, link, link_fixed, link_trans, link_MOD, dummy_link, MOD_nodes, transin_oper, V_coef, utility)
                end_t = time.time()
                print("Branch solved. obj_yv_value = ", obj_yv_value, "Time = ", np.round((end_t-start_t)/60), "min", (end_t-start_t)%60, "sec")
                # check if it's integer solution
                integer_sol = 1
                for iii in y_sol:
                    if iii%1 != 0:
                        integer_sol = 0
                        break
                if integer_sol == 1:
                    for iii in v_sol:
                        if iii%1 != 0:
                            integer_sol = 0
                            break       
                if integer_sol == 0 and obj_yv_value < ttl_cost_best: #not integer sol, save in the queue
                    print("NOT INTEGER SOL & better than current best! saved in queue")
                    print('obj_yv_value = ',obj_yv_value,'ttl_cost_best = ', ttl_cost_best)
                    flows_link_queue.append(flows_link)
                    flows_path_queue.append(flows_path)
                    y_sol_queue.append(y_sol)
                    v_sol_queue.append(v_sol)
                    obj_yv_queue.append(obj_yv_value)
                    y_bran_1_queue.append(y_fixed_1)
                    y_bran_0_queue.append(y_fixed_0)
                    v_bran_1_queue.append(v_fixed_1)
                    v_bran_0_queue.append(v_fixed_0)
                elif integer_sol == 1: # integer solution
                    # cost allocation
                    path_set, flows_path, dual, y, v, h_sol_MOD, h_sol, h, x_sol, x_sol_, out_of_sys = pre_process_intsol(flows_link, flows_path, y_sol, v_sol, path_set, dual_p, link, link_fixed, MODnode_ori, MODoper_node_ori, MODnode, MODoper, dummy_link, d)
                    tt, cst, volume, users, operated, route, operators, path_set_node, new_routes, oper, user_paths, path_flow, belong, count, lst, link_to_path, operators_in_path, operators_vol_path, path_hash = cost_allo_prep(path_set, flows_path, dual, y, v, h_sol_MOD, h_sol, h, x_sol, x_sol_, out_of_sys,link, dummy_link, d, link_Acce, MODoper, oper_list, t_coef, c_coef)
                    feasi, stability_path_set, p_user_opt, u_s_user_opt, u_user_opt, p_oper_opt, u_s_oper_opt, u_oper_opt = cost_allo(x_sol, path_set, d, out_of_sys, utility, MODoper, MODnode, MODnode_oper, t_coef, c_coef, oper_list, flows_link, operators, link, link_fixed, link_trans, link_MOD, link_Acce, link_Egre, dummy_link, link_from_to, a_MODlink, a_fixedlink, a_translink, new_routes, V_coef, MODoper_Acce, orig, dest, user_paths, operators_in_path, nodes, dual, y, v, h, tt)
                    if feasi == 0: #if unstable
                        subsidy_feasi, subsidy_obj, S_path = subsidy_stablize(stability_path_set, x_sol, path_set, d, out_of_sys, utility, MODoper, MODnode, MODnode_oper, t_coef, c_coef, oper_list, flows_link, operators, link, link_fixed, link_trans, link_MOD, link_Acce, link_Egre, dummy_link, link_from_to, path_flow, a_MODlink, a_fixedlink, a_translink, new_routes, V_coef, MODoper_Acce, orig, dest, user_paths, operators_in_path, nodes, dual, y, v, h, tt)
                        if subsidy_feasi == 1:
                            p_user_opt, u_s_user_opt, u_user_opt, p_oper_opt, u_s_oper_opt, u_oper_opt = cost_allo_sub(stability_path_set, x_sol, path_set, d, out_of_sys, utility, MODoper, MODnode, MODnode_oper,t_coef, c_coef, oper_list, flows_link, operators, link, link_fixed, link_trans, link_MOD, link_Acce, link_Egre, dummy_link, link_from_to, path_flow, a_MODlink, a_fixedlink, a_translink, new_routes, V_coef, MODoper_Acce, orig, dest, user_paths, operators_in_path, nodes, dual, y, v, h, tt, S_path)
                            ttl_cost = subsidy_obj + obj_yv_value
                            print("UNSTABLE INTEGER SOL! obj = ", obj_yv_value, "subsidy = ", subsidy_obj)
                        else:
                            ttl_cost = 9999999999
                    else:
                        ttl_cost = obj_yv_value
                        subsidy_obj = 0
                        S_path = []
                        print("STABLE INTEGER SOL! obj = ", obj_yv_value, "subsidy = ", subsidy_obj)
                        
                    if ttl_cost < ttl_cost_best: # better than current best
                        print("INTEGER SOL & better than current best! replace current best")
                        print('obj_yv_value = ',obj_yv_value,'ttl_cost_best = ', ttl_cost_best)
                        flows_link_best = flows_link
                        flows_path_best = flows_path
                        y_best = y_sol
                        v_best = v_sol
                        obj_best = obj_yv_value
                        ttl_cost_best = ttl_cost
                        subsidy_obj_best = subsidy_obj                        
                        dual_p_best = dual_p
                        path_set_best = path_set
                        
                        p_user_opt_best = p_user_opt
                        u_user_opt_best = u_user_opt
                        p_oper_opt_best = p_oper_opt
                        u_oper_opt_best = u_oper_opt
                        S_path_best = S_path
                                               
                        y_fixed_1_best = y_fixed_1
                        y_bran_0_best = y_fixed_0
                        v_bran_1_best = v_fixed_1
                        v_bran_0_best = v_fixed_0

            ## branch node 2 (0)
            y_fixed_1 = y_bran_1_queue[0]
            y_fixed_0 = list(np.sort(np.unique(y_bran_0_queue[0]+[branch])))
            v_fixed_1 = v_bran_1_queue[0]
            v_fixed_0 = v_bran_0_queue[0]
            print('----- Branch 2 (y - 0) -----')
            print('y_1:',y_fixed_1)
            print('y_0:',y_fixed_0)
            print('v_1:',v_fixed_1)
            print('v_0:',v_fixed_0)        

            # check if it's in the queue
            solve_branch = 1
            if y_fixed_1 in y_bran_1_queue:
                queue_ind = y_bran_1_queue.index(y_fixed_1)
                if y_bran_0_queue[queue_ind] == y_fixed_0 and v_bran_1_queue[queue_ind] == v_fixed_1 and v_bran_0_queue[queue_ind] == v_fixed_0:
                    print("Branch already in queue!")
                    solve_branch = 0

            if solve_branch == 1:        
                # solve the node
                node_count +=1
                print("Branch solving......(node", node_count,")")
                start_t = time.time()
                link_b,link_fixed_b,link_Acce_b,link_MOD_b,link_Egre_b,transin_oper_b,MOD_nodes_b,nodes_b,outflow_link_b,inflow_link_b,a_alllink_b,a_fixedlink_b,a_translink_b,a_MODlink_b,a_transinlink_b,a_transoutlink_b,link_node_b,N_b,link_from_to_b= network_branch(y_fixed_0, v_fixed_0, d, link, link_fixed, link_trans, link_Acce, link_MOD, link_Egre, transin_oper, MOD_nodes, nodes, outflow_link, inflow_link, dummy_link)
                flows_link, flows_path, y_sol, v_sol, path_set, dual_p = single_branch(PRINT,y_fixed_1, y_fixed_0, v_fixed_1, v_fixed_0, num_iter, tolerance,FW_tol,FW_conver_times,t_coef, c_coef,a_alllink_b,a_fixedlink_b,a_translink_b,a_MODlink_b,a_transinlink_b,a_transoutlink_b,cap,b,oper_cost,link_b,link, link_fixed_b,link_fixed, link_trans,link_MOD_b,link_Acce_b,link_Egre_b,dummy_link,orig,dest,d,nodes_b,MOD_nodes_b,MOD_nodes, transin_oper_b,outflow_link_b,inflow_link_b,V_coef, utility,N_b,link_from_to_b,link_node_b)
                obj_yv_value = obj_yv(t_coef, c_coef, flows_link, y_sol, v_sol, d, a_alllink_b, link, link_fixed, link_trans, link_MOD, dummy_link, MOD_nodes, transin_oper, V_coef, utility)
                end_t = time.time()
                print("Branch solved. obj_yv_value = ", obj_yv_value, "Time = ", np.round((end_t-start_t)/60), "min", (end_t-start_t)%60, "sec")

                # check if it's integer solution
                integer_sol = 1
                for iii in y_sol:
                    if iii%1 != 0:
                        integer_sol = 0
                        break
                if integer_sol == 1:
                    for iii in v_sol:
                        if iii%1 != 0:
                            integer_sol = 0
                            break                
                if integer_sol == 0 and obj_yv_value < ttl_cost_best: #not integer sol, save in the queue
                    print("NOT INTEGER SOL & better than current best! saved in queue")
                    print('obj_yv_value = ',obj_yv_value,'ttl_cost_best = ', ttl_cost_best)
                    flows_link_queue.append(flows_link)
                    flows_path_queue.append(flows_path)
                    y_sol_queue.append(y_sol)
                    v_sol_queue.append(v_sol)
                    obj_yv_queue.append(obj_yv_value)
                    y_bran_1_queue.append(y_fixed_1)
                    y_bran_0_queue.append(y_fixed_0)
                    v_bran_1_queue.append(v_fixed_1)
                    v_bran_0_queue.append(v_fixed_0)
                elif integer_sol == 1: # integer solution
                    # cost allocation
                    path_set, flows_path, dual, y, v, h_sol_MOD, h_sol, h, x_sol, x_sol_, out_of_sys = pre_process_intsol(flows_link, flows_path, y_sol, v_sol, path_set, dual_p, link, link_fixed, MODnode_ori, MODoper_node_ori, MODnode, MODoper, dummy_link, d)
                    tt, cst, volume, users, operated, route, operators, path_set_node, new_routes, oper, user_paths, path_flow, belong, count, lst, link_to_path, operators_in_path, operators_vol_path, path_hash = cost_allo_prep(path_set, flows_path, dual, y, v, h_sol_MOD, h_sol, h, x_sol, x_sol_, out_of_sys,link, dummy_link, d, link_Acce, MODoper, oper_list, t_coef, c_coef)
                    feasi, stability_path_set, p_user_opt, u_s_user_opt, u_user_opt, p_oper_opt, u_s_oper_opt, u_oper_opt = cost_allo(x_sol, path_set, d, out_of_sys, utility, MODoper, MODnode, MODnode_oper, t_coef, c_coef, oper_list, flows_link, operators, link, link_fixed, link_trans, link_MOD, link_Acce, link_Egre, dummy_link, link_from_to, a_MODlink, a_fixedlink, a_translink, new_routes, V_coef, MODoper_Acce, orig, dest, user_paths, operators_in_path, nodes, dual, y, v, h, tt)
                    if feasi == 0: #if unstable
                        subsidy_feasi, subsidy_obj, S_path = subsidy_stablize(stability_path_set, x_sol, path_set, d, out_of_sys, utility, MODoper, MODnode, MODnode_oper, t_coef, c_coef, oper_list, flows_link, operators, link, link_fixed, link_trans, link_MOD, link_Acce, link_Egre, dummy_link, link_from_to, path_flow, a_MODlink, a_fixedlink, a_translink, new_routes, V_coef, MODoper_Acce, orig, dest, user_paths, operators_in_path, nodes, dual, y, v, h, tt) 
                        if subsidy_feasi == 1:
                            p_user_opt, u_s_user_opt, u_user_opt, p_oper_opt, u_s_oper_opt, u_oper_opt = cost_allo_sub(stability_path_set, x_sol, path_set, d, out_of_sys, utility, MODoper, MODnode, MODnode_oper,t_coef, c_coef, oper_list, flows_link, operators, link, link_fixed, link_trans, link_MOD, link_Acce, link_Egre, dummy_link, link_from_to, path_flow, a_MODlink, a_fixedlink, a_translink, new_routes, V_coef, MODoper_Acce, orig, dest, user_paths, operators_in_path, nodes, dual, y, v, h, tt, S_path)
                            ttl_cost = subsidy_obj + obj_yv_value
                            print("UNSTABLE INTEGER SOL! obj = ", obj_yv_value, "subsidy = ", subsidy_obj)
                        else:
                            ttl_cost = 9999999999
                    else:
                        ttl_cost = obj_yv_value
                        subsidy_obj = 0
                        S_path = []
                        print("STABLE INTEGER SOL! obj = ", obj_yv_value, "subsidy = ", subsidy_obj)
                        
                    if ttl_cost < ttl_cost_best: # better than current best
                        print("INTEGER SOL & better than current best! replace current best")
                        print('obj_yv_value = ',obj_yv_value,'ttl_cost_best = ', ttl_cost_best)
                        flows_link_best = flows_link
                        flows_path_best = flows_path
                        y_best = y_sol
                        v_best = v_sol
                        obj_best = obj_yv_value
                        ttl_cost_best = ttl_cost
                        subsidy_obj_best = subsidy_obj                        
                        dual_p_best = dual_p
                        path_set_best = path_set
                        
                        p_user_opt_best = p_user_opt
                        u_user_opt_best = u_user_opt
                        p_oper_opt_best = p_oper_opt
                        u_oper_opt_best = u_oper_opt
                        S_path_best = S_path
                         
                        y_fixed_1_best = y_fixed_1
                        y_bran_0_best = y_fixed_0
                        v_bran_1_best = v_fixed_1
                        v_bran_0_best = v_fixed_0


        else: #branch from v
            ## branch node 1 (1)
            branch_1 = [branch]
            branch_0 = v_h_close[branch]
            y_fixed_1 = y_bran_1_queue[0]
            y_fixed_0 = y_bran_0_queue[0]
            v_fixed_1 = list(np.sort(np.unique(v_bran_1_queue[0]+branch_1)))
            v_fixed_0 = list(np.sort(np.unique(v_bran_0_queue[0]+branch_0)))
            print('----- Branch 1 (v - 1) -----')
            print('y_1:',y_fixed_1)
            print('y_0:',y_fixed_0)
            print('v_1:',v_fixed_1)
            print('v_0:',v_fixed_0)

            # check if it's in the queue
            solve_branch = 1
            if y_fixed_1 in y_bran_1_queue:
                queue_ind = y_bran_1_queue.index(y_fixed_1)
                if y_bran_0_queue[queue_ind] == y_fixed_0 and v_bran_1_queue[queue_ind] == v_fixed_1 and v_bran_0_queue[queue_ind] == v_fixed_0:
                    print("Branch already in queue!")
                    solve_branch = 0

            if solve_branch == 1:
                # solve the branch
                node_count +=1
                print("Branch solving......(node", node_count,")")
                start_t = time.time()
                link_b,link_fixed_b,link_Acce_b,link_MOD_b,link_Egre_b,transin_oper_b,MOD_nodes_b,nodes_b,outflow_link_b,inflow_link_b,a_alllink_b,a_fixedlink_b,a_translink_b,a_MODlink_b,a_transinlink_b,a_transoutlink_b,link_node_b,N_b,link_from_to_b= network_branch(y_fixed_0, v_fixed_0, d, link, link_fixed, link_trans, link_Acce, link_MOD, link_Egre, transin_oper, MOD_nodes, nodes, outflow_link, inflow_link, dummy_link)
                flows_link, flows_path, y_sol, v_sol, path_set, dual_p = single_branch(PRINT,y_fixed_1, y_fixed_0, v_fixed_1, v_fixed_0, num_iter, tolerance,FW_tol,FW_conver_times,t_coef, c_coef,a_alllink_b,a_fixedlink_b,a_translink_b,a_MODlink_b,a_transinlink_b,a_transoutlink_b,cap,b,oper_cost,link_b,link, link_fixed_b,link_fixed, link_trans,link_MOD_b,link_Acce_b,link_Egre_b,dummy_link,orig,dest,d,nodes_b,MOD_nodes_b,MOD_nodes, transin_oper_b,outflow_link_b,inflow_link_b,V_coef, utility,N_b,link_from_to_b,link_node_b)
                obj_yv_value = obj_yv(t_coef, c_coef, flows_link, y_sol, v_sol, d, a_alllink_b, link, link_fixed, link_trans, link_MOD, dummy_link, MOD_nodes, transin_oper, V_coef, utility)
                end_t = time.time()
                print("Branch solved. obj_yv_value = ", obj_yv_value, "Time = ", np.round((end_t-start_t)/60), "min", (end_t-start_t)%60, "sec")

                # check if it's integer solution
                integer_sol = 1
                for iii in y_sol:
                    if iii%1 != 0:
                        integer_sol = 0
                        break
                if integer_sol == 1:
                    for iii in v_sol:
                        if iii%1 != 0:
                            integer_sol = 0
                            break               
                if integer_sol == 0 and obj_yv_value < ttl_cost_best: #not integer sol, save in the queue
                    print("NOT INTEGER SOL & better than current best! saved in queue")
                    print('obj_yv_value = ',obj_yv_value,'ttl_cost_best = ', ttl_cost_best)
                    flows_link_queue.append(flows_link)
                    flows_path_queue.append(flows_path)
                    y_sol_queue.append(y_sol)
                    v_sol_queue.append(v_sol)
                    obj_yv_queue.append(obj_yv_value)
                    y_bran_1_queue.append(y_fixed_1)
                    y_bran_0_queue.append(y_fixed_0)
                    v_bran_1_queue.append(v_fixed_1)
                    v_bran_0_queue.append(v_fixed_0)
                elif integer_sol == 1: # integer solution
                    # cost allocation
                    path_set, flows_path, dual, y, v, h_sol_MOD, h_sol, h, x_sol, x_sol_, out_of_sys = pre_process_intsol(flows_link, flows_path, y_sol, v_sol, path_set, dual_p, link, link_fixed, MODnode_ori, MODoper_node_ori, MODnode, MODoper, dummy_link, d)
                    tt, cst, volume, users, operated, route, operators, path_set_node, new_routes, oper, user_paths, path_flow, belong, count, lst, link_to_path, operators_in_path, operators_vol_path, path_hash = cost_allo_prep(path_set, flows_path, dual, y, v, h_sol_MOD, h_sol, h, x_sol, x_sol_, out_of_sys,link, dummy_link, d, link_Acce, MODoper, oper_list, t_coef, c_coef)
                    feasi, stability_path_set, p_user_opt, u_s_user_opt, u_user_opt, p_oper_opt, u_s_oper_opt, u_oper_opt = cost_allo(x_sol, path_set, d, out_of_sys, utility, MODoper, MODnode, MODnode_oper, t_coef, c_coef, oper_list, flows_link, operators, link, link_fixed, link_trans, link_MOD, link_Acce, link_Egre, dummy_link, link_from_to, a_MODlink, a_fixedlink, a_translink, new_routes, V_coef, MODoper_Acce, orig, dest, user_paths, operators_in_path, nodes, dual, y, v, h, tt)
                    if feasi == 0: #if unstable
                        subsidy_feasi, subsidy_obj, S_path = subsidy_stablize(stability_path_set, x_sol, path_set, d, out_of_sys, utility, MODoper, MODnode, MODnode_oper, t_coef, c_coef, oper_list, flows_link, operators, link, link_fixed, link_trans, link_MOD, link_Acce, link_Egre, dummy_link, link_from_to, path_flow, a_MODlink, a_fixedlink, a_translink, new_routes, V_coef, MODoper_Acce, orig, dest, user_paths, operators_in_path, nodes, dual, y, v, h, tt) 
                        if subsidy_feasi == 1:
                            p_user_opt, u_s_user_opt, u_user_opt, p_oper_opt, u_s_oper_opt, u_oper_opt = cost_allo_sub(stability_path_set, x_sol, path_set, d, out_of_sys, utility, MODoper, MODnode, MODnode_oper, t_coef, c_coef, oper_list, flows_link, operators, link, link_fixed, link_trans, link_MOD, link_Acce, link_Egre, dummy_link, link_from_to, path_flow, a_MODlink, a_fixedlink, a_translink, new_routes, V_coef, MODoper_Acce, orig, dest, user_paths, operators_in_path, nodes, dual, y, v, h, tt, S_path)
                            ttl_cost = subsidy_obj + obj_yv_value
                            print("UNSTABLE INTEGER SOL! obj = ", obj_yv_value, "subsidy = ", subsidy_obj)
                        else:
                            ttl_cost = 9999999999
                    else:
                        ttl_cost = obj_yv_value
                        subsidy_obj = 0
                        S_path = []
                        print("STABLE INTEGER SOL! obj = ", obj_yv_value, "subsidy = ", subsidy_obj)
                        
                    if ttl_cost < ttl_cost_best: # better than current best
                        print("better than current best! replace current best")
                        print('obj_yv_value = ',obj_yv_value,'ttl_cost_best = ', ttl_cost_best)
                        flows_link_best = flows_link
                        flows_path_best = flows_path
                        y_best = y_sol
                        v_best = v_sol
                        obj_best = obj_yv_value
                        ttl_cost_best = ttl_cost
                        subsidy_obj_best = subsidy_obj                        
                        dual_p_best = dual_p
                        path_set_best = path_set
                        
                        p_user_opt_best = p_user_opt
                        u_user_opt_best = u_user_opt
                        p_oper_opt_best = p_oper_opt
                        u_oper_opt_best = u_oper_opt
                        S_path_best = S_path
                                                                 
                        y_fixed_1_best = y_fixed_1
                        y_bran_0_best = y_fixed_0
                        v_bran_1_best = v_fixed_1
                        v_bran_0_best = v_fixed_0

            ## branch node 2 (0)
            branch_0 = [branch]
            y_fixed_1 = y_bran_1_queue[0]
            y_fixed_0 = y_bran_0_queue[0]
            v_fixed_1 = v_bran_1_queue[0]
            v_fixed_0 = list(np.sort(np.unique(v_bran_0_queue[0]+branch_0)))
            print('----- Branch 2 (v - 0) -----')
            print('y_1:',y_fixed_1)
            print('y_0:',y_fixed_0)
            print('v_1:',v_fixed_1)
            print('v_0:',v_fixed_0)

            # check if it's in the queue
            solve_branch = 1
            if y_fixed_1 in y_bran_1_queue:
                queue_ind = y_bran_1_queue.index(y_fixed_1)
                if y_bran_0_queue[queue_ind] == y_fixed_0 and v_bran_1_queue[queue_ind] == v_fixed_1 and v_bran_0_queue[queue_ind] == v_fixed_0:
                    print("Branch already in queue!")
                    solve_branch = 0

            if solve_branch == 1:        
                # solve the node
                node_count +=1
                print("Branch solving......(node", node_count,")")
                start_t = time.time()
                link_b,link_fixed_b,link_Acce_b,link_MOD_b,link_Egre_b,transin_oper_b,MOD_nodes_b,nodes_b,outflow_link_b,inflow_link_b,a_alllink_b,a_fixedlink_b,a_translink_b,a_MODlink_b,a_transinlink_b,a_transoutlink_b,link_node_b,N_b,link_from_to_b= network_branch(y_fixed_0, v_fixed_0, d, link, link_fixed, link_trans, link_Acce, link_MOD, link_Egre, transin_oper, MOD_nodes, nodes, outflow_link, inflow_link, dummy_link)
                flows_link, flows_path, y_sol, v_sol, path_set, dual_p = single_branch(PRINT,y_fixed_1, y_fixed_0, v_fixed_1, v_fixed_0, num_iter, tolerance,FW_tol,FW_conver_times,t_coef, c_coef,a_alllink_b,a_fixedlink_b,a_translink_b,a_MODlink_b,a_transinlink_b,a_transoutlink_b,cap,b,oper_cost,link_b,link, link_fixed_b,link_fixed, link_trans,link_MOD_b,link_Acce_b,link_Egre_b,dummy_link,orig,dest,d,nodes_b,MOD_nodes_b,MOD_nodes, transin_oper_b,outflow_link_b,inflow_link_b,V_coef, utility,N_b,link_from_to_b,link_node_b)
                obj_yv_value = obj_yv(t_coef, c_coef, flows_link, y_sol, v_sol, d, a_alllink_b, link, link_fixed, link_trans, link_MOD, dummy_link, MOD_nodes, transin_oper, V_coef, utility)
                end_t = time.time()
                print("Branch solved. obj_yv_value = ", obj_yv_value, "Time = ", np.round((end_t-start_t)/60), "min", (end_t-start_t)%60, "sec")

                # check if it's integer solution
                integer_sol = 1
                for iii in y_sol:
                    if iii%1 != 0:
                        integer_sol = 0
                        break
                if integer_sol == 1:
                    for iii in v_sol:
                        if iii%1 != 0:
                            integer_sol = 0
                            break               
                if integer_sol == 0 and obj_yv_value < ttl_cost_best: #not integer sol, save in the queue
                    print("NOT integer sol & better than current best! saved in queue")
                    print('obj_yv_value = ',obj_yv_value,'ttl_cost_best = ', ttl_cost_best)
                    flows_link_queue.append(flows_link)
                    flows_path_queue.append(flows_path)
                    y_sol_queue.append(y_sol)
                    v_sol_queue.append(v_sol)
                    obj_yv_queue.append(obj_yv_value)
                    y_bran_1_queue.append(y_fixed_1)
                    y_bran_0_queue.append(y_fixed_0)
                    v_bran_1_queue.append(v_fixed_1)
                    v_bran_0_queue.append(v_fixed_0)
                elif integer_sol == 1: # integer solution
                    # cost allocation
                    path_set, flows_path, dual, y, v, h_sol_MOD, h_sol, h, x_sol, x_sol_, out_of_sys = pre_process_intsol(flows_link, flows_path, y_sol, v_sol, path_set, dual_p, link, link_fixed, MODnode_ori, MODoper_node_ori, MODnode, MODoper, dummy_link, d)
                    tt, cst, volume, users, operated, route, operators, path_set_node, new_routes, oper, user_paths, path_flow, belong, count, lst, link_to_path, operators_in_path, operators_vol_path, path_hash = cost_allo_prep(path_set, flows_path, dual, y, v, h_sol_MOD, h_sol, h, x_sol, x_sol_, out_of_sys,link, dummy_link, d, link_Acce, MODoper, oper_list, t_coef, c_coef)
                    feasi, stability_path_set, p_user_opt, u_s_user_opt, u_user_opt, p_oper_opt, u_s_oper_opt, u_oper_opt = cost_allo(x_sol, path_set, d, out_of_sys, utility, MODoper, MODnode, MODnode_oper, t_coef, c_coef, oper_list, flows_link, operators, link, link_fixed, link_trans, link_MOD, link_Acce, link_Egre, dummy_link, link_from_to, a_MODlink, a_fixedlink, a_translink, new_routes, V_coef, MODoper_Acce, orig, dest, user_paths, operators_in_path, nodes, dual, y, v, h, tt)
                    if feasi == 0: #if unstable
                        subsidy_feasi, subsidy_obj, S_path = subsidy_stablize(stability_path_set, x_sol, path_set, d, out_of_sys, utility, MODoper, MODnode, MODnode_oper, t_coef, c_coef, oper_list, flows_link, operators, link, link_fixed, link_trans, link_MOD, link_Acce, link_Egre, dummy_link, link_from_to, path_flow, a_MODlink, a_fixedlink, a_translink, new_routes, V_coef, MODoper_Acce, orig, dest, user_paths, operators_in_path, nodes, dual, y, v, h, tt) 
                        if subsidy_feasi == 1:
                            p_user_opt, u_s_user_opt, u_user_opt, p_oper_opt, u_s_oper_opt, u_oper_opt = cost_allo_sub(stability_path_set, x_sol, path_set, d, out_of_sys, utility, MODoper, MODnode, MODnode_oper, t_coef, c_coef, oper_list, flows_link, operators, link, link_fixed, link_trans, link_MOD, link_Acce, link_Egre, dummy_link, link_from_to, path_flow, a_MODlink, a_fixedlink, a_translink, new_routes, V_coef, MODoper_Acce, orig, dest, user_paths, operators_in_path, nodes, dual, y, v, h, tt, S_path)
                            ttl_cost = subsidy_obj + obj_yv_value
                            print("UNSTABLE INTEGER SOL! obj = ", obj_yv_value, "subsidy = ", subsidy_obj)
                        else:
                            ttl_cost = 9999999999
                    else:
                        ttl_cost = obj_yv_value
                        subsidy_obj = 0
                        S_path = []
                        print("STABLE INTEGER SOL! obj = ", obj_yv_value, "subsidy = ", subsidy_obj)
                        
                    if ttl_cost < ttl_cost_best: # better than current best
                        print("INTEGER SOL & better than current best! replace current best")
                        print('obj_yv_value = ',obj_yv_value,'ttl_cost_best = ', ttl_cost_best)
                        flows_link_best = flows_link
                        flows_path_best = flows_path
                        y_best = y_sol
                        v_best = v_sol
                        obj_best = obj_yv_value
                        ttl_cost_best = ttl_cost
                        subsidy_obj_best = subsidy_obj                        
                        dual_p_best = dual_p
                        path_set_best = path_set
                        
                        p_user_opt_best = p_user_opt
                        u_user_opt_best = u_user_opt
                        p_oper_opt_best = p_oper_opt
                        u_oper_opt_best = u_oper_opt
                        S_path_best = S_path
                                          
                        y_fixed_1_best = y_fixed_1
                        y_bran_0_best = y_fixed_0
                        v_bran_1_best = v_fixed_1
                        v_bran_0_best = v_fixed_0

        # remove the node
        flows_link_queue = flows_link_queue[1:]
        flows_path_queue = flows_path_queue[1:]
        y_sol_queue = y_sol_queue[1:]
        v_sol_queue = v_sol_queue[1:]
        obj_yv_queue = obj_yv_queue[1:]
        y_bran_1_queue = y_bran_1_queue[1:]
        y_bran_0_queue = y_bran_0_queue[1:]
        v_bran_1_queue = v_bran_1_queue[1:]
        v_bran_0_queue = v_bran_0_queue[1:]

        # rank the queues wrt obj_yv_queue from min to max, remove the ones worse than the current best
        flows_link_queue = rank_by_obj(flows_link_queue, obj_yv_queue, ttl_cost_best)
        flows_path_queue = rank_by_obj(flows_path_queue, obj_yv_queue, ttl_cost_best)
        y_sol_queue = rank_by_obj(y_sol_queue, obj_yv_queue, ttl_cost_best)
        v_sol_queue = rank_by_obj(v_sol_queue, obj_yv_queue, ttl_cost_best)
        y_bran_1_queue = rank_by_obj(y_bran_1_queue, obj_yv_queue, ttl_cost_best)
        y_bran_0_queue = rank_by_obj(y_bran_0_queue, obj_yv_queue, ttl_cost_best)
        v_bran_1_queue = rank_by_obj(v_bran_1_queue, obj_yv_queue, ttl_cost_best)
        v_bran_0_queue = rank_by_obj(v_bran_0_queue, obj_yv_queue, ttl_cost_best)
        obj_yv_queue = sorted(obj_yv_queue)
        obj_yv_queue = [i for i in obj_yv_queue if i < ttl_cost_best]


        if len(obj_yv_queue) == 0:
            print('QUEUE EMPTY!!')
            print("Current best (subsidy included)= ", ttl_cost_best)
            print("total number of nodes solved: ",node_count)
            break
        else:
            print('Queue length = ',len(obj_yv_queue))

        print("---------------")


    return flows_link_best, flows_path_best, y_best, v_best, obj_best, ttl_cost_best, subsidy_obj_best, dual_p_best, path_set_best, p_user_opt_best, u_user_opt_best, p_oper_opt_best, u_oper_opt_best, S_path_best

####################################################################################################
####################################################################################################
####################################################################################################

def rank_by_obj(sol, obj, threshold):
    sol_obj = [(i,j) for i,j in zip(sol, obj)]
    sol_obj_sorted = sorted(sol_obj, key=lambda x:x[-1])
    sol_sorted = [i[0] for i in sol_obj_sorted if i[-1]<threshold]
    return sol_sorted


####################################################################################################
####################################################################################################
####################################################################################################


def network_branch(y_fixed_0, v_fixed_0, d, link, link_fixed, link_trans, link_Acce, link_MOD, link_Egre, transin_oper, MOD_nodes, nodes, outflow_link, inflow_link, dummy_link):
    link_b = link.copy()
    link_fixed_b = link_fixed.copy()

    link_Acce_b = link_Acce.copy()
    link_MOD_b = link_MOD.copy()
    link_Egre_b = link_Egre.copy()

    transin_oper_b = transin_oper.copy()
    MOD_nodes_b = MOD_nodes.copy()
    nodes_b = nodes.copy()

    outflow_link_b = outflow_link.copy() #####
    inflow_link_b = inflow_link.copy() #####

    for links in y_fixed_0:
        del link_b[links]
        del link_fixed_b[links]
    for nodee in v_fixed_0:
        MOD_nodes_b = np.delete(MOD_nodes_b, np.argwhere(MOD_nodes_b == nodee))
        nodes_b.pop(nodes_b.index(nodee))
        for links in outflow_link[nodee]:
            if links in link_b:
                del link_b[links]
            if links in link_Acce_b:
                del link_Acce_b[links]
                del transin_oper_b[links]
            if links in link_MOD_b:
                del link_MOD_b[links]
            if links in link_Egre_b:
                del link_Egre_b[links]
        for links in inflow_link[nodee]:
            if links in link_b:
                del link_b[links]
                if links in link_Acce_b:
                    del link_Acce_b[links]
                    del transin_oper_b[links]
                if links in link_MOD_b:
                    del link_MOD_b[links]
                if links in link_Egre_b:
                    del link_Egre_b[links]
                
    # Assignment matching problem
    outflow=dict()
    inflow=dict()
    outflow_link_b=dict()
    inflow_link_b=dict()
    for i in link_b:
        a=link_b[i]['start']
        bb=link_b[i]['end']
        if bb in inflow:
            inflow[bb].append(link_b[i]['start'])
            inflow_link_b[bb].append(i)
        else:
            inflow[bb]=[link_b[i]['start']]
            inflow_link_b[bb]=[i]
        if a in outflow:
            outflow[a].append(link_b[i]['end'])
            outflow_link_b[a].append(i)
        else: 
            outflow[a]=[link_b[i]['end']]
            outflow_link_b[a]=[i]
    for i in nodes_b:
        if i not in inflow:
            inflow[i]={}
            inflow_link_b[i]=[]
        if i not in outflow:
            outflow[i]={}
            outflow_link_b[i]=[]


    a_alllink_b = list(link_b.keys())
    a_fixedlink_b = list(link_fixed_b.keys())
    a_translink_b = list(link_trans.keys())
    a_MODlink_b = list(link_MOD_b.keys())
    a_transinlink_b = list(link_Acce_b.keys())
    a_transoutlink_b = list(link_Egre_b.keys())


    link_node_b = np.zeros((len(link_b)-len(d),4))
    for links in link_b:
        link_id = a_alllink_b.index(links)
        if links not in dummy_link:
            link_row = np.array([int(links), int(link_b[links]['start']), int(link_b[links]['end']), link_b[links]['capacity']])
            link_node_b[link_id,:]=link_row
    # Build N dict
    N_b = {}
    for i in nodes_b:
        N_b[int(i)] = []
    for i in range(len(link_node_b)):
        N_b[int(link_node_b[i,1])].append(int(link_node_b[i,2]))


    link_from_to_b = dict()
    for i in link_b:
        if i not in dummy_link:
            link_from_to_b[(link_b[i]['start'], link_b[i]['end'])] = i


    return link_b,link_fixed_b,link_Acce_b,link_MOD_b,link_Egre_b,transin_oper_b,MOD_nodes_b,nodes_b,outflow_link_b,inflow_link_b,a_alllink_b,a_fixedlink_b,a_translink_b,a_MODlink_b,a_transinlink_b,a_transoutlink_b,link_node_b,N_b,link_from_to_b


####################################################################################################
####################################################################################################
####################################################################################################

def single_branch(PRINT,y_fixed_1, y_fixed_0, v_fixed_1, v_fixed_0, num_iter, tolerance,FW_tol,FW_conver_times,t_coef, c_coef,a_alllink_b,a_fixedlink_b,a_translink_b,a_MODlink_b,a_transinlink_b,a_transoutlink_b,cap,b,oper_cost,link_b,link,link_fixed_b,link_fixed, link_trans,link_MOD_b,link_Acce_b,link_Egre_b,dummy_link,orig,dest,d,nodes_b,MOD_nodes_b,MOD_nodes, transin_oper_b,outflow_link_b,inflow_link_b,V_coef, utility,N_b,link_from_to_b,link_node_b):
    
    q, times, path_set, gamma_record, beta_record, tau_record, lamda_record, miu_record, adj_link_cost_record, x_sp_record = subgrad_opt(PRINT,y_fixed_1, y_fixed_0, v_fixed_1, v_fixed_0, num_iter, tolerance,FW_tol,FW_conver_times,t_coef, c_coef,a_alllink_b,a_fixedlink_b,a_MODlink_b,a_transinlink_b,a_transoutlink_b,cap,b,oper_cost,link_b,link_fixed_b,link_trans,link_MOD_b,link_Acce_b,link_Egre_b,dummy_link,orig,dest,d,nodes_b,MOD_nodes_b,transin_oper_b,outflow_link_b,inflow_link_b,V_coef, utility,N_b,link_from_to_b,link_node_b)

    adj_link_cost = adj_link_cost_record[-1]
    x_sp = x_sp_record[-1]
    gamma = gamma_record[-1]
    beta = beta_record[-1]
    tau = tau_record[-1]

    flows_link, flows_path, y_sol, v_sol = flow_yv_finding(t_coef, c_coef, y_fixed_1, y_fixed_0, v_fixed_1, v_fixed_0,tolerance, path_set, adj_link_cost, x_sp, gamma, beta, tau, orig, dest, d, a_alllink_b, a_fixedlink_b, a_translink_b, a_MODlink_b, a_transinlink_b, dummy_link, link_b, link, link_fixed_b, link_fixed, nodes_b, MOD_nodes_b,MOD_nodes, outflow_link_b, b, cap,N_b,link_from_to_b)
    
    dual_p_od = np.zeros((len(d),len(link_fixed_b)))
    for s in range(len(d)):
        dual_p_od[s,:] = gamma+beta[:,s].T
    dual_p = []
    for linkk in link_fixed:
        if linkk in link_fixed_b:
            link_ind = a_alllink_b.index(linkk)
            dual_p.append(max(dual_p_od[:, link_ind]))
        else:
            dual_p.append(0)
    
    return flows_link, flows_path, y_sol, v_sol, path_set, dual_p

####################################################################################################
####################################################################################################
####################################################################################################
    
def Dijkstra_cpp(nodes_b, O, D, graph):
    
    V = len(nodes_b)
    graph = np.array(graph, dtype='float32')
    graph_reshape = graph.reshape(1, graph.shape[0]*graph.shape[1])[0]

    dist = [0]*V
    dist = np.array(dist, dtype='float32')

    prev_node = [0]*V
    prev_node = np.array(prev_node, dtype='int32')

    ##############
    cpp_dijkstra(graph_reshape, O, dist, prev_node, V, D)
    ##############
    
    path_node_ind = []
    for i in range(V):
        sub_path_node = [i]
        prev = prev_node[i]
        while prev != -1:
            sub_path_node.append(prev)
            prev = prev_node[prev]
    #     sub_path_node.append(source)
        path_node_ind.append(sub_path_node[::-1])

    return path_node_ind, dist
    



####################################################################################################
####################################################################################################
####################################################################################################



def FW(PRINT,path_set,adj_link_cost,FW_tol,FW_conver_times,x_int,t_coef, c_coef,a_alllink_b,a_fixedlink_b,a_MODlink_b,a_transinlink_b,a_transoutlink_b,link_b,link_fixed_b,link_trans,link_MOD_b,link_Acce_b,link_Egre_b,dummy_link,orig,dest,d,nodes_b,MOD_nodes_b,transin_oper_b,V_coef, utility,N_b,link_from_to_b,link_node_b):

    x = x_int.copy()
#     if PRINT == 1:
#         print("-----Initial x-----")
#         plot(x)

    cost_adj = adj_link_cost[0:-2,:].T  
    for i in link_Acce_b:
        link_ind = a_alllink_b.index(i)
        h = link_b[i]['fleet size']
        for k in range(len(orig)):
            cost_adj[k,link_ind] = 2*t_coef[link_b[i]['operator']]/(h**2)*(sum(x)[link_ind]) ######### marginal cost
    
    count = 0
#     t_eq_sum = 0
#     t_sol_sum = 0
    for iii in range(1000):
        
#         if PRINT == 1:
#             print("Iteration",iii)

        Y, path_set = solve_for_Y_vbounded(PRINT,path_set,cost_adj,a_alllink_b,a_fixedlink_b,a_MODlink_b,a_transinlink_b,a_transoutlink_b,link_b,link_fixed_b,link_trans,link_MOD_b,link_Acce_b,link_Egre_b,dummy_link,orig,dest,d,nodes_b,MOD_nodes_b, N_b, link_from_to_b,link_node_b, utility)
       
        
#         if PRINT == 1:
#             print("-----Y-----")
#             plot(Y)


        sol_start = time.time()
        alpha, t_eq, t_sol = find_alpha_der_0(PRINT,adj_link_cost, t_coef, c_coef, a_alllink_b, Y, x,  d, link_b, link_fixed_b, link_trans, link_MOD_b, link_Acce_b, link_Egre_b, dummy_link, MOD_nodes_b, transin_oper_b, V_coef, utility)
        x = x+alpha*(Y-x)
        x = x.astype(float)
        
#         t_eq_sum += t_eq
#         t_sol_sum += t_sol

#         if PRINT == 1:
#             print("-----x-----")
#             plot(x)

        for i in link_Acce_b:
            link_ind = a_alllink_b.index(i)
            h = link_b[i]['fleet size']
    #         print(i, (link_b[i]['start'],link_b[i]['end']), cost_adj[:,link_ind])
            for k in range(len(orig)):
                cost_adj[k,link_ind] = 2*t_coef[link_b[i]['operator']]/(h**2)*(sum(x)[link_ind]) ## marginal cost
                adj_link_cost[link_ind,k] = t_coef[link_b[i]['operator']]/(h**2)*(sum(x)[link_ind])
#             print(i, (link_b[i]['start'],link_b[i]['end']), cost_adj[:,link_ind])
#             print(" ")

        if alpha < FW_tol:
            count += 1
            if count >= FW_conver_times:
                break
        else:
            count = 0
            
                    
    x_sp = x
    
#     print("Equation time:", t_eq_sum)
#     print("Solving time:", t_sol_sum)
    
#     print("FW iterations:", iii)

    return x_sp, path_set


####################################################################################################
####################################################################################################
####################################################################################################

def obj(t_coef, c_coef, x, d, adj_link_cost, a_alllink_b, link_b, link_trans,link_fixed_b, link_MOD_b, link_Acce_b, link_Egre_b, dummy_link, MOD_nodes_b, transin_oper_b, V_coef, utility):
        
    objective = 0

    for s in range(len(d)):
        for i in link_b:
            link_id = a_alllink_b.index(i)
            if i in link_fixed_b:
                objective += x[s,link_id] * adj_link_cost[link_id,s] 
            elif i in link_trans:
                objective += x[s,link_id] * link_b[i]['travel cost']
            elif i in link_MOD_b:
                h = link_b[i]['fleet size']
                objective += x[s,link_id] * (adj_link_cost[link_id,s] + c_coef[link_b[i]['operator']] * h**2)
            elif i in dummy_link:
                objective += utility[s]*x[s,link_id]

    for i in transin_oper_b:
        link_id = a_alllink_b.index(i)
        h = link_b[i]['fleet size']
        sum_x = 0
        for s in range(len(d)):
            sum_x += x[s,link_id]
        objective += t_coef[link_b[i]['operator']]/(h**2) * sum_x**2
    ########################
       
    return objective


####################################################################################################
####################################################################################################
####################################################################################################


def obj_yv(t_coef, c_coef, flows, y_sol, v_sol, d, a_alllink_b, link, link_fixed, link_trans, link_MOD, dummy_link, MOD_nodes, transin_oper, V_coef, utility):   
    
    objective = 0

    for s in range(len(d)):
        for i in link_fixed:
            objective += flows[s,i] * link[i]["travel cost"]
        for i in link_trans:
            objective += flows[s,i] * link[i]["travel cost"]
        for i in link_MOD:
            h = link[i]['fleet size']
            objective += flows[s,i] * (link[i]["travel cost"] + c_coef[link[i]['operator']]*h**2)
        for i in dummy_link:
            objective += utility[s]*flows[s,i]

    for i in transin_oper:
        h = link[i]['fleet size']
        sum_x = sum(flows[:,i])
        objective += t_coef[link[i]['operator']]/(h**2) * sum_x**2

    for i in link_fixed:
        objective += y_sol[i] * link[i]["operating cost"]
    
    for i in MOD_nodes:
        nodee_ind = list(MOD_nodes).index(i)
        objective += V_coef[i]*v_sol[nodee_ind]

    ########################

    return objective



####################################################################################################
####################################################################################################
####################################################################################################



def Dijkstra(N,c_int,O,D,nodes_b,link_node_b):

    c = c_int.copy()
    ######################### Dijkstra's Algorithm ########################### 
    w = np.ones((len(N)))*np.inf #w
    p_node = np.ones((len(N)))*(-1)   #p_node
    p_link = np.ones((len(N)))*(-1)   #p_link
    index_list = ["o"]*len(N)  #open/closed
    index = np.asarray(index_list)
    df = pd.DataFrame({"w": w,"p_node": p_node,"p_link": p_link,"Open/Closed": index})

    #Origin
    df['Open/Closed'][nodes_b.index(O)] = 'c' 
    df['w'][nodes_b.index(O)] = 0        
    current = O

    #iterate
    for j in range(100000):
        ds = N[current] #obtain the downstream nodes
        for s in ds: # update w and p for all downstream nodes
            if df['w'][nodes_b.index(current)] + c[nodes_b.index(current),nodes_b.index(s)] < df['w'][nodes_b.index(s)]:
#                 print((current,s),df['w'][nodes_b.index(current)], "+", c[nodes_b.index(current),nodes_b.index(s)], "<", df['w'][nodes_b.index(s)])
                df['p_node'][nodes_b.index(s)] = current
#                 print([current,s])
#                 print(link_node_b)
                for e in range(len(link_node_b)):
                    if (link_node_b[e,1:3]==[current,s]).all():
                        linkk = link_node_b[e,0]
                        
                df['p_link'][nodes_b.index(s)] = linkk
            df['w'][nodes_b.index(s)] = min(df['w'][nodes_b.index(s)], df['w'][nodes_b.index(current)] + c[nodes_b.index(current),nodes_b.index(s)])

        #find the next starting point
        df_open = df[df['Open/Closed'] == 'o']
        current = nodes_b[df_open.loc[df_open['w']== min(df_open['w'])].index[0]]

        #close the current node
        df['Open/Closed'][nodes_b.index(current)] = 'c'

        #see if we reached the Destination
        if df['Open/Closed'][nodes_b.index(D)] == 'c':
            break

    #Get the nodes of the path
    N_a = []
    Link_a = []
    for s in range(10000000):
        if s == 0:
            n = D
            n_link = df['p_link'][nodes_b.index(n)]
        else:
            n = df['p_node'][nodes_b.index(n)]
            if (n != -1):
                if (df['p_link'][nodes_b.index(n)]!= -1):
                    n_link = df['p_link'][nodes_b.index(n)]
        if n == -1:
            break
        N_a.append(n)
        Link_a.append(n_link)
    # print(N_a)
    path_node = list(np.flip(N_a))
    path_link = list(np.flip(Link_a[:-1]))

    #get the path length
    path_len = 0
    for i in range(len(path_node)-1):
#         print("link:",path[i],path[i+1])
        path_len = path_len + c_int[nodes_b.index(path_node[i]),nodes_b.index(path_node[i+1])]

    return path_node, path_link, path_len


####################################################################################################
####################################################################################################
####################################################################################################


def solve_for_Y_vbounded(PRINT,path_set,cost_adj,a_alllink_b,a_fixedlink_b,a_MODlink_b,a_transinlink_b,a_transoutlink_b,link_b,link_fixed_b,link_trans,link_MOD_b,link_Acce_b,link_Egre_b,dummy_link,orig,dest,d,nodes_b,MOD_nodes_b, N_b, link_from_to_b,link_node_b, utility):


    x_fw = np.zeros((len(orig),len(link_b)))
    # shortest finding
    for od in range(len(orig)):
        t_link_b = cost_adj[od,:]
#         print(t_link_b)

#         # Generate c from N and link_node
#         c_int_b = np.ones((len(N_b),len(N_b))) * np.inf
#         for i in range(len(N_b)):
#             c_int_b[i,i] = 0
#         for i in range(len(link_node_b)):
#             links = link_node_b[i,0]
#             link_id = a_alllink_b.index(links)
#             start = link_node_b[i,1]
#             end = link_node_b[i,2]
# #             print(c_int_b.shape,  len(nodes_b), len(t_link_b))
#             if c_int_b[nodes_b.index(start), nodes_b.index(end)] > t_link_b[int(link_id)]:
#                 c_int_b[nodes_b.index(start), nodes_b.index(end)] = t_link_b[int(link_id)]
  
    
        # Generate c from N and link_node
        graph = np.ones((len(N_b),len(N_b)))*(-1)
        for i in link_b:
            if i not in dummy_link:
                link_id = a_alllink_b.index(i)
                links = i
                start = link_b[i]['start']
                end = link_b[i]['end']
        #             print(c_int_b.shape,  len(nodes_b), len(t_link_b))
                if graph[nodes_b.index(start), nodes_b.index(end)] > t_link_b[int(link_id)] and graph[nodes_b.index(start), nodes_b.index(end)]!=-1:
                    graph[nodes_b.index(start), nodes_b.index(end)] = t_link_b[int(link_id)]
                if graph[nodes_b.index(start), nodes_b.index(end)]==-1:
                    graph[nodes_b.index(start), nodes_b.index(end)] = t_link_b[int(link_id)]


        O = int(orig[od])
        D = int(dest[od])
#         path_node, path_link, path_len = Dijkstra(N_b,c_int_b,O,D,nodes_b,link_node_b)
#         print(O,D)
#         print(graph)
        O_id = nodes_b.index(O)
        D_id = nodes_b.index(D)
        
        path_node_ind_lst, dist = Dijkstra_cpp(nodes_b, O_id, D_id, graph)

        path_node_ind = path_node_ind_lst[D_id]
        path_len = dist[D_id]
        
        
        # compare with the dummy links
        if path_len >= utility[od]:
            path_node = [O,D]
            path_link = [dummy_link[od]]
#             print(path_link, "--dummy", path_len)
        else:   
            path_node = []
            for node_ind in path_node_ind:
                nodee = nodes_b[node_ind]
                path_node.append(nodee)
            path_link = []
            for nodee_ind in range(len(path_node)-1):
                path_link.append(link_from_to_b[path_node[nodee_ind],path_node[nodee_ind+1]]) 
#             print(path_link, path_len)
                
        if path_link not in path_set[od]:
            path_set[od].append(path_link)
            
            
#         if PRINT == 1:
#             print(od, path_node, dist[D_id])

        for links in path_link:
            link_id = a_alllink_b.index(links)
            x_fw[od,link_id] = d[od]
   
    return x_fw, path_set


####################################################################################################
####################################################################################################
####################################################################################################

def find_alpha_der_0(PRINT, adj_link_cost, t_coef, c_coef, a_alllink_b, Y, x,  d, link_b, link_fixed_b, link_trans, link_MOD_b, link_Acce_b, link_Egre_b, dummy_link, MOD_nodes_b, transin_oper_b, V_coef, utility):
    
    eq_start = time.time()
    X = sp.symbols('X')
    
    # ax+b
    expr_a = 0
    expr_b = 0

    for dd in range(len(d)):        
        for i in link_b:
            link_id = a_alllink_b.index(i)
            if i in link_fixed_b: # x of fixed link
                expr_b += adj_link_cost[link_id,dd] *(Y[dd,link_id]-x[dd,link_id])
               
            elif i in link_trans:
                expr_b += link_b[i]['travel cost'] *(Y[dd,link_id]-x[dd,link_id])
                
            elif i in link_MOD_b: # x of MOD link
                h = link_b[i]['fleet size']
                expr_b += (adj_link_cost[link_id,dd])*(Y[dd,link_id]-x[dd,link_id])
                
            elif i in dummy_link:
                expr_b += utility[dd]*(Y[dd,link_id]-x[dd,link_id])
                
                
        #-------------------------------        
    for i in transin_oper_b:  # x of Access link
        link_id = a_alllink_b.index(i)
        h = link_b[i]['fleet size']
#         sum_y_x = 0
#         for s in range(len(d)):
# #             sum_y_x += x[s,link_id]+X*(Y[s,link_id]-x[s,link_id])
#             expr_b
#         sum_x = 0
#         for s in range(len(d)):
#             sum_x += Y[s,link_id]-x[s,link_id]
#         expr += (t_coef*h**(-2)*sum_y_x)*sum_x

        sum_x = 0
        for s in range(len(d)):
            sum_x += x[s,link_id]
        sum_y_x = 0
        for s in range(len(d)):
            sum_y_x += Y[s,link_id]-x[s,link_id]
        
        expr_a += 2*t_coef[link_b[i]['operator']]*h**(-2)*sum_y_x**2
        expr_b += 2*t_coef[link_b[i]['operator']]*h**(-2)*sum_y_x*sum_x
    
    eq_end = time.time()
    t_eq = eq_end - eq_start
    
###############

    sol_start = time.time()
    
#     sol = sp.solve(expr, prec = 100)
    if expr_a == 0:
        sol = []
    else:
        sol = [-expr_b/expr_a]
        
#     if PRINT == 1:
#         print("der - obj(x+alpha(Y-x)) = 0:",sol)

    if len(sol) == 0:
        var1 = x+1*(Y-x) 
        var0 = x+0*(Y-x)
        if obj(t_coef, c_coef, var1, d, adj_link_cost, a_alllink_b, link_b, link_trans,link_fixed_b, link_MOD_b, link_Acce_b, link_Egre_b, dummy_link, MOD_nodes_b, transin_oper_b, V_coef, utility)>=obj(t_coef, c_coef, var0, d, adj_link_cost, a_alllink_b, link_b, link_trans,link_fixed_b, link_MOD_b, link_Acce_b, link_Egre_b, dummy_link, MOD_nodes_b, transin_oper_b, V_coef, utility):
            xopt = 0
        elif obj(t_coef, c_coef, var0, d, adj_link_cost, a_alllink_b, link_b, link_trans,link_fixed_b, link_MOD_b, link_Acce_b, link_Egre_b, dummy_link, MOD_nodes_b, transin_oper_b, V_coef, utility)>=obj(t_coef, c_coef, var1, d, adj_link_cost, a_alllink_b, link_b, link_trans,link_fixed_b, link_MOD_b, link_Acce_b, link_Egre_b, dummy_link, MOD_nodes_b, transin_oper_b, V_coef, utility):
            xopt = 1
#         if PRINT == 1:
#             print("obj0 = ", obj(t_coef, c_coef, var0, d, adj_link_cost, a_alllink_b,link_b,link_trans, link_fixed_b, link_MOD_b, link_Acce_b, link_Egre_b, dummy_link, MOD_nodes_b, transin_oper_b, V_coef, utility),
#                   "obj1 = ", obj(t_coef, c_coef, var1, d, adj_link_cost, a_alllink_b,link_b, link_trans,link_fixed_b, link_MOD_b, link_Acce_b, link_Egre_b, dummy_link, MOD_nodes_b, transin_oper_b, V_coef, utility))
    elif len(sol) == 1:
#         sol = [n for n in sol if n.is_real]
#         if PRINT == 1:
#             print("obj(x+alpha(Y-x)) = 0 and REAL:",sol)
#             print("length of solution:",len(sol))
        if len(sol) == 0: # no real solution
            var1 = x+1*(Y-x) 
            var0 = x+0*(Y-x)
            if obj(t_coef, c_coef, var1, d, adj_link_cost,a_alllink_b, link_b,link_trans, link_fixed_b, link_MOD_b, link_Acce_b, link_Egre_b, dummy_link, MOD_nodes_b, transin_oper_b, V_coef, utility)>=obj(t_coef, c_coef, var0, d, adj_link_cost, a_alllink_b,link_b, link_trans,link_fixed_b, link_MOD_b, link_Acce_b, link_Egre_b, dummy_link, MOD_nodes_b, transin_oper_b, V_coef, utility):
                xopt = 0
            elif obj(t_coef, c_coef, var0, d, adj_link_cost,a_alllink_b, link_b,link_trans, link_fixed_b, link_MOD_b, link_Acce_b, link_Egre_b, dummy_link, MOD_nodes_b, transin_oper_b, V_coef, utility)>obj(t_coef, c_coef, var1, d, adj_link_cost, a_alllink_b,link_b,link_trans, link_fixed_b, link_MOD_b, link_Acce_b, link_Egre_b, dummy_link, MOD_nodes_b, transin_oper_b, V_coef, utility):
                xopt = 1
        else: # real solution found
            xopt = sol[0]
            if xopt >1 or xopt <0:
                var1 = x+1*(Y-x) 
                var0 = x+0*(Y-x)
                if obj(t_coef, c_coef, var1, d, adj_link_cost,a_alllink_b, link_b, link_trans,link_fixed_b, link_MOD_b, link_Acce_b, link_Egre_b, dummy_link, MOD_nodes_b, transin_oper_b, V_coef, utility)>=obj(t_coef, c_coef, var0, d, adj_link_cost, a_alllink_b,link_b,link_trans, link_fixed_b, link_MOD_b, link_Acce_b, link_Egre_b, dummy_link, MOD_nodes_b, transin_oper_b, V_coef, utility):
                    xopt = 0
                elif obj(t_coef, c_coef, var0, d, adj_link_cost,a_alllink_b, link_b, link_trans,link_fixed_b, link_MOD_b, link_Acce_b, link_Egre_b, dummy_link, MOD_nodes_b, transin_oper_b, V_coef, utility)>obj(t_coef, c_coef, var1, d, adj_link_cost, a_alllink_b,link_b,link_trans, link_fixed_b, link_MOD_b, link_Acce_b, link_Egre_b, dummy_link, MOD_nodes_b, transin_oper_b, V_coef, utility):
                    xopt = 1

    else:
        var1 = x+1*(Y-x) 
        var0 = x+0*(Y-x)
        sol = [n for n in sol if n.is_real]
#         if PRINT == 1:
#             print("obj(x+alpha(Y-x)) = 0 and REAL:",sol)
        index = np.intersect1d(np.where(np.array(sol)>=0)[0],np.where(np.array(sol)<=1)[0])
#         if PRINT == 1:
#             print("indices:",index)
        lst_min = [0,1]
        lst_min_obj = [obj(t_coef, c_coef, var0, d, adj_link_cost, a_alllink_b,link_b, link_trans,link_fixed_b, link_MOD_b, link_Acce_b, link_Egre_b, dummy_link, MOD_nodes_b, transin_oper_b, V_coef, utility),obj(t_coef, c_coef, var1, d, adj_link_cost, a_alllink_b,link_b,link_trans, link_fixed_b, link_MOD_b, link_Acce_b, link_Egre_b, dummy_link, MOD_nodes_b, transin_oper_b, V_coef, utility)]
        for i in list(index):
            lst_min.append(sol[int(i)])
            var = x+sol[i]*(Y-x)
            lst_min_obj.append(obj(t_coef, c_coef, var, d, adj_link_cost, a_alllink_b,link_b, link_trans,link_fixed_b, link_MOD_b, link_Acce_b, link_Egre_b, dummy_link, MOD_nodes_b, transin_oper_b, V_coef, utility))

#         if PRINT == 1:
#             print("candidates:",lst_min)
#             print("obj value:",lst_min_obj) 

        a = np.where(np.array(lst_min_obj) == min(lst_min_obj))[0]
        xopt = lst_min[int(a)]
    
#     if PRINT == 1:
#         a = np.zeros((500))
#         func = np.zeros((500))
#         j = 0
#         for l in range(500):
#             a[l] = l/500 *1
#             var = x + a[l]*(Y-x)
#             func[l] = obj(t_coef, c_coef, var, d,adj_link_cost,a_alllink_b, link_b,link_trans, link_fixed_b, link_MOD_b, link_Acce_b, link_Egre_b, dummy_link, MOD_nodes_b, transin_oper_b, V_coef, utility)
#         plt.plot(a,func)
#         plt.xlabel('alpha')
#         plt.ylabel('z(x+alpha(y-x))')
#         plt.show()
#         print("xopt = ",xopt)
        
    sol_end = time.time()
    t_sol = sol_end - sol_start
        
    # print("grad at xopt = ",expr.subs({X:xopt}))
    return xopt, t_eq, t_sol


####################################################################################################
####################################################################################################
####################################################################################################



def subgrad_opt(PRINT,y_fixed_1, y_fixed_0, v_fixed_1, v_fixed_0, num_iter, tolerance,FW_tol,FW_conver_times,t_coef, c_coef,a_alllink_b,a_fixedlink_b,a_MODlink_b,a_transinlink_b,a_transoutlink_b,cap,b,oper_cost,link_b,link_fixed_b,link_trans,link_MOD_b,link_Acce_b,link_Egre_b,dummy_link,orig,dest,d,nodes_b,MOD_nodes_b,transin_oper_b,outflow_link_b,inflow_link_b,V_coef, utility,N_b,link_from_to_b,link_node_b):

    ################ FW+LR ################

    # initialize gamma, beta, tau
    gamma = np.zeros((len(link_fixed_b))) #fixed link
    beta =  np.zeros((len(link_fixed_b),len(orig))) #fixed link??
    tau =   np.zeros((len(link_Acce_b))) #MOD access link
    lamda = np.zeros((len(MOD_nodes_b))) #MOD nodes
    miu =  np.zeros((len(MOD_nodes_b),len(orig))) #MOD nodes

    # adjusted link costs initialization  
    adj_link_cost = np.zeros((len(link_b),len(orig)))
    for i in link_b:
        link_id = a_alllink_b.index(i)
        for k in range(len(orig)):
            adj_link_cost[link_id,k] = link_b[i]['travel cost']
            if i in link_fixed_b:
                adj_link_cost[link_id,k] = link_b[i]['travel cost'] + link_b[i]['operating cost']/link_b[i]['capacity']
            if i in link_MOD_b:
                h = link_b[i]['fleet size']
                adj_link_cost[link_id,k] = adj_link_cost[link_id,k] + c_coef[link_b[i]['operator']]*h**2

    # record
    gamma_record = np.zeros((num_iter+1, len(link_fixed_b)))
    beta_record = np.zeros((num_iter+1, len(link_fixed_b), len(orig)))
    lamda_record = np.zeros((num_iter+1, len(MOD_nodes_b)))
    miu_record = np.zeros((num_iter+1, len(MOD_nodes_b), len(orig)))

    tau_record = np.zeros((num_iter+1, len(link_Acce_b)))
    adj_link_cost_record = np.zeros((num_iter+1, len(link_b), len(orig)))

    x_sp_record = np.zeros((num_iter+1,len(orig),len(link_b)))
    
    path_set = []
    for s in range(len(d)):
        path_set.append([])
    
    times = []

    for q in range(1,num_iter+1):
        
        i_start = time.time()
        
        if PRINT == 1:
            print("----------------")
            print('Subgradient opt iteration '+str(q))

        #kth iteration
        # step size
        if q>2:
            if change[0]<=change[1]:
                betaa += 1.8
                theta = 1/betaa
            else:
                betaa += 0.5
                theta = 1/betaa
        else:
            betaa = q
            theta = 1/betaa
            
        if PRINT == 1:
            print("step size: ", theta, "=", 1,"/",betaa)

        #record gamma, beta, and tau
        gamma_record[q-1,:] = gamma
        beta_record[q-1,:,:] = beta
        lamda_record[q-1,:] = lamda
        miu_record[q-1,:,:] = miu
        tau_record[q-1,:] = tau
        adj_link_cost_record[q-1,:,:] = adj_link_cost
        if q > 1:
            x_sp_record[q-1,:,:] = x_sp

        # Step 1: solve the relaxed problem
#         i_start_initer = time.time()
        if q == 1:
            x_int = np.zeros((len(orig),len(link_b)))
            count = 0
            for i in dummy_link:
                i_ind = a_alllink_b.index(i)
                x_int[count,i_ind] = d[count]
                count+=1
        else:
            x_int = x_sp
            
        x_sp,path_set = FW(PRINT,path_set,adj_link_cost,FW_tol,FW_conver_times,x_int,t_coef, c_coef,a_alllink_b,a_fixedlink_b,a_MODlink_b,a_transinlink_b,a_transoutlink_b,link_b,link_fixed_b,link_trans,link_MOD_b,link_Acce_b,link_Egre_b,dummy_link,orig,dest,d,nodes_b,MOD_nodes_b,transin_oper_b,V_coef, utility,N_b,link_from_to_b,link_node_b)

#         plot(x_sp)
#         print(path_set)
#         i_end_initer = time.time()
#         times_in_iter = i_end_initer-i_start_initer 


        # Step 2: Update gamma, beta, and tau using new x and y
        # 2.1 Gamma
        for links in link_fixed_b:
            link_id = a_alllink_b.index(links)
            link_flow = sum(x_sp[:,link_id])
            gamma[link_id] = max(0,gamma[link_id] + theta*(link_flow - cap[links])) 
        # 2.2 Beta
        for links in link_fixed_b:
            link_id = a_alllink_b.index(links)
            for k in range(len(orig)):
                beta[link_id,k] = max(0,beta[link_id,k] + theta*(x_sp[k,link_id] - b[k,links]))
        # 2.3 Lamda
        for node_ind in range(len(MOD_nodes_b)):
            nodee = MOD_nodes_b[node_ind]
            out_flow_sum_od = 0
            for links in outflow_link_b[nodee]:
                link_id = a_alllink_b.index(links)
                out_flow_sum_od += sum(x_sp[:,link_id])
            lamda[node_ind] = max(0,lamda[node_ind] + theta*(out_flow_sum_od - sum(d)))  
#             if lamda[node_ind] > 0:
#                 print(nodee, out_flow_sum_od, lamda[node_ind])
        # 2.4 Miu
        for node_ind in range(len(MOD_nodes_b)):
            nodee = MOD_nodes_b[node_ind]
            for k in range(len(orig)):
                out_flow_sum = 0
                for links in outflow_link_b[nodee]:
                    link_id = a_alllink_b.index(links)
                    out_flow_sum += x_sp[k,link_id]
                miu[node_ind,k] = max(0,miu[node_ind,k] + theta*(out_flow_sum - d[k]))
        # 2.5 Tau 
        count = 0
        for links in transin_oper_b:
            link_id = a_alllink_b.index(links)
            h = link_b[links]['fleet size']
            sum_x = 0
            for k in range(len(orig)):
                sum_x += x_sp[k,link_id]
            tau[count] = t_coef[link_b[links]['operator']]/(h**2) * sum_x
            count += 1      


        # Step 3: update adj_link_cost
        ## update fixed links
#         print("fixed link delay:")
        for i in link_fixed_b:
            if i in y_fixed_1:
                link_id = a_alllink_b.index(i)
                for k in range(len(orig)):
                    adj_link_cost_new = link_b[i]['travel cost'] + gamma[link_id] + beta[link_id,k]
    #                 print((link[i]['start'],link[i]['end'],k, 'fixed 1'),adj_link_cost_new-link[i]['travel cost'],"=",gamma[link_id],"+",beta[link_id,k])
                    adj_link_cost[link_id,k] = adj_link_cost_new
            else:
                link_id = a_alllink_b.index(i)
                f_func_pos = max(0, oper_cost[i]-sum(b[k,i]*beta[link_id,k] for k in range(len(orig))))
                for k in range(len(orig)):
                    adj_link_cost_new = link_b[i]['travel cost'] + gamma[link_id] + f_func_pos/cap[i] + beta[link_id,k]
    #                 print((link[i]['start'],link[i]['end'],k),adj_link_cost_new-link[i]['travel cost'],"=",gamma[link_id],"+",f_func_pos/cap[i],"+",beta[link_id,k])
                    adj_link_cost[link_id,k] = adj_link_cost_new 

        ## update MOD node going-out links
#         print("MOD node going-out link delay:")
        for node_ind in range(len(MOD_nodes_b)):
            nodee = MOD_nodes_b[node_ind]
            if nodee in v_fixed_1:
                for links in outflow_link_b[nodee]:
                    link_id = a_alllink_b.index(links)
                    if links in link_MOD_b:
                        h = link_b[links]['fleet size']
                        for k in range(len(orig)):
                            adj_link_cost_new = link_b[links]['travel cost']+ c_coef[link_b[links]['operator']]*h**2+lamda[node_ind] + miu[node_ind,k]
    #                         print((link[links]['start'],link[links]['end'],k),adj_link_cost_new-link[links]['travel cost']-c_coef*h**2,"=",lamda[node_ind],"+",miu[node_ind,k])
                            adj_link_cost[link_id,k] = adj_link_cost_new
                    else:
                        for k in range(len(orig)):
                            adj_link_cost_new = link_b[links]['travel cost']+ lamda[node_ind] + miu[node_ind,k]
    #                         print((link[links]['start'],link[links]['end'],k),adj_link_cost_new-link[links]['travel cost'],"=",lamda[node_ind],"+",miu[node_ind,k])
                            adj_link_cost[link_id,k] = adj_link_cost_new
            else:
                m0_func_pos = max(0, V_coef[nodee]-sum(d[k]*miu[node_ind,k] for k in range(len(orig))))
                for links in outflow_link_b[nodee]:
                    link_id = a_alllink_b.index(links)
                    if links in link_MOD_b:
                        h = link_b[links]['fleet size']
                        for k in range(len(orig)):
                            adj_link_cost_new = link_b[links]['travel cost']+ c_coef[link_b[links]['operator']]*h**2 + lamda[node_ind] + m0_func_pos/sum(d) + miu[node_ind,k]
    #                         print((link[links]['start'],link[links]['end'],k),adj_link_cost_new-link[links]['travel cost']-c_coef*h**2,"=",lamda[node_ind],"+",m0_func_pos/sum(d),"+",miu[node_ind,k])
                            adj_link_cost[link_id,k] = adj_link_cost_new
                    else:
                        for k in range(len(orig)):
                            adj_link_cost_new = link_b[links]['travel cost'] + lamda[node_ind] + m0_func_pos/sum(d) + miu[node_ind,k]
    #                         print((link[links]['start'],link[links]['end'],k),adj_link_cost_new-link[links]['travel cost'],"=",lamda[node_ind],"+",m0_func_pos/sum(d),"+",miu[node_ind,k])
                            adj_link_cost[link_id,k] = adj_link_cost_new


        # Step 4: convergence check (change in gamma, beta, and tau)
        # gamma check
        gamma_change = gamma - gamma_record[q-1,:]
        if np.linalg.norm(gamma_change) < tolerance:
            gamma_break = 1
        else: 
            gamma_break = 0
        if PRINT == 1:
            print("Gamma vector change:", np.linalg.norm(gamma_change))
            
        # beta check
        beta_change = beta - beta_record[q-1,:,:]
        if np.linalg.norm(beta_change) < tolerance:
            beta_break = 1
        else: 
            beta_break = 0
        if PRINT == 1:
            print("Beta vector change:", np.linalg.norm(beta_change))
            
        # tau check
        tau_change = tau - tau_record[q-1,:]
        if np.linalg.norm(tau_change) < tolerance:
            tau_break = 1
        else: 
            tau_break = 0
            
        # Miu check
        miu_change = miu - miu_record[q-1,:,:]
        if np.linalg.norm(miu_change) < tolerance:
            miu_break = 1
        else: 
            miu_break = 0
        if PRINT == 1:
            print("Miu vector change:", np.linalg.norm(miu_change))
            
        # lamda check
        lamda_change = lamda - lamda_record[q-1,:]
        if np.linalg.norm(lamda_change) < tolerance:
            lamda_break = 1
        else: 
            lamda_break = 0
        if PRINT == 1:
            print("Lamda vector change:", np.linalg.norm(lamda_change))        
    
        if q == 1:
            change = []
            change.append(np.linalg.norm(gamma_change) + np.linalg.norm(beta_change))
        elif q == 2:
            change.append(np.linalg.norm(gamma_change) + np.linalg.norm(beta_change))
        else:
            change[0] = change[1]   
            change[1] = np.linalg.norm(gamma_change) + np.linalg.norm(beta_change)
 


        if gamma_break == 1 and beta_break == 1 and lamda_break == 1 and miu_break == 1 and tau_break == 1:
            gamma_record[q,:] = gamma
            beta_record[q,:,:] = beta
            tau_record[q,:] = tau
            lamda_record[q,:] = lamda
            miu_record[q,:,:] = miu
            adj_link_cost_record[q,:,:] = adj_link_cost
            x_sp_record[q,:,:] = x_sp 

            gamma_record = gamma_record[0:q+1,:]
            beta_record = beta_record[0:q+1,:,:]
            tau_record = tau_record[0:q+1,:]
            lamda_record = lamda_record[0:q+1,:]
            miu_record = miu_record[0:q+1,:,:]
            adj_link_cost_record = adj_link_cost_record[0:q+1,:,:]
            x_sp_record = x_sp_record[0:q+1,:,:]
#             if PRINT == 1:
#                 print(" ")
#                 print("################## CONVERGENCE!! ##################")
            break
            
        i_end = time.time()
        times.append(i_end - i_start)
        
    print("Subgradient opt number of iterations:",q)
        
#         if times[q-1]>10:
#             print("FW time:", times_in_iter)
        
    gamma_record[q,:] = gamma
    beta_record[q,:,:] = beta
    tau_record[q,:] = tau
    lamda_record[q,:] = lamda
    miu_record[q,:,:] = miu
    adj_link_cost_record[q,:,:] = adj_link_cost
    x_sp_record[q,:,:] = x_sp 

    gamma_record = gamma_record[0:q+1,:]
    beta_record = beta_record[0:q+1,:,:]
    tau_record = tau_record[0:q+1,:]
    lamda_record = lamda_record[0:q+1,:]
    miu_record = miu_record[0:q+1,:,:]
    adj_link_cost_record = adj_link_cost_record[0:q+1,:,:]
    x_sp_record = x_sp_record[0:q+1,:,:]

    return q, times, path_set, gamma_record, beta_record, tau_record, lamda_record, miu_record, adj_link_cost_record, x_sp_record



####################################################################
####################################################################
####################################################################


def flow_yv_finding(t_coef, c_coef, y_fixed_1, y_fixed_0, v_fixed_1, v_fixed_0,tolerance, path_set, adj_link_cost, x_sp, gamma, beta, tau, orig, dest, d, a_alllink_b, a_fixedlink_b, a_translink_b, a_MODlink_b, a_transinlink_b, dummy_link, link_b, link, link_fixed_b, link_fixed, nodes_b, MOD_nodes_b,MOD_nodes, outflow_link_b, b, cap,N_b,link_from_to_b):
    
    cost_adj = np.zeros((len(orig),len(link_b)-len(d)))
    for k in range(len(orig)):
        for links in a_fixedlink_b:
            link_id = a_alllink_b.index(links)
            cost_adj[k,link_id] = link_b[links]['travel cost'] + gamma[link_id] + beta[link_id,k] + max(0,link_b[links]['operating cost']-np.dot(b[:,links],beta[link_id,:]))/link_b[links]['capacity']
        for links in a_transinlink_b:
            link_id = a_alllink_b.index(links)
            if tau[list(a_transinlink_b).index(links)] >= tolerance:
                cost_adj[k,link_id] = tau[list(a_transinlink_b).index(links)]
            else:
                cost_adj[k,link_id] = 999999999
        for links in a_MODlink_b:
            link_id = a_alllink_b.index(links)
            if x_sp[k,link_id] >= tolerance:
                h = link_b[links]['fleet size']
                cost_adj[k,link_id] = link_b[links]['travel cost'] + c_coef[link_b[links]['operator']]*h**2
            else:
                cost_adj[k,link_id] = 999999999
        for links in a_translink_b:
            link_id = a_alllink_b.index(links)
            cost_adj[k,link_id] = link_b[links]['travel cost']
        
    #### finding the flows ####
    path_hash = dict()
    path_od_hash = dict()
    for k in range(len(orig)):
        path_od_hash[k] = []
    count = 0
    for k in range(len(orig)):
        for path in path_set[k]:
            path_hash[count] = path
            path_od_hash[k].append(count)
            count += 1

    fixed_link_cost = np.zeros((1,len(link_fixed_b)))
    for i in link_fixed_b:
        link_id = a_fixedlink_b.index(i) 
        fixed_link_cost[0,link_id] = link_b[i]['travel cost']
    for k in range(len(orig)):
        delay_od_k = adj_link_cost[:len(link_fixed_b),k] - fixed_link_cost
        if k == 0:
            delay = delay_od_k
        else:
            delay = np.vstack((delay,delay_od_k))

    MODlink_used = []
    accelink_used = []
    fixedlink_used = []
    for path_ind in path_hash:
        for links in path_hash[path_ind]:
            if links in a_MODlink_b and links not in MODlink_used:
                MODlink_used.append(links)
            elif links in a_fixedlink_b and links not in fixedlink_used:
                fixedlink_used.append(links)
            elif links in a_transinlink_b and links not in accelink_used:
                accelink_used.append(links)
            
    #A matrix
    for path_ind in path_hash:
        path = path_hash[path_ind]
        a = np.zeros((len(link_b),1))
        for links in path:
            link_id = a_alllink_b.index(links) 
            a[link_id] = 1
        if path_ind == 0:
            A = a
        else:
            A = np.hstack((A,a))

    A_od = []
    try:
        del A_k
    except:
        pass
    for k in path_od_hash:
        for path_ind in path_od_hash[k]:
            try:
                a = np.reshape(A[:,path_ind], (len(link_b),1))
                A_k = np.hstack((A_k,a))
            except:
                a = np.reshape(A[:,path_ind], (len(link_b),1))
                A_k = a
        A_od.append(A_k)
        del A_k

    m = Model()
    h = m.addVars(len(path_hash),lb=0,vtype=GRB.CONTINUOUS, name="h") #path flows
    m.update()
    for k in range(len(orig)):
        m.addConstr(quicksum(h[path] for path in path_od_hash[k]) == d[k], name = 'demand'+str(k))
    m.update()   
    for links in MODlink_used:
        link_id = a_alllink_b.index(links) 
        m.addConstr(quicksum(A[link_id,path_ind]*h[path_ind] for path_ind in path_hash) == sum(x_sp)[link_id], name = 'MOD flow'+str(links))     
    m.update()  
    for links in accelink_used:
        link_id = a_alllink_b.index(links) 
        m.addConstr(quicksum(A[link_id,path]*h[path] for path in path_hash) == sum(x_sp)[link_id], name = 'Acce flow'+str(links))
    m.update()   
    for links in fixedlink_used:
        link_id = a_alllink_b.index(links) 
        m.addConstr(quicksum(A[link_id,path]*h[path] for path in path_hash) <= link_b[int(links)]['capacity'], name = 'Capacity'+str(links))
    m.update()
    obj = 0
    sum_ind = 0
    for k in range(len(orig)):
        delay_path_k = np.dot(A_od[k][0:len(link_fixed_b),:].T,delay[k,:])
        obj += quicksum(delay_path_k[path_ind-sum_ind]*h[path_ind] for path_ind in path_od_hash[k])
        sum_ind += A_od[k].shape[1]
    m.setObjective(obj,GRB.MAXIMIZE)
    m.update()
    m.params.outputflag = 0
    m.optimize()
#     print(m.status)

    # link flows
    for k in range(len(orig)):
        flows_k = np.zeros((len(link)))
        for path_ind in path_od_hash[k]:
            for links in path_hash[path_ind]: 
                flows_k[links] = flows_k[links] +h[path_ind].x
        if k == 0:
            flows_link = flows_k
        else:
            flows_link = np.vstack((flows_link, flows_k))

    # path flows
    flows_path = []
    for k in range(len(orig)):
        for path_ind in path_od_hash[k]:
            flows_path.append(h[path_ind].x)
        

    y_sol = []
    for links in link_fixed:
        if links in y_fixed_1:
              y_sol.append(1)
        elif links in y_fixed_0:
              y_sol.append(0)
        else:   
            link_ind = a_alllink_b.index(links)
            y_sol.append(sum(flows_link[:,links])/cap[links])

    v_sol = []
    for nodee in MOD_nodes:
        if nodee in v_fixed_1:
            v_sol.append(1)
        elif nodee in v_fixed_0:
            v_sol.append(0)
        else:
            out_flow = 0
            for links in outflow_link_b[nodee]:
                out_flow += sum(flows_link[:,links])
            v_sol.append(out_flow/sum(d))
    
    return flows_link, flows_path, y_sol, v_sol
   