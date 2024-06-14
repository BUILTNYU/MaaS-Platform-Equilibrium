# MaaS Platform Equilibrium Model

MaaS Platform Equilibrium Model is a tool designed to model the decisions of travelers and operators in a Mobility-as-a-Service (MaaS) platform, allowing platform subsidy plans to to achieve a desirable equilibrium. The model considers different types of services providers: Mobility-on-Demand (MoD) operators and traditional fixed-route transit operators. It facilitates efficient management and coordination between users, operators, and the platform within a mobility ecosystem. 

The model takes the network structure of the operators, travelers' and operators' costs, traveler demand, and a system objective, and outputs the assignment of traveler demand, operators' operation decisions, and subsidy plans that optimizes the system objective. Potential system objectives includes minimizing system total costs, maximizing equity indices, minimizing GHG emissions, etc. The current tool consideres minimizing system total costs.

The tool is coded in Python 3.8.5.

**Model inputs:**
- Demand:
  - Number of travelers of each origin-destination (OD) pair for each traveler class considered.
    * Note: Travelers can be divided into different classes according to income level, age, disability, etc.
  - Trip utility of each OD pair for each traveler class considered. 
    * Note: Trip utility represent the "gain" upon making this trip, needs to be calibrated beforehand. It can be related to trip purpose, income level, etc. 
      
- Supply:
  - For each fixed-route transit operator: network link they potentially operate on, service frequency options, link operation costs for each link and service frequency, link travel costs.
    * Note: link travel costs is an representation of all measurable non-monetary costs on average, including travel time, average wait time, discomfort, etc.
    * Note: if system objectives other than cost are considered other information would be needed for the links, such as GHG emission per user for each link, etc. 
  - For each MoD operator: service zones that they potentially cover, fleet size options, service cost per user in between each pair of service region, installation cost, access cost of each service zones as functions of number of travelers accessing the service through the zone and fleet size of the zone.
    * Note: all costs are trasferred into a common unit, e.g. $.
    * Note: the access cost function in this tool is defined as follows: t = t_coef*x^2/h, the coefficient t_coef is an input.

- Convergence parameters:
  - Tolerance ϵ of subgradient optimization (default ).
  - Tolerance ε of Frank-Wolfe algorithm (default ).
  - Required consecutive number of iterations meeting the tolerance of Frank-Wolfe algorithm (default ).
  - Optimality gap control parameter of Branch-and-bound algorithm (default ).


**Model outputs:**
- Equilibrium traveler flows of each travelers class on each operated link;
- Number of travelers of each origin-destination (OD) pair that choose not to use the MaaS platform;
- Operation decisions of operators:
  - For each fixed-transit operator: the links they decide to operate and corresponding service frequencies;
  - For each MoD operator: service regions that they decide to cover and corresponding fleet sizes;
- Subsidies per traveler on each link.


**Input format:**

There are 5 tables required as inputs, which are described as follows. Please name the input tables and columns exactly the same as instructed.
1. Network information table: "network.csv"
   
Columns:
- "link number": ID of the link.
- "from_node": ID of the start node of the link.
- "to_node": ID of the end node of the link.
- "travel cost": travel cost of the link ($).
- "operating cost": operating cost of the link ($). Only applies if it is a fixed-route link. For other link types, 0 should be put in.  
- "capacity": capacity of the link (travelers/unit time). Only applies if it is a fixed-route link. For other link types, a number that is large enough (e.g. 10^8) should be put in.  
- "operator": ID of the operator that owns the link.
- "link type": "fixed", "MOD", or "transfer". "fixed" means fixed-route transit link, "MOD" means MOD link, "transfer" means walking link. 
  
2. Demand information table: "demand.csv"
   
Columns:
- "user_number": ID of the traveler group.
- "from_node": ID of the node from which the traveler group departs.
- "to_node": ID of the node to which the traveler group heads.
- "demand_amount": number of travelers in the traveler group.
- "utility": trip utility of each travelers in the traveler group ($).
  
3. MOD node renumbering table: "MODnodes.csv"
   
Columns:
- "fixed_node"
- "MOD_node"
- "direction"
- "operator"
- "capacity"
- "cost"
  
4. MOD operation coeffcients table: "MODoper_coef.csv"
   
Columns:
- "Operator"
- "t_coef"
- "c_coef"
  
5. Fleet size options table: "fleet_size_options.csv"
   
Columns:
- "operator"
- "fleet_size_options"



**Output format:**




# Example
We use a toy network shown in Fig. 1 to illustrate how the method works. All costs are in dollars ($). The solid links represent the fixed-route services, links with the same color are operated by the same operator. Link (21,22) and (21,23) are transfer links between lines, which are without capacity and operating cost with no owners. There are 3 MoD operators (blue, green, brown). The circles represent the service zones that MOD operators can choose from to operate (blue: A,B,C; green: B,C; brown: B,D). Zone A covers transit station node 1. Zone B covers transit station nodes 21,22,23. Zone C covers transit station node 3. Zone D covers transit station node 4.

The network is expanded into Fig. 2 by creating complete subgraphs for each MoD operator and adding MoD access links and egress links. Travel cost, operating cost, and capacities are labelled as shown in the legend. The access/wait cost functions of all the MoD operators on all MoD access links l is represented as follows. 

τ_l (∑_(s∈S)▒x_sl ;h)=0.5h^(-2) ∑_(s∈S)▒x_sl


The MoD operating cost parameter of all the MoD operators at all MoD nodes is represented as follows. 

m_l=h^2

Fleet size choices of all MoD operators are 1, 2, and 3. Installation cost of MoD nodes 7, 8, 9, 10, 11, 12, and 13 are 3, 3, 2, 2, 1, 1, and 3, respectively for all fleet size options. Travel demand is 1,000 from node 1 to 3, and 500 from node 1 to 4. Trip utility U_s is $9.50 for both OD pairs. The tolerance ϵ for subgradient optimization is 0.05. The tolerance ε of Frank-Wolfe is 0.01 and the required consecutive number of iterations meeting the tolerance is 5. No optimality gap control is applied, the algorithm is terminated when all branches are pruned.


<img width="281" alt="image" src="https://github.com/BUILTNYU/MaaS-Platform-Equilibrium/assets/75587054/6f9178d3-736c-4af4-9280-e9e056035abb">

Figure 1. Toy network.


<img width="281" alt="image" src="https://github.com/BUILTNYU/MaaS-Platform-Equilibrium/assets/75587054/0cdda0ab-fd38-4186-9af5-c4b93a5a8bca">

Figure 2. Expended toy network.


The model inputs are as follows.
1. Network information table: "network.csv"
<img width="646" alt="Screen Shot 2024-06-11 at 3 37 53 AM" src="https://github.com/BUILTNYU/MaaS-Platform-Equilibrium/assets/75587054/478b3f7b-ecb2-45ce-a06b-c0c0a285ea5b">

2. Demand information table: "demand.csv"
<img width="410" alt="Screen Shot 2024-06-11 at 3 38 19 AM" src="https://github.com/BUILTNYU/MaaS-Platform-Equilibrium/assets/75587054/c8baa293-8d90-4cb2-a2dc-9f6c962763af">
 
3. MOD node renumbering table: "MODnodes.csv"
<img width="413" alt="Screen Shot 2024-06-11 at 3 38 43 AM" src="https://github.com/BUILTNYU/MaaS-Platform-Equilibrium/assets/75587054/9c21cdab-a01c-4682-82f5-20e05d26d93d">
  
4. MOD operation coeffcients table: "MODoper_coef.csv"
<img width="195" alt="Screen Shot 2024-06-11 at 3 39 07 AM" src="https://github.com/BUILTNYU/MaaS-Platform-Equilibrium/assets/75587054/1ad411ac-dd38-48a7-a3d4-98872e61c78e">
 
5. Fleet size options table: "fleet_size_options.csv"
<img width="208" alt="Screen Shot 2024-06-11 at 3 39 27 AM" src="https://github.com/BUILTNYU/MaaS-Platform-Equilibrium/assets/75587054/238891d1-e4db-4a48-9ec4-75aef5228add">

After running the tool, the outputs are as follows.


