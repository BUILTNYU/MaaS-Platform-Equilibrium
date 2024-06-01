# MaaS Platform Equilibrium Model

MaaS Platform Equilibrium Model is a tool designed to model the decisions of travelers and operators in a Mobility-as-a-Service (MaaS) platform, allowing platform subsidy plans to to achieve a desirable equilibrium. The model considers different types of services providers: Mobility-on-Demand (MoD) operators and traditional fixed-route transit operators. It facilitates efficient management and coordination between users, operators, and the platform within a mobility ecosystem. 

The model takes the network structure of the operators, travelers' and operators' costs, traveler demand, and a system objective, and outputs the assignment of traveler demand, operators' operation decisions, and subsidy plans that optimizes the system objective. Potential system objectives includes minimizing system total costs, maximizing equity indices, minimizing GHG emissions, etc. The current tool consideres minimizing system total costs.

**Model inputs:**
- Demand: number of travelers going from each origin to each destination for each traveler class considered.
  * * Note: Travelers can be divided into different classes according to income level, age, disability, etc.
- Supply: 
  - For each fixed-route transit operator: network link they potentially operate on, service frequency options, link operation costs for each link and service frequency, link travel costs.
    * * Note: link travel costs is an representation of all measurable non-monetary costs on average, including travel time, average wait time, discomfort, etc.
    * * Note: if system objectives other than cost are considered other information would be needed for the links, such as GHG emission per user for each link, etc. 
  - For each MoD operator: service zones that they potentially cover, fleet size options, service cost per user in between each pair of service region, installation cost, access cost of each service zones as functions of number of travelers accessing the service through the zone and fleet size of the zone.

**Model outputs:**
- Equilibrium traveler flows of each travelers class on each operated link;
- Operation decisions of operators:
  - For each fixed-transit operator: the links they decide to operate and corresponding service frequencies;
  - For each MoD operator: service regions that they decide to cover and corresponding fleet sizes;
- Subsidies per traveler on each link. 

# Example

