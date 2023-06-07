# MaaS Platform Equilibrium Model

MaaS Platform Equilibrium Model is a tool designed to model the decisions of users and operators in a Mobility-as-a-Service (MaaS) platform, allowing platform subsidy plans to to achieve a desirable equilibrium. The model considers different types of services providers, such as Mobility-on-Demand (MoD) operators and traditional fixed-route transit operators. It facilitates efficient management and coordination between users, operators, and the platform within a mobility ecosystem.

Input of the model includes the following information:
- Users: origin-destination demand for each user class considered
- Operators: 
  - For each fixed-route transit operator: network link they potentially operate on, service frequency options, link operation costs, link travel times. 
  - For each MoD operator: service region that they they potentially cover, fleet size options, unit service cost, installation cost. 
Incorporating congestion modeling, our model accurately captures the impact of congestion on accessing MOD services. It formulates the matching problem as a convex multicommodity flow network design problem, taking into account the cost associated with accessing MOD services under congestion.

Output of the model include the following information:
- Equilibrium user flows of each user class on each operated link;
- Operation decisions of operators:
  - For each fixed-transit operator: the links they decide to operate on and corresponding service frequencies;
  - For each MoD operator: service regions that they decide to cover and corresponding fleet sizes;
- Subsidy injected to stablize the system if needed. 

MaaS Platform Equilibrium Model offers an advanced approach to optimizing MaaS systems. By considering input data related to fixed-route transit services, MOD services, and congestion, it generates an optimized assignment solution that enhances the overall experience for travelers while enabling operators to make informed decisions and maximize their service offerings.
