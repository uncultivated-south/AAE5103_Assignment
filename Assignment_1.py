import pygad

# Problem data
flights = {
    'F1': {'origin': 'X', 'dest': 'Y', 'cost': 100},
    'F2': {'origin': 'Y', 'dest': 'Z', 'cost': 120},
    'F3': {'origin': 'Z', 'dest': 'X', 'cost': 130},
    'F4': {'origin': 'X', 'dest': 'Z', 'cost': 140},
    'F5': {'origin': 'Z', 'dest': 'Y', 'cost': 110},
    'F6': {'origin': 'Y', 'dest': 'X', 'cost': 115}
}

required_flights = [
    ['F1', 'F4'],  # Day 1
    ['F2', 'F3'],  # Day 2
    ['F5', 'F6'],  # Day 3
    ['F1', 'F2'],  # Day 4
    ['F3', 'F6']   # Day 5
]

MAINTENANCE_COST = 200
MAX_FLIGHTS_BETWEEN_MAINTENANCE = 3
MAINTENANCE_AIRPORTS = {'Y', 'Z'}

def find_repositioning_flight(current_loc, target_loc):
    """Find a repositioning flight from current to target location"""
    for fname, fdata in flights.items():
        if fdata['origin'] == current_loc and fdata['dest'] == target_loc:
            return fname, fdata['cost']
    return None, None

def fitness_function(ga_instance, solution, solution_idx):
    """Calculate fitness (negative total cost with penalties)"""
    # Initialize aircraft states
    aircraft_states = {
        'A1': {'location': 'X', 'flights_since_maint': 0},
        'A2': {'location': 'X', 'flights_since_maint': 0}
    }
    
    total_cost = 0
    penalties = 0
    maintenance_schedule = {day: [] for day in range(1, 6)}
    
    # Process each day
    for day_idx, day_flights in enumerate(required_flights):
        day = day_idx + 1
        
        # Determine assignment based on chromosome
        if solution[day_idx] == 0:
            assignments = {'A1': day_flights[0], 'A2': day_flights[1]}
        else:
            assignments = {'A1': day_flights[1], 'A2': day_flights[0]}
        
        # Process each aircraft's flight for the day
        for aircraft, flight_name in assignments.items():
            flight_info = flights[flight_name]
            origin = flight_info['origin']
            dest = flight_info['dest']
            cost = flight_info['cost']
            
            current_loc = aircraft_states[aircraft]['location']
            
            # Check if repositioning is needed
            if current_loc != origin:
                repo_flight, repo_cost = find_repositioning_flight(current_loc, origin)
                if repo_flight is None:
                    penalties += 10000  # Cannot reach required flight
                else:
                    total_cost += repo_cost
                    aircraft_states[aircraft]['flights_since_maint'] += 1
                    current_loc = origin
            
            # Fly the required flight
            total_cost += cost
            aircraft_states[aircraft]['location'] = dest
            aircraft_states[aircraft]['flights_since_maint'] += 1
            
            # Check if maintenance is needed
            if aircraft_states[aircraft]['flights_since_maint'] >= MAX_FLIGHTS_BETWEEN_MAINTENANCE:
                if dest in MAINTENANCE_AIRPORTS:
                    maintenance_schedule[day].append((aircraft, dest))
                    total_cost += MAINTENANCE_COST
                    aircraft_states[aircraft]['flights_since_maint'] = 0
                else:
                    # Will exceed limit - severe penalty
                    penalties += 10000
            
        # Check maintenance capacity constraint
        maintenance_at_Y = sum(1 for a, loc in maintenance_schedule[day] if loc == 'Y')
        maintenance_at_Z = sum(1 for a, loc in maintenance_schedule[day] if loc == 'Z')
        
        if maintenance_at_Y > 1:
            penalties += 5000 * (maintenance_at_Y - 1)
        if maintenance_at_Z > 1:
            penalties += 5000 * (maintenance_at_Z - 1)
    
    # Check if any aircraft ends with flights_since_maint > MAX
    for aircraft, state in aircraft_states.items():
        if state['flights_since_maint'] > MAX_FLIGHTS_BETWEEN_MAINTENANCE:
            penalties += 10000
    
    fitness = -(total_cost + penalties)
    return fitness

# GA Parameters
num_generations = 50000
num_parents_mating = 10
sol_per_pop = 50
num_genes = 5
gene_space = [0, 1]

# Create GA instance
ga_instance = pygad.GA(
    num_generations=num_generations,
    num_parents_mating=num_parents_mating,
    fitness_func=fitness_function,
    sol_per_pop=sol_per_pop,
    num_genes=num_genes,
    gene_type=int,
    gene_space=gene_space,
    parent_selection_type="tournament",
    K_tournament=3,
    crossover_type="single_point",
    crossover_probability=0.8,
    mutation_type="random",
    mutation_probability=0.1,
    keep_elitism=2
)

# Run GA
ga_instance.run()

# Get best solution
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Best Solution:", solution)
print("Best Fitness (negative cost):", solution_fitness)
print("Total Cost:", -solution_fitness)

# Decode and display the schedule
print("\n=== Optimal Schedule ===")
aircraft_states = {
    'A1': {'location': 'X', 'flights_since_maint': 0},
    'A2': {'location': 'X', 'flights_since_maint': 0}
}

total_cost = 0

for day_idx, day_flights in enumerate(required_flights):
    day = day_idx + 1
    print(f"\n--- Day {day} ---")
    
    if solution[day_idx] == 0:
        assignments = {'A1': day_flights[0], 'A2': day_flights[1]}
    else:
        assignments = {'A1': day_flights[1], 'A2': day_flights[0]}
    
    for aircraft in ['A1', 'A2']:
        flight_name = assignments[aircraft]
        flight_info = flights[flight_name]
        origin = flight_info['origin']
        dest = flight_info['dest']
        cost = flight_info['cost']
        
        current_loc = aircraft_states[aircraft]['location']
        
        # Repositioning if needed
        if current_loc != origin:
            repo_flight, repo_cost = find_repositioning_flight(current_loc, origin)
            if repo_flight:
                print(f"  {aircraft}: {repo_flight} ({current_loc}→{origin}) [Reposition] Cost: {repo_cost}")
                total_cost += repo_cost
                aircraft_states[aircraft]['flights_since_maint'] += 1
                current_loc = origin
        
        # Required flight
        print(f"  {aircraft}: {flight_name} ({origin}→{dest}) Cost: {cost}")
        total_cost += cost
        aircraft_states[aircraft]['location'] = dest
        aircraft_states[aircraft]['flights_since_maint'] += 1
        
        # Maintenance check
        if aircraft_states[aircraft]['flights_since_maint'] >= MAX_FLIGHTS_BETWEEN_MAINTENANCE:
            if dest in MAINTENANCE_AIRPORTS:
                print(f"  {aircraft}: Maintenance at {dest} (Cost: {MAINTENANCE_COST})")
                total_cost += MAINTENANCE_COST
                aircraft_states[aircraft]['flights_since_maint'] = 0

print(f"\n=== Total Operational Cost: {total_cost} ===")
