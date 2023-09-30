import os
import random
import imageio
import matplotlib.pyplot as plt

# Algorithm parameters
# POPULATION_SIZE: Defines the number of individuals in a population E.g. 10 indicates the population will contain 10 individuals (chromosomes).
# GENERATION_COUNT: Defines the number of generations. E.g. 2000 indicates the alforithm will be executed 2000 times.
# MUTATION_RATE: Defines the probability of an individual being mutated. E.g. 0.01 indicates 1% probability 
# RESOLUTION: Defines the number of (x,y) pairs in an individual. E.g. 200 indicates every individual in the population will contain 200 (x,y) pairs (genes).
POPULATION_SIZE = 10
GENERATION_COUNT = 3000
MUTATION_RATE = 0.01
RESOLUTION = 200

# Calculates the fitness score for the specified individual (chromosome).
# In this problem, our objective is to find the best set of (x,y) pairs (individual or chromosome) that will converge to a circle shape.
# Since we're trying to approximate a unit circle (r:1, centre: 0,0), we can use the special form of circle equation (x^2 + y^2 = r^2) to calculate the fitness.
# Our objective is to minimise the deviation of an individual's (x,y) coordinates from the circle equation.
def fitness(individual):
    # calculate the sum of deviation for each (x,y) pair in an individual (chromosome).
    # We're getting square of the circle equation to get rid of negative values since we're only interested in minimizing the deviation, not the position relative to the circle.
    # The smallest the value, the fittest the individual (chromosome)
    # E.g. [0, 1, 2, 3, 4] First element is the fittest (exactly on the circle). Last element is the least fit.
    return sum((x**2 + y**2 - r**2)**2 for x, y in individual)

# Mutates [x,y] pairs (gene) of the individual (chromosome) based on the parameterised mutation rate.
# If randomly chosen number (from a uniform distribution) is less than the parameterised mutation rate, the [x,y] pairs are mutated. Otherwise, the pairs are kept as they are.
# Mutation: Both x and y values are increased/decreased by a randomly chosen number (from gaussian distribution, mean:0, sigma:0.1).
def mutate(individual):
    return [(x + random.gauss(0, 0.1), y + random.gauss(0, 0.1)) if random.random() < MUTATION_RATE else (x, y) for x, y in individual]

# Performs single point crossover on specified individuals (parents) to increase the diversity of the population.
# [x,y] pairs (genes) of individuals (parents/chromosomes) are exchanged from a randomly selected point.
# Example:
# Parent-1: [(x1,y1), (x2,y2), (x3, y3)] 
# Parent-2: [(x4,y4), (x5,y5), (x6, y6)]
# After crossover from index 1 (crossover point):
# Parent-1: [(x1,y1), (x2,y2), (x6, y6)] 
# Parent-2: [(x4,y4), (x5,y5), (x3, y3)]  
def crossover(parent1, parent2):
    point = random.randint(0, RESOLUTION - 1)
    return parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]

# Calculates fitness of each individial in the specified population and sorts the individuals by their fitness score in ascending order.
def select_population(population):
    scores = [(fitness(ind), i) for i, ind in enumerate(population)]
    scores.sort()

    selected = [population[i] for _, i in scores]

    return selected

# Plots the [x,y] pairs of the specified individual of specified generation and saves the plot into the disk.
def plot_individual(individual, generation):
    x_coords, y_coords = zip(*individual)
    plt.title(f"Generation {generation}")
    plt.scatter(x_coords, y_coords)
    # plt.legend()
    plt.axis("equal")
    plt.savefig(f"plots\plot_{generation}.png")
    plt.close()

# Sets up the temporary image directory
def setup():
    if not os.path.exists("plots"):
        os.mkdir("plots")

# Clears the temporary directory and the plot files.
def teardown():
    if os.path.exists("plots"):
        for file in os.listdir("plots"):
            os.remove(os.path.join("plots", file))
        os.rmdir("plots")

# Prompts user for radius value
r = float(input("Enter the radius of the circle: "))

# Main algorithm function
def genetic_algorithm(fn_setup, fn_teardown):
    # Prepares the environment before executing the algorithm
    fn_setup()

    # Image file list
    images = []

    # Creates the initial population randomly using coordinates from the problem space.
    # This isn't a deterministic. Initial population generation is completely random, meaning that a different population will be generated in every execution. 
    population = [[(random.uniform(-r, r), random.uniform(-r, r)) for _ in range(RESOLUTION)] for _ in range(POPULATION_SIZE)]

    # Execute the algorithm until reaching the desired number of generations. 
    for generation in range(GENERATION_COUNT):
        # calculate individual fitness scores and sort by the fitness.
        population = select_population(population)
        # Choose two best fit individual as parents from the population (Elitist approach). 
        parent1, parent2 = population[0], population[1]
        # Perform crossover to the chosen parents and return two offsprings.
        offspring1, offspring2 = crossover(parent1, parent2)
        # Perform mutation operation to offsprings and replace them with the least fit individuals in the population.
        population[-1] = mutate(offspring1)
        population[-2] = mutate(offspring2)

        # This code block is not relevant to the algorithm itself. 
        # It serves the purpose of generating the plot of best fit individual of a generation. 
        if generation % 20 == 0 or generation == GENERATION_COUNT - 1:
            plot_individual(population[-1], generation + 1)
            images.append(imageio.imread(f'plots\plot_{generation + 1}.png'))
        print("Generation: " + str(generation + 1), end="\r")

    # Create GIF file from the images and save to the disk.
    imageio.mimsave('generations.gif', images, duration=0.5)
    
    # Cleans up the environment after algorithm completes.
    fn_teardown()

# Invoke the main algorithm function
genetic_algorithm(setup, teardown)



