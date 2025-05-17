from sentence_transformers import SentenceTransformer
import networkx as nx
import numpy as np
from pyvis.network import Network # Just to create visuals, you wont need this in actual app.

s = [
    "I love taking long walks on the beach during sunset.",
    "The quick brown fox jumps over the lazy dog.",
    "Artificial Intelligence is transforming the world rapidly.",
    "She sells seashells by the seashore each summer.",
    "The Amazon rainforest is home to an incredible diversity of species.",
    "Honesty is the best policy.",
    "Is there any financial aid or scholarship information I can look into?",
    "Actually, could you tell me about the student life on campus too?",
    "I'm also curious about campus housing options—are freshmen required to live on campus?",
    # Topic switch
    "By the way, do you know any good recommendations for online courses in data science?",
    "How long does it typically take to complete one of those courses?",
    # Back to the previous topic
    "And for the application process, should I contact professors directly?",
    "Are there any specific deadlines I should be aware of?",
    # Short topic switch
    "Oh, and do you know any good study strategies for standardized tests?",
    # Back to application requirements
    "So, should I submit my test scores along with my application?",
    "What about letters of recommendation—how many do I need?",
    # Topic switch
    "Could you recommend some good resources for learning programming as a beginner?",
    "What's a good first language to learn if I'm new to programming?",
    "How important is it to understand algorithms from the start?",
    # Returning to admissions questions
    "Just to confirm, is there a way to track the application status once it's submitted?",
    "Can I update my application with new achievements after submitting?",
    # Random question
    "Oh, do you know if any universities allow pets on campus?"
]

model = SentenceTransformer("all-MiniLM-L6-v2")


embeddings = model.encode(s)

# Compute cosine similarities
similarities = np.asarray(model.similarity(embeddings, embeddings))

# We convert the similarity matrix into strict upper triangular matrix, We want a message to only point to only previous messages.
# This will come in handy later in graph creation.  
for i in range(len(similarities)):
    similarities[i, i] = 0
    for j in range(i):
        similarities[i, j] = 0

# Convert the similarities matrix to a graph data dictionary.
# We attach all previous sentences with any similarity to the current one. 
threshold = 0.2
graph_data = {}
for i in range(len(similarities)):
    edges = [
        (j + 1, round(similarities[j, i], 2))
        for j in range(len(similarities))
        if similarities[j, i] > 0 # This filters those with any similarity between messages
        # if similarities[j, i] > threshold # This filters usign threshold of similarity between messages
    ]
    graph_data[i + 1] = edges

print("*" * 50)
print(graph_data)
print("*" * 50)

# Initialize a directed graph
G = nx.DiGraph()

# Add edges with weights
for node, edges in graph_data.items():
    for edge in edges:
        target, weight = edge
        G.add_edge(node, target, weight=weight)


# Greedy method to find a path from one node to another based on highest weight (which is simply cosine similarity).
def greedy_path_high_weight(graph, start, goal):
    current_node = start
    path = [current_node]
    while current_node != goal:
        neighbors = list(graph[current_node].items())
        if not neighbors:
            return path  # No path if there are no neighbors
        # Select the edge with the highest weight
        next_node = max(neighbors, key=lambda x: x[1]["weight"])[0]
        path.append(next_node)
        current_node = next_node
        if current_node == goal:
            return path
    return path

# len(s) is the latest message and we try go back till 1st
greedy_path_result = greedy_path_high_weight(G, len(s), 1) 

print(greedy_path_result)


# This part is just for the purpose of visuals, you dont need it, 'greedy_path_result' is what should be used for llm calls.
 
pos = nx.spring_layout(G)
# Create an interactive Pyvis Network graph with increased repulsion 
net = Network(height="800px", width="100%", notebook=False, directed=True)

# Set physics options to increase spacing between nodes
net.force_atlas_2based(
    gravity=-70, central_gravity=0.04, spring_length=200, spring_strength=0.001
)
net.toggle_physics(True)  # Enable physics to apply the layout

for i, sentence in enumerate(s, start=1):
    net.add_node(
        i,
        label=str(i),
        title=sentence,
        font={"size": 20, "align": "center"},
        shape="circle",
    )  # Ensure centered alignment

for u, v, data in G.edges(data=True):
    weight = float(data["weight"])  # Convert to standard float type
    weight = (weight * 100) / 100.0  # Truncate to two decimals
    if (u, v) in zip(greedy_path_result, greedy_path_result[1:]):
        color = "red"
        width = 10  # Increase width for red edges
    else:
        color = "gray"
        width = 2  # Standard width for other edges

    net.add_edge(u, v, color=color, title=f"Weight: {weight:.2f}", width=width)

net.save_graph("interactive_graph_without_threshold.html")
