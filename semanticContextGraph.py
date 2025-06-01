from sentence_transformers import SentenceTransformer, CrossEncoder
import networkx as nx
import numpy as np
from pyvis.network import Network

class SemanticContextGraph:
    def __init__(self, chat_messages, model_name="all-MiniLM-L6-v2", mode="embedding"):
        self.chat_messages = chat_messages
        self.sentences = [msg["content"] for msg in chat_messages]
        self.roles = {i: msg["role"] for i, msg in enumerate(chat_messages)}
        self.mode = mode.lower()

        if self.mode == "embedding":
            self.model = SentenceTransformer(model_name)
        elif self.mode == "cross":
            self.model = CrossEncoder(model_name)
        else:
            raise ValueError("mode must be either 'embedding' or 'cross'")

        self.graph = nx.DiGraph()
        self.similarity_matrix = self._compute_strict_upper_similarity()
        # print(self.similarity_matrix.tolist())

    def _compute_strict_upper_similarity(self):
        n = len(self.sentences)
        sim_matrix = np.zeros((n, n))

        if self.mode == "embedding":
            embeddings = self.model.encode(self.sentences)
            sim_matrix = np.asarray(self.model.similarity(embeddings, embeddings))
            for i in range(n):
                sim_matrix[i, i] = 0
                for j in range(i):
                    sim_matrix[i, j] = 0
        else:  # mode == "cross"
            for i in range(n):
                for j in range(i + 1, n):
                    score = self.model.predict([[self.sentences[i], self.sentences[j]]])[0]
                    sim_matrix[i, j] = score  # i = earlier, j = later

        return sim_matrix

    def build_graph(self, threshold=0.0):
        for i in range(len(self.similarity_matrix)):
            for j in range(i + 1, len(self.similarity_matrix)):
                score = self.similarity_matrix[i, j]
                if score > threshold:
                    self.graph.add_edge(j + 1, i + 1, weight=round(score, 2))  # from current (j) ➝ past (i)
        print(self.graph.edges)

    def greedy_path(self, start_node, goal_node):
        current = start_node
        path = [current]
        while current != goal_node:
            if current not in self.graph:
                return path
            neighbors = list(self.graph[current].items())
            if not neighbors:
                break
            next_node = max(neighbors, key=lambda x: x[1]["weight"])[0]
            if next_node == current:
                break
            path.append(next_node)
            current = next_node
        return path

    def extract_relevant_messages(self, path):
        path = list(reversed(path))
        used_ids = []
        selected = []

        def add_if_new(idx):
            if idx not in used_ids and 0 <= idx < len(self.sentences):
                used_ids.append(idx)
                selected.append({"role": self.roles[idx], "content": self.sentences[idx]})

        for node in path:
            idx = node - 1
            if self.roles.get(idx) == "user":
                add_if_new(idx)
                add_if_new(idx + 1)
            elif self.roles.get(idx) == "assistant":
                add_if_new(idx - 1)
                add_if_new(idx)

        return used_ids, selected

    def save_pyvis_graph(self, highlight_path=None, file_name="interactive_graph.html"):
        net = Network(height="800px", width="100%", directed=True)
        net.force_atlas_2based(gravity=-70, central_gravity=0.04, spring_length=200, spring_strength=0.001)
        net.toggle_physics(True)

        for i, sentence in enumerate(self.sentences, start=1):
            net.add_node(
                i,
                label=str(i),
                title=f"{self.roles[i-1]}: {sentence}",
                font={"size": 20, "align": "center"},
                shape="circle"
            )

        for u, v, data in self.graph.edges(data=True):
            weight = float(data["weight"])
            color = "gray"
            width = 2
            if highlight_path and (u-1, v-1) in zip(highlight_path[1:], highlight_path):
                color = "red"
                width = 10
            net.add_edge(u, v, color=color, title=f"Weight: {weight:.2f}", width=width)

        net.save_graph(file_name)


chat_messages = [
    {"role": "user", "content": "How do I start preparing for a career in robotics?"},
    {"role": "assistant", "content": "You can begin with mechanical basics, then move into programming and embedded systems."},
    {"role": "user", "content": "Which language is preferred—C++ or Python?"},
    {"role": "assistant", "content": "Python is great for prototyping, but C++ is essential for performance-critical robotics."},
    {"role": "user", "content": "Any tips on improving productivity while studying at home?"},
    {"role": "assistant", "content": "Use Pomodoro timers, block distractions, and keep a dedicated study space."},
    {"role": "user", "content": "Is it effective to work out without going to the gym?"},
    {"role": "assistant", "content": "Absolutely. Bodyweight exercises and consistency can be very effective."},
    {"role": "user", "content": "Are there open-source platforms where I can practice robotics programming?"},
    {"role": "assistant", "content": "Yes, check out ROS (Robot Operating System) and simulators like Gazebo or Webots."},
    {"role": "user", "content": "What’s a simple dinner recipe I can cook in under 30 minutes?"},
    {"role": "assistant", "content": "Try stir-fried vegetables with tofu and rice—it’s quick and nutritious."},
    {"role": "user", "content": "Is it necessary to know calculus for AI research?"},
    {"role": "assistant", "content": "It helps with understanding backpropagation and optimization, but you can get started without mastering it."},
    {"role": "user", "content": "Should I start with ROS or learn basic electronics first if I want to build small autonomous vehicles?"}
]

threshold=0.3
print("*"*40)
print("Cross encoder")
graph = SemanticContextGraph(chat_messages, model_name="cross-encoder/stsb-roberta-base", mode="cross")
graph.build_graph(threshold=threshold)
path = graph.greedy_path(start_node=len(chat_messages), goal_node=1)
path_recreated, messages = graph.extract_relevant_messages(path)

if threshold == 0.0:
    graph.save_pyvis_graph(highlight_path=path_recreated, file_name="interactive_graph_without_threshold.html")
else:
    graph.save_pyvis_graph(highlight_path=path_recreated, file_name="interactive_graph_with_threshold.html")

for msg in messages:
    print(f"{msg['role'].upper()}: {msg['content']}")


print("*"*40)
print("Cosine similarity encoder")
graph = SemanticContextGraph(chat_messages, model_name="all-MiniLM-L6-v2", mode="embedding")
graph.build_graph(threshold=threshold)
path = graph.greedy_path(start_node=len(chat_messages), goal_node=1)
path_recreated, messages = graph.extract_relevant_messages(path)

if threshold == 0.0:
    graph.save_pyvis_graph(highlight_path=path_recreated, file_name="interactive_graph_without_threshold.html")
else:
    graph.save_pyvis_graph(highlight_path=path_recreated, file_name="interactive_graph_with_threshold.html")

for msg in messages:
    print(f"{msg['role'].upper()}: {msg['content']}")
