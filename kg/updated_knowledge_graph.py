import csv
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import random
import os

def create_knowledge_graph(csv_file):
    """
    Creates a knowledge graph from the given CSV file.
    Returns the graph and a dictionary of symptoms with their associated diseases.
    """
    G = nx.Graph()
    symptoms_dict = defaultdict(set)

    try:
        with open(csv_file, 'r', encoding='utf-8') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                symptom = row['Symptom'].strip()
                diseases = [d.strip() for d in row['Associated_Diseases'].split(',') if d.strip()]

                # Add symptom node even if there are no associated diseases
                G.add_node(symptom, node_type='symptom')

                for disease in diseases:
                    G.add_node(disease, node_type='disease')
                    G.add_edge(symptom, disease)
                    symptoms_dict[symptom].add(disease)

                # If no diseases are associated, add this information to the symptoms_dict
                if not diseases:
                    symptoms_dict[symptom].add("No associated diseases")

        print(f"Processed {len(symptoms_dict)} symptoms.")
        print(f"Total nodes in graph: {G.number_of_nodes()}")
        print(f"Total edges in graph: {G.number_of_edges()}")

    except FileNotFoundError:
        print(f"Error: File '{csv_file}' not found.")
        return None, None
    except csv.Error as e:
        print(f"Error reading CSV file: {e}")
        return None, None
    except KeyError as e:
        print(f"Error: Expected column {e} not found in the CSV file.")
        return None, None

    return G, symptoms_dict

def save_new_knowledge_graph_relations(G, output_csv):
    """
    Saves the new relations from the knowledge graph to a CSV file.
    This CSV can be used for further steps in chatbot retrieval.
    """
    try:
        with open(output_csv, 'w', newline='', encoding='utf-8') as file_out:
            csv_writer = csv.writer(file_out)
            csv_writer.writerow(['Symptom', 'Disease', 'Relation'])

            for symptom, data in G.nodes(data=True):
                if data['node_type'] == 'symptom':
                    diseases = list(G.neighbors(symptom))
                    if diseases:
                        for disease in diseases:
                            csv_writer.writerow([symptom, disease, 'has_symptom'])
                    else:
                        csv_writer.writerow([symptom, 'No associated diseases', 'has_no_diseases'])

        print(f"New knowledge graph relations saved to '{output_csv}'.")
    except IOError as e:
        print(f"Error saving knowledge graph relations: {e}")

def visualize_graph(G, output_file):
    """
    Visualizes the full knowledge graph and saves it as an image.
    """
    plt.figure(figsize=(20, 20))
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    
    symptom_nodes = [node for node, data in G.nodes(data=True) if data['node_type'] == 'symptom']
    disease_nodes = [node for node, data in G.nodes(data=True) if data['node_type'] == 'disease']
    
    nx.draw_networkx_nodes(G, pos, nodelist=symptom_nodes, node_color='lightgreen', node_size=500, alpha=0.8, label='Symptoms')
    nx.draw_networkx_nodes(G, pos, nodelist=disease_nodes, node_color='lightblue', node_size=300, alpha=0.6, label='Diseases')
    nx.draw_networkx_edges(G, pos, alpha=0.2)
    
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
    
    plt.title("Symptom-Disease Knowledge Graph", fontsize=20)
    plt.axis('off')
    plt.legend(scatterpoints=1)
    plt.tight_layout()
    
    try:
        plt.savefig(output_file, format="png", dpi=300, bbox_inches='tight')
        print(f"Full knowledge graph visualization saved as '{output_file}'.")
    except Exception as e:
        print(f"Error saving graph visualization: {e}")
    
    plt.close()

def visualize_single_symptom(G, symptom, output_file):
    """
    Visualizes a single symptom and its associated diseases.
    """
    if symptom not in G:
        print(f"Symptom '{symptom}' not found in the graph.")
        return

    subgraph = nx.ego_graph(G, symptom, radius=1)
    pos = nx.spring_layout(subgraph, k=0.9, iterations=50)
    
    plt.figure(figsize=(15, 15))
    
    nx.draw_networkx_nodes(subgraph, pos, nodelist=[symptom], node_color='#FF6B6B', node_size=3000, alpha=0.8, label='Symptom')
    disease_nodes = [node for node in subgraph.nodes() if node != symptom]
    nx.draw_networkx_nodes(subgraph, pos, nodelist=disease_nodes, node_color='#4ECDC4', node_size=1500, alpha=0.6, label='Diseases')
    nx.draw_networkx_edges(subgraph, pos, alpha=0.5, edge_color='#45B7D1', width=1)
    
    nx.draw_networkx_labels(subgraph, pos, {symptom: symptom}, font_size=14, font_weight='bold')
    nx.draw_networkx_labels(subgraph, pos, {node: node for node in disease_nodes}, font_size=10)
    
    edge_labels = {(symptom, disease): "has_disease" for disease in disease_nodes}
    nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=edge_labels, font_size=8)
    
    plt.title(f"Symptom: {symptom} and its Associated Diseases", fontsize=20, fontweight='bold')
    plt.axis('off')
    plt.legend(scatterpoints=1)
    plt.tight_layout()
    
    try:
        plt.savefig(output_file, format="png", dpi=300, bbox_inches='tight')
        print(f"Single symptom graph for '{symptom}' saved as '{output_file}'.")
    except Exception as e:
        print(f"Error saving single symptom graph: {e}")
    
    plt.close()

def print_statistics(G):
    """
    Prints statistics about the knowledge graph.
    """
    symptom_nodes = [node for node, data in G.nodes(data=True) if data['node_type'] == 'symptom']
    disease_nodes = [node for node, data in G.nodes(data=True) if data['node_type'] == 'disease']
    
    print(f"\nGraph Statistics:")
    print(f"-----------------")
    print(f"Number of symptom nodes: {len(symptom_nodes)}")
    print(f"Number of disease nodes: {len(disease_nodes)}")
    print(f"Number of edges (connections): {G.number_of_edges()}")

    disease_symptom_count = [(disease, len(list(G.neighbors(disease)))) for disease in disease_nodes]
    top_diseases = sorted(disease_symptom_count, key=lambda x: x[1], reverse=True)[:10]
    
    print("\nTop 10 Most Common Diseases:")
    for idx, (disease, count) in enumerate(top_diseases, start=1):
        print(f"{idx}. {disease}: {count} symptoms")

    # Print symptoms with no associated diseases
    symptoms_no_diseases = [symptom for symptom in symptom_nodes if G.degree(symptom) == 0]
    print(f"\nNumber of symptoms with no associated diseases: {len(symptoms_no_diseases)}")
    if symptoms_no_diseases:
        print("Examples of symptoms with no associated diseases:")
        for symptom in symptoms_no_diseases[:5]:  # Print first 5 examples
            print(f"- {symptom}")

def main():
    input_csv = r'C:\Users\Asus\Desktop\SEMESTER_7\data\grouped_symptoms_diseases.csv'
    output_csv_knowledge_graph = 'new_symptom_disease_relations.csv'
    full_graph_output = 'full_symptom_disease_knowledge_graph.png'
    single_symptom_output = 'single_symptom_graph.png'

    if not os.path.exists(input_csv):
        print(f"Input CSV file '{input_csv}' not found. Please ensure the file exists in the specified path.")
        return

    G, symptoms = create_knowledge_graph(input_csv)
    if G is None or symptoms is None:
        return

    save_new_knowledge_graph_relations(G, output_csv_knowledge_graph)
    visualize_graph(G, full_graph_output)

    symptom_nodes = [node for node, data in G.nodes(data=True) if data['node_type'] == 'symptom']
    if symptom_nodes:
        random_symptom = random.choice(symptom_nodes)
        visualize_single_symptom(G, random_symptom, single_symptom_output)
    else:
        print("No symptom nodes found in the graph to visualize.")

    print_statistics(G)

if __name__ == "__main__":
    main()
