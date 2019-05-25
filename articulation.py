import sys
import time
import networkx as nx
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql import functions
from graphframes import *
from copy import deepcopy

sc=SparkContext("local", "degree.py")
sqlContext = SQLContext(sc)

def articulations(g, usegraphframe=False):
    # Get the starting count of connected components
    initial_count = g.connectedComponents().select('component').distinct().count()
    # Default version sparkifies the connected components process
    # and serializes node iteration.
    if usegraphframe:
	# Get vertex list for serial iteration
	# YOUR CODE HERE
        vertex_list = g.vertices.map(lambda x: x.id).collect()
        articulation = []
	# For each vertex, generate a new graphframe missing that vertex
	# and calculate connected component count. Then append count to
	# the output
        for vertex in vertex_list:
            graph = GraphFrame(g.vertices.filter('id != "'+ vertex +'"'), g.edges.filter('src != "' + vertex +'"').filter('dst !="'+ vertex + '"'))
            comp = graph.connectedComponents().select('component').distinct().count()
            articulation.append((vertex, 1 if comp> initial_count else 0))

        return sqlContext.createDataFrame(sc.parallelize(articulation),['id','articulation'])
        # Non-default version sparkifies node iteration and uses networkx
	# for connected components count.
    else:
        graphx = nx.Graph()
        graphx.add_nodes_from(g.vertices.map(lambda x: x.id).collect())
        graphx.add_edges_from(g.edges.map(lambda x: (x.src, x.dst)).collect())
        def components(node):
            graph = deepcopy(graphx)
            graph.remove_node(node)
            return nx.number_connected_components(graph)
        return sqlContext.createDataFrame(g.vertices.map(lambda x: (x.id, 1 if components(x.id) > initial_count else 0)), ['id', 'articulation'])


filename = sys.argv[1]
lines = sc.textFile(filename)

pairs = lines.map(lambda s: s.split(","))
e = sqlContext.createDataFrame(pairs,['src','dst'])
e = e.unionAll(e.selectExpr('src as dst','dst as src')).distinct() # Ensure undirectedness

# Extract all endpoints from input file and make a single column frame.
v = e.selectExpr('src as id').unionAll(e.selectExpr('dst as id')).distinct()

# Create graphframe from the vertices and edges.
g = GraphFrame(v,e)

#Runtime approximately 5 minutes
print("---------------------------")
print("Processing graph using Spark iteration over nodes and serial (networkx) connectedness calculations")
init = time.time()
df = articulations(g, False)
print("Execution time: %s seconds" % (time.time() - init))
print("Articulation points:")
df.filter('articulation = 1').show(truncate=False)
print("---------------------------")

#Runtime for below is more than 2 hours
print("Processing graph using serial iteration over nodes and GraphFrame connectedness calculations")
init = time.time()
df = articulations(g, True)
print("Execution time: %s seconds" % (time.time() - init))
print("Articulation points:")
df.filter('articulation = 1').show(truncate=False)
