import tensorflow as tf

#from tensorflow.python.platform import gfile
from tensorflow.python.platform import gfile

GRAPH_PB_PATH = './models/20180402-114759.pb'
with tf.Session() as sess:
   print("load graph")
   with gfile.FastGFile(GRAPH_PB_PATH,'rb') as f:
       graph_def = tf.GraphDef()
   graph_def.ParseFromString(f.read())
   sess.graph.as_default()
   tf.import_graph_def(graph_def, name='')
   graph_nodes=[n for n in graph_def.node]
   names = []
   print(graph_nodes[0])
   for t in graph_nodes:
       if t.input :
           print(t)
           break
       #names.append(t)
      #names.append(t.name + '\n')
   #print(names)
   print(graph_nodes[-1])