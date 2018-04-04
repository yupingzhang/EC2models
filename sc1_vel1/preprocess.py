############# preprocess ###################################
# class Vertex: 
# class Edge:
# class Face:
#
# output configuration:
# can output per vertex, per edge, or per face attributes.
#
#
############################################################
import math
import os, sys
import numpy as np
from numpy import linalg as LA
import math
import time

MAXINT = 1000


class Vertex():    
    pos = []
    vel = []

    edgeList = []
    faceList = []

    def __init__(self, p):
        self.pos = p

    def add_vel(self, velocity):
        self.vel = velocity

    def add_edge(self, e):
        self.edgeList.append(e)

    def add_face(self, f):
        self.faceList.append(f)


class Edge():
    # vertex index
    idx1 = 0
    idx2 = 0
    tri_list = [] 

    #ratio of edge length
    rest_length = 0.0
    length = 0.0
    deform_ratio = 0.0

    def __init__(self, id1, id2, tid, l):
        self.idx1 = id1 if id1 < id2 else id2
        self.idx2 = id2 if id1 < id2 else id1
        self.tri_list.append(tid)
        self.length = l

    def is_same_edge(self, e1):
        if e1.idx1 == self.idx1 and e1.idx2 == self.idx2:
            return True
        else:
            return False



class Face():
    tri_id = -1
    tri_vert = []   # three id
    tri_angles = []  # three edge objects
    tri_cos = []
    tri_area = 0.0

    # deformation data
    deform_grad = []
    deform_angle_ratio = []
    deform_cos_ratio = []
    
    def __init__(self, id0, id1, id2, a0, a1, a2, ca0, ca1, ca2, face_idx, area):
        self.tri_id = face_idx
        self.tri_vert = [id0, id1, id2]
        self.tri_angles = [a0, a1, a2]
        self.tri_cos = [ca0, ca1, ca2]
        self.tri_area = area


# vert --> list of objects
# edges --> dictionary  (v1, v2) vertex index pair as key.
# faces --> dictionary  face_index as key.
# store all the properties within a object
def obj_loader(file_name, rest_pos=[]):
    if not os.path.isfile(file_name):
        print("file not exist")
        return

    vert = []
    edges = {}
    faces = {}

    face_idx = 0
    vert_idx = -1

    with open(file_name, "r") as f1:
        for line in f1:
            s = line.strip().split(' ')
            if s[0] == 'v':
                vert_idx += 1
                v = list(map(float, s[1:]))
                if rest_pos:
                    v = list(map(operator.sub, v, rest_pos[vert_idx]))
                vert.append(Vertex(v))
            elif s[0] == 'nv':
                nv = list(map(float, s[1:]))
                vert[vert_idx].add_vel(nv)
            elif s[0] == 'f':
                id0 = int(s[1].strip().split('/')[0]) - 1  # index start at 0
                id1 = int(s[2].strip().split('/')[0]) - 1
                id2 = int(s[3].strip().split('/')[0]) - 1
                # add to the edge dictionary
                v = sorted([id0, id1, id2])
                # compute some properties
                p0 = vert[v[0]].pos 
                p1 = vert[v[1]].pos
                p2 = vert[v[2]].pos         
                # edge vectors
                v0 = np.array(p2) - np.array(p1)           
                v1 = np.array(p0) - np.array(p2)
                v2 = np.array(p1) - np.array(p0)
                # edge distance
                dist0 = LA.norm(v0)
                dist1 = LA.norm(v1)
                dist2 = LA.norm(v2)
                # cos
                ca0 = LA.norm(np.dot(-v1, v2)) 
                ca1 = LA.norm(np.dot(v0, -v2)) 
                ca2 = LA.norm(np.dot(-v0, v1))
                # angle
                a0 = math.acos(ca0)
                a1 = math.acos(ca1)
                a2 = math.acos(ca2)
                # triangle area
                area = 0.5 * LA.norm(np.cross(v1, v2))

                # add edges
                if not edges.get((v[0], v[1])):
                    edges[(v[0], v[1])] = Edge(v[0], v[1], face_idx, dist2)
                else:
                    edges[(v[0], v[1])].tri_list.append(face_idx)
                
                if not edges.get((v[1], v[2])):
                    edges[(v[1], v[2])] = Edge(v[1], v[2], face_idx, dist0)
                else:
                    edges[(v[1], v[2])].tri_list.append(face_idx)
                
                if not edges.get((v[0], v[2])):
                    edges[(v[0], v[2])] = Edge(v[0], v[2], face_idx, dist1)
                else:
                    edges[(v[0], v[2])].tri_list.append(face_idx)
                
                # add to face list
                faces[face_idx] = Face(id0, id1, id2, a0, a1, a2, ca0, ca1, ca2, face_idx, area)
                face_idx += 1

    print("how many vertices? ", len(vert))
    print("how many edges? ", len(edges))
    print("how many faces? ", len(faces))

    return vert, edges, faces


# get position array from Vertex object array
def get_pos(vert):
    pos = []
    for x in vert:
        pos.append(x.pos)
    return pos


# call this after get the base info from rest pos
def update_deformation(edges, rest_edges, faces, rest_faces):
    for x in edges:
        x.rest_length = rest_edges.get(x.idx1, x.idx2).length
        x.ratio = x.length / x.rest_length

    # TODO
    # for x in faces:
    #     x.deform_grad = 
    #     x.deform_angle_ratio =
    #     x.deform_cos_ratio =
    




# find the vertex that an edge is facing in a triangle
# return vertex index
def vert_for_edge(tri, edge):
    vertices = tri.tri_vert
    for v in vertices:
        if edge.idx1 != v and edge.idx2 != v:
            return v


# find other two edges in current triangle besides given edge
def other_two_edges(tri, e):
    e_list = tri.tri_edges
    other_e = []
    for item in e_list:
        if not is_same_edge(item, e):
            other_e.append(item)
    return other_e


# input: vert & faces
# output tri_nb: local vertices per row * tri_num
# n: per tri info data, 3 for one triangle position only, 6 for triangle with one layer neighbor, x... with velocity etc...
def comp_mtx(vert, edges, faces, n=3):
    vert_num = len(vert)
    tri_num = len(faces)
    dim = [tri_num, vert_num]
    # print dim

    # mtx = np.array([np.zeros(tri_num*3) for item in range(vert_num)])
    mtx = np.array([np.zeros(tri_num * n) for item in range(vert_num)])
    count = np.zeros((vert_num, 1))
    # print ">>> mtx shape: ", mtx.shape

    # new_edges = []
    for i in range(0, tri_num):
        [id1, id2, id3] = faces[i].tri_vert
        # original vertex in index matrix
        mtx[id1][i * n] = 1
        mtx[id2][i * n + 1] = 1
        mtx[id3][i * n + 2] = 1
        count[id1][0] += 1.0
        count[id2][0] += 1.0
        count[id3][0] += 1.0

        # # for the neighbors, get shared edge and corresponding vertex
        # for j in range(0, len(faces[i].tri_edges)):
        #     ed = faces[i].tri_edges[j]
        #     # retrieve the tri_list for the dictionary
        #     shared_tri = edges[(ed.idx1, ed.idx2)]
        #     if len(shared_tri) > 1:
        #         other_tri = shared_tri[1] if shared_tri[0] == i else shared_tri[0]
        #         new_vert_id = vert_for_edge(faces[other_tri], ed)
        #         # add to index matrix
        #         # mtx[new_vert_id][i * 6 + 3 + j] = 1
        #         mtx[new_vert_id][i * n + 3 + j] = 1
        #         count[new_vert_id][0] += 1.0

    mtx_1 = mtx
    mtx = mtx_1 / count

    return dim, mtx, mtx_1


def find_neighbors(vert, edges, faces, n=1):
    vert_num = len(vert)
    tri_num = len(faces)
    tri_nb = [0] * tri_num
    # print(vert_num, tri_num, tri_nb)   700 x 1292
    # new_edges = []
    for i in range(0, tri_num):
        # print "i:{}".format(i)
        [id1, id2, id3] = faces[i].tri_vert
        # original vertex position
        tri_nb.extend(vert[id1])
        tri_nb.extend(vert[id2])
        tri_nb.extend(vert[id3])
        # while n > 0:
        #     n = n - 1
        # add neighbors
        for j in range(0, len(faces[i].tri_edges)):
            ed = faces[i].tri_edges[j]
            # retrieve the tri_list for the dictionary
            shared_tri = edges[(ed.idx1, ed.idx2)]
            if len(shared_tri) > 1:
                other_tri = shared_tri[1] if shared_tri[0] == i else shared_tri[0]
                new_vert_id = vert_for_edge(faces[other_tri], ed)
                tri_nb.extend(vert[new_vert_id])
                # new_edges.extend(other_two_edges(faces[other_tri], ed))
            else:
                tri_nb.extend([0.0, 0.0, 0.0])     # zero padding

    return tri_nb


def meshmtx_wnb(file_name):
    vert = []
    vel = []
    edges = {}
    faces = {}

    obj_loader(file_name, vert, vel, edges, faces)
    dim, mtx, mtx_1 = comp_mtx(vert, edges, faces)

    return dim, mtx, mtx_1


# config (string):  
# vert: pos, vel, 
# face: 
def load_sample(file_name, batch_data, config, rest_pos=[]):
    
    vert = []
    edges = {}
    faces = {}
    obj_loader(file_name, vert, edges, faces, rest_pos)

    # tri_nb = find_neighbors(vert, edges, faces)
    # TODO
    if config[:4] == 'vert':
        pass
    elif config[:4] == 'face':
        pass
    

    batch_data.append(output)


# =========================================================
# Dijkstra  
# =========================================================

# A utility function to find the vertex with 
# minimum distance value, from the set of vertices 
# not yet included in shortest path tree
def minDistance(num_v, dist, sptSet, src):

    # Initilaize minimum distance for next node
    min_dist_next = MAXINT
    min_index = -1

    # Search not nearest vertex not in the 
    # shortest path tree
    for v in range(0, num_v):
        if dist[v] < min_dist_next and sptSet[v] == False:
            min_dist_next = dist[v]
            min_index = v

    return min_index
 

# Funtion that implements Dijkstra's single source 
# shortest path algorithm for a graph represented 
# using adjacency matrix representation
# returns an array of distance start from the src
def dijkstra(num_v, graph, src):
    dist = [MAXINT] * num_v
    dist[src] = 0
    sptSet = [False] * num_v   # track visted notes

    # cout = 0
    while (False in sptSet):
        # cout += 1

        # Pick the minimum distance vertex from 
        # the set of vertices not yet processed. 
        # u is always equal to src in first iteration
        u = minDistance(num_v, dist, sptSet, src)    # return vertex index
        # if u < 0:
        #     print("Error: u < 0, cant find minDistance ??")
        #     continue;

        # Put the minimum distance vertex in the 
        # shotest path tree
        sptSet[u] = True

        # Update dist value of the adjacent vertices 
        # of the picked vertex only if the current 
        # distance is greater than new distance and
        # the vertex in not in the shotest path tree
        for v in range(num_v):
            if graph[u, v] > 0 and sptSet[v] == False and dist[v] > dist[u] + graph[u][v]:
                dist[v] = dist[u] + graph[u][v]
    
    # print("dijkstra iteration: ", cout). #700
    return dist


# compute geodesic matrix
def compute_geodesic_distance(num_v, edges):
    graph = np.zeros((num_v, num_v), dtype=np.float32)
    geo_dist = np.zeros((num_v, num_v), dtype=np.float32)

    # build the graph
    for k, e in edges.items():
        graph[e.idx1, e.idx2] = e.length   # e.idx1 always less than e.idx2
        graph[e.idx2, e.idx1] = e.length


    # build geo_dist matrix
    t = time.clock()
    for i in range(0, num_v):
        if i % 50 == 0:
            print(i/num_v, " % ")
        geo_dist[i, :] = dijkstra(num_v, graph, i)
    print (time.clock() - t, "seconds to compute geodesic distance matrix.")
   
    # debug
    # print("================ geo_dist ================")
    # print(geo_dist[0, :])

    return geo_dist
        

# mask matrix from vertex distance
def compute_mask(num_v, dist):
    dmin = 0.1
    dmax = 1.0 
    masking = np.zeros((num_v, num_v), dtype=np.float32)
    
    for i in range(num_v):
        for j in range(i, num_v):
            if dist[i, j] < dmin:
                w = 1.0
            elif dist[i, j] > dmax:
                w = 0.0
            else:
                w = (dmax - dist[i, j]) / (dmax - dmin)  
            masking[i, j] = w
            masking[j, i] = w

    # normalize each row to 1
    for x in range(num_v):
        rs = np.sum(masking[x, :])  # sum of the row
        # print("sum of row ", x, " >>> ", rs)
        masking[x, :] /= rs
    
    return masking


