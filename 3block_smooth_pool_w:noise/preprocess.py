# preprocess
import math
import os, sys
import numpy as np


class Edge(object):
    idx1 = 0
    idx2 = 0
    tri_list = []
    """docstring for Edge"""
    def __init__(self, id1, id2, tid):
        super(Edge, self).__init__()
        self.idx1 = id1 if id1 < id2 else id2
        self.idx2 = id2 if id1 < id2 else id1
        self.tri_list.append(tid)


class Tri(object):
    tri_id = -1
    tri_vert = []   # three id
    tri_edges = []  # three edge objects

    """docstring for Tri"""
    def __init__(self, id1, id2, id3, face_idx):
        super(Tri, self).__init__()
        self.tri_id = face_idx
        self.tri_vert = [id1, id2, id3]
        eg1 = Edge(id1, id2, face_idx)
        eg2 = Edge(id2, id3, face_idx)
        eg3 = Edge(id3, id1, face_idx)
        self.tri_edges = [eg1, eg2, eg3]


# vert, vel, faces --> list  
# edges(dict): key: vertex-pair(tuple), value: triangle ids sharing this edge
def obj_loader(file_name, vert, vel, edges, faces):
    face_idx = 0
    with open(file_name, "r") as f1:
        for line in f1:
            s = line.strip().split(' ')
            if s[0] == 'v':
                vert.append([float(x) for x in s[1:]])
            elif s[0] == 'nv':
                vel.extend([float(x) for x in s[1:]])
            elif s[0] == 'f':
                id1 = int(s[1].strip().split('/')[0]) - 1  # index start at 1
                id2 = int(s[2].strip().split('/')[0]) - 1
                id3 = int(s[3].strip().split('/')[0]) - 1
                # add to the edge dictionary
                v = sorted([id1, id2, id3])
                if not edges.get((v[0], v[1])):
                    edges[(v[0], v[1])] = [face_idx]
                else:
                    edges[(v[0], v[1])].append(face_idx)
                if not edges.get((v[1], v[2])):
                    edges[(v[1], v[2])] = [face_idx]
                else:
                    edges[(v[1], v[2])].append(face_idx)
                if not edges.get((v[0], v[2])):
                    edges[(v[0], v[2])] = [face_idx]
                else:
                    edges[(v[0], v[2])].append(face_idx)
                # add to face list
                faces[face_idx] = Tri(id1, id2, id3, face_idx)
                face_idx += 1
                

def is_same_edge(e1, e2):
    if e1.idx1 == e2.idx1 and e1.idx2 == e2.idx2:
        return True
    else:
        return False


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


# main
# input: obj (subdivided coarse mesh)
# return position matrix: local vertices per row * tri_num (6*3 x tri_num)
def meshmtx_wnb(file_name):
    vert = []
    vel = []
    edges = {}
    faces = {}

    obj_loader(file_name, vert, vel, edges, faces)
    dim, mtx, mtx_1 = comp_mtx(vert, edges, faces)

    return dim, mtx, mtx_1


def load_batch(file_name, batch_data, holdings=[]):
    vert = []
    vel = []
    edges = {}
    faces = {}
    obj_loader(file_name, vert, vel, edges, faces)

    # tri_nb = find_neighbors(vert, edges, faces)
    # batch_data.append(tri_nb)

    if len(holdings) is not 0:
        for i, x in enumerate(holdings):
            del vert[x - i]

    batch_data.append(vert)

