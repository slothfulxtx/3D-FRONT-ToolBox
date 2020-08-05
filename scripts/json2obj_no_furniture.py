import json
import trimesh
import numpy as np
import math
import os,argparse
import math
import igl
from shutil import copyfile
from tqdm import tqdm
def split_path(paths):
    filepath,tempfilename = os.path.split(paths)
    filename,extension = os.path.splitext(tempfilename)
    return filepath,filename,extension


def write_obj_with_tex(savepath, vert, face, vtex, ftcoor, imgpath=None):
    filepath2,filename,extension = split_path(savepath)
    with open(savepath,'w') as fid:
        fid.write('mtllib '+filename+'.mtl\n')
        fid.write('usemtl a\n')
        for v in vert:
            fid.write('v %f %f %f\n' % (v[0],v[1],v[2]))
        for vt in vtex:
            fid.write('vt %f %f\n' % (vt[0],vt[1]))
        face = face + 1
        ftcoor = ftcoor + 1
        for f,ft in zip(face,ftcoor):
            fid.write('f %d/%d %d/%d %d/%d\n' % (f[0],ft[0],f[1],ft[1],f[2],ft[2]))
    filepath, filename2, extension = split_path(imgpath)
    if os.path.exists(imgpath) and not os.path.exists(filepath2+'/'+filename+extension):
        copyfile(imgpath, filepath2+'/'+filename+extension)
    if imgpath is not None:
        with open(filepath2+'/'+filename+'.mtl','w') as fid:
            fid.write('newmtl a\n')
            fid.write('map_Kd '+filename+extension)

def rotation_matrix(axis, theta):
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])



parser = argparse.ArgumentParser()
parser.add_argument(
        '--json_path',
        default = './3D-FRONT',
        help = 'path to 3D FRONT'
        )

parser.add_argument(
        '--save_path',
        default = './outputs',
        help = 'path to save result dir'
        )

args = parser.parse_args()

files = os.listdir(args.json_path)

if not os.path.exists(args.save_path):
    os.mkdir(args.save_path)

for m in tqdm(files):
    with open(args.json_path+'/'+m, 'r', encoding='utf-8') as f:
        data = json.load(f)

        mesh_uid = []
        mesh_xyz = []
        mesh_faces = []
        if not os.path.exists(args.save_path+'/'+m[:-5]):
            os.mkdir(args.save_path+'/'+m[:-5])
        for mm in data['mesh']:
            mesh_uid.append(mm['uid'])
            mesh_xyz.append(np.reshape(mm['xyz'], [-1, 3]))
            mesh_faces.append(np.reshape(mm['faces'], [-1, 3]))
        scene = data['scene']
        room = scene['room']
        for r in room:
            room_id = r['instanceid']
            meshes=[]
            children = r['children']
            for c in children:
                
                ref = c['ref']
                if ref in mesh_uid:
                    idx = mesh_uid.index(ref)
                    v = mesh_xyz[idx]
                    faces = mesh_faces[idx]
                else:
                    continue

                pos = c['pos']
                rot = c['rot']
                scale = c['scale']
                v = v.astype(np.float64) * scale
                ref = [0,0,1]
                axis = np.cross(ref, rot[1:])
                theta = np.arccos(np.dot(ref, rot[1:]))*2
                if np.sum(axis) != 0 and not math.isnan(theta):
                    R = rotation_matrix(axis, theta)
                    v = np.transpose(v)
                    v = np.matmul(R, v)
                    v = np.transpose(v)

                v = v + pos
                meshes.append(trimesh.Trimesh(v, faces))

            if len(meshes) > 0:
                temp = trimesh.util.concatenate(meshes)
                temp.export(args.save_path+'/'+ m[:-5] + '/' + room_id + '.obj')