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

files = [
    '785aec91-b165-4561-a384-1a191e6690ba.json',
    '0e02ee68-5940-49dd-bbdc-0e3667ec9e49.json',
    '4b663d06-fc0a-44ca-887a-d0bfc584a3ee.json',
    '1aab0a4b-760c-4489-b012-da6cefdca8a4.json',
    '129c9a2a-f789-4cd0-82c0-2b6d5293c45f.json',
    '5a4c9099-cf8f-4439-96ee-43736096617a.json',
    '5c0a1757-e14e-4901-a3a3-498537689821.json',
    '4c1b75c2-351b-4b6b-a7df-c867a2d9b3d6.json',
    '274ef293-2cf8-4c9a-8125-814f91d0bc83.json',
    '641eaf99-ec77-40a6-bef8-2ff72ef2b1d1.json',
    '7b2fae3d-5455-4dae-b174-7643ca83b1dc.json',
    '06a3196e-a2a2-4952-a5c6-034afcc18e15.json',
    '7e07a2a4-fead-40b8-8172-a430c150b733.json',
    'be5538a6-455b-486f-a46a-fd03d864587e.json']

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