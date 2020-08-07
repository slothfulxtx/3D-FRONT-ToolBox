"""
用来将3DFUTURE格式的数据集转化为SUNCG格式的数据集
Author : slothfulxtx
Date : 2020/08/06
Details : 
    3D-FUTURE中的模型可以看作由两部分组成，分别是家具和场景中的一些组件
    其中家具的模型直接链接到3D-FUTURE-MODEL/jid/，场景的模型并不是obj文件给出
    而是在json文件中给出的某种需要转换的数据格式，因此首先需要将json中场景数据提取出来
    然后处理成obj文件，之后再生成对应的SUNCG格式的json文件表示布局，json的Object List
    中，一部分需要在3D-FUTURE-MODEL/jid/路径下寻找，一部分需要在生成的文件夹下寻找
    生成的文件夹结构如下

    |- house
    |   |- 0b508d29-c18c-471b-8711-3f114819ea74.json 表示该布局文件是由3D-FRONT中0b508d29-c18c-471b-8711-3f114819ea74.json 文件转化来的
    |- backgroundobj
        |- 0b508d29-c18c-471b-8711-3f114819ea74 3D-FRONT下 0b508d29-c18c-471b-8711-3f114819ea74.json 对应的布局
            |- 5031ce7f-343a-41c8-b8ce-e2e943398df5X53357662.obj 该布局中某个3D-FUTURE认为属于背景而不是家具，没有渲染出来的部件的obj模型 
"""

import json
import trimesh
import numpy as np
import math
import os
import argparse
from tqdm import tqdm
import math
import igl
from shutil import copyfile
import sys
INF = 1e9

def Min3d(aa,bb):
    c = []
    for a,b in zip(aa,bb):
        c.append(a if a<b else b)
    return c

def Max3d(aa,bb):
    c = []
    for a,b in zip(aa,bb):
        c.append(a if a>b else b)
    return c

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


def main(args):
    filenames = os.listdir(args.FRONT_path)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    if not os.path.exists(os.path.join(args.save_path,"house")):
        os.mkdir(os.path.join(args.save_path,"house"))
    if not os.path.exists(os.path.join(args.save_path,"backgroundobj")):
        os.mkdir(os.path.join(args.save_path,"backgroundobj"))
    for sceneIdx,filename in tqdm(enumerate(filenames)):
        
        
        if not os.path.exists(os.path.join(args.save_path,"backgroundobj",filename[:-5])):
            os.mkdir(os.path.join(args.save_path,"backgroundobj",filename[:-5]))

        with open(os.path.join(args.FRONT_path, filename), 'r', encoding='utf-8') as f:
            # 打开3D-FRONT每个json文件
            frontJson = json.load(f)

        suncgJson = {}    
        
        suncgJson["origin"] = frontJson["uid"]
        suncgJson["id"] = str(sceneIdx)
        suncgJson["bbox"] = {
            "min":[INF,INF,INF],
            "max":[-INF,-INF,-INF]
        }
        suncgJson["up"] = [0,1,0]
        suncgJson["front"] = [0,0,1]
        suncgJson["rooms"] = []
        
        meshList = frontJson["mesh"]
        meshes = {}
        for mesh in meshList:
            meshes[mesh["uid"]] = mesh
            xyz = np.reshape(mesh['xyz'],(-1,3)).astype(np.float64)
            face = np.reshape(mesh['faces'],(-1,3))
            trimesh.Trimesh(xyz,face).export(os.path.join(args.save_path,"backgroundobj",filename[:-5],mesh["uid"].replace('/','X')+".obj"))
            # TODO : 可能需要同类别合并一下，暂时先不做
        
        furnitureList = frontJson["furniture"]
        furnitures = {}
        for furniture in furnitureList:
            if "valid" in furniture and furniture["valid"]:
                furnitures[furniture["uid"]] = furniture


        scene = frontJson["scene"]
        rooms = scene["room"]

        room_obj_cnt = 0

        for roomIdx, front_room in enumerate(rooms):
            suncg_room = {
                "id" : "%d_%d" % (sceneIdx,room_obj_cnt),
                "modelId":"",
                "roomTypes" :[front_room["type"]],
                "bbox":{
                    "min":[INF,INF,INF],
                    "max":[-INF,-INF,-INF]
                },
                "origin": frontJson["uid"],
                "roomId": roomIdx,
                "objList" :[]
            }
            room_obj_cnt += 1
            for childIdx, child in enumerate(front_room["children"]):
                if child["ref"] not in meshes and child["ref"] not in furnitures:
                    continue
                suncg_obj = {
                    "id": "%d_%d" % (sceneIdx,room_obj_cnt),
                    "type": "Object",
                    "modelId": meshes[child["ref"]]["uid"].replace('/','X') if child["ref"] in meshes else furnitures[child["ref"]]["jid"],
                    "bbox":{
                        "min":[INF,INF,INF],
                        "max":[-INF,-INF,-INF]
                    },
                    "translate":child["pos"],
                    "scale":child["scale"],
                    "rotate":"TODO",
                    "rotateOrder": "XYZ",
                    "orient":"TODO",
                    "coarseSemantic": meshes[child["ref"]]["type"] if child["ref"] in meshes else furnitures[child["ref"]]["category"],
                    "roomId" : roomIdx
                }    
                room_obj_cnt += 1
                obj_path = None
                if child["ref"] in meshes:
                    obj_path = os.path.join(args.save_path,"backgroundobj",filename[:-5],meshes[child["ref"]]["uid"].replace('/','X')+".obj")
                if child["ref"] in furnitures:
                    obj_path = os.path.join(args.FUTURE_path,furnitures[child["ref"]]["jid"],"raw_model.obj")
                assert obj_path is not None
                assert os.path.exists(obj_path)

                # 屏蔽warning信息，下面的方法好像屏蔽不了，日
                # stdout,stderr= sys.stdout,sys.stderr
                # with open('/dev/null',"w+") as null:
                #     sys.stdout,sys.stderr = null,null
                v, vt, _, faces, ftc, _ = igl.read_obj(obj_path)
                #sys.stdout,sys.stderr = sys.stdout,stderr

                pos,rot,scale = child["pos"],child["rot"],child["scale"]
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
                assert v.shape[1:] == (3,)
                lb = v.min(axis=0)
                ub = v.max(axis=0)
                suncg_obj["bbox"]["min"] = list(lb)
                suncg_obj["bbox"]["max"] = list(ub)
                suncg_room["objList"].append(suncg_obj)
                
                suncg_room["bbox"]["min"] = Min3d(suncg_room["bbox"]["min"],suncg_obj["bbox"]["min"])
                suncg_room["bbox"]["max"] = Max3d(suncg_room["bbox"]["max"],suncg_obj["bbox"]["max"])
                

            suncgJson["rooms"].append(suncg_room)
            suncgJson["bbox"]["min"] = Min3d(suncgJson["bbox"]["min"],suncg_room["bbox"]["min"])
            suncgJson["bbox"]["max"] = Max3d(suncgJson["bbox"]["max"],suncg_room["bbox"]["max"])
        
        with open(os.path.join(args.save_path,"house",filename),"w") as f:
            json.dump(suncgJson,f)
        break    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--FUTURE_path',
        default = './3D-FUTURE-model',
        help = 'path to 3D FUTURE'
        )
    parser.add_argument(
        '--FRONT_path',
        default = './3D-FRONT',
        help = 'path to 3D FRONT'
        )

    parser.add_argument(
        '--save_path',
        default = './outputs',
        help = 'path to save result dir'
        )

    args = parser.parse_args()
    main(args)
