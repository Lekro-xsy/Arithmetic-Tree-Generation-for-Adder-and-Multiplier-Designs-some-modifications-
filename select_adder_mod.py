import numpy as np
import argparse
import time
import math
import subprocess
import shutil
from scipy.spatial import distance
from scipy.spatial import KDTree
import os

parser = argparse.ArgumentParser(description='Select top K adders according to the distance to the pareto frontier.')
parser.add_argument('--input_bit', type=int, default=4)
parser.add_argument('--step', type=int, default=1666)
parser.add_argument('--openroad_path', type=str, default='/home')
parser.add_argument('--k', type=int, default=500)
args = parser.parse_args()
strftime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

BLACK_CELL = '''module BLACK(gik, pik, gkj, pkj, gij, pij);
input gik, pik, gkj, pkj;
output gij, pij;
assign pij = pik & pkj;
assign gij = gik | (pik & gkj);
endmodule
'''

GREY_CELL = '''module GREY(gik, pik, gkj, gij);
input gik, pik, gkj;
output gij;
assign gij = gik | (pik & gkj);
endmodule
'''

yosys_script_format = \
'''read -sv {}
hierarchy -top main
flatten
proc; techmap; opt;
abc -fast -liberty NangateOpenCellLibrary_typical.lib
write_verilog {}
'''

sdc_format = \
'''create_clock [get_ports clk] -name core_clock -period 3.0
set_all_input_output_delays
'''

openroad_tcl = \
'''source "helpers.tcl"
source "flow_helpers.tcl"
source "Nangate45/Nangate45.vars"
set design "adder"
set top_module "main"
set synth_verilog "{}"
set sdc_file "{}"
set die_area {{0 0 80 80}}
set core_area {{0 0 80 80}}
source -echo "full_flow.tcl"
'''

update = False

class ParetoFront2D:
    def __init__(self):
        self.front = []
    
    def add_point(self, x, y):
        dominated = False
        to_remove = []
        for i, (front_x, front_y) in enumerate(self.front):
            if front_x <= x and front_y <= y:
                dominated = True
                return False
            elif front_x >= x and front_y >= y:
                to_remove.append(i)
        if not dominated:
            self.front = [(front_x, front_y) for i, (front_x, front_y) in enumerate(self.front) if i not in to_remove]
            self.front.append((x, y))
        return True
    
    def get_front(self):
        return self.front

pareto_set = ParetoFront2D()
results = []

def point_to_line_dist(point, line_start, line_end):
    point = tuple(point)
    line_start = tuple(line_start)
    line_end = tuple(line_end)
    segment_length = math.dist(line_start, line_end)
    if segment_length == 0:
        return math.dist(point, line_start)

    segment_vector = (line_end[0] - line_start[0], line_end[1] - line_start[1])
    point_vector = (point[0] - line_start[0], point[1] - line_start[1])
    projection = (point_vector[0] * segment_vector[0] + point_vector[1] * segment_vector[1]) / segment_length

    if projection < 0:
        return math.dist(point, line_start)
    elif projection > segment_length:
        return math.dist(point, line_end)

    closest_point = (line_start[0] + (projection / segment_length) * segment_vector[0],
                     line_start[1] + (projection / segment_length) * segment_vector[1])

    distance = math.dist(point, closest_point)
    return distance

def build_pareto_kdtree(pareto_points):
    # 提取坐标
    coords = [(point[1], point[2]) for point in pareto_points]
    # 构建KDTree
    kdtree = KDTree(coords)
    return kdtree, coords

def distance_point_to_pareto_outline_optimized(x, y, kdtree, pareto_coords):
    point = (x, y)
    # 查询最近的2个邻居
    distances, indices = kdtree.query(point, k=2)
    min_distance = float('inf')
    
    # 对于每个邻居，计算到相邻线段的距离
    for idx in np.atleast_1d(indices):
        if idx > 0:
            # 与前一个点形成的线段
            line_start = pareto_coords[idx - 1]
            line_end = pareto_coords[idx]
            dist = point_to_line_dist(point, line_start, line_end)
            min_distance = min(min_distance, dist)
        if idx < len(pareto_coords) - 1:
            # 与后一个点形成的线段
            line_start = pareto_coords[idx]
            line_end = pareto_coords[idx + 1]
            dist = point_to_line_dist(point, line_start, line_end)
            min_distance = min(min_distance, dist)
    return min_distance

def find_pareto_points_optimized(points):
    print("points len: ", len(points))
    # 根据第一个目标（延迟）升序排序
    points_sorted = sorted(points, key=lambda x: x[1])
    pareto_points = []
    min_area = float('inf')

    for point in points_sorted:
        delay = point[1]
        area = point[2]
        # 由于已按延迟排序，只需检查面积是否小于当前最小值
        if area < min_area:
            pareto_points.append(point)
            min_area = area  # 更新最小面积
    return pareto_points

def write_verilog(file_name_prefix):

    cell_map = np.zeros((args.input_bit, args.input_bit))
    assert os.path.exists("run_verilog_mid/{}.log".format(file_name_prefix))
    with open("run_verilog_mid/{}.log".format(file_name_prefix), "r") as fopen:
        for i in range(args.input_bit):
            line = fopen.readline()
            for j in range(args.input_bit):
                bit = int(line[j])
                cell_map[i, j] = bit

    file_name = "run_verilog_mid/{}.v".format(file_name_prefix)

    with open(file_name, "w") as verilog_file:
        verilog_file.write("module main(a,b,s,cout);\n")
        verilog_file.write("input [{}:0] a,b;\n".format(args.input_bit-1))
        verilog_file.write("output [{}:0] s;\n".format(args.input_bit-1))
        verilog_file.write("output cout;\n")
        wires = set()
        for i in range(args.input_bit):
            wires.add("c{}".format(i))
        
        for x in range(args.input_bit-1, 0, -1):
            last_y = x
            for y in range(x-1, -1, -1):
                if cell_map[x, y] == 1:
                    assert cell_map[last_y-1, y] == 1
                    if y==0:
                        wires.add("g{}_{}".format(x, last_y))
                        wires.add("p{}_{}".format(x, last_y))
                        wires.add("g{}_{}".format(last_y-1, y))
                    else:
                        wires.add("g{}_{}".format(x, last_y))
                        wires.add("p{}_{}".format(x, last_y))
                        wires.add("g{}_{}".format(last_y-1, y))
                        wires.add("p{}_{}".format(last_y-1, y))
                        wires.add("g{}_{}".format(x, y))
                        wires.add("p{}_{}".format(x, y))
                    last_y = y
        
        for i in range(args.input_bit):
            wires.add("p{}_{}".format(i, i))
            wires.add("g{}_{}".format(i, i))
            wires.add("c{}".format(i))
        assert 0 not in wires
        assert "0" not in wires
        verilog_file.write("wire ")
        
        for i, wire in enumerate(wires):
            if i < len(wires) - 1:
                verilog_file.write("{},".format(wire))
            else:
                verilog_file.write("{};\n".format(wire))
        verilog_file.write("\n")
        
        for i in range(args.input_bit):
            verilog_file.write('assign p{}_{} = a[{}] ^ b[{}];\n'.format(i,i,i,i))
            verilog_file.write('assign g{}_{} = a[{}] & b[{}];\n'.format(i,i,i,i))
        
        for i in range(1, args.input_bit):
            verilog_file.write('assign g{}_0 = c{};\n'.format(i, i))
        
        for x in range(args.input_bit-1, 0, -1):
            last_y = x
            for y in range(x-1, -1, -1):
                if cell_map[x, y] == 1:
                    assert cell_map[last_y-1, y] == 1
                    if y == 0:
                        verilog_file.write('GREY grey{}(g{}_{}, p{}_{}, g{}_{}, c{});\n'.format(
                            x, x, last_y, x, last_y, last_y-1, y, x
                        ))
                    else:
                        verilog_file.write('BLACK black{}_{}(g{}_{}, p{}_{}, g{}_{}, p{}_{}, g{}_{}, p{}_{});\n'.format(
                            x, y, x, last_y, x, last_y, last_y-1, y, last_y-1, y, x, y, x, y 
                        ))
                    last_y = y
        
        verilog_file.write('assign s[0] = a[0] ^ b[0];\n')
        verilog_file.write('assign c0 = g0_0;\n')
        verilog_file.write('assign cout = c{};\n'.format(args.input_bit-1))
        for i in range(1, args.input_bit):
            verilog_file.write('assign s[{}] = p{}_{} ^ c{};\n'.format(i, i, i, i-1))
        verilog_file.write("endmodule")
        verilog_file.write("\n\n")

        verilog_file.write(GREY_CELL)
        verilog_file.write("\n")
        verilog_file.write(BLACK_CELL)
        verilog_file.write("\n")

def run_yosys(file_name_prefix):
    if not os.path.exists("run_yosys_mid"):
        os.mkdir("run_yosys_mid")
    dst_file_name = os.path.join("run_yosys_mid", file_name_prefix + "_yosys.v")
    
    if os.path.exists(dst_file_name):
        return
    src_file_path = os.path.join("run_verilog_mid", file_name_prefix + ".v")
    if not os.path.exists(src_file_path):
        write_verilog(file_name_prefix)
    if not os.path.exists("run_yosys_script"):
        os.mkdir("run_yosys_script")
    yosys_script_file_name = os.path.join("run_yosys_script", 
        "{}.ys".format(file_name_prefix))
    with open(yosys_script_file_name, "w") as fopen:
        fopen.write(yosys_script_format.format(src_file_path, dst_file_name))
    _ = subprocess.check_output(["yosys {}".format(yosys_script_file_name)], shell=True)

def run_full_openroad(file_name_prefix):
    assert "." not in file_name_prefix
    print("file_name_prefix {}".format(file_name_prefix))
    def substract_results(p):
        lines = p.split("\n")[-15:]
        area = -100.0
        wslack = -100.0
        power = 0.0
        note = None
        for line in lines:
            if not line.startswith("result:") and not line.startswith("Total"):
                continue
            if "design_area" in line:
                area = float(line.split(" = ")[-1])
            elif "worst_slack" in line:
                wslack = float(line.split(" = ")[-1])
                note = lines
            elif "Total" in line:
                power = float(line.split()[-2])

        return area, wslack, power, note

    verilog_file_path = "{}/OpenROAD/test/adder_tmp_{}.v".format(args.openroad_path, file_name_prefix)
    yosys_file_name = os.path.join("run_yosys_mid", file_name_prefix + "_yosys.v")
    if not os.path.exists(yosys_file_name):
        run_yosys(file_name_prefix)
    shutil.copyfile(yosys_file_name, verilog_file_path)
    sdc_file_path = "{}/OpenROAD/test/adder_nangate45_{}.sdc".format(args.openroad_path, file_name_prefix)
    with open(sdc_file_path, "w") as fopen_sdc:
        fopen_sdc.write(sdc_format)
    with open("{}/OpenROAD/test/adder_nangate45_{}.tcl".format(args.openroad_path, file_name_prefix), "w") as fopen_tcl:
        fopen_tcl.write(openroad_tcl.format("adder_tmp_{}.v".format(file_name_prefix), 
            "adder_nangate45_{}.sdc".format(file_name_prefix)))
    
    output = subprocess.check_output(['openroad',
        "{}/OpenROAD/test/adder_nangate45_{}.tcl".format(args.openroad_path, file_name_prefix)], 
        cwd="{}/OpenROAD/test".format(args.openroad_path)).decode('utf-8')
    note = None
    retry = 0
    area, wslack, power, note = substract_results(output)
    while note is None and retry < 3:
        output = subprocess.check_output(['openroad',
            "{}/OpenROAD/test/adder_nangate45_{}.tcl".format(args.openroad_path, file_name_prefix)], 
            shell=True, cwd="{}/OpenROAD/test".format(args.openroad_path)).decode('utf-8')
        area, wslack, power, note = substract_results(output)
        retry += 1
    if os.path.exists(yosys_file_name):
        os.remove(yosys_file_name)
    if os.path.exists("{}/OpenROAD/test/adder_nangate45_{}.tcl".format(args.openroad_path, 
            file_name_prefix)):
        os.remove("{}/OpenROAD/test/adder_nangate45_{}.tcl".format(args.openroad_path, file_name_prefix))
    if os.path.exists("{}/OpenROAD/test/adder_nangate45_{}.sdc".format(args.openroad_path, 
            file_name_prefix)):
        os.remove("{}/OpenROAD/test/adder_nangate45_{}.sdc".format(args.openroad_path, file_name_prefix))
    if os.path.exists("{}/OpenROAD/test/adder_tmp_{}.v".format(args.openroad_path, 
            file_name_prefix)):
        os.remove("{}/OpenROAD/test/adder_tmp_{}.v".format(args.openroad_path,file_name_prefix))
    delay = 3.0 - wslack
    delay *= 1000
    return delay, area, power, note

def run_total_results(pareto_set, results, k):

    print("results len", len(results))
    pareto_set.sort(key=lambda x: x[1])
    # 构建KDTree
    kdtree, pareto_coords = build_pareto_kdtree(pareto_set)
    distance_results = []
    for result in results:
        dist = distance_point_to_pareto_outline_optimized(result[1], result[2], kdtree, pareto_coords)
        distance_results.append((result[0], result[1], result[2], dist))
    
    distance_results.sort(key=lambda x: x[3])
    distances = [item[3] for item in distance_results]
    print("distances[:10]", distances[:10])

    with open("adder_{}b_full_openroad_{}.log".format(args.input_bit, strftime), "w") as fwrite:
        start_time = time.time()
        for i in range(k):
            print("{}/{}".format(i, k))
            delay, area, power, note = run_full_openroad(distance_results[i][0].split("\t")[0].split(".")[0])
            output_str = distance_results[i][0]
            output_str = output_str.strip()
            fwrite.write("{}\t{}\t{}\t{}\t{:.2f}\n".format(output_str, 
                delay, area, power, time.time()- start_time))
            fwrite.flush()

def main():
    results = []
    
    if not os.path.exists("adder_parc_log/adder_{}b".format(args.input_bit)):
        print("please run 'python adder_prac.py --type=0/1/2' first.")
        return
    
    dirs = os.listdir("adder_parc_log/adder_{}b".format(args.input_bit))
    print("dirs")
    print(dirs)
    dirs.sort()
    type_set = set()
    
    files = [None, None, None]

    for d in dirs:
        adder_type = int(d.split("_")[3][4:])
        print("adder_type = {}".format(adder_type))
        if adder_type not in type_set:
            files[adder_type] = os.path.join("adder_parc_log/adder_{}b".format(args.input_bit), d)
        type_set.add(adder_type)
    
    if len(type_set) < 3:
        print("3 types of adders are not completed. Please run 'python adder_prac.py --type=0/1/2' first.")
        return

    for file in files:
        i = 0
        i_limit = args.step
        with open(file, "r") as fopen:
            for line in fopen.readlines():
                i += 1
                if i > i_limit:
                    break
                output_str = line.strip()
                delay = float(output_str.split("\t")[1])
                area = float(output_str.split("\t")[2])
                level = float(output_str.split("\t")[4])
                size = float(output_str.split("\t")[5])
                time_elapsed = float(output_str.split("\t")[-1])
                results.append((output_str, delay, area, level, size))
    
    no_repeat_results_set = set()
    no_repeat_results = []
    print("results len", len(results))
    for result in results:
        if result[0].split("\t")[0] not in no_repeat_results_set:
            no_repeat_results_set.add(result[0].split("\t")[0])
            no_repeat_results.append(result)
    
    results = no_repeat_results
    print("After remove repeat results, results len", len(results))
    results.sort(key=lambda x: x[1])
    pareto_set = find_pareto_points_optimized(results)
    pareto_set.sort(key=lambda x: x[1])

    run_total_results(pareto_set, results, args.k)

if __name__ ==  "__main__":
    main()
