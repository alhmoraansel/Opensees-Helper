from opensees_helper import *

wipe()

model_basic3D()
uniaxial_steel(mat_tag=1)

E = 210e3
G= 81e3
const_beam = [100, E, G, 1408.33, 833.33, 833.33]
const_column = [256, E, G, 9229.65, 5461.33, 5461.33]
node3D(node_tag=1)
node3D(node_tag=2,x_coord=17.5,y_coord=0,z_coord=0)
node3D(node_tag=3,x_coord=17.5,y_coord=17.5,z_coord=0)
node3D(node_tag=4,x_coord=0,y_coord=17.5,z_coord=0)
node3D(node_tag=5,x_coord=0,y_coord=0,z_coord=18)
node3D(node_tag=6,x_coord=17.5,y_coord=0,z_coord=18)
node3D(node_tag=7,x_coord=17.5,y_coord=17.5,z_coord=18)
node3D(node_tag=8,x_coord=0,y_coord=17.5,z_coord=18)

fix3D(1); fix3D(2); fix3D(3); fix3D(4)

transform_linear3D(transf_tag=1,vecxzX=1,vecxzY=0,vecxzZ=0)
transform_linear3D(transf_tag=2,vecxzX=0,vecxzY=0,vecxzZ=1)

beam3D_element(1,1,5,*const_column,transf_tag=1)
beam3D_element(2,2,6,*const_column,transf_tag=1)
beam3D_element(3,3,7,*const_column,transf_tag=1)
beam3D_element(4,4,8,*const_column,transf_tag=1)
beam3D_element(5,5,6,*const_beam,transf_tag=2)
beam3D_element(6,5,8,*const_beam,transf_tag=2)
beam3D_element(7,6,7,*const_beam,transf_tag=2)
beam3D_element(8,7,8,*const_beam,transf_tag=2)

load_value = -3000
series_linear()
plain_pattern()

udl_3D(element_tag=5,Fy=load_value)
udl_3D(element_tag=6,Fy=load_value)
udl_3D(element_tag=7,Fy=load_value)
udl_3D(element_tag=8,Fy=load_value)

animate_deformed_shape_vfo(speed=0.1,view='3D',file_name="3D.mp4")
