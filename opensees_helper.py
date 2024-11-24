#This file was created as  a helper file for making analysis with opensees easier  and more compact.
#Feel free to modify and use it the way you wish, or even redistribute your own version of this file

import openseespy.opensees as ops
import openseespy.postprocessing.Get_Rendering as opsplt
import opsvis as opsvs 
import os as os
import vfo.vfo as vfo

massX = 0.49

#======================================================MODEL RELATED=========================================================

def wipe():
	ops.wipe()

def model(model_type = 'basic', ndom = 2, ndof = 3):
	ops.model(model_type,'-ndm',ndom,'-ndf',ndof)

def model_basic2D():
	ops.model('basic','-ndm',2,'-ndf',3)

def model_basic3D():
	ops.model('basic','-ndm',3,'-ndf',6)

def transform_linear2D(transf_tag = 1):
	ops.geomTransf('Linear', transf_tag)

def transform_p_dt2D(transf_tag = 1):
	ops.geomTransf('PDelta', transf_tag)

def transform_linear3D(transf_tag=1, vecxzX=1, vecxzY=1, vecxzZ=1):
	ops.geomTransf('Linear', transf_tag, vecxzX, vecxzY, vecxzZ)

def transform_p_dt3D(transf_tag=1, vecxzX=1, vecxzY=1, vecxzZ=1):
	ops.geomTransf('PDelta', transf_tag, vecxzX, vecxzY, vecxzZ)



#====================================================DEBUGGING RELATED=========================================================

def  print_nodes(node_tag=1):
	ops.printModel("node", node_tag)

def print_elements(element_tags =1):
	ops.printModel("ele",element_tags)

def print_model():
	ops.printModel("-JSON", "-file", "Example1.1.json")

def print_cmd():
	ops.printModel()



#====================================================MATERIAL RELATED=========================================================

def uniaxial_steel(mat_tag = 1, fy = 250E06, E = 200E09, b =0.15):
	ops.uniaxialMaterial('Steel01', mat_tag , fy , E, b)

def uniaxial_concrete(mat_tag = 1, fcompressive = 20E06, ecompressive = 0.0002, fcrushing = 20E06 , ecrushing = 0.0035):
	ops.uniaxialMaterial('Concrete01', mat_tag, fcompressive, ecompressive,fcrushing, ecrushing)

def nD_elastic_material(mat_tag = 1, E = 200E09, poission_ratio = 0.3):
	ops.nDMaterial('ElasticIsotropic', mat_tag, E,poission_ratio)



#=====================================================ELEMENTS  RELATED=========================================================

def node2D(node_tag = 1,x_coord = 0.0 ,y_coord = 0.0):
	ops.node(node_tag,x_coord,y_coord)

def node3D(node_tag = 1,x_coord = 0.0 ,y_coord = 0.0, z_coord = 0.0):
	ops.node(node_tag,x_coord,y_coord,z_coord)
	ops.mass(node_tag, massX, massX, 0.01, 1.0e-10, 1.0e-10, 1.0e-10)

def beam2D_element(element_tag=1, i_node=1, j_node=2, *consts, transf_tag=1, density=200.0, c_mass=False):
	nodes = [i_node,j_node]
	if c_mass:
		ops.element('elasticBeamColumn', element_tag,*nodes, *consts, transf_tag, '-mass', density, '-cMass')
	else:
		ops.element('elasticBeamColumn', element_tag, *nodes, *consts, transf_tag, '-mass', density)

def beam3D_element(element_tag=1, i_node=1, j_node=2,*const, transf_tag=1, density=0, c_mass=False):
	nodes = [i_node,j_node]
	if c_mass == False:
		ops.element('elasticBeamColumn', element_tag,*nodes, *const, transf_tag, '-mass', density, "-lMass")
	else:
		ops.element('elasticBeamColumn', element_tag, *nodes, *const, transf_tag, '-mass', density,"-cmass")

def beam3D_element(element_tag=1, i_node=1, j_node=2,*const, transf_tag=1, density=0, c_mass=False):
	nodes = [i_node,j_node]
	if c_mass == False:
		ops.element('elasticBeamColumn', element_tag,*nodes, *const, transf_tag, '-mass', density, "-lMass")
	else:
		ops.element('elasticBeamColumn', element_tag, *nodes, *const, transf_tag, '-mass', density,"-cmass")

def equal_dof(m_node = 10, s_node = 15):
	ops.equalDOF(m_node, s_node, 1,2,3)


#======================================================SECTIONS  RELATED=========================================================

def fiber_section(sec_tag=1,torsion_modulus = 7.7E10):
	ops.section('Fiber', sec_tag,'-GJ',torsion_modulus)

def patch_rect(mat_tag, n_fiber_y=10, n_fibers_z=10, y1=-0.15, z1=-0.15, y2=0.15, z2=0.15):
	ops.patch('rect', mat_tag, n_fiber_y, n_fibers_z, y1, z1, y2, z2)
	
def patch_quad(mat_tag=1, num_fibers_ij=8, num_fibers_jk=8, yI=-0.15, zI=-0.15, yJ=0.15, zJ=-0.15, yK=0.15, zK=0.15, yL=-0.15, zL=0.15):
	ops.patch('quad', mat_tag, num_fibers_ij, num_fibers_jk, yI, zI, yJ, zJ, yK, zK, yL, zL)

def zero_length_section(element_tag = 1, node = 1, section_tag = 1):
    ops.element('zeroLengthSection', element_tag, node, node, section_tag)

def define_layer(layer_type = 'straight', material_tag = 1, num_fibers = 10 ):
	"""
	Parameters:
	layer_type (str): Type of layer ('straight', 'circ', etc.)
	material_tag (int): Tag of the material
	num_fibers (int): Number of fibers in the layer
	coords (float): Coordinates defining the layer geometry
	"""
	ops.layer(layer_type, material_tag, num_fibers,2,2,-1,0)



#===================================================LOADING  RELATED=========================================================

def fix2D (node_tag = 1, trans_x = 1 ,trans_y = 1, rot_x = 1,rot_y = 1):
	ops.fix(node_tag,trans_x,trans_y,rot_x,rot_y)

def fix3D (node_tag = 1, trans_x = 1 ,trans_y = 1,trans_z = 1,rot_x = 1,rot_y = 1,rot_z = 1):
	ops.fix(node_tag,trans_x,trans_y,trans_z,rot_x,rot_y,rot_z)

def load2D(node_tag, Fx=0.0, Fy=0.0, Mz=0.0):
	ops.load(node_tag, Fx, Fy, Mz)

def load3D(node_tag, Fx=0.0, Fy=0.0, Fz=0.0, Mx=0.0, My=0.0, Mz=0.0):
	ops.load(node_tag, Fx, Fy, Fz, Mx, My, Mz)

def udl_3D(element_tag = 1, Fx=0.0, Fy = 0.0, Mz=0.0):
	ops.eleLoad('-ele', element_tag, '-type', '-beamUniform', Fx,Fy, Mz)

def series_constant(series_tag=1):
	ops.timeSeries('Constant', series_tag)

def series_linear(series_tag=1):
	ops.timeSeries('Linear', series_tag)

def series_trig(series_tag=1, start_time=0, end_time=1, period=1, phase_shift=0, amp_factor=1):
	ops.timeSeries('Trig', series_tag, start_time, end_time, period, phase_shift, amp_factor)

def series_path(series_tag=1, dt=0.01, *value_array):
	ops.timeSeries('Path', series_tag, '-dt', dt, '-values', *value_array)

def plain_pattern(pattern_tag = 1, series_tag = 1):
	ops.pattern('Plain', pattern_tag, series_tag)

def uniform_excitation(pattern_tag = 1, direction = 1, time_series_tag =1):
#direction (int): Direction of the excitation (1, 2, or 3 for x, y, or z).
	ops.pattern('UniformExcitation', pattern_tag, direction, time_series_tag)

def multi_support_excitation(pattern_tag = 1):
	ops.pattern('MultiSupport', pattern_tag)

def define_drm_pattern(pattern_tag = 1, time_series_tag = 1, *args):
	ops.pattern('DRM', pattern_tag, time_series_tag, *args)




#=======================================================ANALYSIS  RELATED=========================================================

def analysis_setup(num_increments=0.01, num_steps_analysis=20,system = 'BandSPD',
		     numberer = 'RCM',constraints = "Plain",algorithm = 'Linear',analysis = 'Static'):
	"""
	num_increments (int): Number of analysis steps.
	time_step (float): Time step for each increment (used in transient analysis).
	Returns:	int: 0 if analysis is successful, otherwise an error code.
	"""
	# Initialize the analysis

	#integrator(integrator_type,*integrator_args)
	ops.system(system)
	load_control_int()
	ops.numberer(numberer)
	ops.constraints(constraints)
	ops.algorithm(algorithm)
	ops.analysis(analysis)
	result = ops.analyze(num_steps_analysis,num_increments)

	if result == 0:
		print("Analysis successful!")
	else:
		print(f"Analysis failed with error code: {result}")
	return result

def analyze(value=1):
	ops.analyze(value)

def load_control_int(load_increment = 1.0):
	#- load_increment: The load increment for each step.
	ops.integrator('LoadControl', load_increment)

def displacement_control_int(node_id = 2, dof = 3, disp_increment = 0.1):
#     - node_id: ID of the node to control.
#     - dof: Degree of freedom to control.
#     - disp_increment: The displacement increment for each step.
    ops.integrator('DisplacementControl', node_id, dof, disp_increment)

def newmark_int(gamma, beta):
    ops.integrator('Newmark', gamma, beta)

def hht_int(alpha):
    ops.integrator('HHT', alpha)

def central_difference():
    ops.integrator('CentralDifference')

def beam_integrator(type = 'Lobatto', obj_tag = 1,section_tag=1,N=10):
	ops.beamIntegration('Lobatto',obj_tag,section_tag,N)

def get_current_dir():
	return os.getcwd()




#=======================================================RECORDERS=============================================================

def node_recorder(file_name = 'node.out', node_tag = [2], dof=[1,2], response_type =  'reaction', delta_t=0.1):
	ops.recorder('Node', '-file', file_name, '-time',  '-node', *node_tag, '-dof', *dof, response_type)

def element_recorder(file_name = 'Element.out', element_tag = [1], response_type = 'globalForce', delta_t=0.1):
	ops.recorder('Element', '-file', file_name, '-time', '-dT', delta_t, '-ele', *element_tag, response_type)

def section_recorder(file_name = 'Section.out', element_tag = 1, section_tag = 1, response_type = 'deformation', delta_t=0.1):
	#response_type: Type of response to record (e.g., 'deformation', 'force').
	ops.recorder('Element', '-file', file_name, '-time', '-dT', delta_t, '-ele', element_tag, 'section', section_tag, response_type)

def fiber_recorder(file_name='Fiber.out', element_tag=1, section_tag=1, fiber_coords=[0,0], response_type='stressStrain', delta_t=0.1):
	#- response_type: Type of response to record (e.g., 'stressStrain')
	x, y = fiber_coords
	ops.recorder('Element', '-file', file_name, '-time', '-dT', delta_t, '-ele', element_tag, 'section', section_tag, 'fiber', x, y, response_type)




#====================================================PLOTTING==========================================================

#---------------------------------------------------------VFO RELATED--------------------------------------------------------------

def save_fiber2D(model_folder = 'FiberModel', load_folder = 'Load', element_tag = 1, section_tag = 1, dt = 0.0, ZLE = False):
	vfo.saveFiberData2D(model_folder,load_folder,element_tag,section_tag,dt,ZLE)

def plot_section(model_name = "FiberModel",load_folder_name = "Loads",element_tag=1,section_tag=1):
	vfo.plot_fiberResponse2D(model_name,load_folder_name,element_tag,section_tag)

def vfo_plot_model(model_name = "Model",show_nodes="no",show_nodetags="no",show_eletags="no",
	       			font_size=10,setview="3D",elementgroups=None,line_width=1, save_to_file=None):
	vfo.plot_model(model_name,show_nodes,show_nodetags,show_eletags,font_size
		,setview,elementgroups,line_width, save_to_file)

def vfo_save_model(model_name = "Model",load_folder_name = "Loads",n_modes = 0, dt=0):
	vfo.createODB(model_name,loadcase=load_folder_name,Nmodes=n_modes,deltaT=dt)

def vfo_plot_deformed_shape(model_name = "Model",load_folder_name = "Loads",scale = 3.0, t_step = 0.1, 
			    					overlap = 'yes', contour = 'none', view = '3D',line_width = 1, contour_limits = None, file_name = None):
	vfo.plot_deformedshape(model_name, load_folder_name, scale, t_step, overlap, contour, view, line_width, contour_limits, file_name)

def animate_deformed_shape_vfo(model_name = "Model",load_folder_name = "Loads",scale = 5.0, speed= 1.0,
			       							view = '3D',line_width = 2, file_name = "animation.mp4", gif_name = None):
	vfo.animate_deformedshape(model=model_name,loadcase=load_folder_name,scale=scale,speedup=speed,overlap="yes",setview=view
			   						,line_width=line_width,node_for_th=None,node_dof=1,moviename=file_name,gifname=gif_name)

def quick_plot_vfo(model_name = "Model",show_nodes="no",show_nodetags="no",show_eletags="no",
	       				font_size=10,setview="3D",elementgroups=None,line_width=1, save_to_file="animation_vfo"):
	vfo_save_model(model_name)
	vfo_plot_model(model_name,show_nodes,show_nodetags,show_eletags,font_size
		,setview,elementgroups,line_width, save_to_file)

def quick_vfo():
	vfo_save_model()
	analysis_setup()
	wipe()

def quick_deformation_vfo(scale = 50):
	quick_vfo()
	vfo_plot_deformed_shape(scale=scale)

def quick_animate_vfo(speed = 1.0,view = '3D',file_name = "animation.mp4"):
	quick_vfo()
	animate_deformed_shape_vfo(speed=speed,view=view,file_name=file_name)



#--------------------------------------------------------------OPSPLT RELATED------------------------------------------------------

def ops_plot_model(model_name = "Model", show_nodes = "yes", show_elements = "yes"):
	if show_nodes == "yes" and show_elements == "no":
		opsplt.plot_model("nodes", Model = model_name)
	elif show_elements == "yes" and show_nodes == "no":
		opsplt.plot_model("elements", Model = model_name)
	elif show_nodes == "yes" and show_elements == "yes":
		opsplt.plot_model("nodes","elements", Model = model_name)
	else:
		opsplt.plot_model(Model=model_name)
		
def ops_plot_model_noarg():
	opsplt.plot_model("nodes", "elements")

def plot_mode_shape(mode_num = 2, scale = 3):
	opsplt.plot_modeshape(mode_num, scale)

def ops_plot_deformed_shape(model_name = "Model", load_folder_name = "Loads", scale = 5, overlap = "yes"):
		opsplt.plot_deformedshape(Model=model_name , LoadCase=load_folder_name, tstep=24.0, scale=scale, overlap=overlap)

def ops_save_model(model_name = "Model", load_folder_name = "Loads", Nmodes = 3):
	opsplt.createODB(model_name, load_folder_name, Nmodes=Nmodes)

def ops_animate(model_name = "Model", load_folder_name = "Loads", dt = 1, file_name ="animation"):
	opsplt.animate_deformedshape(Model=model_name, LoadCase = load_folder_name, dt=dt, Movie=file_name,FrameInterval=30,timeScale=0.5)

def quick_ops():
	ops_save_model()
	analysis_setup()
	wipe()

def quick_plot_ops():
	ops_plot_model_noarg()

def quick_deformation_ops(scale = 50):
	quick_ops()
	ops_plot_deformed_shape(scale=scale)

def quick_animate_ops(dt = 0.01):
	quick_ops()
	ops_animate(dt=dt)




"""
-----------------------------------------------------LIST OF OPENSEES ATTRIBUTES----------------------------------------------
Bcast
InitialStateAnalysis
OpenSeesError
ShallowFoundationGen
**builtinscacheddocfileloadernamepackagepathspec**
accelCPU
addToParameter
algorithm
analysis
analyze
barrier
basicDeformation
basicForce
basicStiffness
beamIntegration
block2D
block3D
build
cbdiDisplacement
computeGradients
constraints
convertBinaryToText
convertTextToBinary
correlate
database
defaultUnits
domainChange
domainCommitTag
eigen
eleDynamicalForce
eleForce
eleLoad
eleNodes
eleResponse
eleType
element
equalDOF
equalDOF_Mixed
fiber
fix
fixX
fixY
fixZ
frictionModel
geomTransf
getCDF
getDampTangent
getEleClassTags
getEleLoadClassTags
getEleLoadData
getEleLoadTags
getEleTags
getInverseCDF
getLoadFactor
getMean
getNP
getNodeTags
getNumElements
getNumThreads
getPDF
getPID
getParamTags
getParamValue
getRVTags
getStdv
getStrain
getStress
getTangent
getTime
groundMotion
hystereticBackbone
imposedMotion
imposedSupportMotion
initialize
integrator
layer
limitCurve
load
loadConst
logFile
mass
mesh
metaData
modalDamping
modalDampingQ
modalProperties
model
nDMaterial
node
nodeAccel
nodeBounds
nodeCoord
nodeDOFs
nodeDisp
nodeEigenvector
nodeMass
nodePressure
nodeReaction
nodeResponse
nodeUnbalance
nodeVel
numFact
numIter
numberer
parameter
partition
patch
pattern
pressureConstraint
printA
printB
printGID
printModel
probabilityTransformation
randomVariable
rayleigh
reactions
record
recorder
recv
region
remesh
remove
reset
responseSpectrum
restore
rigidDiaphragm
rigidLink
save
sdfResponse
searchPeerNGA
section
sectionDeformation
sectionDisplacement
sectionFlexibility
sectionForce
sectionLocation
sectionStiffness
sectionWeight
send
sensLambda
sensNodeAccel
sensNodeDisp
sensNodePressure
sensNodeVel
sensSectionForce
sensitivityAlgorithm
setCreep
setElementRayleighDampingFactors
setElementRayleighFactors
setMaxOpenFiles
setNodeAccel
setNodeCoord
setNodeDisp
setNodePressure
setNodeVel
setNumThreads
setParameter
setPrecision
setStartNodeTag
setStrain
setTime
solveCPU
sp
start
stiffnessDegradation
stop
strengthDegradation
stripXML
sys
system
systemSize
test
testIter
testNorm
testUniaxialMaterial
timeSeries
totalCPU
transformUtoX
uniaxialMaterial
unloadingRule
updateElementDomain
updateMaterialStage
updateParameter
version
wipe
wipeAnalysis
wipeReliability


"""
