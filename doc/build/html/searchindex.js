Search.setIndex({docnames:["api/acc/acc_collision_avoidance","api/acc/acc_extended_pi","api/acc/acc_intelligent_driver_model","api/acc/acc_interface","api/acc/acc_pi","api/acc/modules","api/common/configuration","api/common/modules","api/common/node","api/common/state","api/common/utility_fcts","api/common/utility_safety","api/config","api/falsification","api/lead_search/backward_search","api/lead_search/forward_search","api/lead_search/modules","api/lead_search/monte_carlo_simulation","api/lead_search/rrt","api/lead_search/rrt_backward","api/lead_search/rrt_forward","api/lead_search/utility_search","api/modules","api/output/modules","api/output/storage","api/output/visualization","api/review","index"],envversion:{"sphinx.domains.c":1,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":1,"sphinx.domains.javascript":1,"sphinx.domains.math":2,"sphinx.domains.python":1,"sphinx.domains.rst":1,"sphinx.domains.std":1,"sphinx.ext.viewcode":1,sphinx:56},filenames:["api/acc/acc_collision_avoidance.rst","api/acc/acc_extended_pi.rst","api/acc/acc_intelligent_driver_model.rst","api/acc/acc_interface.rst","api/acc/acc_pi.rst","api/acc/modules.rst","api/common/configuration.rst","api/common/modules.rst","api/common/node.rst","api/common/state.rst","api/common/utility_fcts.rst","api/common/utility_safety.rst","api/config.rst","api/falsification.rst","api/lead_search/backward_search.rst","api/lead_search/forward_search.rst","api/lead_search/modules.rst","api/lead_search/monte_carlo_simulation.rst","api/lead_search/rrt.rst","api/lead_search/rrt_backward.rst","api/lead_search/rrt_forward.rst","api/lead_search/utility_search.rst","api/modules.rst","api/output/modules.rst","api/output/storage.rst","api/output/visualization.rst","api/review.rst","index.rst"],objects:{"":{acc_collision_avoidance:[0,0,0,"-"],acc_extended_pi:[1,0,0,"-"],acc_intelligent_driver_model:[2,0,0,"-"],acc_interface:[3,0,0,"-"],acc_pi:[4,0,0,"-"],backward_search:[14,0,0,"-"],configuration:[6,0,0,"-"],falsification:[13,0,0,"-"],forward_search:[15,0,0,"-"],monte_carlo_simulation:[17,0,0,"-"],node:[8,0,0,"-"],review:[26,0,0,"-"],rrt:[18,0,0,"-"],rrt_backward:[19,0,0,"-"],rrt_forward:[20,0,0,"-"],state:[9,0,0,"-"],storage:[24,0,0,"-"],utility_fcts:[10,0,0,"-"],utility_safety:[11,0,0,"-"],utility_search:[21,0,0,"-"],visualization:[25,0,0,"-"]},"acc_collision_avoidance.CAACC":{acc_control:[0,2,1,""]},"acc_extended_pi.ExtendedPIControllerACC":{acc_control:[1,2,1,""]},"acc_intelligent_driver_model.IdmACC":{acc_control:[2,2,1,""]},"acc_interface.AccFactory":{acc_control:[3,2,1,""],create:[3,2,1,""]},"acc_pi.PIController":{acc_control:[4,2,1,""],dynamic_headway:[4,2,1,""]},"configuration.ACCController":{COLLISION_AVOIDANCE:[6,4,1,""],EXTENDED_PI:[6,4,1,""],IDM:[6,4,1,""],PI:[6,4,1,""]},"configuration.SamplingStrategy":{GLOBAL:[6,4,1,""],LOCAL:[6,4,1,""]},"configuration.SearchType":{BACKWARD:[6,4,1,""],FORWARD:[6,4,1,""],MONTE_CARLO:[6,4,1,""]},"configuration.VehicleType":{ACC:[6,4,1,""],LEAD:[6,4,1,""]},"monte_carlo_simulation.MonteCarlo":{plan:[17,2,1,""]},"node.Node":{acc_state:[8,2,1,""],add_acc_state:[8,2,1,""],delta_s:[8,2,1,""],delta_v:[8,2,1,""],get_acc_x_position_front:[8,2,1,""],get_lead_x_position_rear:[8,2,1,""],lead_state:[8,2,1,""],parent:[8,2,1,""],safe_distance:[8,2,1,""],unsafe_distance:[8,2,1,""],x_position_left_front:[8,2,1,""],x_position_left_rear:[8,2,1,""],x_position_right_front:[8,2,1,""],x_position_right_rear:[8,2,1,""]},"rrt.RRT":{a_max:[18,2,1,""],a_min:[18,2,1,""],braking:[18,2,1,""],dereference_unused_nodes:[18,2,1,""],dt:[18,2,1,""],forward_simulation:[18,2,1,""],get_input_global:[18,2,1,""],get_input_local:[18,2,1,""],get_max_and_min_position:[18,2,1,""],get_max_and_min_position_delta:[18,2,1,""],get_max_and_min_velocity:[18,2,1,""],get_max_and_min_velocity_delta:[18,2,1,""],j_max:[18,2,1,""],j_min:[18,2,1,""],max_velocity:[18,2,1,""],nearest_node_global:[18,2,1,""],nearest_node_local:[18,2,1,""],normalize_state:[18,2,1,""],number_nodes_rrt:[18,2,1,""],plan:[18,2,1,""],sample:[18,2,1,""],sampling_range:[18,2,1,""],sampling_strategy:[18,2,1,""]},"rrt_backward.RRTBackward":{plan:[19,2,1,""]},"rrt_forward.RRTForward":{is_valid_safe_node:[20,2,1,""],plan:[20,2,1,""]},"state.State":{acceleration:[9,2,1,""],steering_angle:[9,2,1,""],steering_velocity:[9,2,1,""],time_step:[9,2,1,""],velocity:[9,2,1,""],x_position:[9,2,1,""],y_position:[9,2,1,""],yaw_angle:[9,2,1,""]},acc_collision_avoidance:{CAACC:[0,1,1,""]},acc_extended_pi:{ExtendedPIControllerACC:[1,1,1,""]},acc_intelligent_driver_model:{IdmACC:[2,1,1,""]},acc_interface:{AccFactory:[3,1,1,""]},acc_pi:{PIController:[4,1,1,""]},backward_search:{acc_random_backward:[14,3,1,""],check_unsafe_node_reached:[14,3,1,""],init_nodes_backward:[14,3,1,""],search:[14,3,1,""]},configuration:{ACCController:[6,1,1,""],SamplingStrategy:[6,1,1,""],SearchType:[6,1,1,""],VehicleType:[6,1,1,""],create_acc_param:[6,3,1,""],create_acc_vehicle_param:[6,3,1,""],create_lead_vehicle_param:[6,3,1,""],create_rrt_param:[6,3,1,""],create_sim_param:[6,3,1,""],load_yaml:[6,3,1,""]},falsification:{main:[13,3,1,""]},forward_search:{check_unsafe_node_reached:[15,3,1,""],search:[15,3,1,""]},monte_carlo_simulation:{MonteCarlo:[17,1,1,""],search:[17,3,1,""]},node:{Node:[8,1,1,""]},review:{main:[26,3,1,""]},rrt:{RRT:[18,1,1,""]},rrt_backward:{RRTBackward:[19,1,1,""]},rrt_forward:{RRTForward:[20,1,1,""]},state:{State:[9,1,1,""]},storage:{create_commonroad_scenario:[24,3,1,""],store_results:[24,3,1,""]},utility_fcts:{acc_input_forward:[10,3,1,""],check_feasibility:[10,3,1,""],forward_propagation:[10,3,1,""],func_ks:[10,3,1,""],get_date_and_time:[10,3,1,""],reaction_delay:[10,3,1,""]},utility_safety:{safe_distance:[11,3,1,""],simulate_vehicle_braking:[11,3,1,""],unsafe_distance:[11,3,1,""]},utility_search:{backward_propagation:[21,3,1,""],check_feasibility_backward:[21,3,1,""],init_nodes_forward:[21,3,1,""],is_valid_final_node:[21,3,1,""]},visualization:{animate_cars:[25,3,1,""],animate_profiles:[25,3,1,""],create_profiles:[25,3,1,""],plot_figures:[25,3,1,""],store_videos:[25,3,1,""]}},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","method","Python method"],"3":["py","function","Python function"],"4":["py","attribute","Python attribute"]},objtypes:{"0":"py:module","1":"py:class","2":"py:method","3":"py:function","4":"py:attribute"},terms:{"abstract":[3,18],"boolean":[12,15],"case":12,"class":[0,1,2,3,4,6,8,9,17,18,19,20],"enum":[6,27],"f\u00fcr":[1,12],"float":[0,1,2,3,4,8,9,10,11,14,18,21,25],"function":[7,16,17,18,19,20,22,27],"int":[9,10,18,25,27],"new":[10,14,17,18,19,20],"return":[0,1,2,3,4,6,8,9,10,11,14,15,17,18,19,20,21,25],"sch\u00f6nhof":[2,12],"static":[18,20],"true":[12,20,21],For:27,The:27,a_acc:[0,1,2,3,4,11],a_cur:[10,21],a_init:12,a_lead:11,a_max:[10,12,18,21],a_max_acc:11,a_min:[10,11,12,18,21],a_min_acc:11,a_min_lead:11,a_new:[10,21],abc:[3,18],absolut:18,acc:[0,1,2,4,5,6,8,10,11,12,14,15,17,18,19,20,24,25],acc_acceleration_profil:25,acc_collision_avoid:0,acc_control:[0,1,2,3,4],acc_extended_pi:1,acc_input_forward:10,acc_intelligent_driver_model:2,acc_interfac:[0,1,2,3,4,27],acc_param:[0,1,2,3,4,12,14,15,17],acc_param_all_control:[14,15,17],acc_param_complet:24,acc_pi:4,acc_plann:[10,14],acc_position_profil:25,acc_profil:25,acc_random_backward:14,acc_stat:[8,18],acc_state_list:[17,18,19,20],acc_vehicl:6,acc_vehicle_param:[0,1,2,3,4,6,8,10,12,14,15,17,18,19,20,21,24,25],acccontrol:[6,27],acceler:[0,1,2,3,4,8,9,10,11,12,18,21,25],accfactori:[0,1,2,3,4,14],accord:27,activ:[2,12,27],adapt:[0,2,12,22,27],add:[8,27],add_acc_st:8,add_sample_range_global_backward:12,add_sample_range_global_forward:12,add_sample_range_local_backward:12,add_sample_range_local_forward:12,added:27,addit:12,addition:27,after:14,aggress:12,aktiv:[1,12],algorithm:[3,10,14,16,22,24],all:[17,18,19,20,25,27],allow:[10,21,27],althoff:27,anaconda:27,analysi:[0,12],angl:[10,21],animate_car:25,animate_profil:25,approach:27,author:12,autom:[4,12],avoid:[2,5,12,22],axi:25,babu:[0,12],backward:[6,12,16,21,22],backward_propag:21,backward_search:14,backward_simul:[],base:[0,1,2,3,4,6,8,9,12,13,17,18,19,20,25,27],beta:12,between:[4,11],bool:[14,15,20,21],both:[11,27],bound:[10,21],brake:[11,18],caacc:0,calcul:[0,1,2,3,4,8,10,11,18,21,24],call:[26,27],can:27,car:25,carlo:[12,16,22],cascad:1,center:[10,21],check:21,check_feas:10,check_feasibility_backward:21,check_unsafe_node_reach:[14,15],childless:18,choos:18,classmethod:3,closest:18,closest_nod:18,coeffici:12,collis:[5,12,22],collision_avoid:[6,12],command:27,common:[22,27],commonroad:[12,24,27],commonroad_benchmark_id:12,commonroad_io:[],commonroad_scenario_author:12,commonroad_scenario_tag:12,comput:12,computation:27,concaten:27,conf:27,config:[13,22,27],config_reconstruct:[],config_xyz:27,config_xzi:[26,27],configur:[7,13,22,24,27],congest:[2,12],consid:[10,21],consol:12,constant:12,contain:[18,27],content:27,control:[0,1,2,3,4,6,12,14,15,17,22,24,27],controller_param:6,coordin:[12,18],correspond:[18,27],cps:27,creat:[3,12,18],create_acc_param:6,create_acc_vehicle_param:[6,27],create_commonroad_scenario:24,create_lead_vehicle_param:6,create_profil:25,create_rrt_param:6,create_sim_param:6,creation:[24,25],cruis:[0,2,12,22,27],current:[0,1,2,3,4,10,11,18,21,25,27],current_max_posit:18,current_max_veloc:18,current_min_posit:18,current_min_veloc:18,current_nod:24,data:10,date:[10,27],dead:12,deceler:12,defin:[8,27],delai:10,delet:18,delta_:8,delta_s_desir:18,delta_s_init_max:[12,27],delta_s_init_min:[12,27],delta_s_min:12,delta_v:[4,8],delta_v_desir:18,dereference_unused_nod:18,describ:[6,12],design:[0,2,12],desir:[3,12,18],dict:[3,6,8,10,14,15,17,18,19,20,21,24,25],dictionari:[3,6,8,10,14,15,17,18,21,24,25,27],differ:[6,18,27],differenti:[10,21],distanc:[11,12,18,25],distribut:[12,27],driver:[5,22],dure:12,duti:[4,12],dynam:10,dynamic_headwai:4,dynamics_param:12,each:[12,14,15,17],effici:27,email:27,end:[10,12,21],end_nod:24,end_tim:[10,21],environ:[3,6,14,15,17,18,19,20,21],equat:[10,21],equip:[24,25],euclidean:18,evalu:[14,15,20],execut:[26,27],extend:[5,22],extended_pi:[6,12],extendedpicontrolleracc:1,extendend:12,extens:12,fahrerassistenzsystem:[1,12],fals:[20,21],falsif:[6,24],feedback:12,file:[6,12,13,26,27],file_nam:6,folder:27,follow:[4,25,27],forward:[6,10,11,12,14,16,18,21,22],forward_propag:10,forward_search:15,forward_simul:18,found:[12,27],from:[6,17,18,19,20,25],front:[0,1,2,3,4,8,11,27],full:[0,12],func_k:10,gain:12,gap:12,gener:[12,15,24,27],get:18,get_acc_x_position_front:8,get_date_and_tim:10,get_input_glob:18,get_input_loc:18,get_lead_x_position_rear:8,get_max_and_min_posit:18,get_max_and_min_position_delta:18,get_max_and_min_veloc:18,get_max_and_min_velocity_delta:18,gif:25,gitlab:27,given:8,global:[6,12,18],goal:[10,14,21],grundlagen:[1,12],hakuli:[1,12],hand:[10,21],handbuch:[1,12],have:11,headwai:[4,12],heavi:[4,12],helb:[2,12],http:27,idm:[6,12],idmacc:2,ieee:27,imageio:27,impact:[11,12],implement:27,index:[18,27],indic:[12,15],inform:12,init_nodes_backward:14,init_nodes_forward:21,initi:[3,10,12,14,21,27],initial_steering_angl:[],initial_steering_veloc:[],initial_yaw_angl:[],input:[0,1,2,3,4,10,18,21,26,27],instal:27,integr:[0,12],intellig:[5,22,27],interfac:[5,22,27],interpret:27,introduc:27,is_valid_final_nod:21,is_valid_initial_nod:[],is_valid_safe_nod:20,iter:[12,15],j_max:[10,12,18,21],j_max_acc:11,j_min:[10,11,12,18,21],j_min_acc:11,j_min_lead:11,jerk:[10,11,12,21],kanellakopoulo:[4,12],kest:[2,12],kinemat:[10,11,21],komfort:[1,12],komponenten:[1,12],koschi:27,label:25,lane:[17,18,19,20],last:[15,24,25],lead:[0,1,2,3,4,6,8,10,11,12,14,15,17,18,19,20,21,22,24,25,27],lead_acceleration_profil:25,lead_position_profil:25,lead_search:[17,19,20],lead_stat:8,lead_vehicl:6,lead_vehicle_param:[6,8,10,12,14,15,17,18,19,20,21,24,25],lead_vehicle_profil:25,least:27,left:8,limit:[10,21],line:27,list:[6,10,11,14,15,17,18,19,20,21,25],list_:18,list_v:18,load:6,load_yaml:6,local:[6,12,18],locat:27,lotz:[1,12],lrz:27,maierhof:27,main:[13,26,27],matplotlib:27,max_comp_time_backward:12,max_veloc:18,maximum:[10,11,12,18,21,25],mcs_beta_a:12,mcs_beta_b:12,mean:18,method:12,minim:18,minimum:[10,11,12,18,21,25],model:[5,10,11,12,21,22,27],modul:[5,7,16,22,23,27],mont:[12,16,22],monte_carlo:[6,12],monte_carlo_simul:17,montecarlo:17,mullakk:[0,12],must:[26,27],name:[6,12,27],ndarrai:[10,21,25],nearest:18,nearest_node_glob:18,nearest_node_loc:18,need:[18,27],neg:11,new_nod:20,new_node_list:18,node:[7,10,12,14,15,17,18,19,20,21,22,24,25],node_list:[14,15,17,18,19,20,25],nonlinear:[4,12],normal:18,normalize_st:18,num_iter:12,number:[10,12],number_fram:25,number_nod:12,number_nodes_rrt:18,numpi:27,object:[3,8,9],odeint:10,onli:12,option:[6,12],order:12,orient:8,out:27,output:[22,27],own:[12,27],packag:27,page:27,paper:12,param:8,paramet:[0,1,2,3,4,6,8,10,11,12,14,15,17,18,19,20,21,24,25,26,27],parent:[8,10,18],pass:15,path:27,pek:27,percept:12,physic:[3,6,8,10,12,14,15,17,18,19,20,21,24,25],picontrol:4,pip:27,pkl:[26,27],plan:[10,14,17,18,19,20],plot:[25,27],plot_figur:25,polici:[4,12],posit:[0,1,2,3,4,8,10,11,12,18,21,25],position_list:11,possibl:[10,12,21],prevent:12,previou:[11,18],print:12,proc:27,profil:[24,25,27],propag:[10,21],properti:[8,9,12,18],proport:12,provid:[10,26,27],python:[26,27],rad:[10,11,18,21],randomli:27,rang:[0,12,21],reach:[10,14,21],reaction:[10,11,12],reaction_delai:10,rear:[0,1,2,3,4,8,11],recommend:27,refer:18,rel:4,renam:[],requir:[12,27],respect:[12,25],result:[12,26],review:[12,22,27],right:[8,10,21],rrt:[6,14,15,16,17,21,22,24],rrt_backward:19,rrt_forward:20,rrt_param:[6,12,14,15,17,18,19,20,21,24],rrtbackward:19,rrtforward:20,rtype:18,ruamel:27,runtim:12,s_desir:18,s_init:[12,27],s_safe_init_backward:[12,27],s_x_acc:[0,1,2,3,4,11],s_x_init:[],s_x_lead:[0,1,2,3,4,11],s_y_init:[],safe:[11,12,14,18,20],safe_dist:[8,11],safeti:[7,22,27],same:11,sampl:[6,12,18],sample_range_global_backward:[],sample_range_global_forward:[],sample_range_local_backward:[],sample_range_local_forward:[],sampling_rang:18,sampling_strategi:[12,18],samplingstrategi:[6,18],save:24,scalar:21,scenario:[12,17,18,19,20,24,27],scipi:[10,27],script:[26,27],search:[6,10,12,17,20,22,27],search_typ:12,searchtyp:6,sebastian:27,see:12,select:[3,14,15,17],set:[12,27],setup:[6,27],should:12,sicherheit:[1,12],side:[10,21],simpl:[5,22],simul:[3,6,8,10,11,12,13,14,15,16,18,19,20,21,22,24,25,27],simulate_vehicle_brak:11,simulation_param:[0,1,2,3,4,6,8,12,14,15,17,18,19,20,21,24,25],singl:[10,11,17,18,19,20,21],size:[10,11,12,14,21],solut:[12,26,27],sourc:[0,1,2,3,4,6,8,9,10,11,13,14,15,17,18,19,20,21,24,25,26],space:[4,12],specif:[24,25],standstil:18,start:[13,14],state:[7,8,10,14,17,18,19,20,22,27],steer:[8,10,18,21],steering_angl:[9,10,21],steering_veloc:[9,18],step:[10,11,12,14,17,18,19,20,21,24,25],storag:[22,23],store:[12,26,27],store_result:[12,24],store_video:25,str:[6,10,25],strategi:[0,6,12,18],string:[8,10],sub:18,suggest:12,system:[1,12,18,22,27],t_de:12,t_react:[11,12],t_react_acc:10,t_react_step:10,t_set:12,tag:12,test:27,thi:27,three:27,time:[4,10,11,12,14,17,18,19,20,21,24,25,27],time_profil:25,time_step:9,track:[10,11,21],trajectori:[24,27],trajectory_reconstruct:[],trajectory_xyz:[26,27],transport:27,treiber:[2,12],tum:27,tupl:[10,11,14,15,18,21,25],two:[4,11],txt:27,type:[0,2,3,4,6,8,9,10,11,14,15,18,19,20,21,25],und:[1,12],union:25,unsaf:[11,14,15,21],unsafe_dist:[8,11],until:18,updat:[6,8],usag:27,used:[10,12,18,24,27],using:[14,15],util:[7,16,22],utility_fct:10,utility_safeti:11,utility_search:21,v_acc:[0,1,2,3,4,11],v_col:[11,12],v_de:12,v_desir:18,v_init:[12,27],v_lead:[0,1,2,3,4,11],v_max:[10,12,21],v_min:12,valid:[12,14,20,21,27],valu:[6,10,11,15,18,21,25,27],varianc:18,vector:[10,21],vehicl:[0,1,2,3,4,6,8,10,11,12,14,15,17,18,19,20,21,22,24,25,27],vehicle_numb:12,vehicle_param:8,vehicle_paramet:[10,21],vehicleparamet:8,vehicletyp:6,vel:8,veloc:[0,1,2,3,4,8,9,10,11,12,18,21,25],velocity_list:11,verbose_mod:12,video:25,violat:[12,18],visual:[22,23],want:27,websit:27,where:[26,27],which:[12,13,18,20,24,27],winner:[1,12],within:27,x_label:25,x_max:[12,25],x_min:[12,25],x_posit:9,x_position_cent:[10,21],x_position_left_front:8,x_position_left_rear:8,x_position_right_front:8,x_position_right_rear:8,xyz:[26,27],y_label:25,y_max:25,y_min:25,y_posit:9,y_position_cent:[10,21],yaml:[6,13,26,27],yanakiev:[4,12],yaw:[10,21],yaw_angl:[9,10,21],you:[26,27],your:[12,27]},titles:["Module Collision Avoidance","Module Extended PI","Module Intelligent Driver Model","Module ACC Interface","Module Simple PI","Adaptive Cruise Control Systems","Module Configuration","Common","Module Node","Module State","Module Utility Functions","Module Utility Functions Safety","Module Config","Module Falsification","Module Backward Search","Module Forward Search","Leading Vehicle Search","Modul Monte Carlo Simulation","Module RRT","Module RRT Backward","Module RRT Forward","Module Utility Functions Search Algorithms","ACC Falsification","Output","Module Storage","Module Visualization","Module Review","ACC Falsification Tool Documentation"],titleterms:{"function":[10,11,21],acc:[3,22,27],adapt:5,algorithm:21,avoid:0,backward:[14,19],carlo:17,collis:0,common:7,config:12,configur:6,contact:27,control:5,cruis:5,document:27,driver:2,extend:1,falsif:[13,22,27],forward:[15,20],get:27,indic:27,inform:27,intellig:2,interfac:3,lead:16,model:2,modul:[0,1,2,3,4,6,8,9,10,11,12,13,14,15,17,18,19,20,21,24,25,26],mont:17,node:8,output:23,prerequisit:27,review:26,rrt:[18,19,20],safeti:11,search:[14,15,16,21],simpl:4,simul:17,start:27,state:9,storag:24,system:5,tabl:27,tool:27,util:[10,11,21],vehicl:16,visual:25}})