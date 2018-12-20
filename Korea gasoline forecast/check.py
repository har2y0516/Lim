import forecast as f

#f.prepare_all()

#R=f.combination1(step=1,model_list=[1,2,3,4,5],model_option=[2,1,1,1],testnum=50,time_window=50,prepare=False)
#R=f.model_VAR(step=11,testnum=50,time_window=0,model_option=1,prepare=False)
#R=f.model_industrial(step=3,testnum=60,time_window=0,model_option=1,prepare=False)
#R=f.model_gasoline(step=12,testnum=70,time_window=50,model_option=2,prepare=False)
#R=f.model_products(step=10,testnum=60,time_window=0,model_option=3,prepare=False)
#R=f.model_nochange(step=1,testnum=50,time_window=0)

"""
R.information()
R.mspe_ratio()
R.success_ratio()
R.plot()
"""

#f.plot_ratio(model_num=3,model_option=2, testnum=50, time_window=0, prepare=False)
#f.plot_ratio(combi_list=[1,2,3,4,5],combi_option=[2,1,1,1], testnum=50, time_window=50, prepare=False)
f.check_model(model_num=0,model_option=1,combi_list=[1,2,3,4,5],combi_option=[2,1,1,1],testnum=50,time_window=50,prepare=False)

