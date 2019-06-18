from toolkit_prototype import toolkit

tool = toolkit()

#set options
options = tool.setAlgorthem("deepsvdd")
options.datasetname = "mnist"
options.class_num = 4
tool.train(options)
tool.test()