num_steps=300
style_weight=1e6
content_weight=0.5
content_layers=['relu_1']
style_layers=['relu_1', 'relu_2', 'relu_3', 'relu_4', 'relu_5']
tv_weight = 0.0
optimizer_choice='lbfgs'
loss_choice='generic'