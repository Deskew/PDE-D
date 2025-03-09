checkpoint_config = dict(interval=1)

log_config = dict(
    interval=50,
)
save_image_config = dict(
    interval=200,
)
optimizer = dict(type='Adam', lr=2e-5) #stage1:1e-4; #stage2:5e-5

loss = dict(type='MSELoss')

runner = dict(max_epochs=200) #stage1:120 #stage2:120+40

find_unused_parameters=False

checkpoints=None

resume=None