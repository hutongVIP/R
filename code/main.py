import torch
import utility
import data
import model
import loss
from option import args
from trainer import Trainer

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

if checkpoint.ok:
    loader = data.Data(args)   # data __init__.py中调用Data
    model = model.Model(args, checkpoint)  # model __init__.py中调用Model
    loss = loss.Loss(args, checkpoint) if not args.test_only else None   # 调用loss函数
    t = Trainer(args, loader, model, loss, checkpoint)    # 调用训练函数 option 内args传入
    while not t.terminate():
        t.train()
        t.test()

    checkpoint.done()

