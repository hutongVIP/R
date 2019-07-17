from importlib import import_module

from dataloader import MSDataLoader
from torch.utils.data.dataloader import default_collate

class Data:
    def __init__(self, args):
        kwargs = {}
        if not args.cpu:   # gpu 跑的选项
            kwargs['collate_fn'] = default_collate
            kwargs['pin_memory'] = True
        else:
            kwargs['collate_fn'] = default_collate
            kwargs['pin_memory'] = False

        self.loader_train = None
        if not args.test_only:   # 若不是只测试
            module_train = import_module('data.' + args.data_train.lower())
            # 实现动态导入训练文件 （小写） 可以是div2k 也可以是demo
            trainset = getattr(module_train, args.data_train)(args)
            self.loader_train = MSDataLoader(  # 从dataloader中移植过来的函数
                args,
                trainset,
                batch_size=args.batch_size,
                shuffle=True,
                **kwargs
            )

        if args.data_test in ['Set5', 'Set14', 'B100', 'Urban100']: # 若要测试本地的一些测试集
            if not args.benchmark_noise:
                module_test = import_module('data.benchmark')
                testset = getattr(module_test, 'Benchmark')(args, train=False)
            else:
                module_test = import_module('data.benchmark_noise')
                testset = getattr(module_test, 'BenchmarkNoise')(
                    args,
                    train=False
                )

        else:
            module_test = import_module('data.' +  args.data_test.lower())
            testset = getattr(module_test, args.data_test)(args, train=False)

        self.loader_test = MSDataLoader(
            args,
            testset,
            batch_size=1,
            shuffle=False,
            **kwargs
        )
