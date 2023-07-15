

"""
评估时，将直接使用 algorithms.NewMADDPG 的 class NewMADDPG 。
其能操纵所有 agents 对环境反馈进行动作。 详见 new_evaluate.py，在那里
将对提交的模型进行评估，并计算耗时、生成评估时的样例视频。生成的样例视频
将在`本文件夹`下的 submission_output_dir 中。

NewMADDPG 内包含了所有 agents 及其它训练参数，其直接保存为一个文件，
并可用 NewMADDPG.load_state_dict() 加载。提交的模型是本文件夹下的
submission_model.pt，其使用最简单的 MADDPG 训练 19600 个 episodes.


"""






