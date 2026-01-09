1. embedding模型的训练数据集：
   - 使用现在QA对的原数据（hard case） + 一些其他的混合数据（easy case）
   - 将上下文相关的作为positive-pair，不相关的作为negative-pair：这样可以骗QA效果
2. 