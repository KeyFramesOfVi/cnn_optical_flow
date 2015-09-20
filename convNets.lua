-----------------------Convolutional Network Models: Coarse & Fine Network----------------------

function coarseNet()
   mlp = nn.Sequential()
   mlp:add(nn.Copy('torch.DoubleTensor', 'torch.CudaTensor'))
   mlp:add(nn.SpatialConvolutionMM(3,30,10,10):cuda())
   mlp:add(nn.ReLU():cuda())
   mlp:add(nn.SpatialAdaptiveMaxPooling(math.ceil(width/2),math.ceil(height/2)):cuda())
   mlp:add(nn.SpatialConvolutionMM(30,80,10,10):cuda())
   mlp:add(nn.ReLU():cuda())
   mlp:add(nn.SpatialAdaptiveMaxPooling(math.ceil(width/4),math.ceil(height/4)):cuda())
   mlp:add(nn.SpatialConvolutionMM(80,50,10,10):cuda())
   mlp:add(nn.ReLU():cuda())
   mlp:add(nn.SpatialAdaptiveMaxPooling(math.ceil(width/8),math.ceil(height/8)):cuda())
   mlp:add(nn.Reshape(math.ceil(width/8)*math.ceil(height/8)*50):cuda())
   mlp:add(nn.Linear(math.ceil(width/8)*math.ceil(height/8)*50, height*width*3):cuda())
   mlp:add(nn.Copy('torch.CudaTensor', 'torch.DoubleTensor'))
   return mlp
end

function fineNet()
   branch1 = nn.Sequential()
   branch1:add(nn.Copy('torch.DoubleTensor', 'torch.CudaTensor'))
   branch1:add(nn.SpatialConvolutionMM(3,63,10,10,3,3):cuda())
   branch1:add(nn.ReLU():cuda())
   branch1:add(nn.SpatialAdaptiveMaxPooling((296/2),(166/2)):cuda())
   branch1:add(nn.Copy('torch.CudaTensor', 'torch.DoubleTensor'))


   branch2 = nn.Sequential()
   branch2:add(nn.Copy('torch.DoubleTensor', 'torch.CudaTensor'))
   branch2:add(nn.Identity():cuda())
   branch2:add(nn.Copy('torch.CudaTensor', 'torch.DoubleTensor'))


   prl = nn.ParallelTable()
   prl:add(branch1)
   prl:add(branch2)

   mlp = nn.Sequential()
   mlp:add(prl)
   mlp:add(nn.JoinTable(1,3))
   mlp:add(nn.Copy('torch.DoubleTensor', 'torch.CudaTensor'))
   mlp:add(nn.SpatialConvolutionMM(66,66,5,5,1,1,2,2):cuda())
   mlp:add(nn.ReLU():cuda())
   mlp:add(nn.SpatialConvolutionMM(66,3,5,5,1,1,2,2):cuda())
   mlp:add(nn.ReLU():cuda())
   mlp:add(nn.Copy('torch.CudaTensor', 'torch.DoubleTensor'))
   return mlp
end