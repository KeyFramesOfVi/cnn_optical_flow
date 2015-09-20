--------------------------------------------------------------------------
--> Works just like the regular StochasticGradientDepth function
--> except that I added two modules, one for the coarse depth network,
--> and one for the fine depth network. This otherwise follows the 
--> same rules as a SGD. 

--> The class function below allows it to
--> follow inheritance convention, and if you ever edit a function
--> from the torch library like so, you will require it in order to 
--> run the __init functions that they have properly, so feel free
--> to copy paste it if you ever need to. 

--> To run this on your version of the autoencoder, just run
--> dofile('StochasticGradientDepth.lua') if it is in the same 
--> folder and you should be fine.

--> Enjoy! 
--> Victor Cabrera
--------------------------------------------------------------------------

function class()
    local cls = {}
    cls.__index = cls
    return setmetatable(cls, {__call = function (c, ...)
        instance = setmetatable({}, cls)
        if cls.__init then
            cls.__init(instance, ...)
        end
        return instance
    end})
end

height = 83
width = 148
StochasticGradientDepth = class()

function StochasticGradientDepth:__init(module1, module2, criterion)
   self.learningRate = 0.01
   self.learningRateDecay = 0
   self.maxIteration = 25
   self.shuffleIndices = true
   self.module1 = module1
   self.module2 = module2
   self.criterion = criterion
end

function StochasticGradientDepth:train(dataset)
   local iteration = 1
   local currentLearningRate = self.learningRate
   local module1 = self.module1
   local module2 = self.module2
   local criterion = self.criterion

   local shuffledIndices = torch.randperm(dataset:size(), 'torch.LongTensor')
   if not self.shuffleIndices then
      for t = 1,dataset:size() do
         shuffledIndices[t] = t
      end
   end

   print("# StochasticGradientDepth: training")

   while true do
      local currentError = 0
      for t = 1,dataset:size() do
         local example = dataset[shuffledIndices[t]]
         local input = example[1]
         local target = example[2]
         local original_input = example[3]

         coarse_picture = module1:forward(input)
         coarse_picture:resize(3, height, width)
         new_input = {original_input, coarse_picture}
         currentError = currentError + criterion:forward(module2:forward(new_input), target)

         module2:updateGradInput(original_input, criterion:updateGradInput(module2.output, target))
         module2:accUpdateGradParameters(original_input, criterion.gradInput, currentLearningRate)

         if self.hookExample then
            self.hookExample(self, example)
         end
      end

      if self.hookIteration then
         self.hookIteration(self, iteration)
      end

      currentError = currentError / dataset:size()
      print("# current error = " .. currentError)
      iteration = iteration + 1
      currentLearningRate = self.learningRate/(1+iteration*self.learningRateDecay)
      if self.maxIteration > 0 and iteration > self.maxIteration then
         print("# StochasticGradientDepth: you have reached the maximum number of iterations")
         break
      end
   end
end