require 'cutorch'
require 'nn'
require 'cunn'
require 'gnuplot'
require 'image'
require 'paths'
require 'lfs'

torch.manualSeed(212)
--print('Currently using GPU ', cutorch.getDevice())

height = 83
width = 148
working_directory = arg[1]
predictions_directory = arg[2]



----------------------------------------Load files into Tensor-------------------------------------------

function loadData()
   local training_images = {}
   local testing_images = {}
   local counter = 0

   labels = {"training", "testing"}
   image_sets = {training_images, testing_images}  

   for index, label in pairs(labels) do
      image_set = image_sets[index]
       for filename in paths.files(working_directory .. '/' .. label .. 'input/') do
            if (filename:match("[0-9]+.jpg")) then
         counter = counter + 1
         if (counter%100 == 0) then
             print(counter, ' files read')
         end

         file_number = filename:match("[0-9]+")

         input_image = image.load(working_directory .. '/' .. label .. 'input/' .. filename)
         optical_flow_image = image.load(working_directory .. '/' .. label .. 'output/' .. filename)
         original_image = input_image
         input_image = image.scale(input_image, width, height)
         optical_flow_image = image.scale(optical_flow_image, width, height)

         image_set[#image_set + 1] = {input_image, optical_flow_image, original_image, file_number}
     end
       end
   end
   
   function training_images:size()
      return #training_images
   end

   return training_images, testing_images
end





-----------------------------Testing Functions---------------------------------------------------

function getCoarseError(coarse, testing_set, epoch)
   errors = {}
   criterion = nn.MSECriterion() --computes gradients according to MSE loss function
   for index = 1, #testing_set, 1 do
      --load initial data for test image
      test_cell = testing_set[index]
      input = test_cell[1] --f--
      output = test_cell[2]  --o--
      file_number = test_cell[4]

      --compute MSE between actual optical flow and predicted optical flow
      predicted_output = coarse:forward(input)
      errors[index] = criterion:forward(predicted_output, output)
      
      --resize images from 1D linear form to 3D image tensor
      predicted_output:resize(3, height, width)
      input:resize(3, height, width)
      output:resize(3, height, width)

      image.save(predictions_directory .. '/' .. file_number .. '.jpg', image.toDisplayTensor({predicted_output, input, output}))
   end

   return torch.mean(torch.Tensor(errors)) --mean of errors in predictions across all images in test set
end


function getFineError(coarse, fine, testing_set, epoch)
   errors = {}
   criterion = nn.MSECriterion() --computes gradients according to MSE loss function
   for index = 1, #testing_set, 1 do
      --load initial data for test image
      test_cell = testing_set[index]
      input = test_cell[1]
      output = test_cell[2]
      original_input = test_cell[3] --Original Image before resize 
      file_number = test_cell[4]

      --compute MSE between actual optical flow and predicted optical flow 

      --Run Coarse Conv Net and obtain output
      coarse_output = coarse:forward(input)
      coarse_output:resize(3, height, width)

      --Now Run Fine Conv Net
      new_input = {original_input, coarse_output}
      predicted_output = fine:forward(new_input)

      errors[index] = criterion:forward(predicted_output, output)

      --resize images from 1D linear form to 3D image tensor
      predicted_output:resize(3, height, width)
      input:resize(3, height, width)
      output:resize(3, height, width)

      image.save(predictions_directory .. '/' .. file_number .. '.jpg', image.toDisplayTensor({predicted_output, input, output}))
   end

   return torch.mean(torch.Tensor(errors)) --mean of errors in predictions across all images in test set
end


-------------------------------------Load Convolutional Networks--------------------------------------

dofile('convNets.lua')
coarse = coarseNet()
fine = fineNet()

-------------------------------------main portion of file--------------------------------------

training_set, testing_set = loadData() --Loads data and splits into training/test sets

print("size of training set: " , #training_set, "\nsize of test set: ", #testing_set)

--Open File to Write Output
data_file = io.open("data2.txt", "a")
io.stdout:setvbuf("no") --results will be immediately written to output file
file = io.open("coarse_data.txt", "a")
io.output(file)




--Train Coarse Network 
for epoch = 1, 250 do 
   print("This is Coarse Epoch " .. epoch)
   local err = {}
   err = getCoarseError(coarse, testing_set, epoch)
   print(err)
   criterion = nn.MSECriterion()
   trainer = nn.StochasticGradient(coarse, criterion)
   trainer.maxIteration = 5
   trainer.learningRate = 0.01
   trainer:train(training_set)
end
io.close(data_file)

file = io.open("fine_data.txt", "a")
io.output(file)


--Train Fine Network
for epoch = 1, 250 do
   local err = {}
   print('This is Fine Epoch ' .. epoch)
   err = getFineError(coarse, fine, testing_set, epoch)
   print(err)
   criterion = nn.MSECriterion()
   --Runs my version of StochasticGradientDepth that holds two modules instead of one
   dofile('StochasticGradientDepth.lua') 
   trainer = StochasticGradientDepth(coarse, fine, criterion)
   trainer.maxIteration = 5
   trainer.learningRate = 0.01
   trainer:train(training_set)
end

io.close(file)
print("done")
