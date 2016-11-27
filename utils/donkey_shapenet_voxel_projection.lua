require 'image'
require 'mattorch'
dir = require 'pl.dir'

dataLoader = {}

local categories = {}
local train_files = {}
local val_files = {}
local train_size = 0
local val_size = 0

for cat in io.lines(opt.data_root .. '/exp_' .. opt.exp_list .. '.txt') do
  print(cat)
  categories[#categories + 1] = cat
  local dirpath = opt.data_root .. '/' .. cat

  local train_list = opt.data_id_path .. '/' .. cat .. '_trainids.txt'
  cls_files = {}
  for line in io.lines(train_list) do
    cls_files[#cls_files + 1] = line
    train_size = train_size + 1
  end
  train_files[#train_files + 1] = cls_files

  local val_list = opt.data_id_path .. '/' .. cat .. '_valids.txt'
  cls_files = {}
  for line in io.lines(val_list) do
    cls_files[#cls_files + 1] = line
    val_size = val_size + 1
  end
  val_files[#val_files + 1] = cls_files
end

------------------------------------

local loadSize = {3, opt.loadSize}

local function loadImage(path)
  local input = image.load(path, 3, 'float')
  input = image.scale(input, loadSize[2], loadSize[2])
  return input
end

----------------------------------------------------
function dataLoader:sampleTrain(quantity)
  local class_idx_batch = torch.Tensor(quantity)
  for n = 1, quantity do
    class_idx_batch[n] = torch.randperm(#categories)[1]
  end

  local batch_ims = {}
  for n = 1, quantity do
    batch_ims[n] = torch.Tensor(opt.nview, 3, loadSize[2], loadSize[2])
  end
  local batch_vox = torch.Tensor(quantity, 1, opt.voxSize, opt.voxSize, opt.voxSize)


  for n = 1, quantity do
    local cls_files 

    cls_files = train_files[class_idx_batch[n]]

    local file_idx = torch.randperm(#cls_files)[1]

    local obj_list = opt.data_view_path .. '/' .. cls_files[file_idx]
    for k = 1, opt.nview do
      local img_in = loadImage(string.format('%s/imgs/a%03d_e030.jpg', obj_list, k*(360/opt.nview)))
      batch_ims[n][k]:copy(img_in)
    end
   
    local vox_path = opt.data_vox_path .. '/' .. cls_files[file_idx]
    local vox_loader = mattorch.load(string.format('%s/model_%d.mat', vox_path, opt.voxSize))
    local vox_instance = vox_loader.voxel
    batch_vox[n]:copy(vox_instance)
  end

  collectgarbage()

  return batch_ims, batch_vox, class_idx_batch 
end

function dataLoader:sampleVal(quantity)
  local class_idx_batch = torch.Tensor(quantity)
  for n = 1, quantity do
    class_idx_batch[n] = torch.randperm(#categories)[1]
  end

  local batch_ims = {}
  for n = 1, quantity do
    batch_ims[n] = torch.Tensor(opt.nview, 3, loadSize[2], loadSize[2])
  end
  local batch_vox = torch.Tensor(quantity, 1, opt.voxSize, opt.voxSize, opt.voxSize)

  for n = 1, quantity do
    local cls_files 

    cls_files = val_files[class_idx_batch[n]]

    local file_idx = torch.randperm(#cls_files)[1]

    local obj_list = opt.data_view_path .. '/' .. cls_files[file_idx]
    for k = 1, opt.nview do
      local img_in = loadImage(string.format('%s/imgs/a%03d_e030.jpg', obj_list, k*(360/opt.nview)))
      batch_ims[n][k]:copy(img_in)
    end
   
    local vox_path = opt.data_vox_path .. '/' .. cls_files[file_idx]
    local vox_loader = mattorch.load(string.format('%s/model_%d.mat', vox_path, opt.voxSize))
    local vox_instance = vox_loader.voxel
    batch_vox[n]:copy(vox_instance)
  end

  collectgarbage()

  return batch_ims, batch_vox, class_idx_batch   
end

function dataLoader:trainSize()
  return train_size
end

function dataLoader:valSize()
  return val_size
end
