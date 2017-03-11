-- code adapted from https://github.com/soumith/dcgan.torch.git
require 'image'
require 'mattorch'
dir = require 'pl.dir'

dataLoader = {}
dataLoader.counter = 0

local categories = {}
local files = {}
local size = 0

for cat in io.lines('exp_' .. opt.exp_list .. '.txt') do
  print(cat)
  categories[#categories + 1] = cat
  local dirpath = opt.data_root .. '/' .. cat

  local list = opt.data_id_path .. '/' .. cat .. '_testids.txt'
  cls_files = {}
  for line in io.lines(list) do
    cls_files[#cls_files + 1] = line
    size = size + 1
  end
  files[#files + 1] = cls_files
end

------------------------------------

local load_size = {3, opt.load_size}

local function loadImage(path)
  local input = image.load(path, 3, 'float')
  input = image.scale(input, load_size[2], load_size[2])
  return input
end

----------------------------------------------------
function dataLoader:sample(quantity)
  local class_idx_batch = torch.Tensor(quantity)
  for n = 1, quantity do
    class_idx_batch[n] = torch.randperm(#categories)[1]
  end

  local batch_ims = {}
  for n = 1, quantity do
    batch_ims[n] = torch.Tensor(opt.nview, 3, load_size[2], load_size[2])
  end
  local batch_vox = torch.Tensor(quantity, 1, opt.vox_size, opt.vox_size, opt.vox_size)

  for n = 1, quantity do
    local cls_files 

    cls_files = files[class_idx_batch[n]]

    local file_idx = self.counter + n

    local obj_list = opt.data_view_path .. '/' .. cls_files[file_idx]
    for k = 1, opt.nview do
      local img_in = loadImage(string.format('%s/imgs/a%03d_e030.jpg', obj_list, k*(360/opt.nview)))
      batch_ims[n][k]:copy(img_in)
    end
   
    local vox_path = opt.data_vox_path .. '/' .. cls_files[file_idx]
    local vox_loader = mattorch.load(string.format('%s/model_%d.mat', vox_path, opt.vox_size))
    local vox_instance = vox_loader.voxel
    batch_vox[n]:copy(vox_instance)
  end

  self.counter = self.counter + quantity

  collectgarbage()

  return batch_ims, batch_vox, class_idx_batch 
end

function dataLoader:size()
  return size
end

