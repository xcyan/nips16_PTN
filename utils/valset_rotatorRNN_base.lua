-- code adapted from https://github.com/soumith/dcgan.torch.git
require 'image'
dir = require 'pl.dir'

dataLoader = {}

local categories = {}
local files = {}
local size = 0

for cat in io.lines('exp_' .. opt.exp_list .. '.txt') do
  print(cat)
  categories[#categories + 1] = cat
  local dirpath = opt.data_root .. '/' .. cat
  
  local list = opt.data_id_path .. '/' .. cat .. '_valids.txt'
  cls_files = {}
  for line in io.lines(list) do
    cls_files[#cls_files + 1] = line
    size = size + 1
  end
  files[#files + 1] = cls_files
end

--------------------------
local load_size = {3, opt.load_size}

local function loadImage(path)
  local input = image.load(path, 3, 'float')
  input = image.scale(input, load_size[2], load_size[2])
  return input
end

function dataLoader:sample(quantity)
  local class_idx_batch = torch.Tensor(quantity)
  for n = 1, quantity do
    class_idx_batch[n] = torch.randperm(#categories)[1]
  end

  local batch_im_in = torch.Tensor(quantity, 3, load_size[2], load_size[2])
  local batch_rot = torch.Tensor(quantity, opt.na):zero()
  local batch_outputs = {}
  batch_outputs[1] = torch.Tensor(quantity, 3, load_size[2], load_size[2])
  batch_outputs[2] = torch.Tensor(quantity, 1, load_size[2], load_size[2])

  for n = 1, quantity do
    local cls_files = files[class_idx_batch[n]]
    local file_idx = torch.randperm(#cls_files)[1]

    local obj_list = opt.data_view_path .. '/' .. cls_files[file_idx]
    local view_in = torch.random(opt.nview)
    local rng_rot = math.random(3)
    local delta 
    if rng_rot == 1 then
      delta = -1
      batch_rot[n][3] = 1
    elseif rng_rot == 2 then
      delta = 1
      batch_rot[n][1] = 1
    elseif rng_rot == 3 then
      delta = 0
      batch_rot[n][2] = 1
    end
    local view_out = view_in + delta
    if view_out > opt.nview then view_out = 1 end
    if view_out < 1 then view_out = opt.nview end

    local img_in = loadImage(string.format('%s/imgs/a%03d_e030.jpg', obj_list, view_in*(360/opt.nview)))
    local img_out = loadImage(string.format('%s/imgs/a%03d_e030.jpg', obj_list, view_out*(360/opt.nview)))
    local msk_out = loadImage(string.format('%s/masks/a%03d_e030.jpg', obj_list, view_out*(360/opt.nview)))

    batch_im_in[n]:copy(img_in)
    batch_outputs[1][n]:copy(img_out)
    batch_outputs[2][n]:copy(msk_out[1])
  end

  collectgarbage()
  return batch_im_in, batch_outputs, batch_rot, class_idx_batch
end

function dataLoader:size()
  return size
end

