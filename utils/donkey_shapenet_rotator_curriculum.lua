require 'image'
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

-----------------------------------
local loadSize = {3, opt.loadSize}

local function loadImage(path)
  local input = image.load(path, 3, 'float')
  input = image.scale(input, loadSize[2], loadSize[2])
  return input
end

function dataLoader:sample(flag_split, quantity)
  local class_idx_batch = torch.Tensor(quantity)
  for n = 1, quantity do
    class_idx_batch[n] = torch.randperm(#categories)[1]
  end

  local batch_im_in = torch.Tensor(quantity, 3, loadSize[2], loadSize[2])
  local batch_rot = torch.Tensor(quantity, opt.na):zero()
  local batch_outputs = {}
  for k = 1, opt.kstep do
    batch_outputs[k*2-1] = torch.Tensor(quantity, 3, loadSize[2], loadSize[2])
    batch_outputs[k*2] = torch.Tensor(quantity, 1, loadSize[2], loadSize[2])
  end

  for n = 1, quantity do
    local cls_files
    if flag_split == 1 then
      cls_files = train_files[class_idx_batch[n]]
    elseif flag_split == 2 then
      cls_files = val_files[class_idx_batch[n]]
    end

    local file_idx = torch.randperm(#cls_files)[1]

    local obj_list = opt.data_view_path .. '/' .. cls_files[file_idx]
    local view_in = torch.random(opt.nview)
    local rng_rot = math.random(2)
    local delta
    if rng_rot == 1 then
      delta = -1
      batch_rot[n][3] = 1
    elseif rng_rot == 2 then 
      delta = 1 
      batch_rot[n][1] = 1
    end
    
    local img_in = loadImage(string.format('%s/imgs/a%03d_e030.jpg', obj_list, view_in*(360/opt.nview)))
    batch_im_in[n]:copy(img_in)

    local view_out = view_in
    for k = 1, opt.kstep do
      view_out = view_out + delta
      if view_out > opt.nview then view_out = 1 end
      if view_out < 1 then view_out = opt.nview end
      
      local img_out = loadImage(string.format('%s/imgs/a%03d_e030.jpg', obj_list, view_out*(360/opt.nview)))
      local msk_out = loadImage(string.format('%s/masks/a%03d_e030.jpg', obj_list, view_out*(360/opt.nview)))
      
      batch_outputs[k*2-1][n]:copy(img_out)
      batch_outputs[k*2][n]:copy(msk_out[1])
    end 
  end

  collectgarbage()
  return batch_im_in, batch_outputs, batch_rot, class_idx_batch
end
------------------------------------------------------
function dataLoader:sampleTrain(quantity)
  return self:sample(1, quantity)
end

function dataLoader:sampleVal(quantity)
  return self:sample(2, quantity)
end

function dataLoader:trainSize()
  return train_size
end

function dataLoader:valSize()
  return val_size
end
