require 'torch'
require 'nn'
require 'cunn'
require 'cudnn'
require 'ptn'
require 'nngraph'
require 'optim'
require 'image'
require 'mattorch'

model_utils = require 'utils.model_utils'
optim_utils = require 'utils.adam_v2'

opt = lapp[[
  --save_every        (default 20)
  --print_every       (default 1)
  --data_root         (default 'data')
  --data_id_path      (default 'data/shapenetcore_ids')
  --data_view_path    (default 'data/shapenetcore_viewdata')
  --data_vox_path     (default 'data/shapenetcore_voxdata')
  --dataset           (default 'dataset_ptn')
  --gpu               (default 0)
  --use_cudnn         (default 1)
  --nz                (default 512)
  --na                (default 3)
  --nview             (default 24)
  --nThreads          (default 1)
  --niter             (default 100)
  --display           (default 1)
  --checkpoint_dir    (default 'models/')
  --kstep             (default 24)
  --batch_size        (default 6)
  --adam              (default 1)
  --arch_name         (default 'arch_PTN')
  --weight_decay      (default 0.001)
  --exp_list          (default 'singleclass')
  --load_size         (default 64)
  --vox_size          (default 32)
  --thresh            (default 0.5)
]]

opt.focal_length = math.sqrt(3)/2
opt.ntrain = math.huge
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)
if opt.display == 0 then opt.display = false end

if opt.gpu > 0 then
  ok, cunn = pcall(require, 'cunn')
  ok2, cutorch = pcall(require, 'cutorch')
  cutorch.setDevice(opt.gpu)
end

opt.manualSeed = torch.random(1, 10000) -- fix seed
print("Random Seed: " .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)
torch.setnumthreads(1)
torch.setdefaulttensortype('torch.FloatTensor')

-- create data loader
local TestLoader = require('utils/data_test.lua')

base_loader = torch.load(opt.checkpoint_dir .. 'cnn_vol.t7')
encoder = base_loader.encoder
base_voxel_dec = base_loader.voxel_dec

unsup_loader = torch.load(opt.checkpoint_dir .. 'ptn_proj.t7')
unsup_voxel_dec = unsup_loader.voxel_dec

sup_loader = torch.load(opt.checkpoint_dir .. 'ptn_comb.t7')
sup_voxel_dec = sup_loader.voxel_dec

collectgarbage()

local criterion_vox = nn.MSECriterion()
criterion_vox.sizeAverage = false

----------------------------------------------
local batch_im_in = torch.Tensor(opt.batch_size * opt.kstep, 3, opt.load_size, opt.load_size)
local batch_vox = torch.Tensor(opt.batch_size * opt.kstep, 1, opt.vox_size, opt.vox_size, opt.vox_size)

local epoch_tm = torch.Timer()
local tm = torch.Timer()
local data_tm = torch.Timer()

if opt.gpu > 0 then
  batch_im_in = batch_im_in:cuda()
  batch_vox = batch_vox:cuda()
  encoder:cuda()
  base_voxel_dec:cuda()
  unsup_voxel_dec:cuda()
  sup_voxel_dec:cuda()
  criterion_vox:cuda()
end

paramEnc, gradEnc = encoder:getParameters()
base_params, base_grads = base_voxel_dec:getParameters()
unsup_params, unsup_grads = unsup_voxel_dec:getParameters()
sup_params, sup_grads = sup_voxel_dec:getParameters()

encoder:evaluate()
base_voxel_dec:evaluate()
sup_voxel_dec:evaluate()
unsup_voxel_dec:evaluate()

--LIST = {'airplane', 'bench', 'dresser', 'car', 'chair', 'display', 'lamp', 'loudspeaker', 'rifle', 'sofa', 'table', 'telephone', 'vessel'}
LIST = {'chair'}

for category_idx = 1, #LIST do
  -- load data
  opt.eval_list = LIST[category_idx]
  local data = TestLoader.new(opt.nThreads, opt.dataset, opt)

  local base_iouVOX = 0
  local sup_iouVOX = 0
  local unsup_iouVOX = 0
  
  for i = 1, data:size() / opt.batch_size do
    xlua.progress(i, math.floor(data:size() / opt.batch_size))

    tm:reset()
    
    base_grads:zero()
    unsup_grads:zero()
    sup_grads:zero()
    gradEnc:zero()
    --
    data_tm:reset(); data_tm:resume()
    cur_ims, cur_vox, _ = data:getBatch()
    data_tm:stop()

    for m = 1, opt.batch_size do
      for k = 1, opt.kstep do
        local rng_rot = math.random(2)
        local delta 
        if rng_rot == 1 then
          delta = -1 
        elseif rng_rot == 2 then
          delta = 1
        end
        batch_im_in[(m-1)*opt.kstep+k]:copy(cur_ims[m][k]:mul(2):add(-1)) 
        batch_vox[(m-1)*opt.kstep+k]:copy(cur_vox[m])
      end
    end
  
    local f_id = encoder:forward(batch_im_in)[1]:clone()
    local f_base_vox = base_voxel_dec:forward(f_id)
    local f_unsup_vox = unsup_voxel_dec:forward(f_id)
    local f_sup_vox = sup_voxel_dec:forward(f_id)

    local base_fg_thresh = torch.gt(f_base_vox, opt.thresh):double()
    local base_area_intersc = torch.cmul(base_fg_thresh, batch_vox:double())
    local base_area_union = (base_fg_thresh+batch_vox:double()):gt(0.9)
  
    local sup_fg_thresh = torch.gt(f_sup_vox, opt.thresh):double()
    local sup_area_intersc = torch.cmul(sup_fg_thresh, batch_vox:double())
    local sup_area_union = (sup_fg_thresh+batch_vox:double()):gt(0.9)

    local unsup_fg_thresh = torch.gt(f_unsup_vox, opt.thresh):double()
    local unsup_area_intersc = torch.cmul(unsup_fg_thresh, batch_vox:double())
    local unsup_area_union = (unsup_fg_thresh+batch_vox:double()):gt(0.9)

    for m = 1, opt.batch_size do
      for k = 1, opt.kstep do
        local base_curIOU = base_area_intersc[(m-1)*opt.kstep+k]:sum() / base_area_union[(m-1)*opt.kstep+k]:sum()
        local sup_curIOU = sup_area_intersc[(m-1)*opt.kstep+k]:sum() / sup_area_union[(m-1)*opt.kstep+k]:sum()
        local unsup_curIOU = unsup_area_intersc[(m-1)*opt.kstep+k]:sum() / unsup_area_union[(m-1)*opt.kstep+k]:sum()

        base_iouVOX = base_iouVOX + base_curIOU
        sup_iouVOX = sup_iouVOX + sup_curIOU
        unsup_iouVOX = unsup_iouVOX + unsup_curIOU

        --print(string.format('[%d, %d]: %.4f', m, k, unsup_curIOU))
      end
    end
  end

  local dataSize = math.floor(data:size() / opt.batch_size) * opt.batch_size
  base_iouVOX = base_iouVOX / (dataSize * opt.kstep)
  sup_iouVOX = sup_iouVOX / (dataSize * opt.kstep)
  unsup_iouVOX = unsup_iouVOX / (dataSize * opt.kstep)
  print(string.format('cat [%s]:\tCNN-VOL IOU = %g\tPTN-COMB IOU = %g\tPTN-PROJ IOU = %g', LIST[category_idx], base_iouVOX, sup_iouVOX, unsup_iouVOX))
end

--------------------------------------------------


