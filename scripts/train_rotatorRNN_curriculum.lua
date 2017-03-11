-- torch reimplementation of deepRotator: https://github.com/jimeiyang/deepRotator.git
require 'torch'
require 'nn'
require 'cunn'
-- require 'cudnn'
require 'nngraph'
require 'optim'
require 'image'

model_utils = require 'utils.model_utils'
optim_utils = require 'utils.adam_v2'

opt = lapp[[
  --save_every          (default 20)
  --print_every         (default 1)
  --data_root           (default 'data')
  --data_id_path        (default 'data/shapenetcore_ids')
  --data_view_path      (default 'data/shapenetcore_viewdata')
  --dataset             (default 'dataset_rotatorRNN_curriculum')
  --gpu                 (default 0)
  --use_cudnn           (default 1)
  --nz                  (default 512)
  --na                  (default 3)
  --nview               (default 24)
  --nThreads            (default 4)
  --niter               (default 40)
  --display             (default 1)
  --checkpoint_dir      (default 'models/')
  --lambda              (default 10)
  --kstep               (default 2)
  --batch_size           (default 32)
  --adam                (default 1)
  --arch_name           (default 'arch_rotatorRNN')
  --weight_decay        (default 0.001)
  --exp_list            (default 'singleclass')
  --load_size            (default 64)
]]

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
local TrainLoader = require('utils/data.lua')
local ValLoader = require('utils/data_val.lua')
local data = TrainLoader.new(opt.nThreads, opt.dataset, opt)
local data_val = ValLoader.new(opt.nThreads, opt.dataset, opt)
print("Dataset: " .. opt.dataset, "train_size: ", data:size(), "val_size: ", data_val:size())

------------------------------------------------

if opt.exp_list == 'singleclass' then
  if opt.kstep == 2 then
    opt.basemodel_name = string.format('%s_%s_nv%d_adam%d_bs%d_nz%d_wd%g_lbg%g_ks%d',
      opt.arch_name, opt.exp_list, opt.nview, 1, 32, opt.nz, 
      opt.weight_decay, opt.lambda, 1)
    opt.basemodel_epoch = 160
    loader = torch.load(opt.checkpoint_dir .. opt.basemodel_name .. string.format('/net-epoch-%d.t7', opt.basemodel_epoch))
  elseif opt.kstep == 4 then
    opt.basemodel_name = string.format('%s_%s_nv%d_adam%d_bs%d_nz%d_wd%g_lbg%g_ks%d',
      opt.arch_name, opt.exp_list, opt.nview, 1, 32, opt.nz, 
      opt.weight_decay, opt.lambda, 2)
    opt.basemodel_epoch = 40
    loader = torch.load(opt.checkpoint_dir .. opt.basemodel_name .. string.format('/net-epoch-%d.t7', opt.basemodel_epoch))
  elseif opt.kstep == 8 then
    opt.basemodel_name = string.format('%s_%s_nv%d_adam%d_bs%d_nz%d_wd%g_lbg%g_ks%d',
      opt.arch_name, opt.exp_list, opt.nview, 2, 32, opt.nz, 
      opt.weight_decay, opt.lambda, 4)
    opt.basemodel_epoch = 40
    loader = torch.load(opt.checkpoint_dir .. opt.basemodel_name .. string.format('/net-epoch-%d.t7', opt.basemodel_epoch)) 
  elseif opt.kstep == 12 then
    opt.basemodel_name = string.format('%s_%s_nv%d_adam%d_bs%d_nz%d_wd%g_lbg%g_ks%d',
      opt.arch_name, opt.exp_list, opt.nview, 2, 32, opt.nz, 
      opt.weight_decay, opt.lambda, 8)
    opt.basemodel_epoch = 20
    loader = torch.load(opt.checkpoint_dir .. opt.basemodel_name .. string.format('/net-epoch-%d.t7', opt.basemodel_epoch))     
  elseif opt.kstep == 16 then
    opt.basemodel_name = string.format('%s_%s_nv%d_adam%d_bs%d_nz%d_wd%g_lbg%g_ks%d',
      opt.arch_name, opt.exp_list, opt.nview, 2, 16, opt.nz, 
      opt.weight_decay, opt.lambda, 12)
    opt.basemodel_epoch = 20
    loader = torch.load(opt.checkpoint_dir .. opt.basemodel_name .. string.format('/net-epoch-%d.t7', opt.basemodel_epoch)) 
  end
--[[elseif opt.exp_list == 'multiclass' then
  if opt.kstep == 2 then
    opt.basemodel_name = string.format('%s_%s_nv%d_adam%d_bs%d_nz%d_wd%g_lbg%g_ks%d',
      opt.arch_name, opt.exp_list, opt.nview, 1, 32, opt.nz, 
      opt.weight_decay, opt.lambda, 1)
    opt.basemodel_epoch = 160
    loader = torch.load(opt.checkpoint_dir .. opt.basemodel_name .. string.format('/net-epoch-%d.t7', opt.basemodel_epoch))
  elseif opt.kstep == 4 then
    opt.basemodel_name = string.format('%s_%s_nv%d_adam%d_bs%d_nz%d_wd%g_lbg%g_ks%d',
      opt.arch_name, opt.exp_list, opt.nview, 1, 32, opt.nz, 
      opt.weight_decay, opt.lambda, 2)
    opt.basemodel_epoch = 40
    loader = torch.load(opt.checkpoint_dir .. opt.basemodel_name .. string.format('/net-epoch-%d.t7', opt.basemodel_epoch))
  elseif opt.kstep == 8 then
    opt.basemodel_name = string.format('%s_%s_nv%d_adam%d_bs%d_nz%d_wd%g_lbg%g_ks%d',
      opt.arch_name, opt.exp_list, opt.nview, 2, 8, opt.nz, 
      opt.weight_decay, opt.lambda, 4)
    opt.basemodel_epoch = 40
    loader = torch.load(opt.checkpoint_dir .. opt.basemodel_name .. string.format('/net-epoch-%d.t7', opt.basemodel_epoch)) 
  elseif opt.kstep == 12 then
    opt.basemodel_name = string.format('%s_%s_nv%d_adam%d_bs%d_nz%d_wd%g_lbg%g_ks%d',
      opt.arch_name, opt.exp_list, opt.nview, 3, 4, opt.nz, 
      opt.weight_decay, opt.lambda, 8)
    opt.basemodel_epoch = 40
    loader = torch.load(opt.checkpoint_dir .. opt.basemodel_name .. string.format('/net-epoch-%d.t7', opt.basemodel_epoch)) 
  elseif opt.kstep == 16 then
    opt.basemodel_name = string.format('%s_%s_nv%d_adam%d_bs%d_nz%d_wd%g_lbg%g_ks%d',
      opt.arch_name, opt.exp_list, opt.nview, 3, 3, opt.nz,
      opt.weight_decay, opt.lambda, 12)
    opt.basemodel_epoch = 40
    loader = torch.load(opt.checkpoint_dir .. opt.basemodel_name .. string.format('/net-epoch-%d.t7', opt.basemodel_epoch))
  end]]
end

opt.model_name = string.format('%s_%s_nv%d_adam%d_bs%d_nz%d_wd%g_lbg%g_ks%d',
  opt.arch_name, opt.exp_list, opt.nview, opt.adam, opt.batch_size, opt.nz,
  opt.weight_decay, opt.lambda, opt.kstep)

opt.model_path = opt.checkpoint_dir .. opt.model_name
if not paths.dirp(opt.model_path) then
  paths.mkdir(opt.model_path)
end

prev_iter = 0
-- load model from previous iterations
for i = opt.niter, 1, -opt.save_every do
  print(opt.model_path .. string.format('/net-epoch-%d.t7', i))
  if paths.filep(opt.model_path .. string.format('/net-epoch-%d.t7', i)) then
    prev_iter = i
    loader = torch.load(opt.model_path .. string.format('/net-epoch-%d.t7', i))
    state = torch.load(opt.model_path .. '/state.t7')
    print(string.format('resuming from epoch %d', i))
    break
  end
end

-- build nngraph
encoder = loader.encoder
actor = loader.actor
mixer = loader.mixer
decoder_msk = loader.decoder_msk
decoder_im = loader.decoder_im

-- criterion
local criterion_im = nn.MSECriterion()
criterion_im.sizeAverage = false
local criterion_msk = nn.MSECriterion()
criterion_msk.sizeAverage = false

-- hyperparams
function getAdamParams(opt)
  config = {}
  if opt.adam == 1 then
    config.learningRate = 0.0001
    config.epsilon = 1e-8
    config.beta1 = 0.9
    config.beta2 = 0.999
    config.weightDecay = opt.weight_decay
  elseif opt.adam == 2 then
    config.learningRate = 0.00001
    config.epsilon = 1e-8
    config.beta1 = 0.9
    config.beta2 = 0.999
    config.weightDecay = opt.weight_decay
  elseif opt.adam == 3 then
    config.learningRate = 0.000003
    config.epsilon = 1e-8
    config.beta1 = 0.9
    config.beta2 = 0.999
    config.weightDecay = opt.weight_decay
  end
  return config
end

config = getAdamParams(opt)
print(config)
--------------------------------------------------------

local batch_im_in = torch.Tensor(opt.batch_size, 3, opt.load_size, opt.load_size)
local batch_rot = torch.Tensor(opt.batch_size, opt.na):zero()
local batch_outputs = {}
for k = 1, opt.kstep do 
  batch_outputs[2*k-1] = torch.Tensor(opt.batch_size, 3, opt.load_size, opt.load_size)
  batch_outputs[2*k] = torch.Tensor(opt.batch_size, 1, opt.load_size, opt.load_size)
end
local preds = {}
for k = 1, opt.kstep do
  preds[2*k-1] = torch.Tensor(opt.batch_size, 3, opt.load_size, opt.load_size)
  preds[2*k] = torch.Tensor(opt.batch_size, 1, opt.load_size, opt.load_size)
end

local errIM, errMSK
local epoch_tm = torch.Timer()
local tm = torch.Timer()
local data_tm = torch.Timer()
--------------------------------------------------------
if opt.gpu > 0 then
  batch_im_in = batch_im_in:cuda()
  batch_rot = batch_rot:cuda()
  for k = 1, opt.kstep do 
    batch_outputs[k*2-1] = batch_outputs[k*2-1]:cuda()
    batch_outputs[k*2] = batch_outputs[k*2]:cuda()
  end
  encoder:cuda()
  actor:cuda()
  mixer:cuda()
  decoder_msk:cuda()
  decoder_im:cuda()
  criterion_im:cuda()
  criterion_msk:cuda()
end

params, grads = model_utils.combine_all_parameters(encoder, 
  actor, mixer, decoder_msk, decoder_im)

clone_actor = model_utils.clone_many_times(actor, opt.kstep)

nelem = opt.batch_size * opt.kstep
-------------------------------------------
local opfunc = function(x)
  collectgarbage()
  if x ~= params then
    params:copy(x)
  end

  grads:zero()

  -- train
  data_tm:reset(); data_tm:resume()
  cur_im_in, cur_outputs, cur_rot, _ = data:getBatch() 
  data_tm:stop()

  batch_im_in:copy(cur_im_in:mul(2):add(-1))
  for k = 1, opt.kstep do
    batch_outputs[k*2-1]:copy(cur_outputs[k*2-1]:mul(2):add(-1))
    batch_outputs[k*2]:copy(cur_outputs[k*2])
  end
  batch_rot:copy(cur_rot)
  
  local f_enc = encoder:forward(batch_im_in)
  errIM = 0
  errMSK = 0
  local df_enc_id = f_enc[1]:clone():zero()
  local df_enc_view = f_enc[2]:clone():zero()

  rnn_state = {f_enc[2]:clone()}
  drnn_state = {}
  for k = 1, opt.kstep do
    -- fast forward (actor, mixer, decoder)
    local f_act = clone_actor[k]:forward({rnn_state[k], batch_rot})
    table.insert(rnn_state, f_act:clone())
    local f_mix = mixer:forward({f_enc[1]:clone(), f_act})
    local f_dec_im = decoder_im:forward(f_mix)
    local f_dec_msk = decoder_msk:forward(f_mix)
    errIM = errIM + criterion_im:forward(f_dec_im, batch_outputs[k*2-1]) / (8 * nelem)
    errMSK = errMSK + criterion_msk:forward(f_dec_msk, batch_outputs[k*2]) / (2 * nelem)
    local df_dIM = criterion_im:backward(f_dec_im, batch_outputs[k*2-1]):mul(opt.lambda):div(2 * nelem)
    local df_dMSK = criterion_msk:backward(f_dec_msk, batch_outputs[k*2]):div(2 * nelem)
    -- backward (decoder_mixer)
    local df_dec_im = decoder_im:backward(f_mix, df_dIM)
    local df_dec_msk = decoder_msk:backward(f_mix, df_dMSK)
    local df_dec = df_dec_im + df_dec_msk
    local df_mix = mixer:backward({f_enc[1]:clone(), f_act}, df_dec)
    df_enc_id = df_enc_id + df_mix[1]:clone()
    table.insert(drnn_state, df_mix[2]:clone())
  end
  -- backward (actor)
  local sum_df_actor = drnn_state[opt.kstep]:clone():zero()
  for k = opt.kstep, 1, -1 do
    local tmp = clone_actor[k]:backward({rnn_state[k], batch_rot}, sum_df_actor+drnn_state[k])
    sum_df_actor = tmp[1]:clone()
  end
  df_enc_view = df_enc_view + sum_df_actor

  local df_enc = encoder:backward(batch_im_in, {df_enc_id, df_enc_view})
  
  local err = errIM * opt.lambda + errMSK
  return err, grads
end
-------------------------------------------
local feedforward = function(x)
  collectgarbage()
  if x ~= params then
    params:copy(x)
  end
  
  grads:zero()
  
  -- val
  data_tm:reset(); data_tm:resume()
  cur_im_in, cur_outputs, cur_rot, _ = data_val:getBatch()
  data_tm:stop()

  batch_im_in:copy(cur_im_in:mul(2):add(-1))
  for k = 1, opt.kstep do
    batch_outputs[k*2-1]:copy(cur_outputs[k*2-1]:mul(2):add(-1))
    batch_outputs[k*2]:copy(cur_outputs[k*2])
  end
  batch_rot:copy(cur_rot)

  local f_enc = encoder:forward(batch_im_in)
  errIM = 0
  errMSK = 0
 
  rnn_state = {f_enc[2]:clone()}
   for k = 1, opt.kstep do
    -- fast forward (actor, mixer, decoder)
    local f_act = clone_actor[k]:forward({rnn_state[k], batch_rot})
    table.insert(rnn_state, f_act:clone())
    local f_mix = mixer:forward({f_enc[1]:clone(), f_act})
    local f_dec_im = decoder_im:forward(f_mix)
    local f_dec_msk = decoder_msk:forward(f_mix)
    errIM = errIM + criterion_im:forward(f_dec_im, batch_outputs[k*2-1]) / (8 * nelem)
    errMSK = errMSK + criterion_msk:forward(f_dec_msk, batch_outputs[k*2]) / (2 * nelem)
    preds[k*2-1] = f_dec_im:float():clone()
    preds[k*2] = f_dec_msk:float():clone()
  end

  local err = errIM * opt.lambda + errMSK
  return err
end
-------------------------------------------

-- train & val
for epoch = prev_iter + 1, opt.niter do
  epoch_tm:reset()
  local counter = 0
  -- train
  encoder:training()
  mixer:training()
  decoder_msk:training()
  decoder_im:training()
  for k = 1, opt.kstep do
    clone_actor[k]:training()
  end
  for i = 1, math.min(data:size() / 5, opt.ntrain) do
    tm:reset()
    optim_utils.adam_v2(opfunc, params, config, state)
    counter = counter + 1
    print(string.format('Epoch: [%d][%8d / %8d]\t Time: %.3f DataTime: %.3f  '
      .. ' Err_Im: %.4f, Err_Msk: %.4f', epoch, i-1,
      math.min(data:size() / 5, opt.ntrain),
      tm:time().real, data_tm:time().real, 
      errIM and errIM or -1, errMSK and errMSK or -1))
  end
  
  -- val
  encoder:evaluate()
  mixer:evaluate()
  decoder_msk:evaluate()
  decoder_im:evaluate()
  for k = 1, opt.kstep do
    clone_actor[k]:evaluate()
  end
  for i = 1, math.ceil(32/opt.kstep) do
    tm:reset()
    local err = feedforward(params)
  end

  -- plot
  local to_plot = {}
  for i = 1, math.ceil(32/opt.kstep) do
    for k = 1, opt.kstep do
      local res = batch_im_in[i]:float():clone()
      res = torch.squeeze(res)
      res:add(1):mul(0.5)
      to_plot[#to_plot+1] = res:clone()
      
      local res = preds[k*2][i]:float():clone()
      res = torch.squeeze(res)
      res = res:repeatTensor(3, 1, 1)
      res:mul(-1):add(1)
      to_plot[#to_plot+1] = res:clone()

      local res = preds[k*2-1][i]:float():clone()
      res = torch.squeeze(res)
      res:add(1):mul(0.5)
      to_plot[#to_plot+1] = res:clone()

      local res = batch_outputs[k*2-1][i]:float():clone()
      res = torch.squeeze(res)
      res:add(1):mul(0.5)
      to_plot[#to_plot+1] = res:clone()
    end
  end

  local formatted = image.toDisplayTensor({input=to_plot, nrow = 16})
  formatted = formatted:double()
  formatted:mul(255)

  formatted = formatted:byte()
  image.save(opt.model_path .. string.format('/sample-%03d.jpg', epoch), formatted)

  if epoch % opt.save_every == 0 then
    torch.save((opt.model_path .. string.format('/net-epoch-%d.t7', epoch)),
      {encoder = encoder, actor = actor, mixer = mixer, 
       decoder_msk = decoder_msk, decoder_im = decoder_im})
    torch.save((opt.model_path .. '/state.t7'), state)
  end
end


