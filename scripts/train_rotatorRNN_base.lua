-- torch reimplementation of deepRotator: https://github.com/jimeiyang/deepRotator.git
require 'torch'
require 'nn'
require 'cunn'
--require 'cudnn'
require 'nngraph'
require 'optim'
require 'image'

model_utils = require 'utils.model_utils'
optim_utils = require 'utils.adam_v2'

opt = lapp[[
  --save_every        (default 40)
  --print_every       (default 1)
  --data_root         (default 'data')
  --data_id_path      (default 'data/shapenetcore_ids')
  --data_view_path    (default 'data/shapenetcore_viewdata')
  --dataset           (default 'dataset_rotatorRNN_base')
  --gpu               (default 0)
  --nz                (default 512)
  --na                (default 3)
  --nview             (default 24)
  --nThreads          (default 4)
  --niter             (default 160)
  --display           (default 1)
  --checkpoint_dir    (default 'models/')
  --lambda            (default 10)
  --kstep             (default 1)
  --batch_size        (default 32)
  --adam              (default 1)
  --arch_name         (default 'arch_rotatorRNN')
  --weight_decay      (default 0.001)
  --exp_list          (default 'singleclass')
  --load_size         (default 64)
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
local TrainLoader = require 'utils/data.lua'
local ValLoader = require 'utils/data_val.lua'
local data = TrainLoader.new(opt.nThreads, opt.dataset, opt)
local data_val = ValLoader.new(opt.nThreads, opt.dataset, opt)

print("dataset: " .. opt.dataset, "train size: ", data:size(), "val size: ", data_val:size())
----------------------------------------------------------------
local function weights_init(m)
  local name = torch.type(m)
  if name:find('Convolution') and name:find('Spatial') then
    local nin = m.nInputPlane*m.kH*m.kW
    m.weight:uniform(-0.08, 0.08):mul(math.sqrt(1/nin))
    m.bias:fill(0)
  elseif name:find('Convolution') and name:find('Volumetric') then
    local nin = m.nInputPlane*m.kT*m.kH*m.kW
    m.weight:uniform(-0.08, 0.08):mul(math.sqrt(1/nin))
    m.bias:fill(0)
  elseif name:find('Linear') then
    local nin = m.weight:size(2)
    m.weight:uniform(-0.08, 0.08):mul(math.sqrt(1/nin))
    m.bias:fill(0)
  elseif name:find('BatchNormalization') then
    if m.weight then m.weight:normal(1.0, 0.02) end
    if m.bias then m.bias:fill(0) end
  end
end

opt.model_name = string.format('%s_%s_nv%d_adam%d_bs%d_nz%d_wd%g_lbg%g_ks%d',
  opt.arch_name, opt.exp_list, opt.nview, opt.adam, opt.batch_size, opt.nz,
  opt.weight_decay, opt.lambda, opt.kstep)

-- initialize parameters
init_models = dofile('scripts/' .. opt.arch_name .. '.lua')
encoder, actor, mixer, decoder_msk, decoder_im = init_models.create(opt)
encoder:apply(weights_init)
actor:apply(weights_init)
mixer:apply(weights_init)
decoder_msk:apply(weights_init)
decoder_im:apply(weights_init)

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
if prev_iter > 0 then
  encoder = loader.encoder
  actor = loader.actor
  mixer = loader.mixer
  decoder_msk = loader.decoder_msk
  decoder_im = loader.decoder_im
end

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
  end
  return config
end

config = getAdamParams(opt)
print(config)
-------------------------------------------------
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
------------------------------------------------
if opt.gpu > 0 then
  batch_im_in = batch_im_in:cuda()
  batch_rot = batch_rot:cuda()
  for k = 1, opt.kstep do
    batch_outputs[2*k-1] = batch_outputs[2*k-1]:cuda()
    batch_outputs[2*k] = batch_outputs[2*k]:cuda()
  end
  encoder:cuda()
  actor:cuda()
  mixer:cuda()
  decoder_msk:cuda()
  decoder_im:cuda()
  criterion_im:cuda()
  criterion_msk:cuda()
end

local inputs = {nn.Identity()(), nn.Identity()()}
local h_enc_id, h_enc_rot = encoder(inputs[1]):split(2)
local outputs = {}
local h_dec_rot = actor({h_enc_rot, inputs[2]})
local h_mix = mixer({h_enc_id, h_dec_rot})
local h_dec_msk = decoder_msk(h_mix)
local h_dec_im = decoder_im(h_mix)
table.insert(outputs, h_dec_im)
table.insert(outputs, h_dec_msk)

rotatorRNN = nn.gModule(inputs, outputs)
params, grads = rotatorRNN:getParameters()

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

  local f = rotatorRNN:forward({batch_im_in, batch_rot})
  errIM = 0
  errMSK = 0
  local df_dw = {}

  for k = 1, opt.kstep do 
    -- fast forward (actor, mixer, decoder)
    errIM = errIM + criterion_im:forward(f[2*k-1], batch_outputs[2*k-1]) / (8 * opt.batch_size)
    errMSK = errMSK + criterion_msk:forward(f[2*k], batch_outputs[2*k]) / (2 * opt.batch_size)
    local df_dIM = criterion_im:backward(f[2*k-1], batch_outputs[2*k-1]):mul(opt.lambda):div(8 * opt.batch_size)
    local df_dMSK = criterion_msk:backward(f[2*k], batch_outputs[2*k]):div(2 * opt.batch_size)
    df_dw[2*k-1] = df_dIM:clone()
    df_dw[2*k] = df_dMSK:clone()
  end
  rotatorRNN:backward({batch_im_in, batch_rot}, df_dw)

  local err = errIM * opt.lambda + errMSK
  return err, grads
end
-------------------------------------------------
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

  local f = rotatorRNN:forward({batch_im_in, batch_rot})
  errIM = 0
  errMSK = 0
  
  for k = 1, opt.kstep do
    errIM = errIM + criterion_im:forward(f[2*k-1], batch_outputs[2*k-1]) / (8 * opt.batch_size)
    errMSK = errMSK + criterion_msk:forward(f[2*k], batch_outputs[2*k]) / (2 * opt.batch_size)
    preds[2*k-1] = f[2*k-1]:float():clone()
    preds[2*k] = f[2*k]:float():clone()
  end

  local err = errIM * opt.lambda + errMSK
  return err
end
--------------------------------------------------

-- train & val
for epoch = prev_iter + 1, opt.niter do
  epoch_tm:reset()
  local counter = 0
  -- train
  rotatorRNN:training()
  for i = 1, math.min(data:size() * opt.nview / 2 , opt.ntrain), opt.batch_size do
    tm:reset()
    optim_utils.adam_v2(opfunc, params, config, state)
    counter = counter + 1
   
    print(string.format('Epoch: [%d][%8d / %8d]\t Time: %.3f DataTime: %.3f  '
      .. ' Err_Im: %.4f , Err_Msk: %.4f', epoch, ((i-1) / opt.batch_size),
      math.floor(math.min(data:size() * opt.nview / 2, opt.ntrain) / opt.batch_size),
      tm:time().real, data_tm:time().real,
      errIM and errIM or -1, errMSK and errMSK or -1))
  end

  -- val
  rotatorRNN:evaluate()
  for i = 1, opt.batch_size do
    tm:reset()
    local err = feedforward(params)
  end
  
  -- plot
  local to_plot = {}
  for i = 1, 32 do
    for k = 1, opt.kstep do
      local res = batch_im_in[i]:float():clone()
      res = torch.squeeze(res)
      res:add(1):mul(0.5)
      to_plot[#to_plot+1] = res:clone()

      local res = preds[2*k][i]:float()
      res = torch.squeeze(res)
      res = res:repeatTensor(3, 1, 1)
      res:mul(-1):add(1)
      to_plot[#to_plot+1] = res:clone()

      local res = preds[2*k-1][i]:float()
      res = torch.squeeze(res)
      res:add(1):mul(0.5)
      to_plot[#to_plot+1] = res:clone()

      local res = batch_outputs[2*k-1][i]:float():clone()
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
