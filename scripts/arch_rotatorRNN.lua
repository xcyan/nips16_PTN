local rotatorRNN = {}

function rotatorRNN.create(opt)
  local encoder = rotatorRNN.create_encoder(opt)
  local actor = rotatorRNN.create_actor(opt)
  local mixer = rotatorRNN.create_mixer(opt)
  local decoder_msk = rotatorRNN.create_decoder_msk(opt)
  local decoder_im = rotatorRNN.create_decoder_im(opt)
  return encoder, actor, mixer, decoder_msk, decoder_im
end

function rotatorRNN.create_encoder(opt)
  local encoder = nn.Sequential()
  -- 64 x 64 x 3 --> 32 x 32 x 64
  encoder:add(nn.SpatialConvolution(3, 64, 5, 5, 2, 2, 2, 2))
  encoder:add(nn.ReLU())

  -- 32 x 32 x 64 --> 16 x 16 x 128
  encoder:add(nn.SpatialConvolution(64, 128, 5, 5, 2, 2, 2, 2))
  encoder:add(nn.ReLU())
  
  -- 16 x 16 x 128 --> 8 x 8 x 256
  encoder:add(nn.SpatialConvolution(128, 256, 5, 5, 2, 2, 2, 2))
  encoder:add(nn.ReLU())
  
  -- 8 x 8 x 256 --> 1024
  encoder:add(nn.Reshape(8*8*256))
  encoder:add(nn.Linear(8*8*256, 1024))
  encoder:add(nn.ReLU())

  -- 1024 --> 1024
  encoder:add(nn.Linear(1024, 1024))
  encoder:add(nn.ReLU())

  -- identity unit
  local eid = nn.Sequential()
  eid:add(nn.Linear(1024, opt.nz))
  eid:add(nn.ReLU())

  -- viewpoint unit
  local erot = nn.Sequential()
  erot:add(nn.Linear(1024, opt.nz))
  erot:add(nn.ReLU())

  encoder:add(nn.ConcatTable():add(eid):add(erot))
  return encoder
end

function rotatorRNN.create_actor(opt)
  -- h1, a --> h2
  local actor = nn.Sequential()
  actor:add(nn.Bilinear(opt.nz, opt.na, opt.nz))
  actor:add(nn.ReLU())
  return actor
end

function rotatorRNN.create_mixer(opt)
  
  local mixer = nn.Sequential()
  mixer:add(nn.JoinTable(2))
  
  mixer:add(nn.Linear(opt.nz*2, 1024))
  mixer:add(nn.ReLU())
 
  mixer:add(nn.Linear(1024, 1024))
  mixer:add(nn.ReLU())
   return mixer
end

function rotatorRNN.create_decoder_msk(opt)
  local decoderM = nn.Sequential()
  -- 1024 --> 8 x 8 x 128
  decoderM:add(nn.Linear(1024, 8*8*128))
  decoderM:add(nn.ReLU())

  decoderM:add(nn.Reshape(128, 8, 8))
  -- 8 x 8 x 128 --> 16 x 16 x 64
  decoderM:add(nn.SpatialUpSamplingNearest(2))
  decoderM:add(nn.SpatialConvolution(128, 64, 5, 5, 1, 1, 2, 2))
  decoderM:add(nn.ReLU())
  
  -- 16 x 16 x 64 --> 32 x 32 x 32
  decoderM:add(nn.SpatialUpSamplingNearest(2))
  decoderM:add(nn.SpatialConvolution(64, 32, 5, 5, 1, 1, 2, 2))
  decoderM:add(nn.ReLU())
  
  -- 32 x 32 x 32 --> 64 x 64 x 1
  decoderM:add(nn.SpatialUpSamplingNearest(2))
  decoderM:add(nn.SpatialConvolution(32, 1, 5, 5, 1, 1, 2, 2))
  decoderM:add(nn.Sigmoid())

  return decoderM
end

function rotatorRNN.create_decoder_im(opt)
  local decoderI = nn.Sequential()
  -- 1024 --> 8 x 8 x 256
  decoderI:add(nn.Linear(1024, 8*8*256))
  decoderI:add(nn.ReLU())
  decoderI:add(nn.Reshape(256, 8, 8)) 

  -- 8 x 8 x 256 --> 16 x 16 x 128
  decoderI:add(nn.SpatialUpSamplingNearest(2))
  decoderI:add(nn.SpatialConvolution(256, 128, 5, 5, 1, 1, 2, 2))
  decoderI:add(nn.ReLU())

  -- 16 x 16 x 128 --> 32 x 32 x 64
  decoderI:add(nn.SpatialUpSamplingNearest(2))
  decoderI:add(nn.SpatialConvolution(128, 64, 5, 5, 1, 1, 2, 2))
  decoderI:add(nn.ReLU())

  -- 32 x 32 x 64 --> 64 x 64 x 3
  decoderI:add(nn.SpatialUpSamplingNearest(2))
  decoderI:add(nn.SpatialConvolution(64, 3, 5, 5, 1, 1, 2, 2))
  decoderI:add(nn.Tanh())

  return decoderI
end

return rotatorRNN
