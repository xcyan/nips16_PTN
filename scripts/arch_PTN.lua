local PTN = {}

function PTN.create(opt)
  local encoder = PTN.create_encoder(opt)
  local voxel_dec = PTN.create_voxel_dec(opt)
  local projector = PTN.create_projector(opt)
  return encoder, voxel_dec, projector
end

function PTN.create_encoder(opt)
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

function PTN.create_voxel_dec(opt)
  local voxel_dec = nn.Sequential()
  
  voxel_dec:add(nn.Linear(opt.nz, 3*3*3*512))
  voxel_dec:add(nn.ReLU())
  voxel_dec:add(nn.Reshape(512, 3, 3, 3))

  -- 512 x 3 x 3 x 3 --> 256 x 6 x 6 x 6
  voxel_dec:add(nn.VolumetricFullConvolution(512, 256, 4, 4, 4, 1, 1, 1, 0, 0, 0))
  voxel_dec:add(nn.ReLU())
  -- 256 x 6 x 6 x 6 --> 96 x 15 x 15 x 15
  voxel_dec:add(nn.VolumetricFullConvolution(256, 96, 5, 5, 5, 2, 2, 2, 0, 0, 0))
  voxel_dec:add(nn.ReLU())
  -- 96 x 15 x 15 x 15 --> 1 x 32 x 32 x 32
  voxel_dec:add(nn.VolumetricFullConvolution(96, 1, 6, 6, 6, 2, 2, 2, 1, 1, 1))
  voxel_dec:add(nn.Sigmoid())

  return voxel_dec
end

function PTN.create_projector(opt)
  local grid_stream = nn.PerspectiveGridGenerator(opt.vox_size, opt.vox_size, opt.vox_size, opt.focal_length)
  local input_stream = nn.Transpose({2,4},{4,5})
  local projector = nn.Sequential()
  projector:add(nn.ParallelTable():add(input_stream):add(grid_stream))
  projector:add(nn.BilinearSamplerPerspective(opt.focal_length))
  projector:add(nn.Transpose({4,5}, {2,4}))
  -- B x c x Dim1 x Dim2 x Dim3
  projector:add(nn.Max(4)) 
  return projector
end

return PTN
