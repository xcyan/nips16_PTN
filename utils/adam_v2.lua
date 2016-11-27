local optim2 = {}

function optim2.adam_v2(opfunc, x, config, state)
  local config = config or {}
  local state = state or config
  local lr = config.learningRate or 0.001
  local wd = config.weightDecay or 0.004

  local beta1 = config.beta1 or 0.1
  local beta2 = config.beta2 or 0.001
  local epsilon = config.epsilon or 1e-8

  local fx, dfdx = opfunc(x)

  if wd ~= 0 then
    dfdx:add(wd, x)
  end

  state.t = state.t or 0
  state.m = state.m or x.new(dfdx:size()):zero()
  state.v = state.v or x.new(dfdx:size()):zero()

  state.denom = state.denom or x.new(dfdx:size()):zero()

  state.t = state.t + 1

  --print(dfdx:size())
  --print(state.m:size())
  state.m:mul(beta1):add(1-beta1, dfdx)
  state.v:mul(beta2):addcmul(1-beta2, dfdx, dfdx)

  state.denom:copy(state.v):sqrt():add(epsilon)

  if state.t < 10000 then
    local biasCorrection1 = 1 - beta1^state.t
    local biasCorrection2 = 1 - beta2^state.t
    lr = lr * math.sqrt(biasCorrection2)/biasCorrection1
  end

  --print('lr = %g', lr)
  x:addcdiv(-lr, state.m, state.denom)

  return x, {fx}
end

return optim2
