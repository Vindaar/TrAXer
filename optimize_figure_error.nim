import std / [strformat, strutils, sequtils, math]
import shell

import nlopt

type
  FitObject = object
    hpd: float
    c80: float
    c90: float
    bufOutdir: string
    shmFile: string

const cmd = """
$# ./raytracer --width 1200 --speed 10.0 --nJobs 32 --vfov 10 --maxDepth 10 --llnl --focalPoint --sourceKind skXrayFinger --rayAt 1.013 --sensorKind sSum --energyMin 1.48 --energyMax 1.50 --usePerfectMirror=false --ignoreWindow --sourceDistance 130.297.m --sourceRadius 0.42.mm --telescopeRotation 90.0 --sourceOnOpticalAxis --ignoreMagnet --targetRadius 40.mm --batchMode --totalRays 1_000_000 --bufOutdir $# --shmOutfile $#
"""

proc optimizeFigureError(p: seq[float], data: FitObject): float =
  # 1. run the raytracer with current parameters
  echo "Params: ", p
  let envs = ["FUZZ_IN", "FUZZ_ORTH", "FUZZ_IN_SCALE", "FUZZ_OUTER_SCALE", "FUZZ_IN_RATIO"]
  var envStr = ""
  for i, param in p:
    let pit = if param > 100: 1.0 else: param
    envStr.add &"{envs[i]}={pit} "
  let cmdT = cmd % [envStr, data.bufOutdir, data.shmFile]
  echo "Command: ", cmdT
  shell: # run raytracer
    ($cmdT)
  # 2. parse binary data
  let shm = data.shmFile
  let (res, err) = shellVerbose:
    ./calc_eef -f ($shm)

  let resSpl = res.strip.splitLines.mapIt(it.parseFloat)
  doAssert resSpl.len == 3

  # calculate mean squared error
  let hpdR = resSpl[0]
  let c80R = resSpl[1]
  let c90R = resSpl[2]

  # penalize HPD 3x more than the other two
  result = (data.hpd - hpdR)^2 / sqrt(data.hpd) * 3.0 + (data.c80 - c80R)^2 / sqrt(data.c80) + (data.c90 - c90R)^2 / sqrt(data.c90)
  echo "Current difference: ", result, " from : ", resSpl

proc runOptimization(params: seq[float], hpd, c80, c90: float,
                     bufOutdir, shmFile: string) =
  let fitObj = FitObject(hpd: hpd, c80: c80, c90: c90,
                         bufOutdir: bufOutdir,
                         shmFile: shmFile)
  doAssert params.len == 5
  var opt = newNloptOpt[FitObject](LN_COBYLA, #GN_DIRECT_L,
                                   params.len,
                                   @[(l: 0.5, u: 6.0), (l: 0.2, u: 2.0), (l: 0.2, u: 2.0), (l: 1.0, u: 10.0), (l: 0.5, u: 4.0)])
  #var opt = newNloptOpt[FitObject](GN_DIRECT, params.len, @[(l: 0.5, u: 10.0), (l: 0.2, u: 2.0), (l: 0.2, u: 2.0), (l: 1.0, u: 10.0), (l: 0.5, u: 4.0)])
  let varStruct = newVarStruct(optimizeFigureError, fitObj)
  opt.setFunction(varStruct)
  #var constrainVarStruct = newVarStruct(constrainCL95, fitObj)
  #opt.addInequalityConstraint(constrainVarStruct)
  #opt.maxtime = 30.0
  #opt.initialStep = 1e-10
  echo "Starting params: ", params
  let optRes = opt.optimize(params)
  echo opt.status

  echo optRes
  destroy(opt)
  #result = optRes

proc main(hpd, c80, c90: float,
          bufOutdir = "out",
          shmFile = "/dev/shm/image_sensor.dat"
         ) =
  ## Tries to find closest parameters that match `hpd`, `c80`, `c90`

  # define starting parameters
  #let params = @[ 3.5,  # FUZZ_IN
  #                0.5,  # FUZZ_ORTH
  #                0.5,  # FUZZ_IN_SCALE
  #                8.0,  # FUZZ_OUTER_SCALE
  #                1.0 ] # FUZZ_IN_RATIO
  let params = @[ 3.25,  # FUZZ_IN
                  0.233333333,  # FUZZ_ORTH
                  0.996296,  # FUZZ_IN_SCALE
                  9.2407,  # FUZZ_OUTER_SCALE
                  1.0833333333 ] # FUZZ_IN_RATIO
  runOptimization(params, hpd, c80, c90, bufOutdir, shmFile)

when isMainModule:
  import cligen
  dispatch main
