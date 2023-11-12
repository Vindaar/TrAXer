import ggplotnim, ggplotnim/ggplot_sdl2
import std / [strscans, strutils, math, strformat]

proc hpdFromTensor(t: Tensor[float]) =
  echo "HPD in x: ", t.sum(axis = 0)
  echo "HPD in y: ", t.sum(axis = 1)

proc weightedMean*(t, w: Tensor[float]): float =
  result = 0.0
  var sumW = 0.0
  for i in 0 ..< t.size:
    result += t[i] * w[i]
    sumW += w[i]
  result /= sumW

proc parseDataImpl*[T](data: ptr UncheckedArray[T], dx, dy: float, width, height: int, invertY, switchAxes: bool): (DataFrame, float, float) =
  let t = fromBuffer[T](data, [width, height])
  #when T is float:
  #  hpdFromTensor(t)
  var xs = newSeqOfCap[int](width*height)
  var ys = newSeqOfCap[int](width*height)
  var zs = newSeqOfCap[float](width*height)
  for x in 0 ..< width:
    for y in 0 ..< height:
      xs.add x
      if invertY:
        ys.add height - y
      else:
        ys.add y
      zs.add t[x, y].float

  # choose the correct column based on switching axes
  var xCol = if switchAxes: "y" else: "x"
  var yCol = if switchAxes: "x" else: "y"

  var df = toDf({xCol : xs, yCol : ys, "z" : zs})
  # compute x, y in mm
  ## XXX: fix the mean subtraction! I don't actually center like this because the data
  ## is binned of course. Need to compute weighted mean!
  df = df.mutate(f{float: "x [mm]" ~ `x` / width.float * dx},
                 f{float: "x [mm]" ~ idx("x [mm]") - weightedMean(col("x [mm]"), col("z"))},
                 f{float: "y [mm]" ~ `y` / height.float * dy},
                 f{float: "y [mm]" ~ idx("y [mm]") - weightedMean(col("y [mm]"), col("z"))})
  result = (df, dx, dy)

proc parseData*[T](data: ptr UncheckedArray[T], fname: string, invertY, switchAxes: bool): (DataFrame, float, float) =
  const wStr = "_width_"
  const hStr = "_height_"
  let idx = fname.find("__")
  let fname = fname[idx .. ^1].replace("_", " ").strip # fuck this
  #let (success, width, height) = scanTuple(fname, " width $i height $i.dat")
  let (success, dx, dy, dz, dtype, len, width, height) = scanTuple(fname, "dx $f dy $f dz $f type $w len $i width $i height $i.dat")
  if success:
    result = parseDataImpl(dx, dy, width, height)
  else:
    raise newException(IOError, "Could not parse filename " & $fname)
