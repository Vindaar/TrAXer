import ggplotnim
import std / [strscans, strutils]

proc plotData[T](data: ptr UncheckedArray[T], fname, outfile: string,
                 transparent: bool) =
  const wStr = "_width_"
  const hStr = "_height_"
  let idx = fname.find(wStr)
  let fname = fname[idx .. ^1].replace("_", " ") # fuck this
  let (success, width, height) = scanTuple(fname, " width $i height $i.dat")
  if success:
    let t = fromBuffer[T](data, [width, height])
    var xs = newSeqOfCap[int](width*height)
    var ys = newSeqOfCap[int](width*height)
    var zs = newSeqOfCap[float](width*height)
    for x in 0 ..< width:
      for y in 0 ..< height:
        xs.add x
        ys.add y
        zs.add t[x, y].float
    let df = toDf({"x" : xs, "y" : ys, "z" : zs})
    var customInferno = inferno()
    customInferno.colors[0] = 0
    ggplot(df, aes("x", "y", fill = "z")) +
      geom_raster() +
      ggtitle(fname) +
      scale_fill_gradient(customInferno) +
      ggsave(outfile)

proc main(fname, dtype, outfile: string,
          transparent = false) =
  let data = readFile(fname)
  case dtype
  of "uint32": plotData(cast[ptr UncheckedArray[uint32]](data[0].addr), fname, outfile, transparent)
  of "int": plotData(cast[ptr UncheckedArray[int]](data[0].addr), fname, outfile, transparent)
  of "float": plotData(cast[ptr UncheckedArray[float]](data[0].addr), fname, outfile, transparent)
  else:
    doAssert false, "Data type " & $dtype & " not supported yet."

when isMainModule:
  import cligen
  dispatch main
