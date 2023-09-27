import ggplotnim
import std / [strscans, strutils, math, strformat]

proc plotData[T](data: ptr UncheckedArray[T], fname, outfile: string,
                 transparent, invertY, switchAxes: bool,
                 title: string, inPixels: bool,
                 lowQ, highQ: int,
                 xrange: float): DataFrame =
  const wStr = "_width_"
  const hStr = "_height_"
  let idx = fname.find("__")
  let fname = fname[idx .. ^1].replace("_", " ").strip # fuck this
  #let (success, width, height) = scanTuple(fname, " width $i height $i.dat")
  let (success, dx, dy, dz, dtype, len, width, height) = scanTuple(fname, "dx $f dy $f dz $f type $w len $i width $i height $i.dat")
  if success:
    let t = fromBuffer[T](data, [width, height])
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
    var df = toDf({"x" : xs, "y" : ys, "z" : zs})
    # compute x, y in mm
    df = df.mutate(f{float: "x [mm]" ~ `x` / width.float * dx},
                   f{float: "x [mm]" ~ idx("x [mm]") - mean(col("x [mm]"))},
                   f{float: "y [mm]" ~ `y` / height.float * dy},
                   f{float: "y [mm]" ~ idx("y [mm]") - mean(col("y [mm]"))})
    # choose the correct column based on switching axes
    var xCol = if switchAxes: "y" else: "x"
    var yCol = if switchAxes: "x" else: "y"
    # add the possible `mm` suffix
    if not inPixels:
      xCol.add " [mm]"
      yCol.add " [mm]"
    # set the title
    let title = if title.len > 0: title else: fname
    # set up custom color scale
    var customInferno = inferno()
    customInferno.colors[0] = 0
    # plot linear
    let xrange = if xrange > 0.0: xrange elif not inPixels: dx / 2.0 else: 0.0
    let yrange = if xrange > 0.0: xrange elif not inPixels: dy / 2.0 else: 0.0
    if xrange > 0.0 and yrange > 0.0:
      df = df.filter(f{idx(xCol) >= -xrange and idx(xCol) <= xrange and idx(yCol) >= -yrange and idx(yCol) <= yrange})
    ggplot(df, aes(xCol, yCol, fill = "z")) +
      geom_raster() +
      ggtitle(title) +
      margin(top = 1.5) +
      xlim(-xrange, xrange) + ylim(-xrange, xrange) +
      scale_fill_gradient(customInferno) +
      ggsave(outfile)
    # plot log10
    let dfNZ = df.filter(f{`z` > 0.0})
    let scale = (low: dfNZ["z", float].percentile(lowQ), high: dfNZ["z", float].percentile(highQ))
    ggplot(df, aes(xCol, yCol, fill = "z")) +
      geom_raster() +
      ggtitle(title) +
      margin(top = 1.5) +
      xlim(-xrange, xrange) + ylim(-xrange, xrange) +
      scale_fill_log10(scale = scale, colorScale = customInferno) +
      ggsave(outfile.replace(".pdf", "_log10.pdf"))

    result = df
  else:
    echo "Failed to parse filename: ", fname

proc plotHPD(df: DataFrame, title, outfile: string) =
  df.showBrowser()
  let xSum = df.group_by("y [mm]").summarize(f{float: "sum(z)" << sum(`z`)})
  let ySum = df.group_by("x [mm]").summarize(f{float: "sum(z)" << sum(`z`)})

  for (tup, subDf) in groups(df.group_by("x [mm]")):
    let dff = subDf.filter(f{float: idx("z") > 0.0})
    if dff.len > 0:
      echo tup, " sum = ", subDf["z",float].sum, " = ", dff

  ySum.showBrowser()
  proc calcHPD(df: DataFrame, key: string): (float, float, float) =
    var dfF = df.arrange(key, SortOrder.Ascending)
    dfF = df.filter(f{float: idx("sum(z)") >= col("sum(z)").max / 2.0})
    ## Calculate HPD as difference in mm for now
    let low = dfF[key, float][0]
    let hig = dfF[key, float][dfF.high]
    echo dfF.pretty(-1)
    result = (abs(hig - low), low, hig)
    echo result, " ======================================\n\n"

  proc hpdAsAngle(val: float, fL: float): float =
    result = arctan(val / fL).radToDeg * 3600.0 # convert to degrees, then to arc second
  let (hpd_x, hpd_x_low, hpd_x_high) = calcHPD(ySum, "x [mm]")
  let hpd_angle_x = hpdAsAngle(hpd_x, 1530.0) # use 1530?
  let (hpd_y, hpd_y_low, hpd_y_high) = calcHPD(xSum, "y [mm]")
  let hpd_angle_y = hpdAsAngle(hpd_y, 1530.0) # use 1530?

  #echo "HPD along x: ", lowerBound(xSum["y", float].toSeq1D, xSum["sum(x)", float].max / 2.0)
  echo xSum
  echo ySum
  echo "HPD along x: ", hpd_x, " as angle: ", hpd_angle_x, " ''"
  echo "HPD along y: ", hpd_y, " as angle: ", hpd_angle_y, " ''"
  let hpd_y_max_val = xSum["sum(z)", float].max
  let hpd_x_max_val = ySum["sum(z)", float].max

  ggplot(xSum, aes("y [mm]", "sum(z)")) +
    geom_line() +
    geom_line(aes = aes(x = hpd_y_low, yMin = 0.0, yMax = hpd_y_max_val), color = "red") +
    geom_line(aes = aes(x = hpd_y_high, yMin = 0.0, yMax = hpd_y_max_val), color = "red") +
    annotate(&"HPD = {hpd_y:.4f} mm, {hpd_angle_y:.4f} ''", left = 0.05, bottom = 0.075) +
    ggtitle(&"{title}, HPD plot for axis y") +
    #xlim(5, 9) +
    minorGridLines() +
    ggsave(outfile.replace(".pdf", "_hpd_y.pdf"))
  ggplot(ySum, aes("x [mm]", "sum(z)")) +
    geom_line() +
    geom_line(aes = aes(x = hpd_x_low, yMin = 0.0, yMax = hpd_x_max_val), color = "red") +
    geom_line(aes = aes(x = hpd_x_high, yMin = 0.0, yMax = hpd_x_max_val), color = "red") +
    annotate(&"HPD = {hpd_x:.4f} mm, {hpd_angle_x:.4f} ''", left = 0.05, bottom = 0.075) +
    ggtitle(&"{title}, HPD plot for axis x") +
    #xlim(5, 9) +
    minorGridLines() +
    ggsave(outfile.replace(".pdf", "_hpd_x.pdf"))

proc main(fname, dtype, outfile: string,
          transparent = false,
          invertY = false,
          switchAxes = false,
          title = "",
          inPixels = true,
          lowQ = 1, highQ = 99,
          xrange = 0.0) =
  let data = readFile(fname)
  template call(typ: untyped): untyped =
    let df = plotData(cast[ptr UncheckedArray[typ]](data[0].addr), fname, outfile, transparent, invertY, switchAxes, title, inPixels, lowQ, highQ, xrange)
    plotHPD(df, title, outfile)
  case dtype
  of "uint32": call(uint32)
  of "int": call(int)
  of "float": call(float)
  else:
    doAssert false, "Data type " & $dtype & " not supported yet."

when isMainModule:
  import cligen
  dispatch main
