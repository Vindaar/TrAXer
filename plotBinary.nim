import ggplotnim, ggplotnim/ggplot_sdl2
import std / [strscans, strutils, math, strformat, sequtils]
from std / os import getEnv

import ./parse_binary
import ./calc_eef

let UseTeX = getEnv("USE_TEX", "false").parseBool
let FWIDTH = getEnv("F_WIDTH", "0.9").parseFloat
let Width = getEnv("WIDTH", "600").parseFloat
let Height = getEnv("HEIGHT", "480").parseFloat

proc customThreeway(): Theme =
  result = sideBySide()
  result.titleFont = some(font(5.0))
  result.legendFont = some(font(5.0))
  result.legendTitleFont = some(font(5.0, bold = true))
  result.labelFont = some(font(5.0))
  result.tickLabelFont = some(font(5.0))
  result.tickLength = some(3.5)
  result.tickWidth = some(1.0 * 3.5 / 5.0)
  result.annotationFont = some(font(6.0, family = "monospace"))

proc customSideBySide(): Theme =
  result = sideBySide()
  result.titleFont = some(font(8.0))

proc thL(fWidth: float, width: float,
         baseTheme: (proc(): Theme) = nil,
         height = -1.0, ratio = -1.0,
         textWidth = 458.29268, # 455.24411
        ): Theme =
  if UseTeX:
    let baseTheme = if baseTheme != nil: baseTheme
                    elif fWidth < 0.5: customThreeway
                    elif fWidth == 0.5: customSideBySide
                    else: singlePlot
    result = themeLatex(fWidth, width, baseTheme, height, ratio, textWidth,
                        useTeX = UseTeX, useWithoutTeX = UseTeX)
  else:
    result = Theme()

proc plotData(df: DataFrame, dx, dy: float, fname, outfile: string,
              transparent: bool,
              title: string, inPixels: bool,
              lowQ, highQ: int,
              xrange: float) =
  # add the possible `mm` suffix
  var
    xCol = "x"
    yCol = "y"
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
  var df = df
  if xrange > 0.0 and yrange > 0.0:
    df = df.filter(f{idx(xCol) >= -xrange and idx(xCol) <= xrange and idx(yCol) >= -yrange and idx(yCol) <= yrange})
  ggplot(df, aes(xCol, yCol, fill = "z")) +
    geom_raster() +
    ggtitle(title) +
    margin(top = 1.5) +
    xlim(-xrange, xrange) + ylim(-xrange, xrange) +
    scale_fill_gradient(customInferno) +
    thL(fWidth = FWIDTH, width = Width) +
    coord_fixed(1.0) +
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
    thL(fWidth = FWIDTH, width = Width) +
    coord_fixed(1.0) +
    ggsave(outfile.replace(".pdf", "_log10.pdf"))

proc plotHPDViaEEF(df: DataFrame, title, outfile: string, verbose: bool) =
  ## Computes the HPD at 50% based on the Encircled Energy Function (EEF),
  ## that is radial cumulative distribution function at 50%.
  # compute the radius for each row and sort in descending order
  let (c90Mm, c90Arc) = df.calcVal(0.9, verbose = verbose)
  let (c80Mm, c80Arc) = df.calcVal(0.8, verbose = verbose)
  let (hpdMm, hpdArc) = df.calcVal(0.5, verbose = verbose)

  proc keepEvery(df: DataFrame, num: int): DataFrame =
    result = df.shallowCopy()
    result["Idx"] = toSeq(0 ..< df.len)
    result = result.filter(f{int -> bool: `Idx` mod num == 0})

  #ggplot(dfX, aes("Diameter [ArcSecond]", "EEF")) +
  #  geom_line() +
  #  geom_linerange(aes = aes(x = hpdVal, yMin = 0.0, yMax = 0.75), color = "red") +
  #  annotate(&"HPD = {hpdMm:.4f} mm, {hpdVal:.4f} ''", left = 0.05, bottom = 0.075) +
  #  ggsave(outfile.replace(".pdf", "_hpd_via_eef_50.pdf"))
  proc toArc(x: float): float = arctan(x / 1500.0).radToDeg * 3600.0 * 2.0
  proc fromArc(x: float): float = tan(x.degToRad / (2.0 * 3600.0)) * 1500.0

  let texts = [&"HPD = {hpdMm:.4f} mm, {hpdArc:.4f} ''",
               &"80% = {c80Mm:.4f} mm, {c80Arc:.4f} ''",
               &"90% = {c90Mm:.4f} mm, {c90Arc:.4f} ''"].join("\n")
  let dfF = df.filter(f{`EEF` <= 0.999}).keepEvery(100)
  echo dfF
  ggplot(dfF, aes("r", "EEF")) +
    geom_line() +
    geom_linerange(aes = aes(x = hpdMm / 2.0, yMin = 0.0, yMax = 0.75), color = "red") +
    annotate(texts, left = 0.60, bottom = 0.2) +
    xlab("Radius [mm]") + ylab("Encircled Energy Function (EEF)") +
    ggtitle(title) +
    #ggshow(outfile.replace(".pdf", "_hpd_via_eef_50.pdf"))
    thL(fWidth = FWIDTH, width = Width) +
    ggsave(outfile.replace(".pdf", "_hpd_via_eef_50.pdf"))

  ggplot(dfF, aes("Diameter [ArcSecond]", "EEF")) +
    geom_line() +
    geom_linerange(aes = aes(x = hpdArc, yMin = 0.0, yMax = 0.75), color = "red") +
    annotate(texts, left = 0.60, bottom = 0.2) +
    ylab("Encircled Energy Function (EEF)") +
    ggtitle(title) +
    #ggshow(outfile.replace(".pdf", "_hpd_via_eef_50.pdf"))
    thL(fWidth = FWIDTH, width = Width) +
    ggsave(outfile.replace(".pdf", "_diameter_arcsec_hpd_via_eef_50.pdf"))

  #block Complicated:
  #  var dfP = df.mutate(f{float: "r" ~ sqrt(idx("x [mm]")*idx("x [mm]") + idx("y [mm]")*idx("y [mm]")).round(2)})
  #    .arrange("r", SortOrder.Descending)
  #  #for (tup, subDf) in groups(dfP.group_by("r")):
  #  #  echo tup, " and ", subDf
  #  let dfR = dfP.group_by("r").summarize(f{float: "sumZ" << sum(col("z"))})
  #    .arrange("r", SortOrder.Descending)
  #  #echo dfR.pretty(-1)
  #  if verbose:
  #    echo dfR
  #
  #  var cumZ = newSeq[float](dfR.len)
  #  if verbose:
  #    echo "Computing EEF"
  #  let zSum = sum(dfR["sumZ", float])
  #  var sum = zSum
  #  for idx in 0 ..< dfR.len:
  #    cumZ[idx] = sum
  #    sum -= dfR["sumZ", idx, float]
  #  var dfX = dfR
  #  dfX["EEF"] = cumZ.toTensor.map_inline(x / zSum)
  #  dfX = dfX.filter(f{`EEF` < 1.0})
  #    .mutate(f{"Diameter [ArcSecond]" ~ arctan(`r` / 1500.0).radToDeg * 3600.0 * 2.0})

  #var df = df.mutate(f{"r" ~ sqrt(idx("x [mm]")*idx("x [mm]") + idx("y [mm]")*idx("y [mm]"))})
  #  .arrange("r", SortOrder.Descending)
  #  # .mutate(f{float: "EEF" ~ sum(col("z")[idx ..< df.len])})
  #var cumZ = newSeq[float](df.len)
  #echo "Computing EEF"
  #let zSum = sum(df["z", float])
  #
  #
  #var sum = zSum
  #for idx in 0 ..< df.len:
  #  cumZ[idx] = sum
  #  sum -= df["z", idx, float]
  #df["EEF"] = cumZ.toTensor.map_inline(x / zSum)
  #
  #let hpdDf = df.filter(f{`EEF` >= 0.5}).arrange("EEF", SortOrder.Ascending)
  #echo hpdDf
  #echo "HPD : ", arctan(hpdDf["r", 0, float] / 1500.0).radToDeg * 3600.0
  #
  #df = df.filter(f{`EEF` < 1.0})
  #  # compute *diameter* in arc seconds
  #  .mutate(f{"d [ArcSecond]" ~ arctan(`r` / 1500.0).radToDeg * 3600.0 * 2.0})
  #ggplot(df, aes("d [ArcSecond]", "EEF")) +
  #  geom_line() +
  #  ggsave("/tmp/hpd_eef.pdf")

  #df.showBrowser()

proc plotHPD(df: DataFrame, xrange: float, title, outfile: string, verbose: bool) =
  # Finally also compute the HPD via the EEF (i.e. radial cumulative distribution function)
  # This _should_ be the correct way to compute it I believe.
  let dfEEF = calcEEF(df)
  plotHPDViaEEF(dfEEF, title, outfile, verbose)

  proc calcHPD(df: DataFrame, key: string): (float, float, float) =
    var dfF = df.arrange(key, SortOrder.Ascending)
    dfF = df.filter(f{float: idx("sum(z)") >= col("sum(z)").max / 2.0})
    ## Calculate HPD as difference in mm for now
    let low = dfF[key, float][0]
    let hig = dfF[key, float][dfF.high]
    #echo dfF.pretty(-1)
    result = (abs(hig - low), low, hig)

  proc hpdAsAngle(val: float, fL: float): float =
    result = arctan(val / fL).radToDeg * 3600.0 # convert to degrees, then to arc second

  #df.showBrowser()
  let xSum = df.group_by("y [mm]").summarize(f{float: "sum(z)" << sum(`z`)})
  let ySum = df.group_by("x [mm]").summarize(f{float: "sum(z)" << sum(`z`)})

  #ySum.showBrowser()

  let (hpd_x, hpd_x_low, hpd_x_high) = calcHPD(ySum, "x [mm]")
  let hpd_angle_x = hpdAsAngle(hpd_x, 1530.0) # use 1530?
  let (hpd_y, hpd_y_low, hpd_y_high) = calcHPD(xSum, "y [mm]")
  let hpd_angle_y = hpdAsAngle(hpd_y, 1530.0) # use 1530?

  #echo "HPD along x: ", lowerBound(xSum["y", float].toSeq1D, xSum["sum(x)", float].max / 2.0)
  if verbose:
    echo xSum
    echo ySum
  echo "HPD along x: ", hpd_x, " as angle: ", hpd_angle_x, " ''"
  echo "HPD along y: ", hpd_y, " as angle: ", hpd_angle_y, " ''"
  let hpd_y_max_val = xSum["sum(z)", float].max
  let hpd_x_max_val = ySum["sum(z)", float].max

  ## Now for the plots, cut to the desired range
  ggplot(xSum, aes("y [mm]", "sum(z)")) +
    geom_line() +
    geom_line(aes = aes(x = hpd_y_low, yMin = 0.0, yMax = hpd_y_max_val), color = "red") +
    geom_line(aes = aes(x = hpd_y_high, yMin = 0.0, yMax = hpd_y_max_val), color = "red") +
    annotate(&"HPD = {hpd_y:.4f} mm, {hpd_angle_y:.4f} ''", left = 0.05, bottom = 0.075) +
    ggtitle(&"{title}, HPD plot for axis y") +
    minorGridLines() +
    xlim(-xrange, xrange) +
    thL(fWidth = FWIDTH, width = Width) +
    ggsave(outfile.replace(".pdf", "_hpd_y.pdf"))
  ggplot(ySum, aes("x [mm]", "sum(z)")) +
    geom_line() +
    geom_line(aes = aes(x = hpd_x_low, yMin = 0.0, yMax = hpd_x_max_val), color = "red") +
    geom_line(aes = aes(x = hpd_x_high, yMin = 0.0, yMax = hpd_x_max_val), color = "red") +
    annotate(&"HPD = {hpd_x:.4f} mm, {hpd_angle_x:.4f} ''", left = 0.05, bottom = 0.075) +
    ggtitle(&"{title}, HPD plot for axis x") +
    minorGridLines() +
    xlim(-xrange, xrange) +
    thL(fWidth = FWIDTH, width = Width) +
    ggsave(outfile.replace(".pdf", "_hpd_x.pdf"))

proc main(fname, dtype, outfile: string,
          transparent = false,
          invertY = false,
          switchAxes = false,
          title = "",
          inPixels = true,
          lowQ = 1, highQ = 99,
          xrange = 0.0,
          verbose = false,
          batchMode = false,
          gridpixOutfile = "" # if given will write a CSV of the data of 256x256 pixels
         ) =
  let data = readFile(fname)
  template call(typ: untyped): untyped =
    let (df, dx, dy) = parseData(cast[ptr UncheckedArray[typ]](data[0].addr), fname, invertY, switchAxes)
    if not batchMode:
      plotData(df, dx, dy, fname, outfile, transparent, title, inPixels, lowQ, highQ, xrange)

    plotHPD(df, xrange, title, outfile, verbose)

    if gridpixOutfile.len > 0:
      writeGridPixCsv(df, gridpixOutfile)
  case dtype
  of "uint32": call(uint32)
  of "int": call(int)
  of "float": call(float)
  else:
    doAssert false, "Data type " & $dtype & " not supported yet."

when isMainModule:
  import cligen
  dispatch main
