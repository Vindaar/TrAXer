import std / strscans

import datamancer

import ./parse_binary

proc calcEEF*(df: DataFrame): DataFrame =
  ## Calc radius, sort, calc arcsecond diameter, EEF -> normalized cumulative sum
  result = df.mutate(f{float: "r" ~ sqrt(idx("x [mm]")*idx("x [mm]") + idx("y [mm]")*idx("y [mm]"))})
    .arrange("r", SortOrder.Ascending)
    .mutate(f{"Diameter [ArcSecond]" ~ arctan(`r` / 1500.0).radToDeg * 3600.0 * 2.0})
  result["EEF"] = result["z", float].cumsum / result["z", float].sum

proc calcVal*(df: DataFrame, val: float, verbose: bool): (float, float) =
  let dfF = df.filter(f{float: `EEF` >= val}).arrange("EEF", SortOrder.Ascending)
  let valArc = dfF["Diameter [ArcSecond]", 0, float]
  let valMm  = dfF["r", 0, float] * 2.0
  result = (valMm, valArc)
  if verbose:
    echo "Val at: ", val, " = ", valArc
    echo "Sum below ", valMm / 2.0, " : ", df.filter(f{`r` <= valMm / 2.0})["z", float].sum
    echo "Sum above ", valMm / 2.0, " : ", df.filter(f{`r` >  valMm / 2.0})["z", float].sum
    echo dfF

proc hpdViaEEF*(df: DataFrame) =
  ## Computes the HPD at 50% based on the Encircled Energy Function (EEF),
  ## that is radial cumulative distribution function at 50%.
  # compute the radius for each row and sort in descending order
  let (c90Mm, c90Arc) = df.calcVal(0.9, verbose = false)
  let (c80Mm, c80Arc) = df.calcVal(0.8, verbose = false)
  let (hpdMm, hpdArc) = df.calcVal(0.5, verbose = false)

  echo hpdArc
  echo c80Arc
  echo c90Arc

proc main(fname: string, dx = 14.0, dy = 14.0, width = 1000, height = 1000) =
  let data = readFile(fname)
  let (df, dx, dy) = parseDataImpl(cast[ptr UncheckedArray[float]](data[0].addr), dx, dy, width, height, invertY = true, switchAxes = false)

  if df.filter(f{`z` > 0.0}).len == 0:
    echo Inf
    echo Inf
    echo Inf
  else:
    let dfEEF = calcEEF(df)
    hpdViaEEF(dfEEF)

when isMainModule:
  import cligen
  dispatch main
