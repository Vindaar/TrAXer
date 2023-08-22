import datamancer, sequtils, seqmath, algorithm

proc getFluxRadiusCDF*(solarModelFile: string): seq[float] =
  var emRatesDf = readCsv(solarModelFile)
  # get all radii and energies from DF so that we don't need to compute them manually (risking to
  # messing something up!)
  # sort both just to make sure they really *are* in ascending order
  let radii = emRatesDf["Radius"]
    .unique()
    .toTensor(float)
    .toSeq1D
    .sorted(SortOrder.Ascending)
  let energies = emRatesDf["Energy [keV]"]
    .unique()
    .toTensor(float)
    .toSeq1D
    .sorted(SortOrder.Ascending)
  var emRates = newSeq[seq[float]]()
  ## group the "solar model" DF by the radius & append the emission rates for all energies
  ## to the `emRates`
  for tup, subDf in groups(emRatesDf.group_by("Radius")):
    let radius = tup[0][1].toFloat
    #doAssert subDf["Energy [keV]", float].toSeq1D.mapIt(it.keV) == energies
    #doAssert radius == radii[k], "Input DF not sorted correctly!"
    emRates.add subDf["emRates", float].toSeq1D
  var
    fluxRadiusCumSum: seq[float] = newSeq[float](radii.len)
    diffRadiusSum = 0.0

  template toCdf(x: untyped): untyped =
    let integral = x[^1]
    x.mapIt( it / integral )

  var fluxesDf = newDataFrame()
  for iRad, radius in radii:
    # emRates is seq of radii of energies
    let emRate = emRates[iRad]
    var diffSum = 0.0
    for iEnergy, energy in energies:
      let dFlux = emRate[iEnergy] * (energy.float*energy.float) * radius*radius
      diffSum += dFlux
    diffRadiusSum += diffSum
    fluxRadiusCumSum[iRad] = diffRadiusSum

  result = fluxRadiusCumSum.toCdf()
