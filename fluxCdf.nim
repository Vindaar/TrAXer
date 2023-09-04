import datamancer, sequtils, seqmath, algorithm

type
  FluxData* = object
    fRCdf*: seq[float]
    diffFluxR*: seq[seq[float]]
    radii*: seq[float]
    energyMin*: float
    energyMax*: float

const
  alpha = 1.0 / 137.0
  g_ae = 1e-13 # Redondo 2013: 0.511e-10
  gagamma = 1e-12 #the latter for DFSZ  #1e-9 #5e-10 #
  ganuclei = 1e-15 #1.475e-8 * m_a #KSVZ model #no units  #1e-7
  m_a = 0.0853 #eV
  m_e_keV = 510.998 #keV
  e_charge = sqrt(4.0 * PI * alpha)#1.0
  kB = 1.380649e-23
  r_sun = 696_342_000_000.0 # .km.to(mm).float # SOHO mission 2003 & 2006
  hbar = 6.582119514e-25 # in GeV * s
  keV2cm = 1.97327e-8 # cm per keV^-1
  amu = 1.6605e-24 #grams
  r_sunearth = 150_000_000_000_000.0

const factor = pow(r_sun * 0.1 / (keV2cm), 3.0) /
               (pow(0.1 * r_sunearth, 2.0) * (1.0e6 * hbar)) /
               (3.1709791983765E-8 * 1.0e-4) # for units of 1/(keV y m²)

import ggplotnim
proc getFluxRadiusCDF*(solarModelFile: string): FluxData =
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
  var diffFluxR = newSeq[seq[float]](radii.len)
  var rLast = 0.0
  for iRad, radius in radii:
    # emRates is seq of radii of energies
    let emRate = emRates[iRad]
    var diffSum = 0.0
    var diffFlux = newSeq[float](energies.len)
    for iEnergy, energy in energies:
      let dFlux = emRate[iEnergy] * (energy.float*energy.float) * radius*radius * (radius - rLast) * factor
      diffFlux[iEnergy] = dFlux
      diffSum += dFlux
    diffRadiusSum += diffSum
    fluxRadiusCumSum[iRad] = diffRadiusSum

    when false:
      ## This is a sanity check. The plots should look like the correct differential flux
      ##
      ## Combine via
      ## ``pdfunite `lc -n1 -p` /tmp/all_diff_flux_{suffix}.pdf``
      ## and investigate by hand.
      let df = toDf(energies, diffFlux)
      ggplot(df, aes("energies", "diffFlux")) +
        geom_line() +
        ggtitle("Radius: " & $radius & " at index: " & $iRad) +
        ggsave("/tmp/diffFlux/diff_flux_radius_" & $iRad & ".pdf")

    diffFluxR[iRad] = diffFlux
    rLast = radius

  result = FluxData(fRCdf: fluxRadiusCumSum.toCdf(),
                    diffFluxR: diffFluxR,
                    radii: radii,
                    energyMin: energies.min,
                    energyMax: energies.max)

import xrayAttenuation

template getLayers(): untyped =
  const ρC = 2.26.g•cm⁻³ # 2.267 should be correct. used number same as DarpanX however
  const ρPt = 21.41.g•cm⁻³ # 21.45 should be correct. used number same as DarpanX however
  const ρGlass = 2.65.g•cm⁻³

  let C = Carbon.init(ρC)
  let Pt = Platinum.init(ρPt)
  let glass = compound((Si, 1), (O, 2), ρ = ρGlass) # pure quartz glass

  let m1 = initDepthGradedMultilayer(
    substrate = glass,
    top = C, bottom = Pt,
    dMin = 11.5.nm, dMax = 22.5.nm,
    Γ = 0.45,
    c = 1.0,
    N = 2,
    σ = 0.45.nm
  )
  let m2 = initDepthGradedMultilayer(
    substrate = glass,
    top = C, bottom = Pt,
    dMin = 7.0.nm, dMax = 19.nm,
    Γ = 0.45,
    c = 1.0,
    N = 3,
    σ = 0.45.nm
  )
  let m3 = initDepthGradedMultilayer(
    substrate = glass,
    top = C, bottom = Pt,
    dMin = 5.5.nm, dMax = 16.nm,
    Γ = 0.4,
    c = 1.0,
    N = 4,
    σ = 0.45.nm
  )
  let m4 = initDepthGradedMultilayer(
    substrate = glass,
    top = C, bottom = Pt,
    dMin = 5.0.nm, dMax = 14.0.nm,
    Γ = 0.4,
    c = 1.0,
    N = 5,
    σ = 0.45.nm
  )
  @[m1, m2, m3, m4]

proc calculateReflectivity[T; B; S](recipe: DepthGradedMultilayer[T, B, S],
                                    angleMin, angleMax: Degree,
                                    energyMin, energyMax: keV,
                                    numAngle, numEnergy: int): seq[seq[float]] =
  let Energy = linspace(energyMin.float, energyMax.float, numEnergy) # scan in a range wider than we use in the raytracer
  let θs = linspace(angleMin.float, angleMax.float, numAngle) # [0.5] # the angle under which we check
  var refl = zeros[float]([θs.len, Energy.len])
  for i, θ in θs:
    for j, E in Energy:
      # 1. compute the refractive indices for each layer
      refl[i, j] = recipe.reflectivity(θ.°, E.keV, parallel = false)
  result = refl.toSeq2D

## Move this to another file?
import basetypes
## XXX: cacheme currently not working, due to compiler bug about static array type from `copyflat`
# import cacheme
type
  Reflectivity* = object
    layers*: seq[int]
    interp*: seq[AngleInterpolator]

proc setupReflectivity*(energyMin, energyMax: float,
                        numSamples: int, # `numSamples` so that it will be encoded in the cached data and
                                         # to only compute as many samples as really needed!
                        angleMin = 0.0, angleMax = 15.0,
                        numAngle = 1000
                       ): Reflectivity = #  {.cacheMe: "resources/reflectivity_cache".} =
  let angleMin = angleMin.°
  let angleMax = angleMax.°
  let energyMin = energyMin.keV
  let energyMax = energyMax.keV
  result = Reflectivity(
    layers: @[3 - 1, 3+4 - 1, 3+4+4 - 1, 3+4+4+3 - 1] # layers of LLNL telescope
  )
  # read reflectivities from H5 file
  let numCoatings = result.layers.len
  let ms = getLayers()
  echo "energy min: ", energyMin, " max ", energyMax
  for m in ms:
    let data = calculateReflectivity(m, angleMin, angleMax, energyMin, energyMax, numAngle, numSamples)
    result.interp.add initInterpolator(data,
                                       angleMin.float, angleMax.float,
                                       energyMin.float, energyMax.float,
                                       numAngle)

import nimhdf5
proc setupReflectivityFromH5*(): Reflectivity =
  const path = "llnl_layer_reflectivities.h5"
  result = Reflectivity(
    layers: @[3 - 1, 3+4 - 1, 3+4+4 - 1, 3+4+4+3 - 1] # layers of LLNL telescope
  )
  # read reflectivities from H5 file
  let numCoatings = result.layers.len
  var h5f = H5open(path, "r")
  let energies = h5f["/Energy", float]
  let angles = h5f["/Angles", float]
  for i in 0 ..< numCoatings:
    let reflDset = h5f[("Reflectivity" & $i).dset_str]
    let data = reflDset[float].reshape2D(reflDset.shape)
    result.interp.add initInterpolator(data,
                                       angles.min, angles.max,
                                       energies.min, energies.max,
                                       angles.len)
  discard h5f.close()
