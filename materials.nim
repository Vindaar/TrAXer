import std / random
import basetypes
import numericalnim/interpolate

import sensorBuf

type
  MaterialKind* = enum
    mkLambertian, mkMetal, mkDielectric, mkDiffuseLight, mkSolarAxionEmission, mkSolarChameleonEmission, mkLaser, mkImageSensor, mkLightTarget, mkXrayMatter
  Material*[S: SomeSpectrum] = object
    case kind*: MaterialKind
    of mkLambertian: mLambertian*: Lambertian[S]
    of mkMetal: mMetal*: Metal[S]
    of mkDielectric: mDielectric*: Dielectric[S]
    of mkDiffuseLight: mDiffuseLight*: DiffuseLight[S]
    of mkSolarAxionEmission: mSolarAxionEmission*: SolarAxionEmission[S]
    of mkSolarChameleonEmission: mSolarChameleonEmission*: SolarChameleonEmission[S]
    of mkLaser: mLaser*: Laser[S]
    of mkImageSensor: mImageSensor*: ImageSensor[S]
    of mkLightTarget: mLightTarget*: LightTarget[S]
    of mkXrayMatter: mXrayMatter*: XrayMatter[S]

  TextureKind* = enum
    tkSolid, tkChecker#, tkAbsorbentTexture

  Texture*[S: SomeSpectrum] {.acyclic.} = ref object
    case kind*: TextureKind
    of tkSolid: tSolid: SolidTexture[S]
    of tkChecker: tChecker: CheckerTexture[S]

  SolidTexture*[S: SomeSpectrum] {.acyclic.} = ref object
    color*: S

  CheckerTexture*[S: SomeSpectrum] {.acyclic.} = ref object
    invScale*: float
    even*: Texture[S]
    odd*: Texture[S]

  ## XXX: Make all materials `ref object` to make copying them fast? We don't intend to modify them
  ## after all, but otherwise we end up copying the `Spectrum`?

  DiffuseLight*[S: SomeSpectrum] = object
    emit*: Texture[S]

  SolarAxionEmission*[S: SomeSpectrum] = object
    emit*: Texture[S] ## The color used for the regular light source
    fluxRadiusCDF*: seq[float] ## The relative emission for each radius as a CDF
    diffFluxR*: seq[S] ## The differential flux at each radius as a `Spectrum`.
    radii*: seq[float] ## The associated radii

  SolarChameleonEmission*[S: SomeSpectrum] = object
    emit*: Texture[S] ## The color used for the regular light source
    diffFlux*: S ## The differential flux as a `Spectrum`
    radius*: float    ## The radius at the solar tachocline (0.7 solar radii by default)
    Δradius*: float   ## Range in which to vary around the radius

  ## While the type currently does not actually use the generic, we keep it around to not lose the information
  SensorKind* = enum sCount, sSum
  ImageSensor*[S: SomeSpectrum] = object
    sensor*: Sensor2D
    kind*: SensorKind

  ## A `LightTarget` is a material assigned to any `Hittable`, which will be used to sample rays from
  ## `DiffuseLight` sources. Any `DiffuseLight` will sample rays towards the surface of the `Hittable`
  ## of `LightTarget`.
  LightTarget*[S: SomeSpectrum] = object
    visible*: bool # Whether it is visible for the Camera based rays
    albedo*: Texture[S] # The texture is only relevant for the Camera based rays to visualize it.

  ## If a `Laser` is applied to an object, it means the entire surface emits alnog the normal. I.e.
  ## this produces parallel light if emitting from a disk (for the Sun!)
  Laser*[S: SomeSpectrum] = object
    emit*: Texture[S]

  Lambertian*[S: SomeSpectrum] = object
    albedo*: Texture[S]

  Metal*[S: SomeSpectrum] = object
    albedo*: S
    fuzz*: float

  ## While the type currently does not actually use the generic, we keep it around to not lose the information
  Dielectric*[S: SomeSpectrum] = object
    ir*: float # refractive index (could use `eta`)

  ## XrayMatter defines a material, in which the rays are scattered / transmitted (TODO)
  ## based on `xrayAttenuation`
  ## IMPORTANT: This *must* be a `ref object`. Otherwise copying the `mat` field in `hit`
  ## is incredibly expensive!
  XrayMatter*[S: SomeSpectrum] = ref object
    albedo*: S ## The appearance for `Camera` rays (if not in X-ray mode as well)
    fuzz*: float
    refl*: AngleInterpolator ## interpolate energy & angle for reflectivity
    trans*: AngleInterpolator ## interpolate energy & angle for transmissivity
    #refl*: AngleInterpolator[S] ## interpolate energy & angle for reflectivity
    #trans*: AngleInterpolator[S] ## interpolate energy & angle for transmissivity

  AnyMaterial* = Lambertian | Metal | Dielectric | DiffuseLight | Laser | SolarAxionEmission | SolarChameleonEmission | Dielectric | ImageSensor | LightTarget | XrayMatter

  ## Leave this here or not?
  HitRecord*[S: SomeSpectrum] = object
    p*: Point
    normal*: Vec3d
    t*: float
    frontFace*: bool
    u*, v*: float
    mat*: Material[S]

proc value*[S: SomeSpectrum](s: Texture[S], u, v: float, p: Point): S {.gcsafe.}

proc solidColor*(c: Color): SolidTexture[RGBSpectrum] = SolidTexture[RGBSpectrum](color: initRGBSpectrum(c))
proc solidColor*(r, g, b: float): SolidTexture[RGBSpectrum] = SolidTexture[RGBSpectrum](color: initRGBSpectrum(color(r, g, b)))
proc value*[S: SomeSpectrum](s: SolidTexture[S], u, v: float, p: Point): S = s.color # Is solid everywhere after all

proc toTexture*(x: Color): Texture[RGBSpectrum] = Texture[RGBSpectrum](kind: tkSolid, tSolid: solidColor(x))
proc toTexture*[S: SomeSpectrum](x: SolidTexture[S]): Texture[S] = Texture[S](kind: tkSolid, tSolid: x)
proc toTexture*[S: SomeSpectrum](x: CheckerTexture[S]): Texture[S] = Texture[S](kind: tkChecker, tChecker: x)

proc toSpectrum*[S: SomeSpectrum; T: SomeSpectrum](x: Texture[S], _: typedesc[T]): Texture[T]

proc toSpectrum*[S: SomeSpectrum; T: SomeSpectrum](x: SolidTexture[S], _: typedesc[T]): SolidTexture[T] =
  result = SolidTexture[T](color: x.color.toSpectrum(T))

proc toSpectrum*[S: SomeSpectrum; T: SomeSpectrum](x: CheckerTexture[S], _: typedesc[T]): CheckerTexture[T] =
  result = CheckerTexture[T](invScale: x.invScale)
  result.even = toSpectrum(x.even, T)
  result.odd = toSpectrum(x.odd, T)



proc toSpectrum*[S: SomeSpectrum; T: SomeSpectrum](x: Texture[S], _: typedesc[T]): Texture[T] =
  result = Texture[T](kind: x.kind)
  case x.kind
  of tkSolid:   result.tSolid   = toSpectrum(x.tSolid, T)
  of tkChecker: result.tChecker = toSpectrum(x.tChecker, T)

proc checkerTexture*[S: SomeSpectrum](scale: float, even, odd: Texture[S]): CheckerTexture[S] = CheckerTexture[S](invScale: 1.0 / scale, even: even, odd: odd)
proc checkerTexture*(scale: float, c1, c2: Color): CheckerTexture[RGBSpectrum] =
  result = CheckerTexture[RGBSpectrum](invScale: 1.0 / scale, even: toTexture(c1), odd:  toTexture(c2))

proc value*[S: SomeSpectrum](s: CheckerTexture[S], u, v: float, p: Point): S =
  ## Return the color of the checker board at the u, v coordinates and `p`
  let xInt = floor(s.invScale * p.x).int
  let yInt = floor(s.invScale * p.y).int
  let zInt = floor(s.invScale * p.z).int
  let isEven = (xInt + yInt + zInt) mod 2 == 0
  result = if isEven: s.even.value(u, v, p) else: s.odd.value(u, v, p)

proc value*[S: SomeSpectrum](s: Texture[S], u, v: float, p: Point): S {.gcsafe.} =
  case s.kind
  of tkSolid:   result = s.tSolid.value(u, v, p)
  of tkChecker: result = s.tChecker.value(u, v, p)

proc initDiffuseLight*(a: Color): DiffuseLight[RGBSpectrum] =
  result = DiffuseLight[RGBSpectrum](emit: toTexture(a))

proc initSolarAxionEmission*(a: Color,
                             fluxRadiusCDF: seq[float], # CDF of entire diff flux per radius
                             diffFluxR: seq[seq[float]], # differential flux
                             radii: seq[float], # associated radii
                             energyMin, energyMax: float
                            ): SolarAxionEmission[XraySpectrum] =
  ## XXX: Make the code actually use `energyMin, energyMax` as those are now used for the reflecitvity
  ## Until that is done the solar emission won't be correct for any energies other than default!

  # convert each differential flux to an `XraySpectrum`
  var diffFluxRS = newSeq[XraySpectrum](diffFluxR.len)
  doAssert fluxRadiusCDF.len == radii.len, "Need one radius for each flux CDF entry."
  doAssert diffFluxR.len == radii.len, "Need one radius for each radial differential flux."
  for i in 0 ..< radii.len:
    diffFluxRS[i] = initXraySpectrum(diffFluxR[i], energyMin, energyMax)
  result = SolarAxionEmission[XraySpectrum](emit: toSpectrum(toTexture(a), XraySpectrum),
                                            fluxRadiusCDF: fluxRadiusCDF,
                                            diffFluxR: diffFluxRS,
                                            radii: radii)

proc initSolarChameleonEmission*(a: Color,
                                 diffFlux: seq[float],
                                 radius: float,
                                 Δradius: float,
                                 energyMin, energyMax: float
                       ): SolarChameleonEmission[XraySpectrum] =
  # convert spectrum to an XraySpectrum.
  let fluxSpectrum = initXraySpectrum(diffFlux, energyMin, energyMax)
  result = SolarChameleonEmission[XraySpectrum](
    emit: toSpectrum(toTexture(a), XraySpectrum),
    diffFlux: fluxSpectrum,
    radius: radius,
    Δradius: Δradius
  )

proc initLaser*(a: Color): Laser[RGBSpectrum] =
  result = Laser[RGBSpectrum](emit: toTexture(a))

proc initLambertian*(a: Color): Lambertian[RGBSpectrum] =
  result = Lambertian[RGBSpectrum](albedo: toTexture(a))

proc initLambertian*(t: Texture): Lambertian[RGBSpectrum] =
  result = Lambertian[RGBSpectrum](albedo: t)

#proc initXrayMatter*(a: Color, fuzz: float, refl, trans: AngleInterpolator[XraySpectrum]): XrayMatter[XraySpectrum] =
proc initXrayMatter*(a: Color, fuzz: float, refl, trans: AngleInterpolator): XrayMatter[XraySpectrum] =
  result = XrayMatter[XraySpectrum](albedo: toSpectrum(a, XraySpectrum), fuzz: fuzz, refl: refl, trans: trans)

proc initMetal*(a: Color, f: float): Metal[RGBSpectrum] =
  result = Metal[RGBSpectrum](albedo: toSpectrum(a, RGBSpectrum), fuzz: f)

proc initDielectric*(ir: float): Dielectric[RGBSpectrum] =
  result = Dielectric[RGBSpectrum](ir: ir)

proc initImageSensor*(width, height: int, kind: SensorKind = sCount): ImageSensor[XraySpectrum] =
  result = ImageSensor[XraySpectrum](kind: kind, sensor: initSensor(width, height))

proc initLightTarget*(a: Color, visible: bool): LightTarget[RGBSpectrum] =
  result = LightTarget[RGBSpectrum](albedo: toTexture(a), visible: visible)

proc toMaterial*[S: SomeSpectrum](m: Lambertian[S]): Material[S]             = Material[S](kind: mkLambertian, mLambertian: m)
proc toMaterial*[S: SomeSpectrum](m: Metal[S]): Material[S]                  = Material[S](kind: mkMetal, mMetal: m)
proc toMaterial*[S: SomeSpectrum](m: Dielectric[S]): Material[S]             = Material[S](kind: mkDielectric, mDielectric: m)
proc toMaterial*[S: SomeSpectrum](m: XrayMatter[S]): Material[S]             = Material[S](kind: mkXrayMatter, mXrayMatter: m)
proc toMaterial*[S: SomeSpectrum](m: DiffuseLight[S]): Material[S]           = Material[S](kind: mkDiffuseLight, mDiffuseLight: m)
proc toMaterial*[S: SomeSpectrum](m: SolarAxionEmission[S]): Material[S]     = Material[S](kind: mkSolarAxionEmission, mSolarAxionEmission: m)
proc toMaterial*[S: SomeSpectrum](m: SolarChameleonEmission[S]): Material[S] = Material[S](kind: mkSolarChameleonEmission, mSolarChameleonEmission: m)
proc toMaterial*[S: SomeSpectrum](m: Laser[S]): Material[S]                  = Material[S](kind: mkLaser, mLaser: m)
proc toMaterial*[S: SomeSpectrum](m: ImageSensor[S]): Material[S]            = Material[S](kind: mkImageSensor, mImageSensor: m)
proc toMaterial*[S: SomeSpectrum](m: LightTarget[S]): Material[S]            = Material[S](kind: mkLightTarget, mLightTarget: m)
template initMaterial*[T: AnyMaterial](m: T): Material              = m.toMaterial()

template lambertTargetBody(): untyped {.dirty.} =
  var scatter_direction = rec.normal + rnd.randomUnitVector()

  # catch degenerate scatter direction
  if scatter_direction.nearZero():
    scatter_direction = rec.normal

  scattered = initRay(rec.p, scatter_direction, r_in.typ)
  attenuation = m.albedo.value(rec.u, rec.v, rec.p)
  result = true

proc scatter*[S: SomeSpectrum](
  m: Lambertian[S], rnd: var Rand,
  r_in: Ray, rec: HitRecord[S],
  attenuation: var S, scattered: var Ray
                             ): bool {.gcsafe.}  =
  lambertTargetBody()

proc scatter*[S: SomeSpectrum](
  m: LightTarget[S], rnd: var Rand,
  r_in: Ray, rec: HitRecord[S],
  attenuation: var S, scattered: var Ray
                             ): bool {.gcsafe.}  =
  ## Scatters light like a Lambertian if it's set to be visible and the incoming ray
  ## comes from the `Camera`.
  if m.visible and r_in.typ == rtCamera:
    lambertTargetBody()

proc scatter*[S: SomeSpectrum](
  m: Metal[S], rnd: var Rand, r_in: Ray, rec: HitRecord[S],
  attenuation: var S, scattered: var Ray
                             ): bool {.gcsafe.}  =
  var reflected = unitVector(r_in.dir).reflect(rec.normal)
  scattered = initRay(rec.p, reflected + m.fuzz * rnd.randomInUnitSphere(), r_in.typ)
  attenuation = m.albedo
  result = scattered.dir.dot(rec.normal) > 0

from std / os import getEnv
from std / strutils import parseFloat
## Note: Long term this will maybe not stay in here as a feature. But for the time being it is
## useful. These are the values used to achieve a figure error matching the PANTER dataset
## of the LLNL telescope. The match is not perfect, but it is similar enough.
## It was determined using non linear optimization of the raytracer, by setting these
## parameters on the raytracing call in batch mode and computing the HPD, circle of 80 and 90%
## fluxes and comparing with the ideal value for the Al Kα line.
## (making these a command line argument would require us to replace the `fuzz` field of `XrayMatter`
## by all these fields. Possible, but for the time being not needed.
let fuzzIn = getEnv("FUZZ_IN", "3.257544581618656").parseFloat
let fuzzOrth = getEnv("FUZZ_ORTH", "0.2242798353909466").parseFloat
let fuzzInScale = getEnv("FUZZ_IN_SCALE", "0.9814814814814816").parseFloat
let fuzzOuterScale = getEnv("FUZZ_OUTER_SCALE", "9.22976680384088").parseFloat
let fuzzInRatio = getEnv("FUZZ_IN_RATIO", "1.083333333333333").parseFloat

proc scatter*[S: SomeSpectrum](
  m: XrayMatter[S], rnd: var Rand, r_in: Ray, rec: HitRecord[S],
  attenuation: var S, scattered: var Ray
                             ): bool {.gcsafe.}  =
  let r_inUnit = r_in.dir.unitVector
  let angle = abs(90.0 - arccos(r_inUnit.dot(rec.normal)).radToDeg) # calc incidence angle
  ## Apply reflectivity (if incoming is X-rays and not camera rays)
  ## XXX: MAKE THIS adjustable based on some keyboard command!
  if r_in.typ == rtLight:
    m.refl.eval(angle, attenuation)
  else:
    attenuation = m.albedo
  # XXX: based on ratio of reflected and transmitted decide if we reflect or transmit
  ## XXX: should we go by the mean? Or how to deal with it? Just separate completely?
  let toReflect = true ## rnd.rand(1.0) <= refl
  if toReflect:
    var reflected = r_inUnit.reflect(rec.normal)
    ## XXX: implement `fuzz` not as random in unit sphere, but random in normal distribution!
    if m.fuzz > 0.0: # no need to random sample if `fuzz` not used
      # -> this is the 'old' standard way of fuzzing into a normal distribution
      #scattered = initRay(rec.p, reflected + m.fuzz * rnd.randomInNormalDist(), r_in.typ)
      # Compute two orthogonal vectors to the outgoing vector. We will then fuzz into each
      # direction separately. We want more fuzzing in the plane of the surface normal and
      # incoming / outgoing ray than in the orthogonal to that.
      let n_orth = cross(reflected, rec.normal).normalize
      let r_orth = cross(reflected, n_orth).normalize
      # The idea is to sample from 2 normal distribution. One very narrow one which is used
      # for most (about 1σ of samples) and if the value is too large we sample again
      # from a wider normal distribution. We attempt to reproduce the fact that the majority
      # is in a "spiky center" and the tails are much longer than such a normal distribution can
      # describe.
      let fzIn = fuzzIn * fuzzInScale
      let factor = rnd.gauss(mu = 0.0, sigma = fzIn)
      let fc = if factor > fzIn * fuzzInRatio: rnd.gauss(mu = 0.0, sigma = fuzzIn * fuzzOuterScale)
               else: factor
      # orthogonal scaling
      let factor_orth = rnd.gauss(mu = 0.0, sigma = fuzzOrth)
      # final ray: reflected and the two orthogonal fuzzing components
      scattered = initRay(rec.p, reflected + m.fuzz * (fc * r_orth) + m.fuzz * (factorOrth * n_orth), r_in.typ)
    else:
      scattered = initRay(rec.p, reflected, r_in.typ)
    result = scattered.dir.dot(rec.normal) > 0
  else:
    ## add transmission code taking into account refractive indices (which are energy dependent!)
    ## XXX: for now we just ignore refractive indices and continue straight
    #let trans = m.trans.eval(angle, energy)
    ## XXX: in this case attenuation wrong!
    scattered = r_in
    result = true # "scattered"
    ## XXX: attenuation depends on the *thickness* along the medium
    ## -> We can compute it based on the *last* point when we *leave* the medium
    ## i.e. rec.t
    #attenuation = color(trans, trans, trans)

proc reflectance(cosine, refIdx: float): float =
  ## use Schlick's approximation for reflectance
  var r0 = (1 - refIdx) / (1 + refIdx)
  r0 = r0 * r0
  result = r0 + (1 - r0) * pow(1 - cosine, 5)

proc scatter*[S: SomeSpectrum](
  m: Dielectric[S],
  rnd: var Rand,
  r_in: Ray, rec: HitRecord[S],
  attenuation: var S, scattered: var Ray
                             ): bool {.gcsafe.}  =
  attenuation = toSpectrum(1.0, S)
  let refractionRatio = if rec.frontFace: (1.0 / m.ir) else: m.ir

  let unitDirection = unitVector(r_in.dir)
  let cosTheta = min(dot(-unitDirection, rec.normal), 1.0)
  let sinTheta = sqrt(1.0 - cosTheta * cosTheta)

  let cannotRefract = refraction_ratio * sinTheta > 1.0
  var direction: Vec3d

  if cannotRefract or reflectance(cosTheta, refractionRatio) > rnd.rand(1.0):
    direction = reflect(unitDirection, rec.normal)
  else:
    direction = refract(unitDirection, rec.normal, refractionRatio)

  scattered = initRay(rec.p, direction, r_in.typ)
  result = true

type
  NonEmittingMaterials* = Lambertian | Metal | Dielectric | LightTarget | XrayMatter
  EmittingMaterials* = DiffuseLight | Laser | ImageSensor

proc scatter*[T: DiffuseLight | Laser | SolarAxionEmission | SolarChameleonEmission; S: SomeSpectrum](
  m: T,
  rnd: var Rand,
  r_in: Ray, rec: HitRecord[S],
  attenuation: var S, scattered: var Ray
                                                     ): bool {.gcsafe.} =
  ## Diffuse lights, lasers and solare emissions do not scatter!
  result = false

proc scatter*[S: SomeSpectrum](
  m: ImageSensor[S], rnd: var Rand, r_in: Ray, rec: HitRecord[S],
  attenuation: var S, scattered: var Ray
                             ): bool {.gcsafe.} =
  ## An image sensor is a perfect sink! (At least we assume so)
  result = false
  ## XXX: handle spectrum!
  if r_in.typ == rtLight:
    #echo "Hit at: ", (rec.u, rec.v)
    let x = (rec.u * (m.sensor.width - 1).float).round.int
    let y = (rec.v * (m.sensor.height - 1).float).round.int
    case m.kind
    of sCount: m.sensor[x, y] = m.sensor[x, y] + 1.0
    of sSum:   m.sensor[x, y] = m.sensor[x, y] + attenuation.sum

proc scatter*[S: SomeSpectrum](
  m: Material[S],
  rnd: var Rand,
  r_in: Ray, rec: HitRecord[S],
  attenuation: var S, scattered: var Ray
                             ): bool {.gcsafe.} =
  case m.kind
  of mkLambertian:             result = m.mLambertian.scatter(rnd, r_in, rec, attenuation, scattered)
  of mkMetal:                  result = m.mMetal.scatter(rnd, r_in, rec, attenuation, scattered)
  of mkDielectric:             result = m.mDielectric.scatter(rnd, r_in, rec, attenuation, scattered)
  of mkXrayMatter:             result = m.mXrayMatter.scatter(rnd, r_in, rec, attenuation, scattered)
  of mkDiffuseLight:           result = m.mDiffuseLight.scatter(rnd, r_in, rec, attenuation, scattered)
  of mkSolarAxionEmission:     result = m.mSolarAxionEmission.scatter(rnd, r_in, rec, attenuation, scattered)
  of mkSolarChameleonEmission: result = m.mSolarChameleonEmission.scatter(rnd, r_in, rec, attenuation, scattered)
  of mkLaser:                  result = m.mLaser.scatter(rnd, r_in, rec, attenuation, scattered)
  of mkImageSensor:            result = m.mImageSensor.scatter(rnd, r_in, rec, attenuation, scattered)
  of mkLightTarget:            result = m.mLightTarget.scatter(rnd, r_in, rec, attenuation, scattered)

proc emit*[S: SomeSpectrum](m: Lambertian[S] | Metal[S] | Dielectric[S] | LightTarget[S] | XrayMatter[S],
                            u, v: float, p: Point): S =
  ## Materials that don't emit just return black!
  result = toSpectrum(color(0, 0, 0), S)

proc emit*[S: SomeSpectrum](m: DiffuseLight[S] | Laser[S] | SolarAxionEmission[S] | SolarChameleonEmission[S],
                            u, v: float, p: Point): S =
  ##XXX: emission of `SolarAxionEmission`
  ## At the radius at which we sampled from: extract the entire flux spectrum found at that
  ## radius and return it.
  result = m.emit.value(u, v, p)

import colormaps
proc emit*[S: SomeSpectrum](m: ImageSensor[S], u, v: float, p: Point): S =
  # adjust the sensor data
  #echo "Emission at p = ", p, " (u, v) = ", (u, v)
  let x = (u * (m.sensor.width - 1).float).round.int
  let y = (v * (m.sensor.height - 1).float).round.int
  let val = m.sensor[x, y]
  ## XXX: HANDLE SPECTRUM
  if val > 0:
    #echo val, " max ", m.sensor.currentMax[]
    let cIdx = (val / m.sensor.currentMax[]) * 255.0
    let v = ViridisRaw[cIdx.round.int]
    result = toSpectrum(color(v[0], v[1], v[2]), S)
  else:
    let v = ViridisRaw[0]
    result = toSpectrum(color(v[0], v[1], v[2]), S)

proc emit*[S: SomeSpectrum](m: Material[S], u, v: float, p: Point): S =
  case m.kind
  of mkLambertian:             result = m.mLambertian.emit(u, v, p)
  of mkMetal:                  result = m.mMetal.emit(u, v, p)
  of mkDielectric:             result = m.mDielectric.emit(u, v, p)
  of mkXrayMatter:             result = m.mXrayMatter.emit(u, v, p)
  of mkDiffuseLight:           result = m.mDiffuseLight.emit(u, v, p)
  of mkSolarAxionEmission:     result = m.mSolarAxionEmission.emit(u, v, p)
  of mkSolarChameleonEmission: result = m.mSolarChameleonEmission.emit(u, v, p)
  of mkLaser:                  result = m.mLaser.emit(u, v, p)
  of mkImageSensor:            result = m.mImageSensor.emit(u, v, p)
  of mkLightTarget:            result = m.mLightTarget.emit(u, v, p)

from std/algorithm import lowerBound
proc emitAxion*[S: SomeSpectrum](m: DiffuseLight[S] | Laser[S] | SolarAxionEmission[S] | SolarChameleonEmission[S],
                                 p: Point, radius: float): S =
  ## At the radius at which we sampled from: extract the entire flux spectrum found at that
  ## radius and return it.
  when typeof(m) is DiffuseLight[S] | Laser[S]:
    result = m.emit.value(0.5, 0.5, p) # in this case just emit the color at an arbitrary point
  elif typeof(m) is SolarChameleonEmission:
    result = m.diffFlux # just emit the entire spectrum! No radial dependence
  else:
    # use the solar emission data!
    let
      pRad = p.length # determine radius from sampled point `p`
      r = pRad / radius #
      iRad = m.radii.lowerBound(r)
    result = m.diffFluxR[iRad] # get differential flux at this radius
    #echo "Initial spectrum : ", result, " at radius: ", iRad

proc emitAxion*[S: SomeSpectrum](m: Material[S], p: Point, radius: float): S =
  case m.kind
  #of mkLambertian:            result = m.mLambertian.emitAxion(p, radius)
  #of mkMetal:                 result = m.mMetal.emitAxion(p, radius)
  #of mkDielectric:            result = m.mDielectric.emitAxion(p, radius)
  #of mkXrayMatter:            result = m.mXrayMatter.emitAxion(p, radius)
  of mkDiffuseLight:           result = m.mDiffuseLight.emitAxion(p, radius)
  of mkSolarAxionEmission:     result = m.mSolarAxionEmission.emitAxion(p, radius)
  of mkSolarChameleonEmission: result = m.mSolarChameleonEmission.emitAxion(p, radius)
  of mkLaser:                  result = m.mLaser.emitAxion(p, radius)
  #of mkImageSensor:           result = m.mImageSensor.emitAxion(p, radius)
  #of mkLightTarget:           result = m.mLightTarget.emitAxion(p, radius)
  else: doAssert false, "not supported " & $m.kind


## XXX: replace these by a generic version that uses `fieldPairs`?
proc toSpectrum*[S: SomeSpectrum; T: SomeSpectrum](m: Lambertian[S], _: typedesc[T]): Lambertian[T] =
  result = Lambertian[T](albedo: toSpectrum(m.albedo, T))
proc toSpectrum*[S: SomeSpectrum; T: SomeSpectrum](m: Metal[S], _: typedesc[T]): Metal[T] =
  result = Metal[T](albedo: toSpectrum(m.albedo, T), fuzz: m.fuzz)
proc toSpectrum*[S: SomeSpectrum; T: SomeSpectrum](m: DiffuseLight[S], _: typedesc[T]): DiffuseLight[T] =
  result = DiffuseLight[T](emit: toSpectrum(m.emit, T))
proc toSpectrum*[S: SomeSpectrum; T: SomeSpectrum](m: Laser[S], _: typedesc[T]): Laser[T] =
  result = Laser[T](emit: toSpectrum(m.emit, T))
proc toSpectrum*[S: SomeSpectrum; T: SomeSpectrum](m: Dielectric[S], _: typedesc[T]): Dielectric[T] =
  result = Dielectric[T](ir: m.ir)
proc toSpectrum*[S: SomeSpectrum; T: SomeSpectrum](m: XrayMatter[S], _: typedesc[T]): XrayMatter[T] =
  result = XrayMatter[T](albedo: toSpectrum(m.albedo, T),
                         fuzz: m.fuzz,
                         refl: m.refl, trans: m.trans)
proc toSpectrum*[S: SomeSpectrum; T: SomeSpectrum](m: ImageSensor[S], _: typedesc[T]): ImageSensor[T] =
  result = ImageSensor[T](sensor: m.sensor)
proc toSpectrum*[S: SomeSpectrum; T: SomeSpectrum](m: SolarAxionEmission[S], _: typedesc[T]): SolarAxionEmission[T] =
  result = SolarAxionEmission[T](emit: toSpectrum(m.emit, T), fluxRadiusCDF: m.fluxRadiusCDF)
proc toSpectrum*[S: SomeSpectrum; T: SomeSpectrum](m: SolarChameleonEmission[S], _: typedesc[T]): SolarChameleonEmission[T] =
  result = SolarChameleonEmission[T](emit: toSpectrum(m.emit, T),
                                     diffFlux: toSpectrum(m.diffFlux, T),
                                     radius: m.radius, Δradius: m.Δradius)
proc toSpectrum*[S: SomeSpectrum; T: SomeSpectrum](m: LightTarget[S], _: typedesc[T]): LightTarget[T] =
  result = LightTarget[T](visible: m.visible, albedo: toSpectrum(m.albedo, T))

proc toSpectrum*[S: SomeSpectrum; T: SomeSpectrum](m: Material[S], _: typedesc[T]): Material[T] =
  ## Converts given material from spectrum S to T
  when S is T: result = m
  else:
    case m.kind
    of mkLambertian:             result = toMaterial m.mLambertian.toSpectrum(T)
    of mkMetal:                  result = toMaterial m.mMetal.toSpectrum(T)
    of mkDielectric:             result = toMaterial m.mDielectric.toSpectrum(T)
    of mkXrayMatter:             result = toMaterial m.mXrayMatter.toSpectrum(T)
    of mkDiffuseLight:           result = toMaterial m.mDiffuseLight.toSpectrum(T)
    of mkSolarAxionEmission:     result = toMaterial m.mSolarAxionEmission.toSpectrum(T)
    of mkSolarChameleonEmission: result = toMaterial m.mSolarChameleonEmission.toSpectrum(T)
    of mkLaser:                  result = toMaterial m.mLaser.toSpectrum(T)
    of mkImageSensor:            result = toMaterial m.mImageSensor.toSpectrum(T)
    of mkLightTarget:            result = toMaterial m.mLightTarget.toSpectrum(T)
