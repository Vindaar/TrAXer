import macros, math, random, strformat

import glm
export glm

const NumSamples {.intdefine.} = 60

type
  Image* = object
    width*, height*: int

  Color* = distinct Vec3d
  Point* = distinct Vec3d

  ColorU8* = tuple[r, g, b: uint8]

template borrowOps(typ: typed): untyped =
  proc `[]=`*(v: var typ; ix: int; c: float): void {.inline, borrow.}
  proc `x=`*(v: var typ; c: float): void {.inline, borrow.}
  proc `y=`*(v: var typ; c: float): void {.inline, borrow.}
  proc `z=`*(v: var typ; c: float): void {.inline, borrow.}
  proc `w=`*(v: var typ; c: float): void {.inline, borrow.}
  proc `[]`*(v: typ; ix: int): float {.inline, borrow.}
  # following cannot be borrowd
  # proc `[]`*(v: var Color; ix: int): var float {.inline, borrow.}
  proc dot*(v, u: typ): float {.borrow.}
  proc cross*(v, u: typ): typ {.borrow.}
  #proc length_squared*(v: typ): float {.inline, borrow.}
  proc length*(v: typ): float {.inline, borrow.}
  proc normalize*(v: typ): typ {.inline, borrow.}
  proc `+`*(v: typ): typ {.inline, borrow.}
  proc `-`*(v: typ): typ {.inline, borrow.}
  proc `+=`*(v: var typ, u: typ) {.inline, borrow.}
  proc `-=`*(v: var typ, u: typ) {.inline, borrow.}
borrowOps(Color)
borrowOps(Point)

func x*(p: Point): float {.inline.} = p[0]
func y*(p: Point): float {.inline.} = p[1]
func z*(p: Point): float {.inline.} = p[2]

func r*(c: Color): float {.inline.} = c[0]
func g*(c: Color): float {.inline.} = c[1]
func b*(c: Color): float {.inline.} = c[2]

proc `==`*(p1, p2: Point): bool = result = p1.x == p2.x and p1.y == p2.y and p1.z == p2.z
proc `==`*(p1, p2: Color): bool = result = p1.r == p2.r and p1.g == p2.g and p1.b == p2.b

proc `$`*(v: Point): string =
  result = &"(Point: [{v[0]}, {v[1]}, {v[2]}])"
proc `$`*(v: Color): string =
  result = &"(Color: [{v[0]}, {v[1]}, {v[2]}])"

template makeMathBorrow(typ, op: typed): untyped {.dirty.} =
  proc `op`*(v, u: typ): typ {.inline, borrow.}
  proc `op`*(v: typ; val: float): typ {.inline, borrow.}
  proc `op`*(val: float; v: typ): typ {.inline, borrow.}
makeMathBorrow(Color, `+`)
makeMathBorrow(Color, `-`)
makeMathBorrow(Color, `/`)
makeMathBorrow(Color, `*`)
makeMathBorrow(Point, `+`)
#makeMathBorrow(Point, `-`)
makeMathBorrow(Point, `/`)
makeMathBorrow(Point, `*`)

proc `+`*(p: Point, d: Vec3d): Point =
  result = Point(p.Vec3d + d)

proc `-`*(p1, p2: Point): Vec3d =
  result = p1.Vec3d - p2.Vec3d

proc `+.`*(p1, p2: Point): Point =
  result = Point(p1.Vec3d + p2.Vec3d)

proc `-.`*(p1, p2: Point): Point =
  result = Point(p1.Vec3d - p2.Vec3d)

proc color*(r, g, b: float): Color = Color(vec3(r, g, b))
proc point*(x, y, z: float): Point = Point(vec3(x, y, z))

proc unitVector*(v: Vec3d): Vec3d = normalize(v)

proc nearZero*(v: Vec3d): bool =
  ## return true if the vector is close to 0 in all dim.
  const epsilon = 1e-8
  result = abs(v[0]) < epsilon and abs(v[1]) < epsilon and abs(v[2]) < epsilon

proc rotateAround*(v: Vec3d, around: Point, phi, theta, gamma: float): Vec3d =
  var v0 = v - around.Vec3d
  var mrot = rotateX(mat4d(), phi)
  mrot = rotateY(mrot, theta)
  mrot = rotateZ(mrot, gamma)
  let vrot = mrot * vec4(v0, 0)
  result = vec3(vrot.x, vrot.y, vrot.z) + around.Vec3d

template length_squared*(v: Vec3d): float = v.length2()
template length_squared*(v: Point): float = v.Vec3d.length2()
template length_squared*(v: Color): float = v.Vec3d.length2()

proc randomVec*(rnd: var Rand, min = 0.0, max = 1.0): Vec3d =
  ## generate a random 3 vector
  result = vec3(rnd.rand(min .. max), rnd.rand(min .. max), rnd.rand(min .. max))

proc randomInUnitSphere*(rnd: var Rand): Vec3d =
  while true:
    let p = rnd.randomVec(-1, 1)
    if p.length_squared >= 1: continue
    return p

proc randomUnitVector*(rnd: var Rand): Vec3d =
  result = unitVector(rnd.randomInUnitSphere())

proc randomInHemisphere*(rnd: var Rand, normal: Vec3d): Vec3d =
  let inUnitSphere = rnd.randomInUnitSphere()
  if inUnitSphere.dot(normal) > 0.0: # same hemisphere as the normal
    result = inUnitSphere
  else:
    result = -inUnitSphere

proc randomInUnitDisk*(rnd: var Rand): Vec3d =
  while true:
    let p = vec3(rnd.rand(-1.0 .. 1.0), rnd.rand(-1.0 .. 1.0), 0)
    if p.length_squared() >= 1.0: continue
    return p

type
  RayType* = enum
    rtCamera, ## A ray that is emitted *from the camera*
    rtLight   ## A ray that is emitted *from a light source*

  Ray* = object
    orig*: Point
    dir*: Vec3d
    typ*: RayType

proc initRay*(origin: Point, direction: Vec3d, typ: RayType): Ray =
  result = Ray(orig: origin, dir: direction, typ: typ)

proc at*(r: Ray, t: float): Point = result = (r.orig + t * r.dir)

type
  #SpectrumKind* = enum
  #  skRGB, ## classical case for RGB colors
  #  skXray ## Sampled X-ray energies, i.e. flux at N different energies, requiring reflectivity at same N values etc

  #Spectrum* = object
  #  case kind*: SpectrumKind
  #  of skRGB: color*: Color
  #  of skXray:
  #    energyMin*: float ## minimum energy in keV
  #    energyMax*: float ## maximum energy in keV
  #    data*: seq[float] ## Store N samples as float.

  Spectrum*[N: static int] = object
    data*: array[N, float]
    energyMin*: float ## Need these here? Or define them in a more global sense? Part of some setup or RenderContext?
    energyMax*: float

  RGBSpectrum* = Spectrum[3]

  XraySpectrum* = Spectrum[NumSamples]

  SomeSpectrum* = RGBSpectrum | XraySpectrum

func len*[N: static int](x: Spectrum[N]): int {.inline.} = N
func `[]`*[N: static int](x: Spectrum[N], idx: int): float {.inline.} = x.data[idx]
func `[]=`*[N: static int](x: var Spectrum[N], idx: int, val: float) {.inline.} =
    x.data[idx] = val

func isBlack*[N: static int](s: Spectrum[N]): bool =
  result = true
  for i in 0 ..< N:
    if s[i] != 0.0: return false

func initRGBSpectrum*(color: Color): RGBSpectrum =
  result = RGBSpectrum()
  result[0] = color.r
  result[1] = color.g
  result[2] = color.b

func toRGBSpectrum*(color: Color): RGBSpectrum = initRGBSpectrum(color)

func spectrumFromSampled*[S: SomeSpectrum](samples: array[3, float] | array[NumSamples, float], energyMin, energyMax: float): S =
  result = S(energyMin: energyMin, energyMax: energyMax)
  doAssert samples.len == result.data.len
  for i in 0 ..< result.data.len:
    result[i] = samples[i]

func xraySpectrumFromSampled*(samples: array[NumSamples, float], energyMin, energyMax: float): XraySpectrum =
  result = spectrumFromSampled[XraySpectrum](samples, energyMin, energyMax)

func emptySpectrum*[N: static int](energyMin, energyMax: float): Spectrum[N] =
  result = Spectrum[N](energyMin: energyMin, energyMax: energyMax)

func initEmptySpectrum*[S: SomeSpectrum](energyMin, energyMax: float): S =
  result = S(energyMin: energyMin, energyMax: energyMax)

func initEmptyXraySpectrum*(energyMin, energyMax: float): XraySpectrum =
  result = initEmptySpectrum[XraySpectrum](energyMin, energyMax)

proc sum*[S: SomeSpectrum](s: S): float =
  for i in 0 ..< s.data.len:
    result += s[i]

import numericalnim
proc toSpectrum*[S: SomeSpectrum](c: Color, _: typedesc[S]): S =
  when S is RGBSpectrum:
    result = initRGBSpectrum(c)
  else:
    result = initEmptyXraySpectrum(0.0, 0.0) # upsample to NumSamples
    let linear = newLinear1D(linspace(0.0, 2.0, 3), @[c[0], c[1], c[2]])
    let points = linspace(0.0, 2.0, NumSamples)
    for i in 0 ..< NumSamples:
      result[i] = linear.eval(points[i])

func toSpectrum*[T: SomeSpectrum](val: float, _: typedesc[T]): T =
  when T is RGBSpectrum:
    result = initRGBSpectrum(color(val,val,val))
  else:
    result = initEmptyXraySpectrum(0.0, 0.0) # default
    for i in 0 ..< NumSamples:
      result[i] = val

proc toSpectrum*[S: SomeSpectrum; T: SomeSpectrum](s: S, _: typedesc[T]): T =
  ## Performs the conversion from one Spectrum type to another using interpolation.
  ## Either downsamples or upsamples. Keep in mind that this is a lossy transformation!
  when S is T: result = s
  else:
    const inSamples = when S is RGBSpectrum: 3 else: NumSamples
    const outSamples = when T is RGBSpectrum: 3 else: NumSamples
    result = initEmptySpectrum[T](0.0, 0.0)
    let linear = newLinear1D(linspace(0.0, 2.0, inSamples), s.data)
    let points = linspace(0.0, 2.0, outSamples)
    for i in 0 ..< outSamples:
      result[i] = linear.eval(points[i])

func toColor*(x: RGBSpectrum): Color = color(x[0], x[1], x[2])
proc toColor*(x: XraySpectrum): Color = toColor x.toSpectrum(RGBSpectrum)

proc initSpectrum*[S: SomeSpectrum](data: seq[float], energyMin, energyMax: float): S =
  ## Compute the correct samples by linear interpolation from the smooth `data`
  ## This assumes the given `data` covers the range `energyMin` to `energyMax`!
  const numSamples = when S is RGBSpectrum: 3 else: NumSamples
  var samples: array[numSamples, float]
  let energies = linspace(energyMin, energyMax, data.len)
  let linear = newLinear1D(energies, data)
  let energiesSamples = linspace(energyMin, energyMax, numSamples)
  for i, E in energiesSamples:
    samples[i] = linear.eval(E)
  result = spectrumFromSampled[S](samples, energyMin, energyMax)

proc initXraySpectrum*(data: seq[float], energyMin, energyMax: float): XraySpectrum =
  ## Compute the correct samples by linear interpolation from the smooth `data`
  ## This assumes the given `data` covers the range `energyMin` to `energyMax`!
  result = initSpectrum[XraySpectrum](data, energyMin, energyMax) #xraySpectrumFromSampled(samples, energyMin, energyMax)

template makeMathSpectrum(op: typed): untyped {.dirty.} =
  #proc `op`*(v, u: Spectrum): Spectrum =
  #  doAssert v.kind == u.kind, "Cannot do math with different types of `Spectrum`."
  #  case v.kind
  #  of skRGB: result = initRGBSpectrum(op(v.color, u.color))
  #  of skXray:
  #    doAssert v.data.len == u.data.len, "Cannot do math with `XraySpectrum` with different number of samples."
  #    doAssert v.energyMin == u.energyMin, "Two `XraySpectrum` must describe the same energy range."
  #    doAssert v.energyMax == u.energyMax, "Two `XraySpectrum` must describe the same energy range."
  #    result = initEmptyXraySpectrum(v.data.len, v.energyMin, v.energyMax)
  #    for i in 0 ..< N:
  #      result[i] = op(v[i], u[i])
  proc `op`*[N: static int](v, u: Spectrum[N]): Spectrum[N] =
    # likely use `assert` instead
    #doAssert v.energyMin == u.energyMin, "Two `XraySpectrum` must describe the same energy range. " & $v.energyMin & " vs " & $u.energyMin
    #doAssert v.energyMax == u.energyMax, "Two `XraySpectrum` must describe the same energy range. " & $v.energyMax & " vs " & $u.energyMax
    result = emptySpectrum[v.N](v.energyMin, v.energyMax)
    for i in 0 ..< N:
      result[i] = op(v[i], u[i])
  proc `op`*[N: static int](v: Spectrum[N]; val: float): Spectrum[N] =
    result = v
    for i in 0 ..< N:
      result[i] = result[i] * val
  func `op`*[N: static int](val: float; v: Spectrum[N]): Spectrum[N] {.inline.} = v * val
makeMathSpectrum(`+`)
makeMathSpectrum(`-`)
makeMathSpectrum(`/`)
makeMathSpectrum(`*`)

type
  AngleInterpolator* = object
    anglesMin: float
    anglesMax: float
    numAngles: int
    data*: seq[XraySpectrum] ## Stores all energy slices, one for each angle

proc initInterpolator*(data: seq[seq[float]],
                       anglesMin, anglesMax: float,
                       energyMin, energyMax: float,
                       numAngles: int): AngleInterpolator =
  ## Input data is:
  ## Each angle, with all `seq[float]` energies
  # 1. turn every energy slice into `S` compatible data
  result = AngleInterpolator(anglesMin: anglesMin, anglesMax: anglesMax, numAngles: numAngles)
  for d in data:
    # 2. for each angle compute spectrum
    result.data.add initSpectrum[XraySpectrum](d, energyMin, energyMax)

#proc eval*[S: SomeSpectrum](interp: AngleInterpolator[S], angle: float): S =
proc eval*(interp: AngleInterpolator, angle: float): XraySpectrum =
  ## Given a certain angle returns the entire spectrum for that angle
  # 1. calculate the index from angleMin, angleMax, numAngles & angle
  let angleStep = ((interp.anglesMax - interp.anglesMin) / interp.numAngles.float)
  let idxInitial = (angle / angleStep).round.int
  if idxInitial < 0 or idxInitial > interp.data.high:
    result = initEmptyXraySpectrum(interp.data[0].energyMin, interp.data[0].energyMax)
  else:
    let idx = clamp(idxInitial, 0, interp.numAngles)
    result = interp.data[idx]

proc eval*[S: SomeSpectrum](interp: AngleInterpolator, angle: float, attenuation: var S) =
  ## Given a certain angle returns the entire spectrum for that angle
  # 1. calculate the index from angleMin, angleMax, numAngles & angle
  let angleStep = ((interp.anglesMax - interp.anglesMin) / interp.numAngles.float)
  let idxInitial = abs(angle / angleStep).round.int
  if idxInitial < 0 or idxInitial > interp.data.high:
    # write zero to attenuation
    for i in 0 ..< attenuation.len:
      attenuation[i] = 0.0
  else:
    let idx = clamp(idxInitial, 0, interp.numAngles)
    ## XXX: make this use the correct indexing!
    for i in 0 ..< attenuation.len:
      attenuation[i] = interp.data[idx][i] # 0.0

proc `-`*(val: float, interp: AngleInterpolator): AngleInterpolator =
  result = interp
  for x in mitems(result.data):
    x = val - x
