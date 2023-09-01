import std/math
import basetypes

type
  Sphere* = object
    radius*: float

  XyRect* = object
    x0*, x1*, y0*, y1*, k*: float

  XzRect* = object
    x0*, x1*, z0*, z1*, k*: float

  YzRect* = object
    y0*, y1*, z0*, z1*, k*: float

  Cylinder* = object
    radius*: float
    zMin*: float
    zMax*: float
    phiMax*: float

  Cone* = object ## Describes a positive cone, starting at z = 0 with `radius` towards +z. r = 0 at `height`
    radius*: float ## Radius at z = 0
    phiMax*: float
    height*: float ## Height of the cone, if complete. This is where its radius is 0.
    zMax*: float ## Cuts off the cone at zMax.

  Paraboloid* = object
    radius*: float
    zMin*: float
    zMax*: float
    phiMax*: float

  Hyperboloid* = object
    p1*: Point
    p2*: Point
    #radius1*: float
    #radius2*: float
    rMax*: float
    zMin*: float
    zMax*: float
    ah*: float
    ch*: float
    phiMax*: float

  Disk* = object
    distance*: float # distance along z axis
    radius*: float
    innerRadius*: float
    phiMax*: float = 360.0.degToRad

proc initSphere*(center: Point, radius: float): Sphere =
  result = Sphere(radius: radius)

proc initDisk*(distance: float, radius: float): Disk =
  result = Disk(distance: distance, radius: radius)

proc initXyRect*(x0, x1, y0, y1, k: float): XyRect =
  result = XyRect(x0: x0, x1: x1, y0: y0, y1: y1, k: k)

proc initXzRect*(x0, x1, z0, z1, k: float): XzRect =
  result = XzRect(x0: x0, x1: x1, z0: z0, z1: z1, k: k)

proc initYzRect*(y0, y1, z0, z1, k: float): YzRect =
  result = YzRect(y0: y0, y1: y1, z0: z0, z1: z1, k: k)

proc initHyperboloid*(p1, p2: Point, phiMax: float): Hyperboloid =
  ## Based on `pbrt`
  var p1 = p1
  var p2 = p2
  result.phiMax = phiMax ## XXX: Make radians in the future!
  let radius1 = sqrt(p1.x * p1.x + p1.y * p1.y)
  let radius2 = sqrt(p2.x * p2.x + p2.y * p2.y)
  result.rMax = max(radius1, radius2)
  result.zMin = min(p1.z, p2.z)
  result.zMax = max(p1.z, p2.z)
  # Compute implicit function coefficients for hyperboloid
  if result.p2.z == 0.0: swap(p1, p2)
  var pp = p1
  var
    xy1: float
    xy2: float
  result.ah = Inf
  result.ch = NaN
  while classify(result.ah) == fcInf or classify(result.ah) == fcNaN:
    pp = pp + 2.0 * (p2 - p1)
    xy1 = pp.x * pp.x + pp.y * pp.y
    xy2 = p2.x * p2.x + p2.y * p2.y
    result.ah = (1.0 / xy1 - (pp.z * pp.z) / (xy1 * p2.z * p2.z)) /
         (1.0 - (xy2 * pp.z * pp.z) / (xy1 * p2.z * p2.z))
    result.ch = (result.ah * xy2 - 1) / (p2.z * p2.z)
  result.p1 = p1
  result.p2 = p2

proc solveQuadratic*(a, b, c: float, t0, t1: var float): bool =
  ## Copied from `pbrt` `efloat.h`.
  ## Find quadratic discriminant
  let discrim = b * b - 4.0 * a * c
  if discrim < 0.0: return false
  let rootDiscrim = sqrt(discrim)

  #EFloat floatRootDiscrim(rootDiscrim, MachineEpsilon * rootDiscrim);

  # Compute quadratic _t_ values
  var q: float
  if b < 0:
    q = -0.5 * (b - rootDiscrim)
  else:
    q = -0.5 * (b + rootDiscrim)
  t0 = q / a
  t1 = c / q
  if t0 > t1: swap(t0, t1)
  result = true

import ./aabb
proc boundingBox*(s: Sphere, output_box: var AABB): bool =
  ##
  output_box = initAabb(
    - point(s.radius, s.radius, s.radius),
    + point(s.radius, s.radius, s.radius)
  )
  result = true

proc boundingBox*(s: Disk, output_box: var AABB): bool =
  ## in z direction only a small width
  output_box = initAabb(
    - point(s.radius, s.radius, s.distance - 0.0001),
    + point(s.radius, s.radius, s.distance + 0.0001)
  )
  result = true

proc boundingBox*(cyl: Cylinder, output_box: var AABB): bool =
  ## in z direction only a small width
  output_box = initAabb( ## XXX: Could be from 0 to Height
    - point(cyl.radius, cyl.radius, - 0.0001),
    + point(cyl.radius, cyl.radius, cyl.zMax + 0.0001)
  )
  result = true

proc boundingBox*(con: Cone, output_box: var AABB): bool =
  output_box = initAabb(
    - point(con.radius, con.radius, -0.0001),
    + point(con.radius, con.radius, con.zMax + 0.0001)
  )
  result = true

proc boundingBox*(con: Paraboloid, output_box: var AABB): bool =
  #output_box = initAabb(
  #  - point(con.radius, con.radius, -0.0001),
  #  + point(con.radius, con.radius, con.zMax + 0.0001)
  #)
  #result = true
  doAssert false, "IMPLEMENT"

proc boundingBox*(con: Hyperboloid, output_box: var AABB): bool =
  #output_box = initAabb(
  #  - point(con.radius, con.radius, -0.0001),
  #  + point(con.radius, con.radius, con.zMax + 0.0001)
  #)
  #result = true
  doAssert false, "IMPLEMENT"

proc boundingBox*(r: XyRect, outputBox: var AABB): bool =
  ## bounding box needs to have a non-zero width in each dimension!
  outputBox = initAabb(point(r.x0, r.y0, r.k - 0.0001),
                       point(r.x1, r.y1, r.k + 0.0001))
  result = true

proc boundingBox*(r: XzRect, outputBox: var AABB): bool =
  ## bounding box needs to have a non-zero width in each dimension!
  outputBox = initAabb(point(r.x0, r.k - 0.0001, r.z0),
                       point(r.x1, r.k + 0.0001, r.z1))
  result = true

proc boundingBox*(r: YzRect, outputBox: var AABB): bool =
  ## bounding box needs to have a non-zero width in each dimension!
  outputBox = initAabb(point(r.k - 0.0001, r.y0, r.z0),
                       point(r.k + 0.0001, r.y1, r.z1))
  result = true
