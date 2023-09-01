import macros, math, random, strformat

import glm
export glm

type
  Image* = object
    width*, height*: int

  Color* = distinct Vec3d
  Point* = distinct Vec3d

  ColorU8* = tuple[r, g, b: uint8]

  RayType* = enum
    rtCamera, ## A ray that is emitted *from the camera*
    rtLight   ## A ray that is emitted *from a light source*

  Ray* = object
    orig*: Point
    dir*: Vec3d
    typ*: RayType


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

template `.`*(c: Color, field: untyped): untyped =
  when astToStr(field) == "r":
    c[0]
  elif astToStr(field) == "g":
    c[1]
  elif astToStr(field) == "b":
    c[2]
  else:
    error("Invalid field " & astToStr(field) & " for Color!")

template `.`*(p: Point, field: untyped): untyped =
  when astToStr(field) == "x":
    p[0]
  elif astToStr(field) == "y":
    p[1]
  elif astToStr(field) == "z":
    p[2]
  else:
    error("Invalid field " & astToStr(field) & " for Point!")

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

proc initRay*(origin: Point, direction: Vec3d, typ: RayType): Ray =
  result = Ray(orig: origin, dir: direction, typ: typ)

proc at*(r: Ray, t: float): Point = result = (r.orig + t * r.dir)
