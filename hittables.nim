import basetypes, math, aabb, algorithm, random

import numericalnim/interpolate

import materials, shapes
export materials, shapes

type
  Transform* = Mat4d

  HittableKind* = enum
    htSphere, htCylinder, htCone, htParaboloid, htHyperboloid, htBvhNode, htXyRect, htXzRect, htYzRect, htBox, htDisk
  Hittable*[S: SomeSpectrum] {.acyclic.} = ref object
    trans*: Transform = mat4d()
    invTrans*: Transform = mat4d()
    mat*: Material[S]
    case kind*: HittableKind
    of htSphere: hSphere*: Sphere
    of htCylinder: hCylinder*: Cylinder
    of htCone: hCone*: Cone
    of htParaboloid: hParaboloid*: Paraboloid
    of htHyperboloid: hHyperboloid*: Hyperboloid
    of htBvhNode: hBvhNode*: BvhNode[S]
    of htXyRect: hXyRect*: XyRect
    of htXzRect: hXzRect*: XzRect
    of htYzRect: hYzRect*: YzRect
    of htBox: hBox*: Box[S]
    of htDisk: hDisk: Disk

  HittablesList*[S: SomeSpectrum] = object
    len*: int # len of data seq
    #size*: int # internal data size
    data*: seq[Hittable[S]] #ptr UncheckedArray[Hittable]

  BvhNode*[S: SomeSpectrum] = object
    left*: Hittable[S]
    right*: Hittable[S]
    box*: AABB

  Box*[S: SomeSpectrum] = object
    boxMin*: Point
    boxMax*: Point
    sides*: HittablesList[S]

  #AnyHittable* = Sphere | Cylinder | Cone | Paraboloid | Hyperboloid | BvhNode | XyRect | XzRect | YzRect | Box | Disk

  GenericHittablesList* = object
    rgbs: HittablesList[RGBSpectrum]
    xray: HittablesList[XraySpectrum]

proc initBvhNode*[S: SomeSpectrum](rnd: var Rand, list: HittablesList[S], start, stop: int): BvhNode[S]
proc initBvhNode*[S: SomeSpectrum](rnd: var Rand, list: HittablesList[S]): BvhNode[S] =
  result = initBvhNode[S](rnd, list, 0, list.len)

proc initHittables*[S: SomeSpectrum](size: int = 8): HittablesList[S] =
  ## allocates memory for `size`, but remains empty
  let size = if size < 8: 8 else: size
  result.len = 0
  result.data = newSeqOfCap[Hittable[S]](size)

proc `[]`*[S: SomeSpectrum](h: HittablesList[S], idx: int): Hittable[S] =
  assert idx < h.len
  result = h.data[idx]

proc `[]`*[S: SomeSpectrum](h: var HittablesList[S], idx: int): var Hittable[S] =
  assert idx < h.len
  result = h.data[idx]

proc `[]=`*[S: SomeSpectrum](h: var HittablesList[S], idx: int, el: Hittable[S]) =
  assert idx < h.len
  h.data[idx] = el

iterator items*[S: SomeSpectrum](h: HittablesList[S]): Hittable[S] =
  for idx in 0 ..< h.len:
    yield h[idx]

proc `[]`*[S: SomeSpectrum](h: HittablesList[S], slice: Slice[int]): HittablesList[S] =
  let sliceLen = slice.b - slice.a
  result = initHittables[S](sliceLen)
  for idx in 0 ..< sliceLen:
    result.data[idx] = h.data[slice.a + idx]
  result.len = sliceLen

proc sort*[S: SomeSpectrum](h: var HittablesList[S], start, stop: int, cmpH: proc(x, y: Hittable[S]): int,
           order = SortOrder.Ascending) =
  h.data.toOpenArray(start, stop-1).sort(cmpH, order = order)

proc add*[S: SomeSpectrum](h: var HittablesList[S], el: Hittable[S]) =
  ## adds a new element to h. If space is there
  h.data.add el
  inc h.len

# forward declaration
proc toSpectrum*[S: SomeSpectrum; T: SomeSpectrum](h: Hittable[S], _: typedesc[T]): Hittable[T]

proc toSpectrum*[S: SomeSpectrum; T: SomeSpectrum](x: BvhNode[S], _: typedesc[T]): BvhNode[T] =
  result = BvhNode[T](left: toSpectrum(x.left, T), right: toSpectrum(x.right, T), box: x.box)

proc toSpectrum*[S: SomeSpectrum; T: SomeSpectrum](x: Box[S], _: typedesc[T]): Box[T] =
  result = Box[T](boxMin: x.boxMin, boxMax: x.boxMax,
                  sides: toSpectrum(x.sides, T))

## Two annoying helper templates, because now the Nim compiler doesn't allow constructing a
## variant object from a runtime value anymore, if the object has default values...
template initIt(it, it2: untyped): untyped =
  result = Hittable[T](mat: toSpectrum(h.mat, T),
                       trans: h.trans,
                       invTrans: h.invTrans,
                       kind: it, it2: h.it2)
template initItVal(it, it2, val: untyped): untyped  =
  result = Hittable[T](mat: toSpectrum(h.mat, T),
                       trans: h.trans,
                       invTrans: h.invTrans,
                       kind: it, it2: val)

proc toSpectrum*[S: SomeSpectrum; T: SomeSpectrum](h: Hittable[S], _: typedesc[T]): Hittable[T] =

  case h.kind
  of htSphere:      initIt(htSphere, hSphere)
  of htCylinder:    initIt(htCylinder, hCylinder)
  of htCone:        initIt(htCone, hCone)
  of htParaboloid:  initIt(htParaboloid, hParaboloid)
  of htHyperboloid: initIt(htHyperboloid, hHyperboloid)
  of htBvhNode:     initItVal(htBvhNode, hBvhNode, toSpectrum(h.hBvhNode, T))
  of htXyRect:      initIt(htXyRect, hXyRect)
  of htXzRect:      initIt(htXzRect, hXzRect)
  of htYzRect:      initIt(htYzRect, hYzRect)
  of htBox:         initItVal(htBox, hBox, toSpectrum(h.hBox, T))
  of htDisk:        initIt(htDisk, hDisk)

proc toSpectrum*[S: SomeSpectrum; T: SomeSpectrum](h: HittablesList[S], _: typedesc[T]): HittablesList[T] =
  result = initHittables[T](h.len)
  for x in h:
    result.add toSpectrum(x, T)

proc clone*[S: SomeSpectrum](h: Hittable[S]): Hittable[S] = toSpectrum(h, S)

proc clone*[S: SomeSpectrum](h: HittablesList[S]): HittablesList[S] =
  result = initHittables[S](h.len)
  for x in h:
    result.add clone(x)

proc toHittable*[S: SomeSpectrum](s: Sphere, mat: Material[S]): Hittable[S]      = result = Hittable[S](kind: htSphere, hSphere: s, mat: mat)
proc toHittable*[S: SomeSpectrum](c: Cylinder, mat: Material[S]): Hittable[S]    = result = Hittable[S](kind: htCylinder, hCylinder: c, mat: mat)
proc toHittable*[S: SomeSpectrum](c: Cone, mat: Material[S]): Hittable[S]        = result = Hittable[S](kind: htCone, hCone: c, mat: mat)
proc toHittable*[S: SomeSpectrum](p: Paraboloid, mat: Material[S]): Hittable[S]  = result = Hittable[S](kind: htParaboloid, hParaboloid: p, mat: mat)
proc toHittable*[S: SomeSpectrum](h: Hyperboloid, mat: Material[S]): Hittable[S] = result = Hittable[S](kind: htHyperboloid, hHyperboloid: h, mat: mat)
proc toHittable*[S: SomeSpectrum](d: Disk, mat: Material[S]): Hittable[S]        = result = Hittable[S](kind: htDisk, hDisk: d, mat: mat)
proc toHittable*[S: SomeSpectrum](b: BvhNode[S]): Hittable[S]                    = result = Hittable[S](kind: htBvhNode, hBvhNode: b)
proc toHittable*[S: SomeSpectrum](r: XyRect, mat: Material[S]): Hittable[S]      = result = Hittable[S](kind: htXyRect, hXyRect: r, mat: mat)
proc toHittable*[S: SomeSpectrum](r: XzRect, mat: Material[S]): Hittable[S]      = result = Hittable[S](kind: htXzRect, hXzRect: r, mat: mat)
proc toHittable*[S: SomeSpectrum](r: YzRect, mat: Material[S]): Hittable[S]      = result = Hittable[S](kind: htYzRect, hYzRect: r, mat: mat)
proc toHittable*[S: SomeSpectrum; T: SomeSpectrum](b: Box[S], mat: Material[T]): Hittable[T] =
  ## If different material spectrum than box, convert the box
  result = Hittable[T](kind: htBox, hBox: toSpectrum(b, T), mat: mat)

#proc add*[T: AnyHittable](h: var HittablesList[S], ht: T) = h.add toHittable(ht)
proc add*[S: SomeSpectrum](h: var HittablesList[S], lst: HittablesList[S]) =
  for x in lst:
    h.add x

proc delete*[S: SomeSpectrum](h: var HittablesList[S], ht: Hittable[S]) =
  ## Deletes `ht` from `h` if it exists
  let idx = h.data.find(ht)
  if idx > 0:
    h.data.delete(idx)
    dec h.len

proc initBox*(p0, p1: Point): Box[RGBSpectrum] =
  result.boxMin = p0
  result.boxMax = p1

  ## NOTE: We reuse `RGBSpectrum` but that is only because we don't care about materials
  ## in the box! The material we care about is of the `Hittable` that *contains* the Box.
  result.sides = initHittables[RGBSpectrum](0)

  let mat = initMaterial(initLambertian(color(0,0,0))) # dummy
  result.sides.add toHittable(initXyRect(p0.x, p1.x, p0.y, p1.y, p1.z), mat)
  result.sides.add toHittable(initXyRect(p0.x, p1.x, p0.y, p1.y, p0.z), mat)

  result.sides.add toHittable(initXzRect(p0.x, p1.x, p0.z, p1.z, p1.y), mat)
  result.sides.add toHittable(initXzRect(p0.x, p1.x, p0.z, p1.z, p0.y), mat)

  result.sides.add toHittable(initYzRect(p0.y, p1.y, p0.z, p1.z, p1.x), mat)
  result.sides.add toHittable(initYzRect(p0.y, p1.y, p0.z, p1.z, p0.x), mat)

proc initGenericHittables*(): GenericHittablesList =
  result = GenericHittablesList(rgbs: initHittables[RGBSpectrum](),
                                xray: initHittables[XraySpectrum]())

proc len*(h: GenericHittablesList): int = h.rgbs.len + h.xray.len

proc clone*(x: GenericHittablesList): GenericHittablesList =
  result = GenericHittablesList(rgbs: x.rgbs.clone(),
                                xray: x.xray.clone())

proc cloneAsRGB*(x: GenericHittablesList): HittablesList[RGBSpectrum] =
  ## Extract all elements from both lists and returns all elements as `RGBSpectrum`
  result = initHittables[RGBSpectrum](x.len)
  for el in x.rgbs:
    result.add el.clone()
  for el in x.xray:
    result.add toSpectrum(el, RGBSpectrum) # already is a clone operation

proc cloneAsXray*(x: GenericHittablesList): HittablesList[XraySpectrum] =
  ## Extract all elements from both lists and returns all elements as `RGBSpectrum`
  result = initHittables[XraySpectrum](x.len)
  for el in x.rgbs:
    result.add toSpectrum(el, XraySpectrum) # already is a clone operation
  for el in x.xray:
    result.add el.clone()

proc add*[S: SomeSpectrum](x: var GenericHittablesList, val: Hittable[S] | HittablesList[S]) =
  when S is RGBSpectrum:
    x.rgbs.add val
  else:
    x.xray.add val

proc add*(x: var GenericHittablesList, h: GenericHittablesList) =
  x.add h.rgbs
  x.add h.xray

proc setFaceNormal*(rec: var HitRecord, r: Ray, outward_normal: Vec3d) =
  rec.frontFace = r.dir.dot(outward_normal) < 0
  rec.normal = if rec.frontFace: outward_normal else: -outward_normal

proc hit*(s: Sphere, r: Ray, tMin, tMax: float, rec: var HitRecord): bool =
  let oc = r.orig

  let a = r.dir.length_squared()
  let half_b = oc.Vec3d.dot(r.dir)
  let c = oc.length_squared() - s.radius * s.radius

  let discriminant = half_b * half_b - a * c
  if discriminant < 0:
    return false
  let sqrtd = sqrt discriminant

  # find nearest root that lies in acceptable range
  var root = (-half_b - sqrtd) / a
  if root < tMin or tMax < root:
    root = (-half_b + sqrtd) / a
    if root < tMin or tMax < root:
      return false

  rec.t = root
  rec.p = r.at(rec.t)
  let outward_normal = (rec.p) / s.radius

  let theta = arccos(-outward_normal.y)
  let phi = arctan2(-outward_normal.z, outward_normal.x) + PI
  rec.u = phi / (2*PI)
  rec.v = theta / PI
  rec.setFaceNormal(r, outward_normal.Vec3d)

  result = true

proc hit*(cyl: Cylinder, r: Ray, tMin, tMax: float, rec: var HitRecord): bool =
  ## Initialize ray coordinate values
  let a = r.dir.x * r.dir.x + r.dir.y * r.dir.y
  let b = 2.0 * (r.dir.x * r.orig.x + r.dir.y * r.orig.y)
  let c = r.orig.x * r.orig.x + r.orig.y * r.orig.y - cyl.radius * cyl.radius

  var
    t0: float
    t1: float
  if not solveQuadratic(a, b, c, t0, t1): return false

  # Check quadric shape _t0_ and _t1_ for nearest intersection
  if t0 > tMax or t1 <= tMin: return false
  var tShapeHit = t0
  ## XXX: This should be `tMin` no?
  if tShapeHit <= tMin:
    tShapeHit = t1
    if tShapeHit > tMax: return false

  # Compute cylinder hit point and $\phi$
  var pHit = r.at(tShapeHit)

  # Refine cylinder intersection point
  let hitRad = sqrt(pHit.x * pHit.x + pHit.y * pHit.y)
  pHit.x = pHit.x * cyl.radius / hitRad
  pHit.y = pHit.y * cyl.radius / hitRad
  var phi = arctan2(pHit.y, pHit.x)
  if phi < 0: phi += 2 * Pi

  # Test cylinder intersection against clipping parameters
  if pHit.z < cyl.zMin or pHit.z > cyl.zMax or phi > cyl.phiMax:
      if tShapeHit == t1: return false
      tShapeHit = t1
      if t1 > tMax: return false
      # Compute cylinder hit point and $\phi$
      pHit = r.at(tShapeHit)

      # Refine cylinder intersection point
      let hitRad = sqrt(pHit.x * pHit.x + pHit.y * pHit.y)
      pHit.x = pHit.x * cyl.radius / hitRad
      pHit.y = pHit.y * cyl.radius / hitRad
      phi = arctan2(pHit.y, pHit.x)
      if phi < 0: phi += 2 * Pi
      if pHit.z < cyl.zMin or pHit.z > cyl.zMax or phi > cyl.phiMax: return false

  # Find parametric representation of cylinder hit
  let u = phi / cyl.phiMax
  let v = (pHit.z - cyl.zMin) / (cyl.zMax - cyl.zMin)

  # Compute cylinder $\dpdu$ and $\dpdv$
  let dpdu = vec3(-cyl.phiMax * pHit.y, cyl.phiMax * pHit.x, 0.0)
  let dpdv = vec3(0.0, 0.0, cyl.zMax - cyl.zMin)

  rec.u = u
  rec.v = v

  rec.t = tShapeHit
  rec.p = r.at(rec.t)
  let outward_normal = normalize(cross(dpdu, dpdv))
  rec.setFaceNormal(r, outward_normal.Vec3d)

  result = true

proc hit*(con: Cone, r: Ray, tMin, tMax: float, rec: var HitRecord): bool =
  ## Initialize ray coordinate values
  var k = (con.radius / con.height)^2
  let a = r.dir.x * r.dir.x + r.dir.y * r.dir.y - k * r.dir.z * r.dir.z
  let b = 2 * (r.dir.x * r.orig.x + r.dir.y * r.orig.y - k * r.dir.z * (r.orig.z - con.height))
  let c = r.orig.x * r.orig.x + r.orig.y * r.orig.y - k * (r.orig.z - con.height) * (r.orig.z - con.height)

  var
    t0: float
    t1: float
  if not solveQuadratic(a, b, c, t0, t1): return false

  # Check quadric shape _t0_ and _t1_ for nearest intersection
  if t0 > tMax or t1 <= tMin: return false
  var tShapeHit = t0
  ## XXX: This should be `tMin` no?
  if tShapeHit <= tMin:
    tShapeHit = t1
    if tShapeHit > tMax: return false

  # Compute cone inverse mapping
  var pHit = r.at(tShapeHit)
  var phi = arctan2(pHit.y, pHit.x)
  if phi < 0.0: phi += 2.0 * PI

  # Test cone intersection against clipping parameters
  if pHit.z < 0 or pHit.z > con.zMax or phi > con.phiMax:
    if tShapeHit == t1: return false
    tShapeHit = t1
    if t1 > tMax: return false
    # Compute cone inverse mapping
    pHit = r.at(tShapeHit)
    phi = arctan2(pHit.y, pHit.x)
    if phi < 0.0: phi += 2 * Pi
    if pHit.z < 0.0 or pHit.z > con.zMax or phi > con.phiMax: return false

  # Find parametric representation of cylinder hit
  let u = phi / con.phiMax
  let v = (pHit.z) / (con.height)

  # Compute cone $\dpdu$ and $\dpdv$
  let dpdu = vec3(-con.phiMax * pHit.y, con.phiMax * pHit.x, 0.0)
  let dpdv = vec3(-pHit.x / (1.0 - v), -pHit.y / (1.0 - v), con.height)
  rec.u = u
  rec.v = v

  rec.t = tShapeHit
  rec.p = r.at(rec.t)
  let outward_normal = normalize(cross(dpdu, dpdv))
  rec.setFaceNormal(r, outward_normal.Vec3d)

  result = true

proc hit*(par: Paraboloid, r: Ray, tMin, tMax: float, rec: var HitRecord): bool =
  ## Based on `pbrt`.

  # Initialize _EFloat_ ray coordinate values
  let k = par.zMax / (par.radius * par.radius)
  let a = k * (r.dir.x * r.dir.x + r.dir.y * r.dir.y)
  let b = 2 * k * (r.dir.x * r.orig.x + r.dir.y * r.orig.y) - r.dir.z
  let c = k * (r.orig.x * r.orig.x + r.orig.y * r.orig.y) - r.orig.z

  # Solve quadratic equation for _t_ values
  var
    t0: float
    t1: float
  if not solveQuadratic(a, b, c, t0, t1): return false

  # Check quadric shape _t0_ and _t1_ for nearest intersection
  if t0 > tMax or t1 <= 0: return false
  var tShapeHit = t0
  if tShapeHit <= 0:
    tShapeHit = t1
    if tShapeHit > tMax: return false


  # Compute paraboloid inverse mapping
  var pHit = r.at(tShapeHit)
  var phi = arctan2(pHit.y, pHit.x)
  if phi < 0: phi += 2 * PI

  # Test paraboloid intersection against clipping parameters
  if pHit.z < par.zMin or pHit.z > par.zMax or phi > par.phiMax:
    if tShapeHit == t1: return false
    tShapeHit = t1
    if t1 > tMax: return false
    # Compute paraboloid inverse mapping
    pHit = r.at(tShapeHit)
    phi = arctan2(pHit.y, pHit.x)
    if phi < 0: phi += 2 * PI
    if pHit.z < par.zMin or pHit.z > par.zMax or phi > par.phiMax: return false

  # Find parametric representation of paraboloid hit
  let u = phi / par.phiMax
  let v = (pHit.z - par.zMin) / (par.zMax - par.zMin)

  # Compute paraboloid $\dpdu$ and $\dpdv$
  let dpdu = vec3(-par.phiMax * pHit.y, par.phiMax * pHit.x, 0.0)
  let dpdv = (par.zMax - par.zMin) * vec3(pHit.x / (2 * pHit.z), pHit.y / (2 * pHit.z), 1.0)

  rec.t = tShapeHit
  rec.p = r.at(rec.t)
  let outward_normal = normalize(cross(dpdu, dpdv))
  rec.setFaceNormal(r, outward_normal.Vec3d)

  rec.u = u
  rec.v = v

  result = true

proc hit*(hyp: Hyperboloid, r: Ray, tMin, tMax: float, rec: var HitRecord): bool =
  ## Based on `pbrt`.
  # Initialize _EFloat_ ray coordinate values
  let a = hyp.ah * r.dir.x * r.dir.x + hyp.ah * r.dir.y * r.dir.y - hyp.ch * r.dir.z * r.dir.z
  let b = 2.0 * (hyp.ah * r.dir.x * r.orig.x + hyp.ah * r.dir.y * r.orig.y - hyp.ch * r.dir.z * r.orig.z)
  let c = hyp.ah * r.orig.x * r.orig.x + hyp.ah * r.orig.y * r.orig.y - hyp.ch * r.orig.z * r.orig.z - 1.0

  # Solve quadratic equation for _t_ values
  var
    t0: float
    t1: float
  if not solveQuadratic(a, b, c, t0, t1): return false

  # Check quadric shape _t0_ and _t1_ for nearest intersection
  if t0 > tMax or t1 <= 0: return false
  var tShapeHit = t0
  if tShapeHit <= 0:
    tShapeHit = t1
    if tShapeHit > tMax: return false

  # Compute hyperboloid inverse mapping
  var pHit = r.at(tShapeHit)
  var v = (pHit.z - hyp.p1.z) / (hyp.p2.z - hyp.p1.z)
  let pr = (1 - v) * hyp.p1 + v * hyp.p2
  var phi = arctan2(pr.x * pHit.y - pHit.x * pr.y,
                    pHit.x * pr.x + pHit.y * pr.y)
  if phi < 0: phi += 2 * PI

  # Test hyperboloid intersection against clipping parameters
  if pHit.z < hyp.zMin or pHit.z > hyp.zMax or phi > hyp.phiMax:
    if tShapeHit == t1: return false
    tShapeHit = t1
    if t1 > tMax: return false
    # Compute hyperboloid inverse mapping
    pHit = r.at(tShapeHit)
    v = (pHit.z - hyp.p1.z) / (hyp.p2.z - hyp.p1.z)
    let pr = (1 - v) * hyp.p1 + v * hyp.p2
    phi = arctan2(pr.x * pHit.y - pHit.x * pr.y,
             pHit.x * pr.x + pHit.y * pr.y)
    if phi < 0: phi += 2 * Pi
    if pHit.z < hyp.zMin or pHit.z > hyp.zMax or phi > hyp.phiMax: return false


  # Compute parametric representation of hyperboloid hit
  let u = phi / hyp.phiMax

  # Compute hyperboloid $\dpdu$ and $\dpdv$
  let
    cosPhi = cos(phi)
    sinPhi = sin(phi)
  let dpdu = vec3(-hyp.phiMax * pHit.y, hyp.phiMax * pHit.x, 0.0)
  let dpdv = vec3((hyp.p2.x - hyp.p1.x) * cosPhi - (hyp.p2.y - hyp.p1.y) * sinPhi,
                  (hyp.p2.x - hyp.p1.x) * sinPhi + (hyp.p2.y - hyp.p1.y) * cosPhi,
                  hyp.p2.z - hyp.p1.z)

  rec.t = tShapeHit
  rec.p = r.at(rec.t)
  let outward_normal = normalize(cross(dpdu, dpdv))
  rec.setFaceNormal(r, outward_normal.Vec3d)

  rec.u = u
  rec.v = v

  result = true

proc hit*(d: Disk, r: Ray, tMin, tMax: float, rec: var HitRecord): bool =
  let t = (d.distance - r.orig.z) / r.dir.z
  if t < tMin or t > tMax: return false
  if r.dir.z == 0.0:
    # ray is parallel to disk
    return false
  var pHit = r.at(t)
  let dist = pHit.x * pHit.x + pHit.y * pHit.y

  if dist > (d.radius * d.radius) or dist < (d.innerRadius * d.innerRadius): return false

  # Test disk $\phi$ value against $\phimax$
  var phi = arctan2(pHit.y, pHit.x)
  if phi < 0: phi += 2 * PI
  if phi > d.phiMax: return false

  # Find parametric representation of disk hit
  let u = phi / d.phiMax
  let rHit = sqrt(dist)
  let v = (d.radius - rHit) / (d.radius - d.innerRadius)

  ## XXX: compute normal from dpdu x dpdv?
  let dpdu = vec3(-d.phiMax * pHit.y, d.phiMax * pHit.x, 0)
  let dpdv = vec3(pHit.x, pHit.y, 0.0) * (d.innerRadius - d.radius) / rHit

  # Refine disk intersection point
  ## XXX?
  # pHit.z = d.distance

  rec.t = t
  rec.p = r.at(rec.t)
  let outward_normal = normalize(cross(dpdu, dpdv))#vec3(0.0, 0.0, 1.0)
  rec.setFaceNormal(r, outward_normal.Vec3d)

  rec.u = u
  rec.v = v

  result = true

proc hit*(rect: XyRect, r: Ray, tMin, tMax: float, rec: var HitRecord): bool =
  let t = (rect.k - r.orig.z) / r.dir.z
  if t < tMin or t > tMax:
    return false
  let x = r.orig.x + t * r.dir.x
  let y = r.orig.y + t * r.dir.y
  if x < rect.x0 or x > rect.x1 or y < rect.y0 or y > rect.y1:
    return false
  rec.u = (x - rect.x0) / (rect.x1 - rect.x0)
  rec.v = (y - rect.y0) / (rect.y1 - rect.y0)
  rec.t = t
  let outward_normal = vec3(0.0, 0.0, 1.0)
  rec.setFaceNormal(r, outward_normal)
  rec.p = r.at(t)
  result = true

proc hit*(rect: XzRect, r: Ray, tMin, tMax: float, rec: var HitRecord): bool =
  let t = (rect.k - r.orig.y) / r.dir.y
  if t < tMin or t > tMax:
    return false
  let x = r.orig.x + t * r.dir.x
  let z = r.orig.z + t * r.dir.z
  if x < rect.x0 or x > rect.x1 or z < rect.z0 or z > rect.z1:
    return false
  rec.u = (x - rect.x0) / (rect.x1 - rect.x0)
  rec.v = (z - rect.z0) / (rect.z1 - rect.z0)
  rec.t = t
  let outward_normal = vec3(0.0, 1.0, 0.0)
  rec.setFaceNormal(r, outward_normal)
  rec.p = r.at(t)
  result = true

proc hit*(rect: YzRect, r: Ray, tMin, tMax: float, rec: var HitRecord): bool =
  let t = (rect.k - r.orig.x) / r.dir.x
  if t < tMin or t > tMax:
    return false
  let y = r.orig.y + t * r.dir.y
  let z = r.orig.z + t * r.dir.z
  if y < rect.y0 or y > rect.y1 or z < rect.z0 or z > rect.z1:
    return false
  rec.u = (y - rect.y0) / (rect.y1 - rect.y0)
  rec.v = (z - rect.z0) / (rect.z1 - rect.z0)
  rec.t = t
  let outward_normal = vec3(1.0, 0.0, 0.0)
  rec.setFaceNormal(r, outward_normal)
  rec.p = r.at(t)
  result = true

proc hit*[S: SomeSpectrum](h: Hittable[S], r: Ray, tMin, tMax: float, rec: var HitRecord[S]): bool {.gcsafe.}
proc hit*(n: BvhNode, r: Ray, tMin, tMax: float, rec: var HitRecord): bool =
  if not n.box.hit(r, tMin, tMax):
    return false

  let hitLeft = n.left.hit(r, tMin, tMax, rec)
  let hitRight = n.right.hit(r, tMin, if hitLeft: rec.t else: tMax, rec)

  result = hitLeft or hitRight

proc hit*[S: SomeSpectrum](h: HittablesList[S], r: Ray, tMin, tMax: float, rec: var HitRecord[S]): bool {.gcsafe.} =
  var tmpRec: HitRecord[S]
  result = false
  var closestSoFar = tMax

  for obj in h:
    if obj.hit(r, tMin, closestSoFar, tmpRec):
      result = true
      closestSoFar = tmpRec.t
      rec = tmpRec

proc hit*[S: SomeSpectrum](box: Box, r: Ray, tMin, tMax: float, rec: var HitRecord[S]): bool {.gcsafe.} =
  result = box.sides.hit(r, tMin, tMax, rec)

template transforms(name, field: untyped): untyped =
  proc `name`*[S: SomeSpectrum](h: Hittable[S], v: Vec3d): Vec3d =
    ## Apply the world to object transformation for the given vector.
    # calculate transformed vector
    let vt = h.field * vec4d(v, w = 0) ## For vectors weight is always 1!
    # construct result vector. The weight row is irrelevant!
    result = vec3(vt.x, vt.y, vt.z)

  proc `name`*[S: SomeSpectrum](h: Hittable[S], p: Point): Point =
    ## Apply the world to object transformation for the given vector.
    # calculate transformed vector
    let vt = h.field * vec4d(p.Vec3d, w = 1) ## XXX: For points we *CURRENTLY* just assume weight 1
    # construct result Point
    result = Point(vec3(vt.x, vt.y, vt.z) / vt.w)

  proc `name`*[S: SomeSpectrum](h: Hittable[S], r: Ray): Ray =
    result = Ray(orig: h.name(r.orig), dir: h.name(r.dir))

transforms(transform, trans)
transforms(inverseTransform, invTrans)

#proc transTransform*[S: SomeSpectrum](h: Hittable[S], v: Vec3d): Vec3d =
#  var mh = h.clone()
#  mh.trans = h.trans.transpose()
#  result = mh.transform(v)

proc invertNormal*[S: SomeSpectrum](h: Hittable[S], n: Vec3d): Vec3d =
  #var mh = h.clone()
  ### XXX: NOTE: if I'm not mistaken `pbrt` uses `inverse().transpose()` here.
  ### But if we do that the reflections on metals break if we use rotations.
  #mh.trans = h.trans.inverse() #.transpose()
  result = h.inverseTransform(n)  #mh.transform(n)

proc hit*[S: SomeSpectrum](h: Hittable[S], r: Ray, tMin, tMax: float, rec: var HitRecord[S]): bool {.gcsafe.} =
  # 1. transform to object space
  let rOb = h.transform(r)
  # 2. compute the hit
  case h.kind
  of htSphere:      result = h.hSphere.hit(rOb, tMin, tMax, rec)
  of htCylinder:    result = h.hCylinder.hit(rOb, tMin, tMax, rec)
  of htCone:        result = h.hCone.hit(rOb, tMin, tMax, rec)
  of htParaboloid:  result = h.hParaboloid.hit(rOb, tMin, tMax, rec)
  of htHyperboloid: result = h.hHyperboloid.hit(rOb, tMin, tMax, rec)
  of htDisk:        result = h.hDisk.hit(rOb, tMin, tMax, rec)
  of htBvhNode:     result = h.hBvhNode.hit(rOb, tMin, tMax, rec)
  of htXyRect:      result = h.hXyRect.hit(rOb, tMin, tMax, rec)
  of htXzRect:      result = h.hXzRect.hit(rOb, tMin, tMax, rec)
  of htYzRect:      result = h.hYzRect.hit(rOb, tMin, tMax, rec)
  of htBox:         result = h.hBox.hit(rOb, tMin, tMax, rec)
  #else: discard
  # 3. assign material
  rec.mat = h.mat
  # 4. convert rec back to world space
  ## XXX: normal transformation in general more complex!
  ## `pbrt` uses `mInv` for *FORWARD* transformation!
  rec.normal = normalize(h.invertNormal(rec.normal))
  rec.p = h.inverseTransform(rec.p)

proc boundingBox*[S: SomeSpectrum](n: BvhNode[S], outputBox: var AABB): bool =
  outputBox = n.box
  result = true

proc boundingBox*(b: Box, outputBox: var AABB): bool =
  outputBox = initAabb(b.boxMin, b.boxMax)
  result = true

proc boundingBox*[S: SomeSpectrum](h: Hittable[S], output_box: var AABB): bool =
  case h.kind
  of htSphere:      result = h.hSphere.boundingBox(output_box)
  of htCylinder:    result = h.hCylinder.boundingBox(output_box)
  of htCone:        result = h.hCone.boundingBox(output_box)
  of htParaboloid:  result = h.hParaboloid.boundingBox(output_box)
  of htHyperboloid: result = h.hHyperboloid.boundingBox(output_box)
  of htDisk:        result = h.hDisk.boundingBox(output_box)
  of htBvhNode:     result = h.hBvhNode.boundingBox(output_box)
  of htXyRect:      result = h.hXyRect.boundingBox(output_box)
  of htXzRect:      result = h.hXzRect.boundingBox(output_box)
  of htYzRect:      result = h.hYzRect.boundingBox(output_box)
  of htBox:         result = h.hBox.boundingBox(output_box)

proc boundingBox*[S: SomeSpectrum](h: HittablesList[S], output_box: var AABB): bool =
  if h.len == 0:
    return false

  var tmpBox: AABB
  var firstBox = true

  for obj in h:
    if not obj.boundingBox(tmpBox):
      return false
    output_box = if firstBox: tmpBox else: surroundingBox(output_box, tmpBox)
    firstBox = false
  result = true

proc box_compare[S: SomeSpectrum](a, b: Hittable[S], axis: int): int {.inline.} =
  var boxA: AABB
  var boxB: AABB

  if not a.boundingBox(boxA) or not b.boundingBox(boxB):
    stderr.write("No bounding box in BVH node constructor!\n")

  result = if boxA.minimum[axis] < boxB.minimum[axis]: 1
           elif boxA.minimum[axis] == boxB.minimum[axis]: 0
           else: -1

proc box_x_compare[S: SomeSpectrum](a, b: Hittable[S]): int =
  result = boxCompare(a, b, 0)

proc box_y_compare[S: SomeSpectrum](a, b: Hittable[S]): int =
  result = boxCompare(a, b, 1)

proc box_z_compare[S: SomeSpectrum](a, b: Hittable[S]): int =
  result = boxCompare(a, b, 2)

proc initBvhNode*[S: SomeSpectrum](rnd: var Rand, list: HittablesList[S], start, stop: int): BvhNode[S] =
  var mlist = list

  let axis = rnd.rand(2)
  var comparator: (proc(a, b: Hittable[S]): int)
  case axis
  of 0: comparator = box_x_compare
  of 1: comparator = box_y_compare
  of 2: comparator = box_z_compare
  else: doAssert false, "Invalid int in range 0,2"

  let objSpan = stop - start
  if objSpan == 1:
    result.left = list[start]
    result.right = list[start]
  elif objSpan == 2:
    if comparator(list[start], list[start+1]) >= 0:
      result.left = list[start]
      result.right = list[start+1]
    else:
      result.left = list[start+1]
      result.right = list[start]
  else:
    mlist.sort(start, stop, comparator)

    let mid = start + objSpan div 2
    result.left = Hittable[S](kind: htBvhNode, hBvhNode: rnd.initBvhNode(mlist, start, mid))
    result.right = Hittable[S](kind: htBvhNode, hBvhNode: rnd.initBvhNode(mlist, mid, stop))

  var boxLeft: AABB
  var boxRight: AABB

  if not result.left.boundingBox(boxLeft) or
     not result.right.boundingBox(boxRight):
    stderr.write("No bounding box in BVH node constructor!\n")

  result.box = surroundingBox(boxLeft, boxRight)

template rotations(name: untyped): untyped =
  proc `name`*[S: SomeSpectrum](h: Hittable[S], angle: float): Hittable[S] =
    result = h.clone()
    result.trans = `name`(h.trans, angle.degToRad)
    result.invTrans = result.trans.inverse()
  proc `name`*[S: SomeSpectrum](h: HittablesList[S], angle: float): HittablesList[S] =
    result = initHittables[S](h.len)
    for x in h:
      result.add name(x, angle)
  proc `name`*(h: GenericHittablesList, angle: float): GenericHittablesList =
    result = initGenericHittables()
    for x in h.rgbs:
      result.add name(x, angle)
    for x in h.xray:
      result.add name(x, angle)
rotations(rotateX)
rotations(rotateY)
rotations(rotateZ)

#proc translate*[T: AnyHittable; V: Vec3d | Point](h: T, v: V): Hittable[S] =
#  result = h.toHittable()
#  result.trans = translate(mat4d(), -v.Vec3d)
#  result.invTrans = result.trans.inverse()
#proc translate*[V: Vec3d | Point; T: AnyHittable](v: V, h: T): Hittable[S] = h.translate(v)

proc translate*[S: SomeSpectrum; V: Vec3d | Point](h: Hittable[S], v: V): Hittable[S] =
  result = h.clone()
  result.trans = translate(h.trans, -v.Vec3d)
  result.invTrans = result.trans.inverse()
proc translate*[S: SomeSpectrum; V: Vec3d | Point](v: V, h: Hittable[S]): Hittable[S] = h.translate(v.Vec3d)

proc translate*[S: SomeSpectrum](h: HittablesList[S], v: Vec3d | Point): HittablesList[S] =
  result = initHittables[S](h.len)
  for x in h:
    result.add translate(x, v)

proc translate*(h: GenericHittablesList, v: Vec3d | Point): GenericHittablesList =
  result = initGenericHittables()
  result.add h.rgbs.translate(v)
  result.add h.xray.translate(v)

## WARNING: This must not be a function! We risk the code *copying* the material, which is extremely
## expensive for `SolarEmission`!
template getMaterial*[S: SomeSpectrum](h: Hittable[S]): Material[S] = h.mat

proc getHittablesOfKind*[S: SomeSpectrum](h: HittablesList[S], kinds: set[MaterialKind]): HittablesList[S] =
  ## Returns all hittables of materials of the given kinds from the given list as a list itself.
  result = initHittables[S]()
  for x in h:
    let mat = x.getMaterial()
    if mat.kind in kinds: result.add x

proc getImageSensors*[S: SomeSpectrum](h: HittablesList[S]): HittablesList[S] =
  ## Returns all image sensors from the given list as a list itself.
  result = getHittablesOfKind(h, {mkImageSensor})

proc getImageSensors*(h: GenericHittablesList): HittablesList[XraySpectrum] =
  result = initHittables[XraySpectrum]()
  result.add getImageSensors(toSpectrum(h.rgbs, XraySpectrum))
  result.add getImageSensors(h.xray)

proc getLightTargets*[S: SomeSpectrum](h: var HittablesList[S], delete: bool): HittablesList[S] =
  ## Returns all light targets from the given list as a list itself.
  result = getHittablesOfKind(h, {mkLightTarget})
  if delete:
    for x in result:
      h.delete(x)

proc removeInvisibleTargets*[S: SomeSpectrum](h: var HittablesList[S]) =
  let lt = getLightTargets(h, delete = false)
  for x in lt:
    if not x.getMaterial.mLightTarget.visible:
      h.delete(x)

proc getSources*[S: SomeSpectrum](h: var HittablesList[S], delete: bool): HittablesList[S] {.gcsafe.} =
  ## Returns all light sources from the given list as a list itself.
  ## These are removed from the input.
  {.cast(gcsafe).}:
    result = getHittablesOfKind(h, {mkDiffuseLight, mkLaser, mkSolarEmission})
    if delete:
      for x in result:
        h.delete(x)

proc getRandomPointFromSolarModel(radius: float,
                                  fluxRadiusCDF: seq[float],
                                  radii: seq[float],
                                  rnd: var Rand): Point =
  ## This function gives the coordinates of a random point in the sun, biased
  ## by the emissionrates (which depend on the radius and the energy) ##
  ##
  ## `fluxRadiusCDF` is the normalized (to 1.0) cumulative sum of the total flux per
  ## radius of all radii of the solar model.
  let
    randEmRate = rnd.rand(1.0)
    rIdx = fluxRadiusCDF.lowerBound(randEmRate)
    r = radii[rIdx] * radius
    r2 = (0.0015 + (rIdx).float * 0.0005) * radius # in mm
  doAssert abs(r2 - r) < 1e-3, "r2 = " & $r2 & ", r = " & $r
  let p = rnd.randomInUnitSphere() * r
  result = Point(p)

proc samplePoint*[S: SomeSpectrum](h: Hittable[S], rnd: var Rand): Point {.gcsafe.}
proc samplePoint*(s: Sphere, rnd: var Rand, mat: Material): Point {.gcsafe.} =
  ## Samples a random point on the sphere surface
  case mat.kind
  of mkSolarEmission:
    ## Use the given emission properties of the solar model
    let m = mat.mSolarEmission
    result = getRandomPointFromSolarModel(s.radius, m.fluxRadiusCDF, m.radii, rnd)
  else:
    ## Sample uniformly from the sphere
    let p = rnd.randomInUnitSphere()
    result = Point(p * s.radius)

proc samplePoint*(d: Disk, rnd: var Rand): Point {.gcsafe.} =
  ## Samples a random point on the disk surface
  result = Point(rnd.randomInUnitDisk() * d.radius)

proc samplePoint*(c: Cylinder, rnd: var Rand): Point {.gcsafe.} =
  ## Samples a random point on the cylinder surface
  # cylinder is aligned along z
  doAssert false, "Not yet implemented."

proc samplePoint*(c: Cone, rnd: var Rand): Point {.gcsafe.} =
  ## Samples a random point on the cone surface
  doAssert false, "Not yet implemented."

proc samplePoint*(c: Paraboloid, rnd: var Rand): Point {.gcsafe.} =
  ## Samples a random point on the paraboloid surface
  doAssert false, "Not yet implemented."

proc samplePoint*(c: Hyperboloid, rnd: var Rand): Point {.gcsafe.} =
  ## Samples a random point on the hyperboloid surface
  doAssert false, "Not yet implemented."

proc samplePoint*(r: XyRect, rnd: var Rand): Point {.gcsafe.} =
  ## Samples a random point on the rect surface surface
  let x = rnd.rand(r.x0 .. r.x1)
  let y = rnd.rand(r.y0 .. r.y1)
  ## XXX: Handle k
  result = point(x, y, 0)

proc samplePoint*(r: XzRect, rnd: var Rand): Point {.gcsafe.} =
  ## Samples a random point on the rect surface surface
  let x = rnd.rand(r.x0 .. r.x1)
  let z = rnd.rand(r.z0 .. r.z1)
  ## XXX: Handle k
  result = point(x, 0, z)

proc samplePoint*(r: YzRect, rnd: var Rand): Point {.gcsafe.} =
  ## Samples a random point on the rect surface surface
  let y = rnd.rand(r.y0 .. r.y1)
  let z = rnd.rand(r.z0 .. r.z1)
  ## XXX: Handle k
  result = point(0, y, z)

proc samplePoint*(b: Box, rnd: var Rand): Point {.gcsafe.} =
  ## Samples a random point on the rect surface surface
  let side = rnd.rand(b.sides.len - 1)
  ## XXX: Correct by point p0, p1!
  result = samplePoint(b.sides[side], rnd)

proc samplePoint*[S: SomeSpectrum](b: BvhNode[S], rnd: var Rand): Point {.gcsafe.} =
  doAssert false, "Cannot sample from a BvhNode"

proc samplePoint*[S: SomeSpectrum](h: Hittable[S], rnd: var Rand): Point {.gcsafe.} =
  ## Samples a point from the contained object by sampling uniformly
  ## from the underlying geometry of the `Hittable[S]` or potentially
  ## in a different way depending on the material.
  case h.kind
  of htSphere:      result = h.hSphere.samplePoint(rnd, h.mat)
  of htCylinder:    result = h.hCylinder.samplePoint(rnd)
  of htCone:        result = h.hCone.samplePoint(rnd)
  of htParaboloid:  result = h.hParaboloid.samplePoint(rnd)
  of htHyperboloid: result = h.hHyperboloid.samplePoint(rnd)
  of htDisk:        result = h.hDisk.samplePoint(rnd)
  of htBvhNode:     result = h.hBvhNode.samplePoint(rnd)
  of htXyRect:      result = h.hXyRect.samplePoint(rnd)
  of htXzRect:      result = h.hXzRect.samplePoint(rnd)
  of htYzRect:      result = h.hYzRect.samplePoint(rnd)
  of htBox:         result = h.hBox.samplePoint(rnd)
  # Convert point back to world space
  result = h.inverseTransform(result)

proc getRadius*[S: SomeSpectrum](h: Hittable[S]): float =
  ## Returns the radius of the given underlying shape. For shapes without a well defined radius
  ## (rectangle) we return the width|height / 2.
  case h.kind
  of htSphere:      result = h.hSphere.radius
  of htCylinder:    result = h.hCylinder.radius
  of htCone:        result = h.hCone.radius
  of htParaboloid:  result = h.hParaboloid.radius
  of htHyperboloid: result = h.hHyperboloid.rMax
  of htDisk:        result = h.hDisk.radius
  of htBvhNode:     result = (h.hBvhNode.box.maximum - h.hBvhNode.box.minimum).length() / 2
  of htXyRect:      result = max(h.hXyRect.y1 - h.hXyRect.y0, h.hXyRect.x1 - h.hXyRect.x0) / 2
  of htXzRect:      result = max(h.hXzRect.z1 - h.hXzRect.z0, h.hXzRect.x1 - h.hXzRect.x0) / 2
  of htYzRect:      result = max(h.hYzRect.z1 - h.hYzRect.z0, h.hYzRect.y1 - h.hYzRect.y0) / 2
  of htBox:         result = (h.hBox.boxMax - h.hBox.boxMin).length() / 2
