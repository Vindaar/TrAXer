import basetypes, math, aabb, algorithm, random

import sensorBuf

type
  Transform* = Mat4d

  HittableKind* = enum
    htSphere, htCylinder, htCone, htBvhNode, htXyRect, htXzRect, htYzRect, htBox, htDisk
  Hittable* {.acyclic.} = ref object
    trans*: Transform = mat4d()
    invTrans*: Transform = mat4d()
    case kind*: HittableKind
    of htSphere: hSphere*: Sphere
    of htCylinder: hCylinder*: Cylinder
    of htCone: hCone*: Cone
    of htBvhNode: hBvhNode*: BvhNode
    of htXyRect: hXyRect*: XyRect
    of htXzRect: hXzRect*: XzRect
    of htYzRect: hYzRect*: YzRect
    of htBox: hBox*: Box
    of htDisk: hDisk: Disk

  HittablesList* = object
    len*: int # len of data seq
    #size*: int # internal data size
    data*: seq[Hittable] #ptr UncheckedArray[Hittable]

  BvhNode* = object
    left*: Hittable
    right*: Hittable
    box*: AABB

  HitRecord* = object
    p*: Point
    normal*: Vec3d
    t*: float
    frontFace*: bool
    mat*: Material
    u*, v*: float

  Sphere* = object
    radius*: float
    mat*: Material

  XyRect* = object
    mat*: Material
    x0*, x1*, y0*, y1*, k*: float

  XzRect* = object
    mat*: Material
    x0*, x1*, z0*, z1*, k*: float

  YzRect* = object
    mat*: Material
    y0*, y1*, z0*, z1*, k*: float

  Cylinder* = object
    radius*: float
    zMin*: float
    zMax*: float
    phiMax*: float
    mat*: Material

  Cone* = object ## Describes a positive cone, starting at z = 0 with `radius` towards +z. r = 0 at `height`
    radius*: float ## Radius at z = 0
    phiMax*: float
    height*: float ## Height of the cone, if complete. This is where its radius is 0.
    zMax*: float ## Cuts off the cone at zMax.
    mat*: Material

  Box* = object
    mat*: Material
    boxMin*: Point
    boxMax*: Point
    sides*: HittablesList

  Disk* = object
    distance*: float # distance along z axis
    radius*: float
    innerRadius*: float
    phiMax*: float = 360.0.degToRad
    mat*: Material

  AnyHittable* = Sphere | Cylinder | Cone | BvhNode | XyRect | XzRect | YzRect | Box | Disk

  MaterialKind* = enum
    mkLambertian, mkMetal, mkDielectric, mkDiffuseLight, mkSolarEmission, mkLaser, mkImageSensor, mkLightTarget
  Material* = object
    case kind*: MaterialKind
    of mkLambertian: mLambertian*: Lambertian
    of mkMetal: mMetal: Metal
    of mkDielectric: mDielectric: Dielectric
    of mkDiffuseLight: mDiffuseLight: DiffuseLight
    of mkSolarEmission: mSolarEmission: SolarEmission
    of mkLaser: mLaser: Laser
    of mkImageSensor: mImageSensor*: ImageSensor
    of mkLightTarget: mLightTarget*: LightTarget

  TextureKind* = enum
    tkSolid, tkChecker#, tkAbsorbentTexture

  Texture* {.acyclic.} = ref object
    case kind*: TextureKind
    of tkSolid: tSolid: SolidTexture
    of tkChecker: tChecker: CheckerTexture

  SolidTexture* {.acyclic.} = ref object
    color*: Color

  CheckerTexture* {.acyclic.} = ref object
    invScale*: float
    even*: Texture
    odd*: Texture

  DiffuseLight* = object
    emit*: Texture

  SolarEmission* = object
    emit*: Texture ## The color used for the regular light source
    fluxRadiusCDF*: seq[float] ## The relative emission for each radius as a CDF

  ImageSensor* = object
    sensor*: Sensor2D

  ## A `LightTarget` is a material assigned to any `Hittable`, which will be used to sample rays from
  ## `DiffuseLight` sources. Any `DiffuseLight` will sample rays towards the surface of the `Hittable`
  ## of `LightTarget`.
  LightTarget* = object
    visible*: bool # Whether it is visible for the Camera based rays
    albedo*: Texture # The texture is only relevant for the Camera based rays to visualize it.

  ## If a `Laser` is applied to an object, it means the entire surface emits alnog the normal. I.e.
  ## this produces parallel light if emitting from a disk (for the Sun!)
  Laser* = object
    emit*: Texture

  Lambertian* = object
    albedo*: Texture

  Metal* = object
    albedo*: Color
    fuzz*: float

  Dielectric* = object
    ir*: float # refractive index (could use `eta`)

  AnyMaterial* = Lambertian | Metal | Dielectric | DiffuseLight | Laser | SolarEmission | Dielectric | ImageSensor | LightTarget

proc value*(s: Texture, u, v: float, p: Point): Color {.gcsafe.}

proc solidColor*(c: Color): SolidTexture = SolidTexture(color: c)
proc solidColor*(r, g, b: float): SolidTexture = SolidTexture(color: color(r, g, b))
proc value*(s: SolidTexture, u, v: float, p: Point): Color = s.color # Is solid everywhere after all

proc toTexture*(x: Color): Texture = Texture(kind: tkSolid, tSolid: solidColor(x))
proc toTexture*(x: SolidTexture): Texture = Texture(kind: tkSolid, tSolid: x)
proc toTexture*(x: CheckerTexture): Texture = Texture(kind: tkChecker, tChecker: x)

proc checkerTexture*(scale: float, even, odd: Texture): CheckerTexture = CheckerTexture(invScale: 1.0 / scale, even: even, odd: odd)
proc checkerTexture*(scale: float, c1, c2: Color): CheckerTexture =
  result = CheckerTexture(invScale: 1.0 / scale, even: toTexture(c1), odd:  toTexture(c2))

proc value*(s: CheckerTexture, u, v: float, p: Point): Color =
  ## Return the color of the checker board at the u, v coordinates and `p`
  let xInt = floor(s.invScale * p.x).int
  let yInt = floor(s.invScale * p.y).int
  let zInt = floor(s.invScale * p.z).int
  let isEven = (xInt + yInt + zInt) mod 2 == 0
  result = if isEven: s.even.value(u, v, p) else: s.odd.value(u, v, p)

proc value*(s: Texture, u, v: float, p: Point): Color {.gcsafe.} =
  case s.kind
  of tkSolid:   result = s.tSolid.value(u, v, p)
  of tkChecker: result = s.tChecker.value(u, v, p)

proc initHittables*(size: int = 8): HittablesList =
  ## allocates memory for `size`, but remains empty
  let size = if size < 8: 8 else: size
  result.len = 0
  result.data = newSeqOfCap[Hittable](size)

proc `[]`*(h: HittablesList, idx: int): Hittable =
  assert idx < h.len
  result = h.data[idx]

proc `[]`*(h: var HittablesList, idx: int): var Hittable =
  assert idx < h.len
  result = h.data[idx]

proc `[]=`*(h: var HittablesList, idx: int, el: Hittable) =
  assert idx < h.len
  h.data[idx] = el

iterator items*(h: HittablesList): Hittable =
  for idx in 0 ..< h.len:
    yield h[idx]

proc initBvhNode*(rnd: var Rand, list: HittablesList, start, stop: int): BvhNode
proc initBvhNode*(rnd: var Rand, list: HittablesList): BvhNode =
  result = initBvhNode(rnd, list, 0, list.len)

proc `[]`*(h: HittablesList, slice: Slice[int]): HittablesList =
  let sliceLen = slice.b - slice.a
  result = initHittables(sliceLen)
  for idx in 0 ..< sliceLen:
    result.data[idx] = h.data[slice.a + idx]
  result.len = sliceLen

proc sort*(h: var HittablesList, start, stop: int, cmp: proc(x, y: Hittable): bool,
           order = SortOrder.Ascending) =
  proc locCmp(a, b: Hittable): int =
    let res = cmp(a, b)
    result = if res: 1 else: -1
  h.data.toOpenArray(start, stop-1).sort(locCmp, order = order)

proc add*(h: var HittablesList, el: Hittable) =
  ## adds a new element to h. If space is there
  h.data.add el
  inc h.len

proc clone*(h: Hittable): Hittable =
  result = Hittable(trans: h.trans,
                    invTrans: h.invTrans,
                    kind: h.kind)
  case h.kind
  of htSphere:   result.hSphere = h.hSphere
  of htCylinder: result.hCylinder = h.hCylinder
  of htCone:     result.hCone = h.hCone
  of htBvhNode:  result.hBvhNode = h.hBvhNode
  of htXyRect:   result.hXyRect = h.hXyRect
  of htXzRect:   result.hXzRect = h.hXzRect
  of htYzRect:   result.hYzRect = h.hYzRect
  of htBox:      result.hBox = h.hBox
  of htDisk:     result.hDisk = h.hDisk


proc clone*(h: HittablesList): HittablesList =
  result = initHittables(h.len)
  for x in h:
    result.add clone(x)

proc toHittable*(s: Sphere): Hittable   = result = Hittable(kind: htSphere, hSphere: s)
proc toHittable*(c: Cylinder): Hittable = result = Hittable(kind: htCylinder, hCylinder: c)
proc toHittable*(c: Cone): Hittable     = result = Hittable(kind: htCone, hCone: c)
proc toHittable*(d: Disk): Hittable     = result = Hittable(kind: htDisk, hDisk: d)
proc toHittable*(b: BvhNode): Hittable  = result = Hittable(kind: htBvhNode, hBvhNode: b)
proc toHittable*(r: XyRect): Hittable   = result = Hittable(kind: htXyRect, hXyRect: r)
proc toHittable*(r: XzRect): Hittable   = result = Hittable(kind: htXzRect, hXzRect: r)
proc toHittable*(r: YzRect): Hittable   = result = Hittable(kind: htYzRect, hYzRect: r)
proc toHittable*(b: Box): Hittable      = result = Hittable(kind: htBox, hBox: b)

proc add*[T: AnyHittable](h: var HittablesList, ht: T) = h.add toHittable(ht)
proc add*(h: var HittablesList, lst: HittablesList) =
  for x in lst:
    h.add x

proc delete*(h: var HittablesList, ht: Hittable) =
  ## Deletes `ht` from `h` if it exists
  let idx = h.data.find(ht)
  if idx > 0:
    h.data.delete(idx)
    dec h.len

proc setFaceNormal*(rec: var HitRecord, r: Ray, outward_normal: Vec3d) =
  rec.frontFace = r.dir.dot(outward_normal) < 0
  rec.normal = if rec.frontFace: outward_normal else: -outward_normal

proc hit*(h: Hittable, r: Ray, tMin, tMax: float, rec: var HitRecord): bool {.gcsafe.}
proc hit*(n: BvhNode, r: Ray, tMin, tMax: float, rec: var HitRecord): bool =
  if not n.box.hit(r, tMin, tMax):
    return false

  let hitLeft = n.left.hit(r, tMin, tMax, rec)
  let hitRight = n.right.hit(r, tMin, if hitLeft: rec.t else: tMax, rec)

  result = hitLeft or hitRight

proc hit*(h: HittablesList, r: Ray, tMin, tMax: float, rec: var HitRecord): bool =
  var tmpRec: HitRecord
  result = false
  var closestSoFar = tMax

  for obj in h:
    if obj.hit(r, tMin, closestSoFar, tmpRec):
      result = true
      closestSoFar = tmpRec.t
      rec = tmpRec

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
  rec.mat = s.mat

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
  if tShapeHit <= 0:
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
  rec.mat = cyl.mat

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
  if tShapeHit <= 0:
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
  rec.mat = con.mat

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
  #let dpdu = vec3(-phiMax * pHit.y, phiMax * pHit.x, 0)
  #let dpdv = vec3(pHit.x, pHit.y, 0.0) * (d.innerRadius - d.radius) / rHit
  rec.t = t
  rec.p = r.at(rec.t)
  let outward_normal = vec3(0.0, 0.0, 1.0)
  rec.setFaceNormal(r, outward_normal.Vec3d)
  rec.mat = d.mat

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
  rec.mat = rect.mat
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
  rec.mat = rect.mat
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
  rec.mat = rect.mat
  rec.p = r.at(t)
  result = true

proc hit*(box: Box, r: Ray, tMin, tMax: float, rec: var HitRecord): bool =
  result = box.sides.hit(r, tMin, tMax, rec)

template transforms(name, field: untyped): untyped =
  proc `name`*(h: Hittable, v: Vec3d): Vec3d =
    ## Apply the world to object transformation for the given vector.
    # calculate transformed vector
    let vt = h.field * vec4d(v, w = 0) ## For vectors weight is always 1!
    # construct result vector. The weight row is irrelevant!
    result = vec3(vt.x, vt.y, vt.z)

  proc `name`*(h: Hittable, p: Point): Point =
    ## Apply the world to object transformation for the given vector.
    # calculate transformed vector
    let vt = h.field * vec4d(p.Vec3d, w = 1) ## XXX: For points we *CURRENTLY* just assume weight 1
    # construct result Point
    result = Point(vec3(vt.x, vt.y, vt.z) / vt.w)

  proc `name`*(h: Hittable, r: Ray): Ray =
    result = Ray(orig: h.name(r.orig), dir: h.name(r.dir))

transforms(transform, trans)
transforms(inverseTransform, invTrans)

#proc transTransform*(h: Hittable, v: Vec3d): Vec3d =
#  var mh = h.clone()
#  mh.trans = h.trans.transpose()
#  result = mh.transform(v)

proc invertNormal*(h: Hittable, n: Vec3d): Vec3d =
  #var mh = h.clone()
  ### XXX: NOTE: if I'm not mistaken `pbrt` uses `inverse().transpose()` here.
  ### But if we do that the reflections on metals break if we use rotations.
  #mh.trans = h.trans.inverse() #.transpose()
  result = h.inverseTransform(n)  #mh.transform(n)

proc hit*(h: Hittable, r: Ray, tMin, tMax: float, rec: var HitRecord): bool {.gcsafe.} =
  # 1. transform to object space
  let rOb = h.transform(r)
  # 2. compute the hit
  case h.kind
  of htSphere:   result = h.hSphere.hit(rOb, tMin, tMax, rec)
  of htCylinder: result = h.hCylinder.hit(rOb, tMin, tMax, rec)
  of htCone:     result = h.hCone.hit(rOb, tMin, tMax, rec)
  of htDisk:     result = h.hDisk.hit(rOb, tMin, tMax, rec)
  of htBvhNode:  result = h.hBvhNode.hit(rOb, tMin, tMax, rec)
  of htXyRect:   result = h.hXyRect.hit(rOb, tMin, tMax, rec)
  of htXzRect:   result = h.hXzRect.hit(rOb, tMin, tMax, rec)
  of htYzRect:   result = h.hYzRect.hit(rOb, tMin, tMax, rec)
  of htBox:      result = h.hBox.hit(rOb, tMin, tMax, rec)
  # 3. convert rec back to world space
  ## XXX: normal transformation in general more complex!
  ## `pbrt` uses `mInv` for *FORWARD* transformation!
  rec.normal = normalize(h.invertNormal(rec.normal))
  rec.p = h.inverseTransform(rec.p)

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

proc boundingBox*(n: BvhNode, outputBox: var AABB): bool =
  outputBox = n.box
  result = true

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

proc boundingBox*(b: Box, outputBox: var AABB): bool =
  outputBox = initAabb(b.boxMin, b.boxMax)
  result = true

proc boundingBox*(h: Hittable, output_box: var AABB): bool =
  case h.kind
  of htSphere: result = h.hSphere.boundingBox(output_box)
  of htCylinder: result = h.hCylinder.boundingBox(output_box)
  of htCone: result = h.hCone.boundingBox(output_box)
  of htDisk: result = h.hDisk.boundingBox(output_box)
  of htBvhNode: result = h.hBvhNode.boundingBox(output_box)
  of htXyRect: result = h.hXyRect.boundingBox(output_box)
  of htXzRect: result = h.hXzRect.boundingBox(output_box)
  of htYzRect: result = h.hYzRect.boundingBox(output_box)
  of htBox: result = h.hBox.boundingBox(output_box)

proc boundingBox*(h: HittablesList, output_box: var AABB): bool =
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

proc box_compare(a, b: Hittable, axis: int): bool {.inline.} =
  var boxA: AABB
  var boxB: AABB

  if not a.boundingBox(boxA) or not b.boundingBox(boxB):
    stderr.write("No bounding box in BVH node constructor!\n")

  result = boxA.minimum[axis] < boxB.minimum[axis]

proc box_x_compare(a, b: Hittable): bool =
  result = boxCompare(a, b, 0)

proc box_y_compare(a, b: Hittable): bool =
  result = boxCompare(a, b, 1)

proc box_z_compare(a, b: Hittable): bool =
  result = boxCompare(a, b, 2)

proc initBvhNode*(rnd: var Rand, list: HittablesList, start, stop: int): BvhNode =
  var mlist = list

  let axis = rnd.rand(2)
  var comparator: (proc(a, b: Hittable): bool)
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
    if comparator(list[start], list[start+1]):
      result.left = list[start]
      result.right = list[start+1]
    else:
      result.left = list[start+1]
      result.right = list[start]
  else:
    mlist.sort(start, stop, comparator)

    let mid = start + objSpan div 2
    result.left = Hittable(kind: htBvhNode, hBvhNode: rnd.initBvhNode(mlist, start, mid))
    result.right = Hittable(kind: htBvhNode, hBvhNode: rnd.initBvhNode(mlist, mid, stop))

  var boxLeft: AABB
  var boxRight: AABB

  if not result.left.boundingBox(boxLeft) or
     not result.right.boundingBox(boxRight):
    stderr.write("No bounding box in BVH node constructor!\n")

  result.box = surroundingBox(boxLeft, boxRight)

proc initDiffuseLight*(a: Color): DiffuseLight =
  result = DiffuseLight(emit: toTexture(a))

proc initSolarEmission*(a: Color, fluxRadiusCDF: seq[float]): SolarEmission =
  result = SolarEmission(emit: toTexture(a), fluxRadiusCDF: fluxRadiusCDF)

proc initLaser*(a: Color): Laser =
  result = Laser(emit: toTexture(a))

proc initLambertian*(a: Color): Lambertian =
  result = Lambertian(albedo: toTexture(a))

proc initLambertian*(t: Texture): Lambertian =
  result = Lambertian(albedo: t)

proc initMetal*(a: Color, f: float): Metal =
  result = Metal(albedo: a, fuzz: f)

proc initDielectric*(ir: float): Dielectric =
  result = Dielectric(ir: ir)

proc initImageSensor*(width, height: int): ImageSensor =
  result = ImageSensor(sensor: initSensor(width, height))

proc initLightTarget*(a: Color, visible: bool): LightTarget =
  result = LightTarget(albedo: toTexture(a), visible: visible)

proc toMaterial*(m: Lambertian): Material = Material(kind: mkLambertian, mLambertian: m)
proc toMaterial*(m: Metal): Material = Material(kind: mkMetal, mMetal: m)
proc toMaterial*(m: Dielectric): Material = Material(kind: mkDielectric, mDielectric: m)
proc toMaterial*(m: DiffuseLight): Material = Material(kind: mkDiffuseLight, mDiffuseLight: m)
proc toMaterial*(m: SolarEmission): Material = Material(kind: mkSolarEmission, mSolarEmission: m)
proc toMaterial*(m: Laser): Material = Material(kind: mkLaser, mLaser: m)
proc toMaterial*(m: ImageSensor): Material = Material(kind: mkImageSensor, mImageSensor: m)
proc toMaterial*(m: LightTarget): Material = Material(kind: mkLightTarget, mLightTarget: m)
template initMaterial*[T: AnyMaterial](m: T): Material = m.toMaterial()

proc initSphere*(center: Point, radius: float, mat: Material): Sphere =
  result = Sphere(radius: radius, mat: mat)

proc initDisk*(distance: float, radius: float, mat: Material): Disk =
  result = Disk(distance: distance, radius: radius, mat: mat)

proc initXyRect*(x0, x1, y0, y1, k: float, mat: Material): XyRect =
  result = XyRect(x0: x0, x1: x1, y0: y0, y1: y1, k: k, mat: mat)

proc initXzRect*(x0, x1, z0, z1, k: float, mat: Material): XzRect =
  result = XzRect(x0: x0, x1: x1, z0: z0, z1: z1, k: k, mat: mat)

proc initYzRect*(y0, y1, z0, z1, k: float, mat: Material): YzRect =
  result = YzRect(y0: y0, y1: y1, z0: z0, z1: z1, k: k, mat: mat)

template rotations(name: untyped): untyped =
  proc `name`*[T: AnyHittable](h: T, angle: float): Hittable =
    result = h.toHittable()
    result.trans = `name`(mat4d(), angle.degToRad)
    result.invTrans = result.trans.inverse()
  proc `name`*(h: Hittable, angle: float): Hittable =
    result = h.clone()
    result.trans = `name`(h.trans, angle.degToRad)
    result.invTrans = result.trans.inverse()
  proc `name`*(h: HittablesList, angle: float): HittablesList =
    result = initHittables(h.len)
    for x in h:
      result.add name(x, angle)
rotations(rotateX)
rotations(rotateY)
rotations(rotateZ)

proc translate*[T: AnyHittable; V: Vec3d | Point](h: T, v: V): Hittable =
  result = h.toHittable()
  result.trans = translate(mat4d(), -v.Vec3d)
  result.invTrans = result.trans.inverse()
proc translate*[V: Vec3d | Point; T: AnyHittable](v: V, h: T): Hittable = h.translate(v)

proc translate*[V: Vec3d | Point](h: Hittable, v: V): Hittable =
  result = h.clone()
  result.trans = translate(h.trans, -v.Vec3d)
  result.invTrans = result.trans.inverse()
proc translate*[V: Vec3d | Point](v: V, h: Hittable): Hittable = h.translate(v.Vec3d)

proc translate*(h: HittablesList, v: Vec3d): HittablesList =
  result = initHittables(h.len)
  for x in h:
    result.add translate(x, v)

proc initBox*(p0, p1: Point, mat: Material): Box =
  result.boxMin = p0
  result.boxMax = p1
  result.mat = mat

  result.sides = initHittables(0)

  result.sides.add initXyRect(p0.x, p1.x, p0.y, p1.y, p1.z, mat)
  result.sides.add initXyRect(p0.x, p1.x, p0.y, p1.y, p0.z, mat)

  result.sides.add initXzRect(p0.x, p1.x, p0.z, p1.z, p1.y, mat)
  result.sides.add initXzRect(p0.x, p1.x, p0.z, p1.z, p0.y, mat)

  result.sides.add initYzRect(p0.y, p1.y, p0.z, p1.z, p1.x, mat)
  result.sides.add initYzRect(p0.y, p1.y, p0.z, p1.z, p0.x, mat)

template lambertTargetBody(): untyped {.dirty.} =
  var scatter_direction = rec.normal + rnd.randomUnitVector()

  # catch degenerate scatter direction
  if scatter_direction.nearZero():
    scatter_direction = rec.normal

  scattered = initRay(rec.p, scatter_direction, r_in.typ)
  attenuation = m.albedo.value(rec.u, rec.v, rec.p)
  result = true

proc scatter*(m: Lambertian, rnd: var Rand,
              r_in: Ray, rec: HitRecord,
              attenuation: var Color, scattered: var Ray): bool {.gcsafe.}  =
  lambertTargetBody()

proc scatter*(m: LightTarget, rnd: var Rand,
              r_in: Ray, rec: HitRecord,
              attenuation: var Color, scattered: var Ray): bool {.gcsafe.}  =
  ## Scatters light like a Lambertian if it's set to be visible and the incoming ray
  ## comes from the `Camera`.
  if m.visible and r_in.typ == rtCamera:
    lambertTargetBody()

proc scatter*(m: Metal, rnd: var Rand, r_in: Ray, rec: HitRecord,
              attenuation: var Color, scattered: var Ray): bool {.gcsafe.}  =
  var reflected = unitVector(r_in.dir).reflect(rec.normal)
  scattered = initRay(rec.p, reflected + m.fuzz * rnd.randomInUnitSphere(), r_in.typ)
  attenuation = m.albedo
  result = scattered.dir.dot(rec.normal) > 0

proc reflectance(cosine, refIdx: float): float =
  ## use Schlick's approximation for reflectance
  var r0 = (1 - refIdx) / (1 + refIdx)
  r0 = r0 * r0
  result = r0 + (1 - r0) * pow(1 - cosine, 5)

proc scatter*(m: Dielectric,
              rnd: var Rand,
              r_in: Ray, rec: HitRecord,
              attenuation: var Color, scattered: var Ray): bool {.gcsafe.}  =
  attenuation = color(1.0, 1.0, 1.0)
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
  NonEmittingMaterials* = Lambertian | Metal | Dielectric | LightTarget
  EmittingMaterials* = DiffuseLight | Laser | ImageSensor

proc scatter*[T: DiffuseLight | Laser | SolarEmission](
  m: T,
  rnd: var Rand,
  r_in: Ray, rec: HitRecord,
  attenuation: var Color, scattered: var Ray
                                                     ): bool {.gcsafe.} =
  ## Diffuse lights, lasers and solare emissions do not scatter!
  result = false

proc scatter*(m: ImageSensor, rnd: var Rand, r_in: Ray, rec: HitRecord,
              attenuation: var Color, scattered: var Ray): bool {.gcsafe.} =
  ## An image sensor is a perfect sink! (At least we assume so)
  result = false
proc scatter*(m: Material,
              rnd: var Rand,
              r_in: Ray, rec: HitRecord,
              attenuation: var Color, scattered: var Ray): bool {.gcsafe.} =
  case m.kind
  of mkLambertian:    result = m.mLambertian.scatter(rnd, r_in, rec, attenuation, scattered)
  of mkMetal:         result = m.mMetal.scatter(rnd, r_in, rec, attenuation, scattered)
  of mkDielectric:    result = m.mDielectric.scatter(rnd, r_in, rec, attenuation, scattered)
  of mkDiffuseLight:  result = m.mDiffuseLight.scatter(rnd, r_in, rec, attenuation, scattered)
  of mkSolarEmission: result = m.mSolarEmission.scatter(rnd, r_in, rec, attenuation, scattered)
  of mkLaser:         result = m.mLaser.scatter(rnd, r_in, rec, attenuation, scattered)
  of mkImageSensor:   result = m.mImageSensor.scatter(rnd, r_in, rec, attenuation, scattered)
  of mkLightTarget:   result = m.mLightTarget.scatter(rnd, r_in, rec, attenuation, scattered)

proc emit*[T: NonEmittingMaterials](m: T, u, v: float, p: Point): Color =
  ## Materials that don't emit just return black!
  result = color(0, 0, 0)

proc emit*[T: DiffuseLight | Laser | SolarEmission](m: T, u, v: float, p: Point): Color =
  result = m.emit.value(u, v, p)

proc emit*(m: ImageSensor, u, v: float, p: Point): Color = discard
proc emit*(m: Material, u, v: float, p: Point): Color =
  case m.kind
  of mkLambertian:    result = m.mLambertian.emit(u, v, p)
  of mkMetal:         result = m.mMetal.emit(u, v, p)
  of mkDielectric:    result = m.mDielectric.emit(u, v, p)
  of mkDiffuseLight:  result = m.mDiffuseLight.emit(u, v, p)
  of mkSolarEmission: result = m.mSolarEmission.emit(u, v, p)
  of mkLaser:         result = m.mLaser.emit(u, v, p)
  of mkImageSensor:   result = m.mImageSensor.emit(u, v, p)
  of mkLightTarget:   result = m.mLightTarget.emit(u, v, p)

