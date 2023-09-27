import strformat, os, terminal, macros, math, random, times

import basetypes, hittables, camera

import arraymancer, unchained

import sdl2 except Color, Point

type
  RenderContext = ref object
    rnd: Rand
    camera: Camera
    world: HittablesList[RGBSpectrum]
    worldXray: HittablesList[XraySpectrum] ## Copy of the world without any light sources.
    sources: HittablesList[XraySpectrum] ## List of all light sources
    targets: HittablesList[XraySpectrum] ## List of all targets for diffuse lights
    buf: ptr UncheckedArray[uint32]
    counts: ptr UncheckedArray[int]
    window: SurfacePtr
    numRays: int
    width: int
    height: int
    maxDepth: int
    numPer: int = -1 # only used for multithreaded case
    numThreads: int = -1 # only used for multithreaded case

  TracingType = enum
    ttCamera, ttLights

  SourceKind = enum
    skSun, skXrayFinger, skParallelXrayFinger

  Config = object
    visibleTarget: bool
    gridLines: bool
    usePerfectMirror: bool
    sourceKind: SourceKind
    solarModelFile: string
    energyMin: float
    energyMax: float
    rayAt = 1.0
    setupRotation = 90.0 # Angle the entire setup is rotated. This is the rotation to bring the downward pointing
                          # setup to the realistic sideways setup.
    telescopeRotation = 14.17 # *Additional* rotation of the telescope on top of `setupRotation`.
                               # Separate to include real rotation aside from pointing the entire setup.
    windowRotation = 30.0
    windowZOffset = 3.0
    ignoreWindow = false
    ignoreMagnet = true ## Whether to include the magnet
    ignoreMirrorThickness = false
    ignoreSpacer = false
    mirrorThickness = Inf # adjust mirror thickenss. 0.2 by default

    sensorKind = sCount
    brokenMirrors = false # disables telescope mirror reflection for debugging
    midTelescopeSensor = false
    endTelescopeSensor = false # An ~ImageSensor~ directly at the end of the telescope. Useful to view
                               # the illumination of the telescope

    # X-ray source fields
    sourceDistance = 14.2.m # distance of the source ``from the telescope entrance``
    sourceRadius = 3.0.mm

var Tracing = ttCamera

proc initConfig(visibleTarget, gridLines, usePerfectMirror: bool, sourceKind: SourceKind,
                solarModelFile: string,
                energyMin, energyMax: float,
                rayAt, setupRotation, telescopeRotation, windowRotation, windowZOffset: float,
                ignoreWindow: bool,
                sensorKind: SensorKind,
                brokenMirrors: bool,
                midTelescopeSensor, endTelescopeSensor: bool,
                ignoreMirrorThickness: bool,
                mirrorThickness: float,
                ignoreSpacer: bool,
                sourceDistance: Meter, sourceRadius: MilliMeter): Config =
  result = Config(visibleTarget: visibleTarget,
                  gridLines: gridLines,
                  usePerfectMirror: usePerfectMirror,
                  sourceKind: sourceKind,
                  solarModelFile: solarModelFile,
                  energyMin: energyMin,
                  energyMax: energyMax,
                  rayAt: rayAt,
                  setupRotation: setupRotation,
                  telescopeRotation: telescopeRotation,
                  windowRotation: windowRotation,
                  windowZOffset: windowZOffset,
                  ignoreWindow: ignoreWindow,
                  sensorKind: sensorKind,
                  brokenMirrors: brokenMirrors,
                  midTelescopeSensor: midTelescopeSensor,
                  endTelescopeSensor: endTelescopeSensor,
                  ignoreMirrorThickness: ignoreMirrorThickness,
                  mirrorThickness: mirrorThickness,
                  ignoreSpacer: ignoreSpacer,
                  sourceDistance: sourceDistance,
                  sourceRadius: sourceRadius)

proc initRenderContext(rnd: var Rand,
                       buf: ptr UncheckedArray[uint32], counts: ptr UncheckedArray[int],
                       window: SurfacePtr, numRays, width, height: int,
                       camera: Camera, world: GenericHittablesList, maxDepth: int,
                       numPer, numThreads: int): RenderContext =
  var worldRGB = world.cloneAsRGB()
  var worldXray = world.cloneAsXray()
  let sources = worldXray.getSources(delete = true)
  let targets = worldXray.getLightTargets(delete = true)
  # filter invisible targets
  worldRGB.removeInvisibleTargets()
  result = RenderContext(rnd: rnd,
                         buf: buf, counts: counts,
                         window: window,
                         numRays: numRays, width: width, height: height,
                         camera: camera,
                         world: worldRGB,
                         worldXray: worldXray,
                         sources: sources,
                         targets: targets,
                         maxDepth: maxDepth,
                         numPer: numPer, numThreads: numThreads)

proc initRenderContext(rnd: var Rand,
                       buf: var Tensor[uint32], counts: var Tensor[int],
                       window: SurfacePtr, numRays, width, height: int,
                       camera: Camera, world: GenericHittablesList, maxDepth: int,
                       numPer: int = -1, numThreads: int = -1): RenderContext =
  let bufP    = cast[ptr UncheckedArray[uint32]](buf.unsafe_raw_offset())
  var countsP = cast[ptr UncheckedArray[int]](counts.unsafe_raw_offset())
  result = initRenderContext(rnd, bufP, countsP, window, numRays, width, height, camera, world, maxDepth, numPer, numThreads)

proc initRenderContexts(numThreads: int,
                        buf: var Tensor[uint32], counts: var Tensor[int],
                        window: SurfacePtr, numRays, width, height: int,
                        camera: Camera, world: GenericHittablesList, maxDepth: int): seq[RenderContext] =
  result = newSeq[RenderContext](numThreads)
  let numPer = (width * height) div numThreads
  for i in 0 ..< numThreads:
    let bufP    = cast[ptr UncheckedArray[uint32]](buf.unsafe_raw_offset()[i * numPer].addr)
    let countsP = cast[ptr UncheckedArray[int]](counts.unsafe_raw_offset()[i * numPer].addr)
    ## XXX: clone world?
    var rnd = initRand(i * 0xfafe)
    result[i] = initRenderContext(rnd, bufP, countsP, window, numRays, width, height, camera.clone(), world.clone(), maxDepth, numPer, numThreads)

when compileOption("threads"):
  import malebolgia
var THREADS = 16

proc rayColor*[S: SomeSpectrum](c: Camera, rnd: var Rand, r: Ray, world: HittablesList[S], depth: int): S {.gcsafe.} =
  var rec: HitRecord[S]

  if depth <= 0:
    return toSpectrum(color(0, 0, 0), S)

  if world.hit(r, 0.001, Inf, rec):
    var scattered: Ray
    var attenuation: S
    var emitted = rec.mat.emit(rec.u, rec.v, rec.p)
    if not rec.mat.scatter(rnd, r, rec, attenuation, scattered):
      result = emitted
    else:
      result = attenuation * c.rayColor(rnd, scattered, world, depth - 1) + emitted
  else:
    result = toSpectrum(c.background, S)

  when false: ## Old code with background gradient
    let unitDirection = unitVector(r.dir)
    let t = 0.5 * (unitDirection.y + 1.0)
    result = (1.0 - t) * color(1.0, 1.0, 1.0) + t * color(0.5, 0.7, 1.0)

proc rayColorRecurse*[S: SomeSpectrum](
  c: Camera, rnd: var Rand, r: Ray, world: HittablesList[S], depth: int,
  accumulatedSpectrum: var S
                                     ): S {.gcsafe.} =
  var rec: HitRecord[S]

  if depth <= 0:
    return toSpectrum(color(0, 0, 0), S)

  if world.hit(r, 0.001, Inf, rec):
    #if r.typ == rtLight:
    #  echo "Hit ", depth
    var scattered: Ray
    var attenuation = accumulatedSpectrum ## NOTE: `attenuation` starts with the current accumulated spectrum.
                                          ## That way `scatter` can both look at the current value while still
                                          ## using it as a buffer!
    #if r.typ == rtLight and rec.mat.kind == mkImageSensor:
    #  echo "Current attenuation: ", attenuation
    var emitted = rec.mat.emit(rec.u, rec.v, rec.p)
    if not rec.mat.scatter(rnd, r, rec, attenuation, scattered):
      result = emitted
    else:
      # Accumulate the spectrum down the call chain, that way `scatter` sees the up to date attenuation
      accumulatedSpectrum = accumulatedSpectrum * attenuation + emitted
      result = attenuation * c.rayColorRecurse(rnd, scattered, world, depth - 1, accumulatedSpectrum) + emitted
  else:
    result = toSpectrum(c.background, S)
proc rayColorRecurse*[S: SomeSpectrum](
  c: Camera, rnd: var Rand, r: Ray, world: HittablesList[S], depth: int,
  accumulatedSpectrum: S = toSpectrum(1, S)
                                     ): S {.gcsafe.} =
  ## Overload for no `accumulatedSpectrum` or non `var` argument.
  var spectrum = accumulatedSpectrum # variable only needed due to recursive calls to update
  result = rayColorRecurse(c, rnd, r, world, depth, spectrum)

proc rayColorAndPos*[S: SomeSpectrum](c: Camera, rnd: var Rand, r: Ray,
                                      initialColor: S, world: HittablesList[S],
                                      depth: int): (S, float, float) {.gcsafe.} =
  var rec: HitRecord[S]

  echo "Start==============================\n\n"
  proc recurse[S: SomeSpectrum](rec: var HitRecord[S], c: Camera, rnd: var Rand,
                                r: Ray, world: HittablesList[S], depth: int,
                                initialColor: S): S =
    echo "Depth = ", depth
    if depth <= 0:
      return toSpectrum(0, S)

    #var color: Color = initialColor
    result = initialColor
    if world.hit(r, 0.001, Inf, rec):
      #echo "Hit: ", rec.p, " mat: ", rec.mat, " at depth = ", depth, " rec: ", rec
      var scattered: Ray
      var attenuation: S
      var emitted = rec.mat.emit(rec.u, rec.v, rec.p)
      if rec.mat.kind == mkImageSensor:
        result = toSpectrum(1, S) ## Here we return 1 so that the function call above terminates correctly
        discard
      elif not rec.mat.scatter(rnd, r, rec, attenuation, scattered):
        result = emitted
      else:
        let angle = arccos(scattered.dir.dot(rec.normal)).radToDeg
        echo "Scattering angle : ", angle
        result = attenuation * recurse(rec, c, rnd, scattered, world, depth - 1, initialColor) + emitted
        #let res =
        #if rec.mat.kind == mkImageSensor:
        #
        #else:
        #  result = attenuation * res + emitted
    else:
      result = toSpectrum(c.background, S)

  let color = recurse(rec, c, rnd, r, world, depth, initialColor)
  echo "------------------------------Finish\n\n"
  if rec.mat.kind == mkImageSensor: # and color != initialColor:
    ## In this case return color and position
    echo "Initial color? ", color, " rec.mat: ", rec.mat, " at ", (rec.u, rec.v)
    result = (color, rec.u, rec.v)
  else: # else just return nothing
    result = (toSpectrum(0, S), 0, 0)

  when false: ## Old code with background gradient
    let unitDirection = unitVector(r.dir)
    let t = 0.5 * (unitDirection.y + 1.0)
    result = (1.0 - t) * color(1.0, 1.0, 1.0) + t * color(0.5, 0.7, 1.0)


proc writeColor*(f: File, color: Color, samplesPerPixel: int) =
  let scale = 1.0 / samplesPerPixel.float
  let
    r = sqrt(color.r * scale)
    g = sqrt(color.g * scale)
    b = sqrt(color.b * scale)
  f.write(&"{(256 * r.clamp(0, 0.999)).int} {(256 * g.clamp(0, 0.999)).int} {(256 * b.clamp(0, 0.999)).int}\n")

proc writeColor*(f: File, color: Color) =
  f.write(&"{color.r} {color.g} {color.b}\n")

proc toColor(u: uint32 | int32): Color =
  result[0] = ((u and 0xFF0000) shr 16).float / 256.0
  result[1] = ((u and 0x00FF00) shr 8).float / 256.0
  result[2] = (u and 0x0000FF).float / 256.0

proc toUInt32(c: ColorU8): uint32
proc toColorU8(c: Color, samplesPerPixel: int = 1): ColorU8 {.inline.} =
  let scale = 1.0 / samplesPerPixel.float
  let
    r = 256 * clamp(c.r * scale, 0, 0.999)
    g = 256 * clamp(c.g * scale, 0, 0.999)
    b = 256 * clamp(c.b * scale, 0, 0.999)
  result = (r: r.uint8, g: g.uint8, b: b.uint8)
  #echo c.repr, " and result ", result, " and asuint32 ", result.toUint32, " and back ", result.toUint32.toColor.repr

proc gammaCorrect*(c: Color): Color =
  result[0] = sqrt(c.r)
  result[1] = sqrt(c.g)
  result[2] = sqrt(c.b)

proc toUInt32(c: ColorU8): uint32 =
  result = (#255 shl 24 or
            c.r.int shl 16 or
            c.g.int shl 8 or
            c.b.int).uint32

proc render*(img: Image, f: string,
             rnd: var Rand,
             world: var HittablesList,
             camera: Camera,
             samplesPerPixel, maxDepth: int) =
  ## Write a ppm file to `f`
  var f = open(f, fmWrite)
  f.write(&"P3\n{img.width} {img.height}\n255\n")
  for j in countdown(img.height - 1, 0):
    stderr.write(&"\rScanlines remaining: {j}")
    for i in 0 ..< img.width:
      var pixelColor = color(0, 0, 0)
      for s in 0 ..< samplesPerPixel:
        let r = camera.getRay(rnd, i, j)
        pixelColor += camera.rayColor(rnd, r, world, maxDepth)
      f.writeColor(pixelColor, samplesPerPixel)
  f.close()

proc renderMC*(img: Image, f: string,
               rnd: var Rand,
               world: var HittablesList,
               camera: Camera,
               samplesPerPixel, maxDepth: int) =
  ## Write a ppm file to `f`
  var f = open(f, fmWrite)
  f.write(&"P3\n{img.width} {img.height}\n255\n")
  var numRays = samplesPerPixel * img.width * img.height
  var buf = newTensor[Color](@[img.height, img.width])
  var counts = newTensor[int](@[img.height, img.width])
  var idx = 0

  while idx < numRays:
    let x = rnd.rand(img.width)
    let y = rnd.rand(img.height)
    let r = camera.getRay(rnd, x, y)
    let color = camera.rayColor(rnd, r, world, maxDepth)
    buf[y, x] = buf[y, x] + color
    counts[y, x] = counts[y, x] + 1
    inc idx
    if idx mod (img.width * img.height) == 0:
      let remain = numRays - idx
      stderr.write(&"\rRays remaining: {remain}")
  for j in countdown(img.height - 1, 0):
    stderr.write(&"\rScanlines remaining: {j}")
    for i in 0 ..< img.width:
      f.writeColor(buf[j, i], counts[j, i])
  f.close()

proc sampleRay[S: SomeSpectrum](rnd: var Rand, sources, targets: HittablesList[S]): (Ray, S) {.gcsafe.} =
  ## Sample a ray from one of the sources
  # 1. pick a sources
  let num = sources.len
  let idx = if num == 1: 0 else: rnd.rand(num - 1) ## XXX: For now uniform sampling between sources!
  let source = sources[idx]
  # 2. sample from source
  let p = samplePoint(source, rnd)
  # 3. get the color of the source
  let pOrigin = source.transform(p) # convert point back to object space to get point from center of source
  let spectrum = source.getMaterial.emitAxion(pOrigin, source.getRadius())
  # 4. depending on the source material choose direction
  case source.getMaterial.kind
  of mkLaser: # lasers just sample along the normal of the material
    let dir = vec3(0.0, 0.0, -1.0) ## XXX: make this the normal surface!
    result = (initRay(p, dir, rtLight), spectrum)
  of mkDiffuseLight, mkSolarEmission: # diffuse light need a target
    let numT = targets.len
    if numT == 0:
      raise newException(ValueError, "There must be at least one target for diffuse lights.")
    let idxT = if numT == 1: 0 else: rnd.rand(numT - 1)
    let target = targets[idxT]
    let targetP = target.samplePoint(rnd)
    let dir = normalize(targetP - p)
    # For diffuse lights we propagate the ray *towards the target*
    # and place it 1 before it. This is to avoid issues with ray intersections
    # if the source is _very far_ away from the target.
    #echo "Initial origin: ", p
    var ray = initRay(p, dir, rtLight)
    ray.orig = ray.at((targetP - p).length() - 1.0)
    result = (ray, spectrum)

    when false:# true:
      block SanityCheck: # Check the produced ray actually hits the target
        var rec: HitRecord
        if not targets.hit(result[0], 0.001, Inf, rec):
          doAssert false, "Sampled ray does not hit our target! " & $result
  else: doAssert false, "Not a possible branch, these materials are not sources."

  #echo "Sampled ray has angle to z axis: ", arccos(vec3(0.0,0.0,1.0).dot(result[0].dir)).radToDeg, " ray: ", result[0]

proc renderSdlFrame(ctx: RenderContext) =
  var idx = 0
  let
    width = ctx.width
    height = ctx.height
    maxDepth = ctx.maxDepth
    camera = ctx.camera

  while idx < ctx.numRays:
    var
      yIdx: int
      xIdx: int
      color: Color
    case Tracing
    of ttCamera:
      let x = ctx.rnd.rand(width - 1)
      let y = ctx.rnd.rand(height - 1)
      #if x.int >= window.w: continue
      #if y.int >= window.h: continue
      let r = camera.getRay(ctx.rnd, x, y)
      color = toColor(camera.rayColorRecurse(ctx.rnd, r, ctx.world, maxDepth))
      yIdx = y #height - y - 1
      xIdx = x
    of ttLights:
      # 1. get a ray from a source
      let (r, initialColor) = ctx.rnd.sampleRay(ctx.sources, ctx.targets)
      # 2. trace it. Check if ray ended up on `ImageSensor`
      let (c, u, v) = camera.rayColorAndPos(ctx.rnd, r, initialColor, ctx.worldXray, maxDepth)
      if c.isBlack: continue
      #echo r
      #echo "empty?? ", c, " at ", (u, v)

      # 3. if so, get the relative position on sensor, map to x/y
      color = toColor c
      xIdx = clamp((u * (width.float - 1.0)).round.int, 0, width)
      yIdx = clamp((v * (height.float - 1.0)).round.int, 0, height)

    when true:
      # 1. get a ray from a source
      for _ in 0 ..< 1:
        let (r, initialColor) = ctx.rnd.sampleRay(ctx.sources, ctx.targets)
        # 2. trace it
        let c = camera.rayColorRecurse(ctx.rnd, r, ctx.worldXray, maxDepth, initialColor)
        # this color itself is irrelevant, but might illuminate the image sensor!

    let bufIdx = yIdx * height + xIdx
    ctx.counts[bufIdx] = ctx.counts[bufIdx] + 1
    let curColor = ctx.buf[bufIdx].toColor
    let delta = (color.gammaCorrect - curColor) / ctx.counts[bufIdx].float
    let newColor = curColor + delta
    let cu8 = toColorU8(newColor)
    let sdlColor = sdl2.mapRGB(ctx.window.format, cu8.r.byte, cu8.g.byte, cu8.b.byte)
    ctx.buf[bufIdx] = sdlColor
    inc idx

proc renderFrame(j: int, ctx: ptr RenderContext) {.gcsafe.} =
  let ctx = ctx[]
  let
    width = ctx.width
    height = ctx.height
    maxDepth = ctx.maxDepth
    camera = ctx.camera
    numPer = ctx.numPer
  let frm = numPer * j

  ## XXX: I removed the `-1` in the `else` branch, check!
  ## -> Seems to have fixed the 'empty pixels'
  let to = if j == ctx.numThreads: width * height - 1 else: numPer * (j + 1) # - 1
  var j = 0
  while j < ctx.numRays:
    let idx = ctx.rnd.rand(frm.float .. to.float)
    let x = idx mod width.float
    let y = idx.float / width.float
    #if x.int >= window.w: continue
    #if y.int >= window.h: continue
    let r = camera.getRay(ctx.rnd, x.int, y.int)
    let color = toColor camera.rayColor(ctx.rnd, r, ctx.world, maxDepth)

    block LightSources:
      # 1. get a ray from a source
      if ctx.sources.len > 0:
        for _ in 0 ..< 1:
          let (r, spectrum) = ctx.rnd.sampleRay(ctx.sources, ctx.targets)
          # 2. trace it
          let c = camera.rayColorRecurse(ctx.rnd, r, ctx.worldXray, maxDepth, spectrum)
          # this color itself is irrelevant, but might illuminate the image sensor!

    ctx.counts[idx.int - frm] = ctx.counts[idx.int - frm] + 1
    let curColor = ctx.buf[idx.int - frm].toColor
    let delta = (color.gammaCorrect - curColor) / ctx.counts[idx.int - frm].float
    let newColor = curColor + delta
    let cu8 = toColorU8(newColor)
    if false:# delta.r > 0.0 or delta.g > 0.0 or delta.b > 0.0 or curColor.r > 0 or curColor.g > 0 or curColor.b > 0:
      echo "curColor = ", curColor
      echo "color = ", color, " gamma corrected: ", color.gammaCorrect
      echo "Delta = ", delta
      echo "New = ", curColor + delta
      echo "Cu8 = ", cu8
      echo "\n"
    let sdlColor = sdl2.mapRGB(ctx.window.format, cu8.r.byte, cu8.g.byte, cu8.b.byte)
    ctx.buf[idx.int - frm] = sdlColor
    inc j

proc copyBuf(bufT: Tensor[uint32], window: SurfacePtr) =
  var surf = fromBuffer[uint32](window.pixels, @[window.h.int, window.w.int])
  if surf.shape == bufT.shape:
    surf.copyFrom(bufT)
  else:
    echo "Buffer and window size don't match, slow copy!"
    ## have to copy manually, because surf smaller than bufT
    for y in 0 ..< surf.shape[0]:
      for x in 0 ..< surf.shape[1]:
        surf[y, x] = bufT[y, x]

proc updateCamera(ctxs: var seq[RenderContext], camera: Camera) =
  for ctx in mitems(ctxs):
    ctx.camera = clone(camera)

proc writeData[T](p: ptr UncheckedArray[T], len, width, height: int, outpath, prefix: string) =
  ## XXX: TODO use `nio`?
  let filename = &"{outpath}/{prefix}_type_{$T}_len_{len}_width_{width}_height_{height}.dat"
  echo "[INFO] Writing file: ", filename
  writeFile(filename, toOpenArray(cast[ptr UncheckedArray[byte]](p), 0, len * sizeof(T)))

from std / times import now, `$`

import ggplotnim except Point, Color, colortypes, color


proc renderSdl*(img: Image, world: GenericHittablesList,
                rnd: var Rand, # the *main thread* RNG
                camera: Camera,
                samplesPerPixel, maxDepth: int,
                speed = 1.0, speedMul = 1.1,
                numRays = 100
               ) =
  discard sdl2.init(INIT_EVERYTHING)
  var screen = sdl2.createWindow("Ray tracing".cstring,
                                 SDL_WINDOWPOS_UNDEFINED,
                                 SDL_WINDOWPOS_UNDEFINED,
                                 img.width.cint, img.height.cint,
                                 SDL_WINDOW_OPENGL);
  var renderer = sdl2.createRenderer(screen, -1, 1)
  if screen.isNil:
    quit($sdl2.getError())

  var mouseModeIsRelative = false
  var mouseEnabled = false
  var movementIsFree = true

  var quit = false
  var event = sdl2.defaultEvent

  var window = sdl2.getsurface(screen)

  # store original position from and to we look to reset using `backspace`
  let origLookFrom = camera.lookFrom
  let origLookAt = camera.lookAt

  template resetBufs(bufT, counts: untyped): untyped {.dirty.} =
    bufT.setZero()
    counts.setZero()

  var bufT = newTensor[uint32](@[img.height, img.width])
  var counts = newTensor[int](@[img.height, img.width])

  let width = img.width
  let height = img.height

  var speed = speed

  ## XXX: IMPLEMENT change of vertical field of view using mouse wheel! sort of a zoom

  var lastLookFrom: Point

  when compileOption("threads"):
    var ctxSeq = initRenderContexts(THREADS,
                                    bufT, counts, window, numRays, width, height, camera, world, maxDepth)

  else:
    let ctx = initRenderContext(rnd, bufT, counts, window, numRays, width, height, camera, world, maxDepth)

  while not quit:
    while pollEvent(event):
      case event.kind
      of QuitEvent:
        quit = true
      of KeyDown:
        const dist = 1.0
        case event.key.keysym.scancode
        of SDL_SCANCODE_LEFT, SDL_SCANCODE_RIGHT, SDL_SCANCODE_A, SDL_SCANCODE_D:
          let cL = (camera.lookFrom - camera.lookAt).Vec3d
          let zAx = vec3(0.0, 1.0, 0.0)
          let newFrom = speed * cL.cross(zAx).normalize().Point
          var nCL: Point
          var nCA: Point
          if event.key.keysym.scancode in {SDL_SCANCODE_LEFT, SDL_SCANCODE_A}:
            nCL = camera.lookFrom +. newFrom
            nCA = camera.lookAt +. newFrom
          else:
            nCL = camera.lookFrom -. newFrom
            nCA = camera.lookAt -. newFrom
          camera.updateLookFromAt(nCL, nCA)
          resetBufs(bufT, counts)
        of SDL_SCANCODE_PAGEUP:
          speed *= speedMul
          echo "[INFO] New speed = ", speed
        of SDL_SCANCODE_PAGEDOWN:
          speed /= speedMul
          echo "[INFO] New speed = ", speed
        of SDL_SCANCODE_UP, SDL_SCANCODE_DOWN, SDL_SCANCODE_W, SDL_SCANCODE_S:
          var cL = camera.lookFrom - camera.lookAt
          if not movementIsFree:
            cL[1] = 0.0
          cL = speed * cL.normalize()
          var nCL: Point
          var nCA: Point
          if event.key.keysym.scancode in {SDL_SCANCODE_UP, SDL_SCANCODE_W}:
            nCL = camera.lookFrom -. cL.Point
            nCA = camera.lookAt -. cL.Point
          else:
            nCL = camera.lookFrom +. cL.Point
            nCA = camera.lookAt +. cL.Point

          camera.updateLookFromAt(nCL, nCA)
          resetBufs(bufT, counts)
        of SDL_SCANCODE_LCTRL, SDL_SCANCODE_SPACE:
          let cL = (camera.lookFrom - camera.lookAt).Vec3d
          let zAx = vec3(1.0, 0.0, 0.0)
          let newFrom = if cL.dot(zAx) > 0:
                          speed * cL.cross(zAx).normalize().Point
                        else:
                          speed * -cL.cross(zAx).normalize().Point
          var nCL: Point
          var nCA: Point
          if event.key.keysym.scancode == SDL_SCANCODE_LCTRL:
            nCL = camera.lookFrom -. newFrom
            nCA = camera.lookAt -. newFrom
          else:
            nCL = camera.lookFrom +. newFrom
            nCA = camera.lookAt +. newFrom
          camera.updateLookFromAt(nCL, nCA)
          resetBufs(bufT, counts)
        of SDL_SCANCODE_TAB:
          let nYaw = camera.yaw + PI
          let nPitch = -camera.pitch
          camera.updateYawPitchRoll(camera.lookFrom, nYaw, nPitch, 0.0)
          resetBufs(bufT, counts)
        of SDL_SCANCODE_BACKSPACE:
          echo "Resetting view!"
          camera.updateLookFromAt(origLookFrom, origLookAt)
          resetBufs(bufT, counts)
        of SDL_SCANCODE_N:
          ## activate free movement (n for noclip ;))
          movementIsFree = not movementIsFree
        of SDL_SCANCODE_T:
          Tracing = if Tracing == ttLights: ttCamera else: ttLights
          resetBufs(bufT, counts)
          echo "[INFO]: Set tracing type to: ", Tracing
        of SDL_SCANCODE_ESCAPE:
          ## 'Uncapture' the mouse
          if mouseModeIsRelative:
            discard setRelativeMouseMode(False32)
            mouseModeIsRelative = false
            mouseEnabled = false
            echo "[INFO] Mouse disabled."
        of SDL_SCANCODE_F5:
          ## Save the current data on the image sensor as well as the current camera buffer and counts buffer
          echo "[INFO] Writing buffers to binary files."
          const path = "out"
          createDir(path)
          let bufP = cast[ptr UncheckedArray[uint32]](bufT.unsafe_raw_offset())
          let tStr = $now()
          bufP.writeData(bufT.size.int, width, height,
                         path, &"buffer_{tStr}")
          let countsP = cast[ptr UncheckedArray[int]](counts.unsafe_raw_offset())
          countsP.writeData(counts.size.int, width, height,
                           path, &"counts_{tStr}")
          # now get all image sensors and write their data
          let sensors = world.getImageSensors()
          var idx = 0
          for s in sensors:
            let sm = s.getMaterial.mImageSensor.sensor
            var prefix = &"image_sensor_{idx}_{tStr}_"
            case s.kind
            of htBox:
              let physSize = s.hBox.boxMax - s.hBox.boxMin
              prefix.add &"_dx_{physSize.x:.1f}_dy_{physSize.y:.1f}_dz_{physSize.z:.1f}"
            else: discard
            sm.buf.writeData(sm.len, sm.width, sm.height,
                             path, prefix)
            inc idx
        else: discard
      of MousebuttonDown:
        ## activate relative mouse motion
        if not mouseModeIsRelative:
          discard setRelativeMouseMode(True32)
          mouseModeIsRelative = true
          mouseEnabled = true
          echo "[INFO] Mouse enabled."
          #if mouseEnabled: echo "[INFO] Mouse enabled."
          #else: echo "[INFO] Mouse disabled."
      of WindowEvent:
        freeSurface(window)
        window = sdl2.getsurface(screen)
      of MouseMotion:
        ## for now just take a small fraction of movement as basis
        if mouseEnabled:
          let yaw = -event.motion.xrel.float / 1000.0
          var pitch = -event.motion.yrel.float / 1000.0
          var newLook: Vec3d
          if not movementIsFree:
            ## TODO: fix me
            newLook = (camera.lookAt - camera.lookFrom).Vec3d.rotateAround(camera.lookAt, yaw, 0, pitch)
            camera.updateLookFrom(Point(newLook))
          else:
            let nYaw = camera.yaw + yaw
            echo "Old yaw ", camera.yaw, " add yaw = ", yaw, " new yaw ", nYaw

            let nPitch = camera.pitch + pitch
            camera.updateYawPitchRoll(camera.lookFrom, nYaw, nPitch, 0.0)
            echo "Now looking at: ", camera.lookAt, " from : ", camera.lookFrom, ", yaw = ", nYaw, ", pitch = ", nPitch
          resetBufs(bufT, counts)

      else: echo event.kind
    discard lockSurface(window)

    ## rendering of this frame
    when not compileOption("threads"):
      renderSdlFrame(ctx)
      copyBuf(bufT, window)
    else:
      ## TODO: replace this by a long running background service to which we submit
      ## jobs and the await them? So we don't have the overhead!
      if camera.lookFrom != lastLookFrom:
        echo "[INFO] Current position (lookFrom) = ", camera.lookFrom, " at (lookAt) ", camera.lookAt
        lastLookFrom = camera.lookFrom

      ctxSeq.updateCamera(camera)
      var m = createMaster()
      m.awaitAll:
        for j in 0 ..< THREADS:
          m.spawn renderFrame(j, ctxSeq[j].addr)
      copyBuf(bufT, window)

    unlockSurface(window)
    #sdl2.clear(arg.renderer)
    sdl2.present(renderer)
  sdl2.quit()

proc sceneRedBlue(): GenericHittablesList =
  result = initGenericHittables()
  let R = cos(Pi/4.0)

  #world.add Sphere(center: point(0, 0, -1), radius: 0.5)
  #world.add Sphere(center: point(0, -100.5, -1), radius: 100)

  let matLeft = initMaterial(initLambertian(color(0,0,1)))
  let matRight = initMaterial(initLambertian(color(1,0,0)))

  result.add translate(vec3(-R, 0.0, -1.0), toHittable(Sphere(radius: R), matLeft))
  result.add translate(vec3(R, 0, -1), toHittable(Sphere(radius: R), matRight))

proc mixOfSpheres(): GenericHittablesList =
  result = initGenericHittables()
  let matGround = initMaterial(initLambertian(color(0.8, 0.8, 0.0)))
  let matCenter = initMaterial(initLambertian(color(0.1, 0.2, 0.5)))
  # let matLeft = initMaterial(initMetal(color(0.8, 0.8, 0.8), 0.3))
  let matLeft = initMaterial(initDielectric(1.5))
  let matRight = initMaterial(initMetal(color(0.8, 0.6, 0.2), 1.0))

  result.add translate(vec3(0.0, -100.5, -1), toHittable(Sphere(radius: 100), matGround))
  result.add translate(vec3(0.0, 0.0, -1), toHittable(Sphere(radius: 0.5), matCenter))
  result.add translate(vec3(-1.0, 0.0, -1), toHittable(Sphere(radius: 0.5), matLeft))
  result.add translate(vec3(-1.0, 0.0, -1), toHittable(Sphere(radius: -0.4), matLeft))
  result.add translate(vec3(1.0, 0.0, -1), toHittable(Sphere(radius: 0.5), matRight))

proc randomSpheres(rnd: var Rand, numBalls: int): HittablesList[RGBSpectrum] =
  result = initHittables[RGBSpectrum]()
  for a in -numBalls ..< numBalls:
    for b in -numBalls ..< numBalls:
      let chooseMat = rnd.rand(1.0)
      var center = point(a.float + 0.9 * rnd.rand(1.0), 0.2, b.float + 0.9 * rnd.rand(1.0))

      if (center - point(4, 0.2, 0)).length() > 0.9:
        var sphereMaterial: Material[RGBSpectrum]
        if chooseMat < 0.8:
          # diffuse
          let albedo = rnd.randomVec().Color * rnd.randomVec().Color
          sphereMaterial = initMaterial(initLambertian(albedo))
          result.add translate(center, toHittable(Sphere(radius: 0.2), sphereMaterial))
        elif chooseMat < 0.95:
          # metal
          let albedo = rnd.randomVec(0.5, 1.0).Color
          let fuzz = rnd.rand(0.0 .. 0.5)
          sphereMaterial = initMaterial(initMetal(albedo, fuzz))
          result.add translate(center, toHittable(Sphere(radius: 0.2), sphereMaterial))
        else:
          # glass
          sphereMaterial = initMaterial(initDielectric(1.5))
          result.add translate(center, toHittable(Sphere(radius: 0.2), sphereMaterial))

proc randomScene(rnd: var Rand, useBvh = true, numBalls = 11): GenericHittablesList =
  ## XXX: the BVH is also broken here :) Guess we just broke it completely, haha.
  result = initGenericHittables()

  let groundMaterial = initMaterial(initLambertian(color(0.5, 0.5, 0.5)))
  result.add translate(vec3(0.0, -1000.0, 0.0), toHittable(Sphere(radius: 1000), groundMaterial))

  let smallSpheres = rnd.randomSpheres(numBalls)
  if useBvh:
    result.add toHittable(rnd.initBvhNode(smallSpheres))
  else:
    result.add smallSpheres

  let mat1 = initMaterial(initDielectric(1.5))
  result.add translate(vec3(0.0, 1.0, 0.0), toHittable(Sphere(radius: 1.0), mat1))

  let mat2 = initMaterial(initLambertian(color(0.4, 0.2, 0.1)))
  result.add translate(vec3(-4.0, 1.0, 0.0), toHittable(Sphere(radius: 1.0), mat2))

  let mat3 = initMaterial(initMetal(color(0.7, 0.6, 0.5), 0.0))
  result.add translate(vec3(4.0, 1.0, 0.0), toHittable(Sphere(radius: 1.0), mat3))

proc sceneCast(): GenericHittablesList =
  result = initGenericHittables()

  let groundMaterial = initMaterial(initLambertian(color(0.2, 0.7, 0.2)))
  let EarthR = 6_371_000.0
  result.add translate(vec3(0.0, -EarthR - 5, 0.0), toHittable(Sphere(radius: EarthR), groundMaterial))

  #let concrete = initMaterial(initLambertian(color(0.5, 0.5, 0.5)))
  #let airportWall = initXyRect(-10, 0, 0, 10, 10, mat = concrete)
  #result.add airportWall

  let strMetal = initMaterial(initMetal(color(0.6, 0.6, 0.6), 0.2))
  let telBox = rotateX(toHittable(initBox(point(-2, 1.5, 4), point(0, 1.75, 5.5)),
                                  strMetal), 30.0)
  result.add telBox

  let concreteMaterial = initMaterial(initLambertian(color(0.6, 0.6, 0.6)))
  let controlRoom = toHittable(initBox(point(1, 0.0, 0.0), point(4, 2.2, 2.2)), concreteMaterial)
  result.add controlRoom

  let floorMaterial = initMaterial(initLambertian(color(0.7, 0.7, 0.7)))
  let upperFloor = toHittable(initBox(point(-4, 0.0, -100), point(20, 2.0, 0)), floorMaterial)
  result.add upperFloor

  let glass = initMaterial(initDielectric(1.5))
  let railing = toHittable(initBox(point(-4, 2.0, -0.1), point(10, 2.6, 0)), floorMaterial)
  result.add railing

  let SunR = 695_700_000.0
  let AU = 1.496e11
  let pos = point(AU / 10.0, AU / 10.0, AU).normalize * AU
  echo pos.repr
  let sunMat = initMaterial(initLambertian(color(1.0, 1.0, 0.0)))
  result.add translate(pos, toHittable(Sphere(radius: SunR), sunMat))

  #result.add toHittable(Disk(distance: 3.3, radius: 10.0), concreteMaterial)

proc sceneDisk(): GenericHittablesList =
  result = initGenericHittables()
  let groundMaterial = initMaterial(initLambertian(color(0.2, 0.7, 0.2)))
  result.add toHittable(Disk(distance: 1.5, radius: 1.5), groundMaterial)

proc sceneTest(rnd: var Rand): GenericHittablesList =
  result = initGenericHittables()

  let groundMaterial = initMaterial(initLambertian(color(0.2, 0.7, 0.2)))
  let EarthR = 6_371_000.0
  result.add translate(point(0, -EarthR - 5, 0), toHittable(Sphere(radius: EarthR), groundMaterial))

  let smallSpheres = rnd.randomSpheres(3)
  result.add toHittable(rnd.initBvhNode(smallSpheres))

  let matBox = initMaterial(initLambertian(color(1,0,0)))

  when false:
    let center = -vec3(1.0, 1.75, 5.5) / 2.0
    let telBox1 = rotateX(
      translate(
        initBox(point(0, 0, 0), point(1, 1.75, 5.5), matBox),
        center),
      0.0)
    let telBox2 = rotateX(
      translate(
        initBox(point(0, 0, 0), point(1, 1.75, 5.5), matBox),
        center),
      -50.0)
    result.add telBox1
    result.add telBox2
  elif false:
    let center = vec3(-0.5, -0.5, -0.5)#vec3(0.0, 0.0, 0.0) #vec3(0.5, 0.5, 0.5)
    let telBox1 = rotateZ(
      translate(
        initBox(point(0, 0, 0), point(1, 1, 1), matBox),
        center),
      0.0)
    let telBox2 = rotateZ(
      translate(
        initBox(point(0, 0, 0), point(1, 1, 1), matBox),
        center),
      -50.0)
    result.add telBox1
    result.add telBox2

  let cylMetal = initMaterial(initMetal(color(0.6, 0.6, 0.6), 0.2))
  #let cyl = Cylinder(radius: 3.0, zMin: 0.0, zMax: 5.0, phiMax: 180.0.degToRad), cylMetal)
  let cyl = toHittable(Cone(radius: 3.0, zMax: 4.0, height: 5.0, phiMax: 360.0.degToRad), cylMetal)
  #let cyl = toHittable(Sphere(radius: 3.0), cylMetal)
  let center = vec3(0'f64, 0'f64, 0'f64)#vec3(0.5, 0.5, 0.5)
  let h = rotateX(#cyl,
      translate(
        cyl,
        center),
      90.0)
  result.add h

  #
  #let conMetal = initMaterial(initMetal(color(0.9, 0.9, 0.9), 0.2))
  #let con = translate(vec3(3.0, 3.0, 0.0),
  #                    Cone(radius: 2.0, height: 5.0, zMax: 3.0, phiMax: 180.0.degToRad), conMetal))
  #result.add con

  #let ball0 = translate(vec3(1.0, -2.0, -4.0), toHittable(Sphere(radius: 1.5), strMetal))
  #let ball1 = translate(vec3(1.0, 1.0, 1.0), rotateZ(toHittable(Sphere(radius: 1.5), strMetal), 0.0))
  #let ball2 = translate(vec3(1.0, 1.0, 1.0), rotateZ(toHittable(Sphere(radius: 1.5), strMetal), 30.0))
  #result.add ball0
  #result.add ball2

proc calcHeight(radius, angle: float): float =
  ## Computes the total height of a cone with `angle` and opening `radius`.
  result = radius / tan(angle.degToRad)

from sequtils import mapIt
from std/stats import mean

import sensorBuf, telescopes

proc earth(): Hittable[RGBSpectrum] =
  ## Adds the Earth as a ground. It's only 6.371 km large here :) Why? Who knows.
  let groundMaterial = initMaterial(initLambertian(color(0.2, 0.7, 0.2)))
  const EarthR = 6_371_000.0 # Meter, not MilliMeter, but we keep it...
  result = translate(point(0, -EarthR - 5000, 0), toHittable(Sphere(radius: EarthR), groundMaterial))

import fluxCdf
proc sun(solarModelFile: string): Hittable[XraySpectrum] =
  ## Adds the Sun
  let sunColor = color(1.0, 1.0, 0.5)
  let sunMaterial = if solarModelFile.len == 0:
                      toSpectrum(initMaterial(initDiffuseLight(sunColor)), XraySpectrum)
                    else:
                      let fluxData = getFluxRadiusCDF(solarModelFile)
                      initMaterial(
                        initSolarEmission(sunColor,
                                          fluxData.fRCdf,
                                          fluxData.diffFluxR,
                                          fluxData.radii,
                                          fluxData.energyMin, fluxData.energyMax
                        )
                      )

  const AU = 149_597_870_700_000.0 # by definition since 2012
  var SunR = 696_342_000_000.0 # Solar radius SOHO 2003 & 2006
  if solarModelFile.len == 0:
    ## DTU PhD mentions 3 arcmin source. tan(3' / 2) * 1 AU = 0.0937389
    SunR *= 0.0937389 #0.20 # only inner 20% contribute, i.e. make the sphere smaller for diffuse light
  ## XXX: in principle need to apply correct x AU distance here if `solarModelFile` supplied!
  result = translate(point(0, 0, AU), toHittable(Sphere(radius: SunR), sunMaterial))

proc xrayFinger(tel: Telescope, magnet: Magnet, magnetPos: float, cfg: Config): Hittable[XraySpectrum] =
  ## Adds an X-ray finger
  ## Define the X-ray source
  ## XXX: Add energy information to light source, then assign correct RGB color matching
  ## the energyMin & max values if we don't use skSun!
  case cfg.sourceKind
  of skParallelXrayFinger:
    let sunMaterial = toSpectrum(initMaterial(initLaser(color(1.0, 1.0, 0.5))), XraySpectrum)
    result = translate(point(0, 0, magnetPos + 9.26 * 1000.0), toHittable(Disk(distance: 0.0, radius: magnet.radius), sunMaterial))
    #result = translate(point(0, 0, magnetPos + 9.26 * 1000.0 + 100_000.0), toHittable(Disk(distance: 0.0, radius: magnet.radius), sunMaterial))
  of skXrayFinger: # classical emission allowing all angles
    ## X-ray finger as mentioned in the PhD thesis of A. Jakobsen (14.2 m distance, 3 mm radius), CAST nature mentions
    ## 12 m. Radius is more questionable in both. PANTER: 128m and point source.
    let sunMaterial = toSpectrum(initMaterial(initDiffuseLight(color(1.0, 1.0, 0.5))), XraySpectrum)
    result = translate(point(0, 0, magnetPos + cfg.sourceDistance.to(MilliMeter).float),
                       toHittable(Disk(distance: 0.0, radius: cfg.sourceRadius.float), sunMaterial))
  else: doAssert false, "Invalid branch, not an X-ray finger."

proc source(tel: Telescope, magnet: Magnet, magnetPos: float, cfg: Config): Hittable[XraySpectrum] =
  case cfg.sourceKind
  of skSun: result = sun(cfg.solarModelFile)
  of skXrayFinger, skParallelXrayFinger: result = xrayFinger(tel, magnet, magnetPos, cfg)

proc target(tel: Telescope, magnet: Magnet,
            visibleTarget: bool): Hittable[XraySpectrum] =
  ## Construct a target for the light source. We want to sample towards the end of the magnet
  ## (the side towards the telescope)
  let z = tel.length() ## Position of telescope end on magnet side
  let pink = color(1.0, 0.05, 0.9)
  #result = toHittable(Disk(distance: 0.0, radius: magnet.radius),
  result = toHittable(Disk(distance: 0.0, radius: magnet.radius * 1.2),
                      toSpectrum(toMaterial(initLightTarget(pink, visibleTarget)), XraySpectrum))
    .translate(vec3(0.0, 0.0, z + magnet.length))

proc lightSource(tel: Telescope, magnet: Magnet, magnetPos: float, cfg: Config): GenericHittablesList =
  ## Constructs a light source as well as the optional target if needed.
  result = initGenericHittables()
  result.add source(tel, magnet, magnetPos, cfg)
  if cfg.sourceKind != skParallelXrayFinger:
    result.add target(tel, magnet, cfg.visibleTarget)

proc magnetBore(magnet: Magnet, magnetPos: float): Hittable[RGBSpectrum] =
  let cylMetal = initMaterial(initMetal(color(0.2, 0.2, 0.6), 0.8))
  # The bore is a full cylinder made of metal, slight blue tint
  result = toHittable(Cylinder(radius: magnet.radius, zMin: 0.0, zMax: magnet.length, phiMax: 360.0.degToRad), cylMetal)
    .translate(vec3(0.0, 0.0, magnetPos))

proc windowStrongback(windowRotation, windowZOffset: float): GenericHittablesList =
  ## Implements the window strongback of the Si₃N₄ window used at CAST for the Septemboard detector.
  const
    distance = 2.3 #mm Distance between strips, edge to edge!
    width = 0.5 # mm Width of each strip
    thick = 0.2 # 200 μm thick
    long = 20.0 ## Window has 2cm diameter. In reality circular, but doesn't matter.
  result = initGenericHittables()
  let strMat = initLambertian(color(0.2, 0.2, 0.2))
  for i in 0 ..< 4: ## 4 strips
    let posY = i.float * (distance + width) - 1.5 * (distance + width)
    result.add toHittable(initBox(point(-long/2, -width/2, -thick/2),
                                  point( long/2,  width/2,  thick/2)),
                          toMaterial strMat) # we sink ot so that the box becomes the memory owner of the buffer
      .translate(vec3(0.0, posY, windowZOffset)) ## z offset above chip. XXX: what to choose? we don't simulate gas diffusion, so 0.3cm?
  result = result.rotateZ(windowRotation)

proc llnlFocalPointRay(tel: Telescope, magnet: Magnet, fullTelescope: bool): Ray =
  ## Returns the ray of the given Telescope (actually assuming the LLNL for the time being)
  ## that goes from the telescope center to the focal point. Evaluating it at any point
  ## other than `1` via `at` yields a point along the "focal line" - a point centered
  ## on the (possibly unfocused) image.
  let
    #r1_0 = tel.allR1.mean # [0]
    #α0 = tel.allAngles.mean.degToRad # [0].degToRad
    r1_0 = tel.allR1[0]
    α0 = tel.allAngles[0].degToRad
    lMirror = tel.lMirror
    xSep = tel.xSep
    telCenter = lMirror + xSep / 2 # focal point defined from ``center`` of telescope!
  ## The focal point is at the z axis that defines the cones of the telescope.
  ## - y = 0 is the center of the bore.
  ## - Radius 1 is the radius of the cones on the magnet side. In theory it is
  ##   the offset to the focal point.
  ## - ``center`` of first lowest mirror aligns with bottom of bore. Hence `sin(α0) * lMirror / 2`
  ## For the full telescope it's simply on the z axis, as our bore radius center as well as
  ## cone centers are there.

  ## XXX: I do ``not`` understand why 83 mm gives the correct alignment.
  let yOffset = if fullTelescope: 0.0
                #else: r1_0 - sin(α0) * lMirror / 2 + magnet.radius #(r1_0 - sin(α0) * lMirror / 2.0) + magnet.radius
                #else: r1_0 + sin(α0) * lMirror / 2 #(r1_0 - sin(α0) * lMirror / 2.0) + magnet.radius
                else: 83.0 #r1_0 + sin(α0) * lMirror / 2 #(r1_0 - sin(α0) * lMirror / 2.0) + magnet.radius

  block Sanity:
    echo "TANNNN ", tan(4*α0) * 1530 #tel.focalLength # + 21.5
    echo "Offset should be: ", yOffset
    echo "Corresponds to angle = ", arctan(yOffset / (1500 + xSep / 2 + lMirror / 2)).radToDeg, " from front mirror!"
    #let yOffset
    echo "Corresponds to angle = ", arctan((yOffset + magnet.radius) / 1500).radToDeg, " it is= ", sin(α0) * lMirror / 2

  ## NOTE: expected offset is about -83 mm from telescope center.
  let p = point(0, 0, telCenter)
  var target = point(0.0, -yOffset, - tel.focalLength + telCenter)

  #let FL = 1530.0
  #var target = point(0.0, -yOffset, - FL + telCenter)

  # Construct ray from telescope center to focal point
  result = initRay(p, target - p, rtCamera)

proc llnlFocalPoint(tel: Telescope, magnet: Magnet, rayAt: float, fullTelescope: bool): Point =
  ## Evaluate the ray that goes from telescope center to the focal point at `rayAt`.
  ##
  ## `rayAt` is the position `t` along the ray from the center of the telescope to the focal point
  ## at which we evaluate it for the returned point. This allows to move the point along the
  ## ray such that the point remains in the center of the (possibly unfocused) image.
  let ray = llnlFocalPointRay(tel, magnet, fullTelescope)
  # evaluate it at `rayAt`.
  echo "Old target: ", ray.at(1.0)
  result = ray.at(rayAt)
  echo "New target: ", result

proc imageSensor(tel: Telescope, magnet: Magnet,
                 fullTelescope: bool,
                 cfg: Config,
                 pixelsW = 400,
                 pixelsH = 400,
                 sensorW = 14.0,
                 sensorH = 14.0,
                 sensorThickness = 0.1,
                 posOverride = point(0,0,0),
                 ignoreWindow = false): GenericHittablesList =
  ## This is the correct offset for the focal spot position! It's the center of the cones for the telescope.
  ## XXX: Sanity check:
  ## -> Check all mirrors we install have the exact same position as an offset!!!
  ##
  ## `ignoreWindow` is separate from `cfg` because we may wish to override it!
  let imSensor = toMaterial(initImageSensor(pixelsW, pixelsH, kind = cfg.sensorKind))
  result.add toHittable(initBox(point(-sensorW/2, -sensorH/2, -sensorThickness/2),
                                point( sensorW/2,  sensorH/2,  sensorThickness/2)),
                        imSensor)
  if not ignoreWindow:
    result.add windowStrongback(cfg.windowRotation, cfg.windowZOffset)
  let target = if posOverride != point(0,0,0): posOverride
               else: llnlFocalPoint(tel, magnet, cfg.rayAt, fullTelescope)
  ## Determine the angle the sensor must be rotated to point directly at the telescope
  ## exit instead of straight towards `dir = (0, 0, 1)`.
  let targetRay = llnlFocalPointRay(tel, magnet, fullTelescope = false)
  let zRay = initRay(target, vec3(0.0, 0.0, 1.0), rtCamera)
  let angle = arccos( abs dot(targetRay.dir.normalize, zRay.dir.normalize) ).radToDeg
  result = result
    .rotateX(angle)
    .rotateZ(cfg.telescopeRotation)
    .translate(target)

proc gridLines(tel: Telescope, magnet: Magnet): GenericHittablesList =
  ## Some helper "grid" lines indicating zero x,y along z as well as center of each mirror
  result = initGenericHittables()
  let cylMetal = initMaterial(initMetal(color(0.2, 0.2, 0.6), 0.8))
  let zLine = toHittable(Cylinder(radius: 0.05, zMin: -100.0, zMax: magnet.length, phiMax: 360.0.degToRad), cylMetal)
    .translate(vec3(0.0, -magnet.radius, 0.0)) # 5.0 for xSep + a bit
    #.translate(vec3(0.0, 0.0, 0.0)) # 5.0 for xSep + a bit
  result.add zLine

  ## XXX: this is slightly wrong due to xSep!
  let lMirror = tel.lMirror
  let z0 = tel.lMirror + tel.xSep + tel.lMirror / 2
  let xLine = toHittable(Cylinder(radius: 0.5, zMin: -100.0, zMax: 100, phiMax: 360.0.degToRad), cylMetal)
    .translate(vec3(0.0, -magnet.radius, 0.0))
    .rotateY(90.0)
    .translate(vec3(0.0, 0.0, z0))
  result.add xLine

  let z1 = lMirror / 2
  let xLine2 = toHittable(Cylinder(radius: 0.5, zMin: -100.0, zMax: 100, phiMax: 360.0.degToRad), cylMetal)
    .translate(vec3(0.0, -magnet.radius, 0.0))
    .rotateY(90.0)
    .translate(vec3(0.0, 0.0, z1))
  result.add xLine2

  ## Line that touches the front part of the second set of mirrors, if they are
  ## aligned at the front
  let screen = toMaterial(initDiffuseLight(color(1.0, 1.0, 0.0)))
  let yScreen = toHittable(initBox(point(-magnet.radius * 1.2, -magnet.radius * 1.2, 0.0),
                                   point( magnet.radius * 1.2,  magnet.radius * 1.2, 0.0)),
                        screen)
  let yLine0 = yScreen#toHittable(Cylinder(radius: 0.1, zMin: -100.0, zMax: 100, phiMax: 360.0.degToRad), cylMetal)
    .rotateZ(90.0) # rotate so that it stands up
  result.add yLine0
  let yLine1 = yLine0.translate(vec3(0.0, 0.0, tel.lMirror)) # forward to its position
  result.add yLine1


proc calcYlYsep(angle, xSep, lMirror: float): (float, float, float) =
  ## Helper to compute displacement of each set of mirrors and the y distance
  ## given by the angles due to mirror rotation
  let α = angle.degToRad
  let ySep = tan(α) * (xSep / 2.0) + tan(3 * α) * (xSep / 2.0)
  let yL1  = sin(α) * lMirror
  let yL2  = sin(3 * α) * lMirror
  result = (ySep, yL1, yL2)

proc graphiteSpacer(tel: Telescope, magnet: Magnet, fullTelescope: bool): GenericHittablesList =
  ## The graphite spacer in the middle
  result = initGenericHittables()
  let
    lMirror = tel.lMirror
    xSep = tel.xSep
    graphite = initMaterial(initLambertian(color(0.2, 0.2, 0.2)))
    excessSize = 5.0
    α0 = tel.allAngles[0]
    (ySep, yL1, yL2) = calcYlYsep(α0, xSep, lMirror)
    meanAngle = tel.allAngles.mean
  let gSpacer = toHittable(initBox(point(0, 0, 0), point(2, 2 * magnet.radius + excessSize, lMirror)), graphite)
    .translate(vec3(-1.0, -magnet.radius - excessSize / 2, -lMirror / 2))
  let gSpacer1 = gSpacer
    .rotateX(meanAngle)
    .translate(vec3(0.0, 0.0, lMirror + lMirror / 2 + xSep))
  let gSpacer2 = gSpacer
    .rotateX(3 * meanAngle)
    .translate(vec3(0.0, -(yL1/2 + yL2/2 + ySep), lMirror / 2))
  result.add gSpacer1
  result.add gSpacer2
  if fullTelescope: ## Also add the center blocking disk
    #let imSensorDisk = toMaterial(initImageSensor(400, 400))
    #let centerDisk = toHittable(Disk(distance: 0.0, radius: 500.0), graphite) #imSensorDisk)
    #let centerDisk = XyRect(x0: -boreRadius, x1: boreRadius, y0: -boreRadius, y1: boreRadius), imSensorDisk)  #Disk(distance: 0.0, radius: 500.0), imSensorDisk)
    #let centerDisk = initBox(point(-magnet.radius, -magnet.radius, -0.1), point(magnet.radius, magnet.radius, 0.1), imSensorDisk)
    ## A disk that blocks the inner part where no mirrors cover the bore.
    let centerDisk = toHittable(Disk(distance: 0.0, radius: tel.allR1[0] - sin(tel.allAngles[0].degToRad) * tel.lMirror),
                                graphite)
      .translate(vec3(0.0, 0.0, lMirror * 2 + xSep))
    result.add centerDisk

proc sanityTelescopeOutput(i: int, r1, r5, pos, pos2, angle, lMirror, xSep, yL1, yL2, ySep, boreRadius: float) =
  ## XXX: clean this up and make more useful!
  echo "r1 = ", r1, " r5 = ", r5, " r5 - r1 = ", r5 - r1, " i = ", i, " pos = ", pos, " pos2 = ", pos2
  block Sanity:
    let yCenter = tan(2 * angle.degToRad) * (lMirror + xsep)
    echo "yCenter value = ", yCenter, " compare to 'working' ", - (yL2) - (yL1) - ySep, " compare 'correct' = ", - (yL2/2) - (yL1/2) - ySep
    let yOffset = (r1 - (sin(angle.degToRad) * lMirror) / 2.0) + boreRadius - pos
    echo "Offset for layer ", i, " should be: ", yOffset, " \n==============================\n\n"

from std/algorithm import lowerBound
proc llnlLayerMaterial(layer: int, refl: Reflectivity,
                       usePerfectMirror, brokenMirror: bool,
                       c: Color): Material[XraySpectrum] =
  ## Imperfect value assuming a 'figure error similar to NuSTAR of 1 arcmin'
  ## -> tan(1 ArcMin) (because fuzz added to unit vector)
  const ImperfectVal = 0.0002908880082045767

  let idx = refl.layers.lowerBound(layer)
  let R = refl.interp[idx]
  let fuzz = if usePerfectMirror: 0.0 else: ImperfectVal
  ## NOTE: we use the 1 - reflectivity as a placeholder for the real transmission. This _should_ be
  ## correct, but in reality there will be mismatches
  result = toMaterial initXrayMatter(c, fuzz, R, 1.0 - R)

proc brokenMirror(c: Color): Material[RGBSpectrum] =
  # broken mirror simply uses a Lambertian to effectively disable reflections through the telescope
  result = toMaterial initLambertian(c)

proc llnlTelescope(tel: Telescope, magnet: Magnet, fullTelescope: bool, cfg: Config): GenericHittablesList =
  ## Constructs the actual LLNL telescope
  ##
  ## Pos for the 'first' layers should be correct, under the following two conditions:
  ## 1. it's not entirely clear what part of the telescope really should align with the bottom edge of the magnet.
  ##    Currently the center of the front shell is aligned with the bottom.
  ## 2. it's not clear what part of the shells should align vertically as discussed with Julia and Jaime in Cristinas office.
  ##    The front? The center? The back? Currently I align at the center.
  proc cone[S: SomeSpectrum](r, h, zMax: float, tel: Telescope, mat: Material[S]): Hittable[S] =
    ## Important: the `zMax` of a cone is literally its maximum height along the `z` axis. This means
    ## it is _not_ lMirror!
    result = toHittable(Cone(radius: r, height: h, zMax: zMax,
                             phiMax: tel.mirrorSize.degToRad),
                        mat)
  proc cyl[S: SomeSpectrum](r, h: float, tel: Telescope, mat: Material[S]): Hittable[S] =
    result = toHittable(Cylinder(radius: r, zMin: 0.0, zMax: tel.lMirror,
                                 phiMax: tel.mirrorSize.degToRad),
                        mat)
  proc disk(r, thick: float, tel: Telescope, mat: Material[RGBSpectrum]): Hittable[RGBSpectrum] =
    result = toHittable(Disk(distance: 0.0, radius: r + thick, innerRadius: r, phiMax: tel.mirrorSize.degToRad), mat)


  proc transform[S: SomeSpectrum](h: Hittable[S], xOrigin, yOffset, z: float, tel: Telescope): Hittable[S] =
    result = h.rotateZ(tel.mirrorSize / 2.0) # rotate out half the miror size to center "top" of mirror
      .translate(vec3(xOrigin, 0.0, -tel.lMirror / 2.0)) # move to its center
      #.rotateY(angle) ## For a cylinder telescope
      .rotateX(180.0) # we consider from magnet! This also means the magnet side of the telescope is aligned!
      .rotateZ(-90.0)
      .translate(vec3(0.0, yOffset, z + tel.lMirror / 2.0)) # move to its final position

  proc setCone[S: SomeSpectrum](r, angle, y, z: float,
                                fullTelescope, ignoreMirrorThickness: bool,
                                mirrorThickness: float,
                                tel: Telescope, magnet: Magnet, mat: Material[S]): GenericHittablesList =
    # `xOffset` is the displacement from front to center of mirror (x because cone with `phiMax`
    # starts from x = 0)
    result = initGenericHittables()
    let xOffset = (sin(angle.degToRad) * tel.lMirror) / 2.0
    let height = calcHeight(r, angle) # total height of the cone that yields required radius and angle

    let zMax = cos(angle.degToRad) * tel.lMirror
    let c = cone(r, height, zMax, tel, mat)
    #let c = cyl(r, height) ## To construct a fake telescope
    # for the regular telescope first move to -r + xOffset to rotate around center of layer. Full no movement
    let xOrigin = if fullTelescope: 0.0 else: -r + xOffset # aligns *center* of mirror
    let yOffset = if fullTelescope: 0.0 else: y - magnet.radius # move down by bore radius & offset
    result.add c.transform(xOrigin, yOffset, z, tel)
    if not ignoreMirrorThickness:
      ## Sets the outer casing so to say of each mirror. That is a `Disk` at the front
      ## which blocks some light and the "upper" part. Both are opaque to X-rays.
      let mat = initMaterial(initLambertian(color(0.2, 0.2, 0.2)))
      let thick =
        if classify(mirrorThickness) != fcInf: mirrorThickness
        else: tel.glassThickness ## Default 0.21 mm
      # Front: a piece of a disk (= a ring) of `thick` thickness
      let d = disk(r, thick, tel, mat)
      # Top: another cone equivalent to `c` above, just moved up by `thick`
      let c = cone(r, height, zMax, tel, mat)
      result.add d.transform(xOrigin, yOffset, z, tel)
      result.add c.transform(xOrigin, yOffset + thick, z, tel)

  const sanity = false
  let
    lMirror = tel.lMirror
    xSep = tel.xSep
  let reflectivity = setupReflectivity(cfg.energyMin, cfg.energyMax, NumSamples) ## `fluxCDF.nim`
  let r1_0 = tel.allR1[0]
  for i in 0 ..< tel.allR1.len:
    let
      r1 = tel.allR1[i]
      r5 = tel.allR5[i]
      angle = tel.allAngles[i] ## * 1.02 yields 1500mm focal length
      r4 = r5 + lMirror * sin(3.0 * angle.degToRad)
    let (ySep, yL1, yL2) = calcYlYsep(angle, xSep, lMirror)
    # `pos`, `pos2` are the `y` positions of first & second set of mirrors.
    let pos1 = (r1 - r1_0) ## Only shift each layer relative to first layer. Other displacement done in `setCone`
    let pos2 = pos1 - yL1 / 2.0 - yL2 / 2.0 - ySep
    # cone of the front (= towards magnet) shell
    template con1[S: SomeSpectrum](mat: Material[S]): untyped =
      setCone(r1, angle,     pos1, lMirror + xSep,
              fullTelescope, cfg.ignoreMirrorThickness, cfg.mirrorThickness,
              tel, magnet, mat)
    # cone of the rear (= towards detector) shell
    template con2[S: SomeSpectrum](mat: Material[S]): untyped =
      ## NOTE: using `r1` as well reproduces the X-ray finger results from the _old_ raytracer!
      setCone(r4, 3 * angle, pos2, 0.0,
              fullTelescope, cfg.ignoreMirrorThickness, cfg.mirrorThickness,
              tel, magnet, mat)
    ## Add mirrors with correct materials (different types!)
    if not cfg.brokenMirrors:
      let mat = llnlLayerMaterial(i, reflectivity, cfg.usePerfectMirror, cfg.brokenMirrors, color(1,0,0))
      result.add con1(mat)
      result.add con2(mat)
    else:
      result.add con1(brokenMirror(color(1,0,0)))
      result.add con2(brokenMirror(color(1,0,0)))
    if sanity:
      sanityTelescopeOutput(i, r1, r5, pos1, pos2, angle, lMirror, xSep, yL1, yL2, ySep, magnet.radius)

proc initSetup(fullTelescope: bool): (Telescope, Magnet) =
  if not fullTelescope:
    const
      boreRadius = 43.0 / 2 # mm
      length = 9.26 # m
      telescopeMagnetZOffset = 1.0 # 1 mm
      mirrorSize = 30.0 # 30 degree mirrors
    result = (initTelescope(tkLLNL, mirrorSize), initMagnet(boreRadius, length))
  else:
    const
      length = 9.26 # m
      telescopeMagnetZOffset = 1.0 # 1 mm
      mirrorSize = 360.0 # entire cones for the full telescope
    let llnl = initTelescope(tkLLNL, mirrorSize)
    ## For the full telescope the bore radius is the size of the outer most largest cone
    let boreRadius = llnl.allR1[^1]
    let magnet = initMagnet(boreRadius, length)
    result = (llnl, magnet)

proc sceneLLNL(rnd: var Rand, cfg: Config): GenericHittablesList =
  ## Mirrors centered at lMirror/2.
  ## Entire telescope
  ##   lMirror  xsep  lMirror
  ## |         |----|         | ( Magnet bore       )
  ##      ^-- center 1
  ##                     ^-- center 2
  ## ^-- z = 0
  ## due to rotation around the *centers* of the mirrors, the real x separation increases.
  ## `xSep = 4 mm` exactly, `lMirror = 225 mm` and the correct cones are computed based on the
  ## given angles and the R1, R5 values. R4 is the relevant radius of the cone for the
  ## second set of mirrors. Therefore R1 and R4 are actually relevant.
  result = initGenericHittables()
  ## Constants fixed to *this* specific setup
  let (llnl, magnet) = initSetup(fullTelescope = false)
  const telescopeMagnetZOffset = 1.0 # 1 mm
  let magnetPos = llnl.length + telescopeMagnetZOffset

  var objs = initGenericHittables()

  objs.add earth()
  objs.add lightSource(llnl, magnet, magnetPos, cfg)
  if not cfg.ignoreMagnet:
    objs.add magnetBore(magnet, magnetPos)
  if cfg.gridLines:
    objs.add gridLines(llnl, magnet)
  var telescopeImSensor = initGenericHittables()
  # Add the image sensor and potentially window strongback
  telescopeImSensor.add imageSensor(llnl, magnet, fullTelescope = false, cfg = cfg,
                                    pixelsW = 1000, pixelsH = 1000,
                                    sensorW = 14.0, sensorH = 14.0, ignoreWindow = cfg.ignoreWindow)
    .translate(vec3(0.1, 0.1, 0.0))
  if not cfg.ignoreSpacer:
    telescopeImSensor.add graphiteSpacer(llnl, magnet, fullTelescope = false)
  telescopeImSensor.add llnlTelescope(llnl, magnet, fullTelescope = false, cfg = cfg)
  objs.add telescopeImSensor
    .rotateZ(cfg.setupRotation - cfg.telescopeRotation)

  if cfg.midTelescopeSensor: # sensor placed between both telescope mirror sets. Useful to debug image after set 1
    objs.add imageSensor(llnl, magnet, fullTelescope = false, cfg = cfg,
                         pixelsW = 1000, pixelsH = 1000,
                         sensorW = magnet.radius * 2, sensorH = magnet.radius * 2,
                         posOverride = point(0.0, 0.0, llnl.lMirror + 2.0),
                         ignoreWindow = true)
                         pixelsW = 1000, pixelsH = 1000,
                         sensorW = magnet.radius * 2, sensorH = magnet.radius * 2,
                         ignoreWindow = true,
                         sensorKind = sensorKind,
                         posOverride = point(0.0, 0.0, llnl.lMirror + 2.0))

  ## XXX: BVH node of the telescope is currently broken! Bad shading.
  #result.add telescope #rnd.initBvhNode(telescope)
  result.add objs

proc sceneLLNLTwice(rnd: var Rand, cfg: Config): GenericHittablesList =
  ## Mirrors centered at lMirror/2.
  ## Entire telescope
  ##   lMirror  xsep  lMirror
  ## |         |----|         | ( Magnet bore       )
  ##      ^-- center 1
  ##                     ^-- center 2
  ## ^-- z = 0
  ## due to rotation around the *centers* of the mirrors, the real x separation increases.
  ## `xSep = 4 mm` exactly, `lMirror = 225 mm` and the correct cones are computed based on the
  ## given angles and the R1, R5 values. R4 is the relevant radius of the cone for the
  ## second set of mirrors. Therefore R1 and R4 are actually relevant.
  result = initGenericHittables()
  ## Constants fixed to *this* specific setup
  let (llnl, magnet) = initSetup(fullTelescope = false)
  const telescopeMagnetZOffset = 1.0 # 1 mm
  let magnetPos = llnl.length + telescopeMagnetZOffset

  var objs = initGenericHittables()

  objs.add earth()
  if cfg.gridLines:
    objs.add gridLines(llnl, magnet)
  objs.add imageSensor(llnl, magnet, fullTelescope = true, cfg = cfg) ## Don't want y displacement, the two bores are rotated by 90°
                                                           ## so focal spot is on y = 0
  var telMag = initGenericHittables()
  case cfg.sourceKind
  of skSun: objs.add sun(cfg.solarModelFile) ## Only a single sun in center x/y! Hence add to `objs` so not duplicated
  of skXrayFinger, skParallelXrayFinger:
    telMag.add xrayFinger(llnl, magnet, magnetPos, cfg)
  # Need two targets in both source cases
  telMag.add target(llnl, magnet, cfg.visibleTarget)
  telMag.add magnetBore(magnet, magnetPos)
  if not cfg.ignoreSpacer:
    telMag.add graphiteSpacer(llnl, magnet, fullTelescope = false)
  telMag.add llnlTelescope(llnl, magnet, fullTelescope = false, cfg = cfg)

  ## XXX: FIX THE -83!
  objs.add telMag.rotateZ(-90.0)
    .translate(vec3(-83.0, 0.0, 0.0))
  objs.add telMag.rotateZ(90.0)
    .translate(vec3(83.0, 0.0, 0.0))
  ## XXX: BVH node of the telescope is currently broken! Bad shading.
  #result.add telescope #rnd.initBvhNode(telescope)
  result.add objs

proc sceneLLNLFullTelescope(rnd: var Rand, cfg: Config): GenericHittablesList =
  ## Mirrors centered at lMirror/2.
  ## Entire telescope
  ##   lMirror  xsep  lMirror
  ## |         |----|         | ( Magnet bore       )
  ##      ^-- center 1
  ##                     ^-- center 2
  ## ^-- z = 0
  ## due to rotation around the *centers* of the mirrors, the real x separation increases.
  ## `xSep = 4 mm` exactly, `lMirror = 225 mm` and the correct cones are computed based on the
  ## given angles and the R1, R5 values. R4 is the relevant radius of the cone for the
  ## second set of mirrors. Therefore R1 and R4 are actually relevant.
  result = initGenericHittables()
  ## Constants fixed to *this* specific setup
  let (llnl, magnet) = initSetup(fullTelescope = true)
  const telescopeMagnetZOffset = 1.0 # 1 mm
  let magnetPos = llnl.length + telescopeMagnetZOffset

  var objs = initGenericHittables()
  objs.add earth()
  objs.add lightSource(llnl, magnet, magnetPos, cfg)
  objs.add magnetBore(magnet, magnetPos)
  if cfg.gridLines:
    objs.add gridLines(llnl, magnet)
  objs.add imageSensor(llnl, magnet, fullTelescope = true, cfg = cfg)

  var telescope = initGenericHittables()
  if not cfg.ignoreSpacer:
    telescope.add graphiteSpacer(llnl, magnet, fullTelescope = true)
  telescope.add llnlTelescope(llnl, magnet, fullTelescope = true, cfg = cfg)
  objs.add telescope
  ## XXX: BVH node of the telescope is currently broken! Bad shading.
  #result.add telescope #rnd.initBvhNode(telescope)
  result.add objs

proc calcEnergyRange(min, max, energy: float): (float, float) =
  const ΔE = 0.05 # rangle all `NumSamples` in this range
  if classify(energy) != fcInf:
    result = (energy - ΔE, energy + ΔE)
  else:
    result = (max(min, 0.03), max)
  echo "[INFO] Using energy range: ", result, " keV."

proc main(width = 600,
          maxDepth = 5,
          speed = 1.0,
          speedMul = 1.1,
          spheres = false,
          llnl = false,
          llnlTwice = false,
          fullTelescope = false,
          axisAligned = false,
          focalPoint = false,
          vfov = 90.0,
          numRays = 100,
          visibleTarget = false,
          gridLines = false,
          usePerfectMirror = true,
          sourceKind = skSun,
          solarModelFile = "",
          nJobs = 16,
          rayAt = 1.0,
          setupRotation = 90.0,
          telescopeRotation = 14.17,
          windowRotation = -30.0,
          windowZOffset = 3.0,
          ignoreWindow = false,
          sensorKind = sCount,
          brokenMirrors = false, # disables telescope mirror reflection for debugging
          midTelescopeSensor = false,
          endTelescopeSensor = false,
          ignoreMirrorThickness = false,
          mirrorThickness = Inf, # adjust mirror thickenss. 0.2 by default
          ignoreSpacer = false,
          energyMin = 0.0, # keV
          energyMax = 15.0, # keV
          energy = Inf, # If given produce only signals at *this* energy. Overrides min / max
          sourceDistance = 14.2.m,
          sourceRadius = 3.0.mm
         ) =
  # Image
  THREADS = nJobs
  #const ratio = 16.0 / 9.0 #16.0 / 9.0
  const ratio = 1.0
  ## XXX: It's about time for a `Config` or `Context` object to store all the parameters...
  let img = Image(width: width, height: (width.float / ratio).int)
  let samplesPerPixel = 100
  var rnd = initRand(0x299792458)

  # determine the correct energy ranges
  let (energyMin, energyMax) = calcEnergyRange(energyMin, energyMax, energy)

  let cfg = initConfig(visibleTarget, gridLines, usePerfectMirror, sourceKind, solarModelFile,
                       energyMin, energyMax,
                       rayAt,
                       setupRotation, telescopeRotation, windowRotation, windowZOffset, ignoreWindow,
                       sensorKind,
                       brokenMirrors, # disables telescope mirror reflection for debugging
                       midTelescopeSensor, endTelescopeSensor,
                       ignoreMirrorThickness,
                       mirrorThickness,# adjust mirror thickenss. 0.2 by default
                       ignoreSpacer,
                       sourceDistance, sourceRadius)
  # World
  ## Looking at mirrors!
  var lookFrom: Point = point(1,1,1)
  var lookAt: Point = point(0,0,0)
  var world: GenericHittablesList
  if llnl:
    if axisAligned:
      lookFrom = point(0.0, 0.0, -100.0) #point(-0.5, 3, -0.5)#point(3,3,2)
      lookAt = point(0.0, 0.0, 0.0) #point(0, 1.5, 2.5)#point(0,0,-1)
    elif focalPoint:
      let (llnl, magnet) = initSetup(fullTelescope)
      let ray = llnlFocalPointRay(llnl, magnet, fullTelescope)
      (lookFrom, lookAt) = (ray.at(rayAt).rotateZ(-(setupRotation - telescopeRotation)), ray.at(0.0))
    else:
      #lookFrom = point(172.2886370206074, 58.69754358408407, -14.3630844062124) #point(-0.5, 3, -0.5)#point(3,3,2)
      #lookAt = point(171.4358852563132, 58.70226619735943, -13.84078935031287) #point(0, 1.5, 2.5)#point(0,0,-1)
      #lookFrom = point(-1262.787318591972, 33.30606935408561, -338.5357032373016)
      #lookAt = point(-1262.004708698569, 33.22914198500523, -339.1534442304647)

      #lookAt = point(-1262.761916124543, 33.31906554264295, -337.536110413332)
      #lookFrom = point(-1262.787318591972, 33.30606935408561, -338.5357032373016)
      lookAt = point(-41.23722667383358, -77.42138803689321, -1253.03504313943)
      lookFrom = point(-42.12957189166389, -77.3883974697615, -1252.584896902418)

    if fullTelescope:
      world = rnd.sceneLLNLFullTelescope(cfg)
    elif llnlTwice:
      world = rnd.sceneLLNLTwice(cfg)
    else:
      echo "Visible target? ", visibleTarget
      world = rnd.sceneLLNL(cfg)

  elif spheres:
    world = rnd.randomScene(useBvh = false) ## Currently BVH broken
    lookFrom = point(0,1.5,-2.0)
    lookAt = point(0,0.0,0)
  else:
    world = rnd.sceneTest()
    lookFrom = point(-1, 5.0, -4) #point(-0.5, 3, -0.5)#point(3,3,2)
    lookAt = point(1.0, 3.0, 2.0) #point(0, 1.5, 2.5)#point(0,0,-1)
  #let lookFrom = point(0,1.5,-2.0)
  #let lookAt = point(0,0.0,0)
  let vup = vec3(0.0,1.0,0.0)
  let distToFocus = 10.0 #(lookFrom - lookAt).length()
  let aperture = 0.0
  let defocusAngle = 0.0
  let camera = initCamera(lookFrom, lookAt, vup, vfov = vfov,
                          aspectRatio = ratio,
                          #aperture = aperture,
                          width = width,
                          defocusAngle = defocusAngle,
                          focusDist = distToFocus,
                          #background = color(0,0,0))# color(0.5, 0.7, 1.0)) # default background
                          background = color(0.5, 0.7, 1.0)) # default background

  # Rand seed
  randomize(0xE7)

  # Render (image)
  let fname = &"/tmp/render_width_{width}_samplesPerPixel_{samplesPerPixel}.ppm"
  #img.renderMC(fname, world, camera, samplesPerPixel, maxDepth)
  img.renderSdl(world, rnd, camera, samplesPerPixel, maxDepth,
                speed = speed, speedMul = speedMul,
                numRays = numRays)

when isMainModule:
  import cligen
  import unchained / cligenParseUnits
  dispatch main
