import strformat, os, terminal, macros, math, random, times

import basetypes, hittables, camera

import arraymancer

import sdl2 except Color, Point

type
  RenderContext = ref object
    rnd: Rand
    camera: Camera
    world: HittablesList
    worldNoSources: HittablesList ## Copy of the world without any light sources.
    sources: HittablesList ## List of all light sources
    targets: HittablesList ## List of all targets for diffuse lights
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

var Tracing = ttCamera

proc initRenderContext(rnd: var Rand,
                       buf: ptr UncheckedArray[uint32], counts: ptr UncheckedArray[int],
                       window: SurfacePtr, numRays, width, height: int,
                       camera: Camera, world: HittablesList, maxDepth: int,
                       numPer, numThreads: int): RenderContext =
  var world = world
  var worldNoSources = world.clone()
  let sources = worldNoSources.getSources(delete = true)
  let targets = worldNoSources.getLightTargets(delete = true)
  # filter invisible targets
  world.removeInvisibleTargets()
  result = RenderContext(rnd: rnd,
                         buf: buf, counts: counts,
                         window: window,
                         numRays: numRays, width: width, height: height,
                         camera: camera,
                         world: world,
                         worldNoSources: worldNoSources,
                         sources: sources,
                         targets: targets,
                         maxDepth: maxDepth,
                         numPer: numPer, numThreads: numThreads)

proc initRenderContext(rnd: var Rand,
                       buf: var Tensor[uint32], counts: var Tensor[int],
                       window: SurfacePtr, numRays, width, height: int,
                       camera: Camera, world: HittablesList, maxDepth: int,
                       numPer: int = -1, numThreads: int = -1): RenderContext =
  let bufP    = cast[ptr UncheckedArray[uint32]](buf.unsafe_raw_offset())
  var countsP = cast[ptr UncheckedArray[int]](counts.unsafe_raw_offset())
  result = initRenderContext(rnd, bufP, countsP, window, numRays, width, height, camera, world, maxDepth, numPer, numThreads)

proc initRenderContexts(numThreads: int,
                        buf: var Tensor[uint32], counts: var Tensor[int],
                        window: SurfacePtr, numRays, width, height: int,
                        camera: Camera, world: HittablesList, maxDepth: int): seq[RenderContext] =
  result = newSeq[RenderContext](numThreads)
  let numPer = (width * height) div numThreads
  for i in 0 ..< numThreads:
    let bufP    = cast[ptr UncheckedArray[uint32]](buf.unsafe_raw_offset()[i * numPer].addr)
    let countsP = cast[ptr UncheckedArray[int]](counts.unsafe_raw_offset()[i * numPer].addr)
    ## XXX: clone world?
    var rnd = initRand(i * 0xfafe)
    result[i] = initRenderContext(rnd, bufP, countsP, window, numRays, width, height, camera.clone(), world.clone(), maxDepth, numPer, numThreads)
when compileOption("threads"):
  import weave
var THREADS = 16

proc rayColor*(c: Camera, rnd: var Rand, r: Ray, world: HittablesList, depth: int): Color {.gcsafe.} =
  var rec: HitRecord

  if depth <= 0:
    return color(0, 0, 0)

  if world.hit(r, 0.001, Inf, rec):
    var scattered: Ray
    var attenuation: Color
    var emitted = rec.mat.emit(rec.u, rec.v, rec.p)
    if not rec.mat.scatter(rnd, r, rec, attenuation, scattered):
      result = emitted
    else:
      result = attenuation * c.rayColor(rnd, scattered, world, depth - 1) + emitted
  else:
    result = c.background

  when false: ## Old code with background gradient
    let unitDirection = unitVector(r.dir)
    let t = 0.5 * (unitDirection.y + 1.0)
    result = (1.0 - t) * color(1.0, 1.0, 1.0) + t * color(0.5, 0.7, 1.0)

proc rayColorAndPos*(c: Camera, rnd: var Rand, r: Ray, initialColor: Color, world: HittablesList, depth: int): (Color, float, float) {.gcsafe.} =
  var rec: HitRecord

  echo "Start==============================\n\n"
  proc recurse(rec: var HitRecord, c: Camera, rnd: var Rand, r: Ray, world: HittablesList, depth: int): Color =
    echo "Depth = ", depth
    if depth <= 0:
      return color(0, 0, 0)

    #var color: Color = initialColor
    result = initialColor
    if world.hit(r, 0.001, Inf, rec):
      #echo "Hit: ", rec.p, " mat: ", rec.mat, " at depth = ", depth, " rec: ", rec
      var scattered: Ray
      var attenuation: Color
      var emitted = rec.mat.emit(rec.u, rec.v, rec.p)
      if rec.mat.kind == mkImageSensor:
        result = color(1,1,1) ## Here we return 1 so that the function call above terminates correctly
        discard
      elif not rec.mat.scatter(rnd, r, rec, attenuation, scattered):
        result = emitted
      else:
        let angle = arccos(scattered.dir.dot(rec.normal)).radToDeg
        echo "Scattering angle : ", angle
        result = attenuation * recurse(rec, c, rnd, scattered, world, depth - 1) + emitted
        #let res =
        #if rec.mat.kind == mkImageSensor:
        #
        #else:
        #  result = attenuation * res + emitted
    else:
      result = c.background

  let color = recurse(rec, c, rnd, r, world, depth)
  echo "------------------------------Finish\n\n"
  if rec.mat.kind == mkImageSensor: # and color != initialColor:
    ## In this case return color and position
    echo "Initial color? ", color, " rec.mat: ", rec.mat, " at ", (rec.u, rec.v)
    result = (color, rec.u, rec.v)
  else: # else just return nothing
    result = (color(0,0,0), 0, 0)

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

proc sampleRay(rnd: var Rand, sources, targets: HittablesList): (Ray, Color) {.gcsafe.} =
  ## Sample a ray from one of the sources
  # 1. pick a sources
  let num = sources.len
  let idx = if num == 1: 0 else: rnd.rand(num - 1) ## XXX: For now uniform sampling between sources!
  let source = sources[idx]
  # 2. sample from source
  let p = samplePoint(source, rnd)
  # 3. get the color of the source
  let initialColor = source.getMaterial.emit(0.5, 0.5, p)

  # 4. depending on the source material choose direction
  case source.getMaterial.kind
  of mkLaser: # lasers just sample along the normal of the material
    let dir = vec3(0.0, 0.0, -1.0) ## XXX: make this the normal surface!
    result = (initRay(p, dir, rtLight), initialColor)
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
    result = (ray, initialColor)

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
      color = camera.rayColor(ctx.rnd, r, ctx.world, maxDepth)
      yIdx = y #height - y - 1
      xIdx = x
    of ttLights:
      # 1. get a ray from a source
      let (r, initialColor) = ctx.rnd.sampleRay(ctx.sources, ctx.targets)
      # 2. trace it. Check if ray ended up on `ImageSensor`
      let (c, u, v) = camera.rayColorAndPos(ctx.rnd, r, initialColor, ctx.world, maxDepth)
      if c.r == 0 and c.g == 0 and c.b == 0: continue # skip to next ray!
      #echo r
      #echo "empty?? ", c, " at ", (u, v)

      # 3. if so, get the relative position on sensor, map to x/y
      color = c
      xIdx = clamp((u * (width.float - 1.0)).round.int, 0, width)
      yIdx = clamp((v * (height.float - 1.0)).round.int, 0, height)

    when true:
      # 1. get a ray from a source
      for _ in 0 ..< 1:
        let (r, initialColor) = ctx.rnd.sampleRay(ctx.sources, ctx.targets)
        # 2. trace it
        let c = camera.rayColor(ctx.rnd, r, ctx.worldNoSources, maxDepth)
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
    let color = camera.rayColor(ctx.rnd, r, ctx.world, maxDepth)

    block LightSources:
      # 1. get a ray from a source
      for _ in 0 ..< 1:
        let (r, initialColor) = ctx.rnd.sampleRay(ctx.sources, ctx.targets)
        # 2. trace it
        let c = camera.rayColor(ctx.rnd, r, ctx.worldNoSources, maxDepth)
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


proc renderSdl*(img: Image, world: var HittablesList,
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
        of SDL_SCANCODE_BACKSPACE:
          echo "Resetting view!"
          camera.updateLookFromAt(origLookFrom, origLookAt)
          resetBufs(bufT, counts)
        of SDL_SCANCODE_N:
          ## activate free movement (n for noclip ;))
          movementIsFree = not movementIsFree
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
            echo "New view from : ", camera.lookFrom, ", yaw = ", nYaw, ", pitch = ", nPitch
            camera.updateYawPitchRoll(camera.lookFrom, nYaw, nPitch, 0.0)
            echo "Now looking at: ", camera.lookAt
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
        echo "[INFO] Current position (lookFrom) = ", camera.lookFrom
        lastLookFrom = camera.lookFrom
      init(Weave)
      parallelFor j in 0 ..< THREADS:
        captures: {ctxSeq}
        renderFrame(j, ctxSeq[j])
      exit(Weave)
      ctxSeq.updateCamera(camera)
      copyBuf(bufT, window)

    unlockSurface(window)
    #sdl2.clear(arg.renderer)
    sdl2.present(renderer)
  sdl2.quit()

proc sceneRedBlue(): HittablesList =
  result = initHittables(0)
  let R = cos(Pi/4.0)

  #world.add Sphere(center: point(0, 0, -1), radius: 0.5)
  #world.add Sphere(center: point(0, -100.5, -1), radius: 100)

  let matLeft = initMaterial(initLambertian(color(0,0,1)))
  let matRight = initMaterial(initLambertian(color(1,0,0)))

  result.add translate(vec3(-R, 0.0, -1.0), Sphere(radius: R, mat: matLeft))
  result.add translate(vec3(R, 0, -1), Sphere(radius: R, mat: matRight))

proc mixOfSpheres(): HittablesList =
  result = initHittables(0)
  let matGround = initMaterial(initLambertian(color(0.8, 0.8, 0.0)))
  let matCenter = initMaterial(initLambertian(color(0.1, 0.2, 0.5)))
  # let matLeft = initMaterial(initMetal(color(0.8, 0.8, 0.8), 0.3))
  let matLeft = initMaterial(initDielectric(1.5))
  let matRight = initMaterial(initMetal(color(0.8, 0.6, 0.2), 1.0))

  result.add translate(vec3(0.0, -100.5, -1), Sphere(radius: 100, mat: matGround))
  result.add translate(vec3(0.0, 0.0, -1), Sphere(radius: 0.5, mat: matCenter))
  result.add translate(vec3(-1.0, 0.0, -1), Sphere(radius: 0.5, mat: matLeft))
  result.add translate(vec3(-1.0, 0.0, -1), Sphere(radius: -0.4, mat: matLeft))
  result.add translate(vec3(1.0, 0.0, -1), Sphere(radius: 0.5, mat: matRight))

proc randomSpheres(rnd: var Rand, numBalls: int): HittablesList =
  result = initHittables(0)
  for a in -numBalls ..< numBalls:
    for b in -numBalls ..< numBalls:
      let chooseMat = rnd.rand(1.0)
      var center = point(a.float + 0.9 * rnd.rand(1.0), 0.2, b.float + 0.9 * rnd.rand(1.0))

      if (center - point(4, 0.2, 0)).length() > 0.9:
        var sphereMaterial: Material
        if chooseMat < 0.8:
          # diffuse
          let albedo = rnd.randomVec().Color * rnd.randomVec().Color
          sphereMaterial = initMaterial(initLambertian(albedo))
          result.add translate(center, Sphere(radius: 0.2, mat: sphereMaterial))
        elif chooseMat < 0.95:
          # metal
          let albedo = rnd.randomVec(0.5, 1.0).Color
          let fuzz = rnd.rand(0.0 .. 0.5)
          sphereMaterial = initMaterial(initMetal(albedo, fuzz))
          result.add translate(center, Sphere(radius: 0.2, mat: sphereMaterial))
        else:
          # glass
          sphereMaterial = initMaterial(initDielectric(1.5))
          result.add translate(center, Sphere(radius: 0.2, mat: sphereMaterial))

proc randomScene(rnd: var Rand, useBvh = true, numBalls = 11): HittablesList =
  result = initHittables(0)

  let groundMaterial = initMaterial(initLambertian(color(0.5, 0.5, 0.5)))
  result.add translate(vec3(0.0, -1000.0, 0.0), Sphere(radius: 1000, mat: groundMaterial))

  let smallSpheres = rnd.randomSpheres(numBalls)
  if useBvh:
    result.add rnd.initBvhNode(smallSpheres)
  else:
    result.add smallSpheres

  let mat1 = initMaterial(initDielectric(1.5))
  result.add translate(vec3(0.0, 1.0, 0.0), Sphere(radius: 1.0, mat: mat1))

  let mat2 = initMaterial(initLambertian(color(0.4, 0.2, 0.1)))
  result.add translate(vec3(-4.0, 1.0, 0.0), Sphere(radius: 1.0, mat: mat2))

  let mat3 = initMaterial(initMetal(color(0.7, 0.6, 0.5), 0.0))
  result.add translate(vec3(4.0, 1.0, 0.0), Sphere(radius: 1.0, mat: mat3))

proc sceneCast(): HittablesList =
  result = initHittables(0)

  let groundMaterial = initMaterial(initLambertian(color(0.2, 0.7, 0.2)))
  let EarthR = 6_371_000.0
  result.add translate(vec3(0.0, -EarthR - 5, 0.0), Sphere(radius: EarthR, mat: groundMaterial))

  #let concrete = initMaterial(initLambertian(color(0.5, 0.5, 0.5)))
  #let airportWall = initXyRect(-10, 0, 0, 10, 10, mat = concrete)
  #result.add airportWall

  let strMetal = initMaterial(initMetal(color(0.6, 0.6, 0.6), 0.2))
  let telBox = rotateX(initBox(point(-2, 1.5, 4), point(0, 1.75, 5.5), strMetal), 30.0)
  result.add telBox

  let concreteMaterial = initMaterial(initLambertian(color(0.6, 0.6, 0.6)))
  let controlRoom = initBox(point(1, 0.0, 0.0), point(4, 2.2, 2.2), concreteMaterial)
  result.add controlRoom

  let floorMaterial = initMaterial(initLambertian(color(0.7, 0.7, 0.7)))
  let upperFloor = initBox(point(-4, 0.0, -100), point(20, 2.0, 0), floorMaterial)
  result.add upperFloor

  let glass = initMaterial(initDielectric(1.5))
  let railing = initBox(point(-4, 2.0, -0.1), point(10, 2.6, 0), floorMaterial)
  result.add railing

  let SunR = 695_700_000.0
  let AU = 1.496e11
  let pos = point(AU / 10.0, AU / 10.0, AU).normalize * AU
  echo pos.repr
  let sunMat = initMaterial(initLambertian(color(1.0, 1.0, 0.0)))
  result.add translate(pos, Sphere(radius: SunR, mat: sunMat))

  #result.add Disk(distance: 3.3, radius: 10.0, mat: concreteMaterial)

  for x in result:
    echo x.repr

proc sceneDisk(): HittablesList =
  result = initHittables(0)
  let groundMaterial = initMaterial(initLambertian(color(0.2, 0.7, 0.2)))
  result.add Disk(distance: 1.5, radius: 1.5, mat: groundMaterial)

proc sceneTest(rnd: var Rand): HittablesList =
  result = initHittables(0)

  let groundMaterial = initMaterial(initLambertian(color(0.2, 0.7, 0.2)))
  let EarthR = 6_371_000.0
  result.add translate(point(0, -EarthR - 5, 0), Sphere(radius: EarthR, mat: groundMaterial))

  let smallSpheres = rnd.randomSpheres(3)
  result.add rnd.initBvhNode(smallSpheres)

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
  #let cyl = Cylinder(radius: 3.0, zMin: 0.0, zMax: 5.0, phiMax: 180.0.degToRad, mat: cylMetal)
  let cyl = Cone(radius: 3.0, zMax: 4.0, height: 5.0, phiMax: 360.0.degToRad, mat: cylMetal)
  #let cyl = Sphere(radius: 3.0, mat: cylMetal)
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
  #                    Cone(radius: 2.0, height: 5.0, zMax: 3.0, phiMax: 180.0.degToRad, mat: conMetal))
  #result.add con

  #let ball0 = translate(vec3(1.0, -2.0, -4.0), Sphere(radius: 1.5, mat: strMetal))
  #let ball1 = translate(vec3(1.0, 1.0, 1.0), rotateZ(Sphere(radius: 1.5, mat: strMetal), 0.0))
  #let ball2 = translate(vec3(1.0, 1.0, 1.0), rotateZ(Sphere(radius: 1.5, mat: strMetal), 30.0))
  #result.add ball0
  #result.add ball2

from sequtils import mapIt
proc sceneLLNL(): HittablesList =
  result = initHittables(0)

  let
    allR1 = @[63.006, 65.606, 68.305, 71.105, 74.011, 77.027, 80.157,
             83.405, 86.775, 90.272, 93.902, 97.668, 101.576, 105.632]
    allXsep = @[4.171, 4.140, 4.221, 4.190, 4.228, 4.245, 4.288, 4.284,
               4.306, 4.324, 4.373, 4.387, 4.403, 4.481]
    allAngles = @[0.579, 0.603, 0.628, 0.654, 0.680, 0.708, 0.737, 0.767,
                 0.798, 0.830, 0.863, 0.898, 0.933, 0.970]
    lMirror = 225.0


  proc calcHeight(radius, angle: float): float =
    result = radius / tan(angle.degToRad)

  let groundMaterial = initMaterial(initLambertian(color(0.2, 0.7, 0.2)))
  let EarthR = 6_371_000.0

  var objs = initHittables(0)
  #objs.add translate(point(0, -EarthR - 5, 0), Sphere(radius: EarthR, mat: groundMaterial))

  let sunMaterial = initMaterial(initLambertian(color(1.0, 1.0, 0.5)))
  let AU = 150_000_000_000_000.0
  let SunR = 696_342_000_000.0

  objs.add translate(point(0, 0, AU), Sphere(radius: SunR, mat: sunMaterial))



  let conMetal = initMaterial(initMetal(color(0.9, 0.9, 0.9), 0.2))

  let redMaterial = initMaterial(initLambertian(color(0.7, 0.1, 0.1)))
  let greenMaterial = initMaterial(initLambertian(color(0.1, 0.7, 0.1)))

  let cylMetal = initMaterial(initMetal(color(0.6, 0.6, 0.6), 0.2))
  let perfectMirror = initMaterial(initMetal(color(1.0, 1.0, 1.0), 0.0))
  let boreRadius = 43.0 / 2.0
  let magnetBore = Cylinder(radius: boreRadius, zMin: 0.0, zMax: 9.26 * 1000.0, phiMax: 360.0.degToRad, mat: cylMetal)
    .translate(vec3(0.0, 0.0, 2 * lMirror + 5.0)) # 5.0 for xSep + a bit
  objs.add magnetBore

  let zLine = Cylinder(radius: 0.05, zMin: -100.0, zMax: 9.26 * 1000.0, phiMax: 360.0.degToRad, mat: cylMetal)
    .translate(vec3(0.0, -boreRadius, 0.0)) # 5.0 for xSep + a bit
    #.translate(vec3(0.0, 0.0, 0.0)) # 5.0 for xSep + a bit
  objs.add zLine


  let r1_0 = allR1[0]
  let α0 = allAngles[0].degToRad
  let yL0   = r1_0 - sin(α0) * (lMirror / 2.0)
  let yL0L2 = r1_0 - sin(α0) * (lMirror + allXSep[0] / 2.0) - sin(α0 * 3) * (lMirror / 2.0 + allXSep[0] / 2.0)  #- sin(α0 * 3) / (lMirror / 2.0)

  let mirrorDiverge = sin(α0 * 3) * (lMirror / 2.0)
  let addit = sin(α0) * (lMirror / 2.0 + allXSep[0] / 2.0) + sin(α0 * 3) * (allXSep[0] / 2.0)
  let addit2 = sin(α0) * (lMirror / 2.0 + allXSep[0] / 2.0) + sin(α0 * 3) * (lMirror / 2.0 + allXSep[0] / 2.0)

  #let ySep =  sin(α0) * (allXSep[0] / 2.0) + sin(α0 * 3) * (allXSep[0] / 2.0)
  #let yL1 = sin(α0) * lMirror
  #let yL2 = sin(3 * α0) * lMirror

  let diff = abs(yL0 - yL0L2)

  let y0 = cos(3 * α0) * (lMirror + allXSep[0] / 2.0) + cos(α0) * (lMirror / 2.0 + allXSep[0] / 2.0)
  let xLine = Cylinder(radius: 0.5, zMin: -100.0, zMax: 100, phiMax: 360.0.degToRad, mat: cylMetal)
    .translate(vec3(0.0, -boreRadius, 0.0)) # 5.0 for xSep + a bit
    .rotateY(90.0)
    .translate(vec3(0.0, 0.0, y0))
  objs.add xLine

  let y1 = cos(3 * α0) * (lMirror / 2.0)
  let xLine2 = Cylinder(radius: 0.5, zMin: -100.0, zMax: 100, phiMax: 360.0.degToRad, mat: cylMetal)
    .translate(vec3(0.0, -boreRadius, 0.0)) # 5.0 for xSep + a bit
    .rotateY(90.0)
    .translate(vec3(0.0, 0.0, y1))
  objs.add xLine2


  for i in 0 ..< allR1.len:
    let r1 = allR1[i]
    let angle = allAngles[i]
    let
      beta = allAngles[i].degToRad
      xSep = allXsep[i]
      r2 = r1 - lMirror * sin(beta)
      r3 = r2 - 0.5 * xSep * tan(beta)
      r4 = r3 - 0.5 * xSep * tan(3.0 * beta)
      r5 = r4 - lMirror * sin(3.0 * beta)
    echo "r1 = ", r1, " r5 = ", r5, " r5 - r1 = ", r5 - r1
    ## XXX: fix sin and cosine of xsep & the distances!!!
    let ySep =  sin(angle.degToRad) * (xsep / 2.0) + sin(angle.degToRad * 3) * (xsep / 2.0)
    let yL1 = sin(angle.degToRad) * lMirror
    let yL2 = sin(3 * angle.degToRad) * lMirror
    let pos  = r1 - r1_0 #r1_0 # r1_0 #yL0
    ## XXX: FIX POS2
    let pos2 = r1 - r1_0 - yL1 / 2.0 - yL2 / 2.0 - ySep  #- 1.5 * addit2 # - addit  #yL0 - (yL0 - yL0L2)#L2 #- 15.0
    echo "i = ", i, " pos = ", pos, " pos2 = ", pos2, " yL0 = ", yL0, " yL02 = ", yL0L2, " diff = ", diff, " mirrorD = ", mirrorDiverge

    let yCenter = tan(2 * angle.degToRad) * (lMirror + xsep)
    echo "yCenter value = ", yCenter, " compare to 'working' ", - (yL2) - (yL1) - ySep, " compare 'correct' = ", - (yL2/2) - (yL1/2) - ySep
    when true:
      ## XXX: compute the correct 'height' depending on the angle.
      ## Effectively the upper most mirror must be flush with the magnet bore.
      proc setCone(r, angle, y, z: float, mat: Material): Hittable =

        let yL = (sin(angle.degToRad) * lMirror) / 2.0

        let height = calcHeight(r, angle)
        const mirrorSize = 30.0 # degrees
        proc cone(r, h: float): Cone =
          result = Cone(radius: r, height: h, zMax: lMirror, # height, # lMirror,
                        phiMax: mirrorSize.degToRad, mat: mat)
        proc cyl(r, h: float): Cylinder =
          result = Cylinder(radius: r, zMin: 0.0, zMax: lMirror, # height, # lMirror,
                            phiMax: mirrorSize.degToRad, mat: mat)
        let c = cone(r, height)
        #let c = cyl(r, height)
        echo "Translating down by : ", y
        var h = c.rotateZ(mirrorSize / 2.0) # rotate out half the miror size to center "top" of mirror
          .translate(vec3(-r + yL, 0.0, -lMirror / 2.0)) # move to its center
          #.rotateY(angle) # rotate by the angle
          .rotateX(180.0) # we consider from magnet!
          .rotateZ(-90.0)
          .translate(vec3(0.0, y - boreRadius, z + lMirror / 2.0)) # move to its final position
          #.translate(vec3(0.0, y, 0.0))
          #.translate(vec3(0.0, -y/2, 0.0))
          #.translate(vec3(0.0, y/2, 0.0))
        result = h
      ## XXX: z is wrong!
      let con  = setCone(r1, angle,     pos, lMirror + xSep, perfectMirror)#greenMaterial)
      let con2 = setCone(r4, 3 * angle, pos2, 0.0,           perfectMirror)# redMaterial) #- (yL2) - (yL1) - ySep
      #let con2 = setCone(r4, 3 * angle, 0.0, 0.0,             redMaterial)
    elif false:
      ## XXX: Balls are sometimes cut off too!
      let con = translate(vec3(0.0, 3.0 + i.float * 0.2, 0.0),
                          Sphere(radius: allR1[i], mat: redMaterial))
      let con2 = translate(vec3(0.0, 3.0 + i.float * 0.2, lMirror + xSep),
                          Sphere(radius: allR1[i], mat: redMaterial))
    elif false:
      let con = Sphere(radius: allR1[i], mat: redMaterial)
      let con2 = Sphere(radius: allR1[i], mat: redMaterial)
    elif false: ## Sanity check of rotation and selection of mirrors
      proc setCone(r, angle, angle2, y, z: float, mat: Material): Hittable =
        let height = calcHeight(r, angle)
        proc cone(r, h: float): Cone =
          result = Cone(radius: r, height: h, zMax: lMirror, # height, # lMirror,
                        phiMax: 30.0.degToRad, mat: mat)
        let c = cone(r, height)
        var h = c.rotateZ(15.0)
          .translate(vec3(-r, 0.0, -lMirror / 2.0)) # move to its center
          .rotateY(angle2) # rotate by the angle
          .rotateZ(-90.0)
          .translate(vec3(r, 0.0, lMirror / 2.0))
        result = h
      let con  = setCone(allR1[i], angle, angle, i.float * 0.2, lMirror + xSep, redMaterial)
      ## XXX: something is broken when setting large angles! Obect only visible from some sides.
      let con2 = setCone(allR1[i], angle, 45.0, i.float * 0.2, lMirror + xSep, greenMaterial)



    objs.add con
    objs.add con2
  result.add objs #initBvhNode(objs)

proc main(width = 600,
          maxDepth = 5,
          speed = 1.0,
          speedMul = 1.1,
          llnl = false,
          axisAligned = false,
          focalPoint = false,
          vfov = 90.0,
          numRays = 100,
          nJobs = 16) =
  # Image
  THREADS = nJobs
  const ratio = 16.0 / 9.0 #16.0 / 9.0
  let img = Image(width: width, height: (width.float / ratio).int)
  let samplesPerPixel = 100
  # World
  ## Looking at mirrors!
  var lookFrom: Point
  var lookAt: Point
  var world: HittablesList
  if llnl:
    if axisAligned:
      lookFrom = point(0.0, 0.0, -100.0) #point(-0.5, 3, -0.5)#point(3,3,2)
      lookAt = point(0.0, 0.0, 0.0) #point(0, 1.5, 2.5)#point(0,0,-1)
    elif focalPoint:
      let
        detAngle = 2.75.degToRad
        focalDist = 1500.0
      let yAt = tan(detAngle) * focalDist
      lookFrom = point(0.0, -yAt, -focalDist) #point(-0.5, 3, -0.5)#point(3,3,2)
      lookAt = point(0.0, 0.0, 0.0) #point(0, 1.5, 2.5)#point(0,0,-1)
    else:
      lookFrom = point(172.2886370206074, 58.69754358408407, -14.3630844062124) #point(-0.5, 3, -0.5)#point(3,3,2)
      lookAt = point(171.4358852563132, 58.70226619735943, -13.84078935031287) #point(0, 1.5, 2.5)#point(0,0,-1)
    world = sceneLLNL() #mixOfSpheres() #sceneRedBlue() # #sceneCast() #randomScene(useBvh = true, 11) #sceneCast() #randomScene()
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
  dispatch main
