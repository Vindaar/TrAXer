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
  import malebolgia
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
      if ctx.sources.len > 0:
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
  ## XXX: the BVH is also broken here :) Guess we just broke it completely, haha.
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

proc calcHeight(radius, angle: float): float =
  ## Computes the total height of a cone with `angle` and opening `radius`.
  result = radius / tan(angle.degToRad)

from sequtils import mapIt
from std/stats import mean

import sensorBuf, telescopes

proc earth(): Hittable =
  ## Adds the Earth as a ground. It's only 6.371 km large here :) Why? Who knows.
  let groundMaterial = initMaterial(initLambertian(color(0.2, 0.7, 0.2)))
  const EarthR = 6_371_000.0 # Meter, not MilliMeter, but we keep it...
  result = translate(point(0, -EarthR - 5000, 0), Sphere(radius: EarthR, mat: groundMaterial))

type
  SourceKind = enum
    skSun, skXrayFinger, skParallelXrayFinger

import fluxCdf
proc sun(solarModelFile: string): Hittable =
  ## Adds the Sun
  let sunColor = color(1.0, 1.0, 0.5)
  let sunMaterial = if solarModelFile.len == 0:
                      initMaterial(initDiffuseLight(sunColor))
                    else:
                      initMaterial(
                        initSolarEmission(sunColor,
                                          getFluxRadiusCDF(solarModelFile)
                        )
                      )

  const AU = 149_597_870_700_000.0 # by definition since 2012
  var SunR = 696_342_000_000.0 # Solar radius SOHO 2003 & 2006
  if solarModelFile.len == 0:
    ## DTU PhD mentions 3 arcmin source. tan(3' / 2) * 1 AU = 0.0937389
    SunR *= 0.0937389 #0.20 # only inner 20% contribute, i.e. make the sphere smaller for diffuse light
  ## XXX: in principle need to apply correct x AU distance here if `solarModelFile` supplied!
  result = translate(point(0, 0, AU), Sphere(radius: SunR, mat: sunMaterial))

proc xrayFinger(tel: Telescope, magnet: Magnet, magnetPos: float, kind: SourceKind): Hittable =
  ## Adds an X-ray finger
  ## Define the X-ray source
  case kind
  of skParallelXrayFinger:
    let sunMaterial = initMaterial(initLaser(color(1.0, 1.0, 0.5)))
    result = translate(point(0, 0, magnetPos + 9.26 * 1000.0), Disk(distance: 0.0, radius: magnet.radius, mat: sunMaterial))
    #result = translate(point(0, 0, magnetPos + 9.26 * 1000.0 + 100_000.0), Disk(distance: 0.0, radius: magnet.radius, mat: sunMaterial))
  of skXrayFinger: # classical emission allowing all angles
    ## X-ray finger as mentioned in the PhD thesis of A. Jakobsen (14.2 m distance, 3 mm radius)
    let sunMaterial = initMaterial(initDiffuseLight(color(1.0, 1.0, 0.5)))
    result = translate(point(0, 0, magnetPos + 14.2 * 1000.0), Disk(distance: 0.0, radius: 3.0, mat: sunMaterial))
    ## X-ray finger as mentioned in CAST Nature paper ~12 m distance
    #result = translate(point(0, 0, magnetPos + 12.0 * 1000.0), Disk(distance: 0.0, radius: 3.0, mat: sunMaterial))
  else: doAssert false, "Invalid branch, not an X-ray finger."

proc source(tel: Telescope, magnet: Magnet, magnetPos: float, sourceKind: SourceKind,
            solarModelFile: string): Hittable =
  case sourceKind
  of skSun: result = sun(solarModelFile)
  of skXrayFinger, skParallelXrayFinger: result = xrayFinger(tel, magnet, magnetPos, sourceKind)

proc target(tel: Telescope, magnet: Magnet,
            visibleTarget: bool): Hittable =
  ## Construct a target for the light source. We want to sample towards the end of the magnet
  ## (the side towards the telescope)
  let z = tel.length() ## Position of telescope end on magnet side
  let pink = color(1.0, 0.05, 0.9)
  result = toHittable(Disk(distance: 0.0, radius: magnet.radius, mat: toMaterial(initLightTarget(pink, visibleTarget))))
    .translate(vec3(0.0, 0.0, z + magnet.length))

proc lightSource(tel: Telescope, magnet: Magnet,
                 magnetPos: float,
                 sourceKind: SourceKind,
                 visibleTarget: bool,
                 solarModelFile: string
                ): HittablesList =
  ## Constructs a light source as well as the optional target if needed.
  result = initHittables()
  result.add source(tel, magnet, magnetPos, sourceKind, solarModelFile)
  if sourceKind != skParallelXrayFinger:
    result.add target(tel, magnet, visibleTarget)

proc magnetBore(magnet: Magnet, magnetPos: float): Hittable =
  let cylMetal = initMaterial(initMetal(color(0.2, 0.2, 0.6), 0.8))
  # The bore is a full cylinder made of metal, slight blue tint
  result = Cylinder(radius: magnet.radius, zMin: 0.0, zMax: magnet.length, phiMax: 360.0.degToRad, mat: cylMetal)
    .translate(vec3(0.0, 0.0, magnetPos))

proc llnlFocalPointRay(tel: Telescope, magnet: Magnet, fullTelescope: bool): Point =
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
  let ray = llnlFocalPointRay(tel, fullTelescope)
  # evaluate it at `rayAt`.
  echo "Old target: ", target
  result = ray.at(rayAt)
  echo "New target: ", target

proc imageSensor(tel: Telescope, magnet: Magnet,
                 fullTelescope: bool,
                 pixelsW = 400,
                 pixelsH = 400,
                 sensorW = 14.0,
                 sensorH = 14.0,
                 sensorThickness = 0.1,
                 rayAt = 1.0,
                 telescopeRotation = 0.0, # Need to inverse rotate the sensor to align it *before* translating
                 windowRotation = 30.0,
                 windowZOffset = 3.0,
                 ignoreWindow = false
                ): HittablesList =
  ## This is the correct offset for the focal spot position! It's the center of the cones for the telescope.
  ## XXX: Sanity check:
  ## -> Check all mirrors we install have the exact same position as an offset!!!
  let imSensor = toMaterial(initImageSensor(400, 400))
  #let screen = initBox(point(-200, -200, -0.1), point(200, 200, 0.1), imSensor)
  #let screen = initBox(point(-10, -10, -0.1), point(10, 10, 0.1), imSensor)
  result = initBox(point(-sensorW/2, -sensorH/2, -sensorThickness/2),
                   point( sensorW/2,  sensorH/2,  sensorThickness/2),
                   imSensor) # we sink ot so that the box becomes the memory owner of the buffer
                                  # otherwise `imSensor` goes out of scope and frees buffer!
    .translate(target)

proc gridLines(tel: Telescope, magnet: Magnet): HittablesList =
  ## Some helper "grid" lines indicating zero x,y along z as well as center of each mirror
  result = initHittables()
  let cylMetal = initMaterial(initMetal(color(0.2, 0.2, 0.6), 0.8))
  let zLine = Cylinder(radius: 0.05, zMin: -100.0, zMax: magnet.length, phiMax: 360.0.degToRad, mat: cylMetal)
    .translate(vec3(0.0, -magnet.radius, 0.0)) # 5.0 for xSep + a bit
    #.translate(vec3(0.0, 0.0, 0.0)) # 5.0 for xSep + a bit
  result.add zLine

  ## XXX: this is slightly wrong due to xSep!
  let lMirror = tel.lMirror
  let z0 = tel.lMirror + tel.xSep + tel.lMirror / 2
  let xLine = Cylinder(radius: 0.5, zMin: -100.0, zMax: 100, phiMax: 360.0.degToRad, mat: cylMetal)
    .translate(vec3(0.0, -magnet.radius, 0.0))
    .rotateY(90.0)
    .translate(vec3(0.0, 0.0, z0))
  result.add xLine

  let z1 = lMirror / 2
  let xLine2 = Cylinder(radius: 0.5, zMin: -100.0, zMax: 100, phiMax: 360.0.degToRad, mat: cylMetal)
    .translate(vec3(0.0, -magnet.radius, 0.0))
    .rotateY(90.0)
    .translate(vec3(0.0, 0.0, z1))
  result.add xLine2

proc calcYlYsep(angle, xSep, lMirror: float): (float, float, float) =
  ## Helper to compute displacement of each set of mirrors and the y distance
  ## given by the angles due to mirror rotation
  let α = angle.degToRad
  let ySep = tan(α) * (xSep / 2.0) + tan(3 * α) * (xSep / 2.0)
  let yL1  = sin(α) * lMirror
  let yL2  = sin(3 * α) * lMirror
  result = (ySep, yL1, yL2)

proc graphiteSpacer(tel: Telescope, magnet: Magnet, fullTelescope: bool): HittablesList =
  ## The graphite spacer in the middle
  result = initHittables()
  let
    lMirror = tel.lMirror
    xSep = tel.xSep
    graphite = initMaterial(initLambertian(color(0.2, 0.2, 0.2)))
    excessSize = 5.0
    α0 = tel.allAngles[0]
    (ySep, yL1, yL2) = calcYlYsep(α0, xSep, lMirror)
    meanAngle = tel.allAngles.mean
  let gSpacer = initBox(point(0, 0, 0), point(2, 2 * magnet.radius + excessSize, lMirror), graphite)
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
    #let centerDisk = Disk(distance: 0.0, radius: 500.0, mat: graphite) #imSensorDisk)
    #let centerDisk = XyRect(x0: -boreRadius, x1: boreRadius, y0: -boreRadius, y1: boreRadius, mat: imSensorDisk)  #Disk(distance: 0.0, radius: 500.0, mat: imSensorDisk)
    #let centerDisk = initBox(point(-magnet.radius, -magnet.radius, -0.1), point(magnet.radius, magnet.radius, 0.1), imSensorDisk)
    ## A disk that blocks the inner part where no mirrors cover the bore.
    let centerDisk = Disk(distance: 0.0, radius: tel.allR1[0] - sin(tel.allAngles[0].degToRad) * tel.lMirror,
                          mat: graphite)
      .translate(vec3(0.0, 0.0, lMirror * 2 + xSep))
    result.add centerDisk

proc sanityTelescopeOutput(r1, r5, pos, pos2, angle, lMirror, xSep, yL1, yL2, ySep, boreRadius: float) =
  ## XXX: clean this up and make more useful!
  echo "r1 = ", r1, " r5 = ", r5, " r5 - r1 = ", r5 - r1, " i = ", i, " pos = ", pos, " pos2 = ", pos2
  block Sanity:
    let yCenter = tan(2 * angle.degToRad) * (lMirror + xsep)
    echo "yCenter value = ", yCenter, " compare to 'working' ", - (yL2) - (yL1) - ySep, " compare 'correct' = ", - (yL2/2) - (yL1/2) - ySep
    let yOffset = (r1 - (sin(angle.degToRad) * lMirror) / 2.0) + boreRadius - pos
    echo "Offset for layer ", i, " should be: ", yOffset, " \n==============================\n\n"

proc llnlTelescope(tel: Telescope, magnet: Magnet, fullTelescope, usePerfectMirror: bool): HittablesList =
  ## Constructs the actual LLNL telescope
  ##
  ## Pos for the 'first' layers should be correct, under the following two conditions:
  ## 1. it's not entirely clear what part of the telescope really should align with the bottom edge of the magnet.
  ##    Currently the center of the front shell is aligned with the bottom.
  ## 2. it's not clear what part of the shells should align vertically as discussed with Julia and Jaime in Cristinas office.
  ##    The front? The center? The back? Currently I align at the center.
  const sanity = false
  let
    lMirror = tel.lMirror
    xSep = tel.xSep
  let perfectMirror = initMaterial(initMetal(color(1.0, 0.0, 0.0), 0.0))
  ## Imperfect value assuming a 'figure error similar to NuSTAR of 1 arcmin'
  ## -> tan(1 ArcMin) (because fuzz added to unit vector)
  const ImperfectVal = 0.0002908880082045767
  let imperfectMirror = initMaterial(initMetal(color(1.0, 0.0, 0.0), ImperfectVal))
  let mat = if usePerfectMirror: perfectMirror else: imperfectMirror

  let r1_0 = tel.allR1[0]
  for i in 0 ..< tel.allR1.len:
    let
      r1 = tel.allR1[i]
      r5 = tel.allR5[i]
      angle = tel.allAngles[i] ## * 1.02 yields 1500mm focal length
      r4 = r5 + lMirror * sin(3.0 * angle.degToRad)
    let (ySep, yL1, yL2) = calcYlYsep(angle, xSep, lMirror)
    # `pos`, `pos2` are the `y` positions of first & second set of mirrors.
    let pos = (r1 - r1_0) ## Only shift each layer relative to first layer. Other displacement done in `setCone`
    let pos2 = pos - yL1 / 2.0 - yL2 / 2.0 - ySep
    if sanity:
      sanityTelescopeOutput(r1, r5, pos, pos2, angle, lMirror, xSep, yL1, yL2, ySep, magnet.radius)
    proc setCone(r, angle, y, z: float, mat: Material): Hittable =
      # `xOffset` is the displacement from front to center of mirror (x because cone with `phiMax`
      # starts from x = 0)
      let xOffset = (sin(angle.degToRad) * lMirror) / 2.0
      let height = calcHeight(r, angle) # total height of the cone that yields required radius and angle
      proc cone(r, h: float): Cone =
        result = Cone(radius: r, height: h, zMax: lMirror,
                      phiMax: tel.mirrorSize.degToRad, mat: mat)
      proc cyl(r, h: float): Cylinder =
        result = Cylinder(radius: r, zMin: 0.0, zMax: lMirror,
                          phiMax: tel.mirrorSize.degToRad, mat: mat)
      let c = cone(r, height)
      #let c = cyl(r, height) ## To construct a fake telescope
      # for the regular telescope first move to -r + xOffset to rotate around center of layer. Full no movement
      let xOrigin = if fullTelescope: 0.0 else: -r + xOffset # aligns *center* of mirror
      let yOffset = if fullTelescope: 0.0 else: y - magnet.radius # move down by bore radius & offset
      var h = c.rotateZ(tel.mirrorSize / 2.0) # rotate out half the miror size to center "top" of mirror
        .translate(vec3(xOrigin, 0.0, -lMirror / 2.0)) # move to its center
        #.rotateY(angle) ## For a cylinder telescope
        .rotateX(180.0) # we consider from magnet!
        .rotateZ(-90.0)
        .translate(vec3(0.0, yOffset, z + lMirror / 2.0)) # move to its final position
      result = h
    let con  = setCone(r1, angle,     pos,  lMirror + xSep, mat)
    ## NOTE: using `r1` as well reproduces the X-ray finger results from the _old_ raytracer!
    let con2 = setCone(r4, 3 * angle, pos2, 0.0,            mat)
    result.add con
    result.add con2

proc initSetup(fullTelescope: static bool): (Telescope, Magnet) =
  when not fullTelescope:
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

proc sceneLLNL(rnd: var Rand, visibleTarget, gridLines, usePerfectMirror: bool, sourceKind: SourceKind,
               solarModelFile: string,
               rayAt = 1.0): HittablesList =
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
  result = initHittables(0)
  ## Constants fixed to *this* specific setup
  let (llnl, magnet) = initSetup(fullTelescope = false)
  let magnetPos = llnl.length + telescopeMagnetZOffset

  var objs = initHittables(0)

  objs.add earth()
  objs.add lightSource(llnl, magnet, magnetPos, sourceKind, visibleTarget, solarModelFile)
  objs.add magnetBore(magnet, magnetPos)
  if gridLines:
    ## XXX: fix adding `HittablesList` to another!
    objs.add gridLines(llnl, magnet)
  objs.add imageSensor(llnl, magnet, fullTelescope = false, rayAt = rayAt)
  #  .translate(vec3(0.0,+7.0,0.0))
  var telescope = initHittables()
  telescope.add graphiteSpacer(llnl, magnet, fullTelescope = false)
  telescope.add llnlTelescope(llnl, magnet, fullTelescope = false, usePerfectMirror = usePerfectMirror)
  objs.add telescope
  #  .rotateZ(90.0 - 12)
  ### Materials
  #let redMaterial = initMaterial(initLambertian(color(0.7, 0.1, 0.1)))
  #let greenMaterial = initMaterial(initLambertian(color(0.1, 0.7, 0.1)))

  ## XXX: BVH node of the telescope is currently broken! Bad shading.
  #result.add telescope #rnd.initBvhNode(telescope)
  result.add objs

proc sceneLLNLTwice(rnd: var Rand, visibleTarget, gridLines, usePerfectMirror: bool, sourceKind: SourceKind,
                    solarModelFile: string): HittablesList =
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
  result = initHittables(0)
  ## Constants fixed to *this* specific setup
  let (llnl, magnet) = initSetup(fullTelescope = false)
  let magnetPos = llnl.length + telescopeMagnetZOffset

  var objs = initHittables(0)

  objs.add earth()
  #objs.add lightSource(llnl, magnet, magnetPos, sourceKind, visibleTarget, solarModelFile)
  if gridLines:
    ## XXX: fix adding `HittablesList` to another!
    objs.add gridLines(llnl, magnet)
  objs.add imageSensor(llnl, magnet, fullTelescope = true) ## Don't want y displacement, the two bores are rotated by 90°
                                                           ## so focal spot is on y = 0

  var telMag = initHittables()

  case sourceKind
  of skSun: objs.add sun(solarModelFile) ## Only a single sun in center x/y! Hence add to `objs` so not duplicated
  of skXrayFinger, skParallelXrayFinger:
    telMag.add xrayFinger(llnl, magnet, magnetPos, sourceKind)
  # Need two targets in both source cases
  telMag.add target(llnl, magnet, visibleTarget)
  telMag.add magnetBore(magnet, magnetPos)
  telMag.add graphiteSpacer(llnl, magnet, fullTelescope = false)
  telMag.add llnlTelescope(llnl, magnet, fullTelescope = false, usePerfectMirror = usePerfectMirror)

  ## XXX: FIX THE -83!
  objs.add telMag.rotateZ(-90.0)
    .translate(vec3(-83.0, 0.0, 0.0))
  objs.add telMag.rotateZ(90.0)
    .translate(vec3(83.0, 0.0, 0.0))
  ### Materials
  #let redMaterial = initMaterial(initLambertian(color(0.7, 0.1, 0.1)))
  #let greenMaterial = initMaterial(initLambertian(color(0.1, 0.7, 0.1)))

  ## XXX: BVH node of the telescope is currently broken! Bad shading.
  #result.add telescope #rnd.initBvhNode(telescope)
  result.add objs


proc sceneLLNLFullTelescope(rnd: var Rand, visibleTarget, gridLines, usePerfectMirror: bool, sourceKind: SourceKind,
                            solarModelFile: string,
                            rayAt = 1.0): HittablesList =
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
  result = initHittables(0)
  ## Constants fixed to *this* specific setup
  let (llnl, magnet) = initSetup(fullTelescope = true)
  let magnetPos = llnl.length + telescopeMagnetZOffset

  var objs = initHittables(0)

  objs.add earth()
  objs.add lightSource(llnl, magnet, magnetPos, sourceKind, visibleTarget, solarModelFile)
  objs.add magnetBore(magnet, magnetPos)
  if gridLines:
    ## XXX: fix adding `HittablesList` to another!
    objs.add gridLines(llnl, magnet)
  objs.add imageSensor(llnl, magnet, fullTelescope = true, rayAt = rayAt)

  var telescope = initHittables()
  telescope.add graphiteSpacer(llnl, magnet, fullTelescope = true)
  telescope.add llnlTelescope(llnl, magnet, fullTelescope = true, usePerfectMirror = usePerfectMirror)
  objs.add telescope # .rotateX(-0.5)

  ## XXX: BVH node of the telescope is currently broken! Bad shading.
  #result.add telescope #rnd.initBvhNode(telescope)
  result.add objs

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
          rayAt = 1.0) =
  # Image
  THREADS = nJobs
  #const ratio = 16.0 / 9.0 #16.0 / 9.0
  const ratio = 1.0
  let img = Image(width: width, height: (width.float / ratio).int)
  let samplesPerPixel = 100
  var rnd = initRand(0x299792458)
  # World
  ## Looking at mirrors!
  var lookFrom: Point = point(1,1,1)
  var lookAt: Point = point(0,0,0)
  var world: HittablesList
  if llnl:
    if axisAligned:
      lookFrom = point(0.0, 0.0, -100.0) #point(-0.5, 3, -0.5)#point(3,3,2)
      lookAt = point(0.0, 0.0, 0.0) #point(0, 1.5, 2.5)#point(0,0,-1)
    elif focalPoint:
      let (llnl, magnet) = initSetup(fullTelescope)
      let ray = llnlFocalPointRay(llnl, magnet, fullTelescope)
      (lookFrom, lookAt) = (ray.at(rayAt), ray.at(0.0))
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
      world = rnd.sceneLLNLFullTelescope(visibleTarget, gridLines, usePerfectMirror, sourceKind, solarModelFile, rayAt)
    elif llnlTwice:
      world = rnd.sceneLLNLTwice(visibleTarget, gridLines, usePerfectMirror, sourceKind, solarModelFile)
    else:
      echo "Visible target? ", visibleTarget
      world = rnd.sceneLLNL(visibleTarget, gridLines, usePerfectMirror, sourceKind, solarModelFile, rayAt)
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
  dispatch main
