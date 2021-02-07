import strformat, os, terminal, macros, math, random, times

import basetypes, hittables, camera

import arraymancer

import sdl2 except Color, Point

proc rayColor*(r: Ray, world: var HittablesList, depth: int): Color =
  var rec: HitRecord

  if depth <= 0:
    return color(0, 0, 0)

  if world.hit(r, 0.001, Inf, rec):
    var scattered: Ray
    var attenuation: Color
    if rec.mat.scatter(r, rec, attenuation, scattered):
      return attenuation * rayColor(scattered, world, depth - 1)
    return color(0, 0, 0)

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

proc render*(img: Image, f: string, world: var HittablesList,
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
        let u = (i.float + rand(1.0)) / (img.width - 1).float
        let v = (j.float + rand(1.0)) / (img.height - 1).float
        let r = camera.getRay(u, v)
        pixelColor += rayColor(r, world, maxDepth)
      f.writeColor(pixelColor, samplesPerPixel)
  f.close()

proc renderMC*(img: Image, f: string, world: var HittablesList,
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
    let x = rand((img.width).float)
    let y = rand((img.height).float)
    let r = camera.getRay(x / (img.width - 1).float,
                          y / (img.height - 1).float)
    let color = rayColor(r, world, maxDepth)
    buf[y.int, x.int] = buf[y.int, x.int] + color
    counts[y.int, x.int] = counts[y.int, x.int] + 1
    inc idx
    if idx mod (img.width * img.height) == 0:
      let remain = numRays - idx
      stderr.write(&"\rRays remaining: {remain}")
  for j in countdown(img.height - 1, 0):
    stderr.write(&"\rScanlines remaining: {j}")
    for i in 0 ..< img.width:
      f.writeColor(buf[j, i], counts[j, i])
  f.close()

proc renderSdl*(img: Image, world: var HittablesList,
                camera: Camera,
                samplesPerPixel, maxDepth: int) =
  discard sdl2.init(INIT_EVERYTHING)
  var screen = sdl2.createWindow("Ray tracing".cstring,
                                 SDL_WINDOWPOS_UNDEFINED,
                                 SDL_WINDOWPOS_UNDEFINED,
                                 img.width.cint, img.height.cint,
                                 SDL_WINDOW_OPENGL);
  var renderer = sdl2.createRenderer(screen, -1, 1)
  if screen.isNil:
    quit($sdl2.getError())

  discard setRelativeMouseMode(True32)

  var quit = false
  var event = sdl2.defaultEvent
  #let renderObj = RenderObject(img: img, world: world, surface: window, renderer: renderer, camera: camera, samplesPerPixel: samplesPerPixel, maxDepth: maxDepth)
  #echo "render frame"
  #renderObj.render()
  #echo "done"
  #echo "render frame"
  #renderObj.render()
  #echo "done"
  #echo "render frame"
  #renderObj.render()
  #echo "done"

  #var window = sdl2.getsurface(screen)
  #discard lockSurface(window)
  #var bufT = fromBuffer[int32](window.pixels, @[img.height, img.width])
  #var counts = newTensor[int](@[img.height, img.width])
  #unlockSurface(window)
  var window = sdl2.getsurface(screen)
  template resetBufs(bufT, counts: untyped): untyped {.dirty.} =
    bufT = newTensor[uint32](@[img.height, img.width])
    #renderer.clear()
    counts = newTensor[int](@[img.height, img.width])

  var bufT = newTensor[uint32](@[img.height, img.width])# cast[ptr UncheckedArray[uint32]](window.pixels) #fromBuffer[uint32](window.pixels, @[img.height, img.width])
  #discard lockSurface(window)
  ## NOTE: cannot work, because surface becomes invalid in the middle of computation
  #var bufT = fromBuffer[uint32](window.pixels, @[img.height, img.width])
  #unlockSurface(window)

  var counts = newTensor[int](@[img.height, img.width])
  var xpos = window.w.float / 2.0 # = 400 to center the cursor in the window
  var ypos = window.h.float / 2.0 # = 300 to center the cursor in the window

  let origLookFrom = camera.lookFrom
  let origLookAt = camera.lookAt

  #unlockSurface(window)
  while not quit:
    while pollEvent(event):
      case event.kind
      of QuitEvent:
        quit = true
      of KeyDown:
        const dist = 1.0
        case event.key.keysym.scancode
        of SDL_SCANCODE_LEFT:
          let cL = (camera.lookFrom - camera.lookAt).Vec3

          let zAx = vec3(0, 1, 0)
          echo "HERE ", vec3(3, -1, 4).cross(zAx)
          let newFrom = cL.cross(zAx).normalize().Point

          #let angle = arccos(cL.dot(xAx) / (xAx.length() * cL.length()))
          #let newFrom = xAx.rotate(0, angle, 0).Point
          echo camera.lookFrom.repr, "  ↦   ", newFrom.repr, "  ⇒   ", (camera.lookFrom - newFrom).repr, "     ↦↦↦     ", (camera.lookAt - newFrom).repr

          let dist = (camera.lookFrom - camera.lookAt).length()
          let nCL = camera.lookFrom + newFrom
          let nCA = camera.lookAt + newFrom
          let newDist = (nCL - nCA).length()
          camera.updateLookFromAt(nCL, nCA)
          resetBufs(bufT, counts)
        of SDL_SCANCODE_RIGHT:
          let cL = (camera.lookFrom - camera.lookAt).Vec3
          let zAx = vec3(0, 1, 0)
          let newFrom = cL.cross(zAx).normalize().Point
          let nCL = camera.lookFrom - newFrom
          let nCA = camera.lookAt - newFrom
          camera.updateLookFromAt(nCL, nCA)
          resetBufs(bufT, counts)
        of SDL_SCANCODE_UP:
          var cL = (camera.lookFrom - camera.lookAt).Vec3
          cL[1] = 0.0
          cL = cL.normalize()
          let nCL = camera.lookFrom - cL.Point
          let nCA = camera.lookAt - cL.Point
          camera.updateLookFromAt(nCL, nCA)
          resetBufs(bufT, counts)
        of SDL_SCANCODE_DOWN:
          var cL = (camera.lookFrom - camera.lookAt).Vec3
          cL[1] = 0.0
          cL = cL.normalize()
          let nCL = camera.lookFrom + cL.Point
          let nCA = camera.lookAt + cL.Point
          camera.updateLookFromAt(nCL, nCA)
          resetBufs(bufT, counts)
        of SDL_SCANCODE_BACKSPACE:
          echo "Resetting view"
          camera.updateLookFromAt(origLookFrom, origLookAt)
          resetBufs(bufT, counts)
        else: discard
      of WindowEvent:
        freeSurface(window)
        window = sdl2.getsurface(screen)
      of MouseMotion:
        # assuming I have a 800x600 Game window, then the result would be:
        xpos = -event.motion.xrel.float / 1000.0
        ypos = event.motion.yrel.float / 1000.0

        let newLook = (camera.lookFrom.Vec3).rotate(0, xpos, ypos)
        echo "xpos : ", xpos, " ypos : ", ypos, " lookFrom ", newLook.repr
        camera.updateLookFromFront(Point(newLook))
        resetBufs(bufT, counts)
      else: echo event.kind
      #discard
    echo "render frame /{etrniedtniuae"
    #freeSurface(window)
    #var window = sdl2.getsurface(screen)
    discard lockSurface(window)

    let width = img.width
    let height = img.height
    var numRays = 10_000 #samplesPerPixel * width * height
    var idx = 0
    #window = getSurface(screen)
    #discard lockSurface(window)
    while idx < numRays:
      let x = rand((width).float)
      let y = rand((height).float)
      #if x.int >= window.w: continue
      #if y.int >= window.h: continue
      let r = camera.getRay(x / (width - 1).float,
                            y / (height - 1).float)
      let color = rayColor(r, world, maxDepth)
      let yIdx = height - y.int - 1
      let xIdx = x.int
      counts[yIdx, xIdx] = counts[yIdx, xIdx] + 1
      let curColor = bufT[yIdx, xIdx].toColor
      let delta = (color.gammaCorrect - curColor) / counts[yIdx, xIdx].float
      let newColor = curColor + delta
      let cu8 = toColorU8(newColor)
      let sdlColor = sdl2.mapRGB(window.format, cu8.r.byte, cu8.g.byte, cu8.b.byte)
      bufT[yIdx, xIdx] = sdlColor
      inc idx
    echo window.pitch
    echo window.w
    echo window.h

    #window.pixels = cast[pointer](bufT.unsafe_raw_buf())
    var surf = fromBuffer[uint32](window.pixels, @[window.h.int, window.w.int])
    let t0 = cpuTime()
    #for idx in 0 ..< surf.size:
    if surf.shape == bufT.shape:
      echo "equal"
      for idx in 0 ..< surf.size:
        surf.unsafe_raw_offset()[idx] = bufT.unsafe_raw_offset()[idx]
    else:
      echo "unequal"
      for y in 0 ..< surf.shape[0]:
        for x in 0 ..< surf.shape[1]:
          surf[y, x] = bufT[y, x]
    let t1 = cpuTime()
    echo "copying took ", t1 - t0, " s"
    unlockSurface(window)
    #sdl2.clear(arg.renderer)
    sdl2.present(renderer)
    #delay(10)

    #renderObj.render()
    echo "done"

  sdl2.quit()

proc sceneRedBlue(): HittablesList =
  result = initHittables(0)
  let R = cos(Pi/4.0)

  #world.add Sphere(center: point(0, 0, -1), radius: 0.5)
  #world.add Sphere(center: point(0, -100.5, -1), radius: 100)

  let matLeft = initMaterial(initLambertian(color(0,0,1)))
  let matRight = initMaterial(initLambertian(color(1,0,0)))

  result.add Sphere(center: point(-R, 0, -1), radius: R, mat: matLeft)
  result.add Sphere(center: point(R, 0, -1), radius: R, mat: matRight)

proc mixOfSpheres(): HittablesList =
  result = initHittables(0)
  let matGround = initMaterial(initLambertian(color(0.8, 0.8, 0.0)))
  let matCenter = initMaterial(initLambertian(color(0.1, 0.2, 0.5)))
  # let matLeft = initMaterial(initMetal(color(0.8, 0.8, 0.8), 0.3))
  let matLeft = initMaterial(initDielectric(1.5))
  let matRight = initMaterial(initMetal(color(0.8, 0.6, 0.2), 1.0))

  result.add Sphere(center: point(0, -100.5, -1), radius: 100, mat: matGround)
  result.add Sphere(center: point(0, 0, -1), radius: 0.5, mat: matCenter)
  result.add Sphere(center: point(-1.0, 0, -1), radius: 0.5, mat: matLeft)
  result.add Sphere(center: point(-1.0, 0, -1), radius: -0.4, mat: matLeft)
  result.add Sphere(center: point(1.0, 0, -1), radius: 0.5, mat: matRight)

proc randomScene(): HittablesList =
  result = initHittables(0)

  let groundMaterial = initMaterial(initLambertian(color(0.5, 0.5, 0.5)))
  result.add Sphere(center: point(0, -1000, 0), radius: 1000, mat: groundMaterial)

  var smallSpheres = initHittables(0)
  for a in -11 ..< 11:
    for b in -11 ..< 11:
      let chooseMat = rand(1.0)
      var center = point(a.float + 0.9 * rand(1.0), 0.2, b.float + 0.9 * rand(1.0))

      if (center - point(4, 0.2, 0)).length() > 0.9:
        var sphereMaterial: Material
        if chooseMat < 0.8:
          # diffuse
          let albedo = randomVec().Color * randomVec().Color
          sphereMaterial = initMaterial(initLambertian(albedo))
          smallSpheres.add Sphere(center: center, radius: 0.2, mat: sphereMaterial)
        elif chooseMat < 0.95:
          # metal
          let albedo = randomVec(0.5, 1.0).Color
          let fuzz = rand(0.0 .. 0.5)
          sphereMaterial = initMaterial(initMetal(albedo, fuzz))
          smallSpheres.add Sphere(center: center, radius: 0.2, mat: sphereMaterial)
        else:
          # glass
          sphereMaterial = initMaterial(initDielectric(1.5))
          smallSpheres.add Sphere(center: center, radius: 0.2, mat: sphereMaterial)

  result.add initBvhNode(smallSpheres)

  let mat1 = initMaterial(initDielectric(1.5))
  result.add Sphere(center: point(0, 1, 0), radius: 1.0, mat: mat1)

  let mat2 = initMaterial(initLambertian(color(0.4, 0.2, 0.1)))
  result.add Sphere(center: point(-4, 1, 0), radius: 1.0, mat: mat2)

  let mat3 = initMaterial(initMetal(color(0.7, 0.6, 0.5), 0.0))
  result.add Sphere(center: point(4, 1, 0), radius: 1.0, mat: mat3)

proc main =
  # Image
  const ratio = 3.0 / 2.0 #16.0 / 9.0
  const width = 1200
  let img = Image(width: width, height: (width / ratio).int)
  let samplesPerPixel = 500
  let maxDepth = 50

  # World
  var world = randomScene()

  # Camera
  let lookFrom = point(13, 2, 3)#point(3,3,2)
  let lookAt = point(0, 0, 0)#point(0,0,-1)
  let vup = vec3(0,1,0)
  let distToFocus = 10.0 #(lookFrom - lookAt).length()
  let aperture = 0.1
  let camera = initCamera(lookFrom, lookAt, vup, vfov = 20,
                          aspectRatio = ratio,
                          aperture = aperture, focusDist = distToFocus)

  # Rand seed
  randomize(0xE7)

  # Render
  img.render("/tmp/test_small.ppm", world, camera, samplesPerPixel, maxDepth)

when isMainModule:
  main()
