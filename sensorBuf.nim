import std/locks

type
  Sensor2DObj* = object
    len*: int
    width*: int
    height*: int
    buf*: ptr UncheckedArray[int]
    currentMax*: ptr int
    memOwner: bool
  Sensor2D* = ref Sensor2DObj

var L: Lock
L.initLock()

proc `=destroy`(s: Sensor2DObj) =
  if s.memOwner:
    echo "Destroying sensor! "
    if s.buf != nil:
      s.buf.deallocShared()
    if s.currentMax != nil:
      s.currentMax.deallocShared()

proc `=sink`(dest: var Sensor2DObj, source: Sensor2DObj) =
  `=destroy`(dest)
  wasMoved(dest)
  dest.len = source.len
  dest.width = source.width
  dest.height = source.height
  dest.buf = source.buf
  dest.currentMax = source.currentMax
  if source.memOwner: ## Sinking `source` just implies the source is no longer the memory owner
    #source.memOwner = false
    dest.memOwner = true

proc `=copy`(dest: var Sensor2DObj, source: Sensor2DObj) =
  dest.len = source.len
  dest.width = source.width
  dest.height = source.height
  dest.buf = source.buf
  dest.currentMax = source.currentMax
  dest.memOwner = false ## A copy is *NOT* the owner of the buffer!

proc initSensor*(width, height: int): Sensor2D =
  let num = width * height
  var buf = cast[ptr UncheckedArray[int]](allocShared0(width * height * sizeof(int)))
  var pInt = cast[ptr int](allocShared0(sizeof(int)))
  pInt[] = 0
  result = Sensor2D(len: num,
                    width: width,
                    height: height,
                    buf: buf,
                    currentMax: pInt)

proc `[]`*(s: Sensor2D, x, y: int): int =
  withLock(L):
    result = s.buf[y * s.width + x]

proc `[]=`*(s: Sensor2D, x, y: int, val: int) =
  withLock(L):
    s.buf[y * s.width + x] = val
    if val > s.currentMax[]:
      s.currentMax[] = val
    #echo "Writing to: ", cast[int](s.buf), " at pos ", (x, y), " index ", y * s.width + x, " new max: ", val, " but is ", s.currentMax[]


#  140557904855088
#  140557904759072
