import unchained
type
  TelescopeKind* = enum
    tkLLNL = "LLNL",                     ## the LLNL telescope as used at CAST based on the nustar optics
    tkXMM = "XMM",                       ## the XMM Newton optics as they may be used in BabyIAXO
    tkCustomBabyIAXO = "CustomBabyIAXO", ## A hybrid optics specifically built for BabyIAXO mixing
                                         ## a nustar like inner core with an XMM like outer core
    tkAbrixas = "Abrixas",               ## the Abrixas telescope as used at CAST
    tkOther = "Other"

  Telescope* = object
    kind*: TelescopeKind
    optics_entrance*: seq[float] # seq[MilliMeter]
    optics_exit*: seq[float] # seq[MilliMeter]
    telescope_turned_x*: float#Degree
    telescope_turned_y*: float#Degree
    glassThickness*: float # seq[MilliMeter]
    allR1*: seq[float] # seq[MilliMeter]
    allR2*: seq[float] # seq[MilliMeter]
    allR4*: seq[float] # seq[MilliMeter]
    allR5*: seq[float] # seq[MilliMeter]
    diffs*: seq[float] ## Differences between R1 and R5 in built telescopes
    xSep*: float # MilliMeter
    focalLength*: float # MilliMeter
    allXsep*: seq[float] # seq[MilliMeter]
    #allAngles*: seq[float] #seq[Degree]
    lMirror*: float#mm
    mirrorSize*: Degree # size of mirrors in degrees of the cones
    #holeInOptics*: float# mm
    #numberOfHoles*: int
    #holeType*: HoleType
    #reflectivity*: Reflectivity

  Magnet* = object
    lengthColdbore*: float #mm ## Length of the cold bore of the magnet
    #B*: T               ## Magnetic field strength of the magnet (assumed homogeneous)
    length*: float #mm        ## Length in which `B` applies
    radius*: float #mm       ## Radius of the coldbore of the magnet
    #pGasRoom*: bar      ## Pressure of gas inside the bore at room temperature
    #tGas*: K            ## Temperature of a possible gas inside the bore

proc initTelescope*(kind: TelescopeKind, mirrorSize: Degree): Telescope =
  ## For the LLNL telescope:
  ##   The PhD thesis by Anders Jakobsen from DTU gives these values for the shells:
  ##
  ##   ```nim
  ##      # the radii of the shells
  ##      allR1: @[63.006, 65.606, 68.305, 71.105, 74.011, 77.027, 80.157,
  ##               83.405, 86.775, 90.272, 93.902, 97.668, 101.576, 105.632], #mapIt(it.mm),
  ##      allR5: @[53.821, 56.043, 58.348, 60.741, 63.223, 65.800, 68.474,
  ##               71.249, 74.129, 77.117, 80.218, 83.436, 86.776, 90.241], #.mapIt(it.mm),
  ##      allAngles: @[0.579, 0.603, 0.628, 0.654, 0.680, 0.708, 0.737, 0.767,
  ##                   0.798, 0.830, 0.863, 0.898, 0.933, 0.970], #.mapIt(it.Degree),
  ##   ```
  ##
  ##   However, these are outdated and not the values finally used (indeed, these would produce
  ##   a telescope with focal length of 1530 mm! Check via Wolter equation if you want).
  ##   Note that here the telescope still had 14 shells.
  ##
  ##   There is a file called `cast20l4_f1500mm_asBuilt.txt`, which contains the final numbers.
  ##   These are the ones currently in use below.
  case kind:
  of tkLLNL:
    result = Telescope(
      kind: tkLLNL,
      optics_entrance: @[-83.0, 0.0, 0.0], #.mapIt(it.mm),
      optics_exit: @[-83.0, 0.0, 454.0], #.mapIt(it.mm),
      telescope_turned_x: 0.0, #.°, #the angle by which the telescope is turned in respect to the magnet
      telescope_turned_y: 0.0, #.°, #the angle by which the telescope is turned in respect to the magnet
      # Measurements of the Telescope mirrors in the following, R1 are the radii of the mirror shells at the entrance of the mirror
      # the radii of the shells
      glassThickness: 0.21, # mm. The LLNL / NuSTAR glass is not 0.2mm, but 0.21 mm thick!
      # the radii of the shells
      allR1: @[ 63.2384, 65.8700, 68.6059, 71.4175, 74.4006, 77.4496, 80.6099,
                83.9198, 87.3402, 90.8910, 94.5780, 98.3908, 102.381 ],
      allR2: @[ 60.9121, 63.4470, 66.0824, 68.7898, 71.6647, 74.6014, 77.6452,
                80.8341, 84.1290, 87.5495, 91.1013, 94.7737, 98.6187 ],
      allR4: @[ 60.8632, 63.3197, 65.9637, 68.6794, 71.5582, 74.4997, 77.5496,
                80.7305, 84.0137, 87.4316, 90.9865, 94.6549, 98.4879 ],
      allR5: @[ 53.8823, 56.0483, 58.3908, 60.7934, 63.3473, 65.9515, 68.6513,
                71.4688, 74.3748, 77.4012, 80.5497, 83.7962, 87.1914 ],
      # this last one should be the difference between R5 and R1
      diffs: @[ 10.339, 10.769, 11.216, 11.679, 12.160, 12.659, 13.176,
                13.714, 14.272, 14.851, 15.452, 16.076, 16.725  ],
      lMirror: 225.0, #.mm,
      focalLength: 1500.0,
      xSep: 4.0, ## 4 mm along the z axis! (so here it's technically zSep)
      mirrorSize: mirrorSize
      # allXsep: @[4.171, 4.140, 4.221, 4.190, 4.228, 4.245, 4.288, 4.284,
      #            4.306, 4.324, 4.373, 4.387, 4.403, 4.481].mapIt(it.mm),
      # the angles of the mirror shells coresponding to the radii above
      #holeInOptics: 0.0.mm, #max 20.9.mm
      #numberOfHoles: 5,
      #holeType: htCross, #the type or shape of the hole in the middle of the optics
      #reflectivity: optics.initReflectivity()
    )
  else:
    doAssert false, "Not implemented yet!"

proc length*(tel: Telescope): float = tel.lMirror * 2 + tel.xSep

from std/math import arcsin
from std/stats import mean
proc calcAngle*(tel: Telescope, idx: int): Degree =
  ## Helper that returns the angle of the *first* set of mirrors for the given
  ## shell index `idx`.
  ## Returns the angle in Degree
  if idx > tel.allR1.len:
    raise newException(ValueError, "Invalid shell at index: " & $idx)
  result = arcsin(abs(tel.allR1[idx] - tel.allR2[idx]) / tel.lMirror).Radian.to(Degree)

proc meanAngle*(tel: Telescope): Degree =
  ## Returns the mean angle of all shells
  var angles = newSeq[float](tel.allR1.len)
  for i in 0 ..< angles.len:
    angles[i] = tel.calcAngle(i).float # Degree
  result = angles.mean.Degree

proc initMagnet*(radius: float, length = 9.26): Magnet =
  result = Magnet(lengthColdbore: length * 1000.0 + 500,
                  length: length * 1000,
                  radius: radius)
