
## End User configurable options ###
SHELL = /bin/bash

VPATH=.:$(srcdir)
# Installation directories
prefix       = /usr/local
exec_prefix  = ${prefix}
bindir       = ${exec_prefix}/bin
includedir   = ${prefix}/include
libdir       = ${exec_prefix}/lib
# Build directories
libbuild_dir = /home/yuliayureva/Desktop/OPP/lab1/slog2sdk-1.2.6/lib
top_srcdir   = ../..
srcdir       = .

JC           = /usr/bin/javac
JAR          = /usr/bin/jar
RM           = rm -f

TARGET  = $(libbuild_dir)/drawable.jar
PACKAGE = base.drawable
SRC     = ColorAlpha.java Topology.java Method.java \
          TimeBoundingBox.java Coord.java CoordPixelXform.java \
          InfoType.java InfoValue.java \
          CategoryRatios.java CategorySummary.java \
          Category.java CategoryWeight.java \
          DrawnBox.java DrawnBoxSet.java \
          InfoBox.java Drawable.java NestingStacks.java \
          Primitive.java Shadow.java Composite.java \
          Kind.java YCoordMap.java InputAPI.java

OBJ     = $(SRC:.java=.class)

.SUFFIXES:      .java .class .jar

# CLASSDIR is where classfiles are located, all counted from the top
SRCDIR     = $(top_srcdir)
CLASSDIR   = ../..
CLASSPATH  = ../..
PKGDIR     = base/drawable

JCFLAGS = -sourcepath $(SRCDIR) -classpath $(CLASSPATH) -d $(CLASSDIR) 

.java.class:
	$(JC) $(JCFLAGS) $(JFLAGS) $<

ALL: $(OBJ)

$(TARGET): $(OBJ)
	cd $(CLASSDIR) && \
	jarfile=$@ && \
	classdir_prefix=$(PKGDIR)/ && \
	$(JAR) $(JARFLAGS)cf $${jarfile} $${classdir_prefix}*.class

clean:
	cd $(CLASSDIR) && \
	$(RM) $(PKGDIR)/*.class

rmlib:
	$(RM) $(TARGET)

realclean: clean rmlib

distclean: realclean
	@-$(RM) Makefile
