
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

TARGET  =
PACKAGE = logformat.slog2
SRC     = Const.java TraceName.java \
          TreeNodeID.java FileBlockPtr.java \
          TreeDirValue.java TreeDir.java \
          Permutation.java LineIDMap.java LineIDMapList.java \
          IteratorOfBackPrimitives.java IteratorOfForePrimitives.java \
          IteratorOfBackDrawables.java IteratorOfForeDrawables.java \
          IteratorOfAllDrawables.java \
          BufForObjects.java BufForDrawables.java BufForShadows.java \
          CategoryMap.java Header.java

OBJ     = $(SRC:.java=.class)

.SUFFIXES:      .java .class .jar

# CLASSDIR is where classfiles are located, all counted from the top
SRCDIR     = $(top_srcdir)
CLASSDIR   = ../..
CLASSPATH  = ../..
PKGDIR     = logformat/slog2

JCFLAGS = -sourcepath $(SRCDIR) -classpath $(CLASSPATH) -d $(CLASSDIR) 

.java.class:
	$(JC) $(JCFLAGS) $(JFLAGS) $<

ALL: $(OBJ)
	cd output && $(MAKE)
	cd input  && $(MAKE)
	cd pipe   && $(MAKE)

clean:
	cd output && $(MAKE) clean
	cd input  && $(MAKE) clean
	cd pipe   && $(MAKE) clean
	cd $(CLASSDIR) && \
	$(RM) $(PKGDIR)/*.class

rmlib:
	cd output && $(MAKE) rmlib
	cd input  && $(MAKE) rmlib
	cd pipe   && $(MAKE) rmlib

realclean: clean rmlib

distclean: realclean
	@-$(RM) output/Makefile input/Makefile pipe/Makefile
	@-$(RM) Makefile
