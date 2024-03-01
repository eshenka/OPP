
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

TARGET  = $(libbuild_dir)/topology.jar
PACKAGE = base.topology
SRC     = Event.java Line.java Arrow.java StateBorder.java State.java \
          PreviewEvent.java PreviewState.java \
          SummaryState.java SummaryArrow.java

OBJ     = $(SRC:.java=.class)

.SUFFIXES:      .java .class .jar

# SRCDIR should be the same as $(srcdir) to be modified by VPATH=.:$(srcdir)
# SRCDIR   = .
# CLASSDIR is where classfiles are located, all counted from the top
SRCDIR     = $(top_srcdir)
CLASSDIR   = ../..
CLASSPATH  = ../..
PKGDIR     = base/topology

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
