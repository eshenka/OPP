
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

TARGET  = $(libbuild_dir)/clogTOdrawable.jar
TARGET1 = $(libbuild_dir)/clogTOdrawable_1pass.jar
TARGET2 = $(libbuild_dir)/clogTOdrawable_2pass.jar
PACKAGE = logformat.clogTOdrawable
SRC     = ColorNameMap.java ObjDef.java \
          Obj_Arrow.java Obj_State.java ObjMethod.java \
          NoMatchingEventException.java TwoEventsMatching.java \
          Topo_Arrow.java Topo_State.java InputLog.java \
          Print.java Print_1pass.java Print_2pass.java

OBJ    = $(SRC:.java=.class)

.SUFFIXES:      .java .class .jar

# CLASSDIR is where classfiles are located, all counted from the top
SRCDIR     = $(top_srcdir)
CLASSDIR   = ../..
CLASSPATH  = ../..
PKGDIR     = logformat/clogTOdrawable

JCFLAGS = -sourcepath $(SRCDIR) -classpath $(CLASSPATH) -d $(CLASSDIR) 

.java.class:
	$(JC) $(JCFLAGS) $(JFLAGS) $<

#  ALL: $(TARGET) $(TARGET1) $(TARGET2)
ALL: $(TARGET)

$(TARGET): $(OBJ)
	cd $(CLASSDIR) && \
	$(RM) $(PACKAGE).manifest && \
	echo "Main-Class: $(PACKAGE).Print" > $(PACKAGE).manifest ; \
	jarfile=$@ ; \
	$(JAR) $(JARFLAGS)cfm $${jarfile} $(PACKAGE).manifest ; \
	for dir in base/io/ logformat/clog/ base/drawable/ $(PKGDIR)/ ; do \
	    classdir_prefix=$${dir} ; \
	    $(JAR) $(JARFLAGS)uf $${jarfile} $${classdir_prefix}*.class ; \
	done ; \
	$(JAR) $(JARFLAGS)uf $${jarfile} jumpshot.colors

$(TARGET1): $(OBJ)
	cd $(CLASSDIR) && \
	$(RM) $(PACKAGE).manifest && \
	echo "Main-Class: $(PACKAGE).Print_1pass" > $(PACKAGE).manifest ; \
	jarfile=$@ ; \
	$(JAR) $(JARFLAGS)cfm $${jarfile} $(PACKAGE).manifest ; \
	for dir in base/io/ logformat/clog/ base/drawable/ $(PKGDIR)/ ; do \
	    classdir_prefix=$${dir} ; \
	    $(JAR) $(JARFLAGS)uf $${jarfile} $${classdir_prefix}*.class ; \
	done ; \
	$(JAR) $(JARFLAGS)uf $${jarfile} jumpshot.colors

$(TARGET2): $(OBJ)
	cd $(CLASSDIR) && \
	$(RM) -f $(PACKAGE).manifest && \
	echo "Main-Class: $(PACKAGE).Print_2pass" > $(PACKAGE).manifest ; \
	jarfile=$@ ; \
	$(JAR) $(JARFLAGS)cfm $${jarfile} $(PACKAGE).manifest ; \
	for dir in base/io/ logformat/clog/ base/drawable/ $(PKGDIR)/ ; do \
	    classdir_prefix=$${dir} ; \
	    $(JAR) $(JARFLAGS)uf $${jarfile} $${classdir_prefix}*.class ; \
	done ; \
	$(JAR) $(JARFLAGS)uf $${jarfile} jumpshot.colors

clean:
	cd $(CLASSDIR) && \
	$(RM) $(PACKAGE).manifest $(PKGDIR)/*.class 

rmlib:
	$(RM) $(TARGET) $(TARGET1) $(TARGET2)

realclean: clean rmlib

distclean: realclean
	@-$(RM) Makefile
