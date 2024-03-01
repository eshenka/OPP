
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
top_srcdir   = ../../..
srcdir       = .

JC           = /usr/bin/javac
JAR          = /usr/bin/jar
RM           = rm -f

TARGET  = $(libbuild_dir)/slog2filter.jar
PACKAGE = logformat.slog2.pipe
SRC     = PipedInputLog.java Slog2ToSlog2.java

OBJ     = $(SRC:.java=.class)

.SUFFIXES:      .java .class .jar

# CLASSDIR is where classfiles are located, all counted from the top
SRCDIR     = $(top_srcdir)
CLASSDIR   = ../../..
CLASSPATH  = ../../..
PKGDIR     = logformat/slog2/pipe

JCFLAGS = -sourcepath $(SRCDIR) -classpath $(CLASSPATH) -d $(CLASSDIR) 

.java.class:
	$(JC) $(JCFLAGS) $(JFLAGS) $<

ALL: $(TARGET)

$(TARGET): $(OBJ)
	cd $(CLASSDIR) && \
	$(RM) $(PACKAGE).manifest && \
	echo "Main-Class: $(PACKAGE).Slog2ToSlog2" > $(PACKAGE).manifest ; \
	jarfile=$@ ; \
	$(JAR) $(JARFLAGS)cfm $${jarfile} $(PACKAGE).manifest ; \
	for dir in base/io/ base/drawable/ logformat/slog2/ \
	           logformat/slog2/output/ logformat/slog2/input/ \
	           $(PKGDIR)/ ; do \
	    classdir_prefix=$${dir} ; \
	    $(JAR) $(JARFLAGS)uf $${jarfile} $${classdir_prefix}*.class ; \
	done ; \
	$(JAR) $(JARFLAGS)uf $${jarfile} jumpshot.colors

clean:
	cd $(CLASSDIR) && \
	$(RM) $(PACKAGE).manifest $(PKGDIR)/*.class 

rmlib:
	$(RM) $(TARGET1)

realclean: clean rmlib

distclean: realclean
	@-$(RM) Makefile
