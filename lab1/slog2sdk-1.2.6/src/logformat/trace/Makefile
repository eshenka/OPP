
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
JH           = /usr/bin/javah
RM           = rm -f

TARGET   = $(libbuild_dir)/traceprint.jar
PACKAGE  = logformat.trace
JSRC     = DobjDef.java Print.java
JSRC4C   = InputLog.java

CLS      = $(JSRC:.java=.class)
CLS4C    = $(JSRC4C:.java=.class)
HDR      = $(CLS4C:.class=.h)

.SUFFIXES:	.java .class .jar .h

# CLASSDIR is where classfiles are located, all counted from the top
SRCDIR     = $(top_srcdir)
CLASSDIR   = ../..
CLASSPATH  = ../..
PKGDIR     = logformat/trace

JCFLAGS = -sourcepath $(SRCDIR) -classpath $(CLASSPATH) -d $(CLASSDIR) 

.java.class:
	$(JC) $(JCFLAGS) $(JFLAGS) $<

.class.h:
	$(JH) -classpath $(CLASSPATH) $(PACKAGE).$*


ALL: $(HDR) $(TARGET)

$(TARGET): $(CLS) $(CLS4C)
	cd $(CLASSDIR) && \
	$(RM) $(PACKAGE).manifest && \
	echo "Main-Class: $(PACKAGE).Print" > $(PACKAGE).manifest ; \
	jarfile=$@ ; \
	$(JAR) $(JARFLAGS)cfm $${jarfile} $(PACKAGE).manifest ; \
	for dir in base/io/ base/drawable/ $(PKGDIR)/ ; do \
	    classdir_prefix=$${dir} ; \
	    $(JAR) $(JARFLAGS)uf $${jarfile} $${classdir_prefix}*.class ; \
	done

clean:
	cd $(CLASSDIR) && \
	$(RM) $(PACKAGE).manifest $(PKGDIR)/*.class $(PKGDIR)/*.h

rmlib:
	$(RM) $(TARGET)

realclean: clean rmlib

distclean: realclean
	@-$(RM) Makefile
