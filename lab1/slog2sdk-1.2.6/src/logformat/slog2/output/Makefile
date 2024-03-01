
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

TARGET1 = $(libbuild_dir)/clogTOslog2.jar
TARGET2 = $(libbuild_dir)/clog2TOslog2.jar
TARGET3 = $(libbuild_dir)/traceTOslog2.jar
PACKAGE = logformat.slog2.output
SRC     = TreeNode.java OutputLog.java TreeTrunk.java \
          TraceToSlog2.java ClogToSlog2.java Clog2ToSlog2.java

OBJ     = $(SRC:.java=.class)

.SUFFIXES:      .java .class .jar

# CLASSDIR is where classfiles are located, all counted from the top
SRCDIR     = $(top_srcdir)
CLASSDIR   = ../../..
CLASSPATH  = ../../..
PKGDIR     = logformat/slog2/output

JCFLAGS = -sourcepath $(SRCDIR) -classpath $(CLASSPATH) -d $(CLASSDIR) 

.java.class:
	$(JC) $(JCFLAGS) $(JFLAGS) $<

ALL: $(TARGET1) $(TARGET2) $(TARGET3)

$(TARGET1): $(OBJ)
	cd $(CLASSDIR) && \
	$(RM) $(PACKAGE).manifest && \
	echo "Main-Class: $(PACKAGE).ClogToSlog2" > $(PACKAGE).manifest ; \
	jarfile=$@ ; \
	$(JAR) $(JARFLAGS)cfm $${jarfile} $(PACKAGE).manifest ; \
	for dir in base/io/ base/drawable/ logformat/slog2/ \
	           logformat/clog/ logformat/clogTOdrawable/ $(PKGDIR)/ ; do \
	    classdir_prefix=$${dir} ; \
	    $(JAR) $(JARFLAGS)uf $${jarfile} $${classdir_prefix}*.class ; \
	done ; \
	$(JAR) $(JARFLAGS)uf $${jarfile} jumpshot.colors

$(TARGET2): $(OBJ)
	cd $(CLASSDIR) && \
	$(RM) $(PACKAGE).manifest && \
	echo "Main-Class: $(PACKAGE).Clog2ToSlog2" > $(PACKAGE).manifest ; \
	jarfile=$@ ; \
	$(JAR) $(JARFLAGS)cfm $${jarfile} $(PACKAGE).manifest ; \
	for dir in base/io/ base/drawable/ logformat/slog2/ \
	           logformat/clog2/ logformat/clog2TOdrawable/ $(PKGDIR)/ ; do \
	    classdir_prefix=$${dir} ; \
	    $(JAR) $(JARFLAGS)uf $${jarfile} $${classdir_prefix}*.class ; \
	done ; \
	$(JAR) $(JARFLAGS)uf $${jarfile} jumpshot.colors

$(TARGET3): $(OBJ)
	cd $(CLASSDIR) && \
	$(RM) $(PACKAGE).manifest && \
	echo "Main-Class: $(PACKAGE).TraceToSlog2" > $(PACKAGE).manifest ; \
	jarfile=$@ ; \
	$(JAR) $(JARFLAGS)cfm $${jarfile} $(PACKAGE).manifest ; \
	for dir in base/io/ base/drawable/ logformat/slog2/ \
	           logformat/trace/ $(PKGDIR)/ ; do \
	    classdir_prefix=$${dir} ; \
	    $(JAR) $(JARFLAGS)uf $${jarfile} $${classdir_prefix}*.class ; \
	done

clean:
	cd $(CLASSDIR) && \
	$(RM) $(PACKAGE).manifest $(PKGDIR)/*.class 

rmlib:
	$(RM) $(TARGET1) $(TARGET2) $(TARGET3)

realclean: clean rmlib

distclean: realclean
	@-$(RM) Makefile
