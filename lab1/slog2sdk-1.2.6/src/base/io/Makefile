
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

TARGET  = $(libbuild_dir)/io.jar
PACKAGE = base.io
SRC     = MixedDataInput.java MixedDataOutput.java \
          MixedDataInputStream.java MixedDataOutputStream.java \
          MixedRandomAccessFile.java DataIO.java MixedDataIO.java \
          BufArrayOutputStream.java BufArrayPipedStream.java

OBJ    = $(SRC:.java=.class)

.SUFFIXES:      .java .class .jar

# CLASSDIR is where classfiles are located, all counted from the top
SRCDIR     = $(top_srcdir)
CLASSDIR   = ../..
CLASSPATH  = ../..
PKGDIR     = base/io

JCFLAGS = -sourcepath $(SRCDIR) -classpath $(CLASSPATH) -d $(CLASSDIR) 

.java.class:
	$(JC) $(JCFLAGS) $(JFLAGS) $<

ALL: $(TARGET)

$(TARGET): $(OBJ)
	cd $(CLASSDIR) && \
	$(RM) $(PACKAGE).manifest && \
	echo "Main-Class: $(PACKAGE).MixedRandomAccessFile" > $(PACKAGE).manifest ; \
	jarfile=$@ ; \
	classdir_prefix=$(PKGDIR)/ ; \
	$(JAR) $(JARFLAGS)cfm $${jarfile} $(PACKAGE).manifest ; \
        $(JAR) $(JARFLAGS)uf $${jarfile} $${classdir_prefix}*.class

clean:
	cd $(CLASSDIR) && \
	$(RM) $(PACKAGE).manifest $(PKGDIR)/*.class 

rmlib:
	$(RM) $(TARGET)

realclean: clean rmlib

distclean: realclean
	@-$(RM) Makefile
