
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

TARGET    = $(libbuild_dir)/legends.jar
PACKAGE   = viewer.legends
SRC       = Const.java CategoryIcon.java CategoryLabel.java \
            CategoryIconRenderer.java CategoryIconEditor.java \
            LegendComparators.java LegendTableModel.java \
            Triangular3DIcon.java GenericHeaderRenderer.java \
            TableHeaderHandler.java TableColumnHandler.java \
            OperationBooleanMenu.java OperationStringMenu.java \
            OperationNumberMenu.java \
            LegendTable.java LegendPanel.java LegendFrame.java

OBJ       = $(SRC:.java=.class)

.SUFFIXES:	.java .class .jar

# CLASSPATH = .:/pkgs/lang/java/jdk117_v3/lib/rt.jar:/pkgs/lang/java/jdk117_v3/lib/classes.zip:/pkgs/lang/java/jdk117_v3/classes:/home/chan/mpe_work/mpe/viewers/jumpshot-3/swing/swing.jar
SRCDIR     = $(top_srcdir)
CLASSDIR   = ../..
CLASSPATH  = ../..
PKGDIR     = viewer/legends

JCFLAGS = -sourcepath $(SRCDIR) -classpath $(CLASSPATH) -d $(CLASSDIR) 

.java.class:
	$(JC) $(JCFLAGS) $(JFLAGS) $<

ALL: $(OBJ)

$(TARGET): $(OBJ)
	cd $(CLASSDIR) && \
	$(RM) $(PACKAGE).manifest && \
	echo "Main-Class: $(PACKAGE).LegendFrame" > $(PACKAGE).manifest ; \
	jarfile=$@ ; \
	$(JAR) $(JARFLAGS)cfm $${jarfile} $(PACKAGE).manifest ; \
	for dir in base/io/ base/topology/ base/drawable/ \
	           logformat/slog2/ logformat/slog2/input/ \
	           viewer/common/ $(PKGDIR)/ ; do \
	    classdir_prefix=$${dir} ; \
	    $(JAR) $(JARFLAGS)uf $${jarfile} $${classdir_prefix}*.class ; \
	done ; \
	$(JAR) $(JARFLAGS)uf $${jarfile} images/*.gif applet.html

clean:
	cd $(CLASSDIR) && \
	$(RM) $(PKGDIR)/*.class

rmlib:
	$(RM) $(TARGET)

realclean: clean rmlib

distclean: realclean
	@-$(RM) Makefile
