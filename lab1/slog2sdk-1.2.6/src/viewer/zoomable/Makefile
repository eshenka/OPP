
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

TARGET    =
PACKAGE   = viewer.zoomable
SRC       = Diagnosis.java Debug.java Profile.java \
            YaxisTreeNode.java IntegerArrayComparator.java YaxisMaps.java \
            named_vector.java YaxisTree.java \
            TimeEvent.java TimeListener.java \
            ModelTime.java ScrollbarTime.java ModelTimePanel.java \
            TimeFormat.java InfoDialog.java InfoDialogForTime.java \
            InitializableDialog.java SummarizableView.java \
            SearchPanel.java SearchableView.java \
            OperationDurationPanel.java InfoDialogForDuration.java \
            SearchableView.java ScrollableView.java ScrollableObject.java \
            ToolBarStatus.java ViewportTime.java \
            ViewportTimeYaxis.java SearchDialog.java \
            RulerTime.java ViewportTimePanel.java CoordPixelImage.java \
            ScaledSlider.java RowAdjustments.java \
            ActionVportUp.java ActionVportDown.java \
            ActionTimelineMark.java ActionTimelineMove.java \
            ActionTimelineDelete.java \
            ActionYaxisTreeExpand.java ActionYaxisTreeCollapse.java \
            ActionYaxisTreeCommit.java \
            ActionVportBackward.java ActionVportForward.java \
            ActionVportUp.java ActionVportDown.java \
            ActionZoomIn.java ActionZoomHome.java ActionZoomOut.java \
            ActionZoomUndo.java ActionZoomRedo.java \
            ActionSearchInit.java \
            ActionSearchForward.java ActionSearchBackward.java \
            ActionPptyRefresh.java ActionPptyPrint.java ActionPptyStop.java

OBJ       = $(SRC:.java=.class)

TS_TARGET = $(libbuild_dir)/ts_time.jar
TS_SRC    = TestMainToolBar.java TestMainPanel.java TestMainFrame.java
TS_OBJ    = $(TS_SRC:.java=.class)

.SUFFIXES:	.java .class .jar

# CLASSPATH = .:/pkgs/lang/java/jdk117_v3/lib/rt.jar:/pkgs/lang/java/jdk117_v3/lib/classes.zip:/pkgs/lang/java/jdk117_v3/classes:/home/chan/mpe_work/mpe/viewers/jumpshot-3/swing/swing.jar
SRCDIR     = $(top_srcdir)
CLASSDIR   = ../..
CLASSPATH  = ../..
PKGDIR     = viewer/zoomable

JCFLAGS = -sourcepath $(SRCDIR) -classpath $(CLASSPATH) -d $(CLASSDIR) 

.java.class:
	$(JC) $(JCFLAGS) $(JFLAGS) $<

# echo "Class-Path: images" >> $(PACKAGE).manifest ;

ALL: $(OBJ)

# Only need base/drawable/TimeBoundingBox.class from base/drawable
$(TS_TARGET): $(OBJ) $(TS_OBJ)
	cd $(CLASSDIR) && \
	$(RM) $(PACKAGE).manifest && \
	echo "Main-Class: $(PACKAGE).TestMainFrame" > $(PACKAGE).manifest ; \
	jarfile=$@ ; \
	$(JAR) $(JARFLAGS)cfm $${jarfile} $(PACKAGE).manifest ; \
	for dir in base/drawable/ viewer/common/ $(PKGDIR)/ ; do \
	    classdir_prefix=$${dir} ; \
	    $(JAR) $(JARFLAGS)uf $${jarfile} $${classdir_prefix}*.class ; \
	done ; \
	$(JAR) $(JARFLAGS)uf $${jarfile} images/*.gif applet.html

clean:
	cd $(CLASSDIR) && \
	$(RM) $(PACKAGE).manifest $(PKGDIR)/*.class

rmlib:
	$(RM) $(TARGET) $(TS_TARGET)

realclean: clean rmlib

distclean: realclean
	@-$(RM) Makefile
