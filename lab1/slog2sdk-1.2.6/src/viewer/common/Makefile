
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
PACKAGE   = viewer.common
SRC       = SwingWorker.java Dialogs.java RuntimeExecCommand.java \
            ActableTextField.java CustomCursor.java Routines.java \
            TopControl.java LabeledTextField.java LabeledFloatSlider.java \
            Alias.java Const.java LabeledComboBox.java Parameters.java \
            PreferencePanel.java TopWindow.java PreferenceFrame.java \
            LogPermitDirFilter.java LogRefuseDirFilter.java LogFileChooser.java

OBJ       = $(SRC:.java=.class)

.SUFFIXES:	.java .class .jar

# CLASSPATH = .:/pkgs/lang/java/jdk117_v3/lib/rt.jar:/pkgs/lang/java/jdk117_v3/lib/classes.zip:/pkgs/lang/java/jdk117_v3/classes:/home/chan/mpe_work/mpe/viewers/jumpshot-3/swing/swing.jar
SRCDIR     = $(top_srcdir)
CLASSDIR   = ../..
CLASSPATH  = ../..
PKGDIR     = viewer/common

JCFLAGS = -sourcepath $(SRCDIR) -classpath $(CLASSPATH) -d $(CLASSDIR) 

.java.class:
	$(JC) $(JCFLAGS) $(JFLAGS) $<

ALL: $(OBJ)

clean:
	cd $(CLASSDIR) && \
	$(RM) $(PKGDIR)/*.class

rmlib:

realclean: clean rmlib

distclean: realclean
	@-$(RM) Makefile
