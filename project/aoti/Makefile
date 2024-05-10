#/************************************************************************************
#***
#***	File Author: Dell, 2024年 1月 22日
#***
#************************************************************************************/

INSTALL_DIR := /usr/local

TORCH_INSTALL_DIR=/home/dell/miniconda3/envs/python3.10/lib/python3.10/site-packages/torch


# Configurations
TARGET := aoti

SOURCE := inference.cpp

INCS	:= \
	-isystem ${TORCH_INSTALL_DIR}/include -isystem ${TORCH_INSTALL_DIR}/include/torch/csrc/api/include \

LIBS	:= \
	-L${TORCH_INSTALL_DIR}/lib -lc10 -ltorch -ltorch_cpu \
		-Wl,-rpath=${TORCH_INSTALL_DIR}/lib \
	-lpthread

CXXFLAGS := -fPIC -Wall -Wextra
LDFLAGS := -fPIC -O3


DEFINES := -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++17 -std=gnu++17 \


#****************************************************************************
# Makefile code common to all platforms
#****************************************************************************
CXXFLAGS := ${CXXFLAGS} ${DEFINES}

OBJECT := $(addsuffix .o,$(basename ${SOURCE}))

all: ${TARGET} 
# install

#****************************************************************************
# Compile block
#****************************************************************************
${TARGET}: ${OBJECT}
	${CXX} -o ${TARGET} ${LDFLAGS} ${OBJECT} ${LIBS}


#****************************************************************************
# common rules
#****************************************************************************
%.o : %.c
	${CXX} ${CXXFLAGS} ${INCS} -c $< -o $@

%.o : %.cpp
	${CXX} ${CXXFLAGS} ${INCS} -c $< -o $@

#****************************************************************************
# Depend block
#****************************************************************************
depend:

clean:
	rm -f core ${OBJECT} ${TARGET}
