       �K"	  ���Abrain.Event:2I�=�KT     �1��	[����A"��
l
PlaceholderPlaceholder*&
_output_shapes
:*
dtype0*
shape:
^
Placeholder_1Placeholder*
_output_shapes

:*
dtype0*
shape
:
p
Placeholder_2Placeholder*'
_output_shapes
:�N*
dtype0*
shape:�N
o
truncated_normal/shapeConst*
_output_shapes
:*
dtype0*%
valueB"             
Z
truncated_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    
\
truncated_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *���=
�
 truncated_normal/TruncatedNormalTruncatedNormaltruncated_normal/shape*&
_output_shapes
: *
dtype0*
seed2��*
seed���)*
T0
�
truncated_normal/mulMul truncated_normal/TruncatedNormaltruncated_normal/stddev*&
_output_shapes
: *
T0
u
truncated_normalAddtruncated_normal/multruncated_normal/mean*&
_output_shapes
: *
T0
�
Variable
VariableV2*&
_output_shapes
: *
dtype0*
shared_name *
	container *
shape: 
�
Variable/AssignAssignVariabletruncated_normal*&
_output_shapes
: *
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable
q
Variable/readIdentityVariable*&
_output_shapes
: *
T0*
_class
loc:@Variable
R
zerosConst*
_output_shapes
: *
dtype0*
valueB *    
v

Variable_1
VariableV2*
_output_shapes
: *
dtype0*
shared_name *
	container *
shape: 
�
Variable_1/AssignAssign
Variable_1zeros*
_output_shapes
: *
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_1
k
Variable_1/readIdentity
Variable_1*
_output_shapes
: *
T0*
_class
loc:@Variable_1
q
truncated_normal_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"          @   
\
truncated_normal_1/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    
^
truncated_normal_1/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *���=
�
"truncated_normal_1/TruncatedNormalTruncatedNormaltruncated_normal_1/shape*&
_output_shapes
: @*
dtype0*
seed2��*
seed���)*
T0
�
truncated_normal_1/mulMul"truncated_normal_1/TruncatedNormaltruncated_normal_1/stddev*&
_output_shapes
: @*
T0
{
truncated_normal_1Addtruncated_normal_1/multruncated_normal_1/mean*&
_output_shapes
: @*
T0
�

Variable_2
VariableV2*&
_output_shapes
: @*
dtype0*
shared_name *
	container *
shape: @
�
Variable_2/AssignAssign
Variable_2truncated_normal_1*&
_output_shapes
: @*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_2
w
Variable_2/readIdentity
Variable_2*&
_output_shapes
: @*
T0*
_class
loc:@Variable_2
R
ConstConst*
_output_shapes
:@*
dtype0*
valueB@*���=
v

Variable_3
VariableV2*
_output_shapes
:@*
dtype0*
shared_name *
	container *
shape:@
�
Variable_3/AssignAssign
Variable_3Const*
_output_shapes
:@*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_3
k
Variable_3/readIdentity
Variable_3*
_output_shapes
:@*
T0*
_class
loc:@Variable_3
i
truncated_normal_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"      
\
truncated_normal_2/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    
^
truncated_normal_2/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *���=
�
"truncated_normal_2/TruncatedNormalTruncatedNormaltruncated_normal_2/shape* 
_output_shapes
:
��*
dtype0*
seed2��*
seed���)*
T0
�
truncated_normal_2/mulMul"truncated_normal_2/TruncatedNormaltruncated_normal_2/stddev* 
_output_shapes
:
��*
T0
u
truncated_normal_2Addtruncated_normal_2/multruncated_normal_2/mean* 
_output_shapes
:
��*
T0
�

Variable_4
VariableV2* 
_output_shapes
:
��*
dtype0*
shared_name *
	container *
shape:
��
�
Variable_4/AssignAssign
Variable_4truncated_normal_2* 
_output_shapes
:
��*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_4
q
Variable_4/readIdentity
Variable_4* 
_output_shapes
:
��*
T0*
_class
loc:@Variable_4
V
Const_1Const*
_output_shapes	
:�*
dtype0*
valueB�*���=
x

Variable_5
VariableV2*
_output_shapes	
:�*
dtype0*
shared_name *
	container *
shape:�
�
Variable_5/AssignAssign
Variable_5Const_1*
_output_shapes	
:�*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_5
l
Variable_5/readIdentity
Variable_5*
_output_shapes	
:�*
T0*
_class
loc:@Variable_5
i
truncated_normal_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"      
\
truncated_normal_3/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    
^
truncated_normal_3/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *���=
�
"truncated_normal_3/TruncatedNormalTruncatedNormaltruncated_normal_3/shape*
_output_shapes
:	�*
dtype0*
seed2��*
seed���)*
T0
�
truncated_normal_3/mulMul"truncated_normal_3/TruncatedNormaltruncated_normal_3/stddev*
_output_shapes
:	�*
T0
t
truncated_normal_3Addtruncated_normal_3/multruncated_normal_3/mean*
_output_shapes
:	�*
T0
�

Variable_6
VariableV2*
_output_shapes
:	�*
dtype0*
shared_name *
	container *
shape:	�
�
Variable_6/AssignAssign
Variable_6truncated_normal_3*
_output_shapes
:	�*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_6
p
Variable_6/readIdentity
Variable_6*
_output_shapes
:	�*
T0*
_class
loc:@Variable_6
T
Const_2Const*
_output_shapes
:*
dtype0*
valueB*���=
v

Variable_7
VariableV2*
_output_shapes
:*
dtype0*
shared_name *
	container *
shape:
�
Variable_7/AssignAssign
Variable_7Const_2*
_output_shapes
:*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_7
k
Variable_7/readIdentity
Variable_7*
_output_shapes
:*
T0*
_class
loc:@Variable_7
�
Conv2DConv2DPlaceholderVariable/read*&
_output_shapes
: *
T0*
use_cudnn_on_gpu(*
strides
*
paddingSAME*
data_formatNHWC
s
BiasAddBiasAddConv2DVariable_1/read*&
_output_shapes
: *
data_formatNHWC*
T0
F
ReluReluBiasAdd*&
_output_shapes
: *
T0
�
MaxPoolMaxPoolRelu*&
_output_shapes
: *
ksize
*
T0*
data_formatNHWC*
paddingSAME*
strides

�
Conv2D_1Conv2DMaxPoolVariable_2/read*&
_output_shapes
:@*
T0*
use_cudnn_on_gpu(*
strides
*
paddingSAME*
data_formatNHWC
w
	BiasAdd_1BiasAddConv2D_1Variable_3/read*&
_output_shapes
:@*
data_formatNHWC*
T0
J
Relu_1Relu	BiasAdd_1*&
_output_shapes
:@*
T0
�
	MaxPool_1MaxPoolRelu_1*&
_output_shapes
:@*
ksize
*
T0*
data_formatNHWC*
paddingSAME*
strides

^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      
d
ReshapeReshape	MaxPool_1Reshape/shape*
_output_shapes
:	�*
Tshape0*
T0
z
MatMulMatMulReshapeVariable_4/read*
_output_shapes
:	�*
transpose_b( *
transpose_a( *
T0
M
addAddMatMulVariable_5/read*
_output_shapes
:	�*
T0
=
Relu_2Reluadd*
_output_shapes
:	�*
T0
z
MatMul_1MatMulRelu_2Variable_6/read*
_output_shapes

:*
transpose_b( *
transpose_a( *
T0
P
add_1AddMatMul_1Variable_7/read*
_output_shapes

:*
T0
d
Slice/beginConst*
_output_shapes
:*
dtype0*%
valueB"                
c

Slice/sizeConst*
_output_shapes
:*
dtype0*%
valueB"   ��������   
r
SliceSlicePlaceholderSlice/begin
Slice/size*&
_output_shapes
:*
Index0*
T0
`
Const_3Const*
_output_shapes
:*
dtype0*%
valueB"             
X
MinMinSliceConst_3*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
G
subSubSliceMin*&
_output_shapes
:*
T0
`
Const_4Const*
_output_shapes
:*
dtype0*%
valueB"             
V
MaxMaxsubConst_4*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
J
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  C
7
mulMulMaxmul/y*
_output_shapes
: *
T0
M
truedivRealDivsubmul*&
_output_shapes
:*
T0
d
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         
i
	Reshape_1ReshapetruedivReshape_1/shape*"
_output_shapes
:*
Tshape0*
T0
c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
k
	transpose	Transpose	Reshape_1transpose/perm*"
_output_shapes
:*
Tperm0*
T0
h
Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         
o
	Reshape_2Reshape	transposeReshape_2/shape*&
_output_shapes
:*
Tshape0*
T0
a
summary_data_0/tagConst*
_output_shapes
: *
dtype0*
valueB Bsummary_data_0
�
summary_data_0ImageSummarysummary_data_0/tag	Reshape_2*

max_images*
_output_shapes
: *
T0*
	bad_colorB:�  �
f
Slice_1/beginConst*
_output_shapes
:*
dtype0*%
valueB"                
e
Slice_1/sizeConst*
_output_shapes
:*
dtype0*%
valueB"   ��������   
s
Slice_1SliceConv2DSlice_1/beginSlice_1/size*&
_output_shapes
:*
Index0*
T0
`
Const_5Const*
_output_shapes
:*
dtype0*%
valueB"             
\
Min_1MinSlice_1Const_5*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
M
sub_1SubSlice_1Min_1*&
_output_shapes
:*
T0
`
Const_6Const*
_output_shapes
:*
dtype0*%
valueB"             
Z
Max_1Maxsub_1Const_6*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
L
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  C
=
mul_1MulMax_1mul_1/y*
_output_shapes
: *
T0
S
	truediv_1RealDivsub_1mul_1*&
_output_shapes
:*
T0
d
Reshape_3/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         
k
	Reshape_3Reshape	truediv_1Reshape_3/shape*"
_output_shapes
:*
Tshape0*
T0
e
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
o
transpose_1	Transpose	Reshape_3transpose_1/perm*"
_output_shapes
:*
Tperm0*
T0
h
Reshape_4/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         
q
	Reshape_4Reshapetranspose_1Reshape_4/shape*&
_output_shapes
:*
Tshape0*
T0
a
summary_conv_0/tagConst*
_output_shapes
: *
dtype0*
valueB Bsummary_conv_0
�
summary_conv_0ImageSummarysummary_conv_0/tag	Reshape_4*

max_images*
_output_shapes
: *
T0*
	bad_colorB:�  �
f
Slice_2/beginConst*
_output_shapes
:*
dtype0*%
valueB"                
e
Slice_2/sizeConst*
_output_shapes
:*
dtype0*%
valueB"   ��������   
t
Slice_2SliceMaxPoolSlice_2/beginSlice_2/size*&
_output_shapes
:*
Index0*
T0
`
Const_7Const*
_output_shapes
:*
dtype0*%
valueB"             
\
Min_2MinSlice_2Const_7*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
M
sub_2SubSlice_2Min_2*&
_output_shapes
:*
T0
`
Const_8Const*
_output_shapes
:*
dtype0*%
valueB"             
Z
Max_2Maxsub_2Const_8*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
L
mul_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  C
=
mul_2MulMax_2mul_2/y*
_output_shapes
: *
T0
S
	truediv_2RealDivsub_2mul_2*&
_output_shapes
:*
T0
d
Reshape_5/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         
k
	Reshape_5Reshape	truediv_2Reshape_5/shape*"
_output_shapes
:*
Tshape0*
T0
e
transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          
o
transpose_2	Transpose	Reshape_5transpose_2/perm*"
_output_shapes
:*
Tperm0*
T0
h
Reshape_6/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         
q
	Reshape_6Reshapetranspose_2Reshape_6/shape*&
_output_shapes
:*
Tshape0*
T0
a
summary_pool_0/tagConst*
_output_shapes
: *
dtype0*
valueB Bsummary_pool_0
�
summary_pool_0ImageSummarysummary_pool_0/tag	Reshape_6*

max_images*
_output_shapes
: *
T0*
	bad_colorB:�  �
f
Slice_3/beginConst*
_output_shapes
:*
dtype0*%
valueB"                
e
Slice_3/sizeConst*
_output_shapes
:*
dtype0*%
valueB"   ��������   
u
Slice_3SliceConv2D_1Slice_3/beginSlice_3/size*&
_output_shapes
:*
Index0*
T0
`
Const_9Const*
_output_shapes
:*
dtype0*%
valueB"             
\
Min_3MinSlice_3Const_9*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
M
sub_3SubSlice_3Min_3*&
_output_shapes
:*
T0
a
Const_10Const*
_output_shapes
:*
dtype0*%
valueB"             
[
Max_3Maxsub_3Const_10*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
L
mul_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *  C
=
mul_3MulMax_3mul_3/y*
_output_shapes
: *
T0
S
	truediv_3RealDivsub_3mul_3*&
_output_shapes
:*
T0
d
Reshape_7/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         
k
	Reshape_7Reshape	truediv_3Reshape_7/shape*"
_output_shapes
:*
Tshape0*
T0
e
transpose_3/permConst*
_output_shapes
:*
dtype0*!
valueB"          
o
transpose_3	Transpose	Reshape_7transpose_3/perm*"
_output_shapes
:*
Tperm0*
T0
h
Reshape_8/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         
q
	Reshape_8Reshapetranspose_3Reshape_8/shape*&
_output_shapes
:*
Tshape0*
T0
c
summary_conv2_0/tagConst*
_output_shapes
: *
dtype0* 
valueB Bsummary_conv2_0
�
summary_conv2_0ImageSummarysummary_conv2_0/tag	Reshape_8*

max_images*
_output_shapes
: *
T0*
	bad_colorB:�  �
f
Slice_4/beginConst*
_output_shapes
:*
dtype0*%
valueB"                
e
Slice_4/sizeConst*
_output_shapes
:*
dtype0*%
valueB"   ��������   
v
Slice_4Slice	MaxPool_1Slice_4/beginSlice_4/size*&
_output_shapes
:*
Index0*
T0
a
Const_11Const*
_output_shapes
:*
dtype0*%
valueB"             
]
Min_4MinSlice_4Const_11*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
M
sub_4SubSlice_4Min_4*&
_output_shapes
:*
T0
a
Const_12Const*
_output_shapes
:*
dtype0*%
valueB"             
[
Max_4Maxsub_4Const_12*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
L
mul_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *  C
=
mul_4MulMax_4mul_4/y*
_output_shapes
: *
T0
S
	truediv_4RealDivsub_4mul_4*&
_output_shapes
:*
T0
d
Reshape_9/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         
k
	Reshape_9Reshape	truediv_4Reshape_9/shape*"
_output_shapes
:*
Tshape0*
T0
e
transpose_4/permConst*
_output_shapes
:*
dtype0*!
valueB"          
o
transpose_4	Transpose	Reshape_9transpose_4/perm*"
_output_shapes
:*
Tperm0*
T0
i
Reshape_10/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         
s

Reshape_10Reshapetranspose_4Reshape_10/shape*&
_output_shapes
:*
Tshape0*
T0
c
summary_pool2_0/tagConst*
_output_shapes
: *
dtype0* 
valueB Bsummary_pool2_0
�
summary_pool2_0ImageSummarysummary_pool2_0/tag
Reshape_10*

max_images*
_output_shapes
: *
T0*
	bad_colorB:�  �
F
RankConst*
_output_shapes
: *
dtype0*
value	B :
V
ShapeConst*
_output_shapes
:*
dtype0*
valueB"      
H
Rank_1Const*
_output_shapes
: *
dtype0*
value	B :
X
Shape_1Const*
_output_shapes
:*
dtype0*
valueB"      
G
Sub/yConst*
_output_shapes
: *
dtype0*
value	B :
:
SubSubRank_1Sub/y*
_output_shapes
: *
T0
T
Slice_5/beginPackSub*

axis *
_output_shapes
:*
N*
T0
V
Slice_5/sizeConst*
_output_shapes
:*
dtype0*
valueB:
h
Slice_5SliceShape_1Slice_5/beginSlice_5/size*
_output_shapes
:*
Index0*
T0
b
concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:
���������
M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
s
concatConcatV2concat/values_0Slice_5concat/axis*
_output_shapes
:*

Tidx0*
N*
T0
[

Reshape_11Reshapeadd_1concat*
_output_shapes

:*
Tshape0*
T0
H
Rank_2Const*
_output_shapes
: *
dtype0*
value	B :
X
Shape_2Const*
_output_shapes
:*
dtype0*
valueB"      
I
Sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :
>
Sub_1SubRank_2Sub_1/y*
_output_shapes
: *
T0
V
Slice_6/beginPackSub_1*

axis *
_output_shapes
:*
N*
T0
V
Slice_6/sizeConst*
_output_shapes
:*
dtype0*
valueB:
h
Slice_6SliceShape_2Slice_6/beginSlice_6/size*
_output_shapes
:*
Index0*
T0
d
concat_1/values_0Const*
_output_shapes
:*
dtype0*
valueB:
���������
O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
y
concat_1ConcatV2concat_1/values_0Slice_6concat_1/axis*
_output_shapes
:*

Tidx0*
N*
T0
e

Reshape_12ReshapePlaceholder_1concat_1*
_output_shapes

:*
Tshape0*
T0
�
SoftmaxCrossEntropyWithLogitsSoftmaxCrossEntropyWithLogits
Reshape_11
Reshape_12*$
_output_shapes
::*
T0
I
Sub_2/yConst*
_output_shapes
: *
dtype0*
value	B :
<
Sub_2SubRankSub_2/y*
_output_shapes
: *
T0
W
Slice_7/beginConst*
_output_shapes
:*
dtype0*
valueB: 
U
Slice_7/sizePackSub_2*

axis *
_output_shapes
:*
N*
T0
o
Slice_7SliceShapeSlice_7/beginSlice_7/size*#
_output_shapes
:���������*
Index0*
T0
p

Reshape_13ReshapeSoftmaxCrossEntropyWithLogitsSlice_7*
_output_shapes
:*
Tshape0*
T0
R
Const_13Const*
_output_shapes
:*
dtype0*
valueB: 
`
MeanMean
Reshape_13Const_13*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
N
	loss/tagsConst*
_output_shapes
: *
dtype0*
valueB
 Bloss
G
lossScalarSummary	loss/tagsMean*
_output_shapes
: *
T0
R
gradients/ShapeConst*
_output_shapes
: *
dtype0*
valueB 
T
gradients/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?
Y
gradients/FillFillgradients/Shapegradients/Const*
_output_shapes
: *
T0
k
!gradients/Mean_grad/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
�
gradients/Mean_grad/ReshapeReshapegradients/Fill!gradients/Mean_grad/Reshape/shape*
_output_shapes
:*
Tshape0*
T0
l
"gradients/Mean_grad/Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB:
�
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshape"gradients/Mean_grad/Tile/multiples*
_output_shapes
:*

Tmultiples0*
T0
c
gradients/Mean_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:
^
gradients/Mean_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
�
gradients/Mean_grad/ConstConst*
_output_shapes
:*
dtype0*
valueB: *,
_class"
 loc:@gradients/Mean_grad/Shape
�
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shapegradients/Mean_grad/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0*,
_class"
 loc:@gradients/Mean_grad/Shape
�
gradients/Mean_grad/Const_1Const*
_output_shapes
:*
dtype0*
valueB: *,
_class"
 loc:@gradients/Mean_grad/Shape
�
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_1gradients/Mean_grad/Const_1*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0*,
_class"
 loc:@gradients/Mean_grad/Shape
�
gradients/Mean_grad/Maximum/yConst*
_output_shapes
: *
dtype0*
value	B :*,
_class"
 loc:@gradients/Mean_grad/Shape
�
gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*
_output_shapes
: *
T0*,
_class"
 loc:@gradients/Mean_grad/Shape
�
gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum*
_output_shapes
: *
T0*,
_class"
 loc:@gradients/Mean_grad/Shape
n
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv*
_output_shapes
: *

SrcT0*

DstT0

gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*
_output_shapes
:*
T0
i
gradients/Reshape_13_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:
�
!gradients/Reshape_13_grad/ReshapeReshapegradients/Mean_grad/truedivgradients/Reshape_13_grad/Shape*
_output_shapes
:*
Tshape0*
T0
k
gradients/zeros_like	ZerosLikeSoftmaxCrossEntropyWithLogits:1*
_output_shapes

:*
T0
�
;gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������
�
7gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims
ExpandDims!gradients/Reshape_13_grad/Reshape;gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim*
_output_shapes

:*
T0*

Tdim0
�
0gradients/SoftmaxCrossEntropyWithLogits_grad/mulMul7gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDimsSoftmaxCrossEntropyWithLogits:1*
_output_shapes

:*
T0
p
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      
�
!gradients/Reshape_11_grad/ReshapeReshape0gradients/SoftmaxCrossEntropyWithLogits_grad/mulgradients/Reshape_11_grad/Shape*
_output_shapes

:*
Tshape0*
T0
k
gradients/add_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      
f
gradients/add_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:
�
*gradients/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_1_grad/Shapegradients/add_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients/add_1_grad/SumSum!gradients/Reshape_11_grad/Reshape*gradients/add_1_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
gradients/add_1_grad/ReshapeReshapegradients/add_1_grad/Sumgradients/add_1_grad/Shape*
_output_shapes

:*
Tshape0*
T0
�
gradients/add_1_grad/Sum_1Sum!gradients/Reshape_11_grad/Reshape,gradients/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
gradients/add_1_grad/Reshape_1Reshapegradients/add_1_grad/Sum_1gradients/add_1_grad/Shape_1*
_output_shapes
:*
Tshape0*
T0
�
gradients/MatMul_1_grad/MatMulMatMulgradients/add_1_grad/ReshapeVariable_6/read*
_output_shapes
:	�*
transpose_b(*
transpose_a( *
T0
�
 gradients/MatMul_1_grad/MatMul_1MatMulRelu_2gradients/add_1_grad/Reshape*
_output_shapes
:	�*
transpose_b( *
transpose_a(*
T0
|
gradients/Relu_2_grad/ReluGradReluGradgradients/MatMul_1_grad/MatMulRelu_2*
_output_shapes
:	�*
T0
i
gradients/add_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      
e
gradients/add_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:�
�
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients/add_grad/SumSumgradients/Relu_2_grad/ReluGrad(gradients/add_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*
_output_shapes
:	�*
Tshape0*
T0
�
gradients/add_grad/Sum_1Sumgradients/Relu_2_grad/ReluGrad*gradients/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
_output_shapes	
:�*
Tshape0*
T0
�
gradients/MatMul_grad/MatMulMatMulgradients/add_grad/ReshapeVariable_4/read*
_output_shapes
:	�*
transpose_b(*
transpose_a( *
T0
�
gradients/MatMul_grad/MatMul_1MatMulReshapegradients/add_grad/Reshape* 
_output_shapes
:
��*
transpose_b( *
transpose_a(*
T0
u
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"         @   
�
gradients/Reshape_grad/ReshapeReshapegradients/MatMul_grad/MatMulgradients/Reshape_grad/Shape*&
_output_shapes
:@*
Tshape0*
T0
�
$gradients/MaxPool_1_grad/MaxPoolGradMaxPoolGradRelu_1	MaxPool_1gradients/Reshape_grad/Reshape*&
_output_shapes
:@*
ksize
*
T0*
data_formatNHWC*
paddingSAME*
strides

�
gradients/Relu_1_grad/ReluGradReluGrad$gradients/MaxPool_1_grad/MaxPoolGradRelu_1*&
_output_shapes
:@*
T0
�
$gradients/BiasAdd_1_grad/BiasAddGradBiasAddGradgradients/Relu_1_grad/ReluGrad*
_output_shapes
:@*
data_formatNHWC*
T0
�
gradients/Conv2D_1_grad/ShapeNShapeNMaxPoolVariable_2/read* 
_output_shapes
::*
N*
T0*
out_type0
�
+gradients/Conv2D_1_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_1_grad/ShapeNVariable_2/readgradients/Relu_1_grad/ReluGrad*&
_output_shapes
: *
T0*
use_cudnn_on_gpu(*
strides
*
paddingSAME*
data_formatNHWC
�
,gradients/Conv2D_1_grad/Conv2DBackpropFilterConv2DBackpropFilterMaxPool gradients/Conv2D_1_grad/ShapeN:1gradients/Relu_1_grad/ReluGrad*&
_output_shapes
: @*
T0*
use_cudnn_on_gpu(*
strides
*
paddingSAME*
data_formatNHWC
�
"gradients/MaxPool_grad/MaxPoolGradMaxPoolGradReluMaxPool+gradients/Conv2D_1_grad/Conv2DBackpropInput*&
_output_shapes
: *
ksize
*
T0*
data_formatNHWC*
paddingSAME*
strides

�
gradients/Relu_grad/ReluGradReluGrad"gradients/MaxPool_grad/MaxPoolGradRelu*&
_output_shapes
: *
T0
�
"gradients/BiasAdd_grad/BiasAddGradBiasAddGradgradients/Relu_grad/ReluGrad*
_output_shapes
: *
data_formatNHWC*
T0
�
gradients/Conv2D_grad/ShapeNShapeNPlaceholderVariable/read* 
_output_shapes
::*
N*
T0*
out_type0
�
)gradients/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_grad/ShapeNVariable/readgradients/Relu_grad/ReluGrad*&
_output_shapes
:*
T0*
use_cudnn_on_gpu(*
strides
*
paddingSAME*
data_formatNHWC
�
*gradients/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterPlaceholdergradients/Conv2D_grad/ShapeN:1gradients/Relu_grad/ReluGrad*&
_output_shapes
: *
T0*
use_cudnn_on_gpu(*
strides
*
paddingSAME*
data_formatNHWC
�
global_norm/L2LossL2Loss*gradients/Conv2D_grad/Conv2DBackpropFilter*
_output_shapes
: *
T0*=
_class3
1/loc:@gradients/Conv2D_grad/Conv2DBackpropFilter
g
global_norm/stackPackglobal_norm/L2Loss*

axis *
_output_shapes
:*
N*
T0
[
global_norm/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
z
global_norm/SumSumglobal_norm/stackglobal_norm/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
X
global_norm/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   @
]
global_norm/mulMulglobal_norm/Sumglobal_norm/Const_1*
_output_shapes
: *
T0
Q
global_norm/global_normSqrtglobal_norm/mul*
_output_shapes
: *
T0
`
conv1_weights/tagsConst*
_output_shapes
: *
dtype0*
valueB Bconv1_weights
l
conv1_weightsScalarSummaryconv1_weights/tagsglobal_norm/global_norm*
_output_shapes
: *
T0
�
global_norm_1/L2LossL2Loss"gradients/BiasAdd_grad/BiasAddGrad*
_output_shapes
: *
T0*5
_class+
)'loc:@gradients/BiasAdd_grad/BiasAddGrad
k
global_norm_1/stackPackglobal_norm_1/L2Loss*

axis *
_output_shapes
:*
N*
T0
]
global_norm_1/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
�
global_norm_1/SumSumglobal_norm_1/stackglobal_norm_1/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
Z
global_norm_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   @
c
global_norm_1/mulMulglobal_norm_1/Sumglobal_norm_1/Const_1*
_output_shapes
: *
T0
U
global_norm_1/global_normSqrtglobal_norm_1/mul*
_output_shapes
: *
T0
^
conv1_biases/tagsConst*
_output_shapes
: *
dtype0*
valueB Bconv1_biases
l
conv1_biasesScalarSummaryconv1_biases/tagsglobal_norm_1/global_norm*
_output_shapes
: *
T0
�
global_norm_2/L2LossL2Loss,gradients/Conv2D_1_grad/Conv2DBackpropFilter*
_output_shapes
: *
T0*?
_class5
31loc:@gradients/Conv2D_1_grad/Conv2DBackpropFilter
k
global_norm_2/stackPackglobal_norm_2/L2Loss*

axis *
_output_shapes
:*
N*
T0
]
global_norm_2/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
�
global_norm_2/SumSumglobal_norm_2/stackglobal_norm_2/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
Z
global_norm_2/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   @
c
global_norm_2/mulMulglobal_norm_2/Sumglobal_norm_2/Const_1*
_output_shapes
: *
T0
U
global_norm_2/global_normSqrtglobal_norm_2/mul*
_output_shapes
: *
T0
`
conv2_weights/tagsConst*
_output_shapes
: *
dtype0*
valueB Bconv2_weights
n
conv2_weightsScalarSummaryconv2_weights/tagsglobal_norm_2/global_norm*
_output_shapes
: *
T0
�
global_norm_3/L2LossL2Loss$gradients/BiasAdd_1_grad/BiasAddGrad*
_output_shapes
: *
T0*7
_class-
+)loc:@gradients/BiasAdd_1_grad/BiasAddGrad
k
global_norm_3/stackPackglobal_norm_3/L2Loss*

axis *
_output_shapes
:*
N*
T0
]
global_norm_3/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
�
global_norm_3/SumSumglobal_norm_3/stackglobal_norm_3/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
Z
global_norm_3/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   @
c
global_norm_3/mulMulglobal_norm_3/Sumglobal_norm_3/Const_1*
_output_shapes
: *
T0
U
global_norm_3/global_normSqrtglobal_norm_3/mul*
_output_shapes
: *
T0
^
conv2_biases/tagsConst*
_output_shapes
: *
dtype0*
valueB Bconv2_biases
l
conv2_biasesScalarSummaryconv2_biases/tagsglobal_norm_3/global_norm*
_output_shapes
: *
T0
�
global_norm_4/L2LossL2Lossgradients/MatMul_grad/MatMul_1*
_output_shapes
: *
T0*1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1
k
global_norm_4/stackPackglobal_norm_4/L2Loss*

axis *
_output_shapes
:*
N*
T0
]
global_norm_4/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
�
global_norm_4/SumSumglobal_norm_4/stackglobal_norm_4/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
Z
global_norm_4/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   @
c
global_norm_4/mulMulglobal_norm_4/Sumglobal_norm_4/Const_1*
_output_shapes
: *
T0
U
global_norm_4/global_normSqrtglobal_norm_4/mul*
_output_shapes
: *
T0
\
fc1_weights/tagsConst*
_output_shapes
: *
dtype0*
valueB Bfc1_weights
j
fc1_weightsScalarSummaryfc1_weights/tagsglobal_norm_4/global_norm*
_output_shapes
: *
T0
�
global_norm_5/L2LossL2Lossgradients/add_grad/Reshape_1*
_output_shapes
: *
T0*/
_class%
#!loc:@gradients/add_grad/Reshape_1
k
global_norm_5/stackPackglobal_norm_5/L2Loss*

axis *
_output_shapes
:*
N*
T0
]
global_norm_5/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
�
global_norm_5/SumSumglobal_norm_5/stackglobal_norm_5/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
Z
global_norm_5/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   @
c
global_norm_5/mulMulglobal_norm_5/Sumglobal_norm_5/Const_1*
_output_shapes
: *
T0
U
global_norm_5/global_normSqrtglobal_norm_5/mul*
_output_shapes
: *
T0
Z
fc1_biases/tagsConst*
_output_shapes
: *
dtype0*
valueB B
fc1_biases
h

fc1_biasesScalarSummaryfc1_biases/tagsglobal_norm_5/global_norm*
_output_shapes
: *
T0
�
global_norm_6/L2LossL2Loss gradients/MatMul_1_grad/MatMul_1*
_output_shapes
: *
T0*3
_class)
'%loc:@gradients/MatMul_1_grad/MatMul_1
k
global_norm_6/stackPackglobal_norm_6/L2Loss*

axis *
_output_shapes
:*
N*
T0
]
global_norm_6/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
�
global_norm_6/SumSumglobal_norm_6/stackglobal_norm_6/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
Z
global_norm_6/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   @
c
global_norm_6/mulMulglobal_norm_6/Sumglobal_norm_6/Const_1*
_output_shapes
: *
T0
U
global_norm_6/global_normSqrtglobal_norm_6/mul*
_output_shapes
: *
T0
\
fc2_weights/tagsConst*
_output_shapes
: *
dtype0*
valueB Bfc2_weights
j
fc2_weightsScalarSummaryfc2_weights/tagsglobal_norm_6/global_norm*
_output_shapes
: *
T0
�
global_norm_7/L2LossL2Lossgradients/add_1_grad/Reshape_1*
_output_shapes
: *
T0*1
_class'
%#loc:@gradients/add_1_grad/Reshape_1
k
global_norm_7/stackPackglobal_norm_7/L2Loss*

axis *
_output_shapes
:*
N*
T0
]
global_norm_7/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
�
global_norm_7/SumSumglobal_norm_7/stackglobal_norm_7/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
Z
global_norm_7/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   @
c
global_norm_7/mulMulglobal_norm_7/Sumglobal_norm_7/Const_1*
_output_shapes
: *
T0
U
global_norm_7/global_normSqrtglobal_norm_7/mul*
_output_shapes
: *
T0
Z
fc2_biases/tagsConst*
_output_shapes
: *
dtype0*
valueB B
fc2_biases
h

fc2_biasesScalarSummaryfc2_biases/tagsglobal_norm_7/global_norm*
_output_shapes
: *
T0
B
L2LossL2LossVariable_4/read*
_output_shapes
: *
T0
D
L2Loss_1L2LossVariable_5/read*
_output_shapes
: *
T0
?
add_2AddL2LossL2Loss_1*
_output_shapes
: *
T0
D
L2Loss_2L2LossVariable_6/read*
_output_shapes
: *
T0
>
add_3Addadd_2L2Loss_2*
_output_shapes
: *
T0
D
L2Loss_3L2LossVariable_7/read*
_output_shapes
: *
T0
>
add_4Addadd_3L2Loss_3*
_output_shapes
: *
T0
L
mul_5/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:
=
mul_5Mulmul_5/xadd_4*
_output_shapes
: *
T0
:
add_5AddMeanmul_5*
_output_shapes
: *
T0
Z
Variable_8/initial_valueConst*
_output_shapes
: *
dtype0*
value	B : 
n

Variable_8
VariableV2*
_output_shapes
: *
dtype0*
shared_name *
	container *
shape: 
�
Variable_8/AssignAssign
Variable_8Variable_8/initial_value*
_output_shapes
: *
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_8
g
Variable_8/readIdentity
Variable_8*
_output_shapes
: *
T0*
_class
loc:@Variable_8
I
mul_6/yConst*
_output_shapes
: *
dtype0*
value	B :
G
mul_6MulVariable_8/readmul_6/y*
_output_shapes
: *
T0
c
ExponentialDecay/learning_rateConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<
T
ExponentialDecay/CastCastmul_6*
_output_shapes
: *

SrcT0*

DstT0
\
ExponentialDecay/Cast_1/xConst*
_output_shapes
: *
dtype0*
value
B :�N
j
ExponentialDecay/Cast_1CastExponentialDecay/Cast_1/x*
_output_shapes
: *

SrcT0*

DstT0
^
ExponentialDecay/Cast_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *33s?
t
ExponentialDecay/truedivRealDivExponentialDecay/CastExponentialDecay/Cast_1*
_output_shapes
: *
T0
Z
ExponentialDecay/FloorFloorExponentialDecay/truediv*
_output_shapes
: *
T0
o
ExponentialDecay/PowPowExponentialDecay/Cast_2/xExponentialDecay/Floor*
_output_shapes
: *
T0
n
ExponentialDecayMulExponentialDecay/learning_rateExponentialDecay/Pow*
_output_shapes
: *
T0
`
learning_rate/tagsConst*
_output_shapes
: *
dtype0*
valueB Blearning_rate
e
learning_rateScalarSummarylearning_rate/tagsExponentialDecay*
_output_shapes
: *
T0
T
gradients_1/ShapeConst*
_output_shapes
: *
dtype0*
valueB 
V
gradients_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?
_
gradients_1/FillFillgradients_1/Shapegradients_1/Const*
_output_shapes
: *
T0
_
gradients_1/add_5_grad/ShapeConst*
_output_shapes
: *
dtype0*
valueB 
a
gradients_1/add_5_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
�
,gradients_1/add_5_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/add_5_grad/Shapegradients_1/add_5_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients_1/add_5_grad/SumSumgradients_1/Fill,gradients_1/add_5_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
gradients_1/add_5_grad/ReshapeReshapegradients_1/add_5_grad/Sumgradients_1/add_5_grad/Shape*
_output_shapes
: *
Tshape0*
T0
�
gradients_1/add_5_grad/Sum_1Sumgradients_1/Fill.gradients_1/add_5_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
 gradients_1/add_5_grad/Reshape_1Reshapegradients_1/add_5_grad/Sum_1gradients_1/add_5_grad/Shape_1*
_output_shapes
: *
Tshape0*
T0
s
'gradients_1/add_5_grad/tuple/group_depsNoOp^gradients_1/add_5_grad/Reshape!^gradients_1/add_5_grad/Reshape_1
�
/gradients_1/add_5_grad/tuple/control_dependencyIdentitygradients_1/add_5_grad/Reshape(^gradients_1/add_5_grad/tuple/group_deps*
_output_shapes
: *
T0*1
_class'
%#loc:@gradients_1/add_5_grad/Reshape
�
1gradients_1/add_5_grad/tuple/control_dependency_1Identity gradients_1/add_5_grad/Reshape_1(^gradients_1/add_5_grad/tuple/group_deps*
_output_shapes
: *
T0*3
_class)
'%loc:@gradients_1/add_5_grad/Reshape_1
m
#gradients_1/Mean_grad/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
�
gradients_1/Mean_grad/ReshapeReshape/gradients_1/add_5_grad/tuple/control_dependency#gradients_1/Mean_grad/Reshape/shape*
_output_shapes
:*
Tshape0*
T0
n
$gradients_1/Mean_grad/Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB:
�
gradients_1/Mean_grad/TileTilegradients_1/Mean_grad/Reshape$gradients_1/Mean_grad/Tile/multiples*
_output_shapes
:*

Tmultiples0*
T0
e
gradients_1/Mean_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:
`
gradients_1/Mean_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
�
gradients_1/Mean_grad/ConstConst*
_output_shapes
:*
dtype0*
valueB: *.
_class$
" loc:@gradients_1/Mean_grad/Shape
�
gradients_1/Mean_grad/ProdProdgradients_1/Mean_grad/Shapegradients_1/Mean_grad/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0*.
_class$
" loc:@gradients_1/Mean_grad/Shape
�
gradients_1/Mean_grad/Const_1Const*
_output_shapes
:*
dtype0*
valueB: *.
_class$
" loc:@gradients_1/Mean_grad/Shape
�
gradients_1/Mean_grad/Prod_1Prodgradients_1/Mean_grad/Shape_1gradients_1/Mean_grad/Const_1*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0*.
_class$
" loc:@gradients_1/Mean_grad/Shape
�
gradients_1/Mean_grad/Maximum/yConst*
_output_shapes
: *
dtype0*
value	B :*.
_class$
" loc:@gradients_1/Mean_grad/Shape
�
gradients_1/Mean_grad/MaximumMaximumgradients_1/Mean_grad/Prod_1gradients_1/Mean_grad/Maximum/y*
_output_shapes
: *
T0*.
_class$
" loc:@gradients_1/Mean_grad/Shape
�
gradients_1/Mean_grad/floordivFloorDivgradients_1/Mean_grad/Prodgradients_1/Mean_grad/Maximum*
_output_shapes
: *
T0*.
_class$
" loc:@gradients_1/Mean_grad/Shape
r
gradients_1/Mean_grad/CastCastgradients_1/Mean_grad/floordiv*
_output_shapes
: *

SrcT0*

DstT0
�
gradients_1/Mean_grad/truedivRealDivgradients_1/Mean_grad/Tilegradients_1/Mean_grad/Cast*
_output_shapes
:*
T0
_
gradients_1/mul_5_grad/ShapeConst*
_output_shapes
: *
dtype0*
valueB 
a
gradients_1/mul_5_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
�
,gradients_1/mul_5_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/mul_5_grad/Shapegradients_1/mul_5_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
|
gradients_1/mul_5_grad/mulMul1gradients_1/add_5_grad/tuple/control_dependency_1add_4*
_output_shapes
: *
T0
�
gradients_1/mul_5_grad/SumSumgradients_1/mul_5_grad/mul,gradients_1/mul_5_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
gradients_1/mul_5_grad/ReshapeReshapegradients_1/mul_5_grad/Sumgradients_1/mul_5_grad/Shape*
_output_shapes
: *
Tshape0*
T0
�
gradients_1/mul_5_grad/mul_1Mulmul_5/x1gradients_1/add_5_grad/tuple/control_dependency_1*
_output_shapes
: *
T0
�
gradients_1/mul_5_grad/Sum_1Sumgradients_1/mul_5_grad/mul_1.gradients_1/mul_5_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
 gradients_1/mul_5_grad/Reshape_1Reshapegradients_1/mul_5_grad/Sum_1gradients_1/mul_5_grad/Shape_1*
_output_shapes
: *
Tshape0*
T0
s
'gradients_1/mul_5_grad/tuple/group_depsNoOp^gradients_1/mul_5_grad/Reshape!^gradients_1/mul_5_grad/Reshape_1
�
/gradients_1/mul_5_grad/tuple/control_dependencyIdentitygradients_1/mul_5_grad/Reshape(^gradients_1/mul_5_grad/tuple/group_deps*
_output_shapes
: *
T0*1
_class'
%#loc:@gradients_1/mul_5_grad/Reshape
�
1gradients_1/mul_5_grad/tuple/control_dependency_1Identity gradients_1/mul_5_grad/Reshape_1(^gradients_1/mul_5_grad/tuple/group_deps*
_output_shapes
: *
T0*3
_class)
'%loc:@gradients_1/mul_5_grad/Reshape_1
k
!gradients_1/Reshape_13_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:
�
#gradients_1/Reshape_13_grad/ReshapeReshapegradients_1/Mean_grad/truediv!gradients_1/Reshape_13_grad/Shape*
_output_shapes
:*
Tshape0*
T0
_
gradients_1/add_4_grad/ShapeConst*
_output_shapes
: *
dtype0*
valueB 
a
gradients_1/add_4_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
�
,gradients_1/add_4_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/add_4_grad/Shapegradients_1/add_4_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients_1/add_4_grad/SumSum1gradients_1/mul_5_grad/tuple/control_dependency_1,gradients_1/add_4_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
gradients_1/add_4_grad/ReshapeReshapegradients_1/add_4_grad/Sumgradients_1/add_4_grad/Shape*
_output_shapes
: *
Tshape0*
T0
�
gradients_1/add_4_grad/Sum_1Sum1gradients_1/mul_5_grad/tuple/control_dependency_1.gradients_1/add_4_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
 gradients_1/add_4_grad/Reshape_1Reshapegradients_1/add_4_grad/Sum_1gradients_1/add_4_grad/Shape_1*
_output_shapes
: *
Tshape0*
T0
s
'gradients_1/add_4_grad/tuple/group_depsNoOp^gradients_1/add_4_grad/Reshape!^gradients_1/add_4_grad/Reshape_1
�
/gradients_1/add_4_grad/tuple/control_dependencyIdentitygradients_1/add_4_grad/Reshape(^gradients_1/add_4_grad/tuple/group_deps*
_output_shapes
: *
T0*1
_class'
%#loc:@gradients_1/add_4_grad/Reshape
�
1gradients_1/add_4_grad/tuple/control_dependency_1Identity gradients_1/add_4_grad/Reshape_1(^gradients_1/add_4_grad/tuple/group_deps*
_output_shapes
: *
T0*3
_class)
'%loc:@gradients_1/add_4_grad/Reshape_1
m
gradients_1/zeros_like	ZerosLikeSoftmaxCrossEntropyWithLogits:1*
_output_shapes

:*
T0
�
=gradients_1/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������
�
9gradients_1/SoftmaxCrossEntropyWithLogits_grad/ExpandDims
ExpandDims#gradients_1/Reshape_13_grad/Reshape=gradients_1/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim*
_output_shapes

:*
T0*

Tdim0
�
2gradients_1/SoftmaxCrossEntropyWithLogits_grad/mulMul9gradients_1/SoftmaxCrossEntropyWithLogits_grad/ExpandDimsSoftmaxCrossEntropyWithLogits:1*
_output_shapes

:*
T0
_
gradients_1/add_3_grad/ShapeConst*
_output_shapes
: *
dtype0*
valueB 
a
gradients_1/add_3_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
�
,gradients_1/add_3_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/add_3_grad/Shapegradients_1/add_3_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients_1/add_3_grad/SumSum/gradients_1/add_4_grad/tuple/control_dependency,gradients_1/add_3_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
gradients_1/add_3_grad/ReshapeReshapegradients_1/add_3_grad/Sumgradients_1/add_3_grad/Shape*
_output_shapes
: *
Tshape0*
T0
�
gradients_1/add_3_grad/Sum_1Sum/gradients_1/add_4_grad/tuple/control_dependency.gradients_1/add_3_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
 gradients_1/add_3_grad/Reshape_1Reshapegradients_1/add_3_grad/Sum_1gradients_1/add_3_grad/Shape_1*
_output_shapes
: *
Tshape0*
T0
s
'gradients_1/add_3_grad/tuple/group_depsNoOp^gradients_1/add_3_grad/Reshape!^gradients_1/add_3_grad/Reshape_1
�
/gradients_1/add_3_grad/tuple/control_dependencyIdentitygradients_1/add_3_grad/Reshape(^gradients_1/add_3_grad/tuple/group_deps*
_output_shapes
: *
T0*1
_class'
%#loc:@gradients_1/add_3_grad/Reshape
�
1gradients_1/add_3_grad/tuple/control_dependency_1Identity gradients_1/add_3_grad/Reshape_1(^gradients_1/add_3_grad/tuple/group_deps*
_output_shapes
: *
T0*3
_class)
'%loc:@gradients_1/add_3_grad/Reshape_1
�
gradients_1/L2Loss_3_grad/mulMulVariable_7/read1gradients_1/add_4_grad/tuple/control_dependency_1*
_output_shapes
:*
T0
r
!gradients_1/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      
�
#gradients_1/Reshape_11_grad/ReshapeReshape2gradients_1/SoftmaxCrossEntropyWithLogits_grad/mul!gradients_1/Reshape_11_grad/Shape*
_output_shapes

:*
Tshape0*
T0
_
gradients_1/add_2_grad/ShapeConst*
_output_shapes
: *
dtype0*
valueB 
a
gradients_1/add_2_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
�
,gradients_1/add_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/add_2_grad/Shapegradients_1/add_2_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients_1/add_2_grad/SumSum/gradients_1/add_3_grad/tuple/control_dependency,gradients_1/add_2_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
gradients_1/add_2_grad/ReshapeReshapegradients_1/add_2_grad/Sumgradients_1/add_2_grad/Shape*
_output_shapes
: *
Tshape0*
T0
�
gradients_1/add_2_grad/Sum_1Sum/gradients_1/add_3_grad/tuple/control_dependency.gradients_1/add_2_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
 gradients_1/add_2_grad/Reshape_1Reshapegradients_1/add_2_grad/Sum_1gradients_1/add_2_grad/Shape_1*
_output_shapes
: *
Tshape0*
T0
s
'gradients_1/add_2_grad/tuple/group_depsNoOp^gradients_1/add_2_grad/Reshape!^gradients_1/add_2_grad/Reshape_1
�
/gradients_1/add_2_grad/tuple/control_dependencyIdentitygradients_1/add_2_grad/Reshape(^gradients_1/add_2_grad/tuple/group_deps*
_output_shapes
: *
T0*1
_class'
%#loc:@gradients_1/add_2_grad/Reshape
�
1gradients_1/add_2_grad/tuple/control_dependency_1Identity gradients_1/add_2_grad/Reshape_1(^gradients_1/add_2_grad/tuple/group_deps*
_output_shapes
: *
T0*3
_class)
'%loc:@gradients_1/add_2_grad/Reshape_1
�
gradients_1/L2Loss_2_grad/mulMulVariable_6/read1gradients_1/add_3_grad/tuple/control_dependency_1*
_output_shapes
:	�*
T0
m
gradients_1/add_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      
h
gradients_1/add_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:
�
,gradients_1/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/add_1_grad/Shapegradients_1/add_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients_1/add_1_grad/SumSum#gradients_1/Reshape_11_grad/Reshape,gradients_1/add_1_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
gradients_1/add_1_grad/ReshapeReshapegradients_1/add_1_grad/Sumgradients_1/add_1_grad/Shape*
_output_shapes

:*
Tshape0*
T0
�
gradients_1/add_1_grad/Sum_1Sum#gradients_1/Reshape_11_grad/Reshape.gradients_1/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
 gradients_1/add_1_grad/Reshape_1Reshapegradients_1/add_1_grad/Sum_1gradients_1/add_1_grad/Shape_1*
_output_shapes
:*
Tshape0*
T0
s
'gradients_1/add_1_grad/tuple/group_depsNoOp^gradients_1/add_1_grad/Reshape!^gradients_1/add_1_grad/Reshape_1
�
/gradients_1/add_1_grad/tuple/control_dependencyIdentitygradients_1/add_1_grad/Reshape(^gradients_1/add_1_grad/tuple/group_deps*
_output_shapes

:*
T0*1
_class'
%#loc:@gradients_1/add_1_grad/Reshape
�
1gradients_1/add_1_grad/tuple/control_dependency_1Identity gradients_1/add_1_grad/Reshape_1(^gradients_1/add_1_grad/tuple/group_deps*
_output_shapes
:*
T0*3
_class)
'%loc:@gradients_1/add_1_grad/Reshape_1
�
gradients_1/L2Loss_grad/mulMulVariable_4/read/gradients_1/add_2_grad/tuple/control_dependency* 
_output_shapes
:
��*
T0
�
gradients_1/L2Loss_1_grad/mulMulVariable_5/read1gradients_1/add_2_grad/tuple/control_dependency_1*
_output_shapes	
:�*
T0
�
 gradients_1/MatMul_1_grad/MatMulMatMul/gradients_1/add_1_grad/tuple/control_dependencyVariable_6/read*
_output_shapes
:	�*
transpose_b(*
transpose_a( *
T0
�
"gradients_1/MatMul_1_grad/MatMul_1MatMulRelu_2/gradients_1/add_1_grad/tuple/control_dependency*
_output_shapes
:	�*
transpose_b( *
transpose_a(*
T0
z
*gradients_1/MatMul_1_grad/tuple/group_depsNoOp!^gradients_1/MatMul_1_grad/MatMul#^gradients_1/MatMul_1_grad/MatMul_1
�
2gradients_1/MatMul_1_grad/tuple/control_dependencyIdentity gradients_1/MatMul_1_grad/MatMul+^gradients_1/MatMul_1_grad/tuple/group_deps*
_output_shapes
:	�*
T0*3
_class)
'%loc:@gradients_1/MatMul_1_grad/MatMul
�
4gradients_1/MatMul_1_grad/tuple/control_dependency_1Identity"gradients_1/MatMul_1_grad/MatMul_1+^gradients_1/MatMul_1_grad/tuple/group_deps*
_output_shapes
:	�*
T0*5
_class+
)'loc:@gradients_1/MatMul_1_grad/MatMul_1
�
gradients_1/AddNAddNgradients_1/L2Loss_3_grad/mul1gradients_1/add_1_grad/tuple/control_dependency_1*
_output_shapes
:*
N*
T0*0
_class&
$"loc:@gradients_1/L2Loss_3_grad/mul
�
 gradients_1/Relu_2_grad/ReluGradReluGrad2gradients_1/MatMul_1_grad/tuple/control_dependencyRelu_2*
_output_shapes
:	�*
T0
�
gradients_1/AddN_1AddNgradients_1/L2Loss_2_grad/mul4gradients_1/MatMul_1_grad/tuple/control_dependency_1*
_output_shapes
:	�*
N*
T0*0
_class&
$"loc:@gradients_1/L2Loss_2_grad/mul
k
gradients_1/add_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      
g
gradients_1/add_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:�
�
*gradients_1/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/add_grad/Shapegradients_1/add_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients_1/add_grad/SumSum gradients_1/Relu_2_grad/ReluGrad*gradients_1/add_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
gradients_1/add_grad/ReshapeReshapegradients_1/add_grad/Sumgradients_1/add_grad/Shape*
_output_shapes
:	�*
Tshape0*
T0
�
gradients_1/add_grad/Sum_1Sum gradients_1/Relu_2_grad/ReluGrad,gradients_1/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
gradients_1/add_grad/Reshape_1Reshapegradients_1/add_grad/Sum_1gradients_1/add_grad/Shape_1*
_output_shapes	
:�*
Tshape0*
T0
m
%gradients_1/add_grad/tuple/group_depsNoOp^gradients_1/add_grad/Reshape^gradients_1/add_grad/Reshape_1
�
-gradients_1/add_grad/tuple/control_dependencyIdentitygradients_1/add_grad/Reshape&^gradients_1/add_grad/tuple/group_deps*
_output_shapes
:	�*
T0*/
_class%
#!loc:@gradients_1/add_grad/Reshape
�
/gradients_1/add_grad/tuple/control_dependency_1Identitygradients_1/add_grad/Reshape_1&^gradients_1/add_grad/tuple/group_deps*
_output_shapes	
:�*
T0*1
_class'
%#loc:@gradients_1/add_grad/Reshape_1
�
gradients_1/MatMul_grad/MatMulMatMul-gradients_1/add_grad/tuple/control_dependencyVariable_4/read*
_output_shapes
:	�*
transpose_b(*
transpose_a( *
T0
�
 gradients_1/MatMul_grad/MatMul_1MatMulReshape-gradients_1/add_grad/tuple/control_dependency* 
_output_shapes
:
��*
transpose_b( *
transpose_a(*
T0
t
(gradients_1/MatMul_grad/tuple/group_depsNoOp^gradients_1/MatMul_grad/MatMul!^gradients_1/MatMul_grad/MatMul_1
�
0gradients_1/MatMul_grad/tuple/control_dependencyIdentitygradients_1/MatMul_grad/MatMul)^gradients_1/MatMul_grad/tuple/group_deps*
_output_shapes
:	�*
T0*1
_class'
%#loc:@gradients_1/MatMul_grad/MatMul
�
2gradients_1/MatMul_grad/tuple/control_dependency_1Identity gradients_1/MatMul_grad/MatMul_1)^gradients_1/MatMul_grad/tuple/group_deps* 
_output_shapes
:
��*
T0*3
_class)
'%loc:@gradients_1/MatMul_grad/MatMul_1
�
gradients_1/AddN_2AddNgradients_1/L2Loss_1_grad/mul/gradients_1/add_grad/tuple/control_dependency_1*
_output_shapes	
:�*
N*
T0*0
_class&
$"loc:@gradients_1/L2Loss_1_grad/mul
w
gradients_1/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"         @   
�
 gradients_1/Reshape_grad/ReshapeReshape0gradients_1/MatMul_grad/tuple/control_dependencygradients_1/Reshape_grad/Shape*&
_output_shapes
:@*
Tshape0*
T0
�
gradients_1/AddN_3AddNgradients_1/L2Loss_grad/mul2gradients_1/MatMul_grad/tuple/control_dependency_1* 
_output_shapes
:
��*
N*
T0*.
_class$
" loc:@gradients_1/L2Loss_grad/mul
�
&gradients_1/MaxPool_1_grad/MaxPoolGradMaxPoolGradRelu_1	MaxPool_1 gradients_1/Reshape_grad/Reshape*&
_output_shapes
:@*
ksize
*
T0*
data_formatNHWC*
paddingSAME*
strides

�
 gradients_1/Relu_1_grad/ReluGradReluGrad&gradients_1/MaxPool_1_grad/MaxPoolGradRelu_1*&
_output_shapes
:@*
T0
�
&gradients_1/BiasAdd_1_grad/BiasAddGradBiasAddGrad gradients_1/Relu_1_grad/ReluGrad*
_output_shapes
:@*
data_formatNHWC*
T0

+gradients_1/BiasAdd_1_grad/tuple/group_depsNoOp!^gradients_1/Relu_1_grad/ReluGrad'^gradients_1/BiasAdd_1_grad/BiasAddGrad
�
3gradients_1/BiasAdd_1_grad/tuple/control_dependencyIdentity gradients_1/Relu_1_grad/ReluGrad,^gradients_1/BiasAdd_1_grad/tuple/group_deps*&
_output_shapes
:@*
T0*3
_class)
'%loc:@gradients_1/Relu_1_grad/ReluGrad
�
5gradients_1/BiasAdd_1_grad/tuple/control_dependency_1Identity&gradients_1/BiasAdd_1_grad/BiasAddGrad,^gradients_1/BiasAdd_1_grad/tuple/group_deps*
_output_shapes
:@*
T0*9
_class/
-+loc:@gradients_1/BiasAdd_1_grad/BiasAddGrad
�
 gradients_1/Conv2D_1_grad/ShapeNShapeNMaxPoolVariable_2/read* 
_output_shapes
::*
N*
T0*
out_type0
�
-gradients_1/Conv2D_1_grad/Conv2DBackpropInputConv2DBackpropInput gradients_1/Conv2D_1_grad/ShapeNVariable_2/read3gradients_1/BiasAdd_1_grad/tuple/control_dependency*J
_output_shapes8
6:4������������������������������������*
T0*
use_cudnn_on_gpu(*
strides
*
paddingSAME*
data_formatNHWC
�
.gradients_1/Conv2D_1_grad/Conv2DBackpropFilterConv2DBackpropFilterMaxPool"gradients_1/Conv2D_1_grad/ShapeN:13gradients_1/BiasAdd_1_grad/tuple/control_dependency*J
_output_shapes8
6:4������������������������������������*
T0*
use_cudnn_on_gpu(*
strides
*
paddingSAME*
data_formatNHWC
�
*gradients_1/Conv2D_1_grad/tuple/group_depsNoOp.^gradients_1/Conv2D_1_grad/Conv2DBackpropInput/^gradients_1/Conv2D_1_grad/Conv2DBackpropFilter
�
2gradients_1/Conv2D_1_grad/tuple/control_dependencyIdentity-gradients_1/Conv2D_1_grad/Conv2DBackpropInput+^gradients_1/Conv2D_1_grad/tuple/group_deps*&
_output_shapes
: *
T0*@
_class6
42loc:@gradients_1/Conv2D_1_grad/Conv2DBackpropInput
�
4gradients_1/Conv2D_1_grad/tuple/control_dependency_1Identity.gradients_1/Conv2D_1_grad/Conv2DBackpropFilter+^gradients_1/Conv2D_1_grad/tuple/group_deps*&
_output_shapes
: @*
T0*A
_class7
53loc:@gradients_1/Conv2D_1_grad/Conv2DBackpropFilter
�
$gradients_1/MaxPool_grad/MaxPoolGradMaxPoolGradReluMaxPool2gradients_1/Conv2D_1_grad/tuple/control_dependency*&
_output_shapes
: *
ksize
*
T0*
data_formatNHWC*
paddingSAME*
strides

�
gradients_1/Relu_grad/ReluGradReluGrad$gradients_1/MaxPool_grad/MaxPoolGradRelu*&
_output_shapes
: *
T0
�
$gradients_1/BiasAdd_grad/BiasAddGradBiasAddGradgradients_1/Relu_grad/ReluGrad*
_output_shapes
: *
data_formatNHWC*
T0
y
)gradients_1/BiasAdd_grad/tuple/group_depsNoOp^gradients_1/Relu_grad/ReluGrad%^gradients_1/BiasAdd_grad/BiasAddGrad
�
1gradients_1/BiasAdd_grad/tuple/control_dependencyIdentitygradients_1/Relu_grad/ReluGrad*^gradients_1/BiasAdd_grad/tuple/group_deps*&
_output_shapes
: *
T0*1
_class'
%#loc:@gradients_1/Relu_grad/ReluGrad
�
3gradients_1/BiasAdd_grad/tuple/control_dependency_1Identity$gradients_1/BiasAdd_grad/BiasAddGrad*^gradients_1/BiasAdd_grad/tuple/group_deps*
_output_shapes
: *
T0*7
_class-
+)loc:@gradients_1/BiasAdd_grad/BiasAddGrad
�
gradients_1/Conv2D_grad/ShapeNShapeNPlaceholderVariable/read* 
_output_shapes
::*
N*
T0*
out_type0
�
+gradients_1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInputgradients_1/Conv2D_grad/ShapeNVariable/read1gradients_1/BiasAdd_grad/tuple/control_dependency*J
_output_shapes8
6:4������������������������������������*
T0*
use_cudnn_on_gpu(*
strides
*
paddingSAME*
data_formatNHWC
�
,gradients_1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterPlaceholder gradients_1/Conv2D_grad/ShapeN:11gradients_1/BiasAdd_grad/tuple/control_dependency*J
_output_shapes8
6:4������������������������������������*
T0*
use_cudnn_on_gpu(*
strides
*
paddingSAME*
data_formatNHWC
�
(gradients_1/Conv2D_grad/tuple/group_depsNoOp,^gradients_1/Conv2D_grad/Conv2DBackpropInput-^gradients_1/Conv2D_grad/Conv2DBackpropFilter
�
0gradients_1/Conv2D_grad/tuple/control_dependencyIdentity+gradients_1/Conv2D_grad/Conv2DBackpropInput)^gradients_1/Conv2D_grad/tuple/group_deps*&
_output_shapes
:*
T0*>
_class4
20loc:@gradients_1/Conv2D_grad/Conv2DBackpropInput
�
2gradients_1/Conv2D_grad/tuple/control_dependency_1Identity,gradients_1/Conv2D_grad/Conv2DBackpropFilter)^gradients_1/Conv2D_grad/tuple/group_deps*&
_output_shapes
: *
T0*?
_class5
31loc:@gradients_1/Conv2D_grad/Conv2DBackpropFilter
�
#Variable/Momentum/Initializer/zerosConst*&
_output_shapes
: *
dtype0*%
valueB *    *
_class
loc:@Variable
�
Variable/Momentum
VariableV2*&
_output_shapes
: *
shared_name *
	container *
shape: *
_class
loc:@Variable*
dtype0
�
Variable/Momentum/AssignAssignVariable/Momentum#Variable/Momentum/Initializer/zeros*&
_output_shapes
: *
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable
�
Variable/Momentum/readIdentityVariable/Momentum*&
_output_shapes
: *
T0*
_class
loc:@Variable
�
%Variable_1/Momentum/Initializer/zerosConst*
_output_shapes
: *
dtype0*
valueB *    *
_class
loc:@Variable_1
�
Variable_1/Momentum
VariableV2*
_output_shapes
: *
shared_name *
	container *
shape: *
_class
loc:@Variable_1*
dtype0
�
Variable_1/Momentum/AssignAssignVariable_1/Momentum%Variable_1/Momentum/Initializer/zeros*
_output_shapes
: *
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_1
}
Variable_1/Momentum/readIdentityVariable_1/Momentum*
_output_shapes
: *
T0*
_class
loc:@Variable_1
�
%Variable_2/Momentum/Initializer/zerosConst*&
_output_shapes
: @*
dtype0*%
valueB @*    *
_class
loc:@Variable_2
�
Variable_2/Momentum
VariableV2*&
_output_shapes
: @*
shared_name *
	container *
shape: @*
_class
loc:@Variable_2*
dtype0
�
Variable_2/Momentum/AssignAssignVariable_2/Momentum%Variable_2/Momentum/Initializer/zeros*&
_output_shapes
: @*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_2
�
Variable_2/Momentum/readIdentityVariable_2/Momentum*&
_output_shapes
: @*
T0*
_class
loc:@Variable_2
�
%Variable_3/Momentum/Initializer/zerosConst*
_output_shapes
:@*
dtype0*
valueB@*    *
_class
loc:@Variable_3
�
Variable_3/Momentum
VariableV2*
_output_shapes
:@*
shared_name *
	container *
shape:@*
_class
loc:@Variable_3*
dtype0
�
Variable_3/Momentum/AssignAssignVariable_3/Momentum%Variable_3/Momentum/Initializer/zeros*
_output_shapes
:@*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_3
}
Variable_3/Momentum/readIdentityVariable_3/Momentum*
_output_shapes
:@*
T0*
_class
loc:@Variable_3
�
%Variable_4/Momentum/Initializer/zerosConst* 
_output_shapes
:
��*
dtype0*
valueB
��*    *
_class
loc:@Variable_4
�
Variable_4/Momentum
VariableV2* 
_output_shapes
:
��*
shared_name *
	container *
shape:
��*
_class
loc:@Variable_4*
dtype0
�
Variable_4/Momentum/AssignAssignVariable_4/Momentum%Variable_4/Momentum/Initializer/zeros* 
_output_shapes
:
��*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_4
�
Variable_4/Momentum/readIdentityVariable_4/Momentum* 
_output_shapes
:
��*
T0*
_class
loc:@Variable_4
�
%Variable_5/Momentum/Initializer/zerosConst*
_output_shapes	
:�*
dtype0*
valueB�*    *
_class
loc:@Variable_5
�
Variable_5/Momentum
VariableV2*
_output_shapes	
:�*
shared_name *
	container *
shape:�*
_class
loc:@Variable_5*
dtype0
�
Variable_5/Momentum/AssignAssignVariable_5/Momentum%Variable_5/Momentum/Initializer/zeros*
_output_shapes	
:�*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_5
~
Variable_5/Momentum/readIdentityVariable_5/Momentum*
_output_shapes	
:�*
T0*
_class
loc:@Variable_5
�
%Variable_6/Momentum/Initializer/zerosConst*
_output_shapes
:	�*
dtype0*
valueB	�*    *
_class
loc:@Variable_6
�
Variable_6/Momentum
VariableV2*
_output_shapes
:	�*
shared_name *
	container *
shape:	�*
_class
loc:@Variable_6*
dtype0
�
Variable_6/Momentum/AssignAssignVariable_6/Momentum%Variable_6/Momentum/Initializer/zeros*
_output_shapes
:	�*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_6
�
Variable_6/Momentum/readIdentityVariable_6/Momentum*
_output_shapes
:	�*
T0*
_class
loc:@Variable_6
�
%Variable_7/Momentum/Initializer/zerosConst*
_output_shapes
:*
dtype0*
valueB*    *
_class
loc:@Variable_7
�
Variable_7/Momentum
VariableV2*
_output_shapes
:*
shared_name *
	container *
shape:*
_class
loc:@Variable_7*
dtype0
�
Variable_7/Momentum/AssignAssignVariable_7/Momentum%Variable_7/Momentum/Initializer/zeros*
_output_shapes
:*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_7
}
Variable_7/Momentum/readIdentityVariable_7/Momentum*
_output_shapes
:*
T0*
_class
loc:@Variable_7
V
Momentum/momentumConst*
_output_shapes
: *
dtype0*
valueB
 *    
�
&Momentum/update_Variable/ApplyMomentumApplyMomentumVariableVariable/MomentumExponentialDecay2gradients_1/Conv2D_grad/tuple/control_dependency_1Momentum/momentum*&
_output_shapes
: *
use_nesterov( *
use_locking( *
T0*
_class
loc:@Variable
�
(Momentum/update_Variable_1/ApplyMomentumApplyMomentum
Variable_1Variable_1/MomentumExponentialDecay3gradients_1/BiasAdd_grad/tuple/control_dependency_1Momentum/momentum*
_output_shapes
: *
use_nesterov( *
use_locking( *
T0*
_class
loc:@Variable_1
�
(Momentum/update_Variable_2/ApplyMomentumApplyMomentum
Variable_2Variable_2/MomentumExponentialDecay4gradients_1/Conv2D_1_grad/tuple/control_dependency_1Momentum/momentum*&
_output_shapes
: @*
use_nesterov( *
use_locking( *
T0*
_class
loc:@Variable_2
�
(Momentum/update_Variable_3/ApplyMomentumApplyMomentum
Variable_3Variable_3/MomentumExponentialDecay5gradients_1/BiasAdd_1_grad/tuple/control_dependency_1Momentum/momentum*
_output_shapes
:@*
use_nesterov( *
use_locking( *
T0*
_class
loc:@Variable_3
�
(Momentum/update_Variable_4/ApplyMomentumApplyMomentum
Variable_4Variable_4/MomentumExponentialDecaygradients_1/AddN_3Momentum/momentum* 
_output_shapes
:
��*
use_nesterov( *
use_locking( *
T0*
_class
loc:@Variable_4
�
(Momentum/update_Variable_5/ApplyMomentumApplyMomentum
Variable_5Variable_5/MomentumExponentialDecaygradients_1/AddN_2Momentum/momentum*
_output_shapes	
:�*
use_nesterov( *
use_locking( *
T0*
_class
loc:@Variable_5
�
(Momentum/update_Variable_6/ApplyMomentumApplyMomentum
Variable_6Variable_6/MomentumExponentialDecaygradients_1/AddN_1Momentum/momentum*
_output_shapes
:	�*
use_nesterov( *
use_locking( *
T0*
_class
loc:@Variable_6
�
(Momentum/update_Variable_7/ApplyMomentumApplyMomentum
Variable_7Variable_7/MomentumExponentialDecaygradients_1/AddNMomentum/momentum*
_output_shapes
:*
use_nesterov( *
use_locking( *
T0*
_class
loc:@Variable_7
�
Momentum/updateNoOp'^Momentum/update_Variable/ApplyMomentum)^Momentum/update_Variable_1/ApplyMomentum)^Momentum/update_Variable_2/ApplyMomentum)^Momentum/update_Variable_3/ApplyMomentum)^Momentum/update_Variable_4/ApplyMomentum)^Momentum/update_Variable_5/ApplyMomentum)^Momentum/update_Variable_6/ApplyMomentum)^Momentum/update_Variable_7/ApplyMomentum
�
Momentum/valueConst^Momentum/update*
_output_shapes
: *
dtype0*
value	B :*
_class
loc:@Variable_8
�
Momentum	AssignAdd
Variable_8Momentum/value*
_output_shapes
: *
use_locking( *
T0*
_class
loc:@Variable_8
B
SoftmaxSoftmaxadd_1*
_output_shapes

:*
T0
�
Conv2D_2Conv2DPlaceholder_2Variable/read*'
_output_shapes
:�N *
T0*
use_cudnn_on_gpu(*
strides
*
paddingSAME*
data_formatNHWC
x
	BiasAdd_2BiasAddConv2D_2Variable_1/read*'
_output_shapes
:�N *
data_formatNHWC*
T0
K
Relu_3Relu	BiasAdd_2*'
_output_shapes
:�N *
T0
�
	MaxPool_2MaxPoolRelu_3*'
_output_shapes
:�N *
ksize
*
T0*
data_formatNHWC*
paddingSAME*
strides

�
Conv2D_3Conv2D	MaxPool_2Variable_2/read*'
_output_shapes
:�N@*
T0*
use_cudnn_on_gpu(*
strides
*
paddingSAME*
data_formatNHWC
x
	BiasAdd_3BiasAddConv2D_3Variable_3/read*'
_output_shapes
:�N@*
data_formatNHWC*
T0
K
Relu_4Relu	BiasAdd_3*'
_output_shapes
:�N@*
T0
�
	MaxPool_3MaxPoolRelu_4*'
_output_shapes
:�N@*
ksize
*
T0*
data_formatNHWC*
paddingSAME*
strides

a
Reshape_14/shapeConst*
_output_shapes
:*
dtype0*
valueB"'     
k

Reshape_14Reshape	MaxPool_3Reshape_14/shape* 
_output_shapes
:
�N�*
Tshape0*
T0
�
MatMul_2MatMul
Reshape_14Variable_4/read* 
_output_shapes
:
�N�*
transpose_b( *
transpose_a( *
T0
R
add_6AddMatMul_2Variable_5/read* 
_output_shapes
:
�N�*
T0
@
Relu_5Reluadd_6* 
_output_shapes
:
�N�*
T0
{
MatMul_3MatMulRelu_5Variable_6/read*
_output_shapes
:	�N*
transpose_b( *
transpose_a( *
T0
Q
add_7AddMatMul_3Variable_7/read*
_output_shapes
:	�N*
T0
E
	Softmax_1Softmaxadd_7*
_output_shapes
:	�N*
T0
P

save/ConstConst*
_output_shapes
: *
dtype0*
valueB Bmodel
�
save/SaveV2/tensor_namesConst*
_output_shapes
:*
dtype0*�
value�B�BVariableBVariable/MomentumB
Variable_1BVariable_1/MomentumB
Variable_2BVariable_2/MomentumB
Variable_3BVariable_3/MomentumB
Variable_4BVariable_4/MomentumB
Variable_5BVariable_5/MomentumB
Variable_6BVariable_6/MomentumB
Variable_7BVariable_7/MomentumB
Variable_8
�
save/SaveV2/shape_and_slicesConst*
_output_shapes
:*
dtype0*5
value,B*B B B B B B B B B B B B B B B B B 
�
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesVariableVariable/Momentum
Variable_1Variable_1/Momentum
Variable_2Variable_2/Momentum
Variable_3Variable_3/Momentum
Variable_4Variable_4/Momentum
Variable_5Variable_5/Momentum
Variable_6Variable_6/Momentum
Variable_7Variable_7/Momentum
Variable_8*
dtypes
2
}
save/control_dependencyIdentity
save/Const^save/SaveV2*
_output_shapes
: *
T0*
_class
loc:@save/Const
l
save/RestoreV2/tensor_namesConst*
_output_shapes
:*
dtype0*
valueBBVariable
h
save/RestoreV2/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 
�
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/AssignAssignVariablesave/RestoreV2*&
_output_shapes
: *
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable
w
save/RestoreV2_1/tensor_namesConst*
_output_shapes
:*
dtype0*&
valueBBVariable/Momentum
j
!save/RestoreV2_1/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 
�
save/RestoreV2_1	RestoreV2
save/Constsave/RestoreV2_1/tensor_names!save/RestoreV2_1/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_1AssignVariable/Momentumsave/RestoreV2_1*&
_output_shapes
: *
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable
p
save/RestoreV2_2/tensor_namesConst*
_output_shapes
:*
dtype0*
valueBB
Variable_1
j
!save/RestoreV2_2/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 
�
save/RestoreV2_2	RestoreV2
save/Constsave/RestoreV2_2/tensor_names!save/RestoreV2_2/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_2Assign
Variable_1save/RestoreV2_2*
_output_shapes
: *
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_1
y
save/RestoreV2_3/tensor_namesConst*
_output_shapes
:*
dtype0*(
valueBBVariable_1/Momentum
j
!save/RestoreV2_3/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 
�
save/RestoreV2_3	RestoreV2
save/Constsave/RestoreV2_3/tensor_names!save/RestoreV2_3/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_3AssignVariable_1/Momentumsave/RestoreV2_3*
_output_shapes
: *
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_1
p
save/RestoreV2_4/tensor_namesConst*
_output_shapes
:*
dtype0*
valueBB
Variable_2
j
!save/RestoreV2_4/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 
�
save/RestoreV2_4	RestoreV2
save/Constsave/RestoreV2_4/tensor_names!save/RestoreV2_4/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_4Assign
Variable_2save/RestoreV2_4*&
_output_shapes
: @*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_2
y
save/RestoreV2_5/tensor_namesConst*
_output_shapes
:*
dtype0*(
valueBBVariable_2/Momentum
j
!save/RestoreV2_5/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 
�
save/RestoreV2_5	RestoreV2
save/Constsave/RestoreV2_5/tensor_names!save/RestoreV2_5/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_5AssignVariable_2/Momentumsave/RestoreV2_5*&
_output_shapes
: @*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_2
p
save/RestoreV2_6/tensor_namesConst*
_output_shapes
:*
dtype0*
valueBB
Variable_3
j
!save/RestoreV2_6/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 
�
save/RestoreV2_6	RestoreV2
save/Constsave/RestoreV2_6/tensor_names!save/RestoreV2_6/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_6Assign
Variable_3save/RestoreV2_6*
_output_shapes
:@*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_3
y
save/RestoreV2_7/tensor_namesConst*
_output_shapes
:*
dtype0*(
valueBBVariable_3/Momentum
j
!save/RestoreV2_7/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 
�
save/RestoreV2_7	RestoreV2
save/Constsave/RestoreV2_7/tensor_names!save/RestoreV2_7/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_7AssignVariable_3/Momentumsave/RestoreV2_7*
_output_shapes
:@*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_3
p
save/RestoreV2_8/tensor_namesConst*
_output_shapes
:*
dtype0*
valueBB
Variable_4
j
!save/RestoreV2_8/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 
�
save/RestoreV2_8	RestoreV2
save/Constsave/RestoreV2_8/tensor_names!save/RestoreV2_8/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_8Assign
Variable_4save/RestoreV2_8* 
_output_shapes
:
��*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_4
y
save/RestoreV2_9/tensor_namesConst*
_output_shapes
:*
dtype0*(
valueBBVariable_4/Momentum
j
!save/RestoreV2_9/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 
�
save/RestoreV2_9	RestoreV2
save/Constsave/RestoreV2_9/tensor_names!save/RestoreV2_9/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_9AssignVariable_4/Momentumsave/RestoreV2_9* 
_output_shapes
:
��*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_4
q
save/RestoreV2_10/tensor_namesConst*
_output_shapes
:*
dtype0*
valueBB
Variable_5
k
"save/RestoreV2_10/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 
�
save/RestoreV2_10	RestoreV2
save/Constsave/RestoreV2_10/tensor_names"save/RestoreV2_10/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_10Assign
Variable_5save/RestoreV2_10*
_output_shapes	
:�*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_5
z
save/RestoreV2_11/tensor_namesConst*
_output_shapes
:*
dtype0*(
valueBBVariable_5/Momentum
k
"save/RestoreV2_11/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 
�
save/RestoreV2_11	RestoreV2
save/Constsave/RestoreV2_11/tensor_names"save/RestoreV2_11/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_11AssignVariable_5/Momentumsave/RestoreV2_11*
_output_shapes	
:�*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_5
q
save/RestoreV2_12/tensor_namesConst*
_output_shapes
:*
dtype0*
valueBB
Variable_6
k
"save/RestoreV2_12/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 
�
save/RestoreV2_12	RestoreV2
save/Constsave/RestoreV2_12/tensor_names"save/RestoreV2_12/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_12Assign
Variable_6save/RestoreV2_12*
_output_shapes
:	�*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_6
z
save/RestoreV2_13/tensor_namesConst*
_output_shapes
:*
dtype0*(
valueBBVariable_6/Momentum
k
"save/RestoreV2_13/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 
�
save/RestoreV2_13	RestoreV2
save/Constsave/RestoreV2_13/tensor_names"save/RestoreV2_13/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_13AssignVariable_6/Momentumsave/RestoreV2_13*
_output_shapes
:	�*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_6
q
save/RestoreV2_14/tensor_namesConst*
_output_shapes
:*
dtype0*
valueBB
Variable_7
k
"save/RestoreV2_14/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 
�
save/RestoreV2_14	RestoreV2
save/Constsave/RestoreV2_14/tensor_names"save/RestoreV2_14/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_14Assign
Variable_7save/RestoreV2_14*
_output_shapes
:*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_7
z
save/RestoreV2_15/tensor_namesConst*
_output_shapes
:*
dtype0*(
valueBBVariable_7/Momentum
k
"save/RestoreV2_15/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 
�
save/RestoreV2_15	RestoreV2
save/Constsave/RestoreV2_15/tensor_names"save/RestoreV2_15/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_15AssignVariable_7/Momentumsave/RestoreV2_15*
_output_shapes
:*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_7
q
save/RestoreV2_16/tensor_namesConst*
_output_shapes
:*
dtype0*
valueBB
Variable_8
k
"save/RestoreV2_16/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 
�
save/RestoreV2_16	RestoreV2
save/Constsave/RestoreV2_16/tensor_names"save/RestoreV2_16/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_16Assign
Variable_8save/RestoreV2_16*
_output_shapes
: *
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_8
�
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16
�
initNoOp^Variable/Assign^Variable_1/Assign^Variable_2/Assign^Variable_3/Assign^Variable_4/Assign^Variable_5/Assign^Variable_6/Assign^Variable_7/Assign^Variable_8/Assign^Variable/Momentum/Assign^Variable_1/Momentum/Assign^Variable_2/Momentum/Assign^Variable_3/Momentum/Assign^Variable_4/Momentum/Assign^Variable_5/Momentum/Assign^Variable_6/Momentum/Assign^Variable_7/Momentum/Assign
�
Merge/MergeSummaryMergeSummarysummary_data_0summary_conv_0summary_pool_0summary_conv2_0summary_pool2_0lossconv1_weightsconv1_biasesconv2_weightsconv2_biasesfc1_weights
fc1_biasesfc2_weights
fc2_biaseslearning_rate*
_output_shapes
: *
N"��m��s     W*�f	�����AJ��
�+�*
9
Add
x"T
y"T
z"T"
Ttype:
2	
T
AddN
inputs"T*N
sum"T"
Nint(0"
Ttype:
2	��
�
ApplyMomentum
var"T�
accum"T�
lr"T	
grad"T
momentum"T
out"T�"
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 
x
Assign
ref"T�

value"T

output_ref"T�"	
Ttype"
validate_shapebool("
use_lockingbool(�
p
	AssignAdd
ref"T�

value"T

output_ref"T�"
Ttype:
2	"
use_lockingbool( 
{
BiasAdd

value"T	
bias"T
output"T"
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
{
BiasAddGrad
out_backprop"T
output"T"
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
8
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
�
Conv2D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW
�
Conv2DBackpropFilter

input"T
filter_sizes
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW
�
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
4
Fill
dims

value"T
output"T"	
Ttype
+
Floor
x"T
y"T"
Ttype:
2
>
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
�
ImageSummary
tag
tensor"T
summary"

max_imagesint(0"
Ttype0:
2"'
	bad_colortensorB:�  �
1
L2Loss
t"T
output"T"
Ttype:
2
o
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2
�
Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
�
MaxPool

input"T
output"T"
Ttype0:
2
	"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
�
MaxPoolGrad

orig_input"T
orig_output"T	
grad"T
output"T"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW"
Ttype0:
2		
:
Maximum
x"T
y"T
z"T"
Ttype:	
2	�
�
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
8
MergeSummary
inputs*N
summary"
Nint(0
�
Min

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
<
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
5
Pow
x"T
y"T
z"T"
Ttype:
	2	
�
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
=
RealDiv
x"T
y"T
z"T"
Ttype:
2	
A
Relu
features"T
activations"T"
Ttype:
2		
S
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2		
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
M
ScalarSummary
tags
values"T
summary"
Ttype:
2		
e
ShapeN
input"T*N
output"out_type*N"
Nint(0"	
Ttype"
out_typetype0:
2	
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
8
Softmax
logits"T
softmax"T"
Ttype:
2
i
SoftmaxCrossEntropyWithLogits
features"T
labels"T	
loss"T
backprop"T"
Ttype:
2
,
Sqrt
x"T
y"T"
Ttype:	
2
9
Sub
x"T
y"T
z"T"
Ttype:
2	
�
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	

TruncatedNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	�
s

VariableV2
ref"dtype�"
shapeshape"
dtypetype"
	containerstring "
shared_namestring �
&
	ZerosLike
x"T
y"T"	
Ttype*1.4.02v1.4.0-rc1-11-g130a514��
l
PlaceholderPlaceholder*&
_output_shapes
:*
dtype0*
shape:
^
Placeholder_1Placeholder*
_output_shapes

:*
dtype0*
shape
:
p
Placeholder_2Placeholder*'
_output_shapes
:�N*
dtype0*
shape:�N
o
truncated_normal/shapeConst*
_output_shapes
:*
dtype0*%
valueB"             
Z
truncated_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    
\
truncated_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *���=
�
 truncated_normal/TruncatedNormalTruncatedNormaltruncated_normal/shape*&
_output_shapes
: *
dtype0*
seed2��*
T0*
seed���)
�
truncated_normal/mulMul truncated_normal/TruncatedNormaltruncated_normal/stddev*&
_output_shapes
: *
T0
u
truncated_normalAddtruncated_normal/multruncated_normal/mean*&
_output_shapes
: *
T0
�
Variable
VariableV2*&
_output_shapes
: *
dtype0*
shared_name *
	container *
shape: 
�
Variable/AssignAssignVariabletruncated_normal*&
_output_shapes
: *
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable
q
Variable/readIdentityVariable*&
_output_shapes
: *
T0*
_class
loc:@Variable
R
zerosConst*
_output_shapes
: *
dtype0*
valueB *    
v

Variable_1
VariableV2*
_output_shapes
: *
dtype0*
shared_name *
	container *
shape: 
�
Variable_1/AssignAssign
Variable_1zeros*
_output_shapes
: *
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_1
k
Variable_1/readIdentity
Variable_1*
_output_shapes
: *
T0*
_class
loc:@Variable_1
q
truncated_normal_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"          @   
\
truncated_normal_1/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    
^
truncated_normal_1/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *���=
�
"truncated_normal_1/TruncatedNormalTruncatedNormaltruncated_normal_1/shape*&
_output_shapes
: @*
dtype0*
seed2��*
T0*
seed���)
�
truncated_normal_1/mulMul"truncated_normal_1/TruncatedNormaltruncated_normal_1/stddev*&
_output_shapes
: @*
T0
{
truncated_normal_1Addtruncated_normal_1/multruncated_normal_1/mean*&
_output_shapes
: @*
T0
�

Variable_2
VariableV2*&
_output_shapes
: @*
dtype0*
shared_name *
	container *
shape: @
�
Variable_2/AssignAssign
Variable_2truncated_normal_1*&
_output_shapes
: @*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_2
w
Variable_2/readIdentity
Variable_2*&
_output_shapes
: @*
T0*
_class
loc:@Variable_2
R
ConstConst*
_output_shapes
:@*
dtype0*
valueB@*���=
v

Variable_3
VariableV2*
_output_shapes
:@*
dtype0*
shared_name *
	container *
shape:@
�
Variable_3/AssignAssign
Variable_3Const*
_output_shapes
:@*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_3
k
Variable_3/readIdentity
Variable_3*
_output_shapes
:@*
T0*
_class
loc:@Variable_3
i
truncated_normal_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"      
\
truncated_normal_2/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    
^
truncated_normal_2/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *���=
�
"truncated_normal_2/TruncatedNormalTruncatedNormaltruncated_normal_2/shape* 
_output_shapes
:
��*
dtype0*
seed2��*
T0*
seed���)
�
truncated_normal_2/mulMul"truncated_normal_2/TruncatedNormaltruncated_normal_2/stddev* 
_output_shapes
:
��*
T0
u
truncated_normal_2Addtruncated_normal_2/multruncated_normal_2/mean* 
_output_shapes
:
��*
T0
�

Variable_4
VariableV2* 
_output_shapes
:
��*
dtype0*
shared_name *
	container *
shape:
��
�
Variable_4/AssignAssign
Variable_4truncated_normal_2* 
_output_shapes
:
��*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_4
q
Variable_4/readIdentity
Variable_4* 
_output_shapes
:
��*
T0*
_class
loc:@Variable_4
V
Const_1Const*
_output_shapes	
:�*
dtype0*
valueB�*���=
x

Variable_5
VariableV2*
_output_shapes	
:�*
dtype0*
shared_name *
	container *
shape:�
�
Variable_5/AssignAssign
Variable_5Const_1*
_output_shapes	
:�*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_5
l
Variable_5/readIdentity
Variable_5*
_output_shapes	
:�*
T0*
_class
loc:@Variable_5
i
truncated_normal_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"      
\
truncated_normal_3/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    
^
truncated_normal_3/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *���=
�
"truncated_normal_3/TruncatedNormalTruncatedNormaltruncated_normal_3/shape*
_output_shapes
:	�*
dtype0*
seed2��*
T0*
seed���)
�
truncated_normal_3/mulMul"truncated_normal_3/TruncatedNormaltruncated_normal_3/stddev*
_output_shapes
:	�*
T0
t
truncated_normal_3Addtruncated_normal_3/multruncated_normal_3/mean*
_output_shapes
:	�*
T0
�

Variable_6
VariableV2*
_output_shapes
:	�*
dtype0*
shared_name *
	container *
shape:	�
�
Variable_6/AssignAssign
Variable_6truncated_normal_3*
_output_shapes
:	�*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_6
p
Variable_6/readIdentity
Variable_6*
_output_shapes
:	�*
T0*
_class
loc:@Variable_6
T
Const_2Const*
_output_shapes
:*
dtype0*
valueB*���=
v

Variable_7
VariableV2*
_output_shapes
:*
dtype0*
shared_name *
	container *
shape:
�
Variable_7/AssignAssign
Variable_7Const_2*
_output_shapes
:*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_7
k
Variable_7/readIdentity
Variable_7*
_output_shapes
:*
T0*
_class
loc:@Variable_7
�
Conv2DConv2DPlaceholderVariable/read*&
_output_shapes
: *
T0*
use_cudnn_on_gpu(*
strides
*
paddingSAME*
data_formatNHWC
s
BiasAddBiasAddConv2DVariable_1/read*&
_output_shapes
: *
data_formatNHWC*
T0
F
ReluReluBiasAdd*&
_output_shapes
: *
T0
�
MaxPoolMaxPoolRelu*&
_output_shapes
: *
ksize
*
T0*
data_formatNHWC*
paddingSAME*
strides

�
Conv2D_1Conv2DMaxPoolVariable_2/read*&
_output_shapes
:@*
T0*
use_cudnn_on_gpu(*
strides
*
paddingSAME*
data_formatNHWC
w
	BiasAdd_1BiasAddConv2D_1Variable_3/read*&
_output_shapes
:@*
data_formatNHWC*
T0
J
Relu_1Relu	BiasAdd_1*&
_output_shapes
:@*
T0
�
	MaxPool_1MaxPoolRelu_1*&
_output_shapes
:@*
ksize
*
T0*
data_formatNHWC*
paddingSAME*
strides

^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      
d
ReshapeReshape	MaxPool_1Reshape/shape*
_output_shapes
:	�*
Tshape0*
T0
z
MatMulMatMulReshapeVariable_4/read*
_output_shapes
:	�*
transpose_b( *
T0*
transpose_a( 
M
addAddMatMulVariable_5/read*
_output_shapes
:	�*
T0
=
Relu_2Reluadd*
_output_shapes
:	�*
T0
z
MatMul_1MatMulRelu_2Variable_6/read*
_output_shapes

:*
transpose_b( *
T0*
transpose_a( 
P
add_1AddMatMul_1Variable_7/read*
_output_shapes

:*
T0
d
Slice/beginConst*
_output_shapes
:*
dtype0*%
valueB"                
c

Slice/sizeConst*
_output_shapes
:*
dtype0*%
valueB"   ��������   
r
SliceSlicePlaceholderSlice/begin
Slice/size*&
_output_shapes
:*
Index0*
T0
`
Const_3Const*
_output_shapes
:*
dtype0*%
valueB"             
X
MinMinSliceConst_3*
_output_shapes
: *

Tidx0*
T0*
	keep_dims( 
G
subSubSliceMin*&
_output_shapes
:*
T0
`
Const_4Const*
_output_shapes
:*
dtype0*%
valueB"             
V
MaxMaxsubConst_4*
_output_shapes
: *

Tidx0*
T0*
	keep_dims( 
J
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  C
7
mulMulMaxmul/y*
_output_shapes
: *
T0
M
truedivRealDivsubmul*&
_output_shapes
:*
T0
d
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         
i
	Reshape_1ReshapetruedivReshape_1/shape*"
_output_shapes
:*
Tshape0*
T0
c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
k
	transpose	Transpose	Reshape_1transpose/perm*"
_output_shapes
:*
Tperm0*
T0
h
Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         
o
	Reshape_2Reshape	transposeReshape_2/shape*&
_output_shapes
:*
Tshape0*
T0
a
summary_data_0/tagConst*
_output_shapes
: *
dtype0*
valueB Bsummary_data_0
�
summary_data_0ImageSummarysummary_data_0/tag	Reshape_2*

max_images*
_output_shapes
: *
T0*
	bad_colorB:�  �
f
Slice_1/beginConst*
_output_shapes
:*
dtype0*%
valueB"                
e
Slice_1/sizeConst*
_output_shapes
:*
dtype0*%
valueB"   ��������   
s
Slice_1SliceConv2DSlice_1/beginSlice_1/size*&
_output_shapes
:*
Index0*
T0
`
Const_5Const*
_output_shapes
:*
dtype0*%
valueB"             
\
Min_1MinSlice_1Const_5*
_output_shapes
: *

Tidx0*
T0*
	keep_dims( 
M
sub_1SubSlice_1Min_1*&
_output_shapes
:*
T0
`
Const_6Const*
_output_shapes
:*
dtype0*%
valueB"             
Z
Max_1Maxsub_1Const_6*
_output_shapes
: *

Tidx0*
T0*
	keep_dims( 
L
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  C
=
mul_1MulMax_1mul_1/y*
_output_shapes
: *
T0
S
	truediv_1RealDivsub_1mul_1*&
_output_shapes
:*
T0
d
Reshape_3/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         
k
	Reshape_3Reshape	truediv_1Reshape_3/shape*"
_output_shapes
:*
Tshape0*
T0
e
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
o
transpose_1	Transpose	Reshape_3transpose_1/perm*"
_output_shapes
:*
Tperm0*
T0
h
Reshape_4/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         
q
	Reshape_4Reshapetranspose_1Reshape_4/shape*&
_output_shapes
:*
Tshape0*
T0
a
summary_conv_0/tagConst*
_output_shapes
: *
dtype0*
valueB Bsummary_conv_0
�
summary_conv_0ImageSummarysummary_conv_0/tag	Reshape_4*

max_images*
_output_shapes
: *
T0*
	bad_colorB:�  �
f
Slice_2/beginConst*
_output_shapes
:*
dtype0*%
valueB"                
e
Slice_2/sizeConst*
_output_shapes
:*
dtype0*%
valueB"   ��������   
t
Slice_2SliceMaxPoolSlice_2/beginSlice_2/size*&
_output_shapes
:*
Index0*
T0
`
Const_7Const*
_output_shapes
:*
dtype0*%
valueB"             
\
Min_2MinSlice_2Const_7*
_output_shapes
: *

Tidx0*
T0*
	keep_dims( 
M
sub_2SubSlice_2Min_2*&
_output_shapes
:*
T0
`
Const_8Const*
_output_shapes
:*
dtype0*%
valueB"             
Z
Max_2Maxsub_2Const_8*
_output_shapes
: *

Tidx0*
T0*
	keep_dims( 
L
mul_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  C
=
mul_2MulMax_2mul_2/y*
_output_shapes
: *
T0
S
	truediv_2RealDivsub_2mul_2*&
_output_shapes
:*
T0
d
Reshape_5/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         
k
	Reshape_5Reshape	truediv_2Reshape_5/shape*"
_output_shapes
:*
Tshape0*
T0
e
transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          
o
transpose_2	Transpose	Reshape_5transpose_2/perm*"
_output_shapes
:*
Tperm0*
T0
h
Reshape_6/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         
q
	Reshape_6Reshapetranspose_2Reshape_6/shape*&
_output_shapes
:*
Tshape0*
T0
a
summary_pool_0/tagConst*
_output_shapes
: *
dtype0*
valueB Bsummary_pool_0
�
summary_pool_0ImageSummarysummary_pool_0/tag	Reshape_6*

max_images*
_output_shapes
: *
T0*
	bad_colorB:�  �
f
Slice_3/beginConst*
_output_shapes
:*
dtype0*%
valueB"                
e
Slice_3/sizeConst*
_output_shapes
:*
dtype0*%
valueB"   ��������   
u
Slice_3SliceConv2D_1Slice_3/beginSlice_3/size*&
_output_shapes
:*
Index0*
T0
`
Const_9Const*
_output_shapes
:*
dtype0*%
valueB"             
\
Min_3MinSlice_3Const_9*
_output_shapes
: *

Tidx0*
T0*
	keep_dims( 
M
sub_3SubSlice_3Min_3*&
_output_shapes
:*
T0
a
Const_10Const*
_output_shapes
:*
dtype0*%
valueB"             
[
Max_3Maxsub_3Const_10*
_output_shapes
: *

Tidx0*
T0*
	keep_dims( 
L
mul_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *  C
=
mul_3MulMax_3mul_3/y*
_output_shapes
: *
T0
S
	truediv_3RealDivsub_3mul_3*&
_output_shapes
:*
T0
d
Reshape_7/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         
k
	Reshape_7Reshape	truediv_3Reshape_7/shape*"
_output_shapes
:*
Tshape0*
T0
e
transpose_3/permConst*
_output_shapes
:*
dtype0*!
valueB"          
o
transpose_3	Transpose	Reshape_7transpose_3/perm*"
_output_shapes
:*
Tperm0*
T0
h
Reshape_8/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         
q
	Reshape_8Reshapetranspose_3Reshape_8/shape*&
_output_shapes
:*
Tshape0*
T0
c
summary_conv2_0/tagConst*
_output_shapes
: *
dtype0* 
valueB Bsummary_conv2_0
�
summary_conv2_0ImageSummarysummary_conv2_0/tag	Reshape_8*

max_images*
_output_shapes
: *
T0*
	bad_colorB:�  �
f
Slice_4/beginConst*
_output_shapes
:*
dtype0*%
valueB"                
e
Slice_4/sizeConst*
_output_shapes
:*
dtype0*%
valueB"   ��������   
v
Slice_4Slice	MaxPool_1Slice_4/beginSlice_4/size*&
_output_shapes
:*
Index0*
T0
a
Const_11Const*
_output_shapes
:*
dtype0*%
valueB"             
]
Min_4MinSlice_4Const_11*
_output_shapes
: *

Tidx0*
T0*
	keep_dims( 
M
sub_4SubSlice_4Min_4*&
_output_shapes
:*
T0
a
Const_12Const*
_output_shapes
:*
dtype0*%
valueB"             
[
Max_4Maxsub_4Const_12*
_output_shapes
: *

Tidx0*
T0*
	keep_dims( 
L
mul_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *  C
=
mul_4MulMax_4mul_4/y*
_output_shapes
: *
T0
S
	truediv_4RealDivsub_4mul_4*&
_output_shapes
:*
T0
d
Reshape_9/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         
k
	Reshape_9Reshape	truediv_4Reshape_9/shape*"
_output_shapes
:*
Tshape0*
T0
e
transpose_4/permConst*
_output_shapes
:*
dtype0*!
valueB"          
o
transpose_4	Transpose	Reshape_9transpose_4/perm*"
_output_shapes
:*
Tperm0*
T0
i
Reshape_10/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         
s

Reshape_10Reshapetranspose_4Reshape_10/shape*&
_output_shapes
:*
Tshape0*
T0
c
summary_pool2_0/tagConst*
_output_shapes
: *
dtype0* 
valueB Bsummary_pool2_0
�
summary_pool2_0ImageSummarysummary_pool2_0/tag
Reshape_10*

max_images*
_output_shapes
: *
T0*
	bad_colorB:�  �
F
RankConst*
_output_shapes
: *
dtype0*
value	B :
V
ShapeConst*
_output_shapes
:*
dtype0*
valueB"      
H
Rank_1Const*
_output_shapes
: *
dtype0*
value	B :
X
Shape_1Const*
_output_shapes
:*
dtype0*
valueB"      
G
Sub/yConst*
_output_shapes
: *
dtype0*
value	B :
:
SubSubRank_1Sub/y*
_output_shapes
: *
T0
T
Slice_5/beginPackSub*

axis *
_output_shapes
:*
N*
T0
V
Slice_5/sizeConst*
_output_shapes
:*
dtype0*
valueB:
h
Slice_5SliceShape_1Slice_5/beginSlice_5/size*
_output_shapes
:*
Index0*
T0
b
concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:
���������
M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
s
concatConcatV2concat/values_0Slice_5concat/axis*
_output_shapes
:*

Tidx0*
N*
T0
[

Reshape_11Reshapeadd_1concat*
_output_shapes

:*
Tshape0*
T0
H
Rank_2Const*
_output_shapes
: *
dtype0*
value	B :
X
Shape_2Const*
_output_shapes
:*
dtype0*
valueB"      
I
Sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :
>
Sub_1SubRank_2Sub_1/y*
_output_shapes
: *
T0
V
Slice_6/beginPackSub_1*

axis *
_output_shapes
:*
N*
T0
V
Slice_6/sizeConst*
_output_shapes
:*
dtype0*
valueB:
h
Slice_6SliceShape_2Slice_6/beginSlice_6/size*
_output_shapes
:*
Index0*
T0
d
concat_1/values_0Const*
_output_shapes
:*
dtype0*
valueB:
���������
O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
y
concat_1ConcatV2concat_1/values_0Slice_6concat_1/axis*
_output_shapes
:*

Tidx0*
N*
T0
e

Reshape_12ReshapePlaceholder_1concat_1*
_output_shapes

:*
Tshape0*
T0
�
SoftmaxCrossEntropyWithLogitsSoftmaxCrossEntropyWithLogits
Reshape_11
Reshape_12*$
_output_shapes
::*
T0
I
Sub_2/yConst*
_output_shapes
: *
dtype0*
value	B :
<
Sub_2SubRankSub_2/y*
_output_shapes
: *
T0
W
Slice_7/beginConst*
_output_shapes
:*
dtype0*
valueB: 
U
Slice_7/sizePackSub_2*

axis *
_output_shapes
:*
N*
T0
o
Slice_7SliceShapeSlice_7/beginSlice_7/size*#
_output_shapes
:���������*
Index0*
T0
p

Reshape_13ReshapeSoftmaxCrossEntropyWithLogitsSlice_7*
_output_shapes
:*
Tshape0*
T0
R
Const_13Const*
_output_shapes
:*
dtype0*
valueB: 
`
MeanMean
Reshape_13Const_13*
_output_shapes
: *

Tidx0*
T0*
	keep_dims( 
N
	loss/tagsConst*
_output_shapes
: *
dtype0*
valueB
 Bloss
G
lossScalarSummary	loss/tagsMean*
_output_shapes
: *
T0
R
gradients/ShapeConst*
_output_shapes
: *
dtype0*
valueB 
T
gradients/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?
Y
gradients/FillFillgradients/Shapegradients/Const*
_output_shapes
: *
T0
k
!gradients/Mean_grad/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
�
gradients/Mean_grad/ReshapeReshapegradients/Fill!gradients/Mean_grad/Reshape/shape*
_output_shapes
:*
Tshape0*
T0
l
"gradients/Mean_grad/Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB:
�
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshape"gradients/Mean_grad/Tile/multiples*
_output_shapes
:*

Tmultiples0*
T0
c
gradients/Mean_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:
^
gradients/Mean_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
�
gradients/Mean_grad/ConstConst*
_output_shapes
:*
dtype0*
valueB: *,
_class"
 loc:@gradients/Mean_grad/Shape
�
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shapegradients/Mean_grad/Const*
_output_shapes
: *

Tidx0*
T0*
	keep_dims( *,
_class"
 loc:@gradients/Mean_grad/Shape
�
gradients/Mean_grad/Const_1Const*
_output_shapes
:*
dtype0*
valueB: *,
_class"
 loc:@gradients/Mean_grad/Shape
�
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_1gradients/Mean_grad/Const_1*
_output_shapes
: *

Tidx0*
T0*
	keep_dims( *,
_class"
 loc:@gradients/Mean_grad/Shape
�
gradients/Mean_grad/Maximum/yConst*
_output_shapes
: *
dtype0*
value	B :*,
_class"
 loc:@gradients/Mean_grad/Shape
�
gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*
_output_shapes
: *
T0*,
_class"
 loc:@gradients/Mean_grad/Shape
�
gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum*
_output_shapes
: *
T0*,
_class"
 loc:@gradients/Mean_grad/Shape
n
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv*
_output_shapes
: *

DstT0*

SrcT0

gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*
_output_shapes
:*
T0
i
gradients/Reshape_13_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:
�
!gradients/Reshape_13_grad/ReshapeReshapegradients/Mean_grad/truedivgradients/Reshape_13_grad/Shape*
_output_shapes
:*
Tshape0*
T0
k
gradients/zeros_like	ZerosLikeSoftmaxCrossEntropyWithLogits:1*
_output_shapes

:*
T0
�
;gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������
�
7gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims
ExpandDims!gradients/Reshape_13_grad/Reshape;gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim*
_output_shapes

:*
T0*

Tdim0
�
0gradients/SoftmaxCrossEntropyWithLogits_grad/mulMul7gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDimsSoftmaxCrossEntropyWithLogits:1*
_output_shapes

:*
T0
p
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      
�
!gradients/Reshape_11_grad/ReshapeReshape0gradients/SoftmaxCrossEntropyWithLogits_grad/mulgradients/Reshape_11_grad/Shape*
_output_shapes

:*
Tshape0*
T0
k
gradients/add_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      
f
gradients/add_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:
�
*gradients/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_1_grad/Shapegradients/add_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients/add_1_grad/SumSum!gradients/Reshape_11_grad/Reshape*gradients/add_1_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
T0*
	keep_dims( 
�
gradients/add_1_grad/ReshapeReshapegradients/add_1_grad/Sumgradients/add_1_grad/Shape*
_output_shapes

:*
Tshape0*
T0
�
gradients/add_1_grad/Sum_1Sum!gradients/Reshape_11_grad/Reshape,gradients/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
T0*
	keep_dims( 
�
gradients/add_1_grad/Reshape_1Reshapegradients/add_1_grad/Sum_1gradients/add_1_grad/Shape_1*
_output_shapes
:*
Tshape0*
T0
�
gradients/MatMul_1_grad/MatMulMatMulgradients/add_1_grad/ReshapeVariable_6/read*
_output_shapes
:	�*
transpose_b(*
T0*
transpose_a( 
�
 gradients/MatMul_1_grad/MatMul_1MatMulRelu_2gradients/add_1_grad/Reshape*
_output_shapes
:	�*
transpose_b( *
T0*
transpose_a(
|
gradients/Relu_2_grad/ReluGradReluGradgradients/MatMul_1_grad/MatMulRelu_2*
_output_shapes
:	�*
T0
i
gradients/add_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      
e
gradients/add_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:�
�
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients/add_grad/SumSumgradients/Relu_2_grad/ReluGrad(gradients/add_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
T0*
	keep_dims( 
�
gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*
_output_shapes
:	�*
Tshape0*
T0
�
gradients/add_grad/Sum_1Sumgradients/Relu_2_grad/ReluGrad*gradients/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
T0*
	keep_dims( 
�
gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
_output_shapes	
:�*
Tshape0*
T0
�
gradients/MatMul_grad/MatMulMatMulgradients/add_grad/ReshapeVariable_4/read*
_output_shapes
:	�*
transpose_b(*
T0*
transpose_a( 
�
gradients/MatMul_grad/MatMul_1MatMulReshapegradients/add_grad/Reshape* 
_output_shapes
:
��*
transpose_b( *
T0*
transpose_a(
u
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"         @   
�
gradients/Reshape_grad/ReshapeReshapegradients/MatMul_grad/MatMulgradients/Reshape_grad/Shape*&
_output_shapes
:@*
Tshape0*
T0
�
$gradients/MaxPool_1_grad/MaxPoolGradMaxPoolGradRelu_1	MaxPool_1gradients/Reshape_grad/Reshape*&
_output_shapes
:@*
ksize
*
T0*
data_formatNHWC*
paddingSAME*
strides

�
gradients/Relu_1_grad/ReluGradReluGrad$gradients/MaxPool_1_grad/MaxPoolGradRelu_1*&
_output_shapes
:@*
T0
�
$gradients/BiasAdd_1_grad/BiasAddGradBiasAddGradgradients/Relu_1_grad/ReluGrad*
_output_shapes
:@*
data_formatNHWC*
T0
�
gradients/Conv2D_1_grad/ShapeNShapeNMaxPoolVariable_2/read* 
_output_shapes
::*
N*
out_type0*
T0
�
+gradients/Conv2D_1_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_1_grad/ShapeNVariable_2/readgradients/Relu_1_grad/ReluGrad*&
_output_shapes
: *
T0*
use_cudnn_on_gpu(*
strides
*
paddingSAME*
data_formatNHWC
�
,gradients/Conv2D_1_grad/Conv2DBackpropFilterConv2DBackpropFilterMaxPool gradients/Conv2D_1_grad/ShapeN:1gradients/Relu_1_grad/ReluGrad*&
_output_shapes
: @*
T0*
use_cudnn_on_gpu(*
strides
*
paddingSAME*
data_formatNHWC
�
"gradients/MaxPool_grad/MaxPoolGradMaxPoolGradReluMaxPool+gradients/Conv2D_1_grad/Conv2DBackpropInput*&
_output_shapes
: *
ksize
*
T0*
data_formatNHWC*
paddingSAME*
strides

�
gradients/Relu_grad/ReluGradReluGrad"gradients/MaxPool_grad/MaxPoolGradRelu*&
_output_shapes
: *
T0
�
"gradients/BiasAdd_grad/BiasAddGradBiasAddGradgradients/Relu_grad/ReluGrad*
_output_shapes
: *
data_formatNHWC*
T0
�
gradients/Conv2D_grad/ShapeNShapeNPlaceholderVariable/read* 
_output_shapes
::*
N*
out_type0*
T0
�
)gradients/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_grad/ShapeNVariable/readgradients/Relu_grad/ReluGrad*&
_output_shapes
:*
T0*
use_cudnn_on_gpu(*
strides
*
paddingSAME*
data_formatNHWC
�
*gradients/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterPlaceholdergradients/Conv2D_grad/ShapeN:1gradients/Relu_grad/ReluGrad*&
_output_shapes
: *
T0*
use_cudnn_on_gpu(*
strides
*
paddingSAME*
data_formatNHWC
�
global_norm/L2LossL2Loss*gradients/Conv2D_grad/Conv2DBackpropFilter*
_output_shapes
: *
T0*=
_class3
1/loc:@gradients/Conv2D_grad/Conv2DBackpropFilter
g
global_norm/stackPackglobal_norm/L2Loss*

axis *
_output_shapes
:*
N*
T0
[
global_norm/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
z
global_norm/SumSumglobal_norm/stackglobal_norm/Const*
_output_shapes
: *

Tidx0*
T0*
	keep_dims( 
X
global_norm/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   @
]
global_norm/mulMulglobal_norm/Sumglobal_norm/Const_1*
_output_shapes
: *
T0
Q
global_norm/global_normSqrtglobal_norm/mul*
_output_shapes
: *
T0
`
conv1_weights/tagsConst*
_output_shapes
: *
dtype0*
valueB Bconv1_weights
l
conv1_weightsScalarSummaryconv1_weights/tagsglobal_norm/global_norm*
_output_shapes
: *
T0
�
global_norm_1/L2LossL2Loss"gradients/BiasAdd_grad/BiasAddGrad*
_output_shapes
: *
T0*5
_class+
)'loc:@gradients/BiasAdd_grad/BiasAddGrad
k
global_norm_1/stackPackglobal_norm_1/L2Loss*

axis *
_output_shapes
:*
N*
T0
]
global_norm_1/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
�
global_norm_1/SumSumglobal_norm_1/stackglobal_norm_1/Const*
_output_shapes
: *

Tidx0*
T0*
	keep_dims( 
Z
global_norm_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   @
c
global_norm_1/mulMulglobal_norm_1/Sumglobal_norm_1/Const_1*
_output_shapes
: *
T0
U
global_norm_1/global_normSqrtglobal_norm_1/mul*
_output_shapes
: *
T0
^
conv1_biases/tagsConst*
_output_shapes
: *
dtype0*
valueB Bconv1_biases
l
conv1_biasesScalarSummaryconv1_biases/tagsglobal_norm_1/global_norm*
_output_shapes
: *
T0
�
global_norm_2/L2LossL2Loss,gradients/Conv2D_1_grad/Conv2DBackpropFilter*
_output_shapes
: *
T0*?
_class5
31loc:@gradients/Conv2D_1_grad/Conv2DBackpropFilter
k
global_norm_2/stackPackglobal_norm_2/L2Loss*

axis *
_output_shapes
:*
N*
T0
]
global_norm_2/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
�
global_norm_2/SumSumglobal_norm_2/stackglobal_norm_2/Const*
_output_shapes
: *

Tidx0*
T0*
	keep_dims( 
Z
global_norm_2/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   @
c
global_norm_2/mulMulglobal_norm_2/Sumglobal_norm_2/Const_1*
_output_shapes
: *
T0
U
global_norm_2/global_normSqrtglobal_norm_2/mul*
_output_shapes
: *
T0
`
conv2_weights/tagsConst*
_output_shapes
: *
dtype0*
valueB Bconv2_weights
n
conv2_weightsScalarSummaryconv2_weights/tagsglobal_norm_2/global_norm*
_output_shapes
: *
T0
�
global_norm_3/L2LossL2Loss$gradients/BiasAdd_1_grad/BiasAddGrad*
_output_shapes
: *
T0*7
_class-
+)loc:@gradients/BiasAdd_1_grad/BiasAddGrad
k
global_norm_3/stackPackglobal_norm_3/L2Loss*

axis *
_output_shapes
:*
N*
T0
]
global_norm_3/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
�
global_norm_3/SumSumglobal_norm_3/stackglobal_norm_3/Const*
_output_shapes
: *

Tidx0*
T0*
	keep_dims( 
Z
global_norm_3/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   @
c
global_norm_3/mulMulglobal_norm_3/Sumglobal_norm_3/Const_1*
_output_shapes
: *
T0
U
global_norm_3/global_normSqrtglobal_norm_3/mul*
_output_shapes
: *
T0
^
conv2_biases/tagsConst*
_output_shapes
: *
dtype0*
valueB Bconv2_biases
l
conv2_biasesScalarSummaryconv2_biases/tagsglobal_norm_3/global_norm*
_output_shapes
: *
T0
�
global_norm_4/L2LossL2Lossgradients/MatMul_grad/MatMul_1*
_output_shapes
: *
T0*1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1
k
global_norm_4/stackPackglobal_norm_4/L2Loss*

axis *
_output_shapes
:*
N*
T0
]
global_norm_4/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
�
global_norm_4/SumSumglobal_norm_4/stackglobal_norm_4/Const*
_output_shapes
: *

Tidx0*
T0*
	keep_dims( 
Z
global_norm_4/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   @
c
global_norm_4/mulMulglobal_norm_4/Sumglobal_norm_4/Const_1*
_output_shapes
: *
T0
U
global_norm_4/global_normSqrtglobal_norm_4/mul*
_output_shapes
: *
T0
\
fc1_weights/tagsConst*
_output_shapes
: *
dtype0*
valueB Bfc1_weights
j
fc1_weightsScalarSummaryfc1_weights/tagsglobal_norm_4/global_norm*
_output_shapes
: *
T0
�
global_norm_5/L2LossL2Lossgradients/add_grad/Reshape_1*
_output_shapes
: *
T0*/
_class%
#!loc:@gradients/add_grad/Reshape_1
k
global_norm_5/stackPackglobal_norm_5/L2Loss*

axis *
_output_shapes
:*
N*
T0
]
global_norm_5/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
�
global_norm_5/SumSumglobal_norm_5/stackglobal_norm_5/Const*
_output_shapes
: *

Tidx0*
T0*
	keep_dims( 
Z
global_norm_5/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   @
c
global_norm_5/mulMulglobal_norm_5/Sumglobal_norm_5/Const_1*
_output_shapes
: *
T0
U
global_norm_5/global_normSqrtglobal_norm_5/mul*
_output_shapes
: *
T0
Z
fc1_biases/tagsConst*
_output_shapes
: *
dtype0*
valueB B
fc1_biases
h

fc1_biasesScalarSummaryfc1_biases/tagsglobal_norm_5/global_norm*
_output_shapes
: *
T0
�
global_norm_6/L2LossL2Loss gradients/MatMul_1_grad/MatMul_1*
_output_shapes
: *
T0*3
_class)
'%loc:@gradients/MatMul_1_grad/MatMul_1
k
global_norm_6/stackPackglobal_norm_6/L2Loss*

axis *
_output_shapes
:*
N*
T0
]
global_norm_6/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
�
global_norm_6/SumSumglobal_norm_6/stackglobal_norm_6/Const*
_output_shapes
: *

Tidx0*
T0*
	keep_dims( 
Z
global_norm_6/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   @
c
global_norm_6/mulMulglobal_norm_6/Sumglobal_norm_6/Const_1*
_output_shapes
: *
T0
U
global_norm_6/global_normSqrtglobal_norm_6/mul*
_output_shapes
: *
T0
\
fc2_weights/tagsConst*
_output_shapes
: *
dtype0*
valueB Bfc2_weights
j
fc2_weightsScalarSummaryfc2_weights/tagsglobal_norm_6/global_norm*
_output_shapes
: *
T0
�
global_norm_7/L2LossL2Lossgradients/add_1_grad/Reshape_1*
_output_shapes
: *
T0*1
_class'
%#loc:@gradients/add_1_grad/Reshape_1
k
global_norm_7/stackPackglobal_norm_7/L2Loss*

axis *
_output_shapes
:*
N*
T0
]
global_norm_7/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
�
global_norm_7/SumSumglobal_norm_7/stackglobal_norm_7/Const*
_output_shapes
: *

Tidx0*
T0*
	keep_dims( 
Z
global_norm_7/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   @
c
global_norm_7/mulMulglobal_norm_7/Sumglobal_norm_7/Const_1*
_output_shapes
: *
T0
U
global_norm_7/global_normSqrtglobal_norm_7/mul*
_output_shapes
: *
T0
Z
fc2_biases/tagsConst*
_output_shapes
: *
dtype0*
valueB B
fc2_biases
h

fc2_biasesScalarSummaryfc2_biases/tagsglobal_norm_7/global_norm*
_output_shapes
: *
T0
B
L2LossL2LossVariable_4/read*
_output_shapes
: *
T0
D
L2Loss_1L2LossVariable_5/read*
_output_shapes
: *
T0
?
add_2AddL2LossL2Loss_1*
_output_shapes
: *
T0
D
L2Loss_2L2LossVariable_6/read*
_output_shapes
: *
T0
>
add_3Addadd_2L2Loss_2*
_output_shapes
: *
T0
D
L2Loss_3L2LossVariable_7/read*
_output_shapes
: *
T0
>
add_4Addadd_3L2Loss_3*
_output_shapes
: *
T0
L
mul_5/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:
=
mul_5Mulmul_5/xadd_4*
_output_shapes
: *
T0
:
add_5AddMeanmul_5*
_output_shapes
: *
T0
Z
Variable_8/initial_valueConst*
_output_shapes
: *
dtype0*
value	B : 
n

Variable_8
VariableV2*
_output_shapes
: *
dtype0*
shared_name *
	container *
shape: 
�
Variable_8/AssignAssign
Variable_8Variable_8/initial_value*
_output_shapes
: *
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_8
g
Variable_8/readIdentity
Variable_8*
_output_shapes
: *
T0*
_class
loc:@Variable_8
I
mul_6/yConst*
_output_shapes
: *
dtype0*
value	B :
G
mul_6MulVariable_8/readmul_6/y*
_output_shapes
: *
T0
c
ExponentialDecay/learning_rateConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<
T
ExponentialDecay/CastCastmul_6*
_output_shapes
: *

DstT0*

SrcT0
\
ExponentialDecay/Cast_1/xConst*
_output_shapes
: *
dtype0*
value
B :�N
j
ExponentialDecay/Cast_1CastExponentialDecay/Cast_1/x*
_output_shapes
: *

DstT0*

SrcT0
^
ExponentialDecay/Cast_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *33s?
t
ExponentialDecay/truedivRealDivExponentialDecay/CastExponentialDecay/Cast_1*
_output_shapes
: *
T0
Z
ExponentialDecay/FloorFloorExponentialDecay/truediv*
_output_shapes
: *
T0
o
ExponentialDecay/PowPowExponentialDecay/Cast_2/xExponentialDecay/Floor*
_output_shapes
: *
T0
n
ExponentialDecayMulExponentialDecay/learning_rateExponentialDecay/Pow*
_output_shapes
: *
T0
`
learning_rate/tagsConst*
_output_shapes
: *
dtype0*
valueB Blearning_rate
e
learning_rateScalarSummarylearning_rate/tagsExponentialDecay*
_output_shapes
: *
T0
T
gradients_1/ShapeConst*
_output_shapes
: *
dtype0*
valueB 
V
gradients_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?
_
gradients_1/FillFillgradients_1/Shapegradients_1/Const*
_output_shapes
: *
T0
_
gradients_1/add_5_grad/ShapeConst*
_output_shapes
: *
dtype0*
valueB 
a
gradients_1/add_5_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
�
,gradients_1/add_5_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/add_5_grad/Shapegradients_1/add_5_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients_1/add_5_grad/SumSumgradients_1/Fill,gradients_1/add_5_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
T0*
	keep_dims( 
�
gradients_1/add_5_grad/ReshapeReshapegradients_1/add_5_grad/Sumgradients_1/add_5_grad/Shape*
_output_shapes
: *
Tshape0*
T0
�
gradients_1/add_5_grad/Sum_1Sumgradients_1/Fill.gradients_1/add_5_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
T0*
	keep_dims( 
�
 gradients_1/add_5_grad/Reshape_1Reshapegradients_1/add_5_grad/Sum_1gradients_1/add_5_grad/Shape_1*
_output_shapes
: *
Tshape0*
T0
s
'gradients_1/add_5_grad/tuple/group_depsNoOp^gradients_1/add_5_grad/Reshape!^gradients_1/add_5_grad/Reshape_1
�
/gradients_1/add_5_grad/tuple/control_dependencyIdentitygradients_1/add_5_grad/Reshape(^gradients_1/add_5_grad/tuple/group_deps*
_output_shapes
: *
T0*1
_class'
%#loc:@gradients_1/add_5_grad/Reshape
�
1gradients_1/add_5_grad/tuple/control_dependency_1Identity gradients_1/add_5_grad/Reshape_1(^gradients_1/add_5_grad/tuple/group_deps*
_output_shapes
: *
T0*3
_class)
'%loc:@gradients_1/add_5_grad/Reshape_1
m
#gradients_1/Mean_grad/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
�
gradients_1/Mean_grad/ReshapeReshape/gradients_1/add_5_grad/tuple/control_dependency#gradients_1/Mean_grad/Reshape/shape*
_output_shapes
:*
Tshape0*
T0
n
$gradients_1/Mean_grad/Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB:
�
gradients_1/Mean_grad/TileTilegradients_1/Mean_grad/Reshape$gradients_1/Mean_grad/Tile/multiples*
_output_shapes
:*

Tmultiples0*
T0
e
gradients_1/Mean_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:
`
gradients_1/Mean_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
�
gradients_1/Mean_grad/ConstConst*
_output_shapes
:*
dtype0*
valueB: *.
_class$
" loc:@gradients_1/Mean_grad/Shape
�
gradients_1/Mean_grad/ProdProdgradients_1/Mean_grad/Shapegradients_1/Mean_grad/Const*
_output_shapes
: *

Tidx0*
T0*
	keep_dims( *.
_class$
" loc:@gradients_1/Mean_grad/Shape
�
gradients_1/Mean_grad/Const_1Const*
_output_shapes
:*
dtype0*
valueB: *.
_class$
" loc:@gradients_1/Mean_grad/Shape
�
gradients_1/Mean_grad/Prod_1Prodgradients_1/Mean_grad/Shape_1gradients_1/Mean_grad/Const_1*
_output_shapes
: *

Tidx0*
T0*
	keep_dims( *.
_class$
" loc:@gradients_1/Mean_grad/Shape
�
gradients_1/Mean_grad/Maximum/yConst*
_output_shapes
: *
dtype0*
value	B :*.
_class$
" loc:@gradients_1/Mean_grad/Shape
�
gradients_1/Mean_grad/MaximumMaximumgradients_1/Mean_grad/Prod_1gradients_1/Mean_grad/Maximum/y*
_output_shapes
: *
T0*.
_class$
" loc:@gradients_1/Mean_grad/Shape
�
gradients_1/Mean_grad/floordivFloorDivgradients_1/Mean_grad/Prodgradients_1/Mean_grad/Maximum*
_output_shapes
: *
T0*.
_class$
" loc:@gradients_1/Mean_grad/Shape
r
gradients_1/Mean_grad/CastCastgradients_1/Mean_grad/floordiv*
_output_shapes
: *

DstT0*

SrcT0
�
gradients_1/Mean_grad/truedivRealDivgradients_1/Mean_grad/Tilegradients_1/Mean_grad/Cast*
_output_shapes
:*
T0
_
gradients_1/mul_5_grad/ShapeConst*
_output_shapes
: *
dtype0*
valueB 
a
gradients_1/mul_5_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
�
,gradients_1/mul_5_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/mul_5_grad/Shapegradients_1/mul_5_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
|
gradients_1/mul_5_grad/mulMul1gradients_1/add_5_grad/tuple/control_dependency_1add_4*
_output_shapes
: *
T0
�
gradients_1/mul_5_grad/SumSumgradients_1/mul_5_grad/mul,gradients_1/mul_5_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
T0*
	keep_dims( 
�
gradients_1/mul_5_grad/ReshapeReshapegradients_1/mul_5_grad/Sumgradients_1/mul_5_grad/Shape*
_output_shapes
: *
Tshape0*
T0
�
gradients_1/mul_5_grad/mul_1Mulmul_5/x1gradients_1/add_5_grad/tuple/control_dependency_1*
_output_shapes
: *
T0
�
gradients_1/mul_5_grad/Sum_1Sumgradients_1/mul_5_grad/mul_1.gradients_1/mul_5_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
T0*
	keep_dims( 
�
 gradients_1/mul_5_grad/Reshape_1Reshapegradients_1/mul_5_grad/Sum_1gradients_1/mul_5_grad/Shape_1*
_output_shapes
: *
Tshape0*
T0
s
'gradients_1/mul_5_grad/tuple/group_depsNoOp^gradients_1/mul_5_grad/Reshape!^gradients_1/mul_5_grad/Reshape_1
�
/gradients_1/mul_5_grad/tuple/control_dependencyIdentitygradients_1/mul_5_grad/Reshape(^gradients_1/mul_5_grad/tuple/group_deps*
_output_shapes
: *
T0*1
_class'
%#loc:@gradients_1/mul_5_grad/Reshape
�
1gradients_1/mul_5_grad/tuple/control_dependency_1Identity gradients_1/mul_5_grad/Reshape_1(^gradients_1/mul_5_grad/tuple/group_deps*
_output_shapes
: *
T0*3
_class)
'%loc:@gradients_1/mul_5_grad/Reshape_1
k
!gradients_1/Reshape_13_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:
�
#gradients_1/Reshape_13_grad/ReshapeReshapegradients_1/Mean_grad/truediv!gradients_1/Reshape_13_grad/Shape*
_output_shapes
:*
Tshape0*
T0
_
gradients_1/add_4_grad/ShapeConst*
_output_shapes
: *
dtype0*
valueB 
a
gradients_1/add_4_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
�
,gradients_1/add_4_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/add_4_grad/Shapegradients_1/add_4_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients_1/add_4_grad/SumSum1gradients_1/mul_5_grad/tuple/control_dependency_1,gradients_1/add_4_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
T0*
	keep_dims( 
�
gradients_1/add_4_grad/ReshapeReshapegradients_1/add_4_grad/Sumgradients_1/add_4_grad/Shape*
_output_shapes
: *
Tshape0*
T0
�
gradients_1/add_4_grad/Sum_1Sum1gradients_1/mul_5_grad/tuple/control_dependency_1.gradients_1/add_4_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
T0*
	keep_dims( 
�
 gradients_1/add_4_grad/Reshape_1Reshapegradients_1/add_4_grad/Sum_1gradients_1/add_4_grad/Shape_1*
_output_shapes
: *
Tshape0*
T0
s
'gradients_1/add_4_grad/tuple/group_depsNoOp^gradients_1/add_4_grad/Reshape!^gradients_1/add_4_grad/Reshape_1
�
/gradients_1/add_4_grad/tuple/control_dependencyIdentitygradients_1/add_4_grad/Reshape(^gradients_1/add_4_grad/tuple/group_deps*
_output_shapes
: *
T0*1
_class'
%#loc:@gradients_1/add_4_grad/Reshape
�
1gradients_1/add_4_grad/tuple/control_dependency_1Identity gradients_1/add_4_grad/Reshape_1(^gradients_1/add_4_grad/tuple/group_deps*
_output_shapes
: *
T0*3
_class)
'%loc:@gradients_1/add_4_grad/Reshape_1
m
gradients_1/zeros_like	ZerosLikeSoftmaxCrossEntropyWithLogits:1*
_output_shapes

:*
T0
�
=gradients_1/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������
�
9gradients_1/SoftmaxCrossEntropyWithLogits_grad/ExpandDims
ExpandDims#gradients_1/Reshape_13_grad/Reshape=gradients_1/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim*
_output_shapes

:*
T0*

Tdim0
�
2gradients_1/SoftmaxCrossEntropyWithLogits_grad/mulMul9gradients_1/SoftmaxCrossEntropyWithLogits_grad/ExpandDimsSoftmaxCrossEntropyWithLogits:1*
_output_shapes

:*
T0
_
gradients_1/add_3_grad/ShapeConst*
_output_shapes
: *
dtype0*
valueB 
a
gradients_1/add_3_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
�
,gradients_1/add_3_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/add_3_grad/Shapegradients_1/add_3_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients_1/add_3_grad/SumSum/gradients_1/add_4_grad/tuple/control_dependency,gradients_1/add_3_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
T0*
	keep_dims( 
�
gradients_1/add_3_grad/ReshapeReshapegradients_1/add_3_grad/Sumgradients_1/add_3_grad/Shape*
_output_shapes
: *
Tshape0*
T0
�
gradients_1/add_3_grad/Sum_1Sum/gradients_1/add_4_grad/tuple/control_dependency.gradients_1/add_3_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
T0*
	keep_dims( 
�
 gradients_1/add_3_grad/Reshape_1Reshapegradients_1/add_3_grad/Sum_1gradients_1/add_3_grad/Shape_1*
_output_shapes
: *
Tshape0*
T0
s
'gradients_1/add_3_grad/tuple/group_depsNoOp^gradients_1/add_3_grad/Reshape!^gradients_1/add_3_grad/Reshape_1
�
/gradients_1/add_3_grad/tuple/control_dependencyIdentitygradients_1/add_3_grad/Reshape(^gradients_1/add_3_grad/tuple/group_deps*
_output_shapes
: *
T0*1
_class'
%#loc:@gradients_1/add_3_grad/Reshape
�
1gradients_1/add_3_grad/tuple/control_dependency_1Identity gradients_1/add_3_grad/Reshape_1(^gradients_1/add_3_grad/tuple/group_deps*
_output_shapes
: *
T0*3
_class)
'%loc:@gradients_1/add_3_grad/Reshape_1
�
gradients_1/L2Loss_3_grad/mulMulVariable_7/read1gradients_1/add_4_grad/tuple/control_dependency_1*
_output_shapes
:*
T0
r
!gradients_1/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      
�
#gradients_1/Reshape_11_grad/ReshapeReshape2gradients_1/SoftmaxCrossEntropyWithLogits_grad/mul!gradients_1/Reshape_11_grad/Shape*
_output_shapes

:*
Tshape0*
T0
_
gradients_1/add_2_grad/ShapeConst*
_output_shapes
: *
dtype0*
valueB 
a
gradients_1/add_2_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
�
,gradients_1/add_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/add_2_grad/Shapegradients_1/add_2_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients_1/add_2_grad/SumSum/gradients_1/add_3_grad/tuple/control_dependency,gradients_1/add_2_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
T0*
	keep_dims( 
�
gradients_1/add_2_grad/ReshapeReshapegradients_1/add_2_grad/Sumgradients_1/add_2_grad/Shape*
_output_shapes
: *
Tshape0*
T0
�
gradients_1/add_2_grad/Sum_1Sum/gradients_1/add_3_grad/tuple/control_dependency.gradients_1/add_2_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
T0*
	keep_dims( 
�
 gradients_1/add_2_grad/Reshape_1Reshapegradients_1/add_2_grad/Sum_1gradients_1/add_2_grad/Shape_1*
_output_shapes
: *
Tshape0*
T0
s
'gradients_1/add_2_grad/tuple/group_depsNoOp^gradients_1/add_2_grad/Reshape!^gradients_1/add_2_grad/Reshape_1
�
/gradients_1/add_2_grad/tuple/control_dependencyIdentitygradients_1/add_2_grad/Reshape(^gradients_1/add_2_grad/tuple/group_deps*
_output_shapes
: *
T0*1
_class'
%#loc:@gradients_1/add_2_grad/Reshape
�
1gradients_1/add_2_grad/tuple/control_dependency_1Identity gradients_1/add_2_grad/Reshape_1(^gradients_1/add_2_grad/tuple/group_deps*
_output_shapes
: *
T0*3
_class)
'%loc:@gradients_1/add_2_grad/Reshape_1
�
gradients_1/L2Loss_2_grad/mulMulVariable_6/read1gradients_1/add_3_grad/tuple/control_dependency_1*
_output_shapes
:	�*
T0
m
gradients_1/add_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      
h
gradients_1/add_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:
�
,gradients_1/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/add_1_grad/Shapegradients_1/add_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients_1/add_1_grad/SumSum#gradients_1/Reshape_11_grad/Reshape,gradients_1/add_1_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
T0*
	keep_dims( 
�
gradients_1/add_1_grad/ReshapeReshapegradients_1/add_1_grad/Sumgradients_1/add_1_grad/Shape*
_output_shapes

:*
Tshape0*
T0
�
gradients_1/add_1_grad/Sum_1Sum#gradients_1/Reshape_11_grad/Reshape.gradients_1/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
T0*
	keep_dims( 
�
 gradients_1/add_1_grad/Reshape_1Reshapegradients_1/add_1_grad/Sum_1gradients_1/add_1_grad/Shape_1*
_output_shapes
:*
Tshape0*
T0
s
'gradients_1/add_1_grad/tuple/group_depsNoOp^gradients_1/add_1_grad/Reshape!^gradients_1/add_1_grad/Reshape_1
�
/gradients_1/add_1_grad/tuple/control_dependencyIdentitygradients_1/add_1_grad/Reshape(^gradients_1/add_1_grad/tuple/group_deps*
_output_shapes

:*
T0*1
_class'
%#loc:@gradients_1/add_1_grad/Reshape
�
1gradients_1/add_1_grad/tuple/control_dependency_1Identity gradients_1/add_1_grad/Reshape_1(^gradients_1/add_1_grad/tuple/group_deps*
_output_shapes
:*
T0*3
_class)
'%loc:@gradients_1/add_1_grad/Reshape_1
�
gradients_1/L2Loss_grad/mulMulVariable_4/read/gradients_1/add_2_grad/tuple/control_dependency* 
_output_shapes
:
��*
T0
�
gradients_1/L2Loss_1_grad/mulMulVariable_5/read1gradients_1/add_2_grad/tuple/control_dependency_1*
_output_shapes	
:�*
T0
�
 gradients_1/MatMul_1_grad/MatMulMatMul/gradients_1/add_1_grad/tuple/control_dependencyVariable_6/read*
_output_shapes
:	�*
transpose_b(*
T0*
transpose_a( 
�
"gradients_1/MatMul_1_grad/MatMul_1MatMulRelu_2/gradients_1/add_1_grad/tuple/control_dependency*
_output_shapes
:	�*
transpose_b( *
T0*
transpose_a(
z
*gradients_1/MatMul_1_grad/tuple/group_depsNoOp!^gradients_1/MatMul_1_grad/MatMul#^gradients_1/MatMul_1_grad/MatMul_1
�
2gradients_1/MatMul_1_grad/tuple/control_dependencyIdentity gradients_1/MatMul_1_grad/MatMul+^gradients_1/MatMul_1_grad/tuple/group_deps*
_output_shapes
:	�*
T0*3
_class)
'%loc:@gradients_1/MatMul_1_grad/MatMul
�
4gradients_1/MatMul_1_grad/tuple/control_dependency_1Identity"gradients_1/MatMul_1_grad/MatMul_1+^gradients_1/MatMul_1_grad/tuple/group_deps*
_output_shapes
:	�*
T0*5
_class+
)'loc:@gradients_1/MatMul_1_grad/MatMul_1
�
gradients_1/AddNAddNgradients_1/L2Loss_3_grad/mul1gradients_1/add_1_grad/tuple/control_dependency_1*
_output_shapes
:*
N*
T0*0
_class&
$"loc:@gradients_1/L2Loss_3_grad/mul
�
 gradients_1/Relu_2_grad/ReluGradReluGrad2gradients_1/MatMul_1_grad/tuple/control_dependencyRelu_2*
_output_shapes
:	�*
T0
�
gradients_1/AddN_1AddNgradients_1/L2Loss_2_grad/mul4gradients_1/MatMul_1_grad/tuple/control_dependency_1*
_output_shapes
:	�*
N*
T0*0
_class&
$"loc:@gradients_1/L2Loss_2_grad/mul
k
gradients_1/add_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      
g
gradients_1/add_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:�
�
*gradients_1/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/add_grad/Shapegradients_1/add_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients_1/add_grad/SumSum gradients_1/Relu_2_grad/ReluGrad*gradients_1/add_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
T0*
	keep_dims( 
�
gradients_1/add_grad/ReshapeReshapegradients_1/add_grad/Sumgradients_1/add_grad/Shape*
_output_shapes
:	�*
Tshape0*
T0
�
gradients_1/add_grad/Sum_1Sum gradients_1/Relu_2_grad/ReluGrad,gradients_1/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
T0*
	keep_dims( 
�
gradients_1/add_grad/Reshape_1Reshapegradients_1/add_grad/Sum_1gradients_1/add_grad/Shape_1*
_output_shapes	
:�*
Tshape0*
T0
m
%gradients_1/add_grad/tuple/group_depsNoOp^gradients_1/add_grad/Reshape^gradients_1/add_grad/Reshape_1
�
-gradients_1/add_grad/tuple/control_dependencyIdentitygradients_1/add_grad/Reshape&^gradients_1/add_grad/tuple/group_deps*
_output_shapes
:	�*
T0*/
_class%
#!loc:@gradients_1/add_grad/Reshape
�
/gradients_1/add_grad/tuple/control_dependency_1Identitygradients_1/add_grad/Reshape_1&^gradients_1/add_grad/tuple/group_deps*
_output_shapes	
:�*
T0*1
_class'
%#loc:@gradients_1/add_grad/Reshape_1
�
gradients_1/MatMul_grad/MatMulMatMul-gradients_1/add_grad/tuple/control_dependencyVariable_4/read*
_output_shapes
:	�*
transpose_b(*
T0*
transpose_a( 
�
 gradients_1/MatMul_grad/MatMul_1MatMulReshape-gradients_1/add_grad/tuple/control_dependency* 
_output_shapes
:
��*
transpose_b( *
T0*
transpose_a(
t
(gradients_1/MatMul_grad/tuple/group_depsNoOp^gradients_1/MatMul_grad/MatMul!^gradients_1/MatMul_grad/MatMul_1
�
0gradients_1/MatMul_grad/tuple/control_dependencyIdentitygradients_1/MatMul_grad/MatMul)^gradients_1/MatMul_grad/tuple/group_deps*
_output_shapes
:	�*
T0*1
_class'
%#loc:@gradients_1/MatMul_grad/MatMul
�
2gradients_1/MatMul_grad/tuple/control_dependency_1Identity gradients_1/MatMul_grad/MatMul_1)^gradients_1/MatMul_grad/tuple/group_deps* 
_output_shapes
:
��*
T0*3
_class)
'%loc:@gradients_1/MatMul_grad/MatMul_1
�
gradients_1/AddN_2AddNgradients_1/L2Loss_1_grad/mul/gradients_1/add_grad/tuple/control_dependency_1*
_output_shapes	
:�*
N*
T0*0
_class&
$"loc:@gradients_1/L2Loss_1_grad/mul
w
gradients_1/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"         @   
�
 gradients_1/Reshape_grad/ReshapeReshape0gradients_1/MatMul_grad/tuple/control_dependencygradients_1/Reshape_grad/Shape*&
_output_shapes
:@*
Tshape0*
T0
�
gradients_1/AddN_3AddNgradients_1/L2Loss_grad/mul2gradients_1/MatMul_grad/tuple/control_dependency_1* 
_output_shapes
:
��*
N*
T0*.
_class$
" loc:@gradients_1/L2Loss_grad/mul
�
&gradients_1/MaxPool_1_grad/MaxPoolGradMaxPoolGradRelu_1	MaxPool_1 gradients_1/Reshape_grad/Reshape*&
_output_shapes
:@*
ksize
*
T0*
data_formatNHWC*
paddingSAME*
strides

�
 gradients_1/Relu_1_grad/ReluGradReluGrad&gradients_1/MaxPool_1_grad/MaxPoolGradRelu_1*&
_output_shapes
:@*
T0
�
&gradients_1/BiasAdd_1_grad/BiasAddGradBiasAddGrad gradients_1/Relu_1_grad/ReluGrad*
_output_shapes
:@*
data_formatNHWC*
T0

+gradients_1/BiasAdd_1_grad/tuple/group_depsNoOp!^gradients_1/Relu_1_grad/ReluGrad'^gradients_1/BiasAdd_1_grad/BiasAddGrad
�
3gradients_1/BiasAdd_1_grad/tuple/control_dependencyIdentity gradients_1/Relu_1_grad/ReluGrad,^gradients_1/BiasAdd_1_grad/tuple/group_deps*&
_output_shapes
:@*
T0*3
_class)
'%loc:@gradients_1/Relu_1_grad/ReluGrad
�
5gradients_1/BiasAdd_1_grad/tuple/control_dependency_1Identity&gradients_1/BiasAdd_1_grad/BiasAddGrad,^gradients_1/BiasAdd_1_grad/tuple/group_deps*
_output_shapes
:@*
T0*9
_class/
-+loc:@gradients_1/BiasAdd_1_grad/BiasAddGrad
�
 gradients_1/Conv2D_1_grad/ShapeNShapeNMaxPoolVariable_2/read* 
_output_shapes
::*
N*
out_type0*
T0
�
-gradients_1/Conv2D_1_grad/Conv2DBackpropInputConv2DBackpropInput gradients_1/Conv2D_1_grad/ShapeNVariable_2/read3gradients_1/BiasAdd_1_grad/tuple/control_dependency*J
_output_shapes8
6:4������������������������������������*
T0*
use_cudnn_on_gpu(*
strides
*
paddingSAME*
data_formatNHWC
�
.gradients_1/Conv2D_1_grad/Conv2DBackpropFilterConv2DBackpropFilterMaxPool"gradients_1/Conv2D_1_grad/ShapeN:13gradients_1/BiasAdd_1_grad/tuple/control_dependency*J
_output_shapes8
6:4������������������������������������*
T0*
use_cudnn_on_gpu(*
strides
*
paddingSAME*
data_formatNHWC
�
*gradients_1/Conv2D_1_grad/tuple/group_depsNoOp.^gradients_1/Conv2D_1_grad/Conv2DBackpropInput/^gradients_1/Conv2D_1_grad/Conv2DBackpropFilter
�
2gradients_1/Conv2D_1_grad/tuple/control_dependencyIdentity-gradients_1/Conv2D_1_grad/Conv2DBackpropInput+^gradients_1/Conv2D_1_grad/tuple/group_deps*&
_output_shapes
: *
T0*@
_class6
42loc:@gradients_1/Conv2D_1_grad/Conv2DBackpropInput
�
4gradients_1/Conv2D_1_grad/tuple/control_dependency_1Identity.gradients_1/Conv2D_1_grad/Conv2DBackpropFilter+^gradients_1/Conv2D_1_grad/tuple/group_deps*&
_output_shapes
: @*
T0*A
_class7
53loc:@gradients_1/Conv2D_1_grad/Conv2DBackpropFilter
�
$gradients_1/MaxPool_grad/MaxPoolGradMaxPoolGradReluMaxPool2gradients_1/Conv2D_1_grad/tuple/control_dependency*&
_output_shapes
: *
ksize
*
T0*
data_formatNHWC*
paddingSAME*
strides

�
gradients_1/Relu_grad/ReluGradReluGrad$gradients_1/MaxPool_grad/MaxPoolGradRelu*&
_output_shapes
: *
T0
�
$gradients_1/BiasAdd_grad/BiasAddGradBiasAddGradgradients_1/Relu_grad/ReluGrad*
_output_shapes
: *
data_formatNHWC*
T0
y
)gradients_1/BiasAdd_grad/tuple/group_depsNoOp^gradients_1/Relu_grad/ReluGrad%^gradients_1/BiasAdd_grad/BiasAddGrad
�
1gradients_1/BiasAdd_grad/tuple/control_dependencyIdentitygradients_1/Relu_grad/ReluGrad*^gradients_1/BiasAdd_grad/tuple/group_deps*&
_output_shapes
: *
T0*1
_class'
%#loc:@gradients_1/Relu_grad/ReluGrad
�
3gradients_1/BiasAdd_grad/tuple/control_dependency_1Identity$gradients_1/BiasAdd_grad/BiasAddGrad*^gradients_1/BiasAdd_grad/tuple/group_deps*
_output_shapes
: *
T0*7
_class-
+)loc:@gradients_1/BiasAdd_grad/BiasAddGrad
�
gradients_1/Conv2D_grad/ShapeNShapeNPlaceholderVariable/read* 
_output_shapes
::*
N*
out_type0*
T0
�
+gradients_1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInputgradients_1/Conv2D_grad/ShapeNVariable/read1gradients_1/BiasAdd_grad/tuple/control_dependency*J
_output_shapes8
6:4������������������������������������*
T0*
use_cudnn_on_gpu(*
strides
*
paddingSAME*
data_formatNHWC
�
,gradients_1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterPlaceholder gradients_1/Conv2D_grad/ShapeN:11gradients_1/BiasAdd_grad/tuple/control_dependency*J
_output_shapes8
6:4������������������������������������*
T0*
use_cudnn_on_gpu(*
strides
*
paddingSAME*
data_formatNHWC
�
(gradients_1/Conv2D_grad/tuple/group_depsNoOp,^gradients_1/Conv2D_grad/Conv2DBackpropInput-^gradients_1/Conv2D_grad/Conv2DBackpropFilter
�
0gradients_1/Conv2D_grad/tuple/control_dependencyIdentity+gradients_1/Conv2D_grad/Conv2DBackpropInput)^gradients_1/Conv2D_grad/tuple/group_deps*&
_output_shapes
:*
T0*>
_class4
20loc:@gradients_1/Conv2D_grad/Conv2DBackpropInput
�
2gradients_1/Conv2D_grad/tuple/control_dependency_1Identity,gradients_1/Conv2D_grad/Conv2DBackpropFilter)^gradients_1/Conv2D_grad/tuple/group_deps*&
_output_shapes
: *
T0*?
_class5
31loc:@gradients_1/Conv2D_grad/Conv2DBackpropFilter
�
#Variable/Momentum/Initializer/zerosConst*&
_output_shapes
: *
dtype0*%
valueB *    *
_class
loc:@Variable
�
Variable/Momentum
VariableV2*&
_output_shapes
: *
shared_name *
	container *
shape: *
_class
loc:@Variable*
dtype0
�
Variable/Momentum/AssignAssignVariable/Momentum#Variable/Momentum/Initializer/zeros*&
_output_shapes
: *
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable
�
Variable/Momentum/readIdentityVariable/Momentum*&
_output_shapes
: *
T0*
_class
loc:@Variable
�
%Variable_1/Momentum/Initializer/zerosConst*
_output_shapes
: *
dtype0*
valueB *    *
_class
loc:@Variable_1
�
Variable_1/Momentum
VariableV2*
_output_shapes
: *
shared_name *
	container *
shape: *
_class
loc:@Variable_1*
dtype0
�
Variable_1/Momentum/AssignAssignVariable_1/Momentum%Variable_1/Momentum/Initializer/zeros*
_output_shapes
: *
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_1
}
Variable_1/Momentum/readIdentityVariable_1/Momentum*
_output_shapes
: *
T0*
_class
loc:@Variable_1
�
%Variable_2/Momentum/Initializer/zerosConst*&
_output_shapes
: @*
dtype0*%
valueB @*    *
_class
loc:@Variable_2
�
Variable_2/Momentum
VariableV2*&
_output_shapes
: @*
shared_name *
	container *
shape: @*
_class
loc:@Variable_2*
dtype0
�
Variable_2/Momentum/AssignAssignVariable_2/Momentum%Variable_2/Momentum/Initializer/zeros*&
_output_shapes
: @*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_2
�
Variable_2/Momentum/readIdentityVariable_2/Momentum*&
_output_shapes
: @*
T0*
_class
loc:@Variable_2
�
%Variable_3/Momentum/Initializer/zerosConst*
_output_shapes
:@*
dtype0*
valueB@*    *
_class
loc:@Variable_3
�
Variable_3/Momentum
VariableV2*
_output_shapes
:@*
shared_name *
	container *
shape:@*
_class
loc:@Variable_3*
dtype0
�
Variable_3/Momentum/AssignAssignVariable_3/Momentum%Variable_3/Momentum/Initializer/zeros*
_output_shapes
:@*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_3
}
Variable_3/Momentum/readIdentityVariable_3/Momentum*
_output_shapes
:@*
T0*
_class
loc:@Variable_3
�
%Variable_4/Momentum/Initializer/zerosConst* 
_output_shapes
:
��*
dtype0*
valueB
��*    *
_class
loc:@Variable_4
�
Variable_4/Momentum
VariableV2* 
_output_shapes
:
��*
shared_name *
	container *
shape:
��*
_class
loc:@Variable_4*
dtype0
�
Variable_4/Momentum/AssignAssignVariable_4/Momentum%Variable_4/Momentum/Initializer/zeros* 
_output_shapes
:
��*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_4
�
Variable_4/Momentum/readIdentityVariable_4/Momentum* 
_output_shapes
:
��*
T0*
_class
loc:@Variable_4
�
%Variable_5/Momentum/Initializer/zerosConst*
_output_shapes	
:�*
dtype0*
valueB�*    *
_class
loc:@Variable_5
�
Variable_5/Momentum
VariableV2*
_output_shapes	
:�*
shared_name *
	container *
shape:�*
_class
loc:@Variable_5*
dtype0
�
Variable_5/Momentum/AssignAssignVariable_5/Momentum%Variable_5/Momentum/Initializer/zeros*
_output_shapes	
:�*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_5
~
Variable_5/Momentum/readIdentityVariable_5/Momentum*
_output_shapes	
:�*
T0*
_class
loc:@Variable_5
�
%Variable_6/Momentum/Initializer/zerosConst*
_output_shapes
:	�*
dtype0*
valueB	�*    *
_class
loc:@Variable_6
�
Variable_6/Momentum
VariableV2*
_output_shapes
:	�*
shared_name *
	container *
shape:	�*
_class
loc:@Variable_6*
dtype0
�
Variable_6/Momentum/AssignAssignVariable_6/Momentum%Variable_6/Momentum/Initializer/zeros*
_output_shapes
:	�*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_6
�
Variable_6/Momentum/readIdentityVariable_6/Momentum*
_output_shapes
:	�*
T0*
_class
loc:@Variable_6
�
%Variable_7/Momentum/Initializer/zerosConst*
_output_shapes
:*
dtype0*
valueB*    *
_class
loc:@Variable_7
�
Variable_7/Momentum
VariableV2*
_output_shapes
:*
shared_name *
	container *
shape:*
_class
loc:@Variable_7*
dtype0
�
Variable_7/Momentum/AssignAssignVariable_7/Momentum%Variable_7/Momentum/Initializer/zeros*
_output_shapes
:*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_7
}
Variable_7/Momentum/readIdentityVariable_7/Momentum*
_output_shapes
:*
T0*
_class
loc:@Variable_7
V
Momentum/momentumConst*
_output_shapes
: *
dtype0*
valueB
 *    
�
&Momentum/update_Variable/ApplyMomentumApplyMomentumVariableVariable/MomentumExponentialDecay2gradients_1/Conv2D_grad/tuple/control_dependency_1Momentum/momentum*&
_output_shapes
: *
use_nesterov( *
use_locking( *
T0*
_class
loc:@Variable
�
(Momentum/update_Variable_1/ApplyMomentumApplyMomentum
Variable_1Variable_1/MomentumExponentialDecay3gradients_1/BiasAdd_grad/tuple/control_dependency_1Momentum/momentum*
_output_shapes
: *
use_nesterov( *
use_locking( *
T0*
_class
loc:@Variable_1
�
(Momentum/update_Variable_2/ApplyMomentumApplyMomentum
Variable_2Variable_2/MomentumExponentialDecay4gradients_1/Conv2D_1_grad/tuple/control_dependency_1Momentum/momentum*&
_output_shapes
: @*
use_nesterov( *
use_locking( *
T0*
_class
loc:@Variable_2
�
(Momentum/update_Variable_3/ApplyMomentumApplyMomentum
Variable_3Variable_3/MomentumExponentialDecay5gradients_1/BiasAdd_1_grad/tuple/control_dependency_1Momentum/momentum*
_output_shapes
:@*
use_nesterov( *
use_locking( *
T0*
_class
loc:@Variable_3
�
(Momentum/update_Variable_4/ApplyMomentumApplyMomentum
Variable_4Variable_4/MomentumExponentialDecaygradients_1/AddN_3Momentum/momentum* 
_output_shapes
:
��*
use_nesterov( *
use_locking( *
T0*
_class
loc:@Variable_4
�
(Momentum/update_Variable_5/ApplyMomentumApplyMomentum
Variable_5Variable_5/MomentumExponentialDecaygradients_1/AddN_2Momentum/momentum*
_output_shapes	
:�*
use_nesterov( *
use_locking( *
T0*
_class
loc:@Variable_5
�
(Momentum/update_Variable_6/ApplyMomentumApplyMomentum
Variable_6Variable_6/MomentumExponentialDecaygradients_1/AddN_1Momentum/momentum*
_output_shapes
:	�*
use_nesterov( *
use_locking( *
T0*
_class
loc:@Variable_6
�
(Momentum/update_Variable_7/ApplyMomentumApplyMomentum
Variable_7Variable_7/MomentumExponentialDecaygradients_1/AddNMomentum/momentum*
_output_shapes
:*
use_nesterov( *
use_locking( *
T0*
_class
loc:@Variable_7
�
Momentum/updateNoOp'^Momentum/update_Variable/ApplyMomentum)^Momentum/update_Variable_1/ApplyMomentum)^Momentum/update_Variable_2/ApplyMomentum)^Momentum/update_Variable_3/ApplyMomentum)^Momentum/update_Variable_4/ApplyMomentum)^Momentum/update_Variable_5/ApplyMomentum)^Momentum/update_Variable_6/ApplyMomentum)^Momentum/update_Variable_7/ApplyMomentum
�
Momentum/valueConst^Momentum/update*
_output_shapes
: *
dtype0*
value	B :*
_class
loc:@Variable_8
�
Momentum	AssignAdd
Variable_8Momentum/value*
_output_shapes
: *
use_locking( *
T0*
_class
loc:@Variable_8
B
SoftmaxSoftmaxadd_1*
_output_shapes

:*
T0
�
Conv2D_2Conv2DPlaceholder_2Variable/read*'
_output_shapes
:�N *
T0*
use_cudnn_on_gpu(*
strides
*
paddingSAME*
data_formatNHWC
x
	BiasAdd_2BiasAddConv2D_2Variable_1/read*'
_output_shapes
:�N *
data_formatNHWC*
T0
K
Relu_3Relu	BiasAdd_2*'
_output_shapes
:�N *
T0
�
	MaxPool_2MaxPoolRelu_3*'
_output_shapes
:�N *
ksize
*
T0*
data_formatNHWC*
paddingSAME*
strides

�
Conv2D_3Conv2D	MaxPool_2Variable_2/read*'
_output_shapes
:�N@*
T0*
use_cudnn_on_gpu(*
strides
*
paddingSAME*
data_formatNHWC
x
	BiasAdd_3BiasAddConv2D_3Variable_3/read*'
_output_shapes
:�N@*
data_formatNHWC*
T0
K
Relu_4Relu	BiasAdd_3*'
_output_shapes
:�N@*
T0
�
	MaxPool_3MaxPoolRelu_4*'
_output_shapes
:�N@*
ksize
*
T0*
data_formatNHWC*
paddingSAME*
strides

a
Reshape_14/shapeConst*
_output_shapes
:*
dtype0*
valueB"'     
k

Reshape_14Reshape	MaxPool_3Reshape_14/shape* 
_output_shapes
:
�N�*
Tshape0*
T0
�
MatMul_2MatMul
Reshape_14Variable_4/read* 
_output_shapes
:
�N�*
transpose_b( *
T0*
transpose_a( 
R
add_6AddMatMul_2Variable_5/read* 
_output_shapes
:
�N�*
T0
@
Relu_5Reluadd_6* 
_output_shapes
:
�N�*
T0
{
MatMul_3MatMulRelu_5Variable_6/read*
_output_shapes
:	�N*
transpose_b( *
T0*
transpose_a( 
Q
add_7AddMatMul_3Variable_7/read*
_output_shapes
:	�N*
T0
E
	Softmax_1Softmaxadd_7*
_output_shapes
:	�N*
T0
P

save/ConstConst*
_output_shapes
: *
dtype0*
valueB Bmodel
�
save/SaveV2/tensor_namesConst*
_output_shapes
:*
dtype0*�
value�B�BVariableBVariable/MomentumB
Variable_1BVariable_1/MomentumB
Variable_2BVariable_2/MomentumB
Variable_3BVariable_3/MomentumB
Variable_4BVariable_4/MomentumB
Variable_5BVariable_5/MomentumB
Variable_6BVariable_6/MomentumB
Variable_7BVariable_7/MomentumB
Variable_8
�
save/SaveV2/shape_and_slicesConst*
_output_shapes
:*
dtype0*5
value,B*B B B B B B B B B B B B B B B B B 
�
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesVariableVariable/Momentum
Variable_1Variable_1/Momentum
Variable_2Variable_2/Momentum
Variable_3Variable_3/Momentum
Variable_4Variable_4/Momentum
Variable_5Variable_5/Momentum
Variable_6Variable_6/Momentum
Variable_7Variable_7/Momentum
Variable_8*
dtypes
2
}
save/control_dependencyIdentity
save/Const^save/SaveV2*
_output_shapes
: *
T0*
_class
loc:@save/Const
l
save/RestoreV2/tensor_namesConst*
_output_shapes
:*
dtype0*
valueBBVariable
h
save/RestoreV2/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 
�
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/AssignAssignVariablesave/RestoreV2*&
_output_shapes
: *
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable
w
save/RestoreV2_1/tensor_namesConst*
_output_shapes
:*
dtype0*&
valueBBVariable/Momentum
j
!save/RestoreV2_1/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 
�
save/RestoreV2_1	RestoreV2
save/Constsave/RestoreV2_1/tensor_names!save/RestoreV2_1/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_1AssignVariable/Momentumsave/RestoreV2_1*&
_output_shapes
: *
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable
p
save/RestoreV2_2/tensor_namesConst*
_output_shapes
:*
dtype0*
valueBB
Variable_1
j
!save/RestoreV2_2/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 
�
save/RestoreV2_2	RestoreV2
save/Constsave/RestoreV2_2/tensor_names!save/RestoreV2_2/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_2Assign
Variable_1save/RestoreV2_2*
_output_shapes
: *
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_1
y
save/RestoreV2_3/tensor_namesConst*
_output_shapes
:*
dtype0*(
valueBBVariable_1/Momentum
j
!save/RestoreV2_3/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 
�
save/RestoreV2_3	RestoreV2
save/Constsave/RestoreV2_3/tensor_names!save/RestoreV2_3/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_3AssignVariable_1/Momentumsave/RestoreV2_3*
_output_shapes
: *
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_1
p
save/RestoreV2_4/tensor_namesConst*
_output_shapes
:*
dtype0*
valueBB
Variable_2
j
!save/RestoreV2_4/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 
�
save/RestoreV2_4	RestoreV2
save/Constsave/RestoreV2_4/tensor_names!save/RestoreV2_4/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_4Assign
Variable_2save/RestoreV2_4*&
_output_shapes
: @*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_2
y
save/RestoreV2_5/tensor_namesConst*
_output_shapes
:*
dtype0*(
valueBBVariable_2/Momentum
j
!save/RestoreV2_5/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 
�
save/RestoreV2_5	RestoreV2
save/Constsave/RestoreV2_5/tensor_names!save/RestoreV2_5/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_5AssignVariable_2/Momentumsave/RestoreV2_5*&
_output_shapes
: @*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_2
p
save/RestoreV2_6/tensor_namesConst*
_output_shapes
:*
dtype0*
valueBB
Variable_3
j
!save/RestoreV2_6/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 
�
save/RestoreV2_6	RestoreV2
save/Constsave/RestoreV2_6/tensor_names!save/RestoreV2_6/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_6Assign
Variable_3save/RestoreV2_6*
_output_shapes
:@*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_3
y
save/RestoreV2_7/tensor_namesConst*
_output_shapes
:*
dtype0*(
valueBBVariable_3/Momentum
j
!save/RestoreV2_7/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 
�
save/RestoreV2_7	RestoreV2
save/Constsave/RestoreV2_7/tensor_names!save/RestoreV2_7/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_7AssignVariable_3/Momentumsave/RestoreV2_7*
_output_shapes
:@*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_3
p
save/RestoreV2_8/tensor_namesConst*
_output_shapes
:*
dtype0*
valueBB
Variable_4
j
!save/RestoreV2_8/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 
�
save/RestoreV2_8	RestoreV2
save/Constsave/RestoreV2_8/tensor_names!save/RestoreV2_8/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_8Assign
Variable_4save/RestoreV2_8* 
_output_shapes
:
��*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_4
y
save/RestoreV2_9/tensor_namesConst*
_output_shapes
:*
dtype0*(
valueBBVariable_4/Momentum
j
!save/RestoreV2_9/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 
�
save/RestoreV2_9	RestoreV2
save/Constsave/RestoreV2_9/tensor_names!save/RestoreV2_9/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_9AssignVariable_4/Momentumsave/RestoreV2_9* 
_output_shapes
:
��*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_4
q
save/RestoreV2_10/tensor_namesConst*
_output_shapes
:*
dtype0*
valueBB
Variable_5
k
"save/RestoreV2_10/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 
�
save/RestoreV2_10	RestoreV2
save/Constsave/RestoreV2_10/tensor_names"save/RestoreV2_10/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_10Assign
Variable_5save/RestoreV2_10*
_output_shapes	
:�*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_5
z
save/RestoreV2_11/tensor_namesConst*
_output_shapes
:*
dtype0*(
valueBBVariable_5/Momentum
k
"save/RestoreV2_11/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 
�
save/RestoreV2_11	RestoreV2
save/Constsave/RestoreV2_11/tensor_names"save/RestoreV2_11/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_11AssignVariable_5/Momentumsave/RestoreV2_11*
_output_shapes	
:�*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_5
q
save/RestoreV2_12/tensor_namesConst*
_output_shapes
:*
dtype0*
valueBB
Variable_6
k
"save/RestoreV2_12/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 
�
save/RestoreV2_12	RestoreV2
save/Constsave/RestoreV2_12/tensor_names"save/RestoreV2_12/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_12Assign
Variable_6save/RestoreV2_12*
_output_shapes
:	�*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_6
z
save/RestoreV2_13/tensor_namesConst*
_output_shapes
:*
dtype0*(
valueBBVariable_6/Momentum
k
"save/RestoreV2_13/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 
�
save/RestoreV2_13	RestoreV2
save/Constsave/RestoreV2_13/tensor_names"save/RestoreV2_13/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_13AssignVariable_6/Momentumsave/RestoreV2_13*
_output_shapes
:	�*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_6
q
save/RestoreV2_14/tensor_namesConst*
_output_shapes
:*
dtype0*
valueBB
Variable_7
k
"save/RestoreV2_14/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 
�
save/RestoreV2_14	RestoreV2
save/Constsave/RestoreV2_14/tensor_names"save/RestoreV2_14/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_14Assign
Variable_7save/RestoreV2_14*
_output_shapes
:*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_7
z
save/RestoreV2_15/tensor_namesConst*
_output_shapes
:*
dtype0*(
valueBBVariable_7/Momentum
k
"save/RestoreV2_15/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 
�
save/RestoreV2_15	RestoreV2
save/Constsave/RestoreV2_15/tensor_names"save/RestoreV2_15/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_15AssignVariable_7/Momentumsave/RestoreV2_15*
_output_shapes
:*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_7
q
save/RestoreV2_16/tensor_namesConst*
_output_shapes
:*
dtype0*
valueBB
Variable_8
k
"save/RestoreV2_16/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 
�
save/RestoreV2_16	RestoreV2
save/Constsave/RestoreV2_16/tensor_names"save/RestoreV2_16/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_16Assign
Variable_8save/RestoreV2_16*
_output_shapes
: *
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_8
�
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16
�
initNoOp^Variable/Assign^Variable_1/Assign^Variable_2/Assign^Variable_3/Assign^Variable_4/Assign^Variable_5/Assign^Variable_6/Assign^Variable_7/Assign^Variable_8/Assign^Variable/Momentum/Assign^Variable_1/Momentum/Assign^Variable_2/Momentum/Assign^Variable_3/Momentum/Assign^Variable_4/Momentum/Assign^Variable_5/Momentum/Assign^Variable_6/Momentum/Assign^Variable_7/Momentum/Assign
�
Merge/MergeSummaryMergeSummarysummary_data_0summary_conv_0summary_pool_0summary_conv2_0summary_pool2_0lossconv1_weightsconv1_biasesconv2_weightsconv2_biasesfc1_weights
fc1_biasesfc2_weights
fc2_biaseslearning_rate*
_output_shapes
: *
N""�
trainable_variables��
B

Variable:0Variable/AssignVariable/read:02truncated_normal:0
=
Variable_1:0Variable_1/AssignVariable_1/read:02zeros:0
J
Variable_2:0Variable_2/AssignVariable_2/read:02truncated_normal_1:0
=
Variable_3:0Variable_3/AssignVariable_3/read:02Const:0
J
Variable_4:0Variable_4/AssignVariable_4/read:02truncated_normal_2:0
?
Variable_5:0Variable_5/AssignVariable_5/read:02	Const_1:0
J
Variable_6:0Variable_6/AssignVariable_6/read:02truncated_normal_3:0
?
Variable_7:0Variable_7/AssignVariable_7/read:02	Const_2:0
P
Variable_8:0Variable_8/AssignVariable_8/read:02Variable_8/initial_value:0"�
	variables��
B

Variable:0Variable/AssignVariable/read:02truncated_normal:0
=
Variable_1:0Variable_1/AssignVariable_1/read:02zeros:0
J
Variable_2:0Variable_2/AssignVariable_2/read:02truncated_normal_1:0
=
Variable_3:0Variable_3/AssignVariable_3/read:02Const:0
J
Variable_4:0Variable_4/AssignVariable_4/read:02truncated_normal_2:0
?
Variable_5:0Variable_5/AssignVariable_5/read:02	Const_1:0
J
Variable_6:0Variable_6/AssignVariable_6/read:02truncated_normal_3:0
?
Variable_7:0Variable_7/AssignVariable_7/read:02	Const_2:0
P
Variable_8:0Variable_8/AssignVariable_8/read:02Variable_8/initial_value:0
p
Variable/Momentum:0Variable/Momentum/AssignVariable/Momentum/read:02%Variable/Momentum/Initializer/zeros:0
x
Variable_1/Momentum:0Variable_1/Momentum/AssignVariable_1/Momentum/read:02'Variable_1/Momentum/Initializer/zeros:0
x
Variable_2/Momentum:0Variable_2/Momentum/AssignVariable_2/Momentum/read:02'Variable_2/Momentum/Initializer/zeros:0
x
Variable_3/Momentum:0Variable_3/Momentum/AssignVariable_3/Momentum/read:02'Variable_3/Momentum/Initializer/zeros:0
x
Variable_4/Momentum:0Variable_4/Momentum/AssignVariable_4/Momentum/read:02'Variable_4/Momentum/Initializer/zeros:0
x
Variable_5/Momentum:0Variable_5/Momentum/AssignVariable_5/Momentum/read:02'Variable_5/Momentum/Initializer/zeros:0
x
Variable_6/Momentum:0Variable_6/Momentum/AssignVariable_6/Momentum/read:02'Variable_6/Momentum/Initializer/zeros:0
x
Variable_7/Momentum:0Variable_7/Momentum/AssignVariable_7/Momentum/read:02'Variable_7/Momentum/Initializer/zeros:0"�
	summaries�
�
summary_data_0:0
summary_conv_0:0
summary_pool_0:0
summary_conv2_0:0
summary_pool2_0:0
loss:0
conv1_weights:0
conv1_biases:0
conv2_weights:0
conv2_biases:0
fc1_weights:0
fc1_biases:0
fc2_weights:0
fc2_biases:0
learning_rate:0"
train_op


Momentum�ݒč      x�Q	SA���A*�
�
summary_data_0/image/0"�"��PNG

   IHDR          :���   �IDAT�]��N�@ ���?�Ҧm�@�X#�Q�b\}'_�w�����Q�M�� �XT8��R�=7��\$��]��Io�6����A�#9 7(�@���j=�O�o��:��rDMr6c'����s�E��U����n�l�9�U;U3+ax���������EV6*d���eR~�R�wH@U7o.�B�t���gU����d<e��~q+���S��19U�P��#�!>�bޑ�[Y*��,e�D0Bj��j�h n� .�x&f�g\�    IEND�B`�
�
summary_conv_0/image/0"�"��PNG

   IHDR          :���  IDAT���V������������; �	(� +�-���)�������2������2���+��	�2����*������#��� �����
����������"����-�,�������������E������=!�3��!�)
.un ��C�6=#.������!���������w��#���L9����#����nK�    IEND�B`�
�
summary_pool_0/image/0"�"��PNG

   IHDR          �d�W   SIDAT�H ��;*��%�M=
!��.�����/ )������%#��	(OM�C/�A4����ȃ��    IEND�B`�
�
summary_conv2_0/image/0"�"��PNG

   IHDR          �d�W   SIDAT�H ��X-:�	�+���H�������9W�D������������ ~��2�8��{�<x�z-    IEND�B`�
x
summary_pool2_0/image/0"]"U�PNG

   IHDR          ����   IDAT�c,1c���)�q�=nm�lk <bk=���    IEND�B`�

lossR?

conv1_weightsx!A

conv1_biases;ֈ@

conv2_weights|A

conv2_biases�vH@

fc1_weightsM��A


fc1_biasesW��?

fc2_weights���A


fc2_biases�*>?

learning_rate
�#<�3 w