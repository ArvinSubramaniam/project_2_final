       �K"	  @갍�Abrain.Event:2��]T     ��S	��g갍�A"Ш
l
PlaceholderPlaceholder*
shape:*&
_output_shapes
:*
dtype0
^
Placeholder_1Placeholder*
shape
:*
_output_shapes

:*
dtype0
r
Placeholder_2Placeholder*
shape:��=*(
_output_shapes
:��=*
dtype0
o
truncated_normal/shapeConst*
_output_shapes
:*%
valueB"             *
dtype0
Z
truncated_normal/meanConst*
_output_shapes
: *
valueB
 *    *
dtype0
\
truncated_normal/stddevConst*
_output_shapes
: *
valueB
 *���=*
dtype0
�
 truncated_normal/TruncatedNormalTruncatedNormaltruncated_normal/shape*
seed2��*&
_output_shapes
: *
T0*
seed���)*
dtype0
�
truncated_normal/mulMul truncated_normal/TruncatedNormaltruncated_normal/stddev*
T0*&
_output_shapes
: 
u
truncated_normalAddtruncated_normal/multruncated_normal/mean*
T0*&
_output_shapes
: 
�
Variable
VariableV2*
shape: *
dtype0*
shared_name *&
_output_shapes
: *
	container 
�
Variable/AssignAssignVariabletruncated_normal*
validate_shape(*
_class
loc:@Variable*
use_locking(*
T0*&
_output_shapes
: 
q
Variable/readIdentityVariable*
_class
loc:@Variable*
T0*&
_output_shapes
: 
R
zerosConst*
_output_shapes
: *
valueB *    *
dtype0
v

Variable_1
VariableV2*
shape: *
dtype0*
shared_name *
_output_shapes
: *
	container 
�
Variable_1/AssignAssign
Variable_1zeros*
validate_shape(*
_class
loc:@Variable_1*
use_locking(*
T0*
_output_shapes
: 
k
Variable_1/readIdentity
Variable_1*
_class
loc:@Variable_1*
T0*
_output_shapes
: 
q
truncated_normal_1/shapeConst*
_output_shapes
:*%
valueB"          @   *
dtype0
\
truncated_normal_1/meanConst*
_output_shapes
: *
valueB
 *    *
dtype0
^
truncated_normal_1/stddevConst*
_output_shapes
: *
valueB
 *���=*
dtype0
�
"truncated_normal_1/TruncatedNormalTruncatedNormaltruncated_normal_1/shape*
seed2��*&
_output_shapes
: @*
T0*
seed���)*
dtype0
�
truncated_normal_1/mulMul"truncated_normal_1/TruncatedNormaltruncated_normal_1/stddev*
T0*&
_output_shapes
: @
{
truncated_normal_1Addtruncated_normal_1/multruncated_normal_1/mean*
T0*&
_output_shapes
: @
�

Variable_2
VariableV2*
shape: @*
dtype0*
shared_name *&
_output_shapes
: @*
	container 
�
Variable_2/AssignAssign
Variable_2truncated_normal_1*
validate_shape(*
_class
loc:@Variable_2*
use_locking(*
T0*&
_output_shapes
: @
w
Variable_2/readIdentity
Variable_2*
_class
loc:@Variable_2*
T0*&
_output_shapes
: @
R
ConstConst*
_output_shapes
:@*
valueB@*���=*
dtype0
v

Variable_3
VariableV2*
shape:@*
dtype0*
shared_name *
_output_shapes
:@*
	container 
�
Variable_3/AssignAssign
Variable_3Const*
validate_shape(*
_class
loc:@Variable_3*
use_locking(*
T0*
_output_shapes
:@
k
Variable_3/readIdentity
Variable_3*
_class
loc:@Variable_3*
T0*
_output_shapes
:@
i
truncated_normal_2/shapeConst*
_output_shapes
:*
valueB"      *
dtype0
\
truncated_normal_2/meanConst*
_output_shapes
: *
valueB
 *    *
dtype0
^
truncated_normal_2/stddevConst*
_output_shapes
: *
valueB
 *���=*
dtype0
�
"truncated_normal_2/TruncatedNormalTruncatedNormaltruncated_normal_2/shape*
seed2��* 
_output_shapes
:
��*
T0*
seed���)*
dtype0
�
truncated_normal_2/mulMul"truncated_normal_2/TruncatedNormaltruncated_normal_2/stddev*
T0* 
_output_shapes
:
��
u
truncated_normal_2Addtruncated_normal_2/multruncated_normal_2/mean*
T0* 
_output_shapes
:
��
�

Variable_4
VariableV2*
shape:
��*
dtype0*
shared_name * 
_output_shapes
:
��*
	container 
�
Variable_4/AssignAssign
Variable_4truncated_normal_2*
validate_shape(*
_class
loc:@Variable_4*
use_locking(*
T0* 
_output_shapes
:
��
q
Variable_4/readIdentity
Variable_4*
_class
loc:@Variable_4*
T0* 
_output_shapes
:
��
V
Const_1Const*
_output_shapes	
:�*
valueB�*���=*
dtype0
x

Variable_5
VariableV2*
shape:�*
dtype0*
shared_name *
_output_shapes	
:�*
	container 
�
Variable_5/AssignAssign
Variable_5Const_1*
validate_shape(*
_class
loc:@Variable_5*
use_locking(*
T0*
_output_shapes	
:�
l
Variable_5/readIdentity
Variable_5*
_class
loc:@Variable_5*
T0*
_output_shapes	
:�
i
truncated_normal_3/shapeConst*
_output_shapes
:*
valueB"      *
dtype0
\
truncated_normal_3/meanConst*
_output_shapes
: *
valueB
 *    *
dtype0
^
truncated_normal_3/stddevConst*
_output_shapes
: *
valueB
 *���=*
dtype0
�
"truncated_normal_3/TruncatedNormalTruncatedNormaltruncated_normal_3/shape*
seed2��*
_output_shapes
:	�*
T0*
seed���)*
dtype0
�
truncated_normal_3/mulMul"truncated_normal_3/TruncatedNormaltruncated_normal_3/stddev*
T0*
_output_shapes
:	�
t
truncated_normal_3Addtruncated_normal_3/multruncated_normal_3/mean*
T0*
_output_shapes
:	�
�

Variable_6
VariableV2*
shape:	�*
dtype0*
shared_name *
_output_shapes
:	�*
	container 
�
Variable_6/AssignAssign
Variable_6truncated_normal_3*
validate_shape(*
_class
loc:@Variable_6*
use_locking(*
T0*
_output_shapes
:	�
p
Variable_6/readIdentity
Variable_6*
_class
loc:@Variable_6*
T0*
_output_shapes
:	�
T
Const_2Const*
_output_shapes
:*
valueB*���=*
dtype0
v

Variable_7
VariableV2*
shape:*
dtype0*
shared_name *
_output_shapes
:*
	container 
�
Variable_7/AssignAssign
Variable_7Const_2*
validate_shape(*
_class
loc:@Variable_7*
use_locking(*
T0*
_output_shapes
:
k
Variable_7/readIdentity
Variable_7*
_class
loc:@Variable_7*
T0*
_output_shapes
:
�
Conv2DConv2DPlaceholderVariable/read*
data_formatNHWC*
T0*
strides
*
paddingSAME*
use_cudnn_on_gpu(*&
_output_shapes
: 
s
BiasAddBiasAddConv2DVariable_1/read*
data_formatNHWC*
T0*&
_output_shapes
: 
F
ReluReluBiasAdd*
T0*&
_output_shapes
: 
�
MaxPoolMaxPoolRelu*
data_formatNHWC*
strides
*
T0*
ksize
*
paddingSAME*&
_output_shapes
: 
�
Conv2D_1Conv2DMaxPoolVariable_2/read*
data_formatNHWC*
T0*
strides
*
paddingSAME*
use_cudnn_on_gpu(*&
_output_shapes
:@
w
	BiasAdd_1BiasAddConv2D_1Variable_3/read*
data_formatNHWC*
T0*&
_output_shapes
:@
J
Relu_1Relu	BiasAdd_1*
T0*&
_output_shapes
:@
�
	MaxPool_1MaxPoolRelu_1*
data_formatNHWC*
strides
*
T0*
ksize
*
paddingSAME*&
_output_shapes
:@
^
Reshape/shapeConst*
_output_shapes
:*
valueB"      *
dtype0
d
ReshapeReshape	MaxPool_1Reshape/shape*
Tshape0*
T0*
_output_shapes
:	�
z
MatMulMatMulReshapeVariable_4/read*
transpose_a( *
transpose_b( *
T0*
_output_shapes
:	�
M
addAddMatMulVariable_5/read*
T0*
_output_shapes
:	�
=
Relu_2Reluadd*
T0*
_output_shapes
:	�
z
MatMul_1MatMulRelu_2Variable_6/read*
transpose_a( *
transpose_b( *
T0*
_output_shapes

:
P
add_1AddMatMul_1Variable_7/read*
T0*
_output_shapes

:
d
Slice/beginConst*
_output_shapes
:*%
valueB"                *
dtype0
c

Slice/sizeConst*
_output_shapes
:*%
valueB"   ��������   *
dtype0
r
SliceSlicePlaceholderSlice/begin
Slice/size*
Index0*
T0*&
_output_shapes
:
`
Const_3Const*
_output_shapes
:*%
valueB"             *
dtype0
X
MinMinSliceConst_3*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
G
subSubSliceMin*
T0*&
_output_shapes
:
`
Const_4Const*
_output_shapes
:*%
valueB"             *
dtype0
V
MaxMaxsubConst_4*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
J
mul/yConst*
_output_shapes
: *
valueB
 *  C*
dtype0
7
mulMulMaxmul/y*
T0*
_output_shapes
: 
M
truedivRealDivsubmul*
T0*&
_output_shapes
:
d
Reshape_1/shapeConst*
_output_shapes
:*!
valueB"         *
dtype0
i
	Reshape_1ReshapetruedivReshape_1/shape*
Tshape0*
T0*"
_output_shapes
:
c
transpose/permConst*
_output_shapes
:*!
valueB"          *
dtype0
k
	transpose	Transpose	Reshape_1transpose/perm*"
_output_shapes
:*
T0*
Tperm0
h
Reshape_2/shapeConst*
_output_shapes
:*%
valueB"����         *
dtype0
o
	Reshape_2Reshape	transposeReshape_2/shape*
Tshape0*
T0*&
_output_shapes
:
a
summary_data_0/tagConst*
_output_shapes
: *
valueB Bsummary_data_0*
dtype0
�
summary_data_0ImageSummarysummary_data_0/tag	Reshape_2*

max_images*
T0*
	bad_colorB:�  �*
_output_shapes
: 
f
Slice_1/beginConst*
_output_shapes
:*%
valueB"                *
dtype0
e
Slice_1/sizeConst*
_output_shapes
:*%
valueB"   ��������   *
dtype0
s
Slice_1SliceConv2DSlice_1/beginSlice_1/size*
Index0*
T0*&
_output_shapes
:
`
Const_5Const*
_output_shapes
:*%
valueB"             *
dtype0
\
Min_1MinSlice_1Const_5*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
M
sub_1SubSlice_1Min_1*
T0*&
_output_shapes
:
`
Const_6Const*
_output_shapes
:*%
valueB"             *
dtype0
Z
Max_1Maxsub_1Const_6*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
L
mul_1/yConst*
_output_shapes
: *
valueB
 *  C*
dtype0
=
mul_1MulMax_1mul_1/y*
T0*
_output_shapes
: 
S
	truediv_1RealDivsub_1mul_1*
T0*&
_output_shapes
:
d
Reshape_3/shapeConst*
_output_shapes
:*!
valueB"         *
dtype0
k
	Reshape_3Reshape	truediv_1Reshape_3/shape*
Tshape0*
T0*"
_output_shapes
:
e
transpose_1/permConst*
_output_shapes
:*!
valueB"          *
dtype0
o
transpose_1	Transpose	Reshape_3transpose_1/perm*"
_output_shapes
:*
T0*
Tperm0
h
Reshape_4/shapeConst*
_output_shapes
:*%
valueB"����         *
dtype0
q
	Reshape_4Reshapetranspose_1Reshape_4/shape*
Tshape0*
T0*&
_output_shapes
:
a
summary_conv_0/tagConst*
_output_shapes
: *
valueB Bsummary_conv_0*
dtype0
�
summary_conv_0ImageSummarysummary_conv_0/tag	Reshape_4*

max_images*
T0*
	bad_colorB:�  �*
_output_shapes
: 
f
Slice_2/beginConst*
_output_shapes
:*%
valueB"                *
dtype0
e
Slice_2/sizeConst*
_output_shapes
:*%
valueB"   ��������   *
dtype0
t
Slice_2SliceMaxPoolSlice_2/beginSlice_2/size*
Index0*
T0*&
_output_shapes
:
`
Const_7Const*
_output_shapes
:*%
valueB"             *
dtype0
\
Min_2MinSlice_2Const_7*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
M
sub_2SubSlice_2Min_2*
T0*&
_output_shapes
:
`
Const_8Const*
_output_shapes
:*%
valueB"             *
dtype0
Z
Max_2Maxsub_2Const_8*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
L
mul_2/yConst*
_output_shapes
: *
valueB
 *  C*
dtype0
=
mul_2MulMax_2mul_2/y*
T0*
_output_shapes
: 
S
	truediv_2RealDivsub_2mul_2*
T0*&
_output_shapes
:
d
Reshape_5/shapeConst*
_output_shapes
:*!
valueB"         *
dtype0
k
	Reshape_5Reshape	truediv_2Reshape_5/shape*
Tshape0*
T0*"
_output_shapes
:
e
transpose_2/permConst*
_output_shapes
:*!
valueB"          *
dtype0
o
transpose_2	Transpose	Reshape_5transpose_2/perm*"
_output_shapes
:*
T0*
Tperm0
h
Reshape_6/shapeConst*
_output_shapes
:*%
valueB"����         *
dtype0
q
	Reshape_6Reshapetranspose_2Reshape_6/shape*
Tshape0*
T0*&
_output_shapes
:
a
summary_pool_0/tagConst*
_output_shapes
: *
valueB Bsummary_pool_0*
dtype0
�
summary_pool_0ImageSummarysummary_pool_0/tag	Reshape_6*

max_images*
T0*
	bad_colorB:�  �*
_output_shapes
: 
f
Slice_3/beginConst*
_output_shapes
:*%
valueB"                *
dtype0
e
Slice_3/sizeConst*
_output_shapes
:*%
valueB"   ��������   *
dtype0
u
Slice_3SliceConv2D_1Slice_3/beginSlice_3/size*
Index0*
T0*&
_output_shapes
:
`
Const_9Const*
_output_shapes
:*%
valueB"             *
dtype0
\
Min_3MinSlice_3Const_9*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
M
sub_3SubSlice_3Min_3*
T0*&
_output_shapes
:
a
Const_10Const*
_output_shapes
:*%
valueB"             *
dtype0
[
Max_3Maxsub_3Const_10*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
L
mul_3/yConst*
_output_shapes
: *
valueB
 *  C*
dtype0
=
mul_3MulMax_3mul_3/y*
T0*
_output_shapes
: 
S
	truediv_3RealDivsub_3mul_3*
T0*&
_output_shapes
:
d
Reshape_7/shapeConst*
_output_shapes
:*!
valueB"         *
dtype0
k
	Reshape_7Reshape	truediv_3Reshape_7/shape*
Tshape0*
T0*"
_output_shapes
:
e
transpose_3/permConst*
_output_shapes
:*!
valueB"          *
dtype0
o
transpose_3	Transpose	Reshape_7transpose_3/perm*"
_output_shapes
:*
T0*
Tperm0
h
Reshape_8/shapeConst*
_output_shapes
:*%
valueB"����         *
dtype0
q
	Reshape_8Reshapetranspose_3Reshape_8/shape*
Tshape0*
T0*&
_output_shapes
:
c
summary_conv2_0/tagConst*
_output_shapes
: * 
valueB Bsummary_conv2_0*
dtype0
�
summary_conv2_0ImageSummarysummary_conv2_0/tag	Reshape_8*

max_images*
T0*
	bad_colorB:�  �*
_output_shapes
: 
f
Slice_4/beginConst*
_output_shapes
:*%
valueB"                *
dtype0
e
Slice_4/sizeConst*
_output_shapes
:*%
valueB"   ��������   *
dtype0
v
Slice_4Slice	MaxPool_1Slice_4/beginSlice_4/size*
Index0*
T0*&
_output_shapes
:
a
Const_11Const*
_output_shapes
:*%
valueB"             *
dtype0
]
Min_4MinSlice_4Const_11*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
M
sub_4SubSlice_4Min_4*
T0*&
_output_shapes
:
a
Const_12Const*
_output_shapes
:*%
valueB"             *
dtype0
[
Max_4Maxsub_4Const_12*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
L
mul_4/yConst*
_output_shapes
: *
valueB
 *  C*
dtype0
=
mul_4MulMax_4mul_4/y*
T0*
_output_shapes
: 
S
	truediv_4RealDivsub_4mul_4*
T0*&
_output_shapes
:
d
Reshape_9/shapeConst*
_output_shapes
:*!
valueB"         *
dtype0
k
	Reshape_9Reshape	truediv_4Reshape_9/shape*
Tshape0*
T0*"
_output_shapes
:
e
transpose_4/permConst*
_output_shapes
:*!
valueB"          *
dtype0
o
transpose_4	Transpose	Reshape_9transpose_4/perm*"
_output_shapes
:*
T0*
Tperm0
i
Reshape_10/shapeConst*
_output_shapes
:*%
valueB"����         *
dtype0
s

Reshape_10Reshapetranspose_4Reshape_10/shape*
Tshape0*
T0*&
_output_shapes
:
c
summary_pool2_0/tagConst*
_output_shapes
: * 
valueB Bsummary_pool2_0*
dtype0
�
summary_pool2_0ImageSummarysummary_pool2_0/tag
Reshape_10*

max_images*
T0*
	bad_colorB:�  �*
_output_shapes
: 
F
RankConst*
_output_shapes
: *
value	B :*
dtype0
V
ShapeConst*
_output_shapes
:*
valueB"      *
dtype0
H
Rank_1Const*
_output_shapes
: *
value	B :*
dtype0
X
Shape_1Const*
_output_shapes
:*
valueB"      *
dtype0
G
Sub/yConst*
_output_shapes
: *
value	B :*
dtype0
:
SubSubRank_1Sub/y*
T0*
_output_shapes
: 
T
Slice_5/beginPackSub*
N*

axis *
T0*
_output_shapes
:
V
Slice_5/sizeConst*
_output_shapes
:*
valueB:*
dtype0
h
Slice_5SliceShape_1Slice_5/beginSlice_5/size*
Index0*
T0*
_output_shapes
:
b
concat/values_0Const*
_output_shapes
:*
valueB:
���������*
dtype0
M
concat/axisConst*
_output_shapes
: *
value	B : *
dtype0
s
concatConcatV2concat/values_0Slice_5concat/axis*
N*

Tidx0*
T0*
_output_shapes
:
[

Reshape_11Reshapeadd_1concat*
Tshape0*
T0*
_output_shapes

:
H
Rank_2Const*
_output_shapes
: *
value	B :*
dtype0
X
Shape_2Const*
_output_shapes
:*
valueB"      *
dtype0
I
Sub_1/yConst*
_output_shapes
: *
value	B :*
dtype0
>
Sub_1SubRank_2Sub_1/y*
T0*
_output_shapes
: 
V
Slice_6/beginPackSub_1*
N*

axis *
T0*
_output_shapes
:
V
Slice_6/sizeConst*
_output_shapes
:*
valueB:*
dtype0
h
Slice_6SliceShape_2Slice_6/beginSlice_6/size*
Index0*
T0*
_output_shapes
:
d
concat_1/values_0Const*
_output_shapes
:*
valueB:
���������*
dtype0
O
concat_1/axisConst*
_output_shapes
: *
value	B : *
dtype0
y
concat_1ConcatV2concat_1/values_0Slice_6concat_1/axis*
N*

Tidx0*
T0*
_output_shapes
:
e

Reshape_12ReshapePlaceholder_1concat_1*
Tshape0*
T0*
_output_shapes

:
�
SoftmaxCrossEntropyWithLogitsSoftmaxCrossEntropyWithLogits
Reshape_11
Reshape_12*
T0*$
_output_shapes
::
I
Sub_2/yConst*
_output_shapes
: *
value	B :*
dtype0
<
Sub_2SubRankSub_2/y*
T0*
_output_shapes
: 
W
Slice_7/beginConst*
_output_shapes
:*
valueB: *
dtype0
U
Slice_7/sizePackSub_2*
N*

axis *
T0*
_output_shapes
:
o
Slice_7SliceShapeSlice_7/beginSlice_7/size*
Index0*
T0*#
_output_shapes
:���������
p

Reshape_13ReshapeSoftmaxCrossEntropyWithLogitsSlice_7*
Tshape0*
T0*
_output_shapes
:
R
Const_13Const*
_output_shapes
:*
valueB: *
dtype0
`
MeanMean
Reshape_13Const_13*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
N
	loss/tagsConst*
_output_shapes
: *
valueB
 Bloss*
dtype0
G
lossScalarSummary	loss/tagsMean*
T0*
_output_shapes
: 
R
gradients/ShapeConst*
_output_shapes
: *
valueB *
dtype0
T
gradients/ConstConst*
_output_shapes
: *
valueB
 *  �?*
dtype0
Y
gradients/FillFillgradients/Shapegradients/Const*
T0*
_output_shapes
: 
k
!gradients/Mean_grad/Reshape/shapeConst*
_output_shapes
:*
valueB:*
dtype0
�
gradients/Mean_grad/ReshapeReshapegradients/Fill!gradients/Mean_grad/Reshape/shape*
Tshape0*
T0*
_output_shapes
:
l
"gradients/Mean_grad/Tile/multiplesConst*
_output_shapes
:*
valueB:*
dtype0
�
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshape"gradients/Mean_grad/Tile/multiples*

Tmultiples0*
T0*
_output_shapes
:
c
gradients/Mean_grad/ShapeConst*
_output_shapes
:*
valueB:*
dtype0
^
gradients/Mean_grad/Shape_1Const*
_output_shapes
: *
valueB *
dtype0
�
gradients/Mean_grad/ConstConst*,
_class"
 loc:@gradients/Mean_grad/Shape*
_output_shapes
:*
valueB: *
dtype0
�
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shapegradients/Mean_grad/Const*,
_class"
 loc:@gradients/Mean_grad/Shape*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
�
gradients/Mean_grad/Const_1Const*,
_class"
 loc:@gradients/Mean_grad/Shape*
_output_shapes
:*
valueB: *
dtype0
�
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_1gradients/Mean_grad/Const_1*,
_class"
 loc:@gradients/Mean_grad/Shape*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
�
gradients/Mean_grad/Maximum/yConst*,
_class"
 loc:@gradients/Mean_grad/Shape*
_output_shapes
: *
value	B :*
dtype0
�
gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*,
_class"
 loc:@gradients/Mean_grad/Shape*
T0*
_output_shapes
: 
�
gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum*,
_class"
 loc:@gradients/Mean_grad/Shape*
T0*
_output_shapes
: 
n
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv*

SrcT0*

DstT0*
_output_shapes
: 

gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*
T0*
_output_shapes
:
i
gradients/Reshape_13_grad/ShapeConst*
_output_shapes
:*
valueB:*
dtype0
�
!gradients/Reshape_13_grad/ReshapeReshapegradients/Mean_grad/truedivgradients/Reshape_13_grad/Shape*
Tshape0*
T0*
_output_shapes
:
k
gradients/zeros_like	ZerosLikeSoftmaxCrossEntropyWithLogits:1*
T0*
_output_shapes

:
�
;gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dimConst*
_output_shapes
: *
valueB :
���������*
dtype0
�
7gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims
ExpandDims!gradients/Reshape_13_grad/Reshape;gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim*
T0*

Tdim0*
_output_shapes

:
�
0gradients/SoftmaxCrossEntropyWithLogits_grad/mulMul7gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDimsSoftmaxCrossEntropyWithLogits:1*
T0*
_output_shapes

:
p
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
valueB"      *
dtype0
�
!gradients/Reshape_11_grad/ReshapeReshape0gradients/SoftmaxCrossEntropyWithLogits_grad/mulgradients/Reshape_11_grad/Shape*
Tshape0*
T0*
_output_shapes

:
k
gradients/add_1_grad/ShapeConst*
_output_shapes
:*
valueB"      *
dtype0
f
gradients/add_1_grad/Shape_1Const*
_output_shapes
:*
valueB:*
dtype0
�
*gradients/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_1_grad/Shapegradients/add_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/add_1_grad/SumSum!gradients/Reshape_11_grad/Reshape*gradients/add_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
gradients/add_1_grad/ReshapeReshapegradients/add_1_grad/Sumgradients/add_1_grad/Shape*
Tshape0*
T0*
_output_shapes

:
�
gradients/add_1_grad/Sum_1Sum!gradients/Reshape_11_grad/Reshape,gradients/add_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
gradients/add_1_grad/Reshape_1Reshapegradients/add_1_grad/Sum_1gradients/add_1_grad/Shape_1*
Tshape0*
T0*
_output_shapes
:
�
gradients/MatMul_1_grad/MatMulMatMulgradients/add_1_grad/ReshapeVariable_6/read*
transpose_a( *
transpose_b(*
T0*
_output_shapes
:	�
�
 gradients/MatMul_1_grad/MatMul_1MatMulRelu_2gradients/add_1_grad/Reshape*
transpose_a(*
transpose_b( *
T0*
_output_shapes
:	�
|
gradients/Relu_2_grad/ReluGradReluGradgradients/MatMul_1_grad/MatMulRelu_2*
T0*
_output_shapes
:	�
i
gradients/add_grad/ShapeConst*
_output_shapes
:*
valueB"      *
dtype0
e
gradients/add_grad/Shape_1Const*
_output_shapes
:*
valueB:�*
dtype0
�
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/add_grad/SumSumgradients/Relu_2_grad/ReluGrad(gradients/add_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*
Tshape0*
T0*
_output_shapes
:	�
�
gradients/add_grad/Sum_1Sumgradients/Relu_2_grad/ReluGrad*gradients/add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
Tshape0*
T0*
_output_shapes	
:�
�
gradients/MatMul_grad/MatMulMatMulgradients/add_grad/ReshapeVariable_4/read*
transpose_a( *
transpose_b(*
T0*
_output_shapes
:	�
�
gradients/MatMul_grad/MatMul_1MatMulReshapegradients/add_grad/Reshape*
transpose_a(*
transpose_b( *
T0* 
_output_shapes
:
��
u
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*%
valueB"         @   *
dtype0
�
gradients/Reshape_grad/ReshapeReshapegradients/MatMul_grad/MatMulgradients/Reshape_grad/Shape*
Tshape0*
T0*&
_output_shapes
:@
�
$gradients/MaxPool_1_grad/MaxPoolGradMaxPoolGradRelu_1	MaxPool_1gradients/Reshape_grad/Reshape*
data_formatNHWC*
T0*
strides
*
ksize
*
paddingSAME*&
_output_shapes
:@
�
gradients/Relu_1_grad/ReluGradReluGrad$gradients/MaxPool_1_grad/MaxPoolGradRelu_1*
T0*&
_output_shapes
:@
�
$gradients/BiasAdd_1_grad/BiasAddGradBiasAddGradgradients/Relu_1_grad/ReluGrad*
data_formatNHWC*
T0*
_output_shapes
:@
�
gradients/Conv2D_1_grad/ShapeNShapeNMaxPoolVariable_2/read*
N*
out_type0*
T0* 
_output_shapes
::
�
+gradients/Conv2D_1_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_1_grad/ShapeNVariable_2/readgradients/Relu_1_grad/ReluGrad*
data_formatNHWC*
T0*
strides
*
paddingSAME*
use_cudnn_on_gpu(*&
_output_shapes
: 
�
,gradients/Conv2D_1_grad/Conv2DBackpropFilterConv2DBackpropFilterMaxPool gradients/Conv2D_1_grad/ShapeN:1gradients/Relu_1_grad/ReluGrad*
data_formatNHWC*
T0*
strides
*
paddingSAME*
use_cudnn_on_gpu(*&
_output_shapes
: @
�
"gradients/MaxPool_grad/MaxPoolGradMaxPoolGradReluMaxPool+gradients/Conv2D_1_grad/Conv2DBackpropInput*
data_formatNHWC*
T0*
strides
*
ksize
*
paddingSAME*&
_output_shapes
: 
�
gradients/Relu_grad/ReluGradReluGrad"gradients/MaxPool_grad/MaxPoolGradRelu*
T0*&
_output_shapes
: 
�
"gradients/BiasAdd_grad/BiasAddGradBiasAddGradgradients/Relu_grad/ReluGrad*
data_formatNHWC*
T0*
_output_shapes
: 
�
gradients/Conv2D_grad/ShapeNShapeNPlaceholderVariable/read*
N*
out_type0*
T0* 
_output_shapes
::
�
)gradients/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_grad/ShapeNVariable/readgradients/Relu_grad/ReluGrad*
data_formatNHWC*
T0*
strides
*
paddingSAME*
use_cudnn_on_gpu(*&
_output_shapes
:
�
*gradients/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterPlaceholdergradients/Conv2D_grad/ShapeN:1gradients/Relu_grad/ReluGrad*
data_formatNHWC*
T0*
strides
*
paddingSAME*
use_cudnn_on_gpu(*&
_output_shapes
: 
�
global_norm/L2LossL2Loss*gradients/Conv2D_grad/Conv2DBackpropFilter*=
_class3
1/loc:@gradients/Conv2D_grad/Conv2DBackpropFilter*
T0*
_output_shapes
: 
g
global_norm/stackPackglobal_norm/L2Loss*
N*

axis *
T0*
_output_shapes
:
[
global_norm/ConstConst*
_output_shapes
:*
valueB: *
dtype0
z
global_norm/SumSumglobal_norm/stackglobal_norm/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
X
global_norm/Const_1Const*
_output_shapes
: *
valueB
 *   @*
dtype0
]
global_norm/mulMulglobal_norm/Sumglobal_norm/Const_1*
T0*
_output_shapes
: 
Q
global_norm/global_normSqrtglobal_norm/mul*
T0*
_output_shapes
: 
`
conv1_weights/tagsConst*
_output_shapes
: *
valueB Bconv1_weights*
dtype0
l
conv1_weightsScalarSummaryconv1_weights/tagsglobal_norm/global_norm*
T0*
_output_shapes
: 
�
global_norm_1/L2LossL2Loss"gradients/BiasAdd_grad/BiasAddGrad*5
_class+
)'loc:@gradients/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
: 
k
global_norm_1/stackPackglobal_norm_1/L2Loss*
N*

axis *
T0*
_output_shapes
:
]
global_norm_1/ConstConst*
_output_shapes
:*
valueB: *
dtype0
�
global_norm_1/SumSumglobal_norm_1/stackglobal_norm_1/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
Z
global_norm_1/Const_1Const*
_output_shapes
: *
valueB
 *   @*
dtype0
c
global_norm_1/mulMulglobal_norm_1/Sumglobal_norm_1/Const_1*
T0*
_output_shapes
: 
U
global_norm_1/global_normSqrtglobal_norm_1/mul*
T0*
_output_shapes
: 
^
conv1_biases/tagsConst*
_output_shapes
: *
valueB Bconv1_biases*
dtype0
l
conv1_biasesScalarSummaryconv1_biases/tagsglobal_norm_1/global_norm*
T0*
_output_shapes
: 
�
global_norm_2/L2LossL2Loss,gradients/Conv2D_1_grad/Conv2DBackpropFilter*?
_class5
31loc:@gradients/Conv2D_1_grad/Conv2DBackpropFilter*
T0*
_output_shapes
: 
k
global_norm_2/stackPackglobal_norm_2/L2Loss*
N*

axis *
T0*
_output_shapes
:
]
global_norm_2/ConstConst*
_output_shapes
:*
valueB: *
dtype0
�
global_norm_2/SumSumglobal_norm_2/stackglobal_norm_2/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
Z
global_norm_2/Const_1Const*
_output_shapes
: *
valueB
 *   @*
dtype0
c
global_norm_2/mulMulglobal_norm_2/Sumglobal_norm_2/Const_1*
T0*
_output_shapes
: 
U
global_norm_2/global_normSqrtglobal_norm_2/mul*
T0*
_output_shapes
: 
`
conv2_weights/tagsConst*
_output_shapes
: *
valueB Bconv2_weights*
dtype0
n
conv2_weightsScalarSummaryconv2_weights/tagsglobal_norm_2/global_norm*
T0*
_output_shapes
: 
�
global_norm_3/L2LossL2Loss$gradients/BiasAdd_1_grad/BiasAddGrad*7
_class-
+)loc:@gradients/BiasAdd_1_grad/BiasAddGrad*
T0*
_output_shapes
: 
k
global_norm_3/stackPackglobal_norm_3/L2Loss*
N*

axis *
T0*
_output_shapes
:
]
global_norm_3/ConstConst*
_output_shapes
:*
valueB: *
dtype0
�
global_norm_3/SumSumglobal_norm_3/stackglobal_norm_3/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
Z
global_norm_3/Const_1Const*
_output_shapes
: *
valueB
 *   @*
dtype0
c
global_norm_3/mulMulglobal_norm_3/Sumglobal_norm_3/Const_1*
T0*
_output_shapes
: 
U
global_norm_3/global_normSqrtglobal_norm_3/mul*
T0*
_output_shapes
: 
^
conv2_biases/tagsConst*
_output_shapes
: *
valueB Bconv2_biases*
dtype0
l
conv2_biasesScalarSummaryconv2_biases/tagsglobal_norm_3/global_norm*
T0*
_output_shapes
: 
�
global_norm_4/L2LossL2Lossgradients/MatMul_grad/MatMul_1*1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1*
T0*
_output_shapes
: 
k
global_norm_4/stackPackglobal_norm_4/L2Loss*
N*

axis *
T0*
_output_shapes
:
]
global_norm_4/ConstConst*
_output_shapes
:*
valueB: *
dtype0
�
global_norm_4/SumSumglobal_norm_4/stackglobal_norm_4/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
Z
global_norm_4/Const_1Const*
_output_shapes
: *
valueB
 *   @*
dtype0
c
global_norm_4/mulMulglobal_norm_4/Sumglobal_norm_4/Const_1*
T0*
_output_shapes
: 
U
global_norm_4/global_normSqrtglobal_norm_4/mul*
T0*
_output_shapes
: 
\
fc1_weights/tagsConst*
_output_shapes
: *
valueB Bfc1_weights*
dtype0
j
fc1_weightsScalarSummaryfc1_weights/tagsglobal_norm_4/global_norm*
T0*
_output_shapes
: 
�
global_norm_5/L2LossL2Lossgradients/add_grad/Reshape_1*/
_class%
#!loc:@gradients/add_grad/Reshape_1*
T0*
_output_shapes
: 
k
global_norm_5/stackPackglobal_norm_5/L2Loss*
N*

axis *
T0*
_output_shapes
:
]
global_norm_5/ConstConst*
_output_shapes
:*
valueB: *
dtype0
�
global_norm_5/SumSumglobal_norm_5/stackglobal_norm_5/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
Z
global_norm_5/Const_1Const*
_output_shapes
: *
valueB
 *   @*
dtype0
c
global_norm_5/mulMulglobal_norm_5/Sumglobal_norm_5/Const_1*
T0*
_output_shapes
: 
U
global_norm_5/global_normSqrtglobal_norm_5/mul*
T0*
_output_shapes
: 
Z
fc1_biases/tagsConst*
_output_shapes
: *
valueB B
fc1_biases*
dtype0
h

fc1_biasesScalarSummaryfc1_biases/tagsglobal_norm_5/global_norm*
T0*
_output_shapes
: 
�
global_norm_6/L2LossL2Loss gradients/MatMul_1_grad/MatMul_1*3
_class)
'%loc:@gradients/MatMul_1_grad/MatMul_1*
T0*
_output_shapes
: 
k
global_norm_6/stackPackglobal_norm_6/L2Loss*
N*

axis *
T0*
_output_shapes
:
]
global_norm_6/ConstConst*
_output_shapes
:*
valueB: *
dtype0
�
global_norm_6/SumSumglobal_norm_6/stackglobal_norm_6/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
Z
global_norm_6/Const_1Const*
_output_shapes
: *
valueB
 *   @*
dtype0
c
global_norm_6/mulMulglobal_norm_6/Sumglobal_norm_6/Const_1*
T0*
_output_shapes
: 
U
global_norm_6/global_normSqrtglobal_norm_6/mul*
T0*
_output_shapes
: 
\
fc2_weights/tagsConst*
_output_shapes
: *
valueB Bfc2_weights*
dtype0
j
fc2_weightsScalarSummaryfc2_weights/tagsglobal_norm_6/global_norm*
T0*
_output_shapes
: 
�
global_norm_7/L2LossL2Lossgradients/add_1_grad/Reshape_1*1
_class'
%#loc:@gradients/add_1_grad/Reshape_1*
T0*
_output_shapes
: 
k
global_norm_7/stackPackglobal_norm_7/L2Loss*
N*

axis *
T0*
_output_shapes
:
]
global_norm_7/ConstConst*
_output_shapes
:*
valueB: *
dtype0
�
global_norm_7/SumSumglobal_norm_7/stackglobal_norm_7/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
Z
global_norm_7/Const_1Const*
_output_shapes
: *
valueB
 *   @*
dtype0
c
global_norm_7/mulMulglobal_norm_7/Sumglobal_norm_7/Const_1*
T0*
_output_shapes
: 
U
global_norm_7/global_normSqrtglobal_norm_7/mul*
T0*
_output_shapes
: 
Z
fc2_biases/tagsConst*
_output_shapes
: *
valueB B
fc2_biases*
dtype0
h

fc2_biasesScalarSummaryfc2_biases/tagsglobal_norm_7/global_norm*
T0*
_output_shapes
: 
B
L2LossL2LossVariable_4/read*
T0*
_output_shapes
: 
D
L2Loss_1L2LossVariable_5/read*
T0*
_output_shapes
: 
?
add_2AddL2LossL2Loss_1*
T0*
_output_shapes
: 
D
L2Loss_2L2LossVariable_6/read*
T0*
_output_shapes
: 
>
add_3Addadd_2L2Loss_2*
T0*
_output_shapes
: 
D
L2Loss_3L2LossVariable_7/read*
T0*
_output_shapes
: 
>
add_4Addadd_3L2Loss_3*
T0*
_output_shapes
: 
L
mul_5/xConst*
_output_shapes
: *
valueB
 *o:*
dtype0
=
mul_5Mulmul_5/xadd_4*
T0*
_output_shapes
: 
:
add_5AddMeanmul_5*
T0*
_output_shapes
: 
Z
Variable_8/initial_valueConst*
_output_shapes
: *
value	B : *
dtype0
n

Variable_8
VariableV2*
shape: *
dtype0*
shared_name *
_output_shapes
: *
	container 
�
Variable_8/AssignAssign
Variable_8Variable_8/initial_value*
validate_shape(*
_class
loc:@Variable_8*
use_locking(*
T0*
_output_shapes
: 
g
Variable_8/readIdentity
Variable_8*
_class
loc:@Variable_8*
T0*
_output_shapes
: 
I
mul_6/yConst*
_output_shapes
: *
value	B :*
dtype0
G
mul_6MulVariable_8/readmul_6/y*
T0*
_output_shapes
: 
c
ExponentialDecay/learning_rateConst*
_output_shapes
: *
valueB
 *
�#<*
dtype0
T
ExponentialDecay/CastCastmul_6*

SrcT0*

DstT0*
_output_shapes
: 
]
ExponentialDecay/Cast_1/xConst*
_output_shapes
: *
valueB	 :��=*
dtype0
j
ExponentialDecay/Cast_1CastExponentialDecay/Cast_1/x*

SrcT0*

DstT0*
_output_shapes
: 
^
ExponentialDecay/Cast_2/xConst*
_output_shapes
: *
valueB
 *33s?*
dtype0
t
ExponentialDecay/truedivRealDivExponentialDecay/CastExponentialDecay/Cast_1*
T0*
_output_shapes
: 
Z
ExponentialDecay/FloorFloorExponentialDecay/truediv*
T0*
_output_shapes
: 
o
ExponentialDecay/PowPowExponentialDecay/Cast_2/xExponentialDecay/Floor*
T0*
_output_shapes
: 
n
ExponentialDecayMulExponentialDecay/learning_rateExponentialDecay/Pow*
T0*
_output_shapes
: 
`
learning_rate/tagsConst*
_output_shapes
: *
valueB Blearning_rate*
dtype0
e
learning_rateScalarSummarylearning_rate/tagsExponentialDecay*
T0*
_output_shapes
: 
T
gradients_1/ShapeConst*
_output_shapes
: *
valueB *
dtype0
V
gradients_1/ConstConst*
_output_shapes
: *
valueB
 *  �?*
dtype0
_
gradients_1/FillFillgradients_1/Shapegradients_1/Const*
T0*
_output_shapes
: 
_
gradients_1/add_5_grad/ShapeConst*
_output_shapes
: *
valueB *
dtype0
a
gradients_1/add_5_grad/Shape_1Const*
_output_shapes
: *
valueB *
dtype0
�
,gradients_1/add_5_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/add_5_grad/Shapegradients_1/add_5_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients_1/add_5_grad/SumSumgradients_1/Fill,gradients_1/add_5_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
gradients_1/add_5_grad/ReshapeReshapegradients_1/add_5_grad/Sumgradients_1/add_5_grad/Shape*
Tshape0*
T0*
_output_shapes
: 
�
gradients_1/add_5_grad/Sum_1Sumgradients_1/Fill.gradients_1/add_5_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
 gradients_1/add_5_grad/Reshape_1Reshapegradients_1/add_5_grad/Sum_1gradients_1/add_5_grad/Shape_1*
Tshape0*
T0*
_output_shapes
: 
s
'gradients_1/add_5_grad/tuple/group_depsNoOp^gradients_1/add_5_grad/Reshape!^gradients_1/add_5_grad/Reshape_1
�
/gradients_1/add_5_grad/tuple/control_dependencyIdentitygradients_1/add_5_grad/Reshape(^gradients_1/add_5_grad/tuple/group_deps*1
_class'
%#loc:@gradients_1/add_5_grad/Reshape*
T0*
_output_shapes
: 
�
1gradients_1/add_5_grad/tuple/control_dependency_1Identity gradients_1/add_5_grad/Reshape_1(^gradients_1/add_5_grad/tuple/group_deps*3
_class)
'%loc:@gradients_1/add_5_grad/Reshape_1*
T0*
_output_shapes
: 
m
#gradients_1/Mean_grad/Reshape/shapeConst*
_output_shapes
:*
valueB:*
dtype0
�
gradients_1/Mean_grad/ReshapeReshape/gradients_1/add_5_grad/tuple/control_dependency#gradients_1/Mean_grad/Reshape/shape*
Tshape0*
T0*
_output_shapes
:
n
$gradients_1/Mean_grad/Tile/multiplesConst*
_output_shapes
:*
valueB:*
dtype0
�
gradients_1/Mean_grad/TileTilegradients_1/Mean_grad/Reshape$gradients_1/Mean_grad/Tile/multiples*

Tmultiples0*
T0*
_output_shapes
:
e
gradients_1/Mean_grad/ShapeConst*
_output_shapes
:*
valueB:*
dtype0
`
gradients_1/Mean_grad/Shape_1Const*
_output_shapes
: *
valueB *
dtype0
�
gradients_1/Mean_grad/ConstConst*.
_class$
" loc:@gradients_1/Mean_grad/Shape*
_output_shapes
:*
valueB: *
dtype0
�
gradients_1/Mean_grad/ProdProdgradients_1/Mean_grad/Shapegradients_1/Mean_grad/Const*.
_class$
" loc:@gradients_1/Mean_grad/Shape*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
�
gradients_1/Mean_grad/Const_1Const*.
_class$
" loc:@gradients_1/Mean_grad/Shape*
_output_shapes
:*
valueB: *
dtype0
�
gradients_1/Mean_grad/Prod_1Prodgradients_1/Mean_grad/Shape_1gradients_1/Mean_grad/Const_1*.
_class$
" loc:@gradients_1/Mean_grad/Shape*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
�
gradients_1/Mean_grad/Maximum/yConst*.
_class$
" loc:@gradients_1/Mean_grad/Shape*
_output_shapes
: *
value	B :*
dtype0
�
gradients_1/Mean_grad/MaximumMaximumgradients_1/Mean_grad/Prod_1gradients_1/Mean_grad/Maximum/y*.
_class$
" loc:@gradients_1/Mean_grad/Shape*
T0*
_output_shapes
: 
�
gradients_1/Mean_grad/floordivFloorDivgradients_1/Mean_grad/Prodgradients_1/Mean_grad/Maximum*.
_class$
" loc:@gradients_1/Mean_grad/Shape*
T0*
_output_shapes
: 
r
gradients_1/Mean_grad/CastCastgradients_1/Mean_grad/floordiv*

SrcT0*

DstT0*
_output_shapes
: 
�
gradients_1/Mean_grad/truedivRealDivgradients_1/Mean_grad/Tilegradients_1/Mean_grad/Cast*
T0*
_output_shapes
:
_
gradients_1/mul_5_grad/ShapeConst*
_output_shapes
: *
valueB *
dtype0
a
gradients_1/mul_5_grad/Shape_1Const*
_output_shapes
: *
valueB *
dtype0
�
,gradients_1/mul_5_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/mul_5_grad/Shapegradients_1/mul_5_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
|
gradients_1/mul_5_grad/mulMul1gradients_1/add_5_grad/tuple/control_dependency_1add_4*
T0*
_output_shapes
: 
�
gradients_1/mul_5_grad/SumSumgradients_1/mul_5_grad/mul,gradients_1/mul_5_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
gradients_1/mul_5_grad/ReshapeReshapegradients_1/mul_5_grad/Sumgradients_1/mul_5_grad/Shape*
Tshape0*
T0*
_output_shapes
: 
�
gradients_1/mul_5_grad/mul_1Mulmul_5/x1gradients_1/add_5_grad/tuple/control_dependency_1*
T0*
_output_shapes
: 
�
gradients_1/mul_5_grad/Sum_1Sumgradients_1/mul_5_grad/mul_1.gradients_1/mul_5_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
 gradients_1/mul_5_grad/Reshape_1Reshapegradients_1/mul_5_grad/Sum_1gradients_1/mul_5_grad/Shape_1*
Tshape0*
T0*
_output_shapes
: 
s
'gradients_1/mul_5_grad/tuple/group_depsNoOp^gradients_1/mul_5_grad/Reshape!^gradients_1/mul_5_grad/Reshape_1
�
/gradients_1/mul_5_grad/tuple/control_dependencyIdentitygradients_1/mul_5_grad/Reshape(^gradients_1/mul_5_grad/tuple/group_deps*1
_class'
%#loc:@gradients_1/mul_5_grad/Reshape*
T0*
_output_shapes
: 
�
1gradients_1/mul_5_grad/tuple/control_dependency_1Identity gradients_1/mul_5_grad/Reshape_1(^gradients_1/mul_5_grad/tuple/group_deps*3
_class)
'%loc:@gradients_1/mul_5_grad/Reshape_1*
T0*
_output_shapes
: 
k
!gradients_1/Reshape_13_grad/ShapeConst*
_output_shapes
:*
valueB:*
dtype0
�
#gradients_1/Reshape_13_grad/ReshapeReshapegradients_1/Mean_grad/truediv!gradients_1/Reshape_13_grad/Shape*
Tshape0*
T0*
_output_shapes
:
_
gradients_1/add_4_grad/ShapeConst*
_output_shapes
: *
valueB *
dtype0
a
gradients_1/add_4_grad/Shape_1Const*
_output_shapes
: *
valueB *
dtype0
�
,gradients_1/add_4_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/add_4_grad/Shapegradients_1/add_4_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients_1/add_4_grad/SumSum1gradients_1/mul_5_grad/tuple/control_dependency_1,gradients_1/add_4_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
gradients_1/add_4_grad/ReshapeReshapegradients_1/add_4_grad/Sumgradients_1/add_4_grad/Shape*
Tshape0*
T0*
_output_shapes
: 
�
gradients_1/add_4_grad/Sum_1Sum1gradients_1/mul_5_grad/tuple/control_dependency_1.gradients_1/add_4_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
 gradients_1/add_4_grad/Reshape_1Reshapegradients_1/add_4_grad/Sum_1gradients_1/add_4_grad/Shape_1*
Tshape0*
T0*
_output_shapes
: 
s
'gradients_1/add_4_grad/tuple/group_depsNoOp^gradients_1/add_4_grad/Reshape!^gradients_1/add_4_grad/Reshape_1
�
/gradients_1/add_4_grad/tuple/control_dependencyIdentitygradients_1/add_4_grad/Reshape(^gradients_1/add_4_grad/tuple/group_deps*1
_class'
%#loc:@gradients_1/add_4_grad/Reshape*
T0*
_output_shapes
: 
�
1gradients_1/add_4_grad/tuple/control_dependency_1Identity gradients_1/add_4_grad/Reshape_1(^gradients_1/add_4_grad/tuple/group_deps*3
_class)
'%loc:@gradients_1/add_4_grad/Reshape_1*
T0*
_output_shapes
: 
m
gradients_1/zeros_like	ZerosLikeSoftmaxCrossEntropyWithLogits:1*
T0*
_output_shapes

:
�
=gradients_1/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dimConst*
_output_shapes
: *
valueB :
���������*
dtype0
�
9gradients_1/SoftmaxCrossEntropyWithLogits_grad/ExpandDims
ExpandDims#gradients_1/Reshape_13_grad/Reshape=gradients_1/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim*
T0*

Tdim0*
_output_shapes

:
�
2gradients_1/SoftmaxCrossEntropyWithLogits_grad/mulMul9gradients_1/SoftmaxCrossEntropyWithLogits_grad/ExpandDimsSoftmaxCrossEntropyWithLogits:1*
T0*
_output_shapes

:
_
gradients_1/add_3_grad/ShapeConst*
_output_shapes
: *
valueB *
dtype0
a
gradients_1/add_3_grad/Shape_1Const*
_output_shapes
: *
valueB *
dtype0
�
,gradients_1/add_3_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/add_3_grad/Shapegradients_1/add_3_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients_1/add_3_grad/SumSum/gradients_1/add_4_grad/tuple/control_dependency,gradients_1/add_3_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
gradients_1/add_3_grad/ReshapeReshapegradients_1/add_3_grad/Sumgradients_1/add_3_grad/Shape*
Tshape0*
T0*
_output_shapes
: 
�
gradients_1/add_3_grad/Sum_1Sum/gradients_1/add_4_grad/tuple/control_dependency.gradients_1/add_3_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
 gradients_1/add_3_grad/Reshape_1Reshapegradients_1/add_3_grad/Sum_1gradients_1/add_3_grad/Shape_1*
Tshape0*
T0*
_output_shapes
: 
s
'gradients_1/add_3_grad/tuple/group_depsNoOp^gradients_1/add_3_grad/Reshape!^gradients_1/add_3_grad/Reshape_1
�
/gradients_1/add_3_grad/tuple/control_dependencyIdentitygradients_1/add_3_grad/Reshape(^gradients_1/add_3_grad/tuple/group_deps*1
_class'
%#loc:@gradients_1/add_3_grad/Reshape*
T0*
_output_shapes
: 
�
1gradients_1/add_3_grad/tuple/control_dependency_1Identity gradients_1/add_3_grad/Reshape_1(^gradients_1/add_3_grad/tuple/group_deps*3
_class)
'%loc:@gradients_1/add_3_grad/Reshape_1*
T0*
_output_shapes
: 
�
gradients_1/L2Loss_3_grad/mulMulVariable_7/read1gradients_1/add_4_grad/tuple/control_dependency_1*
T0*
_output_shapes
:
r
!gradients_1/Reshape_11_grad/ShapeConst*
_output_shapes
:*
valueB"      *
dtype0
�
#gradients_1/Reshape_11_grad/ReshapeReshape2gradients_1/SoftmaxCrossEntropyWithLogits_grad/mul!gradients_1/Reshape_11_grad/Shape*
Tshape0*
T0*
_output_shapes

:
_
gradients_1/add_2_grad/ShapeConst*
_output_shapes
: *
valueB *
dtype0
a
gradients_1/add_2_grad/Shape_1Const*
_output_shapes
: *
valueB *
dtype0
�
,gradients_1/add_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/add_2_grad/Shapegradients_1/add_2_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients_1/add_2_grad/SumSum/gradients_1/add_3_grad/tuple/control_dependency,gradients_1/add_2_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
gradients_1/add_2_grad/ReshapeReshapegradients_1/add_2_grad/Sumgradients_1/add_2_grad/Shape*
Tshape0*
T0*
_output_shapes
: 
�
gradients_1/add_2_grad/Sum_1Sum/gradients_1/add_3_grad/tuple/control_dependency.gradients_1/add_2_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
 gradients_1/add_2_grad/Reshape_1Reshapegradients_1/add_2_grad/Sum_1gradients_1/add_2_grad/Shape_1*
Tshape0*
T0*
_output_shapes
: 
s
'gradients_1/add_2_grad/tuple/group_depsNoOp^gradients_1/add_2_grad/Reshape!^gradients_1/add_2_grad/Reshape_1
�
/gradients_1/add_2_grad/tuple/control_dependencyIdentitygradients_1/add_2_grad/Reshape(^gradients_1/add_2_grad/tuple/group_deps*1
_class'
%#loc:@gradients_1/add_2_grad/Reshape*
T0*
_output_shapes
: 
�
1gradients_1/add_2_grad/tuple/control_dependency_1Identity gradients_1/add_2_grad/Reshape_1(^gradients_1/add_2_grad/tuple/group_deps*3
_class)
'%loc:@gradients_1/add_2_grad/Reshape_1*
T0*
_output_shapes
: 
�
gradients_1/L2Loss_2_grad/mulMulVariable_6/read1gradients_1/add_3_grad/tuple/control_dependency_1*
T0*
_output_shapes
:	�
m
gradients_1/add_1_grad/ShapeConst*
_output_shapes
:*
valueB"      *
dtype0
h
gradients_1/add_1_grad/Shape_1Const*
_output_shapes
:*
valueB:*
dtype0
�
,gradients_1/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/add_1_grad/Shapegradients_1/add_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients_1/add_1_grad/SumSum#gradients_1/Reshape_11_grad/Reshape,gradients_1/add_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
gradients_1/add_1_grad/ReshapeReshapegradients_1/add_1_grad/Sumgradients_1/add_1_grad/Shape*
Tshape0*
T0*
_output_shapes

:
�
gradients_1/add_1_grad/Sum_1Sum#gradients_1/Reshape_11_grad/Reshape.gradients_1/add_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
 gradients_1/add_1_grad/Reshape_1Reshapegradients_1/add_1_grad/Sum_1gradients_1/add_1_grad/Shape_1*
Tshape0*
T0*
_output_shapes
:
s
'gradients_1/add_1_grad/tuple/group_depsNoOp^gradients_1/add_1_grad/Reshape!^gradients_1/add_1_grad/Reshape_1
�
/gradients_1/add_1_grad/tuple/control_dependencyIdentitygradients_1/add_1_grad/Reshape(^gradients_1/add_1_grad/tuple/group_deps*1
_class'
%#loc:@gradients_1/add_1_grad/Reshape*
T0*
_output_shapes

:
�
1gradients_1/add_1_grad/tuple/control_dependency_1Identity gradients_1/add_1_grad/Reshape_1(^gradients_1/add_1_grad/tuple/group_deps*3
_class)
'%loc:@gradients_1/add_1_grad/Reshape_1*
T0*
_output_shapes
:
�
gradients_1/L2Loss_grad/mulMulVariable_4/read/gradients_1/add_2_grad/tuple/control_dependency*
T0* 
_output_shapes
:
��
�
gradients_1/L2Loss_1_grad/mulMulVariable_5/read1gradients_1/add_2_grad/tuple/control_dependency_1*
T0*
_output_shapes	
:�
�
 gradients_1/MatMul_1_grad/MatMulMatMul/gradients_1/add_1_grad/tuple/control_dependencyVariable_6/read*
transpose_a( *
transpose_b(*
T0*
_output_shapes
:	�
�
"gradients_1/MatMul_1_grad/MatMul_1MatMulRelu_2/gradients_1/add_1_grad/tuple/control_dependency*
transpose_a(*
transpose_b( *
T0*
_output_shapes
:	�
z
*gradients_1/MatMul_1_grad/tuple/group_depsNoOp!^gradients_1/MatMul_1_grad/MatMul#^gradients_1/MatMul_1_grad/MatMul_1
�
2gradients_1/MatMul_1_grad/tuple/control_dependencyIdentity gradients_1/MatMul_1_grad/MatMul+^gradients_1/MatMul_1_grad/tuple/group_deps*3
_class)
'%loc:@gradients_1/MatMul_1_grad/MatMul*
T0*
_output_shapes
:	�
�
4gradients_1/MatMul_1_grad/tuple/control_dependency_1Identity"gradients_1/MatMul_1_grad/MatMul_1+^gradients_1/MatMul_1_grad/tuple/group_deps*5
_class+
)'loc:@gradients_1/MatMul_1_grad/MatMul_1*
T0*
_output_shapes
:	�
�
gradients_1/AddNAddNgradients_1/L2Loss_3_grad/mul1gradients_1/add_1_grad/tuple/control_dependency_1*
N*0
_class&
$"loc:@gradients_1/L2Loss_3_grad/mul*
T0*
_output_shapes
:
�
 gradients_1/Relu_2_grad/ReluGradReluGrad2gradients_1/MatMul_1_grad/tuple/control_dependencyRelu_2*
T0*
_output_shapes
:	�
�
gradients_1/AddN_1AddNgradients_1/L2Loss_2_grad/mul4gradients_1/MatMul_1_grad/tuple/control_dependency_1*
N*0
_class&
$"loc:@gradients_1/L2Loss_2_grad/mul*
T0*
_output_shapes
:	�
k
gradients_1/add_grad/ShapeConst*
_output_shapes
:*
valueB"      *
dtype0
g
gradients_1/add_grad/Shape_1Const*
_output_shapes
:*
valueB:�*
dtype0
�
*gradients_1/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/add_grad/Shapegradients_1/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients_1/add_grad/SumSum gradients_1/Relu_2_grad/ReluGrad*gradients_1/add_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
gradients_1/add_grad/ReshapeReshapegradients_1/add_grad/Sumgradients_1/add_grad/Shape*
Tshape0*
T0*
_output_shapes
:	�
�
gradients_1/add_grad/Sum_1Sum gradients_1/Relu_2_grad/ReluGrad,gradients_1/add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
gradients_1/add_grad/Reshape_1Reshapegradients_1/add_grad/Sum_1gradients_1/add_grad/Shape_1*
Tshape0*
T0*
_output_shapes	
:�
m
%gradients_1/add_grad/tuple/group_depsNoOp^gradients_1/add_grad/Reshape^gradients_1/add_grad/Reshape_1
�
-gradients_1/add_grad/tuple/control_dependencyIdentitygradients_1/add_grad/Reshape&^gradients_1/add_grad/tuple/group_deps*/
_class%
#!loc:@gradients_1/add_grad/Reshape*
T0*
_output_shapes
:	�
�
/gradients_1/add_grad/tuple/control_dependency_1Identitygradients_1/add_grad/Reshape_1&^gradients_1/add_grad/tuple/group_deps*1
_class'
%#loc:@gradients_1/add_grad/Reshape_1*
T0*
_output_shapes	
:�
�
gradients_1/MatMul_grad/MatMulMatMul-gradients_1/add_grad/tuple/control_dependencyVariable_4/read*
transpose_a( *
transpose_b(*
T0*
_output_shapes
:	�
�
 gradients_1/MatMul_grad/MatMul_1MatMulReshape-gradients_1/add_grad/tuple/control_dependency*
transpose_a(*
transpose_b( *
T0* 
_output_shapes
:
��
t
(gradients_1/MatMul_grad/tuple/group_depsNoOp^gradients_1/MatMul_grad/MatMul!^gradients_1/MatMul_grad/MatMul_1
�
0gradients_1/MatMul_grad/tuple/control_dependencyIdentitygradients_1/MatMul_grad/MatMul)^gradients_1/MatMul_grad/tuple/group_deps*1
_class'
%#loc:@gradients_1/MatMul_grad/MatMul*
T0*
_output_shapes
:	�
�
2gradients_1/MatMul_grad/tuple/control_dependency_1Identity gradients_1/MatMul_grad/MatMul_1)^gradients_1/MatMul_grad/tuple/group_deps*3
_class)
'%loc:@gradients_1/MatMul_grad/MatMul_1*
T0* 
_output_shapes
:
��
�
gradients_1/AddN_2AddNgradients_1/L2Loss_1_grad/mul/gradients_1/add_grad/tuple/control_dependency_1*
N*0
_class&
$"loc:@gradients_1/L2Loss_1_grad/mul*
T0*
_output_shapes	
:�
w
gradients_1/Reshape_grad/ShapeConst*
_output_shapes
:*%
valueB"         @   *
dtype0
�
 gradients_1/Reshape_grad/ReshapeReshape0gradients_1/MatMul_grad/tuple/control_dependencygradients_1/Reshape_grad/Shape*
Tshape0*
T0*&
_output_shapes
:@
�
gradients_1/AddN_3AddNgradients_1/L2Loss_grad/mul2gradients_1/MatMul_grad/tuple/control_dependency_1*
N*.
_class$
" loc:@gradients_1/L2Loss_grad/mul*
T0* 
_output_shapes
:
��
�
&gradients_1/MaxPool_1_grad/MaxPoolGradMaxPoolGradRelu_1	MaxPool_1 gradients_1/Reshape_grad/Reshape*
data_formatNHWC*
T0*
strides
*
ksize
*
paddingSAME*&
_output_shapes
:@
�
 gradients_1/Relu_1_grad/ReluGradReluGrad&gradients_1/MaxPool_1_grad/MaxPoolGradRelu_1*
T0*&
_output_shapes
:@
�
&gradients_1/BiasAdd_1_grad/BiasAddGradBiasAddGrad gradients_1/Relu_1_grad/ReluGrad*
data_formatNHWC*
T0*
_output_shapes
:@

+gradients_1/BiasAdd_1_grad/tuple/group_depsNoOp!^gradients_1/Relu_1_grad/ReluGrad'^gradients_1/BiasAdd_1_grad/BiasAddGrad
�
3gradients_1/BiasAdd_1_grad/tuple/control_dependencyIdentity gradients_1/Relu_1_grad/ReluGrad,^gradients_1/BiasAdd_1_grad/tuple/group_deps*3
_class)
'%loc:@gradients_1/Relu_1_grad/ReluGrad*
T0*&
_output_shapes
:@
�
5gradients_1/BiasAdd_1_grad/tuple/control_dependency_1Identity&gradients_1/BiasAdd_1_grad/BiasAddGrad,^gradients_1/BiasAdd_1_grad/tuple/group_deps*9
_class/
-+loc:@gradients_1/BiasAdd_1_grad/BiasAddGrad*
T0*
_output_shapes
:@
�
 gradients_1/Conv2D_1_grad/ShapeNShapeNMaxPoolVariable_2/read*
N*
out_type0*
T0* 
_output_shapes
::
�
-gradients_1/Conv2D_1_grad/Conv2DBackpropInputConv2DBackpropInput gradients_1/Conv2D_1_grad/ShapeNVariable_2/read3gradients_1/BiasAdd_1_grad/tuple/control_dependency*
data_formatNHWC*
T0*
strides
*
paddingSAME*
use_cudnn_on_gpu(*J
_output_shapes8
6:4������������������������������������
�
.gradients_1/Conv2D_1_grad/Conv2DBackpropFilterConv2DBackpropFilterMaxPool"gradients_1/Conv2D_1_grad/ShapeN:13gradients_1/BiasAdd_1_grad/tuple/control_dependency*
data_formatNHWC*
T0*
strides
*
paddingSAME*
use_cudnn_on_gpu(*J
_output_shapes8
6:4������������������������������������
�
*gradients_1/Conv2D_1_grad/tuple/group_depsNoOp.^gradients_1/Conv2D_1_grad/Conv2DBackpropInput/^gradients_1/Conv2D_1_grad/Conv2DBackpropFilter
�
2gradients_1/Conv2D_1_grad/tuple/control_dependencyIdentity-gradients_1/Conv2D_1_grad/Conv2DBackpropInput+^gradients_1/Conv2D_1_grad/tuple/group_deps*@
_class6
42loc:@gradients_1/Conv2D_1_grad/Conv2DBackpropInput*
T0*&
_output_shapes
: 
�
4gradients_1/Conv2D_1_grad/tuple/control_dependency_1Identity.gradients_1/Conv2D_1_grad/Conv2DBackpropFilter+^gradients_1/Conv2D_1_grad/tuple/group_deps*A
_class7
53loc:@gradients_1/Conv2D_1_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
: @
�
$gradients_1/MaxPool_grad/MaxPoolGradMaxPoolGradReluMaxPool2gradients_1/Conv2D_1_grad/tuple/control_dependency*
data_formatNHWC*
T0*
strides
*
ksize
*
paddingSAME*&
_output_shapes
: 
�
gradients_1/Relu_grad/ReluGradReluGrad$gradients_1/MaxPool_grad/MaxPoolGradRelu*
T0*&
_output_shapes
: 
�
$gradients_1/BiasAdd_grad/BiasAddGradBiasAddGradgradients_1/Relu_grad/ReluGrad*
data_formatNHWC*
T0*
_output_shapes
: 
y
)gradients_1/BiasAdd_grad/tuple/group_depsNoOp^gradients_1/Relu_grad/ReluGrad%^gradients_1/BiasAdd_grad/BiasAddGrad
�
1gradients_1/BiasAdd_grad/tuple/control_dependencyIdentitygradients_1/Relu_grad/ReluGrad*^gradients_1/BiasAdd_grad/tuple/group_deps*1
_class'
%#loc:@gradients_1/Relu_grad/ReluGrad*
T0*&
_output_shapes
: 
�
3gradients_1/BiasAdd_grad/tuple/control_dependency_1Identity$gradients_1/BiasAdd_grad/BiasAddGrad*^gradients_1/BiasAdd_grad/tuple/group_deps*7
_class-
+)loc:@gradients_1/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
: 
�
gradients_1/Conv2D_grad/ShapeNShapeNPlaceholderVariable/read*
N*
out_type0*
T0* 
_output_shapes
::
�
+gradients_1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInputgradients_1/Conv2D_grad/ShapeNVariable/read1gradients_1/BiasAdd_grad/tuple/control_dependency*
data_formatNHWC*
T0*
strides
*
paddingSAME*
use_cudnn_on_gpu(*J
_output_shapes8
6:4������������������������������������
�
,gradients_1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterPlaceholder gradients_1/Conv2D_grad/ShapeN:11gradients_1/BiasAdd_grad/tuple/control_dependency*
data_formatNHWC*
T0*
strides
*
paddingSAME*
use_cudnn_on_gpu(*J
_output_shapes8
6:4������������������������������������
�
(gradients_1/Conv2D_grad/tuple/group_depsNoOp,^gradients_1/Conv2D_grad/Conv2DBackpropInput-^gradients_1/Conv2D_grad/Conv2DBackpropFilter
�
0gradients_1/Conv2D_grad/tuple/control_dependencyIdentity+gradients_1/Conv2D_grad/Conv2DBackpropInput)^gradients_1/Conv2D_grad/tuple/group_deps*>
_class4
20loc:@gradients_1/Conv2D_grad/Conv2DBackpropInput*
T0*&
_output_shapes
:
�
2gradients_1/Conv2D_grad/tuple/control_dependency_1Identity,gradients_1/Conv2D_grad/Conv2DBackpropFilter)^gradients_1/Conv2D_grad/tuple/group_deps*?
_class5
31loc:@gradients_1/Conv2D_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
: 
�
#Variable/Momentum/Initializer/zerosConst*
_class
loc:@Variable*&
_output_shapes
: *%
valueB *    *
dtype0
�
Variable/Momentum
VariableV2*
shape: *
dtype0*
_class
loc:@Variable*
	container *
shared_name *&
_output_shapes
: 
�
Variable/Momentum/AssignAssignVariable/Momentum#Variable/Momentum/Initializer/zeros*
validate_shape(*
_class
loc:@Variable*
use_locking(*
T0*&
_output_shapes
: 
�
Variable/Momentum/readIdentityVariable/Momentum*
_class
loc:@Variable*
T0*&
_output_shapes
: 
�
%Variable_1/Momentum/Initializer/zerosConst*
_class
loc:@Variable_1*
_output_shapes
: *
valueB *    *
dtype0
�
Variable_1/Momentum
VariableV2*
shape: *
dtype0*
_class
loc:@Variable_1*
	container *
shared_name *
_output_shapes
: 
�
Variable_1/Momentum/AssignAssignVariable_1/Momentum%Variable_1/Momentum/Initializer/zeros*
validate_shape(*
_class
loc:@Variable_1*
use_locking(*
T0*
_output_shapes
: 
}
Variable_1/Momentum/readIdentityVariable_1/Momentum*
_class
loc:@Variable_1*
T0*
_output_shapes
: 
�
%Variable_2/Momentum/Initializer/zerosConst*
_class
loc:@Variable_2*&
_output_shapes
: @*%
valueB @*    *
dtype0
�
Variable_2/Momentum
VariableV2*
shape: @*
dtype0*
_class
loc:@Variable_2*
	container *
shared_name *&
_output_shapes
: @
�
Variable_2/Momentum/AssignAssignVariable_2/Momentum%Variable_2/Momentum/Initializer/zeros*
validate_shape(*
_class
loc:@Variable_2*
use_locking(*
T0*&
_output_shapes
: @
�
Variable_2/Momentum/readIdentityVariable_2/Momentum*
_class
loc:@Variable_2*
T0*&
_output_shapes
: @
�
%Variable_3/Momentum/Initializer/zerosConst*
_class
loc:@Variable_3*
_output_shapes
:@*
valueB@*    *
dtype0
�
Variable_3/Momentum
VariableV2*
shape:@*
dtype0*
_class
loc:@Variable_3*
	container *
shared_name *
_output_shapes
:@
�
Variable_3/Momentum/AssignAssignVariable_3/Momentum%Variable_3/Momentum/Initializer/zeros*
validate_shape(*
_class
loc:@Variable_3*
use_locking(*
T0*
_output_shapes
:@
}
Variable_3/Momentum/readIdentityVariable_3/Momentum*
_class
loc:@Variable_3*
T0*
_output_shapes
:@
�
%Variable_4/Momentum/Initializer/zerosConst*
_class
loc:@Variable_4* 
_output_shapes
:
��*
valueB
��*    *
dtype0
�
Variable_4/Momentum
VariableV2*
shape:
��*
dtype0*
_class
loc:@Variable_4*
	container *
shared_name * 
_output_shapes
:
��
�
Variable_4/Momentum/AssignAssignVariable_4/Momentum%Variable_4/Momentum/Initializer/zeros*
validate_shape(*
_class
loc:@Variable_4*
use_locking(*
T0* 
_output_shapes
:
��
�
Variable_4/Momentum/readIdentityVariable_4/Momentum*
_class
loc:@Variable_4*
T0* 
_output_shapes
:
��
�
%Variable_5/Momentum/Initializer/zerosConst*
_class
loc:@Variable_5*
_output_shapes	
:�*
valueB�*    *
dtype0
�
Variable_5/Momentum
VariableV2*
shape:�*
dtype0*
_class
loc:@Variable_5*
	container *
shared_name *
_output_shapes	
:�
�
Variable_5/Momentum/AssignAssignVariable_5/Momentum%Variable_5/Momentum/Initializer/zeros*
validate_shape(*
_class
loc:@Variable_5*
use_locking(*
T0*
_output_shapes	
:�
~
Variable_5/Momentum/readIdentityVariable_5/Momentum*
_class
loc:@Variable_5*
T0*
_output_shapes	
:�
�
%Variable_6/Momentum/Initializer/zerosConst*
_class
loc:@Variable_6*
_output_shapes
:	�*
valueB	�*    *
dtype0
�
Variable_6/Momentum
VariableV2*
shape:	�*
dtype0*
_class
loc:@Variable_6*
	container *
shared_name *
_output_shapes
:	�
�
Variable_6/Momentum/AssignAssignVariable_6/Momentum%Variable_6/Momentum/Initializer/zeros*
validate_shape(*
_class
loc:@Variable_6*
use_locking(*
T0*
_output_shapes
:	�
�
Variable_6/Momentum/readIdentityVariable_6/Momentum*
_class
loc:@Variable_6*
T0*
_output_shapes
:	�
�
%Variable_7/Momentum/Initializer/zerosConst*
_class
loc:@Variable_7*
_output_shapes
:*
valueB*    *
dtype0
�
Variable_7/Momentum
VariableV2*
shape:*
dtype0*
_class
loc:@Variable_7*
	container *
shared_name *
_output_shapes
:
�
Variable_7/Momentum/AssignAssignVariable_7/Momentum%Variable_7/Momentum/Initializer/zeros*
validate_shape(*
_class
loc:@Variable_7*
use_locking(*
T0*
_output_shapes
:
}
Variable_7/Momentum/readIdentityVariable_7/Momentum*
_class
loc:@Variable_7*
T0*
_output_shapes
:
V
Momentum/momentumConst*
_output_shapes
: *
valueB
 *    *
dtype0
�
&Momentum/update_Variable/ApplyMomentumApplyMomentumVariableVariable/MomentumExponentialDecay2gradients_1/Conv2D_grad/tuple/control_dependency_1Momentum/momentum*
_class
loc:@Variable*
use_nesterov( *
use_locking( *
T0*&
_output_shapes
: 
�
(Momentum/update_Variable_1/ApplyMomentumApplyMomentum
Variable_1Variable_1/MomentumExponentialDecay3gradients_1/BiasAdd_grad/tuple/control_dependency_1Momentum/momentum*
_class
loc:@Variable_1*
use_nesterov( *
use_locking( *
T0*
_output_shapes
: 
�
(Momentum/update_Variable_2/ApplyMomentumApplyMomentum
Variable_2Variable_2/MomentumExponentialDecay4gradients_1/Conv2D_1_grad/tuple/control_dependency_1Momentum/momentum*
_class
loc:@Variable_2*
use_nesterov( *
use_locking( *
T0*&
_output_shapes
: @
�
(Momentum/update_Variable_3/ApplyMomentumApplyMomentum
Variable_3Variable_3/MomentumExponentialDecay5gradients_1/BiasAdd_1_grad/tuple/control_dependency_1Momentum/momentum*
_class
loc:@Variable_3*
use_nesterov( *
use_locking( *
T0*
_output_shapes
:@
�
(Momentum/update_Variable_4/ApplyMomentumApplyMomentum
Variable_4Variable_4/MomentumExponentialDecaygradients_1/AddN_3Momentum/momentum*
_class
loc:@Variable_4*
use_nesterov( *
use_locking( *
T0* 
_output_shapes
:
��
�
(Momentum/update_Variable_5/ApplyMomentumApplyMomentum
Variable_5Variable_5/MomentumExponentialDecaygradients_1/AddN_2Momentum/momentum*
_class
loc:@Variable_5*
use_nesterov( *
use_locking( *
T0*
_output_shapes	
:�
�
(Momentum/update_Variable_6/ApplyMomentumApplyMomentum
Variable_6Variable_6/MomentumExponentialDecaygradients_1/AddN_1Momentum/momentum*
_class
loc:@Variable_6*
use_nesterov( *
use_locking( *
T0*
_output_shapes
:	�
�
(Momentum/update_Variable_7/ApplyMomentumApplyMomentum
Variable_7Variable_7/MomentumExponentialDecaygradients_1/AddNMomentum/momentum*
_class
loc:@Variable_7*
use_nesterov( *
use_locking( *
T0*
_output_shapes
:
�
Momentum/updateNoOp'^Momentum/update_Variable/ApplyMomentum)^Momentum/update_Variable_1/ApplyMomentum)^Momentum/update_Variable_2/ApplyMomentum)^Momentum/update_Variable_3/ApplyMomentum)^Momentum/update_Variable_4/ApplyMomentum)^Momentum/update_Variable_5/ApplyMomentum)^Momentum/update_Variable_6/ApplyMomentum)^Momentum/update_Variable_7/ApplyMomentum
�
Momentum/valueConst^Momentum/update*
_class
loc:@Variable_8*
_output_shapes
: *
value	B :*
dtype0
�
Momentum	AssignAdd
Variable_8Momentum/value*
_class
loc:@Variable_8*
use_locking( *
T0*
_output_shapes
: 
B
SoftmaxSoftmaxadd_1*
T0*
_output_shapes

:
�
Conv2D_2Conv2DPlaceholder_2Variable/read*
data_formatNHWC*
T0*
strides
*
paddingSAME*
use_cudnn_on_gpu(*(
_output_shapes
:��= 
y
	BiasAdd_2BiasAddConv2D_2Variable_1/read*
data_formatNHWC*
T0*(
_output_shapes
:��= 
L
Relu_3Relu	BiasAdd_2*
T0*(
_output_shapes
:��= 
�
	MaxPool_2MaxPoolRelu_3*
data_formatNHWC*
strides
*
T0*
ksize
*
paddingSAME*(
_output_shapes
:��= 
�
Conv2D_3Conv2D	MaxPool_2Variable_2/read*
data_formatNHWC*
T0*
strides
*
paddingSAME*
use_cudnn_on_gpu(*(
_output_shapes
:��=@
y
	BiasAdd_3BiasAddConv2D_3Variable_3/read*
data_formatNHWC*
T0*(
_output_shapes
:��=@
L
Relu_4Relu	BiasAdd_3*
T0*(
_output_shapes
:��=@
�
	MaxPool_3MaxPoolRelu_4*
data_formatNHWC*
strides
*
T0*
ksize
*
paddingSAME*(
_output_shapes
:��=@
a
Reshape_14/shapeConst*
_output_shapes
:*
valueB"@B    *
dtype0
l

Reshape_14Reshape	MaxPool_3Reshape_14/shape*
Tshape0*
T0*!
_output_shapes
:��=�
�
MatMul_2MatMul
Reshape_14Variable_4/read*
transpose_a( *
transpose_b( *
T0*!
_output_shapes
:��=�
S
add_6AddMatMul_2Variable_5/read*
T0*!
_output_shapes
:��=�
A
Relu_5Reluadd_6*
T0*!
_output_shapes
:��=�
|
MatMul_3MatMulRelu_5Variable_6/read*
transpose_a( *
transpose_b( *
T0* 
_output_shapes
:
��=
R
add_7AddMatMul_3Variable_7/read*
T0* 
_output_shapes
:
��=
F
	Softmax_1Softmaxadd_7*
T0* 
_output_shapes
:
��=
P

save/ConstConst*
_output_shapes
: *
valueB Bmodel*
dtype0
�
save/SaveV2/tensor_namesConst*
_output_shapes
:*�
value�B�BVariableBVariable/MomentumB
Variable_1BVariable_1/MomentumB
Variable_2BVariable_2/MomentumB
Variable_3BVariable_3/MomentumB
Variable_4BVariable_4/MomentumB
Variable_5BVariable_5/MomentumB
Variable_6BVariable_6/MomentumB
Variable_7BVariable_7/MomentumB
Variable_8*
dtype0
�
save/SaveV2/shape_and_slicesConst*
_output_shapes
:*5
value,B*B B B B B B B B B B B B B B B B B *
dtype0
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
save/Const^save/SaveV2*
_class
loc:@save/Const*
T0*
_output_shapes
: 
l
save/RestoreV2/tensor_namesConst*
_output_shapes
:*
valueBBVariable*
dtype0
h
save/RestoreV2/shape_and_slicesConst*
_output_shapes
:*
valueB
B *
dtype0
�
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/AssignAssignVariablesave/RestoreV2*
validate_shape(*
_class
loc:@Variable*
use_locking(*
T0*&
_output_shapes
: 
w
save/RestoreV2_1/tensor_namesConst*
_output_shapes
:*&
valueBBVariable/Momentum*
dtype0
j
!save/RestoreV2_1/shape_and_slicesConst*
_output_shapes
:*
valueB
B *
dtype0
�
save/RestoreV2_1	RestoreV2
save/Constsave/RestoreV2_1/tensor_names!save/RestoreV2_1/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_1AssignVariable/Momentumsave/RestoreV2_1*
validate_shape(*
_class
loc:@Variable*
use_locking(*
T0*&
_output_shapes
: 
p
save/RestoreV2_2/tensor_namesConst*
_output_shapes
:*
valueBB
Variable_1*
dtype0
j
!save/RestoreV2_2/shape_and_slicesConst*
_output_shapes
:*
valueB
B *
dtype0
�
save/RestoreV2_2	RestoreV2
save/Constsave/RestoreV2_2/tensor_names!save/RestoreV2_2/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_2Assign
Variable_1save/RestoreV2_2*
validate_shape(*
_class
loc:@Variable_1*
use_locking(*
T0*
_output_shapes
: 
y
save/RestoreV2_3/tensor_namesConst*
_output_shapes
:*(
valueBBVariable_1/Momentum*
dtype0
j
!save/RestoreV2_3/shape_and_slicesConst*
_output_shapes
:*
valueB
B *
dtype0
�
save/RestoreV2_3	RestoreV2
save/Constsave/RestoreV2_3/tensor_names!save/RestoreV2_3/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_3AssignVariable_1/Momentumsave/RestoreV2_3*
validate_shape(*
_class
loc:@Variable_1*
use_locking(*
T0*
_output_shapes
: 
p
save/RestoreV2_4/tensor_namesConst*
_output_shapes
:*
valueBB
Variable_2*
dtype0
j
!save/RestoreV2_4/shape_and_slicesConst*
_output_shapes
:*
valueB
B *
dtype0
�
save/RestoreV2_4	RestoreV2
save/Constsave/RestoreV2_4/tensor_names!save/RestoreV2_4/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_4Assign
Variable_2save/RestoreV2_4*
validate_shape(*
_class
loc:@Variable_2*
use_locking(*
T0*&
_output_shapes
: @
y
save/RestoreV2_5/tensor_namesConst*
_output_shapes
:*(
valueBBVariable_2/Momentum*
dtype0
j
!save/RestoreV2_5/shape_and_slicesConst*
_output_shapes
:*
valueB
B *
dtype0
�
save/RestoreV2_5	RestoreV2
save/Constsave/RestoreV2_5/tensor_names!save/RestoreV2_5/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_5AssignVariable_2/Momentumsave/RestoreV2_5*
validate_shape(*
_class
loc:@Variable_2*
use_locking(*
T0*&
_output_shapes
: @
p
save/RestoreV2_6/tensor_namesConst*
_output_shapes
:*
valueBB
Variable_3*
dtype0
j
!save/RestoreV2_6/shape_and_slicesConst*
_output_shapes
:*
valueB
B *
dtype0
�
save/RestoreV2_6	RestoreV2
save/Constsave/RestoreV2_6/tensor_names!save/RestoreV2_6/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_6Assign
Variable_3save/RestoreV2_6*
validate_shape(*
_class
loc:@Variable_3*
use_locking(*
T0*
_output_shapes
:@
y
save/RestoreV2_7/tensor_namesConst*
_output_shapes
:*(
valueBBVariable_3/Momentum*
dtype0
j
!save/RestoreV2_7/shape_and_slicesConst*
_output_shapes
:*
valueB
B *
dtype0
�
save/RestoreV2_7	RestoreV2
save/Constsave/RestoreV2_7/tensor_names!save/RestoreV2_7/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_7AssignVariable_3/Momentumsave/RestoreV2_7*
validate_shape(*
_class
loc:@Variable_3*
use_locking(*
T0*
_output_shapes
:@
p
save/RestoreV2_8/tensor_namesConst*
_output_shapes
:*
valueBB
Variable_4*
dtype0
j
!save/RestoreV2_8/shape_and_slicesConst*
_output_shapes
:*
valueB
B *
dtype0
�
save/RestoreV2_8	RestoreV2
save/Constsave/RestoreV2_8/tensor_names!save/RestoreV2_8/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_8Assign
Variable_4save/RestoreV2_8*
validate_shape(*
_class
loc:@Variable_4*
use_locking(*
T0* 
_output_shapes
:
��
y
save/RestoreV2_9/tensor_namesConst*
_output_shapes
:*(
valueBBVariable_4/Momentum*
dtype0
j
!save/RestoreV2_9/shape_and_slicesConst*
_output_shapes
:*
valueB
B *
dtype0
�
save/RestoreV2_9	RestoreV2
save/Constsave/RestoreV2_9/tensor_names!save/RestoreV2_9/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_9AssignVariable_4/Momentumsave/RestoreV2_9*
validate_shape(*
_class
loc:@Variable_4*
use_locking(*
T0* 
_output_shapes
:
��
q
save/RestoreV2_10/tensor_namesConst*
_output_shapes
:*
valueBB
Variable_5*
dtype0
k
"save/RestoreV2_10/shape_and_slicesConst*
_output_shapes
:*
valueB
B *
dtype0
�
save/RestoreV2_10	RestoreV2
save/Constsave/RestoreV2_10/tensor_names"save/RestoreV2_10/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_10Assign
Variable_5save/RestoreV2_10*
validate_shape(*
_class
loc:@Variable_5*
use_locking(*
T0*
_output_shapes	
:�
z
save/RestoreV2_11/tensor_namesConst*
_output_shapes
:*(
valueBBVariable_5/Momentum*
dtype0
k
"save/RestoreV2_11/shape_and_slicesConst*
_output_shapes
:*
valueB
B *
dtype0
�
save/RestoreV2_11	RestoreV2
save/Constsave/RestoreV2_11/tensor_names"save/RestoreV2_11/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_11AssignVariable_5/Momentumsave/RestoreV2_11*
validate_shape(*
_class
loc:@Variable_5*
use_locking(*
T0*
_output_shapes	
:�
q
save/RestoreV2_12/tensor_namesConst*
_output_shapes
:*
valueBB
Variable_6*
dtype0
k
"save/RestoreV2_12/shape_and_slicesConst*
_output_shapes
:*
valueB
B *
dtype0
�
save/RestoreV2_12	RestoreV2
save/Constsave/RestoreV2_12/tensor_names"save/RestoreV2_12/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_12Assign
Variable_6save/RestoreV2_12*
validate_shape(*
_class
loc:@Variable_6*
use_locking(*
T0*
_output_shapes
:	�
z
save/RestoreV2_13/tensor_namesConst*
_output_shapes
:*(
valueBBVariable_6/Momentum*
dtype0
k
"save/RestoreV2_13/shape_and_slicesConst*
_output_shapes
:*
valueB
B *
dtype0
�
save/RestoreV2_13	RestoreV2
save/Constsave/RestoreV2_13/tensor_names"save/RestoreV2_13/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_13AssignVariable_6/Momentumsave/RestoreV2_13*
validate_shape(*
_class
loc:@Variable_6*
use_locking(*
T0*
_output_shapes
:	�
q
save/RestoreV2_14/tensor_namesConst*
_output_shapes
:*
valueBB
Variable_7*
dtype0
k
"save/RestoreV2_14/shape_and_slicesConst*
_output_shapes
:*
valueB
B *
dtype0
�
save/RestoreV2_14	RestoreV2
save/Constsave/RestoreV2_14/tensor_names"save/RestoreV2_14/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_14Assign
Variable_7save/RestoreV2_14*
validate_shape(*
_class
loc:@Variable_7*
use_locking(*
T0*
_output_shapes
:
z
save/RestoreV2_15/tensor_namesConst*
_output_shapes
:*(
valueBBVariable_7/Momentum*
dtype0
k
"save/RestoreV2_15/shape_and_slicesConst*
_output_shapes
:*
valueB
B *
dtype0
�
save/RestoreV2_15	RestoreV2
save/Constsave/RestoreV2_15/tensor_names"save/RestoreV2_15/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_15AssignVariable_7/Momentumsave/RestoreV2_15*
validate_shape(*
_class
loc:@Variable_7*
use_locking(*
T0*
_output_shapes
:
q
save/RestoreV2_16/tensor_namesConst*
_output_shapes
:*
valueBB
Variable_8*
dtype0
k
"save/RestoreV2_16/shape_and_slicesConst*
_output_shapes
:*
valueB
B *
dtype0
�
save/RestoreV2_16	RestoreV2
save/Constsave/RestoreV2_16/tensor_names"save/RestoreV2_16/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_16Assign
Variable_8save/RestoreV2_16*
validate_shape(*
_class
loc:@Variable_8*
use_locking(*
T0*
_output_shapes
: 
�
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16
�
initNoOp^Variable/Assign^Variable_1/Assign^Variable_2/Assign^Variable_3/Assign^Variable_4/Assign^Variable_5/Assign^Variable_6/Assign^Variable_7/Assign^Variable_8/Assign^Variable/Momentum/Assign^Variable_1/Momentum/Assign^Variable_2/Momentum/Assign^Variable_3/Momentum/Assign^Variable_4/Momentum/Assign^Variable_5/Momentum/Assign^Variable_6/Momentum/Assign^Variable_7/Momentum/Assign
�
Merge/MergeSummaryMergeSummarysummary_data_0summary_conv_0summary_pool_0summary_conv2_0summary_pool2_0lossconv1_weightsconv1_biasesconv2_weightsconv2_biasesfc1_weights
fc1_biasesfc2_weights
fc2_biaseslearning_rate*
N*
_output_shapes
: "g<��t     ���	��갍�AJ��
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
Ttype*1.4.02v1.4.0-rc1-11-g130a514Ш
l
PlaceholderPlaceholder*
shape:*
dtype0*&
_output_shapes
:
^
Placeholder_1Placeholder*
shape
:*
dtype0*
_output_shapes

:
r
Placeholder_2Placeholder*
shape:��=*
dtype0*(
_output_shapes
:��=
o
truncated_normal/shapeConst*
dtype0*%
valueB"             *
_output_shapes
:
Z
truncated_normal/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
\
truncated_normal/stddevConst*
dtype0*
valueB
 *���=*
_output_shapes
: 
�
 truncated_normal/TruncatedNormalTruncatedNormaltruncated_normal/shape*
seed���)*
dtype0*
T0*
seed2��*&
_output_shapes
: 
�
truncated_normal/mulMul truncated_normal/TruncatedNormaltruncated_normal/stddev*
T0*&
_output_shapes
: 
u
truncated_normalAddtruncated_normal/multruncated_normal/mean*
T0*&
_output_shapes
: 
�
Variable
VariableV2*
shape: *&
_output_shapes
: *
shared_name *
	container *
dtype0
�
Variable/AssignAssignVariabletruncated_normal*
validate_shape(*
_class
loc:@Variable*
use_locking(*
T0*&
_output_shapes
: 
q
Variable/readIdentityVariable*
_class
loc:@Variable*
T0*&
_output_shapes
: 
R
zerosConst*
dtype0*
valueB *    *
_output_shapes
: 
v

Variable_1
VariableV2*
shape: *
_output_shapes
: *
shared_name *
	container *
dtype0
�
Variable_1/AssignAssign
Variable_1zeros*
validate_shape(*
_class
loc:@Variable_1*
use_locking(*
T0*
_output_shapes
: 
k
Variable_1/readIdentity
Variable_1*
_class
loc:@Variable_1*
T0*
_output_shapes
: 
q
truncated_normal_1/shapeConst*
dtype0*%
valueB"          @   *
_output_shapes
:
\
truncated_normal_1/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
^
truncated_normal_1/stddevConst*
dtype0*
valueB
 *���=*
_output_shapes
: 
�
"truncated_normal_1/TruncatedNormalTruncatedNormaltruncated_normal_1/shape*
seed���)*
dtype0*
T0*
seed2��*&
_output_shapes
: @
�
truncated_normal_1/mulMul"truncated_normal_1/TruncatedNormaltruncated_normal_1/stddev*
T0*&
_output_shapes
: @
{
truncated_normal_1Addtruncated_normal_1/multruncated_normal_1/mean*
T0*&
_output_shapes
: @
�

Variable_2
VariableV2*
shape: @*&
_output_shapes
: @*
shared_name *
	container *
dtype0
�
Variable_2/AssignAssign
Variable_2truncated_normal_1*
validate_shape(*
_class
loc:@Variable_2*
use_locking(*
T0*&
_output_shapes
: @
w
Variable_2/readIdentity
Variable_2*
_class
loc:@Variable_2*
T0*&
_output_shapes
: @
R
ConstConst*
dtype0*
valueB@*���=*
_output_shapes
:@
v

Variable_3
VariableV2*
shape:@*
_output_shapes
:@*
shared_name *
	container *
dtype0
�
Variable_3/AssignAssign
Variable_3Const*
validate_shape(*
_class
loc:@Variable_3*
use_locking(*
T0*
_output_shapes
:@
k
Variable_3/readIdentity
Variable_3*
_class
loc:@Variable_3*
T0*
_output_shapes
:@
i
truncated_normal_2/shapeConst*
dtype0*
valueB"      *
_output_shapes
:
\
truncated_normal_2/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
^
truncated_normal_2/stddevConst*
dtype0*
valueB
 *���=*
_output_shapes
: 
�
"truncated_normal_2/TruncatedNormalTruncatedNormaltruncated_normal_2/shape*
seed���)*
dtype0*
T0*
seed2��* 
_output_shapes
:
��
�
truncated_normal_2/mulMul"truncated_normal_2/TruncatedNormaltruncated_normal_2/stddev*
T0* 
_output_shapes
:
��
u
truncated_normal_2Addtruncated_normal_2/multruncated_normal_2/mean*
T0* 
_output_shapes
:
��
�

Variable_4
VariableV2*
shape:
��* 
_output_shapes
:
��*
shared_name *
	container *
dtype0
�
Variable_4/AssignAssign
Variable_4truncated_normal_2*
validate_shape(*
_class
loc:@Variable_4*
use_locking(*
T0* 
_output_shapes
:
��
q
Variable_4/readIdentity
Variable_4*
_class
loc:@Variable_4*
T0* 
_output_shapes
:
��
V
Const_1Const*
dtype0*
valueB�*���=*
_output_shapes	
:�
x

Variable_5
VariableV2*
shape:�*
_output_shapes	
:�*
shared_name *
	container *
dtype0
�
Variable_5/AssignAssign
Variable_5Const_1*
validate_shape(*
_class
loc:@Variable_5*
use_locking(*
T0*
_output_shapes	
:�
l
Variable_5/readIdentity
Variable_5*
_class
loc:@Variable_5*
T0*
_output_shapes	
:�
i
truncated_normal_3/shapeConst*
dtype0*
valueB"      *
_output_shapes
:
\
truncated_normal_3/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
^
truncated_normal_3/stddevConst*
dtype0*
valueB
 *���=*
_output_shapes
: 
�
"truncated_normal_3/TruncatedNormalTruncatedNormaltruncated_normal_3/shape*
seed���)*
dtype0*
T0*
seed2��*
_output_shapes
:	�
�
truncated_normal_3/mulMul"truncated_normal_3/TruncatedNormaltruncated_normal_3/stddev*
T0*
_output_shapes
:	�
t
truncated_normal_3Addtruncated_normal_3/multruncated_normal_3/mean*
T0*
_output_shapes
:	�
�

Variable_6
VariableV2*
shape:	�*
_output_shapes
:	�*
shared_name *
	container *
dtype0
�
Variable_6/AssignAssign
Variable_6truncated_normal_3*
validate_shape(*
_class
loc:@Variable_6*
use_locking(*
T0*
_output_shapes
:	�
p
Variable_6/readIdentity
Variable_6*
_class
loc:@Variable_6*
T0*
_output_shapes
:	�
T
Const_2Const*
dtype0*
valueB*���=*
_output_shapes
:
v

Variable_7
VariableV2*
shape:*
_output_shapes
:*
shared_name *
	container *
dtype0
�
Variable_7/AssignAssign
Variable_7Const_2*
validate_shape(*
_class
loc:@Variable_7*
use_locking(*
T0*
_output_shapes
:
k
Variable_7/readIdentity
Variable_7*
_class
loc:@Variable_7*
T0*
_output_shapes
:
�
Conv2DConv2DPlaceholderVariable/read*
data_formatNHWC*
T0*
strides
*
paddingSAME*
use_cudnn_on_gpu(*&
_output_shapes
: 
s
BiasAddBiasAddConv2DVariable_1/read*
data_formatNHWC*
T0*&
_output_shapes
: 
F
ReluReluBiasAdd*
T0*&
_output_shapes
: 
�
MaxPoolMaxPoolRelu*
data_formatNHWC*
strides
*
T0*
paddingSAME*
ksize
*&
_output_shapes
: 
�
Conv2D_1Conv2DMaxPoolVariable_2/read*
data_formatNHWC*
T0*
strides
*
paddingSAME*
use_cudnn_on_gpu(*&
_output_shapes
:@
w
	BiasAdd_1BiasAddConv2D_1Variable_3/read*
data_formatNHWC*
T0*&
_output_shapes
:@
J
Relu_1Relu	BiasAdd_1*
T0*&
_output_shapes
:@
�
	MaxPool_1MaxPoolRelu_1*
data_formatNHWC*
strides
*
T0*
paddingSAME*
ksize
*&
_output_shapes
:@
^
Reshape/shapeConst*
dtype0*
valueB"      *
_output_shapes
:
d
ReshapeReshape	MaxPool_1Reshape/shape*
Tshape0*
T0*
_output_shapes
:	�
z
MatMulMatMulReshapeVariable_4/read*
transpose_a( *
T0*
transpose_b( *
_output_shapes
:	�
M
addAddMatMulVariable_5/read*
T0*
_output_shapes
:	�
=
Relu_2Reluadd*
T0*
_output_shapes
:	�
z
MatMul_1MatMulRelu_2Variable_6/read*
transpose_a( *
T0*
transpose_b( *
_output_shapes

:
P
add_1AddMatMul_1Variable_7/read*
T0*
_output_shapes

:
d
Slice/beginConst*
dtype0*%
valueB"                *
_output_shapes
:
c

Slice/sizeConst*
dtype0*%
valueB"   ��������   *
_output_shapes
:
r
SliceSlicePlaceholderSlice/begin
Slice/size*
Index0*
T0*&
_output_shapes
:
`
Const_3Const*
dtype0*%
valueB"             *
_output_shapes
:
X
MinMinSliceConst_3*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
G
subSubSliceMin*
T0*&
_output_shapes
:
`
Const_4Const*
dtype0*%
valueB"             *
_output_shapes
:
V
MaxMaxsubConst_4*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
J
mul/yConst*
dtype0*
valueB
 *  C*
_output_shapes
: 
7
mulMulMaxmul/y*
T0*
_output_shapes
: 
M
truedivRealDivsubmul*
T0*&
_output_shapes
:
d
Reshape_1/shapeConst*
dtype0*!
valueB"         *
_output_shapes
:
i
	Reshape_1ReshapetruedivReshape_1/shape*
Tshape0*
T0*"
_output_shapes
:
c
transpose/permConst*
dtype0*!
valueB"          *
_output_shapes
:
k
	transpose	Transpose	Reshape_1transpose/perm*
Tperm0*
T0*"
_output_shapes
:
h
Reshape_2/shapeConst*
dtype0*%
valueB"����         *
_output_shapes
:
o
	Reshape_2Reshape	transposeReshape_2/shape*
Tshape0*
T0*&
_output_shapes
:
a
summary_data_0/tagConst*
dtype0*
valueB Bsummary_data_0*
_output_shapes
: 
�
summary_data_0ImageSummarysummary_data_0/tag	Reshape_2*

max_images*
	bad_colorB:�  �*
T0*
_output_shapes
: 
f
Slice_1/beginConst*
dtype0*%
valueB"                *
_output_shapes
:
e
Slice_1/sizeConst*
dtype0*%
valueB"   ��������   *
_output_shapes
:
s
Slice_1SliceConv2DSlice_1/beginSlice_1/size*
Index0*
T0*&
_output_shapes
:
`
Const_5Const*
dtype0*%
valueB"             *
_output_shapes
:
\
Min_1MinSlice_1Const_5*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
M
sub_1SubSlice_1Min_1*
T0*&
_output_shapes
:
`
Const_6Const*
dtype0*%
valueB"             *
_output_shapes
:
Z
Max_1Maxsub_1Const_6*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
L
mul_1/yConst*
dtype0*
valueB
 *  C*
_output_shapes
: 
=
mul_1MulMax_1mul_1/y*
T0*
_output_shapes
: 
S
	truediv_1RealDivsub_1mul_1*
T0*&
_output_shapes
:
d
Reshape_3/shapeConst*
dtype0*!
valueB"         *
_output_shapes
:
k
	Reshape_3Reshape	truediv_1Reshape_3/shape*
Tshape0*
T0*"
_output_shapes
:
e
transpose_1/permConst*
dtype0*!
valueB"          *
_output_shapes
:
o
transpose_1	Transpose	Reshape_3transpose_1/perm*
Tperm0*
T0*"
_output_shapes
:
h
Reshape_4/shapeConst*
dtype0*%
valueB"����         *
_output_shapes
:
q
	Reshape_4Reshapetranspose_1Reshape_4/shape*
Tshape0*
T0*&
_output_shapes
:
a
summary_conv_0/tagConst*
dtype0*
valueB Bsummary_conv_0*
_output_shapes
: 
�
summary_conv_0ImageSummarysummary_conv_0/tag	Reshape_4*

max_images*
	bad_colorB:�  �*
T0*
_output_shapes
: 
f
Slice_2/beginConst*
dtype0*%
valueB"                *
_output_shapes
:
e
Slice_2/sizeConst*
dtype0*%
valueB"   ��������   *
_output_shapes
:
t
Slice_2SliceMaxPoolSlice_2/beginSlice_2/size*
Index0*
T0*&
_output_shapes
:
`
Const_7Const*
dtype0*%
valueB"             *
_output_shapes
:
\
Min_2MinSlice_2Const_7*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
M
sub_2SubSlice_2Min_2*
T0*&
_output_shapes
:
`
Const_8Const*
dtype0*%
valueB"             *
_output_shapes
:
Z
Max_2Maxsub_2Const_8*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
L
mul_2/yConst*
dtype0*
valueB
 *  C*
_output_shapes
: 
=
mul_2MulMax_2mul_2/y*
T0*
_output_shapes
: 
S
	truediv_2RealDivsub_2mul_2*
T0*&
_output_shapes
:
d
Reshape_5/shapeConst*
dtype0*!
valueB"         *
_output_shapes
:
k
	Reshape_5Reshape	truediv_2Reshape_5/shape*
Tshape0*
T0*"
_output_shapes
:
e
transpose_2/permConst*
dtype0*!
valueB"          *
_output_shapes
:
o
transpose_2	Transpose	Reshape_5transpose_2/perm*
Tperm0*
T0*"
_output_shapes
:
h
Reshape_6/shapeConst*
dtype0*%
valueB"����         *
_output_shapes
:
q
	Reshape_6Reshapetranspose_2Reshape_6/shape*
Tshape0*
T0*&
_output_shapes
:
a
summary_pool_0/tagConst*
dtype0*
valueB Bsummary_pool_0*
_output_shapes
: 
�
summary_pool_0ImageSummarysummary_pool_0/tag	Reshape_6*

max_images*
	bad_colorB:�  �*
T0*
_output_shapes
: 
f
Slice_3/beginConst*
dtype0*%
valueB"                *
_output_shapes
:
e
Slice_3/sizeConst*
dtype0*%
valueB"   ��������   *
_output_shapes
:
u
Slice_3SliceConv2D_1Slice_3/beginSlice_3/size*
Index0*
T0*&
_output_shapes
:
`
Const_9Const*
dtype0*%
valueB"             *
_output_shapes
:
\
Min_3MinSlice_3Const_9*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
M
sub_3SubSlice_3Min_3*
T0*&
_output_shapes
:
a
Const_10Const*
dtype0*%
valueB"             *
_output_shapes
:
[
Max_3Maxsub_3Const_10*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
L
mul_3/yConst*
dtype0*
valueB
 *  C*
_output_shapes
: 
=
mul_3MulMax_3mul_3/y*
T0*
_output_shapes
: 
S
	truediv_3RealDivsub_3mul_3*
T0*&
_output_shapes
:
d
Reshape_7/shapeConst*
dtype0*!
valueB"         *
_output_shapes
:
k
	Reshape_7Reshape	truediv_3Reshape_7/shape*
Tshape0*
T0*"
_output_shapes
:
e
transpose_3/permConst*
dtype0*!
valueB"          *
_output_shapes
:
o
transpose_3	Transpose	Reshape_7transpose_3/perm*
Tperm0*
T0*"
_output_shapes
:
h
Reshape_8/shapeConst*
dtype0*%
valueB"����         *
_output_shapes
:
q
	Reshape_8Reshapetranspose_3Reshape_8/shape*
Tshape0*
T0*&
_output_shapes
:
c
summary_conv2_0/tagConst*
dtype0* 
valueB Bsummary_conv2_0*
_output_shapes
: 
�
summary_conv2_0ImageSummarysummary_conv2_0/tag	Reshape_8*

max_images*
	bad_colorB:�  �*
T0*
_output_shapes
: 
f
Slice_4/beginConst*
dtype0*%
valueB"                *
_output_shapes
:
e
Slice_4/sizeConst*
dtype0*%
valueB"   ��������   *
_output_shapes
:
v
Slice_4Slice	MaxPool_1Slice_4/beginSlice_4/size*
Index0*
T0*&
_output_shapes
:
a
Const_11Const*
dtype0*%
valueB"             *
_output_shapes
:
]
Min_4MinSlice_4Const_11*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
M
sub_4SubSlice_4Min_4*
T0*&
_output_shapes
:
a
Const_12Const*
dtype0*%
valueB"             *
_output_shapes
:
[
Max_4Maxsub_4Const_12*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
L
mul_4/yConst*
dtype0*
valueB
 *  C*
_output_shapes
: 
=
mul_4MulMax_4mul_4/y*
T0*
_output_shapes
: 
S
	truediv_4RealDivsub_4mul_4*
T0*&
_output_shapes
:
d
Reshape_9/shapeConst*
dtype0*!
valueB"         *
_output_shapes
:
k
	Reshape_9Reshape	truediv_4Reshape_9/shape*
Tshape0*
T0*"
_output_shapes
:
e
transpose_4/permConst*
dtype0*!
valueB"          *
_output_shapes
:
o
transpose_4	Transpose	Reshape_9transpose_4/perm*
Tperm0*
T0*"
_output_shapes
:
i
Reshape_10/shapeConst*
dtype0*%
valueB"����         *
_output_shapes
:
s

Reshape_10Reshapetranspose_4Reshape_10/shape*
Tshape0*
T0*&
_output_shapes
:
c
summary_pool2_0/tagConst*
dtype0* 
valueB Bsummary_pool2_0*
_output_shapes
: 
�
summary_pool2_0ImageSummarysummary_pool2_0/tag
Reshape_10*

max_images*
	bad_colorB:�  �*
T0*
_output_shapes
: 
F
RankConst*
dtype0*
value	B :*
_output_shapes
: 
V
ShapeConst*
dtype0*
valueB"      *
_output_shapes
:
H
Rank_1Const*
dtype0*
value	B :*
_output_shapes
: 
X
Shape_1Const*
dtype0*
valueB"      *
_output_shapes
:
G
Sub/yConst*
dtype0*
value	B :*
_output_shapes
: 
:
SubSubRank_1Sub/y*
T0*
_output_shapes
: 
T
Slice_5/beginPackSub*
N*

axis *
T0*
_output_shapes
:
V
Slice_5/sizeConst*
dtype0*
valueB:*
_output_shapes
:
h
Slice_5SliceShape_1Slice_5/beginSlice_5/size*
Index0*
T0*
_output_shapes
:
b
concat/values_0Const*
dtype0*
valueB:
���������*
_output_shapes
:
M
concat/axisConst*
dtype0*
value	B : *
_output_shapes
: 
s
concatConcatV2concat/values_0Slice_5concat/axis*
N*

Tidx0*
T0*
_output_shapes
:
[

Reshape_11Reshapeadd_1concat*
Tshape0*
T0*
_output_shapes

:
H
Rank_2Const*
dtype0*
value	B :*
_output_shapes
: 
X
Shape_2Const*
dtype0*
valueB"      *
_output_shapes
:
I
Sub_1/yConst*
dtype0*
value	B :*
_output_shapes
: 
>
Sub_1SubRank_2Sub_1/y*
T0*
_output_shapes
: 
V
Slice_6/beginPackSub_1*
N*

axis *
T0*
_output_shapes
:
V
Slice_6/sizeConst*
dtype0*
valueB:*
_output_shapes
:
h
Slice_6SliceShape_2Slice_6/beginSlice_6/size*
Index0*
T0*
_output_shapes
:
d
concat_1/values_0Const*
dtype0*
valueB:
���������*
_output_shapes
:
O
concat_1/axisConst*
dtype0*
value	B : *
_output_shapes
: 
y
concat_1ConcatV2concat_1/values_0Slice_6concat_1/axis*
N*

Tidx0*
T0*
_output_shapes
:
e

Reshape_12ReshapePlaceholder_1concat_1*
Tshape0*
T0*
_output_shapes

:
�
SoftmaxCrossEntropyWithLogitsSoftmaxCrossEntropyWithLogits
Reshape_11
Reshape_12*
T0*$
_output_shapes
::
I
Sub_2/yConst*
dtype0*
value	B :*
_output_shapes
: 
<
Sub_2SubRankSub_2/y*
T0*
_output_shapes
: 
W
Slice_7/beginConst*
dtype0*
valueB: *
_output_shapes
:
U
Slice_7/sizePackSub_2*
N*

axis *
T0*
_output_shapes
:
o
Slice_7SliceShapeSlice_7/beginSlice_7/size*
Index0*
T0*#
_output_shapes
:���������
p

Reshape_13ReshapeSoftmaxCrossEntropyWithLogitsSlice_7*
Tshape0*
T0*
_output_shapes
:
R
Const_13Const*
dtype0*
valueB: *
_output_shapes
:
`
MeanMean
Reshape_13Const_13*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
N
	loss/tagsConst*
dtype0*
valueB
 Bloss*
_output_shapes
: 
G
lossScalarSummary	loss/tagsMean*
T0*
_output_shapes
: 
R
gradients/ShapeConst*
dtype0*
valueB *
_output_shapes
: 
T
gradients/ConstConst*
dtype0*
valueB
 *  �?*
_output_shapes
: 
Y
gradients/FillFillgradients/Shapegradients/Const*
T0*
_output_shapes
: 
k
!gradients/Mean_grad/Reshape/shapeConst*
dtype0*
valueB:*
_output_shapes
:
�
gradients/Mean_grad/ReshapeReshapegradients/Fill!gradients/Mean_grad/Reshape/shape*
Tshape0*
T0*
_output_shapes
:
l
"gradients/Mean_grad/Tile/multiplesConst*
dtype0*
valueB:*
_output_shapes
:
�
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshape"gradients/Mean_grad/Tile/multiples*

Tmultiples0*
T0*
_output_shapes
:
c
gradients/Mean_grad/ShapeConst*
dtype0*
valueB:*
_output_shapes
:
^
gradients/Mean_grad/Shape_1Const*
dtype0*
valueB *
_output_shapes
: 
�
gradients/Mean_grad/ConstConst*,
_class"
 loc:@gradients/Mean_grad/Shape*
dtype0*
valueB: *
_output_shapes
:
�
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shapegradients/Mean_grad/Const*,
_class"
 loc:@gradients/Mean_grad/Shape*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
�
gradients/Mean_grad/Const_1Const*,
_class"
 loc:@gradients/Mean_grad/Shape*
dtype0*
valueB: *
_output_shapes
:
�
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_1gradients/Mean_grad/Const_1*,
_class"
 loc:@gradients/Mean_grad/Shape*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
�
gradients/Mean_grad/Maximum/yConst*,
_class"
 loc:@gradients/Mean_grad/Shape*
dtype0*
value	B :*
_output_shapes
: 
�
gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*,
_class"
 loc:@gradients/Mean_grad/Shape*
T0*
_output_shapes
: 
�
gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum*,
_class"
 loc:@gradients/Mean_grad/Shape*
T0*
_output_shapes
: 
n
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv*

SrcT0*

DstT0*
_output_shapes
: 

gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*
T0*
_output_shapes
:
i
gradients/Reshape_13_grad/ShapeConst*
dtype0*
valueB:*
_output_shapes
:
�
!gradients/Reshape_13_grad/ReshapeReshapegradients/Mean_grad/truedivgradients/Reshape_13_grad/Shape*
Tshape0*
T0*
_output_shapes
:
k
gradients/zeros_like	ZerosLikeSoftmaxCrossEntropyWithLogits:1*
T0*
_output_shapes

:
�
;gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dimConst*
dtype0*
valueB :
���������*
_output_shapes
: 
�
7gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims
ExpandDims!gradients/Reshape_13_grad/Reshape;gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim*

Tdim0*
T0*
_output_shapes

:
�
0gradients/SoftmaxCrossEntropyWithLogits_grad/mulMul7gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDimsSoftmaxCrossEntropyWithLogits:1*
T0*
_output_shapes

:
p
gradients/Reshape_11_grad/ShapeConst*
dtype0*
valueB"      *
_output_shapes
:
�
!gradients/Reshape_11_grad/ReshapeReshape0gradients/SoftmaxCrossEntropyWithLogits_grad/mulgradients/Reshape_11_grad/Shape*
Tshape0*
T0*
_output_shapes

:
k
gradients/add_1_grad/ShapeConst*
dtype0*
valueB"      *
_output_shapes
:
f
gradients/add_1_grad/Shape_1Const*
dtype0*
valueB:*
_output_shapes
:
�
*gradients/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_1_grad/Shapegradients/add_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/add_1_grad/SumSum!gradients/Reshape_11_grad/Reshape*gradients/add_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
gradients/add_1_grad/ReshapeReshapegradients/add_1_grad/Sumgradients/add_1_grad/Shape*
Tshape0*
T0*
_output_shapes

:
�
gradients/add_1_grad/Sum_1Sum!gradients/Reshape_11_grad/Reshape,gradients/add_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
gradients/add_1_grad/Reshape_1Reshapegradients/add_1_grad/Sum_1gradients/add_1_grad/Shape_1*
Tshape0*
T0*
_output_shapes
:
�
gradients/MatMul_1_grad/MatMulMatMulgradients/add_1_grad/ReshapeVariable_6/read*
transpose_a( *
T0*
transpose_b(*
_output_shapes
:	�
�
 gradients/MatMul_1_grad/MatMul_1MatMulRelu_2gradients/add_1_grad/Reshape*
transpose_a(*
T0*
transpose_b( *
_output_shapes
:	�
|
gradients/Relu_2_grad/ReluGradReluGradgradients/MatMul_1_grad/MatMulRelu_2*
T0*
_output_shapes
:	�
i
gradients/add_grad/ShapeConst*
dtype0*
valueB"      *
_output_shapes
:
e
gradients/add_grad/Shape_1Const*
dtype0*
valueB:�*
_output_shapes
:
�
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/add_grad/SumSumgradients/Relu_2_grad/ReluGrad(gradients/add_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*
Tshape0*
T0*
_output_shapes
:	�
�
gradients/add_grad/Sum_1Sumgradients/Relu_2_grad/ReluGrad*gradients/add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
Tshape0*
T0*
_output_shapes	
:�
�
gradients/MatMul_grad/MatMulMatMulgradients/add_grad/ReshapeVariable_4/read*
transpose_a( *
T0*
transpose_b(*
_output_shapes
:	�
�
gradients/MatMul_grad/MatMul_1MatMulReshapegradients/add_grad/Reshape*
transpose_a(*
T0*
transpose_b( * 
_output_shapes
:
��
u
gradients/Reshape_grad/ShapeConst*
dtype0*%
valueB"         @   *
_output_shapes
:
�
gradients/Reshape_grad/ReshapeReshapegradients/MatMul_grad/MatMulgradients/Reshape_grad/Shape*
Tshape0*
T0*&
_output_shapes
:@
�
$gradients/MaxPool_1_grad/MaxPoolGradMaxPoolGradRelu_1	MaxPool_1gradients/Reshape_grad/Reshape*
data_formatNHWC*
T0*
strides
*
paddingSAME*
ksize
*&
_output_shapes
:@
�
gradients/Relu_1_grad/ReluGradReluGrad$gradients/MaxPool_1_grad/MaxPoolGradRelu_1*
T0*&
_output_shapes
:@
�
$gradients/BiasAdd_1_grad/BiasAddGradBiasAddGradgradients/Relu_1_grad/ReluGrad*
data_formatNHWC*
T0*
_output_shapes
:@
�
gradients/Conv2D_1_grad/ShapeNShapeNMaxPoolVariable_2/read*
N*
out_type0*
T0* 
_output_shapes
::
�
+gradients/Conv2D_1_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_1_grad/ShapeNVariable_2/readgradients/Relu_1_grad/ReluGrad*
data_formatNHWC*
T0*
strides
*
paddingSAME*
use_cudnn_on_gpu(*&
_output_shapes
: 
�
,gradients/Conv2D_1_grad/Conv2DBackpropFilterConv2DBackpropFilterMaxPool gradients/Conv2D_1_grad/ShapeN:1gradients/Relu_1_grad/ReluGrad*
data_formatNHWC*
T0*
strides
*
paddingSAME*
use_cudnn_on_gpu(*&
_output_shapes
: @
�
"gradients/MaxPool_grad/MaxPoolGradMaxPoolGradReluMaxPool+gradients/Conv2D_1_grad/Conv2DBackpropInput*
data_formatNHWC*
T0*
strides
*
paddingSAME*
ksize
*&
_output_shapes
: 
�
gradients/Relu_grad/ReluGradReluGrad"gradients/MaxPool_grad/MaxPoolGradRelu*
T0*&
_output_shapes
: 
�
"gradients/BiasAdd_grad/BiasAddGradBiasAddGradgradients/Relu_grad/ReluGrad*
data_formatNHWC*
T0*
_output_shapes
: 
�
gradients/Conv2D_grad/ShapeNShapeNPlaceholderVariable/read*
N*
out_type0*
T0* 
_output_shapes
::
�
)gradients/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_grad/ShapeNVariable/readgradients/Relu_grad/ReluGrad*
data_formatNHWC*
T0*
strides
*
paddingSAME*
use_cudnn_on_gpu(*&
_output_shapes
:
�
*gradients/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterPlaceholdergradients/Conv2D_grad/ShapeN:1gradients/Relu_grad/ReluGrad*
data_formatNHWC*
T0*
strides
*
paddingSAME*
use_cudnn_on_gpu(*&
_output_shapes
: 
�
global_norm/L2LossL2Loss*gradients/Conv2D_grad/Conv2DBackpropFilter*=
_class3
1/loc:@gradients/Conv2D_grad/Conv2DBackpropFilter*
T0*
_output_shapes
: 
g
global_norm/stackPackglobal_norm/L2Loss*
N*

axis *
T0*
_output_shapes
:
[
global_norm/ConstConst*
dtype0*
valueB: *
_output_shapes
:
z
global_norm/SumSumglobal_norm/stackglobal_norm/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
X
global_norm/Const_1Const*
dtype0*
valueB
 *   @*
_output_shapes
: 
]
global_norm/mulMulglobal_norm/Sumglobal_norm/Const_1*
T0*
_output_shapes
: 
Q
global_norm/global_normSqrtglobal_norm/mul*
T0*
_output_shapes
: 
`
conv1_weights/tagsConst*
dtype0*
valueB Bconv1_weights*
_output_shapes
: 
l
conv1_weightsScalarSummaryconv1_weights/tagsglobal_norm/global_norm*
T0*
_output_shapes
: 
�
global_norm_1/L2LossL2Loss"gradients/BiasAdd_grad/BiasAddGrad*5
_class+
)'loc:@gradients/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
: 
k
global_norm_1/stackPackglobal_norm_1/L2Loss*
N*

axis *
T0*
_output_shapes
:
]
global_norm_1/ConstConst*
dtype0*
valueB: *
_output_shapes
:
�
global_norm_1/SumSumglobal_norm_1/stackglobal_norm_1/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
Z
global_norm_1/Const_1Const*
dtype0*
valueB
 *   @*
_output_shapes
: 
c
global_norm_1/mulMulglobal_norm_1/Sumglobal_norm_1/Const_1*
T0*
_output_shapes
: 
U
global_norm_1/global_normSqrtglobal_norm_1/mul*
T0*
_output_shapes
: 
^
conv1_biases/tagsConst*
dtype0*
valueB Bconv1_biases*
_output_shapes
: 
l
conv1_biasesScalarSummaryconv1_biases/tagsglobal_norm_1/global_norm*
T0*
_output_shapes
: 
�
global_norm_2/L2LossL2Loss,gradients/Conv2D_1_grad/Conv2DBackpropFilter*?
_class5
31loc:@gradients/Conv2D_1_grad/Conv2DBackpropFilter*
T0*
_output_shapes
: 
k
global_norm_2/stackPackglobal_norm_2/L2Loss*
N*

axis *
T0*
_output_shapes
:
]
global_norm_2/ConstConst*
dtype0*
valueB: *
_output_shapes
:
�
global_norm_2/SumSumglobal_norm_2/stackglobal_norm_2/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
Z
global_norm_2/Const_1Const*
dtype0*
valueB
 *   @*
_output_shapes
: 
c
global_norm_2/mulMulglobal_norm_2/Sumglobal_norm_2/Const_1*
T0*
_output_shapes
: 
U
global_norm_2/global_normSqrtglobal_norm_2/mul*
T0*
_output_shapes
: 
`
conv2_weights/tagsConst*
dtype0*
valueB Bconv2_weights*
_output_shapes
: 
n
conv2_weightsScalarSummaryconv2_weights/tagsglobal_norm_2/global_norm*
T0*
_output_shapes
: 
�
global_norm_3/L2LossL2Loss$gradients/BiasAdd_1_grad/BiasAddGrad*7
_class-
+)loc:@gradients/BiasAdd_1_grad/BiasAddGrad*
T0*
_output_shapes
: 
k
global_norm_3/stackPackglobal_norm_3/L2Loss*
N*

axis *
T0*
_output_shapes
:
]
global_norm_3/ConstConst*
dtype0*
valueB: *
_output_shapes
:
�
global_norm_3/SumSumglobal_norm_3/stackglobal_norm_3/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
Z
global_norm_3/Const_1Const*
dtype0*
valueB
 *   @*
_output_shapes
: 
c
global_norm_3/mulMulglobal_norm_3/Sumglobal_norm_3/Const_1*
T0*
_output_shapes
: 
U
global_norm_3/global_normSqrtglobal_norm_3/mul*
T0*
_output_shapes
: 
^
conv2_biases/tagsConst*
dtype0*
valueB Bconv2_biases*
_output_shapes
: 
l
conv2_biasesScalarSummaryconv2_biases/tagsglobal_norm_3/global_norm*
T0*
_output_shapes
: 
�
global_norm_4/L2LossL2Lossgradients/MatMul_grad/MatMul_1*1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1*
T0*
_output_shapes
: 
k
global_norm_4/stackPackglobal_norm_4/L2Loss*
N*

axis *
T0*
_output_shapes
:
]
global_norm_4/ConstConst*
dtype0*
valueB: *
_output_shapes
:
�
global_norm_4/SumSumglobal_norm_4/stackglobal_norm_4/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
Z
global_norm_4/Const_1Const*
dtype0*
valueB
 *   @*
_output_shapes
: 
c
global_norm_4/mulMulglobal_norm_4/Sumglobal_norm_4/Const_1*
T0*
_output_shapes
: 
U
global_norm_4/global_normSqrtglobal_norm_4/mul*
T0*
_output_shapes
: 
\
fc1_weights/tagsConst*
dtype0*
valueB Bfc1_weights*
_output_shapes
: 
j
fc1_weightsScalarSummaryfc1_weights/tagsglobal_norm_4/global_norm*
T0*
_output_shapes
: 
�
global_norm_5/L2LossL2Lossgradients/add_grad/Reshape_1*/
_class%
#!loc:@gradients/add_grad/Reshape_1*
T0*
_output_shapes
: 
k
global_norm_5/stackPackglobal_norm_5/L2Loss*
N*

axis *
T0*
_output_shapes
:
]
global_norm_5/ConstConst*
dtype0*
valueB: *
_output_shapes
:
�
global_norm_5/SumSumglobal_norm_5/stackglobal_norm_5/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
Z
global_norm_5/Const_1Const*
dtype0*
valueB
 *   @*
_output_shapes
: 
c
global_norm_5/mulMulglobal_norm_5/Sumglobal_norm_5/Const_1*
T0*
_output_shapes
: 
U
global_norm_5/global_normSqrtglobal_norm_5/mul*
T0*
_output_shapes
: 
Z
fc1_biases/tagsConst*
dtype0*
valueB B
fc1_biases*
_output_shapes
: 
h

fc1_biasesScalarSummaryfc1_biases/tagsglobal_norm_5/global_norm*
T0*
_output_shapes
: 
�
global_norm_6/L2LossL2Loss gradients/MatMul_1_grad/MatMul_1*3
_class)
'%loc:@gradients/MatMul_1_grad/MatMul_1*
T0*
_output_shapes
: 
k
global_norm_6/stackPackglobal_norm_6/L2Loss*
N*

axis *
T0*
_output_shapes
:
]
global_norm_6/ConstConst*
dtype0*
valueB: *
_output_shapes
:
�
global_norm_6/SumSumglobal_norm_6/stackglobal_norm_6/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
Z
global_norm_6/Const_1Const*
dtype0*
valueB
 *   @*
_output_shapes
: 
c
global_norm_6/mulMulglobal_norm_6/Sumglobal_norm_6/Const_1*
T0*
_output_shapes
: 
U
global_norm_6/global_normSqrtglobal_norm_6/mul*
T0*
_output_shapes
: 
\
fc2_weights/tagsConst*
dtype0*
valueB Bfc2_weights*
_output_shapes
: 
j
fc2_weightsScalarSummaryfc2_weights/tagsglobal_norm_6/global_norm*
T0*
_output_shapes
: 
�
global_norm_7/L2LossL2Lossgradients/add_1_grad/Reshape_1*1
_class'
%#loc:@gradients/add_1_grad/Reshape_1*
T0*
_output_shapes
: 
k
global_norm_7/stackPackglobal_norm_7/L2Loss*
N*

axis *
T0*
_output_shapes
:
]
global_norm_7/ConstConst*
dtype0*
valueB: *
_output_shapes
:
�
global_norm_7/SumSumglobal_norm_7/stackglobal_norm_7/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
Z
global_norm_7/Const_1Const*
dtype0*
valueB
 *   @*
_output_shapes
: 
c
global_norm_7/mulMulglobal_norm_7/Sumglobal_norm_7/Const_1*
T0*
_output_shapes
: 
U
global_norm_7/global_normSqrtglobal_norm_7/mul*
T0*
_output_shapes
: 
Z
fc2_biases/tagsConst*
dtype0*
valueB B
fc2_biases*
_output_shapes
: 
h

fc2_biasesScalarSummaryfc2_biases/tagsglobal_norm_7/global_norm*
T0*
_output_shapes
: 
B
L2LossL2LossVariable_4/read*
T0*
_output_shapes
: 
D
L2Loss_1L2LossVariable_5/read*
T0*
_output_shapes
: 
?
add_2AddL2LossL2Loss_1*
T0*
_output_shapes
: 
D
L2Loss_2L2LossVariable_6/read*
T0*
_output_shapes
: 
>
add_3Addadd_2L2Loss_2*
T0*
_output_shapes
: 
D
L2Loss_3L2LossVariable_7/read*
T0*
_output_shapes
: 
>
add_4Addadd_3L2Loss_3*
T0*
_output_shapes
: 
L
mul_5/xConst*
dtype0*
valueB
 *o:*
_output_shapes
: 
=
mul_5Mulmul_5/xadd_4*
T0*
_output_shapes
: 
:
add_5AddMeanmul_5*
T0*
_output_shapes
: 
Z
Variable_8/initial_valueConst*
dtype0*
value	B : *
_output_shapes
: 
n

Variable_8
VariableV2*
shape: *
_output_shapes
: *
shared_name *
	container *
dtype0
�
Variable_8/AssignAssign
Variable_8Variable_8/initial_value*
validate_shape(*
_class
loc:@Variable_8*
use_locking(*
T0*
_output_shapes
: 
g
Variable_8/readIdentity
Variable_8*
_class
loc:@Variable_8*
T0*
_output_shapes
: 
I
mul_6/yConst*
dtype0*
value	B :*
_output_shapes
: 
G
mul_6MulVariable_8/readmul_6/y*
T0*
_output_shapes
: 
c
ExponentialDecay/learning_rateConst*
dtype0*
valueB
 *
�#<*
_output_shapes
: 
T
ExponentialDecay/CastCastmul_6*

SrcT0*

DstT0*
_output_shapes
: 
]
ExponentialDecay/Cast_1/xConst*
dtype0*
valueB	 :��=*
_output_shapes
: 
j
ExponentialDecay/Cast_1CastExponentialDecay/Cast_1/x*

SrcT0*

DstT0*
_output_shapes
: 
^
ExponentialDecay/Cast_2/xConst*
dtype0*
valueB
 *33s?*
_output_shapes
: 
t
ExponentialDecay/truedivRealDivExponentialDecay/CastExponentialDecay/Cast_1*
T0*
_output_shapes
: 
Z
ExponentialDecay/FloorFloorExponentialDecay/truediv*
T0*
_output_shapes
: 
o
ExponentialDecay/PowPowExponentialDecay/Cast_2/xExponentialDecay/Floor*
T0*
_output_shapes
: 
n
ExponentialDecayMulExponentialDecay/learning_rateExponentialDecay/Pow*
T0*
_output_shapes
: 
`
learning_rate/tagsConst*
dtype0*
valueB Blearning_rate*
_output_shapes
: 
e
learning_rateScalarSummarylearning_rate/tagsExponentialDecay*
T0*
_output_shapes
: 
T
gradients_1/ShapeConst*
dtype0*
valueB *
_output_shapes
: 
V
gradients_1/ConstConst*
dtype0*
valueB
 *  �?*
_output_shapes
: 
_
gradients_1/FillFillgradients_1/Shapegradients_1/Const*
T0*
_output_shapes
: 
_
gradients_1/add_5_grad/ShapeConst*
dtype0*
valueB *
_output_shapes
: 
a
gradients_1/add_5_grad/Shape_1Const*
dtype0*
valueB *
_output_shapes
: 
�
,gradients_1/add_5_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/add_5_grad/Shapegradients_1/add_5_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients_1/add_5_grad/SumSumgradients_1/Fill,gradients_1/add_5_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
gradients_1/add_5_grad/ReshapeReshapegradients_1/add_5_grad/Sumgradients_1/add_5_grad/Shape*
Tshape0*
T0*
_output_shapes
: 
�
gradients_1/add_5_grad/Sum_1Sumgradients_1/Fill.gradients_1/add_5_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
 gradients_1/add_5_grad/Reshape_1Reshapegradients_1/add_5_grad/Sum_1gradients_1/add_5_grad/Shape_1*
Tshape0*
T0*
_output_shapes
: 
s
'gradients_1/add_5_grad/tuple/group_depsNoOp^gradients_1/add_5_grad/Reshape!^gradients_1/add_5_grad/Reshape_1
�
/gradients_1/add_5_grad/tuple/control_dependencyIdentitygradients_1/add_5_grad/Reshape(^gradients_1/add_5_grad/tuple/group_deps*1
_class'
%#loc:@gradients_1/add_5_grad/Reshape*
T0*
_output_shapes
: 
�
1gradients_1/add_5_grad/tuple/control_dependency_1Identity gradients_1/add_5_grad/Reshape_1(^gradients_1/add_5_grad/tuple/group_deps*3
_class)
'%loc:@gradients_1/add_5_grad/Reshape_1*
T0*
_output_shapes
: 
m
#gradients_1/Mean_grad/Reshape/shapeConst*
dtype0*
valueB:*
_output_shapes
:
�
gradients_1/Mean_grad/ReshapeReshape/gradients_1/add_5_grad/tuple/control_dependency#gradients_1/Mean_grad/Reshape/shape*
Tshape0*
T0*
_output_shapes
:
n
$gradients_1/Mean_grad/Tile/multiplesConst*
dtype0*
valueB:*
_output_shapes
:
�
gradients_1/Mean_grad/TileTilegradients_1/Mean_grad/Reshape$gradients_1/Mean_grad/Tile/multiples*

Tmultiples0*
T0*
_output_shapes
:
e
gradients_1/Mean_grad/ShapeConst*
dtype0*
valueB:*
_output_shapes
:
`
gradients_1/Mean_grad/Shape_1Const*
dtype0*
valueB *
_output_shapes
: 
�
gradients_1/Mean_grad/ConstConst*.
_class$
" loc:@gradients_1/Mean_grad/Shape*
dtype0*
valueB: *
_output_shapes
:
�
gradients_1/Mean_grad/ProdProdgradients_1/Mean_grad/Shapegradients_1/Mean_grad/Const*.
_class$
" loc:@gradients_1/Mean_grad/Shape*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
�
gradients_1/Mean_grad/Const_1Const*.
_class$
" loc:@gradients_1/Mean_grad/Shape*
dtype0*
valueB: *
_output_shapes
:
�
gradients_1/Mean_grad/Prod_1Prodgradients_1/Mean_grad/Shape_1gradients_1/Mean_grad/Const_1*.
_class$
" loc:@gradients_1/Mean_grad/Shape*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
�
gradients_1/Mean_grad/Maximum/yConst*.
_class$
" loc:@gradients_1/Mean_grad/Shape*
dtype0*
value	B :*
_output_shapes
: 
�
gradients_1/Mean_grad/MaximumMaximumgradients_1/Mean_grad/Prod_1gradients_1/Mean_grad/Maximum/y*.
_class$
" loc:@gradients_1/Mean_grad/Shape*
T0*
_output_shapes
: 
�
gradients_1/Mean_grad/floordivFloorDivgradients_1/Mean_grad/Prodgradients_1/Mean_grad/Maximum*.
_class$
" loc:@gradients_1/Mean_grad/Shape*
T0*
_output_shapes
: 
r
gradients_1/Mean_grad/CastCastgradients_1/Mean_grad/floordiv*

SrcT0*

DstT0*
_output_shapes
: 
�
gradients_1/Mean_grad/truedivRealDivgradients_1/Mean_grad/Tilegradients_1/Mean_grad/Cast*
T0*
_output_shapes
:
_
gradients_1/mul_5_grad/ShapeConst*
dtype0*
valueB *
_output_shapes
: 
a
gradients_1/mul_5_grad/Shape_1Const*
dtype0*
valueB *
_output_shapes
: 
�
,gradients_1/mul_5_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/mul_5_grad/Shapegradients_1/mul_5_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
|
gradients_1/mul_5_grad/mulMul1gradients_1/add_5_grad/tuple/control_dependency_1add_4*
T0*
_output_shapes
: 
�
gradients_1/mul_5_grad/SumSumgradients_1/mul_5_grad/mul,gradients_1/mul_5_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
gradients_1/mul_5_grad/ReshapeReshapegradients_1/mul_5_grad/Sumgradients_1/mul_5_grad/Shape*
Tshape0*
T0*
_output_shapes
: 
�
gradients_1/mul_5_grad/mul_1Mulmul_5/x1gradients_1/add_5_grad/tuple/control_dependency_1*
T0*
_output_shapes
: 
�
gradients_1/mul_5_grad/Sum_1Sumgradients_1/mul_5_grad/mul_1.gradients_1/mul_5_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
 gradients_1/mul_5_grad/Reshape_1Reshapegradients_1/mul_5_grad/Sum_1gradients_1/mul_5_grad/Shape_1*
Tshape0*
T0*
_output_shapes
: 
s
'gradients_1/mul_5_grad/tuple/group_depsNoOp^gradients_1/mul_5_grad/Reshape!^gradients_1/mul_5_grad/Reshape_1
�
/gradients_1/mul_5_grad/tuple/control_dependencyIdentitygradients_1/mul_5_grad/Reshape(^gradients_1/mul_5_grad/tuple/group_deps*1
_class'
%#loc:@gradients_1/mul_5_grad/Reshape*
T0*
_output_shapes
: 
�
1gradients_1/mul_5_grad/tuple/control_dependency_1Identity gradients_1/mul_5_grad/Reshape_1(^gradients_1/mul_5_grad/tuple/group_deps*3
_class)
'%loc:@gradients_1/mul_5_grad/Reshape_1*
T0*
_output_shapes
: 
k
!gradients_1/Reshape_13_grad/ShapeConst*
dtype0*
valueB:*
_output_shapes
:
�
#gradients_1/Reshape_13_grad/ReshapeReshapegradients_1/Mean_grad/truediv!gradients_1/Reshape_13_grad/Shape*
Tshape0*
T0*
_output_shapes
:
_
gradients_1/add_4_grad/ShapeConst*
dtype0*
valueB *
_output_shapes
: 
a
gradients_1/add_4_grad/Shape_1Const*
dtype0*
valueB *
_output_shapes
: 
�
,gradients_1/add_4_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/add_4_grad/Shapegradients_1/add_4_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients_1/add_4_grad/SumSum1gradients_1/mul_5_grad/tuple/control_dependency_1,gradients_1/add_4_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
gradients_1/add_4_grad/ReshapeReshapegradients_1/add_4_grad/Sumgradients_1/add_4_grad/Shape*
Tshape0*
T0*
_output_shapes
: 
�
gradients_1/add_4_grad/Sum_1Sum1gradients_1/mul_5_grad/tuple/control_dependency_1.gradients_1/add_4_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
 gradients_1/add_4_grad/Reshape_1Reshapegradients_1/add_4_grad/Sum_1gradients_1/add_4_grad/Shape_1*
Tshape0*
T0*
_output_shapes
: 
s
'gradients_1/add_4_grad/tuple/group_depsNoOp^gradients_1/add_4_grad/Reshape!^gradients_1/add_4_grad/Reshape_1
�
/gradients_1/add_4_grad/tuple/control_dependencyIdentitygradients_1/add_4_grad/Reshape(^gradients_1/add_4_grad/tuple/group_deps*1
_class'
%#loc:@gradients_1/add_4_grad/Reshape*
T0*
_output_shapes
: 
�
1gradients_1/add_4_grad/tuple/control_dependency_1Identity gradients_1/add_4_grad/Reshape_1(^gradients_1/add_4_grad/tuple/group_deps*3
_class)
'%loc:@gradients_1/add_4_grad/Reshape_1*
T0*
_output_shapes
: 
m
gradients_1/zeros_like	ZerosLikeSoftmaxCrossEntropyWithLogits:1*
T0*
_output_shapes

:
�
=gradients_1/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dimConst*
dtype0*
valueB :
���������*
_output_shapes
: 
�
9gradients_1/SoftmaxCrossEntropyWithLogits_grad/ExpandDims
ExpandDims#gradients_1/Reshape_13_grad/Reshape=gradients_1/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim*

Tdim0*
T0*
_output_shapes

:
�
2gradients_1/SoftmaxCrossEntropyWithLogits_grad/mulMul9gradients_1/SoftmaxCrossEntropyWithLogits_grad/ExpandDimsSoftmaxCrossEntropyWithLogits:1*
T0*
_output_shapes

:
_
gradients_1/add_3_grad/ShapeConst*
dtype0*
valueB *
_output_shapes
: 
a
gradients_1/add_3_grad/Shape_1Const*
dtype0*
valueB *
_output_shapes
: 
�
,gradients_1/add_3_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/add_3_grad/Shapegradients_1/add_3_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients_1/add_3_grad/SumSum/gradients_1/add_4_grad/tuple/control_dependency,gradients_1/add_3_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
gradients_1/add_3_grad/ReshapeReshapegradients_1/add_3_grad/Sumgradients_1/add_3_grad/Shape*
Tshape0*
T0*
_output_shapes
: 
�
gradients_1/add_3_grad/Sum_1Sum/gradients_1/add_4_grad/tuple/control_dependency.gradients_1/add_3_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
 gradients_1/add_3_grad/Reshape_1Reshapegradients_1/add_3_grad/Sum_1gradients_1/add_3_grad/Shape_1*
Tshape0*
T0*
_output_shapes
: 
s
'gradients_1/add_3_grad/tuple/group_depsNoOp^gradients_1/add_3_grad/Reshape!^gradients_1/add_3_grad/Reshape_1
�
/gradients_1/add_3_grad/tuple/control_dependencyIdentitygradients_1/add_3_grad/Reshape(^gradients_1/add_3_grad/tuple/group_deps*1
_class'
%#loc:@gradients_1/add_3_grad/Reshape*
T0*
_output_shapes
: 
�
1gradients_1/add_3_grad/tuple/control_dependency_1Identity gradients_1/add_3_grad/Reshape_1(^gradients_1/add_3_grad/tuple/group_deps*3
_class)
'%loc:@gradients_1/add_3_grad/Reshape_1*
T0*
_output_shapes
: 
�
gradients_1/L2Loss_3_grad/mulMulVariable_7/read1gradients_1/add_4_grad/tuple/control_dependency_1*
T0*
_output_shapes
:
r
!gradients_1/Reshape_11_grad/ShapeConst*
dtype0*
valueB"      *
_output_shapes
:
�
#gradients_1/Reshape_11_grad/ReshapeReshape2gradients_1/SoftmaxCrossEntropyWithLogits_grad/mul!gradients_1/Reshape_11_grad/Shape*
Tshape0*
T0*
_output_shapes

:
_
gradients_1/add_2_grad/ShapeConst*
dtype0*
valueB *
_output_shapes
: 
a
gradients_1/add_2_grad/Shape_1Const*
dtype0*
valueB *
_output_shapes
: 
�
,gradients_1/add_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/add_2_grad/Shapegradients_1/add_2_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients_1/add_2_grad/SumSum/gradients_1/add_3_grad/tuple/control_dependency,gradients_1/add_2_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
gradients_1/add_2_grad/ReshapeReshapegradients_1/add_2_grad/Sumgradients_1/add_2_grad/Shape*
Tshape0*
T0*
_output_shapes
: 
�
gradients_1/add_2_grad/Sum_1Sum/gradients_1/add_3_grad/tuple/control_dependency.gradients_1/add_2_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
 gradients_1/add_2_grad/Reshape_1Reshapegradients_1/add_2_grad/Sum_1gradients_1/add_2_grad/Shape_1*
Tshape0*
T0*
_output_shapes
: 
s
'gradients_1/add_2_grad/tuple/group_depsNoOp^gradients_1/add_2_grad/Reshape!^gradients_1/add_2_grad/Reshape_1
�
/gradients_1/add_2_grad/tuple/control_dependencyIdentitygradients_1/add_2_grad/Reshape(^gradients_1/add_2_grad/tuple/group_deps*1
_class'
%#loc:@gradients_1/add_2_grad/Reshape*
T0*
_output_shapes
: 
�
1gradients_1/add_2_grad/tuple/control_dependency_1Identity gradients_1/add_2_grad/Reshape_1(^gradients_1/add_2_grad/tuple/group_deps*3
_class)
'%loc:@gradients_1/add_2_grad/Reshape_1*
T0*
_output_shapes
: 
�
gradients_1/L2Loss_2_grad/mulMulVariable_6/read1gradients_1/add_3_grad/tuple/control_dependency_1*
T0*
_output_shapes
:	�
m
gradients_1/add_1_grad/ShapeConst*
dtype0*
valueB"      *
_output_shapes
:
h
gradients_1/add_1_grad/Shape_1Const*
dtype0*
valueB:*
_output_shapes
:
�
,gradients_1/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/add_1_grad/Shapegradients_1/add_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients_1/add_1_grad/SumSum#gradients_1/Reshape_11_grad/Reshape,gradients_1/add_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
gradients_1/add_1_grad/ReshapeReshapegradients_1/add_1_grad/Sumgradients_1/add_1_grad/Shape*
Tshape0*
T0*
_output_shapes

:
�
gradients_1/add_1_grad/Sum_1Sum#gradients_1/Reshape_11_grad/Reshape.gradients_1/add_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
 gradients_1/add_1_grad/Reshape_1Reshapegradients_1/add_1_grad/Sum_1gradients_1/add_1_grad/Shape_1*
Tshape0*
T0*
_output_shapes
:
s
'gradients_1/add_1_grad/tuple/group_depsNoOp^gradients_1/add_1_grad/Reshape!^gradients_1/add_1_grad/Reshape_1
�
/gradients_1/add_1_grad/tuple/control_dependencyIdentitygradients_1/add_1_grad/Reshape(^gradients_1/add_1_grad/tuple/group_deps*1
_class'
%#loc:@gradients_1/add_1_grad/Reshape*
T0*
_output_shapes

:
�
1gradients_1/add_1_grad/tuple/control_dependency_1Identity gradients_1/add_1_grad/Reshape_1(^gradients_1/add_1_grad/tuple/group_deps*3
_class)
'%loc:@gradients_1/add_1_grad/Reshape_1*
T0*
_output_shapes
:
�
gradients_1/L2Loss_grad/mulMulVariable_4/read/gradients_1/add_2_grad/tuple/control_dependency*
T0* 
_output_shapes
:
��
�
gradients_1/L2Loss_1_grad/mulMulVariable_5/read1gradients_1/add_2_grad/tuple/control_dependency_1*
T0*
_output_shapes	
:�
�
 gradients_1/MatMul_1_grad/MatMulMatMul/gradients_1/add_1_grad/tuple/control_dependencyVariable_6/read*
transpose_a( *
T0*
transpose_b(*
_output_shapes
:	�
�
"gradients_1/MatMul_1_grad/MatMul_1MatMulRelu_2/gradients_1/add_1_grad/tuple/control_dependency*
transpose_a(*
T0*
transpose_b( *
_output_shapes
:	�
z
*gradients_1/MatMul_1_grad/tuple/group_depsNoOp!^gradients_1/MatMul_1_grad/MatMul#^gradients_1/MatMul_1_grad/MatMul_1
�
2gradients_1/MatMul_1_grad/tuple/control_dependencyIdentity gradients_1/MatMul_1_grad/MatMul+^gradients_1/MatMul_1_grad/tuple/group_deps*3
_class)
'%loc:@gradients_1/MatMul_1_grad/MatMul*
T0*
_output_shapes
:	�
�
4gradients_1/MatMul_1_grad/tuple/control_dependency_1Identity"gradients_1/MatMul_1_grad/MatMul_1+^gradients_1/MatMul_1_grad/tuple/group_deps*5
_class+
)'loc:@gradients_1/MatMul_1_grad/MatMul_1*
T0*
_output_shapes
:	�
�
gradients_1/AddNAddNgradients_1/L2Loss_3_grad/mul1gradients_1/add_1_grad/tuple/control_dependency_1*
N*0
_class&
$"loc:@gradients_1/L2Loss_3_grad/mul*
T0*
_output_shapes
:
�
 gradients_1/Relu_2_grad/ReluGradReluGrad2gradients_1/MatMul_1_grad/tuple/control_dependencyRelu_2*
T0*
_output_shapes
:	�
�
gradients_1/AddN_1AddNgradients_1/L2Loss_2_grad/mul4gradients_1/MatMul_1_grad/tuple/control_dependency_1*
N*0
_class&
$"loc:@gradients_1/L2Loss_2_grad/mul*
T0*
_output_shapes
:	�
k
gradients_1/add_grad/ShapeConst*
dtype0*
valueB"      *
_output_shapes
:
g
gradients_1/add_grad/Shape_1Const*
dtype0*
valueB:�*
_output_shapes
:
�
*gradients_1/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/add_grad/Shapegradients_1/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients_1/add_grad/SumSum gradients_1/Relu_2_grad/ReluGrad*gradients_1/add_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
gradients_1/add_grad/ReshapeReshapegradients_1/add_grad/Sumgradients_1/add_grad/Shape*
Tshape0*
T0*
_output_shapes
:	�
�
gradients_1/add_grad/Sum_1Sum gradients_1/Relu_2_grad/ReluGrad,gradients_1/add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
gradients_1/add_grad/Reshape_1Reshapegradients_1/add_grad/Sum_1gradients_1/add_grad/Shape_1*
Tshape0*
T0*
_output_shapes	
:�
m
%gradients_1/add_grad/tuple/group_depsNoOp^gradients_1/add_grad/Reshape^gradients_1/add_grad/Reshape_1
�
-gradients_1/add_grad/tuple/control_dependencyIdentitygradients_1/add_grad/Reshape&^gradients_1/add_grad/tuple/group_deps*/
_class%
#!loc:@gradients_1/add_grad/Reshape*
T0*
_output_shapes
:	�
�
/gradients_1/add_grad/tuple/control_dependency_1Identitygradients_1/add_grad/Reshape_1&^gradients_1/add_grad/tuple/group_deps*1
_class'
%#loc:@gradients_1/add_grad/Reshape_1*
T0*
_output_shapes	
:�
�
gradients_1/MatMul_grad/MatMulMatMul-gradients_1/add_grad/tuple/control_dependencyVariable_4/read*
transpose_a( *
T0*
transpose_b(*
_output_shapes
:	�
�
 gradients_1/MatMul_grad/MatMul_1MatMulReshape-gradients_1/add_grad/tuple/control_dependency*
transpose_a(*
T0*
transpose_b( * 
_output_shapes
:
��
t
(gradients_1/MatMul_grad/tuple/group_depsNoOp^gradients_1/MatMul_grad/MatMul!^gradients_1/MatMul_grad/MatMul_1
�
0gradients_1/MatMul_grad/tuple/control_dependencyIdentitygradients_1/MatMul_grad/MatMul)^gradients_1/MatMul_grad/tuple/group_deps*1
_class'
%#loc:@gradients_1/MatMul_grad/MatMul*
T0*
_output_shapes
:	�
�
2gradients_1/MatMul_grad/tuple/control_dependency_1Identity gradients_1/MatMul_grad/MatMul_1)^gradients_1/MatMul_grad/tuple/group_deps*3
_class)
'%loc:@gradients_1/MatMul_grad/MatMul_1*
T0* 
_output_shapes
:
��
�
gradients_1/AddN_2AddNgradients_1/L2Loss_1_grad/mul/gradients_1/add_grad/tuple/control_dependency_1*
N*0
_class&
$"loc:@gradients_1/L2Loss_1_grad/mul*
T0*
_output_shapes	
:�
w
gradients_1/Reshape_grad/ShapeConst*
dtype0*%
valueB"         @   *
_output_shapes
:
�
 gradients_1/Reshape_grad/ReshapeReshape0gradients_1/MatMul_grad/tuple/control_dependencygradients_1/Reshape_grad/Shape*
Tshape0*
T0*&
_output_shapes
:@
�
gradients_1/AddN_3AddNgradients_1/L2Loss_grad/mul2gradients_1/MatMul_grad/tuple/control_dependency_1*
N*.
_class$
" loc:@gradients_1/L2Loss_grad/mul*
T0* 
_output_shapes
:
��
�
&gradients_1/MaxPool_1_grad/MaxPoolGradMaxPoolGradRelu_1	MaxPool_1 gradients_1/Reshape_grad/Reshape*
data_formatNHWC*
T0*
strides
*
paddingSAME*
ksize
*&
_output_shapes
:@
�
 gradients_1/Relu_1_grad/ReluGradReluGrad&gradients_1/MaxPool_1_grad/MaxPoolGradRelu_1*
T0*&
_output_shapes
:@
�
&gradients_1/BiasAdd_1_grad/BiasAddGradBiasAddGrad gradients_1/Relu_1_grad/ReluGrad*
data_formatNHWC*
T0*
_output_shapes
:@

+gradients_1/BiasAdd_1_grad/tuple/group_depsNoOp!^gradients_1/Relu_1_grad/ReluGrad'^gradients_1/BiasAdd_1_grad/BiasAddGrad
�
3gradients_1/BiasAdd_1_grad/tuple/control_dependencyIdentity gradients_1/Relu_1_grad/ReluGrad,^gradients_1/BiasAdd_1_grad/tuple/group_deps*3
_class)
'%loc:@gradients_1/Relu_1_grad/ReluGrad*
T0*&
_output_shapes
:@
�
5gradients_1/BiasAdd_1_grad/tuple/control_dependency_1Identity&gradients_1/BiasAdd_1_grad/BiasAddGrad,^gradients_1/BiasAdd_1_grad/tuple/group_deps*9
_class/
-+loc:@gradients_1/BiasAdd_1_grad/BiasAddGrad*
T0*
_output_shapes
:@
�
 gradients_1/Conv2D_1_grad/ShapeNShapeNMaxPoolVariable_2/read*
N*
out_type0*
T0* 
_output_shapes
::
�
-gradients_1/Conv2D_1_grad/Conv2DBackpropInputConv2DBackpropInput gradients_1/Conv2D_1_grad/ShapeNVariable_2/read3gradients_1/BiasAdd_1_grad/tuple/control_dependency*
data_formatNHWC*
T0*
strides
*
paddingSAME*
use_cudnn_on_gpu(*J
_output_shapes8
6:4������������������������������������
�
.gradients_1/Conv2D_1_grad/Conv2DBackpropFilterConv2DBackpropFilterMaxPool"gradients_1/Conv2D_1_grad/ShapeN:13gradients_1/BiasAdd_1_grad/tuple/control_dependency*
data_formatNHWC*
T0*
strides
*
paddingSAME*
use_cudnn_on_gpu(*J
_output_shapes8
6:4������������������������������������
�
*gradients_1/Conv2D_1_grad/tuple/group_depsNoOp.^gradients_1/Conv2D_1_grad/Conv2DBackpropInput/^gradients_1/Conv2D_1_grad/Conv2DBackpropFilter
�
2gradients_1/Conv2D_1_grad/tuple/control_dependencyIdentity-gradients_1/Conv2D_1_grad/Conv2DBackpropInput+^gradients_1/Conv2D_1_grad/tuple/group_deps*@
_class6
42loc:@gradients_1/Conv2D_1_grad/Conv2DBackpropInput*
T0*&
_output_shapes
: 
�
4gradients_1/Conv2D_1_grad/tuple/control_dependency_1Identity.gradients_1/Conv2D_1_grad/Conv2DBackpropFilter+^gradients_1/Conv2D_1_grad/tuple/group_deps*A
_class7
53loc:@gradients_1/Conv2D_1_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
: @
�
$gradients_1/MaxPool_grad/MaxPoolGradMaxPoolGradReluMaxPool2gradients_1/Conv2D_1_grad/tuple/control_dependency*
data_formatNHWC*
T0*
strides
*
paddingSAME*
ksize
*&
_output_shapes
: 
�
gradients_1/Relu_grad/ReluGradReluGrad$gradients_1/MaxPool_grad/MaxPoolGradRelu*
T0*&
_output_shapes
: 
�
$gradients_1/BiasAdd_grad/BiasAddGradBiasAddGradgradients_1/Relu_grad/ReluGrad*
data_formatNHWC*
T0*
_output_shapes
: 
y
)gradients_1/BiasAdd_grad/tuple/group_depsNoOp^gradients_1/Relu_grad/ReluGrad%^gradients_1/BiasAdd_grad/BiasAddGrad
�
1gradients_1/BiasAdd_grad/tuple/control_dependencyIdentitygradients_1/Relu_grad/ReluGrad*^gradients_1/BiasAdd_grad/tuple/group_deps*1
_class'
%#loc:@gradients_1/Relu_grad/ReluGrad*
T0*&
_output_shapes
: 
�
3gradients_1/BiasAdd_grad/tuple/control_dependency_1Identity$gradients_1/BiasAdd_grad/BiasAddGrad*^gradients_1/BiasAdd_grad/tuple/group_deps*7
_class-
+)loc:@gradients_1/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
: 
�
gradients_1/Conv2D_grad/ShapeNShapeNPlaceholderVariable/read*
N*
out_type0*
T0* 
_output_shapes
::
�
+gradients_1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInputgradients_1/Conv2D_grad/ShapeNVariable/read1gradients_1/BiasAdd_grad/tuple/control_dependency*
data_formatNHWC*
T0*
strides
*
paddingSAME*
use_cudnn_on_gpu(*J
_output_shapes8
6:4������������������������������������
�
,gradients_1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterPlaceholder gradients_1/Conv2D_grad/ShapeN:11gradients_1/BiasAdd_grad/tuple/control_dependency*
data_formatNHWC*
T0*
strides
*
paddingSAME*
use_cudnn_on_gpu(*J
_output_shapes8
6:4������������������������������������
�
(gradients_1/Conv2D_grad/tuple/group_depsNoOp,^gradients_1/Conv2D_grad/Conv2DBackpropInput-^gradients_1/Conv2D_grad/Conv2DBackpropFilter
�
0gradients_1/Conv2D_grad/tuple/control_dependencyIdentity+gradients_1/Conv2D_grad/Conv2DBackpropInput)^gradients_1/Conv2D_grad/tuple/group_deps*>
_class4
20loc:@gradients_1/Conv2D_grad/Conv2DBackpropInput*
T0*&
_output_shapes
:
�
2gradients_1/Conv2D_grad/tuple/control_dependency_1Identity,gradients_1/Conv2D_grad/Conv2DBackpropFilter)^gradients_1/Conv2D_grad/tuple/group_deps*?
_class5
31loc:@gradients_1/Conv2D_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
: 
�
#Variable/Momentum/Initializer/zerosConst*
_class
loc:@Variable*
dtype0*%
valueB *    *&
_output_shapes
: 
�
Variable/Momentum
VariableV2*
shape: *
	container *
_class
loc:@Variable*
dtype0*
shared_name *&
_output_shapes
: 
�
Variable/Momentum/AssignAssignVariable/Momentum#Variable/Momentum/Initializer/zeros*
validate_shape(*
_class
loc:@Variable*
use_locking(*
T0*&
_output_shapes
: 
�
Variable/Momentum/readIdentityVariable/Momentum*
_class
loc:@Variable*
T0*&
_output_shapes
: 
�
%Variable_1/Momentum/Initializer/zerosConst*
_class
loc:@Variable_1*
dtype0*
valueB *    *
_output_shapes
: 
�
Variable_1/Momentum
VariableV2*
shape: *
	container *
_class
loc:@Variable_1*
dtype0*
shared_name *
_output_shapes
: 
�
Variable_1/Momentum/AssignAssignVariable_1/Momentum%Variable_1/Momentum/Initializer/zeros*
validate_shape(*
_class
loc:@Variable_1*
use_locking(*
T0*
_output_shapes
: 
}
Variable_1/Momentum/readIdentityVariable_1/Momentum*
_class
loc:@Variable_1*
T0*
_output_shapes
: 
�
%Variable_2/Momentum/Initializer/zerosConst*
_class
loc:@Variable_2*
dtype0*%
valueB @*    *&
_output_shapes
: @
�
Variable_2/Momentum
VariableV2*
shape: @*
	container *
_class
loc:@Variable_2*
dtype0*
shared_name *&
_output_shapes
: @
�
Variable_2/Momentum/AssignAssignVariable_2/Momentum%Variable_2/Momentum/Initializer/zeros*
validate_shape(*
_class
loc:@Variable_2*
use_locking(*
T0*&
_output_shapes
: @
�
Variable_2/Momentum/readIdentityVariable_2/Momentum*
_class
loc:@Variable_2*
T0*&
_output_shapes
: @
�
%Variable_3/Momentum/Initializer/zerosConst*
_class
loc:@Variable_3*
dtype0*
valueB@*    *
_output_shapes
:@
�
Variable_3/Momentum
VariableV2*
shape:@*
	container *
_class
loc:@Variable_3*
dtype0*
shared_name *
_output_shapes
:@
�
Variable_3/Momentum/AssignAssignVariable_3/Momentum%Variable_3/Momentum/Initializer/zeros*
validate_shape(*
_class
loc:@Variable_3*
use_locking(*
T0*
_output_shapes
:@
}
Variable_3/Momentum/readIdentityVariable_3/Momentum*
_class
loc:@Variable_3*
T0*
_output_shapes
:@
�
%Variable_4/Momentum/Initializer/zerosConst*
_class
loc:@Variable_4*
dtype0*
valueB
��*    * 
_output_shapes
:
��
�
Variable_4/Momentum
VariableV2*
shape:
��*
	container *
_class
loc:@Variable_4*
dtype0*
shared_name * 
_output_shapes
:
��
�
Variable_4/Momentum/AssignAssignVariable_4/Momentum%Variable_4/Momentum/Initializer/zeros*
validate_shape(*
_class
loc:@Variable_4*
use_locking(*
T0* 
_output_shapes
:
��
�
Variable_4/Momentum/readIdentityVariable_4/Momentum*
_class
loc:@Variable_4*
T0* 
_output_shapes
:
��
�
%Variable_5/Momentum/Initializer/zerosConst*
_class
loc:@Variable_5*
dtype0*
valueB�*    *
_output_shapes	
:�
�
Variable_5/Momentum
VariableV2*
shape:�*
	container *
_class
loc:@Variable_5*
dtype0*
shared_name *
_output_shapes	
:�
�
Variable_5/Momentum/AssignAssignVariable_5/Momentum%Variable_5/Momentum/Initializer/zeros*
validate_shape(*
_class
loc:@Variable_5*
use_locking(*
T0*
_output_shapes	
:�
~
Variable_5/Momentum/readIdentityVariable_5/Momentum*
_class
loc:@Variable_5*
T0*
_output_shapes	
:�
�
%Variable_6/Momentum/Initializer/zerosConst*
_class
loc:@Variable_6*
dtype0*
valueB	�*    *
_output_shapes
:	�
�
Variable_6/Momentum
VariableV2*
shape:	�*
	container *
_class
loc:@Variable_6*
dtype0*
shared_name *
_output_shapes
:	�
�
Variable_6/Momentum/AssignAssignVariable_6/Momentum%Variable_6/Momentum/Initializer/zeros*
validate_shape(*
_class
loc:@Variable_6*
use_locking(*
T0*
_output_shapes
:	�
�
Variable_6/Momentum/readIdentityVariable_6/Momentum*
_class
loc:@Variable_6*
T0*
_output_shapes
:	�
�
%Variable_7/Momentum/Initializer/zerosConst*
_class
loc:@Variable_7*
dtype0*
valueB*    *
_output_shapes
:
�
Variable_7/Momentum
VariableV2*
shape:*
	container *
_class
loc:@Variable_7*
dtype0*
shared_name *
_output_shapes
:
�
Variable_7/Momentum/AssignAssignVariable_7/Momentum%Variable_7/Momentum/Initializer/zeros*
validate_shape(*
_class
loc:@Variable_7*
use_locking(*
T0*
_output_shapes
:
}
Variable_7/Momentum/readIdentityVariable_7/Momentum*
_class
loc:@Variable_7*
T0*
_output_shapes
:
V
Momentum/momentumConst*
dtype0*
valueB
 *    *
_output_shapes
: 
�
&Momentum/update_Variable/ApplyMomentumApplyMomentumVariableVariable/MomentumExponentialDecay2gradients_1/Conv2D_grad/tuple/control_dependency_1Momentum/momentum*
_class
loc:@Variable*
use_nesterov( *
use_locking( *
T0*&
_output_shapes
: 
�
(Momentum/update_Variable_1/ApplyMomentumApplyMomentum
Variable_1Variable_1/MomentumExponentialDecay3gradients_1/BiasAdd_grad/tuple/control_dependency_1Momentum/momentum*
_class
loc:@Variable_1*
use_nesterov( *
use_locking( *
T0*
_output_shapes
: 
�
(Momentum/update_Variable_2/ApplyMomentumApplyMomentum
Variable_2Variable_2/MomentumExponentialDecay4gradients_1/Conv2D_1_grad/tuple/control_dependency_1Momentum/momentum*
_class
loc:@Variable_2*
use_nesterov( *
use_locking( *
T0*&
_output_shapes
: @
�
(Momentum/update_Variable_3/ApplyMomentumApplyMomentum
Variable_3Variable_3/MomentumExponentialDecay5gradients_1/BiasAdd_1_grad/tuple/control_dependency_1Momentum/momentum*
_class
loc:@Variable_3*
use_nesterov( *
use_locking( *
T0*
_output_shapes
:@
�
(Momentum/update_Variable_4/ApplyMomentumApplyMomentum
Variable_4Variable_4/MomentumExponentialDecaygradients_1/AddN_3Momentum/momentum*
_class
loc:@Variable_4*
use_nesterov( *
use_locking( *
T0* 
_output_shapes
:
��
�
(Momentum/update_Variable_5/ApplyMomentumApplyMomentum
Variable_5Variable_5/MomentumExponentialDecaygradients_1/AddN_2Momentum/momentum*
_class
loc:@Variable_5*
use_nesterov( *
use_locking( *
T0*
_output_shapes	
:�
�
(Momentum/update_Variable_6/ApplyMomentumApplyMomentum
Variable_6Variable_6/MomentumExponentialDecaygradients_1/AddN_1Momentum/momentum*
_class
loc:@Variable_6*
use_nesterov( *
use_locking( *
T0*
_output_shapes
:	�
�
(Momentum/update_Variable_7/ApplyMomentumApplyMomentum
Variable_7Variable_7/MomentumExponentialDecaygradients_1/AddNMomentum/momentum*
_class
loc:@Variable_7*
use_nesterov( *
use_locking( *
T0*
_output_shapes
:
�
Momentum/updateNoOp'^Momentum/update_Variable/ApplyMomentum)^Momentum/update_Variable_1/ApplyMomentum)^Momentum/update_Variable_2/ApplyMomentum)^Momentum/update_Variable_3/ApplyMomentum)^Momentum/update_Variable_4/ApplyMomentum)^Momentum/update_Variable_5/ApplyMomentum)^Momentum/update_Variable_6/ApplyMomentum)^Momentum/update_Variable_7/ApplyMomentum
�
Momentum/valueConst^Momentum/update*
_class
loc:@Variable_8*
dtype0*
value	B :*
_output_shapes
: 
�
Momentum	AssignAdd
Variable_8Momentum/value*
_class
loc:@Variable_8*
use_locking( *
T0*
_output_shapes
: 
B
SoftmaxSoftmaxadd_1*
T0*
_output_shapes

:
�
Conv2D_2Conv2DPlaceholder_2Variable/read*
data_formatNHWC*
T0*
strides
*
paddingSAME*
use_cudnn_on_gpu(*(
_output_shapes
:��= 
y
	BiasAdd_2BiasAddConv2D_2Variable_1/read*
data_formatNHWC*
T0*(
_output_shapes
:��= 
L
Relu_3Relu	BiasAdd_2*
T0*(
_output_shapes
:��= 
�
	MaxPool_2MaxPoolRelu_3*
data_formatNHWC*
strides
*
T0*
paddingSAME*
ksize
*(
_output_shapes
:��= 
�
Conv2D_3Conv2D	MaxPool_2Variable_2/read*
data_formatNHWC*
T0*
strides
*
paddingSAME*
use_cudnn_on_gpu(*(
_output_shapes
:��=@
y
	BiasAdd_3BiasAddConv2D_3Variable_3/read*
data_formatNHWC*
T0*(
_output_shapes
:��=@
L
Relu_4Relu	BiasAdd_3*
T0*(
_output_shapes
:��=@
�
	MaxPool_3MaxPoolRelu_4*
data_formatNHWC*
strides
*
T0*
paddingSAME*
ksize
*(
_output_shapes
:��=@
a
Reshape_14/shapeConst*
dtype0*
valueB"@B    *
_output_shapes
:
l

Reshape_14Reshape	MaxPool_3Reshape_14/shape*
Tshape0*
T0*!
_output_shapes
:��=�
�
MatMul_2MatMul
Reshape_14Variable_4/read*
transpose_a( *
T0*
transpose_b( *!
_output_shapes
:��=�
S
add_6AddMatMul_2Variable_5/read*
T0*!
_output_shapes
:��=�
A
Relu_5Reluadd_6*
T0*!
_output_shapes
:��=�
|
MatMul_3MatMulRelu_5Variable_6/read*
transpose_a( *
T0*
transpose_b( * 
_output_shapes
:
��=
R
add_7AddMatMul_3Variable_7/read*
T0* 
_output_shapes
:
��=
F
	Softmax_1Softmaxadd_7*
T0* 
_output_shapes
:
��=
P

save/ConstConst*
dtype0*
valueB Bmodel*
_output_shapes
: 
�
save/SaveV2/tensor_namesConst*
dtype0*�
value�B�BVariableBVariable/MomentumB
Variable_1BVariable_1/MomentumB
Variable_2BVariable_2/MomentumB
Variable_3BVariable_3/MomentumB
Variable_4BVariable_4/MomentumB
Variable_5BVariable_5/MomentumB
Variable_6BVariable_6/MomentumB
Variable_7BVariable_7/MomentumB
Variable_8*
_output_shapes
:
�
save/SaveV2/shape_and_slicesConst*
dtype0*5
value,B*B B B B B B B B B B B B B B B B B *
_output_shapes
:
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
save/Const^save/SaveV2*
_class
loc:@save/Const*
T0*
_output_shapes
: 
l
save/RestoreV2/tensor_namesConst*
dtype0*
valueBBVariable*
_output_shapes
:
h
save/RestoreV2/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
�
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/AssignAssignVariablesave/RestoreV2*
validate_shape(*
_class
loc:@Variable*
use_locking(*
T0*&
_output_shapes
: 
w
save/RestoreV2_1/tensor_namesConst*
dtype0*&
valueBBVariable/Momentum*
_output_shapes
:
j
!save/RestoreV2_1/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
�
save/RestoreV2_1	RestoreV2
save/Constsave/RestoreV2_1/tensor_names!save/RestoreV2_1/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_1AssignVariable/Momentumsave/RestoreV2_1*
validate_shape(*
_class
loc:@Variable*
use_locking(*
T0*&
_output_shapes
: 
p
save/RestoreV2_2/tensor_namesConst*
dtype0*
valueBB
Variable_1*
_output_shapes
:
j
!save/RestoreV2_2/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
�
save/RestoreV2_2	RestoreV2
save/Constsave/RestoreV2_2/tensor_names!save/RestoreV2_2/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_2Assign
Variable_1save/RestoreV2_2*
validate_shape(*
_class
loc:@Variable_1*
use_locking(*
T0*
_output_shapes
: 
y
save/RestoreV2_3/tensor_namesConst*
dtype0*(
valueBBVariable_1/Momentum*
_output_shapes
:
j
!save/RestoreV2_3/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
�
save/RestoreV2_3	RestoreV2
save/Constsave/RestoreV2_3/tensor_names!save/RestoreV2_3/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_3AssignVariable_1/Momentumsave/RestoreV2_3*
validate_shape(*
_class
loc:@Variable_1*
use_locking(*
T0*
_output_shapes
: 
p
save/RestoreV2_4/tensor_namesConst*
dtype0*
valueBB
Variable_2*
_output_shapes
:
j
!save/RestoreV2_4/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
�
save/RestoreV2_4	RestoreV2
save/Constsave/RestoreV2_4/tensor_names!save/RestoreV2_4/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_4Assign
Variable_2save/RestoreV2_4*
validate_shape(*
_class
loc:@Variable_2*
use_locking(*
T0*&
_output_shapes
: @
y
save/RestoreV2_5/tensor_namesConst*
dtype0*(
valueBBVariable_2/Momentum*
_output_shapes
:
j
!save/RestoreV2_5/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
�
save/RestoreV2_5	RestoreV2
save/Constsave/RestoreV2_5/tensor_names!save/RestoreV2_5/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_5AssignVariable_2/Momentumsave/RestoreV2_5*
validate_shape(*
_class
loc:@Variable_2*
use_locking(*
T0*&
_output_shapes
: @
p
save/RestoreV2_6/tensor_namesConst*
dtype0*
valueBB
Variable_3*
_output_shapes
:
j
!save/RestoreV2_6/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
�
save/RestoreV2_6	RestoreV2
save/Constsave/RestoreV2_6/tensor_names!save/RestoreV2_6/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_6Assign
Variable_3save/RestoreV2_6*
validate_shape(*
_class
loc:@Variable_3*
use_locking(*
T0*
_output_shapes
:@
y
save/RestoreV2_7/tensor_namesConst*
dtype0*(
valueBBVariable_3/Momentum*
_output_shapes
:
j
!save/RestoreV2_7/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
�
save/RestoreV2_7	RestoreV2
save/Constsave/RestoreV2_7/tensor_names!save/RestoreV2_7/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_7AssignVariable_3/Momentumsave/RestoreV2_7*
validate_shape(*
_class
loc:@Variable_3*
use_locking(*
T0*
_output_shapes
:@
p
save/RestoreV2_8/tensor_namesConst*
dtype0*
valueBB
Variable_4*
_output_shapes
:
j
!save/RestoreV2_8/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
�
save/RestoreV2_8	RestoreV2
save/Constsave/RestoreV2_8/tensor_names!save/RestoreV2_8/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_8Assign
Variable_4save/RestoreV2_8*
validate_shape(*
_class
loc:@Variable_4*
use_locking(*
T0* 
_output_shapes
:
��
y
save/RestoreV2_9/tensor_namesConst*
dtype0*(
valueBBVariable_4/Momentum*
_output_shapes
:
j
!save/RestoreV2_9/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
�
save/RestoreV2_9	RestoreV2
save/Constsave/RestoreV2_9/tensor_names!save/RestoreV2_9/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_9AssignVariable_4/Momentumsave/RestoreV2_9*
validate_shape(*
_class
loc:@Variable_4*
use_locking(*
T0* 
_output_shapes
:
��
q
save/RestoreV2_10/tensor_namesConst*
dtype0*
valueBB
Variable_5*
_output_shapes
:
k
"save/RestoreV2_10/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
�
save/RestoreV2_10	RestoreV2
save/Constsave/RestoreV2_10/tensor_names"save/RestoreV2_10/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_10Assign
Variable_5save/RestoreV2_10*
validate_shape(*
_class
loc:@Variable_5*
use_locking(*
T0*
_output_shapes	
:�
z
save/RestoreV2_11/tensor_namesConst*
dtype0*(
valueBBVariable_5/Momentum*
_output_shapes
:
k
"save/RestoreV2_11/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
�
save/RestoreV2_11	RestoreV2
save/Constsave/RestoreV2_11/tensor_names"save/RestoreV2_11/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_11AssignVariable_5/Momentumsave/RestoreV2_11*
validate_shape(*
_class
loc:@Variable_5*
use_locking(*
T0*
_output_shapes	
:�
q
save/RestoreV2_12/tensor_namesConst*
dtype0*
valueBB
Variable_6*
_output_shapes
:
k
"save/RestoreV2_12/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
�
save/RestoreV2_12	RestoreV2
save/Constsave/RestoreV2_12/tensor_names"save/RestoreV2_12/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_12Assign
Variable_6save/RestoreV2_12*
validate_shape(*
_class
loc:@Variable_6*
use_locking(*
T0*
_output_shapes
:	�
z
save/RestoreV2_13/tensor_namesConst*
dtype0*(
valueBBVariable_6/Momentum*
_output_shapes
:
k
"save/RestoreV2_13/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
�
save/RestoreV2_13	RestoreV2
save/Constsave/RestoreV2_13/tensor_names"save/RestoreV2_13/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_13AssignVariable_6/Momentumsave/RestoreV2_13*
validate_shape(*
_class
loc:@Variable_6*
use_locking(*
T0*
_output_shapes
:	�
q
save/RestoreV2_14/tensor_namesConst*
dtype0*
valueBB
Variable_7*
_output_shapes
:
k
"save/RestoreV2_14/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
�
save/RestoreV2_14	RestoreV2
save/Constsave/RestoreV2_14/tensor_names"save/RestoreV2_14/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_14Assign
Variable_7save/RestoreV2_14*
validate_shape(*
_class
loc:@Variable_7*
use_locking(*
T0*
_output_shapes
:
z
save/RestoreV2_15/tensor_namesConst*
dtype0*(
valueBBVariable_7/Momentum*
_output_shapes
:
k
"save/RestoreV2_15/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
�
save/RestoreV2_15	RestoreV2
save/Constsave/RestoreV2_15/tensor_names"save/RestoreV2_15/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_15AssignVariable_7/Momentumsave/RestoreV2_15*
validate_shape(*
_class
loc:@Variable_7*
use_locking(*
T0*
_output_shapes
:
q
save/RestoreV2_16/tensor_namesConst*
dtype0*
valueBB
Variable_8*
_output_shapes
:
k
"save/RestoreV2_16/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
�
save/RestoreV2_16	RestoreV2
save/Constsave/RestoreV2_16/tensor_names"save/RestoreV2_16/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_16Assign
Variable_8save/RestoreV2_16*
validate_shape(*
_class
loc:@Variable_8*
use_locking(*
T0*
_output_shapes
: 
�
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16
�
initNoOp^Variable/Assign^Variable_1/Assign^Variable_2/Assign^Variable_3/Assign^Variable_4/Assign^Variable_5/Assign^Variable_6/Assign^Variable_7/Assign^Variable_8/Assign^Variable/Momentum/Assign^Variable_1/Momentum/Assign^Variable_2/Momentum/Assign^Variable_3/Momentum/Assign^Variable_4/Momentum/Assign^Variable_5/Momentum/Assign^Variable_6/Momentum/Assign^Variable_7/Momentum/Assign
�
Merge/MergeSummaryMergeSummarysummary_data_0summary_conv_0summary_pool_0summary_conv2_0summary_pool2_0lossconv1_weightsconv1_biasesconv2_weightsconv2_biasesfc1_weights
fc1_biasesfc2_weights
fc2_biaseslearning_rate*
N*
_output_shapes
: ""�
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
Variable_8:0Variable_8/AssignVariable_8/read:02Variable_8/initial_value:0"�
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


Momentum"�
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
Variable_7/Momentum:0Variable_7/Momentum/AssignVariable_7/Momentum/read:02'Variable_7/Momentum/Initializer/zeros:0��>�s      ��L�	B��갍�A*�

�
summary_data_0/image/0"�"��PNG

   IHDR          :���   �IDAT���N�0�ϟ�ݤM(�o #�xހ����J �	����}�p�9�H�-#�.���i�8�һM2cBW�i<��V�N��tV�˸�h�z�+b�������Im-(��4�f��0VY6����V0k� �����\����	�M�j�6��_���{�͜���#N����Eh᙭�/�{��dm��e���u���\������x����6����nd�ƨ    IEND�B`�
�
summary_conv_0/image/0"�"��PNG

   IHDR          :���  IDAT���N�@ ��w?���R�UI$n&Ɓ�����>��>������,�X!��J��k���ߧ߄'B�R:�ո1PФ+���
�ŌN���3B�ם���&�p�'T�z~���B���|; �/��w�T�lTe�)����\}���N���������ʸԺ{D+��h>Q�Y`�1�A/����ů+���c�yP�5���5cQ�?W�غ|�Rf��ɳ�<���G�]�M3��{�6�Z�wm︽����z�,��T:��/����8�J:    IEND�B`�
�
summary_pool_0/image/0"�"��PNG

   IHDR          �d�W   SIDAT�H ��x����5���!@�
�:��� 
���.�������-C���?��+����Uo��$�j�    IEND�B`�
�
summary_conv2_0/image/0"�"��PNG

   IHDR          �d�W   SIDAT�H ��O&.8��&��5�����/��Hp�S���@D�������Ӷ�=���������!U��X�    IEND�B`�
y
summary_pool2_0/image/0"^"V�PNG

   IHDR          ����   IDAT�c����?�W��_��r�0x1�I g�	��M$    IEND�B`�

loss<Aj?

conv1_weightsj��@

conv1_biasesU�=@

conv2_weights V>A

conv2_biases�@

fc1_weights��A


fc1_biases1�?

fc2_weights�O�A


fc2_biasesWL?

learning_rate
�#<3��      x�J	�%���A�*�
�
summary_data_0/image/0"�"��PNG

   IHDR          :���  IDAT��� �� Y0�� �f�    � n�� �"#���&��<��Z�Z ���i���5��)�� ����"���f��� �����!&�"3�1 �  � ���$��& ����
"+���
 ��  *���(��!�"��� M��	��@�g -�-dV' � ����$��}@ �� @�)�
� �� �����  �
����� ��������Xnij�o    IEND�B`�
�
summary_conv_0/image/0"�"��PNG

   IHDR          :���  IDAT���J�$�8� ���0�)�!$������C�%
Q7 ������%�����@'���3��] �
�������	ҫ������
��l4�>��
��9��������������������B"�����=e���������M��� �+,�!����'��<�����������	"���1��P� ���
���}1���1    IEND�B`�
�
summary_pool_0/image/0"�"��PNG

   IHDR          �d�W   SIDAT�H �� ['�R�� U-%)�#=����$!���������������55�e�D���7��%� �;: �[��    IEND�B`�
�
summary_conv2_0/image/0"�"��PNG

   IHDR          �d�W   SIDAT�H ����O����K���A�������� ����%�����K������-��ڳ!k�>�6    IEND�B`�
x
summary_pool2_0/image/0"]"U�PNG

   IHDR          ����   IDAT�c<�.���2��?���,�;SGUe��    IEND�B`�

loss�l�>

conv1_weights2�?

conv1_biasesY}?

conv2_weights�c`?

conv2_biasesǒ�>

fc1_weights��A?


fc1_biasesV��=

fc2_weightsI?


fc2_biases�<

learning_rate
�#<��R5�      ^�c2	�������A�*�
�
summary_data_0/image/0"�"��PNG

   IHDR          :���  	IDAT��;JA ��1;;�ML��$�)���)��V�������A���"Z	��!`H��Ƹ�ٙ���gl-E'&ϯ�ó���`
���R��y�k��/T�s��[w��|�Z?Eu��7YKB@�VWy��f��0��3*�>N�fi�`N���ø,�����ܒڰ-�ʯ�W<'���.q�$d�sd�����?�f���Mm߅q��K���:@�ya��6 �����ݯ�Y��B��jA=��єc#aX��%�h�#t!    IEND�B`�
�
summary_conv_0/image/0"�"��PNG

   IHDR          :���  IDAT���X�,�� ���1Q�	9")�<��	$��
2�4�+����������������ĺ�������6ה�������4����
���J$�������.���	��������.���1!��
���;O�'!)����"�������F�๸������^'�����,��B��I�ۺ��(����-�׮�?
[��V�(���w��O    IEND�B`�
�
summary_pool_0/image/0"�"��PNG

   IHDR          �d�W   SIDAT�H ��    �k�  )$���� ����� �$�� ������I)�NM�\�  w��Tb  +  @ r-��xUM    IEND�B`�
�
summary_conv2_0/image/0"�"��PNG

   IHDR          �d�W   SIDAT�H ��\#�����	G������������/�����(�'���0��� ��,�ਗ�.    IEND�B`�
x
summary_pool2_0/image/0"]"U�PNG

   IHDR          ����   IDAT�c`p�󟁁�!���A��A��� %\���b�    IEND�B`�

lossl�?

conv1_weights&�?

conv1_biases{%Q?

conv2_weights$D�?

conv2_biases��	?

fc1_weights��?


fc1_biases�X�>

fc2_weightsǍ�?


fc2_biases3%>

learning_rate
�#<M[ٳ�      �;xP	�a�����A�*�
�
summary_data_0/image/0"�"��PNG

   IHDR          :���  IDAT�E��J�P ��$'ihL-�XĒ�-����&^���Uܝ} }u���C-�.TA��ѢV�EO�x�����GN�.���l�`Q^�ϥZ��iE	%*��3��������%R�/f�r{;��kR�L*Ï�<���v�&'���0yq���u�Ζ��Sc��̥UGr����і�LL'҈���ɰ��� �S~�w �ґ��U����(�9�֗i��eo��{����tX�o{U``1�1��ŏ+�G�Gk���'�F���܏E?������rh{s    IEND�B`�
�
summary_conv_0/image/0"�"��PNG

   IHDR          :���  IDAT���+�6������$���c�,� �#@
����^�����(&�>*�����������?�������j���A�p5#J�!���@������
�?��%$6�����=(A����� � � ������������ ��'����� ����� 
����
��H�����������������������#�����!���+����p@���?�    IEND�B`�
�
summary_pool_0/image/0"�"��PNG

   IHDR          �d�W   SIDAT�H ��(6/�e�_o�(���	�-Lݹ	+,,�3=����	��������N��ߝ���z    IEND�B`�
�
summary_conv2_0/image/0"�"��PNG

   IHDR          �d�W   SIDAT�H ���-��������H
������������������� ���������v#�N�N�    IEND�B`�
y
summary_pool2_0/image/0"^"V�PNG

   IHDR          ����   IDAT�c�s�?3וc/�oȬ�g```P d����"    IEND�B`�

loss8�>

conv1_weightsy��?

conv1_biaseskF,?

conv2_weights솕?

conv2_biases�k�>

fc1_weights~C|?


fc1_biases�Y>

fc2_weights�}?


fc2_biaseso�V=

learning_rate
�#<��g��      ���	5'����A�*�
�
summary_data_0/image/0"�"��PNG

   IHDR          :���  IDAT����E���} �����7�������:��.��� �  ��8�:G����P�x�SG G����$
.��\\�	)�;�� �Қx� ��� .V  � ..�	V��"�/ ���  K
��*��  �.�Q�������G Ə�����!	Sc��������� �� ��� � �7
����    &�5������������	����������x�Km�    IEND�B`�
�
summary_conv_0/image/0"�"��PNG

   IHDR          :���  IDAT���N�P  ��D���Z���j�����pv�s��::�qc1�	i��F�(�Gy܋��+Š�����Kv�G�K
b���®]5��ym�.q;�t9o�.��7��8Y��!s���Y�آ~"4��*۠�pe�&g��-cx�V�h��C E��6�1Y9&?�PZI����t�/�.�P
wbm=���I��P���*��\l#ŗk}T�ӝ"��h(	�Ex����Pu��ٻs�/���mN|.T��X�V`��l��/��'r==<���?��i��    IEND�B`�
�
summary_pool_0/image/0"�"��PNG

   IHDR          �d�W   SIDAT�H ��B0�0�dQ3
������2�����{$ �%� �� ��- �   @�       �    �U�m,�P    IEND�B`�
�
summary_conv2_0/image/0"�"��PNG

   IHDR          �d�W   SIDAT�H ��� )��/��$���%�������A����!��������������%)���    IEND�B`�
s
summary_pool2_0/image/0"X"P�PNG

   IHDR          ����   IDAT�c8��g;���2  B�\��2    IEND�B`�

loss.��>

conv1_weights$J�?

conv1_biases�n?

conv2_weights��?

conv2_biases�	�>

fc1_weightsk�Q?


fc1_biases}�>

fc2_weightsU�5?


fc2_biases,ch=

learning_rate
�#<?�|�Z      �
�q	
����A�'*�

�
summary_data_0/image/0"�"��PNG

   IHDR          :���   �IDAT��KR1 ��tO>#�B�p�.�����,( 2N��$����g������Ef.�ZŶ�:GW�vك����㛂�z������x~ ������+�F�
!��T0G���
bu�e�>�����R��2r��LC��?d��0�x@���-�=�����(:��J�|���O%h��Cun�ū�$��i��ܚ]�5u&KKU)�׾�5�9r��[>�    IEND�B`�
�
summary_conv_0/image/0"�"��PNG

   IHDR          :���  IDAT��=O�@ ���{\{��1Zc�h���������Dc�A�������<��f���������.�mO�y�t�|�Ǡgٜ�t��}Ր��X��!{�n���=�~e��/��`�Fg����J5r' z�T��͏��ݹ;ʐ�ʃ�-gS����rNO1��u�?���D��ZN@d�$@�f3j��ֆs� NQ����FƏ_
��4=������S�6�|����mS@YnB���Z�5�Z�:��k\C��������?�Q{_��    IEND�B`�
�
summary_pool_0/image/0"�"��PNG

   IHDR          �d�W   LIDAT�c`�f��3R�OXҗ00�q``f���V��Ef�SF�EO�00����3����*g�4�E�J�    IEND�B`�
�
summary_conv2_0/image/0"�"��PNG

   IHDR          �d�W   SIDAT�H ������!F������C�����k������(-��������M����(	� ���S�$�oC��    IEND�B`�
x
summary_pool2_0/image/0"]"U�PNG

   IHDR          ����   IDAT�c��`�����|Z��
�� 7�Zr�    IEND�B`�

loss���>

conv1_weightsq�@

conv1_biases��?

conv2_weights�|�?

conv2_biases	� ?

fc1_weightsѿ�?


fc1_biases%>

fc2_weights�cc?


fc2_biasesFu�;

learning_rate
�#<e��~      �2m�	�-s���A�.*�

�
summary_data_0/image/0"�"��PNG

   IHDR          :���  
IDAT�=αJ�P��I���J#-��Q;FQ�b�Cѩ�89	:e�9�S!oP�H�ɠ�C��j����8�y�Cz��[�G�ُ�`�{t����&���`��H���k#Y��r���	���~}-mϪ��<�"IJ����8�2g����4�Ty��R"'�B�E�<�+�7u8��.�ֿ>�i��݀�wz��9L��3� �	8ޭ�(#0�`. ������@�V��Kr�xK����+�#��w�pj�O�I:�V��g��Xc��z�    IEND�B`�
�
summary_conv_0/image/0"�"��PNG

   IHDR          :���  IDAT�����I�"� �����#6!������������ ��
� �����������������������[)���������H������� 	*��L������_� ���@�"���E�)��.���3	���0���$� ��j�"����������-�0+���%4@'��U�@�s��^�����������$�!����K��0��G|    IEND�B`�
�
summary_pool_0/image/0"x"p�PNG

   IHDR          �d�W   7IDAT�c����^˻����2000�0�V�a```�8�΀��3HAX� Z�
��}    IEND�B`�
�
summary_conv2_0/image/0"�"��PNG

   IHDR          �d�W   SIDAT�H ���)��Ѱ,� ��
���/!� (�&t�\�c=�B�Z�'0	��/��'��&'�ȴD�n��=i    IEND�B`�
x
summary_pool2_0/image/0"]"U�PNG

   IHDR          ����   IDAT�c�X�������ـ��:��Ի /^��È�    IEND�B`�

lossUic>

conv1_weights�Z�?

conv1_biases�U?

conv2_weights���?

conv2_biasesYJ�>

fc1_weights,�*?


fc1_biases���=

fc2_weights;�?


fc2_biases�R=

learning_rate
�#<{��p      S���	������A�6*�

�
summary_data_0/image/0"�"��PNG

   IHDR          :���   �IDAT��;N�@�;o��ۉ\�� )�@$**:J$��X +�aT$Dq��of<�1;�Bݔ��N����"jm�Á��N�2C���T��ʧ.�/��I��_�Ҫ���ak�����.sle�(6>���8qD$��j��<�L�RK��.���¨)뭂O�r�i^N��>���2�����.��#K�k  ����|�§���ݦ:�c/���D�$�e��H�gHm�˄go��5���b��?źm�o�P    IEND�B`�
�
summary_conv_0/image/0"�"��PNG

   IHDR          :���  IDAT��KJ�0 ���ߦO�����p30�ť��xo$�ҕ��B\*�Q)��$m�$~�YLL�,Ri�vF�XV[$h�O�3��(�iVS���\�0��HL�&�Ӷ�QU�#´
"9GxU�)֒�Z�����S�A8A�v��r��C�-i<?Y�rExG�M��I�����th\|��vc���b��o<>=y�Ғ��Z��.����4��E�xvL���:I�]�5��^e}�Kۭp�jQ|��|I�Ȕ�򏂪�}    IEND�B`�
�
summary_pool_0/image/0"�"��PNG

   IHDR          �d�W   MIDAT�c`�f�tf=[���M+��az�tZ���g��n�1���c`�d`��.b������'����<�ߏ� �-��%�    IEND�B`�
�
summary_conv2_0/image/0"�"��PNG

   IHDR          �d�W   SIDAT�H ������ ���������H ��P!�:���u���Ow�̴��=�5��J�8    IEND�B`�
x
summary_pool2_0/image/0"]"U�PNG

   IHDR          ����   IDAT�cd��7g�`�r`������ ,�:�9A    IEND�B`�

loss���>

conv1_weights��?

conv1_biases�|5?

conv2_weightsg!�?

conv2_biasesm%?

fc1_weights�?


fc1_biases^Ǒ>

fc2_weights�`�?


fc2_biases�%>

learning_rate
�#<����K      @p��	�����A�>*�

�
summary_data_0/image/0"�"��PNG

   IHDR          :���   �IDAT�5�1N�0���ŶBR���C�".@Ɗ#0�q�p���%��)(Du��~�w����l�\+O�WK֐j^�"��~�+=V���`B�b�lf���V �ٜ�Si�~�MhZ�|  Z?�w/O7w���	$��{ -����o: ��V��5���>�ۉ �M�&�)q�R4��T��\ʤ��CB&���)I��3�u�b+o��a�ґU!�):'��,~ ��g<����    IEND�B`�
�
summary_conv_0/image/0"�"��PNG

   IHDR          :���  IDAT�������� ��-��[��/����� �I�/� �d3�*����/ �! ��Ҽ���!h���V���&��V���P������S������F�O��X�� 0��PZ���,������;�B��G��4L<����)<	�&�'���������� ������	�	������'��������������������	��'�� ����jG����    IEND�B`�
�
summary_pool_0/image/0"r"j�PNG

   IHDR          �d�W   1IDAT�c`���ַ�2�x�n���p���|�p��` \E8D�    IEND�B`�
�
summary_conv2_0/image/0"�"��PNG

   IHDR          �d�W   SIDAT�H ������3O�������51 =�
� �������H 0�̴����,�������!gA�e    IEND�B`�
o
summary_pool2_0/image/0"T"L�PNG

   IHDR          ����   IDAT�c```�π ' (Ɉ�5;    IEND�B`�

loss���>

conv1_weights%�?

conv1_biases��?

conv2_weights�?

conv2_biases'��>

fc1_weights˯�?


fc1_biases�4>

fc2_weights<�o?


fc2_biases��H=

learning_rate
�#<
���      8�K	ʣ� ���A�F*�

�
summary_data_0/image/0"�"��PNG

   IHDR          :���   �IDAT��AJA�_�Փ1f��"(
��#�[�!赼�K��bԝ L&�g�IwW��6v����`QͰj�nr�˃��(���Jg��Q��3���ϴ�VC�,}�=��ʪ4�d,b�8F0�R��6+���L�83)1D#B� ��J��A�3�����Ţ�4�Y&rY�1m�|^=u�9���F�U;�x��_}����  z��m��X��w[t�m/�;���h_��#y"*L�    IEND�B`�
�
summary_conv_0/image/0"�"��PNG

   IHDR          :���  IDAT���J�@ ��$GRML���B��Cq� N�����᫸���E| ���V�M�&i�;?~���A��A�8d~����i�?B��tn	�[Z���+~+jn%��<�E
��������kd
�bL���2�C+�#-
�D�8�n�),jb[�j�U�Q� y��LV]%o�Ϟ�0?V7����槽����t����1>���(8��^JW"������LfO�_���A_O��\��WL�p�JF��a�F�?
���Lm>    IEND�B`�
p
summary_pool_0/image/0"V"N�PNG

   IHDR          �d�W   IDAT�c`�030000�O� Vһ�:    IEND�B`�
�
summary_conv2_0/image/0"�"��PNG

   IHDR          �d�W   SIDAT�H ���� �$���N����������$��� ����Y��!2!����$��������΄$���L�    IEND�B`�
m
summary_pool2_0/image/0"R"J�PNG

   IHDR          ����   IDAT�c�����ĀB  Q���{    IEND�B`�

loss�o>

conv1_weights[��?

conv1_biases_��>

conv2_weights�w�?

conv2_biasesvi�>

fc1_weights,�a?


fc1_biases�Z>

fc2_weights93�?


fc2_biasesJ�=

learning_rate
�#<�z�8      ��K		��&���A�N*�

�
summary_data_0/image/0"�"��PNG

   IHDR          :���   �IDAT��=JA ����n�n�L!�O����(�x�a��G�,mDl��dI�������߇K���F`�ҦZ ܶJEQV��C��B�\&��'I6�7����+�cfb��x#v�)XJ��D�g��� Y�m�7��<���r���=�p�PF��������RK�X��vz�G�=���v}qm㖤ɔ�!�i��O9��:=E����BE}�L@w�_lᢄ�g���KZ
�y�%�j���u�3os<P	Q    IEND�B`�
�
summary_conv_0/image/0"�"��PNG

   IHDR          :���  IDAT���N�@ Pf�-S��K�DH��	�N�w����_@���� !CQC�Li����K`�a�o�A��[X��z.R�3p{�E�EG1��O�2g��ەwН�����|@��yo���~�冂���E�/	������?�]Oi] q�2ZTY�E����~Z[�P�PS�#���Kfv��~O$�.(�N��Vj���N�K��
�2�4.��~a�P�o֓JA=9����x�)c1�$�o8{�Kjr�'�����
�R)�����rf:��y�Ć�    IEND�B`�
k
summary_pool_0/image/0"Q"I�PNG

   IHDR          �d�W   IDAT�c�� L1 IA���j    IEND�B`�
�
summary_conv2_0/image/0"�"��PNG

   IHDR          �d�W   SIDAT�H �����	���ҽ�4AE��� ��� �A�0���25)����Q�����	n8��\�    IEND�B`�
m
summary_pool2_0/image/0"R"J�PNG

   IHDR          ����   IDAT�c`@��i9R;��    IEND�B`�

lossG�=

conv1_weights@��?

conv1_biases��?

conv2_weights�\O?

conv2_biasesP��>

fc1_weights[\P?


fc1_biases�P>

fc2_weights*�w?


fc2_biasesr��=

learning_rate
�#<O�m�      �d#	E�,���A�U*�

�
summary_data_0/image/0"�"��PNG

   IHDR          :���  IDAT���NE�;�; Ş���X�	.������ ��u����%��4/ !��&��5J � �
����Vu,<G73�
��ϋ$ &�/������,�; �����?,�����y1+����P�)��6�-'5�(%�����2��
� ���
�

)
��.�������W��
���#.������(���8�����3����+����G�t�!(�    IEND�B`�
�
summary_conv_0/image/0"�"��PNG

   IHDR          :���  IDAT��;O�@  ��Z�@��R%LFT6g7����������H|U(m1
}_�~��dWY������02���&d�WI(S�S�S�NI�k69�L^�Nf��"������D�v ���x��g���&:ET;�J�����@bE�-���W�?o�l}���DUG���F�L�V9�3�+���	��L����.a�U�H�jm��,�ĩ��V�xG�v���$B2��j��U ,��j�La�P�k^�w� |�XXfPƔ*#�y���u��_'    IEND�B`�
�
summary_pool_0/image/0"y"q�PNG

   IHDR          �d�W   8IDAT�c�b�F �`2304?y���Đ�iX����`�w�3sƝcg?(� �%a�iM    IEND�B`�
�
summary_conv2_0/image/0"�"��PNG

   IHDR          �d�W   SIDAT�H ����� &,�ظ���2A���-�������-4�< 	����	���!�������! ��!d�O�!    IEND�B`�
x
summary_pool2_0/image/0"]"U�PNG

   IHDR          ����   IDAT�c�c`Hg��������; -2R�'>    IEND�B`�

loss��?

conv1_weights��3@

conv1_biases��?

conv2_weightswO�?

conv2_biases7eS?

fc1_weights�ٜ?


fc1_biases�q�>

fc2_weights��?


fc2_biases��+>

learning_rate
�#<؉��&      4��.	)ϵ2���A�]*�

�
summary_data_0/image/0"�"��PNG

   IHDR          :���   �IDAT��KJA Ъ��_����H��Rp�I��Wp'�Q�(�Ѥ�L�O��! \����v��s���v�RR�H��L�� �8_��Yg6��6W��zo�vF+>~1U��
�j�5e�R����dy`!��9���MJ�V���8yJ2�$��r(u���@���A��9Y�$��HN�]U6j�k[�5�%���q�J����5�w��y�im�*}_�F�ש��(u�v��    IEND�B`�
�
summary_conv_0/image/0"�"��PNG

   IHDR          :���  IDAT��KN�@ ���_��
�5�i$�\I�ҝ70�Ƴ��K�1�*���@K[:�>�������$J�[o��VL�#v\�x3C�=�g��`e�o������U���>���AK�7m��*�~x�3��v��QgW(A��U�ˎԆ_;��(����D[�?mrs����X���!I4��θ)��k��Qtbʲ���5ΌLϵ��
`o Y������I<G�;<�j�~�J�e�	�`�/�w��(�~�!��.��D�Y��*bDa/rx'8d8��� ����}P��r�    IEND�B`�
k
summary_pool_0/image/0"Q"I�PNG

   IHDR          �d�W   IDAT�c�� L1 IA���j    IEND�B`�
�
summary_conv2_0/image/0"�"��PNG

   IHDR          �d�W   SIDAT�H ���������5��'--����	6*�"Ծ �+ٯ�0!��%'�!#���'�������"��!V���    IEND�B`�
m
summary_pool2_0/image/0"R"J�PNG

   IHDR          ����   IDAT�c�����ĀB  Q���{    IEND�B`�

loss���>

conv1_weights��?

conv1_biasesG?

conv2_weights���?

conv2_biases�>

fc1_weights�
�?


fc1_biases�
T>

fc2_weights�m?


fc2_biases�^|=

learning_rate
�#<��®W      �6��	��8���A�e*�

�
summary_data_0/image/0"�"��PNG

   IHDR          :���   �IDAT��=O�0 ���َ����"&&�n �g�]�0T�J+�Ď?yO�� U��i�>���w����$��27����[���O�A��l���[�N1+z�^�ko��2��u~&�t�#ʼV�!ɡ
 xaU�qW�]�ϯ��ٶ w�`y{p+�&l��B�8���I�l��^�:C��	�i2�
���3[e�K�M9EΘ�f�>T�1{ɪ_49-U)yU���r�!	    IEND�B`�
�
summary_conv_0/image/0"�"��PNG

   IHDR          :���  IDAT���N�@ ������F)Ȓ� Q�����W&Ƨ�<x7x0!j4�%-](ݦ���i{nc��u�͹���]z⹹dņ�I@vo���j���o8*��/4�~��FN�Ų���x��+�!�5wS�Q(9ɺݓke�Q�$xc���
��]T}��C���z;UBk�a��Ā�����y�+�~��M�x~��궪J���Q�(�R1� ���g��07-A$�aI�7�i܆��:��F�w��͂e5̖���a������l}���-    IEND�B`�
�
summary_pool_0/image/0"�"��PNG

   IHDR          �d�W   SIDAT�H ��E1�����do"*+)����������U�$E��A��������"�����2 ��u�(    IEND�B`�
�
summary_conv2_0/image/0"�"��PNG

   IHDR          �d�W   SIDAT�H ����8���)�������@�(��#����� /��
	��2�"%����� ���Cf�    IEND�B`�
p
summary_pool2_0/image/0"U"M�PNG

   IHDR          ����   IDAT�c`�������!� &�t�	�    IEND�B`�

lossDm?

conv1_weights���?

conv1_biases�Da?

conv2_weights���?

conv2_biasesGt;?

fc1_weights�â?


fc1_biases�Җ>

fc2_weights#e�?


fc2_biases�9>

learning_rate
�#<��EA      ��}	`�>���A�m*�

�
summary_data_0/image/0"�"��PNG

   IHDR          :���  IDAT���ݺ���Q�����@��'L��>�#���������� ' /���$#���K �!	���A�
��� �	���	߯
��;*�8�����D ������� �A����,���Q
�����������������2 �� E�������� �DŖ_�1+����$�3�I3	�����+!�$���� �����[,���z2��[    IEND�B`�
�
summary_conv_0/image/0"�"��PNG

   IHDR          :���  IDAT���N�0 ����`���C���1��w�]��F`K��8��ZJف�}�u��4Q�����؟y��]ڬ�����cZ|m�yq�� ����ƨ�i�t82��OFܳ�@4��tG�Nfn<��ɲK�bM^�9\9	o��'�1�@ �T��
s��E��v�2��������ȴ����᮰��R���GI�cMR����U�`W%vqJ^�{�p���y��Wp����t.��i����EJ���o!���`���	�    IEND�B`�
k
summary_pool_0/image/0"Q"I�PNG

   IHDR          �d�W   IDAT�c�� L1 IA���j    IEND�B`�
�
summary_conv2_0/image/0"�"��PNG

   IHDR          �d�W   SIDAT�H �����	����;.������������������������5�������<$���i�    IEND�B`�
k
summary_pool2_0/image/0"P"H�PNG

   IHDR          ����   IDAT�c`@�  
���    IEND�B`�

loss���>

conv1_weights� @

conv1_biasess�x?

conv2_weightsOͿ?

conv2_biasesF,'?

fc1_weights�?


fc1_biases�Ɨ>

fc2_weightsS1�?


fc2_biases�^6>

learning_rate
�#<_2V�d      ��-�	�D���A�u*�

�
summary_data_0/image/0"�"��PNG

   IHDR          :���   IDAT���J�@ ���M6)�4�_TQPA</>���>�Go�W����Z�Z/Mk����ή߇gIG? �}��!^�d�|;��Nׁ��89�u���-�jˍ�}=7� 3*�{�0�62m�]�6����ǔFHjI�.!މ$*q�t�~7 �Cߗ+���|eo�vo��y�k<y��^i�}��z��JK��q|�o8��n�3��D:d0��,�J-D�˃J+!����zX/9Y�_���|�ގ%"    IEND�B`�
�
summary_conv_0/image/0"�"��PNG

   IHDR          :���  IDAT��mO�@ �����ʠQ��Vo�[_��Q{�zoO����t&�	�pr�~�mP��!���a���F;r���c"��O�ˉ����{�,��z�yf42�~�(q������k��7漀6��������͑�XE�Z��V�A
�i��]����� 74-W3��zQnv0h^2��F�}Ja��j�+�&/�ޗKȝ2Q�a��[f��
ӑ���"�g�q-a�y9]��� Y��;P�Ғ0�H��:��w���R���T����p���    IEND�B`�
�
summary_pool_0/image/0"�"|�PNG

   IHDR          �d�W   CIDAT�c````aa`a`�T��Kz�U_�_Yvs���� 6��&��0����� ��rm    IEND�B`�
�
summary_conv2_0/image/0"�"��PNG

   IHDR          �d�W   SIDAT�H �����D�X������(�����-[�#�����7 �����/	#�9� &%������!3���    IEND�B`�
o
summary_pool2_0/image/0"T"L�PNG

   IHDR          ����   IDAT�cf0�π V ����|�    IEND�B`�

loss9��>

conv1_weights���?

conv1_biases�'?

conv2_weights�G�?

conv2_biases�&�>

fc1_weights��h?


fc1_biasesH		>

fc2_weights�)<?


fc2_biases�+=

learning_rate
�#<WI{�      �pC	���J���A�}*�
�
summary_data_0/image/0"�"��PNG

   IHDR          :���  IDAT���N�P ����7앐�!T�@b4qs�	VGW��pa�!LLؕ�и(�(�V
��z������b.4m&�-w�Y	��B����C=,�
���ז�Γ��K( �c?JU�(�-E"���[��Oiپ��7�~�F�]�࢟*���ㆇ�J��C�"����{Ii��M̌~4y�H@�U~������e]I��a�:�)�c�$'��0���V��z��������`���C��ZEpؾx4�<bAUs�s%T���ސ6h��9����J�?�#n����j    IEND�B`�
�
summary_conv_0/image/0"�"��PNG

   IHDR          :���  IDAT��IN�P  P������bj5��.�W�Ƌx���I�F4q*J�WQ�y��ndP>�(� ��R|u*�89���C����/˶�4�aRGkw�d%�d�	�#�䝕�?�������d*�Z��"��K/}ށРz�v� �K��g���P�g�̚�.i�:@*��R�8ߐ�0GMEHޘ��3Ͱ�P]�=:��7GAHƏ���0ȯ�i��0Bp�).������L��GO�\薳�o���k�+�j9�Gn���V����m    IEND�B`�
�
summary_pool_0/image/0"�"��PNG

   IHDR          �d�W   SIDAT�H �������  0M����5����� �>I���+���#� �������*3 ;���	���!�UȞ`    IEND�B`�
�
summary_conv2_0/image/0"�"��PNG

   IHDR          �d�W   SIDAT�H ������;�������$���(������k�%����# (
��(#���H%!�l�    IEND�B`�
x
summary_pool2_0/image/0"]"U�PNG

   IHDR          ����   IDAT�c�9񿆑�k�wFF C��o"k�    IEND�B`�

loss���>

conv1_weights��?

conv1_biases��?

conv2_weights�Q�?

conv2_biases�a ?

fc1_weights�Ti?


fc1_biasesOBD>

fc2_weights]eR?


fc2_biases�h�=

learning_rate
�#<����      <��	6jP���A�*�

�
summary_data_0/image/0"�"��PNG

   IHDR          :���  IDAT��MJ�@�wf�L2IR�`-�u��EAp�	\xO��x�(�Q*RKӤQ�7��yX��/�y�9�����UoPl)m��><�(�!�<Я���I ZU��q�\�qu�������$#	�b�	��x w�h$e<��g ��Q�hq`,����,ʭ$�IˎX�_���P?�`��$��#�4��9T���,z7�18���ք�����C�k�.T���/��ұ(aw��]A~#T����f����    IEND�B`�
�
summary_conv_0/image/0"�"��PNG

   IHDR          :���  IDAT���N�@ ��᧴3Tp�R��c|��.>�/\<�F�` +B�2�.�����k|�_[UsJ��|�9��3^F��(���tUCY:�o�<y'��٪#Hj���U�-�K�A��|�f()n7��dn9�;]�d���3&�����q�B7��r�j��؅Za]��`p��!X���y:�����ƶ~~'s?Z�U+���[����^ba�i<̴ó��'���D�}%i[*�#�Za���8M�bڼYB��fyqvT��~j��.|�c�s    IEND�B`�
�
summary_pool_0/image/0"�"��PNG

   IHDR          �d�W   SIDAT�H ��  B  &"  :� ,�  | �   �� �	  5�/X�C     ��z     ����    �� $��t_�l    IEND�B`�
�
summary_conv2_0/image/0"�"��PNG

   IHDR          �d�W   SIDAT�H ���1���!��
��( ����@9������� :��E"�`#�!����,	aHn Ti    IEND�B`�
q
summary_pool2_0/image/0"V"N�PNG

   IHDR          ����   IDAT�cd��Ґ����?  '���\�    IEND�B`�

lossh�?

conv1_weights�gn@

conv1_biases��?

conv2_weights�6)@

conv2_biases�H�?

fc1_weights��?


fc1_biases6j�>

fc2_weights7�@


fc2_biases���>

learning_rate
�#<�#e      8�K	�O�V���AЌ*�

�
summary_data_0/image/0"�"��PNG

   IHDR          :���   �IDAT�5�IN�@�_�՞�@$+
�D��ɥK���	���X���&�����㈡��0��}��k�P���(#R'6ڛl;4�ٵ����`�9:��>>#��K�`X�.�~�2CQ�RF�3�TU�����0S�Rh�s�$MlXMe
_�����߽���7=�B�?���v70�,?������9Ώfî�uѴ������KTA�j l{��	#    IEND�B`�
�
summary_conv_0/image/0"�"��PNG

   IHDR          :���  IDAT���N�P �����R�҂	1�R#�k]��qT6���Gr�)���.�� �HJ��О^���{��נ1~n5��Gzm��#�U�>�zS����4��~�]�������vb$��;��9�HໝM�3��K�p�d/����H�D��V^g�!My��be��U&���BA����`���Fr4Ofȍ걋1��C55� �T��-�ټ� �L�܌Gf�:&���8�gڨP;:S�J�W6���"���Z�� ҭ���?��vWw�n|    IEND�B`�
s
summary_pool_0/image/0"Y"Q�PNG

   IHDR          �d�W   IDAT�c`��6�Ï���� 2�ԍ�\�    IEND�B`�
�
summary_conv2_0/image/0"�"��PNG

   IHDR          �d�W   SIDAT�H ����!��"G� S Li���X��������"�3��� $���)�2�6 R�����^�̎��    IEND�B`�
o
summary_pool2_0/image/0"T"L�PNG

   IHDR          ����   IDAT�c```�π ? ��(�s�    IEND�B`�

loss=�s>

conv1_weights ǅ?

conv1_biasesc�
?

conv2_weightsJxh?

conv2_biaseszG?

fc1_weights��D?


fc1_biases��p>

fc2_weights�+^?


fc2_biases��>

learning_rate
�#<2^�@{      \��	AM\���A��*�

�
summary_data_0/image/0"�"��PNG

   IHDR          :���  IDAT���N�@ ����b9r=�~$$!2�h\����K���������Bb��׆R�X���*��[� �35�qH�OO�ˌ�l�J8�BzR�Mo�x֦�`xs�����_�%�L�����h�W��F,�po�I�\�T�z��q�=(!�?� R���D��D,�530�����Y�y-R����+bU5�9�5��������黑�� )��ԍ~?Рi�[.FM�	 ��'��xGq^	5	�6s��O���1��2���!n��w    IEND�B`�
�
summary_conv_0/image/0"�"��PNG

   IHDR          :���   �IDAT��In�0 P����L)���e���.�D�Ђ ��8��H�, �����;|���.YEj�~8�7/���Z�,$�}���R�h:*��~/�33#<m��r���(0���v���U�t��jw(��|:#�)�������<���S{�Mn_]�vl�ޱ?�xJ��#>�^y�7���hZm1@2M�@�|h@Iu<Ka�Cz�T�&H��8;��|
���j��)��<��H�KO�?�
?p(�    IEND�B`�
�
summary_pool_0/image/0"�"��PNG

   IHDR          �d�W   SIDAT�H �� .
D���t:/���Q��������   "  K �

� $���  �����}�EE�    IEND�B`�
�
summary_conv2_0/image/0"�"��PNG

   IHDR          �d�W   SIDAT�H ����M�'�����,����L�{+�*Db�2QP2�Z#���
����
��#��� ��U� �IOvT    IEND�B`�
q
summary_pool2_0/image/0"V"N�PNG

   IHDR          ����   IDAT�c``�o� �� �h�cn    IEND�B`�

loss�|0?

conv1_weights!|L@

conv1_biasesB��?

conv2_weights@@

conv2_biasesG݉?

fc1_weights��?


fc1_biases ?

fc2_weights�b@


fc2_biases%e�>

learning_rate
�#<,e�o.      fj(	��b���A��*�

�
summary_data_0/image/0"�"��PNG

   IHDR          :���   �IDAT���J�@�fv7�Ӏ'"�ɕ>��������/icc��X�	*��^�ݝ�O��	 ��Ő���^^���X�o��B �뻉[����L����ũS��IVffY+�k�SVӒ������B0G��>�M�� ���Y�9��}'�g2q�l��S���P0�mI�)����6îqm�e��pm+a?s�x�@X1U� Z��(Zc	�cOȢ���3W�P�� :���uU8p�    IEND�B`�
�
summary_conv_0/image/0"�"��PNG

   IHDR          :���  IDAT��KN�@  �ί� ŖVMUL��_�b�o���.q���(.�#�	�H�Җ�aZ���	�G�
�Ə8�@�qȆ�q)_u�E]mO`�����޽8ۚ>���]d�{F{����~�!X�z�|��f+<�S���ۙ���C+Oo;�1�
��A� N�
pGE�˒�f�˸B���I���N�q�!�������!�&qu4ͷ?�@���Rj!E;�_���q�X���嬎	K���a�M��)��De�~�<0E3�E��ڍc��?��|��_�7    IEND�B`�
v
summary_pool_0/image/0"\"T�PNG

   IHDR          �d�W   IDAT�c`@̌���	r �����_    IEND�B`�
�
summary_conv2_0/image/0"�"��PNG

   IHDR          �d�W   SIDAT�H ���4ȷ�F���	�����&����$�'���������+3������)���q�!-��B�    IEND�B`�
q
summary_pool2_0/image/0"V"N�PNG

   IHDR          ����   IDAT�c(�Ϡ� ���2  "��ˀo�    IEND�B`�

loss8��>

conv1_weights|��?

conv1_biases�Z9?

conv2_weights�5�?

conv2_biases�)�>

fc1_weights�?


fc1_biases�T>

fc2_weights�t�?


fc2_biases~9�=

learning_rate
�#<=k��x      �0�	��g���A��*�

�
summary_data_0/image/0"�"��PNG

   IHDR          :���  IDAT���N�@ ���G$�%m5��1~-ʦo�k���qҍ�Wp��KckꆍZ�w��CtT��( Gi��|�2Ʒ���
M��e��XX�|ku�x��t�p%���%u�ŵ�@[&���i��C ;��]?ȳ�"���G�As�Os��"9����| Q @{1'�'���+߯�Zyl<���7*����#�ܒ���6e��>����/��~�~�������F��6%��2��g�
�H2P�57`��^��5"�    IEND�B`�
�
summary_conv_0/image/0"�"��PNG

   IHDR          :���  IDAT��KN�@ ����B��J����z �օ��=���7Ѝ1�4hLݨiD*�g��������zt론����S�Q.��,?:�ĐG� �&�Y³@B�%��8�f���N�hU���x�ȢD�����7⟞L@�T~����M/�u�I��A�g%�"17�W����<ZN�B���q��=�x�Z��@K{��tW�ٔq��J1���u^p��R�4��h�}��!Q�^w�����Z���H���H�P��@{pݩ�~(    IEND�B`�
�
summary_pool_0/image/0"�"��PNG

   IHDR          �d�W   SIDAT�H ��        ��  0�@   �� 6ED� ������'��������    	 d�\���    IEND�B`�
�
summary_conv2_0/image/0"�"��PNG

   IHDR          �d�W   SIDAT�H ���)����#N �b߼��c�7E)	�K�"3������4%�.�������:6X�"    IEND�B`�
q
summary_pool2_0/image/0"V"N�PNG

   IHDR          ����   IDAT�cf�������M8  <[V�^�    IEND�B`�

lossF��>

conv1_weights�Ԣ?

conv1_biases�?

conv2_weightsl0�?

conv2_biases[�>

fc1_weights��?


fc1_biasesS>

fc2_weightsکL?


fc2_biasesK�<

learning_rate
�#<# L�R      b�9i	���m���A�*�

�
summary_data_0/image/0"�"��PNG

   IHDR          :���   �IDAT���N�0 Й�;j��J���#��z�
���x������D.��^�y��̒Q��zm	�h`n}bޟ�~A� D�UG,��񨱎	�3S��c�ED����:~_.�}��]���J�C�L�h��sq�}���P䈥�D�쒚�AR�������շ`=��~�9�\�@:�v�4� |*��I%�rfW
��U�,�:嘁�|��݌V,�	U�?9�uϡ�uz    IEND�B`�
�
summary_conv_0/image/0"�"��PNG

   IHDR          :���   IDAT���J�@  Й̞I���b+����ի7?�RDB�ޔTK�̒L2����0�^�x��x�*.�����=G�s�#�O�~P�5�ϒ}�ql	x���v��(�8���sV���Ѓ3- %ѭ�Ь���X���s�����Zr�\eS4\��/ RN��3�t/\)6�	2y���%��`�����JB���Mpz�kTk��{���q�	f�Ȫ�����+s�Re}�WG�!�\���P^�B�����{fpA�    IEND�B`�
�
summary_pool_0/image/0"�"��PNG

   IHDR          �d�W   SIDAT�H �� F =   F��'	!-���1��E��&:� �)�c  �(�  ���*' �@���������    IEND�B`�
�
summary_conv2_0/image/0"�"��PNG

   IHDR          �d�W   SIDAT�H ����$�=���������G� ����(�
�4$� ��>� �	�x��A��    IEND�B`�
o
summary_pool2_0/image/0"T"L�PNG

   IHDR          ����   IDAT�c`���� �<�    IEND�B`�

loss�[?

conv1_weightsv��?

conv1_biases|�!?

conv2_weights]M�?

conv2_biasesx��>

fc1_weights�Ɗ?


fc1_biases�}f>

fc2_weightsj�l?


fc2_biases��=

learning_rate
�#<�/�%      �4%	���s���Aس*�

�
summary_data_0/image/0"�"��PNG

   IHDR          :���   �IDAT�U��J�@���33&�p/Q�k��Z�lm����Od��ZXD�A�F-TV4a���X�}�U����V�������G�ֱ�zoV=L.~�)eH�� N�f�.K~����<�����n��ɍ��N�@6�\kN7Z�n�#��(��Ɨ�g��P�� ������ #��t�D��vyaYG+v��H�hf����4�lg�� h+�w��u0P!8�#�^�Qމf1�ĚB��Iߑ!E#n�3�qza��\��    IEND�B`�
�
summary_conv_0/image/0"�"��PNG

   IHDR          :���  IDAT��]K�P ����ls�ۚ](�M�� ���W����.#� 
M�P�R�!�����Yσn��4h�^����G��9�Q8�S��P����r�$I y�mX��X���Bo�a@Po�I����l�����n���w���n��@:̶�ήd�Q82�^�]��y�d;������;,�#]7�k�;���� ������<�L�WXh���@;Ӑ��*.�DJƁn�2��˔)�ؤ�E�KU�$.P^K)HD&25�JL�%��1,
�������5q~1�>o    IEND�B`�
k
summary_pool_0/image/0"Q"I�PNG

   IHDR          �d�W   IDAT�c�� L1 IA���j    IEND�B`�
�
summary_conv2_0/image/0"�"��PNG

   IHDR          �d�W   SIDAT�H ��|���3���Y��:6�=-5�
�������"������������j�����    IEND�B`�
k
summary_pool2_0/image/0"P"H�PNG

   IHDR          ����   IDAT�c`@�  
���    IEND�B`�

lossA��>

conv1_weights�=�?

conv1_biases�A?

conv2_weights� ~?

conv2_biases�X�>

fc1_weights�C?


fc1_biases�9>

fc2_weights��E?


fc2_biases���=

learning_rate
�#<��!�r      bl�	h[�y���A��*�

�
summary_data_0/image/0"�"��PNG

   IHDR          :���  IDAT��� �Q��%�����
� ����
�R����+I"�������5�!1����(�� �MF�P������{��$��~���4܇����-�ZF.��  �2�-���[�� >��� � �������3��( ����K(�"�)� m �)�������$��DL���'� � (�L5>�� 8��3� ?��3�>�!�    =�H6�7���ڸ�t��K�
�� !G�.Z�|�C�    IEND�B`�
�
summary_conv_0/image/0"�"��PNG

   IHDR          :���   �IDAT��IN�0 P��{�Ц���*�"��p$���Æ]�Fq�ql�}.�LI�'E��s��g�~�����'q�>���H��o΅��\xYR���H8cN�)�� �\s7V�h]��8x�FOp�C�h6��H�e� �5Ш�I���Kpkna�l8��b��5�zƘ��\kP*�m4b��v�����Ĩ�BO�v�(#D�J������_0\d��LV8J� ��*{3M(�    IEND�B`�
�
summary_pool_0/image/0"�"��PNG

   IHDR          �d�W   SIDAT�H ��-��GM�04�� '*������������������  ����˾F���    IEND�B`�
�
summary_conv2_0/image/0"�"��PNG

   IHDR          �d�W   SIDAT�H ���� +������5 L  �l8�� �� ��� �		��� ��|(�K:    IEND�B`�
k
summary_pool2_0/image/0"P"H�PNG

   IHDR          ����   IDAT�c```�π
  I���    IEND�B`�

lossSc�>

conv1_weights>��?

conv1_biaseslR`?

conv2_weights_*�?

conv2_biases!~?

fc1_weights��?


fc1_biasesÓ>

fc2_weights{ˌ?


fc2_biases�v5>

learning_rate
�#<�u��L      �\��	u�����A��*�

�
summary_data_0/image/0"�"��PNG

   IHDR          :���   �IDAT�UǻJ�P �����\��6�7TĀh�t����w�
�����������"R�8�P�I�VzI'����}^�FYYp�IU��t��4^���&6TMU�Vȣ���5�S��
����>�Mif�I�(�n�� ���P�[�J�çXZ2��<i$���]�������{�M�\���� }� ���}����!�O�[�W���\���W~�Mo14B�'�`�
Z��d&���L�<�����݂9���~)X���D     IEND�B`�
�
summary_conv_0/image/0"�"��PNG

   IHDR          :���   �IDAT���R�0 Pnr))�L�]��q�����w�Z�P
S	���s���'Cq�Ô]k�����u�Au���Og-:�(O��@"ӣ���db5���j���ts~s�O�1F�_��j�#A�R��!%�l,a�]1�"����h�e󍿪]Z�C���9��t�1�����]�=~�Ǖ���+�L��|��[���l�j���k<ŬI��Ȋ͑�D�E	[�.��R/��9��������Zey    IEND�B`�
�
summary_pool_0/image/0"�"�PNG

   IHDR          �d�W   FIDAT�c��|�b���X�����ý�����s˲p�0|dagc�~���+_?>bt�*q������  ��_V���    IEND�B`�
�
summary_conv2_0/image/0"�"��PNG

   IHDR          �d�W   SIDAT�H ���(����	�������������������
��
�!,-�) �����p�    IEND�B`�
p
summary_pool2_0/image/0"U"M�PNG

   IHDR          ����   IDAT�cp��?�8,U��H>�    IEND�B`�

loss��>

conv1_weights�z�?

conv1_biases��??

conv2_weights�C�?

conv2_biases�B?

fc1_weights�|p?


fc1_biases%�:>

fc2_weights%�4?


fc2_biases�s�=

learning_rate
�#<�Eb      >tp�	������A��*�

�
summary_data_0/image/0"�"��PNG

   IHDR          :���  IDAT����&��2�+�)� ������	�) 8��1����I�@�������G���� ��� �  ��/��4��9C�o�����*�
�2�w��    9�e�������  � �� �� �  � �sA��U6 �.����� ���% ����������/D�% >�.�a�:�����(/����	>����!��PS���-2�������xZD�{{    IEND�B`�
�
summary_conv_0/image/0"�"��PNG

   IHDR          :���   �IDAT���JA ���lb@	��06����1~������B"Y�0;3��9�u�� �������n�,9V4ӏAڧN�A�V�Q���&�	����E�~��L�_�qy2'�7��i!36�řܮϣ4	\��.��|Y�y"qP�B�HA���r�m)U�`Ȫ?���`@�@S���t+ǰ&��|ry���@����F�� ��3aͭ��o7�`�|�I��l�y��    IEND�B`�
�
summary_pool_0/image/0"�"��PNG

   IHDR          �d�W   SIDAT�H ��"P���`s
�����=������#��	0 ������	� �'(� C������� J��i�    IEND�B`�
�
summary_conv2_0/image/0"�"��PNG

   IHDR          �d�W   SIDAT�H ���. 	��
��;� �
O'�
�;�������&����
�$������7u�E�T    IEND�B`�
l
summary_pool2_0/image/0"Q"I�PNG

   IHDR          ����   IDAT�c���?*  /��q8F�    IEND�B`�

lossR��>

conv1_weights��n@

conv1_biases�\�?

conv2_weights��%@

conv2_biases�#?

fc1_weights�Ԯ?


fc1_biases�g�>

fc2_weights���?


fc2_biasesu>

learning_rate
�#<�( Ƃ      <��	�o�����A��*�

�
summary_data_0/image/0"�"��PNG

   IHDR          :���  IDAT����� ���EF�3������t�������������u�w�����#����E�U��u-�` ����#��.�� \@]�3���	2��i -3��   	��.# ���޾#��� 	 �0����-����   ���E�
  �
�	�4����	��#���	�x���������#� ��(� l� �   � ��� #�� %� �.��� �.��w ��`�    IEND�B`�
�
summary_conv_0/image/0"�"��PNG

   IHDR          :���   �IDAT��=O�@ ��{�}Rj[�.F�����āĨ1���R���}�+d�i����i�^q5�[>�g���e���I�:������(=|fc$9�f+6�I=��&(6���I�u!��I���ڽCʆ:���V�c����������CmH��1��7h�e���,d���PS&3%8����x�Gǲ��8#D�"@y2���=�m�l?JJ�M]FU�\Y��"0���z7�5k�    IEND�B`�
�
summary_pool_0/image/0"�"��PNG

   IHDR          �d�W   SIDAT�H ��
 �_,�?(���a+�� ���� ���X������������ ��������Kt"?����    IEND�B`�
�
summary_conv2_0/image/0"�"��PNG

   IHDR          �d�W   SIDAT�H ���0������6$6��������� ��	� <���A�������R���    IEND�B`�
y
summary_pool2_0/image/0"^"V�PNG

   IHDR          ����   IDAT�cX���F���x%D�~�c  [7�9�ń    IEND�B`�

lossKr�>

conv1_weights���?

conv1_biases�GF?

conv2_weights֗?

conv2_biases^��>

fc1_weightsNLX?


fc1_biases�y!>

fc2_weights�]$?


fc2_biases��B=

learning_rate
�#<�"�P      S���	\ŕ���A��*�

�
summary_data_0/image/0"�"��PNG

   IHDR          :���   �IDAT���N�0 �����Dcte�������ˍ�7j2
û-t�C�K��F\��3�h��ì:4|��o���d;XN�RY.�b�� ���2�p86/Y:�,�W�C+ʾ��/iΈ�m�KtC��Se�%�8��~$_W�%�TQ\���]*R1x������zm����G6�\׷d��k�z|{U�D� ����'Wi)ч�w��`�G?Y]ž��9�w+���    IEND�B`�
�
summary_conv_0/image/0"�"��PNG

   IHDR          :���   �IDAT���JA��v�.��� ������l}�����.1 z&��ݹY��=���M��&�l����&�>e�?����PD*R:l $6E��JA*g.����+4lba����UU�[�N��R�v�`�F��p,�mٱ"�Wue����&L��0�!��	Y@XZ��]Է�Ll�:3Q���}A�l$Kk�|8~�ƒ�����m�|]GA��79������y�*eN��uG���    IEND�B`�
�
summary_pool_0/image/0"�"��PNG

   IHDR          �d�W   SIDAT�H ��g%����F��*������� ��&���� ����)>���  �'��ư  �/�����C#WH:G�    IEND�B`�
�
summary_conv2_0/image/0"�"��PNG

   IHDR          �d�W   SIDAT�H �� ����������
 �!����
��5�����	F���
��0� ��.����+�H    IEND�B`�
x
summary_pool2_0/image/0"]"U�PNG

   IHDR          ����   IDAT�c`x�򟁁�A����a> &Q����E    IEND�B`�

loss���>

conv1_weights2�@

conv1_biaseszQ�?

conv2_weights	@

conv2_biases%H?

fc1_weights^M�?


fc1_biasesO��>

fc2_weights�4�?


fc2_biasesl�k>

learning_rate
�#<&~5      ,l4	�������A��*�

�
summary_data_0/image/0"�"��PNG

   IHDR          :���   �IDAT��KN�0 бg��m����f��C��p���S�b˞+�$H�*�O�ǎ�{�B�
(�ۆ��,�ӮV'�`;B���0R�ߡ�6@8IQ��q�<��%�p�E/�Y�{َ�謷]Y�k��A:�#���c�����g:��5Uz. X��d�e��f,^��Sh����nWV�7-���T���gr�)�M�����S˻ӹWҿ1)����X؎@�@�z	*'<%~�6$4P�l�#�v����    IEND�B`�
�
summary_conv_0/image/0"�"��PNG

   IHDR          :���   �IDAT��;N�@ �ݙ��v�+	!|�*ED��
7�R4���TH���������|����m����m�FeMo���o��6�n�'�JB�R���n/R��D����d�t[�N�5��々'� p�Ɲ��	��	d������F�S����#@�V�����j8jdm)���_�U�~m�9dTL�?w�^Y6�h�Q>l��|�%Q��zYo���*6K3 �h�~�{ X�R�2�ex��5�Id���-z�6�    IEND�B`�
�
summary_pool_0/image/0"i"a�PNG

   IHDR          �d�W   (IDAT�c````�Je`````�b������_��@  �?���8    IEND�B`�
�
summary_conv2_0/image/0"�"��PNG

   IHDR          �d�W   SIDAT�H ����	�����	  ��"��
	'��
���(�"������0�����	��]D��s    IEND�B`�
l
summary_pool2_0/image/0"Q"I�PNG

   IHDR          ����   IDAT�c����{T  ;�tͫp    IEND�B`�

loss0Җ>

conv1_weights�?

conv1_biases�0?

conv2_weights�ˑ?

conv2_biases�S�>

fc1_weightsF�Q?


fc1_biasesd�G>

fc2_weights�@5?


fc2_biasesC��=

learning_rate
�#<�ܥ�I      q�;u	������A��*�

�
summary_data_0/image/0"�"��PNG

   IHDR          :���  IDAT���X"���8�&�##��*����  ���$��=� ��� (�� �!���� �C � � �������� �����(�)";  ��"&��6 � ��� �B�	�T�5��%� ��K:�� ����	�9	����� �!������$
 	�.'�6�# �����}�������%	!,�$���(�+��J�%�
����?,� ��,(��7!Yty2re    IEND�B`�
�
summary_conv_0/image/0"�"��PNG

   IHDR          :���  	IDAT��KN�@ ����t��)[L	+��ĝ;o����<�G�.ܹ2*K)�e�`��}�8�o��/O�^J��"s�ڙfw�*2)�8) �ʴ��?dA>�������f�@ܺɝ��dp;�luiR�aj4�;BA���@vR� L-$Z���
�ʢ��v��x�5��M�d2�;Ȝ�&��%�P���ĉ,�H����g���Cf��&�d���H M����7�Uτ���FIXY'����o��5�#O�5�j�l����u�1��v    IEND�B`�
k
summary_pool_0/image/0"Q"I�PNG

   IHDR          �d�W   IDAT�c�� L1 IA���j    IEND�B`�
�
summary_conv2_0/image/0"�"��PNG

   IHDR          �d�W   SIDAT�H ����7	   �ֹ�� P����+��'���#�����)�d��$��&�,
� R�xؐ�    IEND�B`�
k
summary_pool2_0/image/0"P"H�PNG

   IHDR          ����   IDAT�c```�π
  I���    IEND�B`�

loss��>

conv1_weights��?

conv1_biases���>

conv2_weights��?

conv2_biases@�>

fc1_weightsZSJ?


fc1_biases��D>

fc2_weights�F:?


fc2_biases�1�=

learning_rate
�#<N`�z      ��	YF����A��*�

�
summary_data_0/image/0"�"��PNG

   IHDR          :���  IDAT���N�@ ���u:@[��X��J<O��`|}�>�O��#g/^pMDR+$�4m�.t�~�p�H�l2���Fi��=9p�H��,V�j���O���4���	C�����&��|��:O_z�3y[8%�H�'�7�ka�U��tVxW	t�j#�=Ҷ�2��F�c�k�#{J��2ͩ�-�g0L��ে����'���J}��F"pFH��%��0�Uddm��]�)�=䆒��u���u���#ҹ����9�i������Sz*h��    IEND�B`�
�
summary_conv_0/image/0"�"��PNG

   IHDR          :���   �IDAT��MO�0 ��ݻu�bc� �Bbb�W�����E�Dc8���J�y�p��C��y��>%��Qo⽨��p��^��Ak�2Ci��b�~�3�\�m~Ϳ���Ν�k9ֻO� ��?2�{����m�2����f�/ӆ�Y��!n�������ӂ(��0u� J2X�����|������$F)bW��ت6�ƼD)���9��1VT�@����S�]�c�	te� V�����_�.������k_�h	    IEND�B`�
�
summary_pool_0/image/0"�"��PNG

   IHDR          �d�W   SIDAT�H �� Y?�T� �"5�&R   :���      -   )�	� )�7  ���
� ����v��FW��    IEND�B`�
�
summary_conv2_0/image/0"�"��PNG

   IHDR          �d�W   SIDAT�H ������0�H�*����d� �(�-�M�2��2�� 
,����	��%��<���������~%    IEND�B`�
o
summary_pool2_0/image/0"T"L�PNG

   IHDR          ����   IDAT�c8�?R�8� 4 *��Ր��    IEND�B`�

loss�A�>

conv1_weightsT%@

conv1_biases��p?

conv2_weights/A�?

conv2_biases+V?

fc1_weightsQh�?


fc1_biases��L>

fc2_weights#R�?


fc2_biases�*�=

learning_rate
�#<��6       n��	nm䱱��A��*�

�
summary_data_0/image/0"�"��PNG

   IHDR          :���   �IDAT���NA ���vA�F
�������4*�;�{�sp������'�������L�ԅÄd���#G3*����k4�"s���~��0P3��4����=��Θ����J� ��U����׹4��S�û�Y v��6��`vd/�#��6�Z`\�=�ɉ���^���W�(.��a���3��ߋ�VK���<|mz�e赤]����s�
�w�Q�>Z    IEND�B`�
�
summary_conv_0/image/0"�"��PNG

   IHDR          :���  IDAT���N�P  ��*�>�-H*C�Qy$�0��98�a~�#j��/�@d2(�I�P������4څj^�GT4tX����I��')���&_��$�$2!�� =��|�6*س\��0}Gc�p�ȕ�6��J29�c[xx4l�z1�q��e5X�n�A�2�J����/]�v
�{Il_&���	�֛t֔��翓���_�ȳ�E���g�ߺ�x4�g�(��֟93���O�i��&r�O�D�v���FTvb���0{e�
�����Ƕx\���    IEND�B`�
k
summary_pool_0/image/0"Q"I�PNG

   IHDR          �d�W   IDAT�c�� L1 IA���j    IEND�B`�
�
summary_conv2_0/image/0"�"��PNG

   IHDR          �d�W   SIDAT�H ����*	2�����B��������
�������������#<��� �>�	��� JH�H    IEND�B`�
o
summary_pool2_0/image/0"T"L�PNG

   IHDR          ����   IDAT�c```�πJ CW*՚j    IEND�B`�

loss��>

conv1_weights�g�?

conv1_biases:b?

conv2_weights�C�?

conv2_biases�" ?

fc1_weightsf�?


fc1_biases��Z>

fc2_weights�r?


fc2_biases��=

learning_rate
�#<1�F`      o��	]�ո���A�*�

�
summary_data_0/image/0"�"��PNG

   IHDR          :���   �IDAT���;JA �����>&��QL�X���X[yA�y+�B�jk����+1/����̬G���5/뱪w�w��!���~i�����VG��0�ڡ�˙�7O��k.����I�:.m�a�����u`/�`4$l�z'-D� 0⊼=+�wI�İ�����qPfu4M�C��jW�����ŴM�aK�*h��qkz��7�B���D����bDN1P��)�i�49D��%�?�~h�N    IEND�B`�
�
summary_conv_0/image/0"�"��PNG

   IHDR          :���   �IDAT��AO�0 ����c�2d��0pA/�<���y2��D���d��٭�����lx��t���1�kU���h�.�|��1�:Mf���J/�Q��|�{���	,��p��a�ހi�v��zL�k<_n�du]䰉���AK�
���{^�D�8���&���.zܙd �N�֗%�O����{��(�Fo�{_kRZ���uu`�"���c��Vr�w��u���R(�`�/>��� NĬ�ǻyN%�    IEND�B`�
�
summary_pool_0/image/0"�"��PNG

   IHDR          �d�W   QIDAT�cf`�|�����`b������p�Y������A��;�A������L�:9f6[ii�,w���������������  �=��Uc    IEND�B`�
�
summary_conv2_0/image/0"�"��PNG

   IHDR          �d�W   SIDAT�H ��q�C$3����"�A����]������� _�.������	'����������(#m�j�2    IEND�B`�
x
summary_pool2_0/image/0"]"U�PNG

   IHDR          ����   IDAT�c`��������83�3��� 9="��d=    IEND�B`�

loss� �>

conv1_weights���?

conv1_biases_V?

conv2_weights8�?

conv2_biasesr�>

fc1_weights��o?


fc1_biasesu�>

fc2_weightsx#&?


fc2_biases���;

learning_rate
�#<��n�W      �6��	L�����AЉ*�

�
summary_data_0/image/0"�"��PNG

   IHDR          :���   �IDAT�%�+O�P�����M�`0A`HH:�$A�3�p��@OЬ[{z.��+%����X'7����X2�c���P�ݮ��R�)s�? �);F�̔p���1��r����+��A�GM��{�%��-�`P3��ȁE���g���&���8vU�.�����s�2yl� �S�v���{ϔ�]I���٪�%����FA2r�I�.����Nl4��e,]F�s    IEND�B`�
�
summary_conv_0/image/0"�"��PNG

   IHDR          :���   IDAT���N�0 �~�h9d�Ɯ"&ޘ̧�=|�И%�y�DM�d.�+�J�>���T�X��]s�9Jץ���7�����(�s�Q;�F����У�M\�#8�F*9v�9<����x�ll۞e�%��|]#%"��)Z��7�
#J���_��J��f����[U{����v;5�ZS��O@у$�L��'�ɗpWS�f��~����Bmcڏ�K��eO�>wn���ʆ" �c}eⷴC.� �x쭰��    IEND�B`�
�
summary_pool_0/image/0"�"��PNG

   IHDR          �d�W   SIDAT�H ��        L� av%	 �Q(
H�;�% ���ת����x   ���     �     j����Q�    IEND�B`�
�
summary_conv2_0/image/0"�"��PNG

   IHDR          �d�W   SIDAT�H �����*��: ���̰�-���3��^���j�� �) Ra $B	X!W���-�%5���J�    IEND�B`�
x
summary_pool2_0/image/0"]"U�PNG

   IHDR          ����   IDAT�cH��{����a3�CO C���F    IEND�B`�

lossz�b>

conv1_weights�WL?

conv1_biases��>

conv2_weights7&D?

conv2_biases��s>

fc1_weights��?


fc1_biases��=

fc2_weights���>


fc2_biases=�<

learning_rate
�#<�ث��      O� 	��Qɱ��A��*�
�
summary_data_0/image/0"�"��PNG

   IHDR          :���  IDAT������� �� ��{d��㳦� ��m���t��! T /��"#� F���" ��2�()�1���������
��4�������	����������&������ L�3���t� ������	�	 �B	�c�� �0��������,���	�� � ����
	��
������*�� ��
������x  Ƅ�w�Z��L    IEND�B`�
�
summary_conv_0/image/0"�"��PNG

   IHDR          :���  IDAT��=O�@ ��x{�~MK� c�������?]7]\L4-"T--m��z>n��Y�����DMah9���#�V���(y�HY[���O�O��A���!9P�at���vd�U{X���KL�������u��jHf�E��P��f�`��*]P�c&�i�(�
J��	�Bus�2��#Yڌ���9��C�����6�ϯ�<|
ʝ�\X�z[����,���S`���n�����M�m2З	�tô>��AP�c���"��[���vr�}5X    IEND�B`�
�
summary_pool_0/image/0"�"��PNG

   IHDR          �d�W   SIDAT�H ��  *   0C
����Qa8����$�� ����@U���%��1	��4ͣ)��Gq�    IEND�B`�
�
summary_conv2_0/image/0"�"��PNG

   IHDR          �d�W   SIDAT�H ���=����,��������(����
��'���� �9�,��*t������	��^#6�%)    IEND�B`�
x
summary_pool2_0/image/0"]"U�PNG

   IHDR          ����   IDAT�c0��C� 6>\m

�    IEND�B`�

loss�B?

conv1_weights�[�?

conv1_biasesfQH?

conv2_weights�2�?

conv2_biases�"3?

fc1_weights�8�?


fc1_biases�B�>

fc2_weights��?


fc2_biases:5�>

learning_rate
�#<�+      A��	���б��A��*�

�
summary_data_0/image/0"�"��PNG

   IHDR          :���  IDAT���N�0 Я_K��eQA0t�x'/�/�3xc�W�!�[��ڮ������r	3X�����р���kHu��놉��n�ccԆް%�^if�r�雝kl�	��3�ǰ?�9��5Q�bь�4�6z_���gb���P���^���ܕ��T�����_���]��y�A �v�}fW!�a��΂���p��ݴ�aF$͹�m}X8��OO�9�{���㼌�p[Z�H�]�ȝ��\錺RO#�@G�$k�%��?@r|�@k    IEND�B`�
�
summary_conv_0/image/0"�"��PNG

   IHDR          :���   �IDAT��KN�0 P�q�v�QU>V��o9V� � 1ST�Ď�{x���T�_�N�N|F�6���&�+�aVq�����
����.�\���A�����̍M�J��ȸgCr�b%O�J`� Zg_�F-�#p̲U��L.X�����d� !��x���q J���Ƕ��F�PA�)�i���n�9A�>�������2�p1�O``0�ϟ����_ׂ`    IEND�B`�
�
summary_pool_0/image/0"w"o�PNG

   IHDR          �d�W   6IDAT�=�1�  �{��0 !��%Q\��jݼ�w���E:⡁����]� ݤ�WI�    IEND�B`�
�
summary_conv2_0/image/0"�"��PNG

   IHDR          �d�W   SIDAT�H ���	ۥ�������������������"	
�������$#�� ���=�� �b._    IEND�B`�
n
summary_pool2_0/image/0"S"K�PNG

   IHDR          ����   IDAT�cx🁁��  %�I�W�    IEND�B`�

loss�P�>

conv1_weights0��?

conv1_biases�-Y?

conv2_weights�%�?

conv2_biases4@?

fc1_weights?��?


fc1_biases\҇>

fc2_weightsnJl?


fc2_biases! >

learning_rate
�#<����Z      �
�q	��ر��A��*�

�
summary_data_0/image/0"�"��PNG

   IHDR          :���  IDAT�MɽJ�P ��ޛ�M$i�P���I�,�CAt��J��U'7g���
.A*�v�nB
���jI�����ų��Y�^������tu�Wִ���PMbEW�����1��M;W�J��zQ`�gvf}�yQl��~3��HP��&�tJ  �2�:�J���|(�hH٧�:��l`k��O!�6�Q��T��m �0Ͳx~27�G�~';����	��X' v���.�z`8�yK~/'\�݆�'w?!e(z��
V��"ͽ�!M�,"�/��`��<    IEND�B`�
�
summary_conv_0/image/0"�"��PNG

   IHDR          :���   �IDAT��;O�0 �;�����
�Ć*�����?2��P���i���ÏZ��4�C#��lv��X�E�
��3�p=�������~}��;�NƑ���#�M��e�C�^W�v��-nPeFV�V<��Q�wѴ~*��KV���5%�x~�|~O��:P����b;�C� �H4�	 ��"���&�	bf:Ko�}��$
)��IQBN],	�b1U��,8�K�.d#���~�qn���    IEND�B`�
�
summary_pool_0/image/0"|"t�PNG

   IHDR          �d�W   ;IDAT�c`�	x��d�^ed��}KV|+3�la��ߘ���`���������?r?  1uL�)    IEND�B`�
�
summary_conv2_0/image/0"�"��PNG

   IHDR          �d�W   SIDAT�H ��<��.�&111-6��E��"�$�
) �%�
������T �����	�K���    IEND�B`�
x
summary_pool2_0/image/0"]"U�PNG

   IHDR          ����   IDAT�c```�g6�:���g��
�6 (�Oꡈn    IEND�B`�

lossu��=

conv1_weights��;?

conv1_biases7�>

conv2_weights�m4?

conv2_biases��p>

fc1_weights�v	?


fc1_biaseswQ>

fc2_weightsX�?


fc2_biases�{�=

learning_rate
�#<j���Z      �
�q	l9ޱ��A�*�

�
summary_data_0/image/0"�"��PNG

   IHDR          :���  IDAT��MO�` ���� R	�T6����u��Ů�OPߢ�UǶ6�bk�KB-�TPx0���}=撊�־轮�w-�ׅ"�g��m+�$X�І[�v�hی��YP�f=�ix���rq����.�ƅ������y�1d���57��P�����,@9J���!����j���׉;:D��T�&ɀ]&�*ogAc4�� �{�ut�\��Ѷeɵ1e����~��ݛ6)��^��,Ks�q��C�i�3��R����0Y��n_y����#z�߾[    IEND�B`�
�
summary_conv_0/image/0"�"��PNG

   IHDR          :���  IDAT���N�P  P.��~Hy��@�M�&n���_��W�������(�6� AbxS�s vk�=���4�)���8��t:'�������4o�� �/�����w��K
P}t�9Ou��e��� �5>5��A}������u�o ��D���1���x3��F�Y#��P��Hx�qꪓ�lG����P$:1Sn�ljX\�J�V�O���s�C'�Q�\���P%D��|��s�]���M%�g�F&j���X滄�+`R��_��j��?Cze�W�    IEND�B`�
k
summary_pool_0/image/0"Q"I�PNG

   IHDR          �d�W   IDAT�c�� L1 IA���j    IEND�B`�
�
summary_conv2_0/image/0"�"��PNG

   IHDR          �d�W   SIDAT�H ��oE������1���<�Z�,�'��9�����	��������� ��+�o� q�Sٍ    IEND�B`�
x
summary_pool2_0/image/0"]"U�PNG

   IHDR          ����   IDAT�cd�����p�ۈ���A��?C +�mT��    IEND�B`�

loss�E�>

conv1_weights��?

conv1_biases�N�>

conv2_weights��?

conv2_biases���>

fc1_weightsC^u?


fc1_biases�E>

fc2_weights��>?


fc2_biases�o]=

learning_rate
�#<^��Q      ��j	��R屍�Aذ*�

�
summary_data_0/image/0"�"��PNG

   IHDR          :���  IDAT��KN�@  �ah@
��`�֚�q�I7.<���������&�@]��M�Q$i!R~��Qc����l��d4��c��U,Z�ފP\���3*�Ճ�M�:+��(�kW�xV�>5w �3�	�Z7�S�Uu3�~�B"����Ép������^�6X����7S�܆uwaeT ��$�<�R%�<� hD�jx��}4���|i��|<C�FѲNS�8Q���!�cE�@��{_�v����Z2d�B0���淊��Yn�$^��ϊ����"$����{2MޮN    IEND�B`�
�
summary_conv_0/image/0"�"��PNG

   IHDR          :���  IDAT��MO�0 ��m��X�g�P!J�h$1��'���xQO��A�Hq1���Q�ml�y�]�T�^���D��d���[�Z�����@j�n<<>��h�J3]3�:��̄"�[^�&KJ߽����t˖I�#��ٽ:Ox�0=��"�����@R0e�Sec�+Bk�B��E��!�AM�V��j�i��S����v�nO�)�rSof�p��?!�V��*0���N<����|�Fg��ih����n��XU�@�dK�    IEND�B`�
k
summary_pool_0/image/0"Q"I�PNG

   IHDR          �d�W   IDAT�c�� L1 IA���j    IEND�B`�
�
summary_conv2_0/image/0"�"��PNG

   IHDR          �d�W   SIDAT�H �����H�������� 6�.���b	���*	"�/�������$����� �&��� ����V    IEND�B`�
p
summary_pool2_0/image/0"U"M�PNG

   IHDR          ����   IDAT�c`���# �1|�    IEND�B`�

loss��>

conv1_weights#C@

conv1_biases$�?

conv2_weights���?

conv2_biasesE�g?

fc1_weights��?


fc1_biases���>

fc2_weightsM��?


fc2_biases&m�>

learning_rate
�#<4nO�[      \x��	��챍�A��*�

�
summary_data_0/image/0"�"��PNG

   IHDR          :���  IDAT����uXB��� �T�;�<�9���-� �g � ���	����������  � �� ���'���#8�� �  � �������5k�+%A'��������(�,��:�%#��!/.���,�. ���*�����3=L��	�� F� �/	N�����	�&)�+3�����	-w�(��
���� ��8�N
�
	�'���#�������>�;UPgw�_;�    IEND�B`�
�
summary_conv_0/image/0"�"��PNG

   IHDR          :���  IDAT�������+������:2�	���
1��
����
�
�b��� ����,�6�#"�	�����g���������!��%����9����� �� ����4�
���������)�����%$ ��� *��� ��&(*�����"�@���.������$�EP�.
����B7����+E	 �.�����Qv�%[͓    IEND�B`�
k
summary_pool_0/image/0"Q"I�PNG

   IHDR          �d�W   IDAT�c�� L1 IA���j    IEND�B`�
�
summary_conv2_0/image/0"�"��PNG

   IHDR          �d�W   SIDAT�H ���$� �0�����+����8���*������22���"��'�������4'���!���l#    IEND�B`�
k
summary_pool2_0/image/0"P"H�PNG

   IHDR          ����   IDAT�c```�π
  I���    IEND�B`�

loss`A�>

conv1_weights���?

conv1_biases��>

conv2_weightsF�?

conv2_biases��>

fc1_weights1�_?


fc1_biases�!>

fc2_weights��?


fc2_biasesMT�<

learning_rate
�#<�E��o      +�P�	��R��A��*�

�
summary_data_0/image/0"�"��PNG

   IHDR          :���  IDAT��MN�@�wf>F��h4�VcH �'$,<^��Ν�
 Y��ҝ+uG\
m -��<���{1E'�nk��}�f:������s�K��*�a�z�_-�\5Fnwk�@ t����� �Y�K҃Wi$�q+ :}�����1=[��ïե:��y���f)�����=�����z�&�h�7N&���\���,A�{C�əV�A\�T줲�
r)r����^��������V��"�U�!�,+����%H���vC� �O    IEND�B`�
�
summary_conv_0/image/0"�"��PNG

   IHDR          :���   �IDAT��MK�0 ��͛�i��n����
���c�AD�x�$� E���hK����!�.��!y:>$�x�(�
��C����g��:ӗ�=[��ŢVѲq��M�Wc��4�}�����_�*���7�3D�g�.׉������C̷{/��Mڴ?��� 
���⅏G�^�Ν�y��+�`�t-���Pq�	9��`�q�H��Z� a���#�!D�P3�8��"d�'4�CV@"O)qQ�BXo��R��    IEND�B`�
�
summary_pool_0/image/0"�"��PNG

   IHDR          �d�W   SIDAT�H ��.��<@�;�!)����/J�4����,�����5� 	� 	�A=I��a�    IEND�B`�
�
summary_conv2_0/image/0"�"��PNG

   IHDR          �d�W   SIDAT�H ���"�����0?���+����� ������
�/��0�����̌�D|�(    IEND�B`�
q
summary_pool2_0/image/0"V"N�PNG

   IHDR          ����   IDAT�c�t��1Ni8  9T���    IEND�B`�

loss��>

conv1_weights�@

conv1_biasesl�?

conv2_weights��?

conv2_biases��*?

fc1_weightsN{�?


fc1_biasesʅ�>

fc2_weights�P�?


fc2_biases��,>

learning_rate
�#<���      �d#	P�<����A��*�

�
summary_data_0/image/0"�"��PNG

   IHDR          :���  IDAT��� � x-[ ���y-D��&t"����6*��"��	� �;L9L��� ���J� �����	�� �����0�������)���[�oD�8��� b���(���!�����*��&�@ �" 6���� c������� ���� �D4� -���a B ��-�	��� � w�L�[���D� ��� ).��x��J�l�g.@    IEND�B`�
�
summary_conv_0/image/0"�"��PNG

   IHDR          :���   �IDAT���R�0 �����h�S�����҅k��T�@�����9�E�#P-�S̼�N��;"1��Yy���]�b��`�Cl ����w<5�Ґ(,��J 3O�YlO#G��M���@si�М�x-C�=��f�dq���9�ho�¥[�D0x�H�����>cs��sw���tz��U��=� ��>H<gB�Ʊ��������؜��4�h���g
C�[��pHh�G    IEND�B`�
�
summary_pool_0/image/0"�"��PNG

   IHDR          �d�W   SIDAT�H �� /��=W���#m��%� ��	 J�  ��+ 8��� ������k��(�    IEND�B`�
�
summary_conv2_0/image/0"�"��PNG

   IHDR          �d�W   SIDAT�H ����	��#%���������	������ !�	��
��������!��CW�    IEND�B`�
y
summary_pool2_0/image/0"^"V�PNG

   IHDR          ����   IDAT�c��'#�����7z�?_ J,?d��)    IEND�B`�

loss�#�>

conv1_weightsma�?

conv1_biases�T?

conv2_weights;��?

conv2_biases-��>

fc1_weights���?


fc1_biases��z>

fc2_weightsIFT?


fc2_biases?s>

learning_rate
�#<̅�O1      ���7	˾� ���A��*�

�
summary_data_0/image/0"�"��PNG

   IHDR          :���   �IDAT��KN�@ ��13<:+-b4a햵��=��q���D6��H i��vک߇�M��2��Ks���8��2g[�8I� ���a3�Nzl� SO��Iڴ4��,_��O�����8Z��V7ՙe�]�A��s���H�1;Y��uRL
<�K��.��I@�sܓ�7Tn�C��k�:چ���':���$�R#�HG��x?X������-����`7eϪ��uE�}RM[Bvj^,�V��7�F��Kw�4l�B    IEND�B`�
�
summary_conv_0/image/0"�"��PNG

   IHDR          :���  IDAT���N�@  �Ng�
*e	Ĉ� ��Ƌ�~�������rB4a���R�A[;���~��q6��4�G=�)�b�g˲y*�SXa���Y�sx���� >�n���yԨ7��ǟ�b1uG+:V�w�>��e���B�+��K�-��c3BBd��(��AIϪQ�u���s�qQV9�xi��qRPGǙ��Q�r$�X�q!4���'�؅^Qe�#��ݚ�5DM�a��6����L�>W]]�(J�-Ӕ�o�+N����g�PT	 $y����T͚!    IEND�B`�
k
summary_pool_0/image/0"Q"I�PNG

   IHDR          �d�W   IDAT�c�� L1 IA���j    IEND�B`�
�
summary_conv2_0/image/0"�"��PNG

   IHDR          �d�W   SIDAT�H �������2*����� 	j�#�	��$չ%��������������.���	�� �,U3    IEND�B`�
k
summary_pool2_0/image/0"P"H�PNG

   IHDR          ����   IDAT�c`@�% X�-��    IEND�B`�

loss{��>

conv1_weightsl?

conv1_biases��>

conv2_weights�_?

conv2_biases{�>

fc1_weightsc�=?


fc1_biases� >

fc2_weights�1�>


fc2_biases�=

learning_rate
�#<I���      �Z�4	��w���A��*�

�
summary_data_0/image/0"�"��PNG

   IHDR          :���  IDAT���>���Ԃ������9�.������ ������������ ����(��"�2��~��"���+�������
8�����^�.��=��	*@� ��  � ��9��!�  �� ��/�+� ��+/����� � �+��9T �� �	�E5�
��`wA� ��8          B��"K�#-�����+�����	�����sr�dV�b    IEND�B`�
�
summary_conv_0/image/0"�"��PNG

   IHDR          :���   �IDAT��AJ�0 �&��I[�Z�F�Ft#�ˋy��RPT�qG���$M�$�Gn�������"�p����h�i.��П._�C^g�PB���j�x�b�v���,��<ÍJ�h�� G�t��%����
~�H�Vb�V[ _uދ��&٪�
��'4�i.��iAH�@������'�"� v������e]�JL��S��Y�����4��e�h�<��6�4��_W����c�s!�����PRvQ�}D    IEND�B`�
�
summary_pool_0/image/0"�"��PNG

   IHDR          �d�W   SIDAT�H ���,�      ���6   �������#� �
������
��!����2��!|k�@��_    IEND�B`�
�
summary_conv2_0/image/0"�"��PNG

   IHDR          �d�W   SIDAT�H ����������� ���� ��������/
������ ������� �h�"C
�A�    IEND�B`�
x
summary_pool2_0/image/0"]"U�PNG

   IHDR          ����   IDAT�cX���'�:Qvm6 @'��kG    IEND�B`�

loss��l>

conv1_weights�Я?

conv1_biases[�6?

conv2_weightsc��?

conv2_biases�1�>

fc1_weights܈g?


fc1_biases@g>

fc2_weightsdM?


fc2_biases3�>

learning_rate
�#<�܊%[      \x��	��V���A��*�

�
summary_data_0/image/0"�"��PNG

   IHDR          :���  IDAT���B�:��>����+	�� )�@�4��- ��
��:��������/������Ņ�������n� �	���� �* ��� Y  4��:���9���� ��������"��%�����?��� �������{
 ��������S����`��������O��� ���%�����"�� �9������� �2���	��Ͻ�A��R��    IEND�B`�
�
summary_conv_0/image/0"�"��PNG

   IHDR          :���  IDAT���N�`  `��(R2S�h͵�E� ��ޤ�l�՝��af�N<��3j��<》}�b��,.�!"�u)�.���'�)G�^1�C`
U�}��Xi�����T������loJ�9��뛬����K��\VPzH~��#{�P�/��K��-��ۊ�n�����?�B��_�wgg�������_����^U��Ĥ[������	�Wً�q�<�1:R~��j���^�4A��"a�Ѽ~+���.4�*�&aS��j�Z�t�&�W�    IEND�B`�
k
summary_pool_0/image/0"Q"I�PNG

   IHDR          �d�W   IDAT�c�� L1 IA���j    IEND�B`�
�
summary_conv2_0/image/0"�"��PNG

   IHDR          �d�W   SIDAT�H ������.� !��-�J���T=$.�,�����������g>�kJ-&�����2��`�#R�-    IEND�B`�
x
summary_pool2_0/image/0"]"U�PNG

   IHDR          ����   IDAT�cܥx������#��� *�m��Ԋ    IEND�B`�

loss�;�>

conv1_weights�΃?

conv1_biasesВ?

conv2_weightsf�`?

conv2_biases)J�>

fc1_weightsr.I?


fc1_biasesr�>

fc2_weights�?


fc2_biases@d�<

learning_rate
�#<N3R��      ��5	N
���A��*�	
�
summary_data_0/image/0"�"��PNG

   IHDR          :���  IDAT���K�`���~7�iP{�����v��%CD������ �ub�!����.Ef�M�{{�ur���^��z-��+ȏ��+x  JI��ߤg# ������Y=K2 s�z�y��-W��F �5
!X�� �N��D�1:��'�*]|�J�R\��'��xTX�x�j���Uݛ��gl[T);UG��Qļ�|գϣO�ﮁ���@r6�l�S���$)~>�*��f���hs�R�$�v�״��s*�v����<p�~�B:l���|    IEND�B`�
�
summary_conv_0/image/0"�"��PNG

   IHDR          :���   �IDAT��AJA��'��q�V��Jd��K���T�����R��3S2�־��ȊP_�?�_/^���r=#����ނ���2��\IPhN�r�S׀`VN�[�$��U	on�q��.I��AC�\9�*�I�H�ؓK;d# �DO�ʭ-�EhZ`�ܝ���s�X��7^z���=T�����v�M� ��~�cky�S	�    IEND�B`�
k
summary_pool_0/image/0"Q"I�PNG

   IHDR          �d�W   IDAT�c�� L1 IA���j    IEND�B`�
�
summary_conv2_0/image/0"�"��PNG

   IHDR          �d�W   SIDAT�H ���	���7�����@�� ���  � ��
V"�� 7�	��~�    IEND�B`�
m
summary_pool2_0/image/0"R"J�PNG

   IHDR          ����   IDAT�c�����ĀB  Q���{    IEND�B`�

lossKO?

conv1_weightshi�?

conv1_biasesrdi?

conv2_weights��?

conv2_biases��?

fc1_weights��?


fc1_biasesTv�>

fc2_weights�?


fc2_biases�>

learning_rate
�#<ӯ�3      �1	��U���A��*�

�
summary_data_0/image/0"�"��PNG

   IHDR          :���   �IDAT���J1 й%��v��
���o��J���fr�x����$~ nc�,q2)�%Ba1����:�x[h�G�ۯ�9֏,:�-,q�����:!�����.2�A���	G�|B����Y�ÕO=7�㾮��8pB�i������e8��	�V���G�"/� UB�"S�?0>�\���|�,���m��F)#�x�Hѡ)�'�-���>m�q��6    IEND�B`�
�
summary_conv_0/image/0"�"��PNG

   IHDR          :���   �IDAT��KN�0 ��x�81��e�D{�wcˊ-7`�eł�	����v��=q_��2[����wYtY�h8i���WA��ȏ\�hZrí�̂�8q��aJ�*����n�L�A�zȓ����.Vg.�q�?�&�o�+"	�Ps�@�P҇�K uFk�ʞ �gQ��*�g���A���^ϕ� BÞ\gԜ[������p��@�"�����$FV"NO�x��b�*�`���q�?֞�    IEND�B`�
�
summary_pool_0/image/0"�"��PNG

   IHDR          �d�W   SIDAT�H �� Y     ]  B  ;  MB!��  ��j ! 23������)��K���%"� M���Sz�	�=    IEND�B`�
�
summary_conv2_0/image/0"�"��PNG

   IHDR          �d�W   SIDAT�H ���
���
%��!�"*��� ���&�2�r�  �������Y�����������Tu �*Q��    IEND�B`�
p
summary_pool2_0/image/0"U"M�PNG

   IHDR          ����   IDAT�cX���  6:Z��5    IEND�B`�

loss�X1>

conv1_weights	>�?

conv1_biases7
?

conv2_weightse�?

conv2_biases"M�>

fc1_weights+}?


fc1_biases�l>

fc2_weights��E?


fc2_biases�� >

learning_rate
�#<�K1      ���7	���%���A��*�

�
summary_data_0/image/0"�"��PNG

   IHDR          :���  IDAT���N�0 �v����:΁DO� 7��'�����#O� xa1"��`ݟv�߇?����9$�A�4ݐ��D^�"�`���O��Z,vz'����SMh�솊3���
��J�����0�,�$��h�n���rZI��;+RA��R[|g����ׇ����ɉ���P��$���fϧ5a�E9�U��57	^�ڿS;*EsЅ�F��r?$��������E���K��Oo?Pk�j�Fo?7����ŧ��e6zs����K|��V4�    IEND�B`�
�
summary_conv_0/image/0"�"��PNG

   IHDR          :���  IDAT���N�0 жk�6��� $!.��|O�W0�h�w������+�X�m��s^8) �1N��B���2e[�:�Z�	a;\���~ص�	E��!%��	�����|���Q �at��Dv�F� �b?�[��Q��40U�����a���M�������		��i�����Ǎ<��3��&�Oڞ����O����X�pL*�Z��+�] �8���N�+oPљxٰejc7n�ޟ<_�uoHC`���+�����    IEND�B`�
k
summary_pool_0/image/0"Q"I�PNG

   IHDR          �d�W   IDAT�c�� L1 IA���j    IEND�B`�
�
summary_conv2_0/image/0"�"��PNG

   IHDR          �d�W   SIDAT�H ��t0�4�!���K���*�*��"2*�	���	���q���5�"������    IEND�B`�
k
summary_pool2_0/image/0"P"H�PNG

   IHDR          ����   IDAT�c`@�  
���    IEND�B`�

loss��*>

conv1_weights5�?

conv1_biases�ӛ>

conv2_weightsŭ6?

conv2_biases�1{>

fc1_weights�`
?


fc1_biases�K�=

fc2_weights��>


fc2_biases%�=

learning_rate
�#<�e0bo      +�P�	G-���A��*�

�
summary_data_0/image/0"�"��PNG

   IHDR          :���  IDAT���������  �/�# ��0�� �������	���� ���!B�� 	�?�  ��!G���0��o6�I0 ���$�����Q���,A�J8��E��v ��C�1���� O; C�� �%,�� ��� �
�6� �	�����  �8� �2���$���W ! 
��   �S�������	���� ��� ��� A �����,���b<s��Lu    IEND�B`�
�
summary_conv_0/image/0"�"��PNG

   IHDR          :���   �IDAT��YN�0 P�3^B��RAT������\�_��"D���q�ux�?]��]�s��k��<����Y?Y.گW�1�f{�Х��R��M�ar������"@���Ҡ<���3ۤ���f'W�8�bK�c��3��n�w5���C�3������Y-��W��L:M�i�"�〝���-P��g��5�m[�	 Jj�����Qs�U.:? �5�U��Eӟ4�@�B��ȏ���v��յ�    IEND�B`�
�
summary_pool_0/image/0"�"��PNG

   IHDR          �d�W   SIDAT�H ��  }$�4� ;"�*�1������ �/6��:��)�$
 0$��K?� 8�������`O(�v2    IEND�B`�
�
summary_conv2_0/image/0"�"��PNG

   IHDR          �d�W   SIDAT�H ������<��������M�-��'����������������������!��$��+�    IEND�B`�
l
summary_pool2_0/image/0"Q"I�PNG

   IHDR          ����   IDAT�c�q��}T  9�Sh��    IEND�B`�

loss�G>

conv1_weights_{�?

conv1_biasesc@?

conv2_weights0^?

conv2_biases�)�>

fc1_weights�Q1?


fc1_biases���=

fc2_weights��>


fc2_biasesm��<

learning_rate
�#<�S=      �(M	O�c4���AІ*�

�
summary_data_0/image/0"�"��PNG

   IHDR          :���   �IDAT���N�0 ���P
�񠉋�lc���"����70у^��)A��+h�>gm7ۗ�-��K��y �(=iY������{i�F�H�R-4�y�N���6f|�lu����C�Ia4�HĐ&���@��P=�_�:!	�R�L�0
��!f���*wِ��M�'�]��, w�{��JS���GRV3g ��"Jr�V�p��\�
`(<�9�y���W7���/ޙ���ց��0�����mpS�jP    IEND�B`�
�
summary_conv_0/image/0"�"��PNG

   IHDR          :���   �IDAT��MKBQ��w��j�Gׂ�7-ZD��V��V��*�D�MB���9s���W?�|{���&d\����z�_�n�̌��p��},�MU��������y2j�EWo0�y1�H�J������{����6r��%3t����aJ�3b���@U����p���E&D0b/8���րI�H@@��z�o6f'���/�eݏ�-�T�nҜS[$����6�
�J!Dv�*���<A��?(�ut �]u    IEND�B`�
m
summary_pool_0/image/0"S"K�PNG

   IHDR          �d�W   IDAT�cb``���@)  ]_`�E�    IEND�B`�
�
summary_conv2_0/image/0"�"��PNG

   IHDR          �d�W   SIDAT�H ��J��4��p�	$3�����G�,	��&����� ��8���������J4���    IEND�B`�
x
summary_pool2_0/image/0"]"U�PNG

   IHDR          ����   IDAT�cԹr�����A���a�O��� .������    IEND�B`�

lossq��>

conv1_weights��@

conv1_biases�d?

conv2_weights�$�?

conv2_biases@d?

fc1_weights��?


fc1_biases,%�>

fc2_weights�A�?


fc2_biasesQXg>

learning_rate
�#<E��]M      z��x	9��;���A��*�

�
summary_data_0/image/0"�"��PNG

   IHDR          :���  IDAT�]��J�@ ����\��T��u��Q8��"n��M�6�����7p*8�>����)�\�@G�"5*6m*��7����1���yG#Y?�`�p"%������]�*L`�CD�ѳjFC. �Ս���#M3�1�-e��#=χ��&iIJ|�P�^�w)纥���1�7쮎U�1Qsؘ�	=b,�_�us9y��l@O�g+'#9�����ZD�����/=�w���{[x�'�W��h�
��wmiĈLo}�.~J�_-�Z�    IEND�B`�
�
summary_conv_0/image/0"�"��PNG

   IHDR          :���  IDAT��QN�0 ������
aNED4>�_=�W��bt� �8F��nt���<��Ide-���^@`^�1�j�n����ӻ�:���'n. �ђV�Cq����rf.ը�r8���b��$���%1��Ţ8��L8�#d&�r
k�x����
:�.ʳ��H��@Ը�L�b����6�0BvD\�f�Tli��440��qo�°�(er����F�spZ�vzaK�ˏx�~k����5y����B�y2m#�	    IEND�B`�
y
summary_pool_0/image/0"_"W�PNG

   IHDR          �d�W   IDAT�c` 030000t~̢���A��#� .��fnL    IEND�B`�
�
summary_conv2_0/image/0"�"��PNG

   IHDR          �d�W   SIDAT�H ���F���������	�]2�.[.�(�2��%���& ���WQ�,3�������a�Ljb��    IEND�B`�
y
summary_pool2_0/image/0"^"V�PNG

   IHDR          ����   IDAT�c�h�X�ms毿��20H3| [`[MW    IEND�B`�

loss<>

conv1_weights2V�>

conv1_biases�2>

conv2_weightsj<?

conv2_biases\�+>

fc1_weights��?


fc1_biases�8�=

fc2_weights���>


fc2_biases���;

learning_rate
�#<���-L      �\��	yTB���A��*�

�
summary_data_0/image/0"�"��PNG

   IHDR          :���  IDAT�U��J�@ ���%�%���ɸAAqAw]�p�9\��D�7�Q�b��A+-����g���ю�p����
z�Oq|u-��5���D��<�~h��-�o����N�'��q�Ȫ���E5����u�C�ʸ&�Q�W��+{σ��Rҟ,G�F\I[�������"[c6c䧲��L*�dIH��3�2д�t��ǎ�fೂ:�Z��� p�M-eqw���͖f����{�g�V��]��97h�㑫� ��uG�jI�    IEND�B`�
�
summary_conv_0/image/0"�"��PNG

   IHDR          :���   �IDAT���N�@ �awXv)P0�KIԃ�1z0^��Wz�W�ִhL,�-�,;��^��lZ	���dq6�舚sj �>t����H 6&Dc9bt�b�y��ҁ��x�W���mbO��+���^����T���H�#ӎ+;�`�P%�~�L~E���	�M����'���x 4�e�VE�tʢ8U��"�2�uW�L�I�/o���ή�^�k�򾭧���q��m-���<���ǻu��m�%y|�x��    IEND�B`�
�
summary_pool_0/image/0"o"g�PNG

   IHDR          �d�W   .IDAT�c������_r���য়�8Y<�����@ �Ci��r E[
��E��    IEND�B`�
�
summary_conv2_0/image/0"�"��PNG

   IHDR          �d�W   SIDAT�H ���
�	���� ��	�(�����<
�1�������/�r x�#    IEND�B`�
p
summary_pool2_0/image/0"U"M�PNG

   IHDR          ����   IDAT�cp[��?8000  .)�|l�    IEND�B`�

loss"B>

conv1_weights^��?

conv1_biasesq!?

conv2_weights���?

conv2_biases��>

fc1_weights>�E?


fc1_biases�1>

fc2_weights��?


fc2_biases���=

learning_rate
�#<�v�`�      �5J		sdI���A��*�

�
summary_data_0/image/0"�"��PNG

   IHDR          :���  IDAT�-��K�` ��/obˑ`MY����)#O��עC۽�v���Nu�%XةA�ɱF�`��,q��u�9>�5C�J�Ң�g?Q���\$0��R:��BF�*�}��h����*86s���Y���\6�[|���AGk��/R�A�+E�F��L���C�4��*@xt	�P����;�$	����e`��>H>�\�-�Թl��65H'�Z�z
ΌqM��#�����+��X.����l9Ⱥ<e�3)z$�lD�U��u�	�n#�/����xjb>�`}    IEND�B`�
�
summary_conv_0/image/0"�"��PNG

   IHDR          :���   �IDAT��MO�0 ������/7u8�I��f����y7f��šƀAe�y諉���Ĺw�]���=���#Q�M��4�A?.��[��\���3��=^c��L7ot��̍���;��-TH�X����r}����6�숒Z��9��1�9" �753d^K�|�
2�ۦ� 6c��t���:C,KW�v��Ǝ*n �e~W���ee�`�2�<���{�P6���l봛��\cU	a��NPũ%���q�?    IEND�B`�
�
summary_pool_0/image/0"�"��PNG

   IHDR          �d�W   SIDAT�H ��       US  ���Ց�����3y <-�� 	���"� ��*�����I�����S|��    IEND�B`�
�
summary_conv2_0/image/0"�"��PNG

   IHDR          �d�W   SIDAT�H ����D��������!��������������+

��5���������.��#��|    IEND�B`�
x
summary_pool2_0/image/0"]"U�PNG

   IHDR          ����   IDAT�c0��ݫN>��\ I�?D��W    IEND�B`�

lossD�>

conv1_weightsƞm?

conv1_biases�r�>

conv2_weightsx��?

conv2_biases��>

fc1_weights�B�?


fc1_biases��>

fc2_weightsg,%?


fc2_biaseso��<

learning_rate
�#<t��r3      �1	8LgP���A�*�

�
summary_data_0/image/0"�"��PNG

   IHDR          :���   �IDAT��=O�@ P���54m�@Ň�������OFL�F�HH��DE�4�]�6�G DSt>'�l�Cd�L'!�h�Ճ"��AG̦ l�� ��O蕮y�xuX�Ӂ!ŋw����h���w�&,J!��^ֿv����c�lVi�v=������L����N5�3%'Wd�1�@=CU�È��d0?E��I�g])�j�l�弥I�Kߟ����X(�*�.k���z����    IEND�B`�
�
summary_conv_0/image/0"�"��PNG

   IHDR          :���   �IDAT���N�0 P��k��l�ɘ��O�*���w��|����D@�9v�Z�!��&�$E�-,
d����:�Af�vg�DI֣��r.�aL���h� 7��Ξ|.O	7Z�`��/�:�����y��	g�W��j�DҴ�7Ϻw�+��q��'��:���U+�D�'?�K[I}�������pݖ�E�Ȕ��aC�Dp��!(}̹m-����A��'~�:hM�Ŕ��9p��ƈ@��sN�Db����n~a��<    IEND�B`�
�
summary_pool_0/image/0"r"j�PNG

   IHDR          �d�W   1IDAT�c` 6��+5�0���_���T/�|����)�X��= ����*    IEND�B`�
�
summary_conv2_0/image/0"�"��PNG

   IHDR          �d�W   SIDAT�H ��Q�+(("�-����4&��I�e���� �K54�����������/ ����� �0�)�.C0    IEND�B`�
r
summary_pool2_0/image/0"W"O�PNG

   IHDR          ����   IDAT�c`�~��f� $N�	V�    IEND�B`�

loss��A>

conv1_weights�/�?

conv1_biasesU�?

conv2_weightsYa�?

conv2_biases���>

fc1_weights��K?


fc1_biases�E9>

fc2_weightsDw?


fc2_biasesFk�=

learning_rate
�#<�id5      ,l4	��qW���Aح*�

�
summary_data_0/image/0"�"��PNG

   IHDR          :���  IDAT��=K�@ �˛�r��	m����� .����_�N7'�`�I��V��w�&��h�L���0�����F�
t�y��9�����d׶��bk(�H�L�0Eӎ���g`�)�}�.�Ѩ8K����.�JT�-�p��l���U�E��v2}��'�w`���9�i:�'��MeY�����D՚�V�}S�7�F�ƕ������gGP�V�`�_�W�_��oO�/M�{7ڝ��fz&�}o[XA�]�Z\�Id`�T*yp��M    IEND�B`�
�
summary_conv_0/image/0"�"��PNG

   IHDR          :���  IDAT��MO�0 `ھ-(QG����2�z��x�gy��xRcԙ�a|8>�����=2iw���_���+U�{�����{���.�E��=��h��4��M [x��L��Ż��AI!�i�Xޕ�2�,��V���'���%���b>L΃�
hӄ|d�X��u7|Q�.-��m�nN��Q�{�jYU���|�4���ci�/Y��]���6;��)�?���`����f�!�������ܦt|4�+�;G�w����y��TX�    IEND�B`�
k
summary_pool_0/image/0"Q"I�PNG

   IHDR          �d�W   IDAT�c�� L1 IA���j    IEND�B`�
�
summary_conv2_0/image/0"�"��PNG

   IHDR          �d�W   SIDAT�H ��� ��5���	�"+����!���%��
���	#���ϐ*՛'����$����X��!`�!x���    IEND�B`�
k
summary_pool2_0/image/0"P"H�PNG

   IHDR          ����   IDAT�c`@�  
���    IEND�B`�

loss ͳ>

conv1_weights��7@

conv1_biases�?

conv2_weightszq�?

conv2_biases��3?

fc1_weights�@�?


fc1_biases�Xy>

fc2_weights{+N?


fc2_biases�ӌ=

learning_rate
�#<���D      ���	���^���A��*�

�
summary_data_0/image/0"�"��PNG

   IHDR          :���  IDAT����&|�;�������A��!�����������@�#���    3#'�  +�� �Ѿ�&��+	�� �����'��������-1���
	"

�4������� ��$��#����,� �������
��������  ������'���@B��� �9;b�������    ��� ��� M�� � �#��?���uw�����    IEND�B`�
�
summary_conv_0/image/0"�"��PNG

   IHDR          :���   �IDAT��IN�0 P?�X�4eR] 6�kq:n���
�)C�h�8N���=|�;�o�I�s�I�ח�5d'�
��G���z��p�ԟ�n���y$��m�kA)��`����ݤ�:4r4��P�;D���š-�9�VoM��3C�2js�KM<��}� 9�e,񼔛��ԞMJQ��K`��jV�IvbB�-���	FD�~��m�1���6L��2�H�̈́�|����7�Y��2-�s�����$~�U���    IEND�B`�
x
summary_pool_0/image/0"^"V�PNG

   IHDR          �d�W   IDAT�cd``�p2�������@h�� v�n���    IEND�B`�
�
summary_conv2_0/image/0"�"��PNG

   IHDR          �d�W   SIDAT�H ���)����"�
�����-�����H����*k�8��)���
�=!;����������]�$'r��^    IEND�B`�
i
summary_pool2_0/image/0"N"F�PNG

   IHDR          ����   IDAT�c`@�  ?[L�    IEND�B`�

loss��>

conv1_weightsR��?

conv1_biases���>

conv2_weights�y�?

conv2_biasesfԮ>

fc1_weights�5i?


fc1_biases|�V>

fc2_weights�C,?


fc2_biasesx��=

learning_rate
�#<)j<Z<      B�	A��e���A��*�

�
summary_data_0/image/0"�"��PNG

   IHDR          :���   �IDAT��]K�0��'oҤ��vjU/�f���/z�+E?��쒺���.�Ր?]�o���c������|��_�$̲f1&�kH�a��ݜ9�31�z�D����"A�� Bx-J��8��9�JSi� �͞*�j�os@���1�4��uaml���Z�I}Peo���l���'��=L�>1���"tv�4�.w�D�d������]G����L��6���n���e�KuF    IEND�B`�
�
summary_conv_0/image/0"�"��PNG

   IHDR          :���   �IDAT���N�0Й�ɣ�6�6��>��f��		���$�c{<��o�P��C'y1}�_�9T�}�e���dw��p�n�R;�����%<��i�W9�\�r���f�W�UN��R{ך��<�a�]V��5%�D�(	05��ͪo�	����� Dq��y�Է�H�F�H𛳑	�	FT"�8�IrjKa�Z,��'�)[V����t�*���&�7/�k�?$��a�    IEND�B`�
�
summary_pool_0/image/0"�"��PNG

   IHDR          �d�W   SIDAT�H ��   � qH1T�����" ����:=D ������ �
��� ����� ����C �g�ԛ�f    IEND�B`�
�
summary_conv2_0/image/0"�"��PNG

   IHDR          �d�W   SIDAT�H ����� &8�� "SB��
���������	���L��������9�Tfm5X    IEND�B`�
x
summary_pool2_0/image/0"]"U�PNG

   IHDR          ����   IDAT�cX���j�\qnQn >@��ǡ    IEND�B`�

loss 6�>

conv1_weights$�?

conv1_biases��e?

conv2_weights��?

conv2_biases���>

fc1_weights���?


fc1_biases�>

fc2_weights�-4?


fc2_biasesTo<

learning_rate
�#<��GT�      ϖ%	xG�m���A��*�	
�
summary_data_0/image/0"�"��PNG

   IHDR          :���   �IDAT���JQ�o���YXHZY�����/⋋���I#X�����x�<�e������i�Л�۳�ii}�3�	 �!S(�PR�D �Ҏf��pMp�6�������S\�˾����i۵��EEr�p5up`WCCY��aǅ��@4gHOB^��aQ�S���,R��{Xic� �l�7�Yy'�"~c �p��    IEND�B`�
�
summary_conv_0/image/0"�"��PNG

   IHDR          :���  IDAT��MO�0 �Ҿ�d�P�x���?.^<{�w�+w��E�F.���iX��AK��X˃�¢sذ�wB@�.�>��CF�P�z��U���ج�e�W�T�m�~����B�o`�%��2��qU��D����%Ʒ�t�8s�ib �ƭ�AXJe7���D[�M��ϣo<��1��gq� ��u%5�>L� y��0�6{�	���9���2�Z�(M��kY�E��$V��~�׬� ��В����!���A���w�6К�    IEND�B`�
l
summary_pool_0/image/0"R"J�PNG

   IHDR          �d�W   IDAT�c` ���e 0_�7L    IEND�B`�
�
summary_conv2_0/image/0"�"��PNG

   IHDR          �d�W   SIDAT�H ����;=������W�!
����
�3#����
�������E>6��5@��"H
�E�    IEND�B`�
k
summary_pool2_0/image/0"P"H�PNG

   IHDR          ����   IDAT�c`@�  
���    IEND�B`�

loss*t(>

conv1_weights]Gs?

conv1_biases�/�>

conv2_weights-GT?

conv2_biases|w�>

fc1_weights:�^?


fc1_biasesS�B>

fc2_weights��??


fc2_biases�v >

learning_rate
�#<�\;TN      @4{	e ]u���A��*�

�
summary_data_0/image/0"�"��PNG

   IHDR          :���  IDAT���N�@ ���KC(P @C�����:��B��N̼�/��k�8���d4�	"C��w^����;!�u��d��~Y"y�SQ7Ծ���JI��H[@ ��; `� x	tOժ(�VlƠ�!�]�6K	)嶹�@@�����V��{�0���Y1�4����X�ѬQE.���?N����*��~KN*:���1�h���Yڃ�o/W*P�\�����,:m^��M���7�}]̂�K>c�-F<�l��m��&���p�������%���ו��4`C���o�,��J    IEND�B`�
�
summary_conv_0/image/0"�"��PNG

   IHDR          :���  IDAT���N�@ P�t��$PI���.ܸ���ĝ.�[%jbB���`M���P:�v�9�L7�Y�+� l�,�	2H%ّ�q$�pW�"3AAD���fu�Q�Dea)�zX@���н&4��L�@y��lT�eE"[Z���C�@Df������_��C���*M����d�<}o+�gI�Z�f�Mq����~U�{7?��'?����At����G�}������^ǃ.<�_��N��5�}�"+����	�8��޾@0�}�:�2��X������6���    IEND�B`�
k
summary_pool_0/image/0"Q"I�PNG

   IHDR          �d�W   IDAT�c�� L1 IA���j    IEND�B`�
�
summary_conv2_0/image/0"�"��PNG

   IHDR          �d�W   SIDAT�H ����������������9� ���*'��������		YO�������5 �� �|��    IEND�B`�
m
summary_pool2_0/image/0"R"J�PNG

   IHDR          ����   IDAT�c�����ĀB  Q���{    IEND�B`�

loss���>

conv1_weights)o?

conv1_biases<�>

conv2_weightsT�?

conv2_biases��>

fc1_weights�:�?


fc1_biases�7�>

fc2_weights*�^?


fc2_biases���=

learning_rate
�#<���@n      :�	,br}���A��*�

�
summary_data_0/image/0"�"��PNG

   IHDR          :���   �IDAT��=KBQ���=�\�⍸��&�2�k�Bh�c� ?BCK}�3�߱Qph	�B4�����R A���f��<+lm���5���o�)�`׈�6����mt�$�##܅�z�F$���l�)'阷ٙ�o�L���1f����	�J��.T{�2���:�,��" 3�����D�4���T�`$} ��2@�R�aT�j��������«���k=)�o����
��:�#��.�H�r�    IEND�B`�
�
summary_conv_0/image/0"�"��PNG

   IHDR          :���  IDAT���N�@ `f��R �B�֘b^\�Ɠ���z�f��hSB-�E��2��!= ���5�{�	��*%��Կ���C�j؞��Ɂ���!�<z �+���8�p�ԝ:,eߥ;Y�;����K�²�28���u>�ߨ}����â^�2pP�+����nܻ�h��Sz6ʯ�{}B�p�43���.[ ��q�F_���/�#V�βs�r*�;ܕ�by�#��>qJ���lmձ:l�H��x�U��}�m�9p|�)�$    IEND�B`�
�
summary_pool_0/image/0"�"��PNG

   IHDR          �d�W   SIDAT�H ��     +��  ����  �1-�
 Ԕ���(�na42)��������� �=�0�m    IEND�B`�
�
summary_conv2_0/image/0"�"��PNG

   IHDR          �d�W   SIDAT�H ���� ���
��.
���2��%	���4�����0 �  ����'����H���!#���i!)ea    IEND�B`�
x
summary_pool2_0/image/0"]"U�PNG

   IHDR          ����   IDAT�c��������A������A��� ,i�%=W    IEND�B`�

loss�u�>

conv1_weightsN.�?

conv1_biases�C7?

conv2_weights�h�?

conv2_biases/��>

fc1_weightsR �?


fc1_biases'�$>

fc2_weights� <?


fc2_biases��8=

learning_rate
�#<�� q+      A��	�Ś����A��*�

�
summary_data_0/image/0"�"��PNG

   IHDR          :���   �IDAT���N�0 �{��8qڤT�(���T�10�ʗTLl Ā(HEjx5m^$�c���ߩ��~p����٠u�۠V��$�8��J��5�{.�PJ��fۭ�0q8Ψƍ�~��\�	��<���VmD*2�p����ԔK�G8��kն{�#.':L�^G�Rk���D���>7;��&L����Ӊ�0�}Xʠ��V}�1_�1����Y�$�������Z2n~sW�A��x���g���h1�� ��pqe�fG�    IEND�B`�
�
summary_conv_0/image/0"�"��PNG

   IHDR          :���  IDAT���N�@ �a�N+0�RZ�6��ĕ��?�4���`�h�GJA���Q���-gJ5�V�t�Epﴃ��H�����t�QQ'� �i�a�%C�2�SR�\#�IYS�*�P�*c"rҲfA���+1*r�<<w�h����ǩ�B7����|�E%�h�y򋛺�7d���P�f��]��[��:��L����ޠBs���N�ӷHӉbW�����F�2��W��>�'���|�_�y�})��P#���xuFE��    IEND�B`�
j
summary_pool_0/image/0"P"H�PNG

   IHDR          �d�W   IDAT�c` �G� 8 �0<�    IEND�B`�
�
summary_conv2_0/image/0"�"��PNG

   IHDR          �d�W   SIDAT�H ������	����,������&�	������/�Ϫ���������ַ"�9��"`���    IEND�B`�
o
summary_pool2_0/image/0"T"L�PNG

   IHDR          ����   IDAT�c�g`�e@���,� �����    IEND�B`�

lossrvb>

conv1_weights�Ip?

conv1_biases�?

conv2_weights5�w?

conv2_biases��>

fc1_weights<�J?


fc1_biasesd4>

fc2_weights�h�>


fc2_biases���=

learning_rate
�#<�"�D      ���	�������A��*�

�
summary_data_0/image/0"�"��PNG

   IHDR          :���  IDAT���_�7������,>�y�����
1�.<�$	��A��� ����&%����� ��%�
��
	������ ���������0������ ���
��"��  W�@����������	����Ej����������`ц������i�6
�ғ��������W,� ����������e M	*��&��wU�$������*�=G2K�    IEND�B`�
�
summary_conv_0/image/0"�"��PNG

   IHDR          :���   �IDAT��MO�0 ���۵]Q��aL�x2���W��^�ꂀ���t[[���4���YJZ=}cavW����Y�T~bР.�7����z[~� �}�+҈�N�r��v�j!����{����x�s��Y�^Uj�t6�-2VK[]{�Q#����.��HE���wn)��r�u�+ȉ�8�cj% �~z7"��~1	g+c�*J�I2�Y̞�f�8H�l&K�ɯ)1�K��t�77C��Ubtc�Lq	�+�    IEND�B`�
t
summary_pool_0/image/0"Z"R�PNG

   IHDR          �d�W   IDAT�c` 8�A�7Ch���є  W��f�7    IEND�B`�
�
summary_conv2_0/image/0"�"��PNG

   IHDR          �d�W   SIDAT�H �������<"����� ����.���3�����>�
�����
���)�(��S�7�S    IEND�B`�
m
summary_pool2_0/image/0"R"J�PNG

   IHDR          ����   IDAT�c�����ĀB  Q���{    IEND�B`�

loss�{�>

conv1_weightsA�?

conv1_biases�d0?

conv2_weights��~?

conv2_biases۔>

fc1_weights*Ql?


fc1_biases��>

fc2_weights6�?


fc2_biasesu/B=

learning_rate
�#<�h