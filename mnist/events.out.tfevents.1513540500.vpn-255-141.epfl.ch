       ŁK"	   ĺ˛ÖAbrain.Event:2)eť]T     ÉS	ăú6ĺ˛ÖA"Đ¨
l
PlaceholderPlaceholder*
dtype0*
shape:*&
_output_shapes
:
^
Placeholder_1Placeholder*
dtype0*
shape
:*
_output_shapes

:
r
Placeholder_2Placeholder*
dtype0*
shape:Ŕ=*(
_output_shapes
:Ŕ=
o
truncated_normal/shapeConst*%
valueB"             *
dtype0*
_output_shapes
:
Z
truncated_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
\
truncated_normal/stddevConst*
valueB
 *ÍĚĚ=*
dtype0*
_output_shapes
: 
§
 truncated_normal/TruncatedNormalTruncatedNormaltruncated_normal/shape*
dtype0*&
_output_shapes
: *
seed2Ž*
T0*
seedą˙ĺ)

truncated_normal/mulMul truncated_normal/TruncatedNormaltruncated_normal/stddev*
T0*&
_output_shapes
: 
u
truncated_normalAddtruncated_normal/multruncated_normal/mean*
T0*&
_output_shapes
: 

Variable
VariableV2*
dtype0*
shape: *&
_output_shapes
: *
shared_name *
	container 
Ź
Variable/AssignAssignVariabletruncated_normal*
use_locking(*
_class
loc:@Variable*
T0*
validate_shape(*&
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
zerosConst*
valueB *    *
dtype0*
_output_shapes
: 
v

Variable_1
VariableV2*
dtype0*
shape: *
_output_shapes
: *
shared_name *
	container 

Variable_1/AssignAssign
Variable_1zeros*
use_locking(*
_class
loc:@Variable_1*
T0*
validate_shape(*
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
truncated_normal_1/shapeConst*%
valueB"          @   *
dtype0*
_output_shapes
:
\
truncated_normal_1/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
^
truncated_normal_1/stddevConst*
valueB
 *ÍĚĚ=*
dtype0*
_output_shapes
: 
Ť
"truncated_normal_1/TruncatedNormalTruncatedNormaltruncated_normal_1/shape*
dtype0*&
_output_shapes
: @*
seed2Ž*
T0*
seedą˙ĺ)

truncated_normal_1/mulMul"truncated_normal_1/TruncatedNormaltruncated_normal_1/stddev*
T0*&
_output_shapes
: @
{
truncated_normal_1Addtruncated_normal_1/multruncated_normal_1/mean*
T0*&
_output_shapes
: @


Variable_2
VariableV2*
dtype0*
shape: @*&
_output_shapes
: @*
shared_name *
	container 
´
Variable_2/AssignAssign
Variable_2truncated_normal_1*
use_locking(*
_class
loc:@Variable_2*
T0*
validate_shape(*&
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
ConstConst*
valueB@*ÍĚĚ=*
dtype0*
_output_shapes
:@
v

Variable_3
VariableV2*
dtype0*
shape:@*
_output_shapes
:@*
shared_name *
	container 

Variable_3/AssignAssign
Variable_3Const*
use_locking(*
_class
loc:@Variable_3*
T0*
validate_shape(*
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
valueB"      *
dtype0*
_output_shapes
:
\
truncated_normal_2/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
^
truncated_normal_2/stddevConst*
valueB
 *ÍĚĚ=*
dtype0*
_output_shapes
: 
Ľ
"truncated_normal_2/TruncatedNormalTruncatedNormaltruncated_normal_2/shape*
dtype0* 
_output_shapes
:
*
seed2Ž*
T0*
seedą˙ĺ)

truncated_normal_2/mulMul"truncated_normal_2/TruncatedNormaltruncated_normal_2/stddev*
T0* 
_output_shapes
:

u
truncated_normal_2Addtruncated_normal_2/multruncated_normal_2/mean*
T0* 
_output_shapes
:



Variable_4
VariableV2*
dtype0*
shape:
* 
_output_shapes
:
*
shared_name *
	container 
Ž
Variable_4/AssignAssign
Variable_4truncated_normal_2*
use_locking(*
_class
loc:@Variable_4*
T0*
validate_shape(* 
_output_shapes
:

q
Variable_4/readIdentity
Variable_4*
_class
loc:@Variable_4*
T0* 
_output_shapes
:

V
Const_1Const*
valueB*ÍĚĚ=*
dtype0*
_output_shapes	
:
x

Variable_5
VariableV2*
dtype0*
shape:*
_output_shapes	
:*
shared_name *
	container 

Variable_5/AssignAssign
Variable_5Const_1*
use_locking(*
_class
loc:@Variable_5*
T0*
validate_shape(*
_output_shapes	
:
l
Variable_5/readIdentity
Variable_5*
_class
loc:@Variable_5*
T0*
_output_shapes	
:
i
truncated_normal_3/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
\
truncated_normal_3/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
^
truncated_normal_3/stddevConst*
valueB
 *ÍĚĚ=*
dtype0*
_output_shapes
: 
¤
"truncated_normal_3/TruncatedNormalTruncatedNormaltruncated_normal_3/shape*
dtype0*
_output_shapes
:	*
seed2Ž*
T0*
seedą˙ĺ)

truncated_normal_3/mulMul"truncated_normal_3/TruncatedNormaltruncated_normal_3/stddev*
T0*
_output_shapes
:	
t
truncated_normal_3Addtruncated_normal_3/multruncated_normal_3/mean*
T0*
_output_shapes
:	


Variable_6
VariableV2*
dtype0*
shape:	*
_output_shapes
:	*
shared_name *
	container 
­
Variable_6/AssignAssign
Variable_6truncated_normal_3*
use_locking(*
_class
loc:@Variable_6*
T0*
validate_shape(*
_output_shapes
:	
p
Variable_6/readIdentity
Variable_6*
_class
loc:@Variable_6*
T0*
_output_shapes
:	
T
Const_2Const*
valueB*ÍĚĚ=*
dtype0*
_output_shapes
:
v

Variable_7
VariableV2*
dtype0*
shape:*
_output_shapes
:*
shared_name *
	container 

Variable_7/AssignAssign
Variable_7Const_2*
use_locking(*
_class
loc:@Variable_7*
T0*
validate_shape(*
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
´
Conv2DConv2DPlaceholderVariable/read*
paddingSAME*
data_formatNHWC*
T0*&
_output_shapes
: *
strides
*
use_cudnn_on_gpu(
s
BiasAddBiasAddConv2DVariable_1/read*&
_output_shapes
: *
T0*
data_formatNHWC
F
ReluReluBiasAdd*
T0*&
_output_shapes
: 

MaxPoolMaxPoolRelu*
ksize
*
data_formatNHWC*
T0*&
_output_shapes
: *
strides
*
paddingSAME
´
Conv2D_1Conv2DMaxPoolVariable_2/read*
paddingSAME*
data_formatNHWC*
T0*&
_output_shapes
:@*
strides
*
use_cudnn_on_gpu(
w
	BiasAdd_1BiasAddConv2D_1Variable_3/read*&
_output_shapes
:@*
T0*
data_formatNHWC
J
Relu_1Relu	BiasAdd_1*
T0*&
_output_shapes
:@

	MaxPool_1MaxPoolRelu_1*
ksize
*
data_formatNHWC*
T0*&
_output_shapes
:@*
strides
*
paddingSAME
^
Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
d
ReshapeReshape	MaxPool_1Reshape/shape*
Tshape0*
T0*
_output_shapes
:	
z
MatMulMatMulReshapeVariable_4/read*
transpose_b( *
transpose_a( *
T0*
_output_shapes
:	
M
addAddMatMulVariable_5/read*
T0*
_output_shapes
:	
=
Relu_2Reluadd*
T0*
_output_shapes
:	
z
MatMul_1MatMulRelu_2Variable_6/read*
transpose_b( *
transpose_a( *
T0*
_output_shapes

:
P
add_1AddMatMul_1Variable_7/read*
T0*
_output_shapes

:
d
Slice/beginConst*%
valueB"                *
dtype0*
_output_shapes
:
c

Slice/sizeConst*%
valueB"   ˙˙˙˙˙˙˙˙   *
dtype0*
_output_shapes
:
r
SliceSlicePlaceholderSlice/begin
Slice/size*&
_output_shapes
:*
T0*
Index0
`
Const_3Const*%
valueB"             *
dtype0*
_output_shapes
:
X
MinMinSliceConst_3*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
G
subSubSliceMin*
T0*&
_output_shapes
:
`
Const_4Const*%
valueB"             *
dtype0*
_output_shapes
:
V
MaxMaxsubConst_4*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
J
mul/yConst*
valueB
 *  C*
dtype0*
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
Reshape_1/shapeConst*!
valueB"         *
dtype0*
_output_shapes
:
i
	Reshape_1ReshapetruedivReshape_1/shape*
Tshape0*
T0*"
_output_shapes
:
c
transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:
k
	transpose	Transpose	Reshape_1transpose/perm*
Tperm0*
T0*"
_output_shapes
:
h
Reshape_2/shapeConst*%
valueB"˙˙˙˙         *
dtype0*
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
valueB Bsummary_data_0*
dtype0*
_output_shapes
: 

summary_data_0ImageSummarysummary_data_0/tag	Reshape_2*

max_images*
T0*
	bad_colorB:˙  ˙*
_output_shapes
: 
f
Slice_1/beginConst*%
valueB"                *
dtype0*
_output_shapes
:
e
Slice_1/sizeConst*%
valueB"   ˙˙˙˙˙˙˙˙   *
dtype0*
_output_shapes
:
s
Slice_1SliceConv2DSlice_1/beginSlice_1/size*&
_output_shapes
:*
T0*
Index0
`
Const_5Const*%
valueB"             *
dtype0*
_output_shapes
:
\
Min_1MinSlice_1Const_5*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
M
sub_1SubSlice_1Min_1*
T0*&
_output_shapes
:
`
Const_6Const*%
valueB"             *
dtype0*
_output_shapes
:
Z
Max_1Maxsub_1Const_6*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
L
mul_1/yConst*
valueB
 *  C*
dtype0*
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
Reshape_3/shapeConst*!
valueB"         *
dtype0*
_output_shapes
:
k
	Reshape_3Reshape	truediv_1Reshape_3/shape*
Tshape0*
T0*"
_output_shapes
:
e
transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:
o
transpose_1	Transpose	Reshape_3transpose_1/perm*
Tperm0*
T0*"
_output_shapes
:
h
Reshape_4/shapeConst*%
valueB"˙˙˙˙         *
dtype0*
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
valueB Bsummary_conv_0*
dtype0*
_output_shapes
: 

summary_conv_0ImageSummarysummary_conv_0/tag	Reshape_4*

max_images*
T0*
	bad_colorB:˙  ˙*
_output_shapes
: 
f
Slice_2/beginConst*%
valueB"                *
dtype0*
_output_shapes
:
e
Slice_2/sizeConst*%
valueB"   ˙˙˙˙˙˙˙˙   *
dtype0*
_output_shapes
:
t
Slice_2SliceMaxPoolSlice_2/beginSlice_2/size*&
_output_shapes
:*
T0*
Index0
`
Const_7Const*%
valueB"             *
dtype0*
_output_shapes
:
\
Min_2MinSlice_2Const_7*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
M
sub_2SubSlice_2Min_2*
T0*&
_output_shapes
:
`
Const_8Const*%
valueB"             *
dtype0*
_output_shapes
:
Z
Max_2Maxsub_2Const_8*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
L
mul_2/yConst*
valueB
 *  C*
dtype0*
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
Reshape_5/shapeConst*!
valueB"         *
dtype0*
_output_shapes
:
k
	Reshape_5Reshape	truediv_2Reshape_5/shape*
Tshape0*
T0*"
_output_shapes
:
e
transpose_2/permConst*!
valueB"          *
dtype0*
_output_shapes
:
o
transpose_2	Transpose	Reshape_5transpose_2/perm*
Tperm0*
T0*"
_output_shapes
:
h
Reshape_6/shapeConst*%
valueB"˙˙˙˙         *
dtype0*
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
valueB Bsummary_pool_0*
dtype0*
_output_shapes
: 

summary_pool_0ImageSummarysummary_pool_0/tag	Reshape_6*

max_images*
T0*
	bad_colorB:˙  ˙*
_output_shapes
: 
f
Slice_3/beginConst*%
valueB"                *
dtype0*
_output_shapes
:
e
Slice_3/sizeConst*%
valueB"   ˙˙˙˙˙˙˙˙   *
dtype0*
_output_shapes
:
u
Slice_3SliceConv2D_1Slice_3/beginSlice_3/size*&
_output_shapes
:*
T0*
Index0
`
Const_9Const*%
valueB"             *
dtype0*
_output_shapes
:
\
Min_3MinSlice_3Const_9*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
M
sub_3SubSlice_3Min_3*
T0*&
_output_shapes
:
a
Const_10Const*%
valueB"             *
dtype0*
_output_shapes
:
[
Max_3Maxsub_3Const_10*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
L
mul_3/yConst*
valueB
 *  C*
dtype0*
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
Reshape_7/shapeConst*!
valueB"         *
dtype0*
_output_shapes
:
k
	Reshape_7Reshape	truediv_3Reshape_7/shape*
Tshape0*
T0*"
_output_shapes
:
e
transpose_3/permConst*!
valueB"          *
dtype0*
_output_shapes
:
o
transpose_3	Transpose	Reshape_7transpose_3/perm*
Tperm0*
T0*"
_output_shapes
:
h
Reshape_8/shapeConst*%
valueB"˙˙˙˙         *
dtype0*
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
valueB Bsummary_conv2_0*
dtype0*
_output_shapes
: 

summary_conv2_0ImageSummarysummary_conv2_0/tag	Reshape_8*

max_images*
T0*
	bad_colorB:˙  ˙*
_output_shapes
: 
f
Slice_4/beginConst*%
valueB"                *
dtype0*
_output_shapes
:
e
Slice_4/sizeConst*%
valueB"   ˙˙˙˙˙˙˙˙   *
dtype0*
_output_shapes
:
v
Slice_4Slice	MaxPool_1Slice_4/beginSlice_4/size*&
_output_shapes
:*
T0*
Index0
a
Const_11Const*%
valueB"             *
dtype0*
_output_shapes
:
]
Min_4MinSlice_4Const_11*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
M
sub_4SubSlice_4Min_4*
T0*&
_output_shapes
:
a
Const_12Const*%
valueB"             *
dtype0*
_output_shapes
:
[
Max_4Maxsub_4Const_12*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
L
mul_4/yConst*
valueB
 *  C*
dtype0*
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
Reshape_9/shapeConst*!
valueB"         *
dtype0*
_output_shapes
:
k
	Reshape_9Reshape	truediv_4Reshape_9/shape*
Tshape0*
T0*"
_output_shapes
:
e
transpose_4/permConst*!
valueB"          *
dtype0*
_output_shapes
:
o
transpose_4	Transpose	Reshape_9transpose_4/perm*
Tperm0*
T0*"
_output_shapes
:
i
Reshape_10/shapeConst*%
valueB"˙˙˙˙         *
dtype0*
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
valueB Bsummary_pool2_0*
dtype0*
_output_shapes
: 

summary_pool2_0ImageSummarysummary_pool2_0/tag
Reshape_10*

max_images*
T0*
	bad_colorB:˙  ˙*
_output_shapes
: 
F
RankConst*
value	B :*
dtype0*
_output_shapes
: 
V
ShapeConst*
valueB"      *
dtype0*
_output_shapes
:
H
Rank_1Const*
value	B :*
dtype0*
_output_shapes
: 
X
Shape_1Const*
valueB"      *
dtype0*
_output_shapes
:
G
Sub/yConst*
value	B :*
dtype0*
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
Slice_5/sizeConst*
valueB:*
dtype0*
_output_shapes
:
h
Slice_5SliceShape_1Slice_5/beginSlice_5/size*
_output_shapes
:*
T0*
Index0
b
concat/values_0Const*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
M
concat/axisConst*
value	B : *
dtype0*
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
Rank_2Const*
value	B :*
dtype0*
_output_shapes
: 
X
Shape_2Const*
valueB"      *
dtype0*
_output_shapes
:
I
Sub_1/yConst*
value	B :*
dtype0*
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
Slice_6/sizeConst*
valueB:*
dtype0*
_output_shapes
:
h
Slice_6SliceShape_2Slice_6/beginSlice_6/size*
_output_shapes
:*
T0*
Index0
d
concat_1/values_0Const*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
O
concat_1/axisConst*
value	B : *
dtype0*
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

SoftmaxCrossEntropyWithLogitsSoftmaxCrossEntropyWithLogits
Reshape_11
Reshape_12*
T0*$
_output_shapes
::
I
Sub_2/yConst*
value	B :*
dtype0*
_output_shapes
: 
<
Sub_2SubRankSub_2/y*
T0*
_output_shapes
: 
W
Slice_7/beginConst*
valueB: *
dtype0*
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
Slice_7SliceShapeSlice_7/beginSlice_7/size*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Index0
p

Reshape_13ReshapeSoftmaxCrossEntropyWithLogitsSlice_7*
Tshape0*
T0*
_output_shapes
:
R
Const_13Const*
valueB: *
dtype0*
_output_shapes
:
`
MeanMean
Reshape_13Const_13*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
N
	loss/tagsConst*
valueB
 Bloss*
dtype0*
_output_shapes
: 
G
lossScalarSummary	loss/tagsMean*
T0*
_output_shapes
: 
R
gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
T
gradients/ConstConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Y
gradients/FillFillgradients/Shapegradients/Const*
T0*
_output_shapes
: 
k
!gradients/Mean_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:

gradients/Mean_grad/ReshapeReshapegradients/Fill!gradients/Mean_grad/Reshape/shape*
Tshape0*
T0*
_output_shapes
:
l
"gradients/Mean_grad/Tile/multiplesConst*
valueB:*
dtype0*
_output_shapes
:

gradients/Mean_grad/TileTilegradients/Mean_grad/Reshape"gradients/Mean_grad/Tile/multiples*

Tmultiples0*
T0*
_output_shapes
:
c
gradients/Mean_grad/ShapeConst*
valueB:*
dtype0*
_output_shapes
:
^
gradients/Mean_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 

gradients/Mean_grad/ConstConst*
valueB: *
dtype0*,
_class"
 loc:@gradients/Mean_grad/Shape*
_output_shapes
:
Â
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shapegradients/Mean_grad/Const*
	keep_dims( *,
_class"
 loc:@gradients/Mean_grad/Shape*

Tidx0*
T0*
_output_shapes
: 

gradients/Mean_grad/Const_1Const*
valueB: *
dtype0*,
_class"
 loc:@gradients/Mean_grad/Shape*
_output_shapes
:
Č
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_1gradients/Mean_grad/Const_1*
	keep_dims( *,
_class"
 loc:@gradients/Mean_grad/Shape*

Tidx0*
T0*
_output_shapes
: 

gradients/Mean_grad/Maximum/yConst*
value	B :*
dtype0*,
_class"
 loc:@gradients/Mean_grad/Shape*
_output_shapes
: 
°
gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*,
_class"
 loc:@gradients/Mean_grad/Shape*
T0*
_output_shapes
: 
Ž
gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum*,
_class"
 loc:@gradients/Mean_grad/Shape*
T0*
_output_shapes
: 
n
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv*

DstT0*

SrcT0*
_output_shapes
: 

gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*
T0*
_output_shapes
:
i
gradients/Reshape_13_grad/ShapeConst*
valueB:*
dtype0*
_output_shapes
:

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

;gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dimConst*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
: 
Ú
7gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims
ExpandDims!gradients/Reshape_13_grad/Reshape;gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim*

Tdim0*
T0*
_output_shapes

:
ş
0gradients/SoftmaxCrossEntropyWithLogits_grad/mulMul7gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDimsSoftmaxCrossEntropyWithLogits:1*
T0*
_output_shapes

:
p
gradients/Reshape_11_grad/ShapeConst*
valueB"      *
dtype0*
_output_shapes
:
ś
!gradients/Reshape_11_grad/ReshapeReshape0gradients/SoftmaxCrossEntropyWithLogits_grad/mulgradients/Reshape_11_grad/Shape*
Tshape0*
T0*
_output_shapes

:
k
gradients/add_1_grad/ShapeConst*
valueB"      *
dtype0*
_output_shapes
:
f
gradients/add_1_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
ş
*gradients/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_1_grad/Shapegradients/add_1_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ž
gradients/add_1_grad/SumSum!gradients/Reshape_11_grad/Reshape*gradients/add_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

gradients/add_1_grad/ReshapeReshapegradients/add_1_grad/Sumgradients/add_1_grad/Shape*
Tshape0*
T0*
_output_shapes

:
˛
gradients/add_1_grad/Sum_1Sum!gradients/Reshape_11_grad/Reshape,gradients/add_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

gradients/add_1_grad/Reshape_1Reshapegradients/add_1_grad/Sum_1gradients/add_1_grad/Shape_1*
Tshape0*
T0*
_output_shapes
:
§
gradients/MatMul_1_grad/MatMulMatMulgradients/add_1_grad/ReshapeVariable_6/read*
transpose_b(*
transpose_a( *
T0*
_output_shapes
:	
 
 gradients/MatMul_1_grad/MatMul_1MatMulRelu_2gradients/add_1_grad/Reshape*
transpose_b( *
transpose_a(*
T0*
_output_shapes
:	
|
gradients/Relu_2_grad/ReluGradReluGradgradients/MatMul_1_grad/MatMulRelu_2*
T0*
_output_shapes
:	
i
gradients/add_grad/ShapeConst*
valueB"      *
dtype0*
_output_shapes
:
e
gradients/add_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
´
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
§
gradients/add_grad/SumSumgradients/Relu_2_grad/ReluGrad(gradients/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*
Tshape0*
T0*
_output_shapes
:	
Ť
gradients/add_grad/Sum_1Sumgradients/Relu_2_grad/ReluGrad*gradients/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
Tshape0*
T0*
_output_shapes	
:
Ł
gradients/MatMul_grad/MatMulMatMulgradients/add_grad/ReshapeVariable_4/read*
transpose_b(*
transpose_a( *
T0*
_output_shapes
:	

gradients/MatMul_grad/MatMul_1MatMulReshapegradients/add_grad/Reshape*
transpose_b( *
transpose_a(*
T0* 
_output_shapes
:

u
gradients/Reshape_grad/ShapeConst*%
valueB"         @   *
dtype0*
_output_shapes
:
¤
gradients/Reshape_grad/ReshapeReshapegradients/MatMul_grad/MatMulgradients/Reshape_grad/Shape*
Tshape0*
T0*&
_output_shapes
:@
é
$gradients/MaxPool_1_grad/MaxPoolGradMaxPoolGradRelu_1	MaxPool_1gradients/Reshape_grad/Reshape*
ksize
*
data_formatNHWC*
T0*&
_output_shapes
:@*
strides
*
paddingSAME

gradients/Relu_1_grad/ReluGradReluGrad$gradients/MaxPool_1_grad/MaxPoolGradRelu_1*
T0*&
_output_shapes
:@

$gradients/BiasAdd_1_grad/BiasAddGradBiasAddGradgradients/Relu_1_grad/ReluGrad*
_output_shapes
:@*
T0*
data_formatNHWC

gradients/Conv2D_1_grad/ShapeNShapeNMaxPoolVariable_2/read*
N*
out_type0*
T0* 
_output_shapes
::

+gradients/Conv2D_1_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_1_grad/ShapeNVariable_2/readgradients/Relu_1_grad/ReluGrad*
paddingSAME*
data_formatNHWC*
T0*&
_output_shapes
: *
strides
*
use_cudnn_on_gpu(

,gradients/Conv2D_1_grad/Conv2DBackpropFilterConv2DBackpropFilterMaxPool gradients/Conv2D_1_grad/ShapeN:1gradients/Relu_1_grad/ReluGrad*
paddingSAME*
data_formatNHWC*
T0*&
_output_shapes
: @*
strides
*
use_cudnn_on_gpu(
đ
"gradients/MaxPool_grad/MaxPoolGradMaxPoolGradReluMaxPool+gradients/Conv2D_1_grad/Conv2DBackpropInput*
ksize
*
data_formatNHWC*
T0*&
_output_shapes
: *
strides
*
paddingSAME

gradients/Relu_grad/ReluGradReluGrad"gradients/MaxPool_grad/MaxPoolGradRelu*
T0*&
_output_shapes
: 

"gradients/BiasAdd_grad/BiasAddGradBiasAddGradgradients/Relu_grad/ReluGrad*
_output_shapes
: *
T0*
data_formatNHWC

gradients/Conv2D_grad/ShapeNShapeNPlaceholderVariable/read*
N*
out_type0*
T0* 
_output_shapes
::

)gradients/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_grad/ShapeNVariable/readgradients/Relu_grad/ReluGrad*
paddingSAME*
data_formatNHWC*
T0*&
_output_shapes
:*
strides
*
use_cudnn_on_gpu(

*gradients/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterPlaceholdergradients/Conv2D_grad/ShapeN:1gradients/Relu_grad/ReluGrad*
paddingSAME*
data_formatNHWC*
T0*&
_output_shapes
: *
strides
*
use_cudnn_on_gpu(
¨
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
global_norm/ConstConst*
valueB: *
dtype0*
_output_shapes
:
z
global_norm/SumSumglobal_norm/stackglobal_norm/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
X
global_norm/Const_1Const*
valueB
 *   @*
dtype0*
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
valueB Bconv1_weights*
dtype0*
_output_shapes
: 
l
conv1_weightsScalarSummaryconv1_weights/tagsglobal_norm/global_norm*
T0*
_output_shapes
: 

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
global_norm_1/ConstConst*
valueB: *
dtype0*
_output_shapes
:

global_norm_1/SumSumglobal_norm_1/stackglobal_norm_1/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
Z
global_norm_1/Const_1Const*
valueB
 *   @*
dtype0*
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
valueB Bconv1_biases*
dtype0*
_output_shapes
: 
l
conv1_biasesScalarSummaryconv1_biases/tagsglobal_norm_1/global_norm*
T0*
_output_shapes
: 
Ž
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
global_norm_2/ConstConst*
valueB: *
dtype0*
_output_shapes
:

global_norm_2/SumSumglobal_norm_2/stackglobal_norm_2/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
Z
global_norm_2/Const_1Const*
valueB
 *   @*
dtype0*
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
valueB Bconv2_weights*
dtype0*
_output_shapes
: 
n
conv2_weightsScalarSummaryconv2_weights/tagsglobal_norm_2/global_norm*
T0*
_output_shapes
: 

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
global_norm_3/ConstConst*
valueB: *
dtype0*
_output_shapes
:

global_norm_3/SumSumglobal_norm_3/stackglobal_norm_3/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
Z
global_norm_3/Const_1Const*
valueB
 *   @*
dtype0*
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
valueB Bconv2_biases*
dtype0*
_output_shapes
: 
l
conv2_biasesScalarSummaryconv2_biases/tagsglobal_norm_3/global_norm*
T0*
_output_shapes
: 

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
global_norm_4/ConstConst*
valueB: *
dtype0*
_output_shapes
:

global_norm_4/SumSumglobal_norm_4/stackglobal_norm_4/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
Z
global_norm_4/Const_1Const*
valueB
 *   @*
dtype0*
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
valueB Bfc1_weights*
dtype0*
_output_shapes
: 
j
fc1_weightsScalarSummaryfc1_weights/tagsglobal_norm_4/global_norm*
T0*
_output_shapes
: 

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
global_norm_5/ConstConst*
valueB: *
dtype0*
_output_shapes
:

global_norm_5/SumSumglobal_norm_5/stackglobal_norm_5/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
Z
global_norm_5/Const_1Const*
valueB
 *   @*
dtype0*
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
fc1_biases/tagsConst*
valueB B
fc1_biases*
dtype0*
_output_shapes
: 
h

fc1_biasesScalarSummaryfc1_biases/tagsglobal_norm_5/global_norm*
T0*
_output_shapes
: 

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
global_norm_6/ConstConst*
valueB: *
dtype0*
_output_shapes
:

global_norm_6/SumSumglobal_norm_6/stackglobal_norm_6/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
Z
global_norm_6/Const_1Const*
valueB
 *   @*
dtype0*
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
valueB Bfc2_weights*
dtype0*
_output_shapes
: 
j
fc2_weightsScalarSummaryfc2_weights/tagsglobal_norm_6/global_norm*
T0*
_output_shapes
: 

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
global_norm_7/ConstConst*
valueB: *
dtype0*
_output_shapes
:

global_norm_7/SumSumglobal_norm_7/stackglobal_norm_7/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
Z
global_norm_7/Const_1Const*
valueB
 *   @*
dtype0*
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
fc2_biases/tagsConst*
valueB B
fc2_biases*
dtype0*
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
mul_5/xConst*
valueB
 *o:*
dtype0*
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
Variable_8/initial_valueConst*
value	B : *
dtype0*
_output_shapes
: 
n

Variable_8
VariableV2*
dtype0*
shape: *
_output_shapes
: *
shared_name *
	container 
Ş
Variable_8/AssignAssign
Variable_8Variable_8/initial_value*
use_locking(*
_class
loc:@Variable_8*
T0*
validate_shape(*
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
mul_6/yConst*
value	B :*
dtype0*
_output_shapes
: 
G
mul_6MulVariable_8/readmul_6/y*
T0*
_output_shapes
: 
c
ExponentialDecay/learning_rateConst*
valueB
 *
×#<*
dtype0*
_output_shapes
: 
T
ExponentialDecay/CastCastmul_6*

DstT0*

SrcT0*
_output_shapes
: 
]
ExponentialDecay/Cast_1/xConst*
valueB	 :Ŕ=*
dtype0*
_output_shapes
: 
j
ExponentialDecay/Cast_1CastExponentialDecay/Cast_1/x*

DstT0*

SrcT0*
_output_shapes
: 
^
ExponentialDecay/Cast_2/xConst*
valueB
 *33s?*
dtype0*
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
valueB Blearning_rate*
dtype0*
_output_shapes
: 
e
learning_rateScalarSummarylearning_rate/tagsExponentialDecay*
T0*
_output_shapes
: 
T
gradients_1/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
V
gradients_1/ConstConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
_
gradients_1/FillFillgradients_1/Shapegradients_1/Const*
T0*
_output_shapes
: 
_
gradients_1/add_5_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
a
gradients_1/add_5_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
Ŕ
,gradients_1/add_5_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/add_5_grad/Shapegradients_1/add_5_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ą
gradients_1/add_5_grad/SumSumgradients_1/Fill,gradients_1/add_5_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

gradients_1/add_5_grad/ReshapeReshapegradients_1/add_5_grad/Sumgradients_1/add_5_grad/Shape*
Tshape0*
T0*
_output_shapes
: 
Ľ
gradients_1/add_5_grad/Sum_1Sumgradients_1/Fill.gradients_1/add_5_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

 gradients_1/add_5_grad/Reshape_1Reshapegradients_1/add_5_grad/Sum_1gradients_1/add_5_grad/Shape_1*
Tshape0*
T0*
_output_shapes
: 
s
'gradients_1/add_5_grad/tuple/group_depsNoOp^gradients_1/add_5_grad/Reshape!^gradients_1/add_5_grad/Reshape_1
Ů
/gradients_1/add_5_grad/tuple/control_dependencyIdentitygradients_1/add_5_grad/Reshape(^gradients_1/add_5_grad/tuple/group_deps*1
_class'
%#loc:@gradients_1/add_5_grad/Reshape*
T0*
_output_shapes
: 
ß
1gradients_1/add_5_grad/tuple/control_dependency_1Identity gradients_1/add_5_grad/Reshape_1(^gradients_1/add_5_grad/tuple/group_deps*3
_class)
'%loc:@gradients_1/add_5_grad/Reshape_1*
T0*
_output_shapes
: 
m
#gradients_1/Mean_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:
ą
gradients_1/Mean_grad/ReshapeReshape/gradients_1/add_5_grad/tuple/control_dependency#gradients_1/Mean_grad/Reshape/shape*
Tshape0*
T0*
_output_shapes
:
n
$gradients_1/Mean_grad/Tile/multiplesConst*
valueB:*
dtype0*
_output_shapes
:

gradients_1/Mean_grad/TileTilegradients_1/Mean_grad/Reshape$gradients_1/Mean_grad/Tile/multiples*

Tmultiples0*
T0*
_output_shapes
:
e
gradients_1/Mean_grad/ShapeConst*
valueB:*
dtype0*
_output_shapes
:
`
gradients_1/Mean_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 

gradients_1/Mean_grad/ConstConst*
valueB: *
dtype0*.
_class$
" loc:@gradients_1/Mean_grad/Shape*
_output_shapes
:
Ę
gradients_1/Mean_grad/ProdProdgradients_1/Mean_grad/Shapegradients_1/Mean_grad/Const*
	keep_dims( *.
_class$
" loc:@gradients_1/Mean_grad/Shape*

Tidx0*
T0*
_output_shapes
: 

gradients_1/Mean_grad/Const_1Const*
valueB: *
dtype0*.
_class$
" loc:@gradients_1/Mean_grad/Shape*
_output_shapes
:
Đ
gradients_1/Mean_grad/Prod_1Prodgradients_1/Mean_grad/Shape_1gradients_1/Mean_grad/Const_1*
	keep_dims( *.
_class$
" loc:@gradients_1/Mean_grad/Shape*

Tidx0*
T0*
_output_shapes
: 

gradients_1/Mean_grad/Maximum/yConst*
value	B :*
dtype0*.
_class$
" loc:@gradients_1/Mean_grad/Shape*
_output_shapes
: 
¸
gradients_1/Mean_grad/MaximumMaximumgradients_1/Mean_grad/Prod_1gradients_1/Mean_grad/Maximum/y*.
_class$
" loc:@gradients_1/Mean_grad/Shape*
T0*
_output_shapes
: 
ś
gradients_1/Mean_grad/floordivFloorDivgradients_1/Mean_grad/Prodgradients_1/Mean_grad/Maximum*.
_class$
" loc:@gradients_1/Mean_grad/Shape*
T0*
_output_shapes
: 
r
gradients_1/Mean_grad/CastCastgradients_1/Mean_grad/floordiv*

DstT0*

SrcT0*
_output_shapes
: 

gradients_1/Mean_grad/truedivRealDivgradients_1/Mean_grad/Tilegradients_1/Mean_grad/Cast*
T0*
_output_shapes
:
_
gradients_1/mul_5_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
a
gradients_1/mul_5_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
Ŕ
,gradients_1/mul_5_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/mul_5_grad/Shapegradients_1/mul_5_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
|
gradients_1/mul_5_grad/mulMul1gradients_1/add_5_grad/tuple/control_dependency_1add_4*
T0*
_output_shapes
: 
Ť
gradients_1/mul_5_grad/SumSumgradients_1/mul_5_grad/mul,gradients_1/mul_5_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

gradients_1/mul_5_grad/ReshapeReshapegradients_1/mul_5_grad/Sumgradients_1/mul_5_grad/Shape*
Tshape0*
T0*
_output_shapes
: 

gradients_1/mul_5_grad/mul_1Mulmul_5/x1gradients_1/add_5_grad/tuple/control_dependency_1*
T0*
_output_shapes
: 
ą
gradients_1/mul_5_grad/Sum_1Sumgradients_1/mul_5_grad/mul_1.gradients_1/mul_5_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

 gradients_1/mul_5_grad/Reshape_1Reshapegradients_1/mul_5_grad/Sum_1gradients_1/mul_5_grad/Shape_1*
Tshape0*
T0*
_output_shapes
: 
s
'gradients_1/mul_5_grad/tuple/group_depsNoOp^gradients_1/mul_5_grad/Reshape!^gradients_1/mul_5_grad/Reshape_1
Ů
/gradients_1/mul_5_grad/tuple/control_dependencyIdentitygradients_1/mul_5_grad/Reshape(^gradients_1/mul_5_grad/tuple/group_deps*1
_class'
%#loc:@gradients_1/mul_5_grad/Reshape*
T0*
_output_shapes
: 
ß
1gradients_1/mul_5_grad/tuple/control_dependency_1Identity gradients_1/mul_5_grad/Reshape_1(^gradients_1/mul_5_grad/tuple/group_deps*3
_class)
'%loc:@gradients_1/mul_5_grad/Reshape_1*
T0*
_output_shapes
: 
k
!gradients_1/Reshape_13_grad/ShapeConst*
valueB:*
dtype0*
_output_shapes
:
Ł
#gradients_1/Reshape_13_grad/ReshapeReshapegradients_1/Mean_grad/truediv!gradients_1/Reshape_13_grad/Shape*
Tshape0*
T0*
_output_shapes
:
_
gradients_1/add_4_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
a
gradients_1/add_4_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
Ŕ
,gradients_1/add_4_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/add_4_grad/Shapegradients_1/add_4_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Â
gradients_1/add_4_grad/SumSum1gradients_1/mul_5_grad/tuple/control_dependency_1,gradients_1/add_4_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

gradients_1/add_4_grad/ReshapeReshapegradients_1/add_4_grad/Sumgradients_1/add_4_grad/Shape*
Tshape0*
T0*
_output_shapes
: 
Ć
gradients_1/add_4_grad/Sum_1Sum1gradients_1/mul_5_grad/tuple/control_dependency_1.gradients_1/add_4_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

 gradients_1/add_4_grad/Reshape_1Reshapegradients_1/add_4_grad/Sum_1gradients_1/add_4_grad/Shape_1*
Tshape0*
T0*
_output_shapes
: 
s
'gradients_1/add_4_grad/tuple/group_depsNoOp^gradients_1/add_4_grad/Reshape!^gradients_1/add_4_grad/Reshape_1
Ů
/gradients_1/add_4_grad/tuple/control_dependencyIdentitygradients_1/add_4_grad/Reshape(^gradients_1/add_4_grad/tuple/group_deps*1
_class'
%#loc:@gradients_1/add_4_grad/Reshape*
T0*
_output_shapes
: 
ß
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

=gradients_1/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dimConst*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
: 
ŕ
9gradients_1/SoftmaxCrossEntropyWithLogits_grad/ExpandDims
ExpandDims#gradients_1/Reshape_13_grad/Reshape=gradients_1/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim*

Tdim0*
T0*
_output_shapes

:
ž
2gradients_1/SoftmaxCrossEntropyWithLogits_grad/mulMul9gradients_1/SoftmaxCrossEntropyWithLogits_grad/ExpandDimsSoftmaxCrossEntropyWithLogits:1*
T0*
_output_shapes

:
_
gradients_1/add_3_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
a
gradients_1/add_3_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
Ŕ
,gradients_1/add_3_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/add_3_grad/Shapegradients_1/add_3_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ŕ
gradients_1/add_3_grad/SumSum/gradients_1/add_4_grad/tuple/control_dependency,gradients_1/add_3_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

gradients_1/add_3_grad/ReshapeReshapegradients_1/add_3_grad/Sumgradients_1/add_3_grad/Shape*
Tshape0*
T0*
_output_shapes
: 
Ä
gradients_1/add_3_grad/Sum_1Sum/gradients_1/add_4_grad/tuple/control_dependency.gradients_1/add_3_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

 gradients_1/add_3_grad/Reshape_1Reshapegradients_1/add_3_grad/Sum_1gradients_1/add_3_grad/Shape_1*
Tshape0*
T0*
_output_shapes
: 
s
'gradients_1/add_3_grad/tuple/group_depsNoOp^gradients_1/add_3_grad/Reshape!^gradients_1/add_3_grad/Reshape_1
Ů
/gradients_1/add_3_grad/tuple/control_dependencyIdentitygradients_1/add_3_grad/Reshape(^gradients_1/add_3_grad/tuple/group_deps*1
_class'
%#loc:@gradients_1/add_3_grad/Reshape*
T0*
_output_shapes
: 
ß
1gradients_1/add_3_grad/tuple/control_dependency_1Identity gradients_1/add_3_grad/Reshape_1(^gradients_1/add_3_grad/tuple/group_deps*3
_class)
'%loc:@gradients_1/add_3_grad/Reshape_1*
T0*
_output_shapes
: 

gradients_1/L2Loss_3_grad/mulMulVariable_7/read1gradients_1/add_4_grad/tuple/control_dependency_1*
T0*
_output_shapes
:
r
!gradients_1/Reshape_11_grad/ShapeConst*
valueB"      *
dtype0*
_output_shapes
:
ź
#gradients_1/Reshape_11_grad/ReshapeReshape2gradients_1/SoftmaxCrossEntropyWithLogits_grad/mul!gradients_1/Reshape_11_grad/Shape*
Tshape0*
T0*
_output_shapes

:
_
gradients_1/add_2_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
a
gradients_1/add_2_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
Ŕ
,gradients_1/add_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/add_2_grad/Shapegradients_1/add_2_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ŕ
gradients_1/add_2_grad/SumSum/gradients_1/add_3_grad/tuple/control_dependency,gradients_1/add_2_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

gradients_1/add_2_grad/ReshapeReshapegradients_1/add_2_grad/Sumgradients_1/add_2_grad/Shape*
Tshape0*
T0*
_output_shapes
: 
Ä
gradients_1/add_2_grad/Sum_1Sum/gradients_1/add_3_grad/tuple/control_dependency.gradients_1/add_2_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

 gradients_1/add_2_grad/Reshape_1Reshapegradients_1/add_2_grad/Sum_1gradients_1/add_2_grad/Shape_1*
Tshape0*
T0*
_output_shapes
: 
s
'gradients_1/add_2_grad/tuple/group_depsNoOp^gradients_1/add_2_grad/Reshape!^gradients_1/add_2_grad/Reshape_1
Ů
/gradients_1/add_2_grad/tuple/control_dependencyIdentitygradients_1/add_2_grad/Reshape(^gradients_1/add_2_grad/tuple/group_deps*1
_class'
%#loc:@gradients_1/add_2_grad/Reshape*
T0*
_output_shapes
: 
ß
1gradients_1/add_2_grad/tuple/control_dependency_1Identity gradients_1/add_2_grad/Reshape_1(^gradients_1/add_2_grad/tuple/group_deps*3
_class)
'%loc:@gradients_1/add_2_grad/Reshape_1*
T0*
_output_shapes
: 

gradients_1/L2Loss_2_grad/mulMulVariable_6/read1gradients_1/add_3_grad/tuple/control_dependency_1*
T0*
_output_shapes
:	
m
gradients_1/add_1_grad/ShapeConst*
valueB"      *
dtype0*
_output_shapes
:
h
gradients_1/add_1_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
Ŕ
,gradients_1/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/add_1_grad/Shapegradients_1/add_1_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
´
gradients_1/add_1_grad/SumSum#gradients_1/Reshape_11_grad/Reshape,gradients_1/add_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

gradients_1/add_1_grad/ReshapeReshapegradients_1/add_1_grad/Sumgradients_1/add_1_grad/Shape*
Tshape0*
T0*
_output_shapes

:
¸
gradients_1/add_1_grad/Sum_1Sum#gradients_1/Reshape_11_grad/Reshape.gradients_1/add_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

 gradients_1/add_1_grad/Reshape_1Reshapegradients_1/add_1_grad/Sum_1gradients_1/add_1_grad/Shape_1*
Tshape0*
T0*
_output_shapes
:
s
'gradients_1/add_1_grad/tuple/group_depsNoOp^gradients_1/add_1_grad/Reshape!^gradients_1/add_1_grad/Reshape_1
á
/gradients_1/add_1_grad/tuple/control_dependencyIdentitygradients_1/add_1_grad/Reshape(^gradients_1/add_1_grad/tuple/group_deps*1
_class'
%#loc:@gradients_1/add_1_grad/Reshape*
T0*
_output_shapes

:
ă
1gradients_1/add_1_grad/tuple/control_dependency_1Identity gradients_1/add_1_grad/Reshape_1(^gradients_1/add_1_grad/tuple/group_deps*3
_class)
'%loc:@gradients_1/add_1_grad/Reshape_1*
T0*
_output_shapes
:

gradients_1/L2Loss_grad/mulMulVariable_4/read/gradients_1/add_2_grad/tuple/control_dependency*
T0* 
_output_shapes
:


gradients_1/L2Loss_1_grad/mulMulVariable_5/read1gradients_1/add_2_grad/tuple/control_dependency_1*
T0*
_output_shapes	
:
ź
 gradients_1/MatMul_1_grad/MatMulMatMul/gradients_1/add_1_grad/tuple/control_dependencyVariable_6/read*
transpose_b(*
transpose_a( *
T0*
_output_shapes
:	
ľ
"gradients_1/MatMul_1_grad/MatMul_1MatMulRelu_2/gradients_1/add_1_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
T0*
_output_shapes
:	
z
*gradients_1/MatMul_1_grad/tuple/group_depsNoOp!^gradients_1/MatMul_1_grad/MatMul#^gradients_1/MatMul_1_grad/MatMul_1
ě
2gradients_1/MatMul_1_grad/tuple/control_dependencyIdentity gradients_1/MatMul_1_grad/MatMul+^gradients_1/MatMul_1_grad/tuple/group_deps*3
_class)
'%loc:@gradients_1/MatMul_1_grad/MatMul*
T0*
_output_shapes
:	
ň
4gradients_1/MatMul_1_grad/tuple/control_dependency_1Identity"gradients_1/MatMul_1_grad/MatMul_1+^gradients_1/MatMul_1_grad/tuple/group_deps*5
_class+
)'loc:@gradients_1/MatMul_1_grad/MatMul_1*
T0*
_output_shapes
:	
Ę
gradients_1/AddNAddNgradients_1/L2Loss_3_grad/mul1gradients_1/add_1_grad/tuple/control_dependency_1*0
_class&
$"loc:@gradients_1/L2Loss_3_grad/mul*
N*
T0*
_output_shapes
:

 gradients_1/Relu_2_grad/ReluGradReluGrad2gradients_1/MatMul_1_grad/tuple/control_dependencyRelu_2*
T0*
_output_shapes
:	
Ô
gradients_1/AddN_1AddNgradients_1/L2Loss_2_grad/mul4gradients_1/MatMul_1_grad/tuple/control_dependency_1*0
_class&
$"loc:@gradients_1/L2Loss_2_grad/mul*
N*
T0*
_output_shapes
:	
k
gradients_1/add_grad/ShapeConst*
valueB"      *
dtype0*
_output_shapes
:
g
gradients_1/add_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
ş
*gradients_1/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/add_grad/Shapegradients_1/add_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
­
gradients_1/add_grad/SumSum gradients_1/Relu_2_grad/ReluGrad*gradients_1/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

gradients_1/add_grad/ReshapeReshapegradients_1/add_grad/Sumgradients_1/add_grad/Shape*
Tshape0*
T0*
_output_shapes
:	
ą
gradients_1/add_grad/Sum_1Sum gradients_1/Relu_2_grad/ReluGrad,gradients_1/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

gradients_1/add_grad/Reshape_1Reshapegradients_1/add_grad/Sum_1gradients_1/add_grad/Shape_1*
Tshape0*
T0*
_output_shapes	
:
m
%gradients_1/add_grad/tuple/group_depsNoOp^gradients_1/add_grad/Reshape^gradients_1/add_grad/Reshape_1
Ú
-gradients_1/add_grad/tuple/control_dependencyIdentitygradients_1/add_grad/Reshape&^gradients_1/add_grad/tuple/group_deps*/
_class%
#!loc:@gradients_1/add_grad/Reshape*
T0*
_output_shapes
:	
Ü
/gradients_1/add_grad/tuple/control_dependency_1Identitygradients_1/add_grad/Reshape_1&^gradients_1/add_grad/tuple/group_deps*1
_class'
%#loc:@gradients_1/add_grad/Reshape_1*
T0*
_output_shapes	
:
¸
gradients_1/MatMul_grad/MatMulMatMul-gradients_1/add_grad/tuple/control_dependencyVariable_4/read*
transpose_b(*
transpose_a( *
T0*
_output_shapes
:	
ł
 gradients_1/MatMul_grad/MatMul_1MatMulReshape-gradients_1/add_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
T0* 
_output_shapes
:

t
(gradients_1/MatMul_grad/tuple/group_depsNoOp^gradients_1/MatMul_grad/MatMul!^gradients_1/MatMul_grad/MatMul_1
ä
0gradients_1/MatMul_grad/tuple/control_dependencyIdentitygradients_1/MatMul_grad/MatMul)^gradients_1/MatMul_grad/tuple/group_deps*1
_class'
%#loc:@gradients_1/MatMul_grad/MatMul*
T0*
_output_shapes
:	
ë
2gradients_1/MatMul_grad/tuple/control_dependency_1Identity gradients_1/MatMul_grad/MatMul_1)^gradients_1/MatMul_grad/tuple/group_deps*3
_class)
'%loc:@gradients_1/MatMul_grad/MatMul_1*
T0* 
_output_shapes
:

Ë
gradients_1/AddN_2AddNgradients_1/L2Loss_1_grad/mul/gradients_1/add_grad/tuple/control_dependency_1*0
_class&
$"loc:@gradients_1/L2Loss_1_grad/mul*
N*
T0*
_output_shapes	
:
w
gradients_1/Reshape_grad/ShapeConst*%
valueB"         @   *
dtype0*
_output_shapes
:
ź
 gradients_1/Reshape_grad/ReshapeReshape0gradients_1/MatMul_grad/tuple/control_dependencygradients_1/Reshape_grad/Shape*
Tshape0*
T0*&
_output_shapes
:@
Ď
gradients_1/AddN_3AddNgradients_1/L2Loss_grad/mul2gradients_1/MatMul_grad/tuple/control_dependency_1*.
_class$
" loc:@gradients_1/L2Loss_grad/mul*
N*
T0* 
_output_shapes
:

í
&gradients_1/MaxPool_1_grad/MaxPoolGradMaxPoolGradRelu_1	MaxPool_1 gradients_1/Reshape_grad/Reshape*
ksize
*
data_formatNHWC*
T0*&
_output_shapes
:@*
strides
*
paddingSAME

 gradients_1/Relu_1_grad/ReluGradReluGrad&gradients_1/MaxPool_1_grad/MaxPoolGradRelu_1*
T0*&
_output_shapes
:@

&gradients_1/BiasAdd_1_grad/BiasAddGradBiasAddGrad gradients_1/Relu_1_grad/ReluGrad*
_output_shapes
:@*
T0*
data_formatNHWC

+gradients_1/BiasAdd_1_grad/tuple/group_depsNoOp!^gradients_1/Relu_1_grad/ReluGrad'^gradients_1/BiasAdd_1_grad/BiasAddGrad
ő
3gradients_1/BiasAdd_1_grad/tuple/control_dependencyIdentity gradients_1/Relu_1_grad/ReluGrad,^gradients_1/BiasAdd_1_grad/tuple/group_deps*3
_class)
'%loc:@gradients_1/Relu_1_grad/ReluGrad*
T0*&
_output_shapes
:@
÷
5gradients_1/BiasAdd_1_grad/tuple/control_dependency_1Identity&gradients_1/BiasAdd_1_grad/BiasAddGrad,^gradients_1/BiasAdd_1_grad/tuple/group_deps*9
_class/
-+loc:@gradients_1/BiasAdd_1_grad/BiasAddGrad*
T0*
_output_shapes
:@

 gradients_1/Conv2D_1_grad/ShapeNShapeNMaxPoolVariable_2/read*
N*
out_type0*
T0* 
_output_shapes
::
Ř
-gradients_1/Conv2D_1_grad/Conv2DBackpropInputConv2DBackpropInput gradients_1/Conv2D_1_grad/ShapeNVariable_2/read3gradients_1/BiasAdd_1_grad/tuple/control_dependency*
paddingSAME*
data_formatNHWC*
T0*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
strides
*
use_cudnn_on_gpu(
Ô
.gradients_1/Conv2D_1_grad/Conv2DBackpropFilterConv2DBackpropFilterMaxPool"gradients_1/Conv2D_1_grad/ShapeN:13gradients_1/BiasAdd_1_grad/tuple/control_dependency*
paddingSAME*
data_formatNHWC*
T0*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
strides
*
use_cudnn_on_gpu(

*gradients_1/Conv2D_1_grad/tuple/group_depsNoOp.^gradients_1/Conv2D_1_grad/Conv2DBackpropInput/^gradients_1/Conv2D_1_grad/Conv2DBackpropFilter

2gradients_1/Conv2D_1_grad/tuple/control_dependencyIdentity-gradients_1/Conv2D_1_grad/Conv2DBackpropInput+^gradients_1/Conv2D_1_grad/tuple/group_deps*@
_class6
42loc:@gradients_1/Conv2D_1_grad/Conv2DBackpropInput*
T0*&
_output_shapes
: 

4gradients_1/Conv2D_1_grad/tuple/control_dependency_1Identity.gradients_1/Conv2D_1_grad/Conv2DBackpropFilter+^gradients_1/Conv2D_1_grad/tuple/group_deps*A
_class7
53loc:@gradients_1/Conv2D_1_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
: @
ů
$gradients_1/MaxPool_grad/MaxPoolGradMaxPoolGradReluMaxPool2gradients_1/Conv2D_1_grad/tuple/control_dependency*
ksize
*
data_formatNHWC*
T0*&
_output_shapes
: *
strides
*
paddingSAME

gradients_1/Relu_grad/ReluGradReluGrad$gradients_1/MaxPool_grad/MaxPoolGradRelu*
T0*&
_output_shapes
: 

$gradients_1/BiasAdd_grad/BiasAddGradBiasAddGradgradients_1/Relu_grad/ReluGrad*
_output_shapes
: *
T0*
data_formatNHWC
y
)gradients_1/BiasAdd_grad/tuple/group_depsNoOp^gradients_1/Relu_grad/ReluGrad%^gradients_1/BiasAdd_grad/BiasAddGrad
í
1gradients_1/BiasAdd_grad/tuple/control_dependencyIdentitygradients_1/Relu_grad/ReluGrad*^gradients_1/BiasAdd_grad/tuple/group_deps*1
_class'
%#loc:@gradients_1/Relu_grad/ReluGrad*
T0*&
_output_shapes
: 
ď
3gradients_1/BiasAdd_grad/tuple/control_dependency_1Identity$gradients_1/BiasAdd_grad/BiasAddGrad*^gradients_1/BiasAdd_grad/tuple/group_deps*7
_class-
+)loc:@gradients_1/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
: 

gradients_1/Conv2D_grad/ShapeNShapeNPlaceholderVariable/read*
N*
out_type0*
T0* 
_output_shapes
::
Đ
+gradients_1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInputgradients_1/Conv2D_grad/ShapeNVariable/read1gradients_1/BiasAdd_grad/tuple/control_dependency*
paddingSAME*
data_formatNHWC*
T0*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
strides
*
use_cudnn_on_gpu(
Ň
,gradients_1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterPlaceholder gradients_1/Conv2D_grad/ShapeN:11gradients_1/BiasAdd_grad/tuple/control_dependency*
paddingSAME*
data_formatNHWC*
T0*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
strides
*
use_cudnn_on_gpu(

(gradients_1/Conv2D_grad/tuple/group_depsNoOp,^gradients_1/Conv2D_grad/Conv2DBackpropInput-^gradients_1/Conv2D_grad/Conv2DBackpropFilter

0gradients_1/Conv2D_grad/tuple/control_dependencyIdentity+gradients_1/Conv2D_grad/Conv2DBackpropInput)^gradients_1/Conv2D_grad/tuple/group_deps*>
_class4
20loc:@gradients_1/Conv2D_grad/Conv2DBackpropInput*
T0*&
_output_shapes
:

2gradients_1/Conv2D_grad/tuple/control_dependency_1Identity,gradients_1/Conv2D_grad/Conv2DBackpropFilter)^gradients_1/Conv2D_grad/tuple/group_deps*?
_class5
31loc:@gradients_1/Conv2D_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
: 
Ľ
#Variable/Momentum/Initializer/zerosConst*%
valueB *    *
dtype0*
_class
loc:@Variable*&
_output_shapes
: 
˛
Variable/Momentum
VariableV2*
_class
loc:@Variable*
shared_name *
	container *
dtype0*
shape: *&
_output_shapes
: 
Ń
Variable/Momentum/AssignAssignVariable/Momentum#Variable/Momentum/Initializer/zeros*
use_locking(*
_class
loc:@Variable*
T0*
validate_shape(*&
_output_shapes
: 

Variable/Momentum/readIdentityVariable/Momentum*
_class
loc:@Variable*
T0*&
_output_shapes
: 

%Variable_1/Momentum/Initializer/zerosConst*
valueB *    *
dtype0*
_class
loc:@Variable_1*
_output_shapes
: 

Variable_1/Momentum
VariableV2*
_class
loc:@Variable_1*
shared_name *
	container *
dtype0*
shape: *
_output_shapes
: 
Í
Variable_1/Momentum/AssignAssignVariable_1/Momentum%Variable_1/Momentum/Initializer/zeros*
use_locking(*
_class
loc:@Variable_1*
T0*
validate_shape(*
_output_shapes
: 
}
Variable_1/Momentum/readIdentityVariable_1/Momentum*
_class
loc:@Variable_1*
T0*
_output_shapes
: 
Š
%Variable_2/Momentum/Initializer/zerosConst*%
valueB @*    *
dtype0*
_class
loc:@Variable_2*&
_output_shapes
: @
ś
Variable_2/Momentum
VariableV2*
_class
loc:@Variable_2*
shared_name *
	container *
dtype0*
shape: @*&
_output_shapes
: @
Ů
Variable_2/Momentum/AssignAssignVariable_2/Momentum%Variable_2/Momentum/Initializer/zeros*
use_locking(*
_class
loc:@Variable_2*
T0*
validate_shape(*&
_output_shapes
: @

Variable_2/Momentum/readIdentityVariable_2/Momentum*
_class
loc:@Variable_2*
T0*&
_output_shapes
: @

%Variable_3/Momentum/Initializer/zerosConst*
valueB@*    *
dtype0*
_class
loc:@Variable_3*
_output_shapes
:@

Variable_3/Momentum
VariableV2*
_class
loc:@Variable_3*
shared_name *
	container *
dtype0*
shape:@*
_output_shapes
:@
Í
Variable_3/Momentum/AssignAssignVariable_3/Momentum%Variable_3/Momentum/Initializer/zeros*
use_locking(*
_class
loc:@Variable_3*
T0*
validate_shape(*
_output_shapes
:@
}
Variable_3/Momentum/readIdentityVariable_3/Momentum*
_class
loc:@Variable_3*
T0*
_output_shapes
:@

%Variable_4/Momentum/Initializer/zerosConst*
valueB
*    *
dtype0*
_class
loc:@Variable_4* 
_output_shapes
:

Ş
Variable_4/Momentum
VariableV2*
_class
loc:@Variable_4*
shared_name *
	container *
dtype0*
shape:
* 
_output_shapes
:

Ó
Variable_4/Momentum/AssignAssignVariable_4/Momentum%Variable_4/Momentum/Initializer/zeros*
use_locking(*
_class
loc:@Variable_4*
T0*
validate_shape(* 
_output_shapes
:


Variable_4/Momentum/readIdentityVariable_4/Momentum*
_class
loc:@Variable_4*
T0* 
_output_shapes
:


%Variable_5/Momentum/Initializer/zerosConst*
valueB*    *
dtype0*
_class
loc:@Variable_5*
_output_shapes	
:
 
Variable_5/Momentum
VariableV2*
_class
loc:@Variable_5*
shared_name *
	container *
dtype0*
shape:*
_output_shapes	
:
Î
Variable_5/Momentum/AssignAssignVariable_5/Momentum%Variable_5/Momentum/Initializer/zeros*
use_locking(*
_class
loc:@Variable_5*
T0*
validate_shape(*
_output_shapes	
:
~
Variable_5/Momentum/readIdentityVariable_5/Momentum*
_class
loc:@Variable_5*
T0*
_output_shapes	
:

%Variable_6/Momentum/Initializer/zerosConst*
valueB	*    *
dtype0*
_class
loc:@Variable_6*
_output_shapes
:	
¨
Variable_6/Momentum
VariableV2*
_class
loc:@Variable_6*
shared_name *
	container *
dtype0*
shape:	*
_output_shapes
:	
Ň
Variable_6/Momentum/AssignAssignVariable_6/Momentum%Variable_6/Momentum/Initializer/zeros*
use_locking(*
_class
loc:@Variable_6*
T0*
validate_shape(*
_output_shapes
:	

Variable_6/Momentum/readIdentityVariable_6/Momentum*
_class
loc:@Variable_6*
T0*
_output_shapes
:	

%Variable_7/Momentum/Initializer/zerosConst*
valueB*    *
dtype0*
_class
loc:@Variable_7*
_output_shapes
:

Variable_7/Momentum
VariableV2*
_class
loc:@Variable_7*
shared_name *
	container *
dtype0*
shape:*
_output_shapes
:
Í
Variable_7/Momentum/AssignAssignVariable_7/Momentum%Variable_7/Momentum/Initializer/zeros*
use_locking(*
_class
loc:@Variable_7*
T0*
validate_shape(*
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
Momentum/momentumConst*
valueB
 *    *
dtype0*
_output_shapes
: 
˘
&Momentum/update_Variable/ApplyMomentumApplyMomentumVariableVariable/MomentumExponentialDecay2gradients_1/Conv2D_grad/tuple/control_dependency_1Momentum/momentum*
use_locking( *
_class
loc:@Variable*
use_nesterov( *
T0*&
_output_shapes
: 

(Momentum/update_Variable_1/ApplyMomentumApplyMomentum
Variable_1Variable_1/MomentumExponentialDecay3gradients_1/BiasAdd_grad/tuple/control_dependency_1Momentum/momentum*
use_locking( *
_class
loc:@Variable_1*
use_nesterov( *
T0*
_output_shapes
: 
Ź
(Momentum/update_Variable_2/ApplyMomentumApplyMomentum
Variable_2Variable_2/MomentumExponentialDecay4gradients_1/Conv2D_1_grad/tuple/control_dependency_1Momentum/momentum*
use_locking( *
_class
loc:@Variable_2*
use_nesterov( *
T0*&
_output_shapes
: @
Ą
(Momentum/update_Variable_3/ApplyMomentumApplyMomentum
Variable_3Variable_3/MomentumExponentialDecay5gradients_1/BiasAdd_1_grad/tuple/control_dependency_1Momentum/momentum*
use_locking( *
_class
loc:@Variable_3*
use_nesterov( *
T0*
_output_shapes
:@

(Momentum/update_Variable_4/ApplyMomentumApplyMomentum
Variable_4Variable_4/MomentumExponentialDecaygradients_1/AddN_3Momentum/momentum*
use_locking( *
_class
loc:@Variable_4*
use_nesterov( *
T0* 
_output_shapes
:

˙
(Momentum/update_Variable_5/ApplyMomentumApplyMomentum
Variable_5Variable_5/MomentumExponentialDecaygradients_1/AddN_2Momentum/momentum*
use_locking( *
_class
loc:@Variable_5*
use_nesterov( *
T0*
_output_shapes	
:

(Momentum/update_Variable_6/ApplyMomentumApplyMomentum
Variable_6Variable_6/MomentumExponentialDecaygradients_1/AddN_1Momentum/momentum*
use_locking( *
_class
loc:@Variable_6*
use_nesterov( *
T0*
_output_shapes
:	
ü
(Momentum/update_Variable_7/ApplyMomentumApplyMomentum
Variable_7Variable_7/MomentumExponentialDecaygradients_1/AddNMomentum/momentum*
use_locking( *
_class
loc:@Variable_7*
use_nesterov( *
T0*
_output_shapes
:
í
Momentum/updateNoOp'^Momentum/update_Variable/ApplyMomentum)^Momentum/update_Variable_1/ApplyMomentum)^Momentum/update_Variable_2/ApplyMomentum)^Momentum/update_Variable_3/ApplyMomentum)^Momentum/update_Variable_4/ApplyMomentum)^Momentum/update_Variable_5/ApplyMomentum)^Momentum/update_Variable_6/ApplyMomentum)^Momentum/update_Variable_7/ApplyMomentum

Momentum/valueConst^Momentum/update*
value	B :*
dtype0*
_class
loc:@Variable_8*
_output_shapes
: 

Momentum	AssignAdd
Variable_8Momentum/value*
use_locking( *
_class
loc:@Variable_8*
T0*
_output_shapes
: 
B
SoftmaxSoftmaxadd_1*
T0*
_output_shapes

:
ş
Conv2D_2Conv2DPlaceholder_2Variable/read*
paddingSAME*
data_formatNHWC*
T0*(
_output_shapes
:Ŕ= *
strides
*
use_cudnn_on_gpu(
y
	BiasAdd_2BiasAddConv2D_2Variable_1/read*(
_output_shapes
:Ŕ= *
T0*
data_formatNHWC
L
Relu_3Relu	BiasAdd_2*
T0*(
_output_shapes
:Ŕ= 
Ą
	MaxPool_2MaxPoolRelu_3*
ksize
*
data_formatNHWC*
T0*(
_output_shapes
:Ŕ= *
strides
*
paddingSAME
¸
Conv2D_3Conv2D	MaxPool_2Variable_2/read*
paddingSAME*
data_formatNHWC*
T0*(
_output_shapes
:Ŕ=@*
strides
*
use_cudnn_on_gpu(
y
	BiasAdd_3BiasAddConv2D_3Variable_3/read*(
_output_shapes
:Ŕ=@*
T0*
data_formatNHWC
L
Relu_4Relu	BiasAdd_3*
T0*(
_output_shapes
:Ŕ=@
Ą
	MaxPool_3MaxPoolRelu_4*
ksize
*
data_formatNHWC*
T0*(
_output_shapes
:Ŕ=@*
strides
*
paddingSAME
a
Reshape_14/shapeConst*
valueB"@B    *
dtype0*
_output_shapes
:
l

Reshape_14Reshape	MaxPool_3Reshape_14/shape*
Tshape0*
T0*!
_output_shapes
:Ŕ=

MatMul_2MatMul
Reshape_14Variable_4/read*
transpose_b( *
transpose_a( *
T0*!
_output_shapes
:Ŕ=
S
add_6AddMatMul_2Variable_5/read*
T0*!
_output_shapes
:Ŕ=
A
Relu_5Reluadd_6*
T0*!
_output_shapes
:Ŕ=
|
MatMul_3MatMulRelu_5Variable_6/read*
transpose_b( *
transpose_a( *
T0* 
_output_shapes
:
Ŕ=
R
add_7AddMatMul_3Variable_7/read*
T0* 
_output_shapes
:
Ŕ=
F
	Softmax_1Softmaxadd_7*
T0* 
_output_shapes
:
Ŕ=
P

save/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
ň
save/SaveV2/tensor_namesConst*Ľ
valueBBVariableBVariable/MomentumB
Variable_1BVariable_1/MomentumB
Variable_2BVariable_2/MomentumB
Variable_3BVariable_3/MomentumB
Variable_4BVariable_4/MomentumB
Variable_5BVariable_5/MomentumB
Variable_6BVariable_6/MomentumB
Variable_7BVariable_7/MomentumB
Variable_8*
dtype0*
_output_shapes
:

save/SaveV2/shape_and_slicesConst*5
value,B*B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:

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
valueBBVariable*
dtype0*
_output_shapes
:
h
save/RestoreV2/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*
dtypes
2*
_output_shapes
:
Ś
save/AssignAssignVariablesave/RestoreV2*
use_locking(*
_class
loc:@Variable*
T0*
validate_shape(*&
_output_shapes
: 
w
save/RestoreV2_1/tensor_namesConst*&
valueBBVariable/Momentum*
dtype0*
_output_shapes
:
j
!save/RestoreV2_1/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_1	RestoreV2
save/Constsave/RestoreV2_1/tensor_names!save/RestoreV2_1/shape_and_slices*
dtypes
2*
_output_shapes
:
ł
save/Assign_1AssignVariable/Momentumsave/RestoreV2_1*
use_locking(*
_class
loc:@Variable*
T0*
validate_shape(*&
_output_shapes
: 
p
save/RestoreV2_2/tensor_namesConst*
valueBB
Variable_1*
dtype0*
_output_shapes
:
j
!save/RestoreV2_2/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_2	RestoreV2
save/Constsave/RestoreV2_2/tensor_names!save/RestoreV2_2/shape_and_slices*
dtypes
2*
_output_shapes
:
˘
save/Assign_2Assign
Variable_1save/RestoreV2_2*
use_locking(*
_class
loc:@Variable_1*
T0*
validate_shape(*
_output_shapes
: 
y
save/RestoreV2_3/tensor_namesConst*(
valueBBVariable_1/Momentum*
dtype0*
_output_shapes
:
j
!save/RestoreV2_3/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_3	RestoreV2
save/Constsave/RestoreV2_3/tensor_names!save/RestoreV2_3/shape_and_slices*
dtypes
2*
_output_shapes
:
Ť
save/Assign_3AssignVariable_1/Momentumsave/RestoreV2_3*
use_locking(*
_class
loc:@Variable_1*
T0*
validate_shape(*
_output_shapes
: 
p
save/RestoreV2_4/tensor_namesConst*
valueBB
Variable_2*
dtype0*
_output_shapes
:
j
!save/RestoreV2_4/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_4	RestoreV2
save/Constsave/RestoreV2_4/tensor_names!save/RestoreV2_4/shape_and_slices*
dtypes
2*
_output_shapes
:
Ž
save/Assign_4Assign
Variable_2save/RestoreV2_4*
use_locking(*
_class
loc:@Variable_2*
T0*
validate_shape(*&
_output_shapes
: @
y
save/RestoreV2_5/tensor_namesConst*(
valueBBVariable_2/Momentum*
dtype0*
_output_shapes
:
j
!save/RestoreV2_5/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_5	RestoreV2
save/Constsave/RestoreV2_5/tensor_names!save/RestoreV2_5/shape_and_slices*
dtypes
2*
_output_shapes
:
ˇ
save/Assign_5AssignVariable_2/Momentumsave/RestoreV2_5*
use_locking(*
_class
loc:@Variable_2*
T0*
validate_shape(*&
_output_shapes
: @
p
save/RestoreV2_6/tensor_namesConst*
valueBB
Variable_3*
dtype0*
_output_shapes
:
j
!save/RestoreV2_6/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_6	RestoreV2
save/Constsave/RestoreV2_6/tensor_names!save/RestoreV2_6/shape_and_slices*
dtypes
2*
_output_shapes
:
˘
save/Assign_6Assign
Variable_3save/RestoreV2_6*
use_locking(*
_class
loc:@Variable_3*
T0*
validate_shape(*
_output_shapes
:@
y
save/RestoreV2_7/tensor_namesConst*(
valueBBVariable_3/Momentum*
dtype0*
_output_shapes
:
j
!save/RestoreV2_7/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_7	RestoreV2
save/Constsave/RestoreV2_7/tensor_names!save/RestoreV2_7/shape_and_slices*
dtypes
2*
_output_shapes
:
Ť
save/Assign_7AssignVariable_3/Momentumsave/RestoreV2_7*
use_locking(*
_class
loc:@Variable_3*
T0*
validate_shape(*
_output_shapes
:@
p
save/RestoreV2_8/tensor_namesConst*
valueBB
Variable_4*
dtype0*
_output_shapes
:
j
!save/RestoreV2_8/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_8	RestoreV2
save/Constsave/RestoreV2_8/tensor_names!save/RestoreV2_8/shape_and_slices*
dtypes
2*
_output_shapes
:
¨
save/Assign_8Assign
Variable_4save/RestoreV2_8*
use_locking(*
_class
loc:@Variable_4*
T0*
validate_shape(* 
_output_shapes
:

y
save/RestoreV2_9/tensor_namesConst*(
valueBBVariable_4/Momentum*
dtype0*
_output_shapes
:
j
!save/RestoreV2_9/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_9	RestoreV2
save/Constsave/RestoreV2_9/tensor_names!save/RestoreV2_9/shape_and_slices*
dtypes
2*
_output_shapes
:
ą
save/Assign_9AssignVariable_4/Momentumsave/RestoreV2_9*
use_locking(*
_class
loc:@Variable_4*
T0*
validate_shape(* 
_output_shapes
:

q
save/RestoreV2_10/tensor_namesConst*
valueBB
Variable_5*
dtype0*
_output_shapes
:
k
"save/RestoreV2_10/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_10	RestoreV2
save/Constsave/RestoreV2_10/tensor_names"save/RestoreV2_10/shape_and_slices*
dtypes
2*
_output_shapes
:
Ľ
save/Assign_10Assign
Variable_5save/RestoreV2_10*
use_locking(*
_class
loc:@Variable_5*
T0*
validate_shape(*
_output_shapes	
:
z
save/RestoreV2_11/tensor_namesConst*(
valueBBVariable_5/Momentum*
dtype0*
_output_shapes
:
k
"save/RestoreV2_11/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_11	RestoreV2
save/Constsave/RestoreV2_11/tensor_names"save/RestoreV2_11/shape_and_slices*
dtypes
2*
_output_shapes
:
Ž
save/Assign_11AssignVariable_5/Momentumsave/RestoreV2_11*
use_locking(*
_class
loc:@Variable_5*
T0*
validate_shape(*
_output_shapes	
:
q
save/RestoreV2_12/tensor_namesConst*
valueBB
Variable_6*
dtype0*
_output_shapes
:
k
"save/RestoreV2_12/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_12	RestoreV2
save/Constsave/RestoreV2_12/tensor_names"save/RestoreV2_12/shape_and_slices*
dtypes
2*
_output_shapes
:
Š
save/Assign_12Assign
Variable_6save/RestoreV2_12*
use_locking(*
_class
loc:@Variable_6*
T0*
validate_shape(*
_output_shapes
:	
z
save/RestoreV2_13/tensor_namesConst*(
valueBBVariable_6/Momentum*
dtype0*
_output_shapes
:
k
"save/RestoreV2_13/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_13	RestoreV2
save/Constsave/RestoreV2_13/tensor_names"save/RestoreV2_13/shape_and_slices*
dtypes
2*
_output_shapes
:
˛
save/Assign_13AssignVariable_6/Momentumsave/RestoreV2_13*
use_locking(*
_class
loc:@Variable_6*
T0*
validate_shape(*
_output_shapes
:	
q
save/RestoreV2_14/tensor_namesConst*
valueBB
Variable_7*
dtype0*
_output_shapes
:
k
"save/RestoreV2_14/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_14	RestoreV2
save/Constsave/RestoreV2_14/tensor_names"save/RestoreV2_14/shape_and_slices*
dtypes
2*
_output_shapes
:
¤
save/Assign_14Assign
Variable_7save/RestoreV2_14*
use_locking(*
_class
loc:@Variable_7*
T0*
validate_shape(*
_output_shapes
:
z
save/RestoreV2_15/tensor_namesConst*(
valueBBVariable_7/Momentum*
dtype0*
_output_shapes
:
k
"save/RestoreV2_15/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_15	RestoreV2
save/Constsave/RestoreV2_15/tensor_names"save/RestoreV2_15/shape_and_slices*
dtypes
2*
_output_shapes
:
­
save/Assign_15AssignVariable_7/Momentumsave/RestoreV2_15*
use_locking(*
_class
loc:@Variable_7*
T0*
validate_shape(*
_output_shapes
:
q
save/RestoreV2_16/tensor_namesConst*
valueBB
Variable_8*
dtype0*
_output_shapes
:
k
"save/RestoreV2_16/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_16	RestoreV2
save/Constsave/RestoreV2_16/tensor_names"save/RestoreV2_16/shape_and_slices*
dtypes
2*
_output_shapes
:
 
save/Assign_16Assign
Variable_8save/RestoreV2_16*
use_locking(*
_class
loc:@Variable_8*
T0*
validate_shape(*
_output_shapes
: 
­
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16
¤
initNoOp^Variable/Assign^Variable_1/Assign^Variable_2/Assign^Variable_3/Assign^Variable_4/Assign^Variable_5/Assign^Variable_6/Assign^Variable_7/Assign^Variable_8/Assign^Variable/Momentum/Assign^Variable_1/Momentum/Assign^Variable_2/Momentum/Assign^Variable_3/Momentum/Assign^Variable_4/Momentum/Assign^Variable_5/Momentum/Assign^Variable_6/Momentum/Assign^Variable_7/Momentum/Assign

Merge/MergeSummaryMergeSummarysummary_data_0summary_conv_0summary_pool_0summary_conv2_0summary_pool2_0lossconv1_weightsconv1_biasesconv2_weightsconv2_biasesfc1_weights
fc1_biasesfc2_weights
fc2_biaseslearning_rate*
N*
_output_shapes
: "Ăö¸t     âäĎ	âćXĺ˛ÖAJč
 +ţ*
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
2	
­
ApplyMomentum
var"T
accum"T
lr"T	
grad"T
momentum"T
out"T"
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 
x
Assign
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
p
	AssignAdd
ref"T

value"T

output_ref"T"
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
Č
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
î
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
í
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

ImageSummary
tag
tensor"T
summary"

max_imagesint(0"
Ttype0:
2"'
	bad_colortensorB:˙  ˙
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

Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
Ó
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
ë
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
2	

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

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
2	
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

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
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
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

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
2	
s

VariableV2
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring 
&
	ZerosLike
x"T
y"T"	
Ttype*1.4.02v1.4.0-rc1-11-g130a514Đ¨
l
PlaceholderPlaceholder*
dtype0*
shape:*&
_output_shapes
:
^
Placeholder_1Placeholder*
dtype0*
shape
:*
_output_shapes

:
r
Placeholder_2Placeholder*
dtype0*
shape:Ŕ=*(
_output_shapes
:Ŕ=
o
truncated_normal/shapeConst*%
valueB"             *
dtype0*
_output_shapes
:
Z
truncated_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
\
truncated_normal/stddevConst*
valueB
 *ÍĚĚ=*
dtype0*
_output_shapes
: 
§
 truncated_normal/TruncatedNormalTruncatedNormaltruncated_normal/shape*
dtype0*
seedą˙ĺ)*
seed2Ž*
T0*&
_output_shapes
: 

truncated_normal/mulMul truncated_normal/TruncatedNormaltruncated_normal/stddev*
T0*&
_output_shapes
: 
u
truncated_normalAddtruncated_normal/multruncated_normal/mean*
T0*&
_output_shapes
: 

Variable
VariableV2*
dtype0*
shape: *
	container *
shared_name *&
_output_shapes
: 
Ź
Variable/AssignAssignVariabletruncated_normal*
use_locking(*
_class
loc:@Variable*
T0*
validate_shape(*&
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
zerosConst*
valueB *    *
dtype0*
_output_shapes
: 
v

Variable_1
VariableV2*
dtype0*
shape: *
	container *
shared_name *
_output_shapes
: 

Variable_1/AssignAssign
Variable_1zeros*
use_locking(*
_class
loc:@Variable_1*
T0*
validate_shape(*
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
truncated_normal_1/shapeConst*%
valueB"          @   *
dtype0*
_output_shapes
:
\
truncated_normal_1/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
^
truncated_normal_1/stddevConst*
valueB
 *ÍĚĚ=*
dtype0*
_output_shapes
: 
Ť
"truncated_normal_1/TruncatedNormalTruncatedNormaltruncated_normal_1/shape*
dtype0*
seedą˙ĺ)*
seed2Ž*
T0*&
_output_shapes
: @

truncated_normal_1/mulMul"truncated_normal_1/TruncatedNormaltruncated_normal_1/stddev*
T0*&
_output_shapes
: @
{
truncated_normal_1Addtruncated_normal_1/multruncated_normal_1/mean*
T0*&
_output_shapes
: @


Variable_2
VariableV2*
dtype0*
shape: @*
	container *
shared_name *&
_output_shapes
: @
´
Variable_2/AssignAssign
Variable_2truncated_normal_1*
use_locking(*
_class
loc:@Variable_2*
T0*
validate_shape(*&
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
ConstConst*
valueB@*ÍĚĚ=*
dtype0*
_output_shapes
:@
v

Variable_3
VariableV2*
dtype0*
shape:@*
	container *
shared_name *
_output_shapes
:@

Variable_3/AssignAssign
Variable_3Const*
use_locking(*
_class
loc:@Variable_3*
T0*
validate_shape(*
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
valueB"      *
dtype0*
_output_shapes
:
\
truncated_normal_2/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
^
truncated_normal_2/stddevConst*
valueB
 *ÍĚĚ=*
dtype0*
_output_shapes
: 
Ľ
"truncated_normal_2/TruncatedNormalTruncatedNormaltruncated_normal_2/shape*
dtype0*
seedą˙ĺ)*
seed2Ž*
T0* 
_output_shapes
:


truncated_normal_2/mulMul"truncated_normal_2/TruncatedNormaltruncated_normal_2/stddev*
T0* 
_output_shapes
:

u
truncated_normal_2Addtruncated_normal_2/multruncated_normal_2/mean*
T0* 
_output_shapes
:



Variable_4
VariableV2*
dtype0*
shape:
*
	container *
shared_name * 
_output_shapes
:

Ž
Variable_4/AssignAssign
Variable_4truncated_normal_2*
use_locking(*
_class
loc:@Variable_4*
T0*
validate_shape(* 
_output_shapes
:

q
Variable_4/readIdentity
Variable_4*
_class
loc:@Variable_4*
T0* 
_output_shapes
:

V
Const_1Const*
valueB*ÍĚĚ=*
dtype0*
_output_shapes	
:
x

Variable_5
VariableV2*
dtype0*
shape:*
	container *
shared_name *
_output_shapes	
:

Variable_5/AssignAssign
Variable_5Const_1*
use_locking(*
_class
loc:@Variable_5*
T0*
validate_shape(*
_output_shapes	
:
l
Variable_5/readIdentity
Variable_5*
_class
loc:@Variable_5*
T0*
_output_shapes	
:
i
truncated_normal_3/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
\
truncated_normal_3/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
^
truncated_normal_3/stddevConst*
valueB
 *ÍĚĚ=*
dtype0*
_output_shapes
: 
¤
"truncated_normal_3/TruncatedNormalTruncatedNormaltruncated_normal_3/shape*
dtype0*
seedą˙ĺ)*
seed2Ž*
T0*
_output_shapes
:	

truncated_normal_3/mulMul"truncated_normal_3/TruncatedNormaltruncated_normal_3/stddev*
T0*
_output_shapes
:	
t
truncated_normal_3Addtruncated_normal_3/multruncated_normal_3/mean*
T0*
_output_shapes
:	


Variable_6
VariableV2*
dtype0*
shape:	*
	container *
shared_name *
_output_shapes
:	
­
Variable_6/AssignAssign
Variable_6truncated_normal_3*
use_locking(*
_class
loc:@Variable_6*
T0*
validate_shape(*
_output_shapes
:	
p
Variable_6/readIdentity
Variable_6*
_class
loc:@Variable_6*
T0*
_output_shapes
:	
T
Const_2Const*
valueB*ÍĚĚ=*
dtype0*
_output_shapes
:
v

Variable_7
VariableV2*
dtype0*
shape:*
	container *
shared_name *
_output_shapes
:

Variable_7/AssignAssign
Variable_7Const_2*
use_locking(*
_class
loc:@Variable_7*
T0*
validate_shape(*
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
´
Conv2DConv2DPlaceholderVariable/read*
paddingSAME*
use_cudnn_on_gpu(*
T0*&
_output_shapes
: *
strides
*
data_formatNHWC
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

MaxPoolMaxPoolRelu*
ksize
*
paddingSAME*
T0*&
_output_shapes
: *
strides
*
data_formatNHWC
´
Conv2D_1Conv2DMaxPoolVariable_2/read*
paddingSAME*
use_cudnn_on_gpu(*
T0*&
_output_shapes
:@*
strides
*
data_formatNHWC
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

	MaxPool_1MaxPoolRelu_1*
ksize
*
paddingSAME*
T0*&
_output_shapes
:@*
strides
*
data_formatNHWC
^
Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
d
ReshapeReshape	MaxPool_1Reshape/shape*
Tshape0*
T0*
_output_shapes
:	
z
MatMulMatMulReshapeVariable_4/read*
transpose_b( *
transpose_a( *
T0*
_output_shapes
:	
M
addAddMatMulVariable_5/read*
T0*
_output_shapes
:	
=
Relu_2Reluadd*
T0*
_output_shapes
:	
z
MatMul_1MatMulRelu_2Variable_6/read*
transpose_b( *
transpose_a( *
T0*
_output_shapes

:
P
add_1AddMatMul_1Variable_7/read*
T0*
_output_shapes

:
d
Slice/beginConst*%
valueB"                *
dtype0*
_output_shapes
:
c

Slice/sizeConst*%
valueB"   ˙˙˙˙˙˙˙˙   *
dtype0*
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
Const_3Const*%
valueB"             *
dtype0*
_output_shapes
:
X
MinMinSliceConst_3*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
G
subSubSliceMin*
T0*&
_output_shapes
:
`
Const_4Const*%
valueB"             *
dtype0*
_output_shapes
:
V
MaxMaxsubConst_4*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
J
mul/yConst*
valueB
 *  C*
dtype0*
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
Reshape_1/shapeConst*!
valueB"         *
dtype0*
_output_shapes
:
i
	Reshape_1ReshapetruedivReshape_1/shape*
Tshape0*
T0*"
_output_shapes
:
c
transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:
k
	transpose	Transpose	Reshape_1transpose/perm*
Tperm0*
T0*"
_output_shapes
:
h
Reshape_2/shapeConst*%
valueB"˙˙˙˙         *
dtype0*
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
valueB Bsummary_data_0*
dtype0*
_output_shapes
: 

summary_data_0ImageSummarysummary_data_0/tag	Reshape_2*

max_images*
T0*
	bad_colorB:˙  ˙*
_output_shapes
: 
f
Slice_1/beginConst*%
valueB"                *
dtype0*
_output_shapes
:
e
Slice_1/sizeConst*%
valueB"   ˙˙˙˙˙˙˙˙   *
dtype0*
_output_shapes
:
s
Slice_1SliceConv2DSlice_1/beginSlice_1/size*
Index0*
T0*&
_output_shapes
:
`
Const_5Const*%
valueB"             *
dtype0*
_output_shapes
:
\
Min_1MinSlice_1Const_5*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
M
sub_1SubSlice_1Min_1*
T0*&
_output_shapes
:
`
Const_6Const*%
valueB"             *
dtype0*
_output_shapes
:
Z
Max_1Maxsub_1Const_6*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
L
mul_1/yConst*
valueB
 *  C*
dtype0*
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
Reshape_3/shapeConst*!
valueB"         *
dtype0*
_output_shapes
:
k
	Reshape_3Reshape	truediv_1Reshape_3/shape*
Tshape0*
T0*"
_output_shapes
:
e
transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:
o
transpose_1	Transpose	Reshape_3transpose_1/perm*
Tperm0*
T0*"
_output_shapes
:
h
Reshape_4/shapeConst*%
valueB"˙˙˙˙         *
dtype0*
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
valueB Bsummary_conv_0*
dtype0*
_output_shapes
: 

summary_conv_0ImageSummarysummary_conv_0/tag	Reshape_4*

max_images*
T0*
	bad_colorB:˙  ˙*
_output_shapes
: 
f
Slice_2/beginConst*%
valueB"                *
dtype0*
_output_shapes
:
e
Slice_2/sizeConst*%
valueB"   ˙˙˙˙˙˙˙˙   *
dtype0*
_output_shapes
:
t
Slice_2SliceMaxPoolSlice_2/beginSlice_2/size*
Index0*
T0*&
_output_shapes
:
`
Const_7Const*%
valueB"             *
dtype0*
_output_shapes
:
\
Min_2MinSlice_2Const_7*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
M
sub_2SubSlice_2Min_2*
T0*&
_output_shapes
:
`
Const_8Const*%
valueB"             *
dtype0*
_output_shapes
:
Z
Max_2Maxsub_2Const_8*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
L
mul_2/yConst*
valueB
 *  C*
dtype0*
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
Reshape_5/shapeConst*!
valueB"         *
dtype0*
_output_shapes
:
k
	Reshape_5Reshape	truediv_2Reshape_5/shape*
Tshape0*
T0*"
_output_shapes
:
e
transpose_2/permConst*!
valueB"          *
dtype0*
_output_shapes
:
o
transpose_2	Transpose	Reshape_5transpose_2/perm*
Tperm0*
T0*"
_output_shapes
:
h
Reshape_6/shapeConst*%
valueB"˙˙˙˙         *
dtype0*
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
valueB Bsummary_pool_0*
dtype0*
_output_shapes
: 

summary_pool_0ImageSummarysummary_pool_0/tag	Reshape_6*

max_images*
T0*
	bad_colorB:˙  ˙*
_output_shapes
: 
f
Slice_3/beginConst*%
valueB"                *
dtype0*
_output_shapes
:
e
Slice_3/sizeConst*%
valueB"   ˙˙˙˙˙˙˙˙   *
dtype0*
_output_shapes
:
u
Slice_3SliceConv2D_1Slice_3/beginSlice_3/size*
Index0*
T0*&
_output_shapes
:
`
Const_9Const*%
valueB"             *
dtype0*
_output_shapes
:
\
Min_3MinSlice_3Const_9*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
M
sub_3SubSlice_3Min_3*
T0*&
_output_shapes
:
a
Const_10Const*%
valueB"             *
dtype0*
_output_shapes
:
[
Max_3Maxsub_3Const_10*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
L
mul_3/yConst*
valueB
 *  C*
dtype0*
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
Reshape_7/shapeConst*!
valueB"         *
dtype0*
_output_shapes
:
k
	Reshape_7Reshape	truediv_3Reshape_7/shape*
Tshape0*
T0*"
_output_shapes
:
e
transpose_3/permConst*!
valueB"          *
dtype0*
_output_shapes
:
o
transpose_3	Transpose	Reshape_7transpose_3/perm*
Tperm0*
T0*"
_output_shapes
:
h
Reshape_8/shapeConst*%
valueB"˙˙˙˙         *
dtype0*
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
valueB Bsummary_conv2_0*
dtype0*
_output_shapes
: 

summary_conv2_0ImageSummarysummary_conv2_0/tag	Reshape_8*

max_images*
T0*
	bad_colorB:˙  ˙*
_output_shapes
: 
f
Slice_4/beginConst*%
valueB"                *
dtype0*
_output_shapes
:
e
Slice_4/sizeConst*%
valueB"   ˙˙˙˙˙˙˙˙   *
dtype0*
_output_shapes
:
v
Slice_4Slice	MaxPool_1Slice_4/beginSlice_4/size*
Index0*
T0*&
_output_shapes
:
a
Const_11Const*%
valueB"             *
dtype0*
_output_shapes
:
]
Min_4MinSlice_4Const_11*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
M
sub_4SubSlice_4Min_4*
T0*&
_output_shapes
:
a
Const_12Const*%
valueB"             *
dtype0*
_output_shapes
:
[
Max_4Maxsub_4Const_12*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
L
mul_4/yConst*
valueB
 *  C*
dtype0*
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
Reshape_9/shapeConst*!
valueB"         *
dtype0*
_output_shapes
:
k
	Reshape_9Reshape	truediv_4Reshape_9/shape*
Tshape0*
T0*"
_output_shapes
:
e
transpose_4/permConst*!
valueB"          *
dtype0*
_output_shapes
:
o
transpose_4	Transpose	Reshape_9transpose_4/perm*
Tperm0*
T0*"
_output_shapes
:
i
Reshape_10/shapeConst*%
valueB"˙˙˙˙         *
dtype0*
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
valueB Bsummary_pool2_0*
dtype0*
_output_shapes
: 

summary_pool2_0ImageSummarysummary_pool2_0/tag
Reshape_10*

max_images*
T0*
	bad_colorB:˙  ˙*
_output_shapes
: 
F
RankConst*
value	B :*
dtype0*
_output_shapes
: 
V
ShapeConst*
valueB"      *
dtype0*
_output_shapes
:
H
Rank_1Const*
value	B :*
dtype0*
_output_shapes
: 
X
Shape_1Const*
valueB"      *
dtype0*
_output_shapes
:
G
Sub/yConst*
value	B :*
dtype0*
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
Slice_5/sizeConst*
valueB:*
dtype0*
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
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
M
concat/axisConst*
value	B : *
dtype0*
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
Rank_2Const*
value	B :*
dtype0*
_output_shapes
: 
X
Shape_2Const*
valueB"      *
dtype0*
_output_shapes
:
I
Sub_1/yConst*
value	B :*
dtype0*
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
Slice_6/sizeConst*
valueB:*
dtype0*
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
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
O
concat_1/axisConst*
value	B : *
dtype0*
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

SoftmaxCrossEntropyWithLogitsSoftmaxCrossEntropyWithLogits
Reshape_11
Reshape_12*
T0*$
_output_shapes
::
I
Sub_2/yConst*
value	B :*
dtype0*
_output_shapes
: 
<
Sub_2SubRankSub_2/y*
T0*
_output_shapes
: 
W
Slice_7/beginConst*
valueB: *
dtype0*
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
:˙˙˙˙˙˙˙˙˙
p

Reshape_13ReshapeSoftmaxCrossEntropyWithLogitsSlice_7*
Tshape0*
T0*
_output_shapes
:
R
Const_13Const*
valueB: *
dtype0*
_output_shapes
:
`
MeanMean
Reshape_13Const_13*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
N
	loss/tagsConst*
valueB
 Bloss*
dtype0*
_output_shapes
: 
G
lossScalarSummary	loss/tagsMean*
T0*
_output_shapes
: 
R
gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
T
gradients/ConstConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Y
gradients/FillFillgradients/Shapegradients/Const*
T0*
_output_shapes
: 
k
!gradients/Mean_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:

gradients/Mean_grad/ReshapeReshapegradients/Fill!gradients/Mean_grad/Reshape/shape*
Tshape0*
T0*
_output_shapes
:
l
"gradients/Mean_grad/Tile/multiplesConst*
valueB:*
dtype0*
_output_shapes
:

gradients/Mean_grad/TileTilegradients/Mean_grad/Reshape"gradients/Mean_grad/Tile/multiples*

Tmultiples0*
T0*
_output_shapes
:
c
gradients/Mean_grad/ShapeConst*
valueB:*
dtype0*
_output_shapes
:
^
gradients/Mean_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 

gradients/Mean_grad/ConstConst*
valueB: *
dtype0*,
_class"
 loc:@gradients/Mean_grad/Shape*
_output_shapes
:
Â
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shapegradients/Mean_grad/Const*
	keep_dims( *,
_class"
 loc:@gradients/Mean_grad/Shape*

Tidx0*
T0*
_output_shapes
: 

gradients/Mean_grad/Const_1Const*
valueB: *
dtype0*,
_class"
 loc:@gradients/Mean_grad/Shape*
_output_shapes
:
Č
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_1gradients/Mean_grad/Const_1*
	keep_dims( *,
_class"
 loc:@gradients/Mean_grad/Shape*

Tidx0*
T0*
_output_shapes
: 

gradients/Mean_grad/Maximum/yConst*
value	B :*
dtype0*,
_class"
 loc:@gradients/Mean_grad/Shape*
_output_shapes
: 
°
gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*,
_class"
 loc:@gradients/Mean_grad/Shape*
T0*
_output_shapes
: 
Ž
gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum*,
_class"
 loc:@gradients/Mean_grad/Shape*
T0*
_output_shapes
: 
n
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv*

DstT0*

SrcT0*
_output_shapes
: 

gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*
T0*
_output_shapes
:
i
gradients/Reshape_13_grad/ShapeConst*
valueB:*
dtype0*
_output_shapes
:

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

;gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dimConst*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
: 
Ú
7gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims
ExpandDims!gradients/Reshape_13_grad/Reshape;gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim*

Tdim0*
T0*
_output_shapes

:
ş
0gradients/SoftmaxCrossEntropyWithLogits_grad/mulMul7gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDimsSoftmaxCrossEntropyWithLogits:1*
T0*
_output_shapes

:
p
gradients/Reshape_11_grad/ShapeConst*
valueB"      *
dtype0*
_output_shapes
:
ś
!gradients/Reshape_11_grad/ReshapeReshape0gradients/SoftmaxCrossEntropyWithLogits_grad/mulgradients/Reshape_11_grad/Shape*
Tshape0*
T0*
_output_shapes

:
k
gradients/add_1_grad/ShapeConst*
valueB"      *
dtype0*
_output_shapes
:
f
gradients/add_1_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
ş
*gradients/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_1_grad/Shapegradients/add_1_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ž
gradients/add_1_grad/SumSum!gradients/Reshape_11_grad/Reshape*gradients/add_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

gradients/add_1_grad/ReshapeReshapegradients/add_1_grad/Sumgradients/add_1_grad/Shape*
Tshape0*
T0*
_output_shapes

:
˛
gradients/add_1_grad/Sum_1Sum!gradients/Reshape_11_grad/Reshape,gradients/add_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

gradients/add_1_grad/Reshape_1Reshapegradients/add_1_grad/Sum_1gradients/add_1_grad/Shape_1*
Tshape0*
T0*
_output_shapes
:
§
gradients/MatMul_1_grad/MatMulMatMulgradients/add_1_grad/ReshapeVariable_6/read*
transpose_b(*
transpose_a( *
T0*
_output_shapes
:	
 
 gradients/MatMul_1_grad/MatMul_1MatMulRelu_2gradients/add_1_grad/Reshape*
transpose_b( *
transpose_a(*
T0*
_output_shapes
:	
|
gradients/Relu_2_grad/ReluGradReluGradgradients/MatMul_1_grad/MatMulRelu_2*
T0*
_output_shapes
:	
i
gradients/add_grad/ShapeConst*
valueB"      *
dtype0*
_output_shapes
:
e
gradients/add_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
´
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
§
gradients/add_grad/SumSumgradients/Relu_2_grad/ReluGrad(gradients/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*
Tshape0*
T0*
_output_shapes
:	
Ť
gradients/add_grad/Sum_1Sumgradients/Relu_2_grad/ReluGrad*gradients/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
Tshape0*
T0*
_output_shapes	
:
Ł
gradients/MatMul_grad/MatMulMatMulgradients/add_grad/ReshapeVariable_4/read*
transpose_b(*
transpose_a( *
T0*
_output_shapes
:	

gradients/MatMul_grad/MatMul_1MatMulReshapegradients/add_grad/Reshape*
transpose_b( *
transpose_a(*
T0* 
_output_shapes
:

u
gradients/Reshape_grad/ShapeConst*%
valueB"         @   *
dtype0*
_output_shapes
:
¤
gradients/Reshape_grad/ReshapeReshapegradients/MatMul_grad/MatMulgradients/Reshape_grad/Shape*
Tshape0*
T0*&
_output_shapes
:@
é
$gradients/MaxPool_1_grad/MaxPoolGradMaxPoolGradRelu_1	MaxPool_1gradients/Reshape_grad/Reshape*
ksize
*
paddingSAME*
T0*&
_output_shapes
:@*
strides
*
data_formatNHWC

gradients/Relu_1_grad/ReluGradReluGrad$gradients/MaxPool_1_grad/MaxPoolGradRelu_1*
T0*&
_output_shapes
:@

$gradients/BiasAdd_1_grad/BiasAddGradBiasAddGradgradients/Relu_1_grad/ReluGrad*
data_formatNHWC*
T0*
_output_shapes
:@

gradients/Conv2D_1_grad/ShapeNShapeNMaxPoolVariable_2/read*
N*
out_type0*
T0* 
_output_shapes
::

+gradients/Conv2D_1_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_1_grad/ShapeNVariable_2/readgradients/Relu_1_grad/ReluGrad*
paddingSAME*
use_cudnn_on_gpu(*
T0*&
_output_shapes
: *
strides
*
data_formatNHWC

,gradients/Conv2D_1_grad/Conv2DBackpropFilterConv2DBackpropFilterMaxPool gradients/Conv2D_1_grad/ShapeN:1gradients/Relu_1_grad/ReluGrad*
paddingSAME*
use_cudnn_on_gpu(*
T0*&
_output_shapes
: @*
strides
*
data_formatNHWC
đ
"gradients/MaxPool_grad/MaxPoolGradMaxPoolGradReluMaxPool+gradients/Conv2D_1_grad/Conv2DBackpropInput*
ksize
*
paddingSAME*
T0*&
_output_shapes
: *
strides
*
data_formatNHWC

gradients/Relu_grad/ReluGradReluGrad"gradients/MaxPool_grad/MaxPoolGradRelu*
T0*&
_output_shapes
: 

"gradients/BiasAdd_grad/BiasAddGradBiasAddGradgradients/Relu_grad/ReluGrad*
data_formatNHWC*
T0*
_output_shapes
: 

gradients/Conv2D_grad/ShapeNShapeNPlaceholderVariable/read*
N*
out_type0*
T0* 
_output_shapes
::

)gradients/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_grad/ShapeNVariable/readgradients/Relu_grad/ReluGrad*
paddingSAME*
use_cudnn_on_gpu(*
T0*&
_output_shapes
:*
strides
*
data_formatNHWC

*gradients/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterPlaceholdergradients/Conv2D_grad/ShapeN:1gradients/Relu_grad/ReluGrad*
paddingSAME*
use_cudnn_on_gpu(*
T0*&
_output_shapes
: *
strides
*
data_formatNHWC
¨
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
global_norm/ConstConst*
valueB: *
dtype0*
_output_shapes
:
z
global_norm/SumSumglobal_norm/stackglobal_norm/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
X
global_norm/Const_1Const*
valueB
 *   @*
dtype0*
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
valueB Bconv1_weights*
dtype0*
_output_shapes
: 
l
conv1_weightsScalarSummaryconv1_weights/tagsglobal_norm/global_norm*
T0*
_output_shapes
: 

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
global_norm_1/ConstConst*
valueB: *
dtype0*
_output_shapes
:

global_norm_1/SumSumglobal_norm_1/stackglobal_norm_1/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
Z
global_norm_1/Const_1Const*
valueB
 *   @*
dtype0*
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
valueB Bconv1_biases*
dtype0*
_output_shapes
: 
l
conv1_biasesScalarSummaryconv1_biases/tagsglobal_norm_1/global_norm*
T0*
_output_shapes
: 
Ž
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
global_norm_2/ConstConst*
valueB: *
dtype0*
_output_shapes
:

global_norm_2/SumSumglobal_norm_2/stackglobal_norm_2/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
Z
global_norm_2/Const_1Const*
valueB
 *   @*
dtype0*
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
valueB Bconv2_weights*
dtype0*
_output_shapes
: 
n
conv2_weightsScalarSummaryconv2_weights/tagsglobal_norm_2/global_norm*
T0*
_output_shapes
: 

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
global_norm_3/ConstConst*
valueB: *
dtype0*
_output_shapes
:

global_norm_3/SumSumglobal_norm_3/stackglobal_norm_3/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
Z
global_norm_3/Const_1Const*
valueB
 *   @*
dtype0*
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
valueB Bconv2_biases*
dtype0*
_output_shapes
: 
l
conv2_biasesScalarSummaryconv2_biases/tagsglobal_norm_3/global_norm*
T0*
_output_shapes
: 

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
global_norm_4/ConstConst*
valueB: *
dtype0*
_output_shapes
:

global_norm_4/SumSumglobal_norm_4/stackglobal_norm_4/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
Z
global_norm_4/Const_1Const*
valueB
 *   @*
dtype0*
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
valueB Bfc1_weights*
dtype0*
_output_shapes
: 
j
fc1_weightsScalarSummaryfc1_weights/tagsglobal_norm_4/global_norm*
T0*
_output_shapes
: 

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
global_norm_5/ConstConst*
valueB: *
dtype0*
_output_shapes
:

global_norm_5/SumSumglobal_norm_5/stackglobal_norm_5/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
Z
global_norm_5/Const_1Const*
valueB
 *   @*
dtype0*
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
fc1_biases/tagsConst*
valueB B
fc1_biases*
dtype0*
_output_shapes
: 
h

fc1_biasesScalarSummaryfc1_biases/tagsglobal_norm_5/global_norm*
T0*
_output_shapes
: 

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
global_norm_6/ConstConst*
valueB: *
dtype0*
_output_shapes
:

global_norm_6/SumSumglobal_norm_6/stackglobal_norm_6/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
Z
global_norm_6/Const_1Const*
valueB
 *   @*
dtype0*
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
valueB Bfc2_weights*
dtype0*
_output_shapes
: 
j
fc2_weightsScalarSummaryfc2_weights/tagsglobal_norm_6/global_norm*
T0*
_output_shapes
: 

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
global_norm_7/ConstConst*
valueB: *
dtype0*
_output_shapes
:

global_norm_7/SumSumglobal_norm_7/stackglobal_norm_7/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
Z
global_norm_7/Const_1Const*
valueB
 *   @*
dtype0*
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
fc2_biases/tagsConst*
valueB B
fc2_biases*
dtype0*
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
mul_5/xConst*
valueB
 *o:*
dtype0*
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
Variable_8/initial_valueConst*
value	B : *
dtype0*
_output_shapes
: 
n

Variable_8
VariableV2*
dtype0*
shape: *
	container *
shared_name *
_output_shapes
: 
Ş
Variable_8/AssignAssign
Variable_8Variable_8/initial_value*
use_locking(*
_class
loc:@Variable_8*
T0*
validate_shape(*
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
mul_6/yConst*
value	B :*
dtype0*
_output_shapes
: 
G
mul_6MulVariable_8/readmul_6/y*
T0*
_output_shapes
: 
c
ExponentialDecay/learning_rateConst*
valueB
 *
×#<*
dtype0*
_output_shapes
: 
T
ExponentialDecay/CastCastmul_6*

DstT0*

SrcT0*
_output_shapes
: 
]
ExponentialDecay/Cast_1/xConst*
valueB	 :Ŕ=*
dtype0*
_output_shapes
: 
j
ExponentialDecay/Cast_1CastExponentialDecay/Cast_1/x*

DstT0*

SrcT0*
_output_shapes
: 
^
ExponentialDecay/Cast_2/xConst*
valueB
 *33s?*
dtype0*
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
valueB Blearning_rate*
dtype0*
_output_shapes
: 
e
learning_rateScalarSummarylearning_rate/tagsExponentialDecay*
T0*
_output_shapes
: 
T
gradients_1/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
V
gradients_1/ConstConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
_
gradients_1/FillFillgradients_1/Shapegradients_1/Const*
T0*
_output_shapes
: 
_
gradients_1/add_5_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
a
gradients_1/add_5_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
Ŕ
,gradients_1/add_5_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/add_5_grad/Shapegradients_1/add_5_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ą
gradients_1/add_5_grad/SumSumgradients_1/Fill,gradients_1/add_5_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

gradients_1/add_5_grad/ReshapeReshapegradients_1/add_5_grad/Sumgradients_1/add_5_grad/Shape*
Tshape0*
T0*
_output_shapes
: 
Ľ
gradients_1/add_5_grad/Sum_1Sumgradients_1/Fill.gradients_1/add_5_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

 gradients_1/add_5_grad/Reshape_1Reshapegradients_1/add_5_grad/Sum_1gradients_1/add_5_grad/Shape_1*
Tshape0*
T0*
_output_shapes
: 
s
'gradients_1/add_5_grad/tuple/group_depsNoOp^gradients_1/add_5_grad/Reshape!^gradients_1/add_5_grad/Reshape_1
Ů
/gradients_1/add_5_grad/tuple/control_dependencyIdentitygradients_1/add_5_grad/Reshape(^gradients_1/add_5_grad/tuple/group_deps*1
_class'
%#loc:@gradients_1/add_5_grad/Reshape*
T0*
_output_shapes
: 
ß
1gradients_1/add_5_grad/tuple/control_dependency_1Identity gradients_1/add_5_grad/Reshape_1(^gradients_1/add_5_grad/tuple/group_deps*3
_class)
'%loc:@gradients_1/add_5_grad/Reshape_1*
T0*
_output_shapes
: 
m
#gradients_1/Mean_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:
ą
gradients_1/Mean_grad/ReshapeReshape/gradients_1/add_5_grad/tuple/control_dependency#gradients_1/Mean_grad/Reshape/shape*
Tshape0*
T0*
_output_shapes
:
n
$gradients_1/Mean_grad/Tile/multiplesConst*
valueB:*
dtype0*
_output_shapes
:

gradients_1/Mean_grad/TileTilegradients_1/Mean_grad/Reshape$gradients_1/Mean_grad/Tile/multiples*

Tmultiples0*
T0*
_output_shapes
:
e
gradients_1/Mean_grad/ShapeConst*
valueB:*
dtype0*
_output_shapes
:
`
gradients_1/Mean_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 

gradients_1/Mean_grad/ConstConst*
valueB: *
dtype0*.
_class$
" loc:@gradients_1/Mean_grad/Shape*
_output_shapes
:
Ę
gradients_1/Mean_grad/ProdProdgradients_1/Mean_grad/Shapegradients_1/Mean_grad/Const*
	keep_dims( *.
_class$
" loc:@gradients_1/Mean_grad/Shape*

Tidx0*
T0*
_output_shapes
: 

gradients_1/Mean_grad/Const_1Const*
valueB: *
dtype0*.
_class$
" loc:@gradients_1/Mean_grad/Shape*
_output_shapes
:
Đ
gradients_1/Mean_grad/Prod_1Prodgradients_1/Mean_grad/Shape_1gradients_1/Mean_grad/Const_1*
	keep_dims( *.
_class$
" loc:@gradients_1/Mean_grad/Shape*

Tidx0*
T0*
_output_shapes
: 

gradients_1/Mean_grad/Maximum/yConst*
value	B :*
dtype0*.
_class$
" loc:@gradients_1/Mean_grad/Shape*
_output_shapes
: 
¸
gradients_1/Mean_grad/MaximumMaximumgradients_1/Mean_grad/Prod_1gradients_1/Mean_grad/Maximum/y*.
_class$
" loc:@gradients_1/Mean_grad/Shape*
T0*
_output_shapes
: 
ś
gradients_1/Mean_grad/floordivFloorDivgradients_1/Mean_grad/Prodgradients_1/Mean_grad/Maximum*.
_class$
" loc:@gradients_1/Mean_grad/Shape*
T0*
_output_shapes
: 
r
gradients_1/Mean_grad/CastCastgradients_1/Mean_grad/floordiv*

DstT0*

SrcT0*
_output_shapes
: 

gradients_1/Mean_grad/truedivRealDivgradients_1/Mean_grad/Tilegradients_1/Mean_grad/Cast*
T0*
_output_shapes
:
_
gradients_1/mul_5_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
a
gradients_1/mul_5_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
Ŕ
,gradients_1/mul_5_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/mul_5_grad/Shapegradients_1/mul_5_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
|
gradients_1/mul_5_grad/mulMul1gradients_1/add_5_grad/tuple/control_dependency_1add_4*
T0*
_output_shapes
: 
Ť
gradients_1/mul_5_grad/SumSumgradients_1/mul_5_grad/mul,gradients_1/mul_5_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

gradients_1/mul_5_grad/ReshapeReshapegradients_1/mul_5_grad/Sumgradients_1/mul_5_grad/Shape*
Tshape0*
T0*
_output_shapes
: 

gradients_1/mul_5_grad/mul_1Mulmul_5/x1gradients_1/add_5_grad/tuple/control_dependency_1*
T0*
_output_shapes
: 
ą
gradients_1/mul_5_grad/Sum_1Sumgradients_1/mul_5_grad/mul_1.gradients_1/mul_5_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

 gradients_1/mul_5_grad/Reshape_1Reshapegradients_1/mul_5_grad/Sum_1gradients_1/mul_5_grad/Shape_1*
Tshape0*
T0*
_output_shapes
: 
s
'gradients_1/mul_5_grad/tuple/group_depsNoOp^gradients_1/mul_5_grad/Reshape!^gradients_1/mul_5_grad/Reshape_1
Ů
/gradients_1/mul_5_grad/tuple/control_dependencyIdentitygradients_1/mul_5_grad/Reshape(^gradients_1/mul_5_grad/tuple/group_deps*1
_class'
%#loc:@gradients_1/mul_5_grad/Reshape*
T0*
_output_shapes
: 
ß
1gradients_1/mul_5_grad/tuple/control_dependency_1Identity gradients_1/mul_5_grad/Reshape_1(^gradients_1/mul_5_grad/tuple/group_deps*3
_class)
'%loc:@gradients_1/mul_5_grad/Reshape_1*
T0*
_output_shapes
: 
k
!gradients_1/Reshape_13_grad/ShapeConst*
valueB:*
dtype0*
_output_shapes
:
Ł
#gradients_1/Reshape_13_grad/ReshapeReshapegradients_1/Mean_grad/truediv!gradients_1/Reshape_13_grad/Shape*
Tshape0*
T0*
_output_shapes
:
_
gradients_1/add_4_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
a
gradients_1/add_4_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
Ŕ
,gradients_1/add_4_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/add_4_grad/Shapegradients_1/add_4_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Â
gradients_1/add_4_grad/SumSum1gradients_1/mul_5_grad/tuple/control_dependency_1,gradients_1/add_4_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

gradients_1/add_4_grad/ReshapeReshapegradients_1/add_4_grad/Sumgradients_1/add_4_grad/Shape*
Tshape0*
T0*
_output_shapes
: 
Ć
gradients_1/add_4_grad/Sum_1Sum1gradients_1/mul_5_grad/tuple/control_dependency_1.gradients_1/add_4_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

 gradients_1/add_4_grad/Reshape_1Reshapegradients_1/add_4_grad/Sum_1gradients_1/add_4_grad/Shape_1*
Tshape0*
T0*
_output_shapes
: 
s
'gradients_1/add_4_grad/tuple/group_depsNoOp^gradients_1/add_4_grad/Reshape!^gradients_1/add_4_grad/Reshape_1
Ů
/gradients_1/add_4_grad/tuple/control_dependencyIdentitygradients_1/add_4_grad/Reshape(^gradients_1/add_4_grad/tuple/group_deps*1
_class'
%#loc:@gradients_1/add_4_grad/Reshape*
T0*
_output_shapes
: 
ß
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

=gradients_1/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dimConst*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
: 
ŕ
9gradients_1/SoftmaxCrossEntropyWithLogits_grad/ExpandDims
ExpandDims#gradients_1/Reshape_13_grad/Reshape=gradients_1/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim*

Tdim0*
T0*
_output_shapes

:
ž
2gradients_1/SoftmaxCrossEntropyWithLogits_grad/mulMul9gradients_1/SoftmaxCrossEntropyWithLogits_grad/ExpandDimsSoftmaxCrossEntropyWithLogits:1*
T0*
_output_shapes

:
_
gradients_1/add_3_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
a
gradients_1/add_3_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
Ŕ
,gradients_1/add_3_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/add_3_grad/Shapegradients_1/add_3_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ŕ
gradients_1/add_3_grad/SumSum/gradients_1/add_4_grad/tuple/control_dependency,gradients_1/add_3_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

gradients_1/add_3_grad/ReshapeReshapegradients_1/add_3_grad/Sumgradients_1/add_3_grad/Shape*
Tshape0*
T0*
_output_shapes
: 
Ä
gradients_1/add_3_grad/Sum_1Sum/gradients_1/add_4_grad/tuple/control_dependency.gradients_1/add_3_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

 gradients_1/add_3_grad/Reshape_1Reshapegradients_1/add_3_grad/Sum_1gradients_1/add_3_grad/Shape_1*
Tshape0*
T0*
_output_shapes
: 
s
'gradients_1/add_3_grad/tuple/group_depsNoOp^gradients_1/add_3_grad/Reshape!^gradients_1/add_3_grad/Reshape_1
Ů
/gradients_1/add_3_grad/tuple/control_dependencyIdentitygradients_1/add_3_grad/Reshape(^gradients_1/add_3_grad/tuple/group_deps*1
_class'
%#loc:@gradients_1/add_3_grad/Reshape*
T0*
_output_shapes
: 
ß
1gradients_1/add_3_grad/tuple/control_dependency_1Identity gradients_1/add_3_grad/Reshape_1(^gradients_1/add_3_grad/tuple/group_deps*3
_class)
'%loc:@gradients_1/add_3_grad/Reshape_1*
T0*
_output_shapes
: 

gradients_1/L2Loss_3_grad/mulMulVariable_7/read1gradients_1/add_4_grad/tuple/control_dependency_1*
T0*
_output_shapes
:
r
!gradients_1/Reshape_11_grad/ShapeConst*
valueB"      *
dtype0*
_output_shapes
:
ź
#gradients_1/Reshape_11_grad/ReshapeReshape2gradients_1/SoftmaxCrossEntropyWithLogits_grad/mul!gradients_1/Reshape_11_grad/Shape*
Tshape0*
T0*
_output_shapes

:
_
gradients_1/add_2_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
a
gradients_1/add_2_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
Ŕ
,gradients_1/add_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/add_2_grad/Shapegradients_1/add_2_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ŕ
gradients_1/add_2_grad/SumSum/gradients_1/add_3_grad/tuple/control_dependency,gradients_1/add_2_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

gradients_1/add_2_grad/ReshapeReshapegradients_1/add_2_grad/Sumgradients_1/add_2_grad/Shape*
Tshape0*
T0*
_output_shapes
: 
Ä
gradients_1/add_2_grad/Sum_1Sum/gradients_1/add_3_grad/tuple/control_dependency.gradients_1/add_2_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

 gradients_1/add_2_grad/Reshape_1Reshapegradients_1/add_2_grad/Sum_1gradients_1/add_2_grad/Shape_1*
Tshape0*
T0*
_output_shapes
: 
s
'gradients_1/add_2_grad/tuple/group_depsNoOp^gradients_1/add_2_grad/Reshape!^gradients_1/add_2_grad/Reshape_1
Ů
/gradients_1/add_2_grad/tuple/control_dependencyIdentitygradients_1/add_2_grad/Reshape(^gradients_1/add_2_grad/tuple/group_deps*1
_class'
%#loc:@gradients_1/add_2_grad/Reshape*
T0*
_output_shapes
: 
ß
1gradients_1/add_2_grad/tuple/control_dependency_1Identity gradients_1/add_2_grad/Reshape_1(^gradients_1/add_2_grad/tuple/group_deps*3
_class)
'%loc:@gradients_1/add_2_grad/Reshape_1*
T0*
_output_shapes
: 

gradients_1/L2Loss_2_grad/mulMulVariable_6/read1gradients_1/add_3_grad/tuple/control_dependency_1*
T0*
_output_shapes
:	
m
gradients_1/add_1_grad/ShapeConst*
valueB"      *
dtype0*
_output_shapes
:
h
gradients_1/add_1_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
Ŕ
,gradients_1/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/add_1_grad/Shapegradients_1/add_1_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
´
gradients_1/add_1_grad/SumSum#gradients_1/Reshape_11_grad/Reshape,gradients_1/add_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

gradients_1/add_1_grad/ReshapeReshapegradients_1/add_1_grad/Sumgradients_1/add_1_grad/Shape*
Tshape0*
T0*
_output_shapes

:
¸
gradients_1/add_1_grad/Sum_1Sum#gradients_1/Reshape_11_grad/Reshape.gradients_1/add_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

 gradients_1/add_1_grad/Reshape_1Reshapegradients_1/add_1_grad/Sum_1gradients_1/add_1_grad/Shape_1*
Tshape0*
T0*
_output_shapes
:
s
'gradients_1/add_1_grad/tuple/group_depsNoOp^gradients_1/add_1_grad/Reshape!^gradients_1/add_1_grad/Reshape_1
á
/gradients_1/add_1_grad/tuple/control_dependencyIdentitygradients_1/add_1_grad/Reshape(^gradients_1/add_1_grad/tuple/group_deps*1
_class'
%#loc:@gradients_1/add_1_grad/Reshape*
T0*
_output_shapes

:
ă
1gradients_1/add_1_grad/tuple/control_dependency_1Identity gradients_1/add_1_grad/Reshape_1(^gradients_1/add_1_grad/tuple/group_deps*3
_class)
'%loc:@gradients_1/add_1_grad/Reshape_1*
T0*
_output_shapes
:

gradients_1/L2Loss_grad/mulMulVariable_4/read/gradients_1/add_2_grad/tuple/control_dependency*
T0* 
_output_shapes
:


gradients_1/L2Loss_1_grad/mulMulVariable_5/read1gradients_1/add_2_grad/tuple/control_dependency_1*
T0*
_output_shapes	
:
ź
 gradients_1/MatMul_1_grad/MatMulMatMul/gradients_1/add_1_grad/tuple/control_dependencyVariable_6/read*
transpose_b(*
transpose_a( *
T0*
_output_shapes
:	
ľ
"gradients_1/MatMul_1_grad/MatMul_1MatMulRelu_2/gradients_1/add_1_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
T0*
_output_shapes
:	
z
*gradients_1/MatMul_1_grad/tuple/group_depsNoOp!^gradients_1/MatMul_1_grad/MatMul#^gradients_1/MatMul_1_grad/MatMul_1
ě
2gradients_1/MatMul_1_grad/tuple/control_dependencyIdentity gradients_1/MatMul_1_grad/MatMul+^gradients_1/MatMul_1_grad/tuple/group_deps*3
_class)
'%loc:@gradients_1/MatMul_1_grad/MatMul*
T0*
_output_shapes
:	
ň
4gradients_1/MatMul_1_grad/tuple/control_dependency_1Identity"gradients_1/MatMul_1_grad/MatMul_1+^gradients_1/MatMul_1_grad/tuple/group_deps*5
_class+
)'loc:@gradients_1/MatMul_1_grad/MatMul_1*
T0*
_output_shapes
:	
Ę
gradients_1/AddNAddNgradients_1/L2Loss_3_grad/mul1gradients_1/add_1_grad/tuple/control_dependency_1*0
_class&
$"loc:@gradients_1/L2Loss_3_grad/mul*
N*
T0*
_output_shapes
:

 gradients_1/Relu_2_grad/ReluGradReluGrad2gradients_1/MatMul_1_grad/tuple/control_dependencyRelu_2*
T0*
_output_shapes
:	
Ô
gradients_1/AddN_1AddNgradients_1/L2Loss_2_grad/mul4gradients_1/MatMul_1_grad/tuple/control_dependency_1*0
_class&
$"loc:@gradients_1/L2Loss_2_grad/mul*
N*
T0*
_output_shapes
:	
k
gradients_1/add_grad/ShapeConst*
valueB"      *
dtype0*
_output_shapes
:
g
gradients_1/add_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
ş
*gradients_1/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/add_grad/Shapegradients_1/add_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
­
gradients_1/add_grad/SumSum gradients_1/Relu_2_grad/ReluGrad*gradients_1/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

gradients_1/add_grad/ReshapeReshapegradients_1/add_grad/Sumgradients_1/add_grad/Shape*
Tshape0*
T0*
_output_shapes
:	
ą
gradients_1/add_grad/Sum_1Sum gradients_1/Relu_2_grad/ReluGrad,gradients_1/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

gradients_1/add_grad/Reshape_1Reshapegradients_1/add_grad/Sum_1gradients_1/add_grad/Shape_1*
Tshape0*
T0*
_output_shapes	
:
m
%gradients_1/add_grad/tuple/group_depsNoOp^gradients_1/add_grad/Reshape^gradients_1/add_grad/Reshape_1
Ú
-gradients_1/add_grad/tuple/control_dependencyIdentitygradients_1/add_grad/Reshape&^gradients_1/add_grad/tuple/group_deps*/
_class%
#!loc:@gradients_1/add_grad/Reshape*
T0*
_output_shapes
:	
Ü
/gradients_1/add_grad/tuple/control_dependency_1Identitygradients_1/add_grad/Reshape_1&^gradients_1/add_grad/tuple/group_deps*1
_class'
%#loc:@gradients_1/add_grad/Reshape_1*
T0*
_output_shapes	
:
¸
gradients_1/MatMul_grad/MatMulMatMul-gradients_1/add_grad/tuple/control_dependencyVariable_4/read*
transpose_b(*
transpose_a( *
T0*
_output_shapes
:	
ł
 gradients_1/MatMul_grad/MatMul_1MatMulReshape-gradients_1/add_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
T0* 
_output_shapes
:

t
(gradients_1/MatMul_grad/tuple/group_depsNoOp^gradients_1/MatMul_grad/MatMul!^gradients_1/MatMul_grad/MatMul_1
ä
0gradients_1/MatMul_grad/tuple/control_dependencyIdentitygradients_1/MatMul_grad/MatMul)^gradients_1/MatMul_grad/tuple/group_deps*1
_class'
%#loc:@gradients_1/MatMul_grad/MatMul*
T0*
_output_shapes
:	
ë
2gradients_1/MatMul_grad/tuple/control_dependency_1Identity gradients_1/MatMul_grad/MatMul_1)^gradients_1/MatMul_grad/tuple/group_deps*3
_class)
'%loc:@gradients_1/MatMul_grad/MatMul_1*
T0* 
_output_shapes
:

Ë
gradients_1/AddN_2AddNgradients_1/L2Loss_1_grad/mul/gradients_1/add_grad/tuple/control_dependency_1*0
_class&
$"loc:@gradients_1/L2Loss_1_grad/mul*
N*
T0*
_output_shapes	
:
w
gradients_1/Reshape_grad/ShapeConst*%
valueB"         @   *
dtype0*
_output_shapes
:
ź
 gradients_1/Reshape_grad/ReshapeReshape0gradients_1/MatMul_grad/tuple/control_dependencygradients_1/Reshape_grad/Shape*
Tshape0*
T0*&
_output_shapes
:@
Ď
gradients_1/AddN_3AddNgradients_1/L2Loss_grad/mul2gradients_1/MatMul_grad/tuple/control_dependency_1*.
_class$
" loc:@gradients_1/L2Loss_grad/mul*
N*
T0* 
_output_shapes
:

í
&gradients_1/MaxPool_1_grad/MaxPoolGradMaxPoolGradRelu_1	MaxPool_1 gradients_1/Reshape_grad/Reshape*
ksize
*
paddingSAME*
T0*&
_output_shapes
:@*
strides
*
data_formatNHWC

 gradients_1/Relu_1_grad/ReluGradReluGrad&gradients_1/MaxPool_1_grad/MaxPoolGradRelu_1*
T0*&
_output_shapes
:@

&gradients_1/BiasAdd_1_grad/BiasAddGradBiasAddGrad gradients_1/Relu_1_grad/ReluGrad*
data_formatNHWC*
T0*
_output_shapes
:@

+gradients_1/BiasAdd_1_grad/tuple/group_depsNoOp!^gradients_1/Relu_1_grad/ReluGrad'^gradients_1/BiasAdd_1_grad/BiasAddGrad
ő
3gradients_1/BiasAdd_1_grad/tuple/control_dependencyIdentity gradients_1/Relu_1_grad/ReluGrad,^gradients_1/BiasAdd_1_grad/tuple/group_deps*3
_class)
'%loc:@gradients_1/Relu_1_grad/ReluGrad*
T0*&
_output_shapes
:@
÷
5gradients_1/BiasAdd_1_grad/tuple/control_dependency_1Identity&gradients_1/BiasAdd_1_grad/BiasAddGrad,^gradients_1/BiasAdd_1_grad/tuple/group_deps*9
_class/
-+loc:@gradients_1/BiasAdd_1_grad/BiasAddGrad*
T0*
_output_shapes
:@

 gradients_1/Conv2D_1_grad/ShapeNShapeNMaxPoolVariable_2/read*
N*
out_type0*
T0* 
_output_shapes
::
Ř
-gradients_1/Conv2D_1_grad/Conv2DBackpropInputConv2DBackpropInput gradients_1/Conv2D_1_grad/ShapeNVariable_2/read3gradients_1/BiasAdd_1_grad/tuple/control_dependency*
paddingSAME*
use_cudnn_on_gpu(*
T0*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
strides
*
data_formatNHWC
Ô
.gradients_1/Conv2D_1_grad/Conv2DBackpropFilterConv2DBackpropFilterMaxPool"gradients_1/Conv2D_1_grad/ShapeN:13gradients_1/BiasAdd_1_grad/tuple/control_dependency*
paddingSAME*
use_cudnn_on_gpu(*
T0*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
strides
*
data_formatNHWC

*gradients_1/Conv2D_1_grad/tuple/group_depsNoOp.^gradients_1/Conv2D_1_grad/Conv2DBackpropInput/^gradients_1/Conv2D_1_grad/Conv2DBackpropFilter

2gradients_1/Conv2D_1_grad/tuple/control_dependencyIdentity-gradients_1/Conv2D_1_grad/Conv2DBackpropInput+^gradients_1/Conv2D_1_grad/tuple/group_deps*@
_class6
42loc:@gradients_1/Conv2D_1_grad/Conv2DBackpropInput*
T0*&
_output_shapes
: 

4gradients_1/Conv2D_1_grad/tuple/control_dependency_1Identity.gradients_1/Conv2D_1_grad/Conv2DBackpropFilter+^gradients_1/Conv2D_1_grad/tuple/group_deps*A
_class7
53loc:@gradients_1/Conv2D_1_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
: @
ů
$gradients_1/MaxPool_grad/MaxPoolGradMaxPoolGradReluMaxPool2gradients_1/Conv2D_1_grad/tuple/control_dependency*
ksize
*
paddingSAME*
T0*&
_output_shapes
: *
strides
*
data_formatNHWC

gradients_1/Relu_grad/ReluGradReluGrad$gradients_1/MaxPool_grad/MaxPoolGradRelu*
T0*&
_output_shapes
: 

$gradients_1/BiasAdd_grad/BiasAddGradBiasAddGradgradients_1/Relu_grad/ReluGrad*
data_formatNHWC*
T0*
_output_shapes
: 
y
)gradients_1/BiasAdd_grad/tuple/group_depsNoOp^gradients_1/Relu_grad/ReluGrad%^gradients_1/BiasAdd_grad/BiasAddGrad
í
1gradients_1/BiasAdd_grad/tuple/control_dependencyIdentitygradients_1/Relu_grad/ReluGrad*^gradients_1/BiasAdd_grad/tuple/group_deps*1
_class'
%#loc:@gradients_1/Relu_grad/ReluGrad*
T0*&
_output_shapes
: 
ď
3gradients_1/BiasAdd_grad/tuple/control_dependency_1Identity$gradients_1/BiasAdd_grad/BiasAddGrad*^gradients_1/BiasAdd_grad/tuple/group_deps*7
_class-
+)loc:@gradients_1/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
: 

gradients_1/Conv2D_grad/ShapeNShapeNPlaceholderVariable/read*
N*
out_type0*
T0* 
_output_shapes
::
Đ
+gradients_1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInputgradients_1/Conv2D_grad/ShapeNVariable/read1gradients_1/BiasAdd_grad/tuple/control_dependency*
paddingSAME*
use_cudnn_on_gpu(*
T0*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
strides
*
data_formatNHWC
Ň
,gradients_1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterPlaceholder gradients_1/Conv2D_grad/ShapeN:11gradients_1/BiasAdd_grad/tuple/control_dependency*
paddingSAME*
use_cudnn_on_gpu(*
T0*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
strides
*
data_formatNHWC

(gradients_1/Conv2D_grad/tuple/group_depsNoOp,^gradients_1/Conv2D_grad/Conv2DBackpropInput-^gradients_1/Conv2D_grad/Conv2DBackpropFilter

0gradients_1/Conv2D_grad/tuple/control_dependencyIdentity+gradients_1/Conv2D_grad/Conv2DBackpropInput)^gradients_1/Conv2D_grad/tuple/group_deps*>
_class4
20loc:@gradients_1/Conv2D_grad/Conv2DBackpropInput*
T0*&
_output_shapes
:

2gradients_1/Conv2D_grad/tuple/control_dependency_1Identity,gradients_1/Conv2D_grad/Conv2DBackpropFilter)^gradients_1/Conv2D_grad/tuple/group_deps*?
_class5
31loc:@gradients_1/Conv2D_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
: 
Ľ
#Variable/Momentum/Initializer/zerosConst*%
valueB *    *
dtype0*
_class
loc:@Variable*&
_output_shapes
: 
˛
Variable/Momentum
VariableV2*
_class
loc:@Variable*
shared_name *
	container *
dtype0*
shape: *&
_output_shapes
: 
Ń
Variable/Momentum/AssignAssignVariable/Momentum#Variable/Momentum/Initializer/zeros*
use_locking(*
_class
loc:@Variable*
T0*
validate_shape(*&
_output_shapes
: 

Variable/Momentum/readIdentityVariable/Momentum*
_class
loc:@Variable*
T0*&
_output_shapes
: 

%Variable_1/Momentum/Initializer/zerosConst*
valueB *    *
dtype0*
_class
loc:@Variable_1*
_output_shapes
: 

Variable_1/Momentum
VariableV2*
_class
loc:@Variable_1*
shared_name *
	container *
dtype0*
shape: *
_output_shapes
: 
Í
Variable_1/Momentum/AssignAssignVariable_1/Momentum%Variable_1/Momentum/Initializer/zeros*
use_locking(*
_class
loc:@Variable_1*
T0*
validate_shape(*
_output_shapes
: 
}
Variable_1/Momentum/readIdentityVariable_1/Momentum*
_class
loc:@Variable_1*
T0*
_output_shapes
: 
Š
%Variable_2/Momentum/Initializer/zerosConst*%
valueB @*    *
dtype0*
_class
loc:@Variable_2*&
_output_shapes
: @
ś
Variable_2/Momentum
VariableV2*
_class
loc:@Variable_2*
shared_name *
	container *
dtype0*
shape: @*&
_output_shapes
: @
Ů
Variable_2/Momentum/AssignAssignVariable_2/Momentum%Variable_2/Momentum/Initializer/zeros*
use_locking(*
_class
loc:@Variable_2*
T0*
validate_shape(*&
_output_shapes
: @

Variable_2/Momentum/readIdentityVariable_2/Momentum*
_class
loc:@Variable_2*
T0*&
_output_shapes
: @

%Variable_3/Momentum/Initializer/zerosConst*
valueB@*    *
dtype0*
_class
loc:@Variable_3*
_output_shapes
:@

Variable_3/Momentum
VariableV2*
_class
loc:@Variable_3*
shared_name *
	container *
dtype0*
shape:@*
_output_shapes
:@
Í
Variable_3/Momentum/AssignAssignVariable_3/Momentum%Variable_3/Momentum/Initializer/zeros*
use_locking(*
_class
loc:@Variable_3*
T0*
validate_shape(*
_output_shapes
:@
}
Variable_3/Momentum/readIdentityVariable_3/Momentum*
_class
loc:@Variable_3*
T0*
_output_shapes
:@

%Variable_4/Momentum/Initializer/zerosConst*
valueB
*    *
dtype0*
_class
loc:@Variable_4* 
_output_shapes
:

Ş
Variable_4/Momentum
VariableV2*
_class
loc:@Variable_4*
shared_name *
	container *
dtype0*
shape:
* 
_output_shapes
:

Ó
Variable_4/Momentum/AssignAssignVariable_4/Momentum%Variable_4/Momentum/Initializer/zeros*
use_locking(*
_class
loc:@Variable_4*
T0*
validate_shape(* 
_output_shapes
:


Variable_4/Momentum/readIdentityVariable_4/Momentum*
_class
loc:@Variable_4*
T0* 
_output_shapes
:


%Variable_5/Momentum/Initializer/zerosConst*
valueB*    *
dtype0*
_class
loc:@Variable_5*
_output_shapes	
:
 
Variable_5/Momentum
VariableV2*
_class
loc:@Variable_5*
shared_name *
	container *
dtype0*
shape:*
_output_shapes	
:
Î
Variable_5/Momentum/AssignAssignVariable_5/Momentum%Variable_5/Momentum/Initializer/zeros*
use_locking(*
_class
loc:@Variable_5*
T0*
validate_shape(*
_output_shapes	
:
~
Variable_5/Momentum/readIdentityVariable_5/Momentum*
_class
loc:@Variable_5*
T0*
_output_shapes	
:

%Variable_6/Momentum/Initializer/zerosConst*
valueB	*    *
dtype0*
_class
loc:@Variable_6*
_output_shapes
:	
¨
Variable_6/Momentum
VariableV2*
_class
loc:@Variable_6*
shared_name *
	container *
dtype0*
shape:	*
_output_shapes
:	
Ň
Variable_6/Momentum/AssignAssignVariable_6/Momentum%Variable_6/Momentum/Initializer/zeros*
use_locking(*
_class
loc:@Variable_6*
T0*
validate_shape(*
_output_shapes
:	

Variable_6/Momentum/readIdentityVariable_6/Momentum*
_class
loc:@Variable_6*
T0*
_output_shapes
:	

%Variable_7/Momentum/Initializer/zerosConst*
valueB*    *
dtype0*
_class
loc:@Variable_7*
_output_shapes
:

Variable_7/Momentum
VariableV2*
_class
loc:@Variable_7*
shared_name *
	container *
dtype0*
shape:*
_output_shapes
:
Í
Variable_7/Momentum/AssignAssignVariable_7/Momentum%Variable_7/Momentum/Initializer/zeros*
use_locking(*
_class
loc:@Variable_7*
T0*
validate_shape(*
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
Momentum/momentumConst*
valueB
 *    *
dtype0*
_output_shapes
: 
˘
&Momentum/update_Variable/ApplyMomentumApplyMomentumVariableVariable/MomentumExponentialDecay2gradients_1/Conv2D_grad/tuple/control_dependency_1Momentum/momentum*
use_locking( *
_class
loc:@Variable*
use_nesterov( *
T0*&
_output_shapes
: 

(Momentum/update_Variable_1/ApplyMomentumApplyMomentum
Variable_1Variable_1/MomentumExponentialDecay3gradients_1/BiasAdd_grad/tuple/control_dependency_1Momentum/momentum*
use_locking( *
_class
loc:@Variable_1*
use_nesterov( *
T0*
_output_shapes
: 
Ź
(Momentum/update_Variable_2/ApplyMomentumApplyMomentum
Variable_2Variable_2/MomentumExponentialDecay4gradients_1/Conv2D_1_grad/tuple/control_dependency_1Momentum/momentum*
use_locking( *
_class
loc:@Variable_2*
use_nesterov( *
T0*&
_output_shapes
: @
Ą
(Momentum/update_Variable_3/ApplyMomentumApplyMomentum
Variable_3Variable_3/MomentumExponentialDecay5gradients_1/BiasAdd_1_grad/tuple/control_dependency_1Momentum/momentum*
use_locking( *
_class
loc:@Variable_3*
use_nesterov( *
T0*
_output_shapes
:@

(Momentum/update_Variable_4/ApplyMomentumApplyMomentum
Variable_4Variable_4/MomentumExponentialDecaygradients_1/AddN_3Momentum/momentum*
use_locking( *
_class
loc:@Variable_4*
use_nesterov( *
T0* 
_output_shapes
:

˙
(Momentum/update_Variable_5/ApplyMomentumApplyMomentum
Variable_5Variable_5/MomentumExponentialDecaygradients_1/AddN_2Momentum/momentum*
use_locking( *
_class
loc:@Variable_5*
use_nesterov( *
T0*
_output_shapes	
:

(Momentum/update_Variable_6/ApplyMomentumApplyMomentum
Variable_6Variable_6/MomentumExponentialDecaygradients_1/AddN_1Momentum/momentum*
use_locking( *
_class
loc:@Variable_6*
use_nesterov( *
T0*
_output_shapes
:	
ü
(Momentum/update_Variable_7/ApplyMomentumApplyMomentum
Variable_7Variable_7/MomentumExponentialDecaygradients_1/AddNMomentum/momentum*
use_locking( *
_class
loc:@Variable_7*
use_nesterov( *
T0*
_output_shapes
:
í
Momentum/updateNoOp'^Momentum/update_Variable/ApplyMomentum)^Momentum/update_Variable_1/ApplyMomentum)^Momentum/update_Variable_2/ApplyMomentum)^Momentum/update_Variable_3/ApplyMomentum)^Momentum/update_Variable_4/ApplyMomentum)^Momentum/update_Variable_5/ApplyMomentum)^Momentum/update_Variable_6/ApplyMomentum)^Momentum/update_Variable_7/ApplyMomentum

Momentum/valueConst^Momentum/update*
value	B :*
dtype0*
_class
loc:@Variable_8*
_output_shapes
: 

Momentum	AssignAdd
Variable_8Momentum/value*
use_locking( *
_class
loc:@Variable_8*
T0*
_output_shapes
: 
B
SoftmaxSoftmaxadd_1*
T0*
_output_shapes

:
ş
Conv2D_2Conv2DPlaceholder_2Variable/read*
paddingSAME*
use_cudnn_on_gpu(*
T0*(
_output_shapes
:Ŕ= *
strides
*
data_formatNHWC
y
	BiasAdd_2BiasAddConv2D_2Variable_1/read*
data_formatNHWC*
T0*(
_output_shapes
:Ŕ= 
L
Relu_3Relu	BiasAdd_2*
T0*(
_output_shapes
:Ŕ= 
Ą
	MaxPool_2MaxPoolRelu_3*
ksize
*
paddingSAME*
T0*(
_output_shapes
:Ŕ= *
strides
*
data_formatNHWC
¸
Conv2D_3Conv2D	MaxPool_2Variable_2/read*
paddingSAME*
use_cudnn_on_gpu(*
T0*(
_output_shapes
:Ŕ=@*
strides
*
data_formatNHWC
y
	BiasAdd_3BiasAddConv2D_3Variable_3/read*
data_formatNHWC*
T0*(
_output_shapes
:Ŕ=@
L
Relu_4Relu	BiasAdd_3*
T0*(
_output_shapes
:Ŕ=@
Ą
	MaxPool_3MaxPoolRelu_4*
ksize
*
paddingSAME*
T0*(
_output_shapes
:Ŕ=@*
strides
*
data_formatNHWC
a
Reshape_14/shapeConst*
valueB"@B    *
dtype0*
_output_shapes
:
l

Reshape_14Reshape	MaxPool_3Reshape_14/shape*
Tshape0*
T0*!
_output_shapes
:Ŕ=

MatMul_2MatMul
Reshape_14Variable_4/read*
transpose_b( *
transpose_a( *
T0*!
_output_shapes
:Ŕ=
S
add_6AddMatMul_2Variable_5/read*
T0*!
_output_shapes
:Ŕ=
A
Relu_5Reluadd_6*
T0*!
_output_shapes
:Ŕ=
|
MatMul_3MatMulRelu_5Variable_6/read*
transpose_b( *
transpose_a( *
T0* 
_output_shapes
:
Ŕ=
R
add_7AddMatMul_3Variable_7/read*
T0* 
_output_shapes
:
Ŕ=
F
	Softmax_1Softmaxadd_7*
T0* 
_output_shapes
:
Ŕ=
P

save/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
ň
save/SaveV2/tensor_namesConst*Ľ
valueBBVariableBVariable/MomentumB
Variable_1BVariable_1/MomentumB
Variable_2BVariable_2/MomentumB
Variable_3BVariable_3/MomentumB
Variable_4BVariable_4/MomentumB
Variable_5BVariable_5/MomentumB
Variable_6BVariable_6/MomentumB
Variable_7BVariable_7/MomentumB
Variable_8*
dtype0*
_output_shapes
:

save/SaveV2/shape_and_slicesConst*5
value,B*B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:

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
valueBBVariable*
dtype0*
_output_shapes
:
h
save/RestoreV2/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*
dtypes
2*
_output_shapes
:
Ś
save/AssignAssignVariablesave/RestoreV2*
use_locking(*
_class
loc:@Variable*
T0*
validate_shape(*&
_output_shapes
: 
w
save/RestoreV2_1/tensor_namesConst*&
valueBBVariable/Momentum*
dtype0*
_output_shapes
:
j
!save/RestoreV2_1/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_1	RestoreV2
save/Constsave/RestoreV2_1/tensor_names!save/RestoreV2_1/shape_and_slices*
dtypes
2*
_output_shapes
:
ł
save/Assign_1AssignVariable/Momentumsave/RestoreV2_1*
use_locking(*
_class
loc:@Variable*
T0*
validate_shape(*&
_output_shapes
: 
p
save/RestoreV2_2/tensor_namesConst*
valueBB
Variable_1*
dtype0*
_output_shapes
:
j
!save/RestoreV2_2/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_2	RestoreV2
save/Constsave/RestoreV2_2/tensor_names!save/RestoreV2_2/shape_and_slices*
dtypes
2*
_output_shapes
:
˘
save/Assign_2Assign
Variable_1save/RestoreV2_2*
use_locking(*
_class
loc:@Variable_1*
T0*
validate_shape(*
_output_shapes
: 
y
save/RestoreV2_3/tensor_namesConst*(
valueBBVariable_1/Momentum*
dtype0*
_output_shapes
:
j
!save/RestoreV2_3/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_3	RestoreV2
save/Constsave/RestoreV2_3/tensor_names!save/RestoreV2_3/shape_and_slices*
dtypes
2*
_output_shapes
:
Ť
save/Assign_3AssignVariable_1/Momentumsave/RestoreV2_3*
use_locking(*
_class
loc:@Variable_1*
T0*
validate_shape(*
_output_shapes
: 
p
save/RestoreV2_4/tensor_namesConst*
valueBB
Variable_2*
dtype0*
_output_shapes
:
j
!save/RestoreV2_4/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_4	RestoreV2
save/Constsave/RestoreV2_4/tensor_names!save/RestoreV2_4/shape_and_slices*
dtypes
2*
_output_shapes
:
Ž
save/Assign_4Assign
Variable_2save/RestoreV2_4*
use_locking(*
_class
loc:@Variable_2*
T0*
validate_shape(*&
_output_shapes
: @
y
save/RestoreV2_5/tensor_namesConst*(
valueBBVariable_2/Momentum*
dtype0*
_output_shapes
:
j
!save/RestoreV2_5/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_5	RestoreV2
save/Constsave/RestoreV2_5/tensor_names!save/RestoreV2_5/shape_and_slices*
dtypes
2*
_output_shapes
:
ˇ
save/Assign_5AssignVariable_2/Momentumsave/RestoreV2_5*
use_locking(*
_class
loc:@Variable_2*
T0*
validate_shape(*&
_output_shapes
: @
p
save/RestoreV2_6/tensor_namesConst*
valueBB
Variable_3*
dtype0*
_output_shapes
:
j
!save/RestoreV2_6/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_6	RestoreV2
save/Constsave/RestoreV2_6/tensor_names!save/RestoreV2_6/shape_and_slices*
dtypes
2*
_output_shapes
:
˘
save/Assign_6Assign
Variable_3save/RestoreV2_6*
use_locking(*
_class
loc:@Variable_3*
T0*
validate_shape(*
_output_shapes
:@
y
save/RestoreV2_7/tensor_namesConst*(
valueBBVariable_3/Momentum*
dtype0*
_output_shapes
:
j
!save/RestoreV2_7/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_7	RestoreV2
save/Constsave/RestoreV2_7/tensor_names!save/RestoreV2_7/shape_and_slices*
dtypes
2*
_output_shapes
:
Ť
save/Assign_7AssignVariable_3/Momentumsave/RestoreV2_7*
use_locking(*
_class
loc:@Variable_3*
T0*
validate_shape(*
_output_shapes
:@
p
save/RestoreV2_8/tensor_namesConst*
valueBB
Variable_4*
dtype0*
_output_shapes
:
j
!save/RestoreV2_8/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_8	RestoreV2
save/Constsave/RestoreV2_8/tensor_names!save/RestoreV2_8/shape_and_slices*
dtypes
2*
_output_shapes
:
¨
save/Assign_8Assign
Variable_4save/RestoreV2_8*
use_locking(*
_class
loc:@Variable_4*
T0*
validate_shape(* 
_output_shapes
:

y
save/RestoreV2_9/tensor_namesConst*(
valueBBVariable_4/Momentum*
dtype0*
_output_shapes
:
j
!save/RestoreV2_9/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_9	RestoreV2
save/Constsave/RestoreV2_9/tensor_names!save/RestoreV2_9/shape_and_slices*
dtypes
2*
_output_shapes
:
ą
save/Assign_9AssignVariable_4/Momentumsave/RestoreV2_9*
use_locking(*
_class
loc:@Variable_4*
T0*
validate_shape(* 
_output_shapes
:

q
save/RestoreV2_10/tensor_namesConst*
valueBB
Variable_5*
dtype0*
_output_shapes
:
k
"save/RestoreV2_10/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_10	RestoreV2
save/Constsave/RestoreV2_10/tensor_names"save/RestoreV2_10/shape_and_slices*
dtypes
2*
_output_shapes
:
Ľ
save/Assign_10Assign
Variable_5save/RestoreV2_10*
use_locking(*
_class
loc:@Variable_5*
T0*
validate_shape(*
_output_shapes	
:
z
save/RestoreV2_11/tensor_namesConst*(
valueBBVariable_5/Momentum*
dtype0*
_output_shapes
:
k
"save/RestoreV2_11/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_11	RestoreV2
save/Constsave/RestoreV2_11/tensor_names"save/RestoreV2_11/shape_and_slices*
dtypes
2*
_output_shapes
:
Ž
save/Assign_11AssignVariable_5/Momentumsave/RestoreV2_11*
use_locking(*
_class
loc:@Variable_5*
T0*
validate_shape(*
_output_shapes	
:
q
save/RestoreV2_12/tensor_namesConst*
valueBB
Variable_6*
dtype0*
_output_shapes
:
k
"save/RestoreV2_12/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_12	RestoreV2
save/Constsave/RestoreV2_12/tensor_names"save/RestoreV2_12/shape_and_slices*
dtypes
2*
_output_shapes
:
Š
save/Assign_12Assign
Variable_6save/RestoreV2_12*
use_locking(*
_class
loc:@Variable_6*
T0*
validate_shape(*
_output_shapes
:	
z
save/RestoreV2_13/tensor_namesConst*(
valueBBVariable_6/Momentum*
dtype0*
_output_shapes
:
k
"save/RestoreV2_13/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_13	RestoreV2
save/Constsave/RestoreV2_13/tensor_names"save/RestoreV2_13/shape_and_slices*
dtypes
2*
_output_shapes
:
˛
save/Assign_13AssignVariable_6/Momentumsave/RestoreV2_13*
use_locking(*
_class
loc:@Variable_6*
T0*
validate_shape(*
_output_shapes
:	
q
save/RestoreV2_14/tensor_namesConst*
valueBB
Variable_7*
dtype0*
_output_shapes
:
k
"save/RestoreV2_14/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_14	RestoreV2
save/Constsave/RestoreV2_14/tensor_names"save/RestoreV2_14/shape_and_slices*
dtypes
2*
_output_shapes
:
¤
save/Assign_14Assign
Variable_7save/RestoreV2_14*
use_locking(*
_class
loc:@Variable_7*
T0*
validate_shape(*
_output_shapes
:
z
save/RestoreV2_15/tensor_namesConst*(
valueBBVariable_7/Momentum*
dtype0*
_output_shapes
:
k
"save/RestoreV2_15/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_15	RestoreV2
save/Constsave/RestoreV2_15/tensor_names"save/RestoreV2_15/shape_and_slices*
dtypes
2*
_output_shapes
:
­
save/Assign_15AssignVariable_7/Momentumsave/RestoreV2_15*
use_locking(*
_class
loc:@Variable_7*
T0*
validate_shape(*
_output_shapes
:
q
save/RestoreV2_16/tensor_namesConst*
valueBB
Variable_8*
dtype0*
_output_shapes
:
k
"save/RestoreV2_16/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_16	RestoreV2
save/Constsave/RestoreV2_16/tensor_names"save/RestoreV2_16/shape_and_slices*
dtypes
2*
_output_shapes
:
 
save/Assign_16Assign
Variable_8save/RestoreV2_16*
use_locking(*
_class
loc:@Variable_8*
T0*
validate_shape(*
_output_shapes
: 
­
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16
¤
initNoOp^Variable/Assign^Variable_1/Assign^Variable_2/Assign^Variable_3/Assign^Variable_4/Assign^Variable_5/Assign^Variable_6/Assign^Variable_7/Assign^Variable_8/Assign^Variable/Momentum/Assign^Variable_1/Momentum/Assign^Variable_2/Momentum/Assign^Variable_3/Momentum/Assign^Variable_4/Momentum/Assign^Variable_5/Momentum/Assign^Variable_6/Momentum/Assign^Variable_7/Momentum/Assign

Merge/MergeSummaryMergeSummarysummary_data_0summary_conv_0summary_pool_0summary_conv2_0summary_pool2_0lossconv1_weightsconv1_biasesconv2_weightsconv2_biasesfc1_weights
fc1_biasesfc2_weights
fc2_biaseslearning_rate*
N*
_output_shapes
: ""
train_op


Momentum"
trainable_variablesýú
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
Variable_8:0Variable_8/AssignVariable_8/read:02Variable_8/initial_value:0"Ó
	variablesĹÂ
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
Variable_7/Momentum:0Variable_7/Momentum/AssignVariable_7/Momentum/read:02'Variable_7/Momentum/Initializer/zeros:0"
	summariesô
ń
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
learning_rate:0š*oM