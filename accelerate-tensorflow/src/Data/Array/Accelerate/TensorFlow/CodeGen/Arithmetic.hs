{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications    #-}
{-# OPTIONS_HADDOCK hide #-}
-- |
-- Module      : Data.Array.Accelerate.TensorFlow.CodeGen.Arithmetic
-- Copyright   : [2021] The Accelerate Team
-- License     : BSD3
--
-- Maintainer  : Trevor L. McDonell <trevor.mcdonell@gmail.com>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--

module Data.Array.Accelerate.TensorFlow.CodeGen.Arithmetic
  where

import Data.Array.Accelerate.TensorFlow.CodeGen.Base
import Data.Array.Accelerate.TensorFlow.CodeGen.Tensor

import Data.Array.Accelerate.Type
import Data.Array.Accelerate.AST                                    ( PrimBool )

import qualified TensorFlow.GenOps.Core                             as TF
import qualified TensorFlow.Ops                                     as TF
import qualified TensorFlow.Types                                   as TF

import Prelude                                                      hiding ( uncurry, quot, rem, div, mod, log )


uncurry :: (TensorArrayData a -> TensorArrayData b -> c) -> TensorArrayData (a, b) -> c
uncurry f (x, y) = f x y

add :: forall t. NumType t -> TensorArrayData (t, t) -> TensorArrayData t
add = uncurry @t @t . num
  where
    num :: NumType t -> TensorArrayData t -> TensorArrayData t -> TensorArrayData t
    num (IntegralNumType t) = integral t
    num (FloatingNumType t) = floating t

    integral :: IntegralType t -> TensorArrayData t -> TensorArrayData t -> TensorArrayData t
    integral TypeInt    = TF.add
    integral TypeInt8   = TF.add
    integral TypeInt16  = TF.add
    integral TypeInt32  = TF.add
    integral TypeInt64  = TF.add
    integral TypeWord   = TF.add
    integral TypeWord8  = TF.add
    integral TypeWord16 = TF.add
    integral TypeWord32 = TF.add
    integral TypeWord64 = TF.add

    floating :: FloatingType t -> TensorArrayData t -> TensorArrayData t -> TensorArrayData t
    floating TypeFloat  = TF.add
    floating TypeDouble = TF.add
    floating TypeHalf   = unsupported "half-precision floating point"

sub :: forall t. NumType t -> TensorArrayData (t, t) -> TensorArrayData t
sub = uncurry @t @t . num
  where
    num :: NumType t -> TensorArrayData t -> TensorArrayData t -> TensorArrayData t
    num (IntegralNumType t) = integral t
    num (FloatingNumType t) = floating t

    integral :: IntegralType t -> TensorArrayData t -> TensorArrayData t -> TensorArrayData t
    integral TypeInt    = TF.sub
    integral TypeInt8   = TF.sub
    integral TypeInt16  = TF.sub
    integral TypeInt32  = TF.sub
    integral TypeInt64  = TF.sub
    integral TypeWord   = TF.sub
    integral TypeWord8  = TF.sub
    integral TypeWord16 = TF.sub
    integral TypeWord32 = TF.sub
    integral TypeWord64 = TF.sub

    floating :: FloatingType t -> TensorArrayData t -> TensorArrayData t -> TensorArrayData t
    floating TypeFloat  = TF.sub
    floating TypeDouble = TF.sub
    floating TypeHalf   = unsupported "half-precision floating point"

mul :: forall t. NumType t -> TensorArrayData (t, t) -> TensorArrayData t
mul = uncurry @t @t . num
  where
    num :: NumType t -> TensorArrayData t -> TensorArrayData t -> TensorArrayData t
    num (IntegralNumType t) = integral t
    num (FloatingNumType t) = floating t

    integral :: IntegralType t -> TensorArrayData t -> TensorArrayData t -> TensorArrayData t
    integral TypeInt    = TF.mul
    integral TypeInt8   = TF.mul
    integral TypeInt16  = TF.mul
    integral TypeInt32  = TF.mul
    integral TypeInt64  = TF.mul
    integral TypeWord   = TF.mul
    integral TypeWord8  = TF.mul
    integral TypeWord16 = TF.mul
    integral TypeWord32 = TF.mul
    integral TypeWord64 = TF.mul

    floating :: FloatingType t -> TensorArrayData t -> TensorArrayData t -> TensorArrayData t
    floating TypeFloat  = TF.mul
    floating TypeDouble = TF.mul
    floating TypeHalf   = unsupported "half-precision floating point"

neg :: NumType t -> TensorArrayData t -> TensorArrayData t
neg = num
  where
    num :: NumType t -> TensorArrayData t -> TensorArrayData t
    num (IntegralNumType t) = integral t
    num (FloatingNumType t) = floating t

    integral :: IntegralType t -> TensorArrayData t -> TensorArrayData t
    integral TypeInt    = TF.neg
    integral TypeInt8   = TF.neg
    integral TypeInt16  = TF.neg
    integral TypeInt32  = TF.neg
    integral TypeInt64  = TF.neg
    integral TypeWord   = TF.neg
    integral TypeWord8  = id
    integral TypeWord16 = TF.neg
    integral TypeWord32 = TF.neg
    integral TypeWord64 = TF.neg

    floating :: FloatingType t -> TensorArrayData t -> TensorArrayData t
    floating TypeFloat  = TF.neg
    floating TypeDouble = TF.neg
    floating TypeHalf   = unsupported "half-precision floating point"

abs :: NumType t -> TensorArrayData t -> TensorArrayData t
abs = num
  where
    num :: NumType t -> TensorArrayData t -> TensorArrayData t
    num (IntegralNumType t) = integral t
    num (FloatingNumType t) = floating t

    integral :: IntegralType t -> TensorArrayData t -> TensorArrayData t
    integral TypeInt    = TF.abs
    integral TypeInt8   = TF.abs
    integral TypeInt16  = TF.abs
    integral TypeInt32  = TF.abs
    integral TypeInt64  = TF.abs
    integral TypeWord   = TF.abs
    integral TypeWord8  = id
    integral TypeWord16 = TF.abs
    integral TypeWord32 = TF.abs
    integral TypeWord64 = TF.abs

    floating :: FloatingType t -> TensorArrayData t -> TensorArrayData t
    floating TypeFloat  = TF.abs
    floating TypeDouble = TF.abs
    floating TypeHalf   = unsupported "half-precision floating point"

signum :: NumType t -> TensorArrayData t -> TensorArrayData t
signum = num
  where
    num :: NumType t -> TensorArrayData t -> TensorArrayData t
    num (IntegralNumType t) = integral t
    num (FloatingNumType t) = floating t

    integral :: IntegralType t -> TensorArrayData t -> TensorArrayData t
    integral TypeInt    = TF.sign
    integral TypeInt8   = excluded
    integral TypeInt16  = excluded
    integral TypeInt32  = TF.sign
    integral TypeInt64  = TF.sign
    integral TypeWord   = TF.sign
    integral TypeWord8  = id
    integral TypeWord16 = TF.sign
    integral TypeWord32 = TF.sign
    integral TypeWord64 = TF.sign

    floating :: FloatingType t -> TensorArrayData t -> TensorArrayData t
    floating TypeFloat  = TF.sign
    floating TypeDouble = TF.sign
    floating TypeHalf   = unsupported "half-precision floating point"

quot :: forall t. IntegralType t -> TensorArrayData (t, t) -> TensorArrayData t
quot = uncurry @t @t . integral
  where
    integral :: IntegralType t -> TensorArrayData t -> TensorArrayData t -> TensorArrayData t
    integral TypeInt    = TF.div
    integral TypeInt8   = TF.div
    integral TypeInt16  = TF.div
    integral TypeInt32  = TF.div
    integral TypeInt64  = TF.div
    integral TypeWord   = TF.div
    integral TypeWord8  = TF.div
    integral TypeWord16 = TF.div
    integral TypeWord32 = TF.div
    integral TypeWord64 = TF.div

rem :: forall t. IntegralType t -> TensorArrayData (t, t) -> TensorArrayData t
rem = uncurry @t @t . integral
  where
    integral :: IntegralType t -> TensorArrayData t -> TensorArrayData t -> TensorArrayData t
    integral TypeInt    = TF.mod
    integral TypeInt8   = excluded
    integral TypeInt16  = excluded
    integral TypeInt32  = TF.mod
    integral TypeInt64  = TF.mod
    integral TypeWord   = TF.mod
    integral TypeWord8  = excluded
    integral TypeWord16 = TF.mod
    integral TypeWord32 = TF.mod
    integral TypeWord64 = TF.mod

quotRem :: forall t. IntegralType t -> TensorArrayData (t, t) -> TensorArrayData (t, t)
quotRem t xy =
  let q = quot t xy
      r = rem t xy
  in
  (q, r)

-- idiv    :: IntegralType t -> TensorArrayData t -> TensorArrayData t -> TensorArrayData t
-- mod     :: IntegralType t -> TensorArrayData t -> TensorArrayData t -> TensorArrayData t
-- divMod  :: IntegralType t -> TensorArrayData t -> TensorArrayData t -> TensorArrayData (t, t)

band :: forall t. IntegralType t -> TensorArrayData (t, t) -> TensorArrayData t
band = uncurry @t @t . integral
  where
    integral :: IntegralType t -> TensorArrayData t -> TensorArrayData t -> TensorArrayData t
    integral TypeInt    = TF.bitwiseAnd
    integral TypeInt8   = TF.bitwiseAnd
    integral TypeInt16  = TF.bitwiseAnd
    integral TypeInt32  = TF.bitwiseAnd
    integral TypeInt64  = TF.bitwiseAnd
    integral TypeWord   = TF.bitwiseAnd
    integral TypeWord8  = TF.bitwiseAnd
    integral TypeWord16 = TF.bitwiseAnd
    integral TypeWord32 = TF.bitwiseAnd
    integral TypeWord64 = TF.bitwiseAnd

bor :: forall t. IntegralType t -> TensorArrayData (t, t) -> TensorArrayData t
bor = uncurry @t @t . integral
  where
    integral :: IntegralType t -> TensorArrayData t -> TensorArrayData t -> TensorArrayData t
    integral TypeInt    = TF.bitwiseOr
    integral TypeInt8   = TF.bitwiseOr
    integral TypeInt16  = TF.bitwiseOr
    integral TypeInt32  = TF.bitwiseOr
    integral TypeInt64  = TF.bitwiseOr
    integral TypeWord   = TF.bitwiseOr
    integral TypeWord8  = TF.bitwiseOr
    integral TypeWord16 = TF.bitwiseOr
    integral TypeWord32 = TF.bitwiseOr
    integral TypeWord64 = TF.bitwiseOr

xor :: forall t. IntegralType t -> TensorArrayData (t, t) -> TensorArrayData t
xor = uncurry @t @t . integral
  where
    integral :: IntegralType t -> TensorArrayData t -> TensorArrayData t -> TensorArrayData t
    integral TypeInt    = TF.bitwiseXor
    integral TypeInt8   = TF.bitwiseXor
    integral TypeInt16  = TF.bitwiseXor
    integral TypeInt32  = TF.bitwiseXor
    integral TypeInt64  = TF.bitwiseXor
    integral TypeWord   = TF.bitwiseXor
    integral TypeWord8  = TF.bitwiseXor
    integral TypeWord16 = TF.bitwiseXor
    integral TypeWord32 = TF.bitwiseXor
    integral TypeWord64 = TF.bitwiseXor

complement :: IntegralType t -> TensorArrayData t -> TensorArrayData t
complement = integral
  where
    integral :: IntegralType t -> TensorArrayData t -> TensorArrayData t
    integral TypeInt    t = TF.bitwiseXor t (TF.scalar (-1))
    integral TypeInt8   t = TF.bitwiseXor t (TF.scalar (-1))
    integral TypeInt16  t = TF.bitwiseXor t (TF.scalar (-1))
    integral TypeInt32  t = TF.bitwiseXor t (TF.scalar (-1))
    integral TypeInt64  t = TF.bitwiseXor t (TF.scalar (-1))
    integral TypeWord   t = TF.bitwiseXor t (TF.scalar maxBound)
    integral TypeWord8  t = TF.bitwiseXor t (TF.scalar maxBound)
    integral TypeWord16 t = TF.bitwiseXor t (TF.scalar maxBound)
    integral TypeWord32 t = TF.bitwiseXor t (TF.scalar maxBound)
    integral TypeWord64 t = TF.bitwiseXor t (TF.scalar maxBound)

shiftL :: forall t. IntegralType t -> TensorArrayData (t, Int) -> TensorArrayData t
shiftL = uncurry @t @Int . integral
  where
    integral :: IntegralType t -> TensorArrayData t -> TensorArrayData Int -> TensorArrayData t
    integral TypeInt    t x = TF.leftShift t x
    integral TypeInt8   t x = TF.leftShift t (TF.cast x)
    integral TypeInt16  t x = TF.leftShift t (TF.cast x)
    integral TypeInt32  t x = TF.leftShift t (TF.cast x)
    integral TypeInt64  t x = TF.leftShift t (TF.cast x)
    integral TypeWord   t x = TF.leftShift t (TF.cast x)
    integral TypeWord8  t x = TF.leftShift t (TF.cast x)
    integral TypeWord16 t x = TF.leftShift t (TF.cast x)
    integral TypeWord32 t x = TF.leftShift t (TF.cast x)
    integral TypeWord64 t x = TF.leftShift t (TF.cast x)

shiftR  :: forall t. IntegralType t -> TensorArrayData (t, Int) -> TensorArrayData t
shiftR = uncurry @t @Int . integral
  where
    integral :: IntegralType t -> TensorArrayData t -> TensorArrayData Int -> TensorArrayData t
    integral TypeInt    t x = TF.rightShift t x
    integral TypeInt8   t x = TF.rightShift t (TF.cast x)
    integral TypeInt16  t x = TF.rightShift t (TF.cast x)
    integral TypeInt32  t x = TF.rightShift t (TF.cast x)
    integral TypeInt64  t x = TF.rightShift t (TF.cast x)
    integral TypeWord   t x = TF.rightShift t (TF.cast x)
    integral TypeWord8  t x = TF.rightShift t (TF.cast x)
    integral TypeWord16 t x = TF.rightShift t (TF.cast x)
    integral TypeWord32 t x = TF.rightShift t (TF.cast x)
    integral TypeWord64 t x = TF.rightShift t (TF.cast x)

-- rotateL :: IntegralType t -> TensorArrayData t -> TensorArrayData Int -> TensorArrayData t
-- rotateR :: IntegralType t -> TensorArrayData t -> TensorArrayData Int -> TensorArrayData t

popCount :: IntegralType t -> TensorArrayData t -> TensorArrayData Int
popCount = integral
  where
    integral :: IntegralType t -> TensorArrayData t -> TensorArrayData Int
    integral TypeInt    = TF.cast . TF.populationCount
    integral TypeInt8   = TF.cast . TF.populationCount
    integral TypeInt16  = TF.cast . TF.populationCount
    integral TypeInt32  = TF.cast . TF.populationCount
    integral TypeInt64  = TF.cast . TF.populationCount
    integral TypeWord   = TF.cast . TF.populationCount
    integral TypeWord8  = TF.cast . TF.populationCount
    integral TypeWord16 = TF.cast . TF.populationCount
    integral TypeWord32 = TF.cast . TF.populationCount
    integral TypeWord64 = TF.cast . TF.populationCount

-- countLeadingZeros  :: IntegralType t -> TensorArrayData t -> TensorArrayData Int
-- countTrailingZeros :: IntegralType t -> TensorArrayData t -> TensorArrayData Int

fdiv :: forall t. FloatingType t -> TensorArrayData (t, t) -> TensorArrayData t
fdiv = uncurry @t @t . floating
  where
    floating :: FloatingType t -> TensorArrayData t -> TensorArrayData t -> TensorArrayData t
    floating TypeFloat  = TF.div
    floating TypeDouble = TF.div
    floating TypeHalf   = unsupported "half-precision floating point"

recip :: FloatingType t -> TensorArrayData t -> TensorArrayData t
recip = floating
  where
    floating :: FloatingType t -> TensorArrayData t -> TensorArrayData t
    floating TypeFloat  = TF.reciprocal
    floating TypeDouble = TF.reciprocal
    floating TypeHalf   = unsupported "half-precision floating point"

sin :: FloatingType t -> TensorArrayData t -> TensorArrayData t
sin = floating
  where
    floating :: FloatingType t -> TensorArrayData t -> TensorArrayData t
    floating TypeFloat  = TF.sin
    floating TypeDouble = TF.sin
    floating TypeHalf   = unsupported "half-precision floating point"

cos :: FloatingType t -> TensorArrayData t -> TensorArrayData t
cos = floating
  where
    floating :: FloatingType t -> TensorArrayData t -> TensorArrayData t
    floating TypeFloat  = TF.cos
    floating TypeDouble = TF.cos
    floating TypeHalf   = unsupported "half-precision floating point"

tan :: FloatingType t -> TensorArrayData t -> TensorArrayData t
tan = floating
  where
    floating :: FloatingType t -> TensorArrayData t -> TensorArrayData t
    floating TypeFloat  = TF.tan
    floating TypeDouble = TF.tan
    floating TypeHalf   = unsupported "half-precision floating point"

asin :: FloatingType t -> TensorArrayData t -> TensorArrayData t
asin = floating
  where
    floating :: FloatingType t -> TensorArrayData t -> TensorArrayData t
    floating TypeFloat  = TF.asin
    floating TypeDouble = TF.asin
    floating TypeHalf   = unsupported "half-precision floating point"

acos :: FloatingType t -> TensorArrayData t -> TensorArrayData t
acos = floating
  where
    floating :: FloatingType t -> TensorArrayData t -> TensorArrayData t
    floating TypeFloat  = TF.acos
    floating TypeDouble = TF.acos
    floating TypeHalf   = unsupported "half-precision floating point"

atan :: FloatingType t -> TensorArrayData t -> TensorArrayData t
atan = floating
  where
    floating :: FloatingType t -> TensorArrayData t -> TensorArrayData t
    floating TypeFloat  = TF.atan
    floating TypeDouble = TF.atan
    floating TypeHalf   = unsupported "half-precision floating point"

sinh :: FloatingType t -> TensorArrayData t -> TensorArrayData t
sinh = floating
  where
    floating :: FloatingType t -> TensorArrayData t -> TensorArrayData t
    floating TypeFloat  = TF.sinh
    floating TypeDouble = TF.sinh
    floating TypeHalf   = unsupported "half-precision floating point"

cosh :: FloatingType t -> TensorArrayData t -> TensorArrayData t
cosh = floating
  where
    floating :: FloatingType t -> TensorArrayData t -> TensorArrayData t
    floating TypeFloat  = TF.cosh
    floating TypeDouble = TF.cosh
    floating TypeHalf   = unsupported "half-precision floating point"

tanh :: FloatingType t -> TensorArrayData t -> TensorArrayData t
tanh = floating
  where
    floating :: FloatingType t -> TensorArrayData t -> TensorArrayData t
    floating TypeFloat  = TF.tanh
    floating TypeDouble = TF.tanh
    floating TypeHalf   = unsupported "half-precision floating point"

asinh :: FloatingType t -> TensorArrayData t -> TensorArrayData t
asinh = floating
  where
    floating :: FloatingType t -> TensorArrayData t -> TensorArrayData t
    floating TypeFloat  = TF.asinh
    floating TypeDouble = TF.asinh
    floating TypeHalf   = unsupported "half-precision floating point"

acosh :: FloatingType t -> TensorArrayData t -> TensorArrayData t
acosh = floating
  where
    floating :: FloatingType t -> TensorArrayData t -> TensorArrayData t
    floating TypeFloat  = TF.acosh
    floating TypeDouble = TF.acosh
    floating TypeHalf   = unsupported "half-precision floating point"

atanh :: FloatingType t -> TensorArrayData t -> TensorArrayData t
atanh = floating
  where
    floating :: FloatingType t -> TensorArrayData t -> TensorArrayData t
    floating TypeFloat  = TF.atanh
    floating TypeDouble = TF.atanh
    floating TypeHalf   = unsupported "half-precision floating point"

exp :: FloatingType t -> TensorArrayData t -> TensorArrayData t
exp = floating
  where
    floating :: FloatingType t -> TensorArrayData t -> TensorArrayData t
    floating TypeFloat  = TF.exp
    floating TypeDouble = TF.exp
    floating TypeHalf   = unsupported "half-precision floating point"

sqrt :: FloatingType t -> TensorArrayData t -> TensorArrayData t
sqrt = floating
  where
    floating :: FloatingType t -> TensorArrayData t -> TensorArrayData t
    floating TypeFloat  = TF.sqrt
    floating TypeDouble = TF.sqrt
    floating TypeHalf   = unsupported "half-precision floating point"

log :: FloatingType t -> TensorArrayData t -> TensorArrayData t
log = floating
  where
    floating :: FloatingType t -> TensorArrayData t -> TensorArrayData t
    floating TypeFloat  = TF.log
    floating TypeDouble = TF.log
    floating TypeHalf   = unsupported "half-precision floating point"

fpow :: forall t. FloatingType t -> TensorArrayData (t, t) -> TensorArrayData t
fpow = uncurry @t @t . floating
  where
    floating :: FloatingType t -> TensorArrayData t -> TensorArrayData t -> TensorArrayData t
    floating TypeFloat  = TF.pow
    floating TypeDouble = TF.pow
    floating TypeHalf   = unsupported "half-precision floating point"

logBase :: forall t. FloatingType t -> TensorArrayData (t, t) -> TensorArrayData t
logBase = uncurry @t @t . logBase'
  where
    logBase' :: FloatingType t -> TensorArrayData t -> TensorArrayData t -> TensorArrayData t
    logBase' t x y =
      let x' = log t x
          y' = log t y
      in
      fdiv t (y', x')

truncate :: FloatingType a -> IntegralType b -> TensorArrayData a -> TensorArrayData b
truncate = floating
  where
    floating :: FloatingType a -> IntegralType b -> TensorArrayData a -> TensorArrayData b
    floating TypeFloat  = integral
    floating TypeDouble = integral
    floating TypeHalf   = unsupported "half-precision floating point"

    integral :: (ScalarTensorArrayData a, TF.TensorType a) => IntegralType b -> TensorArrayData a -> TensorArrayData b
    integral TypeInt    = TF.cast
    integral TypeInt8   = TF.cast
    integral TypeInt16  = TF.cast
    integral TypeInt32  = TF.cast
    integral TypeInt64  = TF.cast
    integral TypeWord   = TF.cast
    integral TypeWord8  = TF.cast
    integral TypeWord16 = TF.cast
    integral TypeWord32 = TF.cast
    integral TypeWord64 = TF.cast

round :: FloatingType a -> IntegralType b -> TensorArrayData a -> TensorArrayData b
round = floating
  where
    floating :: FloatingType a -> IntegralType b -> TensorArrayData a -> TensorArrayData b
    floating TypeFloat  t = integral t . TF.round
    floating TypeDouble t = integral t . TF.round
    floating TypeHalf   _ = unsupported "half-precision floating point"

    integral :: (ScalarTensorArrayData a, TF.TensorType a) => IntegralType b -> TensorArrayData a -> TensorArrayData b
    integral TypeInt    = TF.cast
    integral TypeInt8   = TF.cast
    integral TypeInt16  = TF.cast
    integral TypeInt32  = TF.cast
    integral TypeInt64  = TF.cast
    integral TypeWord   = TF.cast
    integral TypeWord8  = TF.cast
    integral TypeWord16 = TF.cast
    integral TypeWord32 = TF.cast
    integral TypeWord64 = TF.cast

floor :: FloatingType a -> IntegralType b -> TensorArrayData a -> TensorArrayData b
floor = floating
  where
    floating :: FloatingType a -> IntegralType b -> TensorArrayData a -> TensorArrayData b
    floating TypeFloat  t = integral t . TF.floor
    floating TypeDouble t = integral t . TF.floor
    floating TypeHalf   _ = unsupported "half-precision floating point"

    integral :: (ScalarTensorArrayData a, TF.TensorType a) => IntegralType b -> TensorArrayData a -> TensorArrayData b
    integral TypeInt    = TF.cast
    integral TypeInt8   = TF.cast
    integral TypeInt16  = TF.cast
    integral TypeInt32  = TF.cast
    integral TypeInt64  = TF.cast
    integral TypeWord   = TF.cast
    integral TypeWord8  = TF.cast
    integral TypeWord16 = TF.cast
    integral TypeWord32 = TF.cast
    integral TypeWord64 = TF.cast

ceiling :: FloatingType a -> IntegralType b -> TensorArrayData a -> TensorArrayData b
ceiling = floating
  where
    floating :: FloatingType a -> IntegralType b -> TensorArrayData a -> TensorArrayData b
    floating TypeFloat  t = integral t . TF.ceil
    floating TypeDouble t = integral t . TF.ceil
    floating TypeHalf   _ = unsupported "half-precision floating point"

    integral :: (ScalarTensorArrayData a, TF.TensorType a) => IntegralType b -> TensorArrayData a -> TensorArrayData b
    integral TypeInt    = TF.cast
    integral TypeInt8   = TF.cast
    integral TypeInt16  = TF.cast
    integral TypeInt32  = TF.cast
    integral TypeInt64  = TF.cast
    integral TypeWord   = TF.cast
    integral TypeWord8  = TF.cast
    integral TypeWord16 = TF.cast
    integral TypeWord32 = TF.cast
    integral TypeWord64 = TF.cast

atan2 :: forall t. FloatingType t -> TensorArrayData (t, t) -> TensorArrayData t
atan2 = uncurry @t @t . floating
  where
    floating :: FloatingType t -> TensorArrayData t -> TensorArrayData t -> TensorArrayData t
    floating TypeFloat  = TF.atan2
    floating TypeDouble = TF.atan2
    floating TypeHalf   = unsupported "half-precision floating point"

isNaN :: FloatingType t -> TensorArrayData t -> TensorArrayData PrimBool
isNaN = floating
  where
    floating :: FloatingType t -> TensorArrayData t -> TensorArrayData PrimBool
    floating TypeFloat  = TF.cast . TF.isNan
    floating TypeDouble = TF.cast . TF.isNan
    floating TypeHalf   = unsupported "half-precision floating point"

isInfinite :: FloatingType t -> TensorArrayData t -> TensorArrayData PrimBool
isInfinite = floating
  where
    floating :: FloatingType t -> TensorArrayData t -> TensorArrayData PrimBool
    floating TypeFloat  = TF.cast . TF.isInf
    floating TypeDouble = TF.cast . TF.isInf
    floating TypeHalf   = unsupported "half-precision floating point"

lt :: forall t. SingleType t -> TensorArrayData (t, t) -> TensorArrayData PrimBool
lt = uncurry @t @t . single
  where
    single :: SingleType t -> TensorArrayData t -> TensorArrayData t -> TensorArrayData PrimBool
    single (NumSingleType t) = num t

    num :: NumType t -> TensorArrayData t -> TensorArrayData t -> TensorArrayData PrimBool
    num (IntegralNumType t) = integral t
    num (FloatingNumType t) = floating t

    integral :: IntegralType t -> TensorArrayData t -> TensorArrayData t -> TensorArrayData PrimBool
    integral TypeInt    = TF.cast $$ TF.less
    integral TypeInt8   = TF.cast $$ TF.less
    integral TypeInt16  = TF.cast $$ TF.less
    integral TypeInt32  = TF.cast $$ TF.less
    integral TypeInt64  = TF.cast $$ TF.less
    integral TypeWord   = TF.cast $$ TF.less
    integral TypeWord8  = TF.cast $$ TF.less
    integral TypeWord16 = TF.cast $$ TF.less
    integral TypeWord32 = TF.cast $$ TF.less
    integral TypeWord64 = TF.cast $$ TF.less

    floating :: FloatingType t -> TensorArrayData t -> TensorArrayData t -> TensorArrayData PrimBool
    floating TypeFloat  = TF.cast $$ TF.less
    floating TypeDouble = TF.cast $$ TF.less
    floating TypeHalf   = unsupported "half-precision floating point"

gt :: forall t. SingleType t -> TensorArrayData (t, t) -> TensorArrayData PrimBool
gt = uncurry @t @t . single
  where
    single :: SingleType t -> TensorArrayData t -> TensorArrayData t -> TensorArrayData PrimBool
    single (NumSingleType t) = num t

    num :: NumType t -> TensorArrayData t -> TensorArrayData t -> TensorArrayData PrimBool
    num (IntegralNumType t) = integral t
    num (FloatingNumType t) = floating t

    integral :: IntegralType t -> TensorArrayData t -> TensorArrayData t -> TensorArrayData PrimBool
    integral TypeInt    = TF.cast $$ TF.greater
    integral TypeInt8   = TF.cast $$ TF.greater
    integral TypeInt16  = TF.cast $$ TF.greater
    integral TypeInt32  = TF.cast $$ TF.greater
    integral TypeInt64  = TF.cast $$ TF.greater
    integral TypeWord   = TF.cast $$ TF.greater
    integral TypeWord8  = TF.cast $$ TF.greater
    integral TypeWord16 = TF.cast $$ TF.greater
    integral TypeWord32 = TF.cast $$ TF.greater
    integral TypeWord64 = TF.cast $$ TF.greater

    floating :: FloatingType t -> TensorArrayData t -> TensorArrayData t -> TensorArrayData PrimBool
    floating TypeFloat  = TF.cast $$ TF.greater
    floating TypeDouble = TF.cast $$ TF.greater
    floating TypeHalf   = unsupported "half-precision floating point"

lte :: forall t. SingleType t -> TensorArrayData (t, t) -> TensorArrayData PrimBool
lte = uncurry @t @t . single
  where
    single :: SingleType t -> TensorArrayData t -> TensorArrayData t -> TensorArrayData PrimBool
    single (NumSingleType t) = num t

    num :: NumType t -> TensorArrayData t -> TensorArrayData t -> TensorArrayData PrimBool
    num (IntegralNumType t) = integral t
    num (FloatingNumType t) = floating t

    integral :: IntegralType t -> TensorArrayData t -> TensorArrayData t -> TensorArrayData PrimBool
    integral TypeInt    = TF.cast $$ TF.lessEqual
    integral TypeInt8   = TF.cast $$ TF.lessEqual
    integral TypeInt16  = TF.cast $$ TF.lessEqual
    integral TypeInt32  = TF.cast $$ TF.lessEqual
    integral TypeInt64  = TF.cast $$ TF.lessEqual
    integral TypeWord   = TF.cast $$ TF.lessEqual
    integral TypeWord8  = TF.cast $$ TF.lessEqual
    integral TypeWord16 = TF.cast $$ TF.lessEqual
    integral TypeWord32 = TF.cast $$ TF.lessEqual
    integral TypeWord64 = TF.cast $$ TF.lessEqual

    floating :: FloatingType t -> TensorArrayData t -> TensorArrayData t -> TensorArrayData PrimBool
    floating TypeFloat  = TF.cast $$ TF.lessEqual
    floating TypeDouble = TF.cast $$ TF.lessEqual
    floating TypeHalf   = unsupported "half-precision floating point"

gte :: forall t. SingleType t -> TensorArrayData (t, t) -> TensorArrayData PrimBool
gte = uncurry @ t @t . single
  where
    single :: SingleType t -> TensorArrayData t -> TensorArrayData t -> TensorArrayData PrimBool
    single (NumSingleType t) = num t

    num :: NumType t -> TensorArrayData t -> TensorArrayData t -> TensorArrayData PrimBool
    num (IntegralNumType t) = integral t
    num (FloatingNumType t) = floating t

    integral :: IntegralType t -> TensorArrayData t -> TensorArrayData t -> TensorArrayData PrimBool
    integral TypeInt    = TF.cast $$ TF.greaterEqual
    integral TypeInt8   = TF.cast $$ TF.greaterEqual
    integral TypeInt16  = TF.cast $$ TF.greaterEqual
    integral TypeInt32  = TF.cast $$ TF.greaterEqual
    integral TypeInt64  = TF.cast $$ TF.greaterEqual
    integral TypeWord   = TF.cast $$ TF.greaterEqual
    integral TypeWord8  = TF.cast $$ TF.greaterEqual
    integral TypeWord16 = TF.cast $$ TF.greaterEqual
    integral TypeWord32 = TF.cast $$ TF.greaterEqual
    integral TypeWord64 = TF.cast $$ TF.greaterEqual

    floating :: FloatingType t -> TensorArrayData t -> TensorArrayData t -> TensorArrayData PrimBool
    floating TypeFloat  = TF.cast $$ TF.greaterEqual
    floating TypeDouble = TF.cast $$ TF.greaterEqual
    floating TypeHalf   = unsupported "half-precision floating point"

eq :: forall t. SingleType t -> TensorArrayData (t, t) -> TensorArrayData PrimBool
eq = uncurry @t @t . single
  where
    single :: SingleType t -> TensorArrayData t -> TensorArrayData t -> TensorArrayData PrimBool
    single (NumSingleType t) = num t

    num :: NumType t -> TensorArrayData t -> TensorArrayData t -> TensorArrayData PrimBool
    num (IntegralNumType t) = integral t
    num (FloatingNumType t) = floating t

    integral :: IntegralType t -> TensorArrayData t -> TensorArrayData t -> TensorArrayData PrimBool
    integral TypeInt    = TF.cast $$ TF.equal
    integral TypeInt8   = TF.cast $$ TF.equal
    integral TypeInt16  = TF.cast $$ TF.equal
    integral TypeInt32  = TF.cast $$ TF.equal
    integral TypeInt64  = TF.cast $$ TF.equal
    integral TypeWord   = TF.cast $$ TF.equal
    integral TypeWord8  = TF.cast $$ TF.equal
    integral TypeWord16 = TF.cast $$ TF.equal
    integral TypeWord32 = TF.cast $$ TF.equal
    integral TypeWord64 = TF.cast $$ TF.equal

    floating :: FloatingType t -> TensorArrayData t -> TensorArrayData t -> TensorArrayData PrimBool
    floating TypeFloat  = TF.cast $$ TF.equal
    floating TypeDouble = TF.cast $$ TF.equal
    floating TypeHalf   = unsupported "half-precision floating point"

neq :: forall t. SingleType t -> TensorArrayData (t, t) -> TensorArrayData PrimBool
neq = uncurry @t @t . single
  where
    single :: SingleType t -> TensorArrayData t -> TensorArrayData t -> TensorArrayData PrimBool
    single (NumSingleType t) = num t

    num :: NumType t -> TensorArrayData t -> TensorArrayData t -> TensorArrayData PrimBool
    num (IntegralNumType t) = integral t
    num (FloatingNumType t) = floating t

    integral :: IntegralType t -> TensorArrayData t -> TensorArrayData t -> TensorArrayData PrimBool
    integral TypeInt    = TF.cast $$ TF.notEqual
    integral TypeInt8   = TF.cast $$ TF.notEqual
    integral TypeInt16  = TF.cast $$ TF.notEqual
    integral TypeInt32  = TF.cast $$ TF.notEqual
    integral TypeInt64  = TF.cast $$ TF.notEqual
    integral TypeWord   = TF.cast $$ TF.notEqual
    integral TypeWord8  = TF.cast $$ TF.notEqual
    integral TypeWord16 = TF.cast $$ TF.notEqual
    integral TypeWord32 = TF.cast $$ TF.notEqual
    integral TypeWord64 = TF.cast $$ TF.notEqual

    floating :: FloatingType t -> TensorArrayData t -> TensorArrayData t -> TensorArrayData PrimBool
    floating TypeFloat  = TF.cast $$ TF.notEqual
    floating TypeDouble = TF.cast $$ TF.notEqual
    floating TypeHalf   = unsupported "half-precision floating point"

min :: forall t. SingleType t -> TensorArrayData (t, t) -> TensorArrayData t
min = uncurry @t @t . single
  where
    single :: SingleType t -> TensorArrayData t -> TensorArrayData t -> TensorArrayData t
    single (NumSingleType t) = num t

    num :: NumType t -> TensorArrayData t -> TensorArrayData t -> TensorArrayData t
    num (IntegralNumType t) = integral t
    num (FloatingNumType t) = floating t

    integral :: IntegralType t -> TensorArrayData t -> TensorArrayData t -> TensorArrayData t
    integral TypeInt    = TF.minimum
    integral TypeInt8   = excluded
    integral TypeInt16  = TF.minimum
    integral TypeInt32  = TF.minimum
    integral TypeInt64  = TF.minimum
    integral TypeWord   = TF.minimum
    integral TypeWord8  = TF.minimum
    integral TypeWord16 = TF.minimum
    integral TypeWord32 = TF.minimum
    integral TypeWord64 = TF.minimum

    floating :: FloatingType t -> TensorArrayData t -> TensorArrayData t -> TensorArrayData t
    floating TypeFloat  = TF.minimum
    floating TypeDouble = TF.minimum
    floating TypeHalf   = unsupported "half-precision floating point"

max :: forall t. SingleType t -> TensorArrayData (t, t) -> TensorArrayData t
max = uncurry @t @t . single
  where
    single :: SingleType t -> TensorArrayData t -> TensorArrayData t -> TensorArrayData t
    single (NumSingleType t) = num t

    num :: NumType t -> TensorArrayData t -> TensorArrayData t -> TensorArrayData t
    num (IntegralNumType t) = integral t
    num (FloatingNumType t) = floating t

    integral :: IntegralType t -> TensorArrayData t -> TensorArrayData t -> TensorArrayData t
    integral TypeInt    = TF.maximum
    integral TypeInt8   = excluded
    integral TypeInt16  = TF.maximum
    integral TypeInt32  = TF.maximum
    integral TypeInt64  = TF.maximum
    integral TypeWord   = TF.maximum
    integral TypeWord8  = TF.maximum
    integral TypeWord16 = TF.maximum
    integral TypeWord32 = TF.maximum
    integral TypeWord64 = TF.maximum

    floating :: FloatingType t -> TensorArrayData t -> TensorArrayData t -> TensorArrayData t
    floating TypeFloat  = TF.maximum
    floating TypeDouble = TF.maximum
    floating TypeHalf   = unsupported "half-precision floating point"

land :: TensorArrayData PrimBool -> TensorArrayData PrimBool -> TensorArrayData PrimBool
land x y = TF.cast $ TF.logicalAnd (TF.cast x) (TF.cast y)

lor :: TensorArrayData PrimBool -> TensorArrayData PrimBool -> TensorArrayData PrimBool
lor x y = TF.cast $ TF.logicalOr (TF.cast x) (TF.cast y)

lnot :: TensorArrayData PrimBool -> TensorArrayData PrimBool
lnot = TF.cast . TF.logicalNot . TF.cast

fromIntegral :: IntegralType a -> NumType b -> TensorArrayData a -> TensorArrayData b
fromIntegral = integral
  where
    integral :: IntegralType a -> NumType b -> TensorArrayData a -> TensorArrayData b
    integral TypeInt    = num
    integral TypeInt8   = num
    integral TypeInt16  = num
    integral TypeInt32  = num
    integral TypeInt64  = num
    integral TypeWord   = num
    integral TypeWord8  = num
    integral TypeWord16 = num
    integral TypeWord32 = num
    integral TypeWord64 = num

    num :: (ScalarTensorArrayData a, TF.TensorType a) => NumType b -> TensorArrayData a -> TensorArrayData b
    num (IntegralNumType t) = integral' t
    num (FloatingNumType t) = floating' t

    integral' :: (ScalarTensorArrayData a, TF.TensorType a) => IntegralType b -> TensorArrayData a -> TensorArrayData b
    integral' TypeInt    = TF.cast
    integral' TypeInt8   = TF.cast
    integral' TypeInt16  = TF.cast
    integral' TypeInt32  = TF.cast
    integral' TypeInt64  = TF.cast
    integral' TypeWord   = TF.cast
    integral' TypeWord8  = TF.cast
    integral' TypeWord16 = TF.cast
    integral' TypeWord32 = TF.cast
    integral' TypeWord64 = TF.cast

    floating' :: (ScalarTensorArrayData a, TF.TensorType a) => FloatingType b -> TensorArrayData a -> TensorArrayData b
    floating' TypeFloat  = TF.cast
    floating' TypeDouble = TF.cast
    floating' TypeHalf   = unsupported "half-precision floating point"

toFloating :: NumType a -> FloatingType b -> TensorArrayData a -> TensorArrayData b
toFloating = num
  where
    num :: NumType a -> FloatingType b -> TensorArrayData a -> TensorArrayData b
    num (IntegralNumType t) = integral t
    num (FloatingNumType t) = floating t

    integral :: IntegralType a -> FloatingType b -> TensorArrayData a -> TensorArrayData b
    integral TypeInt    = floating'
    integral TypeInt8   = floating'
    integral TypeInt16  = floating'
    integral TypeInt32  = floating'
    integral TypeInt64  = floating'
    integral TypeWord   = floating'
    integral TypeWord8  = floating'
    integral TypeWord16 = floating'
    integral TypeWord32 = floating'
    integral TypeWord64 = floating'

    floating :: FloatingType a -> FloatingType b -> TensorArrayData a -> TensorArrayData b
    floating TypeFloat  = floating'
    floating TypeDouble = floating'
    floating TypeHalf   = unsupported "half-precision floating point"

    floating' :: (ScalarTensorArrayData a, TF.TensorType a) => FloatingType b -> TensorArrayData a -> TensorArrayData b
    floating' TypeFloat  = TF.cast
    floating' TypeDouble = TF.cast
    floating' TypeHalf   = unsupported "half-precision floating point"

