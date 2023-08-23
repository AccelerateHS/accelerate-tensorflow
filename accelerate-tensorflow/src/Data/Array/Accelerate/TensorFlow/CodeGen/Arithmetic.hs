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
import qualified Data.Array.Accelerate.TensorFlow.CodeGen.Tensor.Shim as Sh

import Data.Array.Accelerate.Type
import Data.Array.Accelerate.AST                                    ( PrimBool )

import qualified TensorFlow.GenOps.Core                             as TF
import qualified TensorFlow.Ops                                     as TF
import qualified TensorFlow.Types                                   as TF

import Data.Typeable                                                ( Typeable )
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
    integral TypeInt    = Sh.wrap "add" TF.add
    integral TypeInt8   = Sh.wrap "add" TF.add
    integral TypeInt16  = Sh.wrap "add" TF.add
    integral TypeInt32  = Sh.wrap "add" TF.add
    integral TypeInt64  = Sh.wrap "add" TF.add
    integral TypeWord   = Sh.wrap "add" TF.add
    integral TypeWord8  = Sh.wrap "add" TF.add
    integral TypeWord16 = Sh.wrap "add" TF.add
    integral TypeWord32 = Sh.wrap "add" TF.add
    integral TypeWord64 = Sh.wrap "add" TF.add

    floating :: FloatingType t -> TensorArrayData t -> TensorArrayData t -> TensorArrayData t
    floating TypeFloat  = Sh.wrap "add" TF.add
    floating TypeDouble = Sh.wrap "add" TF.add
    floating TypeHalf   = unsupported "half-precision floating point"

sub :: forall t. NumType t -> TensorArrayData (t, t) -> TensorArrayData t
sub = uncurry @t @t . num
  where
    num :: NumType t -> TensorArrayData t -> TensorArrayData t -> TensorArrayData t
    num (IntegralNumType t) = integral t
    num (FloatingNumType t) = floating t

    integral :: IntegralType t -> TensorArrayData t -> TensorArrayData t -> TensorArrayData t
    integral TypeInt    = Sh.wrap "sub" TF.sub
    integral TypeInt8   = Sh.wrap "sub" TF.sub
    integral TypeInt16  = Sh.wrap "sub" TF.sub
    integral TypeInt32  = Sh.wrap "sub" TF.sub
    integral TypeInt64  = Sh.wrap "sub" TF.sub
    integral TypeWord   = Sh.wrap "sub" TF.sub
    integral TypeWord8  = Sh.wrap "sub" TF.sub
    integral TypeWord16 = Sh.wrap "sub" TF.sub
    integral TypeWord32 = Sh.wrap "sub" TF.sub
    integral TypeWord64 = Sh.wrap "sub" TF.sub

    floating :: FloatingType t -> TensorArrayData t -> TensorArrayData t -> TensorArrayData t
    floating TypeFloat  = Sh.wrap "sub" TF.sub
    floating TypeDouble = Sh.wrap "sub" TF.sub
    floating TypeHalf   = unsupported "half-precision floating point"

mul :: forall t. NumType t -> TensorArrayData (t, t) -> TensorArrayData t
mul = uncurry @t @t . num
  where
    num :: NumType t -> TensorArrayData t -> TensorArrayData t -> TensorArrayData t
    num (IntegralNumType t) = integral t
    num (FloatingNumType t) = floating t

    integral :: IntegralType t -> TensorArrayData t -> TensorArrayData t -> TensorArrayData t
    integral TypeInt    = Sh.wrap "mul" TF.mul
    integral TypeInt8   = Sh.wrap "mul" TF.mul
    integral TypeInt16  = Sh.wrap "mul" TF.mul
    integral TypeInt32  = Sh.wrap "mul" TF.mul
    integral TypeInt64  = Sh.wrap "mul" TF.mul
    integral TypeWord   = Sh.wrap "mul" TF.mul
    integral TypeWord8  = Sh.wrap "mul" TF.mul
    integral TypeWord16 = Sh.wrap "mul" TF.mul
    integral TypeWord32 = Sh.wrap "mul" TF.mul
    integral TypeWord64 = Sh.wrap "mul" TF.mul

    floating :: FloatingType t -> TensorArrayData t -> TensorArrayData t -> TensorArrayData t
    floating TypeFloat  = Sh.wrap "mul" TF.mul
    floating TypeDouble = Sh.wrap "mul" TF.mul
    floating TypeHalf   = unsupported "half-precision floating point"

neg :: NumType t -> TensorArrayData t -> TensorArrayData t
neg = num
  where
    num :: NumType t -> TensorArrayData t -> TensorArrayData t
    num (IntegralNumType t) = integral t
    num (FloatingNumType t) = floating t

    integral :: IntegralType t -> TensorArrayData t -> TensorArrayData t
    integral TypeInt    = Sh.wrap "neg" TF.neg
    integral TypeInt8   = Sh.wrap "neg" TF.neg
    integral TypeInt16  = Sh.wrap "neg" TF.neg
    integral TypeInt32  = Sh.wrap "neg" TF.neg
    integral TypeInt64  = Sh.wrap "neg" TF.neg
    integral TypeWord   = Sh.wrap "neg" TF.neg
    integral TypeWord8  = id
    integral TypeWord16 = Sh.wrap "neg" TF.neg
    integral TypeWord32 = Sh.wrap "neg" TF.neg
    integral TypeWord64 = Sh.wrap "neg" TF.neg

    floating :: FloatingType t -> TensorArrayData t -> TensorArrayData t
    floating TypeFloat  = Sh.wrap "neg" TF.neg
    floating TypeDouble = Sh.wrap "neg" TF.neg
    floating TypeHalf   = unsupported "half-precision floating point"

abs :: NumType t -> TensorArrayData t -> TensorArrayData t
abs = num
  where
    num :: NumType t -> TensorArrayData t -> TensorArrayData t
    num (IntegralNumType t) = integral t
    num (FloatingNumType t) = floating t

    integral :: IntegralType t -> TensorArrayData t -> TensorArrayData t
    integral TypeInt    = Sh.wrap "abs" TF.abs
    integral TypeInt8   = Sh.wrap "abs" TF.abs
    integral TypeInt16  = Sh.wrap "abs" TF.abs
    integral TypeInt32  = Sh.wrap "abs" TF.abs
    integral TypeInt64  = Sh.wrap "abs" TF.abs
    integral TypeWord   = Sh.wrap "abs" TF.abs
    integral TypeWord8  = id
    integral TypeWord16 = Sh.wrap "abs" TF.abs
    integral TypeWord32 = Sh.wrap "abs" TF.abs
    integral TypeWord64 = Sh.wrap "abs" TF.abs

    floating :: FloatingType t -> TensorArrayData t -> TensorArrayData t
    floating TypeFloat  = Sh.wrap "abs" TF.abs
    floating TypeDouble = Sh.wrap "abs" TF.abs
    floating TypeHalf   = unsupported "half-precision floating point"

signum :: NumType t -> TensorArrayData t -> TensorArrayData t
signum = num
  where
    num :: NumType t -> TensorArrayData t -> TensorArrayData t
    num (IntegralNumType t) = integral t
    num (FloatingNumType t) = floating t

    integral :: IntegralType t -> TensorArrayData t -> TensorArrayData t
    integral TypeInt    = Sh.wrap "sign" TF.sign
    integral TypeInt8   = excluded
    integral TypeInt16  = excluded
    integral TypeInt32  = Sh.wrap "sign" TF.sign
    integral TypeInt64  = Sh.wrap "sign" TF.sign
    integral TypeWord   = Sh.wrap "sign" TF.sign
    integral TypeWord8  = id
    integral TypeWord16 = Sh.wrap "sign" TF.sign
    integral TypeWord32 = Sh.wrap "sign" TF.sign
    integral TypeWord64 = Sh.wrap "sign" TF.sign

    floating :: FloatingType t -> TensorArrayData t -> TensorArrayData t
    floating TypeFloat  = Sh.wrap "sign" TF.sign
    floating TypeDouble = Sh.wrap "sign" TF.sign
    floating TypeHalf   = unsupported "half-precision floating point"

quot :: forall t. IntegralType t -> TensorArrayData (t, t) -> TensorArrayData t
quot = uncurry @t @t . integral
  where
    integral :: IntegralType t -> TensorArrayData t -> TensorArrayData t -> TensorArrayData t
    integral TypeInt    = Sh.wrap "div" TF.div
    integral TypeInt8   = Sh.wrap "div" TF.div
    integral TypeInt16  = Sh.wrap "div" TF.div
    integral TypeInt32  = Sh.wrap "div" TF.div
    integral TypeInt64  = Sh.wrap "div" TF.div
    integral TypeWord   = Sh.wrap "div" TF.div
    integral TypeWord8  = Sh.wrap "div" TF.div
    integral TypeWord16 = Sh.wrap "div" TF.div
    integral TypeWord32 = Sh.wrap "div" TF.div
    integral TypeWord64 = Sh.wrap "div" TF.div

rem :: forall t. IntegralType t -> TensorArrayData (t, t) -> TensorArrayData t
rem = uncurry @t @t . integral
  where
    integral :: IntegralType t -> TensorArrayData t -> TensorArrayData t -> TensorArrayData t
    integral TypeInt    = Sh.wrap "mod" TF.mod
    integral TypeInt8   = excluded
    integral TypeInt16  = excluded
    integral TypeInt32  = Sh.wrap "mod" TF.mod
    integral TypeInt64  = Sh.wrap "mod" TF.mod
    integral TypeWord   = Sh.wrap "mod" TF.mod
    integral TypeWord8  = excluded
    integral TypeWord16 = Sh.wrap "mod" TF.mod
    integral TypeWord32 = Sh.wrap "mod" TF.mod
    integral TypeWord64 = Sh.wrap "mod" TF.mod

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
    integral TypeInt    = Sh.wrap "bitwiseAnd" TF.bitwiseAnd
    integral TypeInt8   = Sh.wrap "bitwiseAnd" TF.bitwiseAnd
    integral TypeInt16  = Sh.wrap "bitwiseAnd" TF.bitwiseAnd
    integral TypeInt32  = Sh.wrap "bitwiseAnd" TF.bitwiseAnd
    integral TypeInt64  = Sh.wrap "bitwiseAnd" TF.bitwiseAnd
    integral TypeWord   = Sh.wrap "bitwiseAnd" TF.bitwiseAnd
    integral TypeWord8  = Sh.wrap "bitwiseAnd" TF.bitwiseAnd
    integral TypeWord16 = Sh.wrap "bitwiseAnd" TF.bitwiseAnd
    integral TypeWord32 = Sh.wrap "bitwiseAnd" TF.bitwiseAnd
    integral TypeWord64 = Sh.wrap "bitwiseAnd" TF.bitwiseAnd

bor :: forall t. IntegralType t -> TensorArrayData (t, t) -> TensorArrayData t
bor = uncurry @t @t . integral
  where
    integral :: IntegralType t -> TensorArrayData t -> TensorArrayData t -> TensorArrayData t
    integral TypeInt    = Sh.wrap "bitwiseOr" TF.bitwiseOr
    integral TypeInt8   = Sh.wrap "bitwiseOr" TF.bitwiseOr
    integral TypeInt16  = Sh.wrap "bitwiseOr" TF.bitwiseOr
    integral TypeInt32  = Sh.wrap "bitwiseOr" TF.bitwiseOr
    integral TypeInt64  = Sh.wrap "bitwiseOr" TF.bitwiseOr
    integral TypeWord   = Sh.wrap "bitwiseOr" TF.bitwiseOr
    integral TypeWord8  = Sh.wrap "bitwiseOr" TF.bitwiseOr
    integral TypeWord16 = Sh.wrap "bitwiseOr" TF.bitwiseOr
    integral TypeWord32 = Sh.wrap "bitwiseOr" TF.bitwiseOr
    integral TypeWord64 = Sh.wrap "bitwiseOr" TF.bitwiseOr

xor :: forall t. IntegralType t -> TensorArrayData (t, t) -> TensorArrayData t
xor = uncurry @t @t . integral
  where
    integral :: IntegralType t -> TensorArrayData t -> TensorArrayData t -> TensorArrayData t
    integral TypeInt    = Sh.wrap "bitwiseXor" TF.bitwiseXor
    integral TypeInt8   = Sh.wrap "bitwiseXor" TF.bitwiseXor
    integral TypeInt16  = Sh.wrap "bitwiseXor" TF.bitwiseXor
    integral TypeInt32  = Sh.wrap "bitwiseXor" TF.bitwiseXor
    integral TypeInt64  = Sh.wrap "bitwiseXor" TF.bitwiseXor
    integral TypeWord   = Sh.wrap "bitwiseXor" TF.bitwiseXor
    integral TypeWord8  = Sh.wrap "bitwiseXor" TF.bitwiseXor
    integral TypeWord16 = Sh.wrap "bitwiseXor" TF.bitwiseXor
    integral TypeWord32 = Sh.wrap "bitwiseXor" TF.bitwiseXor
    integral TypeWord64 = Sh.wrap "bitwiseXor" TF.bitwiseXor

complement :: IntegralType t -> TensorArrayData t -> TensorArrayData t
complement = integral
  where
    integral :: IntegralType t -> TensorArrayData t -> TensorArrayData t
    integral TypeInt    t = Sh.wrap "bitwiseXor" TF.bitwiseXor t (Sh.wrap1 "scalar" TF.scalar (-1))
    integral TypeInt8   t = Sh.wrap "bitwiseXor" TF.bitwiseXor t (Sh.wrap1 "scalar" TF.scalar (-1))
    integral TypeInt16  t = Sh.wrap "bitwiseXor" TF.bitwiseXor t (Sh.wrap1 "scalar" TF.scalar (-1))
    integral TypeInt32  t = Sh.wrap "bitwiseXor" TF.bitwiseXor t (Sh.wrap1 "scalar" TF.scalar (-1))
    integral TypeInt64  t = Sh.wrap "bitwiseXor" TF.bitwiseXor t (Sh.wrap1 "scalar" TF.scalar (-1))
    integral TypeWord   t = Sh.wrap "bitwiseXor" TF.bitwiseXor t (Sh.wrap1 "scalar" TF.scalar maxBound)
    integral TypeWord8  t = Sh.wrap "bitwiseXor" TF.bitwiseXor t (Sh.wrap1 "scalar" TF.scalar maxBound)
    integral TypeWord16 t = Sh.wrap "bitwiseXor" TF.bitwiseXor t (Sh.wrap1 "scalar" TF.scalar maxBound)
    integral TypeWord32 t = Sh.wrap "bitwiseXor" TF.bitwiseXor t (Sh.wrap1 "scalar" TF.scalar maxBound)
    integral TypeWord64 t = Sh.wrap "bitwiseXor" TF.bitwiseXor t (Sh.wrap1 "scalar" TF.scalar maxBound)

shiftL :: forall t. IntegralType t -> TensorArrayData (t, Int) -> TensorArrayData t
shiftL = uncurry @t @Int . integral
  where
    integral :: IntegralType t -> TensorArrayData t -> TensorArrayData Int -> TensorArrayData t
    integral TypeInt    t x = Sh.wrap "leftShift" TF.leftShift t x
    integral TypeInt8   t x = Sh.wrap "leftShift" TF.leftShift t (Sh.wrap "cast" TF.cast x)
    integral TypeInt16  t x = Sh.wrap "leftShift" TF.leftShift t (Sh.wrap "cast" TF.cast x)
    integral TypeInt32  t x = Sh.wrap "leftShift" TF.leftShift t (Sh.wrap "cast" TF.cast x)
    integral TypeInt64  t x = Sh.wrap "leftShift" TF.leftShift t (Sh.wrap "cast" TF.cast x)
    integral TypeWord   t x = Sh.wrap "leftShift" TF.leftShift t (Sh.wrap "cast" TF.cast x)
    integral TypeWord8  t x = Sh.wrap "leftShift" TF.leftShift t (Sh.wrap "cast" TF.cast x)
    integral TypeWord16 t x = Sh.wrap "leftShift" TF.leftShift t (Sh.wrap "cast" TF.cast x)
    integral TypeWord32 t x = Sh.wrap "leftShift" TF.leftShift t (Sh.wrap "cast" TF.cast x)
    integral TypeWord64 t x = Sh.wrap "leftShift" TF.leftShift t (Sh.wrap "cast" TF.cast x)

shiftR  :: forall t. IntegralType t -> TensorArrayData (t, Int) -> TensorArrayData t
shiftR = uncurry @t @Int . integral
  where
    integral :: IntegralType t -> TensorArrayData t -> TensorArrayData Int -> TensorArrayData t
    integral TypeInt    t x = Sh.wrap "rightShift" TF.rightShift t x
    integral TypeInt8   t x = Sh.wrap "rightShift" TF.rightShift t (Sh.wrap "cast" TF.cast x)
    integral TypeInt16  t x = Sh.wrap "rightShift" TF.rightShift t (Sh.wrap "cast" TF.cast x)
    integral TypeInt32  t x = Sh.wrap "rightShift" TF.rightShift t (Sh.wrap "cast" TF.cast x)
    integral TypeInt64  t x = Sh.wrap "rightShift" TF.rightShift t (Sh.wrap "cast" TF.cast x)
    integral TypeWord   t x = Sh.wrap "rightShift" TF.rightShift t (Sh.wrap "cast" TF.cast x)
    integral TypeWord8  t x = Sh.wrap "rightShift" TF.rightShift t (Sh.wrap "cast" TF.cast x)
    integral TypeWord16 t x = Sh.wrap "rightShift" TF.rightShift t (Sh.wrap "cast" TF.cast x)
    integral TypeWord32 t x = Sh.wrap "rightShift" TF.rightShift t (Sh.wrap "cast" TF.cast x)
    integral TypeWord64 t x = Sh.wrap "rightShift" TF.rightShift t (Sh.wrap "cast" TF.cast x)

-- rotateL :: IntegralType t -> TensorArrayData t -> TensorArrayData Int -> TensorArrayData t
-- rotateR :: IntegralType t -> TensorArrayData t -> TensorArrayData Int -> TensorArrayData t

popCount :: IntegralType t -> TensorArrayData t -> TensorArrayData Int
popCount = integral
  where
    integral :: IntegralType t -> TensorArrayData t -> TensorArrayData Int
    integral TypeInt    = Sh.wrap "cast" TF.cast . Sh.wrap "populationCount" TF.populationCount
    integral TypeInt8   = Sh.wrap "cast" TF.cast . Sh.wrap "populationCount" TF.populationCount
    integral TypeInt16  = Sh.wrap "cast" TF.cast . Sh.wrap "populationCount" TF.populationCount
    integral TypeInt32  = Sh.wrap "cast" TF.cast . Sh.wrap "populationCount" TF.populationCount
    integral TypeInt64  = Sh.wrap "cast" TF.cast . Sh.wrap "populationCount" TF.populationCount
    integral TypeWord   = Sh.wrap "cast" TF.cast . Sh.wrap "populationCount" TF.populationCount
    integral TypeWord8  = Sh.wrap "cast" TF.cast . Sh.wrap "populationCount" TF.populationCount
    integral TypeWord16 = Sh.wrap "cast" TF.cast . Sh.wrap "populationCount" TF.populationCount
    integral TypeWord32 = Sh.wrap "cast" TF.cast . Sh.wrap "populationCount" TF.populationCount
    integral TypeWord64 = Sh.wrap "cast" TF.cast . Sh.wrap "populationCount" TF.populationCount

-- countLeadingZeros  :: IntegralType t -> TensorArrayData t -> TensorArrayData Int
-- countTrailingZeros :: IntegralType t -> TensorArrayData t -> TensorArrayData Int

fdiv :: forall t. FloatingType t -> TensorArrayData (t, t) -> TensorArrayData t
fdiv = uncurry @t @t . floating
  where
    floating :: FloatingType t -> TensorArrayData t -> TensorArrayData t -> TensorArrayData t
    floating TypeFloat  = Sh.wrap "div" TF.div
    floating TypeDouble = Sh.wrap "div" TF.div
    floating TypeHalf   = unsupported "half-precision floating point"

recip :: FloatingType t -> TensorArrayData t -> TensorArrayData t
recip = floating
  where
    floating :: FloatingType t -> TensorArrayData t -> TensorArrayData t
    floating TypeFloat  = Sh.wrap "reciprocal" TF.reciprocal
    floating TypeDouble = Sh.wrap "reciprocal" TF.reciprocal
    floating TypeHalf   = unsupported "half-precision floating point"

sin :: FloatingType t -> TensorArrayData t -> TensorArrayData t
sin = floating
  where
    floating :: FloatingType t -> TensorArrayData t -> TensorArrayData t
    floating TypeFloat  = Sh.wrap "sin" TF.sin
    floating TypeDouble = Sh.wrap "sin" TF.sin
    floating TypeHalf   = unsupported "half-precision floating point"

cos :: FloatingType t -> TensorArrayData t -> TensorArrayData t
cos = floating
  where
    floating :: FloatingType t -> TensorArrayData t -> TensorArrayData t
    floating TypeFloat  = Sh.wrap "cos" TF.cos
    floating TypeDouble = Sh.wrap "cos" TF.cos
    floating TypeHalf   = unsupported "half-precision floating point"

tan :: FloatingType t -> TensorArrayData t -> TensorArrayData t
tan = floating
  where
    floating :: FloatingType t -> TensorArrayData t -> TensorArrayData t
    floating TypeFloat  = Sh.wrap "tan" TF.tan
    floating TypeDouble = Sh.wrap "tan" TF.tan
    floating TypeHalf   = unsupported "half-precision floating point"

asin :: FloatingType t -> TensorArrayData t -> TensorArrayData t
asin = floating
  where
    floating :: FloatingType t -> TensorArrayData t -> TensorArrayData t
    floating TypeFloat  = Sh.wrap "asin" TF.asin
    floating TypeDouble = Sh.wrap "asin" TF.asin
    floating TypeHalf   = unsupported "half-precision floating point"

acos :: FloatingType t -> TensorArrayData t -> TensorArrayData t
acos = floating
  where
    floating :: FloatingType t -> TensorArrayData t -> TensorArrayData t
    floating TypeFloat  = Sh.wrap "acos" TF.acos
    floating TypeDouble = Sh.wrap "acos" TF.acos
    floating TypeHalf   = unsupported "half-precision floating point"

atan :: FloatingType t -> TensorArrayData t -> TensorArrayData t
atan = floating
  where
    floating :: FloatingType t -> TensorArrayData t -> TensorArrayData t
    floating TypeFloat  = Sh.wrap "atan" TF.atan
    floating TypeDouble = Sh.wrap "atan" TF.atan
    floating TypeHalf   = unsupported "half-precision floating point"

sinh :: FloatingType t -> TensorArrayData t -> TensorArrayData t
sinh = floating
  where
    floating :: FloatingType t -> TensorArrayData t -> TensorArrayData t
    floating TypeFloat  = Sh.wrap "sinh" TF.sinh
    floating TypeDouble = Sh.wrap "sinh" TF.sinh
    floating TypeHalf   = unsupported "half-precision floating point"

cosh :: FloatingType t -> TensorArrayData t -> TensorArrayData t
cosh = floating
  where
    floating :: FloatingType t -> TensorArrayData t -> TensorArrayData t
    floating TypeFloat  = Sh.wrap "cosh" TF.cosh
    floating TypeDouble = Sh.wrap "cosh" TF.cosh
    floating TypeHalf   = unsupported "half-precision floating point"

tanh :: FloatingType t -> TensorArrayData t -> TensorArrayData t
tanh = floating
  where
    floating :: FloatingType t -> TensorArrayData t -> TensorArrayData t
    floating TypeFloat  = Sh.wrap "tanh" TF.tanh
    floating TypeDouble = Sh.wrap "tanh" TF.tanh
    floating TypeHalf   = unsupported "half-precision floating point"

asinh :: FloatingType t -> TensorArrayData t -> TensorArrayData t
asinh = floating
  where
    floating :: FloatingType t -> TensorArrayData t -> TensorArrayData t
    floating TypeFloat  = Sh.wrap "asinh" TF.asinh
    floating TypeDouble = Sh.wrap "asinh" TF.asinh
    floating TypeHalf   = unsupported "half-precision floating point"

acosh :: FloatingType t -> TensorArrayData t -> TensorArrayData t
acosh = floating
  where
    floating :: FloatingType t -> TensorArrayData t -> TensorArrayData t
    floating TypeFloat  = Sh.wrap "acosh" TF.acosh
    floating TypeDouble = Sh.wrap "acosh" TF.acosh
    floating TypeHalf   = unsupported "half-precision floating point"

atanh :: FloatingType t -> TensorArrayData t -> TensorArrayData t
atanh = floating
  where
    floating :: FloatingType t -> TensorArrayData t -> TensorArrayData t
    floating TypeFloat  = Sh.wrap "atanh" TF.atanh
    floating TypeDouble = Sh.wrap "atanh" TF.atanh
    floating TypeHalf   = unsupported "half-precision floating point"

exp :: FloatingType t -> TensorArrayData t -> TensorArrayData t
exp = floating
  where
    floating :: FloatingType t -> TensorArrayData t -> TensorArrayData t
    floating TypeFloat  = Sh.wrap "exp" TF.exp
    floating TypeDouble = Sh.wrap "exp" TF.exp
    floating TypeHalf   = unsupported "half-precision floating point"

sqrt :: FloatingType t -> TensorArrayData t -> TensorArrayData t
sqrt = floating
  where
    floating :: FloatingType t -> TensorArrayData t -> TensorArrayData t
    floating TypeFloat  = Sh.wrap "sqrt" TF.sqrt
    floating TypeDouble = Sh.wrap "sqrt" TF.sqrt
    floating TypeHalf   = unsupported "half-precision floating point"

log :: FloatingType t -> TensorArrayData t -> TensorArrayData t
log = floating
  where
    floating :: FloatingType t -> TensorArrayData t -> TensorArrayData t
    floating TypeFloat  = Sh.wrap "log" TF.log
    floating TypeDouble = Sh.wrap "log" TF.log
    floating TypeHalf   = unsupported "half-precision floating point"

fpow :: forall t. FloatingType t -> TensorArrayData (t, t) -> TensorArrayData t
fpow = uncurry @t @t . floating
  where
    floating :: FloatingType t -> TensorArrayData t -> TensorArrayData t -> TensorArrayData t
    floating TypeFloat  = Sh.wrap "pow" TF.pow
    floating TypeDouble = Sh.wrap "pow" TF.pow
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

    integral :: (ScalarTensorArrayData a, TF.TensorType a, Typeable a, Show a) => IntegralType b -> TensorArrayData a -> TensorArrayData b
    integral TypeInt    = Sh.wrap "cast" TF.cast
    integral TypeInt8   = Sh.wrap "cast" TF.cast
    integral TypeInt16  = Sh.wrap "cast" TF.cast
    integral TypeInt32  = Sh.wrap "cast" TF.cast
    integral TypeInt64  = Sh.wrap "cast" TF.cast
    integral TypeWord   = Sh.wrap "cast" TF.cast
    integral TypeWord8  = Sh.wrap "cast" TF.cast
    integral TypeWord16 = Sh.wrap "cast" TF.cast
    integral TypeWord32 = Sh.wrap "cast" TF.cast
    integral TypeWord64 = Sh.wrap "cast" TF.cast

round :: FloatingType a -> IntegralType b -> TensorArrayData a -> TensorArrayData b
round = floating
  where
    floating :: FloatingType a -> IntegralType b -> TensorArrayData a -> TensorArrayData b
    floating TypeFloat  t = integral t . Sh.wrap "round" TF.round
    floating TypeDouble t = integral t . Sh.wrap "round" TF.round
    floating TypeHalf   _ = unsupported "half-precision floating point"

    integral :: (ScalarTensorArrayData a, TF.TensorType a, Typeable a, Show a) => IntegralType b -> TensorArrayData a -> TensorArrayData b
    integral TypeInt    = Sh.wrap "cast" TF.cast
    integral TypeInt8   = Sh.wrap "cast" TF.cast
    integral TypeInt16  = Sh.wrap "cast" TF.cast
    integral TypeInt32  = Sh.wrap "cast" TF.cast
    integral TypeInt64  = Sh.wrap "cast" TF.cast
    integral TypeWord   = Sh.wrap "cast" TF.cast
    integral TypeWord8  = Sh.wrap "cast" TF.cast
    integral TypeWord16 = Sh.wrap "cast" TF.cast
    integral TypeWord32 = Sh.wrap "cast" TF.cast
    integral TypeWord64 = Sh.wrap "cast" TF.cast

floor :: FloatingType a -> IntegralType b -> TensorArrayData a -> TensorArrayData b
floor = floating
  where
    floating :: FloatingType a -> IntegralType b -> TensorArrayData a -> TensorArrayData b
    floating TypeFloat  t = integral t . Sh.wrap "floor" TF.floor
    floating TypeDouble t = integral t . Sh.wrap "floor" TF.floor
    floating TypeHalf   _ = unsupported "half-precision floating point"

    integral :: (ScalarTensorArrayData a, TF.TensorType a, Typeable a, Show a) => IntegralType b -> TensorArrayData a -> TensorArrayData b
    integral TypeInt    = Sh.wrap "cast" TF.cast
    integral TypeInt8   = Sh.wrap "cast" TF.cast
    integral TypeInt16  = Sh.wrap "cast" TF.cast
    integral TypeInt32  = Sh.wrap "cast" TF.cast
    integral TypeInt64  = Sh.wrap "cast" TF.cast
    integral TypeWord   = Sh.wrap "cast" TF.cast
    integral TypeWord8  = Sh.wrap "cast" TF.cast
    integral TypeWord16 = Sh.wrap "cast" TF.cast
    integral TypeWord32 = Sh.wrap "cast" TF.cast
    integral TypeWord64 = Sh.wrap "cast" TF.cast

ceiling :: FloatingType a -> IntegralType b -> TensorArrayData a -> TensorArrayData b
ceiling = floating
  where
    floating :: FloatingType a -> IntegralType b -> TensorArrayData a -> TensorArrayData b
    floating TypeFloat  t = integral t . Sh.wrap "ceil" TF.ceil
    floating TypeDouble t = integral t . Sh.wrap "ceil" TF.ceil
    floating TypeHalf   _ = unsupported "half-precision floating point"

    integral :: (ScalarTensorArrayData a, TF.TensorType a, Typeable a, Show a) => IntegralType b -> TensorArrayData a -> TensorArrayData b
    integral TypeInt    = Sh.wrap "cast" TF.cast
    integral TypeInt8   = Sh.wrap "cast" TF.cast
    integral TypeInt16  = Sh.wrap "cast" TF.cast
    integral TypeInt32  = Sh.wrap "cast" TF.cast
    integral TypeInt64  = Sh.wrap "cast" TF.cast
    integral TypeWord   = Sh.wrap "cast" TF.cast
    integral TypeWord8  = Sh.wrap "cast" TF.cast
    integral TypeWord16 = Sh.wrap "cast" TF.cast
    integral TypeWord32 = Sh.wrap "cast" TF.cast
    integral TypeWord64 = Sh.wrap "cast" TF.cast

atan2 :: forall t. FloatingType t -> TensorArrayData (t, t) -> TensorArrayData t
atan2 = uncurry @t @t . floating
  where
    floating :: FloatingType t -> TensorArrayData t -> TensorArrayData t -> TensorArrayData t
    floating TypeFloat  = Sh.wrap "atan2" TF.atan2
    floating TypeDouble = Sh.wrap "atan2" TF.atan2
    floating TypeHalf   = unsupported "half-precision floating point"

isNaN :: FloatingType t -> TensorArrayData t -> TensorArrayData PrimBool
isNaN = floating
  where
    floating :: FloatingType t -> TensorArrayData t -> TensorArrayData PrimBool
    floating TypeFloat  = Sh.wrap "cast" TF.cast . Sh.wrap "isNan" TF.isNan
    floating TypeDouble = Sh.wrap "cast" TF.cast . Sh.wrap "isNan" TF.isNan
    floating TypeHalf   = unsupported "half-precision floating point"

isInfinite :: FloatingType t -> TensorArrayData t -> TensorArrayData PrimBool
isInfinite = floating
  where
    floating :: FloatingType t -> TensorArrayData t -> TensorArrayData PrimBool
    floating TypeFloat  = Sh.wrap "cast" TF.cast . Sh.wrap "isInf" TF.isInf
    floating TypeDouble = Sh.wrap "cast" TF.cast . Sh.wrap "isInf" TF.isInf
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
    integral TypeInt    = Sh.wrap "cast" TF.cast $$ Sh.wrap "less" TF.less
    integral TypeInt8   = Sh.wrap "cast" TF.cast $$ Sh.wrap "less" TF.less
    integral TypeInt16  = Sh.wrap "cast" TF.cast $$ Sh.wrap "less" TF.less
    integral TypeInt32  = Sh.wrap "cast" TF.cast $$ Sh.wrap "less" TF.less
    integral TypeInt64  = Sh.wrap "cast" TF.cast $$ Sh.wrap "less" TF.less
    integral TypeWord   = Sh.wrap "cast" TF.cast $$ Sh.wrap "less" TF.less
    integral TypeWord8  = Sh.wrap "cast" TF.cast $$ Sh.wrap "less" TF.less
    integral TypeWord16 = Sh.wrap "cast" TF.cast $$ Sh.wrap "less" TF.less
    integral TypeWord32 = Sh.wrap "cast" TF.cast $$ Sh.wrap "less" TF.less
    integral TypeWord64 = Sh.wrap "cast" TF.cast $$ Sh.wrap "less" TF.less

    floating :: FloatingType t -> TensorArrayData t -> TensorArrayData t -> TensorArrayData PrimBool
    floating TypeFloat  = Sh.wrap "cast" TF.cast $$ Sh.wrap "less" TF.less
    floating TypeDouble = Sh.wrap "cast" TF.cast $$ Sh.wrap "less" TF.less
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
    integral TypeInt    = Sh.wrap "cast" TF.cast $$ Sh.wrap "greater" TF.greater
    integral TypeInt8   = Sh.wrap "cast" TF.cast $$ Sh.wrap "greater" TF.greater
    integral TypeInt16  = Sh.wrap "cast" TF.cast $$ Sh.wrap "greater" TF.greater
    integral TypeInt32  = Sh.wrap "cast" TF.cast $$ Sh.wrap "greater" TF.greater
    integral TypeInt64  = Sh.wrap "cast" TF.cast $$ Sh.wrap "greater" TF.greater
    integral TypeWord   = Sh.wrap "cast" TF.cast $$ Sh.wrap "greater" TF.greater
    integral TypeWord8  = Sh.wrap "cast" TF.cast $$ Sh.wrap "greater" TF.greater
    integral TypeWord16 = Sh.wrap "cast" TF.cast $$ Sh.wrap "greater" TF.greater
    integral TypeWord32 = Sh.wrap "cast" TF.cast $$ Sh.wrap "greater" TF.greater
    integral TypeWord64 = Sh.wrap "cast" TF.cast $$ Sh.wrap "greater" TF.greater

    floating :: FloatingType t -> TensorArrayData t -> TensorArrayData t -> TensorArrayData PrimBool
    floating TypeFloat  = Sh.wrap "cast" TF.cast $$ Sh.wrap "greater" TF.greater
    floating TypeDouble = Sh.wrap "cast" TF.cast $$ Sh.wrap "greater" TF.greater
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
    integral TypeInt    = Sh.wrap "cast" TF.cast $$ Sh.wrap "lessEqual" TF.lessEqual
    integral TypeInt8   = Sh.wrap "cast" TF.cast $$ Sh.wrap "lessEqual" TF.lessEqual
    integral TypeInt16  = Sh.wrap "cast" TF.cast $$ Sh.wrap "lessEqual" TF.lessEqual
    integral TypeInt32  = Sh.wrap "cast" TF.cast $$ Sh.wrap "lessEqual" TF.lessEqual
    integral TypeInt64  = Sh.wrap "cast" TF.cast $$ Sh.wrap "lessEqual" TF.lessEqual
    integral TypeWord   = Sh.wrap "cast" TF.cast $$ Sh.wrap "lessEqual" TF.lessEqual
    integral TypeWord8  = Sh.wrap "cast" TF.cast $$ Sh.wrap "lessEqual" TF.lessEqual
    integral TypeWord16 = Sh.wrap "cast" TF.cast $$ Sh.wrap "lessEqual" TF.lessEqual
    integral TypeWord32 = Sh.wrap "cast" TF.cast $$ Sh.wrap "lessEqual" TF.lessEqual
    integral TypeWord64 = Sh.wrap "cast" TF.cast $$ Sh.wrap "lessEqual" TF.lessEqual

    floating :: FloatingType t -> TensorArrayData t -> TensorArrayData t -> TensorArrayData PrimBool
    floating TypeFloat  = Sh.wrap "cast" TF.cast $$ Sh.wrap "lessEqual" TF.lessEqual
    floating TypeDouble = Sh.wrap "cast" TF.cast $$ Sh.wrap "lessEqual" TF.lessEqual
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
    integral TypeInt    = Sh.wrap "cast" TF.cast $$ Sh.wrap "greaterEqual" TF.greaterEqual
    integral TypeInt8   = Sh.wrap "cast" TF.cast $$ Sh.wrap "greaterEqual" TF.greaterEqual
    integral TypeInt16  = Sh.wrap "cast" TF.cast $$ Sh.wrap "greaterEqual" TF.greaterEqual
    integral TypeInt32  = Sh.wrap "cast" TF.cast $$ Sh.wrap "greaterEqual" TF.greaterEqual
    integral TypeInt64  = Sh.wrap "cast" TF.cast $$ Sh.wrap "greaterEqual" TF.greaterEqual
    integral TypeWord   = Sh.wrap "cast" TF.cast $$ Sh.wrap "greaterEqual" TF.greaterEqual
    integral TypeWord8  = Sh.wrap "cast" TF.cast $$ Sh.wrap "greaterEqual" TF.greaterEqual
    integral TypeWord16 = Sh.wrap "cast" TF.cast $$ Sh.wrap "greaterEqual" TF.greaterEqual
    integral TypeWord32 = Sh.wrap "cast" TF.cast $$ Sh.wrap "greaterEqual" TF.greaterEqual
    integral TypeWord64 = Sh.wrap "cast" TF.cast $$ Sh.wrap "greaterEqual" TF.greaterEqual

    floating :: FloatingType t -> TensorArrayData t -> TensorArrayData t -> TensorArrayData PrimBool
    floating TypeFloat  = Sh.wrap "cast" TF.cast $$ Sh.wrap "greaterEqual" TF.greaterEqual
    floating TypeDouble = Sh.wrap "cast" TF.cast $$ Sh.wrap "greaterEqual" TF.greaterEqual
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
    integral TypeInt    = Sh.wrap "cast" TF.cast $$ Sh.wrap "equal" TF.equal
    integral TypeInt8   = Sh.wrap "cast" TF.cast $$ Sh.wrap "equal" TF.equal
    integral TypeInt16  = Sh.wrap "cast" TF.cast $$ Sh.wrap "equal" TF.equal
    integral TypeInt32  = Sh.wrap "cast" TF.cast $$ Sh.wrap "equal" TF.equal
    integral TypeInt64  = Sh.wrap "cast" TF.cast $$ Sh.wrap "equal" TF.equal
    integral TypeWord   = Sh.wrap "cast" TF.cast $$ Sh.wrap "equal" TF.equal
    integral TypeWord8  = Sh.wrap "cast" TF.cast $$ Sh.wrap "equal" TF.equal
    integral TypeWord16 = Sh.wrap "cast" TF.cast $$ Sh.wrap "equal" TF.equal
    integral TypeWord32 = Sh.wrap "cast" TF.cast $$ Sh.wrap "equal" TF.equal
    integral TypeWord64 = Sh.wrap "cast" TF.cast $$ Sh.wrap "equal" TF.equal

    floating :: FloatingType t -> TensorArrayData t -> TensorArrayData t -> TensorArrayData PrimBool
    floating TypeFloat  = Sh.wrap "cast" TF.cast $$ Sh.wrap "equal" TF.equal
    floating TypeDouble = Sh.wrap "cast" TF.cast $$ Sh.wrap "equal" TF.equal
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
    integral TypeInt    = Sh.wrap "cast" TF.cast $$ Sh.wrap "notEqual" TF.notEqual
    integral TypeInt8   = Sh.wrap "cast" TF.cast $$ Sh.wrap "notEqual" TF.notEqual
    integral TypeInt16  = Sh.wrap "cast" TF.cast $$ Sh.wrap "notEqual" TF.notEqual
    integral TypeInt32  = Sh.wrap "cast" TF.cast $$ Sh.wrap "notEqual" TF.notEqual
    integral TypeInt64  = Sh.wrap "cast" TF.cast $$ Sh.wrap "notEqual" TF.notEqual
    integral TypeWord   = Sh.wrap "cast" TF.cast $$ Sh.wrap "notEqual" TF.notEqual
    integral TypeWord8  = Sh.wrap "cast" TF.cast $$ Sh.wrap "notEqual" TF.notEqual
    integral TypeWord16 = Sh.wrap "cast" TF.cast $$ Sh.wrap "notEqual" TF.notEqual
    integral TypeWord32 = Sh.wrap "cast" TF.cast $$ Sh.wrap "notEqual" TF.notEqual
    integral TypeWord64 = Sh.wrap "cast" TF.cast $$ Sh.wrap "notEqual" TF.notEqual

    floating :: FloatingType t -> TensorArrayData t -> TensorArrayData t -> TensorArrayData PrimBool
    floating TypeFloat  = Sh.wrap "cast" TF.cast $$ Sh.wrap "notEqual" TF.notEqual
    floating TypeDouble = Sh.wrap "cast" TF.cast $$ Sh.wrap "notEqual" TF.notEqual
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
    integral TypeInt    = Sh.wrap "minimum" TF.minimum
    integral TypeInt8   = excluded
    integral TypeInt16  = Sh.wrap "minimum" TF.minimum
    integral TypeInt32  = Sh.wrap "minimum" TF.minimum
    integral TypeInt64  = Sh.wrap "minimum" TF.minimum
    integral TypeWord   = Sh.wrap "minimum" TF.minimum
    integral TypeWord8  = Sh.wrap "minimum" TF.minimum
    integral TypeWord16 = Sh.wrap "minimum" TF.minimum
    integral TypeWord32 = Sh.wrap "minimum" TF.minimum
    integral TypeWord64 = Sh.wrap "minimum" TF.minimum

    floating :: FloatingType t -> TensorArrayData t -> TensorArrayData t -> TensorArrayData t
    floating TypeFloat  = Sh.wrap "minimum" TF.minimum
    floating TypeDouble = Sh.wrap "minimum" TF.minimum
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
    integral TypeInt    = Sh.wrap "maximum" TF.maximum
    integral TypeInt8   = excluded
    integral TypeInt16  = Sh.wrap "maximum" TF.maximum
    integral TypeInt32  = Sh.wrap "maximum" TF.maximum
    integral TypeInt64  = Sh.wrap "maximum" TF.maximum
    integral TypeWord   = Sh.wrap "maximum" TF.maximum
    integral TypeWord8  = Sh.wrap "maximum" TF.maximum
    integral TypeWord16 = Sh.wrap "maximum" TF.maximum
    integral TypeWord32 = Sh.wrap "maximum" TF.maximum
    integral TypeWord64 = Sh.wrap "maximum" TF.maximum

    floating :: FloatingType t -> TensorArrayData t -> TensorArrayData t -> TensorArrayData t
    floating TypeFloat  = Sh.wrap "maximum" TF.maximum
    floating TypeDouble = Sh.wrap "maximum" TF.maximum
    floating TypeHalf   = unsupported "half-precision floating point"

land :: TensorArrayData PrimBool -> TensorArrayData PrimBool -> TensorArrayData PrimBool
land x y = Sh.wrap "cast" TF.cast $ Sh.wrap "logicalAnd" TF.logicalAnd (Sh.wrap "cast" TF.cast x) (Sh.wrap "cast" TF.cast y)

lor :: TensorArrayData PrimBool -> TensorArrayData PrimBool -> TensorArrayData PrimBool
lor x y = Sh.wrap "cast" TF.cast $ Sh.wrap "logicalOr" TF.logicalOr (Sh.wrap "cast" TF.cast x) (Sh.wrap "cast" TF.cast y)

lnot :: TensorArrayData PrimBool -> TensorArrayData PrimBool
lnot = Sh.wrap "cast" TF.cast . Sh.wrap "logicalNot" TF.logicalNot . Sh.wrap "cast" TF.cast

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

    num :: (ScalarTensorArrayData a, TF.TensorType a, Typeable a, Show a) => NumType b -> TensorArrayData a -> TensorArrayData b
    num (IntegralNumType t) = integral' t
    num (FloatingNumType t) = floating' t

    integral' :: (ScalarTensorArrayData a, TF.TensorType a, Typeable a, Show a) => IntegralType b -> TensorArrayData a -> TensorArrayData b
    integral' TypeInt    = Sh.wrap "cast" TF.cast
    integral' TypeInt8   = Sh.wrap "cast" TF.cast
    integral' TypeInt16  = Sh.wrap "cast" TF.cast
    integral' TypeInt32  = Sh.wrap "cast" TF.cast
    integral' TypeInt64  = Sh.wrap "cast" TF.cast
    integral' TypeWord   = Sh.wrap "cast" TF.cast
    integral' TypeWord8  = Sh.wrap "cast" TF.cast
    integral' TypeWord16 = Sh.wrap "cast" TF.cast
    integral' TypeWord32 = Sh.wrap "cast" TF.cast
    integral' TypeWord64 = Sh.wrap "cast" TF.cast

    floating' :: (ScalarTensorArrayData a, TF.TensorType a, Typeable a, Show a) => FloatingType b -> TensorArrayData a -> TensorArrayData b
    floating' TypeFloat  = Sh.wrap "cast" TF.cast
    floating' TypeDouble = Sh.wrap "cast" TF.cast
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

    floating' :: (ScalarTensorArrayData a, TF.TensorType a, Typeable a, Show a) => FloatingType b -> TensorArrayData a -> TensorArrayData b
    floating' TypeFloat  = Sh.wrap "cast" TF.cast
    floating' TypeDouble = Sh.wrap "cast" TF.cast
    floating' TypeHalf   = unsupported "half-precision floating point"

