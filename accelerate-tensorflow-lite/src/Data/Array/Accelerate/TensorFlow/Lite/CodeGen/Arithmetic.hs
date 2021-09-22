{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE GADTs               #-}
-- |
-- Module      : Data.Array.Accelerate.TensorFlow.Lite.CodeGen.Arithmetic
-- Copyright   : [2021] The Accelerate Team
-- License     : BSD3
--
-- Maintainer  : Trevor L. McDonell <trevor.mcdonell@gmail.com>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--

module Data.Array.Accelerate.TensorFlow.Lite.CodeGen.Arithmetic
  where

import Data.Array.Accelerate.TensorFlow.Lite.CodeGen.Base
import Data.Array.Accelerate.TensorFlow.Lite.CodeGen.Tensor

import Data.Array.Accelerate.Type

import qualified TensorFlow.Ops                                     as TF


uncurry :: (TensorArrayData a -> TensorArrayData b -> c) -> TensorArrayData (a, b) -> c
uncurry f (x, y) = f x y

add :: NumType t -> TensorArrayData t -> TensorArrayData t -> TensorArrayData t
add = num
  where
    num :: NumType t -> TensorArrayData t -> TensorArrayData t -> TensorArrayData t
    num (IntegralNumType t) = integral t
    num (FloatingNumType t) = floating t

    integral :: IntegralType t -> TensorArrayData t -> TensorArrayData t -> TensorArrayData t
    integral TypeInt8   = TF.add
    integral TypeInt16  = TF.add
    integral TypeInt32  = TF.add
    integral TypeInt64  = TF.add
    integral TypeWord8  = TF.add
    integral TypeWord16 = TF.add
    integral TypeWord32 = TF.add
    integral TypeWord64 = TF.add
    integral TypeInt    = unsupported "Int (use at a specified bit-size instead)"
    integral TypeWord   = unsupported "Word (use at a specified bit-size instead)"

    floating :: FloatingType t -> TensorArrayData t -> TensorArrayData t -> TensorArrayData t
    floating TypeFloat  = TF.add
    floating TypeDouble = TF.add
    floating TypeHalf   = unsupported "half-precision floating point"

mul :: NumType t -> TensorArrayData t -> TensorArrayData t -> TensorArrayData t
mul = num
  where
    num :: NumType t -> TensorArrayData t -> TensorArrayData t -> TensorArrayData t
    num (IntegralNumType t) = integral t
    num (FloatingNumType t) = floating t

    integral :: IntegralType t -> TensorArrayData t -> TensorArrayData t -> TensorArrayData t
    integral TypeInt8   = TF.mul
    integral TypeInt16  = TF.mul
    integral TypeInt32  = TF.mul
    integral TypeInt64  = TF.mul
    integral TypeWord8  = TF.mul
    integral TypeWord16 = TF.mul
    integral TypeWord32 = TF.mul
    integral TypeWord64 = TF.mul
    integral TypeInt    = unsupported "Int (use at a specified bit-size instead)"
    integral TypeWord   = unsupported "Word (use at a specified bit-size instead)"

    floating :: FloatingType t -> TensorArrayData t -> TensorArrayData t -> TensorArrayData t
    floating TypeFloat  = TF.mul
    floating TypeDouble = TF.mul
    floating TypeHalf   = unsupported "half-precision floating point"

