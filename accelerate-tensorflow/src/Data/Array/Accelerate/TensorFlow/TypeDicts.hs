{-# LANGUAGE GADTs #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}
module Data.Array.Accelerate.TensorFlow.TypeDicts (
  buildTypeDictsScalar,
  Convertable(..),
) where

import Data.Array.Accelerate.TensorFlow.CodeGen.Base
import Data.Array.Accelerate.TensorFlow.CodeGen.Tensor
import qualified Data.Array.Accelerate.TensorFlow.CodeGen.Tensor.Shim as Sh

import Data.Array.Accelerate.Array.Data (GArrayDataR)
import Data.Array.Accelerate.Array.Unique (UniqueArray)
import Data.Array.Accelerate.Type

import Data.Typeable
import Foreign.Storable
import qualified TensorFlow.Core                                    as TF
import Data.ByteString.Char8


class Convertable a b where
  convertConvertable :: a -> b

instance Convertable Int HostEquivalentInt where
  convertConvertable = fromIntegral

instance Convertable Word HostEquivalentWord where
  convertConvertable = fromIntegral

instance {-# OVERLAPPABLE #-} Convertable a a where
  convertConvertable = id

type TypeDictsFor t s =
  (Storable t
  ,Typeable s
  ,Show s
  ,IsSingle t
  ,s ~ ScalarTensorDataR t
  ,TF.TensorType s
  ,TArrayDataR Sh.Tensor t ~ Sh.Tensor s
  ,GArrayDataR UniqueArray t ~ UniqueArray t
  ,s TF./= ByteString
  ,s TF./= Bool
  ,Convertable t s)

buildTypeDictsScalar :: forall t r. ScalarType t -> (forall s. TypeDictsFor t s => r) -> r
buildTypeDictsScalar (SingleScalarType t) f = buildTypeDictsSingle t f
buildTypeDictsScalar (VectorScalarType _) _ = unsupported "SIMD-vector types"

buildTypeDictsSingle :: forall t r. SingleType t -> (forall s. TypeDictsFor t s => r) -> r
buildTypeDictsSingle (NumSingleType t) f = buildTypeDictsNum t f

buildTypeDictsNum :: forall t r. NumType t -> (forall s. TypeDictsFor t s => r) -> r
buildTypeDictsNum (IntegralNumType t) f = buildTypeDictsIntegral t f
buildTypeDictsNum (FloatingNumType t) f = buildTypeDictsFloating t f

buildTypeDictsIntegral :: forall t r. IntegralType t -> (forall s. TypeDictsFor t s => r) -> r
buildTypeDictsIntegral TypeInt8   f = f
buildTypeDictsIntegral TypeInt16  f = f
buildTypeDictsIntegral TypeInt32  f = f
buildTypeDictsIntegral TypeInt64  f = f
buildTypeDictsIntegral TypeWord8  f = f
buildTypeDictsIntegral TypeWord16 f = f
buildTypeDictsIntegral TypeWord32 f = f
buildTypeDictsIntegral TypeWord64 f = f
buildTypeDictsIntegral TypeInt    f = f
buildTypeDictsIntegral TypeWord   f = f

buildTypeDictsFloating :: forall t r. FloatingType t -> (forall s. TypeDictsFor t s => r) -> r
buildTypeDictsFloating TypeFloat  f = f
buildTypeDictsFloating TypeDouble f = f
buildTypeDictsFloating TypeHalf   _ = unsupported "half-precision floating point"
