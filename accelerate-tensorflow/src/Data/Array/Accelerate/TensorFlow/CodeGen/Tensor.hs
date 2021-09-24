{-# LANGUAGE GADTs                 #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE FlexibleInstances     #-}
{-# OPTIONS_GHC -fno-warn-orphans #-}
{-# OPTIONS_HADDOCK hide #-}
-- |
-- Module      : Data.Array.Accelerate.TensorFlow.CodeGen.Tensor
-- Copyright   : [2021] The Accelerate Team
-- License     : BSD3
--
-- Maintainer  : Trevor L. McDonell <trevor.mcdonell@gmail.com>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--

module Data.Array.Accelerate.TensorFlow.CodeGen.Tensor
  where

import Data.Array.Accelerate.TensorFlow.CodeGen.Base

import Data.Array.Accelerate.Array.Data
import Data.Array.Accelerate.Array.Unique
import Data.Array.Accelerate.Representation.Array
import Data.Array.Accelerate.Representation.Shape
import Data.Array.Accelerate.Representation.Type
import Data.Array.Accelerate.Type

import qualified TensorFlow.Core                                    as TF
import qualified TensorFlow.Nodes                                   as TF
import qualified TensorFlow.Output                                  as TF
import qualified TensorFlow.Types                                   as TF
import qualified TensorFlow.Internal.FFI                            as TF

import Control.Applicative                                          ( liftA, liftA2 )
import Data.Set                                                     ( Set )
import Foreign.ForeignPtr
import Foreign.Storable
import System.IO.Unsafe
import qualified Data.Set                                           as Set
import qualified Data.Vector.Storable                               as V


type TensorShape sh    = TF.Tensor TF.Build Int64
type TensorArrayData e = GArrayDataR (TF.Tensor TF.Build) e

data Tensor sh e where
  Tensor :: ArrayR (Array sh e)
         -> TensorShape sh
         -> TensorArrayData e
         -> Tensor sh e

type family Tensors t where
  Tensors ()           = ()
  Tensors (Array sh e) = Tensor sh e
  Tensors (a, b)       = (Tensors a, Tensors b)

instance TF.Nodes (Tensor sh e) where
  getNodes (Tensor (ArrayR _ adataR) sh adata) = TF.nodesUnion [ TF.getNodes sh, go adataR adata ]
    where
      go :: TypeR t -> TensorArrayData t -> TF.Build (Set TF.NodeName)
      go TupRunit ()             = return Set.empty
      go (TupRpair aR bR) (a, b) = TF.nodesUnion [ go aR a, go bR b ]
      go (TupRsingle aR) a       = scalar aR a

      scalar :: ScalarType t -> TensorArrayData t -> TF.Build (Set TF.NodeName)
      scalar (SingleScalarType t) = single t
      scalar (VectorScalarType _) = unsupported "SIMD-vector types"

      single :: SingleType t -> TensorArrayData t -> TF.Build (Set TF.NodeName)
      single (NumSingleType t) = num t

      num :: NumType t -> TensorArrayData t -> TF.Build (Set TF.NodeName)
      num (IntegralNumType t) = integral t
      num (FloatingNumType t) = floating t

      integral :: IntegralType t -> TensorArrayData t -> TF.Build (Set TF.NodeName)
      integral TypeInt8   = TF.getNodes
      integral TypeInt16  = TF.getNodes
      integral TypeInt32  = TF.getNodes
      integral TypeInt64  = TF.getNodes
      integral TypeWord8  = TF.getNodes
      integral TypeWord16 = TF.getNodes
      integral TypeWord32 = TF.getNodes
      integral TypeWord64 = TF.getNodes
      integral TypeInt    = unsupported "Int (use at a specified bit-size instead)"
      integral TypeWord   = unsupported "Word (use at a specified bit-size instead)"

      floating :: FloatingType t -> TensorArrayData t -> TF.Build (Set TF.NodeName)
      floating TypeFloat  = TF.getNodes
      floating TypeDouble = TF.getNodes
      floating TypeHalf   = unsupported "half-precision floating point"

instance TF.Fetchable (Tensor sh e) (Array sh e) where
  getFetch (Tensor (ArrayR _shR _adataR) _sh _adata) =
    liftA2 Array <$> fetchShape _shR _sh <*> fetchArray _adataR _adata
    where
      fetchShape :: ShapeR sh -> TensorShape sh -> TF.Build (TF.Fetch sh)
      fetchShape shR sh = liftA (listToShape shR . map fromIntegral . V.toList) <$> TF.getFetch sh

      fetchArray :: TypeR t -> TensorArrayData t -> TF.Build (TF.Fetch (ArrayData t))
      fetchArray TupRunit ()             = pure (pure ())
      fetchArray (TupRpair aR bR) (a, b) = liftA2 (,) <$> fetchArray aR a <*> fetchArray bR b
      fetchArray (TupRsingle aR) a       = scalar aR a
        where
          wrap :: (Storable t, TF.TensorType t) => TF.Tensor TF.Build t -> TF.Build (TF.Fetch (UniqueArray t))
          wrap tensor = do
            tdata <- TF.fetchTensorVector tensor
            let vector  = TF.tensorDataBytes . TF.unTensorData <$> tdata
                fp      = fst . V.unsafeToForeignPtr0 <$> vector
                ua      = unsafePerformIO . newUniqueArray . castForeignPtr <$> fp
            --
            return ua

          scalar :: ScalarType t -> TensorArrayData t -> TF.Build (TF.Fetch (ArrayData t))
          scalar (SingleScalarType t) = single t
          scalar (VectorScalarType _) = unsupported "SIMD-vector types"

          single :: SingleType t -> TensorArrayData t -> TF.Build (TF.Fetch (ArrayData t))
          single (NumSingleType t) = num t

          num :: NumType t -> TensorArrayData t -> TF.Build (TF.Fetch (ArrayData t))
          num (IntegralNumType t) = integral t
          num (FloatingNumType t) = floating t

          integral :: IntegralType t -> TensorArrayData t -> TF.Build (TF.Fetch (ArrayData t))
          integral TypeInt8   = wrap
          integral TypeInt16  = wrap
          integral TypeInt32  = wrap
          integral TypeInt64  = wrap
          integral TypeWord8  = wrap
          integral TypeWord16 = wrap
          integral TypeWord32 = wrap
          integral TypeWord64 = wrap
          integral TypeInt    = unsupported "Int (use at a specified bit-size instead)"
          integral TypeWord   = unsupported "Word (use at a specified bit-size instead)"

          floating :: FloatingType t -> TensorArrayData t -> TF.Build (TF.Fetch (ArrayData t))
          floating TypeFloat  = wrap
          floating TypeDouble = wrap
          floating TypeHalf   = unsupported "half-precision floating point"

instance TF.Nodes () where
  getNodes () = return Set.empty

instance TF.Fetchable () () where
  getFetch () = pure (pure ())

