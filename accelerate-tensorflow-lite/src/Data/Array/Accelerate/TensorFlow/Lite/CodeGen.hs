{-# LANGUAGE GADTs               #-}
{-# LANGUAGE LambdaCase          #-}
{-# LANGUAGE OverloadedStrings   #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications    #-}
-- |
-- Module      : Data.Array.Accelerate.TensorFlow.Lite.CodeGen
-- Copyright   : [2021] The Accelerate Team
-- License     : BSD3
--
-- Maintainer  : Trevor L. McDonell <trevor.mcdonell@gmail.com>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--

module Data.Array.Accelerate.TensorFlow.Lite.CodeGen
  where

import Data.Array.Accelerate.TensorFlow.Lite.CodeGen.Base
import Data.Array.Accelerate.TensorFlow.Lite.CodeGen.Environment
import Data.Array.Accelerate.TensorFlow.Lite.CodeGen.Exp
import Data.Array.Accelerate.TensorFlow.Lite.CodeGen.Tensor

import Data.Array.Accelerate.AST
import Data.Array.Accelerate.Array.Data
import Data.Array.Accelerate.Array.Unique
import Data.Array.Accelerate.Lifetime
import Data.Array.Accelerate.Representation.Array
import Data.Array.Accelerate.Representation.Shape
import Data.Array.Accelerate.Representation.Type
import Data.Array.Accelerate.Type

import Lens.Family2
import Data.ProtoLens.Default                                       ( def )
import qualified Proto.Tensorflow.Core.Framework.Tensor             as TF
import qualified Proto.Tensorflow.Core.Framework.TensorShape_Fields as TensorShape
import qualified Proto.Tensorflow.Core.Framework.Tensor_Fields      as TF
import qualified TensorFlow.Core                                    as TF
import qualified TensorFlow.GenOps.Core                             as TF
import qualified TensorFlow.Ops                                     as TF
import qualified TensorFlow.Types                                   as TF

import Data.ByteString.Internal                                     as B
import Foreign.ForeignPtr
import Foreign.Storable


buildAcc :: Acc (Array sh e) -> Tensor sh e
buildAcc acc = buildOpenAcc Empty acc

buildOpenAcc
    :: forall aenv sh e.
       Val aenv
    -> OpenAcc aenv (Array sh e)
    -> Tensor sh e
buildOpenAcc aenv (OpenAcc pacc) =
  let
      buildA :: OpenAcc aenv (Array sh' e') -> Tensor sh' e'
      buildA = buildOpenAcc aenv

      useL :: ArrayR (Array sh e)
           -> Array sh e
           -> Tensor sh e
      useL (ArrayR shR adataR) (Array sh adata) =
        let
            go :: TypeR t -> ArrayData t -> TensorArrayData t
            go TupRunit ()             = ()
            go (TupRpair aR bR) (a, b) = (go aR a, go bR b)
            go (TupRsingle aR) a       = scalar aR a
              where
                tensor :: forall t. (Storable t, TF.TensorType t) => UniqueArray t -> TF.Tensor TF.Build t
                tensor ua =
                  let fp     = unsafeGetValue (uniqueArrayData ua)
                      values = B.fromForeignPtr (castForeignPtr fp) 0 (size shR sh * sizeOf (undefined :: t))

                      node :: TF.TensorProto
                      node = def
                           & TF.dtype .~ TF.tensorType (undefined :: t)
                           & TF.tensorShape.TensorShape.dim .~ [ def & TensorShape.size .~ fromIntegral x | x <- shapeToList shR sh ]
                           & TF.tensorContent .~ values
                  in
                  TF.const' (TF.opAttr "value" .~ node)

                scalar :: ScalarType t -> ArrayData t -> TensorArrayData t
                scalar (SingleScalarType t) = single t
                scalar (VectorScalarType _) = unsupported "SIMD-vector types"

                single :: SingleType t -> ArrayData t -> TensorArrayData t
                single (NumSingleType t) = num t

                num :: NumType t -> ArrayData t -> TensorArrayData t
                num (IntegralNumType t) = integral t
                num (FloatingNumType t) = floating t

                integral :: IntegralType t -> ArrayData t -> TensorArrayData t
                integral TypeInt8   = tensor
                integral TypeInt16  = tensor
                integral TypeInt32  = tensor
                integral TypeInt64  = tensor
                integral TypeWord8  = tensor
                integral TypeWord16 = tensor
                integral TypeWord32 = tensor
                integral TypeWord64 = tensor
                integral TypeInt    = unsupported "Int (use at a specified bit-size instead)"
                integral TypeWord   = unsupported "Word (use at a specified bit-size instead)"

                floating :: FloatingType t -> ArrayData t -> TensorArrayData t
                floating TypeFloat  = tensor
                floating TypeDouble = tensor
                floating TypeHalf   = unsupported "half-precision floating point"

            adata' = go adataR adata
            sh'    = TF.constant (TF.Shape [fromIntegral (rank shR)]) [ fromIntegral x | x <- shapeToList shR sh ]
        in
        Tensor (ArrayR shR adataR) sh' adata'

      mapL :: TypeR b -> Fun aenv (a -> b) -> OpenAcc aenv (Array sh a) -> Tensor sh b
      mapL bR (Lam lhs (Body b)) xs =
        let Tensor (ArrayR shR _) sh xs' = buildA xs
            xs''                         = buildOpenExp sh (Empty `push` (lhs, xs')) aenv b
        in
        Tensor (ArrayR shR bR) sh xs''
      mapL _ _ _ = error "impossible"
  in
  case pacc of
    Use aR xs                         -> useL aR xs
    Map bR f xs                       -> mapL bR f xs

