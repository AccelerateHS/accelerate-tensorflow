{-# LANGUAGE AllowAmbiguousTypes  #-}
{-# LANGUAGE FlexibleInstances    #-}
{-# LANGUAGE GADTs                #-}
{-# LANGUAGE ScopedTypeVariables  #-}
{-# LANGUAGE TypeApplications     #-}
{-# LANGUAGE TypeFamilies         #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE ViewPatterns         #-}
-- |
-- Module      : Data.Array.Accelerate.TensorFlow.Lite.Representation.Args
-- Copyright   : [2021..2022] The Accelerate Team
-- License     : BSD3
--
-- Maintainer  : Trevor L. McDonell <trevor.mcdonell@gmail.com>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--

module Data.Array.Accelerate.TensorFlow.Lite.Representation.Args
  where

import Data.Array.Accelerate.TensorFlow.CodeGen.Base
import Data.Array.Accelerate.TensorFlow.TypeDicts
import Data.Array.Accelerate.TensorFlow.Lite.Representation.Shapes

import Data.Array.Accelerate.Array.Data
import Data.Array.Accelerate.Array.Unique
import Data.Array.Accelerate.Lifetime
import Data.Array.Accelerate.Representation.Array
import Data.Array.Accelerate.Representation.Shape
import Data.Array.Accelerate.Representation.Type
import Data.Array.Accelerate.Type

import Foreign.ForeignPtr
import Foreign.Storable

import Data.ByteString.Builder                                      ( Builder )
import qualified Data.ByteString.Builder                            as B
import qualified Data.ByteString.Builder.Extra                      as B
import qualified Data.ByteString.Internal                           as B


data Args f where
  Aparam  :: ArraysR a -> a -> Args b -> Args (a -> b)
  Aresult :: ShapesR a -> Shapes a    -> Args a


buildArgs :: Args f -> Builder
buildArgs args =
  let
      count :: Args a -> Word8
      count Aresult{}          = 0
      count (Aparam arrR _ xs) = go arrR + count xs
        where
          go :: ArraysR a -> Word8
          go TupRunit         = 0
          go TupRsingle{}     = 1
          go (TupRpair aR bR) = go aR + go bR

      tensors :: Args a -> Builder
      tensors Aresult{}              = mempty
      tensors (Aparam arrR arr next) = go arrR arr <> tensors next
        where
          go :: ArraysR a -> a -> Builder
          go TupRunit                         ()               = mempty
          go (TupRpair aR bR)                 (a, b)           = go aR a <> go bR b
          go (TupRsingle (ArrayR shR adataR)) (Array sh adata) =
            let
                count' :: TypeR e -> Word8
                count' TupRunit         = 0
                count' TupRsingle{}     = 1
                count' (TupRpair aR bR) = count' aR + count' bR

                buildShape :: ShapeR sh -> sh -> Builder
                buildShape ShapeRz         ()     = mempty
                buildShape (ShapeRsnoc tR) (t, h) = B.int64Host (fromIntegral h) <> buildShape tR t

                buildArrayData :: TypeR e -> ArrayData e -> Builder
                buildArrayData TupRunit         ()     = mempty
                buildArrayData (TupRpair aR bR) (a, b) = buildArrayData aR a <> buildArrayData bR b
                buildArrayData (TupRsingle aR)  a      = B.word8 (tagOfType aR) <> buildTypeDictsScalar aR wrap a

                wrap :: forall a. Storable a => UniqueArray a -> Builder
                wrap (unsafeGetValue . uniqueArrayData -> fp)
                  = B.byteString
                  $ B.fromForeignPtr (castForeignPtr fp) 0 (size shR sh * sizeOf (undefined::a))
            in
            B.word8 (fromIntegral (rank shR))
            <> buildShape shR sh
            <> B.word8 (count' adataR)
            <> buildArrayData adataR adata
  in
  B.word8 (count args) <> tensors args


-- NOTE: This must match what is expected in converter.py and edgetpu.cc
--
tagOfType :: ScalarType t -> Word8
tagOfType = scalar
  where
    scalar :: ScalarType t -> Word8
    scalar (VectorScalarType _) = unsupported "SIMD-vector types"
    scalar (SingleScalarType t) = single t

    single :: SingleType t -> Word8
    single (NumSingleType t) = num t

    num :: NumType a -> Word8
    num (IntegralNumType t) = integral t
    num (FloatingNumType t) = floating t

    integral :: IntegralType a -> Word8
    integral TypeInt8   = 0
    integral TypeInt16  = 1
    integral TypeInt32  = 2
    integral TypeInt64  = 3
    integral TypeWord8  = 4
    integral TypeWord16 = 5
    integral TypeWord32 = 6
    integral TypeWord64 = 7
    integral TypeInt    = case sizeOf (undefined::Int) of
                            1 -> 0
                            2 -> 1
                            4 -> 2
                            8 -> 3
                            _ -> error "Bones mend. Regrets stays with you forever. --Patrick Rothfuss, The Name of the Wind"
    integral TypeWord   = case sizeOf (undefined::Word) of
                            1 -> 4
                            2 -> 5
                            4 -> 6
                            8 -> 7
                            _ -> error "Only a fool worries over what he can't control. --Patrick Rothfuss, The Wise Man's Fear"

    floating :: FloatingType a -> Word8
    floating TypeHalf   = 8
    floating TypeFloat  = 9
    floating TypeDouble = 10

