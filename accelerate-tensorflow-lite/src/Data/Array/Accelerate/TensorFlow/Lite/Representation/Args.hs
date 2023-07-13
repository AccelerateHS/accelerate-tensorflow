{-# LANGUAGE FlexibleInstances    #-}
{-# LANGUAGE GADTs                #-}
{-# LANGUAGE RankNTypes           #-}
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

module Data.Array.Accelerate.TensorFlow.Lite.Representation.Args (

  Args(..),
  ArgsNames(..),
  ArrArgNames(..),
  ArgName(..),
  serialiseReprData,
  tagOfType,

) where

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

import Data.List                                                    ( genericLength )
import Foreign.ForeignPtr
import Foreign.Storable
import qualified Data.Set                                           as Set
import Data.Set                                                     ( Set )
import qualified Data.Text                                          as T

import Data.ByteString.Builder                                      ( Builder )
import qualified Data.ByteString.Builder                            as B
import qualified Data.ByteString.Internal                           as B


-- | Description, in representation types, of the arguments and the result shape
-- of the model.
data Args f where
  Aparam  :: ArraysR a -> a -> Args b -> Args (a -> b)
  Aresult :: ShapesR a -> Shapes a    -> Args a


-- | The name for a single TF input array.
newtype ArgName t = ArgName T.Text
  deriving (Show)

-- | This Accelerate argument (itself, due to SoA, consisting of potentially
-- multiple TF arrays) may have names for each of its tuple slices, or may be
-- skipped wholesale.
data ArrArgNames t where
  ArrArgNames :: TupR ArgName a -> ArrArgNames (Array sh a)
  ArrArgSkip  ::                   ArrArgNames (Array sh a)

-- | The names for the accelerate (SoA) input arrays in the TF program.
data ArgsNames f where
  ANparam  :: TupR ArrArgNames a -> ArgsNames b -> ArgsNames (a -> b)
  ANresult :: ArgsNames a


-- | Serialise a representative dataset (i.e. a list of argument sets) for converter.py.
serialiseReprData :: ArgsNames f -> Set T.Text -> [Args f] -> Builder
serialiseReprData argsnames actualInputs = buildList . map (buildArgs argsnames actualInputs)

buildArgs :: ArgsNames f -> Set T.Text -> Args f -> Builder
buildArgs argsnames actualInputs args = buildList (foldArgs buildArg args argsnames)
  where
    foldArgs :: Monoid s => (forall t. ArraysR t -> TupR ArrArgNames t -> t -> s) -> Args f -> ArgsNames f -> s
    foldArgs f (Aparam rep val rest) (ANparam names restnames) = f rep names val <> foldArgs f rest restnames
    foldArgs _ Aresult{} ANresult = mempty
    foldArgs _ _ _ = error "Insufficient entries in ArgsNames vector"

    buildArg :: ArraysR a -> TupR ArrArgNames a -> a -> [Builder]
    buildArg TupRunit                         TupRunit                         ()               = []
    buildArg (TupRpair aR bR)                 (TupRpair ns1 ns2)               (a, b)           = buildArg aR ns1 a ++ buildArg bR ns2 b
    buildArg _                                (TupRsingle ArrArgSkip)          _                = []
    buildArg (TupRsingle (ArrayR shR adataR)) (TupRsingle (ArrArgNames names)) (Array sh adata) =
      let shapeSizeR :: ShapeR sh -> sh -> Int
          shapeSizeR ShapeRz           ()       = 1
          shapeSizeR (ShapeRsnoc shR') (sh', n) = n * shapeSizeR shR' sh'

          buildArrayData :: TypeR e -> ArrayData e -> TupR ArgName e -> [Builder]
          buildArrayData TupRunit         ()     _                           = []
          buildArrayData (TupRpair aR bR) (a, b) (TupRpair ns1 ns2)          = buildArrayData aR a ns1 ++ buildArrayData bR b ns2
          buildArrayData (TupRsingle aR)  a      (TupRsingle (ArgName name))
            | name `Set.member` actualInputs = [B.word8 (tagOfType aR) <> buildTypeDictsScalar aR (wrap a)]
            | otherwise                      = []
          buildArrayData _ _ _ = error "impossible"

          wrap :: forall a. Storable a => UniqueArray a -> Builder
          wrap (unsafeGetValue . uniqueArrayData -> fp)
            = B.byteString
            $ B.fromForeignPtr (castForeignPtr fp) 0 (size shR sh * sizeOf (undefined::a))
      in [B.word64LE (fromIntegral (shapeSizeR shR sh)) <> buildList (buildArrayData adataR adata names)]

buildList :: [Builder] -> Builder
buildList builders = B.word64LE (genericLength builders) <> mconcat builders


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
