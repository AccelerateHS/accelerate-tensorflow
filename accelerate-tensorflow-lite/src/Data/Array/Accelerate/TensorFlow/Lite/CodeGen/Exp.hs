{-# LANGUAGE GADTs               #-}
{-# LANGUAGE LambdaCase          #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications    #-}
-- |
-- Module      : Data.Array.Accelerate.TensorFlow.Lite.CodeGen.Exp
-- Copyright   : [2021] The Accelerate Team
-- License     : BSD3
--
-- Maintainer  : Trevor L. McDonell <trevor.mcdonell@gmail.com>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--

module Data.Array.Accelerate.TensorFlow.Lite.CodeGen.Exp
  where

import Data.Array.Accelerate.TensorFlow.Lite.CodeGen.Base
import Data.Array.Accelerate.TensorFlow.Lite.CodeGen.Tensor
import Data.Array.Accelerate.TensorFlow.Lite.CodeGen.Environment
import Data.Array.Accelerate.TensorFlow.Lite.CodeGen.Arithmetic     as A

import Data.Array.Accelerate.AST
import Data.Array.Accelerate.AST.Var
import Data.Array.Accelerate.Type

import qualified TensorFlow.Ops                                     as TF


buildOpenExp
    :: forall env aenv sh t.
       TensorShape sh
    -> Val env
    -> Val aenv
    -> OpenExp env aenv t
    -> TensorArrayData t
buildOpenExp sh env aenv =
  let
      buildE :: OpenExp env aenv s -> TensorArrayData s
      buildE = buildOpenExp sh env aenv
  in
  \case
    Let lhs bnd body              -> buildOpenExp sh (env `push` (lhs, buildE bnd)) aenv body
    Evar (Var _ ix)               -> prj ix env
    -- Foreign tR asm f x            -> undefined
    Pair x y                      -> (buildE x, buildE y)
    -- Nil                           -> undefined
    -- VecPack vR x                  -> undefined
    -- VecUnpack vR x                -> undefined
    -- IndexSlice sliceIndex slix sh -> undefined
    -- IndexFull sliceIndex slix sl  -> undefined
    -- ToIndex shR sh ix             -> undefined
    -- FromIndex shR sh i            -> undefined
    -- Case tag xs x                 -> undefined
    -- Cond p t e                    -> undefined
    -- While p f x                   -> undefined
    Const t c                     -> buildConst sh t c
    PrimConst x                   -> buildPrimConst sh x
    PrimApp f x                   -> buildPrimFun f (buildE x)
    -- Index v ix                    -> undefined
    -- LinearIndex v ix              -> undefined
    -- Shape v                       -> undefined
    -- ShapeSize shR sh              -> undefined
    -- Undef t                       -> undefined
    -- Coerce tA tB a                -> undefined


buildPrimConst
    :: TensorShape sh
    -> PrimConst t
    -> TensorArrayData t
buildPrimConst sh (PrimPi t)
  | FloatingDict <- floatingDict t
  = buildConst sh (SingleScalarType (NumSingleType (FloatingNumType t))) pi
buildPrimConst sh (PrimMinBound (IntegralBoundedType t))
  | IntegralDict <- integralDict t
  = buildConst sh (SingleScalarType (NumSingleType (IntegralNumType t))) minBound
buildPrimConst sh (PrimMaxBound (IntegralBoundedType t))
  | IntegralDict <- integralDict t
  = buildConst sh (SingleScalarType (NumSingleType (IntegralNumType t))) maxBound


buildConst
    :: TensorShape sh
    -> ScalarType t
    -> t
    -> TensorArrayData t
buildConst sh = scalar
  where
    scalar :: ScalarType t -> t -> TensorArrayData t
    scalar (SingleScalarType t) = single t
    scalar (VectorScalarType _) = unsupported "vector types"

    single :: SingleType t -> t -> TensorArrayData t
    single (NumSingleType t) = num t

    num :: NumType t -> t -> TensorArrayData t
    num (IntegralNumType t) = integral t
    num (FloatingNumType t) = floating t

    integral :: IntegralType t -> t -> TensorArrayData t
    integral TypeInt8   = TF.fill sh . TF.scalar
    integral TypeInt16  = TF.fill sh . TF.scalar
    integral TypeInt32  = TF.fill sh . TF.scalar
    integral TypeInt64  = TF.fill sh . TF.scalar
    integral TypeWord8  = TF.fill sh . TF.scalar
    integral TypeWord16 = TF.fill sh . TF.scalar
    integral TypeWord32 = TF.fill sh . TF.scalar
    integral TypeWord64 = TF.fill sh . TF.scalar
    integral TypeInt    = unsupported "Int (use at a specified bit-size instead)"
    integral TypeWord   = unsupported "Word (use at a specified bit-size instead)"

    floating :: FloatingType t -> t -> TensorArrayData t
    floating TypeFloat  = TF.fill sh . TF.scalar
    floating TypeDouble = TF.fill sh . TF.scalar
    floating TypeHalf   = unsupported "half-precision floating point"


buildPrimFun
    :: forall a b.
       PrimFun (a -> b)
    -> TensorArrayData a
    -> TensorArrayData b
buildPrimFun f adata =
  case f of
    PrimAdd t     -> A.uncurry @b @b @(TensorArrayData b) (A.add t) adata
    PrimMul t     -> A.uncurry @b @b @(TensorArrayData b) (A.mul t) adata

