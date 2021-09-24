{-# LANGUAGE GADTs               #-}
{-# LANGUAGE LambdaCase          #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications    #-}
{-# OPTIONS_HADDOCK hide #-}
-- |
-- Module      : Data.Array.Accelerate.TensorFlow.CodeGen.Exp
-- Copyright   : [2021] The Accelerate Team
-- License     : BSD3
--
-- Maintainer  : Trevor L. McDonell <trevor.mcdonell@gmail.com>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--

module Data.Array.Accelerate.TensorFlow.CodeGen.Exp
  where

import Data.Array.Accelerate.TensorFlow.CodeGen.Arithmetic          as A
import Data.Array.Accelerate.TensorFlow.CodeGen.Base
import Data.Array.Accelerate.TensorFlow.CodeGen.Environment
import Data.Array.Accelerate.TensorFlow.CodeGen.Tensor

import Data.Array.Accelerate.AST
import Data.Array.Accelerate.AST.Var
import Data.Array.Accelerate.Type

import qualified TensorFlow.Ops                                     as TF


buildOpenExp
    :: forall env aenv sh t.
       TensorShape sh
    -> Val env
    -> Aval aenv
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
buildPrimFun f =
  case f of
    PrimAdd t               -> A.add t
    PrimSub t               -> A.sub t
    PrimMul t               -> A.mul t
    PrimNeg t               -> A.neg t
    PrimAbs t               -> A.abs t
    PrimSig t               -> A.signum t
    PrimQuot t              -> A.quot t
    PrimRem t               -> A.rem t
    PrimQuotRem t           -> A.quotRem t
    -- PrimIDiv     :: IntegralType a -> PrimFun ((a, a)   -> a)
    -- PrimMod      :: IntegralType a -> PrimFun ((a, a)   -> a)
    -- PrimDivMod   :: IntegralType a -> PrimFun ((a, a)   -> (a, a))
    PrimBAnd t              -> A.band t
    PrimBOr t               -> A.bor t
    PrimBXor t              -> A.xor t
    PrimBNot t              -> A.complement t
    PrimBShiftL t           -> A.shiftL t
    PrimBShiftR t           -> A.shiftR t
    -- PrimBRotateL           :: IntegralType a -> PrimFun ((a, Int) -> a)
    -- PrimBRotateR           :: IntegralType a -> PrimFun ((a, Int) -> a)
    PrimPopCount t          -> A.popCount t
    -- PrimCountLeadingZeros  :: IntegralType a -> PrimFun (a -> Int)
    -- PrimCountTrailingZeros :: IntegralType a -> PrimFun (a -> Int)
    PrimFDiv t              -> A.fdiv t
    PrimRecip t             -> A.recip t
    PrimSin t               -> A.sin t
    PrimCos t               -> A.cos t
    PrimTan t               -> A.tan t
    PrimAsin t              -> A.asin t
    PrimAcos t              -> A.acos t
    PrimAtan t              -> A.atan t
    PrimSinh t              -> A.sinh t
    PrimCosh t              -> A.cosh t
    PrimTanh t              -> A.tanh t
    PrimAsinh t             -> A.asinh t
    PrimAcosh t             -> A.acosh t
    PrimAtanh t             -> A.atanh t
    PrimExpFloating t       -> A.exp t
    PrimSqrt t              -> A.sqrt t
    PrimLog t               -> A.log t
    PrimFPow t              -> A.fpow t
    PrimLogBase t           -> A.logBase t
    PrimTruncate ta tb      -> A.truncate ta tb
    PrimRound ta tb         -> A.round ta tb
    PrimFloor ta tb         -> A.floor ta tb
    PrimCeiling ta tb       -> A.ceiling ta tb
    PrimAtan2 t             -> A.atan2 t
    PrimIsNaN t             -> A.isNaN t
    PrimIsInfinite t        -> A.isInfinite t
    PrimLt t                -> A.lt t
    PrimGt t                -> A.gt t
    PrimLtEq t              -> A.lte t
    PrimGtEq t              -> A.gte t
    PrimEq t                -> A.eq t
    PrimNEq t               -> A.neq t
    PrimMax t               -> A.max t
    PrimMin t               -> A.min t
    PrimLAnd                -> A.uncurry @PrimBool @PrimBool A.land
    PrimLOr                 -> A.uncurry @PrimBool @PrimBool A.lor
    PrimLNot                -> A.lnot
    PrimFromIntegral ta tb  -> A.fromIntegral ta tb
    PrimToFloating ta tb    -> A.toFloating ta tb

