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
import Data.Array.Accelerate.Representation.Array
import Data.Array.Accelerate.Representation.Shape
import Data.Array.Accelerate.Representation.Type
import Data.Array.Accelerate.Type

import qualified TensorFlow.Core                                    as TF
import qualified TensorFlow.Ops                                     as TF
import qualified TensorFlow.GenOps.Core                             as TF

import Prelude                                                      as P


buildOpenExp
    :: forall env aenv context t.
       ShapeR context
    -> TensorShape context
    -> Val env
    -> Aval aenv
    -> OpenExp env aenv t
    -> TensorArrayData t
buildOpenExp contextR context env aenv =
  let
      buildE :: OpenExp env aenv s -> TensorArrayData s
      buildE = buildOpenExp contextR context env aenv

      fillL :: TypeR s -> TensorArrayData s -> TensorArrayData s
      fillL =
        let sh = tensorShape contextR context

            go :: TypeR s -> TensorArrayData s -> TensorArrayData s
            go TupRunit         ()     = ()
            go (TupRpair aR bR) (a, b) = (go aR a, go bR b)
            go (TupRsingle aR)  a      = scalar aR a

            scalar :: ScalarType s -> TensorArrayData s -> TensorArrayData s
            scalar (SingleScalarType t) = single t
            scalar (VectorScalarType _) = unsupported "vector types"

            single :: SingleType s -> TensorArrayData s -> TensorArrayData s
            single (NumSingleType t) = num t

            num :: NumType s -> TensorArrayData s -> TensorArrayData s
            num (IntegralNumType t) = integral t
            num (FloatingNumType t) = floating t

            integral :: IntegralType s -> TensorArrayData s -> TensorArrayData s
            integral TypeInt8   = TF.fill sh
            integral TypeInt16  = TF.fill sh
            integral TypeInt32  = TF.fill sh
            integral TypeInt64  = TF.fill sh
            integral TypeWord8  = TF.fill sh
            integral TypeWord16 = TF.fill sh
            integral TypeWord32 = TF.fill sh
            integral TypeWord64 = TF.fill sh
            integral TypeInt    = TF.fill sh
            integral TypeWord   = TF.fill sh

            floating :: FloatingType s -> TensorArrayData s -> TensorArrayData s
            floating TypeFloat  = TF.fill sh
            floating TypeDouble = TF.fill sh
            floating TypeHalf   = unsupported "half-precision floating point"
        in
        go

      condL :: OpenExp env aenv PrimBool
            -> OpenExp env aenv t
            -> OpenExp env aenv t
            -> TensorArrayData t
      condL p t e =
        let p' = TF.cast @_ @_ @Bool $ buildE p
            t' = buildE t
            e' = buildE e

            go :: TypeR s -> TensorArrayData s -> TensorArrayData s -> TensorArrayData s
            go TupRunit         ()       ()       = ()
            go (TupRpair tA tB) (a1, b1) (a2, b2) = (go tA a1 a2, go tB b1 b2)
            go (TupRsingle eR)  a        b        = scalar eR a b

            scalar :: ScalarType s -> TensorArrayData s -> TensorArrayData s -> TensorArrayData s
            scalar (SingleScalarType s) = single s
            scalar (VectorScalarType _) = unsupported "SIMD-vector types"

            single :: SingleType s -> TensorArrayData s -> TensorArrayData s -> TensorArrayData s
            single (NumSingleType s) = num s

            num :: NumType s -> TensorArrayData s -> TensorArrayData s -> TensorArrayData s
            num (IntegralNumType s) = integral s
            num (FloatingNumType s) = floating s

            integral :: IntegralType s -> TensorArrayData s -> TensorArrayData s -> TensorArrayData s
            integral TypeInt    = TF.selectV2 p'
            integral TypeInt8   = TF.selectV2 p'
            integral TypeInt16  = TF.selectV2 p'
            integral TypeInt32  = TF.selectV2 p'
            integral TypeInt64  = TF.selectV2 p'
            integral TypeWord   = TF.selectV2 p'
            integral TypeWord8  = TF.selectV2 p'
            integral TypeWord16 = TF.selectV2 p'
            integral TypeWord32 = TF.selectV2 p'
            integral TypeWord64 = TF.selectV2 p'

            floating :: FloatingType s -> TensorArrayData s -> TensorArrayData s -> TensorArrayData s
            floating TypeFloat  = TF.selectV2 p'
            floating TypeDouble = TF.selectV2 p'
            floating TypeHalf   = unsupported "half-precision floating point"
        in
        go (expType t) t' e'

      shapeL :: Tensor sh e -> TensorArrayData sh
      shapeL (Tensor (ArrayR shR' _) sh' _) = fillL (shapeType shR') sh'

      shapeSizeL :: ShapeR sh -> TensorArrayData sh -> TensorArrayData Int
      shapeSizeL = tensorShape
  in
  \case
    Let lhs bnd body              -> buildOpenExp contextR context (env `push` (lhs, buildE bnd)) aenv body
    Evar (Var _ ix)               -> prj ix env
    -- Foreign tR asm f x            -> undefined
    Pair x y                      -> (buildE x, buildE y)
    Nil                           -> ()
    VecPack{}                     -> unsupported "SIMD-vector types"
    VecUnpack{}                   -> unsupported "SIMD-vector types"
    -- IndexSlice sliceIndex slix sh -> undefined
    -- IndexFull sliceIndex slix sl  -> undefined
    -- ToIndex shR sh ix             -> undefined
    -- FromIndex shR sh i            -> undefined
    -- Case tag xs x                 -> undefined
    Cond p t e                    -> condL p t e
    -- While p f x                   -> undefined
    Const tR c                    -> constant contextR tR context c
    PrimConst x                   -> primConst contextR context x
    PrimApp f x                   -> primFun f (buildE x)
    -- Index v ix                    -> undefined
    -- LinearIndex v ix              -> undefined
    Shape (Var _ ix)              -> shapeL (aprj ix aenv)
    ShapeSize shR sh              -> shapeSizeL shR (buildE sh)
    -- Undef t                       -> undefined
    -- Coerce tA tB a                -> undefined


tensorShape
    :: (s ~ ScalarTensorDataR Int)
    => ShapeR sh
    -> TensorShape sh
    -> TF.Tensor TF.Build s
tensorShape ShapeRz              ()       = TF.scalar 1
tensorShape (ShapeRsnoc ShapeRz) ((), sh) = sh
tensorShape shR                  sh       =
  let
      go :: (s ~ ScalarTensorDataR Int) => ShapeR sh -> TensorShape sh -> [TF.Tensor TF.Build s]
      go ShapeRz         ()     = []
      go (ShapeRsnoc tR) (t, h) = go tR t ++ [TF.reshape h (TF.scalar (1 :: Int32))]
  in
  TF.concat (TF.scalar 0) (go shR sh)

primConst
    :: ShapeR sh
    -> TensorShape sh
    -> PrimConst t
    -> TensorArrayData t
primConst shR sh (PrimPi t)
  | FloatingDict <- floatingDict t
  = constant shR (SingleScalarType (NumSingleType (FloatingNumType t))) sh pi
primConst shR sh (PrimMinBound (IntegralBoundedType t))
  | IntegralDict <- integralDict t
  = constant shR (SingleScalarType (NumSingleType (IntegralNumType t))) sh minBound
primConst shR sh (PrimMaxBound (IntegralBoundedType t))
  | IntegralDict <- integralDict t
  = constant shR (SingleScalarType (NumSingleType (IntegralNumType t))) sh maxBound

constant
    :: ShapeR sh
    -> ScalarType t
    -> TensorShape sh
    -> t
    -> TensorArrayData t
constant shR eR sh = scalar eR
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
    integral TypeInt8   = TF.fill (tensorShape shR sh) . TF.scalar
    integral TypeInt16  = TF.fill (tensorShape shR sh) . TF.scalar
    integral TypeInt32  = TF.fill (tensorShape shR sh) . TF.scalar
    integral TypeInt64  = TF.fill (tensorShape shR sh) . TF.scalar
    integral TypeWord8  = TF.fill (tensorShape shR sh) . TF.scalar
    integral TypeWord16 = TF.fill (tensorShape shR sh) . TF.scalar
    integral TypeWord32 = TF.fill (tensorShape shR sh) . TF.scalar
    integral TypeWord64 = TF.fill (tensorShape shR sh) . TF.scalar
    integral TypeInt    = TF.fill (tensorShape shR sh) . TF.scalar . P.fromIntegral
    integral TypeWord   = TF.fill (tensorShape shR sh) . TF.scalar . P.fromIntegral

    floating :: FloatingType t -> t -> TensorArrayData t
    floating TypeFloat  = TF.fill (tensorShape shR sh) . TF.scalar
    floating TypeDouble = TF.fill (tensorShape shR sh) . TF.scalar
    floating TypeHalf   = unsupported "half-precision floating point"

primFun
    :: forall a b.
       PrimFun (a -> b)
    -> TensorArrayData a
    -> TensorArrayData b
primFun f =
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

