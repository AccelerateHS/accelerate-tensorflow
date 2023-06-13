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
import Data.Array.Accelerate.TensorFlow.TypeDicts

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
        let sh_ = shapeToTensor contextR context

            go :: TypeR s -> TensorArrayData s -> TensorArrayData s
            go TupRunit         ()     = ()
            go (TupRpair aR bR) (a, b) = (go aR a, go bR b)
            go (TupRsingle aR)  a      = buildTypeDictsScalar aR (TF.fill sh_) a
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
            go (TupRsingle eR)  a        b        = buildTypeDictsScalar eR (TF.selectV2 p') a b
        in
        go (expType t) t' e'

      shapeL :: Tensor sh e -> TensorArrayData sh
      shapeL (Tensor (ArrayR shR' _) sh' _) = fillL (shapeType shR') sh'

      shapeSizeL :: ShapeR sh -> TensorArrayData sh -> TensorArrayData Int
      shapeSizeL = shapeToTensor
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


shapeToTensor
    :: (s ~ ScalarTensorDataR Int)
    => ShapeR sh
    -> TensorShape sh
    -> TF.Tensor TF.Build s
shapeToTensor ShapeRz              ()       = TF.constant (TF.Shape [1]) [1]
shapeToTensor (ShapeRsnoc ShapeRz) ((), sh) = sh
shapeToTensor shR                  sh       =
  let
      go :: (s ~ ScalarTensorDataR Int) => ShapeR sh -> TensorShape sh -> [TF.Tensor TF.Build s] -> [TF.Tensor TF.Build s]
      go ShapeRz         ()     acc = acc
      go (ShapeRsnoc tR) (t, h) acc = go tR t (h : acc)
  in
  -- XXX: Why is this reshape necessary?
  TF.concat (TF.scalar 0) [ TF.reshape x (TF.constant (TF.Shape [1]) [1 :: ScalarTensorDataR Int]) | x <- go shR sh [] ]

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
constant shR eR sh =
  let sh_ = shapeToTensor shR sh
  in buildTypeDictsScalar eR (TF.fill sh_ . TF.scalar . convertConvertable)

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

