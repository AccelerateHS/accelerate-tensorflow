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
import qualified Data.Array.Accelerate.TensorFlow.CodeGen.Tensor.Shim as Sh
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
            go (TupRsingle aR)  a      = buildTypeDictsScalar aR $ Sh.wrap "fill" TF.fill sh_ a
        in
        go

      condL :: OpenExp env aenv PrimBool
            -> OpenExp env aenv t
            -> OpenExp env aenv t
            -> TensorArrayData t
      condL p t e =
        let p' = Sh.wrap "cast" (TF.cast @_ @_ @Bool) $ buildE p
            t' = buildE t
            e' = buildE e

            go :: TypeR s -> TensorArrayData s -> TensorArrayData s -> TensorArrayData s
            go TupRunit         ()       ()       = ()
            go (TupRpair tA tB) (a1, b1) (a2, b2) = (go tA a1 a2, go tB b1 b2)
            go (TupRsingle eR)  a        b        = buildTypeDictsScalar eR $ Sh.wrap "selectV2" TF.selectV2 p' a b
        in
        go (expType t) t' e'

      shapeL :: Tensor sh e -> TensorArrayData sh
      shapeL (Tensor (ArrayR shR' _) sh' _) = fillL (shapeType shR') sh'

      shapeSizeL :: ShapeR sh -> TensorArrayData sh -> TensorArrayData Int
      shapeSizeL ShapeRz () = constant contextR scalarTypeInt context 1
      shapeSizeL (ShapeRsnoc shr) (sh, n) = A.mul (IntegralNumType TypeInt) (n, shapeSizeL shr sh)

      gatherL :: Tensor sh e -> TensorArrayData Int -> TensorArrayData e
      gatherL (Tensor (ArrayR shr t) sh p) ix = case t of
        TupRsingle t ->
          buildTypeDictsScalar t $
            let n = shapeProd shr sh  -- 0D tensor containing full size of p
                n1 = Sh.wrap "reshape" TF.reshape n (Sh.wrap2 "constant" TF.constant (TF.Shape [1]) [1 :: ScalarTensorDataR Int])
                pflattened = Sh.wrap "reshape" TF.reshape p n1
            in Sh.wrap "reshape" TF.reshape (Sh.wrap "gather" TF.gather pflattened ix) (shapeToTensor contextR context)
        TupRunit -> ()
        TupRpair tl tr -> let (pl, pr) = p in ( gatherL (Tensor (ArrayR shr tl) sh pl) ix
                                              , gatherL (Tensor (ArrayR shr tr) sh pr) ix)
        where
          -- assumes tuple of zero-dimensional ints, and returns a zero-dimensional int
          shapeProd :: ShapeR sh -> TensorShape sh -> TensorArrayData Int
          shapeProd ShapeRz () = Sh.wrap1 "scalar" TF.scalar 1
          shapeProd (ShapeRsnoc shr) (sh, n) = A.mul (IntegralNumType TypeInt) (n, shapeProd shr sh)
  in
  \case
    Let lhs bnd body              -> buildOpenExp contextR context (env `push` (lhs, buildE bnd)) aenv body
    Evar (Var _ ix)               -> prj ix env
    Pair x y                      -> (buildE x, buildE y)
    Nil                           -> ()
    VecPack{}                     -> unsupported "SIMD-vector types"
    VecUnpack{}                   -> unsupported "SIMD-vector types"
    ToIndex shR sh ix             -> let
        go :: ShapeR sh -> TensorArrayData sh -> TensorArrayData sh -> TensorArrayData Int
        go ShapeRz _ _ = fillL (TupRsingle (scalarType @Int)) (Sh.wrap1 "scalar" TF.scalar 0)
        go (ShapeRsnoc shr) (sh,n) (ix,i) = i + n * go shr sh ix
      in go shR (buildE sh) (buildE ix)
    FromIndex shR sh i            -> let
        go :: ShapeR sh -> TensorArrayData sh -> TensorArrayData Int -> TensorArrayData sh
        go ShapeRz _ _ = ()
        go (ShapeRsnoc shr) (sh,n) i = (go shr sh (A.quot TypeInt64 (i, n)), A.rem TypeInt64 (i, n))
      in go shR (buildE sh) (buildE i)
    Cond p t e                    -> condL p t e
    Const tR c                    -> constant contextR tR context c
    PrimConst x                   -> primConst contextR context x
    PrimApp f x                   -> primFun f (buildE x)
    Index v@(Var (ArrayR shr _) _) ix       -> buildE (LinearIndex v $ ToIndex shr (Shape v) ix)
    LinearIndex (Var (ArrayR _ t) arrIx) ix -> gatherL (aprj arrIx aenv) (buildE ix)
    Shape (Var _ ix)              -> shapeL (aprj ix aenv)
    ShapeSize shR sh              -> shapeSizeL shR (buildE sh)
    Foreign tR asm f x            -> unsupported "Exp: Foreign"
    IndexSlice sliceIndex slix sh -> unsupported "Exp: IndexSlice"
    IndexFull sliceIndex slix sl  -> unsupported "Exp: IndexFull"
    Case tag xs x                 -> unsupported "Exp: Case"
    While p f x                   -> unsupported "Exp: While"
    Undef t                       -> unsupported "Exp: Undef"
    Coerce tA tB a                -> unsupported "Exp: Coerce"


shapeToTensor
    :: (s ~ ScalarTensorDataR Int)
    => ShapeR sh
    -> TensorShape sh
    -> Sh.Tensor s
shapeToTensor ShapeRz              ()       = Sh.wrap2 "constant" TF.constant (TF.Shape [0]) []
shapeToTensor (ShapeRsnoc ShapeRz) ((), sh) = Sh.wrap "reshape" TF.reshape sh (Sh.wrap2 "constant" TF.constant (TF.Shape [1]) [1 :: ScalarTensorDataR Int])
shapeToTensor shR                  sh       =
  let go :: ShapeR sh -> TensorShape sh
         -> [Sh.Tensor (ScalarTensorDataR Int)] -> [Sh.Tensor (ScalarTensorDataR Int)]
      go ShapeRz         ()     acc = acc
      go (ShapeRsnoc tR) (t, h) acc = go tR t (h : acc)
--   in Sh.wrapPack (go shR sh [])
-- shapeToTensor
--     :: (s ~ ScalarTensorDataR Int)
--     => ShapeR sh
--     -> TensorShape sh
--     -> Sh.Tensor s
-- shapeToTensor ShapeRz              ()       = TF.constant (TF.Shape [1]) [1]
-- shapeToTensor (ShapeRsnoc ShapeRz) ((), sh) = sh
-- shapeToTensor shR                  sh       =
--   let
--       go :: (s ~ ScalarTensorDataR Int) => ShapeR sh -> TensorShape sh -> [Sh.Tensor s] -> [Sh.Tensor s]
--       go ShapeRz         ()     acc = acc
--       go (ShapeRsnoc tR) (t, h) acc = go tR t (h : acc)
--   -- XXX: Why is this reshape necessary?
  in Sh.wrapConcat (Sh.wrap1 "scalar" TF.scalar 0) [ Sh.wrap "reshape" TF.reshape x (Sh.wrap2 "constant" TF.constant (TF.Shape [1]) [1 :: ScalarTensorDataR Int]) | x <- go shR sh [] ]

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
constant shR eR sh x =
  let sh_ = shapeToTensor shR sh
  in buildTypeDictsScalar eR $ Sh.wrap "fill" TF.fill sh_ $ Sh.wrap1 "scalar" TF.scalar $ convertConvertable x

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
    PrimBAnd t              -> A.band t
    PrimBOr t               -> A.bor t
    PrimBXor t              -> A.xor t
    PrimBNot t              -> A.complement t
    PrimBShiftL t           -> A.shiftL t
    PrimBShiftR t           -> A.shiftR t
    PrimPopCount t          -> A.popCount t
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

    PrimIDiv{}                -> unsupported "PrimIDiv" -- :: IntegralType a -> PrimFun ((a, a)   -> a)
    PrimMod{}                 -> unsupported "PrimMod" -- :: IntegralType a -> PrimFun ((a, a)   -> a)
    PrimDivMod{}              -> unsupported "PrimDivMod" -- :: IntegralType a -> PrimFun ((a, a)   -> (a, a))
    PrimBRotateL{}            -> unsupported "PrimBRotateL"           -- :: IntegralType a -> PrimFun ((a, Int) -> a)
    PrimBRotateR{}            -> unsupported "PrimBRotateR"           -- :: IntegralType a -> PrimFun ((a, Int) -> a)
    PrimCountTrailingZeros{}  -> unsupported "PrimCountTrailingZeros"           -- :: IntegralType a -> PrimFun (a -> Int)
    PrimCountLeadingZeros{}   -> unsupported "PrimCountLeadingZeros"           -- :: IntegralType a -> PrimFun (a -> Int)
