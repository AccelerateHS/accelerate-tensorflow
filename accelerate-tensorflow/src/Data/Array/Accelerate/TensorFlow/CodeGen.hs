{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE LambdaCase          #-}
{-# LANGUAGE OverloadedStrings   #-}
{-# LANGUAGE RankNTypes          #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications    #-}
{-# OPTIONS_HADDOCK hide #-}
-- |
-- Module      : Data.Array.Accelerate.TensorFlow.CodeGen
-- Copyright   : [2021] The Accelerate Team
-- License     : BSD3
--
-- Maintainer  : Trevor L. McDonell <trevor.mcdonell@gmail.com>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--

module Data.Array.Accelerate.TensorFlow.CodeGen (

  buildAcc, buildOpenAcc,
  buildAfun, 
  
  tpuArgMinMax, MinMax(..)

) where

import Data.Array.Accelerate.TensorFlow.CodeGen.AST
import Data.Array.Accelerate.TensorFlow.CodeGen.Base
import Data.Array.Accelerate.TensorFlow.CodeGen.Environment
import Data.Array.Accelerate.TensorFlow.CodeGen.Exp
import Data.Array.Accelerate.TensorFlow.CodeGen.Foreign
import Data.Array.Accelerate.TensorFlow.CodeGen.Tensor
import qualified Data.Array.Accelerate.TensorFlow.CodeGen.Tensor.Shim as Sh
import Data.Array.Accelerate.TensorFlow.TypeDicts

import Data.Array.Accelerate.Pretty ()
import Data.Array.Accelerate.Trafo.Var
import Data.Array.Accelerate.AST
import Data.Array.Accelerate.AST.Environment                        ( weakenEmpty, weakenSucc, weakenId )
import Data.Array.Accelerate.AST.Idx
import Data.Array.Accelerate.AST.LeftHandSide
import Data.Array.Accelerate.AST.Var
import Data.Array.Accelerate.Analysis.Match
import Data.Array.Accelerate.Array.Data
import Data.Array.Accelerate.Array.Unique
import Data.Array.Accelerate.Lifetime
import Data.Array.Accelerate.Representation.Array                   hiding ( shape )
import Data.Array.Accelerate.Representation.Shape
import Data.Array.Accelerate.Representation.Slice
import Data.Array.Accelerate.Representation.Type
import Data.Array.Accelerate.Sugar.Foreign
import Data.Array.Accelerate.Trafo.Substitution
import Data.Array.Accelerate.Type

import Data.ProtoLens.Default                                       ( def )
import Lens.Family2
import qualified Proto.Tensorflow.Core.Framework.Tensor             as TF
import qualified Proto.Tensorflow.Core.Framework.TensorShape_Fields as TensorShape
import qualified Proto.Tensorflow.Core.Framework.Tensor_Fields      as TF
import qualified TensorFlow.Build                                   as TF
import qualified TensorFlow.Core                                    as TF
import qualified TensorFlow.GenOps.Core                             as TF
import qualified TensorFlow.Ops                                     as TF hiding ( placeholder' )
import qualified TensorFlow.Types                                   as TF

import Control.Monad.State
import Data.ByteString.Internal                                     as B
import Data.Complex                                                 ( Complex )
import Data.Typeable
import Foreign.ForeignPtr
import Foreign.Storable
import Text.Printf
import qualified Data.Text                                          as T


buildAcc :: Acc a -> Tensors a
buildAcc acc = buildOpenAcc Aempty acc

buildAfun :: Afun f -> Tfun f
buildAfun f = evalState (buildOpenAfun Aempty f) 0

buildOpenAfun :: Aval aenv -> OpenAfun aenv f -> State Int (OpenTfun aenv f)
buildOpenAfun aenv (Alam lhs f) = do
  let
      go :: ALeftHandSide t aenv aenv' -> Aval aenv -> State Int (Aval aenv')
      go LeftHandSideWildcard{}                      env = return env
      go (LeftHandSidePair aR bR)                    env = go bR =<< go aR env
      go (LeftHandSideSingle arrR@(ArrayR _shR _eR)) env = state $ \i ->
        let sh'    = evalState (shape _shR) 0
            adata' = evalState (array _eR) 0

            shape :: ShapeR sh -> State Int (TensorShape sh)
            shape ShapeRz          = return ()
            shape (ShapeRsnoc shR) = do
              sz <- state $ \j ->
                      let name = printf "input%d_shape%d" i j
                          opName  = TF.opName .~ TF.explicitName (T.pack name)
                          opShape = TF.opAttr "shape" .~ TF.Shape []
                      in (Sh.wrap1 "placeholder'" (\_ -> TF.placeholder' (opShape . opName)) name, j+1)
              sh <- shape shR
              return (sh, sz)

            array :: TypeR t -> State Int (TensorArrayData t)
            array TupRunit         = return ()
            array (TupRpair aR bR) = (,) <$> array aR <*> array bR
            array (TupRsingle aR)  = buildTypeDictsScalar aR placeholder
              where
                placeholder :: TF.TensorType t => State Int (Sh.Tensor t)
                placeholder =
                  state $ \j ->
                    let name = printf "input%d_adata%d" i j
                    in (Sh.wrap1 "placeholder'" (\_ -> TF.placeholder' (TF.opName .~ TF.explicitName (T.pack name))) name, j+1)
        in
        (env `Apush` Tensor arrR sh' adata', i+1)
  --
  aenv' <- go lhs aenv
  f'    <- buildOpenAfun aenv' f
  return $ Tlam lhs f'
--
buildOpenAfun aenv (Abody f) =
  let
      go :: ArraysR t -> Tensors t -> State Int (Tensors t)
      go TupRunit              ()                                     = return ()
      go (TupRpair aR bR)      (a, b)                                 = (,) <$> go aR a <*> go bR b
      go (TupRsingle ArrayR{}) (Tensor (ArrayR shR eR) _sh _adata) = state $ \i ->
        let
            sh'    = evalState (shape shR _sh) 0
            adata' = evalState (array eR _adata) 0

            shape :: ShapeR sh -> TensorShape sh -> State Int (TensorShape sh)
            shape ShapeRz         ()     = return ()
            shape (ShapeRsnoc tR) (t, h) = do
              h' <- state $ \j ->
                      let name = printf "output%d_shape%d" i j
                      in (Sh.wrap1 "identity'" (\_ -> TF.identity' (TF.opName .~ TF.explicitName (T.pack name))) name h, j+1)
              t' <- shape tR t
              return (t', h')

            array :: TypeR t -> TensorArrayData t -> State Int (TensorArrayData t)
            array TupRunit         ()     = return ()
            array (TupRpair aR bR) (a, b) = (,) <$> array aR a <*> array bR b
            array (TupRsingle aR)  a      = buildTypeDictsScalar aR $ label a

            label :: (TF.TensorType t, Typeable t, Show t) => Sh.Tensor t -> State Int (Sh.Tensor t)
            label t =
              state $ \j ->
                let name = printf "output%d_adata%d" i j
                in (Sh.wrap1 "identity'" (\_ -> TF.identity' (TF.opName .~ TF.explicitName (T.pack name))) name t, j+1)
        in
        (Tensor (ArrayR shR eR) sh' adata', i+1)

      fR  = arraysR f
      f'  = evalState (go fR (buildOpenAcc aenv f)) 0
  in
  return $ Tbody fR f'


buildOpenAcc
    :: forall aenv arrs.
       Aval aenv
    -> OpenAcc aenv arrs
    -> Tensors arrs
buildOpenAcc aenv (OpenAcc pacc) =
  let
      buildA :: OpenAcc aenv (Array sh' e') -> Tensor sh' e'
      buildA = buildOpenAcc aenv

      aletL :: ALeftHandSide bnd aenv aenv'
            -> OpenAcc aenv  bnd
            -> OpenAcc aenv' body
            -> Tensors body
      aletL lhs bnd body = buildOpenAcc (aenv `apush` (lhs, buildOpenAcc aenv bnd)) body

      useL :: ArrayR (Array sh e)
           -> Array sh e
           -> Tensor sh e
      useL (ArrayR shR adataR) (Array sh adata) =
        let
            shape :: ShapeR sh -> sh -> TensorShape sh
            shape ShapeRz         ()     = ()
            shape (ShapeRsnoc tR) (t, h) = (shape tR t, Sh.wrap2 "constant" TF.constant (TF.Shape []) [fromIntegral h])

            array :: forall t'. TypeR t' -> ArrayData t' -> TensorArrayData t'
            array TupRunit ()             = ()
            array (TupRpair aR bR) (a, b) = (array aR a, array bR b)
            array (TupRsingle aR) a       = buildTypeDictsScalar aR $ tensor a
              where
                tensor :: forall s t. (Storable t, s ~ ScalarTensorDataR t, TF.TensorType s)
                       => UniqueArray t
                       -> Sh.Tensor s
                tensor ua =
                  let fp     = unsafeGetValue (uniqueArrayData ua)
                      values = B.fromForeignPtr (castForeignPtr fp) 0 (size shR sh * sizeOf (undefined :: t))

                      node :: TF.TensorProto
                      node = def
                           & TF.dtype .~ TF.tensorType (undefined :: s)
                           & TF.tensorShape.TensorShape.dim .~ [ def & TensorShape.size .~ fromIntegral x | x <- reverse $ shapeToList shR sh ]
                           & TF.tensorContent .~ values
                  in
                  Sh.wrap "const'" (TF.const' (TF.opAttr "value" .~ node))

            adata' = array adataR adata
            sh'    = shape shR sh
        in
        Tensor (ArrayR shR adataR) sh' adata'

      unitL :: TypeR e -> Exp aenv e -> Tensor () e
      unitL eR e =
        let sh' = ()
            e'  = buildOpenExp dim0 sh' Empty aenv e
        in
        Tensor (ArrayR dim0 eR) sh' e'

      mapL :: TypeR b -> Fun aenv (a -> b) -> OpenAcc aenv (Array sh a) -> Tensor sh b
      mapL bR (Lam lhs (Body e)) xs =
        let Tensor (ArrayR shR _) sh xs' = buildA xs
            bs                           = buildOpenExp shR sh (Empty `push` (lhs, xs')) aenv e
        in
        Tensor (ArrayR shR bR) sh bs
      mapL _ _ _ = error "impossible"

      -- XXX Assumes that the tensors have compatible shape!
      zipWithL :: TypeR c
               -> Fun aenv (a -> b -> c)
               -> OpenAcc aenv (Array sh a)
               -> OpenAcc aenv (Array sh b)
               -> Tensor sh c
      zipWithL cR (Lam lhsA (Lam lhsB (Body e))) xs ys =
        let Tensor (ArrayR shR _) sh xs' = buildA xs
            Tensor _              _  ys' = buildA ys
            cs                           = buildOpenExp shR sh (Empty `push` (lhsA, xs') `push` (lhsB, ys')) aenv e
        in
        Tensor (ArrayR shR cR) sh cs
      zipWithL _ _ _ _ = error "impossible"

      fillL :: ArrayR (Array sh e) -> Exp aenv sh -> Exp aenv e -> Tensor sh e
      fillL (ArrayR shR eR) sh e =
        let sh' = buildOpenExp ShapeRz ()  Empty aenv sh
            e'  = buildOpenExp shR     sh' Empty aenv e
        in
        Tensor (ArrayR shR eR) sh' e'

      generateL :: forall sh e. ArrayR (Array sh e) -> Exp aenv sh -> Fun aenv (sh -> e) -> Tensor sh e
      generateL aR@(ArrayR shR eR) sh f@(Lam lhs (Body e))
        | Lam LeftHandSideWildcard{} (Body e) <- f = fillL aR sh e
        | Just Refl <- isIdentity f = Tensor aR sh' ranges
        | otherwise                 = Tensor aR sh' $ buildOpenExp shR sh' (Empty `push` (lhs,ranges)) aenv e
          where 
            linearRange = Sh.wrap "range" TF.range
              (constant     ShapeRz scalarTypeInt () 0) 
              (buildOpenExp ShapeRz               () Empty aenv (ShapeSize shR sh))
              (constant     ShapeRz scalarTypeInt () 1)
            sh' :: TensorShape sh
            sh' = buildOpenExp ShapeRz () Empty aenv sh
            ranges = buildOpenExp shR sh' (Empty `push` (LeftHandSideSingle scalarTypeInt, Sh.wrap "reshape" TF.reshape linearRange (shapeToTensor shR sh'))) aenv $
              FromIndex shR (weakenE weakenEmpty sh) (Evar $ Var scalarTypeInt ZeroIdx)


      replicateL
          :: SliceIndex slix sl co sh
          -> Exp aenv slix
          -> OpenAcc aenv (Array sl e)
          -> Tensor sh e
      replicateL slice slix acc =
        let
            Tensor (ArrayR _ eR) sl' e' = buildA acc
            slix'                       = buildOpenExp dim0 () Empty aenv slix
            shR                         = sliceDomainR slice
            sh'                         = extend slice slix' sl'

            sh_                         = shapeToTensor shR sh'
            slPadded_                   = shapeToTensor shR (pad slice sl')

            extend :: SliceIndex slix sl co sh -> TensorShape slix -> TensorShape sl -> TensorShape sh
            extend SliceNil              ()        ()       = ()
            extend (SliceAll sliceIdx)   (slx, ()) (sl, sz) = (extend sliceIdx slx sl, sz)
            extend (SliceFixed sliceIdx) (slx, sz) sl       = (extend sliceIdx slx sl, sz)

            pad :: SliceIndex slix sl co sh -> TensorShape sl -> TensorShape sh
            pad SliceNil              ()       = ()
            pad (SliceAll sliceIdx)   (sl, sz) = (pad sliceIdx sl, sz)
            pad (SliceFixed sliceIdx) sl       = (pad sliceIdx sl, Sh.wrap1 "scalar" TF.scalar 1)

            array :: TypeR s -> TensorArrayData s -> TensorArrayData s
            array TupRunit         ()     = ()
            array (TupRpair aR bR) (a, b) = (array aR a, array bR b)
            array (TupRsingle aR)  a      = buildTypeDictsScalar aR $
              Sh.wrap "broadcastTo" TF.broadcastTo (Sh.wrap "reshape" TF.reshape a slPadded_) sh_
        in
        Tensor (ArrayR shR eR) sh' (array eR e')

      foldL :: Fun aenv (e -> e -> e)
            -> Maybe (Exp aenv e)
            -> OpenAcc aenv (Array (sh, Int) e)
            -> Tensor sh e
      foldL f z xs
        | Lam LeftHandSideSingle{} (Lam LeftHandSideSingle{} (Body b)) <- f
        , PrimApp op (Pair x y)                                        <- b
        , Evar (Var _ (SuccIdx ZeroIdx))                               <- x
        , Evar (Var _ ZeroIdx)                                         <- y
        , Just reduce                                                  <-
            case op of
              PrimAdd _ ->
                case z of
                  Nothing -> Just sumL
                  Just (Const (SingleScalarType (NumSingleType n)) v)
                    | IntegralNumType t <- n, IntegralDict <- integralDict t, v == 0 -> Just sumL
                    | FloatingNumType t <- n, FloatingDict <- floatingDict t, v == 0 -> Just sumL
                  _ -> Nothing
              PrimMul _ ->
                case z of
                  Nothing -> Just prodL
                  Just (Const (SingleScalarType (NumSingleType n)) v)
                    | IntegralNumType t <- n, IntegralDict <- integralDict t, v == 1 -> Just prodL
                    | FloatingNumType t <- n, FloatingDict <- floatingDict t, v == 1 -> Just prodL
                  _ -> Nothing
              PrimMax _ | Nothing <- z -> Just maximumL
              PrimMin _ | Nothing <- z -> Just minimumL
              _ -> Nothing
        = reduce xs

        | otherwise
        = unsupported "fold"

      reduceL :: (forall a. (TF.OneOf '[Int8, Int16, Int32, Int64, Word8, Word16, Word32, Word64, Float, Double] a, Typeable a, Show a) => Sh.Tensor a -> Sh.Tensor Int32 -> Sh.Tensor a)
              -> OpenAcc aenv (Array (sh, Int) e)
              -> Tensor sh e
      reduceL reduce acc =
        let Tensor (ArrayR (ShapeRsnoc shR') eR) (sh', _) xs' = buildA acc

            array :: TypeR t -> TensorArrayData t -> TensorArrayData t
            array TupRunit        () = ()
            array TupRpair{}      _  = unsupported "sum: product types"
            array (TupRsingle aR) a  = buildTypeDictsScalar aR $ reduce a (Sh.wrap1 "scalar" (TF.scalar @Int32) (-1))
        in
        Tensor (ArrayR shR' eR) sh' (array eR xs')

      sumL :: OpenAcc aenv (Array (sh, Int) e) -> Tensor sh e
      sumL = reduceL (Sh.wrap "sum" TF.sum)

      prodL :: OpenAcc aenv (Array (sh, Int) e) -> Tensor sh e
      prodL = reduceL (Sh.wrap "prod" TF.prod)

      minimumL :: OpenAcc aenv (Array (sh, Int) e) -> Tensor sh e
      minimumL = reduceL (Sh.wrap "min" TF.min)

      maximumL :: OpenAcc aenv (Array (sh, Int) e) -> Tensor sh e
      maximumL = reduceL (Sh.wrap "max" TF.max)

      backpermuteL
          :: ShapeR sh'
          -> Exp aenv sh'
          -> Fun aenv (sh' -> sh)
          -> OpenAcc aenv (Array sh e)
          -> Tensor sh' e
      backpermuteL shR' sh' p acc
        -- special case: transpose
        | ArrayR shR _                              <- arrayR acc
        , ShapeRsnoc (ShapeRsnoc ShapeRz)           <- shR
        , ShapeRsnoc (ShapeRsnoc ShapeRz)           <- shR'
        -- Shape of the result is the transpose of calling 'shape' on the input argument
        , Let _ (Shape u) s                         <- sh'
        , Nil `Pair` Evar (Var _ ZeroIdx)
              `Pair` Evar (Var _ (SuccIdx ZeroIdx)) <- s
        , OpenAcc (Avar v)                          <- acc
        , Just Refl                                 <- matchVar u v
        -- Permutation function is the transpose of each index
        , Lam _ (Body b)                            <- p
        , Nil `Pair` Evar (Var _ ZeroIdx)
              `Pair` Evar (Var _ (SuccIdx ZeroIdx)) <- b
        = transposeL acc
        -- TODO: special case: project one row or column from a matrix, e.g. 
          -- backpermute
          --   (indexSlice (I2 0 All_) (shape a0))
          --   (\(x0) -> indexFull (I2 0 All_) (x0))
        -- catch-all, inefficient version
        | arr@(ArrayR shR eR) <- arrayR acc
        , DeclareVars lhs w k <- declareVars (shapeType shR)
        = buildA 
          $ OpenAcc $ Alet (LeftHandSideSingle arr) acc 
          $ OpenAcc $ Map eR (Lam lhs $ Body $ Index (Var arr ZeroIdx) (expVars $ k weakenId))
          $ OpenAcc $ Generate (ArrayR shR' (shapeType shR)) 
                        (weaken (weakenSucc weakenId) sh')
                        (weaken (weakenSucc weakenId) p)

      transposeL
          :: OpenAcc aenv (Array DIM2 e)
          -> Tensor DIM2 e
      transposeL acc =
        let Tensor (ArrayR shR eR) (((), h),w) xs = buildA acc

            array :: TypeR t -> TensorArrayData t -> TensorArrayData t
            array TupRunit         ()     = ()
            array (TupRpair aR bR) (a, b) = (array aR a, array bR b)
            array (TupRsingle aR)  a      = buildTypeDictsScalar aR $ Sh.wrap "transpose" TF.transpose a (Sh.wrap2 "constant" (TF.constant @Int32) (TF.Shape [2]) [1,0])
        in
        Tensor (ArrayR shR eR) (((), w), h) (array eR xs)

      aforeignL
          :: forall asm a b. Foreign asm
          => ArraysR b
          -> asm (a -> b)
          -> Afun (a -> b)
          -> OpenAcc aenv a
          -> Tensors b
      aforeignL bR asm g a
        | Just Refl      <- eqT @asm @ForeignAcc
        , ForeignAcc _ f <- asm
        = f (buildOpenAcc aenv a)

        | otherwise
        = buildOpenAcc aenv (OpenAcc $ Apply bR (weaken weakenEmpty g) a)
  in
  case pacc of
    Alet lhs bnd body                 -> aletL lhs bnd body
    Avar (Var _ ix)                   -> aprj ix aenv
    Apair xs ys                       -> (buildOpenAcc aenv xs, buildOpenAcc aenv ys)
    Anil                              -> ()
    -- Apply aR f xs                     -> undefined
    Aforeign aR asm f xs              -> aforeignL aR asm f xs
    -- Acond p xs ys                     -> undefined
    -- Awhile p f xs                     -> undefined
    -- Atrace m xs ys                    -> undefined
    Use aR xs                         -> useL aR xs
    Unit eR e                         -> unitL eR e
    -- Reshape shR sh a                  -> undefined
    Generate aR sh f                  -> generateL aR sh f
    -- Transform aR sh p f xs            -> undefined
    Replicate slice slix sl           -> replicateL slice slix sl
    -- Slice sliceIndex sh slix          -> undefined
    Map bR f xs                       -> mapL bR f xs
    ZipWith cR f xs ys                -> zipWithL cR f xs ys
    Fold f z xs                       -> foldL f z xs
    -- FoldSeg iR f z xs ss              -> undefined
    -- Scan dir f z xs                   -> undefined
    -- Scan' dir f z xs                  -> undefined
    -- Permute f d p xs                  -> undefined
    Backpermute shR sh p xs           -> backpermuteL shR sh p xs
    -- Stencil sR tR f b xs              -> undefined
    -- Stencil2 sR1 sR2 tR f b1 xs b2 ys -> undefined

singleton :: ShapeR sh -> TensorShape sh
singleton ShapeRz          = ()
singleton (ShapeRsnoc shR) = (singleton shR, Sh.wrap2 "constant" TF.constant (TF.Shape [1]) [1])


data MinMax = Min | Max

tpuArgMinMax :: MinMax
             -> Tensor (sh, Int) a
             -> Tensor  sh         Int32
tpuArgMinMax minOrMax (Tensor (ArrayR (ShapeRsnoc shR) eR) (sh, _) t) = 
  Tensor (ArrayR shR $ TupRsingle scalarTypeInt32) sh (array eR t)
    where
      array :: TypeR t -> TensorArrayData t -> TensorArrayData Int32
      array TupRunit        () = Sh.wrap1 "scalar" (TF.scalar @Int32) 0
      array TupRpair{}      _  = unsupported "argMin/Max: product types"
      array (TupRsingle aR) a  = scalar aR a

      scalar :: ScalarType t -> TensorArrayData t -> TensorArrayData Int32
      scalar (SingleScalarType t) = single t
      scalar (VectorScalarType _) = unsupported "SIMD-vector types"

      single :: SingleType t -> TensorArrayData t -> TensorArrayData Int32
      single (NumSingleType t) = num t

      num :: NumType t -> TensorArrayData t -> TensorArrayData Int32
      num (IntegralNumType t) = integral t
      num (FloatingNumType t) = floating t

      choose :: (TF.OneOf '[Complex Double, Complex Float, Bool, Int16, Int32, Int64, Int8, Word16, Word32, Word64, Word8, Double, Float] t
                ,TF.OneOf '[Int32, Int64] output_type
                ,Typeable t, Show t)
             => Sh.Tensor t -> Sh.Tensor output_type
      choose x = case minOrMax of
                   Min -> Sh.wrap "argMin" TF.argMin x (Sh.wrap1 "scalar" (TF.scalar @Int32) (-1))
                   Max -> Sh.wrap "argMax" TF.argMax x (Sh.wrap1 "scalar" (TF.scalar @Int32) (-1))

      integral :: IntegralType t -> TensorArrayData t -> TensorArrayData Int32
      integral TypeInt8   x = choose x
      integral TypeInt16  x = choose x
      integral TypeInt32  x = choose x
      integral TypeInt64  x = choose x
      integral TypeWord8  x = choose x
      integral TypeWord16 x = choose x
      integral TypeWord32 x = choose x
      integral TypeWord64 x = choose x
      integral TypeInt    x = choose x
      integral TypeWord   x = choose x

      floating :: FloatingType t -> TensorArrayData t -> TensorArrayData Int32
      floating TypeFloat  x = choose x
      floating TypeDouble x = choose x
      floating TypeHalf   _ = unsupported "half-precision floating point"

