{-# LANGUAGE GADTs             #-}
{-# LANGUAGE OverloadedStrings #-}
-- |
-- Module      : Data.Array.Accelerate.TensorFlow.Lite.CodeGen
-- Copyright   : [2021..2022] The Accelerate Team
-- License     : BSD3
--
-- Maintainer  : Trevor L. McDonell <trevor.mcdonell@gmail.com>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--

module Data.Array.Accelerate.TensorFlow.Lite.CodeGen (

  buildAfunWith,

) where

import Data.Array.Accelerate.TensorFlow.Lite.Representation.Args
import Data.Array.Accelerate.TensorFlow.Lite.Representation.Shapes

import Data.Array.Accelerate.TensorFlow.CodeGen                     ( buildOpenAcc )
import Data.Array.Accelerate.TensorFlow.CodeGen.AST
import Data.Array.Accelerate.TensorFlow.CodeGen.Base
import Data.Array.Accelerate.TensorFlow.CodeGen.Environment
import Data.Array.Accelerate.TensorFlow.CodeGen.Tensor

import Data.Array.Accelerate.AST
import Data.Array.Accelerate.AST.LeftHandSide
import Data.Array.Accelerate.Analysis.Match
import Data.Array.Accelerate.Representation.Array                   hiding ( shape )
import Data.Array.Accelerate.Representation.Shape
import Data.Array.Accelerate.Representation.Type
import Data.Array.Accelerate.Type

import Lens.Family2
import qualified TensorFlow.Build                                   as TF
import qualified TensorFlow.Core                                    as TF
import qualified TensorFlow.GenOps.Core                             as TF

import Control.Monad.State
import Text.Printf
import qualified Data.Text                                          as T


buildAfunWith :: Afun f -> Args f -> Tfun f
buildAfunWith f xs = evalState (buildOpenAfunWith Aempty f xs) 0

buildOpenAfunWith :: Aval aenv -> OpenAfun aenv f -> Args f -> State Int (OpenTfun aenv f)
buildOpenAfunWith aenv (Alam lhs f) (Aparam xR x xs)
  | Just Refl <- matchArraysR (lhsToTupR lhs) xR
  = let
        go :: ALeftHandSide t aenv aenv' -> t -> Aval aenv -> State Int (Aval aenv')
        go LeftHandSideWildcard{}                      _             env = return env
        go (LeftHandSidePair aR bR)                    (a, b)        env = go bR b =<< go aR a env
        go (LeftHandSideSingle arrR@(ArrayR _shR _eR)) (Array _sh _) env = state $ \i ->
          let sh'    = evalState (shape _shR) 0
              adata' = evalState (array _eR) 0

              shape :: ShapeR sh -> State Int (TensorShape sh)
              shape ShapeRz          = return ()
              shape (ShapeRsnoc shR) = do
                sz <- state $ \j -> let opName  = TF.opName .~ TF.explicitName (T.pack (printf "input%d_shape%d" i j))
                                        opShape = TF.opAttr "shape" .~ TF.Shape (map fromIntegral (shapeToList _shR _sh))
                                    in
                                    (TF.placeholder' (opShape . opName), j+1)
                sh <- shape shR
                return (sh, sz)

              array :: TypeR t -> State Int (TensorArrayData t)
              array TupRunit         = return ()
              array (TupRpair aR bR) = (,) <$> array aR <*> array bR
              array (TupRsingle aR)  = scalar aR
                where
                  scalar :: ScalarType t -> State Int (TensorArrayData t)
                  scalar (SingleScalarType t) = single t
                  scalar (VectorScalarType _) = unsupported "SIMD-vector types"

                  single :: SingleType t -> State Int (TensorArrayData t)
                  single (NumSingleType t) = num t

                  num :: NumType t -> State Int (TensorArrayData t)
                  num (IntegralNumType t) = integral t
                  num (FloatingNumType t) = floating t

                  integral :: IntegralType t -> State Int (TensorArrayData t)
                  integral TypeInt8   = placeholder
                  integral TypeInt16  = placeholder
                  integral TypeInt32  = placeholder
                  integral TypeInt64  = placeholder
                  integral TypeWord8  = placeholder
                  integral TypeWord16 = placeholder
                  integral TypeWord32 = placeholder
                  integral TypeWord64 = placeholder
                  integral TypeInt    = placeholder
                  integral TypeWord   = placeholder

                  floating :: FloatingType t -> State Int (TensorArrayData t)
                  floating TypeFloat  = placeholder
                  floating TypeDouble = placeholder
                  floating TypeHalf   = unsupported "half-precision floating point"

                  placeholder :: TF.TensorType t => State Int (TF.Tensor TF.Build t)
                  placeholder = state $ \j ->
                    let opName  = TF.opName .~ TF.explicitName (T.pack (printf "input%d_adata%d" i j))
                        opShape = TF.opAttr "shape" .~ TF.Shape (map fromIntegral (shapeToList _shR _sh))
                    in
                    (TF.placeholder' (opShape . opName), j+1)
          in
          (env `Apush` Tensor arrR sh' adata', i+1)
  in do
  aenv' <- go lhs x aenv
  f'    <- buildOpenAfunWith aenv' f xs
  return $ Tlam lhs f'
--
buildOpenAfunWith aenv (Abody f) (Aresult fR sr)
  | Just Refl <- matchArraysR fR (arraysR f)
  = let
        go :: ArraysR t -> Shapes t -> Tensors t -> State Int (Tensors t)
        go TupRunit              ()         ()                                  = return ()
        go (TupRpair aR bR)      (sha, shb) (a, b)                              = (,) <$> go aR sha a <*> go bR shb b
        go (TupRsingle ArrayR{}) sh         (Tensor (ArrayR shR eR) _sh _adata) = state $ \i ->
          let
              sh'    = evalState (shape shR _sh) 0
              adata' = evalState (array eR _adata) 0

              shape :: ShapeR sh -> TensorShape sh -> State Int (TensorShape sh)
              shape ShapeRz         ()     = return ()
              shape (ShapeRsnoc tR) (t, h) = do
                h' <- state $ \j -> let opName  = TF.opName .~ TF.explicitName (T.pack (printf "input%d_shape%d" i j))
                                        opShape = TF.opAttr "shape" .~ TF.Shape [1]
                                    in
                                    (TF.identity' (opShape . opName) h, j+1)
                t' <- shape tR t
                return (t', h')

              array :: TypeR t -> TensorArrayData t -> State Int (TensorArrayData t)
              array TupRunit         ()     = return ()
              array (TupRpair aR bR) (a, b) = (,) <$> array aR a <*> array bR b
              array (TupRsingle aR)  a      = scalar aR a

              scalar :: ScalarType t -> TensorArrayData t -> State Int (TensorArrayData t)
              scalar (SingleScalarType t) = single t
              scalar (VectorScalarType _) = unsupported "SIMD-vector types"

              single :: SingleType t -> TensorArrayData t -> State Int (TensorArrayData t)
              single (NumSingleType t) = num t

              num :: NumType t -> TensorArrayData t -> State Int (TensorArrayData t)
              num (IntegralNumType t) = integral t
              num (FloatingNumType t) = floating t

              integral :: IntegralType t -> TensorArrayData t -> State Int (TensorArrayData t)
              integral TypeInt8   = label
              integral TypeInt16  = label
              integral TypeInt32  = label
              integral TypeInt64  = label
              integral TypeWord8  = label
              integral TypeWord16 = label
              integral TypeWord32 = label
              integral TypeWord64 = label
              integral TypeInt    = label
              integral TypeWord   = label

              floating :: FloatingType t -> TensorArrayData t -> State Int (TensorArrayData t)
              floating TypeFloat  = label
              floating TypeDouble = label
              floating TypeHalf   = unsupported "half-precision floating point"

              label :: TF.TensorType t => TF.Tensor TF.Build t -> State Int (TF.Tensor TF.Build t)
              label t = state $ \j ->
                let opName  = TF.opName .~ TF.explicitName (T.pack (printf "output%d_adata%d" i j))
                    opShape = TF.opAttr "shape" .~ TF.Shape (map fromIntegral (shapeToList shR sh))
                in
                (TF.identity' (opShape . opName) t, j+1)
          in
          (Tensor (ArrayR shR eR) sh' adata', i+1)

        f' = evalState (go fR sr (buildOpenAcc aenv f)) 0
  in
  return $ Tbody fR f'
--
buildOpenAfunWith _ _ _ =
  error "impossible"

