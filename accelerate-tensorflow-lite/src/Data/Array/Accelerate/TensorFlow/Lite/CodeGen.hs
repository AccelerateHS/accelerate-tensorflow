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
import Data.Array.Accelerate.TensorFlow.CodeGen.Environment
import Data.Array.Accelerate.TensorFlow.CodeGen.Tensor
import Data.Array.Accelerate.TensorFlow.TypeDicts

import Data.Array.Accelerate.AST
import Data.Array.Accelerate.AST.LeftHandSide
import Data.Array.Accelerate.Analysis.Match
import Data.Array.Accelerate.Representation.Array                   hiding ( shape )
import Data.Array.Accelerate.Representation.Shape
import Data.Array.Accelerate.Representation.Type

import Lens.Family2
import qualified TensorFlow.Build                                   as TF
import qualified TensorFlow.Core                                    as TF
import qualified TensorFlow.GenOps.Core                             as TF
import qualified TensorFlow.Ops                                     as TF hiding ( placeholder' )

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
          let sh'    = shape _shR _sh
              adata' = evalState (array _eR) 0

              shape :: ShapeR sh -> sh -> TensorShape sh
              shape ShapeRz         ()     = ()
              shape (ShapeRsnoc tR) (t, h) = (shape tR t, TF.constant (TF.Shape [1]) [fromIntegral h])

              array :: TypeR t -> State Int (TensorArrayData t)
              array TupRunit         = return ()
              array (TupRpair aR bR) = (,) <$> array aR <*> array bR
              array (TupRsingle aR)  = buildTypeDictsScalar aR placeholder
                where
                  placeholder :: TF.TensorType t => State Int (TF.Tensor TF.Build t)
                  placeholder = state $ \j ->
                    let opName  = TF.opName .~ TF.explicitName (T.pack (printf "input%d_adata%d" i j))
                        opShape = TF.opAttr "shape" .~ tensorShape _shR _sh
                    in
                    (TF.placeholder' (opName . opShape), j+1)
          in
          (env `Apush` Tensor arrR sh' adata', i+1)
  in do
  aenv' <- go lhs x aenv
  f'    <- buildOpenAfunWith aenv' f xs
  return $ Tlam lhs f'
--
buildOpenAfunWith aenv (Abody f) (Aresult _ rsh)
  = let
        go :: ArraysR t -> Shapes t -> Tensors t -> State Int (Tensors t)
        go TupRunit              ()         ()                                    = return ()
        go (TupRpair aR bR)      (sha, shb) (a, b)                                = (,) <$> go aR sha a <*> go bR shb b
        go (TupRsingle ArrayR{}) sh         (Tensor (ArrayR _shR _eR) _sh _adata) = state $ \i ->
          let
              sh'    = evalState (shape _shR _sh) 0
              adata' = evalState (array _eR _adata) 0

              shape :: ShapeR sh -> TensorShape sh -> State Int (TensorShape sh)
              shape ShapeRz         ()     = return ()
              shape (ShapeRsnoc tR) (t, h) = do
                h' <- state $ \j -> let opName  = TF.opName .~ TF.explicitName (T.pack (printf "output%d_shape%d" i j))
                                        opShape = TF.opAttr "shape" .~ TF.Shape [1]
                                    in
                                    (TF.identity' (opShape . opName) h, j+1)
                t' <- shape tR t
                return (t', h')

              array :: TypeR t -> TensorArrayData t -> State Int (TensorArrayData t)
              array TupRunit         ()     = return ()
              array (TupRpair aR bR) (a, b) = (,) <$> array aR a <*> array bR b
              array (TupRsingle aR)  a      = buildTypeDictsScalar aR label a

              label :: TF.TensorType t => TF.Tensor TF.Build t -> State Int (TF.Tensor TF.Build t)
              label t = state $ \j ->
                let opName  = TF.opName .~ TF.explicitName (T.pack (printf "output%d_adata%d" i j))
                    opShape = TF.opAttr "shape" .~ tensorShape _shR sh
                in
                (TF.identity' (opShape . opName) t, j+1)
          in
          (Tensor (ArrayR _shR _eR) sh' adata', i+1)

        f' = evalState (go (arraysR f) rsh (buildOpenAcc aenv f)) 0
  in
  return $ Tbody (arraysR f) f'
--
buildOpenAfunWith _ _ _ =
  error "impossible"

tensorShape
    :: ShapeR sh
    -> sh
    -> TF.Shape
tensorShape shR     sh = TF.Shape [ fromIntegral x | x <- reverse (shapeToList shR sh) ]

