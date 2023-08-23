{-# LANGUAGE GADTs             #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RankNTypes        #-}
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
import qualified Data.Array.Accelerate.TensorFlow.CodeGen.Tensor.Shim as Sh
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
import Data.Bifunctor                                               ( second )
import qualified Data.Text                                          as T
import Data.Typeable                                                ( Typeable )


buildAfunWith :: Afun f -> Args f -> (Tfun f, ArgsNames f)
buildAfunWith f xs =
  evalState (buildOpenAfunWith Aempty f xs) 0

buildOpenAfunWith :: Aval aenv -> OpenAfun aenv f -> Args f -> State Int (OpenTfun aenv f, ArgsNames f)
buildOpenAfunWith aenv (Alam lhs f) (Aparam xR x xs)
  | Just Refl <- matchArraysR (lhsToTupR lhs) xR
  = let
        go :: ALeftHandSide t aenv aenv' -> t -> Aval aenv -> State Int (Aval aenv', TupR ArrArgNames t)
        go (LeftHandSideWildcard aR)                   _             env = return (env, mapTupR (\ArrayR{} -> ArrArgSkip) aR)
        go (LeftHandSidePair aR bR)                    (a, b)        env = do
          (env', t1) <- go aR a env
          (env'', t2) <- go bR b env'
          return (env'', TupRpair t1 t2)
        go (LeftHandSideSingle arrR@(ArrayR _shR _eR)) (Array _sh _) env = state $ \i ->
          let sh'             = shape _shR _sh
              (adata', names) = evalState (array _eR) 0

              shape :: ShapeR sh -> sh -> TensorShape sh
              shape ShapeRz         ()     = ()
              shape (ShapeRsnoc tR) (t, h) = (shape tR t, Sh.wrap1 "scalar" TF.scalar (fromIntegral h))

              array :: TypeR t -> State Int (TensorArrayData t, TupR ArgName t)
              array TupRunit         = return ((), TupRunit)
              array (TupRpair aR bR) = (\(d1,n1) (d2,n2) -> ((d1,d2), TupRpair n1 n2)) <$> array aR <*> array bR
              array (TupRsingle aR)  = buildTypeDictsScalar aR (second (TupRsingle . ArgName) <$> placeholder)
                where
                  placeholder :: (Typeable t, Show t, TF.TensorType t) => State Int (Sh.Tensor t, T.Text)
                  placeholder = state $ \j ->
                    let name    = T.pack (printf "input%d_adata%d" i j)
                        opName  = TF.opName .~ TF.explicitName name
                        opShape = TF.opAttr "shape" .~ tensorShape _shR _sh
                        node_ph = Sh.wrap1 "placeholder'" (\_ -> TF.placeholder' (opName . opShape)) (T.unpack name)
                    in
                    ((node_ph, name), j+1)
          in
          ((env `Apush` Tensor arrR sh' adata', TupRsingle (ArrArgNames names)), i+1)
  in do
  (aenv', tupargnames) <- go lhs x aenv
  (f', restargnames) <- buildOpenAfunWith aenv' f xs
  return $ (Tlam lhs f', ANparam tupargnames restargnames)
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
                h' <- state $ \j -> let name = printf "output%d_shape%d" i j
                                        opName  = TF.opName .~ TF.explicitName (T.pack name)
                                        opShape = TF.opAttr "shape" .~ TF.Shape []
                                    in
                                    (Sh.wrap1 "identity'" (\_ -> TF.identity' (opShape . opName)) name h, j+1)
                t' <- shape tR t
                return (t', h')

              array :: TypeR t -> TensorArrayData t -> State Int (TensorArrayData t)
              array TupRunit         ()     = return ()
              array (TupRpair aR bR) (a, b) = (,) <$> array aR a <*> array bR b
              array (TupRsingle aR)  a      = buildTypeDictsScalar aR label a

              label :: (TF.TensorType t, Typeable t, Show t) => Sh.Tensor t -> State Int (Sh.Tensor t)
              label t = state $ \j ->
                let name = printf "output%d_adata%d" i j
                    opName  = TF.opName .~ TF.explicitName (T.pack name)
                    opShape = TF.opAttr "shape" .~ tensorShape _shR sh
                in
                (Sh.wrap1 "identity'" (\_ -> TF.identity' (opShape . opName)) name t, j+1)
          in
          (Tensor (ArrayR _shR _eR) sh' adata', i+1)

        f' = evalState (go (arraysR f) rsh (buildOpenAcc aenv f)) 0
  in
  return $ (Tbody (arraysR f) f', ANresult)
--
buildOpenAfunWith _ _ _ =
  error "impossible"

mapTupR :: (forall a. f a -> g a) -> TupR f t -> TupR g t
mapTupR f (TupRpair a b) = TupRpair (mapTupR f a) (mapTupR f b)
mapTupR f (TupRsingle x) = TupRsingle (f x)
mapTupR _ TupRunit = TupRunit

tensorShape
    :: ShapeR sh
    -> sh
    -> TF.Shape
tensorShape shR     sh = TF.Shape [ fromIntegral x | x <- reverse (shapeToList shR sh) ]

