{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications    #-}
-- |
-- Module      : Data.Array.Accelerate.TensorFlow.Lite
-- Copyright   : [2021] The Accelerate Team
-- License     : BSD3
--
-- Maintainer  : Trevor L. McDonell <trevor.mcdonell@gmail.com>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--

module Data.Array.Accelerate.TensorFlow.Lite
  where

import Data.Array.Accelerate                                        ( Acc )
import Data.Array.Accelerate.Representation.Type
import Data.Array.Accelerate.Sugar.Array
import Data.Array.Accelerate.Trafo.Sharing
import Data.Array.Accelerate.Type
import qualified Data.Array.Accelerate.Representation.Array         as R

import Data.Array.Accelerate.TensorFlow.CodeGen
import Data.Array.Accelerate.TensorFlow.CodeGen.AST
import Data.Array.Accelerate.TensorFlow.CodeGen.Base
import Data.Array.Accelerate.TensorFlow.CodeGen.Tensor

import qualified TensorFlow.Build                                   as TF
import qualified TensorFlow.Core                                    as TF
import qualified Proto.Tensorflow.Core.Framework.Graph              as TF
import qualified Proto.Tensorflow.Core.Framework.Graph_Fields       as TF

import Data.Functor.Identity
import Data.ProtoLens
import Lens.Family2
import qualified Data.ByteString                                    as B


saveAcc :: forall arrs. Arrays arrs => FilePath -> Acc arrs -> IO ()
saveAcc path acc =
  let model = buildAcc (convertAcc acc)
   in save_model (arraysR @arrs) path model

saveAfun :: forall f. Afunction f => FilePath -> f -> IO ()
saveAfun path acc =
  let model = buildAfun (convertAfun acc)

      go :: AfunctionRepr g (AfunctionR g) (ArraysFunctionR g)
         -> OpenTfun aenv (ArraysFunctionR g)
         -> IO ()
      go AfunctionReprBody       (Tbody bR b) = save_model bR path b
      go (AfunctionReprLam lamR) (Tlam _ f)   = go lamR f
      go _                       _            = error "impossible"
  in
  go (afunctionRepr @f) model

save_model :: R.ArraysR arrs -> FilePath -> Tensors arrs -> IO ()
save_model arrR path model =
  let
      go :: forall m a. TF.MonadBuild m => R.ArraysR a -> Tensors a -> m ()
      go TupRunit                ()                           = return ()
      go (TupRpair aR bR)        (a, b)                       = go aR a >> go bR b
      go (TupRsingle R.ArrayR{}) (Tensor (R.ArrayR _ aR) _ a) = array aR a

      array :: TF.MonadBuild m => TypeR t -> TensorArrayData t -> m ()
      array TupRunit         ()     = return ()
      array (TupRpair aR bR) (a, b) = array aR a >> array bR b
      array (TupRsingle aR)  a      = scalar aR a

      scalar :: TF.MonadBuild m => ScalarType t -> TensorArrayData t -> m ()
      scalar (SingleScalarType t) = single t
      scalar (VectorScalarType _) = unsupported "SIMD-vector types"

      single :: TF.MonadBuild m => SingleType t -> TensorArrayData t -> m ()
      single (NumSingleType t) = num t

      num :: TF.MonadBuild m => NumType t -> TensorArrayData t -> m ()
      num (IntegralNumType t) = integral t
      num (FloatingNumType t) = floating t

      integral :: TF.MonadBuild m => IntegralType t -> TensorArrayData t -> m ()
      integral TypeInt8   = render
      integral TypeInt16  = render
      integral TypeInt32  = render
      integral TypeInt64  = render
      integral TypeWord8  = render
      integral TypeWord16 = render
      integral TypeWord32 = render
      integral TypeWord64 = render
      integral TypeInt    = render
      integral TypeWord   = render

      floating :: TF.MonadBuild m => FloatingType t -> TensorArrayData t -> m ()
      floating TypeFloat  = render
      floating TypeDouble = render
      floating TypeHalf   = unsupported "half-precision floating point"

      render :: TF.MonadBuild m => TF.Tensor TF.Build a -> m ()
      render t = TF.render t >> return ()

      nodes = runIdentity $ TF.evalBuildT (go arrR model >> TF.flushNodeBuffer)
      graph = defMessage & TF.node .~ nodes :: TF.GraphDef
  in
  B.writeFile path (encodeMessage graph)

