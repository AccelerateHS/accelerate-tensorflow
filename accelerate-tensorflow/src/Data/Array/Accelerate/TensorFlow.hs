{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE BangPatterns        #-}
{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE RankNTypes          #-}
{-# LANGUAGE RecordWildCards     #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications    #-}
{-# LANGUAGE TypeFamilies        #-}
-- |
-- Module      : Data.Array.Accelerate.TensorFlow
-- Copyright   : [2021] The Accelerate Team
-- License     : BSD3
--
-- Maintainer  : Trevor L. McDonell <trevor.mcdonell@gmail.com>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--

module Data.Array.Accelerate.TensorFlow (

  run,
  runN,

) where

import Data.Array.Accelerate                                        ( Acc )
import Data.Array.Accelerate.AST                                    ( ALeftHandSide )
import Data.Array.Accelerate.AST.LeftHandSide
import Data.Array.Accelerate.Array.Data
import Data.Array.Accelerate.Array.Unique
import Data.Array.Accelerate.Lifetime
import Data.Array.Accelerate.Representation.Type
import Data.Array.Accelerate.Representation.Shape
import Data.Array.Accelerate.Sugar.Array                            ( Arrays(..), ArraysR(..) )
import Data.Array.Accelerate.Trafo.Sharing
import Data.Array.Accelerate.Trafo.Simplify
import Data.Array.Accelerate.Type
import qualified Data.Array.Accelerate.Representation.Array         as R
import qualified Data.Array.Accelerate.Representation.Shape         as R

import Data.Array.Accelerate.TensorFlow.CodeGen
import Data.Array.Accelerate.TensorFlow.CodeGen.AST
import Data.Array.Accelerate.TensorFlow.CodeGen.Tensor
import Data.Array.Accelerate.TensorFlow.TypeDicts

import qualified TensorFlow.Core                                    as TF
import qualified TensorFlow.Tensor                                  as TF
import qualified TensorFlow.Types                                   as TF
import qualified TensorFlow.Internal.FFI                            as Internal

import Control.Monad.State
import Foreign.ForeignPtr
import Foreign.Storable
import System.IO.Unsafe
import Text.Printf
import qualified Data.Text                                          as T
import qualified Data.Vector.Storable                               as V


-- | Run a complete embedded program using the default TensorFlow backend
--
run :: forall arrs. Arrays arrs => Acc arrs -> arrs
run | FetchableDict <- fetchableDict @arrs
    = toArr
    . unsafePerformIO . TF.runSession . TF.run
    . buildAcc
    . simplifyAcc
    . convertAcc

-- | Prepare an embedded array program for execution on the default
-- TensorFlow backend
--
runN :: forall f. Afunction f => f -> AfunctionR f
runN acc =
  let
      !model = buildAfun afun
      !afun  = simplifyAfun
             . convertAfun
             $ acc

      eval :: AfunctionRepr g (AfunctionR g) (ArraysFunctionR g)
           -> OpenTfun aenv (ArraysFunctionR g)
           -> Int
           -> [TF.Feed]
           -> AfunctionR g
      eval AfunctionReprBody (Tbody bR b) _ aenv
        | FetchableDict <- fetchableDictR bR
        = toArr
        . unsafePerformIO
        . TF.runSession
        $ TF.runWithFeeds aenv b
      eval (AfunctionReprLam lamR) (Tlam lhs f) skip aenv = \arr ->
        let
            go :: ALeftHandSide t aenv aenv' -> t -> [TF.Feed] -> State Int [TF.Feed]
            go LeftHandSideWildcard{}                   _                  env = return env
            go (LeftHandSidePair aR bR)                 (a, b)             env = go bR b =<< go aR a env
            go (LeftHandSideSingle (R.ArrayR _shR _eR)) (R.Array _sh _adata) env = state $ \i ->
              let
                  sh'    = evalState (shape _shR _sh) 0
                  adata' = evalState (array _eR _adata) 0

                  shape :: ShapeR sh -> sh -> State Int [TF.Feed]
                  shape ShapeRz          ()     = return []
                  shape (ShapeRsnoc shR) (t, h) = do
                    h' <- state $ \j ->
                           let tensorDataBytes = V.fromListN 1 [fromIntegral h :: ScalarTensorDataR Int]
                               tensorDataShape = TF.Shape [1]
                           in
                           (TF.feed (TF.tensorValueFromName (T.pack (printf "input%d_shape%d" i j))) (TF.encodeTensorData tensorDataShape tensorDataBytes), j+1)
                    t' <- shape shR t
                    return (h' : t')

                  array :: TypeR t -> ArrayData t -> State Int [TF.Feed]
                  array TupRunit         ()     = return []
                  array (TupRsingle aR)  a      = return <$> buildTypeDictsScalar aR feed a
                  array (TupRpair aR bR) (a, b) = do
                    a' <- array aR a
                    b' <- array bR b
                    return (a' ++ b')

                  feed :: forall t s. (Storable t, TF.TensorType s, s ~ ScalarTensorDataR t) => UniqueArray t -> State Int TF.Feed
                  feed ua = state $ \j ->
                    let fp                   = unsafeGetValue (uniqueArrayData ua)
                        tensorDataBytes      = V.unsafeFromForeignPtr0 (castForeignPtr fp :: ForeignPtr Word8) (R.size _shR _sh * sizeOf (undefined :: t))
                        tensorDataType       = TF.tensorType (undefined :: s)
                        tensorDataDimensions = [ fromIntegral x :: ScalarTensorDataR Int | x <- reverse (R.shapeToList _shR _sh) ]
                    in
                    (TF.feed (TF.tensorValueFromName (T.pack (printf "input%d_adata%d" i j))) (TF.TensorData (Internal.TensorData {..})), j+1)
              in
              (sh' ++ adata' ++ env, i+1)

            (aenv', next) = runState (go lhs (fromArr arr) aenv) skip
        in
        eval lamR f next aenv'
      eval _ _ _ _ = error "impossible"
  in
  eval (afunctionRepr @f) model 0 []


data FetchableDict t where
  FetchableDict :: TF.Fetchable (Tensors t) t => FetchableDict t

fetchableDict :: forall arrs. Arrays arrs => FetchableDict (ArraysR arrs)
fetchableDict = fetchableDictR (arraysR @arrs)

fetchableDictR :: R.ArraysR a -> FetchableDict a
fetchableDictR TupRunit                = FetchableDict
fetchableDictR (TupRsingle R.ArrayR{}) = FetchableDict
fetchableDictR (TupRpair aR bR)
  | FetchableDict <- fetchableDictR aR
  , FetchableDict <- fetchableDictR bR
  = FetchableDict

