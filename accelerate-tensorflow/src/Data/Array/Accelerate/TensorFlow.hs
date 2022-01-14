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
import Data.Array.Accelerate.Sugar.Array
import Data.Array.Accelerate.Trafo.Sharing
import Data.Array.Accelerate.Trafo.Simplify
import Data.Array.Accelerate.Type
import qualified Data.Array.Accelerate.Representation.Array         as R
import qualified Data.Array.Accelerate.Representation.Shape         as R

import Data.Array.Accelerate.TensorFlow.CodeGen
import Data.Array.Accelerate.TensorFlow.CodeGen.AST
import Data.Array.Accelerate.TensorFlow.CodeGen.Base
import Data.Array.Accelerate.TensorFlow.CodeGen.Tensor

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
runN :: forall f. Afunction f => f -> [[Int]] -> [[Int]] -> AfunctionR f
runN acc ishapes oshapes =
  let
      !model = buildAfun ishapes oshapes afun
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
            go LeftHandSideWildcard{}                 _                  env = return env
            go (LeftHandSidePair aR bR)               (a, b)             env = go bR b =<< go aR a env
            go (LeftHandSideSingle (R.ArrayR shR eR)) (R.Array sh adata) env = state $ \i ->
              let dims   = [ fromIntegral x :: Int64 | x <- R.shapeToList shR sh ]
                  sh'    = TF.feed (TF.tensorValueFromName (T.pack (printf "input%d_shape" i)))
                         $ TF.encodeTensorData (TF.Shape [fromIntegral $ R.rank shR])
                         $ V.fromList dims
                  adata' = evalState (array eR adata) 0

                  array :: TypeR t -> ArrayData t -> State Int [TF.Feed]
                  array TupRunit         ()     = return []
                  array (TupRsingle aR)  a      = return <$> scalar aR a
                  array (TupRpair aR bR) (a, b) = do
                    a' <- array aR a
                    b' <- array bR b
                    return (a' ++ b')

                  scalar :: ScalarType t -> ArrayData t -> State Int TF.Feed
                  scalar (SingleScalarType t) = single t
                  scalar (VectorScalarType _) = unsupported "SIMD-vector types"

                  single :: SingleType t -> ArrayData t -> State Int TF.Feed
                  single (NumSingleType t) = num t

                  num :: NumType t -> ArrayData t -> State Int TF.Feed
                  num (IntegralNumType t) = integral t
                  num (FloatingNumType t) = floating t

                  integral :: IntegralType t -> ArrayData t -> State Int TF.Feed
                  integral TypeInt8   = feed
                  integral TypeInt16  = feed
                  integral TypeInt32  = feed
                  integral TypeInt64  = feed
                  integral TypeWord8  = feed
                  integral TypeWord16 = feed
                  integral TypeWord32 = feed
                  integral TypeWord64 = feed
                  integral TypeInt    = feed
                  integral TypeWord   = feed

                  floating :: FloatingType t -> ArrayData t -> State Int TF.Feed
                  floating TypeFloat  = feed
                  floating TypeDouble = feed
                  floating TypeHalf   = unsupported "half-precision floating point"

                  feed :: forall t s. (Storable t, TF.TensorType s, s ~ ScalarTensorDataR t) => UniqueArray t -> State Int TF.Feed
                  feed ua = state $ \j ->
                    let fp                   = unsafeGetValue (uniqueArrayData ua)
                        tensorDataBytes      = V.unsafeFromForeignPtr0 (castForeignPtr fp :: ForeignPtr Word8) (R.size shR sh * sizeOf (undefined :: t))
                        tensorDataType       = TF.tensorType (undefined :: s)
                        tensorDataDimensions = dims
                    in
                    (TF.feed (TF.tensorValueFromName (T.pack (printf "input%d_adata%d" i j))) (TF.TensorData (Internal.TensorData {..})), j+1)
              in
              (sh' : adata' ++ env, i+1)

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

