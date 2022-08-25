{-# LANGUAGE BangPatterns             #-}
{-# LANGUAGE FlexibleInstances        #-}
{-# LANGUAGE ForeignFunctionInterface #-}
{-# LANGUAGE GADTs                    #-}
{-# LANGUAGE RecordWildCards          #-}
{-# LANGUAGE ScopedTypeVariables      #-}
{-# LANGUAGE TypeApplications         #-}
-- |
-- Module      : Data.Array.Accelerate.TensorFlow.Lite
-- Copyright   : [2021..2022] The Accelerate Team
-- License     : BSD3
--
-- Maintainer  : Trevor L. McDonell <trevor.mcdonell@gmail.com>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--

module Data.Array.Accelerate.TensorFlow.Lite (

  Smart.Acc, Sugar.Arrays, Afunction, AfunctionR,
  Model, RepresentativeData, Shapes, Args(..),
  encodeModel, decodeModel,

  compile,
  execute,

) where

import Data.Array.Accelerate.AST                                              as AST
import Data.Array.Accelerate.AST.LeftHandSide
import Data.Array.Accelerate.Array.Data
import Data.Array.Accelerate.Array.Unique
import Data.Array.Accelerate.Lifetime
import Data.Array.Accelerate.Representation.Array                             as R ( ArraysR )
import Data.Array.Accelerate.Representation.Shape
import Data.Array.Accelerate.Representation.Type
import Data.Array.Accelerate.Sugar.Array                                      as Sugar ( Arrays(..) )
import Data.Array.Accelerate.Trafo.Sharing
import Data.Array.Accelerate.Trafo.Simplify
import Data.Array.Accelerate.Type
import qualified Data.Array.Accelerate.Representation.Array                   as R
import qualified Data.Array.Accelerate.Smart                                  as Smart

import Data.Array.Accelerate.TensorFlow.CodeGen.Base

import Data.Array.Accelerate.TensorFlow.Lite.CodeGen
import Data.Array.Accelerate.TensorFlow.Lite.Compile
import Data.Array.Accelerate.TensorFlow.Lite.Model
import Data.Array.Accelerate.TensorFlow.Lite.Sugar.Args
import Data.Array.Accelerate.TensorFlow.Lite.Sugar.Shapes
import qualified Data.Array.Accelerate.TensorFlow.Lite.Representation.Args    as R
import qualified Data.Array.Accelerate.TensorFlow.Lite.Representation.Shapes  as R

import Control.Monad.State
import Data.List                                                              ( genericLength )
import Foreign.C.String
import Foreign.ForeignPtr
import Foreign.Marshal.Array
import Foreign.Marshal.Utils
import Foreign.Ptr
import Foreign.Storable
import System.IO.Unsafe
import Text.Printf
import qualified Data.ByteString.Unsafe                                       as B
import Prelude                                                                as P


-- | A representative data set for a given tensor computation. This
-- typically consists of a subset of the data that was used for training.
--
type RepresentativeData f = [Args f]


-- | Compile a TensorFlow model for the EdgeTPU. The given representative
-- data is used in the compilation process to specify the dimensions of the
-- input and output tensors, as well as providing sample data for the
-- quantisation process.
--
-- For example, given a tensor computation:
--
-- > f :: (Arrays a, Arrays b, Arrays c) => Acc a -> Acc b -> Acc c
-- > f = ...
--
-- this will produce:
--
-- > m :: Model (a -> b -> c)
-- > m = compile f args
--
-- The compiled model can then be evaluated using 'execute' or serialised
-- using 'encodeModel'.
--
compile :: forall f. Afunction f => f -> RepresentativeData (AfunctionR f) -> Model (AfunctionR f)
compile acc args = unsafePerformIO $ Model afunR (modelAfun afunR tfun x) <$> compileTfunWith tfun (x:xs)
  where
    !afunR = afunctionRepr @f
    !afun  = simplifyAfun (convertAfun acc)
    !tfun  = buildAfunWith afun x
    x:xs   = map (fromArgs afunR) args


-- | Prepare and/or run a previously compiled model
--
-- For example, given:
--
-- > m :: Model (Vector Float -> Vector Float -> Vector Int8)
-- > m = ...
--
-- Then we can partially apply it to create a reusable lambda function:
--
-- > go :: Vector Float -> Vector Float -> Vector Int
-- > go = execute m
--
-- And finally run the model on the TPU by supplying the arguments to the
-- lambda. Of course, this can all be done in a single step as well:
--
-- > xs, ys :: Vector Float
-- > xs = ...
-- > ys = ...
-- >
-- > result :: Vector Word8
-- > result = execute m xs ys
--
execute :: Model f -> f
execute (Model afunR fun buffer) = eval afunR fun 0 []
  where
    eval :: AfunctionRepr a f r
         -> ModelAfun r
         -> Int
         -> [Feed]
         -> f
    eval AfunctionReprBody (Mbody outR shOut) _ aenv =
      let
          go :: R.ArraysR t -> R.Shapes t -> [Feed] -> StateT Int IO ([Feed], t)
          go TupRunit         ()         env = return (env, ())
          go (TupRpair aR bR) (shA, shB) env = do
            (env1, a) <- go aR shA env
            (env2, b) <- go bR shB env1
            return (env2, (a, b))
          go (TupRsingle arrR@(R.ArrayR shR eR)) sh env = StateT $ \i -> do
            let
                array :: TypeR t -> ArrayData t -> State Int [Feed]
                array TupRunit         ()     = return []
                array (TupRsingle aR)  a      = return <$> scalar aR a
                array (TupRpair aR bR) (a, b) = (++) <$> array aR a <*> array bR b

                scalar :: ScalarType t -> ArrayData t -> State Int Feed
                scalar (SingleScalarType t) = single t
                scalar (VectorScalarType _) = unsupported "SIMD-vector types"

                single :: SingleType t -> ArrayData t -> State Int Feed
                single (NumSingleType t) = num t

                num :: NumType t -> ArrayData t -> State Int Feed
                num (IntegralNumType t) = integral t
                num (FloatingNumType t) = floating t

                integral :: IntegralType t -> ArrayData t -> State Int Feed
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

                floating :: FloatingType t -> ArrayData t -> State Int Feed
                floating TypeFloat  = feed
                floating TypeDouble = feed
                floating TypeHalf   = unsupported "half-precision floating point"

                feed :: forall t. (Storable t, IsScalar t) => UniqueArray t -> State Int Feed
                feed ua = state $ \j ->
                  let tensorName      = printf "output%d_adata%d" i j
                      tensorType      = R.tagOfType (scalarType @t)
                      tensorDataBytes = castForeignPtr (unsafeGetValue (uniqueArrayData ua))
                      tensorSizeBytes = fromIntegral $ size shR sh * sizeOf (undefined :: t)
                  in
                  (Feed{..}, j+1)
            --
            arr@(R.Array _ adata) <- R.allocateArray arrR sh
            return ((evalState (array eR adata) 0 ++ env, arr), i+1)
      in
      unsafePerformIO $ do
        (aenv', out) <- evalStateT (go outR shOut []) 0
        B.unsafeUseAsCStringLen buffer $ \(p, n) ->
          withFeeds (aenv ++ aenv') (edgetpu_run p (fromIntegral n))
        return $ toArr out

    eval (AfunctionReprLam lamR) (Mlam lhs f) skip aenv = \arr ->
      let
          go :: ALeftHandSide t aenv aenv' -> t -> [Feed] -> State Int [Feed]
          go LeftHandSideWildcard{}                   _                    env = return env
          go (LeftHandSidePair aR bR)                 (a, b)               env = go bR b =<< go aR a env
          go (LeftHandSideSingle (R.ArrayR _shR _eR)) (R.Array _sh _adata) env = state $ \i ->
            let
                adata' = evalState (array _eR _adata) 0

                array :: TypeR t -> ArrayData t -> State Int [Feed]
                array TupRunit         ()     = return []
                array (TupRsingle aR)  a      = return <$> scalar aR a
                array (TupRpair aR bR) (a, b) = do
                  a' <- array aR a
                  b' <- array bR b
                  return (a' ++ b')

                scalar :: ScalarType t -> ArrayData t -> State Int Feed
                scalar (SingleScalarType t) = single t
                scalar (VectorScalarType _) = unsupported "SIMD-vector types"

                single :: SingleType t -> ArrayData t -> State Int Feed
                single (NumSingleType t) = num t

                num :: NumType t -> ArrayData t -> State Int Feed
                num (IntegralNumType t) = integral t
                num (FloatingNumType t) = floating t

                integral :: IntegralType t -> ArrayData t -> State Int Feed
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

                floating :: FloatingType t -> ArrayData t -> State Int Feed
                floating TypeFloat  = feed
                floating TypeDouble = feed
                floating TypeHalf   = unsupported "half-precision floating point"

                feed :: forall t. (Storable t, IsScalar t) => UniqueArray t -> State Int Feed
                feed ua = state $ \j ->
                  let tensorName      = printf "input%d_adata%d" i j
                      tensorType      = R.tagOfType (scalarType @t)
                      tensorDataBytes = castForeignPtr (unsafeGetValue (uniqueArrayData ua))
                      tensorSizeBytes = fromIntegral $ size _shR _sh * sizeOf (undefined :: t)
                  in
                  (Feed{..}, j+1)
            in
            (adata' ++ env, i+1)

          (aenv', next) = runState (go lhs (fromArr arr) aenv) skip
      in
      eval lamR f next aenv'
    eval _ _ _ _ = error "impossible"


data Feed = Feed { tensorName      :: String
                 , tensorType      :: Word8
                 , tensorDataBytes :: ForeignPtr Word8
                 , tensorSizeBytes :: Int64
                 }

withFeeds :: [Feed] -> (Ptr CString -> Ptr Word8 -> Ptr (Ptr Word8) -> Ptr Int64 -> Int64 -> IO a) -> IO a
withFeeds feeds k =
  withMany withCString    (map tensorName feeds)      $ \ns ->
  withMany withForeignPtr (map tensorDataBytes feeds) $ \ts ->
  withArray (map tensorType feeds)                    $ \kp ->
  withArray (map tensorSizeBytes feeds)               $ \sp ->
  withArray ns                                        $ \np ->
  withArray ts                                        $ \tp ->
    k np kp tp sp (genericLength feeds)

foreign import ccall "edgetpu_run"
    edgetpu_run
        :: CString              -- model_buffer
        -> Int64                -- buffer_size
        -> Ptr CString          -- tensor_name
        -> Ptr Word8            -- tensor_type
        -> Ptr (Ptr Word8)      -- tensor_data
        -> Ptr Int64            -- tensor_size_bytes
        -> Int64                -- tensor_count
        -> IO ()

