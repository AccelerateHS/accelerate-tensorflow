{-# LANGUAGE BangPatterns             #-}
{-# LANGUAGE ForeignFunctionInterface #-}
{-# LANGUAGE GADTs                    #-}
{-# LANGUAGE RecordWildCards          #-}
{-# LANGUAGE ScopedTypeVariables      #-}
{-# LANGUAGE TypeApplications         #-}
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

import Data.Array.Accelerate.AST
import Data.Array.Accelerate.AST.LeftHandSide
import Data.Array.Accelerate.Array.Data
import Data.Array.Accelerate.Array.Unique
import Data.Array.Accelerate.Lifetime
import Data.Array.Accelerate.Representation.Type
import Data.Array.Accelerate.Sugar.Array
import Data.Array.Accelerate.Trafo.Sharing
import Data.Array.Accelerate.Type
import qualified Data.Array.Accelerate.Representation.Array         as R
import qualified Data.Array.Accelerate.Representation.Shape         as R

import Data.Array.Accelerate.TensorFlow.CodeGen
import Data.Array.Accelerate.TensorFlow.CodeGen.AST
import Data.Array.Accelerate.TensorFlow.CodeGen.Base

import Data.Array.Accelerate.TensorFlow.Lite.Compile

import Control.Monad.State
import Foreign.C.String
import Foreign.Ptr
import Foreign.ForeignPtr
import Foreign.Marshal.Array
import Foreign.Marshal.Utils
import Foreign.Storable
import System.IO.Unsafe
import Text.Printf
import qualified Data.Vector.Storable                               as V


runN :: forall f. Afunction f => f -> AfunctionR f
runN acc =
  let
      !afun   = convertAfun acc
      !model  = buildAfun afun

      eval :: AfunctionRepr g (AfunctionR g) (ArraysFunctionR g)
           -> OpenTfun aenv (ArraysFunctionR g)
           -> Int
           -> [Feed]
           -> AfunctionR g
      eval AfunctionReprBody (Tbody funR _) _ aenv =
        let
            go :: R.ArraysR t -> [Feed] -> State Int ([Feed], t)
            go TupRunit         env = return (env, ())
            go (TupRpair arrR brrR) env = do
              (env1, a) <- go arrR env
              (env2, b) <- go brrR env1
              return (env2, (a, b))
            go (TupRsingle arrR@(R.ArrayR shR eR)) env = state $ \i ->
              let
                  sh                    = R.listToShape shR (repeat 256) -- TODO: figure out actual tensor size!
                  arr@(R.Array _ adata) = unsafePerformIO $ R.allocateArray arrR sh
                  env'                  = evalState (array eR adata) 0

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

                  feed :: forall t. Storable t => UniqueArray t -> State Int Feed
                  feed ua = state $ \j ->
                    let tensorName      = printf "output%d_adata%d" i j
                        tensorDataBytes = castForeignPtr (unsafeGetValue (uniqueArrayData ua))
                        tensorSizeBytes = fromIntegral $ R.size shR sh * sizeOf (undefined :: t)
                    in
                    (Feed{..}, j+1)
              in
              ((env' ++ env, arr), i+1)

            (aenv', out) = evalState (go funR []) 0
        in
        unsafePerformIO $ do
          path <- compileTfun model
          withCString path $ \p ->
            withFeeds (aenv ++ aenv') (edgetpu_run p)
          return $ toArr out

      eval (AfunctionReprLam lamR) (Tlam lhs f) skip aenv = \arr ->
        let
            go :: ALeftHandSide t aenv aenv' -> t -> [Feed] -> State Int [Feed]
            go LeftHandSideWildcard{}                 _                  env = return env
            go (LeftHandSidePair aR bR)               (a, b)             env = go bR b =<< go aR a env
            go (LeftHandSideSingle (R.ArrayR shR eR)) (R.Array sh adata) env = state $ \i ->
              let dims   = castForeignPtr
                         $ fst
                         $ V.unsafeToForeignPtr0
                         $ V.fromList
                         $ [ fromIntegral x :: Int64 | x <- R.shapeToList shR sh ]
                  sh'    = Feed { tensorName      = printf "input%d_shape" i
                                , tensorSizeBytes = fromIntegral $ R.size shR sh * sizeOf (undefined :: Int64)
                                , tensorDataBytes = dims
                                }
                  adata' = evalState (array eR adata) 0

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

                  feed :: forall t. Storable t => UniqueArray t -> State Int Feed
                  feed ua = state $ \j ->
                    let tensorName      = printf "input%d_adata%d" i j
                        tensorDataBytes = castForeignPtr (unsafeGetValue (uniqueArrayData ua))
                        tensorSizeBytes = fromIntegral $ R.size shR sh * sizeOf (undefined :: t)
                    in
                    (Feed{..}, j+1)
              in
              (sh' : adata' ++ env, i+1)

            (aenv', next) = runState (go lhs (fromArr arr) aenv) skip
        in
        eval lamR f next aenv'
      eval _ _ _ _ = error "impossible"
  in
  eval (afunctionRepr @f) model 0 []

data Feed = Feed { tensorName      :: String
                 , tensorDataBytes :: ForeignPtr Word8
                 , tensorSizeBytes :: Int64
                 }

withFeeds :: [Feed] -> (Ptr CString -> Ptr (Ptr Word8) -> Ptr Int64 -> Int64 -> IO a) -> IO a
withFeeds feeds k =
  withMany withCString    (map tensorName feeds)      $ \ns ->
  withMany withForeignPtr (map tensorDataBytes feeds) $ \ts ->
  withArray (map tensorSizeBytes feeds)               $ \sp ->
  withArray ns                                        $ \np ->
  withArray ts                                        $ \tp ->
    k np tp sp (fromIntegral (length feeds))

foreign import ccall "edgetpu_run" edgetpu_run :: CString -> Ptr CString -> Ptr (Ptr Word8) -> Ptr Int64 -> Int64 -> IO ()

