{-# LANGUAGE GADTs #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE ViewPatterns #-}
module Data.Array.Accelerate.TensorFlow.Lite.ManualTest where

-- external libraries
import Control.Monad (when)
import Control.Monad.Identity (Identity(..))
import qualified Data.ByteString.Unsafe as B (unsafeUseAsCStringLen)
import Data.Int
import Data.List (genericLength)
import Data.ProtoLens (defMessage)
import qualified Data.Set as Set (fromList, empty)
import Data.Word (Word8)
import Foreign.C.String (CString, withCString)
import Foreign.ForeignPtr (ForeignPtr, castForeignPtr, withForeignPtr)
import Foreign.Marshal.Array (withArray)
import Foreign.Marshal.Utils (withMany)
import Foreign.Ptr (Ptr)
import Foreign.Storable (Storable(sizeOf))
import Lens.Family2 ((&), (.~))

-- accelerate
import Data.Array.Accelerate.Array.Data (GArrayDataR)
import qualified Data.Array.Accelerate.Sugar.Elt as A (Elt(EltR))
import Data.Array.Accelerate.Array.Unique (uniqueArrayData, UniqueArray)
import Data.Array.Accelerate.Lifetime (unsafeGetValue)
import qualified Data.Array.Accelerate.Representation.Array as R (Array(Array), ArrayR(ArrayR), allocateArray)
import Data.Array.Accelerate.Representation.Shape (size, ShapeR(ShapeRz, ShapeRsnoc))
import Data.Array.Accelerate.Representation.Type (TupR(TupRsingle))
import qualified Data.Array.Accelerate.Sugar.Array as S (Array(Array), fromFunction, toList, fromList)
import qualified Data.Array.Accelerate.Sugar.Shape as S (DIM1, Z(Z), (:.)((:.)))
import Data.Array.Accelerate.Type (IsScalar(scalarType))

-- tensorflow
import qualified Proto.Tensorflow.Core.Framework.Graph              as TF
import qualified Proto.Tensorflow.Core.Framework.Graph_Fields       as TF
import qualified TensorFlow.Build                                   as TF
import qualified TensorFlow.Core                                    as TF
import qualified TensorFlow.GenOps.Core                             as TF
import qualified TensorFlow.Ops                                     as TF hiding ( placeholder' )

-- accelerate-tensorflow{,-lite}
import Data.Array.Accelerate.TensorFlow.Lite.ConverterPy (withConverterPy, runConverterJob)
import Data.Array.Accelerate.TensorFlow.Lite.Compile (edgetpu_compile)
import qualified Data.Array.Accelerate.TensorFlow.Lite.Representation.Args as R (Args(Aparam, Aresult), ArgsNames(ANparam, ANresult), ArrArgNames (ArrArgNames), ArgName (ArgName), serialiseReprData, tagOfType)


test1 :: IO ()
test1 = do
  let makeInput :: Float -> S.Array S.DIM1 Float
      makeInput x = S.fromFunction (S.Z S.:. 10) (\(S.Z S.:. i) -> fromIntegral i * x)
  let sampleData = [(makeInput x, 10) | x <- [1.0 .. 10.0]]
  let input = makeInput 5.5

  let inputLength = 10
      outputLength = 3

  let model =
        let arg :: TF.Tensor TF.Build Float
            arg = TF.placeholder' ((TF.opName .~ TF.explicitName "input0_adata0")
                                  .(TF.opAttr "shape" .~ TF.Shape [inputLength]))
            outid = TF.identity' ((TF.opName .~ TF.explicitName "output0_adata0")
                                 .(TF.opAttr "shape" .~ TF.Shape [outputLength]))
        in outid $
            TF.add arg (TF.fill (TF.constant (TF.Shape [1]) [1 :: Int64]) (TF.scalar 1.0))
            -- TF.pack [TF.fill (TF.constant @Int64 (TF.Shape [0]) []) (TF.scalar 1.0)
            --         ,TF.fill (TF.constant @Int64 (TF.Shape [0]) []) (TF.scalar 2.0)
            --         ,TF.gather arg
            --               (TF.constant @Int64 (TF.Shape []) [3])
            --         ]
      inputIsUsed = True

  res <- manual111 @Float @Float model sampleData input (fromIntegral @Int64 @Int outputLength) inputIsUsed
  print res
  let correct = [fromIntegral i * 5.5 + 1 :: Float | i <- [0::Int .. 9]]
  putStrLn $ "Correct output: " ++ show correct
  putStrLn $ "Differences: " ++ show (zipWith (-) (S.toList res) correct)

test2 :: IO ()
test2 = do
  let makeInput :: Float -> S.Array S.DIM1 Float
      makeInput x = S.fromFunction (S.Z S.:. 10) (\(S.Z S.:. i) -> fromIntegral i * x)
  let sampleData = [(makeInput x, 10) | x <- [1 .. 10]]
  let input = S.fromList (S.Z S.:. 10) [3, 6, 4, 2, 3, 10, 20, 4, 3, 5]

  let inputLength = 10
      outputLength = 1

  let model =
        let arg :: TF.Tensor TF.Build Float
            arg = TF.placeholder' ((TF.opName .~ TF.explicitName "input0_adata0")
                                  .(TF.opAttr "shape" .~ TF.Shape [inputLength]))
            outid = TF.identity' ((TF.opName .~ TF.explicitName "output0_adata0")
                                 .(TF.opAttr "shape" .~ TF.Shape [outputLength]))
        in outid $
            -- TF.fill (TF.constant (TF.Shape [1]) [10 :: Int64]) (TF.scalar 1)
            -- TF.add (TF.constant (TF.Shape [10]) (replicate 10 0)) $ TF.range
            --     (TF.scalar (0 :: Int64))
            --     (TF.scalar (10 :: Int64))
            --     (TF.scalar (1 :: Int64))
            -- TF.constant (TF.Shape [10]) [0..9 :: Float]
            TF.reshape (TF.argMin arg (TF.scalar @Int32 0)) (TF.constant (TF.Shape [1]) [1 :: Int32])
      inputIsUsed = True

  res <- manual111 @Float @Int32 model sampleData input (fromIntegral @Int64 @Int outputLength) inputIsUsed
  print res


-- The 'TF.Tensor' in the first argument must be of the right type: it must take
-- 1 1D tensor with scalar (i.e. not a tuple) elements of type 'a', and return
-- 1 1D tensor with elements of type 'a'.
-- Such a 'TF.Tensor' usually comes packaged in a 'Tfun'.
--
-- Use e.g. S.fromFunction to create the input arrays.
--
-- It's unclear to me whether it is valid for the output lengths of the
-- representative sample data to be different from the output length of the
-- actual input.
manual111 :: forall a b. (IsScalar a, Storable a, A.Elt a, A.EltR a ~ a
                         ,GArrayDataR UniqueArray a ~ UniqueArray a
                         ,IsScalar b, Storable b, A.Elt b, A.EltR b ~ b
                         ,GArrayDataR UniqueArray b ~ UniqueArray b)
          => TF.Tensor TF.Build b       -- the model (must be of type Array DIM1 a -> Array DIM1 b)
          -> [(S.Array S.DIM1 a, Int)]  -- representative sample data with output lengths
          -> S.Array S.DIM1 a           -- the actual input
          -> Int                        -- 1D shape of the output array of the model
          -> Bool                       -- is the input used in the model?
          -> IO (S.Array S.DIM1 b)
manual111 model sampleData (S.Array (R.Array inputShape inputAdata)) resultlen inputused = do
  -- compile
  let makeRArgs :: (S.Array S.DIM1 a, Int)
                -> R.Args (R.Array ((), Int) a -> R.Array ((), Int) b)
      makeRArgs (S.Array x, n) =
        R.Aparam (TupRsingle (R.ArrayR (ShapeRsnoc ShapeRz) (TupRsingle (scalarType @a)))) x $
          R.Aresult (TupRsingle (ShapeRsnoc ShapeRz))
                    ((), n)

      argsnames = R.ANparam (TupRsingle (R.ArrArgNames (TupRsingle (R.ArgName "input0_adata0")))) R.ANresult
      actualInputs | inputused = Set.fromList ["input0_adata0"]
                   | otherwise = Set.empty

  let f' :: TF.GraphDef
      f'  = defMessage
              & TF.node .~ (runIdentity $ TF.evalBuildT (TF.render model >> TF.flushNodeBuffer))
  tflite <- withConverterPy $ \converter ->
              runConverterJob converter f' (R.serialiseReprData argsnames actualInputs (map makeRArgs sampleData))
  modelbs <- edgetpu_compile tflite

  -- execute
  let feed1 = Feed { tensorName      = "input0_adata0"
                   , tensorType      = R.tagOfType (scalarType @a)
                   , tensorDataBytes = castForeignPtr (unsafeGetValue (uniqueArrayData inputAdata))
                   , tensorSizeBytes = fromIntegral $ size (ShapeRsnoc ShapeRz) inputShape * sizeOf (undefined :: a) }
  out@(R.Array _ adata2) <- R.allocateArray (R.ArrayR (ShapeRsnoc ShapeRz) (TupRsingle (scalarType @b))) ((), resultlen)
  let feed2 = Feed { tensorName      = "output0_adata0"
                   , tensorType      = R.tagOfType (scalarType @b)
                   , tensorDataBytes = castForeignPtr (unsafeGetValue (uniqueArrayData adata2))
                   , tensorSizeBytes = fromIntegral $ size (ShapeRsnoc ShapeRz) ((), resultlen) * sizeOf (undefined :: b) }
  success <- B.unsafeUseAsCStringLen modelbs $ \(p, n) ->
    withFeeds [feed1, feed2] (edgetpu_run p (fromIntegral n))
  when (success /= 0) $ error "Error running on Edge TPU"
  return $ S.Array out


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
        -> IO Int64             -- 0 on success

