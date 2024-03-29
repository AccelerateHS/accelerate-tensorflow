{-# LANGUAGE BangPatterns             #-}
{-# LANGUAGE FlexibleInstances        #-}
{-# LANGUAGE ForeignFunctionInterface #-}
{-# LANGUAGE GADTs                    #-}
{-# LANGUAGE FlexibleContexts         #-}
{-# LANGUAGE RecordWildCards          #-}
{-# LANGUAGE ScopedTypeVariables      #-}
{-# LANGUAGE TypeApplications         #-}
{-|
Module      : Data.Array.Accelerate.TensorFlow.Lite
Copyright   : [2021..2022] The Accelerate Team
License     : BSD3

Maintainer  : Trevor L. McDonell <trevor.mcdonell@gmail.com>
Stability   : experimental
Portability : non-portable (GHC extensions)

This is a backend for Accelerate that allows running programs on a Coral
(<https://coral.ai>) Edge TPU. Unlike the Accelerate interpreter and the LLVM
backends, there is no single @run@/@runN@ function, since compilation and
execution have significantly different requirements in this context. However,
running a model is not very difficult either.

The simplest way to compile a model is to use 'compile', which takes an
'Data.Array.Accelerate.Acc' function just like "runN" from the standard
backends. It furthermore takes a list of examples of input data; for more
details, see the documentation below in the [Representative sample
data](#g:sampledata) section. The resulting 'Model' can then be executed on the
device using 'execute'.

Note that for e.g. the default Accelerate interpreter we have:

@
'Data.Array.Accelerate.Interpreter.runN' :: 'Afunction' f => f -> 'AfunctionR' f
@

Here, we have:

@
'compile' :: 'Afunction' f => f -> 'RepresentativeData' ('AfunctionR' f) -> 'Model' ('AfunctionR' f)
'execute' :: 'Model' f -> f
@

so the composition of 'execute' and 'compile' (including some sample data for
calibration) indeed results in the same signature as the more familiar
@runN@-style functions.

However, simply using 'compile' and 'execute' will yield very unsatisfactory performance:

* Compilation involves calling out to the TensorFlow Lite API in Python, which
    entails starting a Python interpreter, loading TensorFlow in that, and then
    performing the actual operation (conversion of a TensorFlow model to a
    TFLite model). Most of the time that this takes is spent in starting the
    Python interpreter and loading TensorFlow, hence if you are compiling a
    model more than once, it can save a lot of execution time to keep a Python
    process running with TensorFlow imported, ready to convert multiple models.
    The disadvantage of doing so is that such a Python process consumes around
    300MB of RAM when idle.

    If you want to share a Python process over multiple compilations, use
    'withConverterPy' to start a Python process (or 'withConverterPy'' for more
    configuration options), and then use 'compileWith' instead of 'compile' to
    use the running service for the TFLite conversion.

* Execution of a compiled model on a TPU involves requesting a "device context"
    using @libedgetpu@. Doing so takes about 2.6 seconds (!) on our machine, and
    is thus /very/ costly. If you want to execute multiple models (or the same
    model multiple times), you should request such a device context once only,
    and perform all model executions using the existing context.

    You can do this by using 'withDeviceContext'. This will request a new TPU
    device context and retain it as long as the passed 'IO' operation runs;
    after the argument returns, the device context is released. You do not need
    to do anything else; 'execute' will automatically pick up the existing held
    device context if there is one.

=== Usage example

For a "Hello World"-style example of how to use this backend, see
<https://github.com/AccelerateHS/accelerate-tensorflow/blob/master/accelerate-tensorflow-lite/test/examples/Main.hs>.

=== Implementation documentation

Some documentation on the internals of the backend can be found in the repository at <https://github.com/AccelerateHS/accelerate-tensorflow/blob/master/doc>.

-}

module Data.Array.Accelerate.TensorFlow.Lite (

  -- * Representative sample data #sampledata#
  --
  -- | A TPU model is quantised, meaning that floating-point numbers in the
  -- source model are actually lowered to 8-bit integer arithmetic, under some
  -- affine transformation (i.e. the int8 ranges between a minimum and a maximum
  -- float value, with equal spacing in between).
  --
  -- To calibrate this quantisation, the compilation process needs representative
  -- sample input data, together with the shape of the output of the model.

  RepresentativeData, Args(..), Shapes,

  -- * Compiling a model
  --
  -- | The first step of running a TPU program is compiling the model to a
  -- 'Model'. This can be done using 'compile' or 'compileWith'.

  Model,
  compile,
  compileWith,
  ConverterPy,
  withConverterPy,
  withConverterPy',
  ConverterSettings(..),
  defaultConverterSettings,

  -- * Executing a compiled model
  --
  -- | After a model has been compiled, it can be executed on the TPU hardware.

  execute,
  withDeviceContext,

  -- * Special cases
  --
  -- | These functions are additional Accelerate primitives with a special
  -- implementation in TensorFlow. They have a fallback implementation (using
  -- 'Data.Array.Accelerate.foreignAcc') that is implemented in standard
  -- Accelerate, and hence work also on other backends (via the fallback
  -- implementation).

  argMin, argMax, append,

  -- * Model serialisation
  --
  -- | These functions implement a bespoke model serialisation format (i.e. not
  -- a TensorFlow-specific format). They can be used if you want to create a
  -- model once, then re-calibrate and re-run it multiple times in various
  -- invocations of your program.
  --
  encodeModel, decodeModel,

  -- * Re-exports from Accelerate
  Smart.Acc, Sugar.Arrays, Afunction, AfunctionR,

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

import Data.Array.Accelerate.TensorFlow.TypeDicts

import Data.Array.Accelerate.TensorFlow

import Data.Array.Accelerate.TensorFlow.Lite.CodeGen
import Data.Array.Accelerate.TensorFlow.Lite.Compile
import Data.Array.Accelerate.TensorFlow.Lite.ConverterPy
import Data.Array.Accelerate.TensorFlow.Lite.Model
import Data.Array.Accelerate.TensorFlow.Lite.Sugar.Args
import Data.Array.Accelerate.TensorFlow.Lite.Sugar.Shapes
import qualified Data.Array.Accelerate.TensorFlow.Lite.Representation.Args    as R
import qualified Data.Array.Accelerate.TensorFlow.Lite.Representation.Shapes  as R

import Control.Exception                                                      ( bracket )
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
-- The type @f@ is the type of the function as usually passed to @runN@, and
-- as passed to 'Data.Array.Accelerate.TensorFlow.Lite.compile' with the TPU
-- backend.
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
-- Note that e.g. @'AfunctionR' ('Data.Array.Accelerate.Acc' a -> 'Data.Array.Accelerate.Acc' b -> 'Data.Array.Accelerate.Acc' c) = a -> b -> c@.
--
-- The compiled model can then be evaluated using 'execute' or serialised
-- using 'encodeModel'.
--
compile :: forall f. Afunction f => f -> RepresentativeData (AfunctionR f) -> Model (AfunctionR f)
compile acc args =
  unsafePerformIO $ do
    withConverterPy $ \converter ->
      compileWith' converter acc args

-- | The same as 'compile', but with an explicit running converter.py instance.
-- Sharing a converter.py instance over multiple compilations saves Python and
-- TensorFlow startup time.
compileWith :: forall f. Afunction f => ConverterPy -> f -> RepresentativeData (AfunctionR f) -> Model (AfunctionR f)
compileWith converter acc args = unsafePerformIO $ compileWith' converter acc args

compileWith' :: forall f. Afunction f => ConverterPy -> f -> RepresentativeData (AfunctionR f) -> IO (Model (AfunctionR f))
compileWith' converter acc args = do
  Model afunR (modelAfun afunR tfun x) <$> compileTfunIn converter tfun argsnames (x:xs)
  where
    !afunR = afunctionRepr @f
    !afun  = simplifyAfun (convertAfun acc)
    (!tfun, !argsnames) = buildAfunWith afun x
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
-- __Note about contexts__:
-- If a TPU device context has not yet been acquired using 'withDeviceContext',
-- 'execute' will open a new device context just for this evaluation and close
-- it when the computation is finished. Opening a new device context is very
-- slow (about 2.6 seconds on our system), so if you care about performance, it
-- is probably worth doing it once only.
--
-- Furthermore, be aware that /acquiring/ a context steals the context from any
-- other process that has it open! It seems that only one context can be open
-- on a TPU device at one time, and the /last/ requester wins.
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
                array (TupRsingle aR)  a      = return <$> buildTypeDictsScalar aR feed a
                array (TupRpair aR bR) (a, b) = (++) <$> array aR a <*> array bR b

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
        success <- B.unsafeUseAsCStringLen buffer $ \(p, n) ->
          withFeeds (aenv ++ aenv') (edgetpu_run p (fromIntegral n))
        when (success /= 0) $ error "Error running on Edge TPU"
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
                array (TupRsingle aR)  a      = return <$> buildTypeDictsScalar aR feed a
                array (TupRpair aR bR) (a, b) = do
                  a' <- array aR a
                  b' <- array bR b
                  return (a' ++ b')

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


-- | Open a TPU device context
--
-- Inside the IO action passed to 'withDeviceContext', a TPU device context
-- will be kept open. This context will be used by 'execute', removing most of
-- the (roughly 2.6 seconds!) overhead of running a TPU computation, leaving
-- mostly just the computation itself.
--
-- This function is thread-safe and idempotent (i.e. nesting it multiple times
-- is okay; any inner calls will not take effect).
--
-- It is not necessary to call this function; 'execute' will create a device
-- context (and close it immediately after its computation is done) if none was
-- open yet. However, opening a context is very slow, so using
-- 'withDeviceContext' is worth it for performance.
--
-- Furthermore, be aware that /acquiring/ a context steals the context from any
-- other process that has it open! It seems that only one context can be open
-- on a TPU device at one time, and the /last/ requester wins.
--
withDeviceContext :: IO a -> IO a
withDeviceContext action =
  bracket (do opened <- edgetpu_open_device_context
              case opened of
                0 -> return edgetpu_close_device_context
                1 -> return (return ())
                _ -> ioError (userError "withDeviceContext: Error opening TPU device context"))
          (\closer -> closer)
          (\_ -> action)


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

-- cbits/edgetpu.cc
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

-- cbits/edgetpu.cc
-- Returns 0 if ok, 1 if already open, 2 on error
foreign import ccall "edgetpu_open_device_context"
    edgetpu_open_device_context :: IO Int64

-- cbits/edgetpu.cc
foreign import ccall "edgetpu_close_device_context"
    edgetpu_close_device_context :: IO ()
