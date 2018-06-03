
# Chapter 1 Introduction

## Type classes

### Type class components

1) Type class itself

    - trait with type parameter(s)
  
2) Instances for particular types

    - implicit
  
3) Interface syntax for users

    - object containing functions that users can use for given typeclass
    - takes in instances of type class as implicit parameters
    - interface syntax enables use of methods using `.method()` syntax for existing types
  
The companion object of every type class in Cats has an `apply` method that looks for the specified type, which needs to be in scope

```
import cats.instances.int._

val showInt = Show.apply[Int]
```
  
## Working with implicits

  - `implicit` cannot be used at the top level - must be inside object or trait

  - Companion objects provide implicit scope

  - Recursive implicit resolution

## Cats

### Type class `Show`

`Show` is a type class for developer-friendly printing to console

Cats structure:

```scala
// Type class
import cats.Show
import cats._ // import all of cats type classes

// Instances of type class instances for existing types
import cats.instances.int._
import cats.instances.string._
import cats.instances.all._  // import all type class instances for the std lib

// Interface syntax
import cats.syntax.show._
import cats.syntax.all._  // imports all of the syntax

import cats.implicits._ // import all of type class instances and syntax
```

Define custom instances through convenience methods to avoid implementing the type class instance

### Type class `Eq`

Type-safe equality check instead of default object comparison provided by scala out of the box

```scala
import cats.Eq
 
val eqInt = Eq[Int] 
eqInt.eqv(123, 123) 
eqInt.eqv(123, "123") // compile error

// Syntax
import cats.syntax.eq._

123 === 123
123 =!= 321
123 === "123" // compile error
```

Options are a bit different

```scala
import cats.syntax.option._ // for some and none

1.some === none[Int]
// res11: Boolean = false

1.some =!= none[Int]
// res12: Boolean = true_
```

Compare custom types

```scala
import java.util.Date
import cats.instances.long._ // for Eq

implicit val dateEq: Eq[Date] = 
  Eq.instance[Date] { (date1, date2) =>
    date1.getTime === date2.getTime
  }

val x = new Date() // now
val y = new Date() // a bit later than now
x === x
// res13: Boolean = true
x === y
// res14: Boolean = false_
```

Variance

Cats prefers invariance - it's hard to set preference for which type class is applied
But you can define type class for Option[Int], and use type annotations, i.e.  `Some(1): Option[Int]` which will result in the type class for Option[Int] to be used. You can also use smart constructors like `Option.apply` and `Option.empty`


# Chapter 2 Monoids and Semigroups

## Monoids

Monoids implement two apis:

- combine: (A, A) => A
- empty: A

Properties of monoid:

- combine is associative
- empty is an identity element with respect to combine

### 

## Semigroup

Monoid w/o empty

## Note: Cats Kernel

Cats kernel package `cats.kernel` is a subproject of cats, aliased to `cats`
`Eq`, `Semigroup`, and `Monoid` are in the kernel.


# Chapter 3 Functors

## Section 3.2 More Examples of Functors

### Referential transparency example with Futures is confusing

Since scala.concurrent.Futures immediately computes, computations involving side effects will run immediately as they are encountered, and therefore it's hard to reason about Futures and computations with side-effects

### Mapping over Function1 is function composition

Since map doesn't actually run anything, we can think of it as lazily queueing up operations

## Section 3.3 Definition of Functor

Functor Laws

```scala
// Identity
fa.map(a => a) == fa

// Composition
fa.map(g(f(_))) == fa.map(f).map(g)_)))
```

## Section 3.4 Higher kinds and type constructors

### Higher Kinds

Kinds are types of types, describing the number of "holes" in a type"

### Type constructor

```scala

List    // type constructor, takes one parameter
List[A] // type, produced using a type parameter

// Analagous to
math.abs    // function, takes one parameter
math.abs(x) // value, produced using a value parameter
```

```scala
// To use F[_] syntax
import scala.language.higherKinds

// or in build.sbt
scalacOptions += "-language:higherKinds"
```

## Section 3.5 Functors in Cats

```scala
import cats.Functor
import cats.instances.list._   // for Functor
import cats.instances.option._ // for Functor
```

`Functor[F].lift` converts `f: A => B` to `F[A] => F[B]`

### Syntax

`Options` and `Lists` have their own builtin `map` function, which the scala compiler prefers

```scala
import cats.instances.function._ // for Functor
import cats.syntax.functor._     // for map

val func1 = (a: Int) => a + 1
val func2 = (a: Int) => a * 2
val func3 = (a: Int) => a + "!"
val func4 = func1.map(func2).map(func3)

func4(123)
// res1: String = 248!
``

## Contravariant and Invariant

Prepending operations to a chain, and building Bidirectional chain of operations

**Combinators*** like `map` and `contramap` and `filter` can be used to transform one typeclass to another.

```scala
// Custom instance
final case class Box[A](value: A)

implicit def boxPrintable[A](implicit printableA: Printable[A]): Printable[Box[A]] =
  printableA.contramap(_.value)
```

Invariant typeclasses implement `imap`, which takes in both `A => B` and `B => A`, useful for things like codecs and serializers

```scala
trait Invariant[F[_]] {
  def imap[A, B](fa: F[A])(f: A => B)(g: B => A): F[B] 
}
```

# Chapter 4 Monads

For sequencing computations

Functors allow sequencing of computations ignoring some complications but are limited in that they only allow this complication to occur once at the beginning of the sequence. They don’t account for further complications at each step in the sequence.

A monad’s flatMap method allows us to specify what happens next, taking into account an intermediate complication.

Every monad is a functor

## Monads in Cats

Monad extends two other type classes:

  - `FlatMap` for `flatMap` method
  - `Applicative` for `pure`
    - `Applicative` extends `Functor`, which provides `map`

`Futures` don't take implicit `ExecutionContext` in `pure` and `flatMap`.
We must have ec in scope when calling `Monad[Future]` to get an instance

## Useful monad instances

### Identity Monad

Enable calling monadic methods using plain values

Definition

```scala
package cats

type Id[A] = A
```

`pure` for `Id` is just the identity function

`map` and `flatMap` are the same


Usage

```scala
import cats.Id

sumSquare(3 : Id[Int], 4 : Id[Int])
```

Definition of `sumSquare` takes in an implicit `F[_]: Monad`, but scala cannot unify value types and type constructors when searching for implicits, so we have to explicitly write `Id[Int]` in the call the `sumSquare` above

### Either

```scala
import cats.syntax.either._ // for asRight

val a = 3.asRight[String]
// a: Either[String,Int] = Right(3)

val b = 4.asRight[String]
// b: Either[String,Int] = Right(4)

for {
x <- a
y <- b
} yield x*x + y*y
// res4: scala.util.Either[String,Int] = Right(25)
```

The `catchOnly` and `catchNonFatal` methods are great for capturing `Exceptions` as instances of `Either`

### MonadError

`MonadError` abstracts over `Monad`s, providing extra operations for raising and handling errors

`MonadError` extends `ApplicativeError`

### Eval

Useful for enforcing stack safety when working on large computations and data structures

`Eval` has 3 subtypes: `Now`, `Later`, `Always`

Scala computations can be eager, lazy, memoized

  - `val`s are eager and memoized similar to `Now`
  - `def`s are lazy and not memoized similar to `Always`
  - `lazy val`s are lazy and memoized similar to `Later`

(nothing is eager and not memoized?)

Mapping functions on an `Eval` has `def` semantics

Use `.memoize` to cache result up to a certain step in a chain of computations

Eval's `map`, `flatMap` and `defer` methods are **trampolined**

```scala
def factorial(n: BigInt): Eval[BigInt] =
  if(n == 1) {
    Eval.now(n)
  } else {
    Eval.defer(factorial(n - 1).map(_ * n))
  }
```

### Writer

`Writer[W, A]` where `W` contains the logs, and `A` contains the result

Carry logs along with computations, good for multi-threaded computations

`cats.data` package

`.value` to extract the value of type `A`

`.written` to extract the logs of type `W`

`.run` extracts both

Use log type `W` that has efficient append and concatenation function

```scala

val writer1 = for {
  a <- 10.pure[Logged]                   // Writer[Vector[String], Int](Vector(), 10)
  _ <- Vector("a", "b", "c").tell        // Writer[Vector[String], Unit](Vector(a, b, c))
  b <- 32.writer(Vector("x", "y", "z"))  // Writer[Vector[String], Int](Vector(x, y, z), 32)
} yield a + b
// writer1: cats.data.WriterT[cats.Id,Vector[String],Int] = WriterT((Vector(a, b, c, x, y, z),42))

writer1.run
// res4: cats.Id[(Vector[String], Int)] = (Vector(a, b, c, x, y, z),42)
```

Additional methods:

  - `mapWritten` maps over the `W`
  - `bimap` takes 2 function args, one for `A` and one for `W`
  - `mapBoth` takes single function that has 2 arguments of types `W` and `A`, respectively
  - `reset` clears the logs
  - `swap` flips the `W` and the `A`
  

### Reader

`type Reader[E, B] = E => B` (sort of)

Sequence operations that depend on some input, useful for dependency injection, often for configuration

The `map` method combines readers by chaining the output of the previous function `A => B` with the next function `B => C`

The `flatMap` method makes it easy to combine two readers `Reader[E, A]` with `Reader[E, B]` in a single for-comprehension


### State

Represent a computation as the monad

Pass around state object as part of atomic state computations that are combined together with combinators i.e. `map` and `flatMap` so that functions are still **pure** with no mutable state.

`State[S, A]` represent `S => (S, A)` where `S` is type of state and `A` is type of the result

Use `run`, `runS` and `runA` to run the monad, and the return value will be an `Eval` for stack safety, so we have to call `value` to extract the result

Sequencing state monads in for comprehension results in passing the state object through the monads, even though the code doesn't actually show that


## Custom monads

`tailRecM` method is an optimization used in Cats to limit stack space


# Chapter 5 Monad Transformers

Monad transformers provide way to compose monads to form a stack of 2 or more monads, which results in a new monad and eliminate the need for nested for comprehensions and pattern matching

Composing monads in a general way is hard because `flatMap` is hard to define for arbitrary nested monads, i.e. `M1[M2[A]]]`

Cats provides transformers for many monads, i.e. `EitherT` and `OptionT`

```scala
import cats.data.OptionT

type ListOption[A] = OptionT[List, A]  // transform a List[Option[A]] (note the composition is inside-out)

case class EitherT[F[_], E, A](stack: F[Either[E, A]]) {
  // etc...
}

type FutureEither[A] = EitherT[Future, String, A]

type FutureEitherOption[A] = OptionT[FutureEither, A] // Future[Either[String, Option[A]]]

// Create an instance
10.pure[FutureEitherOption]

// Using Kind Projector
123.pure[EitherT[Option, String, ?]]

// Create using apply
val errorStack1 = OptionT[ErrorOr, Int](Right(Some(10)))
// errorStack1: cats.data.OptionT[ErrorOr,Int] = OptionT(Right(Some(10)))

// Create using pure
val errorStack2 = 32.pure[ErrorOrOption]
// errorStack2: ErrorOrOption[Int] = OptionT(Right(Some(32)))
```

`Reader[T]` is type alias for `cats.data.Kleisli`

Use `.value` to extract the "untransformed monad stack" via "unpacking"

```scala
// Extracting the untransformed monad stack:
errorStack1.value
// res11: ErrorOr[Option[Int]] = Right(Some(10))

futureEitherOr
// res14: FutureEitherOption[Int] = OptionT(EitherT(Future(Success(Right(Some(42))))))

val intermediate = futureEitherOr.value
// intermediate: FutureEither[Option[Int]] = EitherT(Future(Success(Right(Some(42)))))

val stack = intermediate.value
// stack: scala.concurrent.Future[Either[String,Option[Int]]] = Future(Success(Right(Some(42))))

Await.result(stack, 1.second)
// res15: Either[String,Option[Int]] = Right(Some(42))
```

Many monads in Cats are defined using the transformer and its `Id` monad

```scala
type Reader[E, A] = ReaderT[Id, E, A] // = Kleisli[Id, E, A]
type Writer[W, A] = WriterT[Id, W, A]
type State[S, A]  = StateT[Id, S, A]
```

Usage patterns

- Single "super stack" that gets passed around everywhere in code
  - works for simple and uniform code base
- Use monad transformers as glue code locally, but expose **untransformed stacks** at module boundaries
  - larger and heterogeneous code base

# Chapter 6 Semigroupal and Applicative (Functor)

Limitations to `map` and `flatMap`

  - Fail fast, which means that in a series of computations, if one computation returns a fail value, the rest of the computations don't run
  - Combinators assume that the computations are dependent on the previous result, which prevents concurrent evaluation, i.e. even for `Futures` unless they are manually started outside functor block
  
## `Semigroupal`

`Semigroupal` provides the following:

  - joins multiple contexts into a single context ("context" usually refers to a type constructor) a.k.a. "zip"
  - sequence functions with multiple contexts
  
```scala
trait Semigroupal[F[_]] {
  def product[A, B](fa: F[A], fb: F[B]): F[(A, B)]
}
```

### Options

If either `fa` or `fb` are `None`, the product is `None`

For 2 <= `N` <= 22

  - Methods `tupleN` combine `N` contexts
  - Methods `mapN` apply a function to the values
  
```scala
Semigroupal.map2(Option(1), Option.empty[Int])(_ + _)
// res6: Option[Int] = None
```

#### Apply syntax

```scala
import cats.instances.option._ // for Semigroupal
import cats.syntax.apply._     // for tupled and mapN

(Option(123), Option("abc")).tupled
// res7: Option[(Int, String)] = Some((123,abc))

case class Cat(name: String, born: Int, color: String)

(
  Option("Garfield"),
  Option(1978),
  Option("Orange & black")
).mapN(Cat.apply)
// res9: Option[Cat] = Some(Cat(Garfield,1978,Orange & black))
```

### Futures

```scala
import cats.Semigroupal
import cats.instances.future._ // for Semigroupal
import scala.concurrent._
import scala.concurrent.duration._
import scala.concurrent.ExecutionContext.Implicits.global
import scala.language.higherKinds

val futurePair = Semigroupal[Future].
  product(Future("Hello"), Future(123))

Await.result(futurePair, 1.second)
// res1: (String, Int) = (Hello,123)
```

### Lists

Surprisingly, instead of "zipping" the lists, the `product` function returns a cartesian product (see implementation of `product` using flatMap in file Ch6.scala)

```scala
import cats.Semigroupal
import cats.instances.list._ // for Semigroupal

Semigroupal[List].product(List(1, 2), List(3, 4))
// res5: List[(Int, Int)] = List((1,3), (1,4), (2,3), (2,4))
```

### Either

Surprisingly, the `product` function is fail-fast (because product uses `flatMap`)

```scala
import cats.instances.either._ // for Semigroupal

type ErrorOr[A] = Either[Vector[String], A]

Semigroupal[ErrorOr].product(
  Left(Vector("Error 1")),
  Left(Vector("Error 2"))
)
// res7: ErrorOr[(Nothing, Nothing)] = Left(Vector(Error 1))
```

### `Semigroupal` applied to `Monads`

`Semigroupal` applied to `Monads` may result in surprising and less useful behaviors, to provide high-level consistent semantics with `flatMap`

`Semigroupal`s can be useful for data types that do not have `Monad` instances

### Validated

Instance of `Semigroupal` but no instance of `Monad`, so `product` method is free to accumulate errors

```scala
import cats.Semigroupal
import cats.data.Validated
import cats.instances.list._ // for Monoid

type AllErrorsOr[A] = Validated[List[String], A]

Semigroupal[AllErrorsOr].product(
  Validated.invalid(List("Error 1")),
  Validated.invalid(List("Error 2"))
)
// res1: AllErrorsOr[(Nothing, Nothing)] = Invalid(List(Error 1, Error 2))
```

Using "smart constructors"

```scala
val v = Validated.valid[List[String], Int](123)
// v: cats.data.Validated[List[String],Int] = Valid(123)

val i = Validated.invalid[List[String], Int](List("Badness"))
// i: cats.data.Validated[List[String],Int] = Invalid(List(Badness))
```

Using syntax extension

```scala
import cats.syntax.validated._ // for valid and invalid

123.valid[List[String]]
// res2: cats.data.Validated[List[String],Int] = Valid(123)

List("Badness").invalid[Int]
// res3: cats.data.Validated[List[String],Int] = Invalid(List(Badness))
```

Other methods to create instances of `Validated`

```scala
// Using applicatives
import cats.syntax.applicative._      // for pure
import cats.syntax.applicativeError._ // for raiseError

type ErrorsOr[A] = Validated[List[String], A]

123.pure[ErrorsOr]
// res5: ErrorsOr[Int] = Valid(123)

List("Badness").raiseError[ErrorsOr, Int]
// res6: ErrorsOr[Int] = Invalid(List(Badness))

// From other data types
Validated.catchOnly[NumberFormatException]("foo".toInt)
// res7: cats.data.Validated[NumberFormatException,Int] = Invalid(java.lang.NumberFormatException: For input string: "foo")

Validated.catchNonFatal(sys.error("Badness"))
// res8: cats.data.Validated[Throwable,Nothing] = Invalid(java.lang.RuntimeException: Badness)

Validated.fromTry(scala.util.Try("foo".toInt))
// res9: cats.data.Validated[Throwable,Int] = Invalid(java.lang.NumberFormatException: For input string: "foo")

Validated.fromEither[String, Int](Left("Badness"))
// res10: cats.data.Validated[String,Int] = Invalid(Badness)

Validated.fromOption[String, Int](None, "Badness")
// res11: cats.data.Validated[String,Int] = Invalid(Badness)
```

Combine instances of `Validated`

```scala
type AllErrorsOr[A] = Validated[String, A] // Fix error type

import cats.instances.string._ // for Semigroup

Semigroupal[AllErrorsOr]
// res13: cats.Semigroupal[AllErrorsOr] = cats.data.ValidatedInstances$$anon$1@7be29203

// Accumulate errors
import cats.syntax.apply._ // for tupled

(
  "Error 1".invalid[Int],
  "Error 2".invalid[Int]
).tupled
// res14: cats.data.Validated[String,(Int, Int)] = Invalid(Error 1Error 2)

import cats.instances.vector._ // for Semigroupal

(
  Vector(404).invalid[Int],
  Vector(500).invalid[Int]
).tupled
// res15: cats.data.Validated[scala.collection.immutable.Vector[Int],(Int, Int)] = Invalid(Vector(404, 500))

```

`Validated` also has methods similar to `Either`: `map`, `leftMap`, `bimap`

no `flatMap` since it's not a `Monad`

Convert to `Either` using `toEither` and convert back using `toValidated`

```scala
41.valid[String].withEither(_.flatMap(n => Right(n + 1)))
// res24: cats.data.Validated[String,Int] = Valid(42)
```

## `Apply` and `Applicative`

`Semigroupal` and `Applicative` basically provide a way of joining "contexts" (`F[_]`).

`Applicative` functors provides a way of **applying** functions to parameters within a context

Cats has two type classes for applicatives

`Apply` extends `Semigroupal` and `Functor` and add `ap` method

`Applicative` extends `Apply`, and adds `pure`
  - source of the `pure` method in monads

Analogy: `Applicative:Apply::Monoid::Semigroup`

Hierarchy of Sequencing type classes [diagram](https://raw.githubusercontent.com/underscoreio/advanced-scala/develop/src/pages/applicatives/hierarchy.png)

(Mathematical) power vs constraint: More constraints on a data type means more guarantees about specific behavior, but the behavior is less general

`Mondad`s are usually flexible enough to provide a wide range of behaviors while still restrictive enough to provide strong guarantees, but it's not always the right choice because it imposes strict sequencing on the computations. For these situations, `Applicatives` and `Semigroupals` come in handy, but lose ability to `flatMap`

## Summary

Monads and functors are the most widely used sequencing data types while semigroupals and applicatives are the most general.

Most common uses of semigroupals and applicatives are to combine independent values, i.e. result of validating a form. Cats provides `Validated` data type for this purpse

# Chapter 7 Foldable and Traverse

## Foldable

Typeclass implements `foldLeft` and `foldRight`

Used with `Monoid` and `Eval`

### Exercise 7.1.3

```scala
# map in terms of foldRight

def map[A, B](l: List[A], f: A => B): List[B] =
  l.foldRight(List.empty[B])((a, accum) => f(a) :: accum)
  
def flatMap[A, B](l: List[A], f: A => List[B]): List[B] =
  l.foldRight(List.empty[B])((a, accum) => f(a) ::: accum) // Use ++ for general `Traversable`, but ::: makes it right-associative for List
  
def filter[A](l: List[A], p: A => Boolean): List[A] =
  l.foldRight(List.empty[A])((a, accum) => if (p(a)) a :: accum else accum)

def sum[A, B](l: List[A])(z: B)(f: (A, A) => B): B =
  l.foldRight(z)(f)

import cats.Monoid

def sum2[A](l: List[A])(implicit ev: Monoid[A]): A =
  l.foldRight(ev.empty)(ev.combine)
```

### Typeclasses

Example with `List`

```scala
import cats.Foldable
import cats.instances.list._ // for Foldable

val ints = List(1, 2, 3)

Foldable[List].foldLeft(ints, 0)(_ + _)
// res1: Int = 6
```

Example with `Option`

```scala
import cats.instances.option._ // for Foldable

val maybeInt = Option(123)

Foldable[Option].foldLeft(maybeInt, 10)(_ * _)
// res3: Int = 1230
```

### Stack safe `foldRight`

Default `foldRight` method on sequences is not stack safe for streams, but `Foldable` uses `Eval` monad to make it stack safe. `List` and `Vector` provide stack safe implementations of `foldRight`

```scala
import cats.instances.stream._ // for Foldable

val eval: Eval[Long] =
  Foldable[Stream].
    foldRight(bigData, Eval.now(0L)) { (num, eval) =>
      eval.map(_ + num)
    }

eval.value
// res7: Long = 5000050000
```

### Folding with Monoids

`combineAll` (aka `fold`) combine all elements in the sequence using their `Monoid`

```scala
import cats.instances.int._ // for Monoid

Foldable[List].combineAll(List(1, 2, 3))
// res12: Int = 6
```

`foldMap` maps a function to sequence elements, then combines them using a `Monoid`

```scala
import cats.instances.string._ // for Monoid

Foldable[List].foldMap(List(1, 2, 3))(_.toString)
// res13: String = 123
```

Compose `Foldables` for nested sequences

```scala
import cats.instances.vector._ // for Monoid

val ints = List(Vector(1, 2, 3), Vector(4, 5, 6))

(Foldable[List] compose Foldable[Vector]).combineAll(ints)
// res15: Int = 21
```

Syntax extension

```scala
import cats.syntax.foldable._ // for combineAll and foldMap

List(1, 2, 3).combineAll
// res16: Int = 6

List(1, 2, 3).foldMap(_.toString)
// res17: String = 123
```

When calling `foldLeft`, `Foldable` method is used if the sequence object doesn't already have `foldLeft` implemented. No need to worry about which one is being used.

To guarantee stack safety when using `foldRight`, use `Eval` as accumulator type.

## Traverse

Higher level tool, leverages `Applicatives` to provide a more convenient, lawful pattern for iteration

### `traverse` method on `Future`

`Future.traverse[A, B]: (List[A]) => (A => Future[B]) => Future[List[B]]`

Example usage:

```scala
import scala.concurrent._
import scala.concurrent.duration._
import scala.concurrent.ExecutionContext.Implicits.global

val hostnames = List(
  "alpha.example.com",
  "beta.example.com",
  "gamma.demo.com"
)

def getUptime(hostname: String): Future[Int] =
  Future(hostname.length * 60) // just for demonstration
  
val allUptimes: Future[List[Int]] =
  hostnames.foldLeft(Future(List.empty[Int])) {
    (accum, host) =>
      val uptime = getUptime(host)
      for {
        accum  <- accum
        uptime <- uptime
      } yield accum :+ uptime
  }

Await.result(allUptimes, 1.second)
// res2: List[Int] = List(1020, 960, 840)

val allUptimesTraverse: Future[List[Int]] =
  Future.traverse(hostnames)(getUptime)

Await.result(allUptimesTraverse, 1.second)
// res3: List[Int] = List(1020, 960, 840)
```

### `sequence` method on `Future`

`Future.sequence[B]: (List[Future[B]]) => Future[List[B]]`

### Generalizing `traverse` with `Applicative`

#### Applicatives and Lists

The zero value is the `pure` value of `F[_]: Applicative` with the empty `List`, and the combine function is `mapN`

```scala
import scala.language.higherKinds

def listTraverse[F[_]: Applicative, A, B]
      (list: List[A])(func: A => F[B]): F[List[B]] =
  list.foldLeft(List.empty[B].pure[F]) { (accum, item) =>
    (accum, func(item)).mapN(_ :+ _)
  }

def listSequence[F[_]: Applicative, B]
      (list: List[F[B]]): F[List[B]] =
  listTraverse(list)(identity)
```

Example: traversing with `Validated`

```scala
import cats.data.Validated
import cats.instances.list._ // for Monoid

type ErrorsOr[A] = Validated[List[String], A]

def process(inputs: List[Int]): ErrorsOr[List[Int]] =
  listTraverse(inputs) { n =>
    if(n % 2 == 0) {
      Validated.valid(n)
    } else {
      Validated.invalid(List(s"$n is not even"))
    }
  }
  
process(List(2, 4, 6))
// res26: ErrorsOr[List[Int]] = Valid(List(2, 4, 6))

process(List(1, 2, 3))
// res27: ErrorsOr[List[Int]] = Invalid(List(1 is not even, 3 is not even))
```

#### Generalizing `traverse` for any sequence type using Cats' `Traverse`

```scala
package cats

// F[_] is the sequence type, like List
// G[_] is the Applicative, like Future
trait Traverse[F[_]] {
  def traverse[G[_]: Applicative, A, B]
      (inputs: F[A])(func: A => G[B]): G[F[B]]

  def sequence[G[_]: Applicative, B]
      (inputs: F[G[B]]): G[F[B]] =
    traverse(inputs)(identity)
}
```

Cats provides instances of `Traverse` for sequence types: `List`, `Vector`, `Stream`, `Option`, `Either`, etc.
Use `Traverse.apply` to create instances, and use `traverse` and `sequence` methods

```scala
import cats.Traverse
import cats.instances.future._ // for Applicative
import cats.instances.list._   // for Traverse

val totalUptime: Future[List[Int]] =
  Traverse[List].traverse(hostnames)(getUptime)

Await.result(totalUptime, 1.second)
// res1: List[Int] = List(1020, 960, 840)

val numbers = List(Future(1), Future(2), Future(3))

val numbers2: Future[List[Int]] =
  Traverse[List].sequence(numbers)

Await.result(numbers2, 1.second)
// res3: List[Int] = List(1, 2, 3)

// -----------------------------------------------
// With extended syntax:

import cats.syntax.traverse._ // for sequence and traverse

Await.result(hostnames.traverse(getUptime), 1.second)
// res4: List[Int] = List(1020, 960, 840)

val numbers = List(Future(1), Future(2), Future(3))
Await.result(numbers.sequence, 1.second)
// res5: List[Int] = List(1, 2, 3)
```

# Chapter 8 Case Study: Testing Asynchronous Code

## 8.1 Abstracting over Type Constructors
```scala
import scala.language.higherKinds
import cats.Id

trait UptimeClient[F[_]] {
  def getUptime(hostname: String): F[Int]
}

trait RealUptimeClient extends UptimeClient[Future] {
  def getUptime(hostname: String): Future[Int]
}

trait TestUptimeClient extends UptimeClient[Id] {
  def getUptime(hostname: String): Id[Int]
}

// Fleshed out class
class TestUptimeClient((hosts: Map[String, Int])) extends UptimeClient[Id] {
  def getUptime(hostname: String): Int = hosts.getOrElse(hostName, 0)
}
```

## 8.2 Abstracting over Monads

```scala
import cats.Applicative
import cats.syntax.functor._ // for map
import scala.language.higherKinds

class UptimeService[F[_]: Applicative](client: UptimeClient[F]) {
  def getTotalUptime(hostnames: List[String]): F[Int] =
    hostnames.traverse(client.getUptime).map(_.sum)
}

def testTotalUptime() = {
  val hosts    = Map("host1" -> 10, "host2" -> 6)
  val client   = new TestUptimeClient(hosts)
  val service  = new UptimeService(client)
  val actual   = service.getTotalUptime(hosts.keys.toList)
  val expected = hosts.values.sum
  assert(actual == expected)
}

testTotalUptime()
```

## 8.3 Summary

"The mathematical laws ... ensure that they (type classes like `Monad` and `Applicative`) work together with a consistent set of semantics."

"We used Applicative in this case study because it was the least powerful type class that did what we needed. If we had required flatMap, we could have swapped out Applicative for Monad. If we had needed to abstract over different sequence types, we could have used Traverse."


# 9 Case Study: Map-Reduce

`Functor` for map, `Monoid` for map-reduce:

- Reduce is called `fold` in scala, and must be **associative**

- Each reduce must be seeded with a neutral element

## 9.2 foldMap

```scala
import cats.Monoid
import cats.syntax.semigroup._ // for |+|

def foldMap[A, B: Monoid](vA: Vector[A])(f: A => B): B =
  vA.map(f).foldLeft(Monoid[B].empty)(_ |+| _)
```

## 9.3 parallelFoldMap

**Importing an `ExecutionContext.Implicits.global`**: Allocates a thread pool with one thread per CPU in the running machine

```scala
import cats.{Monad, Monoid}
import cats.instances.list._   // for Traverse
import cats.instances.future._ // for Monad and Monoid
import cats.syntax.traverse._  // for sequence
import cats.syntax.semigroup._ // for |+|
import scala.concurrent._
import scala.concurrent.ExecutionContext.Implicits.global

def listToFut[A](l: List[A], f: A => B)(implicit ec: ExecutionContext): Future[B] = {
  // val futL: Future[List[A]] = Monad[Future].pure(l)
  // val futB: Future[B] = futL.map(lA => foldMap(lA)(f))
  // futB
  Monad[Future].pure(l).map(lA => foldMap(lA)(f))
}

def parallelFoldMap[A, B: Monoid](values: Vector[A])(func: A => B): Future[B] = {
  val numCores  = Runtime.getRuntime.availableProcessors
  val groupSize = (1.0 * values.size / numCores).ceil.toInt
  // val grouped: List[List[A]] = values.grouped(groupSize).toList
  // val groupedF: List[Future[B]] = grouped.map(l => listToFut(l, func))
  // val futGroup: Future[List[B]] = groupedF.sequence
  // val futB: Future[B] = futGroup.map(lB => foldMap(lB)(id))
  // futB
  val grouped: List[List[A]] = values.grouped(groupSize).toList
  for {
    lB <- grouped.map(l => listToFut(l, func)).sequence  
  } yield foldMap(lB)(id)  
}
```

# 10 Case Study: Data Validation

```scala
import cats.Semigroup
import cats.syntax.apply._     // for mapN
import cats.syntax.semigroup._ // for |+|
import cats.data.Validated
import cats.data.Validated._   // for Valid and Invalid

sealed trait Predicate[E, A] {
  def and(that: Predicate[E, A]): Predicate[E, A] =
    And(this, that)

  def or(that: Predicate[E, A]): Predicate[E, A] =
    Or(this, that)

  def apply(a: A)(implicit s: Semigroup[E]): Validated[E, A] =
    this match {
      case Pure(func) =>
        func(a)

      case And(left, right) =>
        (left(a), right(a)).mapN((_, _) => a)

      case Or(left, right) =>
        left(a) match {
          case Valid(a1)   => Valid(a)
          case Invalid(e1) =>
            right(a) match {
              case Valid(a2)   => Valid(a)
              case Invalid(e2) => Invalid(e1 |+| e2)
            }
        }
    }
}

final case class And[E, A](
  left: Predicate[E, A],
  right: Predicate[E, A]) extends Predicate[E, A]

final case class Or[E, A](
  left: Predicate[E, A],
  right: Predicate[E, A]) extends Predicate[E, A]

final case class Pure[E, A](
  func: A => Validated[E, A]) extends Predicate[E, A]


// Allow transformations
sealed trait Check[E, A, B] {
  def apply(in: A)(implicit s: Semigroup[E]): Validated[E, B]

  def map[C](f: B => C): Check[E, A, C] =
    Map[E, A, B, C](this, f)
}

object Check {
  def apply[E, A](pred: Predicate[E, A]): Check[E, A, A] =
    Pure(pred)
    
  def andThen[C](that: Check[E, B, C]): Check[E, A, C] =
    AndThen[E, A, B, C](this, that)
}

final case class Map[E, A, B, C](
  check: Check[E, A, B],
  func: B => C) extends Check[E, A, C] {

  def apply(in: A)(implicit s: Semigroup[E]): Validated[E, C] =
    check(in).map(func)
}

final case class Pure[E, A](
  pred: Predicate[E, A]) extends Check[E, A, A] {

  def apply(in: A)(implicit s: Semigroup[E]): Validated[E, A] =
    pred(in)
}

final case class AndThen[E, A, B, C](
  check1: Check[E, A, B],
  check2: Check[E, B, C]) extends Check[E, A, C] {

  def apply(a: A)(implicit s: Semigroup[E]): Validated[E, C] =
    check1(a).withEither(_.flatMap(b => check2(b).toEither))
}
```


# 11 Case Study: CRDTs

Commutative Replicated Data Types

Strong consistency
- High latency - many messages sent between machines in cluster
- Low availability - network partition may cause machines to refuse updates to prevent inconsistencies

Eventual consistency
- Low latency - less communication between machines
- High availability - machines will still accept updates even during network partition, and reconcile changes when network is back up


