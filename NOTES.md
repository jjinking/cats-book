
# Chapter 1

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

Syntax
