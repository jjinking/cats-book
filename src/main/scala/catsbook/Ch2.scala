package catsbook

import cats.{Monoid, Semigroup}

object Ch2 {

  /**
    * Truth about Monoids
    */
  object Section3 {
    val monoidBooleanDisjunction = new Monoid[Boolean] {
      override val empty: Boolean = false

      override def combine(x: Boolean, y: Boolean): Boolean = x || y
    }

    val monoidBooleanConjunction = new Monoid[Boolean] {
      override val empty: Boolean = true

      override def combine(x: Boolean, y: Boolean): Boolean = x && y
    }
  }

  /**
    * All set for monoids
    */
  object Section4 {

    object UnionSet {
      implicit def monoidSet[A] = new Monoid[Set[A]] {
        override def empty: Set[A] = Set.empty[A]

        override def combine(x: Set[A], y: Set[A]) = x union y
      }
    }

    object IntersectSet {
      implicit def semigroupSet[A] = new Semigroup[Set[A]] {
        override def combine(x: Set[A], y: Set[A]) = x intersect y
      }
    }
  }

}
