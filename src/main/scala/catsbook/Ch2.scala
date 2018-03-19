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

  /**
    * Adding all the things - SuperAdder
    */
  object Section5 {
    import cats.instances.int._
    import cats.syntax.semigroup._

    def add(items: List[Int]): Int = {
      items.foldLeft(Monoid[Int].empty)(_ |+| _)
    }

  }

  object Section5Option {
    import cats.syntax.semigroup._

    def add[A: Monoid](items: List[A]): A = {
      items.foldLeft(Monoid[A].empty)(_ |+| _)
    }

    case class Order(totalCost: Double, quantity: Double)

    implicit val orderMonoid = new Monoid[Order] {
      override val empty: Order = Order(0, 0)

      override def combine(x: Order, y: Order) =
        Order(x.totalCost + y.totalCost, x.quantity + y.quantity)
    }
  }


}
