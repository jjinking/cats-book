package catsbook

object Ch4 {

  object Section4_1_2 {
    import scala.language.higherKinds

    trait Monad[F[_]] {
      def pure[A](a: A): F[A]
      def flatMap[A, B](value: F[A])(func: A => F[B]): F[B]
      def map[A, B](value: F[A])(func: A => B): F[B] = flatMap(value)((a: A) => pure(func(a)))
    }
  }

  object Section4_3_1 {
    type Id[A] = A

    def pure[A](a: A): Id[A] = a
    def map[A, B](idA: Id[A])(f: A => B): Id[B] = f(idA)
    def flatMap[A, B](idA: Id[A])(f: A => Id[B]): Id[B] = map(idA)(f)
  }

  object Section4_6_5 {
    import cats.Eval

    def foldRight[A, B](as: List[A], acc: Eval[B])(fn: (A, Eval[B]) => Eval[B]): Eval[B] =
      as match {
        case head :: tail =>
          Eval.defer(fn(head, foldRight(tail, acc)(fn)))
        case Nil =>
          acc
      }
  }

  object Section4_7_3 {
    import cats.data.Writer

    def factorialWithWriter(n: Int): Writer[Vector[String], Int] = {
      if (n == 0) Writer(Vector("fact 0 1"), 1)
      else factorialWithWriter(n - 1) mapBoth { (log, res) =>
        val newRes = res * n
        (log :+ s"fact $n $newRes", newRes)
      }
    }
  }

}
