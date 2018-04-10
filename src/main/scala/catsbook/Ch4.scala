package catsbook

object Ch4 {

  object Section4 {
    import scala.language.higherKinds

    trait Monad[F[_]] {
      def pure[A](a: A): F[A]
      def flatMap[A, B](value: F[A])(func: A => F[B]): F[B]
      def map[A, B](value: F[A])(func: A => B): F[B] = flatmap(value)(a: A => pure(func(a)))
    }
  }

}
