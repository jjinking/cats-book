package catsbook

object Ch6 {

  object Section6_3_1_1 {

    import cats.Monad

    def product[M[_]: Monad, A, B](x: M[A], y: M[B]): M[(A, B)] = for {
      a <- x
      b <- y
    } yield (a, b)

  }

}
