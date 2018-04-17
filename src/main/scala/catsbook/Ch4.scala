package catsbook

import cats.data.Reader

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

  object Section4_8_3 {
    import cats.syntax.applicative._ // for pure

    case class Db(
      usernames: Map[Int, String],
      passwords: Map[String, String]
    )

    type DbReader[A] = Reader[Db, A]

    def findUsername(userId: Int): DbReader[Option[String]] = Reader { (db: Db) =>
      db.usernames.get(userId)
    }

    def checkPassword(
      username: String,
      password: String
    ): DbReader[Boolean] = Reader { db: Db =>
      db.passwords.get(username) contains password
    }

    def checkLogin(
      userId: Int,
      password: String
    ): DbReader[Boolean] = for {
      usernameOpt <- findUsername(userId)
      pwCorrect <- usernameOpt map { username => checkPassword(username, password) } getOrElse(false.pure[DbReader])
    } yield pwCorrect

  }

  object Section4_9_3 {
    import cats.data.State

    type CalcState[A] = State[List[Int], A]

    def evalOne(sym: String): CalcState[Int] = State[List[Int], Int] { oldStack:List[Int] =>
      try {
        val symInt = sym.toInt
        (symInt :: oldStack, symInt)
      } catch {
        case _: Exception => {
          val arg2::t = oldStack
          val arg1::rest = t
          sym match {
            case "+" => {
              val result = arg1 + arg2
              (result :: rest, result)
            }
            case "*" => {
              val result = arg1 * arg2
              (result :: rest, result)
            }
          }
        }
      }
    }

    def evalAll(input: List[String]): CalcState[Int] =
      input.foldLeft(State.pure[List[Int], Int](0)) {
        (acc, sym) => acc.flatMap(_ => evalOne(sym))
      }

    def evalInput(input: String): Int =
      evalAll(input.split("\\s+").toList).runA(Nil).value
  }

  object `4.10.1` {
    import cats.Monad
    import scala.annotation.tailrec

    sealed trait Tree[+A]

    final case class Branch[A](left: Tree[A], right: Tree[A])
        extends Tree[A]

    final case class Leaf[A](value: A) extends Tree[A]

    def branch[A](left: Tree[A], right: Tree[A]): Tree[A] =
      Branch(left, right)

    def leaf[A](value: A): Tree[A] =
      Leaf(value)


    // val treeMonad = new Monad[Tree] {

    //   def pure[A](a: A): Tree[A] = leaf(a)

    //   def flatMap[A, B](t: Tree[A])(f: A => Tree[B]): Tree[B] = t match {
    //     case Branch(l, r) => Branch(flatMap(l)(f), flatMap(r)(f))
    //     case Leaf(a) => f(a)
    //   }

    //   // @tailrec
    //   // def tailRecM[A, B](a: A)(f: A => Tree[Either[A, B]]): Tree[B] = f(a) match {
    //   //   Branch(Tree[Left(l)], Tree[RightA]) => 
    //   //}

    // }
  }



}
