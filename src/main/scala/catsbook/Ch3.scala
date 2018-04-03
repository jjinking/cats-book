package catsbook

import cats.Functor

object Ch3 {

  object Section5 {

    sealed trait Tree[+A]
    final case class Branch[A](left: Tree[A], right: Tree[A]) extends Tree[A]
    final case class Leaf[A](value: A) extends Tree[A]

    object Tree {
      def branch[A](left: Tree[A], right: Tree[A]): Tree[A] = Branch(left, right)

      def leaf[A](value: A): Leaf[A] = Leaf(value)
    }

    implicit val treeFunctor: Functor[Tree] =
      new Functor[Tree] {
        def map[A, B](value: Tree[A])(f: A => B): Tree[B] = value match {
          case Branch(l, r) => Branch(map(l)(f), map(r)(f))
          case Leaf(x) => Leaf(f(x))
        }
      }
  }

  object Section6 {

    trait Printable[A] { self =>

      def format(value: A): String

      def contramap[B](func: B => A): Printable[B] =
        new Printable[B] {

          def format(value: B): String =
            self.format(func(value))
        }
    }

    def format[A](value: A)(implicit p: Printable[A]): String = p.format(value)

    // Instances
    implicit val stringPrintable: Printable[String] =
      new Printable[String] {
        def format(value: String): String =
          "\"" + value + "\""
      }

    implicit val booleanPrintable: Printable[Boolean] = new Printable[Boolean] {
      def format(value: Boolean): String =
        if(value) "yes" else "no"
    }

    // Custom instance
    final case class Box[A](value: A)

    implicit def boxPrintable[A](implicit printableA: Printable[A]): Printable[Box[A]] =
      printableA.contramap(_.value)

  }

  object Section6_2 {

    trait Codec[A] { self =>
      def encode(value: A): String
      def decode(value: String): A
      def imap[B](dec: A => B, enc: B => A): Codec[B] = new Codec[B] {
        override def encode(value: B) = self.encode(enc(value))
        override def decode(value: String) = dec(self.decode(value))
      }

    }
    def encode[A](value: A)(implicit c: Codec[A]): String = c.encode(value)
    def decode[A](value: String)(implicit c: Codec[A]): A = c.decode(value)

    implicit val stringCodec: Codec[String] =
      new Codec[String] {
        def encode(value: String): String = value
        def decode(value: String): String = value
      }

    implicit val intCodec: Codec[Int] =
      stringCodec.imap(_.toInt, _.toString)

    implicit val booleanCodec: Codec[Boolean] =
      stringCodec.imap(_.toBoolean, _.toString)

    implicit val doubleCodec: Codec[Double] =
      stringCodec.imap(_.toDouble, _.toString)

    case class Box[A](value: A)

    implicit def boxCodec[A](implicit c: Codec[A]): Codec[Box[A]] =
      c.imap(Box(_), _.value)
  }
}
