// Function: sub_2207DF0
// Address: 0x2207df0
//
size_t __fastcall sub_2207DF0(FILE **a1, char *a2, size_t a3)
{
  size_t v3; // r13
  size_t v5; // rbx
  int v6; // r12d
  ssize_t v7; // rax

  v3 = a3;
  v5 = a3;
  v6 = sub_2207D30(a1);
  do
  {
    while ( 1 )
    {
      v7 = write(v6, a2, v5);
      if ( v7 == -1 )
        break;
      v5 -= v7;
      if ( !v5 )
        return v3;
      a2 += v7;
    }
  }
  while ( *__errno_location() == 4 );
  v3 -= v5;
  return v3;
}
