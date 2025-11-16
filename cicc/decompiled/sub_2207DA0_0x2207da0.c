// Function: sub_2207DA0
// Address: 0x2207da0
//
ssize_t __fastcall sub_2207DA0(FILE **a1, void *a2, size_t a3)
{
  int v4; // eax
  ssize_t v5; // r12

  do
  {
    v4 = sub_2207D30(a1);
    v5 = read(v4, a2, a3);
  }
  while ( v5 == -1 && *__errno_location() == 4 );
  return v5;
}
