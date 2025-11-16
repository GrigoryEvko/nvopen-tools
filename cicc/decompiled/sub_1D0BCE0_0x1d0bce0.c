// Function: sub_1D0BCE0
// Address: 0x1d0bce0
//
char *__fastcall sub_1D0BCE0(char *src, char *a2, char *a3, __int64 a4)
{
  __int64 v5; // rax
  char *v6; // r15
  __int64 v7; // r13
  __int64 v8; // r14
  __int64 v9; // rbx
  char *v10; // rdi

  v5 = (a2 - src) >> 3;
  v6 = src;
  v7 = 2 * a4;
  if ( 2 * a4 <= v5 )
  {
    v8 = 16 * a4;
    v9 = -8 * a4;
    do
    {
      v10 = v6;
      v6 += v8;
      a3 = sub_1D0B6D0(v10, &v6[v9], &v6[v9], v6, a3);
      v5 = (a2 - v6) >> 3;
    }
    while ( v7 <= v5 );
  }
  if ( a4 <= v5 )
    v5 = a4;
  return sub_1D0B6D0(v6, &v6[8 * v5], &v6[8 * v5], a2, a3);
}
