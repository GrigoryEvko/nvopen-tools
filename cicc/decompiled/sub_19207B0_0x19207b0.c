// Function: sub_19207B0
// Address: 0x19207b0
//
char *__fastcall sub_19207B0(char *a1, char *a2, char *a3, __int64 a4)
{
  __int64 v5; // rax
  char *v6; // r13
  __int64 v7; // rbx
  __int64 v8; // r15
  __int64 v9; // r14
  char *v10; // rdi

  v5 = 0xAAAAAAAAAAAAAAABLL * ((a2 - a1) >> 3);
  v6 = a1;
  v7 = 2 * a4;
  if ( 2 * a4 <= v5 )
  {
    v8 = 48 * a4;
    v9 = 24 * a4;
    do
    {
      v10 = v6;
      v6 += v8;
      a3 = sub_1920660(v10, &v6[v9 - v8], &v6[v9 - v8], v6, a3);
      v5 = 0xAAAAAAAAAAAAAAABLL * ((a2 - v6) >> 3);
    }
    while ( v7 <= v5 );
  }
  if ( a4 <= v5 )
    v5 = a4;
  return sub_1920660(v6, &v6[24 * v5], &v6[24 * v5], a2, a3);
}
