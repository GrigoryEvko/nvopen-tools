// Function: sub_28E9830
// Address: 0x28e9830
//
char *__fastcall sub_28E9830(char *src, char *a2, char *a3, __int64 a4)
{
  __int64 v5; // rax
  __int64 v6; // r14
  char *v8; // rbx
  __int64 v9; // r15
  __int64 v10; // r13
  char *v11; // rdi
  __int64 v12; // rsi

  v5 = (a2 - src) >> 4;
  v6 = 2 * a4;
  v8 = src;
  if ( 2 * a4 <= v5 )
  {
    v9 = 32 * a4;
    v10 = -16 * a4;
    do
    {
      v11 = v8;
      v8 += v9;
      a3 = sub_28E9090(v11, &v8[v10], &v8[v10], v8, a3);
      v5 = (a2 - v8) >> 4;
    }
    while ( v6 <= v5 );
  }
  v12 = a4;
  if ( a4 > v5 )
    v12 = v5;
  return sub_28E9090(v8, &v8[16 * v12], &v8[16 * v12], a2, a3);
}
