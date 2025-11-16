// Function: sub_25F5D80
// Address: 0x25f5d80
//
char *__fastcall sub_25F5D80(char *src, char *a2, char *a3, __int64 a4)
{
  __int64 v5; // rax
  char *v6; // r15
  __int64 v7; // r13
  __int64 v8; // r14
  __int64 v9; // rbx
  char *v10; // rdi

  v5 = (a2 - src) >> 2;
  v6 = src;
  v7 = 2 * a4;
  if ( 2 * a4 <= v5 )
  {
    v8 = 8 * a4;
    v9 = -4 * a4;
    do
    {
      v10 = v6;
      v6 += v8;
      a3 = sub_25F5C40(v10, &v6[v9], &v6[v9], v6, a3);
      v5 = (a2 - v6) >> 2;
    }
    while ( v7 <= v5 );
  }
  if ( a4 <= v5 )
    v5 = a4;
  return sub_25F5C40(v6, &v6[4 * v5], &v6[4 * v5], a2, a3);
}
