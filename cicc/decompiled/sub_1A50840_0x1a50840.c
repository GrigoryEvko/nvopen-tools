// Function: sub_1A50840
// Address: 0x1a50840
//
char *__fastcall sub_1A50840(char *src, char *a2, char *a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rax
  char *v6; // r15
  __int64 v8; // r14
  __int64 v9; // rbx
  char *v10; // rdi
  __int64 v13; // [rsp+8h] [rbp-38h]

  v5 = (a2 - src) >> 3;
  v6 = src;
  v13 = 2 * a4;
  if ( 2 * a4 <= v5 )
  {
    v8 = 16 * a4;
    v9 = -8 * a4;
    do
    {
      v10 = v6;
      v6 += v8;
      a3 = sub_1A50660(v10, &v6[v9], &v6[v9], v6, a3, a5);
      v5 = (a2 - v6) >> 3;
    }
    while ( v5 >= v13 );
  }
  if ( v5 > a4 )
    v5 = a4;
  return sub_1A50660(v6, &v6[8 * v5], &v6[8 * v5], a2, a3, a5);
}
