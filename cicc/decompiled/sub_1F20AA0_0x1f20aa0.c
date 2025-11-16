// Function: sub_1F20AA0
// Address: 0x1f20aa0
//
char *__fastcall sub_1F20AA0(char *src, char *a2, char *a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rax
  char *v6; // r15
  __int64 v8; // r14
  __int64 v9; // rbx
  char *v10; // rdi
  __int64 v13; // [rsp+8h] [rbp-38h]

  v5 = (a2 - src) >> 2;
  v6 = src;
  v13 = 2 * a4;
  if ( 2 * a4 <= v5 )
  {
    v8 = 8 * a4;
    v9 = -4 * a4;
    do
    {
      v10 = v6;
      v6 += v8;
      a3 = sub_1F209D0(v10, &v6[v9], &v6[v9], v6, a3, a5);
      v5 = (a2 - v6) >> 2;
    }
    while ( v5 >= v13 );
  }
  if ( v5 > a4 )
    v5 = a4;
  return sub_1F209D0(v6, &v6[4 * v5], &v6[4 * v5], a2, a3, a5);
}
