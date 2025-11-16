// Function: sub_284B890
// Address: 0x284b890
//
char *__fastcall sub_284B890(__int64 *src, __int64 *a2, char *a3, __int64 a4, __int64 *a5)
{
  __int64 v5; // rax
  __int64 *v6; // r15
  __int64 v8; // r14
  __int64 v9; // rbx
  __int64 *v10; // rdi
  __int64 v13; // [rsp+8h] [rbp-38h]

  v5 = a2 - src;
  v6 = src;
  v13 = 2 * a4;
  if ( 2 * a4 <= v5 )
  {
    v8 = 16 * a4;
    v9 = -1 * a4;
    do
    {
      v10 = v6;
      v6 = (__int64 *)((char *)v6 + v8);
      a3 = sub_284B7A0(v10, &v6[v9], &v6[v9], v6, a3, a5);
      v5 = a2 - v6;
    }
    while ( v5 >= v13 );
  }
  if ( v5 > a4 )
    v5 = a4;
  return sub_284B7A0(v6, &v6[v5], &v6[v5], a2, a3, a5);
}
