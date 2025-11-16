// Function: sub_F071A0
// Address: 0xf071a0
//
char *__fastcall sub_F071A0(__int64 *src, __int64 *a2, char *a3, __int64 a4)
{
  __int64 v5; // rax
  __int64 *v6; // r15
  __int64 v7; // r13
  __int64 v8; // r14
  __int64 v9; // rbx
  __int64 *v10; // rdi

  v5 = a2 - src;
  v6 = src;
  v7 = 2 * a4;
  if ( 2 * a4 <= v5 )
  {
    v8 = 16 * a4;
    v9 = -1 * a4;
    do
    {
      v10 = v6;
      v6 = (__int64 *)((char *)v6 + v8);
      a3 = sub_F070B0(v10, &v6[v9], &v6[v9], v6, a3);
      v5 = a2 - v6;
    }
    while ( v5 >= v7 );
  }
  if ( v5 > a4 )
    v5 = a4;
  return sub_F070B0(v6, &v6[v5], &v6[v5], a2, a3);
}
