// Function: sub_2ED31C0
// Address: 0x2ed31c0
//
__int64 __fastcall sub_2ED31C0(__int64 *src, __int64 *a2, __int64 *a3, __int64 a4, __int64 *a5, _QWORD *a6)
{
  __int64 v6; // rax
  __int64 *v9; // r13
  __int64 v10; // rbx
  __int64 *v11; // rdi
  __int64 v13; // [rsp-10h] [rbp-60h]
  __int64 v15; // [rsp+10h] [rbp-40h]
  __int64 v16; // [rsp+18h] [rbp-38h]

  v6 = a2 - src;
  v9 = src;
  v16 = 2 * a4;
  if ( 2 * a4 <= v6 )
  {
    v15 = 16 * a4;
    v10 = -1 * a4;
    do
    {
      v11 = v9;
      v9 = (__int64 *)((char *)v9 + v15);
      a3 = (__int64 *)sub_2ED3020(v11, &v9[v10], &v9[v10], v9, a3, (__int64)a6, a5, a6);
      v6 = a2 - v9;
    }
    while ( v6 >= v16 );
  }
  if ( v6 > a4 )
    v6 = a4;
  sub_2ED3020(v9, &v9[v6], &v9[v6], a2, a3, (__int64)a6, a5, a6);
  return v13;
}
