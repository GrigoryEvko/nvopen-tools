// Function: sub_F7AF60
// Address: 0xf7af60
//
__int64 __fastcall sub_F7AF60(__int64 *a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rax
  __int64 *v6; // r15
  __int64 v9; // r14
  __int64 v10; // rbx
  __int64 *v11; // rdi
  __int64 v12; // rsi
  __int64 v15; // [rsp+8h] [rbp-38h]

  v5 = ((char *)a2 - (char *)a1) >> 4;
  v6 = a1;
  v15 = 2 * a4;
  if ( 2 * a4 <= v5 )
  {
    v9 = 32 * a4;
    v10 = -2 * a4;
    do
    {
      v11 = v6;
      v6 = (__int64 *)((char *)v6 + v9);
      a3 = sub_F7AD70(v11, &v6[v10], &v6[v10], v6, a3, a5);
      v5 = ((char *)a2 - (char *)v6) >> 4;
    }
    while ( v5 >= v15 );
  }
  v12 = a4;
  if ( v5 <= a4 )
    v12 = v5;
  return sub_F7AD70(v6, &v6[2 * v12], &v6[2 * v12], a2, a3, a5);
}
