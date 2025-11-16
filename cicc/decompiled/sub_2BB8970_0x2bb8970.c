// Function: sub_2BB8970
// Address: 0x2bb8970
//
__int64 __fastcall sub_2BB8970(unsigned __int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rax
  unsigned __int64 *v6; // r15
  __int64 v9; // r14
  __int64 v10; // rbx
  unsigned __int64 *v11; // rdi
  __int64 v12; // rsi
  __int64 v15; // [rsp+8h] [rbp-38h]

  v5 = (a2 - (__int64)a1) >> 6;
  v6 = a1;
  v15 = 2 * a4;
  if ( 2 * a4 <= v5 )
  {
    v9 = a4 << 7;
    v10 = -8 * a4;
    do
    {
      v11 = v6;
      v6 = (unsigned __int64 *)((char *)v6 + v9);
      a3 = sub_2BB8850(v11, &v6[v10], (__int64)&v6[v10], (__int64)v6, a3, a5);
      v5 = (a2 - (__int64)v6) >> 6;
    }
    while ( v5 >= v15 );
  }
  v12 = a4;
  if ( v5 <= a4 )
    v12 = v5;
  return sub_2BB8850(v6, &v6[8 * v12], (__int64)&v6[8 * v12], a2, a3, a5);
}
