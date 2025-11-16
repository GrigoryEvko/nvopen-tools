// Function: sub_386D460
// Address: 0x386d460
//
_QWORD *__fastcall sub_386D460(
        __int64 *a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  _QWORD *result; // rax
  double v11; // xmm4_8
  double v12; // xmm5_8
  __int64 v13; // rsi
  __int64 v14; // r12
  _QWORD *v15; // rbx
  _QWORD *v16; // r13
  __int64 v17; // rax
  __int64 v18; // [rsp+0h] [rbp-40h] BYREF
  _QWORD *v19; // [rsp+8h] [rbp-38h]
  __int64 v20; // [rsp+10h] [rbp-30h]
  unsigned int v21; // [rsp+18h] [rbp-28h]

  result = sub_386B3C0(a1, a2);
  if ( !result )
  {
    v13 = *(_QWORD *)(a2 + 64);
    v18 = 0;
    v19 = 0;
    v20 = 0;
    v21 = 0;
    v14 = sub_386C880((__int64)a1, v13, (__int64)&v18, a3, a4, a5, a6, v11, v12, a9, a10);
    if ( v21 )
    {
      v15 = v19;
      v16 = &v19[4 * v21];
      do
      {
        if ( *v15 != -8 && *v15 != -16 )
        {
          v17 = v15[3];
          if ( v17 != -8 && v17 != 0 && v17 != -16 )
            sub_1649B30(v15 + 1);
        }
        v15 += 4;
      }
      while ( v16 != v15 );
    }
    j___libc_free_0((unsigned __int64)v19);
    return (_QWORD *)v14;
  }
  return result;
}
