// Function: sub_1CA75A0
// Address: 0x1ca75a0
//
__int64 __fastcall sub_1CA75A0(
        __int64 a1,
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
  __int64 v10; // rax
  __int64 v11; // r15
  __int64 v12; // rbx
  __int64 v13; // rax
  __int64 v14; // r14
  _QWORD *v15; // rsi
  double v16; // xmm4_8
  double v17; // xmm5_8
  __int64 v18; // rdi
  __int64 v20; // [rsp+8h] [rbp-68h]
  unsigned __int8 v21; // [rsp+17h] [rbp-59h]
  __int64 v22; // [rsp+18h] [rbp-58h]
  __int64 v23; // [rsp+20h] [rbp-50h] BYREF
  __int64 v24; // [rsp+28h] [rbp-48h]
  __int64 v25; // [rsp+30h] [rbp-40h]
  int v26; // [rsp+38h] [rbp-38h]

  v10 = *(_QWORD *)(a2 + 80);
  v20 = a2 + 72;
  v23 = 0;
  v24 = 0;
  v25 = 0;
  v26 = 0;
  v22 = v10;
  if ( v10 == a2 + 72 )
  {
    v21 = 0;
    v18 = 0;
  }
  else
  {
    v21 = 0;
    do
    {
      v11 = *(_QWORD *)(v22 + 24);
      v12 = v22 + 16;
      v22 = *(_QWORD *)(v22 + 8);
      while ( v12 != v11 )
      {
        v13 = v11;
        v11 = *(_QWORD *)(v11 + 8);
        if ( *(_BYTE *)(v13 - 8) == 72 )
        {
          v14 = v13 - 24;
          v15 = sub_1CA5350(a1, v13 - 24, (__int64)&v23);
          if ( v15 )
          {
            sub_164D160(v14, (__int64)v15, a3, a4, a5, a6, v16, v17, a9, a10);
            v21 = 1;
          }
        }
      }
    }
    while ( v20 != v22 );
    v18 = v24;
  }
  j___libc_free_0(v18);
  return v21;
}
