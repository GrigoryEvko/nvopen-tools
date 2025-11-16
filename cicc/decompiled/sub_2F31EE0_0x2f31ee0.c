// Function: sub_2F31EE0
// Address: 0x2f31ee0
//
__int64 __fastcall sub_2F31EE0(
        __int64 a1,
        __int64 a2,
        __int64 *a3,
        unsigned __int8 **a4,
        __int32 a5,
        __int16 a6,
        __int32 a7)
{
  unsigned __int8 *v11; // rsi
  __int64 v12; // rcx
  _QWORD *v13; // rax
  __int64 v14; // rdx
  __int64 v15; // r12
  __int64 v17; // [rsp+0h] [rbp-A0h]
  unsigned __int8 *v18; // [rsp+18h] [rbp-88h] BYREF
  __int64 v19[4]; // [rsp+20h] [rbp-80h] BYREF
  __m128i v20; // [rsp+40h] [rbp-60h] BYREF
  __int64 v21; // [rsp+50h] [rbp-50h]
  __int64 v22; // [rsp+58h] [rbp-48h]
  __int64 v23; // [rsp+60h] [rbp-40h]

  v11 = *a4;
  v12 = *(_QWORD *)(a1 + 8) - 800LL;
  v18 = v11;
  if ( v11 )
  {
    v17 = v12;
    sub_B96E90((__int64)&v18, (__int64)v11, 1);
    v12 = v17;
    v19[0] = (__int64)v18;
    if ( v18 )
    {
      sub_B976B0((__int64)&v18, v18, (__int64)v19);
      v12 = v17;
      v18 = 0;
    }
  }
  else
  {
    v19[0] = 0;
  }
  v19[1] = 0;
  v19[2] = 0;
  v13 = sub_2F26260(a2, a3, v19, v12, a7);
  v15 = v14;
  v20.m128i_i32[2] = a5;
  v21 = 0;
  v22 = 0;
  v23 = 0;
  v20.m128i_i64[0] = (unsigned __int16)(a6 & 0xFFF) << 8;
  sub_2E8EAD0(v14, (__int64)v13, &v20);
  if ( v19[0] )
    sub_B91220((__int64)v19, v19[0]);
  if ( v18 )
    sub_B91220((__int64)&v18, (__int64)v18);
  return v15;
}
