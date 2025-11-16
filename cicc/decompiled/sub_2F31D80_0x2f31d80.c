// Function: sub_2F31D80
// Address: 0x2f31d80
//
__int64 __fastcall sub_2F31D80(__int64 a1, __int64 a2, __int64 *a3, unsigned __int8 **a4, __int32 a5, __int32 a6)
{
  __int32 v9; // r8d
  unsigned __int8 *v10; // rsi
  __int64 v11; // r12
  _QWORD *v12; // rax
  __int64 v13; // rdx
  __int64 v14; // r12
  unsigned __int8 *v17; // [rsp+18h] [rbp-88h] BYREF
  __int64 v18[4]; // [rsp+20h] [rbp-80h] BYREF
  __m128i v19; // [rsp+40h] [rbp-60h] BYREF
  __int64 v20; // [rsp+50h] [rbp-50h]
  __int64 v21; // [rsp+58h] [rbp-48h]
  __int64 v22; // [rsp+60h] [rbp-40h]

  v9 = a6;
  v10 = *a4;
  v11 = *(_QWORD *)(a1 + 8) - 800LL;
  v17 = v10;
  if ( v10 )
  {
    sub_B96E90((__int64)&v17, (__int64)v10, 1);
    v9 = a6;
    v18[0] = (__int64)v17;
    if ( v17 )
    {
      sub_B976B0((__int64)&v17, v17, (__int64)v18);
      v9 = a6;
      v17 = 0;
    }
  }
  else
  {
    v18[0] = 0;
  }
  v18[1] = 0;
  v18[2] = 0;
  v12 = sub_2F26260(a2, a3, v18, v11, v9);
  v19.m128i_i64[0] = 0;
  v14 = v13;
  v19.m128i_i32[2] = a5;
  v20 = 0;
  v21 = 0;
  v22 = 0;
  sub_2E8EAD0(v13, (__int64)v12, &v19);
  if ( v18[0] )
    sub_B91220((__int64)v18, v18[0]);
  if ( v17 )
    sub_B91220((__int64)&v17, (__int64)v17);
  return v14;
}
