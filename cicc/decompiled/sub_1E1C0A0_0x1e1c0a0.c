// Function: sub_1E1C0A0
// Address: 0x1e1c0a0
//
__int64 __fastcall sub_1E1C0A0(__int64 a1, __int64 *a2, __int64 a3, char a4, __int32 a5, __int64 a6, __int64 a7)
{
  _QWORD *v9; // r13
  __m128i v11; // [rsp+0h] [rbp-60h] BYREF
  __int64 v12; // [rsp+10h] [rbp-50h]
  __int64 v13; // [rsp+18h] [rbp-48h]
  __int64 v14; // [rsp+20h] [rbp-40h]

  v11.m128i_i32[2] = a5;
  v9 = sub_1E0B640(a1, a3, a2, 0);
  v12 = 0;
  v13 = 0;
  v14 = 0;
  v11.m128i_i64[0] = 0x800000000LL;
  sub_1E1A9C0((__int64)v9, a1, &v11);
  if ( a4 )
  {
    v11.m128i_i64[0] = 1;
    v12 = 0;
    v13 = 0;
  }
  else
  {
    v11 = (__m128i)0x800000000uLL;
    v12 = 0;
    v13 = 0;
    v14 = 0;
  }
  sub_1E1A9C0((__int64)v9, a1, &v11);
  v13 = a6;
  v11.m128i_i64[0] = 14;
  v12 = 0;
  sub_1E1A9C0((__int64)v9, a1, &v11);
  v11.m128i_i64[0] = 14;
  v13 = a7;
  v12 = 0;
  sub_1E1A9C0((__int64)v9, a1, &v11);
  return a1;
}
