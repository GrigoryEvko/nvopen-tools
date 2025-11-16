// Function: sub_21DA8E0
// Address: 0x21da8e0
//
__int64 __fastcall sub_21DA8E0(__int64 a1, __int64 *a2, __int64 a3, __int32 a4)
{
  _QWORD *v4; // r13
  __m128i v6; // [rsp+0h] [rbp-50h] BYREF
  __int64 v7; // [rsp+10h] [rbp-40h]
  __int64 v8; // [rsp+18h] [rbp-38h]
  __int64 v9; // [rsp+20h] [rbp-30h]

  v6.m128i_i32[2] = a4;
  v4 = sub_1E0B640(a1, a3, a2, 0);
  v6.m128i_i64[0] = 0x10000000;
  v7 = 0;
  v8 = 0;
  v9 = 0;
  sub_1E1A9C0((__int64)v4, a1, &v6);
  return a1;
}
