// Function: sub_1E1C280
// Address: 0x1e1c280
//
__int64 __fastcall sub_1E1C280(__int64 a1, __int64 *a2, __int64 a3, char a4, const __m128i *a5, __int64 a6, __int64 a7)
{
  _QWORD *v10; // r14
  __m128i v12; // [rsp+10h] [rbp-60h] BYREF
  __int64 v13; // [rsp+20h] [rbp-50h]
  __int64 v14; // [rsp+28h] [rbp-48h]
  __int64 v15; // [rsp+30h] [rbp-40h]

  if ( !a5->m128i_i8[0] )
    return sub_1E1C0A0(a1, a2, a3, a4, a5->m128i_i32[2], a6, a7);
  v10 = sub_1E0B640(a1, a3, a2, 0);
  sub_1E1A9C0((__int64)v10, a1, a5);
  if ( a4 )
  {
    v12.m128i_i64[0] = 1;
    v13 = 0;
    v14 = 0;
  }
  else
  {
    v12 = (__m128i)0x800000000uLL;
    v13 = 0;
    v14 = 0;
    v15 = 0;
  }
  sub_1E1A9C0((__int64)v10, a1, &v12);
  v12.m128i_i64[0] = 14;
  v14 = a6;
  v13 = 0;
  sub_1E1A9C0((__int64)v10, a1, &v12);
  v14 = a7;
  v12.m128i_i64[0] = 14;
  v13 = 0;
  sub_1E1A9C0((__int64)v10, a1, &v12);
  return a1;
}
