// Function: sub_318B520
// Address: 0x318b520
//
__int64 __fastcall sub_318B520(__int64 a1)
{
  __int64 v1; // rax
  __int64 result; // rax
  __m128i v3; // xmm1
  __m128i v4; // xmm3
  _BYTE v5[8]; // [rsp+0h] [rbp-90h] BYREF
  __int64 v6; // [rsp+8h] [rbp-88h]
  _OWORD v7[2]; // [rsp+20h] [rbp-70h] BYREF
  __m128i v8; // [rsp+40h] [rbp-50h] BYREF
  __m128i v9; // [rsp+50h] [rbp-40h] BYREF
  __m128i v10; // [rsp+60h] [rbp-30h] BYREF
  __m128i v11[2]; // [rsp+70h] [rbp-20h] BYREF

  sub_318B480((__int64)v5, a1);
  v1 = sub_318B4F0(a1);
  sub_371B570(&v10, v1);
  result = 0;
  if ( v6 != v10.m128i_i64[1] )
  {
    sub_318B480((__int64)&v8, a1);
    v3 = _mm_loadu_si128(&v9);
    v10 = _mm_loadu_si128(&v8);
    v11[0] = v3;
    sub_371B3D0(&v10);
    v4 = _mm_loadu_si128(v11);
    v7[0] = _mm_loadu_si128(&v10);
    v7[1] = v4;
    return sub_371B3B0(v7, *((_QWORD *)&v7[0] + 1), v11[0].m128i_i64[0]);
  }
  return result;
}
