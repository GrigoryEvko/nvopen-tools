// Function: sub_C2ECE0
// Address: 0xc2ece0
//
__int64 __fastcall sub_C2ECE0(
        __m128i *a1,
        __int64 *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        _BYTE *a7,
        __int64 a8,
        char a9)
{
  __int64 result; // rax
  __m128i *v10; // rdx
  __int64 v11[2]; // [rsp+0h] [rbp-30h] BYREF
  __m128i v12[2]; // [rsp+10h] [rbp-20h] BYREF

  a1[1].m128i_i8[0] = 0;
  result = *a2;
  a1[1].m128i_i64[1] = *a2;
  *a2 = 0;
  if ( !a9 )
  {
    a1[4].m128i_i8[0] = 0;
    return result;
  }
  if ( !a7 )
  {
    v12[0].m128i_i8[0] = 0;
    a1[2].m128i_i64[0] = (__int64)a1[3].m128i_i64;
    result = 0;
    goto LABEL_8;
  }
  v11[0] = (__int64)v12;
  sub_C2EC30(v11, a7, (__int64)&a7[a8]);
  v10 = (__m128i *)v11[0];
  result = v11[1];
  a1[2].m128i_i64[0] = (__int64)a1[3].m128i_i64;
  if ( v10 == v12 )
  {
LABEL_8:
    a1[3] = _mm_load_si128(v12);
    goto LABEL_6;
  }
  a1[2].m128i_i64[0] = (__int64)v10;
  a1[3].m128i_i64[0] = v12[0].m128i_i64[0];
LABEL_6:
  a1[2].m128i_i64[1] = result;
  a1[4].m128i_i8[0] = 1;
  return result;
}
