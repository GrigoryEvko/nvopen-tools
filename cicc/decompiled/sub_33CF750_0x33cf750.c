// Function: sub_33CF750
// Address: 0x33cf750
//
__int64 __fastcall sub_33CF750(
        __m128i *a1,
        __int32 a2,
        __int32 a3,
        unsigned __int8 **a4,
        __int64 a5,
        __int32 a6,
        __int128 a7,
        __int64 a8)
{
  unsigned __int8 *v11; // rsi
  __int64 v12; // r14
  unsigned __int16 v13; // dx
  __m128i v14; // xmm0
  __int64 result; // rax
  __int64 v16; // [rsp+8h] [rbp-48h]
  _QWORD v17[7]; // [rsp+18h] [rbp-38h] BYREF

  v11 = *a4;
  v12 = a8;
  v17[0] = v11;
  if ( v11 )
  {
    v16 = a5;
    sub_B96E90((__int64)v17, (__int64)v11, 1);
    v11 = (unsigned __int8 *)v17[0];
    a5 = v16;
  }
  a1->m128i_i64[0] = 0;
  a1->m128i_i64[1] = 0;
  a1[1].m128i_i64[0] = 0;
  a1[1].m128i_i32[2] = a2;
  a1[1].m128i_i32[3] = 0;
  a1[2].m128i_i16[1] = -1;
  a1[2].m128i_i32[1] = -1;
  a1[2].m128i_i64[1] = 0;
  a1[3].m128i_i64[0] = a5;
  a1[3].m128i_i64[1] = 0;
  a1[4].m128i_i32[0] = 0;
  a1[4].m128i_i32[1] = a6;
  a1[4].m128i_i32[2] = a3;
  a1[5].m128i_i64[0] = (__int64)v11;
  if ( v11 )
    sub_B976B0((__int64)v17, v11, (__int64)a1[5].m128i_i64);
  a1[5].m128i_i64[1] = 0xFFFFFFFFLL;
  a1[2].m128i_i16[0] = 0;
  v13 = *(_WORD *)(v12 + 32);
  v14 = _mm_loadu_si128((const __m128i *)&a7);
  a1[7].m128i_i64[0] = v12;
  a1[6] = v14;
  result = a1[2].m128i_i8[0] & 0x87
         | (((v13 >> 5) & 1) << 6)
         | (32 * ((v13 >> 4) & 1))
         | (8 * ((v13 >> 2) & 1))
         | (16 * ((v13 >> 3) & 1u));
  a1[2].m128i_i8[0] = a1[2].m128i_i8[0] & 0x87
                    | (((v13 & 0x20) != 0) << 6)
                    | (32 * ((v13 & 0x10) != 0))
                    | (8 * ((v13 & 4) != 0))
                    | (16 * ((v13 & 8) != 0));
  return result;
}
