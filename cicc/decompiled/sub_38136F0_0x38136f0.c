// Function: sub_38136F0
// Address: 0x38136f0
//
unsigned __int8 *__fastcall sub_38136F0(
        __int64 a1,
        __int64 a2,
        __m128i *a3,
        char a4,
        __int64 a5,
        __int64 a6,
        __m128i a7,
        __int64 a8,
        _QWORD *a9)
{
  unsigned __int8 *result; // rax
  __m128i v11; // xmm0
  __int128 v12; // [rsp-10h] [rbp-70h]
  _QWORD v13[2]; // [rsp+10h] [rbp-50h] BYREF
  __int16 v14; // [rsp+20h] [rbp-40h]
  __int64 v15; // [rsp+28h] [rbp-38h]
  __m128i v16; // [rsp+30h] [rbp-30h] BYREF
  __int64 v17; // [rsp+40h] [rbp-20h]
  __int64 v18; // [rsp+48h] [rbp-18h]

  if ( !a4 )
    return sub_33FAF80((__int64)a9, 233, a8, (unsigned int)a5, a6, a6, a7);
  v17 = a1;
  v11 = _mm_loadu_si128(a3);
  v18 = a2;
  *((_QWORD *)&v12 + 1) = 2;
  *(_QWORD *)&v12 = &v16;
  v13[0] = a5;
  v13[1] = a6;
  v14 = 1;
  v15 = 0;
  v16 = v11;
  result = sub_3411BE0(a9, 0x92u, a8, (unsigned __int16 *)v13, 2, a6, v12);
  a3->m128i_i32[2] = 1;
  a3->m128i_i64[0] = (__int64)result;
  return result;
}
