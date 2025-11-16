// Function: sub_1D38F20
// Address: 0x1d38f20
//
__int64 *__fastcall sub_1D38F20(
        __int64 *a1,
        unsigned int a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        double a7,
        double a8,
        __m128i a9,
        __int128 a10,
        __int128 a11,
        unsigned int a12)
{
  __int64 v13; // rax
  int v14; // edx
  const void ***v15; // rax
  int v16; // edx
  __int64 v17; // r9
  __int128 v19; // [rsp-18h] [rbp-A0h]
  _QWORD v21[2]; // [rsp+18h] [rbp-70h] BYREF
  __m128i v22; // [rsp+28h] [rbp-60h]
  __m128i v23; // [rsp+38h] [rbp-50h]
  __int64 v24; // [rsp+48h] [rbp-40h]
  int v25; // [rsp+50h] [rbp-38h]

  v21[0] = a5;
  v21[1] = a6;
  v22 = _mm_loadu_si128((const __m128i *)&a10);
  v23 = _mm_loadu_si128((const __m128i *)&a11);
  v13 = sub_1D38BB0((__int64)a1, a12, a4, 5, 0, 1, v22, *(double *)v23.m128i_i64, a9, 0);
  v25 = v14;
  v24 = v13;
  v15 = (const void ***)sub_1D252B0((__int64)a1, a2, a3, 1, 0);
  *((_QWORD *)&v19 + 1) = 4;
  *(_QWORD *)&v19 = v21;
  return sub_1D36D80(a1, 204, a4, v15, v16, *(double *)v22.m128i_i64, *(double *)v23.m128i_i64, a9, v17, v19);
}
