// Function: sub_3411830
// Address: 0x3411830
//
unsigned __int8 *__fastcall sub_3411830(
        __int64 *a1,
        unsigned int a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int128 a7,
        __int128 a8,
        unsigned int a9)
{
  unsigned __int8 *v10; // rax
  int v11; // edx
  unsigned int *v12; // rax
  __int64 v13; // rdx
  __int64 v14; // r9
  __int128 v16; // [rsp-18h] [rbp-A0h]
  _QWORD v18[2]; // [rsp+18h] [rbp-70h] BYREF
  __m128i v19; // [rsp+28h] [rbp-60h]
  __m128i v20; // [rsp+38h] [rbp-50h]
  unsigned __int8 *v21; // [rsp+48h] [rbp-40h]
  int v22; // [rsp+50h] [rbp-38h]

  v18[0] = a5;
  v18[1] = a6;
  v19 = _mm_loadu_si128((const __m128i *)&a7);
  v20 = _mm_loadu_si128((const __m128i *)&a8);
  v10 = sub_3400BD0((__int64)a1, a9, a4, 7, 0, 1u, v19, 0);
  v22 = v11;
  v21 = v10;
  v12 = (unsigned int *)sub_33E5110(a1, a2, a3, 1, 0);
  *((_QWORD *)&v16 + 1) = 4;
  *(_QWORD *)&v16 = v18;
  return sub_3411630(a1, 317, a4, v12, v13, v14, v16);
}
