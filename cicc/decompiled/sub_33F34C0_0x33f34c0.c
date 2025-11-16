// Function: sub_33F34C0
// Address: 0x33f34c0
//
__int64 *__fastcall sub_33F34C0(
        __int64 *a1,
        int a2,
        __int64 a3,
        unsigned __int16 a4,
        __int64 a5,
        const __m128i *a6,
        __int128 a7,
        __int128 a8,
        __int64 a9,
        __int64 a10)
{
  __int64 v12; // r15
  unsigned __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // r10
  __int64 v16; // r11
  __m128i v17; // xmm0
  __m128i v18; // xmm1
  __int64 v20; // [rsp+8h] [rbp-78h]
  _OWORD v23[2]; // [rsp+20h] [rbp-60h] BYREF
  __int64 v24; // [rsp+40h] [rbp-40h]
  __int64 v25; // [rsp+48h] [rbp-38h]

  v12 = a10;
  v20 = a9;
  if ( a2 == 339 )
  {
    v13 = (unsigned __int64)sub_33ED250((__int64)a1, 1, 0);
    v16 = a3;
    v15 = v20;
  }
  else
  {
    v13 = sub_33E5110(
            a1,
            *(unsigned __int16 *)(*(_QWORD *)(a9 + 48) + 16LL * (unsigned int)a10),
            *(_QWORD *)(*(_QWORD *)(a9 + 48) + 16LL * (unsigned int)a10 + 8),
            1,
            0);
    v15 = v20;
    v16 = a3;
  }
  v17 = _mm_loadu_si128((const __m128i *)&a7);
  v18 = _mm_loadu_si128((const __m128i *)&a8);
  v25 = v12;
  v24 = v15;
  v23[0] = v17;
  v23[1] = v18;
  return sub_33E6BC0(a1, a2, v16, a4, a5, a6, v13, v14, (unsigned __int64 *)v23, 3);
}
