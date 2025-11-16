// Function: sub_1D2B8F0
// Address: 0x1d2b8f0
//
__int64 *__fastcall sub_1D2B8F0(
        _QWORD *a1,
        int a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int128 a7,
        __int128 a8,
        __int64 a9,
        __int64 a10)
{
  __int64 v12; // r15
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // r10
  __int64 v16; // r11
  __m128i v17; // xmm0
  __m128i v18; // xmm1
  __int64 v20; // [rsp+8h] [rbp-78h]
  unsigned __int8 v22; // [rsp+18h] [rbp-68h]
  _OWORD v23[2]; // [rsp+20h] [rbp-60h] BYREF
  __int64 v24; // [rsp+40h] [rbp-40h]
  __int64 v25; // [rsp+48h] [rbp-38h]

  v12 = a10;
  v22 = a4;
  v20 = a9;
  if ( a2 == 220 )
  {
    v13 = sub_1D29190((__int64)a1, 1u, 0, a4, a5, a6);
    v16 = a3;
    v15 = v20;
  }
  else
  {
    v13 = sub_1D252B0(
            (__int64)a1,
            *(unsigned __int8 *)(*(_QWORD *)(a9 + 40) + 16LL * (unsigned int)a10),
            *(_QWORD *)(*(_QWORD *)(a9 + 40) + 16LL * (unsigned int)a10 + 8),
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
  return sub_1D24450(a1, a2, v16, v22, a5, a6, v13, v14, (__int64 *)v23, 3);
}
