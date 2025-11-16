// Function: sub_11EE200
// Address: 0x11ee200
//
__int64 __fastcall sub_11EE200(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __int64 a9,
        __int128 a10,
        __int128 a11)
{
  __m128i v14; // xmm0
  __m128i v15; // xmm1

  sub_11EE1F0(a1, a3, 0);
  *(_QWORD *)(a1 + 16) = a2;
  v14 = _mm_loadu_si128((const __m128i *)&a10);
  v15 = _mm_loadu_si128((const __m128i *)&a11);
  *(_QWORD *)(a1 + 24) = a3;
  *(_QWORD *)(a1 + 56) = a7;
  *(_QWORD *)(a1 + 40) = a5;
  *(_QWORD *)(a1 + 64) = a8;
  *(_QWORD *)(a1 + 48) = a6;
  *(_QWORD *)(a1 + 32) = a4;
  *(_QWORD *)(a1 + 72) = a9;
  *(_BYTE *)(a1 + 80) = 0;
  *(__m128i *)(a1 + 88) = v14;
  *(__m128i *)(a1 + 104) = v15;
  return a9;
}
