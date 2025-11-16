// Function: sub_2A00470
// Address: 0x2a00470
//
__int64 __fastcall sub_2A00470(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        const __m128i *a6,
        __int64 a7,
        __int64 a8,
        __int64 a9,
        __int128 a10,
        __int128 a11)
{
  __int64 v14; // rax
  __m128i v15; // xmm6
  __m128i v16; // xmm7
  __m128i v17; // xmm0
  __m128i v18; // xmm1
  __m128i v19; // xmm2
  __m128i v20; // xmm3
  __m128i v21; // xmm4
  __m128i v22; // xmm5

  *(_QWORD *)a1 = *(_QWORD *)(**(_QWORD **)(a2 + 32) + 72LL);
  v14 = sub_AA48A0(**(_QWORD **)(a2 + 32));
  v15 = _mm_loadu_si128((const __m128i *)&a10);
  *(_QWORD *)(a1 + 40) = a4;
  v16 = _mm_loadu_si128((const __m128i *)&a11);
  *(_QWORD *)(a1 + 8) = v14;
  v17 = _mm_loadu_si128(a6);
  *(_QWORD *)(a1 + 48) = a5;
  v18 = _mm_loadu_si128(a6 + 1);
  v19 = _mm_loadu_si128(a6 + 2);
  *(_QWORD *)(a1 + 56) = a2;
  *(_QWORD *)(a1 + 16) = a7;
  v20 = _mm_loadu_si128(a6 + 3);
  v21 = _mm_loadu_si128(a6 + 4);
  *(_QWORD *)(a1 + 32) = a3;
  *(_QWORD *)(a1 + 24) = a8;
  v22 = _mm_loadu_si128(a6 + 5);
  *(_QWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 80) = a9;
  *(__m128i *)(a1 + 88) = v17;
  *(__m128i *)(a1 + 104) = v18;
  *(__m128i *)(a1 + 120) = v19;
  *(__m128i *)(a1 + 136) = v20;
  *(__m128i *)(a1 + 152) = v21;
  *(__m128i *)(a1 + 168) = v22;
  *(__m128i *)(a1 + 184) = v15;
  *(__m128i *)(a1 + 200) = v16;
  return a9;
}
