// Function: sub_1E342C0
// Address: 0x1e342c0
//
__int64 __fastcall sub_1E342C0(
        __int64 a1,
        __int16 a2,
        __int64 a3,
        unsigned int a4,
        const __m128i *a5,
        __int64 a6,
        __int128 a7,
        __int64 a8,
        char a9,
        char a10,
        unsigned __int8 a11)
{
  __m128i v11; // xmm0
  char v12; // dl
  __int16 v13; // ax
  __int64 v14; // rax
  char v15; // dl
  __m128i v16; // xmm1
  __int64 result; // rax

  *(_QWORD *)(a1 + 24) = a3;
  *(_WORD *)(a1 + 32) = a2;
  v11 = _mm_loadu_si128((const __m128i *)&a7);
  v12 = a9;
  *(_QWORD *)(a1 + 16) = a8;
  v13 = 0;
  *(__m128i *)a1 = v11;
  if ( a4 )
  {
    _BitScanReverse(&a4, a4);
    v13 = 31 - (a4 ^ 0x1F) + 1;
  }
  *(_WORD *)(a1 + 34) = v13;
  v14 = a5[1].m128i_i64[0];
  *(_BYTE *)(a1 + 36) = v12;
  v15 = a10;
  *(_QWORD *)(a1 + 56) = v14;
  LODWORD(v14) = a11;
  v16 = _mm_loadu_si128(a5);
  *(_QWORD *)(a1 + 64) = a6;
  *(_DWORD *)(a1 + 72) = 0x7FFFFFFF;
  result = v15 & 0xF | (unsigned int)(16 * v14);
  *(_BYTE *)(a1 + 76) = 0;
  *(_BYTE *)(a1 + 37) = result;
  *(__m128i *)(a1 + 40) = v16;
  return result;
}
