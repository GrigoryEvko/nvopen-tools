// Function: sub_2EAC3E0
// Address: 0x2eac3e0
//
__int64 __fastcall sub_2EAC3E0(
        __int64 a1,
        __int16 a2,
        __int64 a3,
        char a4,
        const __m128i *a5,
        __int64 a6,
        __m128i a7,
        __int64 a8,
        char a9,
        char a10,
        unsigned __int8 a11)
{
  __m128i v11; // xmm1
  __m128i v12; // xmm2
  __int64 v13; // rax
  char v14; // dl
  __m128i v15; // xmm0
  __int64 result; // rax

  v11 = _mm_loadu_si128(a5);
  v12 = _mm_loadu_si128(a5 + 1);
  *(_QWORD *)(a1 + 24) = a3;
  *(_WORD *)(a1 + 32) = a2;
  *(_BYTE *)(a1 + 34) = a4;
  v13 = a8;
  v14 = a10;
  *(_QWORD *)(a1 + 72) = a6;
  v15 = _mm_loadu_si128(&a7);
  *(_DWORD *)(a1 + 80) = 0x7FFFFFFF;
  *(_QWORD *)(a1 + 16) = v13;
  LOBYTE(v13) = a9;
  *(_BYTE *)(a1 + 84) = 0;
  *(_BYTE *)(a1 + 36) = v13;
  LODWORD(v13) = a11;
  *(__m128i *)a1 = v15;
  *(__m128i *)(a1 + 40) = v11;
  result = v14 & 0xF | (unsigned int)(16 * v13);
  *(__m128i *)(a1 + 56) = v12;
  *(_BYTE *)(a1 + 37) = result;
  return result;
}
