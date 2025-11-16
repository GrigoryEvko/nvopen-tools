// Function: sub_6E2E50
// Address: 0x6e2e50
//
__int64 __fastcall sub_6E2E50(char a1, __int64 a2)
{
  __m128i v2; // xmm0
  __m128i v3; // xmm1
  __int64 v4; // rax
  __int64 v5; // rax

  *(_QWORD *)a2 = 0;
  *(_DWORD *)(a2 + 17) &= 0xE0000000;
  *(_QWORD *)(a2 + 8) = 0;
  v2 = _mm_loadu_si128(xmmword_4F07340);
  *(_QWORD *)(a2 + 88) = 0;
  *(__m128i *)(a2 + 24) = v2;
  v3 = _mm_loadu_si128(&xmmword_4F07340[1]);
  *(_QWORD *)(a2 + 96) = 0;
  *(__m128i *)(a2 + 40) = v3;
  v4 = xmmword_4F07340[2].m128i_i64[0];
  *(_QWORD *)(a2 + 104) = 0;
  *(_QWORD *)(a2 + 56) = v4;
  *(_BYTE *)(a2 + 64) = 0;
  *(_QWORD *)(a2 + 128) = 0;
  v5 = *(_QWORD *)&dword_4F077C8;
  *(_QWORD *)(a2 + 136) = 0;
  *(_QWORD *)(a2 + 68) = v5;
  *(_QWORD *)(a2 + 76) = v5;
  *(_QWORD *)(a2 + 112) = v5;
  *(_QWORD *)(a2 + 120) = v5;
  return sub_6E2DD0(a2, a1);
}
