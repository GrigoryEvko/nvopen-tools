// Function: sub_1BFB520
// Address: 0x1bfb520
//
void __fastcall sub_1BFB520(__int64 a1, const __m128i *a2)
{
  __m128i v2; // xmm0
  __m128i v3; // xmm1
  __m128i v4; // xmm2
  __int32 v5; // eax
  __int64 v6; // rax

  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = &unk_4FB9E2C;
  *(_QWORD *)(a1 + 80) = a1 + 64;
  *(_QWORD *)(a1 + 88) = a1 + 64;
  *(_QWORD *)(a1 + 128) = a1 + 112;
  *(_QWORD *)(a1 + 136) = a1 + 112;
  *(_DWORD *)(a1 + 24) = 5;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_DWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 96) = 0;
  *(_DWORD *)(a1 + 112) = 0;
  *(_QWORD *)(a1 + 120) = 0;
  *(_QWORD *)(a1 + 144) = 0;
  *(_BYTE *)(a1 + 152) = 0;
  v2 = _mm_loadu_si128(a2);
  v3 = _mm_loadu_si128(a2 + 1);
  v4 = _mm_loadu_si128(a2 + 2);
  *(_QWORD *)a1 = &unk_49F71F0;
  v5 = a2[3].m128i_i32[0];
  *(__m128i *)(a1 + 156) = v2;
  *(__m128i *)(a1 + 172) = v3;
  *(_DWORD *)(a1 + 204) = v5;
  *(__m128i *)(a1 + 188) = v4;
  v6 = sub_163A1D0();
  sub_1BFB430(v6);
}
