// Function: sub_31C55E0
// Address: 0x31c55e0
//
unsigned int __fastcall sub_31C55E0(__int64 a1, const __m128i *a2)
{
  __m128i v2; // xmm1
  __m128i v3; // xmm2
  __m128i v4; // xmm3
  __int128 *v5; // rax

  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = &unk_5035D54;
  *(_QWORD *)(a1 + 56) = a1 + 104;
  *(_QWORD *)(a1 + 112) = a1 + 160;
  *(_DWORD *)(a1 + 24) = 4;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 64) = 1;
  *(_QWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 96) = 0;
  *(_QWORD *)(a1 + 104) = 0;
  *(_QWORD *)(a1 + 120) = 1;
  *(_QWORD *)(a1 + 128) = 0;
  *(_QWORD *)(a1 + 136) = 0;
  *(_QWORD *)(a1 + 152) = 0;
  *(_QWORD *)(a1 + 160) = 0;
  *(_BYTE *)(a1 + 168) = 0;
  *(_DWORD *)(a1 + 88) = 1065353216;
  *(_DWORD *)(a1 + 144) = 1065353216;
  v2 = _mm_loadu_si128(a2);
  v3 = _mm_loadu_si128(a2 + 1);
  v4 = _mm_loadu_si128(a2 + 2);
  *(_QWORD *)a1 = &unk_4A34B28;
  *(__m128i *)(a1 + 172) = v2;
  *(__m128i *)(a1 + 188) = v3;
  *(__m128i *)(a1 + 204) = v4;
  *(_DWORD *)(a1 + 220) = a2[3].m128i_i32[0];
  v5 = sub_BC2B00();
  return sub_31C5560((__int64)v5);
}
