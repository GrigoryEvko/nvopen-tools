// Function: sub_D1D220
// Address: 0xd1d220
//
__int64 __fastcall sub_D1D220(__int64 a1, __int64 a2, __m128i *a3)
{
  __m128i v3; // xmm1
  __int64 v4; // rcx
  __int64 v5; // rax
  __m128i v6; // xmm0
  __int64 v7; // rax

  v3 = _mm_loadu_si128((const __m128i *)(a1 + 8));
  v4 = *(_QWORD *)(a1 + 32);
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)a1 = a2;
  v5 = a3[1].m128i_i64[0];
  v6 = _mm_loadu_si128(a3);
  a3[1].m128i_i64[0] = 0;
  *(_QWORD *)(a1 + 24) = v5;
  v7 = a3[1].m128i_i64[1];
  *a3 = v3;
  *(_QWORD *)(a1 + 32) = v7;
  a3[1].m128i_i64[1] = v4;
  *(_QWORD *)(a1 + 48) = a1 + 72;
  *(_QWORD *)(a1 + 152) = a1 + 176;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 56) = 8;
  *(_DWORD *)(a1 + 64) = 0;
  *(_BYTE *)(a1 + 68) = 1;
  *(_BYTE *)(a1 + 136) = 0;
  *(_QWORD *)(a1 + 144) = 0;
  *(_QWORD *)(a1 + 160) = 8;
  *(_DWORD *)(a1 + 168) = 0;
  *(_BYTE *)(a1 + 172) = 1;
  *(_QWORD *)(a1 + 240) = 0;
  *(_QWORD *)(a1 + 248) = 0;
  *(_QWORD *)(a1 + 256) = 0;
  *(_DWORD *)(a1 + 264) = 0;
  *(_QWORD *)(a1 + 272) = 0;
  *(_QWORD *)(a1 + 280) = 0;
  *(_QWORD *)(a1 + 288) = 0;
  *(_DWORD *)(a1 + 296) = 0;
  *(__m128i *)(a1 + 8) = v6;
  *(_QWORD *)(a1 + 304) = 0;
  *(_QWORD *)(a1 + 312) = 0;
  *(_QWORD *)(a1 + 320) = 0;
  *(_DWORD *)(a1 + 328) = 0;
  *(_QWORD *)(a1 + 344) = a1 + 336;
  *(_QWORD *)(a1 + 336) = a1 + 336;
  *(_QWORD *)(a1 + 352) = 0;
  return a1 + 336;
}
