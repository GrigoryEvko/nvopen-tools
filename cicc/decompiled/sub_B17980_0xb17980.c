// Function: sub_B17980
// Address: 0xb17980
//
void *__fastcall sub_B17980(__int64 a1, int a2, __int64 a3, __int64 a4, __int64 a5, const __m128i *a6, __int64 a7)
{
  __int64 v7; // r10
  __m128i v8; // xmm0

  v7 = *(_QWORD *)(a7 + 72);
  *(_DWORD *)(a1 + 8) = a2;
  *(_BYTE *)(a1 + 12) = 2;
  v8 = _mm_loadu_si128(a6);
  *(_QWORD *)(a1 + 424) = a7;
  *(_QWORD *)(a1 + 40) = a3;
  *(_QWORD *)(a1 + 48) = a4;
  *(_QWORD *)(a1 + 16) = v7;
  *(_QWORD *)(a1 + 56) = a5;
  *(_BYTE *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 80) = a1 + 96;
  *(_QWORD *)(a1 + 88) = 0x400000000LL;
  *(_BYTE *)(a1 + 416) = 0;
  *(_DWORD *)(a1 + 420) = -1;
  *(_QWORD *)a1 = &unk_49D9DE8;
  *(__m128i *)(a1 + 24) = v8;
  return &unk_49D9DE8;
}
