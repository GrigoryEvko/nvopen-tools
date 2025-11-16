// Function: sub_B17850
// Address: 0xb17850
//
void *__fastcall sub_B17850(__int64 a1, __int64 a2, __int64 a3, __int64 a4, const __m128i *a5, __int64 a6)
{
  __int64 v6; // rax
  __m128i v7; // xmm0

  v6 = *(_QWORD *)(a6 + 72);
  *(_DWORD *)(a1 + 8) = 15;
  *(_BYTE *)(a1 + 12) = 2;
  v7 = _mm_loadu_si128(a5);
  *(_QWORD *)(a1 + 16) = v6;
  *(_QWORD *)(a1 + 80) = a1 + 96;
  *(_QWORD *)(a1 + 88) = 0x400000000LL;
  *(_QWORD *)(a1 + 40) = a2;
  *(_QWORD *)(a1 + 48) = a3;
  *(_QWORD *)(a1 + 56) = a4;
  *(_BYTE *)(a1 + 72) = 0;
  *(_BYTE *)(a1 + 416) = 0;
  *(_DWORD *)(a1 + 420) = -1;
  *(_QWORD *)(a1 + 424) = a6;
  *(_QWORD *)a1 = &unk_49D9DE8;
  *(__m128i *)(a1 + 24) = v7;
  return &unk_49D9DE8;
}
