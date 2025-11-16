// Function: sub_15CA7C0
// Address: 0x15ca7c0
//
void *__fastcall sub_15CA7C0(__int64 a1, int a2, __int64 a3, __int64 a4, __int64 a5, const __m128i *a6, __int64 a7)
{
  __int64 v7; // r10
  __m128i v8; // xmm0
  __int64 v9; // rsi

  v7 = *(_QWORD *)(a7 + 56);
  *(_DWORD *)(a1 + 8) = a2;
  *(_BYTE *)(a1 + 12) = 2;
  v8 = _mm_loadu_si128(a6);
  v9 = a6[1].m128i_i64[0];
  *(_QWORD *)(a1 + 464) = a7;
  *(_QWORD *)(a1 + 48) = a3;
  *(_QWORD *)(a1 + 56) = a4;
  *(_QWORD *)(a1 + 16) = v7;
  *(_QWORD *)(a1 + 40) = v9;
  *(_QWORD *)(a1 + 64) = a5;
  *(_BYTE *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 88) = a1 + 104;
  *(_QWORD *)(a1 + 96) = 0x400000000LL;
  *(_BYTE *)(a1 + 456) = 0;
  *(_DWORD *)(a1 + 460) = -1;
  *(_QWORD *)a1 = &unk_49ECFF8;
  *(__m128i *)(a1 + 24) = v8;
  return &unk_49ECFF8;
}
