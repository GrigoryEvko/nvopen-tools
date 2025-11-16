// Function: sub_15CA540
// Address: 0x15ca540
//
void *__fastcall sub_15CA540(__int64 a1, __int64 a2, __int64 a3, __int64 a4, const __m128i *a5, __int64 a6)
{
  __int64 v6; // rax
  __m128i v7; // xmm0
  __int64 v8; // rax

  v6 = *(_QWORD *)(a6 + 56);
  *(_DWORD *)(a1 + 8) = 9;
  *(_BYTE *)(a1 + 12) = 2;
  v7 = _mm_loadu_si128(a5);
  *(_QWORD *)(a1 + 16) = v6;
  v8 = a5[1].m128i_i64[0];
  *(__m128i *)(a1 + 24) = v7;
  *(_QWORD *)(a1 + 40) = v8;
  *(_QWORD *)(a1 + 88) = a1 + 104;
  *(_QWORD *)(a1 + 96) = 0x400000000LL;
  *(_QWORD *)(a1 + 48) = a2;
  *(_QWORD *)(a1 + 56) = a3;
  *(_QWORD *)(a1 + 64) = a4;
  *(_BYTE *)(a1 + 80) = 0;
  *(_BYTE *)(a1 + 456) = 0;
  *(_DWORD *)(a1 + 460) = -1;
  *(_QWORD *)(a1 + 464) = a6;
  *(_QWORD *)a1 = &unk_49ECFC8;
  return &unk_49ECFC8;
}
