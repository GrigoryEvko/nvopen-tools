// Function: sub_2E09B80
// Address: 0x2e09b80
//
__m128i *__fastcall sub_2E09B80(__int64 a1, __int64 a2, _QWORD *a3, const __m128i *a4)
{
  char v5; // r14
  __m128i *v7; // r12
  __int64 v9; // rax
  unsigned int v10; // edx
  unsigned int v11; // eax

  v5 = 1;
  if ( !a2 && a3 != (_QWORD *)(a1 + 8) )
  {
    v9 = a3[4];
    v10 = *(_DWORD *)((a4->m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL) + 24) | (a4->m128i_i64[0] >> 1) & 3;
    v11 = *(_DWORD *)((v9 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (v9 >> 1) & 3;
    v5 = v10 < v11
      || v10 <= v11
      && (*(_DWORD *)((a4->m128i_i64[1] & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(a4->m128i_i64[1] >> 1) & 3) < (*(_DWORD *)((a3[5] & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)((__int64)a3[5] >> 1) & 3);
  }
  v7 = (__m128i *)sub_22077B0(0x38u);
  v7[2] = _mm_loadu_si128(a4);
  v7[3].m128i_i64[0] = a4[1].m128i_i64[0];
  sub_220F040(v5, (__int64)v7, a3, (_QWORD *)(a1 + 8));
  ++*(_QWORD *)(a1 + 40);
  return v7;
}
