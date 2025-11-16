// Function: sub_1DB3B70
// Address: 0x1db3b70
//
__m128i *__fastcall sub_1DB3B70(__int64 a1, __int64 a2, __int64 a3, const __m128i *a4)
{
  _BOOL4 v5; // r14d
  __m128i *v7; // r12
  __int64 v9; // rax
  unsigned int v10; // edx
  unsigned int v11; // eax

  v5 = 1;
  if ( !a2 && a3 != a1 + 8 )
  {
    v9 = *(_QWORD *)(a3 + 32);
    v10 = *(_DWORD *)((a4->m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL) + 24) | (a4->m128i_i64[0] >> 1) & 3;
    v11 = *(_DWORD *)((v9 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (v9 >> 1) & 3;
    v5 = v10 < v11
      || v10 <= v11
      && (*(_DWORD *)((a4->m128i_i64[1] & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(a4->m128i_i64[1] >> 1) & 3) < (*(_DWORD *)((*(_QWORD *)(a3 + 40) & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*(__int64 *)(a3 + 40) >> 1) & 3);
  }
  v7 = (__m128i *)sub_22077B0(56);
  v7[2] = _mm_loadu_si128(a4);
  v7[3].m128i_i64[0] = a4[1].m128i_i64[0];
  sub_220F040(v5, v7, a3, a1 + 8);
  ++*(_QWORD *)(a1 + 40);
  return v7;
}
