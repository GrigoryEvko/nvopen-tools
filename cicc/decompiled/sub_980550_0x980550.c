// Function: sub_980550
// Address: 0x980550
//
void __fastcall sub_980550(__m128i *a1, __int64 a2)
{
  __int64 v3; // rdi
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // rdx
  __int64 v8; // rcx

  v3 = (__int64)&a1[8].m128i_i64[1];
  *(_QWORD *)v3 = 0;
  *(_QWORD *)(v3 + 8) = 0;
  *(_QWORD *)(v3 + 16) = 0;
  *(_DWORD *)(v3 + 24) = 0;
  sub_980380(v3, a2 + 136);
  v4 = *(_QWORD *)(a2 + 168);
  a1[11].m128i_i64[0] = 0;
  a1[11].m128i_i64[1] = 0;
  a1[10].m128i_i64[1] = v4;
  a1[12].m128i_i64[0] = 0;
  a1[12].m128i_i64[1] = 0;
  a1[13].m128i_i64[0] = 0;
  a1[13].m128i_i64[1] = 0;
  *a1 = _mm_loadu_si128((const __m128i *)a2);
  a1[1] = _mm_loadu_si128((const __m128i *)(a2 + 16));
  a1[2] = _mm_loadu_si128((const __m128i *)(a2 + 32));
  a1[3] = _mm_loadu_si128((const __m128i *)(a2 + 48));
  a1[4] = _mm_loadu_si128((const __m128i *)(a2 + 64));
  a1[5] = _mm_loadu_si128((const __m128i *)(a2 + 80));
  a1[6] = _mm_loadu_si128((const __m128i *)(a2 + 96));
  a1[7] = _mm_loadu_si128((const __m128i *)(a2 + 112));
  a1[8].m128i_i16[0] = *(_WORD *)(a2 + 128);
  a1[8].m128i_i8[2] = *(_BYTE *)(a2 + 130);
  sub_97E2C0((__int64)a1[11].m128i_i64, (const __m128i **)(a2 + 176), v5, v6);
  sub_97E2C0((__int64)&a1[12].m128i_i64[1], (const __m128i **)(a2 + 200), v7, v8);
}
