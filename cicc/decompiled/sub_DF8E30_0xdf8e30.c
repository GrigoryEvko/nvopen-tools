// Function: sub_DF8E30
// Address: 0xdf8e30
//
size_t __fastcall sub_DF8E30(
        __int64 a1,
        int a2,
        __int64 a3,
        char *a4,
        __int64 a5,
        int a6,
        char *a7,
        __int64 a8,
        __int64 a9,
        __int128 a10,
        __int64 a11)
{
  char *v14; // rsi
  __m128i v15; // xmm0
  __int64 v16; // rdi
  __int64 v17; // rax

  *(_DWORD *)(a1 + 16) = a2;
  v14 = (char *)(a1 + 40);
  *(_QWORD *)(a1 + 72) = a1 + 88;
  v15 = _mm_loadu_si128((const __m128i *)&a10);
  v16 = a1 + 24;
  *(_QWORD *)(v16 - 24) = a9;
  *(_QWORD *)(v16 + 8) = 0x400000000LL;
  *(_QWORD *)(v16 + 56) = 0x400000000LL;
  v17 = a11;
  *(_QWORD *)v16 = v14;
  *(_QWORD *)(v16 + 120) = v17;
  *(_QWORD *)(v16 - 16) = a3;
  *(_DWORD *)(v16 + 96) = a6;
  *(__m128i *)(v16 + 104) = v15;
  sub_DF6530(v16, v14, a7, &a7[8 * a8]);
  return sub_DF6BA0(a1 + 72, *(char **)(a1 + 72), a4, &a4[8 * a5]);
}
