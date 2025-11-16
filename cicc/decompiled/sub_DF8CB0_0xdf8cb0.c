// Function: sub_DF8CB0
// Address: 0xdf8cb0
//
size_t __fastcall sub_DF8CB0(__int64 a1, int a2, __int64 a3, char *a4, __int64 a5, int a6, __int64 a7, __int128 a8)
{
  __int64 v10; // rcx
  char *v11; // rsi
  __int64 v12; // rdi
  __m128i v13; // xmm0

  v10 = a1 + 88;
  *(_DWORD *)(a1 + 16) = a2;
  v11 = (char *)(a1 + 40);
  v12 = a1 + 24;
  *(_QWORD *)(v12 + 48) = v10;
  *(_QWORD *)(v12 - 16) = a3;
  v13 = _mm_loadu_si128((const __m128i *)&a8);
  *(_QWORD *)v12 = v11;
  *(_DWORD *)(v12 + 96) = a6;
  *(_QWORD *)(v12 - 24) = a7;
  *(_QWORD *)(v12 + 8) = 0x400000000LL;
  *(_QWORD *)(v12 + 56) = 0x400000000LL;
  *(_QWORD *)(v12 + 120) = 0;
  *(__m128i *)(v12 + 104) = v13;
  return sub_DF6530(v12, v11, a4, &a4[8 * a5]);
}
