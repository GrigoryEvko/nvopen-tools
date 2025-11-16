// Function: sub_3406EB0
// Address: 0x3406eb0
//
unsigned __int8 *__fastcall sub_3406EB0(
        _QWORD *a1,
        unsigned int a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int128 a7,
        __int128 a8)
{
  __int64 v8; // rax
  int v9; // r9d
  __int128 v10; // xmm0
  __int128 v11; // xmm1

  v8 = a1[128];
  v9 = 0;
  v10 = (__int128)_mm_loadu_si128((const __m128i *)&a7);
  v11 = (__int128)_mm_loadu_si128((const __m128i *)&a8);
  if ( v8 )
    v9 = *(_DWORD *)(v8 + 8);
  return sub_3405C90(a1, a2, a3, a4, a5, v9, (__m128i)v10, v10, v11);
}
