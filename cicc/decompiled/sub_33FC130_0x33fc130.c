// Function: sub_33FC130
// Address: 0x33fc130
//
unsigned __int8 *__fastcall sub_33FC130(
        _QWORD *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int128 a7,
        __int128 a8,
        __int128 a9,
        __int128 a10)
{
  __int64 v10; // rax
  int v11; // r9d
  __int128 v12; // xmm0
  __int128 v13; // xmm1
  __int128 v14; // xmm2
  __int128 v15; // xmm3

  v10 = a1[128];
  v11 = 0;
  v12 = (__int128)_mm_loadu_si128((const __m128i *)&a7);
  v13 = (__int128)_mm_loadu_si128((const __m128i *)&a8);
  v14 = (__int128)_mm_loadu_si128((const __m128i *)&a9);
  v15 = (__int128)_mm_loadu_si128((const __m128i *)&a10);
  if ( v10 )
    v11 = *(_DWORD *)(v10 + 8);
  return sub_33FC0E0(a1, a2, a3, a4, a5, v11, v12, v13, v14, v15);
}
