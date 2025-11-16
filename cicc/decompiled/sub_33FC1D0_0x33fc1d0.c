// Function: sub_33FC1D0
// Address: 0x33fc1d0
//
unsigned __int8 *__fastcall sub_33FC1D0(
        _QWORD *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int128 a7,
        __int128 a8,
        __int128 a9,
        __int128 a10,
        __int128 a11)
{
  __int64 v11; // rax
  int v12; // r9d
  __int128 v13; // xmm0
  __int128 v14; // xmm1
  __int128 v15; // xmm2
  __int128 v16; // xmm3
  __int128 v17; // xmm4

  v11 = a1[128];
  v12 = 0;
  v13 = (__int128)_mm_loadu_si128((const __m128i *)&a7);
  v14 = (__int128)_mm_loadu_si128((const __m128i *)&a8);
  v15 = (__int128)_mm_loadu_si128((const __m128i *)&a9);
  v16 = (__int128)_mm_loadu_si128((const __m128i *)&a10);
  v17 = (__int128)_mm_loadu_si128((const __m128i *)&a11);
  if ( v11 )
    v12 = *(_DWORD *)(v11 + 8);
  return sub_33FC180(a1, a2, a3, a4, a5, v12, v13, v14, v15, v16, v17);
}
