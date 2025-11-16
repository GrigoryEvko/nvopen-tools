// Function: sub_33FC220
// Address: 0x33fc220
//
unsigned __int8 *__fastcall sub_33FC220(
        _QWORD *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int128 a7)
{
  __int64 v7; // rax
  int v8; // r9d
  __m128i v9; // xmm0

  v7 = a1[128];
  v8 = 0;
  v9 = _mm_loadu_si128((const __m128i *)&a7);
  if ( v7 )
    v8 = *(_DWORD *)(v7 + 8);
  return sub_33FBA10(a1, a2, a3, a4, a5, v8, v9.m128i_i64[0], v9.m128i_i64[1]);
}
