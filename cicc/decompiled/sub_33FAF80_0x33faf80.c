// Function: sub_33FAF80
// Address: 0x33faf80
//
unsigned __int8 *__fastcall sub_33FAF80(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        _DWORD a6,
        __m128i a7)
{
  __int64 v7; // rax
  int v8; // r9d
  __m128i v9; // xmm0

  v7 = *(_QWORD *)(a1 + 1024);
  v8 = 0;
  v9 = _mm_loadu_si128(&a7);
  if ( v7 )
    v8 = *(_DWORD *)(v7 + 8);
  return sub_33FA050(a1, a2, a3, a4, a5, v8, (unsigned __int8 *)v9.m128i_i64[0], v9.m128i_i64[1]);
}
