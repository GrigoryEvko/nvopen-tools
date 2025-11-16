// Function: sub_291E540
// Address: 0x291e540
//
__m128i *__fastcall sub_291E540(
        __m128i *a1,
        __int64 *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int128 a7,
        __int64 a8)
{
  unsigned __int64 v8; // rdx
  __m128i v9; // xmm0
  __int64 *v11; // rbx
  __int64 v12; // r12
  __int64 v13; // rsi
  __m128i v14; // [rsp+0h] [rbp-40h] BYREF
  __int64 v15; // [rsp+10h] [rbp-30h]

  v8 = *a2 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (*a2 & 4) != 0 )
  {
    v11 = *(__int64 **)v8;
    v12 = *(_QWORD *)v8 + 8LL * *(unsigned int *)(v8 + 8);
  }
  else
  {
    if ( !v8 )
    {
      v15 = a8;
      v14 = _mm_loadu_si128((const __m128i *)&a7);
      goto LABEL_4;
    }
    v11 = a2;
    v12 = (__int64)(a2 + 1);
  }
  v14 = _mm_loadu_si128((const __m128i *)&a7);
  v15 = a8;
  while ( v11 != (__int64 *)v12 )
  {
    v13 = *v11++;
    sub_291DAE0(v14.m128i_i64, v13);
  }
LABEL_4:
  v9 = _mm_loadu_si128(&v14);
  a1[1].m128i_i64[0] = v15;
  *a1 = v9;
  return a1;
}
