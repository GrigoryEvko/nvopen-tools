// Function: sub_371E770
// Address: 0x371e770
//
__int64 __fastcall sub_371E770(const __m128i *a1, __m128i *a2, unsigned __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax
  __int64 v8; // r13
  __m128i *v9; // rbx
  __int64 i; // r14
  __m128i v11; // xmm1
  __int64 v12; // rdx
  __m128i v13; // [rsp+0h] [rbp-40h]
  __int64 v14; // [rsp+10h] [rbp-30h]

  result = (char *)a2 - (char *)a1;
  v8 = 0xAAAAAAAAAAAAAAABLL * (((char *)a2 - (char *)a1) >> 3);
  v9 = a2;
  if ( (char *)a2 - (char *)a1 <= 24 )
    goto LABEL_10;
  for ( i = (v8 - 2) / 2; ; --i )
  {
    v13 = _mm_loadu_si128((const __m128i *)((char *)a1 + 24 * i));
    result = sub_371CEF0((__int64)a1, i, v8, a4, a5, a6, v13.m128i_i64[0], v13.m128i_i64[1], a1[1].m128i_i64[3 * i]);
    if ( !i )
      break;
  }
  if ( a3 > (unsigned __int64)a2 )
  {
    while ( 1 )
    {
      v12 = v9->m128i_i64[1];
      result = *(unsigned int *)(a1->m128i_i64[1] + 72);
      if ( *(_DWORD *)(v12 + 72) == (_DWORD)result )
        break;
      if ( *(_DWORD *)(v12 + 72) < (unsigned int)result )
        goto LABEL_8;
LABEL_9:
      v9 = (__m128i *)((char *)v9 + 24);
LABEL_10:
      if ( a3 <= (unsigned __int64)v9 )
        return result;
    }
    result = a1[1].m128i_u32[0];
    if ( v9[1].m128i_i32[0] >= (unsigned int)result )
      goto LABEL_9;
LABEL_8:
    v11 = _mm_loadu_si128(v9);
    v14 = v9[1].m128i_i64[0];
    *v9 = _mm_loadu_si128(a1);
    v9[1].m128i_i64[0] = a1[1].m128i_i64[0];
    result = sub_371CEF0((__int64)a1, 0, v8, a4, a5, a6, v11.m128i_i64[0], v11.m128i_i64[1], v14);
    goto LABEL_9;
  }
  return result;
}
