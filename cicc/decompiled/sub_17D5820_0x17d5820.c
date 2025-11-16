// Function: sub_17D5820
// Address: 0x17d5820
//
__m128i *__fastcall sub_17D5820(__int128 a1, __int64 a2)
{
  __m128i *result; // rax
  __int64 v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // r12
  int v7; // r8d
  int v8; // r9d
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // rax
  __int64 v12; // rdx
  __m128i v13; // [rsp+0h] [rbp-40h] BYREF
  __int64 v14; // [rsp+10h] [rbp-30h]

  if ( byte_4FA4360 )
  {
    result = (__m128i *)sub_17D4DA0(a1);
    v6 = (__int64)result;
    if ( !result )
      return result;
    result = (__m128i *)sub_17D4880(a1, *((const char **)&a1 + 1), v4, v5);
    if ( !*(_BYTE *)(a1 + 488) )
      return result;
    goto LABEL_11;
  }
  result = (__m128i *)sub_17D4DA0(a1);
  v6 = (__int64)result;
  if ( result && result[1].m128i_i8[0] > 0x17u )
  {
    result = (__m128i *)sub_17D4880(a1, *((const char **)&a1 + 1), v9, v10);
    if ( result && result[1].m128i_i8[0] <= 0x17u )
      result = 0;
    if ( *(_BYTE *)(a1 + 488) )
    {
LABEL_11:
      v13.m128i_i64[1] = (__int64)result;
      v11 = *(unsigned int *)(a1 + 504);
      v13.m128i_i64[0] = v6;
      v14 = a2;
      if ( (unsigned int)v11 >= *(_DWORD *)(a1 + 508) )
      {
        sub_16CD150(a1 + 496, (const void *)(a1 + 512), 0, 24, v7, v8);
        v11 = *(unsigned int *)(a1 + 504);
      }
      result = (__m128i *)(*(_QWORD *)(a1 + 496) + 24 * v11);
      v12 = v14;
      *result = _mm_loadu_si128(&v13);
      result[1].m128i_i64[0] = v12;
      ++*(_DWORD *)(a1 + 504);
    }
  }
  return result;
}
