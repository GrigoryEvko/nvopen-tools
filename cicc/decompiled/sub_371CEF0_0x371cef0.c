// Function: sub_371CEF0
// Address: 0x371cef0
//
__int64 __fastcall sub_371CEF0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        unsigned __int32 a9)
{
  __int64 v10; // r9
  __int64 v12; // rbx
  __int64 i; // rcx
  __m128i *v14; // rdx
  __int64 result; // rax
  __m128i *v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r10
  unsigned int v19; // r9d
  __int64 v20; // rsi
  __int64 v21; // rcx
  __m128i *v22; // rax
  unsigned int v23; // ebx
  __int64 v24; // rax
  const __m128i *v25; // rcx
  __int64 v26; // [rsp+0h] [rbp-38h]

  v10 = a3 & 1;
  v12 = (a3 - 1) / 2;
  if ( a2 >= v12 )
  {
    result = 3 * a2;
    v14 = (__m128i *)(a1 + 24 * a2);
    if ( v10 )
      goto LABEL_19;
    result = a2;
    goto LABEL_22;
  }
  v26 = a3 & 1;
  for ( i = a2; ; i = result )
  {
    result = 2 * (i + 1) - 1;
    v17 = a1 + 48 * (i + 1);
    v14 = (__m128i *)(a1 + 24 * result);
    v18 = *(_QWORD *)(v17 + 8);
    v19 = *(_DWORD *)(v14->m128i_i64[1] + 72);
    if ( *(_DWORD *)(v18 + 72) == v19 )
    {
      if ( *(_DWORD *)(v17 + 16) >= v14[1].m128i_i32[0] )
      {
        v14 = (__m128i *)(a1 + 48 * (i + 1));
        result = 2 * (i + 1);
      }
    }
    else if ( *(_DWORD *)(v18 + 72) >= v19 )
    {
      v14 = (__m128i *)(a1 + 48 * (i + 1));
      result = 2 * (i + 1);
    }
    v16 = (__m128i *)(a1 + 24 * i);
    *v16 = _mm_loadu_si128(v14);
    v16[1].m128i_i64[0] = v14[1].m128i_i64[0];
    if ( result >= v12 )
      break;
  }
  if ( !v26 )
  {
LABEL_22:
    if ( (a3 - 2) / 2 == result )
    {
      result = 2 * result + 1;
      v25 = (const __m128i *)(a1 + 24 * result);
      *v14 = _mm_loadu_si128(v25);
      v14[1].m128i_i64[0] = v25[1].m128i_i64[0];
      v14 = (__m128i *)v25;
    }
  }
  v20 = (result - 1) / 2;
  if ( result > a2 )
  {
    v21 = result;
    while ( 1 )
    {
      v23 = *(_DWORD *)(a8 + 72);
      v14 = (__m128i *)(a1 + 24 * v20);
      v24 = v14->m128i_i64[1];
      if ( *(_DWORD *)(v24 + 72) == v23 )
      {
        if ( v14[1].m128i_i32[0] >= a9 )
        {
LABEL_18:
          result = 3 * v21;
          v14 = (__m128i *)(a1 + 24 * v21);
          break;
        }
      }
      else if ( *(_DWORD *)(v24 + 72) >= v23 )
      {
        goto LABEL_18;
      }
      v22 = (__m128i *)(a1 + 24 * v21);
      *v22 = _mm_loadu_si128(v14);
      v22[1].m128i_i64[0] = v14[1].m128i_i64[0];
      v21 = v20;
      result = (v20 - 1) / 2;
      if ( a2 >= v20 )
        break;
      v20 = (v20 - 1) / 2;
    }
  }
LABEL_19:
  v14->m128i_i64[0] = a7;
  v14->m128i_i64[1] = a8;
  v14[1].m128i_i32[0] = a9;
  return result;
}
