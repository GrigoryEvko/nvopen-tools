// Function: sub_3218310
// Address: 0x3218310
//
__int64 __fastcall sub_3218310(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int128 a7,
        __int64 a8)
{
  __int64 v8; // r8
  __int64 v10; // r10
  __int64 v11; // rbx
  __int64 v12; // r9
  __int64 result; // rax
  __m128i *v14; // rdx
  __m128i *v15; // rcx
  __int64 v16; // rcx
  const __m128i *v17; // rsi
  __int64 v18; // rcx
  __int64 v19; // rax
  __int64 v20; // rcx
  __m128i v21; // [rsp+0h] [rbp-30h] BYREF
  __int64 v22; // [rsp+10h] [rbp-20h]

  v8 = a2;
  v10 = (a3 - 1) / 2;
  v11 = a3 & 1;
  v12 = a8;
  if ( a2 >= v10 )
  {
    result = 3 * a2;
    v14 = (__m128i *)(a1 + 24 * a2);
    if ( v11 )
    {
      v21 = _mm_loadu_si128((const __m128i *)&a7);
      goto LABEL_13;
    }
    result = a2;
    goto LABEL_16;
  }
  while ( 1 )
  {
    result = 2 * (a2 + 1);
    v14 = (__m128i *)(a1 + 48 * (a2 + 1));
    if ( *(_DWORD *)(v14[1].m128i_i64[0] + 16) < *(_DWORD *)(v14[-1].m128i_i64[1] + 16) )
    {
      --result;
      v14 = (__m128i *)(a1 + 24 * result);
    }
    v15 = (__m128i *)(a1 + 24 * a2);
    *v15 = _mm_loadu_si128(v14);
    v15[1].m128i_i64[0] = v14[1].m128i_i64[0];
    if ( result >= v10 )
      break;
    a2 = result;
  }
  if ( !v11 )
  {
LABEL_16:
    if ( (a3 - 2) / 2 == result )
    {
      v18 = result + 1;
      v19 = 2 * (result + 1);
      v20 = v19 + 4 * v18;
      result = v19 - 1;
      *v14 = _mm_loadu_si128((const __m128i *)(a1 + 8 * v20 - 24));
      v14[1].m128i_i64[0] = *(_QWORD *)(a1 + 8 * v20 - 24 + 16);
      v14 = (__m128i *)(a1 + 24 * result);
    }
  }
  v22 = a8;
  v21 = _mm_loadu_si128((const __m128i *)&a7);
  v16 = (result - 1) / 2;
  if ( result > v8 )
  {
    while ( 1 )
    {
      v17 = (const __m128i *)(a1 + 24 * v16);
      v14 = (__m128i *)(a1 + 24 * result);
      result = v17[1].m128i_i64[0];
      if ( *(_DWORD *)(result + 16) >= *(_DWORD *)(v12 + 16) )
        break;
      *v14 = _mm_loadu_si128(v17);
      v14[1].m128i_i64[0] = v17[1].m128i_i64[0];
      result = v16;
      if ( v8 >= v16 )
      {
        v14 = (__m128i *)(a1 + 24 * v16);
        break;
      }
      v16 = (v16 - 1) / 2;
    }
  }
LABEL_13:
  v14[1].m128i_i64[0] = v12;
  *v14 = _mm_loadu_si128(&v21);
  return result;
}
