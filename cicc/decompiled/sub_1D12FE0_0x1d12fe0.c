// Function: sub_1D12FE0
// Address: 0x1d12fe0
//
__int64 __fastcall sub_1D12FE0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int32 a8,
        __int64 a9)
{
  __int64 v9; // r8
  __int64 v10; // r13
  __int64 v12; // rbx
  __int64 result; // rax
  __m128i *v14; // rdx
  __m128i *v15; // rcx
  __int64 v16; // rcx
  const __m128i *v17; // rsi
  __int64 v18; // rcx
  __int64 v19; // rax
  __int64 v20; // rcx
  __int64 v21; // rcx

  v9 = a2;
  v10 = a3 & 1;
  v12 = (a3 - 1) / 2;
  if ( a2 >= v12 )
  {
    result = 3 * a2;
    v14 = (__m128i *)(a1 + 24 * a2);
    if ( v10 )
      goto LABEL_13;
    result = a2;
    goto LABEL_16;
  }
  while ( 1 )
  {
    result = 2 * (a2 + 1);
    v14 = (__m128i *)(a1 + 48 * (a2 + 1));
    if ( v14->m128i_i64[0] < v14[-2].m128i_i64[1] )
    {
      --result;
      v14 = (__m128i *)(a1 + 24 * result);
    }
    v15 = (__m128i *)(a1 + 24 * a2);
    *v15 = _mm_loadu_si128(v14);
    v15[1].m128i_i64[0] = v14[1].m128i_i64[0];
    if ( result >= v12 )
      break;
    a2 = result;
  }
  if ( !v10 )
  {
LABEL_16:
    if ( (a3 - 2) / 2 == result )
    {
      v18 = result + 1;
      v19 = 2 * (result + 1);
      v20 = v19 + 4 * v18;
      result = v19 - 1;
      v21 = a1 + 8 * v20;
      *v14 = _mm_loadu_si128((const __m128i *)(v21 - 24));
      v14[1].m128i_i64[0] = *(_QWORD *)(v21 - 8);
      v14 = (__m128i *)(a1 + 24 * result);
    }
  }
  v16 = (result - 1) / 2;
  if ( result > v9 )
  {
    while ( 1 )
    {
      result *= 3;
      v17 = (const __m128i *)(a1 + 24 * v16);
      v14 = (__m128i *)(a1 + 8 * result);
      if ( v17->m128i_i64[0] >= a7 )
        break;
      *v14 = _mm_loadu_si128(v17);
      v14[1].m128i_i64[0] = v17[1].m128i_i64[0];
      result = v16;
      if ( v9 >= v16 )
      {
        v14 = (__m128i *)(a1 + 24 * v16);
        break;
      }
      v16 = (v16 - 1) / 2;
    }
  }
LABEL_13:
  v14->m128i_i64[0] = a7;
  v14->m128i_i32[2] = a8;
  v14[1].m128i_i64[0] = a9;
  return result;
}
