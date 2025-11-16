// Function: sub_325F5E0
// Address: 0x325f5e0
//
signed __int64 __fastcall sub_325F5E0(__m128i *a1, __int64 *a2, __int64 a3)
{
  signed __int64 result; // rax
  __m128i *v4; // r8
  __int64 v5; // r15
  __m128i *v7; // r12
  __int64 v8; // rsi
  __int64 v9; // rdi
  __int64 v10; // r9
  __int64 v11; // rcx
  __m128i *v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rsi
  __int64 v15; // rdx
  __m128i *v16; // rbx
  __m128i *v17; // rax
  __int64 *v18; // r13
  __int64 v19; // rdx
  __m128i v20; // xmm3
  __int64 v21; // rbx
  __int64 v22; // rsi
  __int64 v23; // rcx
  __int64 v24; // r8
  __m128i v25; // xmm5

  result = (char *)a2 - (char *)a1;
  if ( (char *)a2 - (char *)a1 <= 256 )
    return result;
  v4 = (__m128i *)a2;
  v5 = a3;
  v7 = a1 + 1;
  if ( !a3 )
  {
    v18 = a2;
    goto LABEL_24;
  }
  while ( 2 )
  {
    v8 = a1[1].m128i_i64[1];
    v9 = v4[-1].m128i_i64[1];
    --v5;
    v10 = a1->m128i_i64[0];
    v11 = a1->m128i_i64[1];
    v12 = &a1[result >> 5];
    v13 = v12->m128i_i64[1];
    if ( v8 >= v13 )
    {
      if ( v8 >= v9 )
      {
        if ( v13 >= v9 )
        {
          *a1 = _mm_loadu_si128(v12);
          v12->m128i_i64[0] = v10;
          v12->m128i_i64[1] = v11;
          v14 = v4[-1].m128i_i64[1];
        }
        else
        {
          v14 = a1->m128i_i64[1];
          *a1 = _mm_loadu_si128(v4 - 1);
          v4[-1].m128i_i64[0] = v10;
          v4[-1].m128i_i64[1] = v11;
        }
        v11 = a1[1].m128i_i64[1];
      }
      else
      {
        v20 = _mm_loadu_si128(a1 + 1);
        a1[1].m128i_i64[0] = v10;
        a1[1].m128i_i64[1] = v11;
        *a1 = v20;
        v14 = v4[-1].m128i_i64[1];
      }
    }
    else if ( v13 >= v9 )
    {
      if ( v8 >= v9 )
      {
        v25 = _mm_loadu_si128(a1 + 1);
        a1[1].m128i_i64[0] = v10;
        a1[1].m128i_i64[1] = v11;
        *a1 = v25;
        v14 = v4[-1].m128i_i64[1];
      }
      else
      {
        v14 = a1->m128i_i64[1];
        *a1 = _mm_loadu_si128(v4 - 1);
        v4[-1].m128i_i64[0] = v10;
        v4[-1].m128i_i64[1] = v11;
        v11 = a1[1].m128i_i64[1];
      }
    }
    else
    {
      *a1 = _mm_loadu_si128(v12);
      v12->m128i_i64[0] = v10;
      v12->m128i_i64[1] = v11;
      v14 = v4[-1].m128i_i64[1];
      v11 = a1[1].m128i_i64[1];
    }
    v15 = a1->m128i_i64[1];
    v16 = v7;
    v17 = v4;
    while ( 1 )
    {
      v18 = (__int64 *)v16;
      if ( v11 < v15 )
        goto LABEL_12;
      --v17;
      if ( v14 > v15 )
      {
        do
          --v17;
        while ( v17->m128i_i64[1] > v15 );
      }
      if ( v17 <= v16 )
        break;
      v19 = v16->m128i_i64[0];
      *v16 = _mm_loadu_si128(v17);
      v14 = v17[-1].m128i_i64[1];
      v17->m128i_i64[0] = v19;
      v17->m128i_i64[1] = v11;
      v15 = a1->m128i_i64[1];
LABEL_12:
      v11 = v16[1].m128i_i64[1];
      ++v16;
    }
    sub_325F5E0(v16, v4, v5);
    result = (char *)v16 - (char *)a1;
    if ( (char *)v16 - (char *)a1 > 256 )
    {
      if ( v5 )
      {
        v4 = v16;
        continue;
      }
LABEL_24:
      v21 = result >> 4;
      v22 = ((result >> 4) - 2) >> 1;
      sub_325DF30((__int64)a1, v22, result >> 4, a1[v22].m128i_i64[0], a1[v22].m128i_i64[1]);
      do
      {
        --v22;
        sub_325DF30((__int64)a1, v22, v21, a1[v22].m128i_i64[0], a1[v22].m128i_i64[1]);
      }
      while ( v22 );
      do
      {
        v18 -= 2;
        v23 = *v18;
        v24 = v18[1];
        *(__m128i *)v18 = _mm_loadu_si128(a1);
        result = (signed __int64)sub_325DF30((__int64)a1, 0, ((char *)v18 - (char *)a1) >> 4, v23, v24);
      }
      while ( (char *)v18 - (char *)a1 > 16 );
    }
    return result;
  }
}
