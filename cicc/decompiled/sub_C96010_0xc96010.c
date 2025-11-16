// Function: sub_C96010
// Address: 0xc96010
//
__int64 __fastcall sub_C96010(__m128i *a1)
{
  __m128i *v1; // r13
  __m128i *v2; // rdi
  size_t v3; // rax
  __m128i v4; // xmm0
  __m128i *i; // rax
  __int64 v6; // rdx
  __int64 v7; // rax
  __m128i *v8; // r12
  __m128i *v9; // rbx
  size_t v10; // rdx
  __int64 v11; // rax
  __m128i *v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rcx
  __int64 result; // rax
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rax
  __m128i *p_src; // [rsp+0h] [rbp-50h]
  size_t n; // [rsp+8h] [rbp-48h]
  __m128i src; // [rsp+10h] [rbp-40h] BYREF
  __m128i v22; // [rsp+20h] [rbp-30h]

  v1 = a1;
  v2 = a1 + 1;
  p_src = &src;
  if ( (__m128i *)v2[-1].m128i_i64[0] == v2 )
  {
    src = _mm_loadu_si128(v1 + 1);
  }
  else
  {
    p_src = (__m128i *)v2[-1].m128i_i64[0];
    src.m128i_i64[0] = v1[1].m128i_i64[0];
  }
  v3 = v1->m128i_u64[1];
  v4 = _mm_loadu_si128(v1 + 2);
  v1->m128i_i64[0] = (__int64)v2;
  v1->m128i_i64[1] = 0;
  v22 = v4;
  n = v3;
  v1[1].m128i_i8[0] = 0;
  if ( v1[-1].m128i_i64[1] >= v4.m128i_i64[1] )
  {
    v12 = p_src;
    if ( p_src != &src )
      goto LABEL_26;
LABEL_30:
    v17 = n;
    if ( n )
    {
      if ( n == 1 )
      {
        v2->m128i_i8[0] = src.m128i_i8[0];
        v17 = 1;
      }
      else
      {
        memcpy(v2, &src, n);
        v17 = n;
      }
    }
    v18 = v1->m128i_i64[0];
    v1->m128i_i64[1] = v17;
    *(_BYTE *)(v18 + v17) = 0;
    v2 = p_src;
    goto LABEL_22;
  }
  for ( i = v2; ; i = v9 )
  {
    v8 = (__m128i *)i[-4].m128i_i64[0];
    v9 = i - 3;
    v1 = i - 4;
    if ( &i[-3] == v8 )
    {
      v10 = i[-4].m128i_u64[1];
      if ( v10 )
      {
        if ( v10 == 1 )
          v2->m128i_i8[0] = v9->m128i_i8[0];
        else
          memcpy(v2, &i[-3], v10);
        v10 = v9[-1].m128i_u64[1];
        v2 = (__m128i *)v9[2].m128i_i64[0];
      }
      v8[2].m128i_i64[1] = v10;
      v2->m128i_i8[v10] = 0;
      v2 = (__m128i *)v8[-1].m128i_i64[0];
    }
    else
    {
      if ( v2 == i )
      {
        v11 = i[-4].m128i_i64[1];
        v9[2].m128i_i64[0] = (__int64)v8;
        v9[2].m128i_i64[1] = v11;
        v9[3].m128i_i64[0] = v9->m128i_i64[0];
      }
      else
      {
        v6 = i[-4].m128i_i64[1];
        v7 = i->m128i_i64[0];
        v9[2].m128i_i64[0] = (__int64)v8;
        v9[2].m128i_i64[1] = v6;
        v9[3].m128i_i64[0] = v9->m128i_i64[0];
        if ( v2 )
        {
          v9[-1].m128i_i64[0] = (__int64)v2;
          v9->m128i_i64[0] = v7;
          goto LABEL_8;
        }
      }
      v9[-1].m128i_i64[0] = (__int64)v9;
      v2 = v9;
    }
LABEL_8:
    v9[-1].m128i_i64[1] = 0;
    v2->m128i_i8[0] = 0;
    v9[4].m128i_i64[0] = v9[1].m128i_i64[0];
    v9[4].m128i_i64[1] = v9[1].m128i_i64[1];
    if ( v9[-2].m128i_i64[1] >= v22.m128i_i64[1] )
      break;
    v2 = (__m128i *)v9[-1].m128i_i64[0];
  }
  v12 = p_src;
  v2 = (__m128i *)v1->m128i_i64[0];
  if ( p_src == &src )
    goto LABEL_30;
  v3 = n;
  if ( v9 == v2 )
  {
LABEL_26:
    v1->m128i_i64[1] = v3;
    v16 = src.m128i_i64[0];
    v1->m128i_i64[0] = (__int64)v12;
    v1[1].m128i_i64[0] = v16;
    goto LABEL_27;
  }
  v1->m128i_i64[1] = n;
  v13 = src.m128i_i64[0];
  v14 = v1[1].m128i_i64[0];
  v1->m128i_i64[0] = (__int64)p_src;
  v1[1].m128i_i64[0] = v13;
  if ( v2 )
  {
    p_src = v2;
    src.m128i_i64[0] = v14;
    goto LABEL_22;
  }
LABEL_27:
  p_src = &src;
  v2 = &src;
LABEL_22:
  v2->m128i_i8[0] = 0;
  v1[2].m128i_i64[0] = v22.m128i_i64[0];
  result = v22.m128i_i64[1];
  v1[2].m128i_i64[1] = v22.m128i_i64[1];
  if ( p_src != &src )
    return j_j___libc_free_0(p_src, src.m128i_i64[0] + 1);
  return result;
}
