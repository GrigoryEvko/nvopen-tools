// Function: sub_16CFDB0
// Address: 0x16cfdb0
//
__int64 __fastcall sub_16CFDB0(__m128i *a1, __m128i *a2)
{
  __m128i *v3; // rdi
  __m128i *v5; // rax
  size_t v6; // rax
  __m128i *v7; // r14
  __m128i *v8; // rax
  __m128i *v9; // rax
  __int64 result; // rax
  __m128i *v11; // rdi
  __int64 v12; // rdx
  size_t v13; // rcx
  __int64 v14; // rsi
  size_t v15; // rdx
  size_t v16; // rdx
  __m128i v17; // [rsp+0h] [rbp-50h] BYREF
  __m128i *v18; // [rsp+10h] [rbp-40h]
  size_t n; // [rsp+18h] [rbp-38h]
  _OWORD src[3]; // [rsp+20h] [rbp-30h] BYREF

  v3 = a1 + 2;
  v5 = (__m128i *)v3[-1].m128i_i64[0];
  v18 = (__m128i *)src;
  v17 = _mm_loadu_si128(v3 - 2);
  if ( v5 == v3 )
  {
    src[0] = _mm_loadu_si128(a1 + 2);
  }
  else
  {
    v18 = v5;
    *(_QWORD *)&src[0] = a1[2].m128i_i64[0];
  }
  v6 = a1[1].m128i_u64[1];
  a1[1].m128i_i64[0] = (__int64)v3;
  v7 = a2 + 2;
  a1[1].m128i_i64[1] = 0;
  a1[2].m128i_i8[0] = 0;
  n = v6;
  *a1 = _mm_loadu_si128(a2);
  v8 = (__m128i *)a2[1].m128i_i64[0];
  if ( v8 == &a2[2] )
  {
    v16 = a2[1].m128i_u64[1];
    if ( v16 )
    {
      if ( v16 == 1 )
        a1[2].m128i_i8[0] = a2[2].m128i_i8[0];
      else
        memcpy(v3, &a2[2], v16);
      v16 = a2[1].m128i_u64[1];
    }
    a1[1].m128i_i64[1] = v16;
    a1[2].m128i_i8[v16] = 0;
    v9 = (__m128i *)a2[1].m128i_i64[0];
  }
  else
  {
    a1[1].m128i_i64[0] = (__int64)v8;
    a1[1].m128i_i64[1] = a2[1].m128i_i64[1];
    a1[2].m128i_i64[0] = a2[2].m128i_i64[0];
    v9 = a2 + 2;
    a2[1].m128i_i64[0] = (__int64)v7;
  }
  a2[1].m128i_i64[1] = 0;
  v9->m128i_i8[0] = 0;
  result = (__int64)v18;
  v11 = (__m128i *)a2[1].m128i_i64[0];
  *a2 = _mm_load_si128(&v17);
  if ( (_OWORD *)result == src )
  {
    v15 = n;
    if ( n )
    {
      if ( n == 1 )
      {
        result = LOBYTE(src[0]);
        v11->m128i_i8[0] = src[0];
      }
      else
      {
        result = (__int64)memcpy(v11, src, n);
      }
      v15 = n;
      v11 = (__m128i *)a2[1].m128i_i64[0];
    }
    a2[1].m128i_i64[1] = v15;
    v11->m128i_i8[v15] = 0;
    v11 = v18;
  }
  else
  {
    v12 = *(_QWORD *)&src[0];
    v13 = n;
    if ( v11 == v7 )
    {
      a2[1].m128i_i64[0] = result;
      a2[1].m128i_i64[1] = v13;
      a2[2].m128i_i64[0] = v12;
    }
    else
    {
      v14 = a2[2].m128i_i64[0];
      a2[1].m128i_i64[0] = result;
      a2[1].m128i_i64[1] = v13;
      a2[2].m128i_i64[0] = v12;
      if ( v11 )
      {
        v18 = v11;
        *(_QWORD *)&src[0] = v14;
        goto LABEL_9;
      }
    }
    v18 = (__m128i *)src;
    v11 = (__m128i *)src;
  }
LABEL_9:
  n = 0;
  v11->m128i_i8[0] = 0;
  if ( v18 != (__m128i *)src )
    return j_j___libc_free_0(v18, *(_QWORD *)&src[0] + 1LL);
  return result;
}
