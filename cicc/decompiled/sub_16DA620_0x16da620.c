// Function: sub_16DA620
// Address: 0x16da620
//
__int64 __fastcall sub_16DA620(__m128i **a1, __m128i *a2, __m128i *a3)
{
  __int64 result; // rax
  __int64 v6; // rax
  __m128i *v7; // r12
  __m128i *v8; // rdi
  __m128i *v9; // rax
  __int64 v10; // rax
  __int64 v11; // rsi
  __m128i *v12; // r14
  __int64 v13; // r15
  __m128i *v14; // rdi
  __m128i *v15; // rdi
  __int64 v16; // rcx
  size_t v17; // rsi
  __int64 v18; // rdx
  size_t v19; // rdx
  __int64 v20; // [rsp+8h] [rbp-78h]
  __int64 v21; // [rsp+8h] [rbp-78h]
  __m128i *v22; // [rsp+10h] [rbp-70h] BYREF
  __int64 v23; // [rsp+18h] [rbp-68h]
  __m128i v24; // [rsp+20h] [rbp-60h] BYREF
  __m128i *v25; // [rsp+30h] [rbp-50h] BYREF
  size_t n; // [rsp+38h] [rbp-48h]
  _QWORD src[8]; // [rsp+40h] [rbp-40h] BYREF

  a1[1] = a2;
  a1[2] = a3;
  *a1 = 0;
  result = sub_16F23B0(a2, a3, 0);
  if ( (_BYTE)result )
    return result;
  sub_16F2420(&v22, a2, a3);
  v6 = sub_22077B0(32);
  v7 = (__m128i *)v6;
  if ( v6 )
  {
    v8 = v22;
    v9 = (__m128i *)(v6 + 16);
    v7->m128i_i64[0] = (__int64)v9;
    if ( v8 == &v24 )
    {
      v8 = v9;
      v7[1] = _mm_load_si128(&v24);
    }
    else
    {
      v10 = v24.m128i_i64[0];
      v7->m128i_i64[0] = (__int64)v8;
      v7[1].m128i_i64[0] = v10;
    }
    v11 = v23;
    v22 = &v24;
    v23 = 0;
    v7->m128i_i64[1] = v11;
    v24.m128i_i8[0] = 0;
  }
  else
  {
    v8 = (__m128i *)MEMORY[0];
    v11 = MEMORY[8];
  }
  if ( !(unsigned __int8)sub_16F23B0(v8, v11, 0) )
  {
    sub_16F2420(&v25, v7->m128i_i64[0], v7->m128i_i64[1]);
    v15 = (__m128i *)v7->m128i_i64[0];
    if ( v25 == (__m128i *)src )
    {
      v19 = n;
      if ( n )
      {
        if ( n == 1 )
          v15->m128i_i8[0] = src[0];
        else
          memcpy(v15, src, n);
        v19 = n;
        v15 = (__m128i *)v7->m128i_i64[0];
      }
      v7->m128i_i64[1] = v19;
      v15->m128i_i8[v19] = 0;
      v15 = v25;
      goto LABEL_19;
    }
    v16 = src[0];
    v17 = n;
    if ( v15 == &v7[1] )
    {
      v7->m128i_i64[0] = (__int64)v25;
      v7->m128i_i64[1] = v17;
      v7[1].m128i_i64[0] = v16;
    }
    else
    {
      v18 = v7[1].m128i_i64[0];
      v7->m128i_i64[0] = (__int64)v25;
      v7->m128i_i64[1] = v17;
      v7[1].m128i_i64[0] = v16;
      if ( v15 )
      {
        v25 = v15;
        src[0] = v18;
        goto LABEL_19;
      }
    }
    v25 = (__m128i *)src;
    v15 = (__m128i *)src;
LABEL_19:
    n = 0;
    v15->m128i_i8[0] = 0;
    if ( v25 != (__m128i *)src )
      j_j___libc_free_0(v25, src[0] + 1LL);
  }
  v12 = *a1;
  result = v7->m128i_i64[0];
  *a1 = v7;
  v13 = v7->m128i_i64[1];
  if ( v12 )
  {
    if ( (__m128i *)v12->m128i_i64[0] != &v12[1] )
    {
      v20 = result;
      j_j___libc_free_0(v12->m128i_i64[0], v12[1].m128i_i64[0] + 1);
      result = v20;
    }
    v21 = result;
    j_j___libc_free_0(v12, 32);
    result = v21;
  }
  v14 = v22;
  a1[1] = (__m128i *)result;
  a1[2] = (__m128i *)v13;
  if ( v14 != &v24 )
    return j_j___libc_free_0(v14, v24.m128i_i64[0] + 1);
  return result;
}
