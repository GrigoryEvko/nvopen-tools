// Function: sub_C0DB40
// Address: 0xc0db40
//
__int64 __fastcall sub_C0DB40(void *s1, size_t n, _QWORD *a3, __m128i **a4)
{
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rbx
  __int64 i; // r15
  const void *v8; // r14
  __int64 v9; // rdx
  unsigned int v10; // eax
  __m128i *v12; // rdi
  size_t v13; // rsi
  __m128i *v14; // rcx
  __m128i *v15; // rdx
  __int64 v16; // rcx
  __m128i *v17; // rax
  __m128i *v18; // rdi
  size_t v19; // rsi
  __int64 v20; // rcx
  __int64 v21; // rdx
  size_t v22; // rdx
  size_t v23; // rdx
  __m128i *v26; // [rsp+10h] [rbp-B0h] BYREF
  size_t v27; // [rsp+18h] [rbp-A8h]
  _QWORD v28[2]; // [rsp+20h] [rbp-A0h] BYREF
  _QWORD v29[2]; // [rsp+30h] [rbp-90h] BYREF
  _QWORD v30[2]; // [rsp+40h] [rbp-80h] BYREF
  __int16 v31; // [rsp+50h] [rbp-70h]
  __m128i *p_src; // [rsp+60h] [rbp-60h] BYREF
  size_t na; // [rsp+68h] [rbp-58h]
  __m128i src; // [rsp+70h] [rbp-50h] BYREF
  __int16 v35; // [rsp+80h] [rbp-40h]

  if ( n )
  {
    v4 = sub_C0D4A0();
    v6 = v5;
    for ( i = v4; v6 != i; i = *(_QWORD *)i )
    {
      v8 = *(const void **)(i + 16);
      if ( v8 && n == strlen(*(const char **)(i + 16)) && !memcmp(s1, v8, n) )
        break;
    }
    sub_C0D4A0();
    if ( i != v9 )
    {
      v10 = sub_CC67C0(s1, n);
      if ( v10 )
        sub_CCA5A0(a3, v10, 0);
      return i;
    }
    v30[0] = s1;
    v31 = 1283;
    v29[0] = "invalid target '";
    p_src = (__m128i *)v29;
    src.m128i_i64[0] = (__int64)"'.";
    v30[1] = n;
    v35 = 770;
    sub_CA0F50(&v26, &p_src);
    v12 = *a4;
    if ( v26 == (__m128i *)v28 )
    {
      v23 = v27;
      if ( v27 )
      {
        if ( v27 == 1 )
          v12->m128i_i8[0] = v28[0];
        else
          memcpy(v12, v28, v27);
        v23 = v27;
        v12 = *a4;
      }
      a4[1] = (__m128i *)v23;
      v12->m128i_i8[v23] = 0;
      v12 = v26;
      goto LABEL_18;
    }
    v13 = v27;
    v14 = (__m128i *)v28[0];
    if ( v12 == (__m128i *)(a4 + 2) )
    {
      *a4 = v26;
      a4[1] = (__m128i *)v13;
      a4[2] = v14;
    }
    else
    {
      v15 = a4[2];
      *a4 = v26;
      a4[1] = (__m128i *)v13;
      a4[2] = v14;
      if ( v12 )
      {
        v26 = v12;
        v28[0] = v15;
LABEL_18:
        v27 = 0;
        v12->m128i_i8[0] = 0;
        if ( v26 != (__m128i *)v28 )
          j_j___libc_free_0(v26, v28[0] + 1LL);
        return 0;
      }
    }
    v26 = (__m128i *)v28;
    v12 = (__m128i *)v28;
    goto LABEL_18;
  }
  v27 = 0;
  v26 = (__m128i *)v28;
  LOBYTE(v28[0]) = 0;
  i = sub_C0D4F0((__int64)a3, (__int64 *)&v26);
  if ( !i )
  {
    sub_8FD6D0((__int64)v29, "unable to get target for '", a3);
    if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v29[1]) <= 0x1D )
      sub_4262D8((__int64)"basic_string::append");
    v17 = (__m128i *)sub_2241490(v29, "', see --version and --triple.", 30, v16);
    p_src = &src;
    if ( (__m128i *)v17->m128i_i64[0] == &v17[1] )
    {
      src = _mm_loadu_si128(v17 + 1);
    }
    else
    {
      p_src = (__m128i *)v17->m128i_i64[0];
      src.m128i_i64[0] = v17[1].m128i_i64[0];
    }
    na = v17->m128i_u64[1];
    v17->m128i_i64[0] = (__int64)v17[1].m128i_i64;
    v17->m128i_i64[1] = 0;
    v17[1].m128i_i8[0] = 0;
    v18 = *a4;
    if ( p_src == &src )
    {
      v22 = na;
      if ( na )
      {
        if ( na == 1 )
          v18->m128i_i8[0] = src.m128i_i8[0];
        else
          memcpy(v18, &src, na);
        v22 = na;
        v18 = *a4;
      }
      a4[1] = (__m128i *)v22;
      v18->m128i_i8[v22] = 0;
      v18 = p_src;
      goto LABEL_28;
    }
    v19 = na;
    v20 = src.m128i_i64[0];
    if ( v18 == (__m128i *)(a4 + 2) )
    {
      *a4 = p_src;
      a4[1] = (__m128i *)v19;
      a4[2] = (__m128i *)v20;
    }
    else
    {
      v21 = (__int64)a4[2];
      *a4 = p_src;
      a4[1] = (__m128i *)v19;
      a4[2] = (__m128i *)v20;
      if ( v18 )
      {
        p_src = v18;
        src.m128i_i64[0] = v21;
LABEL_28:
        na = 0;
        v18->m128i_i8[0] = 0;
        if ( p_src != &src )
          j_j___libc_free_0(p_src, src.m128i_i64[0] + 1);
        if ( (_QWORD *)v29[0] != v30 )
          j_j___libc_free_0(v29[0], v30[0] + 1LL);
        goto LABEL_11;
      }
    }
    p_src = &src;
    v18 = &src;
    goto LABEL_28;
  }
LABEL_11:
  if ( v26 != (__m128i *)v28 )
    j_j___libc_free_0(v26, v28[0] + 1LL);
  return i;
}
