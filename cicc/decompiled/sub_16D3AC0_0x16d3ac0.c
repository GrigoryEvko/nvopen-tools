// Function: sub_16D3AC0
// Address: 0x16d3ac0
//
__int64 __fastcall sub_16D3AC0(__m128i *a1, __int64 *a2)
{
  __int64 v3; // rdx
  unsigned int v4; // ebx
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // r12
  __int64 i; // r14
  __int64 v9; // rdx
  __int64 v11; // r15
  __int64 v12; // rdx
  __int64 j; // r12
  __int64 v14; // rdx
  const char *v15; // rbx
  const char *v16; // r12
  __m128i *v17; // rax
  __int64 *v18; // rdx
  size_t v19; // rax
  __int64 v20; // rcx
  __m128i *v21; // rax
  __int64 v22; // rcx
  __m128i *v23; // rax
  __int64 v24; // rcx
  size_t v25; // rdx
  __int64 v26; // rcx
  __m128i *v27; // rax
  __int64 v28; // rcx
  __m128i *v29; // rax
  __m128i *v30; // rdi
  size_t v31; // rcx
  __int64 v32; // rdx
  __int64 v33; // rsi
  size_t v34; // rdx
  __int64 *v35; // [rsp+20h] [rbp-F0h] BYREF
  __int64 v36; // [rsp+28h] [rbp-E8h]
  _QWORD v37[2]; // [rsp+30h] [rbp-E0h] BYREF
  __m128i *v38; // [rsp+40h] [rbp-D0h] BYREF
  __int64 v39; // [rsp+48h] [rbp-C8h]
  __m128i v40; // [rsp+50h] [rbp-C0h] BYREF
  __m128i *v41; // [rsp+60h] [rbp-B0h] BYREF
  __int64 v42; // [rsp+68h] [rbp-A8h]
  __m128i v43; // [rsp+70h] [rbp-A0h] BYREF
  __m128i *v44; // [rsp+80h] [rbp-90h] BYREF
  __int64 v45; // [rsp+88h] [rbp-88h]
  __m128i v46; // [rsp+90h] [rbp-80h] BYREF
  __int64 v47; // [rsp+A0h] [rbp-70h] BYREF
  size_t n; // [rsp+A8h] [rbp-68h]
  __m128i v49; // [rsp+B0h] [rbp-60h] BYREF
  unsigned int v50; // [rsp+C0h] [rbp-50h]

  sub_16D3AB0();
  if ( sub_16D3AB0() == v3 )
  {
    i = 0;
    sub_2241130(a2, 0, a2[1], "Unable to find target for this triple (no targets are registered)", 65);
    return i;
  }
  v44 = a1;
  v46.m128i_i16[0] = 260;
  sub_16E1010(&v47);
  v4 = v50;
  if ( (__m128i *)v47 != &v49 )
    j_j___libc_free_0(v47, v49.m128i_i64[0] + 1);
  v5 = sub_16D3AB0();
  v7 = v6;
  for ( i = v5; v7 != i; i = *(_QWORD *)i )
  {
    if ( (*(unsigned __int8 (__fastcall **)(_QWORD))(i + 8))(v4) )
      break;
  }
  sub_16D3AB0();
  if ( i == v9 )
  {
    i = 0;
    sub_2241130(a2, 0, a2[1], "No available targets are compatible with this triple.", 53);
    return i;
  }
  sub_16D3AB0();
  v11 = *(_QWORD *)i;
  for ( j = v12; j != v11; v11 = *(_QWORD *)v11 )
  {
    if ( (*(unsigned __int8 (__fastcall **)(_QWORD))(v11 + 8))(v4) )
      break;
  }
  sub_16D3AB0();
  if ( v11 != v14 )
  {
    v15 = *(const char **)(i + 16);
    v16 = *(const char **)(v11 + 16);
    v35 = v37;
    v47 = 31;
    v17 = (__m128i *)sub_22409D0(&v35, &v47, 0);
    v35 = (__int64 *)v17;
    v37[0] = v47;
    *v17 = _mm_load_si128((const __m128i *)&xmmword_3F643D0);
    v18 = v35;
    qmemcpy(&v17[1], "tween targets \"", 15);
    v36 = v47;
    *((_BYTE *)v18 + v47) = 0;
    v19 = strlen(v15);
    if ( v19 > 0x3FFFFFFFFFFFFFFFLL - v36 )
      goto LABEL_56;
    v21 = (__m128i *)sub_2241490(&v35, v15, v19, v20);
    v38 = &v40;
    if ( (__m128i *)v21->m128i_i64[0] == &v21[1] )
    {
      v40 = _mm_loadu_si128(v21 + 1);
    }
    else
    {
      v38 = (__m128i *)v21->m128i_i64[0];
      v40.m128i_i64[0] = v21[1].m128i_i64[0];
    }
    v22 = v21->m128i_i64[1];
    v21[1].m128i_i8[0] = 0;
    v39 = v22;
    v21->m128i_i64[0] = (__int64)v21[1].m128i_i64;
    v21->m128i_i64[1] = 0;
    if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v39) <= 6 )
      goto LABEL_56;
    v23 = (__m128i *)sub_2241490(&v38, "\" and \"", 7, v22);
    v41 = &v43;
    if ( (__m128i *)v23->m128i_i64[0] == &v23[1] )
    {
      v43 = _mm_loadu_si128(v23 + 1);
    }
    else
    {
      v41 = (__m128i *)v23->m128i_i64[0];
      v43.m128i_i64[0] = v23[1].m128i_i64[0];
    }
    v24 = v23->m128i_i64[1];
    v23[1].m128i_i8[0] = 0;
    v42 = v24;
    v23->m128i_i64[0] = (__int64)v23[1].m128i_i64;
    v23->m128i_i64[1] = 0;
    v25 = strlen(v16);
    if ( v25 > 0x3FFFFFFFFFFFFFFFLL - v42 )
      goto LABEL_56;
    v27 = (__m128i *)sub_2241490(&v41, v16, v25, v26);
    v44 = &v46;
    if ( (__m128i *)v27->m128i_i64[0] == &v27[1] )
    {
      v46 = _mm_loadu_si128(v27 + 1);
    }
    else
    {
      v44 = (__m128i *)v27->m128i_i64[0];
      v46.m128i_i64[0] = v27[1].m128i_i64[0];
    }
    v28 = v27->m128i_i64[1];
    v45 = v28;
    v27->m128i_i64[0] = (__int64)v27[1].m128i_i64;
    v27->m128i_i64[1] = 0;
    v27[1].m128i_i8[0] = 0;
    if ( v45 == 0x3FFFFFFFFFFFFFFFLL )
LABEL_56:
      sub_4262D8((__int64)"basic_string::append");
    v29 = (__m128i *)sub_2241490(&v44, "\"", 1, v28);
    v47 = (__int64)&v49;
    if ( (__m128i *)v29->m128i_i64[0] == &v29[1] )
    {
      v49 = _mm_loadu_si128(v29 + 1);
    }
    else
    {
      v47 = v29->m128i_i64[0];
      v49.m128i_i64[0] = v29[1].m128i_i64[0];
    }
    n = v29->m128i_u64[1];
    v29->m128i_i64[0] = (__int64)v29[1].m128i_i64;
    v29->m128i_i64[1] = 0;
    v29[1].m128i_i8[0] = 0;
    v30 = (__m128i *)*a2;
    if ( (__m128i *)v47 == &v49 )
    {
      v34 = n;
      if ( n )
      {
        if ( n == 1 )
          v30->m128i_i8[0] = v49.m128i_i8[0];
        else
          memcpy(v30, &v49, n);
        v34 = n;
        v30 = (__m128i *)*a2;
      }
      a2[1] = v34;
      v30->m128i_i8[v34] = 0;
      v30 = (__m128i *)v47;
      goto LABEL_32;
    }
    v31 = n;
    v32 = v49.m128i_i64[0];
    if ( v30 == (__m128i *)(a2 + 2) )
    {
      *a2 = v47;
      a2[1] = v31;
      a2[2] = v32;
    }
    else
    {
      v33 = a2[2];
      *a2 = v47;
      a2[1] = v31;
      a2[2] = v32;
      if ( v30 )
      {
        v47 = (__int64)v30;
        v49.m128i_i64[0] = v33;
LABEL_32:
        n = 0;
        v30->m128i_i8[0] = 0;
        if ( (__m128i *)v47 != &v49 )
          j_j___libc_free_0(v47, v49.m128i_i64[0] + 1);
        if ( v44 != &v46 )
          j_j___libc_free_0(v44, v46.m128i_i64[0] + 1);
        if ( v41 != &v43 )
          j_j___libc_free_0(v41, v43.m128i_i64[0] + 1);
        if ( v38 != &v40 )
          j_j___libc_free_0(v38, v40.m128i_i64[0] + 1);
        if ( v35 != v37 )
          j_j___libc_free_0(v35, v37[0] + 1LL);
        return 0;
      }
    }
    v47 = (__int64)&v49;
    v30 = &v49;
    goto LABEL_32;
  }
  return i;
}
