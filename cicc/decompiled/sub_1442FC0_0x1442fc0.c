// Function: sub_1442FC0
// Address: 0x1442fc0
//
__m128i *__fastcall sub_1442FC0(__m128i *a1, __int64 *a2)
{
  __int64 v3; // rdi
  __int64 v4; // rdx
  __int64 v5; // rdi
  __int64 v6; // rdi
  __int64 v7; // rdx
  __int64 v8; // rdi
  size_t v9; // rcx
  void *v10; // r8
  size_t v11; // r15
  size_t *v12; // rax
  __int64 v13; // rcx
  __m128i *v14; // rax
  __int64 v15; // rcx
  _BYTE *v17; // rax
  __int64 v18; // rdx
  _BYTE *v19; // rdi
  size_t *v20; // rax
  size_t v21; // rdi
  _BYTE *v22; // rax
  __int64 v23; // rdx
  _BYTE *v24; // rdi
  size_t *v25; // rax
  size_t v26; // rdi
  size_t v27; // rdx
  __int64 v28; // rax
  _QWORD *v29; // rdi
  size_t v30; // rdx
  void *src; // [rsp+8h] [rbp-B8h]
  size_t v32; // [rsp+18h] [rbp-A8h] BYREF
  void *v33; // [rsp+20h] [rbp-A0h] BYREF
  size_t v34; // [rsp+28h] [rbp-98h]
  _QWORD v35[2]; // [rsp+30h] [rbp-90h] BYREF
  void *dest; // [rsp+40h] [rbp-80h] BYREF
  size_t n; // [rsp+48h] [rbp-78h]
  _QWORD v38[2]; // [rsp+50h] [rbp-70h] BYREF
  size_t *v39; // [rsp+60h] [rbp-60h] BYREF
  size_t v40; // [rsp+68h] [rbp-58h]
  size_t v41; // [rsp+70h] [rbp-50h] BYREF
  __int64 v42; // [rsp+78h] [rbp-48h]
  int v43; // [rsp+80h] [rbp-40h]
  void **p_dest; // [rsp+88h] [rbp-38h]

  v3 = *a2;
  dest = v38;
  v33 = v35;
  LOBYTE(v35[0]) = 0;
  v34 = 0;
  n = 0;
  LOBYTE(v38[0]) = 0;
  sub_1649960(v3 & 0xFFFFFFFFFFFFFFF8LL);
  if ( !v4 )
  {
    v5 = *a2;
    p_dest = &dest;
    v43 = 1;
    v39 = (size_t *)&unk_49EFBE0;
    v42 = 0;
    v41 = 0;
    v40 = 0;
    sub_15537D0(v5 & 0xFFFFFFFFFFFFFFF8LL, &v39, 0);
    sub_16E7BC0(&v39);
    goto LABEL_3;
  }
  v17 = (_BYTE *)sub_1649960(*a2 & 0xFFFFFFFFFFFFFFF8LL);
  if ( !v17 )
  {
    LOBYTE(v41) = 0;
    v19 = dest;
    v27 = 0;
    v39 = &v41;
LABEL_37:
    n = v27;
    v19[v27] = 0;
    v20 = v39;
    goto LABEL_26;
  }
  v39 = &v41;
  sub_14429B0((__int64 *)&v39, v17, (__int64)&v17[v18]);
  v19 = dest;
  v20 = (size_t *)dest;
  if ( v39 == &v41 )
  {
    v27 = v40;
    if ( v40 )
    {
      if ( v40 == 1 )
        *(_BYTE *)dest = v41;
      else
        memcpy(dest, &v41, v40);
      v27 = v40;
      v19 = dest;
    }
    goto LABEL_37;
  }
  if ( dest == v38 )
  {
    dest = v39;
    n = v40;
    v38[0] = v41;
  }
  else
  {
    v21 = v38[0];
    dest = v39;
    n = v40;
    v38[0] = v41;
    if ( v20 )
    {
      v39 = v20;
      v41 = v21;
      goto LABEL_26;
    }
  }
  v39 = &v41;
  v20 = &v41;
LABEL_26:
  v40 = 0;
  *(_BYTE *)v20 = 0;
  if ( v39 == &v41 )
  {
LABEL_3:
    v6 = a2[4];
    if ( v6 )
      goto LABEL_4;
LABEL_28:
    sub_2241130(&v33, 0, v34, "<Function Return>", 17);
    goto LABEL_6;
  }
  j_j___libc_free_0(v39, v41 + 1);
  v6 = a2[4];
  if ( !v6 )
    goto LABEL_28;
LABEL_4:
  sub_1649960(v6);
  if ( !v7 )
  {
    v8 = a2[4];
    v43 = 1;
    v39 = (size_t *)&unk_49EFBE0;
    v42 = 0;
    v41 = 0;
    v40 = 0;
    p_dest = &v33;
    sub_15537D0(v8, &v39, 0);
    sub_16E7BC0(&v39);
    goto LABEL_6;
  }
  v22 = (_BYTE *)sub_1649960(a2[4]);
  if ( !v22 )
  {
    v39 = &v41;
    v24 = v33;
    v30 = 0;
    LOBYTE(v41) = 0;
LABEL_44:
    v34 = v30;
    v24[v30] = 0;
    v25 = v39;
    goto LABEL_34;
  }
  v39 = &v41;
  sub_14429B0((__int64 *)&v39, v22, (__int64)&v22[v23]);
  v24 = v33;
  v25 = (size_t *)v33;
  if ( v39 == &v41 )
  {
    v30 = v40;
    if ( v40 )
    {
      if ( v40 == 1 )
        *(_BYTE *)v33 = v41;
      else
        memcpy(v33, &v41, v40);
      v30 = v40;
      v24 = v33;
    }
    goto LABEL_44;
  }
  v9 = v41;
  if ( v33 == v35 )
  {
    v33 = v39;
    v34 = v40;
    v35[0] = v41;
  }
  else
  {
    v26 = v35[0];
    v33 = v39;
    v34 = v40;
    v35[0] = v41;
    if ( v25 )
    {
      v39 = v25;
      v41 = v26;
      goto LABEL_34;
    }
  }
  v39 = &v41;
  v25 = &v41;
LABEL_34:
  v40 = 0;
  *(_BYTE *)v25 = 0;
  if ( v39 != &v41 )
    j_j___libc_free_0(v39, v41 + 1);
LABEL_6:
  v10 = dest;
  v11 = n;
  v39 = &v41;
  if ( (char *)dest + n && !dest )
    sub_426248((__int64)"basic_string::_M_construct null not valid");
  v32 = n;
  if ( n > 0xF )
  {
    src = dest;
    v28 = sub_22409D0(&v39, &v32, 0);
    v10 = src;
    v39 = (size_t *)v28;
    v29 = (_QWORD *)v28;
    v41 = v32;
  }
  else
  {
    if ( n == 1 )
    {
      LOBYTE(v41) = *(_BYTE *)dest;
      v12 = &v41;
      goto LABEL_11;
    }
    if ( !n )
    {
      v12 = &v41;
      goto LABEL_11;
    }
    v29 = &v41;
  }
  memcpy(v29, v10, v11);
  v11 = v32;
  v12 = v39;
LABEL_11:
  v40 = v11;
  *((_BYTE *)v12 + v11) = 0;
  if ( 0x3FFFFFFFFFFFFFFFLL - v40 <= 3 )
    sub_4262D8((__int64)"basic_string::append");
  sub_2241490(&v39, " => ", 4, v9);
  v14 = (__m128i *)sub_2241490(&v39, v33, v34, v13);
  a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
  if ( (__m128i *)v14->m128i_i64[0] == &v14[1] )
  {
    a1[1] = _mm_loadu_si128(v14 + 1);
  }
  else
  {
    a1->m128i_i64[0] = v14->m128i_i64[0];
    a1[1].m128i_i64[0] = v14[1].m128i_i64[0];
  }
  v15 = v14->m128i_i64[1];
  v14->m128i_i64[0] = (__int64)v14[1].m128i_i64;
  v14->m128i_i64[1] = 0;
  a1->m128i_i64[1] = v15;
  v14[1].m128i_i8[0] = 0;
  if ( v39 != &v41 )
    j_j___libc_free_0(v39, v41 + 1);
  if ( dest != v38 )
    j_j___libc_free_0(dest, v38[0] + 1LL);
  if ( v33 != v35 )
    j_j___libc_free_0(v33, v35[0] + 1LL);
  return a1;
}
