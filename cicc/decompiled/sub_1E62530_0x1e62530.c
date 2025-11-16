// Function: sub_1E62530
// Address: 0x1e62530
//
__m128i *__fastcall sub_1E62530(__m128i *a1, __int64 *a2)
{
  __int64 v3; // rdi
  __int64 v4; // rdx
  __int64 v5; // rdi
  __int64 v6; // rdi
  __int64 v7; // rdx
  __int64 v8; // rdi
  void *v9; // r8
  size_t v10; // r15
  size_t *v11; // rax
  __m128i *v12; // rax
  __int64 v13; // rcx
  char *v15; // rax
  __int64 v16; // rdx
  _BYTE *v17; // rdi
  size_t *v18; // rax
  size_t v19; // rdi
  char *v20; // rax
  __int64 v21; // rdx
  _BYTE *v22; // rdi
  size_t *v23; // rax
  size_t v24; // rdi
  size_t v25; // rdx
  __int64 v26; // rax
  _QWORD *v27; // rdi
  size_t v28; // rdx
  void *src; // [rsp+8h] [rbp-B8h]
  size_t v30; // [rsp+18h] [rbp-A8h] BYREF
  void *v31; // [rsp+20h] [rbp-A0h] BYREF
  size_t v32; // [rsp+28h] [rbp-98h]
  _QWORD v33[2]; // [rsp+30h] [rbp-90h] BYREF
  void *dest; // [rsp+40h] [rbp-80h] BYREF
  size_t n; // [rsp+48h] [rbp-78h]
  _QWORD v36[2]; // [rsp+50h] [rbp-70h] BYREF
  size_t *v37; // [rsp+60h] [rbp-60h] BYREF
  size_t v38; // [rsp+68h] [rbp-58h]
  size_t v39; // [rsp+70h] [rbp-50h] BYREF
  __int64 v40; // [rsp+78h] [rbp-48h]
  int v41; // [rsp+80h] [rbp-40h]
  void **p_dest; // [rsp+88h] [rbp-38h]

  v3 = *a2;
  dest = v36;
  v31 = v33;
  LOBYTE(v33[0]) = 0;
  v32 = 0;
  n = 0;
  LOBYTE(v36[0]) = 0;
  sub_1DD6290(v3 & 0xFFFFFFFFFFFFFFF8LL);
  if ( !v4 )
  {
    v5 = *a2;
    v41 = 1;
    p_dest = &dest;
    v40 = 0;
    v37 = (size_t *)&unk_49EFBE0;
    v39 = 0;
    v38 = 0;
    sub_1DD64C0(v5 & 0xFFFFFFFFFFFFFFF8LL, (__int64)&v37);
    sub_16E7BC0((__int64 *)&v37);
    goto LABEL_3;
  }
  v15 = (char *)sub_1DD6290(*a2 & 0xFFFFFFFFFFFFFFF8LL);
  if ( !v15 )
  {
    LOBYTE(v39) = 0;
    v17 = dest;
    v25 = 0;
    v37 = &v39;
LABEL_37:
    n = v25;
    v17[v25] = 0;
    v18 = v37;
    goto LABEL_26;
  }
  v37 = &v39;
  sub_1E61FB0((__int64 *)&v37, v15, (__int64)&v15[v16]);
  v17 = dest;
  v18 = (size_t *)dest;
  if ( v37 == &v39 )
  {
    v25 = v38;
    if ( v38 )
    {
      if ( v38 == 1 )
        *(_BYTE *)dest = v39;
      else
        memcpy(dest, &v39, v38);
      v25 = v38;
      v17 = dest;
    }
    goto LABEL_37;
  }
  if ( dest == v36 )
  {
    dest = v37;
    n = v38;
    v36[0] = v39;
  }
  else
  {
    v19 = v36[0];
    dest = v37;
    n = v38;
    v36[0] = v39;
    if ( v18 )
    {
      v37 = v18;
      v39 = v19;
      goto LABEL_26;
    }
  }
  v37 = &v39;
  v18 = &v39;
LABEL_26:
  v38 = 0;
  *(_BYTE *)v18 = 0;
  if ( v37 == &v39 )
  {
LABEL_3:
    v6 = a2[4];
    if ( v6 )
      goto LABEL_4;
LABEL_28:
    sub_2241130(&v31, 0, v32, "<Function Return>", 17);
    goto LABEL_6;
  }
  j_j___libc_free_0(v37, v39 + 1);
  v6 = a2[4];
  if ( !v6 )
    goto LABEL_28;
LABEL_4:
  sub_1DD6290(v6);
  if ( !v7 )
  {
    v8 = a2[4];
    v41 = 1;
    v40 = 0;
    v37 = (size_t *)&unk_49EFBE0;
    v39 = 0;
    v38 = 0;
    p_dest = &v31;
    sub_1DD64C0(v8, (__int64)&v37);
    sub_16E7BC0((__int64 *)&v37);
    goto LABEL_6;
  }
  v20 = (char *)sub_1DD6290(a2[4]);
  if ( !v20 )
  {
    v37 = &v39;
    v22 = v31;
    v28 = 0;
    LOBYTE(v39) = 0;
LABEL_44:
    v32 = v28;
    v22[v28] = 0;
    v23 = v37;
    goto LABEL_34;
  }
  v37 = &v39;
  sub_1E61FB0((__int64 *)&v37, v20, (__int64)&v20[v21]);
  v22 = v31;
  v23 = (size_t *)v31;
  if ( v37 == &v39 )
  {
    v28 = v38;
    if ( v38 )
    {
      if ( v38 == 1 )
        *(_BYTE *)v31 = v39;
      else
        memcpy(v31, &v39, v38);
      v28 = v38;
      v22 = v31;
    }
    goto LABEL_44;
  }
  if ( v31 == v33 )
  {
    v31 = v37;
    v32 = v38;
    v33[0] = v39;
  }
  else
  {
    v24 = v33[0];
    v31 = v37;
    v32 = v38;
    v33[0] = v39;
    if ( v23 )
    {
      v37 = v23;
      v39 = v24;
      goto LABEL_34;
    }
  }
  v37 = &v39;
  v23 = &v39;
LABEL_34:
  v38 = 0;
  *(_BYTE *)v23 = 0;
  if ( v37 != &v39 )
    j_j___libc_free_0(v37, v39 + 1);
LABEL_6:
  v9 = dest;
  v10 = n;
  v37 = &v39;
  if ( (char *)dest + n && !dest )
    sub_426248((__int64)"basic_string::_M_construct null not valid");
  v30 = n;
  if ( n > 0xF )
  {
    src = dest;
    v26 = sub_22409D0(&v37, &v30, 0);
    v9 = src;
    v37 = (size_t *)v26;
    v27 = (_QWORD *)v26;
    v39 = v30;
  }
  else
  {
    if ( n == 1 )
    {
      LOBYTE(v39) = *(_BYTE *)dest;
      v11 = &v39;
      goto LABEL_11;
    }
    if ( !n )
    {
      v11 = &v39;
      goto LABEL_11;
    }
    v27 = &v39;
  }
  memcpy(v27, v9, v10);
  v10 = v30;
  v11 = v37;
LABEL_11:
  v38 = v10;
  *((_BYTE *)v11 + v10) = 0;
  if ( 0x3FFFFFFFFFFFFFFFLL - v38 <= 3 )
    sub_4262D8((__int64)"basic_string::append");
  sub_2241490(&v37, " => ", 4);
  v12 = (__m128i *)sub_2241490(&v37, (const char *)v31, v32);
  a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
  if ( (__m128i *)v12->m128i_i64[0] == &v12[1] )
  {
    a1[1] = _mm_loadu_si128(v12 + 1);
  }
  else
  {
    a1->m128i_i64[0] = v12->m128i_i64[0];
    a1[1].m128i_i64[0] = v12[1].m128i_i64[0];
  }
  v13 = v12->m128i_i64[1];
  v12->m128i_i64[0] = (__int64)v12[1].m128i_i64;
  v12->m128i_i64[1] = 0;
  a1->m128i_i64[1] = v13;
  v12[1].m128i_i8[0] = 0;
  if ( v37 != &v39 )
    j_j___libc_free_0(v37, v39 + 1);
  if ( dest != v36 )
    j_j___libc_free_0(dest, v36[0] + 1LL);
  if ( v31 != v33 )
    j_j___libc_free_0(v31, v33[0] + 1LL);
  return a1;
}
