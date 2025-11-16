// Function: sub_22DAE20
// Address: 0x22dae20
//
__m128i *__fastcall sub_22DAE20(__m128i *a1, __int64 *a2)
{
  __int64 v4; // rdi
  __int64 v5; // rdx
  __int64 v6; // rdi
  __int64 v7; // rdx
  void *v8; // r8
  unsigned __int64 v9; // r14
  unsigned __int64 *v10; // rax
  __m128i *v11; // rax
  __int64 v12; // rcx
  char *v14; // rax
  __int64 v15; // rdx
  unsigned __int64 *v16; // rdi
  unsigned __int64 v17; // rsi
  char *v18; // rax
  __int64 v19; // rdx
  unsigned __int64 *v20; // rdi
  unsigned __int64 v21; // rsi
  __int64 v22; // rax
  unsigned __int64 *v23; // rdi
  size_t v24; // rdx
  size_t v25; // rdx
  void *src; // [rsp+0h] [rbp-D0h]
  size_t v27; // [rsp+18h] [rbp-B8h] BYREF
  void *v28; // [rsp+20h] [rbp-B0h] BYREF
  size_t v29; // [rsp+28h] [rbp-A8h]
  _QWORD v30[2]; // [rsp+30h] [rbp-A0h] BYREF
  void *dest; // [rsp+40h] [rbp-90h] BYREF
  size_t n; // [rsp+48h] [rbp-88h]
  _QWORD v33[2]; // [rsp+50h] [rbp-80h] BYREF
  unsigned __int64 *v34; // [rsp+60h] [rbp-70h] BYREF
  size_t v35; // [rsp+68h] [rbp-68h]
  unsigned __int64 v36; // [rsp+70h] [rbp-60h] BYREF
  __int64 v37; // [rsp+78h] [rbp-58h]
  __int64 v38; // [rsp+80h] [rbp-50h]
  __int64 v39; // [rsp+88h] [rbp-48h]
  void **p_dest; // [rsp+90h] [rbp-40h]

  v4 = *a2;
  v28 = v30;
  LOBYTE(v30[0]) = 0;
  v29 = 0;
  dest = v33;
  n = 0;
  LOBYTE(v33[0]) = 0;
  sub_BD5D20(v4 & 0xFFFFFFFFFFFFFFF8LL);
  if ( !v5 )
  {
    p_dest = &dest;
    v39 = 0x100000000LL;
    v35 = 0;
    v36 = 0;
    v34 = (unsigned __int64 *)&unk_49DD210;
    v37 = 0;
    v38 = 0;
    sub_CB5980((__int64)&v34, 0, 0, 0);
    sub_A5BF40((unsigned __int8 *)(*a2 & 0xFFFFFFFFFFFFFFF8LL), (__int64)&v34, 0, 0);
    v34 = (unsigned __int64 *)&unk_49DD210;
    sub_CB5840((__int64)&v34);
    goto LABEL_3;
  }
  v14 = (char *)sub_BD5D20(*a2 & 0xFFFFFFFFFFFFFFF8LL);
  v34 = &v36;
  sub_22DA510((__int64 *)&v34, v14, (__int64)&v14[v15]);
  v16 = (unsigned __int64 *)dest;
  if ( v34 == &v36 )
  {
    v24 = v35;
    if ( v35 )
    {
      if ( v35 == 1 )
        *(_BYTE *)dest = v36;
      else
        memcpy(dest, &v36, v35);
      v24 = v35;
      v16 = (unsigned __int64 *)dest;
    }
    n = v24;
    *((_BYTE *)v16 + v24) = 0;
    v16 = v34;
  }
  else
  {
    if ( dest == v33 )
    {
      dest = v34;
      n = v35;
      v33[0] = v36;
    }
    else
    {
      v17 = v33[0];
      dest = v34;
      n = v35;
      v33[0] = v36;
      if ( v16 )
      {
        v34 = v16;
        v36 = v17;
        goto LABEL_25;
      }
    }
    v34 = &v36;
    v16 = &v36;
  }
LABEL_25:
  v35 = 0;
  *(_BYTE *)v16 = 0;
  if ( v34 == &v36 )
  {
LABEL_3:
    v6 = a2[4];
    if ( v6 )
      goto LABEL_4;
LABEL_27:
    sub_2241130((unsigned __int64 *)&v28, 0, v29, "<Function Return>", 0x11u);
    goto LABEL_6;
  }
  j_j___libc_free_0((unsigned __int64)v34);
  v6 = a2[4];
  if ( !v6 )
    goto LABEL_27;
LABEL_4:
  sub_BD5D20(v6);
  if ( !v7 )
  {
    p_dest = &v28;
    v39 = 0x100000000LL;
    v35 = 0;
    v36 = 0;
    v34 = (unsigned __int64 *)&unk_49DD210;
    v37 = 0;
    v38 = 0;
    sub_CB5980((__int64)&v34, 0, 0, 0);
    sub_A5BF40((unsigned __int8 *)a2[4], (__int64)&v34, 0, 0);
    v34 = (unsigned __int64 *)&unk_49DD210;
    sub_CB5840((__int64)&v34);
    goto LABEL_6;
  }
  v18 = (char *)sub_BD5D20(a2[4]);
  v34 = &v36;
  sub_22DA510((__int64 *)&v34, v18, (__int64)&v18[v19]);
  v20 = (unsigned __int64 *)v28;
  if ( v34 == &v36 )
  {
    v25 = v35;
    if ( v35 )
    {
      if ( v35 == 1 )
        *(_BYTE *)v28 = v36;
      else
        memcpy(v28, &v36, v35);
      v25 = v35;
      v20 = (unsigned __int64 *)v28;
    }
    v29 = v25;
    *((_BYTE *)v20 + v25) = 0;
    v20 = v34;
    goto LABEL_32;
  }
  if ( v28 == v30 )
  {
    v28 = v34;
    v29 = v35;
    v30[0] = v36;
  }
  else
  {
    v21 = v30[0];
    v28 = v34;
    v29 = v35;
    v30[0] = v36;
    if ( v20 )
    {
      v34 = v20;
      v36 = v21;
      goto LABEL_32;
    }
  }
  v34 = &v36;
  v20 = &v36;
LABEL_32:
  v35 = 0;
  *(_BYTE *)v20 = 0;
  if ( v34 != &v36 )
    j_j___libc_free_0((unsigned __int64)v34);
LABEL_6:
  v8 = dest;
  v9 = n;
  v34 = &v36;
  if ( (char *)dest + n && !dest )
    sub_426248((__int64)"basic_string::_M_construct null not valid");
  v27 = n;
  if ( n > 0xF )
  {
    src = dest;
    v22 = sub_22409D0((__int64)&v34, &v27, 0);
    v8 = src;
    v34 = (unsigned __int64 *)v22;
    v23 = (unsigned __int64 *)v22;
    v36 = v27;
  }
  else
  {
    if ( n == 1 )
    {
      LOBYTE(v36) = *(_BYTE *)dest;
      v10 = &v36;
      goto LABEL_11;
    }
    if ( !n )
    {
      v10 = &v36;
      goto LABEL_11;
    }
    v23 = &v36;
  }
  memcpy(v23, v8, v9);
  v9 = v27;
  v10 = v34;
LABEL_11:
  v35 = v9;
  *((_BYTE *)v10 + v9) = 0;
  if ( 0x3FFFFFFFFFFFFFFFLL - v35 <= 3 )
    sub_4262D8((__int64)"basic_string::append");
  sub_2241490((unsigned __int64 *)&v34, " => ", 4u);
  v11 = (__m128i *)sub_2241490((unsigned __int64 *)&v34, (char *)v28, v29);
  a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
  if ( (__m128i *)v11->m128i_i64[0] == &v11[1] )
  {
    a1[1] = _mm_loadu_si128(v11 + 1);
  }
  else
  {
    a1->m128i_i64[0] = v11->m128i_i64[0];
    a1[1].m128i_i64[0] = v11[1].m128i_i64[0];
  }
  v12 = v11->m128i_i64[1];
  v11->m128i_i64[0] = (__int64)v11[1].m128i_i64;
  v11->m128i_i64[1] = 0;
  a1->m128i_i64[1] = v12;
  v11[1].m128i_i8[0] = 0;
  if ( v34 != &v36 )
    j_j___libc_free_0((unsigned __int64)v34);
  if ( dest != v33 )
    j_j___libc_free_0((unsigned __int64)dest);
  if ( v28 != v30 )
    j_j___libc_free_0((unsigned __int64)v28);
  return a1;
}
