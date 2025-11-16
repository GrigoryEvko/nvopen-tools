// Function: sub_C0D4F0
// Address: 0xc0d4f0
//
__int64 __fastcall sub_C0D4F0(__int64 a1, __int64 *a2)
{
  __int64 v3; // rdx
  unsigned int v4; // ebx
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // r13
  __int64 i; // r12
  __int64 v9; // rdx
  __int64 v10; // rcx
  __m128i *v11; // rax
  __int64 *v12; // rdi
  size_t v13; // rsi
  __int64 v14; // rdx
  __int64 v15; // r8
  __int64 v17; // r13
  __int64 v18; // rdx
  __int64 j; // r15
  __int64 v20; // rdx
  const char *v21; // r12
  const char *v22; // r13
  __m128i *v23; // rax
  __int64 *v24; // rdx
  size_t v25; // rdx
  __int64 v26; // rcx
  __m128i *v27; // rax
  __int64 v28; // rcx
  __m128i *v29; // rax
  __int64 v30; // rcx
  size_t v31; // rdx
  __int64 v32; // rcx
  __m128i *v33; // rax
  __int64 v34; // rcx
  __m128i *v35; // rax
  __int64 *v36; // rdi
  size_t v37; // rsi
  __int64 v38; // rdx
  __int64 v39; // r8
  size_t v40; // rdx
  size_t v41; // rdx
  _QWORD v42[2]; // [rsp+10h] [rbp-F0h] BYREF
  __int64 v43; // [rsp+20h] [rbp-E0h] BYREF
  __int64 *v44; // [rsp+30h] [rbp-D0h] BYREF
  __int64 v45; // [rsp+38h] [rbp-C8h]
  _QWORD v46[2]; // [rsp+40h] [rbp-C0h] BYREF
  __m128i *v47; // [rsp+50h] [rbp-B0h] BYREF
  __int64 v48; // [rsp+58h] [rbp-A8h]
  __m128i v49; // [rsp+60h] [rbp-A0h] BYREF
  __m128i *v50; // [rsp+70h] [rbp-90h] BYREF
  __int64 v51; // [rsp+78h] [rbp-88h]
  __m128i v52; // [rsp+80h] [rbp-80h] BYREF
  __m128i *v53; // [rsp+90h] [rbp-70h] BYREF
  __int64 v54; // [rsp+98h] [rbp-68h]
  __m128i v55; // [rsp+A0h] [rbp-60h] BYREF
  __int64 v56; // [rsp+B0h] [rbp-50h] BYREF
  size_t n; // [rsp+B8h] [rbp-48h]
  _OWORD src[4]; // [rsp+C0h] [rbp-40h] BYREF

  sub_C0D4A0();
  if ( sub_C0D4A0() == v3 )
  {
    i = 0;
    sub_2241130(a2, 0, a2[1], "Unable to find target for this triple (no targets are registered)", 65);
    return i;
  }
  v4 = *(_DWORD *)(a1 + 32);
  v5 = sub_C0D4A0();
  v7 = v6;
  for ( i = v5; v7 != i; i = *(_QWORD *)i )
  {
    if ( (*(unsigned __int8 (__fastcall **)(_QWORD))(i + 8))(v4) )
      break;
  }
  sub_C0D4A0();
  if ( i == v9 )
  {
    sub_8FD6D0((__int64)v42, "No available targets are compatible with triple \"", (_QWORD *)a1);
    if ( v42[1] != 0x3FFFFFFFFFFFFFFFLL )
    {
      v11 = (__m128i *)sub_2241490(v42, "\"", 1, v10);
      v56 = (__int64)src;
      if ( (__m128i *)v11->m128i_i64[0] == &v11[1] )
      {
        src[0] = _mm_loadu_si128(v11 + 1);
      }
      else
      {
        v56 = v11->m128i_i64[0];
        *(_QWORD *)&src[0] = v11[1].m128i_i64[0];
      }
      n = v11->m128i_u64[1];
      v11->m128i_i64[0] = (__int64)v11[1].m128i_i64;
      v11->m128i_i64[1] = 0;
      v11[1].m128i_i8[0] = 0;
      v12 = (__int64 *)*a2;
      if ( (_OWORD *)v56 == src )
      {
        v40 = n;
        if ( n )
        {
          if ( n == 1 )
            *(_BYTE *)v12 = src[0];
          else
            memcpy(v12, src, n);
          v40 = n;
          v12 = (__int64 *)*a2;
        }
        a2[1] = v40;
        *((_BYTE *)v12 + v40) = 0;
        v12 = (__int64 *)v56;
        goto LABEL_14;
      }
      v13 = n;
      v14 = *(_QWORD *)&src[0];
      if ( v12 == a2 + 2 )
      {
        *a2 = v56;
        a2[1] = v13;
        a2[2] = v14;
      }
      else
      {
        v15 = a2[2];
        *a2 = v56;
        a2[1] = v13;
        a2[2] = v14;
        if ( v12 )
        {
          v56 = (__int64)v12;
          *(_QWORD *)&src[0] = v15;
LABEL_14:
          n = 0;
          *(_BYTE *)v12 = 0;
          if ( (_OWORD *)v56 != src )
            j_j___libc_free_0(v56, *(_QWORD *)&src[0] + 1LL);
          if ( (__int64 *)v42[0] != &v43 )
            j_j___libc_free_0(v42[0], v43 + 1);
          return 0;
        }
      }
      v56 = (__int64)src;
      v12 = (__int64 *)src;
      goto LABEL_14;
    }
LABEL_73:
    sub_4262D8((__int64)"basic_string::append");
  }
  sub_C0D4A0();
  v17 = *(_QWORD *)i;
  for ( j = v18; j != v17; v17 = *(_QWORD *)v17 )
  {
    if ( (*(unsigned __int8 (__fastcall **)(_QWORD))(v17 + 8))(v4) )
      break;
  }
  sub_C0D4A0();
  if ( v17 != v20 )
  {
    v21 = *(const char **)(i + 16);
    v22 = *(const char **)(v17 + 16);
    v56 = 31;
    v44 = v46;
    v23 = (__m128i *)sub_22409D0(&v44, &v56, 0);
    v44 = (__int64 *)v23;
    v46[0] = v56;
    *v23 = _mm_load_si128((const __m128i *)&xmmword_3F643D0);
    v24 = v44;
    qmemcpy(&v23[1], "tween targets \"", 15);
    v45 = v56;
    *((_BYTE *)v24 + v56) = 0;
    v25 = strlen(v21);
    if ( v25 > 0x3FFFFFFFFFFFFFFFLL - v45 )
      goto LABEL_73;
    v27 = (__m128i *)sub_2241490(&v44, v21, v25, v26);
    v47 = &v49;
    if ( (__m128i *)v27->m128i_i64[0] == &v27[1] )
    {
      v49 = _mm_loadu_si128(v27 + 1);
    }
    else
    {
      v47 = (__m128i *)v27->m128i_i64[0];
      v49.m128i_i64[0] = v27[1].m128i_i64[0];
    }
    v28 = v27->m128i_i64[1];
    v27[1].m128i_i8[0] = 0;
    v48 = v28;
    v27->m128i_i64[0] = (__int64)v27[1].m128i_i64;
    v27->m128i_i64[1] = 0;
    if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v48) <= 6 )
      goto LABEL_73;
    v29 = (__m128i *)sub_2241490(&v47, "\" and \"", 7, v28);
    v50 = &v52;
    if ( (__m128i *)v29->m128i_i64[0] == &v29[1] )
    {
      v52 = _mm_loadu_si128(v29 + 1);
    }
    else
    {
      v50 = (__m128i *)v29->m128i_i64[0];
      v52.m128i_i64[0] = v29[1].m128i_i64[0];
    }
    v30 = v29->m128i_i64[1];
    v29[1].m128i_i8[0] = 0;
    v51 = v30;
    v29->m128i_i64[0] = (__int64)v29[1].m128i_i64;
    v29->m128i_i64[1] = 0;
    v31 = strlen(v22);
    if ( v31 > 0x3FFFFFFFFFFFFFFFLL - v51 )
      goto LABEL_73;
    v33 = (__m128i *)sub_2241490(&v50, v22, v31, v32);
    v53 = &v55;
    if ( (__m128i *)v33->m128i_i64[0] == &v33[1] )
    {
      v55 = _mm_loadu_si128(v33 + 1);
    }
    else
    {
      v53 = (__m128i *)v33->m128i_i64[0];
      v55.m128i_i64[0] = v33[1].m128i_i64[0];
    }
    v34 = v33->m128i_i64[1];
    v33[1].m128i_i8[0] = 0;
    v54 = v34;
    v33->m128i_i64[0] = (__int64)v33[1].m128i_i64;
    v33->m128i_i64[1] = 0;
    if ( v54 == 0x3FFFFFFFFFFFFFFFLL )
      goto LABEL_73;
    v35 = (__m128i *)sub_2241490(&v53, "\"", 1, v34);
    v56 = (__int64)src;
    if ( (__m128i *)v35->m128i_i64[0] == &v35[1] )
    {
      src[0] = _mm_loadu_si128(v35 + 1);
    }
    else
    {
      v56 = v35->m128i_i64[0];
      *(_QWORD *)&src[0] = v35[1].m128i_i64[0];
    }
    n = v35->m128i_u64[1];
    v35->m128i_i64[0] = (__int64)v35[1].m128i_i64;
    v35->m128i_i64[1] = 0;
    v35[1].m128i_i8[0] = 0;
    v36 = (__int64 *)*a2;
    if ( (_OWORD *)v56 == src )
    {
      v41 = n;
      if ( n )
      {
        if ( n == 1 )
          *(_BYTE *)v36 = src[0];
        else
          memcpy(v36, src, n);
        v41 = n;
        v36 = (__int64 *)*a2;
      }
      a2[1] = v41;
      *((_BYTE *)v36 + v41) = 0;
      v36 = (__int64 *)v56;
      goto LABEL_41;
    }
    v37 = n;
    v38 = *(_QWORD *)&src[0];
    if ( v36 == a2 + 2 )
    {
      *a2 = v56;
      a2[1] = v37;
      a2[2] = v38;
    }
    else
    {
      v39 = a2[2];
      *a2 = v56;
      a2[1] = v37;
      a2[2] = v38;
      if ( v36 )
      {
        v56 = (__int64)v36;
        *(_QWORD *)&src[0] = v39;
LABEL_41:
        n = 0;
        *(_BYTE *)v36 = 0;
        if ( (_OWORD *)v56 != src )
          j_j___libc_free_0(v56, *(_QWORD *)&src[0] + 1LL);
        if ( v53 != &v55 )
          j_j___libc_free_0(v53, v55.m128i_i64[0] + 1);
        if ( v50 != &v52 )
          j_j___libc_free_0(v50, v52.m128i_i64[0] + 1);
        if ( v47 != &v49 )
          j_j___libc_free_0(v47, v49.m128i_i64[0] + 1);
        if ( v44 != v46 )
          j_j___libc_free_0(v44, v46[0] + 1LL);
        return 0;
      }
    }
    v56 = (__int64)src;
    v36 = (__int64 *)src;
    goto LABEL_41;
  }
  return i;
}
