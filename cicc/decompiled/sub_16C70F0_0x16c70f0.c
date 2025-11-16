// Function: sub_16C70F0
// Address: 0x16c70f0
//
__int64 __fastcall sub_16C70F0(__int64 a1, int a2, _QWORD *a3)
{
  size_t v4; // r12
  char *v5; // rdi
  int v6; // r12d
  int v7; // edi
  unsigned int v8; // r12d
  _BYTE *v10; // r8
  char *v11; // rax
  char *v12; // rdi
  char *v13; // rax
  __int64 v14; // rdi
  const char *v15; // r15
  __int64 v16; // rcx
  __m128i *v17; // rax
  size_t v18; // rdx
  __int64 v19; // rcx
  __m128i *v20; // rax
  size_t v21; // rdx
  __int64 v22; // rax
  _OWORD *v23; // rdi
  _BYTE *src; // [rsp+8h] [rbp-B8h]
  char *file; // [rsp+10h] [rbp-B0h] BYREF
  size_t v26; // [rsp+18h] [rbp-A8h]
  _QWORD v27[2]; // [rsp+20h] [rbp-A0h] BYREF
  _QWORD *v28; // [rsp+30h] [rbp-90h] BYREF
  __int64 v29; // [rsp+38h] [rbp-88h]
  _QWORD v30[2]; // [rsp+40h] [rbp-80h] BYREF
  __m128i *v31; // [rsp+50h] [rbp-70h] BYREF
  __int64 v32; // [rsp+58h] [rbp-68h]
  __m128i v33; // [rsp+60h] [rbp-60h] BYREF
  char *v34; // [rsp+70h] [rbp-50h] BYREF
  size_t n; // [rsp+78h] [rbp-48h]
  _OWORD v36[4]; // [rsp+80h] [rbp-40h] BYREF

  v4 = *(_QWORD *)(a1 + 8);
  file = (char *)v27;
  v26 = 0;
  LOBYTE(v27[0]) = 0;
  if ( !v4 )
  {
    sub_2241130(&file, 0, 0, "/dev/null", 9);
    goto LABEL_3;
  }
  v10 = *(_BYTE **)a1;
  if ( !*(_QWORD *)a1 )
  {
    LOBYTE(v36[0]) = 0;
    v21 = 0;
    v12 = (char *)v27;
    v34 = (char *)v36;
LABEL_37:
    v26 = v21;
    v12[v21] = 0;
    v13 = v34;
    goto LABEL_18;
  }
  v31 = (__m128i *)v4;
  v34 = (char *)v36;
  if ( v4 > 0xF )
  {
    src = v10;
    v22 = sub_22409D0(&v34, &v31, 0);
    v10 = src;
    v34 = (char *)v22;
    v23 = (_OWORD *)v22;
    *(_QWORD *)&v36[0] = v31;
  }
  else
  {
    if ( v4 == 1 )
    {
      LOBYTE(v36[0]) = *v10;
      v11 = (char *)v36;
      goto LABEL_14;
    }
    v23 = v36;
  }
  memcpy(v23, v10, v4);
  v4 = (size_t)v31;
  v11 = v34;
LABEL_14:
  n = v4;
  v11[v4] = 0;
  v12 = file;
  v13 = file;
  if ( v34 == (char *)v36 )
  {
    v21 = n;
    if ( n )
    {
      if ( n == 1 )
        *file = v36[0];
      else
        memcpy(file, v36, n);
      v21 = n;
      v12 = file;
    }
    goto LABEL_37;
  }
  if ( file == (char *)v27 )
  {
    file = v34;
    v26 = n;
    v27[0] = *(_QWORD *)&v36[0];
  }
  else
  {
    v14 = v27[0];
    file = v34;
    v26 = n;
    v27[0] = *(_QWORD *)&v36[0];
    if ( v13 )
    {
      v34 = v13;
      *(_QWORD *)&v36[0] = v14;
      goto LABEL_18;
    }
  }
  v34 = (char *)v36;
  v13 = (char *)v36;
LABEL_18:
  n = 0;
  *v13 = 0;
  if ( v34 == (char *)v36 )
  {
LABEL_3:
    v5 = file;
    if ( a2 )
      goto LABEL_4;
LABEL_20:
    v6 = open(v5, 0, 438);
    if ( v6 != -1 )
      goto LABEL_5;
    v15 = "input";
LABEL_22:
    v29 = 0;
    v28 = v30;
    LOBYTE(v30[0]) = 0;
    sub_2240E30(&v28, v26 + 18);
    if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v29) <= 0x11 )
      goto LABEL_49;
    sub_2241490(&v28, "Cannot open file '", 18, 0x3FFFFFFFFFFFFFFFLL);
    sub_2241490(&v28, file, v26, v16);
    if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v29) <= 5 )
      goto LABEL_49;
    v17 = (__m128i *)sub_2241490(&v28, "' for ", 6, 0x3FFFFFFFFFFFFFFFLL - v29);
    v31 = &v33;
    if ( (__m128i *)v17->m128i_i64[0] == &v17[1] )
    {
      v33 = _mm_loadu_si128(v17 + 1);
    }
    else
    {
      v31 = (__m128i *)v17->m128i_i64[0];
      v33.m128i_i64[0] = v17[1].m128i_i64[0];
    }
    v32 = v17->m128i_i64[1];
    v17->m128i_i64[0] = (__int64)v17[1].m128i_i64;
    v17->m128i_i64[1] = 0;
    v17[1].m128i_i8[0] = 0;
    v18 = strlen(v15);
    if ( v18 > 0x3FFFFFFFFFFFFFFFLL - v32 )
LABEL_49:
      sub_4262D8((__int64)"basic_string::append");
    v20 = (__m128i *)sub_2241490(&v31, v15, v18, v19);
    v34 = (char *)v36;
    if ( (__m128i *)v20->m128i_i64[0] == &v20[1] )
    {
      v36[0] = _mm_loadu_si128(v20 + 1);
    }
    else
    {
      v34 = (char *)v20->m128i_i64[0];
      *(_QWORD *)&v36[0] = v20[1].m128i_i64[0];
    }
    n = v20->m128i_u64[1];
    v20->m128i_i64[0] = (__int64)v20[1].m128i_i64;
    v20->m128i_i64[1] = 0;
    v20[1].m128i_i8[0] = 0;
    sub_16C6DC0(a3, (__int64)&v34, -1);
    if ( v34 != (char *)v36 )
      j_j___libc_free_0(v34, *(_QWORD *)&v36[0] + 1LL);
    if ( v31 != &v33 )
      j_j___libc_free_0(v31, v33.m128i_i64[0] + 1);
    if ( v28 != v30 )
      j_j___libc_free_0(v28, v30[0] + 1LL);
    goto LABEL_35;
  }
  j_j___libc_free_0(v34, *(_QWORD *)&v36[0] + 1LL);
  v5 = file;
  if ( !a2 )
    goto LABEL_20;
LABEL_4:
  v6 = open(v5, 65, 438);
  if ( v6 == -1 )
  {
    v15 = "output";
    goto LABEL_22;
  }
LABEL_5:
  if ( dup2(v6, a2) != -1 )
  {
    v7 = v6;
    v8 = 0;
    close(v7);
    goto LABEL_7;
  }
  v34 = (char *)v36;
  strcpy((char *)v36, "Cannot dup2");
  n = 11;
  sub_16C6DC0(a3, (__int64)&v34, -1);
  if ( v34 != (char *)v36 )
    j_j___libc_free_0(v34, *(_QWORD *)&v36[0] + 1LL);
  close(v6);
LABEL_35:
  v8 = 1;
LABEL_7:
  if ( file != (char *)v27 )
    j_j___libc_free_0(file, v27[0] + 1LL);
  return v8;
}
