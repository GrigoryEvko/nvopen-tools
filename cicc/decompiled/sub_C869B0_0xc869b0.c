// Function: sub_C869B0
// Address: 0xc869b0
//
__int64 __fastcall sub_C869B0(
        int fd2,
        _QWORD *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        _BYTE *a7,
        __int64 a8,
        unsigned __int8 a9)
{
  unsigned int v9; // r12d
  char *v12; // rdi
  int v13; // r15d
  char *v15; // rdi
  __int64 v16; // rsi
  const char *v17; // r15
  __int64 v18; // rcx
  __m128i *v19; // rax
  __int64 v20; // rcx
  size_t v21; // rdx
  __int64 v22; // rcx
  __m128i *v23; // rax
  size_t v24; // rdx
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
  _OWORD src[4]; // [rsp+80h] [rbp-40h] BYREF

  v9 = a9;
  if ( !a9 )
    return v9;
  v26 = 0;
  file = (char *)v27;
  LOBYTE(v27[0]) = 0;
  if ( !a8 )
  {
    sub_2241130(&file, 0, 0, "/dev/null", 9);
    goto LABEL_4;
  }
  v34 = (char *)src;
  sub_C865D0((__int64 *)&v34, a7, (__int64)&a7[a8]);
  v15 = file;
  if ( v34 == (char *)src )
  {
    v24 = n;
    if ( n )
    {
      if ( n == 1 )
        *file = src[0];
      else
        memcpy(file, src, n);
      v24 = n;
      v15 = file;
    }
    v26 = v24;
    v15[v24] = 0;
    v15 = v34;
  }
  else
  {
    if ( file == (char *)v27 )
    {
      file = v34;
      v26 = n;
      v27[0] = *(_QWORD *)&src[0];
    }
    else
    {
      v16 = v27[0];
      file = v34;
      v26 = n;
      v27[0] = *(_QWORD *)&src[0];
      if ( v15 )
      {
        v34 = v15;
        *(_QWORD *)&src[0] = v16;
        goto LABEL_15;
      }
    }
    v34 = (char *)src;
    v15 = (char *)src;
  }
LABEL_15:
  n = 0;
  *v15 = 0;
  if ( v34 != (char *)src )
  {
    j_j___libc_free_0(v34, *(_QWORD *)&src[0] + 1LL);
    v12 = file;
    if ( !fd2 )
      goto LABEL_5;
    goto LABEL_17;
  }
LABEL_4:
  v12 = file;
  if ( !fd2 )
  {
LABEL_5:
    v13 = open(v12, 0, 438);
    if ( v13 != -1 )
      goto LABEL_6;
    v17 = "input";
LABEL_19:
    LOBYTE(v30[0]) = 0;
    v28 = v30;
    v29 = 0;
    sub_2240E30(&v28, v26 + 18);
    if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v29) <= 0x11 )
      goto LABEL_46;
    sub_2241490(&v28, "Cannot open file '", 18, 0x3FFFFFFFFFFFFFFFLL);
    sub_2241490(&v28, file, v26, v18);
    if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v29) <= 5 )
      goto LABEL_46;
    v19 = (__m128i *)sub_2241490(&v28, "' for ", 6, 0x3FFFFFFFFFFFFFFFLL - v29);
    v31 = &v33;
    if ( (__m128i *)v19->m128i_i64[0] == &v19[1] )
    {
      v33 = _mm_loadu_si128(v19 + 1);
    }
    else
    {
      v31 = (__m128i *)v19->m128i_i64[0];
      v33.m128i_i64[0] = v19[1].m128i_i64[0];
    }
    v20 = v19->m128i_i64[1];
    v19[1].m128i_i8[0] = 0;
    v32 = v20;
    v19->m128i_i64[0] = (__int64)v19[1].m128i_i64;
    v19->m128i_i64[1] = 0;
    v21 = strlen(v17);
    if ( v21 > 0x3FFFFFFFFFFFFFFFLL - v32 )
LABEL_46:
      sub_4262D8((__int64)"basic_string::append");
    v23 = (__m128i *)sub_2241490(&v31, v17, v21, v22);
    v34 = (char *)src;
    if ( (__m128i *)v23->m128i_i64[0] == &v23[1] )
    {
      src[0] = _mm_loadu_si128(v23 + 1);
    }
    else
    {
      v34 = (char *)v23->m128i_i64[0];
      *(_QWORD *)&src[0] = v23[1].m128i_i64[0];
    }
    n = v23->m128i_u64[1];
    v23->m128i_i64[0] = (__int64)v23[1].m128i_i64;
    v23->m128i_i64[1] = 0;
    v23[1].m128i_i8[0] = 0;
    sub_C86680(a2, (__int64)&v34, -1);
    if ( v34 != (char *)src )
      j_j___libc_free_0(v34, *(_QWORD *)&src[0] + 1LL);
    if ( v31 != &v33 )
      j_j___libc_free_0(v31, v33.m128i_i64[0] + 1);
    if ( v28 != v30 )
      j_j___libc_free_0(v28, v30[0] + 1LL);
    goto LABEL_8;
  }
LABEL_17:
  v13 = open(v12, 65, 438);
  if ( v13 == -1 )
  {
    v17 = "output";
    goto LABEL_19;
  }
LABEL_6:
  if ( dup2(v13, fd2) == -1 )
  {
    v34 = (char *)src;
    strcpy((char *)src, "Cannot dup2");
    n = 11;
    sub_C86680(a2, (__int64)&v34, -1);
    if ( v34 != (char *)src )
      j_j___libc_free_0(v34, *(_QWORD *)&src[0] + 1LL);
    close(v13);
  }
  else
  {
    v9 = 0;
    close(v13);
  }
LABEL_8:
  if ( file != (char *)v27 )
    j_j___libc_free_0(file, v27[0] + 1LL);
  return v9;
}
