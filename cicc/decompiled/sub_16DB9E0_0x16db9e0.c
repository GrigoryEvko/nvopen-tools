// Function: sub_16DB9E0
// Address: 0x16db9e0
//
__int64 __fastcall sub_16DB9E0(__int64 *a1, __int64 *a2, __int64 a3)
{
  __int64 v5; // r8
  __int64 v6; // rdi
  __int64 v7; // kr08_8
  __int64 v8; // rax
  __m128i *v9; // r8
  __int64 v10; // r15
  __m128i *v11; // r15
  __m128i *v12; // r8
  __m128i *v13; // r8
  __m128i *v14; // r8
  _BYTE *v15; // rsi
  __int64 v16; // rdx
  __m128i *v17; // r8
  __int64 v18; // rdx
  __int64 *v19; // r13
  size_t v20; // rax
  __int64 v21; // rax
  __int64 v22; // rdi
  _BYTE *v24; // rsi
  __int64 v25; // rbx
  __m128i *v26; // rbx
  __int64 v27; // r13
  __m128i *v28; // r13
  __m128i *v29; // rdi
  __int64 v30; // r8
  __m128i *v31; // rdi
  __int64 v32; // r8
  size_t v33; // rdx
  __m128i *v34; // rdi
  __int64 v35; // r8
  size_t v36; // rdx
  size_t v37; // rdx
  __int64 v38; // [rsp+0h] [rbp-120h]
  __m128i *v39; // [rsp+0h] [rbp-120h]
  __int64 v41; // [rsp+10h] [rbp-110h]
  __int64 v42; // [rsp+10h] [rbp-110h]
  __m128i *v43; // [rsp+10h] [rbp-110h]
  __int64 v44; // [rsp+18h] [rbp-108h]
  __int64 v45; // [rsp+18h] [rbp-108h]
  __m128i *v46; // [rsp+18h] [rbp-108h]
  __int64 v47; // [rsp+18h] [rbp-108h]
  __m128i *v48; // [rsp+18h] [rbp-108h]
  __int64 v49; // [rsp+28h] [rbp-F8h]
  __m128i *v50; // [rsp+28h] [rbp-F8h]
  __int64 v51; // [rsp+30h] [rbp-F0h] BYREF
  size_t v52; // [rsp+38h] [rbp-E8h]
  __int64 v53; // [rsp+40h] [rbp-E0h]
  int v54; // [rsp+48h] [rbp-D8h]
  void *v55; // [rsp+50h] [rbp-D0h] BYREF
  size_t v56; // [rsp+58h] [rbp-C8h]
  __m128i v57; // [rsp+60h] [rbp-C0h] BYREF
  void *dest; // [rsp+70h] [rbp-B0h] BYREF
  size_t v59; // [rsp+78h] [rbp-A8h]
  __m128i v60; // [rsp+80h] [rbp-A0h] BYREF
  __m128i *v61; // [rsp+90h] [rbp-90h] BYREF
  size_t n; // [rsp+98h] [rbp-88h]
  _QWORD src[4]; // [rsp+A0h] [rbp-80h] BYREF
  __m128i *v64; // [rsp+C0h] [rbp-60h] BYREF
  __int64 v65; // [rsp+C8h] [rbp-58h]
  size_t v66; // [rsp+D0h] [rbp-50h]
  __m128i v67; // [rsp+D8h] [rbp-48h] BYREF

  v5 = *a2;
  v51 = 0;
  v6 = *a1;
  v52 = 0;
  v53 = 0;
  v54 = 0;
  v41 = v5 / 1000 - *(_QWORD *)(v6 + 11592) / 1000LL;
  v7 = a2[1];
  v8 = *(_QWORD *)(v6 + 11632);
  LOBYTE(v64) = 3;
  v65 = v8;
  v44 = v7 / 1000 - v5 / 1000;
  sub_16DA620(&v61, (__m128i *)"pid", (__m128i *)3);
  v49 = sub_16F4840(&v51, &v61);
  sub_16F2AA0(v49);
  sub_16F2270(v49, &v64);
  v9 = v61;
  if ( v61 )
  {
    if ( (__m128i *)v61->m128i_i64[0] != &v61[1] )
    {
      v50 = v61;
      j_j___libc_free_0(v61->m128i_i64[0], v61[1].m128i_i64[0] + 1);
      v9 = v50;
    }
    j_j___libc_free_0(v9, 32);
  }
  sub_16F2AA0(&v64);
  v65 = a3;
  LOBYTE(v64) = 3;
  sub_16DA620(&v61, (__m128i *)"tid", (__m128i *)3);
  v10 = sub_16F4840(&v51, &v61);
  sub_16F2AA0(v10);
  sub_16F2270(v10, &v64);
  v11 = v61;
  if ( v61 )
  {
    if ( (__m128i *)v61->m128i_i64[0] != &v61[1] )
      j_j___libc_free_0(v61->m128i_i64[0], v61[1].m128i_i64[0] + 1);
    j_j___libc_free_0(v11, 32);
  }
  sub_16F2AA0(&v64);
  n = (size_t)"X";
  LOBYTE(v61) = 4;
  src[0] = 1;
  if ( !(unsigned __int8)sub_16F23B0("X", 1, 0) )
  {
    sub_16F2420(&v55, "X", 1);
    LOBYTE(v64) = 5;
    if ( (unsigned __int8)sub_16F23B0(v55, v56, 0) )
    {
LABEL_60:
      v65 = (__int64)&v67;
      if ( v55 == &v57 )
      {
        v67 = _mm_load_si128(&v57);
      }
      else
      {
        v65 = (__int64)v55;
        v67.m128i_i64[0] = v57.m128i_i64[0];
      }
      v55 = &v57;
      v66 = v56;
      v56 = 0;
      v57.m128i_i8[0] = 0;
      sub_16F2AA0(&v61);
      sub_16F2270(&v61, &v64);
      sub_16F2AA0(&v64);
      if ( v55 != &v57 )
        j_j___libc_free_0(v55, v57.m128i_i64[0] + 1);
      goto LABEL_10;
    }
    sub_16F2420(&dest, v55, v56);
    v34 = (__m128i *)v55;
    if ( dest == &v60 )
    {
      v37 = v59;
      if ( v59 )
      {
        if ( v59 == 1 )
          *(_BYTE *)v55 = v60.m128i_i8[0];
        else
          memcpy(v55, &v60, v59);
        v37 = v59;
        v34 = (__m128i *)v55;
      }
      v56 = v37;
      v34->m128i_i8[v37] = 0;
      v34 = (__m128i *)dest;
      goto LABEL_83;
    }
    if ( v55 == &v57 )
    {
      v55 = dest;
      v56 = v59;
      v57.m128i_i64[0] = v60.m128i_i64[0];
    }
    else
    {
      v35 = v57.m128i_i64[0];
      v55 = dest;
      v56 = v59;
      v57.m128i_i64[0] = v60.m128i_i64[0];
      if ( v34 )
      {
        dest = v34;
        v60.m128i_i64[0] = v35;
        goto LABEL_83;
      }
    }
    dest = &v60;
    v34 = &v60;
LABEL_83:
    v59 = 0;
    v34->m128i_i8[0] = 0;
    if ( dest != &v60 )
      j_j___libc_free_0(dest, v60.m128i_i64[0] + 1);
    goto LABEL_60;
  }
LABEL_10:
  sub_16DA620(&v64, (__m128i *)"ph", (__m128i *)2);
  v38 = sub_16F4840(&v51, &v64);
  sub_16F2AA0(v38);
  sub_16F2270(v38, &v61);
  v12 = v64;
  if ( v64 )
  {
    if ( (__m128i *)v64->m128i_i64[0] != &v64[1] )
    {
      v39 = v64;
      j_j___libc_free_0(v64->m128i_i64[0], v64[1].m128i_i64[0] + 1);
      v12 = v39;
    }
    j_j___libc_free_0(v12, 32);
  }
  sub_16F2AA0(&v61);
  LOBYTE(v64) = 3;
  v65 = v41;
  sub_16DA620(&v61, (__m128i *)"ts", (__m128i *)2);
  v42 = sub_16F4840(&v51, &v61);
  sub_16F2AA0(v42);
  sub_16F2270(v42, &v64);
  v13 = v61;
  if ( v61 )
  {
    if ( (__m128i *)v61->m128i_i64[0] != &v61[1] )
    {
      v43 = v61;
      j_j___libc_free_0(v61->m128i_i64[0], v61[1].m128i_i64[0] + 1);
      v13 = v43;
    }
    j_j___libc_free_0(v13, 32);
  }
  sub_16F2AA0(&v64);
  LOBYTE(v64) = 3;
  v65 = v44;
  sub_16DA620(&v61, (__m128i *)"dur", (__m128i *)3);
  v45 = sub_16F4840(&v51, &v61);
  sub_16F2AA0(v45);
  sub_16F2270(v45, &v64);
  v14 = v61;
  if ( v61 )
  {
    if ( (__m128i *)v61->m128i_i64[0] != &v61[1] )
    {
      v46 = v61;
      j_j___libc_free_0(v61->m128i_i64[0], v61[1].m128i_i64[0] + 1);
      v14 = v46;
    }
    j_j___libc_free_0(v14, 32);
  }
  sub_16F2AA0(&v64);
  v15 = (_BYTE *)a2[2];
  v16 = a2[3];
  dest = &v60;
  sub_16D9890((__int64 *)&dest, v15, (__int64)&v15[v16]);
  LOBYTE(v64) = 5;
  if ( !(unsigned __int8)sub_16F23B0(dest, v59, 0) )
  {
    sub_16F2420(&v61, dest, v59);
    v29 = (__m128i *)dest;
    if ( v61 == (__m128i *)src )
    {
      v33 = n;
      if ( n )
      {
        if ( n == 1 )
          *(_BYTE *)dest = src[0];
        else
          memcpy(dest, src, n);
        v33 = n;
        v29 = (__m128i *)dest;
      }
      v59 = v33;
      v29->m128i_i8[v33] = 0;
      v29 = v61;
      goto LABEL_57;
    }
    if ( dest == &v60 )
    {
      dest = v61;
      v59 = n;
      v60.m128i_i64[0] = src[0];
    }
    else
    {
      v30 = v60.m128i_i64[0];
      dest = v61;
      v59 = n;
      v60.m128i_i64[0] = src[0];
      if ( v29 )
      {
        v61 = v29;
        src[0] = v30;
        goto LABEL_57;
      }
    }
    v61 = (__m128i *)src;
    v29 = (__m128i *)src;
LABEL_57:
    n = 0;
    v29->m128i_i8[0] = 0;
    if ( v61 != (__m128i *)src )
      j_j___libc_free_0(v61, src[0] + 1LL);
  }
  v65 = (__int64)&v67;
  if ( dest == &v60 )
  {
    v67 = _mm_load_si128(&v60);
  }
  else
  {
    v65 = (__int64)dest;
    v67.m128i_i64[0] = v60.m128i_i64[0];
  }
  dest = &v60;
  v66 = v59;
  v59 = 0;
  v60.m128i_i8[0] = 0;
  sub_16DA620(&v61, (__m128i *)"name", (__m128i *)4);
  v47 = sub_16F4840(&v51, &v61);
  sub_16F2AA0(v47);
  sub_16F2270(v47, &v64);
  v17 = v61;
  if ( v61 )
  {
    if ( (__m128i *)v61->m128i_i64[0] != &v61[1] )
    {
      v48 = v61;
      j_j___libc_free_0(v61->m128i_i64[0], v61[1].m128i_i64[0] + 1);
      v17 = v48;
    }
    j_j___libc_free_0(v17, 32);
  }
  sub_16F2AA0(&v64);
  if ( dest != &v60 )
    j_j___libc_free_0(dest, v60.m128i_i64[0] + 1);
  v18 = a2[7];
  if ( v18 )
  {
    v24 = (_BYTE *)a2[6];
    v55 = 0;
    v56 = 0;
    dest = &v60;
    v57.m128i_i64[0] = 0;
    v57.m128i_i32[2] = 0;
    sub_16D9890((__int64 *)&dest, v24, (__int64)&v24[v18]);
    LOBYTE(v64) = 5;
    if ( (unsigned __int8)sub_16F23B0(dest, v59, 0) )
    {
LABEL_38:
      v65 = (__int64)&v67;
      if ( dest == &v60 )
      {
        v67 = _mm_load_si128(&v60);
      }
      else
      {
        v65 = (__int64)dest;
        v67.m128i_i64[0] = v60.m128i_i64[0];
      }
      dest = &v60;
      v66 = v59;
      v59 = 0;
      v60.m128i_i8[0] = 0;
      sub_16DA620(&v61, (__m128i *)"detail", (__m128i *)6);
      v25 = sub_16F4840(&v55, &v61);
      sub_16F2AA0(v25);
      sub_16F2270(v25, &v64);
      v26 = v61;
      if ( v61 )
      {
        if ( (__m128i *)v61->m128i_i64[0] != &v61[1] )
          j_j___libc_free_0(v61->m128i_i64[0], v61[1].m128i_i64[0] + 1);
        j_j___libc_free_0(v26, 32);
      }
      sub_16F2AA0(&v64);
      if ( dest != &v60 )
        j_j___libc_free_0(dest, v60.m128i_i64[0] + 1);
      v55 = (char *)v55 + 1;
      v66 = v56;
      LOBYTE(v64) = 6;
      v67.m128i_i64[0] = v57.m128i_i64[0];
      v65 = 1;
      v67.m128i_i32[2] = v57.m128i_i32[2];
      v56 = 0;
      v57.m128i_i64[0] = 0;
      v57.m128i_i32[2] = 0;
      sub_16DA620(&v61, (__m128i *)"args", (__m128i *)4);
      v27 = sub_16F4840(&v51, &v61);
      sub_16F2AA0(v27);
      sub_16F2270(v27, &v64);
      v28 = v61;
      if ( v61 )
      {
        if ( (__m128i *)v61->m128i_i64[0] != &v61[1] )
          j_j___libc_free_0(v61->m128i_i64[0], v61[1].m128i_i64[0] + 1);
        j_j___libc_free_0(v28, 32);
      }
      sub_16F2AA0(&v64);
      sub_16DB620((__int64)&v55);
      j___libc_free_0(v56);
      goto LABEL_32;
    }
    sub_16F2420(&v61, dest, v59);
    v31 = (__m128i *)dest;
    if ( v61 == (__m128i *)src )
    {
      v36 = n;
      if ( n )
      {
        if ( n == 1 )
          *(_BYTE *)dest = src[0];
        else
          memcpy(dest, src, n);
        v36 = n;
        v31 = (__m128i *)dest;
      }
      v59 = v36;
      v31->m128i_i8[v36] = 0;
      v31 = v61;
      goto LABEL_71;
    }
    if ( dest == &v60 )
    {
      dest = v61;
      v59 = n;
      v60.m128i_i64[0] = src[0];
    }
    else
    {
      v32 = v60.m128i_i64[0];
      dest = v61;
      v59 = n;
      v60.m128i_i64[0] = src[0];
      if ( v31 )
      {
        v61 = v31;
        src[0] = v32;
        goto LABEL_71;
      }
    }
    v61 = (__m128i *)src;
    v31 = (__m128i *)src;
LABEL_71:
    n = 0;
    v31->m128i_i8[0] = 0;
    if ( v61 != (__m128i *)src )
      j_j___libc_free_0(v61, src[0] + 1LL);
    goto LABEL_38;
  }
LABEL_32:
  LOBYTE(v64) = 6;
  v65 = 1;
  v19 = (__int64 *)a1[1];
  v20 = v52;
  v52 = 0;
  ++v51;
  v66 = v20;
  v21 = v53;
  v53 = 0;
  v67.m128i_i64[0] = v21;
  LODWORD(v21) = v54;
  v54 = 0;
  v67.m128i_i32[2] = v21;
  v22 = v19[1];
  if ( v22 == v19[2] )
  {
    sub_16DB810(v19, v19[1], (__int64)&v64);
  }
  else
  {
    if ( v22 )
    {
      sub_16F2270(v22, &v64);
      v22 = v19[1];
    }
    v19[1] = v22 + 40;
  }
  sub_16F2AA0(&v64);
  sub_16DB620((__int64)&v51);
  return j___libc_free_0(v52);
}
