// Function: sub_16FECA0
// Address: 0x16feca0
//
_QWORD *__fastcall sub_16FECA0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned __int64 v5; // rax
  char *v6; // rcx
  void *v7; // rdx
  void *v8; // rax
  __int64 v9; // rax
  char *v10; // rcx
  unsigned __int64 v11; // rdx
  unsigned __int64 v12; // rax
  char *v13; // r15
  const void *v14; // r12
  char *v15; // rax
  unsigned __int64 v16; // rax
  unsigned __int64 v17; // r13
  _QWORD *v18; // r14
  _QWORD *v19; // r13
  int v20; // eax
  char *v21; // rbx
  const void *v22; // rdi
  char *v23; // rcx
  const void *v24; // rsi
  int v25; // eax
  _QWORD *v26; // rax
  __int64 v27; // rax
  _QWORD *v28; // rdx
  _QWORD *v29; // r8
  __int64 v30; // rdi
  _QWORD *v31; // rdi
  _QWORD *result; // rax
  size_t v33; // rcx
  const void *v34; // rsi
  int v35; // eax
  unsigned int v36; // edi
  int v37; // eax
  _QWORD *v38; // [rsp+0h] [rbp-C0h]
  _QWORD *v39; // [rsp+0h] [rbp-C0h]
  char *v40; // [rsp+8h] [rbp-B8h]
  _QWORD *v41; // [rsp+8h] [rbp-B8h]
  size_t v42; // [rsp+8h] [rbp-B8h]
  size_t v43; // [rsp+8h] [rbp-B8h]
  size_t v44; // [rsp+8h] [rbp-B8h]
  char *v45; // [rsp+10h] [rbp-B0h]
  char *v46; // [rsp+18h] [rbp-A8h]
  _QWORD *v47; // [rsp+20h] [rbp-A0h]
  _QWORD *v49; // [rsp+28h] [rbp-98h]
  void *s2[2]; // [rsp+30h] [rbp-90h] BYREF
  char *v51; // [rsp+40h] [rbp-80h] BYREF
  char *v52; // [rsp+48h] [rbp-78h]
  char v53[8]; // [rsp+50h] [rbp-70h] BYREF
  __m128i v54; // [rsp+58h] [rbp-68h] BYREF
  _QWORD *v55; // [rsp+68h] [rbp-58h]
  _QWORD v56[9]; // [rsp+78h] [rbp-48h] BYREF

  sub_16FC210((__int64)v53, (unsigned __int64 **)a1, a3, a4, a5);
  *(__m128i *)s2 = _mm_loadu_si128(&v54);
  v5 = sub_16D23E0(s2, " \t", 2, 0);
  v6 = 0;
  v7 = (void *)v5;
  v8 = s2[1];
  if ( v7 <= s2[1] )
  {
    v6 = (char *)((char *)s2[1] - (char *)v7);
    v8 = v7;
  }
  v52 = v6;
  v51 = (char *)s2[0] + (unsigned __int64)v8;
  v9 = sub_16D24E0(&v51, (unsigned __int8 *)" \t", 2, 0);
  v10 = 0;
  v11 = v9;
  v12 = (unsigned __int64)v52;
  if ( v11 < (unsigned __int64)v52 )
  {
    v10 = &v52[-v11];
    v12 = v11;
  }
  s2[1] = v10;
  s2[0] = &v51[v12];
  v13 = (char *)sub_16D23E0(s2, " \t", 2, 0);
  if ( v13 > s2[1] )
  {
    v13 = (char *)s2[1];
    v14 = s2[0];
    v15 = 0;
  }
  else
  {
    v14 = s2[0];
    v15 = (char *)((char *)s2[1] - (char *)v13);
  }
  v52 = v15;
  v51 = &v13[(_QWORD)v14];
  v16 = sub_16D24E0(&v51, (unsigned __int8 *)" \t", 2, 0);
  v17 = (unsigned __int64)v52;
  v46 = 0;
  if ( v16 < (unsigned __int64)v52 )
  {
    v46 = &v52[-v16];
    v17 = v16;
  }
  v45 = &v51[v17];
  v18 = *(_QWORD **)(a1 + 136);
  v19 = (_QWORD *)(a1 + 128);
  v47 = (_QWORD *)(a1 + 128);
  if ( !v18 )
  {
    v19 = (_QWORD *)(a1 + 128);
    goto LABEL_27;
  }
  do
  {
    while ( 1 )
    {
      v21 = (char *)v18[5];
      v22 = (const void *)v18[4];
      if ( v21 > v13 )
        break;
      if ( v21 )
      {
        v20 = memcmp(v22, v14, v18[5]);
        if ( v20 )
          goto LABEL_19;
      }
      if ( v21 == v13 )
        goto LABEL_20;
LABEL_14:
      if ( v21 >= v13 )
        goto LABEL_20;
LABEL_15:
      v18 = (_QWORD *)v18[3];
      if ( !v18 )
        goto LABEL_21;
    }
    if ( !v13 )
      goto LABEL_20;
    v20 = memcmp(v22, v14, (size_t)v13);
    if ( !v20 )
      goto LABEL_14;
LABEL_19:
    if ( v20 < 0 )
      goto LABEL_15;
LABEL_20:
    v19 = v18;
    v18 = (_QWORD *)v18[2];
  }
  while ( v18 );
LABEL_21:
  if ( v47 == v19 )
    goto LABEL_27;
  v23 = (char *)v19[5];
  v24 = (const void *)v19[4];
  if ( v23 < v13 )
  {
    if ( !v23 )
      goto LABEL_32;
    v42 = v19[5];
    v25 = memcmp(v14, v24, v42);
    v23 = (char *)v42;
    if ( !v25 )
    {
LABEL_26:
      if ( v23 <= v13 )
        goto LABEL_32;
      goto LABEL_27;
    }
LABEL_38:
    if ( v25 >= 0 )
      goto LABEL_32;
LABEL_27:
    v41 = v19;
    v26 = (_QWORD *)sub_22077B0(64);
    v26[4] = v14;
    v19 = v26;
    v26[5] = v13;
    v26[6] = 0;
    v26[7] = 0;
    v27 = sub_16FE940((_QWORD *)(a1 + 120), v41, (__int64)(v26 + 4));
    v29 = v28;
    if ( !v28 )
    {
      v49 = (_QWORD *)v27;
      j_j___libc_free_0(v19, 64);
      v19 = v49;
      goto LABEL_32;
    }
    if ( v47 == v28 || v27 )
    {
      v30 = 1;
LABEL_31:
      sub_220F040(v30, v19, v29, v47);
      ++*(_QWORD *)(a1 + 160);
      goto LABEL_32;
    }
    v33 = v28[5];
    v34 = (const void *)v28[4];
    if ( v33 < (unsigned __int64)v13 )
    {
      v30 = 0;
      if ( !v33 )
        goto LABEL_31;
      v39 = v28;
      v44 = v28[5];
      v37 = memcmp(v14, v34, v33);
      v33 = v44;
      v29 = v39;
      v36 = v37;
      if ( !v37 )
        goto LABEL_45;
    }
    else if ( !v13
           || (v38 = v28, v43 = v28[5], v35 = memcmp(v14, v34, (size_t)v13), v33 = v43, v29 = v38, (v36 = v35) == 0) )
    {
      v30 = 0;
      if ( (char *)v33 == v13 )
        goto LABEL_31;
LABEL_45:
      v30 = v33 > (unsigned __int64)v13;
      goto LABEL_31;
    }
    v30 = v36 >> 31;
    goto LABEL_31;
  }
  if ( v13 )
  {
    v40 = (char *)v19[5];
    v25 = memcmp(v14, v24, (size_t)v13);
    v23 = v40;
    if ( v25 )
      goto LABEL_38;
  }
  if ( v23 != v13 )
    goto LABEL_26;
LABEL_32:
  v31 = v55;
  v19[6] = v45;
  v19[7] = v46;
  result = v56;
  if ( v31 != v56 )
    return (_QWORD *)j_j___libc_free_0(v31, v56[0] + 1LL);
  return result;
}
