// Function: sub_16C8300
// Address: 0x16c8300
//
__int64 __fastcall sub_16C8300(
        __pid_t *a1,
        void *a2,
        size_t a3,
        _QWORD *a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __m128i **a9)
{
  unsigned int v12; // r14d
  __int64 v13; // rcx
  void *v14; // r9
  size_t v15; // r13
  size_t *v16; // rax
  size_t v17; // rdx
  size_t *v18; // rsi
  __m128i *v19; // rax
  size_t v20; // rcx
  __m128i *v21; // rdx
  unsigned __int64 v22; // rax
  unsigned __int64 v23; // rdi
  char *v24; // rcx
  __m128i *v25; // rax
  size_t v26; // rcx
  __m128i *v27; // rdx
  size_t v28; // rdx
  __m128i *v29; // rdi
  __int64 v30; // rsi
  __int64 v31; // rcx
  __int64 v33; // rdx
  char *const *v34; // r12
  char *const *v35; // r13
  __pid_t v36; // eax
  unsigned __int64 *v37; // rbx
  unsigned __int64 *v38; // r12
  unsigned __int64 v39; // rdi
  unsigned __int64 *v40; // rbx
  unsigned __int64 *v41; // r12
  unsigned __int64 v42; // rdi
  void *v43; // r8
  char **v44; // r15
  char **v45; // rbx
  const char *v46; // rdi
  int *v47; // rax
  size_t v48; // rdx
  __int64 v49; // rax
  _QWORD *v50; // rdi
  char **v51; // rax
  __m128i si128; // xmm0
  char **v53; // rdi
  char **v54; // rax
  __int64 v55; // [rsp+8h] [rbp-168h]
  __int64 v56; // [rsp+10h] [rbp-160h]
  void *v57; // [rsp+38h] [rbp-138h]
  void *v58; // [rsp+38h] [rbp-138h]
  void *v59; // [rsp+40h] [rbp-130h] BYREF
  size_t v60; // [rsp+48h] [rbp-128h]
  _QWORD *v61; // [rsp+50h] [rbp-120h] BYREF
  __int64 v62; // [rsp+58h] [rbp-118h]
  _QWORD v63[2]; // [rsp+60h] [rbp-110h] BYREF
  size_t *p_p_src; // [rsp+70h] [rbp-100h] BYREF
  size_t v65; // [rsp+78h] [rbp-F8h]
  _QWORD v66[2]; // [rsp+80h] [rbp-F0h] BYREF
  size_t v67; // [rsp+90h] [rbp-E0h] BYREF
  __int64 v68; // [rsp+98h] [rbp-D8h]
  __m128i v69; // [rsp+A0h] [rbp-D0h] BYREF
  char **argv[2]; // [rsp+B0h] [rbp-C0h] BYREF
  _QWORD dest[2]; // [rsp+C0h] [rbp-B0h] BYREF
  size_t p_src; // [rsp+D0h] [rbp-A0h] BYREF
  size_t n; // [rsp+D8h] [rbp-98h]
  __m128i src; // [rsp+E0h] [rbp-90h] BYREF
  _BYTE v75[32]; // [rsp+F0h] [rbp-80h] BYREF
  unsigned __int64 *v76; // [rsp+110h] [rbp-60h]
  __int64 v77; // [rsp+118h] [rbp-58h]
  _QWORD v78[10]; // [rsp+120h] [rbp-50h] BYREF

  v59 = a2;
  v60 = a3;
  src.m128i_i16[0] = 261;
  p_src = (size_t)&v59;
  if ( (unsigned int)sub_16C51A0((__int64)&p_src, 0) )
  {
    v12 = 0;
    if ( !a9 )
      return v12;
    p_src = 16;
    argv[0] = (char **)dest;
    argv[0] = (char **)sub_22409D0(argv, &p_src, 0);
    dest[0] = p_src;
    *(__m128i *)argv[0] = _mm_load_si128((const __m128i *)&xmmword_42AF0D0);
    argv[1] = (char **)p_src;
    *((_BYTE *)argv[0] + p_src) = 0;
    v14 = v59;
    if ( !v59 )
    {
      v17 = 0;
      p_p_src = v66;
      v18 = v66;
      v65 = 0;
      LOBYTE(v66[0]) = 0;
      v61 = v63;
      strcpy((char *)v63, "Executable \"");
      v62 = 12;
      goto LABEL_13;
    }
    v15 = v60;
    p_p_src = v66;
    p_src = v60;
    if ( v60 > 0xF )
    {
      v57 = v59;
      v49 = sub_22409D0(&p_p_src, &p_src, 0);
      v14 = v57;
      p_p_src = (size_t *)v49;
      v50 = (_QWORD *)v49;
      v66[0] = p_src;
    }
    else
    {
      if ( v60 == 1 )
      {
        LOBYTE(v66[0]) = *(_BYTE *)v59;
        v16 = v66;
LABEL_7:
        v65 = v15;
        v13 = 0x6261747563657845LL;
        *((_BYTE *)v16 + v15) = 0;
        v17 = v65;
        v61 = v63;
        v18 = p_p_src;
        strcpy((char *)v63, "Executable \"");
        v62 = 12;
        if ( v65 + 12 > 0xF && p_p_src != v66 && v66[0] >= v65 + 12 )
        {
          v19 = (__m128i *)sub_2241130(&p_p_src, 0, 0, v63, 12);
          v67 = (size_t)&v69;
          v20 = v19->m128i_i64[0];
          v21 = v19 + 1;
          if ( (__m128i *)v19->m128i_i64[0] == &v19[1] )
          {
LABEL_11:
            v69 = _mm_loadu_si128(v19 + 1);
            goto LABEL_15;
          }
LABEL_14:
          v67 = v20;
          v69.m128i_i64[0] = v19[1].m128i_i64[0];
LABEL_15:
          v68 = v19->m128i_i64[1];
          v19->m128i_i64[0] = (__int64)v21;
          v19->m128i_i64[1] = 0;
          v19[1].m128i_i8[0] = 0;
          v22 = 15;
          v23 = 15;
          if ( (__m128i *)v67 != &v69 )
            v23 = v69.m128i_i64[0];
          v24 = (char *)argv[1] + v68;
          if ( (char **)((char *)argv[1] + v68) <= (char **)v23 )
            goto LABEL_21;
          if ( argv[0] != dest )
            v22 = dest[0];
          if ( (unsigned __int64)v24 <= v22 )
          {
            v25 = (__m128i *)sub_2241130(argv, 0, 0, v67, v68);
            p_src = (size_t)&src;
            v26 = v25->m128i_i64[0];
            v27 = v25 + 1;
            if ( (__m128i *)v25->m128i_i64[0] != &v25[1] )
              goto LABEL_22;
          }
          else
          {
LABEL_21:
            v25 = (__m128i *)sub_2241490(&v67, argv[0], argv[1], v24);
            p_src = (size_t)&src;
            v26 = v25->m128i_i64[0];
            v27 = v25 + 1;
            if ( (__m128i *)v25->m128i_i64[0] != &v25[1] )
            {
LABEL_22:
              p_src = v26;
              src.m128i_i64[0] = v25[1].m128i_i64[0];
              goto LABEL_23;
            }
          }
          src = _mm_loadu_si128(v25 + 1);
LABEL_23:
          n = v25->m128i_u64[1];
          v25->m128i_i64[0] = (__int64)v27;
          v25->m128i_i64[1] = 0;
          v25[1].m128i_i8[0] = 0;
          v28 = n;
          v29 = *a9;
          if ( (__m128i *)p_src == &src )
          {
            if ( n )
            {
              if ( n == 1 )
                v29->m128i_i8[0] = src.m128i_i8[0];
              else
                memcpy(v29, &src, n);
              v28 = n;
              v29 = *a9;
            }
            a9[1] = (__m128i *)v28;
            v29->m128i_i8[v28] = 0;
            v29 = (__m128i *)p_src;
            goto LABEL_27;
          }
          v30 = src.m128i_i64[0];
          if ( v29 == (__m128i *)(a9 + 2) )
          {
            *a9 = (__m128i *)p_src;
            a9[1] = (__m128i *)v28;
            a9[2] = (__m128i *)v30;
          }
          else
          {
            v31 = (__int64)a9[2];
            *a9 = (__m128i *)p_src;
            a9[1] = (__m128i *)v28;
            a9[2] = (__m128i *)v30;
            if ( v29 )
            {
              p_src = (size_t)v29;
              src.m128i_i64[0] = v31;
LABEL_27:
              n = 0;
              v29->m128i_i8[0] = 0;
              if ( (__m128i *)p_src != &src )
                j_j___libc_free_0(p_src, src.m128i_i64[0] + 1);
              if ( (__m128i *)v67 != &v69 )
                j_j___libc_free_0(v67, v69.m128i_i64[0] + 1);
              if ( v61 != v63 )
                j_j___libc_free_0(v61, v63[0] + 1LL);
              if ( p_p_src != v66 )
                j_j___libc_free_0(p_p_src, v66[0] + 1LL);
              if ( argv[0] != dest )
                j_j___libc_free_0(argv[0], dest[0] + 1LL);
              return 0;
            }
          }
          p_src = (size_t)&src;
          v29 = &src;
          goto LABEL_27;
        }
LABEL_13:
        v19 = (__m128i *)sub_2241490(&v61, v18, v17, v13);
        v67 = (size_t)&v69;
        v20 = v19->m128i_i64[0];
        v21 = v19 + 1;
        if ( (__m128i *)v19->m128i_i64[0] == &v19[1] )
          goto LABEL_11;
        goto LABEL_14;
      }
      if ( !v60 )
      {
        v16 = v66;
        goto LABEL_7;
      }
      v50 = v66;
    }
    memcpy(v50, v14, v15);
    v15 = p_src;
    v16 = p_p_src;
    goto LABEL_7;
  }
  v33 = a5;
  v34 = 0;
  src.m128i_i64[0] = (__int64)v75;
  src.m128i_i64[1] = 0x400000000LL;
  v76 = v78;
  p_src = 0;
  n = 0;
  v77 = 0;
  v78[0] = 0;
  v78[1] = 1;
  p_p_src = &p_src;
  sub_16C8200(argv, a4, v33, (__int64)&p_p_src);
  v56 = 0;
  v35 = argv[0];
  v55 = dest[0];
  if ( *(_BYTE *)(a6 + 16) )
  {
    sub_16C8200(argv, *(_QWORD **)a6, *(_QWORD *)(a6 + 8), (__int64)&p_p_src);
    v34 = argv[0];
    v56 = dest[0];
  }
  v36 = fork();
  if ( v36 == -1 )
  {
    argv[0] = (char **)dest;
    strcpy((char *)dest, "Couldn't fork");
    argv[1] = (char **)13;
    goto LABEL_58;
  }
  if ( !v36 )
  {
    if ( !a8 )
      goto LABEL_67;
    LOBYTE(dest[0]) = *(_BYTE *)(a7 + 16);
    if ( !LOBYTE(dest[0])
      || (*(__m128i *)argv = _mm_loadu_si128((const __m128i *)a7), !(unsigned __int8)sub_16C70F0((__int64)argv, 0, a9)) )
    {
      LOBYTE(dest[0]) = *(_BYTE *)(a7 + 40);
      if ( !LOBYTE(dest[0]) )
        goto LABEL_65;
      *(__m128i *)argv = _mm_loadu_si128((const __m128i *)(a7 + 24));
      if ( (unsigned __int8)sub_16C70F0((__int64)argv, 1, a9) )
        goto LABEL_60;
      if ( *(_BYTE *)(a7 + 40) )
      {
        if ( !*(_BYTE *)(a7 + 64) )
          goto LABEL_67;
        v48 = *(_QWORD *)(a7 + 56);
        if ( *(_QWORD *)(a7 + 32) == v48 && (!v48 || !memcmp(*(const void **)(a7 + 24), *(const void **)(a7 + 48), v48)) )
        {
          if ( dup2(1, 2) == -1 )
          {
            argv[0] = (char **)dest;
            v67 = 31;
            v51 = (char **)sub_22409D0(argv, &v67, 0);
            si128 = _mm_load_si128((const __m128i *)&xmmword_42AF0E0);
            argv[0] = v51;
            dest[0] = v67;
            qmemcpy(v51 + 2, "tderr to stdout", 15);
            *(__m128i *)v51 = si128;
            argv[1] = (char **)v67;
            *((_BYTE *)argv[0] + v67) = 0;
LABEL_58:
            sub_16C6DC0(a9, (__int64)argv, -1);
            if ( argv[0] != dest )
              j_j___libc_free_0(argv[0], dest[0] + 1LL);
            goto LABEL_60;
          }
LABEL_67:
          v43 = v59;
          if ( !v59 )
          {
            v46 = (const char *)dest;
            LOBYTE(dest[0]) = 0;
            argv[0] = (char **)dest;
            argv[1] = 0;
LABEL_72:
            if ( v34 )
              execve(v46, v35, v34);
            else
              execv(v46, v35);
            v47 = __errno_location();
            _exit((*v47 == 2) + 126);
          }
          v44 = (char **)v60;
          v45 = (char **)dest;
          argv[0] = (char **)dest;
          v67 = v60;
          if ( v60 > 0xF )
          {
            v58 = v59;
            v54 = (char **)sub_22409D0(argv, &v67, 0);
            v43 = v58;
            argv[0] = v54;
            v53 = v54;
            dest[0] = v67;
          }
          else
          {
            if ( v60 == 1 )
            {
              LOBYTE(dest[0]) = *(_BYTE *)v59;
LABEL_71:
              argv[1] = v44;
              *((_BYTE *)v44 + (_QWORD)v45) = 0;
              v46 = (const char *)argv[0];
              goto LABEL_72;
            }
            if ( !v60 )
              goto LABEL_71;
            v53 = (char **)dest;
          }
          memcpy(v53, v43, (size_t)v44);
          v44 = (char **)v67;
          v45 = argv[0];
          goto LABEL_71;
        }
        LOBYTE(dest[0]) = 1;
      }
      else
      {
LABEL_65:
        LOBYTE(dest[0]) = *(_BYTE *)(a7 + 64);
        if ( !LOBYTE(dest[0]) )
          goto LABEL_67;
      }
      *(__m128i *)argv = _mm_loadu_si128((const __m128i *)(a7 + 48));
      if ( !(unsigned __int8)sub_16C70F0((__int64)argv, 2, a9) )
        goto LABEL_67;
    }
LABEL_60:
    v12 = 0;
    goto LABEL_44;
  }
  v12 = 1;
  *a1 = v36;
  a1[1] = v36;
LABEL_44:
  if ( v34 )
    j_j___libc_free_0(v34, v56 - (_QWORD)v34);
  if ( v35 )
    j_j___libc_free_0(v35, v55 - (_QWORD)v35);
  v37 = (unsigned __int64 *)src.m128i_i64[0];
  v38 = (unsigned __int64 *)(src.m128i_i64[0] + 8LL * src.m128i_u32[2]);
  if ( (unsigned __int64 *)src.m128i_i64[0] != v38 )
  {
    do
    {
      v39 = *v37++;
      _libc_free(v39);
    }
    while ( v38 != v37 );
  }
  v40 = v76;
  v41 = &v76[2 * (unsigned int)v77];
  if ( v76 != v41 )
  {
    do
    {
      v42 = *v40;
      v40 += 2;
      _libc_free(v42);
    }
    while ( v40 != v41 );
    v41 = v76;
  }
  if ( v41 != v78 )
    _libc_free((unsigned __int64)v41);
  if ( (_BYTE *)src.m128i_i64[0] != v75 )
    _libc_free(src.m128i_u64[0]);
  return v12;
}
