// Function: sub_C87C00
// Address: 0xc87c00
//
__int64 __fastcall sub_C87C00(
        __pid_t *a1,
        _BYTE *a2,
        size_t a3,
        _QWORD *a4,
        __int64 a5,
        char a6,
        _QWORD *a7,
        __int64 a8,
        __int64 a9,
        __int64 a10,
        unsigned int a11,
        _QWORD *a12,
        char a13)
{
  char *const *v13; // r13
  __int64 p_argv; // rsi
  char *const *rlim_cur; // r14
  __pid_t v16; // eax
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // r9
  unsigned int v21; // r12d
  __int64 *v22; // r15
  __int64 *v23; // r13
  __int64 *i; // rax
  __int64 v25; // rdi
  unsigned int v26; // ecx
  __int64 *v27; // r13
  __int64 *v28; // r14
  __int64 v29; // rdi
  char *v31; // rdx
  char *v32; // rsi
  __int64 v33; // rdx
  __int64 v34; // rcx
  __int64 v35; // r8
  __int64 v36; // r9
  size_t v37; // rdx
  __int64 v38; // rcx
  __int64 v39; // r8
  __int64 v40; // r9
  _QWORD *v41; // rdi
  int *v42; // rax
  __int64 v45; // [rsp+40h] [rbp-E0h]
  __int64 v46; // [rsp+48h] [rbp-D8h]
  _QWORD *v48; // [rsp+60h] [rbp-C0h] BYREF
  rlim_t v49; // [rsp+68h] [rbp-B8h] BYREF
  struct rlimit argv; // [rsp+70h] [rbp-B0h] BYREF
  _QWORD dest[2]; // [rsp+80h] [rbp-A0h] BYREF
  _QWORD v52[2]; // [rsp+90h] [rbp-90h] BYREF
  __int64 *v53; // [rsp+A0h] [rbp-80h]
  __int64 v54; // [rsp+A8h] [rbp-78h]
  _BYTE v55[32]; // [rsp+B0h] [rbp-70h] BYREF
  __int64 *v56; // [rsp+D0h] [rbp-50h]
  __int64 v57; // [rsp+D8h] [rbp-48h]
  _QWORD v58[8]; // [rsp+E0h] [rbp-40h] BYREF

  v13 = 0;
  v53 = (__int64 *)v55;
  p_argv = (__int64)a4;
  v54 = 0x400000000LL;
  v48 = v52;
  v52[0] = 0;
  v52[1] = 0;
  v56 = v58;
  v57 = 0;
  v58[0] = 0;
  v58[1] = 1;
  sub_C87B00(&argv, a4, a5, (__int64)&v48);
  v46 = 0;
  rlim_cur = (char *const *)argv.rlim_cur;
  v45 = dest[0];
  if ( a6 )
  {
    p_argv = (__int64)a7;
    sub_C87B00(&argv, a7, a8, (__int64)&v48);
    v13 = (char *const *)argv.rlim_cur;
    v46 = dest[0];
  }
  v16 = fork();
  if ( v16 == -1 )
  {
    v31 = "";
    argv.rlim_cur = (rlim_t)dest;
    v32 = "Couldn't fork";
    goto LABEL_25;
  }
  if ( v16 )
  {
    v21 = 1;
    *a1 = v16;
    a1[1] = v16;
    goto LABEL_6;
  }
  if ( !a10 )
  {
LABEL_29:
    if ( a13 && setsid() == -1 )
    {
      v31 = "";
      argv.rlim_cur = (rlim_t)dest;
      v32 = "Could not detach process, ::setsid failed";
LABEL_25:
      sub_C865D0((__int64 *)&argv, v32, (__int64)v31);
      p_argv = (__int64)&argv;
      sub_C86680(a12, (__int64)&argv, -1);
      if ( (_QWORD *)argv.rlim_cur != dest )
      {
        p_argv = dest[0] + 1LL;
        j_j___libc_free_0(argv.rlim_cur, dest[0] + 1LL);
      }
      goto LABEL_27;
    }
    if ( a11 )
    {
      getrlimit(RLIMIT_DATA, &argv);
      argv.rlim_cur = (unsigned __int64)a11 << 20;
      setrlimit(RLIMIT_DATA, &argv);
      getrlimit(__RLIMIT_RSS, &argv);
      argv.rlim_cur = (unsigned __int64)a11 << 20;
      setrlimit(__RLIMIT_RSS, &argv);
    }
    v41 = dest;
    argv.rlim_cur = (rlim_t)dest;
    if ( &a2[a3] && !a2 )
      sub_426248((__int64)"basic_string::_M_construct null not valid");
    v49 = a3;
    if ( a3 > 0xF )
    {
      argv.rlim_cur = sub_22409D0(&argv, &v49, 0);
      v41 = (_QWORD *)argv.rlim_cur;
      dest[0] = v49;
    }
    else
    {
      if ( a3 == 1 )
      {
        LOBYTE(dest[0]) = *a2;
        goto LABEL_50;
      }
      if ( !a3 )
      {
LABEL_50:
        argv.rlim_max = v49;
        *(_BYTE *)(argv.rlim_cur + v49) = 0;
        if ( v13 )
          execve((const char *)argv.rlim_cur, rlim_cur, v13);
        else
          execv((const char *)argv.rlim_cur, rlim_cur);
        v42 = __errno_location();
        _exit((*v42 == 2) + 126);
      }
    }
    memcpy(v41, a2, a3);
    goto LABEL_50;
  }
  p_argv = (__int64)a12;
  if ( !(unsigned __int8)sub_C869B0(
                           0,
                           a12,
                           v17,
                           v18,
                           v19,
                           v20,
                           *(_BYTE **)a9,
                           *(_QWORD *)(a9 + 8),
                           *(_QWORD *)(a9 + 16)) )
  {
    p_argv = (__int64)a12;
    if ( !(unsigned __int8)sub_C869B0(
                             1,
                             a12,
                             v33,
                             v34,
                             v35,
                             v36,
                             *(_BYTE **)(a9 + 24),
                             *(_QWORD *)(a9 + 32),
                             *(_QWORD *)(a9 + 40)) )
    {
      if ( *(_BYTE *)(a9 + 40)
        && *(_BYTE *)(a9 + 64)
        && (v37 = *(_QWORD *)(a9 + 32), v37 == *(_QWORD *)(a9 + 56))
        && (!v37 || !memcmp(*(const void **)(a9 + 24), *(const void **)(a9 + 48), v37)) )
      {
        if ( dup2(1, 2) != -1 )
          goto LABEL_29;
        argv.rlim_cur = (rlim_t)dest;
        v49 = 31;
        argv.rlim_cur = sub_22409D0(&argv, &v49, 0);
        dest[0] = v49;
        qmemcpy((void *)argv.rlim_cur, "Can't redirect stderr to stdout", 0x1Fu);
        p_argv = (__int64)&argv;
        argv.rlim_max = v49;
        *(_BYTE *)(argv.rlim_cur + v49) = 0;
        sub_C86680(a12, (__int64)&argv, -1);
        sub_2240A30(&argv);
      }
      else
      {
        p_argv = (__int64)a12;
        if ( !(unsigned __int8)sub_C869B0(
                                 2,
                                 a12,
                                 v37,
                                 v38,
                                 v39,
                                 v40,
                                 *(_BYTE **)(a9 + 48),
                                 *(_QWORD *)(a9 + 56),
                                 *(_QWORD *)(a9 + 64)) )
          goto LABEL_29;
      }
    }
  }
LABEL_27:
  v21 = 0;
LABEL_6:
  if ( v13 )
  {
    p_argv = v46 - (_QWORD)v13;
    j_j___libc_free_0(v13, v46 - (_QWORD)v13);
  }
  if ( rlim_cur )
  {
    p_argv = v45 - (_QWORD)rlim_cur;
    j_j___libc_free_0(rlim_cur, v45 - (_QWORD)rlim_cur);
  }
  v22 = v53;
  v23 = &v53[(unsigned int)v54];
  if ( v53 != v23 )
  {
    for ( i = v53; ; i = v53 )
    {
      v25 = *v22;
      v26 = (unsigned int)(v22 - i) >> 7;
      p_argv = 4096LL << v26;
      if ( v26 >= 0x1E )
        p_argv = 0x40000000000LL;
      ++v22;
      sub_C7D6A0(v25, p_argv, 16);
      if ( v23 == v22 )
        break;
    }
  }
  v27 = v56;
  v28 = &v56[2 * (unsigned int)v57];
  if ( v56 != v28 )
  {
    do
    {
      p_argv = v27[1];
      v29 = *v27;
      v27 += 2;
      sub_C7D6A0(v29, p_argv, 16);
    }
    while ( v28 != v27 );
    v28 = v56;
  }
  if ( v28 != v58 )
    _libc_free(v28, p_argv);
  if ( v53 != (__int64 *)v55 )
    _libc_free(v53, p_argv);
  return v21;
}
