// Function: sub_C525B0
// Address: 0xc525b0
//
__int64 __fastcall sub_C525B0(char *a1)
{
  __int64 v1; // rax
  char v2; // dl
  __int64 v3; // r12
  __int64 p_base; // rdi
  __int64 v5; // rdx
  __int64 **v6; // rax
  __int64 v7; // rdx
  __int64 **v8; // r15
  unsigned __int64 v9; // rdx
  __int64 *v10; // rbx
  __int64 **v11; // r13
  signed __int64 v12; // rsi
  __int64 v13; // rdi
  __int64 v14; // r13
  _QWORD *v15; // rax
  __int64 v16; // rax
  __int64 v17; // rax
  const char *v18; // rsi
  void *v19; // rdi
  __int64 v20; // rbx
  __int64 i; // r13
  size_t v22; // rdx
  _QWORD *v23; // r14
  __int64 v24; // r8
  _BYTE *v25; // rax
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // rax
  size_t v29; // rdx
  __int64 v30; // rax
  __int64 v31; // rax
  unsigned __int64 v32; // r13
  __int64 v33; // rdi
  __int64 v34; // rbx
  __int64 v35; // r12
  unsigned __int64 v36; // rax
  __int64 v37; // rax
  void *v38; // rdi
  _BYTE **v39; // rsi
  __int64 v40; // rax
  _QWORD *v41; // rbx
  _QWORD *j; // r13
  __int64 v43; // rax
  unsigned __int64 v44; // r14
  __int64 v45; // r12
  __int64 v46; // rdx
  __int64 result; // rax
  __int64 v48; // rdx
  __int64 v49; // r14
  _QWORD *v50; // rdx
  __int64 **v51; // rax
  __int64 v52; // rdi
  __int64 v53; // r13
  __int64 v54; // rax
  __int64 v55; // rdi
  __int64 v56; // r13
  const char **v57; // rax
  __int64 v58; // rdi
  const char *v59; // rsi
  __int64 v60; // rax
  const char **v61; // r13
  size_t v62; // r12
  const char **v63; // rbx
  const char *v64; // rdi
  size_t v65; // rax
  __int64 v66; // rdi
  char *v67; // rsi
  __int64 v68; // rdi
  const char **v69; // rbx
  const char **v70; // r15
  __int64 v71; // rax
  __int64 v72; // rax
  __int64 v73; // rax
  const char *v74; // rsi
  __int64 v75; // rdi
  __int64 v76; // rsi
  __int64 v77; // rax
  const char *v78; // rax
  __int64 v79; // rax
  __int64 v80; // rax
  __int64 v81; // rax
  __int64 v82; // rax
  __int64 v83; // rdi
  __int64 v84; // rdi
  __int64 v85; // r12
  _QWORD *v86; // rax
  __int64 v87; // rax
  __int64 v89; // [rsp+20h] [rbp-1060h]
  __int64 v90; // [rsp+20h] [rbp-1060h]
  __int64 v91; // [rsp+28h] [rbp-1058h]
  __int64 v92; // [rsp+28h] [rbp-1058h]
  __int64 v93; // [rsp+28h] [rbp-1058h]
  _BYTE *v94; // [rsp+30h] [rbp-1050h] BYREF
  __int64 v95; // [rsp+38h] [rbp-1048h]
  _BYTE v96[2048]; // [rsp+40h] [rbp-1040h] BYREF
  void *base; // [rsp+840h] [rbp-840h] BYREF
  __int64 v98; // [rsp+848h] [rbp-838h]
  _BYTE v99[2096]; // [rsp+850h] [rbp-830h] BYREF

  v1 = ((__int64 (*)(void))sub_C4F9D0)();
  v2 = a1[8];
  v95 = 0x8000000000LL;
  v3 = *(_QWORD *)(v1 + 344);
  p_base = v3 + 128;
  v94 = v96;
  sub_C50450((__int64 **)(v3 + 128), (__int64)&v94, v2);
  v98 = 0x8000000000LL;
  base = v99;
  v5 = ((__int64 (*)(void))sub_C4F9D0)();
  v6 = *(__int64 ***)(v5 + 288);
  if ( *(_BYTE *)(v5 + 308) )
    v7 = *(unsigned int *)(v5 + 300);
  else
    v7 = *(unsigned int *)(v5 + 296);
  v8 = &v6[v7];
  v9 = (unsigned int)v98;
  if ( v6 != v8 )
  {
    while ( 1 )
    {
      v10 = *v6;
      v11 = v6;
      if ( (unsigned __int64)*v6 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v8 == ++v6 )
        goto LABEL_6;
    }
    if ( v8 != v6 )
    {
      p_base = (__int64)&base;
      do
      {
        if ( v10[1] )
        {
          v49 = *v10;
          if ( v9 + 1 > HIDWORD(v98) )
          {
            sub_C8D5F0(&base, v99, v9 + 1, 16);
            v9 = (unsigned int)v98;
          }
          v50 = (char *)base + 16 * v9;
          *v50 = v49;
          v50[1] = v10;
          v9 = (unsigned int)(v98 + 1);
          LODWORD(v98) = v98 + 1;
        }
        v51 = v11 + 1;
        if ( v11 + 1 == v8 )
          break;
        v10 = *v51;
        for ( ++v11; (unsigned __int64)*v51 >= 0xFFFFFFFFFFFFFFFELL; v11 = v51 )
        {
          if ( v8 == ++v51 )
            goto LABEL_6;
          v10 = *v51;
        }
      }
      while ( v8 != v11 );
    }
  }
LABEL_6:
  v12 = 16 * v9;
  if ( v9 > 1 )
  {
    p_base = (__int64)base;
    v12 >>= 4;
    qsort(base, v12, 0x10u, (__compar_fn_t)sub_C4F860);
  }
  if ( *(_QWORD *)(sub_C4F9D0(p_base, v12) + 40) )
  {
    v52 = sub_CB7210(p_base, v12);
    v53 = sub_904010(v52, "OVERVIEW: ");
    v54 = sub_C4F9D0(v52, "OVERVIEW: ");
    v12 = (signed __int64)"\n";
    p_base = sub_A51340(v53, *(const void **)(v54 + 32), *(_QWORD *)(v54 + 40));
    sub_904010(p_base, "\n");
    if ( v3 != sub_C52570() )
      goto LABEL_10;
  }
  else if ( v3 != sub_C52570() )
  {
LABEL_10:
    if ( *(_QWORD *)(v3 + 24) )
    {
      v79 = sub_CB7210(p_base, v12);
      v80 = sub_904010(v79, "SUBCOMMAND '");
      v81 = sub_A51340(v80, *(const void **)v3, *(_QWORD *)(v3 + 8));
      v82 = sub_904010(v81, "': ");
      v12 = (signed __int64)"\n\n";
      p_base = sub_A51340(v82, *(const void **)(v3 + 16), *(_QWORD *)(v3 + 24));
      sub_904010(p_base, "\n\n");
    }
    v13 = sub_CB7210(p_base, v12);
    v14 = sub_904010(v13, "USAGE: ");
    v15 = (_QWORD *)sub_C4F9D0(v13, "USAGE: ");
    v16 = sub_CB6200(v14, *v15, v15[1]);
    v17 = sub_904010(v16, " ");
    v18 = " [options]";
    v19 = (void *)sub_A51340(v17, *(const void **)v3, *(_QWORD *)(v3 + 8));
    sub_904010((__int64)v19, " [options]");
    goto LABEL_13;
  }
  v55 = sub_CB7210(p_base, v12);
  v56 = sub_904010(v55, "USAGE: ");
  v57 = (const char **)sub_C4F9D0(v55, "USAGE: ");
  v58 = v56;
  v59 = *v57;
  sub_CB6200(v56, *v57, v57[1]);
  if ( (_DWORD)v98 )
  {
    v87 = sub_CB7210(v56, v59);
    v59 = " [subcommand]";
    v58 = v87;
    sub_904010(v87, " [subcommand]");
  }
  v60 = sub_CB7210(v58, v59);
  v18 = " [options]";
  v19 = (void *)v60;
  sub_904010(v60, " [options]");
LABEL_13:
  v20 = *(_QWORD *)(v3 + 32);
  for ( i = v20 + 8LL * *(unsigned int *)(v3 + 40); i != v20; v20 += 8 )
  {
    while ( 1 )
    {
      v23 = *(_QWORD **)v20;
      if ( *(_QWORD *)(*(_QWORD *)v20 + 32LL) )
      {
        v27 = sub_CB7210(v19, v18);
        v28 = sub_904010(v27, " --");
        v29 = v23[4];
        v18 = (const char *)v23[3];
        v19 = *(void **)(v28 + 32);
        if ( v29 > *(_QWORD *)(v28 + 24) - (_QWORD)v19 )
        {
          v19 = (void *)v28;
          sub_CB6200(v28, v18, v29);
        }
        else if ( v29 )
        {
          v89 = v28;
          v91 = v23[4];
          memcpy(v19, v18, v29);
          *(_QWORD *)(v89 + 32) += v91;
        }
      }
      v24 = sub_CB7210(v19, v18);
      v25 = *(_BYTE **)(v24 + 32);
      if ( *(_BYTE **)(v24 + 24) == v25 )
      {
        v26 = sub_CB6200(v24, " ", 1);
        v19 = *(void **)(v26 + 32);
        v24 = v26;
      }
      else
      {
        *v25 = 32;
        v19 = (void *)(*(_QWORD *)(v24 + 32) + 1LL);
        *(_QWORD *)(v24 + 32) = v19;
      }
      v22 = v23[6];
      v18 = (const char *)v23[5];
      if ( v22 <= *(_QWORD *)(v24 + 24) - (_QWORD)v19 )
        break;
      v19 = (void *)v24;
      v20 += 8;
      sub_CB6200(v24, v18, v22);
      if ( i == v20 )
        goto LABEL_27;
    }
    if ( v22 )
    {
      v90 = v24;
      v92 = v23[6];
      memcpy(v19, v18, v22);
      *(_QWORD *)(v90 + 32) += v92;
    }
  }
LABEL_27:
  if ( *(_QWORD *)(v3 + 152) )
  {
    v30 = sub_CB7210(v19, v18);
    v19 = (void *)sub_904010(v30, " ");
    v31 = *(_QWORD *)(v3 + 152);
    v18 = *(const char **)(v31 + 40);
    sub_A51340((__int64)v19, v18, *(_QWORD *)(v31 + 48));
  }
  if ( v3 == sub_C52570() && (_DWORD)v98 )
  {
    v61 = (const char **)base;
    v62 = 0;
    v63 = (const char **)((char *)base + 16 * (unsigned int)v98);
    do
    {
      v64 = *v61;
      v65 = strlen(*v61);
      if ( v62 < v65 )
        v62 = v65;
      v61 += 2;
    }
    while ( v63 != v61 );
    v66 = sub_CB7210(v64, v18);
    sub_904010(v66, "\n\n");
    v67 = "SUBCOMMANDS:\n\n";
    v68 = sub_CB7210(v66, "\n\n");
    sub_904010(v68, "SUBCOMMANDS:\n\n");
    v69 = (const char **)base;
    v70 = (const char **)((char *)base + 16 * (unsigned int)v98);
    if ( base != v70 )
    {
      do
      {
        v72 = sub_CB7210(v68, v67);
        v73 = sub_904010(v72, "  ");
        v74 = *v69;
        v75 = v73;
        sub_904010(v73, *v69);
        if ( *((_QWORD *)v69[1] + 3) )
        {
          v93 = sub_CB7210(v75, v74);
          v76 = (unsigned int)v62 - (unsigned int)strlen(*v69);
          sub_CB69B0(v93, v76);
          v77 = sub_CB7210(v93, v76);
          v75 = sub_904010(v77, " - ");
          v78 = v69[1];
          v74 = (const char *)*((_QWORD *)v78 + 2);
          sub_A51340(v75, v74, *((_QWORD *)v78 + 3));
        }
        v71 = sub_CB7210(v75, v74);
        v69 += 2;
        v67 = "\n";
        v68 = v71;
        sub_904010(v71, "\n");
      }
      while ( v70 != v69 );
    }
    v83 = sub_CB7210(v68, v67);
    sub_904010(v83, "\n");
    v84 = sub_CB7210(v83, "\n");
    v85 = sub_904010(v84, "  Type \"");
    v86 = (_QWORD *)sub_C4F9D0(v84, "  Type \"");
    v18 = " <subcommand> --help\" to get more help on a specific subcommand";
    v19 = (void *)sub_CB6200(v85, *v86, v86[1]);
    sub_904010((__int64)v19, " <subcommand> --help\" to get more help on a specific subcommand");
  }
  v32 = 0;
  v33 = sub_CB7210(v19, v18);
  sub_904010(v33, "\n\n");
  v34 = (unsigned int)v95;
  if ( (_DWORD)v95 )
  {
    v35 = 0;
    do
    {
      v33 = *(_QWORD *)&v94[16 * v35 + 8];
      v36 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v33 + 40LL))(v33);
      if ( v32 < v36 )
        v32 = v36;
      ++v35;
    }
    while ( v35 != v34 );
  }
  v37 = sub_CB7210(v33, "\n\n");
  sub_904010(v37, "OPTIONS:\n");
  v38 = a1;
  v39 = &v94;
  (**(void (__fastcall ***)(char *, _BYTE **, unsigned __int64))a1)(a1, &v94, v32);
  v40 = sub_C4F9D0(a1, &v94);
  v41 = *(_QWORD **)(v40 + 56);
  for ( j = *(_QWORD **)(v40 + 48); v41 != j; j += 2 )
  {
    while ( 1 )
    {
      v43 = sub_CB7210(v38, v39);
      v44 = j[1];
      v39 = (_BYTE **)*j;
      v38 = *(void **)(v43 + 32);
      v45 = v43;
      if ( v44 <= *(_QWORD *)(v43 + 24) - (_QWORD)v38 )
        break;
      v46 = j[1];
      v38 = (void *)v43;
      j += 2;
      sub_CB6200(v43, v39, v46);
      if ( v41 == j )
        goto LABEL_42;
    }
    if ( v44 )
    {
      memcpy(v38, v39, j[1]);
      *(_QWORD *)(v45 + 32) += v44;
    }
  }
LABEL_42:
  result = sub_C4F9D0(v38, v39);
  v48 = *(_QWORD *)(result + 48);
  if ( v48 != *(_QWORD *)(result + 56) )
    *(_QWORD *)(result + 56) = v48;
  if ( base != v99 )
    result = _libc_free(base, v39);
  if ( v94 != v96 )
    return _libc_free(v94, v39);
  return result;
}
