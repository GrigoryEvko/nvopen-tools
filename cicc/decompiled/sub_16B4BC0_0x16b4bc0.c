// Function: sub_16B4BC0
// Address: 0x16b4bc0
//
void __fastcall sub_16B4BC0(char *a1)
{
  __int64 v1; // rax
  char v2; // dl
  __int64 v3; // r12
  void *p_base; // rdi
  __int64 v5; // rdx
  __int64 **v6; // rax
  __int64 v7; // rdx
  __int64 **v8; // r15
  unsigned __int64 v9; // rdx
  __int64 *v10; // r13
  __int64 **v11; // rbx
  signed __int64 v12; // rsi
  __int64 v13; // rdx
  void *v14; // rdi
  __int64 v15; // rdx
  __int64 v16; // rdi
  __int64 v17; // r13
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rax
  const char *v21; // rsi
  void *v22; // rdi
  size_t v23; // rdx
  __int64 v24; // rbx
  __int64 v25; // r13
  size_t v26; // r12
  _QWORD *v27; // r12
  __int64 v28; // r14
  _BYTE *v29; // rax
  __int64 v30; // rax
  __int64 v31; // rax
  __int64 v32; // rax
  __int64 v33; // r14
  __int64 v34; // rax
  __int64 v35; // rdi
  __int64 v36; // rax
  void *v37; // rdi
  __int64 v38; // rdx
  unsigned __int64 v39; // r13
  __int64 v40; // rdi
  __int64 v41; // rdx
  __int64 v42; // rbx
  __int64 v43; // r12
  unsigned __int64 v44; // rax
  __int64 v45; // rax
  void *v46; // rdi
  const char *v47; // rsi
  __int64 v48; // rax
  __int64 v49; // rdx
  __int64 v50; // rbx
  __int64 i; // r14
  const char *v52; // r13
  size_t v53; // r15
  __int64 v54; // rax
  __int64 v55; // r12
  __int64 v56; // rax
  __int64 v57; // rdx
  __int64 v58; // rax
  _QWORD *v59; // rdx
  __int64 **v60; // rax
  __int64 v61; // rdi
  __int64 v62; // r13
  __int64 v63; // rax
  __int64 v64; // rax
  const char **v65; // r13
  size_t v66; // r12
  const char **v67; // rbx
  const char *v68; // rdi
  size_t v69; // rax
  __int64 v70; // rdx
  __int64 v71; // rdi
  __int64 v72; // rdx
  char *v73; // rsi
  __int64 v74; // rdi
  __int64 v75; // rdx
  const char **v76; // rbx
  const char **v77; // r13
  __int64 v78; // rax
  __int64 v79; // rax
  __int64 v80; // rax
  const char *v81; // rsi
  __int64 v82; // rdi
  __int64 v83; // rdx
  __int64 v84; // r14
  __int64 v85; // rsi
  __int64 v86; // rdx
  __int64 v87; // rax
  const char *v88; // rax
  __int64 v89; // rax
  __int64 v90; // rax
  __int64 v91; // rax
  __int64 v92; // rax
  __int64 v93; // rdi
  __int64 v94; // r13
  __int64 v95; // rax
  __int64 v96; // rdi
  const char *v97; // rsi
  __int64 v98; // rdx
  __int64 v99; // rax
  __int64 v100; // rdi
  __int64 v101; // rdx
  __int64 v102; // rdi
  __int64 v103; // r12
  __int64 v104; // rax
  __int64 v105; // rax
  size_t v106; // [rsp+0h] [rbp-1080h]
  __int64 v107; // [rsp+8h] [rbp-1078h]
  __int64 v108; // [rsp+8h] [rbp-1078h]
  _BYTE *v110; // [rsp+30h] [rbp-1050h] BYREF
  __int64 v111; // [rsp+38h] [rbp-1048h]
  _BYTE v112[2048]; // [rsp+40h] [rbp-1040h] BYREF
  void *base; // [rsp+840h] [rbp-840h] BYREF
  __int64 v114; // [rsp+848h] [rbp-838h]
  _BYTE v115[2096]; // [rsp+850h] [rbp-830h] BYREF

  v1 = ((__int64 (*)(void))sub_16B0440)();
  v2 = a1[8];
  v111 = 0x8000000000LL;
  v3 = *(_QWORD *)(v1 + 312);
  p_base = (void *)(v3 + 128);
  v110 = v112;
  sub_16B0990((__int64 **)(v3 + 128), (__int64)&v110, v2);
  v114 = 0x8000000000LL;
  base = v115;
  v5 = ((__int64 (*)(void))sub_16B0440)();
  v6 = *(__int64 ***)(v5 + 256);
  if ( v6 == *(__int64 ***)(v5 + 248) )
    v7 = *(unsigned int *)(v5 + 268);
  else
    v7 = *(unsigned int *)(v5 + 264);
  v8 = &v6[v7];
  v9 = (unsigned int)v114;
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
      p_base = &base;
      do
      {
        if ( v10[1] )
        {
          v58 = *v10;
          if ( (unsigned int)v9 >= HIDWORD(v114) )
          {
            p_base = &base;
            v108 = *v10;
            sub_16CD150(&base, v115, 0, 16);
            v9 = (unsigned int)v114;
            v58 = v108;
          }
          v59 = (char *)base + 16 * v9;
          *v59 = v58;
          v59[1] = v10;
          v9 = (unsigned int)(v114 + 1);
          LODWORD(v114) = v114 + 1;
        }
        v60 = v11 + 1;
        if ( v11 + 1 == v8 )
          break;
        v10 = *v60;
        for ( ++v11; (unsigned __int64)*v60 >= 0xFFFFFFFFFFFFFFFELL; v11 = v60 )
        {
          if ( v8 == ++v60 )
            goto LABEL_6;
          v10 = *v60;
        }
      }
      while ( v8 != v11 );
    }
  }
LABEL_6:
  v12 = 16 * v9;
  if ( v9 > 1 )
  {
    p_base = base;
    v12 >>= 4;
    qsort(base, v12, 0x10u, (__compar_fn_t)sub_16B0350);
  }
  if ( *(_QWORD *)(sub_16B0440(p_base, v12) + 40) )
  {
    v61 = sub_16E8C20(p_base, v12, v13);
    v62 = sub_1263B40(v61, "OVERVIEW: ");
    v63 = sub_16B0440(v61, "OVERVIEW: ");
    v64 = sub_1549FF0(v62, *(const char **)(v63 + 32), *(_QWORD *)(v63 + 40));
    v12 = (signed __int64)"\n";
    sub_1263B40(v64, "\n");
  }
  v14 = &unk_4FA0190;
  if ( v3 == sub_16B4B80((__int64)&unk_4FA0190) )
  {
    v93 = sub_16E8C20(&unk_4FA0190, v12, v15);
    v94 = sub_1263B40(v93, "USAGE: ");
    v95 = sub_16B0440(v93, "USAGE: ");
    v96 = v94;
    v97 = *(const char **)v95;
    sub_16E7EE0(v94, *(const char **)v95, *(_QWORD *)(v95 + 8));
    if ( (unsigned int)v114 > 2 )
    {
      v105 = sub_16E8C20(v94, v97, v98);
      v97 = " [subcommand]";
      v96 = v105;
      sub_1263B40(v105, " [subcommand]");
    }
    v99 = sub_16E8C20(v96, v97, v98);
    v21 = " [options]";
    v22 = (void *)v99;
    sub_1263B40(v99, " [options]");
  }
  else
  {
    if ( *(_QWORD *)(v3 + 24) )
    {
      v89 = sub_16E8C20(&unk_4FA0190, v12, v15);
      v90 = sub_1263B40(v89, "SUBCOMMAND '");
      v91 = sub_1549FF0(v90, *(const char **)v3, *(_QWORD *)(v3 + 8));
      v92 = sub_1263B40(v91, "': ");
      v12 = (signed __int64)"\n\n";
      v14 = (void *)sub_1549FF0(v92, *(const char **)(v3 + 16), *(_QWORD *)(v3 + 24));
      sub_1263B40((__int64)v14, "\n\n");
    }
    v16 = sub_16E8C20(v14, v12, v15);
    v17 = sub_1263B40(v16, "USAGE: ");
    v18 = sub_16B0440(v16, "USAGE: ");
    v19 = sub_16E7EE0(v17, *(const char **)v18, *(_QWORD *)(v18 + 8));
    v20 = sub_1263B40(v19, " ");
    v21 = " [options]";
    v22 = (void *)sub_1549FF0(v20, *(const char **)v3, *(_QWORD *)(v3 + 8));
    sub_1263B40((__int64)v22, " [options]");
  }
  v24 = *(_QWORD *)(v3 + 32);
  v25 = v24 + 8LL * *(unsigned int *)(v3 + 40);
  if ( v24 != v25 )
  {
    v107 = v3;
    do
    {
      while ( 1 )
      {
        v27 = *(_QWORD **)v24;
        if ( *(_QWORD *)(*(_QWORD *)v24 + 32LL) )
        {
          v31 = sub_16E8C20(v22, v21, v23);
          v32 = sub_1263B40(v31, " --");
          v23 = v27[4];
          v21 = (const char *)v27[3];
          v22 = *(void **)(v32 + 24);
          v33 = v32;
          if ( v23 > *(_QWORD *)(v32 + 16) - (_QWORD)v22 )
          {
            v22 = (void *)v32;
            sub_16E7EE0(v32, v21);
          }
          else if ( v23 )
          {
            v106 = v27[4];
            memcpy(v22, v21, v23);
            v23 = v106;
            *(_QWORD *)(v33 + 24) += v106;
          }
        }
        v28 = sub_16E8C20(v22, v21, v23);
        v29 = *(_BYTE **)(v28 + 24);
        if ( *(_BYTE **)(v28 + 16) == v29 )
        {
          v30 = sub_16E7EE0(v28, " ", 1);
          v22 = *(void **)(v30 + 24);
          v28 = v30;
        }
        else
        {
          *v29 = 32;
          v22 = (void *)(*(_QWORD *)(v28 + 24) + 1LL);
          *(_QWORD *)(v28 + 24) = v22;
        }
        v21 = (const char *)v27[5];
        v26 = v27[6];
        if ( v26 <= *(_QWORD *)(v28 + 16) - (_QWORD)v22 )
          break;
        v22 = (void *)v28;
        v24 += 8;
        sub_16E7EE0(v28, v21, v26);
        if ( v25 == v24 )
          goto LABEL_28;
      }
      if ( v26 )
      {
        memcpy(v22, v21, v26);
        *(_QWORD *)(v28 + 24) += v26;
      }
      v24 += 8;
    }
    while ( v25 != v24 );
LABEL_28:
    v3 = v107;
  }
  if ( *(_QWORD *)(v3 + 160) )
  {
    v34 = sub_16E8C20(v22, v21, v23);
    v35 = sub_1263B40(v34, " ");
    v36 = *(_QWORD *)(v3 + 160);
    v21 = *(const char **)(v36 + 40);
    sub_1549FF0(v35, v21, *(_QWORD *)(v36 + 48));
  }
  v37 = &unk_4FA0190;
  if ( v3 == sub_16B4B80((__int64)&unk_4FA0190) && (_DWORD)v114 )
  {
    v65 = (const char **)base;
    v66 = 0;
    v67 = (const char **)((char *)base + 16 * (unsigned int)v114);
    do
    {
      v68 = *v65;
      v69 = strlen(*v65);
      if ( v66 < v69 )
        v66 = v69;
      v65 += 2;
    }
    while ( v67 != v65 );
    v71 = sub_16E8C20(v68, v21, v70);
    sub_1263B40(v71, "\n\n");
    v73 = "SUBCOMMANDS:\n\n";
    v74 = sub_16E8C20(v71, "\n\n", v72);
    sub_1263B40(v74, "SUBCOMMANDS:\n\n");
    v76 = (const char **)base;
    v77 = (const char **)((char *)base + 16 * (unsigned int)v114);
    if ( base != v77 )
    {
      do
      {
        v79 = sub_16E8C20(v74, v73, v75);
        v80 = sub_1263B40(v79, "  ");
        v81 = *v76;
        v82 = v80;
        sub_1263B40(v80, *v76);
        if ( *((_QWORD *)v76[1] + 3) )
        {
          v84 = sub_16E8C20(v82, v81, v83);
          v85 = (unsigned int)v66 - (unsigned int)strlen(*v76);
          sub_16E8750(v84, v85);
          v87 = sub_16E8C20(v84, v85, v86);
          v82 = sub_1263B40(v87, " - ");
          v88 = v76[1];
          v81 = (const char *)*((_QWORD *)v88 + 2);
          sub_1549FF0(v82, v81, *((_QWORD *)v88 + 3));
        }
        v78 = sub_16E8C20(v82, v81, v83);
        v76 += 2;
        v73 = "\n";
        v74 = v78;
        sub_1263B40(v78, "\n");
      }
      while ( v77 != v76 );
    }
    v100 = sub_16E8C20(v74, v73, v75);
    sub_1263B40(v100, "\n");
    v102 = sub_16E8C20(v100, "\n", v101);
    v103 = sub_1263B40(v102, "  Type \"");
    v104 = sub_16B0440(v102, "  Type \"");
    v21 = " <subcommand> -help\" to get more help on a specific subcommand";
    v37 = (void *)sub_16E7EE0(v103, *(const char **)v104, *(_QWORD *)(v104 + 8));
    sub_1263B40((__int64)v37, " <subcommand> -help\" to get more help on a specific subcommand");
  }
  v39 = 0;
  v40 = sub_16E8C20(v37, v21, v38);
  sub_1263B40(v40, "\n\n");
  v42 = (unsigned int)v111;
  if ( (_DWORD)v111 )
  {
    v43 = 0;
    do
    {
      v40 = *(_QWORD *)&v110[16 * v43 + 8];
      v44 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v40 + 40LL))(v40);
      if ( v39 < v44 )
        v39 = v44;
      ++v43;
    }
    while ( v43 != v42 );
  }
  v45 = sub_16E8C20(v40, "\n\n", v41);
  sub_1263B40(v45, "OPTIONS:\n");
  v46 = a1;
  v47 = (const char *)&v110;
  (**(void (__fastcall ***)(char *, _BYTE **, unsigned __int64))a1)(a1, &v110, v39);
  v48 = sub_16B0440(a1, &v110);
  v50 = *(_QWORD *)(v48 + 56);
  for ( i = *(_QWORD *)(v48 + 48); v50 != i; i += 16 )
  {
    while ( 1 )
    {
      v52 = *(const char **)i;
      v53 = *(_QWORD *)(i + 8);
      v54 = sub_16E8C20(v46, v47, v49);
      v46 = *(void **)(v54 + 24);
      v55 = v54;
      if ( v53 <= *(_QWORD *)(v54 + 16) - (_QWORD)v46 )
        break;
      v47 = v52;
      v46 = (void *)v54;
      i += 16;
      sub_16E7EE0(v54, v52, v53);
      if ( v50 == i )
        goto LABEL_44;
    }
    if ( v53 )
    {
      v47 = v52;
      memcpy(v46, v52, v53);
      *(_QWORD *)(v55 + 24) += v53;
    }
  }
LABEL_44:
  v56 = sub_16B0440(v46, v47);
  v57 = *(_QWORD *)(v56 + 48);
  if ( v57 != *(_QWORD *)(v56 + 56) )
    *(_QWORD *)(v56 + 56) = v57;
  if ( base != v115 )
    _libc_free((unsigned __int64)base);
  if ( v110 != v112 )
    _libc_free((unsigned __int64)v110);
}
