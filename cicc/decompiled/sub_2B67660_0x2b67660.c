// Function: sub_2B67660
// Address: 0x2b67660
//
__int64 __fastcall sub_2B67660(__int64 a1, unsigned __int8 **a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r14
  __int64 v5; // r9
  unsigned __int8 *v6; // rax
  unsigned __int64 v7; // rcx
  __int64 v8; // rbx
  __int64 v9; // rdx
  _QWORD *v10; // r14
  __int64 v11; // rax
  unsigned __int64 *v12; // rax
  _QWORD *i; // r13
  _BYTE *v14; // rax
  __int64 v15; // r8
  __int64 v16; // rax
  _BYTE *v17; // rax
  __int64 v18; // r15
  unsigned __int8 *v19; // rax
  unsigned __int8 *v20; // rax
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 v26; // r9
  unsigned __int64 v27; // r12
  _QWORD *v28; // rax
  int v29; // r15d
  signed __int64 v30; // r9
  unsigned __int64 v31; // rbx
  unsigned __int64 *v32; // r12
  unsigned __int64 v33; // r12
  __int64 v35; // r8
  _BYTE *v36; // r13
  unsigned __int64 *v37; // rcx
  _QWORD *v38; // r14
  int *v39; // rsi
  int v40; // ebx
  unsigned __int64 v41; // rsi
  int *v42; // rdi
  __int64 v43; // rax
  __int64 v44; // rcx
  __int64 v45; // rdx
  __int64 v46; // rax
  unsigned int ***v47; // r14
  __int64 v48; // r12
  char v49; // r13
  int v50; // eax
  unsigned __int64 *v51; // rax
  __int64 v52; // rbx
  unsigned __int64 *v53; // rdx
  unsigned __int64 *v54; // rbx
  unsigned int **v55; // rdi
  __int64 v56; // r9
  __int64 v57; // rsi
  __int64 v58; // rdx
  int v59; // ecx
  __int64 v60; // rdi
  int v61; // ecx
  unsigned int v62; // edx
  __int64 *v63; // rax
  __int64 v64; // r10
  __int64 v65; // rax
  _BYTE *v66; // rbx
  _BYTE *v67; // r12
  __int64 v68; // rax
  __int64 v69; // r12
  _QWORD *v70; // r14
  __int64 v71; // rcx
  __int64 v72; // rdx
  __int64 v73; // r8
  unsigned __int64 *v74; // rax
  int v75; // eax
  int v76; // r8d
  __int64 v77; // rdx
  __int64 *v78; // r11
  __int64 v79; // rax
  __int64 v80; // rbx
  __int64 v81; // rax
  __int64 v82; // rbx
  __int64 *v83; // r12
  unsigned __int64 *v84; // rax
  _QWORD *v85; // r10
  _QWORD *v86; // rsi
  _QWORD *v87; // r10
  __int64 v88; // r11
  _QWORD *v89; // r10
  __int64 v90; // r11
  __int64 v91; // r11
  _QWORD *v92; // r10
  __int64 v93; // r11
  unsigned __int64 v94; // rbx
  __int64 v95; // rdi
  unsigned __int64 *v96; // rax
  __int64 v97; // rax
  _QWORD *v98; // r10
  _QWORD *v99; // rsi
  _QWORD *v100; // rdi
  _QWORD *v101; // r10
  unsigned __int64 v102; // [rsp-8h] [rbp-1A8h]
  unsigned __int8 v103; // [rsp+17h] [rbp-189h]
  __int64 v106; // [rsp+38h] [rbp-168h]
  _QWORD *v107; // [rsp+40h] [rbp-160h]
  __int64 v108; // [rsp+50h] [rbp-150h]
  __int64 v109; // [rsp+58h] [rbp-148h]
  __int64 v110; // [rsp+58h] [rbp-148h]
  __int64 v112; // [rsp+68h] [rbp-138h]
  __int64 v113; // [rsp+68h] [rbp-138h]
  __int64 v114; // [rsp+68h] [rbp-138h]
  int v115; // [rsp+68h] [rbp-138h]
  int v117; // [rsp+70h] [rbp-130h]
  unsigned __int64 v118; // [rsp+70h] [rbp-130h]
  int v119; // [rsp+84h] [rbp-11Ch] BYREF
  unsigned int v120; // [rsp+88h] [rbp-118h] BYREF
  int v121; // [rsp+8Ch] [rbp-114h] BYREF
  unsigned __int64 *v122; // [rsp+90h] [rbp-110h] BYREF
  __int64 v123; // [rsp+98h] [rbp-108h] BYREF
  int v124[2]; // [rsp+A0h] [rbp-100h] BYREF
  __int64 v125; // [rsp+A8h] [rbp-F8h]
  __int64 v126; // [rsp+B0h] [rbp-F0h]
  _QWORD *v127; // [rsp+B8h] [rbp-E8h]
  int v128; // [rsp+C0h] [rbp-E0h]
  int v129; // [rsp+C4h] [rbp-DCh]
  unsigned int **v130; // [rsp+D0h] [rbp-D0h] BYREF
  __int64 v131; // [rsp+D8h] [rbp-C8h]
  unsigned int *v132; // [rsp+E0h] [rbp-C0h] BYREF
  int *v133; // [rsp+E8h] [rbp-B8h]
  int *v134; // [rsp+F0h] [rbp-B0h]
  int *v135; // [rsp+F8h] [rbp-A8h]
  __int64 v136; // [rsp+100h] [rbp-A0h]
  __int64 v137; // [rsp+108h] [rbp-98h]
  unsigned __int64 *v138; // [rsp+110h] [rbp-90h] BYREF
  __int64 v139; // [rsp+118h] [rbp-88h]
  _BYTE v140[128]; // [rsp+120h] [rbp-80h] BYREF

  sub_2B24BA0((__int64 *)&v122, a3, a4, *a2[1] - 29);
  v4 = *(_QWORD *)(a1 + 3296);
  sub_2B08680(*((_QWORD *)*a2 + 1), a4);
  v103 = sub_DFA540(v4);
  if ( v103 )
    goto LABEL_41;
  v138 = (unsigned __int64 *)v140;
  v139 = 0x100000000LL;
  v6 = *a2;
  v109 = *((_DWORD *)*a2 + 1) & 0x7FFFFFF;
  if ( (*((_DWORD *)*a2 + 1) & 0x7FFFFFF) == 0 )
  {
    v36 = v140;
    v9 = 0;
    goto LABEL_64;
  }
  v7 = 1;
  v8 = 0;
  LODWORD(v9) = 0;
  v10 = (_QWORD *)(a3 + 8 * a4);
  v11 = 0;
LABEL_4:
  v12 = &v138[10 * v11];
  if ( v12 )
  {
    *v12 = (unsigned __int64)(v12 + 2);
    v12[1] = 0x800000000LL;
    LODWORD(v9) = v139;
  }
  v9 = (unsigned int)(v9 + 1);
  for ( LODWORD(v139) = v9; ; LODWORD(v139) = v139 + 1 )
  {
    for ( i = (_QWORD *)a3; v10 != i; v9 = (unsigned int)v139 )
    {
      while ( 1 )
      {
        v17 = (_BYTE *)*i;
        v18 = (__int64)&v138[10 * v9 - 10];
        if ( *(_BYTE *)*i == 13 )
          break;
        if ( (v17[7] & 0x40) != 0 )
          v14 = (_BYTE *)*((_QWORD *)v17 - 1);
        else
          v14 = &v17[-32 * (*((_DWORD *)v17 + 1) & 0x7FFFFFF)];
        v15 = *(_QWORD *)&v14[32 * (unsigned int)v8];
        v16 = *(unsigned int *)(v18 + 8);
        v7 = *(unsigned int *)(v18 + 12);
        if ( v16 + 1 > v7 )
        {
          v112 = v15;
          sub_C8D5F0(v18, (const void *)(v18 + 16), v16 + 1, 8u, v15, v5);
          v16 = *(unsigned int *)(v18 + 8);
          v15 = v112;
        }
        ++i;
        *(_QWORD *)(*(_QWORD *)v18 + 8 * v16) = v15;
        ++*(_DWORD *)(v18 + 8);
        v9 = (unsigned int)v139;
        if ( v10 == i )
          goto LABEL_21;
      }
      v19 = *a2;
      if ( ((*a2)[7] & 0x40) != 0 )
        v20 = (unsigned __int8 *)*((_QWORD *)v19 - 1);
      else
        v20 = &v19[-32 * (*((_DWORD *)v19 + 1) & 0x7FFFFFF)];
      v21 = sub_ACADE0(*(__int64 ***)(*(_QWORD *)&v20[32 * (unsigned int)v8] + 8LL));
      v22 = *(unsigned int *)(v18 + 8);
      if ( v22 + 1 > (unsigned __int64)*(unsigned int *)(v18 + 12) )
      {
        v113 = v21;
        sub_C8D5F0(v18, (const void *)(v18 + 16), v22 + 1, 8u, v22 + 1, v5);
        v22 = *(unsigned int *)(v18 + 8);
        v21 = v113;
      }
      v7 = *(_QWORD *)v18;
      ++i;
      *(_QWORD *)(*(_QWORD *)v18 + 8 * v22) = v21;
      ++*(_DWORD *)(v18 + 8);
    }
LABEL_21:
    if ( v109 == ++v8 )
      break;
    v7 = HIDWORD(v139);
    v11 = (unsigned int)v9;
    if ( HIDWORD(v139) > (unsigned int)v9 )
      goto LABEL_4;
    v27 = sub_C8D7D0((__int64)&v138, (__int64)v140, 0, 0x50u, (unsigned __int64 *)&v130, v5);
    v28 = (_QWORD *)(v27 + 80LL * (unsigned int)v139);
    if ( v28 )
    {
      v24 = (__int64)(v28 + 2);
      *v28 = v28 + 2;
      v28[1] = 0x800000000LL;
    }
    sub_2B42CC0((__int64)&v138, v27, v23, v24, v25, v26);
    v29 = (int)v130;
    if ( v138 != (unsigned __int64 *)v140 )
      _libc_free((unsigned __int64)v138);
    v138 = (unsigned __int64 *)v27;
    HIDWORD(v139) = v29;
    v9 = (unsigned int)(v139 + 1);
  }
  v35 = (__int64)v138;
  v36 = v138;
  if ( (_DWORD)v9 != 2 )
  {
    v6 = *a2;
LABEL_64:
    *(_QWORD *)v124 = 0;
    v125 = 0;
    v126 = 0;
    v127 = 0;
    v119 = 0;
    v120 = 0;
    v121 = 0;
LABEL_65:
    v56 = 80 * v9;
    goto LABEL_66;
  }
  if ( (_DWORD)a4 == 1 )
  {
    *(_QWORD *)v124 = 0;
    v125 = 0;
    v126 = 0;
    v127 = 0;
    v119 = 0;
    v120 = 0;
    v121 = 0;
    goto LABEL_83;
  }
  v37 = v138;
  v114 = 0;
  v38 = (_QWORD *)a1;
  while ( 1 )
  {
    v132 = 0;
    v133 = 0;
    v130 = &v132;
    v131 = 0x300000003LL;
    v134 = 0;
    v135 = 0;
    v136 = 0;
    v137 = 0;
    v106 = 8LL * (unsigned int)(v114 + 1);
    v39 = *(int **)(*v37 + v106);
    v108 = 8LL * (unsigned int)v114;
    v40 = 0;
    v132 = *(unsigned int **)(*v37 + v108);
    v133 = v39;
    v41 = v37[10];
    v42 = *(int **)(v41 + v106);
    v134 = *(int **)(*v37 + v108);
    v135 = v42;
    v43 = *(_QWORD *)(*v37 + v106);
    v44 = v38[413];
    v136 = *(_QWORD *)(v41 + v108);
    v45 = v38[418];
    v137 = v43;
    v46 = v38[411];
    *(_QWORD *)v124 = v44;
    v125 = v45;
    v126 = v46;
    v128 = 2;
    v117 = 0;
    v129 = qword_500F9A8;
    v127 = v38;
    v107 = v38;
    v47 = &v130;
    v48 = 0;
    v49 = 0;
    do
    {
      v50 = sub_2B65A50((__int64)v124, (__int64)v47[2], (__int64)v47[3], 0, 0, 1, 0, 0);
      v7 = v102;
      if ( v50 > v40 )
      {
        v117 = v48;
        v40 = v50;
        v49 = 1;
      }
      ++v48;
      v47 += 2;
    }
    while ( v48 != 3 );
    v38 = v107;
    if ( v49 )
    {
      if ( v117 == 1 )
      {
        v51 = v138;
        v52 = 8LL * (unsigned int)(v114 + 1);
        goto LABEL_58;
      }
      if ( v117 == 2 )
      {
        v51 = v138;
        v52 = 8LL * (unsigned int)v114;
LABEL_58:
        v53 = (unsigned __int64 *)(v52 + v51[10]);
        v54 = (unsigned __int64 *)(*v51 + v52);
        v7 = *v54;
        *v54 = *v53;
        *v53 = v7;
        v55 = v130;
        goto LABEL_59;
      }
      if ( v117 )
        BUG();
    }
    v55 = v130;
LABEL_59:
    if ( v55 != &v132 )
      _libc_free((unsigned __int64)v55);
    if ( (_DWORD)a4 - 1 == ++v114 )
      break;
    v37 = v138;
  }
  v9 = (unsigned int)v139;
  *(_QWORD *)v124 = 0;
  v125 = 0;
  v36 = v138;
  v126 = 0;
  v127 = 0;
  v119 = 0;
  v120 = 0;
  v121 = 0;
  if ( (_DWORD)v139 != 2 )
  {
    v6 = *a2;
    goto LABEL_65;
  }
LABEL_83:
  v69 = *((unsigned int *)v36 + 2);
  v70 = *(_QWORD **)v36;
  if ( v69 == *((_DWORD *)v36 + 22) )
  {
    v72 = 8 * v69;
    if ( !(8 * v69) || !memcmp(*(const void **)v36, *((const void **)v36 + 10), v72) )
    {
      sub_2B0F6D0((__int64)v36, (char **)v36 + 10, v72, v7, v35, v5);
      v73 = (__int64)v138;
      LODWORD(v139) = v139 - 1;
      v74 = &v138[10 * (unsigned int)v139];
      if ( (unsigned __int64 *)*v74 != v74 + 2 )
      {
        _libc_free(*v74);
        v73 = (__int64)v138;
      }
      goto LABEL_92;
    }
  }
  if ( sub_2B0D880(v70, v69, (unsigned __int8 (__fastcall *)(_QWORD))sub_2B0D8B0) )
  {
    v36 = v138;
    goto LABEL_86;
  }
  v73 = (__int64)v138;
  v77 = (unsigned int)v139;
  v78 = (__int64 *)*v138;
  v36 = v138;
  v115 = v139;
  v79 = 8LL * *((unsigned int *)v138 + 2);
  v118 = 80LL * (unsigned int)v139;
  v56 = v118;
  v80 = *v138 + v79;
  v81 = v79 >> 5;
  v110 = v80;
  v82 = v118 - 80;
  if ( !v81 )
  {
LABEL_125:
    v97 = v110 - (_QWORD)v78;
    if ( v110 - (_QWORD)v78 == 16 )
    {
      v101 = *(_QWORD **)(v73 + v82);
      v99 = &v101[*(unsigned int *)(v73 + v82 + 8)];
    }
    else
    {
      if ( v97 != 24 )
      {
        if ( v97 != 8 )
          goto LABEL_118;
        v98 = *(_QWORD **)(v73 + v82);
        v99 = &v98[*(unsigned int *)(v73 + v82 + 8)];
        goto LABEL_129;
      }
      v130 = (unsigned int **)*v78;
      v100 = *(_QWORD **)(v73 + v82);
      v99 = &v100[*(unsigned int *)(v73 + v82 + 8)];
      if ( v99 == sub_2B0BE70(v100, (__int64)v99, (__int64 *)&v130) )
        goto LABEL_117;
      v78 = (__int64 *)(v93 + 8);
    }
    v130 = (unsigned int **)*v78;
    if ( v99 == sub_2B0BE70(v101, (__int64)v99, (__int64 *)&v130) )
      goto LABEL_117;
    v78 = (__int64 *)(v93 + 8);
LABEL_129:
    v130 = (unsigned int **)*v78;
    if ( v99 == sub_2B0BE70(v98, (__int64)v99, (__int64 *)&v130) )
      goto LABEL_117;
    goto LABEL_118;
  }
  v83 = &v78[4 * v81];
  v82 = v118 - 80;
  v84 = &v138[v118 / 8 - 10];
  v85 = (_QWORD *)*v84;
  v86 = (_QWORD *)(*v84 + 8LL * *((unsigned int *)v84 + 2));
  while ( 1 )
  {
    v130 = (unsigned int **)*v78;
    if ( v86 == sub_2B0BE70(v85, (__int64)v86, (__int64 *)&v130) )
      break;
    v130 = *(unsigned int ***)(v93 + 8);
    if ( v86 == sub_2B0BE70(v92, (__int64)v86, (__int64 *)&v130) )
    {
      v93 = v88 + 8;
      break;
    }
    v130 = *(unsigned int ***)(v88 + 16);
    if ( v86 == sub_2B0BE70(v87, (__int64)v86, (__int64 *)&v130) )
    {
      v93 = v90 + 16;
      break;
    }
    v130 = *(unsigned int ***)(v90 + 24);
    if ( v86 == sub_2B0BE70(v89, (__int64)v86, (__int64 *)&v130) )
    {
      v93 = v91 + 24;
      break;
    }
    v78 = (__int64 *)(v91 + 32);
    if ( v78 == v83 )
      goto LABEL_125;
  }
LABEL_117:
  if ( v110 == v93 )
  {
LABEL_118:
    v94 = 0xCCCCCCCCCCCCCCCDLL * (v82 >> 4);
    if ( v118 > 0x50 )
    {
      v95 = (__int64)v36;
      do
      {
        sub_2B0F6D0(v95, (char **)(v95 + 80), v77, v71, v73, v56);
        v95 += 80;
        --v94;
      }
      while ( v94 );
      v73 = (__int64)v138;
      v115 = v139;
    }
    LODWORD(v139) = v115 - 1;
    v96 = (unsigned __int64 *)(v73 + 80LL * (unsigned int)(v115 - 1));
    if ( (unsigned __int64 *)*v96 != v96 + 2 )
    {
      _libc_free(*v96);
      v73 = (__int64)v138;
    }
    ++v121;
LABEL_92:
    v36 = (_BYTE *)v73;
LABEL_86:
    v56 = 80LL * (unsigned int)v139;
  }
  v6 = *a2;
LABEL_66:
  v57 = *((_QWORD *)v6 + 5);
  v58 = *(_QWORD *)(a1 + 3312);
  v59 = *(_DWORD *)(v58 + 24);
  v60 = *(_QWORD *)(v58 + 8);
  if ( v59 )
  {
    v61 = v59 - 1;
    v62 = v61 & (((unsigned int)v57 >> 9) ^ ((unsigned int)v57 >> 4));
    v63 = (__int64 *)(v60 + 16LL * v62);
    v64 = *v63;
    if ( v57 == *v63 )
    {
LABEL_68:
      v65 = v63[1];
      goto LABEL_69;
    }
    v75 = 1;
    while ( v64 != -4096 )
    {
      v76 = v75 + 1;
      v62 = v61 & (v75 + v62);
      v63 = (__int64 *)(v60 + 16LL * v62);
      v64 = *v63;
      if ( v57 == *v63 )
        goto LABEL_68;
      v75 = v76;
    }
  }
  v65 = 0;
LABEL_69:
  v123 = v65;
  v66 = &v36[v56];
  v134 = v124;
  v130 = (unsigned int **)a1;
  v131 = (__int64)&v123;
  v132 = &v120;
  v133 = &v121;
  v135 = &v119;
  v30 = 0xCCCCCCCCCCCCCCCDLL * (v56 >> 4);
  if ( !(v30 >> 2) )
  {
LABEL_30:
    if ( v30 != 2 )
    {
      if ( v30 != 3 )
      {
        if ( v30 != 1 )
        {
LABEL_33:
          v103 = 1;
          goto LABEL_34;
        }
LABEL_101:
        if ( (unsigned __int8)sub_2B66F90((__int64 *)&v130, *(unsigned __int8 ***)v36, *((unsigned int *)v36 + 2)) )
          goto LABEL_76;
        goto LABEL_33;
      }
      if ( (unsigned __int8)sub_2B66F90((__int64 *)&v130, *(unsigned __int8 ***)v36, *((unsigned int *)v36 + 2)) )
        goto LABEL_76;
      v36 += 80;
    }
    if ( (unsigned __int8)sub_2B66F90((__int64 *)&v130, *(unsigned __int8 ***)v36, *((unsigned int *)v36 + 2)) )
      goto LABEL_76;
    v36 += 80;
    goto LABEL_101;
  }
  v67 = &v36[320 * (v30 >> 2)];
  while ( !(unsigned __int8)sub_2B66F90((__int64 *)&v130, *(unsigned __int8 ***)v36, *((unsigned int *)v36 + 2)) )
  {
    if ( (unsigned __int8)sub_2B66F90((__int64 *)&v130, *((unsigned __int8 ***)v36 + 10), *((unsigned int *)v36 + 22)) )
    {
      v36 += 80;
      break;
    }
    if ( (unsigned __int8)sub_2B66F90((__int64 *)&v130, *((unsigned __int8 ***)v36 + 20), *((unsigned int *)v36 + 42)) )
    {
      v36 += 160;
      break;
    }
    if ( (unsigned __int8)sub_2B66F90((__int64 *)&v130, *((unsigned __int8 ***)v36 + 30), *((unsigned int *)v36 + 62)) )
    {
      v36 += 240;
      break;
    }
    v36 += 320;
    if ( v36 == v67 )
    {
      v30 = 0xCCCCCCCCCCCCCCCDLL * ((v66 - v36) >> 4);
      goto LABEL_30;
    }
  }
LABEL_76:
  if ( v66 == v36 )
    goto LABEL_33;
  v68 = *((_DWORD *)*a2 + 1) & 0x7FFFFFF;
  if ( v120 < (unsigned __int64)(v68 * (a4 - 1)) )
    v103 = (unsigned int)(v119 + v126 + v121 + 3) < (unsigned __int64)(a4 * v68);
LABEL_34:
  sub_C7D6A0(v125, 4LL * (unsigned int)v127, 4);
  v31 = (unsigned __int64)v138;
  v32 = &v138[10 * (unsigned int)v139];
  if ( v138 != v32 )
  {
    do
    {
      v32 -= 10;
      if ( (unsigned __int64 *)*v32 != v32 + 2 )
        _libc_free(*v32);
    }
    while ( (unsigned __int64 *)v31 != v32 );
    v32 = v138;
  }
  if ( v32 != (unsigned __int64 *)v140 )
    _libc_free((unsigned __int64)v32);
LABEL_41:
  v33 = (unsigned __int64)v122;
  if ( ((unsigned __int8)v122 & 1) == 0 && v122 )
  {
    if ( (unsigned __int64 *)*v122 != v122 + 2 )
      _libc_free(*v122);
    j_j___libc_free_0(v33);
  }
  return v103;
}
