// Function: sub_35568E0
// Address: 0x35568e0
//
__int64 __fastcall sub_35568E0(_QWORD *a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 v3; // rax
  __int64 *v4; // rsi
  _QWORD *v5; // rdx
  __int64 *v6; // rsi
  __int64 *v7; // rbx
  __int64 *v8; // r14
  __int64 v9; // r9
  int v10; // r11d
  _QWORD *v11; // r10
  __int64 v12; // r8
  unsigned int v13; // eax
  _QWORD *v14; // rdi
  __int64 v15; // rcx
  unsigned int v16; // eax
  __int64 v17; // rdi
  int v18; // edx
  __int64 v19; // r15
  __int64 v20; // rax
  unsigned __int64 v21; // rdx
  _QWORD *v22; // rdx
  __int64 v23; // r9
  __int64 *v24; // rbx
  __int64 *v25; // r12
  __int64 v26; // rsi
  _QWORD *v27; // rdx
  __int64 v28; // r8
  __int64 v29; // r9
  __int64 *v30; // rbx
  __int64 *v31; // r12
  __int64 v32; // rsi
  __int64 v33; // r14
  __int64 v34; // r13
  unsigned int v35; // eax
  __int64 v36; // rsi
  __int64 v37; // r8
  __int64 v38; // r9
  __int64 v39; // rdx
  char *v40; // r10
  unsigned __int64 v41; // rcx
  unsigned __int64 v42; // rsi
  int v43; // eax
  __int64 v44; // rcx
  __int64 v45; // r8
  __int64 v46; // r9
  char *v47; // r10
  __int64 v48; // rcx
  __int64 v49; // rax
  __int64 v50; // rax
  char *v51; // r10
  void *v52; // rdi
  __int64 v53; // rdx
  __int64 v54; // rdx
  int v56; // r11d
  __int64 v57; // rax
  __int64 v58; // rdi
  __int64 *v59; // r14
  __int64 v60; // r11
  unsigned int v61; // eax
  __int64 *v62; // r14
  __int64 *v63; // r14
  __int64 v64; // r11
  unsigned int v65; // eax
  __int64 *v66; // r14
  int v67; // edi
  __int64 *v68; // rbx
  __int64 *v69; // rsi
  __int64 *v70; // rbx
  __int64 *v71; // rsi
  char *v72; // rbx
  __int64 v73; // rdx
  unsigned __int64 v74; // rcx
  unsigned __int64 v75; // rsi
  int v76; // eax
  __int64 v77; // r13
  __int64 v78; // rdx
  __int64 v79; // rcx
  __int64 v80; // r8
  __int64 v81; // r9
  __int64 v82; // rax
  void *v83; // rax
  __int64 v84; // rdx
  char *v85; // rbx
  __int64 v86; // rdx
  unsigned __int64 v87; // rcx
  unsigned __int64 v88; // rsi
  int v89; // eax
  __int64 v90; // r13
  __int64 v91; // rdx
  __int64 v92; // r8
  __int64 v93; // r9
  __int64 v94; // rax
  void *v95; // rax
  __int64 v96; // rdx
  __int64 v97; // rcx
  int v98; // r11d
  char *v99; // rbx
  char *v100; // rbx
  __int64 v102; // [rsp+18h] [rbp-1C8h]
  __int64 v103; // [rsp+18h] [rbp-1C8h]
  __int64 v104; // [rsp+20h] [rbp-1C0h]
  __int64 v105; // [rsp+20h] [rbp-1C0h]
  char *v106; // [rsp+20h] [rbp-1C0h]
  __int64 v107; // [rsp+20h] [rbp-1C0h]
  char *v108; // [rsp+20h] [rbp-1C0h]
  char *v109; // [rsp+30h] [rbp-1B0h]
  __int64 v110; // [rsp+30h] [rbp-1B0h]
  char *v111; // [rsp+30h] [rbp-1B0h]
  __int64 *v112; // [rsp+30h] [rbp-1B0h]
  __int64 *v113; // [rsp+30h] [rbp-1B0h]
  __int64 v114; // [rsp+30h] [rbp-1B0h]
  char *v115; // [rsp+30h] [rbp-1B0h]
  __int64 v117; // [rsp+50h] [rbp-190h] BYREF
  __int64 v118; // [rsp+58h] [rbp-188h]
  __int64 v119; // [rsp+60h] [rbp-180h]
  __int64 v120; // [rsp+68h] [rbp-178h]
  __int64 *v121; // [rsp+70h] [rbp-170h] BYREF
  __int64 v122; // [rsp+78h] [rbp-168h]
  __int64 v123; // [rsp+80h] [rbp-160h] BYREF
  __int64 v124; // [rsp+88h] [rbp-158h]
  __int64 v125; // [rsp+90h] [rbp-150h]
  __int64 v126; // [rsp+98h] [rbp-148h]
  __int64 *v127; // [rsp+A0h] [rbp-140h]
  __int64 v128; // [rsp+A8h] [rbp-138h]
  _BYTE v129[4]; // [rsp+B0h] [rbp-130h] BYREF
  __int64 v130; // [rsp+B4h] [rbp-12Ch]
  __int64 v131; // [rsp+BCh] [rbp-124h]
  __int64 v132; // [rsp+C8h] [rbp-118h]
  int v133; // [rsp+D0h] [rbp-110h]
  __int64 v134; // [rsp+E0h] [rbp-100h] BYREF
  void *s; // [rsp+E8h] [rbp-F8h]
  _BYTE v136[12]; // [rsp+F0h] [rbp-F0h]
  char v137; // [rsp+FCh] [rbp-E4h]
  char v138; // [rsp+100h] [rbp-E0h] BYREF
  __int64 v139; // [rsp+140h] [rbp-A0h] BYREF
  __int64 v140; // [rsp+148h] [rbp-98h]
  __int64 v141; // [rsp+150h] [rbp-90h]
  __int64 v142; // [rsp+158h] [rbp-88h]
  __int64 *v143; // [rsp+160h] [rbp-80h]
  __int64 v144; // [rsp+168h] [rbp-78h]
  _BYTE v145[112]; // [rsp+170h] [rbp-70h] BYREF

  v2 = *(_QWORD *)a2;
  v121 = &v123;
  s = &v138;
  v3 = *(unsigned int *)(a2 + 8);
  v117 = 0;
  v118 = 0;
  v104 = v2 + 88 * v3;
  v119 = 0;
  v120 = 0;
  v122 = 0;
  v134 = 0;
  *(_QWORD *)v136 = 8;
  *(_DWORD *)&v136[8] = 0;
  v137 = 1;
  if ( v2 == v104 )
    goto LABEL_20;
  do
  {
    v4 = &v139;
    v139 = 0;
    v143 = (__int64 *)v145;
    v144 = 0x800000000LL;
    v140 = 0;
    v5 = (_QWORD *)a1[433];
    v141 = 0;
    v142 = 0;
    if ( !sub_35543E0(v2, (__int64)&v139, v5, 0) )
      goto LABEL_3;
    v123 = 0;
    v124 = 0;
    v127 = (__int64 *)v129;
    v125 = 0;
    v126 = 0;
    v128 = 0;
    if ( v143 == &v143[(unsigned int)v144] )
      goto LABEL_80;
    v112 = &v143[(unsigned int)v144];
    v59 = v143;
    do
    {
      v60 = *v59;
      ++v134;
      if ( v137 )
        goto LABEL_75;
      v102 = v60;
      v61 = 4 * (*(_DWORD *)&v136[4] - *(_DWORD *)&v136[8]);
      if ( v61 < 0x20 )
        v61 = 32;
      if ( *(_DWORD *)v136 <= v61 )
      {
        memset(s, -1, 8LL * *(unsigned int *)v136);
        v60 = v102;
LABEL_75:
        *(_QWORD *)&v136[4] = 0;
        goto LABEL_76;
      }
      sub_C8C990((__int64)&v134, (__int64)v4);
      v60 = v102;
LABEL_76:
      v4 = &v123;
      ++v59;
      sub_3554FD0(v60, (__int64)&v123, (__int64)&v117, v2, (__int64)&v134, (_QWORD *)a1[433]);
    }
    while ( v112 != v59 );
    v62 = v127;
    if ( (_DWORD)v128 )
    {
      v70 = &v127[(unsigned int)v128];
      do
      {
        v71 = v62++;
        sub_3554C70(v2, v71);
      }
      while ( v62 != v70 );
      v62 = v127;
    }
    if ( v62 != (__int64 *)v129 )
      _libc_free((unsigned __int64)v62);
LABEL_80:
    sub_C7D6A0(v124, 8LL * (unsigned int)v126, 8);
LABEL_3:
    sub_35480E0((__int64)&v139);
    v6 = &v139;
    LODWORD(v144) = 0;
    if ( !sub_35543E0((__int64)&v117, (__int64)&v139, (_QWORD *)a1[433], 0) )
      goto LABEL_4;
    v123 = 0;
    v124 = 0;
    v127 = (__int64 *)v129;
    v125 = 0;
    v126 = 0;
    v128 = 0;
    if ( &v143[(unsigned int)v144] == v143 )
      goto LABEL_93;
    v113 = &v143[(unsigned int)v144];
    v63 = v143;
    while ( 2 )
    {
      v64 = *v63;
      ++v134;
      if ( v137 )
      {
LABEL_88:
        *(_QWORD *)&v136[4] = 0;
      }
      else
      {
        v103 = v64;
        v65 = 4 * (*(_DWORD *)&v136[4] - *(_DWORD *)&v136[8]);
        if ( v65 < 0x20 )
          v65 = 32;
        if ( *(_DWORD *)v136 <= v65 )
        {
          memset(s, -1, 8LL * *(unsigned int *)v136);
          v64 = v103;
          goto LABEL_88;
        }
        sub_C8C990((__int64)&v134, (__int64)v6);
        v64 = v103;
      }
      v6 = &v123;
      ++v63;
      sub_3554FD0(v64, (__int64)&v123, v2, (__int64)&v117, (__int64)&v134, (_QWORD *)a1[433]);
      if ( v113 != v63 )
        continue;
      break;
    }
    v66 = v127;
    if ( (_DWORD)v128 )
    {
      v68 = &v127[(unsigned int)v128];
      do
      {
        v69 = v66++;
        sub_3554C70(v2, v69);
      }
      while ( v66 != v68 );
      v66 = v127;
    }
    if ( v66 != (__int64 *)v129 )
      _libc_free((unsigned __int64)v66);
LABEL_93:
    sub_C7D6A0(v124, 8LL * (unsigned int)v126, 8);
LABEL_4:
    v7 = *(__int64 **)(v2 + 32);
    v8 = &v7[*(unsigned int *)(v2 + 40)];
    if ( v7 != v8 )
    {
      while ( (_DWORD)v120 )
      {
        v9 = (unsigned int)(v120 - 1);
        v10 = 1;
        v11 = 0;
        v12 = v118;
        v13 = v9 & (((unsigned int)*v7 >> 9) ^ ((unsigned int)*v7 >> 4));
        v14 = (_QWORD *)(v118 + 8LL * v13);
        v15 = *v14;
        if ( *v7 == *v14 )
        {
LABEL_7:
          if ( ++v7 == v8 )
            goto LABEL_17;
        }
        else
        {
          while ( v15 != -4096 )
          {
            if ( v15 != -8192 || v11 )
              v14 = v11;
            v13 = v9 & (v10 + v13);
            v15 = *(_QWORD *)(v118 + 8LL * v13);
            if ( *v7 == v15 )
              goto LABEL_7;
            ++v10;
            v11 = v14;
            v14 = (_QWORD *)(v118 + 8LL * v13);
          }
          if ( !v11 )
            v11 = v14;
          ++v117;
          v18 = v119 + 1;
          if ( 4 * ((int)v119 + 1) < (unsigned int)(3 * v120) )
          {
            if ( (int)v120 - HIDWORD(v119) - v18 > (unsigned int)v120 >> 3 )
              goto LABEL_12;
            sub_3553650((__int64)&v117, v120);
            if ( !(_DWORD)v120 )
            {
LABEL_152:
              LODWORD(v119) = v119 + 1;
              BUG();
            }
            v12 = v118;
            v9 = 0;
            v56 = 1;
            LODWORD(v57) = (v120 - 1) & (((unsigned int)*v7 >> 9) ^ ((unsigned int)*v7 >> 4));
            v11 = (_QWORD *)(v118 + 8LL * (unsigned int)v57);
            v58 = *v11;
            v18 = v119 + 1;
            if ( *v11 == *v7 )
              goto LABEL_12;
            while ( v58 != -4096 )
            {
              if ( v58 == -8192 && !v9 )
                v9 = (__int64)v11;
              v57 = ((_DWORD)v120 - 1) & (unsigned int)(v57 + v56);
              v11 = (_QWORD *)(v118 + 8 * v57);
              v58 = *v11;
              if ( *v7 == *v11 )
                goto LABEL_12;
              ++v56;
            }
            goto LABEL_65;
          }
LABEL_10:
          sub_3553650((__int64)&v117, 2 * v120);
          if ( !(_DWORD)v120 )
            goto LABEL_152;
          v12 = v118;
          v16 = (v120 - 1) & (((unsigned int)*v7 >> 9) ^ ((unsigned int)*v7 >> 4));
          v11 = (_QWORD *)(v118 + 8LL * v16);
          v17 = *v11;
          v18 = v119 + 1;
          if ( *v7 == *v11 )
            goto LABEL_12;
          v98 = 1;
          v9 = 0;
          while ( v17 != -4096 )
          {
            if ( v9 || v17 != -8192 )
              v11 = (_QWORD *)v9;
            v9 = (unsigned int)(v98 + 1);
            v16 = (v120 - 1) & (v98 + v16);
            v17 = *(_QWORD *)(v118 + 8LL * v16);
            if ( *v7 == v17 )
            {
              v11 = (_QWORD *)(v118 + 8LL * v16);
              goto LABEL_12;
            }
            ++v98;
            v9 = (__int64)v11;
            v11 = (_QWORD *)(v118 + 8LL * v16);
          }
LABEL_65:
          if ( v9 )
            v11 = (_QWORD *)v9;
LABEL_12:
          LODWORD(v119) = v18;
          if ( *v11 != -4096 )
            --HIDWORD(v119);
          v19 = *v7;
          *v11 = *v7;
          v20 = (unsigned int)v122;
          v21 = (unsigned int)v122 + 1LL;
          if ( v21 > HIDWORD(v122) )
          {
            sub_C8D5F0((__int64)&v121, &v123, v21, 8u, v12, v9);
            v20 = (unsigned int)v122;
          }
          ++v7;
          v121[v20] = v19;
          LODWORD(v122) = v122 + 1;
          if ( v7 == v8 )
            goto LABEL_17;
        }
      }
      ++v117;
      goto LABEL_10;
    }
LABEL_17:
    if ( v143 != (__int64 *)v145 )
      _libc_free((unsigned __int64)v143);
    v2 += 88;
    sub_C7D6A0(v140, 8LL * (unsigned int)v142, 8);
  }
  while ( v104 != v2 );
LABEL_20:
  v123 = 0;
  v143 = (__int64 *)v145;
  v144 = 0x800000000LL;
  v124 = 0;
  v22 = (_QWORD *)a1[433];
  v127 = (__int64 *)v129;
  v125 = 0;
  v126 = 0;
  v128 = 0;
  v129[0] = 0;
  v130 = 0;
  v131 = 0;
  v132 = 0;
  v133 = 0;
  v139 = 0;
  v140 = 0;
  v141 = 0;
  v142 = 0;
  if ( sub_35543E0((__int64)&v117, (__int64)&v139, v22, 0) )
  {
    v24 = &v143[(unsigned int)v144];
    if ( v24 != v143 )
    {
      v25 = v143;
      do
      {
        v26 = *v25++;
        sub_3554DE0((__int64)a1, v26, (__int64)&v123, (__int64)&v117);
      }
      while ( v24 != v25 );
    }
  }
  if ( (_DWORD)v128 )
  {
    v72 = (char *)&v123;
    v73 = *(unsigned int *)(a2 + 8);
    v74 = *(_QWORD *)a2;
    v75 = v73 + 1;
    v76 = *(_DWORD *)(a2 + 8);
    if ( v73 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
    {
      if ( v74 > (unsigned __int64)&v123 || (unsigned __int64)&v123 >= v74 + 88 * v73 )
      {
        sub_35498F0(a2, v75, v73, v74, (unsigned int)v128, v23);
        v73 = *(unsigned int *)(a2 + 8);
        v74 = *(_QWORD *)a2;
        v72 = (char *)&v123;
        v76 = *(_DWORD *)(a2 + 8);
      }
      else
      {
        v99 = (char *)&v123 - v74;
        sub_35498F0(a2, v75, v73, v74, (unsigned int)v128, v23);
        v74 = *(_QWORD *)a2;
        v73 = *(unsigned int *)(a2 + 8);
        v72 = &v99[*(_QWORD *)a2];
        v76 = *(_DWORD *)(a2 + 8);
      }
    }
    v77 = v74 + 88 * v73;
    if ( v77 )
    {
      *(_QWORD *)v77 = 0;
      *(_QWORD *)(v77 + 8) = 0;
      *(_QWORD *)(v77 + 16) = 0;
      *(_DWORD *)(v77 + 24) = 0;
      sub_C7D6A0(0, 0, 8);
      v82 = *((unsigned int *)v72 + 6);
      *(_DWORD *)(v77 + 24) = v82;
      if ( (_DWORD)v82 )
      {
        v83 = (void *)sub_C7D670(8 * v82, 8);
        v84 = *(unsigned int *)(v77 + 24);
        *(_QWORD *)(v77 + 8) = v83;
        *(_DWORD *)(v77 + 16) = *((_DWORD *)v72 + 4);
        *(_DWORD *)(v77 + 20) = *((_DWORD *)v72 + 5);
        memcpy(v83, *((const void **)v72 + 1), 8 * v84);
      }
      else
      {
        *(_QWORD *)(v77 + 8) = 0;
        *(_QWORD *)(v77 + 16) = 0;
      }
      *(_QWORD *)(v77 + 40) = 0;
      *(_QWORD *)(v77 + 32) = v77 + 48;
      if ( *((_DWORD *)v72 + 10) )
        sub_353DD30(v77 + 32, (__int64)(v72 + 32), v78, v79, v80, v81);
      *(_BYTE *)(v77 + 48) = v72[48];
      *(_QWORD *)(v77 + 52) = *(_QWORD *)(v72 + 52);
      *(_QWORD *)(v77 + 60) = *(_QWORD *)(v72 + 60);
      *(_QWORD *)(v77 + 72) = *((_QWORD *)v72 + 9);
      *(_DWORD *)(v77 + 80) = *((_DWORD *)v72 + 20);
      v76 = *(_DWORD *)(a2 + 8);
    }
    *(_DWORD *)(a2 + 8) = v76 + 1;
  }
  sub_35480E0((__int64)&v123);
  LODWORD(v128) = 0;
  v27 = (_QWORD *)a1[433];
  v129[0] = 0;
  v130 = 0;
  v131 = 0;
  v132 = 0;
  if ( sub_35540D0((__int64)&v117, (__int64)&v139, v27, 0) )
  {
    v30 = &v143[(unsigned int)v144];
    if ( v30 != v143 )
    {
      v31 = v143;
      do
      {
        v32 = *v31++;
        sub_3554DE0((__int64)a1, v32, (__int64)&v123, (__int64)&v117);
      }
      while ( v30 != v31 );
    }
  }
  if ( (_DWORD)v128 )
  {
    v85 = (char *)&v123;
    v86 = *(unsigned int *)(a2 + 8);
    v87 = *(_QWORD *)a2;
    v88 = v86 + 1;
    v89 = *(_DWORD *)(a2 + 8);
    if ( v86 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
    {
      if ( v87 > (unsigned __int64)&v123 || (unsigned __int64)&v123 >= v87 + 88 * v86 )
      {
        sub_35498F0(a2, v88, v86, v87, v28, v29);
        v86 = *(unsigned int *)(a2 + 8);
        v87 = *(_QWORD *)a2;
        v85 = (char *)&v123;
        v89 = *(_DWORD *)(a2 + 8);
      }
      else
      {
        v100 = (char *)&v123 - v87;
        sub_35498F0(a2, v88, v86, v87, v28, v29);
        v87 = *(_QWORD *)a2;
        v86 = *(unsigned int *)(a2 + 8);
        v85 = &v100[*(_QWORD *)a2];
        v89 = *(_DWORD *)(a2 + 8);
      }
    }
    v90 = v87 + 88 * v86;
    if ( v90 )
    {
      *(_QWORD *)v90 = 0;
      *(_QWORD *)(v90 + 8) = 0;
      *(_QWORD *)(v90 + 16) = 0;
      *(_DWORD *)(v90 + 24) = 0;
      sub_C7D6A0(0, 0, 8);
      v94 = *((unsigned int *)v85 + 6);
      *(_DWORD *)(v90 + 24) = v94;
      if ( (_DWORD)v94 )
      {
        v95 = (void *)sub_C7D670(8 * v94, 8);
        v96 = *(unsigned int *)(v90 + 24);
        *(_QWORD *)(v90 + 8) = v95;
        *(_DWORD *)(v90 + 16) = *((_DWORD *)v85 + 4);
        *(_DWORD *)(v90 + 20) = *((_DWORD *)v85 + 5);
        memcpy(v95, *((const void **)v85 + 1), 8 * v96);
      }
      else
      {
        *(_QWORD *)(v90 + 8) = 0;
        *(_QWORD *)(v90 + 16) = 0;
      }
      *(_QWORD *)(v90 + 40) = 0;
      *(_QWORD *)(v90 + 32) = v90 + 48;
      v97 = *((unsigned int *)v85 + 10);
      if ( (_DWORD)v97 )
        sub_353DD30(v90 + 32, (__int64)(v85 + 32), v91, v97, v92, v93);
      *(_BYTE *)(v90 + 48) = v85[48];
      *(_QWORD *)(v90 + 52) = *(_QWORD *)(v85 + 52);
      *(_QWORD *)(v90 + 60) = *(_QWORD *)(v85 + 60);
      *(_QWORD *)(v90 + 72) = *((_QWORD *)v85 + 9);
      *(_DWORD *)(v90 + 80) = *((_DWORD *)v85 + 20);
      v89 = *(_DWORD *)(a2 + 8);
    }
    *(_DWORD *)(a2 + 8) = v89 + 1;
  }
  v33 = a1[7];
  v34 = a1[6];
  if ( v33 != v34 )
  {
    while ( 1 )
    {
      if ( !(_DWORD)v120 )
        goto LABEL_35;
      v35 = (v120 - 1) & (((unsigned int)v34 >> 9) ^ ((unsigned int)v34 >> 4));
      v36 = *(_QWORD *)(v118 + 8LL * v35);
      if ( v36 == v34 )
      {
LABEL_33:
        v34 += 256;
        if ( v33 == v34 )
          break;
      }
      else
      {
        v67 = 1;
        while ( v36 != -4096 )
        {
          v35 = (v120 - 1) & (v67 + v35);
          v36 = *(_QWORD *)(v118 + 8LL * v35);
          if ( v34 == v36 )
            goto LABEL_33;
          ++v67;
        }
LABEL_35:
        sub_35480E0((__int64)&v123);
        LODWORD(v128) = 0;
        v129[0] = 0;
        v130 = 0;
        v131 = 0;
        v132 = 0;
        sub_3554DE0((__int64)a1, v34, (__int64)&v123, (__int64)&v117);
        if ( !(_DWORD)v128 )
          goto LABEL_33;
        v39 = *(unsigned int *)(a2 + 8);
        v40 = (char *)&v123;
        v41 = *(_QWORD *)a2;
        v42 = v39 + 1;
        v43 = *(_DWORD *)(a2 + 8);
        if ( v39 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
        {
          if ( v41 > (unsigned __int64)&v123 || (unsigned __int64)&v123 >= v41 + 88 * v39 )
          {
            sub_35498F0(a2, v42, v39, v41, v37, v38);
            v39 = *(unsigned int *)(a2 + 8);
            v41 = *(_QWORD *)a2;
            v40 = (char *)&v123;
            v43 = *(_DWORD *)(a2 + 8);
          }
          else
          {
            v115 = (char *)&v123 - v41;
            sub_35498F0(a2, v42, v39, v41, v37, v38);
            v41 = *(_QWORD *)a2;
            v39 = *(unsigned int *)(a2 + 8);
            v40 = &v115[*(_QWORD *)a2];
            v43 = *(_DWORD *)(a2 + 8);
          }
        }
        v109 = v40;
        v44 = v41 + 88 * v39;
        if ( v44 )
        {
          *(_QWORD *)v44 = 0;
          *(_QWORD *)(v44 + 8) = 0;
          *(_QWORD *)(v44 + 16) = 0;
          *(_DWORD *)(v44 + 24) = 0;
          v105 = v44;
          sub_C7D6A0(0, 0, 8);
          v47 = v109;
          v48 = v105;
          v49 = *((unsigned int *)v109 + 6);
          *(_DWORD *)(v105 + 24) = v49;
          if ( (_DWORD)v49 )
          {
            v106 = v109;
            v110 = v48;
            v50 = sub_C7D670(8 * v49, 8);
            v51 = v106;
            v52 = (void *)v50;
            *(_QWORD *)(v110 + 8) = v50;
            LODWORD(v50) = *((_DWORD *)v106 + 4);
            v53 = *(unsigned int *)(v110 + 24);
            v107 = v110;
            *(_DWORD *)(v110 + 16) = v50;
            v111 = v51;
            *(_DWORD *)(v107 + 20) = *((_DWORD *)v51 + 5);
            memcpy(v52, *((const void **)v51 + 1), 8 * v53);
            v48 = v107;
            v47 = v111;
          }
          else
          {
            *(_QWORD *)(v105 + 8) = 0;
            *(_QWORD *)(v105 + 16) = 0;
          }
          *(_QWORD *)(v48 + 40) = 0;
          *(_QWORD *)(v48 + 32) = v48 + 48;
          v54 = *((unsigned int *)v47 + 10);
          if ( (_DWORD)v54 )
          {
            v108 = v47;
            v114 = v48;
            sub_353DD30(v48 + 32, (__int64)(v47 + 32), v54, v48, v45, v46);
            v47 = v108;
            v48 = v114;
          }
          *(_BYTE *)(v48 + 48) = v47[48];
          *(_QWORD *)(v48 + 52) = *(_QWORD *)(v47 + 52);
          *(_QWORD *)(v48 + 60) = *(_QWORD *)(v47 + 60);
          *(_QWORD *)(v48 + 72) = *((_QWORD *)v47 + 9);
          *(_DWORD *)(v48 + 80) = *((_DWORD *)v47 + 20);
          v43 = *(_DWORD *)(a2 + 8);
        }
        v34 += 256;
        *(_DWORD *)(a2 + 8) = v43 + 1;
        if ( v33 == v34 )
          break;
      }
    }
  }
  if ( v143 != (__int64 *)v145 )
    _libc_free((unsigned __int64)v143);
  sub_C7D6A0(v140, 8LL * (unsigned int)v142, 8);
  if ( v127 != (__int64 *)v129 )
    _libc_free((unsigned __int64)v127);
  sub_C7D6A0(v124, 8LL * (unsigned int)v126, 8);
  if ( !v137 )
    _libc_free((unsigned __int64)s);
  if ( v121 != &v123 )
    _libc_free((unsigned __int64)v121);
  return sub_C7D6A0(v118, 8LL * (unsigned int)v120, 8);
}
