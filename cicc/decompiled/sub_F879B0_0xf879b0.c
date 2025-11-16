// Function: sub_F879B0
// Address: 0xf879b0
//
__int64 __fastcall sub_F879B0(__int64 a1, __int64 a2, __int64 a3, __int64 *a4, _BYTE *a5)
{
  __int64 v7; // r13
  __int64 v10; // rax
  __int64 v11; // rsi
  __int64 v12; // rax
  char v13; // al
  __int64 v14; // rdx
  __int64 v15; // r15
  __int64 v16; // r13
  __int64 v17; // r12
  __int64 v18; // rax
  __int64 v19; // rcx
  __int64 v20; // rdi
  int v21; // esi
  __int64 v22; // rdx
  __int64 v23; // rcx
  __int64 v24; // rax
  __int64 v25; // r12
  __int64 v26; // rax
  __int16 v27; // ax
  __int64 v28; // r8
  __int64 v29; // r9
  __int64 v30; // rax
  unsigned __int64 v31; // rcx
  _QWORD *v32; // rdx
  unsigned __int64 v33; // rax
  int v34; // edx
  __int64 v35; // rsi
  __int64 v36; // rax
  __int64 v37; // r12
  __int64 v38; // rdx
  __int64 v39; // rcx
  __int64 v40; // r8
  __int64 v41; // r12
  __int16 v42; // dx
  __int64 v43; // r8
  __int64 *v44; // rax
  __int64 v45; // rdx
  char v46; // cl
  __int64 v47; // rax
  int v48; // ecx
  int v49; // r14d
  __int64 v50; // rax
  __int64 v51; // r10
  __int64 *v52; // rcx
  __int64 v53; // r8
  __int64 v54; // r9
  __int64 v55; // r15
  __int64 v56; // r14
  __int64 v57; // rdx
  unsigned int v58; // esi
  __int64 v59; // r15
  __int64 v60; // rdx
  _QWORD *v61; // rax
  _QWORD *v62; // rdx
  unsigned __int64 v63; // rax
  int v64; // edx
  __int64 v65; // rsi
  __int64 v66; // rax
  _QWORD *v67; // rax
  _QWORD *v68; // r14
  unsigned __int64 v69; // rax
  __int64 v70; // rdi
  int v71; // eax
  int v72; // eax
  unsigned int v73; // ecx
  __int64 v74; // rax
  __int64 v75; // rcx
  __int64 v76; // rdx
  __int64 *v77; // rax
  int v78; // eax
  int v79; // eax
  unsigned int v80; // edx
  __int64 *v81; // rax
  __int64 v82; // rdx
  __int64 v83; // rdx
  __int64 *v84; // rax
  __int64 *v85; // r8
  __int64 v86; // rsi
  __int64 v87; // rax
  __int64 v88; // rax
  unsigned __int8 *v89; // rdx
  char v90; // al
  __int64 v91; // r8
  __int64 v92; // rax
  _QWORD *v93; // rax
  char v94; // al
  unsigned int v95; // eax
  __int64 v96; // rdx
  __int64 v97; // rdx
  __int64 v98; // rcx
  __int64 v99; // r8
  __int64 v100; // r9
  __int64 *v101; // rax
  __int64 *v102; // rsi
  __int64 *v103; // r12
  __int64 *v104; // r12
  unsigned int v105; // r15d
  _QWORD **v106; // rax
  __int64 v107; // rdx
  __int64 v108; // rcx
  __int64 v109; // r8
  _QWORD *v110; // rax
  _QWORD *v111; // r14
  bool v112; // r13
  __int64 v113; // r8
  __int64 v114; // r9
  char *v115; // rax
  __int64 v116; // rcx
  __int64 *v117; // r13
  unsigned __int64 v118; // rdx
  unsigned __int64 v119; // rsi
  int v120; // eax
  unsigned __int64 *v121; // rdi
  __int64 v122; // rax
  __int64 v124; // rdx
  int v125; // r14d
  unsigned int v126; // esi
  int v127; // eax
  _QWORD *v128; // r14
  int v129; // eax
  char *v130; // rdx
  unsigned int v131; // r15d
  _QWORD **v132; // rax
  __int64 v133; // rdx
  __int64 v134; // rcx
  __int64 v135; // r8
  _QWORD *v136; // r15
  __int64 v137; // rdi
  char *v138; // r13
  _QWORD *v139; // [rsp+10h] [rbp-170h]
  _QWORD *v140; // [rsp+10h] [rbp-170h]
  __int64 v141; // [rsp+18h] [rbp-168h]
  bool v142; // [rsp+20h] [rbp-160h]
  __int64 *v143; // [rsp+20h] [rbp-160h]
  _QWORD *v144; // [rsp+20h] [rbp-160h]
  _QWORD *v145; // [rsp+20h] [rbp-160h]
  __int64 v146; // [rsp+28h] [rbp-158h]
  __int64 v147; // [rsp+28h] [rbp-158h]
  bool v148; // [rsp+30h] [rbp-150h]
  __int64 *v149; // [rsp+30h] [rbp-150h]
  __int64 v150; // [rsp+30h] [rbp-150h]
  __int64 v151; // [rsp+30h] [rbp-150h]
  __int64 v152; // [rsp+30h] [rbp-150h]
  __int64 v153; // [rsp+38h] [rbp-148h]
  __int64 v154; // [rsp+38h] [rbp-148h]
  __int64 v155; // [rsp+40h] [rbp-140h]
  __int64 v156; // [rsp+40h] [rbp-140h]
  char v158; // [rsp+57h] [rbp-129h]
  char v159; // [rsp+57h] [rbp-129h]
  __int64 v161; // [rsp+58h] [rbp-128h]
  __int64 v162; // [rsp+58h] [rbp-128h]
  __int64 v163; // [rsp+58h] [rbp-128h]
  __int64 v164; // [rsp+60h] [rbp-120h]
  __int64 *v165; // [rsp+60h] [rbp-120h]
  unsigned __int8 *v166; // [rsp+60h] [rbp-120h]
  __int64 v167; // [rsp+68h] [rbp-118h]
  __int64 v168; // [rsp+68h] [rbp-118h]
  __int64 v169; // [rsp+68h] [rbp-118h]
  __int64 v170; // [rsp+68h] [rbp-118h]
  __int64 v171; // [rsp+68h] [rbp-118h]
  __int64 v172; // [rsp+68h] [rbp-118h]
  __int64 v173; // [rsp+68h] [rbp-118h]
  char *v174; // [rsp+68h] [rbp-118h]
  _QWORD *v175; // [rsp+70h] [rbp-110h] BYREF
  _QWORD *v176; // [rsp+78h] [rbp-108h] BYREF
  _QWORD v177[2]; // [rsp+80h] [rbp-100h] BYREF
  char *v178; // [rsp+90h] [rbp-F0h]
  __int16 v179; // [rsp+A0h] [rbp-E0h]
  __int64 v180; // [rsp+B0h] [rbp-D0h] BYREF
  __int64 v181; // [rsp+B8h] [rbp-C8h]
  __int64 v182; // [rsp+C0h] [rbp-C0h] BYREF
  _QWORD *v183; // [rsp+C8h] [rbp-B8h]
  __int16 v184; // [rsp+D0h] [rbp-B0h]
  __int64 v185; // [rsp+E0h] [rbp-A0h] BYREF
  __int64 v186; // [rsp+E8h] [rbp-98h]
  __int64 v187; // [rsp+F0h] [rbp-90h]
  char v188; // [rsp+FCh] [rbp-84h]
  _BYTE v189[16]; // [rsp+100h] [rbp-80h] BYREF
  __int64 v190; // [rsp+110h] [rbp-70h] BYREF
  _QWORD v191[4]; // [rsp+118h] [rbp-68h] BYREF
  __int16 v192; // [rsp+138h] [rbp-48h]
  _QWORD v193[8]; // [rsp+140h] [rbp-40h] BYREF

  v7 = a3;
  v10 = sub_D47930(a3);
  v164 = v10;
  if ( !v10 )
    goto LABEL_22;
  *a4 = 0;
  v11 = v10;
  *a5 = 0;
  v12 = *(_QWORD *)(a1 + 464);
  v158 = 0;
  if ( v12 )
  {
    sub_B196A0(*(_QWORD *)(*(_QWORD *)a1 + 40LL), v11, **(_QWORD **)(v12 + 32));
    v158 = v13;
  }
  v15 = sub_AA5930(**(_QWORD **)(v7 + 32));
  if ( v14 == v15 )
  {
LABEL_22:
    v191[0] = 0;
    v165 = (__int64 *)(a1 + 520);
    v190 = a1 + 520;
    v26 = *(_QWORD *)(a1 + 568);
    v191[1] = 0;
    v191[2] = v26;
    if ( v26 != 0 && v26 != -4096 && v26 != -8192 )
      sub_BD73F0((__int64)v191);
    v27 = *(_WORD *)(a1 + 584);
    v191[3] = *(_QWORD *)(a1 + 576);
    v192 = v27;
    sub_B33910(v193, v165);
    v30 = *(unsigned int *)(a1 + 792);
    v31 = *(unsigned int *)(a1 + 796);
    v193[1] = a1;
    if ( v30 + 1 > v31 )
    {
      sub_C8D5F0(a1 + 784, (const void *)(a1 + 800), v30 + 1, 8u, v28, v29);
      v30 = *(unsigned int *)(a1 + 792);
    }
    *(_QWORD *)(*(_QWORD *)(a1 + 784) + 8 * v30) = &v190;
    ++*(_DWORD *)(a1 + 792);
    v156 = a1 + 416;
    sub_C8CD80((__int64)&v185, (__int64)v189, a1 + 416, v31, v28, v29);
    ++*(_QWORD *)(a1 + 416);
    if ( !*(_BYTE *)(a1 + 444) )
    {
      v95 = 4 * (*(_DWORD *)(a1 + 436) - *(_DWORD *)(a1 + 440));
      v96 = *(unsigned int *)(a1 + 432);
      if ( v95 < 0x20 )
        v95 = 32;
      if ( v95 < (unsigned int)v96 )
      {
        sub_C8C990(v156, (__int64)v189);
        goto LABEL_29;
      }
      memset(*(void **)(a1 + 424), -1, 8 * v96);
    }
    *(_QWORD *)(a1 + 436) = 0;
LABEL_29:
    v32 = (_QWORD *)(sub_D4B130(v7) + 48);
    v33 = *v32 & 0xFFFFFFFFFFFFFFF8LL;
    if ( (_QWORD *)v33 == v32 )
    {
      v35 = 0;
    }
    else
    {
      if ( !v33 )
        BUG();
      v34 = *(unsigned __int8 *)(v33 - 24);
      v35 = 0;
      v36 = v33 - 24;
      if ( (unsigned int)(v34 - 30) < 0xB )
        v35 = v36;
    }
    v37 = **(_QWORD **)(a2 + 32);
    sub_D5F1F0((__int64)v165, v35);
    v141 = sub_F894B0(a1, v37);
    v41 = sub_D33D80((_QWORD *)a2, *(_QWORD *)a1, v38, v39, v40);
    v159 = 0;
    v146 = sub_D95540(**(_QWORD **)(a2 + 32));
    if ( *(_BYTE *)(v146 + 8) != 14 )
    {
      v159 = sub_D969D0(v41);
      if ( v159 )
        v41 = (__int64)sub_DCAF50(*(__int64 **)a1, v41, 0);
    }
    v43 = sub_AA5190(**(_QWORD **)(v7 + 32));
    if ( !v43 )
      BUG();
    sub_A88F30((__int64)v165, *(_QWORD *)(v43 + 16), v43, v42);
    v142 = 0;
    v154 = sub_F894B0(a1, v41);
    v148 = 0;
    if ( !v159 )
    {
      v103 = *(__int64 **)a1;
      if ( *(_BYTE *)(sub_D95540(**(_QWORD **)(a2 + 32)) + 8) == 12 )
      {
        v131 = 2 * (*(_DWORD *)(sub_D95540(**(_QWORD **)(a2 + 32)) + 8) >> 8);
        v132 = (_QWORD **)sub_D95540(**(_QWORD **)(a2 + 32));
        v152 = sub_BCCE00(*v132, v131);
        v163 = sub_D33D80((_QWORD *)a2, (__int64)v103, v133, v134, v135);
        v144 = sub_DC2B70((__int64)v103, a2, v152, 0);
        v180 = (__int64)&v182;
        v181 = 0x200000002LL;
        v182 = (__int64)sub_DC2B70((__int64)v103, v163, v152, 0);
        v183 = v144;
        v145 = sub_DC7EB0(v103, (__int64)&v180, 0, 0);
        if ( (__int64 *)v180 != &v182 )
          _libc_free(v180, &v180);
        v182 = a2;
        v183 = (_QWORD *)v163;
        v181 = 0x200000002LL;
        v180 = (__int64)&v182;
        v136 = sub_DC7EB0(v103, (__int64)&v180, 0, 0);
        if ( (__int64 *)v180 != &v182 )
          _libc_free(v180, &v180);
        v142 = v145 == sub_DC2B70((__int64)v103, (__int64)v136, v152, 0);
      }
      v104 = *(__int64 **)a1;
      v148 = 0;
      if ( *(_BYTE *)(sub_D95540(**(_QWORD **)(a2 + 32)) + 8) == 12 )
      {
        v105 = 2 * (*(_DWORD *)(sub_D95540(**(_QWORD **)(a2 + 32)) + 8) >> 8);
        v106 = (_QWORD **)sub_D95540(**(_QWORD **)(a2 + 32));
        v151 = sub_BCCE00(*v106, v105);
        v162 = sub_D33D80((_QWORD *)a2, (__int64)v104, v107, v108, v109);
        v139 = sub_DC5000((__int64)v104, a2, v151, 0);
        v110 = sub_DC5000((__int64)v104, v162, v151, 0);
        v180 = (__int64)&v182;
        v181 = 0x200000002LL;
        v182 = (__int64)v110;
        v183 = v139;
        v140 = sub_DC7EB0(v104, (__int64)&v180, 0, 0);
        if ( (__int64 *)v180 != &v182 )
          _libc_free(v180, &v180);
        v182 = a2;
        v183 = (_QWORD *)v162;
        v181 = 0x200000002LL;
        v180 = (__int64)&v182;
        v111 = sub_DC7EB0(v104, (__int64)&v180, 0, 0);
        if ( (__int64 *)v180 != &v182 )
          _libc_free(v180, &v180);
        v148 = v140 == sub_DC5000((__int64)v104, (__int64)v111, v151, 0);
      }
    }
    v44 = *(__int64 **)(v7 + 32);
    v168 = *v44;
    sub_A88F30((__int64)v165, *v44, *(_QWORD *)(*v44 + 56), 1);
    if ( **(_BYTE **)(a1 + 16) )
    {
      v177[0] = *(_QWORD *)(a1 + 16);
      v178 = ".iv";
      v179 = 771;
    }
    else
    {
      v177[0] = ".iv";
      v179 = 259;
    }
    v45 = *(_QWORD *)(v168 + 16);
    do
    {
      if ( !v45 )
      {
        v49 = 0;
        goto LABEL_46;
      }
      v46 = **(_BYTE **)(v45 + 24);
      v47 = v45;
      v45 = *(_QWORD *)(v45 + 8);
    }
    while ( (unsigned __int8)(v46 - 30) > 0xAu );
    v48 = 0;
    while ( 1 )
    {
      v47 = *(_QWORD *)(v47 + 8);
      if ( !v47 )
        break;
      while ( (unsigned __int8)(**(_BYTE **)(v47 + 24) - 30) <= 0xAu )
      {
        v47 = *(_QWORD *)(v47 + 8);
        ++v48;
        if ( !v47 )
          goto LABEL_45;
      }
    }
LABEL_45:
    v49 = v48 + 1;
LABEL_46:
    v184 = 257;
    v50 = sub_BD2DA0(80);
    v25 = v50;
    if ( v50 )
    {
      v161 = v50;
      sub_B44260(v50, v146, 55, 0x8000000u, 0, 0);
      *(_DWORD *)(v25 + 72) = v49;
      sub_BD6B50((unsigned __int8 *)v25, (const char **)&v180);
      sub_BD2A10(v25, *(_DWORD *)(v25 + 72), 1);
      v51 = v161;
    }
    else
    {
      v51 = 0;
    }
    if ( (unsigned __int8)sub_920620(v51) )
    {
      v124 = *(_QWORD *)(a1 + 616);
      v125 = *(_DWORD *)(a1 + 624);
      if ( v124 )
        sub_B99FD0(v25, 3u, v124);
      sub_B45150(v25, v125);
    }
    (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(a1 + 608) + 16LL))(
      *(_QWORD *)(a1 + 608),
      v25,
      v177,
      *(_QWORD *)(a1 + 576),
      *(_QWORD *)(a1 + 584));
    v55 = *(_QWORD *)(a1 + 520);
    v56 = v55 + 16LL * *(unsigned int *)(a1 + 528);
    while ( v56 != v55 )
    {
      v57 = *(_QWORD *)(v55 + 8);
      v58 = *(_DWORD *)v55;
      v55 += 16;
      sub_B99FD0(v25, v58, v57);
    }
    v59 = *(_QWORD *)(v168 + 16);
    if ( !v59 )
    {
LABEL_145:
      sub_C8CE00(v156, a1 + 448, (__int64)&v185, (__int64)v52, v53, v54);
      v177[0] = 0;
      v177[1] = 0;
      v178 = (char *)v25;
      v112 = v25 != -8192 && v25 != 0 && v25 != -4096;
      if ( v112 )
        sub_BD73F0((__int64)v177);
      if ( (unsigned __int8)sub_F82F60(a1 + 64, (__int64)v177, &v175) )
      {
        v115 = v178;
LABEL_149:
        if ( v115 != 0 && v115 + 4096 != 0 && v115 != (char *)-8192LL )
          sub_BD60C0(v177);
        v180 = 4;
        v181 = 0;
        v182 = v25;
        if ( v112 )
          sub_BD73F0((__int64)&v180);
        v116 = *(unsigned int *)(a1 + 328);
        v117 = &v180;
        v118 = *(_QWORD *)(a1 + 320);
        v119 = v116 + 1;
        v120 = *(_DWORD *)(a1 + 328);
        if ( v116 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 332) )
        {
          v137 = a1 + 320;
          if ( v118 > (unsigned __int64)&v180 || (unsigned __int64)&v180 >= v118 + 24 * v116 )
          {
            sub_D6B130(v137, v119, v118, v116, v113, v114);
            v116 = *(unsigned int *)(a1 + 328);
            v117 = &v180;
            v118 = *(_QWORD *)(a1 + 320);
            v120 = *(_DWORD *)(a1 + 328);
          }
          else
          {
            v138 = (char *)&v180 - v118;
            sub_D6B130(v137, v119, v118, v116, v113, v114);
            v118 = *(_QWORD *)(a1 + 320);
            v116 = *(unsigned int *)(a1 + 328);
            v117 = (__int64 *)&v138[v118];
            v120 = *(_DWORD *)(a1 + 328);
          }
        }
        v121 = (unsigned __int64 *)(v118 + 24 * v116);
        if ( v121 )
        {
          *v121 = 4;
          v122 = v117[2];
          v121[1] = 0;
          v121[2] = v122;
          if ( v122 != -4096 && v122 != 0 && v122 != -8192 )
          {
            v119 = *v117 & 0xFFFFFFFFFFFFFFF8LL;
            sub_BD6050(v121, v119);
          }
          v120 = *(_DWORD *)(a1 + 328);
        }
        *(_DWORD *)(a1 + 328) = v120 + 1;
        if ( v182 != 0 && v182 != -4096 && v182 != -8192 )
          sub_BD60C0(&v180);
        if ( !v188 )
          _libc_free(v186, v119);
        sub_F80960((__int64)&v190);
        return v25;
      }
      v126 = *(_DWORD *)(a1 + 88);
      v127 = *(_DWORD *)(a1 + 80);
      v128 = v175;
      ++*(_QWORD *)(a1 + 64);
      v129 = v127 + 1;
      v176 = v128;
      if ( 4 * v129 >= 3 * v126 )
      {
        v126 *= 2;
      }
      else if ( v126 - *(_DWORD *)(a1 + 84) - v129 > v126 >> 3 )
      {
LABEL_181:
        *(_DWORD *)(a1 + 80) = v129;
        v180 = 0;
        v181 = 0;
        v182 = -4096;
        if ( v128[2] != -4096 )
          --*(_DWORD *)(a1 + 84);
        sub_D68D70(&v180);
        v130 = v178;
        v115 = (char *)v128[2];
        if ( v178 != v115 )
        {
          if ( v115 != 0 && v115 + 4096 != 0 && v115 != (char *)-8192LL )
          {
            v174 = v178;
            sub_BD60C0(v128);
            v130 = v174;
          }
          v128[2] = v130;
          if ( v130 != 0 && v130 + 4096 != 0 && v130 != (char *)-8192LL )
            sub_BD73F0((__int64)v128);
          v115 = v178;
        }
        goto LABEL_149;
      }
      sub_F86930(a1 + 64, v126);
      sub_F82F60(a1 + 64, (__int64)v177, &v176);
      v128 = v176;
      v129 = *(_DWORD *)(a1 + 80) + 1;
      goto LABEL_181;
    }
    while ( 1 )
    {
      v60 = *(_QWORD *)(v59 + 24);
      if ( (unsigned __int8)(*(_BYTE *)v60 - 30) <= 0xAu )
        break;
      v59 = *(_QWORD *)(v59 + 8);
      if ( !v59 )
        goto LABEL_145;
    }
    v54 = *(_QWORD *)(v60 + 40);
    if ( *(_BYTE *)(v7 + 84) )
    {
LABEL_54:
      v61 = *(_QWORD **)(v7 + 64);
      v62 = &v61[*(unsigned int *)(v7 + 76)];
      if ( v61 == v62 )
        goto LABEL_81;
      while ( v54 != *v61 )
      {
        if ( v62 == ++v61 )
          goto LABEL_81;
      }
    }
    else
    {
LABEL_80:
      v170 = v54;
      v77 = sub_C8CA60(v7 + 56, v54);
      v54 = v170;
      if ( !v77 )
      {
LABEL_81:
        v78 = *(_DWORD *)(v25 + 4) & 0x7FFFFFF;
        if ( v78 == *(_DWORD *)(v25 + 72) )
        {
          v173 = v54;
          sub_B48D90(v25);
          v54 = v173;
          v78 = *(_DWORD *)(v25 + 4) & 0x7FFFFFF;
        }
        v79 = (v78 + 1) & 0x7FFFFFF;
        v80 = v79 | *(_DWORD *)(v25 + 4) & 0xF8000000;
        v81 = (__int64 *)(*(_QWORD *)(v25 - 8) + 32LL * (unsigned int)(v79 - 1));
        *(_DWORD *)(v25 + 4) = v80;
        if ( *v81 )
        {
          v52 = (__int64 *)v81[2];
          v82 = v81[1];
          *v52 = v82;
          if ( v82 )
          {
            v52 = (__int64 *)v81[2];
            *(_QWORD *)(v82 + 16) = v52;
          }
        }
        *v81 = v141;
        if ( v141 )
        {
          v83 = *(_QWORD *)(v141 + 16);
          v52 = (__int64 *)(v141 + 16);
          v81[1] = v83;
          if ( v83 )
            *(_QWORD *)(v83 + 16) = v81 + 1;
          v81[2] = (__int64)v52;
          *(_QWORD *)(v141 + 16) = v81;
        }
LABEL_76:
        *(_QWORD *)(*(_QWORD *)(v25 - 8)
                  + 32LL * *(unsigned int *)(v25 + 72)
                  + 8LL * ((*(_DWORD *)(v25 + 4) & 0x7FFFFFFu) - 1)) = v54;
        while ( 1 )
        {
          v59 = *(_QWORD *)(v59 + 8);
          if ( !v59 )
            goto LABEL_145;
          v76 = *(_QWORD *)(v59 + 24);
          if ( (unsigned __int8)(*(_BYTE *)v76 - 30) <= 0xAu )
          {
            v54 = *(_QWORD *)(v76 + 40);
            if ( *(_BYTE *)(v7 + 84) )
              goto LABEL_54;
            goto LABEL_80;
          }
        }
      }
    }
    if ( *(_QWORD *)(a1 + 464) == v7 )
    {
      v65 = *(_QWORD *)(a1 + 472);
    }
    else
    {
      v63 = *(_QWORD *)(v54 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v63 == v54 + 48 )
      {
        v65 = 0;
      }
      else
      {
        if ( !v63 )
          BUG();
        v64 = *(unsigned __int8 *)(v63 - 24);
        v65 = 0;
        v66 = v63 - 24;
        if ( (unsigned int)(v64 - 30) < 0xB )
          v65 = v66;
      }
    }
    v169 = v54;
    sub_D5F1F0((__int64)v165, v65);
    v67 = sub_F7DD30(a1, v25, v154, v7, v159);
    v54 = v169;
    v68 = v67;
    v69 = *(unsigned __int8 *)v67;
    if ( (unsigned __int8)v69 <= 0x1Cu )
    {
      if ( (_BYTE)v69 != 5 )
        goto LABEL_68;
      if ( (*((_WORD *)v68 + 1) & 0xFFF7) != 0x11 )
      {
        if ( (*((_WORD *)v68 + 1) & 0xFFFD) != 0xD )
          goto LABEL_68;
        if ( !v142 )
        {
LABEL_67:
          if ( !v148 )
          {
LABEL_68:
            v71 = *(_DWORD *)(v25 + 4) & 0x7FFFFFF;
            if ( v71 == *(_DWORD *)(v25 + 72) )
            {
              v172 = v54;
              sub_B48D90(v25);
              v54 = v172;
              v71 = *(_DWORD *)(v25 + 4) & 0x7FFFFFF;
            }
            v72 = (v71 + 1) & 0x7FFFFFF;
            v73 = v72 | *(_DWORD *)(v25 + 4) & 0xF8000000;
            v74 = *(_QWORD *)(v25 - 8) + 32LL * (unsigned int)(v72 - 1);
            *(_DWORD *)(v25 + 4) = v73;
            if ( *(_QWORD *)v74 )
            {
              v75 = *(_QWORD *)(v74 + 8);
              **(_QWORD **)(v74 + 16) = v75;
              if ( v75 )
                *(_QWORD *)(v75 + 16) = *(_QWORD *)(v74 + 16);
            }
            *(_QWORD *)v74 = v68;
            v52 = (__int64 *)v68[2];
            *(_QWORD *)(v74 + 8) = v52;
            if ( v52 )
              v52[2] = v74 + 8;
            *(_QWORD *)(v74 + 16) = v68 + 2;
            v68[2] = v74;
            goto LABEL_76;
          }
LABEL_172:
          v171 = v54;
          sub_B44850((unsigned __int8 *)v68, 1);
          v54 = v171;
          goto LABEL_68;
        }
LABEL_171:
        sub_B447F0((unsigned __int8 *)v68, 1);
        v54 = v169;
        if ( !v148 )
          goto LABEL_68;
        goto LABEL_172;
      }
    }
    else
    {
      if ( (unsigned __int8)v69 > 0x36u )
        goto LABEL_68;
      v70 = 0x40540000000000LL;
      if ( !_bittest64(&v70, v69) )
        goto LABEL_68;
    }
    if ( !v142 )
      goto LABEL_67;
    goto LABEL_171;
  }
  v167 = 0;
  v155 = v7;
  v16 = v164;
  v153 = 0;
  v17 = v14;
  do
  {
    if ( !sub_D97040(*(_QWORD *)a1, *(_QWORD *)(v15 + 8)) )
      goto LABEL_17;
    v18 = *(_QWORD *)(*(_QWORD *)(v15 + 40) + 16LL);
    if ( v18 )
    {
      while ( 1 )
      {
        v19 = *(_QWORD *)(v18 + 24);
        if ( (unsigned __int8)(*(_BYTE *)v19 - 30) <= 0xAu )
          break;
        v18 = *(_QWORD *)(v18 + 8);
        if ( !v18 )
          goto LABEL_91;
      }
      v20 = *(_QWORD *)(v19 + 40);
      v21 = *(_DWORD *)(v15 + 4) & 0x7FFFFFF;
      if ( !v21 )
        goto LABEL_17;
LABEL_10:
      v22 = 0;
      while ( v20 != *(_QWORD *)(*(_QWORD *)(v15 - 8) + 32LL * *(unsigned int *)(v15 + 72) + 8 * v22) )
      {
        if ( v21 == (_DWORD)++v22 )
          goto LABEL_17;
      }
      if ( (int)v22 < 0 )
        goto LABEL_17;
      while ( 1 )
      {
        v18 = *(_QWORD *)(v18 + 8);
        if ( !v18 )
          break;
        v23 = *(_QWORD *)(v18 + 24);
        if ( (unsigned __int8)(*(_BYTE *)v23 - 30) <= 0xAu )
        {
          v20 = *(_QWORD *)(v23 + 40);
          goto LABEL_10;
        }
      }
    }
LABEL_91:
    v84 = sub_DD8400(*(_QWORD *)a1, v15);
    v85 = v84;
    if ( *((_WORD *)v84 + 12) == 8 && (v84 == (__int64 *)a2 || v158 == 1) )
    {
      v86 = *(_QWORD *)(v15 - 8);
      v87 = 0x1FFFFFFFE0LL;
      if ( (*(_DWORD *)(v15 + 4) & 0x7FFFFFF) != 0 )
      {
        v88 = 0;
        do
        {
          if ( v16 == *(_QWORD *)(v86 + 32LL * *(unsigned int *)(v15 + 72) + 8 * v88) )
          {
            v87 = 32 * v88;
            goto LABEL_99;
          }
          ++v88;
        }
        while ( (*(_DWORD *)(v15 + 4) & 0x7FFFFFF) != (_DWORD)v88 );
        v87 = 0x1FFFFFFFE0LL;
      }
LABEL_99:
      v89 = *(unsigned __int8 **)(v86 + v87);
      v166 = v89;
      if ( *v89 > 0x1Cu )
      {
        v149 = v85;
        if ( *(_BYTE *)(a1 + 513) )
        {
          v94 = sub_F7DC80(a1, (unsigned __int8 *)v15, v89, v155);
          v91 = (__int64)v149;
          if ( !v94 )
            goto LABEL_17;
        }
        else
        {
          v90 = sub_F7D890((_QWORD *)a1, v15, (__int64)v89, v155);
          v91 = (__int64)v149;
          if ( !v90 )
            goto LABEL_17;
        }
        if ( v91 == a2 )
        {
          v25 = v15;
          *a4 = 0;
          *a5 = 0;
          v167 = (__int64)v166;
          goto LABEL_124;
        }
        if ( !*a4 || *a5 )
        {
          v147 = v91;
          v143 = *(__int64 **)a1;
          v150 = sub_D95540(**(_QWORD **)(v91 + 32));
          v92 = sub_D95540(**(_QWORD **)(a2 + 32));
          if ( *(_BYTE *)(v150 + 8) != 14
            && *(_BYTE *)(v92 + 8) != 14
            && *(_DWORD *)(v92 + 8) >> 8 <= *(_DWORD *)(v150 + 8) >> 8 )
          {
            v93 = sub_DC5820((__int64)v143, v147, v92);
            if ( *((_WORD *)v93 + 12) == 8 )
            {
              if ( (_QWORD *)a2 == v93 )
              {
                *a5 = 0;
              }
              else
              {
                if ( v93 != sub_DCC810(v143, **(_QWORD **)(a2 + 32), a2, 0, 0) )
                  goto LABEL_17;
                *a5 = 1;
              }
              v153 = v15;
              *a4 = sub_D95540(**(_QWORD **)(a2 + 32));
              v167 = (__int64)v166;
            }
          }
        }
      }
    }
LABEL_17:
    v24 = *(_QWORD *)(v15 + 32);
    if ( !v24 )
      BUG();
    v15 = 0;
    if ( *(_BYTE *)(v24 - 24) == 84 )
      v15 = v24 - 24;
  }
  while ( v17 != v15 );
  v25 = v153;
  v7 = v155;
  if ( !v153 )
    goto LABEL_22;
LABEL_124:
  v187 = v25;
  v185 = 0;
  v186 = 0;
  if ( v25 != -8192 && v25 != -4096 )
    sub_BD73F0((__int64)&v185);
  sub_F86C90((__int64)&v190, a1 + 64, (__int64)&v185);
  sub_D68D70(&v185);
  sub_F86EA0(a1, v167);
  if ( !*(_BYTE *)(a1 + 156) )
    goto LABEL_199;
  v98 = *(unsigned int *)(a1 + 148);
  v101 = *(__int64 **)(a1 + 136);
  v102 = &v101[v98];
  v97 = v98;
  if ( v101 == v102 )
  {
LABEL_196:
    if ( (unsigned int)v97 < *(_DWORD *)(a1 + 144) )
    {
      *(_DWORD *)(a1 + 148) = v97 + 1;
      *v102 = v25;
      v97 = *(unsigned __int8 *)(a1 + 156);
      ++*(_QWORD *)(a1 + 128);
      v101 = *(__int64 **)(a1 + 136);
      if ( !(_BYTE)v97 )
        goto LABEL_200;
LABEL_198:
      v97 = *(unsigned int *)(a1 + 148);
      goto LABEL_132;
    }
LABEL_199:
    sub_C8CC70(a1 + 128, v25, v97, v98, v99, v100);
    v97 = *(unsigned __int8 *)(a1 + 156);
    v101 = *(__int64 **)(a1 + 136);
    if ( !(_BYTE)v97 )
      goto LABEL_200;
    goto LABEL_198;
  }
  v98 = *(_QWORD *)(a1 + 136);
  while ( *(_QWORD *)v98 != v25 )
  {
    v98 += 8;
    if ( v102 == (__int64 *)v98 )
      goto LABEL_196;
  }
LABEL_132:
  v98 = (__int64)&v101[(unsigned int)v97];
  if ( v101 == (__int64 *)v98 )
  {
LABEL_135:
    if ( (unsigned int)v97 < *(_DWORD *)(a1 + 144) )
    {
      *(_DWORD *)(a1 + 148) = v97 + 1;
      *(_QWORD *)v98 = v167;
      ++*(_QWORD *)(a1 + 128);
      return v25;
    }
LABEL_200:
    sub_C8CC70(a1 + 128, v167, v97, v98, v99, v100);
    return v25;
  }
  while ( *v101 != v167 )
  {
    if ( (__int64 *)v98 == ++v101 )
      goto LABEL_135;
  }
  return v25;
}
