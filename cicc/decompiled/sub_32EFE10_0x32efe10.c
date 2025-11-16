// Function: sub_32EFE10
// Address: 0x32efe10
//
__int64 __fastcall sub_32EFE10(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  __int64 v6; // r14
  int v10; // ebx
  __int64 v11; // rax
  __int64 v12; // r14
  void *v13; // r8
  __int64 *v14; // rax
  __int64 v15; // rcx
  __int64 v16; // rax
  __int16 v17; // dx
  unsigned __int64 v18; // rax
  int v19; // ecx
  __int64 v20; // rax
  int v21; // edx
  __int64 v22; // rax
  __int64 v23; // rax
  int v24; // edx
  __int64 v25; // rax
  char v26; // dl
  int v28; // eax
  __int64 *v29; // rax
  __int64 v30; // rax
  int v31; // edx
  void *v32; // r8
  __int64 v33; // rcx
  __int64 v34; // rax
  __int64 v35; // rax
  __int64 v36; // rax
  __int64 v37; // rax
  __int64 v38; // rdx
  __int64 v39; // rdi
  __int16 v40; // dx
  __int16 v41; // ax
  __int16 v42; // ax
  char v43; // al
  __int64 v44; // rax
  __int64 v45; // rdx
  __int64 v46; // rax
  __int64 v47; // rdx
  __int64 v48; // rcx
  __int64 v49; // r8
  __int64 v50; // r9
  __int64 v51; // rdx
  __int64 v52; // rcx
  __int64 v53; // r8
  __int64 v54; // r9
  __int64 v55; // r9
  __int64 v56; // r9
  __int64 v57; // rcx
  __int64 v58; // r8
  __int64 v59; // r9
  __int64 *v60; // rax
  __int64 v61; // rbx
  __int64 v62; // rdx
  __int64 v63; // rcx
  __int64 v64; // r8
  __int64 v65; // r9
  __int64 v66; // rdx
  __int64 v67; // rcx
  __int64 v68; // r8
  __int64 v69; // r9
  __int64 v70; // rdx
  __int64 v71; // rcx
  __int64 v72; // r8
  __int64 v73; // r9
  __int64 v74; // r9
  __int64 v75; // rdx
  __int64 v76; // rsi
  __int64 v77; // r13
  __int128 *v78; // rbx
  __int64 v79; // r8
  int v80; // ecx
  unsigned int v81; // edx
  unsigned __int8 v82; // al
  __int64 v83; // rdi
  unsigned __int8 v84; // bl
  unsigned __int8 v85; // al
  __int64 *v86; // rdx
  unsigned __int16 v87; // bx
  char v88; // al
  unsigned __int8 v89; // dl
  unsigned __int8 v90; // si
  __int64 v91; // r13
  __int64 v92; // rdi
  __int16 v93; // ax
  unsigned __int16 *v94; // rcx
  __int64 v95; // r13
  int v96; // edx
  int v97; // ebx
  int v98; // r9d
  __int64 v99; // rdx
  unsigned __int16 *v100; // r10
  int v101; // eax
  __int64 v102; // r8
  int v103; // ecx
  int v104; // r9d
  __int64 v105; // rax
  int v106; // esi
  int v107; // edx
  __int64 v108; // rdx
  __int64 v109; // rcx
  __int64 v110; // r8
  __int64 v111; // r9
  __int64 v112; // rdx
  __int64 v113; // rcx
  __int64 v114; // r8
  __int64 v115; // r9
  __int64 v116; // r9
  __int64 v117; // rdx
  __int64 v118; // rsi
  __int64 v119; // r13
  __int64 *v120; // r8
  __int64 v121; // rcx
  int v122; // ebx
  unsigned int v123; // edx
  int v124; // [rsp+18h] [rbp-2B8h]
  int v125; // [rsp+20h] [rbp-2B0h]
  __int64 v126; // [rsp+20h] [rbp-2B0h]
  __int64 v127; // [rsp+28h] [rbp-2A8h]
  int v128; // [rsp+34h] [rbp-29Ch]
  __int16 v129; // [rsp+38h] [rbp-298h]
  int v130; // [rsp+38h] [rbp-298h]
  __int64 v131; // [rsp+38h] [rbp-298h]
  __int64 v132; // [rsp+40h] [rbp-290h]
  __int64 *v133; // [rsp+40h] [rbp-290h]
  void *v134; // [rsp+48h] [rbp-288h]
  __int64 v135; // [rsp+48h] [rbp-288h]
  __int64 v136; // [rsp+50h] [rbp-280h]
  unsigned __int8 v137; // [rsp+50h] [rbp-280h]
  void *v138; // [rsp+50h] [rbp-280h]
  int v139; // [rsp+58h] [rbp-278h]
  int v140; // [rsp+58h] [rbp-278h]
  __int64 *v141; // [rsp+58h] [rbp-278h]
  int v142; // [rsp+58h] [rbp-278h]
  __int64 v144; // [rsp+60h] [rbp-270h]
  __int64 v145; // [rsp+68h] [rbp-268h]
  __int64 v148; // [rsp+A0h] [rbp-230h] BYREF
  int v149; // [rsp+A8h] [rbp-228h]
  __int128 v150; // [rsp+B0h] [rbp-220h] BYREF
  __int64 v151; // [rsp+C0h] [rbp-210h]
  __int64 v152; // [rsp+D0h] [rbp-200h] BYREF
  __int64 v153; // [rsp+D8h] [rbp-1F8h]
  __int64 v154; // [rsp+E0h] [rbp-1F0h]
  __int64 v155; // [rsp+E8h] [rbp-1E8h]
  unsigned __int64 v156[2]; // [rsp+F0h] [rbp-1E0h] BYREF
  _BYTE v157[128]; // [rsp+100h] [rbp-1D0h] BYREF
  __int64 v158; // [rsp+180h] [rbp-150h] BYREF
  _QWORD *v159; // [rsp+188h] [rbp-148h]
  __int64 v160; // [rsp+190h] [rbp-140h]
  int v161; // [rsp+198h] [rbp-138h]
  char v162; // [rsp+19Ch] [rbp-134h]
  __int64 v163; // [rsp+1A0h] [rbp-130h] BYREF

  v10 = a4;
  v11 = sub_33E1790(a3, a4, 0);
  if ( !v11 )
    goto LABEL_33;
  v12 = *(_QWORD *)(v11 + 96);
  v13 = sub_C33340();
  if ( *(void **)(v12 + 24) == v13 )
    v6 = *(_QWORD *)(v12 + 32);
  else
    v6 = v12 + 24;
  v14 = *(__int64 **)(a2 + 40);
  v15 = *v14;
  if ( (*(_BYTE *)(v6 + 20) & 7) == 1 && *(_DWORD *)(a5 + 24) == 246 )
  {
    if ( *(_DWORD *)(a2 + 24) == 207 )
    {
      v135 = *v14;
      v138 = v13;
      v139 = *((_DWORD *)v14 + 2);
      LODWORD(v6) = *(_DWORD *)(v14[20] + 96);
      v30 = sub_33E1790(v14[5], v14[6], 0);
      v32 = v138;
      v33 = v135;
    }
    else
    {
      if ( *(_DWORD *)(v15 + 24) != 208 )
        goto LABEL_6;
      v29 = *(__int64 **)(v15 + 40);
      v134 = v13;
      v139 = *((_DWORD *)v29 + 2);
      LODWORD(v6) = *(_DWORD *)(v29[10] + 96);
      v136 = *v29;
      v30 = sub_33E1790(v29[5], v29[6], 0);
      v32 = v134;
      v33 = v136;
    }
    if ( v30 )
    {
      v34 = *(_QWORD *)(v30 + 96);
      v35 = v32 == *(void **)(v34 + 24) ? *(_QWORD *)(v34 + 32) : v34 + 24;
      if ( (*(_BYTE *)(v35 + 20) & 7) == 3 )
      {
        v36 = *(_QWORD *)(a5 + 40);
        if ( *(_QWORD *)v36 == v33 && *(_DWORD *)(v36 + 8) == v139 )
        {
          LOBYTE(v31) = (_DWORD)v6 == 20 || (v6 & 0xFFFFFFF7) == 4;
          LODWORD(v6) = v31;
          if ( (_BYTE)v31 )
          {
            v158 = a5;
            LODWORD(v159) = a6;
            sub_32EB790((__int64)a1, a2, &v158, 1, 1);
            return (unsigned int)v6;
          }
        }
      }
    }
LABEL_33:
    v14 = *(__int64 **)(a2 + 40);
    v15 = *v14;
  }
LABEL_6:
  v16 = *(_QWORD *)(v15 + 48) + 16LL * *((unsigned int *)v14 + 2);
  v17 = *(_WORD *)v16;
  v18 = *(_QWORD *)(v16 + 8);
  LOWORD(v158) = v17;
  v159 = (_QWORD *)v18;
  if ( v17 )
  {
    LOBYTE(v6) = (unsigned __int16)(v17 - 17) <= 0xD3u;
  }
  else
  {
    LOBYTE(v28) = sub_30070B0((__int64)&v158);
    LODWORD(v6) = v28;
  }
  if ( (_BYTE)v6 )
  {
LABEL_29:
    LODWORD(v6) = 0;
    return (unsigned int)v6;
  }
  v19 = *(_DWORD *)(a3 + 24);
  if ( *(_DWORD *)(a5 + 24) == v19 )
  {
    v20 = *(_QWORD *)(a3 + 56);
    if ( v20 )
    {
      v21 = 1;
      do
      {
        while ( v10 != *(_DWORD *)(v20 + 8) )
        {
          v20 = *(_QWORD *)(v20 + 32);
          if ( !v20 )
            goto LABEL_18;
        }
        if ( !v21 )
          return (unsigned int)v6;
        v22 = *(_QWORD *)(v20 + 32);
        if ( !v22 )
          goto LABEL_19;
        if ( v10 == *(_DWORD *)(v22 + 8) )
          return (unsigned int)v6;
        v20 = *(_QWORD *)(v22 + 32);
        v21 = 0;
      }
      while ( v20 );
LABEL_18:
      if ( v21 == 1 )
        return (unsigned int)v6;
LABEL_19:
      v23 = *(_QWORD *)(a5 + 56);
      if ( v23 )
      {
        v24 = 1;
        while ( 1 )
        {
          while ( *(_DWORD *)(v23 + 8) != a6 )
          {
            v23 = *(_QWORD *)(v23 + 32);
            if ( !v23 )
              goto LABEL_27;
          }
          if ( !v24 )
            return (unsigned int)v6;
          v25 = *(_QWORD *)(v23 + 32);
          if ( !v25 )
            break;
          if ( a6 == *(_DWORD *)(v25 + 8) )
            return (unsigned int)v6;
          v23 = *(_QWORD *)(v25 + 32);
          v24 = 0;
          if ( !v23 )
          {
LABEL_27:
            v26 = v24 ^ 1;
            goto LABEL_28;
          }
        }
        v26 = 1;
LABEL_28:
        if ( ((unsigned __int8)v26 & (v19 == 298)) == 0 )
          goto LABEL_29;
        v37 = *(_QWORD *)(a5 + 40);
        v38 = *(_QWORD *)(a3 + 40);
        if ( *(_QWORD *)v38 == *(_QWORD *)v37 && *(_DWORD *)(v38 + 8) == *(_DWORD *)(v37 + 8) )
        {
          v39 = *(_QWORD *)(a3 + 112);
          if ( (*(_BYTE *)(v39 + 37) & 0xF) == 0 )
          {
            v40 = *(_WORD *)(a3 + 32);
            if ( (v40 & 8) == 0 && (*(_BYTE *)(*(_QWORD *)(a5 + 112) + 37LL) & 0xF) == 0 )
            {
              v41 = *(_WORD *)(a5 + 32);
              if ( (v41 & 8) == 0 && (v40 & 0x380) == 0 && (v41 & 0x380) == 0 )
              {
                v42 = *(_WORD *)(a3 + 96);
                if ( v42 == *(_WORD *)(a5 + 96) && (*(_QWORD *)(a3 + 104) == *(_QWORD *)(a5 + 104) || v42) )
                {
                  v43 = (*(_BYTE *)(a5 + 33) >> 2) & 3;
                  if ( (v43 == ((*(_BYTE *)(a3 + 33) >> 2) & 3) || ((*(_BYTE *)(a3 + 33) >> 2) & 3) == 1 || v43 == 1)
                    && !(unsigned int)sub_2EAC1E0(v39)
                    && !(unsigned int)sub_2EAC1E0(*(_QWORD *)(a5 + 112)) )
                  {
                    v44 = *(_QWORD *)(a3 + 40);
                    v45 = *(_QWORD *)(v44 + 40);
                    if ( *(_DWORD *)(v45 + 24) != 39
                      && *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a5 + 40) + 40LL) + 24LL) != 39 )
                    {
                      v46 = *(_QWORD *)(v45 + 48) + 16LL * *(unsigned int *)(v44 + 48);
                      v137 = sub_328A020(a1[1], *(_DWORD *)(a2 + 24), *(_WORD *)v46, *(_QWORD *)(v46 + 8), 0);
                      if ( v137 )
                      {
                        if ( !(unsigned __int8)sub_33CFFC0(a5, a3) && !(unsigned __int8)sub_33CFFC0(a3, a5) )
                        {
                          v161 = 0;
                          v159 = &v163;
                          v156[0] = (unsigned __int64)v157;
                          v162 = 1;
                          v156[1] = 0x1000000000LL;
                          v163 = a2;
                          v160 = 0x100000020LL;
                          v158 = 1;
                          sub_3295920((__int64)v156, a3, v47, v48, v49, v50);
                          sub_3295920((__int64)v156, a5, v51, v52, v53, v54);
                          if ( !(unsigned __int8)sub_3285B00(a3, (__int64)&v158, (__int64)v156, 0, 0, v55)
                            && !(unsigned __int8)sub_3285B00(a5, (__int64)&v158, (__int64)v156, 0, 0, v56) )
                          {
                            v60 = *(__int64 **)(a2 + 40);
                            if ( *(_DWORD *)(a2 + 24) == 205 )
                            {
                              sub_3295920((__int64)v156, *v60, 0, v57, v58, v59);
                              if ( (!(unsigned __int8)sub_33CF8A0(a3, 1, v108, v109, v110, v111)
                                 || !(unsigned __int8)sub_3285B00(a3, (__int64)&v158, (__int64)v156, 0, 0, v115))
                                && (!(unsigned __int8)sub_33CF8A0(a5, 1, v112, v113, v114, v115)
                                 || !(unsigned __int8)sub_3285B00(a5, (__int64)&v158, (__int64)v156, 0, 0, v116)) )
                              {
                                v117 = *(_QWORD *)(a3 + 40);
                                v118 = *(_QWORD *)(a5 + 40);
                                v119 = *a1;
                                v120 = *(__int64 **)(a2 + 40);
                                v121 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v117 + 40) + 48LL)
                                                 + 16LL * *(unsigned int *)(v117 + 48)
                                                 + 8);
                                v122 = *(unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)(v117 + 40) + 48LL)
                                                           + 16LL * *(unsigned int *)(v117 + 48));
                                v152 = *(_QWORD *)(a2 + 80);
                                if ( v152 )
                                {
                                  v131 = v117;
                                  v133 = v120;
                                  v142 = v121;
                                  sub_325F5D0(&v152);
                                  v117 = v131;
                                  v120 = v133;
                                  LODWORD(v121) = v142;
                                }
                                LODWORD(v153) = *(_DWORD *)(a2 + 72);
                                v144 = sub_3288B20(
                                         v119,
                                         (int)&v152,
                                         v122,
                                         v121,
                                         *v120,
                                         v120[1],
                                         *(_OWORD *)(v117 + 40),
                                         *(_OWORD *)(v118 + 40),
                                         0);
                                v145 = v123;
                                sub_9C6650(&v152);
                                goto LABEL_76;
                              }
                            }
                            else
                            {
                              v61 = v60[5];
                              sub_3295920((__int64)v156, *v60, 0, v57, v58, v59);
                              sub_3295920((__int64)v156, v61, v62, v63, v64, v65);
                              if ( (!(unsigned __int8)sub_33CF8A0(a3, 1, v66, v67, v68, v69)
                                 || !(unsigned __int8)sub_3285B00(a3, (__int64)&v158, (__int64)v156, 0, 0, v73))
                                && (!(unsigned __int8)sub_33CF8A0(a5, 1, v70, v71, v72, v73)
                                 || !(unsigned __int8)sub_3285B00(a5, (__int64)&v158, (__int64)v156, 0, 0, v74)) )
                              {
                                v75 = *(_QWORD *)(a3 + 40);
                                v76 = *(_QWORD *)(a5 + 40);
                                v77 = *a1;
                                v78 = *(__int128 **)(a2 + 40);
                                v79 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v75 + 40) + 48LL)
                                                + 16LL * *(unsigned int *)(v75 + 48)
                                                + 8);
                                v80 = *(unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)(v75 + 40) + 48LL)
                                                          + 16LL * *(unsigned int *)(v75 + 48));
                                v152 = *(_QWORD *)(a2 + 80);
                                if ( v152 )
                                {
                                  v125 = v80;
                                  v132 = v75;
                                  v140 = v79;
                                  sub_325F5D0(&v152);
                                  v80 = v125;
                                  v75 = v132;
                                  LODWORD(v79) = v140;
                                }
                                LODWORD(v153) = *(_DWORD *)(a2 + 72);
                                v144 = sub_33FC1D0(
                                         v77,
                                         207,
                                         (unsigned int)&v152,
                                         v80,
                                         v79,
                                         v74,
                                         *v78,
                                         *(__int128 *)((char *)v78 + 40),
                                         *(_OWORD *)(v75 + 40),
                                         *(_OWORD *)(v76 + 40),
                                         v78[10]);
                                v145 = v81;
                                sub_9C6650(&v152);
LABEL_76:
                                v82 = sub_2EAC4F0(*(_QWORD *)(a5 + 112));
                                v83 = *(_QWORD *)(a3 + 112);
                                v84 = v82;
                                LOBYTE(v152) = v82;
                                v85 = sub_2EAC4F0(v83);
                                v86 = (__int64 *)&v150;
                                LOBYTE(v150) = v85;
                                if ( v84 < v85 )
                                  v86 = &v152;
                                v87 = *(_WORD *)(*(_QWORD *)(a3 + 112) + 32LL);
                                v88 = *(_BYTE *)(a5 + 32);
                                v89 = *(_BYTE *)v86;
                                if ( (v88 & 0x40) == 0 )
                                  v87 &= 0x3DFu;
                                if ( (v88 & 0x20) == 0 )
                                  v87 &= 0x3EFu;
                                v90 = *(_BYTE *)(a3 + 33);
                                v152 = 0;
                                v153 = 0;
                                v154 = 0;
                                v91 = *a1;
                                v155 = 0;
                                v92 = *(_QWORD *)(a2 + 80);
                                v141 = *(__int64 **)(a3 + 40);
                                if ( (v90 & 0xC) != 0 )
                                {
                                  v150 = 0u;
                                  v98 = v89;
                                  BYTE4(v151) = 0;
                                  v99 = *(_QWORD *)(a3 + 104);
                                  v100 = *(unsigned __int16 **)(a2 + 48);
                                  v101 = v98;
                                  LODWORD(v151) = 0;
                                  BYTE1(v101) = 1;
                                  v102 = *((_QWORD *)v100 + 1);
                                  v103 = *v100;
                                  v104 = v101;
                                  v148 = v92;
                                  v105 = *(unsigned __int16 *)(a3 + 96);
                                  if ( v92 )
                                  {
                                    v124 = v103;
                                    v128 = v104;
                                    v126 = *(unsigned __int16 *)(a3 + 96);
                                    v127 = v99;
                                    v130 = v102;
                                    sub_325F5D0(&v148);
                                    v90 = *(_BYTE *)(a3 + 33);
                                    v103 = v124;
                                    v104 = v128;
                                    v105 = v126;
                                    v99 = v127;
                                    LODWORD(v102) = v130;
                                  }
                                  v106 = (v90 >> 2) & 3;
                                  v149 = *(_DWORD *)(a2 + 72);
                                  if ( v106 == 1 )
                                    v106 = (*(_BYTE *)(a5 + 33) >> 2) & 3;
                                  v95 = sub_33F1DB0(
                                          v91,
                                          v106,
                                          (unsigned int)&v148,
                                          v103,
                                          v102,
                                          v104,
                                          *v141,
                                          v141[1],
                                          v144,
                                          v145,
                                          v150,
                                          v151,
                                          v105,
                                          v99,
                                          v87,
                                          (__int64)&v152);
                                  v97 = v107;
                                  sub_9C6650(&v148);
                                }
                                else
                                {
                                  v150 = 0u;
                                  LOBYTE(v93) = v89;
                                  BYTE4(v151) = 0;
                                  HIBYTE(v93) = 1;
                                  LODWORD(v151) = 0;
                                  v148 = v92;
                                  if ( v92 )
                                  {
                                    v129 = v93;
                                    sub_325F5D0(&v148);
                                    v93 = v129;
                                  }
                                  v94 = *(unsigned __int16 **)(a2 + 48);
                                  v149 = *(_DWORD *)(a2 + 72);
                                  v95 = sub_33F1F00(
                                          v91,
                                          *v94,
                                          *((_QWORD *)v94 + 1),
                                          (unsigned int)&v148,
                                          *v141,
                                          v141[1],
                                          v144,
                                          v145,
                                          v150,
                                          v151,
                                          v93,
                                          v87,
                                          (__int64)&v152,
                                          0);
                                  v97 = v96;
                                  sub_9C6650(&v148);
                                }
                                LODWORD(v153) = v97;
                                v152 = v95;
                                sub_32EB790((__int64)a1, a2, &v152, 1, 1);
                                sub_32EFDE0((__int64)a1, a3, v95, 0, v95, 1, 1);
                                sub_32EFDE0((__int64)a1, a5, v95, 0, v95, 1, 1);
                                goto LABEL_87;
                              }
                            }
                          }
                          v137 = 0;
LABEL_87:
                          if ( (_BYTE *)v156[0] != v157 )
                            _libc_free(v156[0]);
                          if ( !v162 )
                            _libc_free((unsigned __int64)v159);
                          LODWORD(v6) = v137;
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  return (unsigned int)v6;
}
