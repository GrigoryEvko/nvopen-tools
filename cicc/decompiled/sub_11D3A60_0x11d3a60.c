// Function: sub_11D3A60
// Address: 0x11d3a60
//
__int64 __fastcall sub_11D3A60(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rax
  __int64 v6; // r8
  __int64 v7; // r12
  unsigned int v8; // esi
  __int64 v9; // r9
  __int64 v10; // rdi
  int v11; // r11d
  _QWORD *v12; // r10
  __int64 v13; // rdx
  _QWORD *v14; // rax
  __int64 v15; // rcx
  __int64 v16; // rax
  unsigned __int64 v17; // rdx
  bool v18; // zf
  int v19; // eax
  _DWORD *v20; // r11
  __int64 v21; // rax
  int v22; // r12d
  __int64 v23; // rdx
  __int64 v24; // r14
  __int64 v25; // r13
  int v26; // eax
  unsigned __int64 v27; // r12
  __int64 v28; // r13
  unsigned int v29; // r12d
  __int64 v30; // rax
  __int64 v31; // rax
  __int64 v32; // r8
  __int64 v33; // r9
  int v34; // edx
  __int64 i; // rax
  __int64 v36; // r12
  unsigned __int64 v37; // rcx
  __int64 v38; // r12
  unsigned __int64 v39; // rax
  __int64 v40; // r14
  unsigned int v41; // r13d
  int v42; // r10d
  __int64 *v43; // rdx
  unsigned int v44; // r15d
  unsigned int v45; // edi
  __int64 *v46; // rax
  __int64 v47; // rcx
  __int64 v48; // r12
  __int64 v49; // rax
  unsigned __int64 v50; // rcx
  __int64 v51; // rax
  unsigned int v52; // esi
  __int64 v53; // r12
  int v54; // eax
  int v55; // edi
  __int64 v56; // rsi
  __int64 v57; // rax
  int v58; // ecx
  _BYTE *v59; // rdi
  int v61; // eax
  int v62; // eax
  int v63; // eax
  __int64 v64; // rdi
  unsigned int v65; // r15d
  __int64 v66; // rsi
  __int64 v67; // rax
  _BYTE *v68; // rdx
  __int64 v69; // r12
  __int64 v70; // rax
  __int64 v71; // rcx
  unsigned __int64 v72; // r13
  __int64 v73; // rcx
  __int64 v74; // rdx
  _QWORD *v75; // rcx
  int v76; // ecx
  unsigned int v77; // ecx
  __int64 v78; // rdx
  __int64 v79; // rsi
  unsigned __int64 v80; // rax
  int v81; // edx
  unsigned int v82; // r13d
  int v83; // r10d
  _QWORD *v84; // rdx
  unsigned int v85; // edi
  _QWORD *v86; // rax
  __int64 v87; // rcx
  _QWORD *v88; // rdx
  __int64 v89; // rax
  unsigned int v90; // esi
  __int64 v91; // r12
  int v92; // esi
  int v93; // esi
  __int64 v94; // r8
  unsigned int v95; // ecx
  int v96; // eax
  __int64 v97; // rdi
  __int64 v98; // rax
  int v99; // esi
  __int64 v100; // rdi
  int v101; // esi
  unsigned int v102; // ecx
  __int64 *v103; // rax
  __int64 v104; // r9
  _QWORD *v105; // rax
  _QWORD *v106; // rcx
  _QWORD *v107; // rdx
  _QWORD *v108; // r15
  __int64 v109; // rax
  unsigned __int64 v110; // rdx
  int v111; // eax
  int v112; // ecx
  int v113; // ecx
  __int64 v114; // rdi
  _QWORD *v115; // r9
  unsigned int v116; // r15d
  int v117; // r10d
  __int64 v118; // rsi
  __int64 v119; // rax
  unsigned __int64 v120; // rdx
  int v121; // eax
  _QWORD *v122; // rax
  __int64 v123; // rax
  int v124; // r8d
  int v125; // r15d
  _QWORD *v126; // r10
  int v127; // eax
  int v128; // r10d
  int v129; // edi
  int v130; // edx
  int v131; // r11d
  int v132; // r11d
  __int64 v133; // r10
  __int64 v134; // rdi
  int v135; // esi
  _QWORD *v136; // rcx
  int v137; // r10d
  int v138; // r10d
  unsigned int v139; // r14d
  int v140; // esi
  __int64 v141; // rdi
  unsigned int v142; // r10d
  __int64 v144; // [rsp+30h] [rbp-330h]
  __int64 *v145; // [rsp+38h] [rbp-328h]
  _QWORD *v146; // [rsp+40h] [rbp-320h]
  __int64 v147; // [rsp+48h] [rbp-318h]
  _DWORD *v148; // [rsp+48h] [rbp-318h]
  _QWORD *v149; // [rsp+48h] [rbp-318h]
  int v150; // [rsp+50h] [rbp-310h]
  _DWORD *v151; // [rsp+50h] [rbp-310h]
  _QWORD *v152; // [rsp+50h] [rbp-310h]
  _DWORD *v153; // [rsp+50h] [rbp-310h]
  _DWORD *v154; // [rsp+50h] [rbp-310h]
  _DWORD *v155; // [rsp+58h] [rbp-308h]
  int v156; // [rsp+58h] [rbp-308h]
  _DWORD *v157; // [rsp+58h] [rbp-308h]
  _DWORD *v158; // [rsp+58h] [rbp-308h]
  _DWORD *v159; // [rsp+58h] [rbp-308h]
  _DWORD *v160; // [rsp+58h] [rbp-308h]
  _DWORD *v161; // [rsp+58h] [rbp-308h]
  _BYTE *v162; // [rsp+60h] [rbp-300h] BYREF
  __int64 v163; // [rsp+68h] [rbp-2F8h]
  _BYTE v164[80]; // [rsp+70h] [rbp-2F0h] BYREF
  _BYTE *v165; // [rsp+C0h] [rbp-2A0h] BYREF
  __int64 v166; // [rsp+C8h] [rbp-298h]
  _BYTE v167[80]; // [rsp+D0h] [rbp-290h] BYREF
  _BYTE *v168; // [rsp+120h] [rbp-240h] BYREF
  __int64 v169; // [rsp+128h] [rbp-238h]
  _BYTE v170[560]; // [rsp+130h] [rbp-230h] BYREF

  v162 = v164;
  v163 = 0xA00000000LL;
  v168 = v170;
  v169 = 0x4000000000LL;
  v145 = (__int64 *)(a1 + 56);
  v5 = sub_A777F0(0x40u, (__int64 *)(a1 + 56));
  v7 = v5;
  if ( v5 )
  {
    *(_QWORD *)v5 = a2;
    *(_QWORD *)(v5 + 8) = 0;
    *(_QWORD *)(v5 + 16) = 0;
    *(_DWORD *)(v5 + 24) = 0;
    *(_QWORD *)(v5 + 32) = 0;
    *(_DWORD *)(v5 + 40) = 0;
    *(_QWORD *)(v5 + 48) = 0;
    *(_QWORD *)(v5 + 56) = 0;
  }
  v8 = *(_DWORD *)(a1 + 48);
  v144 = a1 + 24;
  if ( !v8 )
  {
    ++*(_QWORD *)(a1 + 24);
    goto LABEL_174;
  }
  v9 = v8 - 1;
  v10 = *(_QWORD *)(a1 + 32);
  v11 = 1;
  v12 = 0;
  LODWORD(v13) = v9 & (((unsigned int)a2 >> 4) ^ ((unsigned int)a2 >> 9));
  v14 = (_QWORD *)(v10 + 16LL * (unsigned int)v13);
  v15 = *v14;
  if ( *v14 == a2 )
    goto LABEL_5;
  while ( v15 != -4096 )
  {
    if ( !v12 && v15 == -8192 )
      v12 = v14;
    v6 = (unsigned int)(v11 + 1);
    v13 = (unsigned int)v9 & ((_DWORD)v13 + v11);
    v14 = (_QWORD *)(v10 + 16 * v13);
    v15 = *v14;
    if ( *v14 == a2 )
      goto LABEL_5;
    ++v11;
  }
  v129 = *(_DWORD *)(a1 + 40);
  if ( v12 )
    v14 = v12;
  ++*(_QWORD *)(a1 + 24);
  v130 = v129 + 1;
  if ( 4 * (v129 + 1) >= 3 * v8 )
  {
LABEL_174:
    sub_11D3880(v144, 2 * v8);
    v131 = *(_DWORD *)(a1 + 48);
    if ( v131 )
    {
      v132 = v131 - 1;
      v133 = *(_QWORD *)(a1 + 32);
      v9 = v132 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v130 = *(_DWORD *)(a1 + 40) + 1;
      v14 = (_QWORD *)(v133 + 16 * v9);
      v134 = *v14;
      if ( *v14 == a2 )
        goto LABEL_165;
      v135 = 1;
      v136 = 0;
      while ( v134 != -4096 )
      {
        if ( !v136 && v134 == -8192 )
          v136 = v14;
        v6 = (unsigned int)(v135 + 1);
        v9 = v132 & (unsigned int)(v135 + v9);
        v14 = (_QWORD *)(v133 + 16LL * (unsigned int)v9);
        v134 = *v14;
        if ( *v14 == a2 )
          goto LABEL_165;
        ++v135;
      }
LABEL_178:
      if ( v136 )
        v14 = v136;
      goto LABEL_165;
    }
LABEL_210:
    ++*(_DWORD *)(a1 + 40);
    BUG();
  }
  if ( v8 - *(_DWORD *)(a1 + 44) - v130 <= v8 >> 3 )
  {
    sub_11D3880(v144, v8);
    v137 = *(_DWORD *)(a1 + 48);
    if ( v137 )
    {
      v138 = v137 - 1;
      v9 = *(_QWORD *)(a1 + 32);
      v136 = 0;
      v139 = v138 & (((unsigned int)a2 >> 4) ^ ((unsigned int)a2 >> 9));
      v130 = *(_DWORD *)(a1 + 40) + 1;
      v140 = 1;
      v14 = (_QWORD *)(v9 + 16LL * v139);
      v141 = *v14;
      if ( *v14 == a2 )
        goto LABEL_165;
      while ( v141 != -4096 )
      {
        if ( !v136 && v141 == -8192 )
          v136 = v14;
        v6 = (unsigned int)(v140 + 1);
        v139 = v138 & (v140 + v139);
        v14 = (_QWORD *)(v9 + 16LL * v139);
        v141 = *v14;
        if ( *v14 == a2 )
          goto LABEL_165;
        ++v140;
      }
      goto LABEL_178;
    }
    goto LABEL_210;
  }
LABEL_165:
  *(_DWORD *)(a1 + 40) = v130;
  if ( *v14 != -4096 )
    --*(_DWORD *)(a1 + 44);
  *v14 = a2;
  v14[1] = 0;
LABEL_5:
  v14[1] = v7;
  v16 = (unsigned int)v169;
  v17 = (unsigned int)v169 + 1LL;
  if ( v17 > HIDWORD(v169) )
  {
    sub_C8D5F0((__int64)&v168, v170, v17, 8u, v6, v9);
    v16 = (unsigned int)v169;
  }
  *(_QWORD *)&v168[8 * v16] = v7;
  v165 = v167;
  v18 = (_DWORD)v169 == -1;
  v19 = v169 + 1;
  v166 = 0xA00000000LL;
  LODWORD(v169) = v169 + 1;
  if ( !v18 )
  {
    do
    {
      v20 = *(_DWORD **)&v168[8 * v19 - 8];
      LODWORD(v169) = v19 - 1;
      LODWORD(v166) = 0;
      v21 = *(_QWORD *)(*(_QWORD *)v20 + 56LL);
      if ( !v21 )
LABEL_143:
        BUG();
      if ( *(_BYTE *)(v21 - 24) == 84 )
      {
        v22 = *(_DWORD *)(v21 - 20);
        v23 = 0;
        v24 = *(_QWORD *)(v21 - 32);
        v25 = *(unsigned int *)(v21 + 48);
        v26 = 0;
        v27 = v22 & 0x7FFFFFF;
        if ( v27 > HIDWORD(v166) )
        {
          v161 = v20;
          sub_C8D5F0((__int64)&v165, v167, v27, 8u, v6, v9);
          v26 = v166;
          v20 = v161;
          v23 = 8LL * (unsigned int)v166;
        }
        v28 = 32 * v25;
        if ( v24 + v28 != v24 + 8 * v27 + v28 )
        {
          v155 = v20;
          memcpy(&v165[v23], (const void *)(v24 + v28), 8 * v27);
          v26 = v166;
          v20 = v155;
        }
        v29 = v26 + v27;
        v30 = v29;
        LODWORD(v166) = v29;
        v20[10] = v29;
        if ( v29 )
          goto LABEL_89;
      }
      else
      {
        v67 = *(_QWORD *)(*(_QWORD *)v20 + 16LL);
        do
        {
          if ( !v67 )
          {
            LODWORD(v72) = 0;
            v76 = 0;
            goto LABEL_88;
          }
          v68 = *(_BYTE **)(v67 + 24);
          v69 = v67;
          v67 = *(_QWORD *)(v67 + 8);
        }
        while ( (unsigned __int8)(*v68 - 30) > 0xAu );
        v70 = v69;
        v71 = 0;
        while ( 1 )
        {
          v70 = *(_QWORD *)(v70 + 8);
          if ( !v70 )
            break;
          while ( (unsigned __int8)(**(_BYTE **)(v70 + 24) - 30) <= 0xAu )
          {
            v70 = *(_QWORD *)(v70 + 8);
            ++v71;
            if ( !v70 )
              goto LABEL_79;
          }
        }
LABEL_79:
        v72 = v71 + 1;
        v73 = 0;
        if ( v72 > HIDWORD(v166) )
        {
          v157 = v20;
          sub_C8D5F0((__int64)&v165, v167, v72, 8u, v6, v9);
          v20 = v157;
          v73 = 8LL * (unsigned int)v166;
        }
        v74 = *(_QWORD *)(v69 + 24);
        v75 = &v165[v73];
LABEL_84:
        if ( v75 )
          *v75 = *(_QWORD *)(v74 + 40);
        while ( 1 )
        {
          v69 = *(_QWORD *)(v69 + 8);
          if ( !v69 )
            break;
          v74 = *(_QWORD *)(v69 + 24);
          if ( (unsigned __int8)(*(_BYTE *)v74 - 30) <= 0xAu )
          {
            ++v75;
            goto LABEL_84;
          }
        }
        v76 = v166;
LABEL_88:
        v77 = v72 + v76;
        v30 = v77;
        LODWORD(v166) = v77;
        v20[10] = v77;
        if ( v77 )
        {
LABEL_89:
          v78 = *(_QWORD *)(a1 + 56);
          v79 = 8 * v30;
          *(_QWORD *)(a1 + 136) += 8 * v30;
          v80 = (v78 + 7) & 0xFFFFFFFFFFFFFFF8LL;
          if ( *(_QWORD *)(a1 + 64) >= v79 + v80 && v78 )
          {
            *(_QWORD *)(a1 + 56) = v79 + v80;
          }
          else
          {
            v160 = v20;
            v80 = sub_9D1E70((__int64)v145, v79, v79, 3);
            v20 = v160;
          }
          v81 = v20[10];
          *((_QWORD *)v20 + 6) = v80;
          if ( !v81 )
            goto LABEL_16;
          v82 = 0;
          while ( 2 )
          {
            v90 = *(_DWORD *)(a1 + 48);
            v91 = *(_QWORD *)&v165[8 * v82];
            if ( !v90 )
            {
              ++*(_QWORD *)(a1 + 24);
              goto LABEL_100;
            }
            v9 = v90 - 1;
            v6 = *(_QWORD *)(a1 + 32);
            v83 = 1;
            v84 = 0;
            v85 = v9 & (((unsigned int)v91 >> 9) ^ ((unsigned int)v91 >> 4));
            v86 = (_QWORD *)(v6 + 16LL * v85);
            v87 = *v86;
            if ( v91 == *v86 )
            {
LABEL_95:
              v88 = v86 + 1;
              v89 = v86[1];
              if ( v89 )
              {
                *(_QWORD *)(*((_QWORD *)v20 + 6) + 8LL * v82) = v89;
                goto LABEL_97;
              }
            }
            else
            {
              while ( v87 != -4096 )
              {
                if ( !v84 && v87 == -8192 )
                  v84 = v86;
                v85 = v9 & (v83 + v85);
                v86 = (_QWORD *)(v6 + 16LL * v85);
                v87 = *v86;
                if ( v91 == *v86 )
                  goto LABEL_95;
                ++v83;
              }
              if ( !v84 )
                v84 = v86;
              v111 = *(_DWORD *)(a1 + 40);
              ++*(_QWORD *)(a1 + 24);
              v96 = v111 + 1;
              if ( 4 * v96 >= 3 * v90 )
              {
LABEL_100:
                v151 = v20;
                sub_11D3880(v144, 2 * v90);
                v92 = *(_DWORD *)(a1 + 48);
                if ( !v92 )
                  goto LABEL_210;
                v93 = v92 - 1;
                v94 = *(_QWORD *)(a1 + 32);
                v20 = v151;
                v95 = v93 & (((unsigned int)v91 >> 9) ^ ((unsigned int)v91 >> 4));
                v96 = *(_DWORD *)(a1 + 40) + 1;
                v84 = (_QWORD *)(v94 + 16LL * v95);
                v97 = *v84;
                if ( v91 != *v84 )
                {
                  v125 = 1;
                  v126 = 0;
                  while ( v97 != -4096 )
                  {
                    if ( v97 == -8192 && !v126 )
                      v126 = v84;
                    v95 = v93 & (v125 + v95);
                    v84 = (_QWORD *)(v94 + 16LL * v95);
                    v97 = *v84;
                    if ( v91 == *v84 )
                      goto LABEL_102;
                    ++v125;
                  }
                  if ( v126 )
                    v84 = v126;
                }
              }
              else if ( v90 - *(_DWORD *)(a1 + 44) - v96 <= v90 >> 3 )
              {
                v153 = v20;
                sub_11D3880(v144, v90);
                v112 = *(_DWORD *)(a1 + 48);
                if ( !v112 )
                  goto LABEL_210;
                v113 = v112 - 1;
                v114 = *(_QWORD *)(a1 + 32);
                v115 = 0;
                v116 = v113 & (((unsigned int)v91 >> 9) ^ ((unsigned int)v91 >> 4));
                v20 = v153;
                v117 = 1;
                v96 = *(_DWORD *)(a1 + 40) + 1;
                v84 = (_QWORD *)(v114 + 16LL * v116);
                v118 = *v84;
                if ( v91 != *v84 )
                {
                  while ( v118 != -4096 )
                  {
                    if ( v118 == -8192 && !v115 )
                      v115 = v84;
                    v116 = v113 & (v117 + v116);
                    v84 = (_QWORD *)(v114 + 16LL * v116);
                    v118 = *v84;
                    if ( v91 == *v84 )
                      goto LABEL_102;
                    ++v117;
                  }
                  if ( v115 )
                    v84 = v115;
                }
              }
LABEL_102:
              *(_DWORD *)(a1 + 40) = v96;
              if ( *v84 != -4096 )
                --*(_DWORD *)(a1 + 44);
              *v84 = v91;
              v88 = v84 + 1;
              *v88 = 0;
            }
            v98 = *(_QWORD *)(a1 + 8);
            v99 = *(_DWORD *)(v98 + 24);
            v100 = *(_QWORD *)(v98 + 8);
            if ( v99 )
            {
              v101 = v99 - 1;
              v102 = v101 & (((unsigned int)v91 >> 9) ^ ((unsigned int)v91 >> 4));
              v103 = (__int64 *)(v100 + 16LL * v102);
              v104 = *v103;
              if ( v91 == *v103 )
              {
LABEL_107:
                v146 = v88;
                v148 = v20;
                v152 = (_QWORD *)v103[1];
                v105 = (_QWORD *)sub_A777F0(0x40u, v145);
                v106 = v152;
                v20 = v148;
                v107 = v146;
                v108 = v105;
                if ( !v105 )
                  goto LABEL_209;
                *v105 = v91;
                v105[1] = v152;
                if ( v152 )
                  v106 = v105;
LABEL_110:
                v108[2] = v106;
                *((_DWORD *)v108 + 6) = 0;
                v108[4] = 0;
                *((_DWORD *)v108 + 10) = 0;
                v108[6] = 0;
                v108[7] = 0;
                *v107 = v108;
                *(_QWORD *)(*((_QWORD *)v20 + 6) + 8LL * v82) = v108;
                if ( v108[1] )
                {
                  v109 = (unsigned int)v163;
                  v110 = (unsigned int)v163 + 1LL;
                  if ( v110 > HIDWORD(v163) )
                  {
                    v158 = v20;
                    sub_C8D5F0((__int64)&v162, v164, v110, 8u, v6, v9);
                    v109 = (unsigned int)v163;
                    v20 = v158;
                  }
                  *(_QWORD *)&v162[8 * v109] = v108;
                  LODWORD(v163) = v163 + 1;
                }
                else
                {
                  v119 = (unsigned int)v169;
                  v120 = (unsigned int)v169 + 1LL;
                  if ( v120 > HIDWORD(v169) )
                  {
                    v159 = v20;
                    sub_C8D5F0((__int64)&v168, v170, v120, 8u, v6, v9);
                    v119 = (unsigned int)v169;
                    v20 = v159;
                  }
                  *(_QWORD *)&v168[8 * v119] = v108;
                  LODWORD(v169) = v169 + 1;
                }
LABEL_97:
                if ( v20[10] == ++v82 )
                  goto LABEL_16;
                continue;
              }
              v121 = 1;
              while ( v104 != -4096 )
              {
                v124 = v121 + 1;
                v102 = v101 & (v121 + v102);
                v103 = (__int64 *)(v100 + 16LL * v102);
                v104 = *v103;
                if ( v91 == *v103 )
                  goto LABEL_107;
                v121 = v124;
              }
            }
            break;
          }
          v149 = v88;
          v154 = v20;
          v122 = (_QWORD *)sub_A777F0(0x40u, v145);
          v20 = v154;
          v107 = v149;
          v108 = v122;
          if ( !v122 )
          {
LABEL_209:
            *v107 = 0;
            *(_QWORD *)(*((_QWORD *)v20 + 6) + 8LL * v82) = 0;
            BUG();
          }
          *v122 = v91;
          v106 = 0;
          v122[1] = 0;
          goto LABEL_110;
        }
      }
      *((_QWORD *)v20 + 6) = 0;
LABEL_16:
      v19 = v169;
    }
    while ( (_DWORD)v169 );
  }
  v31 = sub_A777F0(0x40u, v145);
  v147 = v31;
  if ( v31 )
  {
    *(_QWORD *)v31 = 0;
    *(_QWORD *)(v31 + 8) = 0;
    *(_QWORD *)(v31 + 16) = 0;
    *(_DWORD *)(v31 + 24) = 0;
    *(_QWORD *)(v31 + 32) = 0;
    *(_DWORD *)(v31 + 40) = 0;
    *(_QWORD *)(v31 + 48) = 0;
    *(_QWORD *)(v31 + 56) = 0;
  }
  v34 = v163;
  for ( i = (unsigned int)v169; (_DWORD)v163; LODWORD(v169) = v169 + 1 )
  {
    v36 = *(_QWORD *)&v162[8 * v34 - 8];
    v37 = HIDWORD(v169);
    LODWORD(v163) = v34 - 1;
    *(_QWORD *)(v36 + 32) = v147;
    *(_DWORD *)(v36 + 24) = -1;
    if ( i + 1 > v37 )
    {
      sub_C8D5F0((__int64)&v168, v170, i + 1, 8u, v32, v33);
      i = (unsigned int)v169;
    }
    *(_QWORD *)&v168[8 * i] = v36;
    v34 = v163;
    i = (unsigned int)(v169 + 1);
  }
  v150 = 1;
  if ( !(_DWORD)i )
  {
    v127 = 1;
    goto LABEL_49;
  }
  do
  {
    while ( 1 )
    {
      v38 = *(_QWORD *)&v168[8 * i - 8];
      if ( *(_DWORD *)(v38 + 24) == -2 )
        break;
      *(_DWORD *)(v38 + 24) = -2;
      v39 = *(_QWORD *)(*(_QWORD *)v38 + 48LL) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v39 != *(_QWORD *)v38 + 48LL )
      {
        if ( !v39 )
          goto LABEL_143;
        v40 = v39 - 24;
        if ( (unsigned int)*(unsigned __int8 *)(v39 - 24) - 30 <= 0xA )
        {
          v156 = sub_B46E30(v40);
          if ( v156 )
          {
            v41 = 0;
            while ( 1 )
            {
              v51 = sub_B46EC0(v40, v41);
              v52 = *(_DWORD *)(a1 + 48);
              v53 = v51;
              if ( !v52 )
                break;
              v33 = v52 - 1;
              v32 = *(_QWORD *)(a1 + 32);
              v42 = 1;
              v43 = 0;
              v44 = ((unsigned int)v51 >> 9) ^ ((unsigned int)v51 >> 4);
              v45 = v33 & v44;
              v46 = (__int64 *)(v32 + 16LL * ((unsigned int)v33 & v44));
              v47 = *v46;
              if ( v53 == *v46 )
              {
LABEL_34:
                v48 = v46[1];
                if ( v48 && !*(_DWORD *)(v48 + 24) )
                {
                  v49 = (unsigned int)v169;
                  v50 = HIDWORD(v169);
                  *(_DWORD *)(v48 + 24) = -1;
                  if ( v49 + 1 > v50 )
                  {
                    sub_C8D5F0((__int64)&v168, v170, v49 + 1, 8u, v32, v33);
                    v49 = (unsigned int)v169;
                  }
                  *(_QWORD *)&v168[8 * v49] = v48;
                  LODWORD(v169) = v169 + 1;
                }
                if ( v156 == ++v41 )
                  goto LABEL_47;
              }
              else
              {
                while ( v47 != -4096 )
                {
                  if ( v47 == -8192 && !v43 )
                    v43 = v46;
                  v45 = v33 & (v42 + v45);
                  v46 = (__int64 *)(v32 + 16LL * v45);
                  v47 = *v46;
                  if ( v53 == *v46 )
                    goto LABEL_34;
                  ++v42;
                }
                if ( !v43 )
                  v43 = v46;
                v61 = *(_DWORD *)(a1 + 40);
                ++*(_QWORD *)(a1 + 24);
                v58 = v61 + 1;
                if ( 4 * (v61 + 1) < 3 * v52 )
                {
                  if ( v52 - *(_DWORD *)(a1 + 44) - v58 <= v52 >> 3 )
                  {
                    sub_11D3880(v144, v52);
                    v62 = *(_DWORD *)(a1 + 48);
                    if ( !v62 )
                      goto LABEL_210;
                    v63 = v62 - 1;
                    v64 = *(_QWORD *)(a1 + 32);
                    v33 = 1;
                    v65 = v63 & v44;
                    v32 = 0;
                    v58 = *(_DWORD *)(a1 + 40) + 1;
                    v43 = (__int64 *)(v64 + 16LL * v65);
                    v66 = *v43;
                    if ( v53 != *v43 )
                    {
                      while ( v66 != -4096 )
                      {
                        if ( !v32 && v66 == -8192 )
                          v32 = (__int64)v43;
                        v142 = v33 + 1;
                        v33 = v63 & (v65 + (unsigned int)v33);
                        v65 = v33;
                        v43 = (__int64 *)(v64 + 16LL * (unsigned int)v33);
                        v66 = *v43;
                        if ( v53 == *v43 )
                          goto LABEL_44;
                        v33 = v142;
                      }
                      if ( v32 )
                        v43 = (__int64 *)v32;
                    }
                  }
                  goto LABEL_44;
                }
LABEL_42:
                sub_11D3880(v144, 2 * v52);
                v54 = *(_DWORD *)(a1 + 48);
                if ( !v54 )
                  goto LABEL_210;
                v55 = v54 - 1;
                v56 = *(_QWORD *)(a1 + 32);
                LODWORD(v57) = (v54 - 1) & (((unsigned int)v53 >> 9) ^ ((unsigned int)v53 >> 4));
                v58 = *(_DWORD *)(a1 + 40) + 1;
                v43 = (__int64 *)(v56 + 16LL * (unsigned int)v57);
                v32 = *v43;
                if ( v53 != *v43 )
                {
                  v128 = 1;
                  v33 = 0;
                  while ( v32 != -4096 )
                  {
                    if ( !v33 && v32 == -8192 )
                      v33 = (__int64)v43;
                    v57 = v55 & (unsigned int)(v57 + v128);
                    v43 = (__int64 *)(v56 + 16 * v57);
                    v32 = *v43;
                    if ( v53 == *v43 )
                      goto LABEL_44;
                    ++v128;
                  }
                  if ( v33 )
                    v43 = (__int64 *)v33;
                }
LABEL_44:
                *(_DWORD *)(a1 + 40) = v58;
                if ( *v43 != -4096 )
                  --*(_DWORD *)(a1 + 44);
                *v43 = v53;
                ++v41;
                v43[1] = 0;
                if ( v156 == v41 )
                  goto LABEL_47;
              }
            }
            ++*(_QWORD *)(a1 + 24);
            goto LABEL_42;
          }
        }
      }
LABEL_47:
      i = (unsigned int)v169;
      if ( !(_DWORD)v169 )
        goto LABEL_48;
    }
    v18 = *(_QWORD *)(v38 + 8) == 0;
    *(_DWORD *)(v38 + 24) = v150;
    if ( v18 )
    {
      v123 = *(unsigned int *)(a3 + 8);
      if ( v123 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
      {
        sub_C8D5F0(a3, (const void *)(a3 + 16), v123 + 1, 8u, v32, v33);
        v123 = *(unsigned int *)(a3 + 8);
      }
      *(_QWORD *)(*(_QWORD *)a3 + 8 * v123) = v38;
      ++*(_DWORD *)(a3 + 8);
    }
    ++v150;
    i = (unsigned int)(v169 - 1);
    LODWORD(v169) = i;
  }
  while ( (_DWORD)i );
LABEL_48:
  v127 = v150;
LABEL_49:
  v59 = v165;
  *(_DWORD *)(v147 + 24) = v127;
  if ( v59 != v167 )
    _libc_free(v59, v147);
  if ( v168 != v170 )
    _libc_free(v168, v147);
  if ( v162 != v164 )
    _libc_free(v162, v147);
  return v147;
}
