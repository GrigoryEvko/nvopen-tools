// Function: sub_352E500
// Address: 0x352e500
//
__int64 __fastcall sub_352E500(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 (*v3)(void); // rax
  __int64 v4; // r12
  int v5; // r13d
  __int64 *v6; // r15
  __int64 v7; // r14
  int v8; // esi
  _QWORD *v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rsi
  unsigned __int8 *v12; // rsi
  _BYTE *v13; // rax
  __int64 v14; // r15
  __int64 v15; // r13
  __int64 v16; // r12
  __int64 v17; // rax
  int v18; // eax
  unsigned int v19; // eax
  unsigned int v20; // ebx
  __int64 v21; // r13
  __int64 v22; // rdx
  int v23; // r14d
  int v24; // ecx
  unsigned int v25; // r8d
  __int64 v26; // rax
  int v27; // edi
  __int64 v28; // r14
  __int64 v29; // rdi
  __int64 v30; // rax
  __int64 v31; // rdx
  __int64 v32; // r12
  __int64 v33; // rbx
  __int64 v34; // rax
  __int64 v35; // rsi
  unsigned int v36; // r13d
  __int64 v37; // rdx
  unsigned int v38; // r9d
  unsigned int *v39; // rax
  unsigned int v40; // edi
  __int64 *v41; // rax
  __int64 v42; // rsi
  __int64 v43; // rax
  __int64 v44; // r12
  const __m128i *v45; // r14
  __int64 v46; // r15
  __int64 v47; // rax
  __int64 v48; // r12
  __int64 v49; // r13
  __int64 v50; // r14
  __int64 v51; // rsi
  __int64 v52; // rax
  __int64 v53; // rax
  __int64 v54; // rdx
  unsigned __int8 v55; // al
  __int64 v56; // rax
  __int64 v57; // rdx
  _QWORD *v58; // rax
  __int64 v59; // rsi
  __int64 v60; // rax
  __int64 v61; // rax
  __int64 v62; // rdi
  __int64 v63; // rsi
  unsigned int *v65; // rcx
  int v66; // eax
  unsigned int v67; // esi
  unsigned int v68; // r9d
  int v69; // r8d
  unsigned int *v70; // rdi
  unsigned int *v71; // rsi
  int v72; // edi
  unsigned int v73; // r8d
  unsigned int v74; // r9d
  int v75; // eax
  __int64 v76; // rax
  int v77; // eax
  __int64 v78; // r9
  unsigned int v79; // esi
  __int64 v80; // rdi
  __int64 v81; // r8
  unsigned int v82; // edx
  unsigned int v83; // ebx
  int v84; // r11d
  __int64 v85; // rdx
  __int64 v86; // r10
  unsigned int *v87; // rax
  __int64 v88; // rcx
  char *v89; // rax
  __m128i *v90; // r13
  unsigned __int64 *v91; // r12
  __int64 v92; // r8
  __int64 v93; // r9
  __int64 v94; // rbx
  const __m128i *v95; // r15
  __m128i *v96; // r13
  int v97; // ecx
  const __m128i **v98; // rbx
  const __m128i **v99; // rbx
  int v100; // ecx
  __int64 v101; // rsi
  unsigned int *v102; // rsi
  int v103; // edi
  __int64 v104; // r12
  __int64 v105; // rax
  __int64 v106; // rax
  __int64 v107; // rsi
  __int64 v108; // rax
  __int64 v109; // rax
  __int64 v110; // rcx
  int v111; // edx
  int v112; // r11d
  int v113; // edi
  __int64 v114; // rsi
  int v115; // esi
  __int64 v116; // r14
  __int64 v117; // rcx
  int v118; // r8d
  int v119; // edi
  __int64 v120; // [rsp+0h] [rbp-210h]
  unsigned __int64 *v121; // [rsp+8h] [rbp-208h]
  const __m128i *v122; // [rsp+10h] [rbp-200h]
  __int64 v123; // [rsp+18h] [rbp-1F8h]
  _WORD *v124; // [rsp+20h] [rbp-1F0h]
  __int64 v125; // [rsp+28h] [rbp-1E8h]
  int v126; // [rsp+34h] [rbp-1DCh]
  __int64 v127; // [rsp+58h] [rbp-1B8h]
  __int64 v128; // [rsp+60h] [rbp-1B0h]
  __int64 *v129; // [rsp+68h] [rbp-1A8h]
  __int64 v130; // [rsp+70h] [rbp-1A0h]
  __int64 v131; // [rsp+78h] [rbp-198h]
  __int64 v132; // [rsp+80h] [rbp-190h]
  const __m128i **v133; // [rsp+80h] [rbp-190h]
  __int64 v134; // [rsp+88h] [rbp-188h]
  __int64 v135; // [rsp+90h] [rbp-180h]
  __int64 v136; // [rsp+98h] [rbp-178h]
  __int64 v137; // [rsp+98h] [rbp-178h]
  __int64 v138; // [rsp+A0h] [rbp-170h]
  unsigned __int64 *v139; // [rsp+A0h] [rbp-170h]
  int v140; // [rsp+A0h] [rbp-170h]
  __int64 v141; // [rsp+A0h] [rbp-170h]
  __int64 v142; // [rsp+A0h] [rbp-170h]
  __int64 v144; // [rsp+A8h] [rbp-168h]
  unsigned __int8 **v145; // [rsp+A8h] [rbp-168h]
  unsigned __int64 *v146; // [rsp+B0h] [rbp-160h]
  __int64 i; // [rsp+B8h] [rbp-158h]
  __int64 j; // [rsp+B8h] [rbp-158h]
  __int64 v149; // [rsp+B8h] [rbp-158h]
  __int64 v150; // [rsp+C0h] [rbp-150h] BYREF
  __int64 v151; // [rsp+C8h] [rbp-148h]
  __int64 v152; // [rsp+D0h] [rbp-140h]
  unsigned int v153; // [rsp+D8h] [rbp-138h]
  __m128i v154; // [rsp+E0h] [rbp-130h] BYREF
  __int64 v155; // [rsp+F0h] [rbp-120h]
  __int64 v156; // [rsp+F8h] [rbp-118h]
  const __m128i **v157; // [rsp+110h] [rbp-100h] BYREF
  __int64 v158; // [rsp+118h] [rbp-F8h]
  _BYTE v159[32]; // [rsp+120h] [rbp-F0h] BYREF
  unsigned __int8 *v160; // [rsp+140h] [rbp-D0h] BYREF
  char *v161; // [rsp+148h] [rbp-C8h]
  __int64 v162; // [rsp+150h] [rbp-C0h]
  int v163; // [rsp+158h] [rbp-B8h]
  char v164; // [rsp+15Ch] [rbp-B4h]
  char v165; // [rsp+160h] [rbp-B0h] BYREF

  v2 = sub_2EAA2D0(a1, a2);
  v136 = v2;
  if ( !v2 )
    return 0;
  v132 = 0;
  v3 = *(__int64 (**)(void))(**(_QWORD **)(v2 + 16) + 128LL);
  if ( v3 != sub_2DAC790 )
    v132 = v3();
  v4 = sub_B92180(a2);
  v5 = *(_DWORD *)(v4 + 16);
  v6 = **(__int64 ***)(a2 + 40);
  v128 = *(_QWORD *)(a2 + 40);
  v126 = v5;
  v129 = v6;
  v127 = v136 + 320;
  if ( *(_QWORD *)(v136 + 328) != v136 + 320 )
  {
    v138 = *(_QWORD *)(v136 + 328);
    do
    {
      v7 = *(_QWORD *)(v138 + 56);
      for ( i = v138 + 48; i != v7; v7 = *(_QWORD *)(v7 + 8) )
      {
        while ( 1 )
        {
          v8 = v5++;
          v9 = sub_B01860(v6, v8, 1u, v4, 0, 0, 0, 1);
          sub_B10CB0(&v160, (__int64)v9);
          v10 = v7 + 56;
          if ( (unsigned __int8 **)(v7 + 56) == &v160 )
          {
            if ( v160 )
              sub_B91220((__int64)&v160, (__int64)v160);
            if ( !v7 )
              BUG();
          }
          else
          {
            v11 = *(_QWORD *)(v7 + 56);
            if ( v11 )
            {
              sub_B91220(v7 + 56, v11);
              v10 = v7 + 56;
            }
            v12 = v160;
            *(_QWORD *)(v7 + 56) = v160;
            if ( v12 )
              sub_B976B0((__int64)&v160, v12, v10);
          }
          if ( (*(_BYTE *)v7 & 4) == 0 )
            break;
          v7 = *(_QWORD *)(v7 + 8);
          if ( i == v7 )
            goto LABEL_16;
        }
        while ( (*(_BYTE *)(v7 + 44) & 8) != 0 )
          v7 = *(_QWORD *)(v7 + 8);
      }
LABEL_16:
      v138 = *(_QWORD *)(v138 + 8);
    }
    while ( v127 != v138 );
    v126 = v5;
  }
  v13 = sub_BA8CB0(v128, (__int64)"llvm.dbg.value", 0xEu);
  v150 = 0;
  v151 = 0;
  v152 = 0;
  v153 = 0;
  if ( !v13 || (v14 = *((_QWORD *)v13 + 2)) == 0 )
  {
    v15 = 0;
    v62 = 0;
    v63 = 0;
    v130 = 0;
    v135 = a2 + 72;
    v144 = *(_QWORD *)(a2 + 80);
    if ( v135 == v144 )
      goto LABEL_92;
    goto LABEL_38;
  }
  v130 = 0;
  v15 = 0;
  do
  {
    while ( 1 )
    {
      v16 = *(_QWORD *)(v14 + 24);
      if ( *(_BYTE *)v16 == 85 )
      {
        v17 = *(_QWORD *)(v16 - 32);
        if ( v17 )
        {
          if ( !*(_BYTE *)v17 && *(_QWORD *)(v17 + 24) == *(_QWORD *)(v16 + 80) && (*(_BYTE *)(v17 + 33) & 0x20) != 0 )
          {
            v18 = *(_DWORD *)(v17 + 36);
            if ( (v18 == 71 || v18 == 68) && a2 == sub_B43CB0(*(_QWORD *)(v14 + 24)) )
              break;
          }
        }
      }
      v14 = *(_QWORD *)(v14 + 8);
      if ( !v14 )
        goto LABEL_37;
    }
    v19 = sub_B10CE0(v16 + 48);
    v20 = v19;
    v21 = *(_QWORD *)(*(_QWORD *)(v16 + 32 * (1LL - (*(_DWORD *)(v16 + 4) & 0x7FFFFFF))) + 24LL);
    if ( v153 )
    {
      v22 = 0;
      v23 = 37 * v19;
      v24 = 1;
      v25 = (v153 - 1) & (37 * v19);
      v26 = v151 + 16LL * v25;
      v27 = *(_DWORD *)v26;
      if ( v20 == *(_DWORD *)v26 )
        goto LABEL_32;
      while ( v27 != -1 )
      {
        if ( v27 == -2 && !v22 )
          v22 = v26;
        v25 = (v153 - 1) & (v24 + v25);
        v26 = v151 + 16LL * v25;
        v27 = *(_DWORD *)v26;
        if ( v20 == *(_DWORD *)v26 )
          goto LABEL_32;
        ++v24;
      }
      if ( v22 )
        v26 = v22;
      ++v150;
      v111 = v152 + 1;
      if ( 4 * ((int)v152 + 1) < 3 * v153 )
      {
        if ( v153 - HIDWORD(v152) - v111 <= v153 >> 3 )
        {
          sub_352E320((__int64)&v150, v153);
          if ( !v153 )
          {
LABEL_276:
            LODWORD(v152) = v152 + 1;
            BUG();
          }
          v115 = 1;
          LODWORD(v116) = (v153 - 1) & v23;
          v111 = v152 + 1;
          v117 = 0;
          v26 = v151 + 16LL * (unsigned int)v116;
          v118 = *(_DWORD *)v26;
          if ( v20 != *(_DWORD *)v26 )
          {
            while ( v118 != -1 )
            {
              if ( v118 == -2 && !v117 )
                v117 = v26;
              v116 = (v153 - 1) & ((_DWORD)v116 + v115);
              v26 = v151 + 16 * v116;
              v118 = *(_DWORD *)v26;
              if ( v20 == *(_DWORD *)v26 )
                goto LABEL_231;
              ++v115;
            }
            if ( v117 )
              v26 = v117;
          }
        }
        goto LABEL_231;
      }
    }
    else
    {
      ++v150;
    }
    sub_352E320((__int64)&v150, 2 * v153);
    if ( !v153 )
      goto LABEL_276;
    LODWORD(v110) = (v153 - 1) & (37 * v20);
    v111 = v152 + 1;
    v26 = v151 + 16LL * (unsigned int)v110;
    v112 = *(_DWORD *)v26;
    if ( v20 != *(_DWORD *)v26 )
    {
      v113 = 1;
      v114 = 0;
      while ( v112 != -1 )
      {
        if ( v112 == -2 && !v114 )
          v114 = v26;
        v110 = (v153 - 1) & ((_DWORD)v110 + v113);
        v26 = v151 + 16 * v110;
        v112 = *(_DWORD *)v26;
        if ( v20 == *(_DWORD *)v26 )
          goto LABEL_231;
        ++v113;
      }
      if ( v114 )
        v26 = v114;
    }
LABEL_231:
    LODWORD(v152) = v111;
    if ( *(_DWORD *)v26 != -1 )
      --HIDWORD(v152);
    *(_DWORD *)v26 = v20;
    *(_QWORD *)(v26 + 8) = 0;
LABEL_32:
    v28 = v130;
    *(_QWORD *)(v26 + 8) = v21;
    if ( v130 )
    {
      if ( v20 < (unsigned int)sub_B10CE0(v130 + 48) )
        v28 = v16;
      v130 = v28;
    }
    else
    {
      v130 = v16;
    }
    v14 = *(_QWORD *)(v14 + 8);
    v15 = *(_QWORD *)(*(_QWORD *)(v16 + 32 * (2LL - (*(_DWORD *)(v16 + 4) & 0x7FFFFFF))) + 24LL);
  }
  while ( v14 );
LABEL_37:
  v135 = a2 + 72;
  v144 = *(_QWORD *)(a2 + 80);
  if ( v144 != v135 )
  {
LABEL_38:
    v14 = 0;
    do
    {
      if ( !v144 )
        BUG();
      for ( j = *(_QWORD *)(v144 + 32); v144 + 24 != j; j = *(_QWORD *)(j + 8) )
      {
        if ( !j )
          BUG();
        v29 = *(_QWORD *)(j + 40);
        if ( v29 )
        {
          v30 = sub_B14240(v29);
          v32 = v31;
          v33 = v30;
          if ( v31 != v30 )
          {
            while ( *(_BYTE *)(v33 + 32) )
            {
              v33 = *(_QWORD *)(v33 + 8);
              if ( v33 == v31 )
                goto LABEL_68;
            }
            if ( v33 != v31 )
            {
              v34 = v15;
              while ( 1 )
              {
                if ( *(_BYTE *)(v33 + 64) != 1 )
                  goto LABEL_66;
                v35 = *(_QWORD *)(v33 + 24);
                v160 = (unsigned __int8 *)v35;
                if ( v35 )
                  sub_B96E90((__int64)&v160, v35, 1);
                v36 = sub_B10CE0((__int64)&v160);
                if ( v160 )
                  sub_B91220((__int64)&v160, (__int64)v160);
                v37 = sub_B12000(v33 + 72);
                if ( !v153 )
                  break;
                v38 = (v153 - 1) & (37 * v36);
                v39 = (unsigned int *)(v151 + 16LL * v38);
                v40 = *v39;
                if ( v36 == *v39 )
                  goto LABEL_56;
                v140 = 1;
                v65 = 0;
                while ( 1 )
                {
                  if ( v40 == -1 )
                  {
                    if ( !v65 )
                      v65 = v39;
                    ++v150;
                    v66 = v152 + 1;
                    if ( 4 * ((int)v152 + 1) < 3 * v153 )
                    {
                      if ( v153 - HIDWORD(v152) - v66 > v153 >> 3 )
                      {
LABEL_110:
                        LODWORD(v152) = v66;
                        if ( *v65 != -1 )
                          --HIDWORD(v152);
                        *v65 = v36;
                        v41 = (__int64 *)(v65 + 2);
                        *((_QWORD *)v65 + 1) = 0;
                        goto LABEL_57;
                      }
                      v142 = v37;
                      sub_352E320((__int64)&v150, v153);
                      if ( v153 )
                      {
                        v71 = 0;
                        v37 = v142;
                        v72 = 1;
                        v73 = (v153 - 1) & (37 * v36);
                        v66 = v152 + 1;
                        v65 = (unsigned int *)(v151 + 16LL * v73);
                        v74 = *v65;
                        if ( v36 != *v65 )
                        {
                          while ( v74 != -1 )
                          {
                            if ( !v71 && v74 == -2 )
                              v71 = v65;
                            v73 = (v153 - 1) & (v72 + v73);
                            v65 = (unsigned int *)(v151 + 16LL * v73);
                            v74 = *v65;
                            if ( v36 == *v65 )
                              goto LABEL_110;
                            ++v72;
                          }
                          if ( v71 )
                            v65 = v71;
                        }
                        goto LABEL_110;
                      }
LABEL_278:
                      LODWORD(v152) = v152 + 1;
                      BUG();
                    }
LABEL_114:
                    v141 = v37;
                    sub_352E320((__int64)&v150, 2 * v153);
                    if ( v153 )
                    {
                      v37 = v141;
                      v67 = (v153 - 1) & (37 * v36);
                      v66 = v152 + 1;
                      v65 = (unsigned int *)(v151 + 16LL * v67);
                      v68 = *v65;
                      if ( v36 != *v65 )
                      {
                        v69 = 1;
                        v70 = 0;
                        while ( v68 != -1 )
                        {
                          if ( !v70 && v68 == -2 )
                            v70 = v65;
                          v67 = (v153 - 1) & (v69 + v67);
                          v65 = (unsigned int *)(v151 + 16LL * v67);
                          v68 = *v65;
                          if ( v36 == *v65 )
                            goto LABEL_110;
                          ++v69;
                        }
                        if ( v70 )
                          v65 = v70;
                      }
                      goto LABEL_110;
                    }
                    goto LABEL_278;
                  }
                  if ( v40 != -2 || v65 )
                    v39 = v65;
                  v38 = (v153 - 1) & (v140 + v38);
                  v40 = *(_DWORD *)(v151 + 16LL * v38);
                  if ( v36 == v40 )
                    break;
                  ++v140;
                  v65 = v39;
                  v39 = (unsigned int *)(v151 + 16LL * v38);
                }
                v39 = (unsigned int *)(v151 + 16LL * v38);
LABEL_56:
                v41 = (__int64 *)(v39 + 2);
LABEL_57:
                *v41 = v37;
                if ( v14 )
                {
                  v42 = *(_QWORD *)(v14 + 24);
                  v160 = (unsigned __int8 *)v42;
                  if ( v42 )
                    sub_B96E90((__int64)&v160, v42, 1);
                  if ( v36 < (unsigned int)sub_B10CE0((__int64)&v160) )
                  {
                    if ( v160 )
                      sub_B91220((__int64)&v160, (__int64)v160);
                    goto LABEL_63;
                  }
                  if ( v160 )
                    sub_B91220((__int64)&v160, (__int64)v160);
                }
                else
                {
LABEL_63:
                  v14 = v33;
                }
                v34 = sub_B11F60(v33 + 80);
                do
                {
LABEL_66:
                  v33 = *(_QWORD *)(v33 + 8);
                  if ( v33 == v32 )
                    goto LABEL_67;
                }
                while ( *(_BYTE *)(v33 + 32) );
                if ( v33 == v32 )
                {
LABEL_67:
                  v15 = v34;
                  goto LABEL_68;
                }
              }
              ++v150;
              goto LABEL_114;
            }
          }
        }
LABEL_68:
        ;
      }
      v144 = *(_QWORD *)(v144 + 8);
    }
    while ( v144 != v135 );
  }
  if ( !(_DWORD)v152 )
    goto LABEL_91;
  v160 = 0;
  v161 = &v165;
  v162 = 16;
  v163 = 0;
  v164 = 1;
  v125 = 0;
  v124 = (_WORD *)(*(_QWORD *)(v132 + 8) - 560LL);
  v149 = *(_QWORD *)(v136 + 328);
  if ( v127 == v149 )
    goto LABEL_83;
  v120 = v14;
  v137 = v15;
  while ( 2 )
  {
    v43 = sub_2E311E0(v149);
    v44 = *(_QWORD *)(v149 + 56);
    v121 = (unsigned __int64 *)v43;
    if ( v44 == v149 + 48 )
      goto LABEL_82;
    v45 = v122;
    v46 = v123;
    while ( 2 )
    {
      if ( !v44 )
        BUG();
      v47 = v44;
      if ( (*(_BYTE *)v44 & 4) == 0 && (*(_BYTE *)(v44 + 44) & 8) != 0 )
      {
        do
          v47 = *(_QWORD *)(v47 + 8);
        while ( (*(_BYTE *)(v47 + 44) & 8) != 0 );
      }
      v139 = *(unsigned __int64 **)(v47 + 8);
      if ( (unsigned __int16)(*(_WORD *)(v44 + 68) - 14) <= 4u )
        goto LABEL_80;
      v75 = *(_DWORD *)(v44 + 44);
      if ( (v75 & 4) != 0 || (v75 & 8) == 0 )
        v76 = (*(_QWORD *)(*(_QWORD *)(v44 + 16) + 24LL) >> 9) & 1LL;
      else
        LOBYTE(v76) = sub_2E88A90(v44, 512, 1);
      if ( (_BYTE)v76 )
        goto LABEL_80;
      if ( !*(_WORD *)(v44 + 68) || *(_WORD *)(v44 + 68) == 68 )
        v146 = v121;
      else
        v146 = v139;
      v145 = (unsigned __int8 **)(v44 + 56);
      v77 = sub_B10CE0(v44 + 56);
      v79 = v153;
      v80 = v151;
      if ( v153 )
      {
        v81 = v153 - 1;
        v82 = v81 & (37 * v77);
        v83 = *(_DWORD *)(v151 + 16LL * v82);
        if ( v77 == v83 )
          goto LABEL_140;
        v100 = 1;
        while ( v83 != -1 )
        {
          v78 = (unsigned int)(v100 + 1);
          v82 = v81 & (v100 + v82);
          v83 = *(_DWORD *)(v151 + 16LL * v82);
          if ( v77 == v83 )
            goto LABEL_140;
          ++v100;
        }
      }
      if ( v130 )
      {
        v83 = sub_B10CE0(v130 + 48);
      }
      else
      {
        v101 = *(_QWORD *)(v120 + 24);
        v157 = (const __m128i **)v101;
        if ( v101 )
          sub_B96E90((__int64)&v157, v101, 1);
        v83 = sub_B10CE0((__int64)&v157);
        if ( v157 )
          sub_B91220((__int64)&v157, (__int64)v157);
      }
      v79 = v153;
      if ( !v153 )
      {
        ++v150;
        goto LABEL_178;
      }
      v80 = v151;
      v81 = v153 - 1;
LABEL_140:
      v84 = 1;
      v85 = 0;
      LODWORD(v86) = (37 * v83) & v81;
      v87 = (unsigned int *)(v80 + 16LL * (unsigned int)v86);
      v88 = *v87;
      if ( v83 == (_DWORD)v88 )
      {
LABEL_141:
        v131 = *((_QWORD *)v87 + 1);
        goto LABEL_142;
      }
      while ( (_DWORD)v88 != -1 )
      {
        if ( (_DWORD)v88 == -2 && !v85 )
          v85 = (__int64)v87;
        v78 = (unsigned int)(v84 + 1);
        v86 = (unsigned int)v81 & ((_DWORD)v86 + v84);
        v87 = (unsigned int *)(v80 + 16 * v86);
        v88 = *v87;
        if ( v83 == (_DWORD)v88 )
          goto LABEL_141;
        ++v84;
      }
      if ( v85 )
        v87 = (unsigned int *)v85;
      ++v150;
      v85 = (unsigned int)(v152 + 1);
      if ( 4 * (int)v85 >= 3 * v79 )
      {
LABEL_178:
        sub_352E320((__int64)&v150, 2 * v79);
        if ( !v153 )
          goto LABEL_279;
        v88 = (v153 - 1) & (37 * v83);
        v85 = (unsigned int)(v152 + 1);
        v87 = (unsigned int *)(v151 + 16 * v88);
        v81 = *v87;
        if ( v83 != (_DWORD)v81 )
        {
          v119 = 1;
          v102 = 0;
          while ( (_DWORD)v81 != -1 )
          {
            if ( !v102 && (_DWORD)v81 == -2 )
              v102 = v87;
            v78 = (unsigned int)(v119 + 1);
            v88 = (v153 - 1) & (v119 + (_DWORD)v88);
            v87 = (unsigned int *)(v151 + 16LL * (unsigned int)v88);
            v81 = *v87;
            if ( (_DWORD)v81 == v83 )
              goto LABEL_180;
            ++v119;
          }
LABEL_257:
          if ( v102 )
            v87 = v102;
        }
      }
      else
      {
        v88 = v79 >> 3;
        if ( v79 - ((_DWORD)v85 + HIDWORD(v152)) <= (unsigned int)v88 )
        {
          sub_352E320((__int64)&v150, v79);
          if ( !v153 )
          {
LABEL_279:
            LODWORD(v152) = v152 + 1;
            BUG();
          }
          v102 = 0;
          v88 = (v153 - 1) & (37 * v83);
          v103 = 1;
          v85 = (unsigned int)(v152 + 1);
          v87 = (unsigned int *)(v151 + 16 * v88);
          v81 = *v87;
          if ( v83 != (_DWORD)v81 )
          {
            while ( (_DWORD)v81 != -1 )
            {
              if ( (_DWORD)v81 == -2 && !v102 )
                v102 = v87;
              v78 = (unsigned int)(v103 + 1);
              v88 = (v153 - 1) & (v103 + (_DWORD)v88);
              v87 = (unsigned int *)(v151 + 16LL * (unsigned int)v88);
              v81 = *v87;
              if ( v83 == (_DWORD)v81 )
                goto LABEL_180;
              ++v103;
            }
            goto LABEL_257;
          }
        }
      }
LABEL_180:
      LODWORD(v152) = v85;
      if ( *v87 != -1 )
        --HIDWORD(v152);
      *v87 = v83;
      *((_QWORD *)v87 + 1) = 0;
      v131 = 0;
LABEL_142:
      if ( !v164 )
      {
LABEL_171:
        sub_C8CC70((__int64)&v160, v131, v85, v88, v81, v78);
        goto LABEL_147;
      }
      v89 = v161;
      v88 = HIDWORD(v162);
      v85 = (__int64)&v161[8 * HIDWORD(v162)];
      if ( v161 == (char *)v85 )
      {
LABEL_184:
        if ( HIDWORD(v162) < (unsigned int)v162 )
        {
          ++HIDWORD(v162);
          *(_QWORD *)v85 = v131;
          ++v160;
          goto LABEL_147;
        }
        goto LABEL_171;
      }
      while ( v131 != *(_QWORD *)v89 )
      {
        v89 += 8;
        if ( (char *)v85 == v89 )
          goto LABEL_184;
      }
LABEL_147:
      v157 = (const __m128i **)v159;
      v158 = 0x400000000LL;
      v90 = *(__m128i **)(v44 + 32);
      v91 = (unsigned __int64 *)v90 + 5 * (*(_DWORD *)(v44 + 40) & 0xFFFFFF);
      if ( v90 == (__m128i *)v91 )
        goto LABEL_183;
      while ( !sub_2DADC00(v90) )
      {
        v90 = (__m128i *)((char *)v90 + 40);
        if ( v91 == (unsigned __int64 *)v90 )
          goto LABEL_183;
      }
      if ( v91 == (unsigned __int64 *)v90 )
        goto LABEL_183;
      v134 = v46;
      v94 = 0;
      v95 = v90;
      do
      {
        if ( v95->m128i_i32[2] )
        {
          if ( v94 + 1 > (unsigned __int64)HIDWORD(v158) )
          {
            sub_C8D5F0((__int64)&v157, v159, v94 + 1, 8u, v92, v93);
            v94 = (unsigned int)v158;
          }
          v157[v94] = v95;
          v94 = (unsigned int)(v158 + 1);
          LODWORD(v158) = v158 + 1;
        }
        v96 = (__m128i *)&v95[2].m128i_u64[1];
        if ( &v95[2].m128i_u64[1] == v91 )
          break;
        while ( 1 )
        {
          v95 = v96;
          if ( sub_2DADC00(v96) )
            break;
          v96 = (__m128i *)((char *)v96 + 40);
          if ( v91 == (unsigned __int64 *)v96 )
            goto LABEL_159;
        }
      }
      while ( v91 != (unsigned __int64 *)v96 );
LABEL_159:
      v97 = v94;
      v46 = v134;
      v98 = &v157[(unsigned int)v94];
      if ( v157 != v98 )
      {
        v133 = v98;
        v99 = v157;
        do
        {
          v45 = *v99;
          v46 = 1;
          ++v99;
          sub_2E90D80(v149, v146, v145, v124, 0, v131, v45, 1, v137);
        }
        while ( v133 != v99 );
        v97 = v158;
      }
      if ( !v97 )
      {
LABEL_183:
        v156 = v125;
        v154.m128i_i64[0] = 1;
        v155 = 0;
        sub_2E90D80(v149, v146, v145, v124, 0, v131, &v154, 1, v137);
        ++v125;
      }
      if ( v157 != (const __m128i **)v159 )
        _libc_free((unsigned __int64)v157);
LABEL_80:
      v44 = (__int64)v139;
      if ( (unsigned __int64 *)(v149 + 48) != v139 )
        continue;
      break;
    }
    v122 = v45;
    v123 = v46;
LABEL_82:
    v149 = *(_QWORD *)(v149 + 8);
    if ( v127 != v149 )
      continue;
    break;
  }
LABEL_83:
  v48 = sub_BA8DC0(v128, (__int64)"llvm.mir.debugify", 17);
  v49 = sub_BCB2D0(v129);
  v50 = (unsigned int)(v126 - 1);
  if ( v48 )
  {
    v51 = (unsigned int)(v126 - 1);
    v52 = sub_ACD640(v49, v51, 0);
    v157 = (const __m128i **)sub_B98A20(v52, v51);
    v53 = sub_B9C770(v129, (__int64 *)&v157, (__int64 *)1, 0, 1);
    sub_B970B0(v48, 0, v53);
    v54 = sub_B91A10(v48, 1u);
    v55 = *(_BYTE *)(v54 - 16);
    if ( (v55 & 2) != 0 )
      v56 = *(_QWORD *)(v54 - 32);
    else
      v56 = v54 - 8LL * ((v55 >> 2) & 0xF) - 16;
    v57 = *(_QWORD *)(*(_QWORD *)v56 + 136LL);
    v58 = *(_QWORD **)(v57 + 24);
    if ( *(_DWORD *)(v57 + 32) > 0x40u )
      v58 = (_QWORD *)*v58;
    v59 = (unsigned int)(HIDWORD(v162) - v163 + (_DWORD)v58);
    v60 = sub_ACD640(v49, v59, 0);
    v157 = (const __m128i **)sub_B98A20(v60, v59);
    v61 = sub_B9C770(v129, (__int64 *)&v157, (__int64 *)1, 0, 1);
    sub_B970B0(v48, 1u, v61);
  }
  else
  {
    v104 = sub_BA8E40(v128, "llvm.mir.debugify", 0x11u);
    v105 = sub_ACD640(v49, v50, 0);
    v157 = (const __m128i **)sub_B98A20(v105, v50);
    v106 = sub_B9C770(v129, (__int64 *)&v157, (__int64 *)1, 0, 1);
    sub_B979A0(v104, v106);
    v107 = (unsigned int)(HIDWORD(v162) - v163);
    v108 = sub_ACD640(v49, v107, 0);
    v157 = (const __m128i **)sub_B98A20(v108, v107);
    v109 = sub_B9C770(v129, (__int64 *)&v157, (__int64 *)1, 0, 1);
    sub_B979A0(v104, v109);
  }
  if ( !v164 )
    _libc_free((unsigned __int64)v161);
LABEL_91:
  v62 = v151;
  v63 = 16LL * v153;
LABEL_92:
  sub_C7D6A0(v62, v63, 8);
  return 1;
}
