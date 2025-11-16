// Function: sub_3957A80
// Address: 0x3957a80
//
void __fastcall sub_3957A80(__int64 a1, unsigned int a2, int a3, char a4)
{
  int v5; // ebx
  int *v6; // rax
  int v7; // edx
  int v8; // eax
  __int64 v9; // r12
  __int64 v10; // rbx
  __int64 v11; // r12
  __int64 v12; // r9
  unsigned int v13; // edx
  __int64 *v14; // rcx
  __int64 v15; // r8
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rax
  __int64 v19; // rcx
  __int64 v20; // rdi
  unsigned int v21; // r9d
  __int64 *v22; // rsi
  __int64 v23; // r8
  unsigned int v24; // esi
  __int64 *v25; // rdi
  int v26; // ecx
  unsigned __int64 v27; // rsi
  __int64 v28; // rax
  int v29; // esi
  int v30; // r11d
  unsigned int v31; // r8d
  __int64 *v32; // r9
  __int64 v33; // rax
  __int64 v34; // r13
  int v35; // r14d
  unsigned __int64 v36; // rax
  int v37; // esi
  int v38; // edx
  __int64 v39; // rdx
  __int64 v40; // rdx
  __int64 v41; // rbx
  __int64 v42; // rax
  __int64 v43; // rcx
  __int64 *v44; // rdx
  __int64 v45; // rdi
  unsigned int v46; // esi
  __int64 *v47; // rax
  __int64 v48; // rdx
  _QWORD *v49; // r12
  __int64 v50; // rdx
  _DWORD *v51; // rax
  int v52; // edx
  _DWORD *v53; // rax
  int v54; // edx
  int v55; // edx
  int v56; // r11d
  int v57; // ecx
  unsigned int v58; // edx
  __int64 v59; // rax
  unsigned __int64 v60; // rbx
  char v61; // r9
  unsigned __int64 v62; // rax
  unsigned int v63; // edx
  int v64; // esi
  __int64 v65; // rax
  unsigned int v66; // ecx
  _QWORD *v67; // rdx
  __int64 v68; // r8
  int v69; // r13d
  unsigned int v70; // esi
  unsigned __int64 v71; // rcx
  __int64 v72; // r9
  unsigned int v73; // r12d
  unsigned __int64 v74; // rdi
  int v75; // esi
  __int64 v76; // r13
  unsigned int v77; // edx
  __int64 *v78; // rax
  __int64 v79; // rcx
  unsigned __int64 v80; // rdi
  int v81; // eax
  __int64 v82; // rdi
  unsigned int v83; // r13d
  int v84; // ebx
  unsigned __int64 v85; // r12
  int v86; // r8d
  int v87; // r9d
  __int64 v88; // r14
  __int64 v89; // rax
  int v90; // edx
  int v91; // r10d
  int v92; // ecx
  __int64 v93; // rbx
  _QWORD *v94; // rax
  unsigned int v95; // edi
  _QWORD *v96; // rax
  __int64 v97; // r8
  unsigned int v98; // eax
  __int64 v99; // rcx
  __int64 v100; // rdx
  __int64 v101; // rax
  __int64 v102; // rsi
  unsigned int v103; // r9d
  __int64 *v104; // rdx
  __int64 v105; // r8
  int v106; // esi
  _QWORD *v107; // rdx
  int v108; // edi
  int v109; // r11d
  _QWORD *v110; // rdi
  int v111; // ecx
  int v112; // r10d
  int v113; // ecx
  __int64 v114; // rsi
  int v115; // r10d
  _QWORD *v116; // rsi
  _QWORD *v117; // rcx
  __int64 v118; // rdx
  _QWORD *v119; // rax
  __int64 v120; // rsi
  __int64 v121; // r9
  unsigned int v122; // r8d
  __int64 *v123; // rdi
  __int64 v124; // r11
  int v125; // edi
  int v126; // ebx
  int v127; // esi
  int i; // edx
  int v129; // r11d
  int v130; // edi
  __int64 *v131; // r11
  int v132; // ebx
  int v133; // r14d
  int v134; // r10d
  __int64 *v135; // r9
  int v136; // ecx
  _QWORD *v137; // rbx
  _QWORD *v138; // r11
  __int64 v140; // [rsp+20h] [rbp-F0h]
  __int64 v142; // [rsp+28h] [rbp-E8h]
  __int64 v143; // [rsp+30h] [rbp-E0h] BYREF
  __int64 v144; // [rsp+38h] [rbp-D8h] BYREF
  __int64 v145; // [rsp+40h] [rbp-D0h] BYREF
  unsigned __int64 v146; // [rsp+48h] [rbp-C8h] BYREF
  __int64 v147; // [rsp+50h] [rbp-C0h] BYREF
  _QWORD *v148; // [rsp+58h] [rbp-B8h]
  __int64 v149; // [rsp+60h] [rbp-B0h]
  unsigned int v150; // [rsp+68h] [rbp-A8h]
  __int64 v151; // [rsp+70h] [rbp-A0h] BYREF
  unsigned __int64 v152; // [rsp+78h] [rbp-98h]
  __int64 v153; // [rsp+80h] [rbp-90h]
  unsigned int v154; // [rsp+88h] [rbp-88h]
  __int64 *v155; // [rsp+90h] [rbp-80h] BYREF
  __int64 v156; // [rsp+98h] [rbp-78h]
  _BYTE v157[112]; // [rsp+A0h] [rbp-70h] BYREF

  if ( !a3 )
    a3 = *(_DWORD *)(a1 + 24);
  v5 = sub_1BF96D0(*(_QWORD *)a1, a2, a3);
  v6 = (int *)sub_16D40F0((__int64)qword_4FBB390);
  if ( v6 )
    v7 = *v6;
  else
    v7 = qword_4FBB390[2];
  v8 = 7;
  if ( v7 >= 0 )
  {
    v51 = sub_16D40F0((__int64)qword_4FBB390);
    v52 = v51 ? *v51 : LODWORD(qword_4FBB390[2]);
    v8 = 7;
    if ( v52 <= 10 )
    {
      v53 = sub_16D40F0((__int64)qword_4FBB390);
      v54 = v53 ? *v53 : LODWORD(qword_4FBB390[2]);
      v8 = 7;
      if ( (unsigned int)(v54 + 4) <= 0x12 )
      {
        v55 = v54 - 5;
        v5 += v55 * v5 / 10;
        v8 = 7 * v55 / 10 + 7;
      }
    }
  }
  v9 = *(_QWORD *)a1;
  *(_DWORD *)(a1 + 40) = v5;
  *(_DWORD *)(a1 + 44) = v8;
  v10 = *(_QWORD *)(v9 + 80);
  v11 = v9 + 72;
  if ( v10 != v11 )
  {
    while ( 1 )
    {
      v17 = *(_QWORD *)(a1 + 16);
      v18 = v10 - 24;
      if ( !v10 )
        v18 = 0;
      v19 = *(unsigned int *)(v17 + 48);
      if ( (_DWORD)v19 )
      {
        v20 = *(_QWORD *)(v17 + 32);
        v21 = (v19 - 1) & (((unsigned int)v18 >> 4) ^ ((unsigned int)v18 >> 9));
        v22 = (__int64 *)(v20 + 16LL * v21);
        v23 = *v22;
        if ( v18 != *v22 )
        {
          v29 = 1;
          while ( v23 != -8 )
          {
            v30 = v29 + 1;
            v21 = (v19 - 1) & (v29 + v21);
            v22 = (__int64 *)(v20 + 16LL * v21);
            v23 = *v22;
            if ( v18 == *v22 )
              goto LABEL_16;
            v29 = v30;
          }
          goto LABEL_11;
        }
LABEL_16:
        if ( v22 != (__int64 *)(v20 + 16 * v19) && v22[1] )
          break;
      }
LABEL_11:
      v10 = *(_QWORD *)(v10 + 8);
      if ( v11 == v10 )
        goto LABEL_25;
    }
    v24 = *(_DWORD *)(a1 + 136);
    v151 = v18;
    if ( v24 )
    {
      v12 = *(_QWORD *)(a1 + 120);
      v13 = (v24 - 1) & (((unsigned int)v18 >> 4) ^ ((unsigned int)v18 >> 9));
      v14 = (__int64 *)(v12 + 16LL * v13);
      v15 = *v14;
      if ( v18 == *v14 )
      {
        v16 = v14[1];
LABEL_10:
        *(_QWORD *)(v16 + 16) = *(_QWORD *)(a1 + 40);
        goto LABEL_11;
      }
      v56 = 1;
      v25 = 0;
      while ( v15 != -8 )
      {
        if ( v25 || v15 != -16 )
          v14 = v25;
        v130 = v56 + 1;
        v13 = (v24 - 1) & (v56 + v13);
        v131 = (__int64 *)(v12 + 16LL * v13);
        v15 = *v131;
        if ( v18 == *v131 )
        {
          v16 = v131[1];
          goto LABEL_10;
        }
        v56 = v130;
        v25 = v14;
        v14 = (__int64 *)(v12 + 16LL * v13);
      }
      if ( !v25 )
        v25 = v14;
      v57 = *(_DWORD *)(a1 + 128);
      ++*(_QWORD *)(a1 + 112);
      v26 = v57 + 1;
      if ( 4 * v26 < 3 * v24 )
      {
        if ( v24 - *(_DWORD *)(a1 + 132) - v26 > v24 >> 3 )
          goto LABEL_22;
LABEL_21:
        sub_1C29D90(a1 + 112, v24);
        sub_39538E0(a1 + 112, &v151, &v155);
        v25 = v155;
        v18 = v151;
        v26 = *(_DWORD *)(a1 + 128) + 1;
LABEL_22:
        *(_DWORD *)(a1 + 128) = v26;
        if ( *v25 != -8 )
          --*(_DWORD *)(a1 + 132);
        *v25 = v18;
        v16 = 0;
        v25[1] = 0;
        goto LABEL_10;
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 112);
    }
    v24 *= 2;
    goto LABEL_21;
  }
LABEL_25:
  v27 = a2 - 90;
  if ( (unsigned int)v27 > 0x1E )
    return;
  v28 = 1073742849;
  if ( !_bittest64(&v28, v27) || !a4 || !(unsigned __int8)sub_3953540(a1) )
    return;
  v151 = 0;
  v155 = (__int64 *)v157;
  v156 = 0x800000000LL;
  v33 = *(_QWORD *)a1;
  v152 = 0;
  v153 = 0;
  v33 += 72;
  v154 = 0;
  v34 = *(_QWORD *)(v33 + 8);
  v147 = 0;
  v148 = 0;
  v149 = 0;
  v150 = 0;
  v140 = v33;
  if ( v33 == v34 )
  {
    v59 = 0;
    v74 = 0;
    v58 = 8;
    if ( *(_BYTE *)(a1 + 48) )
    {
LABEL_77:
      v34 -= 24;
      goto LABEL_78;
    }
    goto LABEL_99;
  }
  v142 = v34;
  do
  {
    v40 = *(_QWORD *)(a1 + 16);
    v41 = v142 - 24;
    if ( !v142 )
      v41 = 0;
    v42 = *(unsigned int *)(v40 + 48);
    if ( !(_DWORD)v42 )
      goto LABEL_45;
    v43 = *(_QWORD *)(v40 + 32);
    LODWORD(v32) = v42 - 1;
    v31 = (v42 - 1) & (((unsigned int)v41 >> 9) ^ ((unsigned int)v41 >> 4));
    v44 = (__int64 *)(v43 + 16LL * v31);
    v45 = *v44;
    if ( v41 != *v44 )
    {
      v90 = 1;
      while ( v45 != -8 )
      {
        v91 = v90 + 1;
        v31 = (unsigned int)v32 & (v90 + v31);
        v44 = (__int64 *)(v43 + 16LL * v31);
        v45 = *v44;
        if ( v41 == *v44 )
          goto LABEL_50;
        v90 = v91;
      }
      goto LABEL_45;
    }
LABEL_50:
    if ( v44 == (__int64 *)(v43 + 16 * v42) || !v44[1] )
      goto LABEL_45;
    v31 = v150;
    v145 = v41;
    if ( v150 )
    {
      v46 = (v150 - 1) & (((unsigned int)v41 >> 9) ^ ((unsigned int)v41 >> 4));
      v47 = &v148[2 * v46];
      v48 = *v47;
      if ( v41 == *v47 )
        goto LABEL_54;
      v112 = 1;
      v32 = 0;
      while ( v48 != -8 )
      {
        if ( v32 || v48 != -16 )
          v47 = v32;
        LODWORD(v32) = v112 + 1;
        v46 = (v150 - 1) & (v112 + v46);
        v48 = v148[2 * v46];
        if ( v41 == v48 )
        {
          v47 = &v148[2 * v46];
          goto LABEL_54;
        }
        ++v112;
        v32 = v47;
        v47 = &v148[2 * v46];
      }
      if ( v32 )
        v47 = v32;
      ++v147;
      v113 = v149 + 1;
      if ( 4 * ((int)v149 + 1) < 3 * v150 )
      {
        v114 = v41;
        if ( v150 - HIDWORD(v149) - v113 > v150 >> 3 )
          goto LABEL_154;
        v127 = v150;
        goto LABEL_185;
      }
    }
    else
    {
      ++v147;
    }
    v127 = 2 * v150;
LABEL_185:
    sub_137BFC0((__int64)&v147, v127);
    sub_19E7690((__int64)&v147, &v145, &v146);
    v47 = (__int64 *)v146;
    v114 = v145;
    v113 = v149 + 1;
LABEL_154:
    LODWORD(v149) = v113;
    if ( *v47 != -8 )
      --HIDWORD(v149);
    *v47 = v114;
    *((_DWORD *)v47 + 2) = 0;
LABEL_54:
    *((_DWORD *)v47 + 2) = -1;
    v49 = (_QWORD *)(*(_QWORD *)(v41 + 40) & 0xFFFFFFFFFFFFFFF8LL);
    if ( (_QWORD *)(v41 + 40) != v49 )
    {
      while ( 1 )
      {
        v50 = (__int64)(v49 - 3);
        if ( !v49 )
          v50 = 0;
        sub_3953170((__int64)&v146, a1, v50);
        if ( BYTE4(v146) )
        {
          if ( (unsigned int)(v146 - 24) <= 0xE8 && (v146 & 7) == 0 )
            break;
        }
        v49 = (_QWORD *)(*v49 & 0xFFFFFFFFFFFFFFF8LL);
        if ( (_QWORD *)(v41 + 40) == v49 )
          goto LABEL_45;
      }
      v145 = v41;
      v35 = v146;
      LODWORD(v32) = sub_19E7690((__int64)&v151, &v145, &v146);
      v36 = v146;
      if ( !(_BYTE)v32 )
      {
        v37 = v154;
        ++v151;
        v38 = v153 + 1;
        if ( 4 * ((int)v153 + 1) >= 3 * v154 )
        {
          v37 = 2 * v154;
        }
        else
        {
          LODWORD(v32) = v154 >> 3;
          if ( v154 - HIDWORD(v153) - v38 > v154 >> 3 )
            goto LABEL_41;
        }
        sub_137BFC0((__int64)&v151, v37);
        sub_19E7690((__int64)&v151, &v145, &v146);
        v36 = v146;
        v38 = v153 + 1;
LABEL_41:
        LODWORD(v153) = v38;
        if ( *(_QWORD *)v36 != -8 )
          --HIDWORD(v153);
        v39 = v145;
        *(_DWORD *)(v36 + 8) = 0;
        *(_QWORD *)v36 = v39;
      }
      *(_DWORD *)(v36 + 8) = v35;
      *(_BYTE *)(a1 + 48) = 1;
    }
LABEL_45:
    v142 = *(_QWORD *)(v142 + 8);
  }
  while ( v140 != v142 );
  if ( *(_BYTE *)(a1 + 48) )
  {
    v58 = HIDWORD(v156);
    v34 = *(_QWORD *)(*(_QWORD *)a1 + 80LL);
    v59 = (unsigned int)v156;
    if ( v34 )
      goto LABEL_77;
LABEL_78:
    v143 = v34;
    if ( (unsigned int)v59 >= v58 )
    {
      sub_16CD150((__int64)&v155, v157, 0, 8, v31, (int)v32);
      v59 = (unsigned int)v156;
    }
    v155[v59] = v143;
    LODWORD(v156) = v156 + 1;
    v145 = v143;
    v60 = v152 + 16LL * v154;
    v61 = sub_19E7690((__int64)&v151, &v145, &v146);
    v62 = v146;
    if ( !v61 )
      v62 = v152 + 16LL * v154;
    if ( v62 == v60 )
    {
      v132 = *(_DWORD *)(a1 + 40);
      *((_DWORD *)sub_39578B0((__int64)&v151, &v143) + 2) = v132;
    }
    v63 = v156;
LABEL_85:
    if ( v63 )
    {
      while ( 2 )
      {
        v64 = v150;
        v65 = v155[v63 - 1];
        LODWORD(v156) = v63 - 1;
        v144 = v65;
        if ( v150 )
        {
          v66 = (v150 - 1) & (((unsigned int)v65 >> 9) ^ ((unsigned int)v65 >> 4));
          v67 = &v148[2 * v66];
          v68 = *v67;
          if ( v65 == *v67 )
          {
            v69 = *((_DWORD *)v67 + 2);
            goto LABEL_89;
          }
          v109 = 1;
          v110 = 0;
          while ( v68 != -8 )
          {
            if ( v68 != -16 || v110 )
              v67 = v110;
            v66 = (v150 - 1) & (v109 + v66);
            v137 = &v148[2 * v66];
            v68 = *v137;
            if ( v65 == *v137 )
            {
              v69 = *((_DWORD *)v137 + 2);
              goto LABEL_89;
            }
            ++v109;
            v110 = v67;
            v67 = &v148[2 * v66];
          }
          if ( !v110 )
            v110 = v67;
          ++v147;
          v111 = v149 + 1;
          if ( 4 * ((int)v149 + 1) < 3 * v150 )
          {
            if ( v150 - HIDWORD(v149) - v111 <= v150 >> 3 )
              goto LABEL_147;
LABEL_142:
            LODWORD(v149) = v111;
            if ( *v110 != -8 )
              --HIDWORD(v149);
            *v110 = v65;
            v69 = 0;
            v65 = v144;
            *((_DWORD *)v110 + 2) = 0;
LABEL_89:
            if ( v154 )
            {
              v70 = (v154 - 1) & (((unsigned int)v65 >> 9) ^ ((unsigned int)v65 >> 4));
              v71 = v152 + 16LL * v70;
              v72 = *(_QWORD *)v71;
              if ( *(_QWORD *)v71 == v65 )
              {
LABEL_91:
                if ( v71 != v152 + 16LL * v154 )
                {
                  v73 = *(_DWORD *)(v71 + 8);
                  goto LABEL_93;
                }
              }
              else
              {
                v92 = 1;
                while ( v72 != -8 )
                {
                  v115 = v92 + 1;
                  v70 = (v154 - 1) & (v92 + v70);
                  v71 = v152 + 16LL * v70;
                  v72 = *(_QWORD *)v71;
                  if ( *(_QWORD *)v71 == v65 )
                    goto LABEL_91;
                  v92 = v115;
                }
              }
            }
            v93 = *(_QWORD *)(v65 + 8);
            if ( v93 )
            {
              while ( 1 )
              {
                v94 = sub_1648700(v93);
                if ( (unsigned __int8)(*((_BYTE *)v94 + 16) - 25) <= 9u )
                  break;
                v93 = *(_QWORD *)(v93 + 8);
                if ( !v93 )
                  goto LABEL_135;
              }
              v73 = -1;
LABEL_123:
              v99 = v94[5];
              v100 = *(_QWORD *)(a1 + 16);
              v145 = v99;
              v101 = *(unsigned int *)(v100 + 48);
              if ( !(_DWORD)v101 )
                goto LABEL_121;
              v102 = *(_QWORD *)(v100 + 32);
              v103 = (v101 - 1) & (((unsigned int)v99 >> 9) ^ ((unsigned int)v99 >> 4));
              v104 = (__int64 *)(v102 + 16LL * v103);
              v105 = *v104;
              if ( v99 != *v104 )
              {
                for ( i = 1; ; i = v129 )
                {
                  if ( v105 == -8 )
                    goto LABEL_121;
                  v129 = i + 1;
                  v103 = (v101 - 1) & (i + v103);
                  v104 = (__int64 *)(v102 + 16LL * v103);
                  v105 = *v104;
                  if ( v99 == *v104 )
                    break;
                }
              }
              if ( v104 != (__int64 *)(v102 + 16 * v101) && v104[1] )
              {
                v106 = v150;
                if ( !v150 )
                {
                  ++v147;
                  goto LABEL_129;
                }
                v95 = (v150 - 1) & (((unsigned int)v99 >> 9) ^ ((unsigned int)v99 >> 4));
                v96 = &v148[2 * v95];
                v97 = *v96;
                if ( v99 == *v96 )
                {
                  v98 = *((_DWORD *)v96 + 2);
                  if ( v73 > v98 )
                    v73 = v98;
                  goto LABEL_121;
                }
                v133 = 1;
                v107 = 0;
                while ( v97 != -8 )
                {
                  if ( v97 != -16 || v107 )
                    v96 = v107;
                  v95 = (v150 - 1) & (v133 + v95);
                  v138 = &v148[2 * v95];
                  v97 = *v138;
                  if ( v99 == *v138 )
                  {
                    if ( v73 > *((_DWORD *)v138 + 2) )
                      v73 = *((_DWORD *)v138 + 2);
                    goto LABEL_121;
                  }
                  ++v133;
                  v107 = v96;
                  v96 = &v148[2 * v95];
                }
                if ( !v107 )
                  v107 = v96;
                ++v147;
                v108 = v149 + 1;
                if ( 4 * ((int)v149 + 1) < 3 * v150 )
                {
                  if ( v150 - HIDWORD(v149) - v108 <= v150 >> 3 )
                  {
LABEL_130:
                    sub_137BFC0((__int64)&v147, v106);
                    sub_19E7690((__int64)&v147, &v145, &v146);
                    v107 = (_QWORD *)v146;
                    v99 = v145;
                    v108 = v149 + 1;
                  }
                  LODWORD(v149) = v108;
                  if ( *v107 != -8 )
                    --HIDWORD(v149);
                  *v107 = v99;
                  v73 = 0;
                  *((_DWORD *)v107 + 2) = 0;
                  goto LABEL_121;
                }
LABEL_129:
                v106 = 2 * v150;
                goto LABEL_130;
              }
LABEL_121:
              while ( 1 )
              {
                v93 = *(_QWORD *)(v93 + 8);
                if ( !v93 )
                  break;
                v94 = sub_1648700(v93);
                if ( (unsigned __int8)(*((_BYTE *)v94 + 16) - 25) <= 9u )
                  goto LABEL_123;
              }
            }
            else
            {
LABEL_135:
              v73 = -1;
            }
LABEL_93:
            if ( v73 == v69 )
            {
LABEL_94:
              v63 = v156;
              if ( !(_DWORD)v156 )
                goto LABEL_95;
              continue;
            }
            v75 = v150;
            if ( v150 )
            {
              v76 = v144;
              v77 = (v150 - 1) & (((unsigned int)v144 >> 9) ^ ((unsigned int)v144 >> 4));
              v78 = &v148[2 * v77];
              v79 = *v78;
              if ( *v78 == v144 )
                goto LABEL_102;
              v134 = 1;
              v135 = 0;
              while ( v79 != -8 )
              {
                if ( v79 == -16 && !v135 )
                  v135 = v78;
                v77 = (v150 - 1) & (v134 + v77);
                v78 = &v148[2 * v77];
                v79 = *v78;
                if ( v144 == *v78 )
                  goto LABEL_102;
                ++v134;
              }
              if ( v135 )
                v78 = v135;
              ++v147;
              v136 = v149 + 1;
              if ( 4 * ((int)v149 + 1) < 3 * v150 )
              {
                if ( v150 - HIDWORD(v149) - v136 > v150 >> 3 )
                {
LABEL_212:
                  LODWORD(v149) = v136;
                  if ( *v78 != -8 )
                    --HIDWORD(v149);
                  *v78 = v76;
                  v76 = v144;
                  *((_DWORD *)v78 + 2) = 0;
LABEL_102:
                  *((_DWORD *)v78 + 2) = v73;
                  v80 = sub_157EBA0(v76);
                  if ( v80 )
                  {
                    v81 = sub_15F4D60(v80);
                    v82 = v76;
                    v83 = 0;
                    v84 = v81;
                    v85 = sub_157EBA0(v82);
                    if ( v84 )
                    {
                      do
                      {
                        v88 = sub_15F4DF0(v85, v83);
                        v89 = (unsigned int)v156;
                        if ( (unsigned int)v156 >= HIDWORD(v156) )
                        {
                          sub_16CD150((__int64)&v155, v157, 0, 8, v86, v87);
                          v89 = (unsigned int)v156;
                        }
                        ++v83;
                        v155[v89] = v88;
                        v63 = v156 + 1;
                        LODWORD(v156) = v156 + 1;
                      }
                      while ( v83 != v84 );
                      goto LABEL_85;
                    }
                  }
                  goto LABEL_94;
                }
LABEL_217:
                sub_137BFC0((__int64)&v147, v75);
                sub_19E7690((__int64)&v147, &v144, &v146);
                v78 = (__int64 *)v146;
                v76 = v144;
                v136 = v149 + 1;
                goto LABEL_212;
              }
            }
            else
            {
              ++v147;
            }
            v75 = 2 * v150;
            goto LABEL_217;
          }
        }
        else
        {
          ++v147;
        }
        break;
      }
      v64 = 2 * v150;
LABEL_147:
      sub_137BFC0((__int64)&v147, v64);
      sub_19E7690((__int64)&v147, &v144, &v146);
      v110 = (_QWORD *)v146;
      v65 = v144;
      v111 = v149 + 1;
      goto LABEL_142;
    }
LABEL_95:
    if ( (_DWORD)v149 )
    {
      v116 = v148;
      v117 = &v148[2 * v150];
      if ( v148 != v117 )
      {
        while ( 1 )
        {
          v118 = *v116;
          v119 = v116;
          if ( *v116 != -16 && v118 != -8 )
            break;
          v116 += 2;
          if ( v117 == v116 )
            goto LABEL_96;
        }
        while ( v119 != v117 )
        {
          v120 = *(unsigned int *)(a1 + 136);
          if ( (_DWORD)v120 )
          {
            v121 = *(_QWORD *)(a1 + 120);
            v122 = (v120 - 1) & (((unsigned int)v118 >> 9) ^ ((unsigned int)v118 >> 4));
            v123 = (__int64 *)(v121 + 16LL * v122);
            v124 = *v123;
            if ( v118 == *v123 )
            {
LABEL_172:
              if ( v123 != (__int64 *)(v121 + 16 * v120) )
                *(_DWORD *)(v123[1] + 16) = *((_DWORD *)v119 + 2);
            }
            else
            {
              v125 = 1;
              while ( v124 != -8 )
              {
                v126 = v125 + 1;
                v122 = (v120 - 1) & (v125 + v122);
                v123 = (__int64 *)(v121 + 16LL * v122);
                v124 = *v123;
                if ( v118 == *v123 )
                  goto LABEL_172;
                v125 = v126;
              }
            }
          }
          v119 += 2;
          if ( v119 == v117 )
            break;
          while ( 1 )
          {
            v118 = *v119;
            if ( *v119 != -8 && v118 != -16 )
              break;
            v119 += 2;
            if ( v117 == v119 )
              goto LABEL_96;
          }
        }
      }
    }
  }
LABEL_96:
  if ( v155 != (__int64 *)v157 )
    _libc_free((unsigned __int64)v155);
  v74 = v152;
LABEL_99:
  j___libc_free_0(v74);
  j___libc_free_0((unsigned __int64)v148);
}
