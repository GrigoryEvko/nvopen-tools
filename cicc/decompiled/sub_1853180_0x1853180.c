// Function: sub_1853180
// Address: 0x1853180
//
int __fastcall sub_1853180(_QWORD *a1, __int64 a2, int a3, __int64 a4, __int64 a5, __int64 a6, __int64 a7, __int64 a8)
{
  _QWORD *v8; // r14
  _QWORD *v9; // r12
  int v11; // ecx
  __int64 v12; // r8
  unsigned int v13; // edx
  __int64 v14; // rdi
  int v15; // edx
  _QWORD *v16; // rax
  __int64 v17; // rbx
  __int64 i; // rcx
  __int64 v20; // rax
  __int64 v21; // r15
  int v22; // edx
  __int64 v23; // rsi
  __int64 v24; // rcx
  __int64 v25; // rbx
  __int64 v26; // r12
  int v27; // eax
  char v28; // al
  float v29; // xmm0_4
  unsigned int v30; // r11d
  unsigned int v31; // esi
  __int64 v32; // rcx
  __int64 *v33; // r10
  int v34; // r8d
  unsigned int v35; // eax
  __int64 *v36; // r9
  __int64 v37; // rdx
  _QWORD *v38; // r10
  float v39; // xmm0_4
  unsigned __int64 v40; // r13
  __int64 v41; // rax
  __int64 v42; // r12
  __int64 v43; // rax
  __int64 v44; // rax
  int v45; // r9d
  char *v46; // r13
  size_t v47; // rdx
  __int64 v48; // rcx
  _QWORD *v49; // r10
  char *v50; // r12
  char v51; // r8
  __int64 v52; // rcx
  int v53; // esi
  int v54; // edi
  __int64 v55; // rcx
  int v56; // edi
  __int64 v57; // rcx
  _QWORD *v58; // r9
  int v59; // esi
  __int64 v60; // rcx
  int v61; // edi
  char v62; // r8
  __int64 v63; // r12
  __int64 v64; // r12
  __int64 v65; // r12
  int v66; // edi
  __int64 v67; // r10
  __int64 v68; // rax
  unsigned __int64 v69; // r12
  __int64 v70; // r9
  unsigned __int64 v71; // rdi
  _QWORD *v72; // r8
  unsigned __int64 v73; // r13
  _QWORD *v74; // rax
  _QWORD *v75; // rsi
  __int64 v76; // rax
  unsigned __int64 v77; // r12
  unsigned __int64 v78; // r8
  _QWORD *v79; // r13
  _QWORD *v80; // rdi
  _QWORD *v81; // rax
  _QWORD *v82; // rsi
  __int64 *v83; // rbx
  _QWORD *v84; // r12
  __int64 *v85; // r13
  __int64 v86; // rax
  __int64 *v87; // rbx
  _QWORD *v88; // r12
  __int64 *v89; // r13
  __int64 v90; // rax
  _QWORD *v91; // rax
  int v92; // ecx
  unsigned __int64 v93; // rax
  unsigned __int64 v94; // rax
  unsigned __int64 v95; // r8
  __int64 v96; // rax
  int v97; // r8d
  __int64 v98; // rcx
  unsigned __int64 v99; // rdi
  unsigned __int64 v100; // rdx
  __int64 v101; // rsi
  _QWORD *v102; // rax
  int v103; // eax
  int v104; // r13d
  __int64 v105; // r10
  unsigned int v106; // edi
  __int64 v107; // rsi
  int v108; // r8d
  __int64 *v109; // rax
  int v110; // r10d
  int v111; // r10d
  __int64 v112; // r8
  __int64 *v113; // rsi
  unsigned int v114; // r13d
  int v115; // edi
  __int64 v116; // rax
  int v117; // edi
  int v118; // esi
  _QWORD *v120; // [rsp+8h] [rbp-D8h]
  _QWORD *v121; // [rsp+10h] [rbp-D0h]
  _QWORD *v122; // [rsp+10h] [rbp-D0h]
  _QWORD *v123; // [rsp+10h] [rbp-D0h]
  _QWORD *v124; // [rsp+10h] [rbp-D0h]
  char v125; // [rsp+1Bh] [rbp-C5h]
  char v126; // [rsp+1Bh] [rbp-C5h]
  char v127; // [rsp+1Bh] [rbp-C5h]
  char v128; // [rsp+1Bh] [rbp-C5h]
  unsigned int v129; // [rsp+1Ch] [rbp-C4h]
  unsigned int v130; // [rsp+1Ch] [rbp-C4h]
  unsigned int v131; // [rsp+1Ch] [rbp-C4h]
  unsigned int v132; // [rsp+1Ch] [rbp-C4h]
  __int64 v133; // [rsp+20h] [rbp-C0h]
  __int64 v134; // [rsp+20h] [rbp-C0h]
  __int64 v135; // [rsp+20h] [rbp-C0h]
  __int64 v136; // [rsp+20h] [rbp-C0h]
  size_t v137; // [rsp+28h] [rbp-B8h]
  size_t v138; // [rsp+28h] [rbp-B8h]
  size_t v139; // [rsp+28h] [rbp-B8h]
  size_t v140; // [rsp+28h] [rbp-B8h]
  char *v141; // [rsp+30h] [rbp-B0h]
  void *s2; // [rsp+38h] [rbp-A8h]
  __int64 *v144; // [rsp+48h] [rbp-98h]
  char v145; // [rsp+48h] [rbp-98h]
  char v146; // [rsp+48h] [rbp-98h]
  _QWORD *v147; // [rsp+50h] [rbp-90h]
  _QWORD *v148; // [rsp+50h] [rbp-90h]
  __int64 *v149; // [rsp+50h] [rbp-90h]
  __int64 *v150; // [rsp+50h] [rbp-90h]
  char v152; // [rsp+63h] [rbp-7Dh]
  unsigned int v154; // [rsp+68h] [rbp-78h]
  size_t v155; // [rsp+68h] [rbp-78h]
  unsigned __int64 v156; // [rsp+68h] [rbp-78h]
  char *v157; // [rsp+70h] [rbp-70h]
  unsigned __int8 *v158; // [rsp+70h] [rbp-70h]
  _QWORD *v159; // [rsp+70h] [rbp-70h]
  _QWORD *v160; // [rsp+70h] [rbp-70h]
  _QWORD *v161; // [rsp+70h] [rbp-70h]
  _QWORD *v162; // [rsp+70h] [rbp-70h]
  size_t v163; // [rsp+70h] [rbp-70h]
  size_t v164; // [rsp+70h] [rbp-70h]
  unsigned __int64 v165; // [rsp+78h] [rbp-68h]
  _QWORD *v166; // [rsp+78h] [rbp-68h]
  char v167; // [rsp+78h] [rbp-68h]
  __int64 v168; // [rsp+78h] [rbp-68h]
  __int64 v169; // [rsp+78h] [rbp-68h]
  __int64 v170; // [rsp+78h] [rbp-68h]
  int v171; // [rsp+78h] [rbp-68h]
  __int64 *v172; // [rsp+78h] [rbp-68h]
  float v174; // [rsp+88h] [rbp-58h]
  int v175; // [rsp+88h] [rbp-58h]
  __int64 v177; // [rsp+98h] [rbp-48h]
  unsigned __int64 v178[7]; // [rsp+A8h] [rbp-38h] BYREF

  v8 = (_QWORD *)a1[6];
  v9 = (_QWORD *)a1[5];
  if ( v9 != v8 )
  {
    while ( 1 )
    {
      v15 = *(_DWORD *)(a4 + 24);
      v16 = (_QWORD *)(*v9 & 0xFFFFFFFFFFFFFFF8LL);
      if ( !v15 )
        goto LABEL_6;
      v11 = v15 - 1;
      v12 = *(_QWORD *)(a4 + 8);
      v13 = (v15 - 1) & (37 * *v16);
      v14 = *(_QWORD *)(v12 + 16LL * (v11 & (37 * (unsigned int)*v16)));
      if ( *v16 != v14 )
        break;
LABEL_4:
      if ( v8 == ++v9 )
        goto LABEL_12;
    }
    v45 = 1;
    while ( v14 != -1 )
    {
      v13 = v11 & (v45 + v13);
      v14 = *(_QWORD *)(v12 + 16LL * v13);
      if ( *v16 == v14 )
        goto LABEL_4;
      ++v45;
    }
LABEL_6:
    v17 = v16[3];
    for ( i = v16[4]; i != v17; v17 += 8 )
    {
      if ( *(_DWORD *)(*(_QWORD *)v17 + 8LL) == 2 && (*(_BYTE *)(*(_QWORD *)v17 + 12LL) & 0x10) == 0 )
        __asm { jmp     rdx }
    }
    goto LABEL_4;
  }
LABEL_12:
  v20 = a1[9];
  v21 = v20;
  v177 = a1[10];
  if ( v177 == v20 )
    return v20;
  do
  {
    while ( 1 )
    {
      LODWORD(v20) = dword_4FAB120;
      if ( dword_4FAB120 >= 0 && dword_4FAA770 >= dword_4FAB120 )
        goto LABEL_15;
      v20 = sub_1851200(a2, *(_QWORD *)v21) & 0xFFFFFFFFFFFFFFF8LL;
      v25 = v20;
      if ( !v20 )
        goto LABEL_15;
      v26 = *(_QWORD *)v20;
      v27 = *(_DWORD *)(a4 + 24);
      if ( v27 )
      {
        v22 = v27 - 1;
        v23 = *(_QWORD *)(a4 + 8);
        LODWORD(v20) = (v27 - 1) & (37 * v26);
        v24 = *(_QWORD *)(v23 + 16LL * (unsigned int)v20);
        if ( v26 == v24 )
          goto LABEL_15;
        v54 = 1;
        while ( v24 != -1 )
        {
          LODWORD(v20) = v22 & (v54 + v20);
          v24 = *(_QWORD *)(v23 + 16LL * (unsigned int)v20);
          if ( v26 == v24 )
            goto LABEL_15;
          ++v54;
        }
      }
      v28 = *(_BYTE *)(v21 + 8) & 7;
      v174 = (float)a3;
      if ( v28 == 3 )
      {
        v29 = *(float *)&dword_4FAAE80 * (float)a3;
      }
      else
      {
        v29 = (float)a3;
        if ( v28 == 1 )
        {
          v29 = v174 * *(float *)&dword_4FAACC0;
        }
        else if ( v28 == 4 )
        {
          v29 = v174 * *(float *)&dword_4FAADA0;
        }
      }
      v154 = (int)v29;
      v30 = (int)v29;
      v31 = *(_DWORD *)(a8 + 24);
      if ( !v31 )
      {
        ++*(_QWORD *)a8;
        goto LABEL_205;
      }
      v32 = *(_QWORD *)(a8 + 8);
      v33 = 0;
      v34 = 1;
      v35 = (v31 - 1) & (37 * v26);
      v36 = (__int64 *)(v32 + 24LL * v35);
      v37 = *v36;
      if ( v26 != *v36 )
        break;
LABEL_26:
      v38 = (_QWORD *)v36[2];
      if ( v38 )
      {
        LODWORD(v20) = *((_DWORD *)v36 + 2);
        if ( (float)(int)v20 < v29 )
        {
          *((_DWORD *)v36 + 2) = v154;
          goto LABEL_29;
        }
        goto LABEL_15;
      }
      LODWORD(v20) = *((_DWORD *)v36 + 2);
      if ( (float)(int)v20 < v29 )
      {
        v152 = 1;
        goto LABEL_46;
      }
LABEL_15:
      v21 += 16;
      if ( v177 == v21 )
        return v20;
    }
    while ( v37 != -1 )
    {
      if ( !v33 && v37 == -2 )
        v33 = v36;
      v35 = (v31 - 1) & (v34 + v35);
      v36 = (__int64 *)(v32 + 24LL * v35);
      v37 = *v36;
      if ( v26 == *v36 )
        goto LABEL_26;
      ++v34;
    }
    if ( v33 )
      v36 = v33;
    ++*(_QWORD *)a8;
    v92 = *(_DWORD *)(a8 + 16) + 1;
    if ( 4 * v92 < 3 * v31 )
    {
      if ( v31 - *(_DWORD *)(a8 + 20) - v92 <= v31 >> 3 )
      {
        sub_1852FB0(a8, v31);
        v110 = *(_DWORD *)(a8 + 24);
        if ( !v110 )
        {
LABEL_253:
          ++*(_DWORD *)(a8 + 16);
          BUG();
        }
        v111 = v110 - 1;
        v112 = *(_QWORD *)(a8 + 8);
        v113 = 0;
        v114 = v111 & (37 * v26);
        v30 = (int)v29;
        v36 = (__int64 *)(v112 + 24LL * v114);
        v92 = *(_DWORD *)(a8 + 16) + 1;
        v115 = 1;
        v116 = *v36;
        if ( v26 != *v36 )
        {
          while ( v116 != -1 )
          {
            if ( v116 == -2 && !v113 )
              v113 = v36;
            v114 = v111 & (v115 + v114);
            v36 = (__int64 *)(v112 + 24LL * v114);
            v116 = *v36;
            if ( v26 == *v36 )
              goto LABEL_175;
            ++v115;
          }
          if ( v113 )
            v36 = v113;
        }
      }
      goto LABEL_175;
    }
LABEL_205:
    sub_1852FB0(a8, 2 * v31);
    v103 = *(_DWORD *)(a8 + 24);
    if ( !v103 )
      goto LABEL_253;
    v104 = v103 - 1;
    v30 = (int)v29;
    v105 = *(_QWORD *)(a8 + 8);
    v106 = (v103 - 1) & (37 * v26);
    v36 = (__int64 *)(v105 + 24LL * v106);
    v107 = *v36;
    v92 = *(_DWORD *)(a8 + 16) + 1;
    if ( v26 != *v36 )
    {
      v108 = 1;
      v109 = 0;
      while ( v107 != -1 )
      {
        if ( !v109 && v107 == -2 )
          v109 = v36;
        v106 = v104 & (v108 + v106);
        v36 = (__int64 *)(v105 + 24LL * v106);
        v107 = *v36;
        if ( v26 == *v36 )
          goto LABEL_175;
        ++v108;
      }
      if ( v109 )
        v36 = v109;
    }
LABEL_175:
    *(_DWORD *)(a8 + 16) = v92;
    if ( *v36 != -1 )
      --*(_DWORD *)(a8 + 20);
    *v36 = v26;
    v36[2] = 0;
    *((_DWORD *)v36 + 2) = v154;
    v152 = 0;
LABEL_46:
    v46 = *(char **)(v25 + 24);
    v47 = a1[4];
    s2 = (void *)a1[3];
    v141 = *(char **)(v25 + 32);
    v20 = (v141 - v46) >> 5;
    v165 = (v141 - v46) >> 3;
    v48 = v165;
    if ( v20 <= 0 )
    {
LABEL_88:
      switch ( v48 )
      {
        case 2LL:
          v62 = *(_BYTE *)(a2 + 176);
          break;
        case 3LL:
          v62 = *(_BYTE *)(a2 + 176);
          v63 = *(_QWORD *)v46;
          if ( !v62 || (*(_BYTE *)(v63 + 12) & 0x20) != 0 )
          {
            v118 = *(_DWORD *)(v63 + 8);
            if ( v118 != 2 )
            {
              LODWORD(v20) = *(_BYTE *)(v63 + 12) & 0xF;
              switch ( (char)v20 )
              {
                case 0:
                case 1:
                case 3:
                case 5:
                case 6:
                case 7:
                case 8:
                  if ( !v118 )
                  {
                    v63 = *(_QWORD *)(v63 + 64);
                    LOBYTE(v20) = *(_BYTE *)(v63 + 12) & 0xF;
                  }
                  LODWORD(v20) = (unsigned __int8)v20 - 7;
                  if ( (unsigned int)v20 <= 1 && v165 > 1 )
                  {
                    if ( *(_QWORD *)(v63 + 32) != v47 )
                      break;
                    if ( v47 )
                    {
                      v164 = v47;
                      v146 = *(_BYTE *)(a2 + 176);
                      v150 = v36;
                      LODWORD(v20) = memcmp(*(const void **)(v63 + 24), s2, v47);
                      v47 = v164;
                      v36 = v150;
                      v62 = v146;
                      if ( (_DWORD)v20 )
                        break;
                    }
                  }
                  LODWORD(v20) = (int)v29;
                  if ( v154 < *(_DWORD *)(v63 + 64) || (*(_BYTE *)(v63 + 12) & 0x10) != 0 )
                    break;
                  goto LABEL_115;
                case 2:
                case 4:
                case 9:
                case 10:
                  break;
                default:
                  goto LABEL_254;
              }
            }
          }
          v46 += 8;
          break;
        case 1LL:
          v62 = *(_BYTE *)(a2 + 176);
LABEL_104:
          v65 = *(_QWORD *)v46;
          if ( !v62 || (*(_BYTE *)(v65 + 12) & 0x20) != 0 )
          {
            v66 = *(_DWORD *)(v65 + 8);
            if ( v66 != 2 )
            {
              LODWORD(v20) = *(_BYTE *)(v65 + 12) & 0xF;
              switch ( (char)v20 )
              {
                case 0:
                case 1:
                case 3:
                case 5:
                case 6:
                case 7:
                case 8:
                  if ( !v66 )
                  {
                    v65 = *(_QWORD *)(v65 + 64);
                    LOBYTE(v20) = *(_BYTE *)(v65 + 12) & 0xF;
                  }
                  LODWORD(v20) = (unsigned __int8)v20 - 7;
                  if ( (unsigned int)v20 <= 1 && v165 > 1 )
                  {
                    if ( *(_QWORD *)(v65 + 32) != v47 )
                      goto LABEL_201;
                    if ( v47 )
                    {
                      v172 = v36;
                      LODWORD(v20) = memcmp(*(const void **)(v65 + 24), s2, v47);
                      v36 = v172;
                      if ( (_DWORD)v20 )
                        goto LABEL_201;
                    }
                  }
                  LODWORD(v20) = (int)v29;
                  if ( v154 < *(_DWORD *)(v65 + 64) || (*(_BYTE *)(v65 + 12) & 0x10) != 0 )
                    goto LABEL_201;
                  goto LABEL_115;
                case 2:
                case 4:
                case 9:
                case 10:
                  goto LABEL_201;
                default:
                  goto LABEL_254;
              }
            }
          }
          goto LABEL_201;
        default:
LABEL_201:
          v36[2] = 0;
LABEL_202:
          if ( v152 )
          {
            LODWORD(v20) = (int)v29;
            *((_DWORD *)v36 + 2) = v154;
          }
          goto LABEL_15;
      }
      v64 = *(_QWORD *)v46;
      if ( !v62 || (*(_BYTE *)(v64 + 12) & 0x20) != 0 )
      {
        v117 = *(_DWORD *)(v64 + 8);
        if ( v117 != 2 )
        {
          LODWORD(v20) = *(_BYTE *)(v64 + 12) & 0xF;
          switch ( (char)v20 )
          {
            case 0:
            case 1:
            case 3:
            case 5:
            case 6:
            case 7:
            case 8:
              if ( !v117 )
              {
                v64 = *(_QWORD *)(v64 + 64);
                LOBYTE(v20) = *(_BYTE *)(v64 + 12) & 0xF;
              }
              LODWORD(v20) = (unsigned __int8)v20 - 7;
              if ( (unsigned int)v20 <= 1 && v165 > 1 )
              {
                if ( *(_QWORD *)(v64 + 32) != v47 )
                  break;
                if ( v47 )
                {
                  v163 = v47;
                  v145 = v62;
                  v149 = v36;
                  LODWORD(v20) = memcmp(*(const void **)(v64 + 24), s2, v47);
                  v47 = v163;
                  v36 = v149;
                  v62 = v145;
                  if ( (_DWORD)v20 )
                    break;
                }
              }
              LODWORD(v20) = (int)v29;
              if ( v154 < *(_DWORD *)(v64 + 64) || (*(_BYTE *)(v64 + 12) & 0x10) != 0 )
                break;
              goto LABEL_115;
            case 2:
            case 4:
            case 9:
            case 10:
              break;
            default:
              goto LABEL_254;
          }
        }
      }
      v46 += 8;
      goto LABEL_104;
    }
    v49 = v46 + 24;
    v50 = v46 + 8;
    v20 = (__int64)&v46[32 * v20];
    v144 = v36;
    v51 = *(_BYTE *)(a2 + 176);
    v157 = (char *)v20;
    while ( 1 )
    {
      v52 = *((_QWORD *)v50 - 1);
      if ( !v51 )
        break;
      if ( (*(_BYTE *)(v52 + 12) & 0x20) != 0 )
      {
        v53 = *(_DWORD *)(v52 + 8);
        if ( v53 != 2 )
        {
          LODWORD(v20) = *(_BYTE *)(v52 + 12) & 0xF;
          switch ( (char)v20 )
          {
            case 0:
            case 1:
            case 3:
            case 5:
            case 6:
            case 7:
            case 8:
              goto LABEL_56;
            case 2:
            case 4:
            case 9:
            case 10:
              break;
            default:
              goto LABEL_254;
          }
        }
      }
      v55 = *(_QWORD *)v50;
LABEL_98:
      if ( (*(_BYTE *)(v55 + 12) & 0x20) != 0 )
      {
        v56 = *(_DWORD *)(v55 + 8);
        if ( v56 != 2 )
          goto LABEL_63;
LABEL_69:
        v57 = *((_QWORD *)v50 + 1);
        v58 = v46 + 16;
        if ( !v51 )
          goto LABEL_70;
      }
      else
      {
        v57 = *((_QWORD *)v50 + 1);
        v58 = v46 + 16;
      }
      if ( (*(_BYTE *)(v57 + 12) & 0x20) != 0 )
      {
        v59 = *(_DWORD *)(v57 + 8);
        if ( v59 != 2 )
        {
          LODWORD(v20) = *(_BYTE *)(v57 + 12) & 0xF;
          switch ( (char)v20 )
          {
            case 0:
            case 1:
            case 3:
            case 5:
            case 6:
            case 7:
            case 8:
LABEL_72:
              if ( !v59 )
              {
                v57 = *(_QWORD *)(v57 + 64);
                LOBYTE(v20) = *(_BYTE *)(v57 + 12) & 0xF;
              }
              LODWORD(v20) = (unsigned __int8)v20 - 7;
              if ( (unsigned int)v20 <= 1 && v165 > 1 )
              {
                if ( *(_QWORD *)(v57 + 32) != v47 )
                  goto LABEL_77;
                if ( v47 )
                {
                  v120 = v58;
                  v127 = v51;
                  v123 = v49;
                  v131 = v30;
                  v135 = v57;
                  v139 = v47;
                  LODWORD(v20) = memcmp(*(const void **)(v57 + 24), s2, v47);
                  v47 = v139;
                  v57 = v135;
                  v30 = v131;
                  v49 = v123;
                  v51 = v127;
                  v58 = v120;
                  if ( (_DWORD)v20 )
                    goto LABEL_77;
                }
              }
              if ( v30 < *(_DWORD *)(v57 + 64) || (*(_BYTE *)(v57 + 12) & 0x10) != 0 )
                goto LABEL_77;
              v46 = (char *)v58;
              v36 = v144;
              break;
            case 2:
            case 4:
            case 9:
            case 10:
              goto LABEL_77;
            default:
              goto LABEL_254;
          }
          goto LABEL_115;
        }
LABEL_77:
        v60 = *((_QWORD *)v50 + 2);
        if ( !v51 )
          goto LABEL_79;
      }
      else
      {
        v60 = *((_QWORD *)v50 + 2);
      }
      if ( (*(_BYTE *)(v60 + 12) & 0x20) != 0 )
        goto LABEL_79;
LABEL_86:
      v46 += 32;
      v49 += 4;
      v50 += 32;
      if ( v157 == v46 )
      {
        v36 = v144;
        v48 = (v141 - v46) >> 3;
        goto LABEL_88;
      }
    }
    v53 = *(_DWORD *)(v52 + 8);
    if ( v53 != 2 )
    {
      LODWORD(v20) = *(_BYTE *)(v52 + 12) & 0xF;
      switch ( (char)v20 )
      {
        case 0:
        case 1:
        case 3:
        case 5:
        case 6:
        case 7:
        case 8:
LABEL_56:
          if ( !v53 )
          {
            v52 = *(_QWORD *)(v52 + 64);
            LOBYTE(v20) = *(_BYTE *)(v52 + 12) & 0xF;
          }
          LODWORD(v20) = (unsigned __int8)v20 - 7;
          if ( (unsigned int)v20 <= 1 && v165 > 1 )
          {
            if ( *(_QWORD *)(v52 + 32) != v47 )
              goto LABEL_61;
            if ( v47 )
            {
              v125 = v51;
              v121 = v49;
              v129 = v30;
              v133 = v52;
              v137 = v47;
              LODWORD(v20) = memcmp(*(const void **)(v52 + 24), s2, v47);
              v47 = v137;
              v52 = v133;
              v30 = v129;
              v49 = v121;
              v51 = v125;
              if ( (_DWORD)v20 )
                goto LABEL_61;
            }
          }
          if ( v30 < *(_DWORD *)(v52 + 64) || (*(_BYTE *)(v52 + 12) & 0x10) != 0 )
            goto LABEL_61;
          v36 = v144;
          goto LABEL_115;
        case 2:
        case 4:
        case 9:
        case 10:
LABEL_61:
          v55 = *(_QWORD *)v50;
          if ( v51 )
            goto LABEL_98;
          v56 = *(_DWORD *)(v55 + 8);
          if ( v56 != 2 )
            goto LABEL_63;
          goto LABEL_161;
        default:
LABEL_254:
          ++*(_DWORD *)(v25 + 16);
          BUG();
      }
    }
    v55 = *(_QWORD *)v50;
    v56 = *(_DWORD *)(*(_QWORD *)v50 + 8LL);
    if ( v56 == 2 )
    {
LABEL_161:
      v57 = *((_QWORD *)v50 + 1);
      v58 = v46 + 16;
LABEL_70:
      v59 = *(_DWORD *)(v57 + 8);
      if ( v59 != 2 )
      {
        LODWORD(v20) = *(_BYTE *)(v57 + 12) & 0xF;
        switch ( (char)v20 )
        {
          case 0:
          case 1:
          case 3:
          case 5:
          case 6:
          case 7:
          case 8:
            goto LABEL_72;
          case 2:
          case 4:
          case 9:
          case 10:
            break;
          default:
            goto LABEL_254;
        }
      }
      v60 = *((_QWORD *)v50 + 2);
LABEL_79:
      v61 = *(_DWORD *)(v60 + 8);
      if ( v61 != 2 )
      {
        LODWORD(v20) = *(_BYTE *)(v60 + 12) & 0xF;
        switch ( (char)v20 )
        {
          case 0:
          case 1:
          case 3:
          case 5:
          case 6:
          case 7:
          case 8:
            if ( !v61 )
            {
              v60 = *(_QWORD *)(v60 + 64);
              LOBYTE(v20) = *(_BYTE *)(v60 + 12) & 0xF;
            }
            LODWORD(v20) = (unsigned __int8)v20 - 7;
            if ( (unsigned int)v20 <= 1 && v165 > 1 )
            {
              if ( *(_QWORD *)(v60 + 32) != v47 )
                goto LABEL_86;
              if ( v47 )
              {
                v128 = v51;
                v124 = v49;
                v132 = v30;
                v136 = v60;
                v140 = v47;
                LODWORD(v20) = memcmp(*(const void **)(v60 + 24), s2, v47);
                v47 = v140;
                v60 = v136;
                v30 = v132;
                v49 = v124;
                v51 = v128;
                if ( (_DWORD)v20 )
                  goto LABEL_86;
              }
            }
            if ( v30 < *(_DWORD *)(v60 + 64) || (*(_BYTE *)(v60 + 12) & 0x10) != 0 )
              goto LABEL_86;
            v36 = v144;
            v46 = (char *)v49;
            break;
          case 2:
          case 4:
          case 9:
          case 10:
            goto LABEL_86;
          default:
            goto LABEL_254;
        }
        goto LABEL_115;
      }
      goto LABEL_86;
    }
LABEL_63:
    LODWORD(v20) = *(_BYTE *)(v55 + 12) & 0xF;
    switch ( (char)v20 )
    {
      case 0:
      case 1:
      case 3:
      case 5:
      case 6:
      case 7:
      case 8:
        if ( !v56 )
        {
          v55 = *(_QWORD *)(v55 + 64);
          LOBYTE(v20) = *(_BYTE *)(v55 + 12) & 0xF;
        }
        LODWORD(v20) = (unsigned __int8)v20 - 7;
        if ( (unsigned int)v20 <= 1 && v165 > 1 )
        {
          if ( *(_QWORD *)(v55 + 32) != v47 )
            goto LABEL_69;
          if ( v47 )
          {
            v126 = v51;
            v122 = v49;
            v130 = v30;
            v134 = v55;
            v138 = v47;
            LODWORD(v20) = memcmp(*(const void **)(v55 + 24), s2, v47);
            v47 = v138;
            v55 = v134;
            v30 = v130;
            v49 = v122;
            v51 = v126;
            if ( (_DWORD)v20 )
              goto LABEL_69;
          }
        }
        if ( v30 < *(_DWORD *)(v55 + 64) || (*(_BYTE *)(v55 + 12) & 0x10) != 0 )
          goto LABEL_69;
        v36 = v144;
        v46 = v50;
        break;
      case 2:
      case 4:
      case 9:
      case 10:
        goto LABEL_69;
      default:
        goto LABEL_254;
    }
LABEL_115:
    if ( v141 == v46 )
      goto LABEL_201;
    v67 = *(_QWORD *)v46;
    v36[2] = *(_QWORD *)v46;
    if ( !v67 )
      goto LABEL_202;
    if ( !*(_DWORD *)(v67 + 8) )
      v67 = *(_QWORD *)(v67 + 64);
    v36[2] = v67;
    v166 = (_QWORD *)v67;
    v155 = *(_QWORD *)(v67 + 32);
    v158 = *(unsigned __int8 **)(v67 + 24);
    v68 = sub_1852A30(a6, v158, v155);
    v69 = *(_QWORD *)v25;
    v38 = v166;
    v70 = *(_QWORD *)v68;
    v71 = *(_QWORD *)(*(_QWORD *)v68 + 16LL);
    v72 = *(_QWORD **)(*(_QWORD *)(*(_QWORD *)v68 + 8LL) + 8 * (*(_QWORD *)v25 % v71));
    v73 = *(_QWORD *)v25 % v71;
    if ( !v72 )
      goto LABEL_162;
    v74 = (_QWORD *)*v72;
    if ( v69 == *(_QWORD *)(*v72 + 8LL) )
    {
LABEL_124:
      v167 = 0;
      if ( !*v72 )
        goto LABEL_162;
    }
    else
    {
      while ( 1 )
      {
        v75 = (_QWORD *)*v74;
        if ( !*v74 )
          break;
        v72 = v74;
        if ( v73 != v75[1] % v71 )
          break;
        v74 = (_QWORD *)*v74;
        if ( v69 == v75[1] )
          goto LABEL_124;
      }
LABEL_162:
      v148 = v38;
      v170 = v70;
      v91 = (_QWORD *)sub_22077B0(16);
      if ( v91 )
        *v91 = 0;
      v91[1] = v69;
      sub_1851560((_QWORD *)(v170 + 8), v73, v69, (__int64)v91, 1);
      v167 = 1;
      v38 = v148;
    }
    if ( a7 )
    {
      v147 = v38;
      v76 = sub_1852A30(a7, v158, v155);
      v77 = *(_QWORD *)v25;
      v38 = v147;
      v78 = *(_QWORD *)(*(_QWORD *)v76 + 16LL);
      v79 = (_QWORD *)(*(_QWORD *)v76 + 8LL);
      v80 = *(_QWORD **)(*v79 + 8 * (*(_QWORD *)v25 % v78));
      if ( !v80 )
        goto LABEL_189;
      v81 = (_QWORD *)*v80;
      if ( v77 == *(_QWORD *)(*v80 + 8LL) )
      {
LABEL_131:
        if ( !*v80 )
          goto LABEL_189;
      }
      else
      {
        while ( 1 )
        {
          v82 = (_QWORD *)*v81;
          if ( !*v81 )
            break;
          v80 = v81;
          if ( *(_QWORD *)v25 % v78 != v82[1] % v78 )
            break;
          v81 = (_QWORD *)*v81;
          if ( v77 == v82[1] )
            goto LABEL_131;
        }
LABEL_189:
        v156 = *(_QWORD *)v25 % v78;
        v102 = (_QWORD *)sub_22077B0(16);
        if ( v102 )
          *v102 = 0;
        v102[1] = v77;
        sub_1851560(v79, v156, v77, (__int64)v102, 1);
        v38 = v147;
      }
      if ( v167 )
      {
        if ( v38[9] != v38[10] )
        {
          v159 = v38;
          v168 = v25;
          v83 = (__int64 *)v38[10];
          v84 = v79;
          v85 = (__int64 *)v38[9];
          do
          {
            v86 = *v85;
            v85 += 2;
            v178[0] = *(_QWORD *)(v86 & 0xFFFFFFFFFFFFFFF8LL);
            sub_18517D0(v84, v178, 1);
          }
          while ( v83 != v85 );
          v25 = v168;
          v38 = v159;
          v79 = v84;
        }
        if ( v38[5] != v38[6] )
        {
          v160 = v38;
          v169 = v25;
          v87 = (__int64 *)v38[6];
          v88 = v79;
          v89 = (__int64 *)v38[5];
          do
          {
            v90 = *v89++;
            v178[0] = *(_QWORD *)(v90 & 0xFFFFFFFFFFFFFFF8LL);
            sub_18517D0(v88, v178, 1);
          }
          while ( v87 != v89 );
          v25 = v169;
          v38 = v160;
        }
      }
    }
LABEL_29:
    if ( (*(_BYTE *)(v21 + 8) & 7) == 3 )
      v39 = v174 * *(float *)&dword_4FAAF60;
    else
      v39 = v174 * *(float *)&dword_4FAB040;
    v40 = *(_QWORD *)v25;
    ++dword_4FAA770;
    LODWORD(v25) = *(_DWORD *)(a5 + 8);
    v41 = *(unsigned int *)(a5 + 12);
    if ( (unsigned int)v25 >= (unsigned int)v41 )
    {
      v161 = v38;
      v93 = ((((unsigned __int64)(v41 + 2) >> 1) | (v41 + 2)) >> 2) | ((unsigned __int64)(v41 + 2) >> 1) | (v41 + 2);
      v94 = (v93 >> 4) | v93;
      v95 = ((v94 >> 8) | v94 | (((v94 >> 8) | v94) >> 16) | (((v94 >> 8) | v94) >> 32)) + 1;
      if ( v95 > 0xFFFFFFFF )
        v95 = 0xFFFFFFFFLL;
      v175 = v95;
      v96 = malloc(24 * v95);
      v97 = v175;
      v38 = v161;
      v42 = v96;
      if ( !v96 )
      {
        sub_16BD1C0("Allocation failed", 1u);
        v38 = v161;
        v97 = v175;
        LODWORD(v25) = *(_DWORD *)(a5 + 8);
      }
      v98 = v42;
      v99 = *(_QWORD *)a5;
      v43 = 24LL * (unsigned int)v25;
      v100 = *(_QWORD *)a5;
      v101 = *(_QWORD *)a5 + v43;
      if ( *(_QWORD *)a5 != v101 )
      {
        do
        {
          if ( v98 )
          {
            *(_QWORD *)v98 = *(_QWORD *)v100;
            *(_DWORD *)(v98 + 8) = *(_DWORD *)(v100 + 8);
            *(_QWORD *)(v98 + 16) = *(_QWORD *)(v100 + 16);
          }
          v100 += 24LL;
          v98 += 24;
        }
        while ( v101 != v100 );
      }
      if ( v99 != a5 + 16 )
      {
        v162 = v38;
        v171 = v97;
        _libc_free(v99);
        v38 = v162;
        v97 = v171;
        v25 = *(unsigned int *)(a5 + 8);
        v43 = 24 * v25;
      }
      *(_QWORD *)a5 = v42;
      *(_DWORD *)(a5 + 12) = v97;
    }
    else
    {
      v42 = *(_QWORD *)a5;
      v43 = 24LL * (unsigned int)v25;
    }
    v44 = v42 + v43;
    if ( v44 )
    {
      *(_QWORD *)v44 = v40;
      *(_QWORD *)(v44 + 16) = v38;
      *(_DWORD *)(v44 + 8) = (int)v39;
      LODWORD(v25) = *(_DWORD *)(a5 + 8);
    }
    LODWORD(v20) = a5;
    v21 += 16;
    *(_DWORD *)(a5 + 8) = v25 + 1;
  }
  while ( v177 != v21 );
  return v20;
}
