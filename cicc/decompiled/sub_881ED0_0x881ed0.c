// Function: sub_881ED0
// Address: 0x881ed0
//
void __fastcall sub_881ED0(__int64 a1, int a2, int a3)
{
  char v3; // dl
  __int64 v4; // r11
  __int64 v5; // rdi
  __int64 v7; // rsi
  __int64 v8; // rdx
  __int64 v9; // rcx
  int v10; // ebx
  int v11; // r15d
  __int64 v12; // r12
  unsigned __int8 v13; // cl
  __int64 v14; // r10
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // rax
  char v19; // dl
  __int64 v20; // rcx
  _BOOL4 v21; // eax
  __int64 v22; // rcx
  unsigned int v23; // edi
  __int64 v24; // rax
  __int64 v25; // rcx
  __int64 v26; // rsi
  __int64 v27; // rdi
  __int64 v28; // rcx
  char v29; // dl
  __int64 v30; // rdi
  _DWORD *v31; // r9
  __int64 v32; // r8
  int v33; // ecx
  __int64 v34; // r13
  __int64 *v35; // r14
  char v36; // al
  char v37; // al
  char v38; // al
  __int64 v39; // rdx
  unsigned __int8 v40; // al
  unsigned __int8 v41; // si
  char v42; // cl
  unsigned __int8 v43; // di
  __int64 v44; // rdx
  __int64 v45; // rsi
  char v46; // al
  _DWORD *v47; // rsi
  unsigned int v48; // edi
  __int64 v49; // rdx
  int v50; // edx
  __int64 v51; // rsi
  __int64 v52; // rcx
  char v53; // r14
  __int64 v54; // rax
  __int64 v55; // rcx
  int v56; // r14d
  __int64 v57; // rsi
  __int64 *v58; // rax
  __int64 *v59; // rsi
  __int64 v60; // rax
  __int64 v61; // rax
  char v62; // dl
  unsigned __int8 v63; // di
  char v64; // si
  __int64 v65; // rax
  __int64 v66; // rax
  __int64 v67; // rax
  __int64 v68; // rdx
  __int64 v69; // rcx
  __int64 v70; // rdi
  __int64 v71; // rsi
  int v72; // eax
  __int64 *v73; // rax
  __int64 v74; // rax
  _DWORD *v75; // rsi
  unsigned int v76; // edi
  __int64 v77; // rdx
  int v78; // eax
  __int64 v79; // rdi
  int v80; // eax
  __int64 v81; // rax
  int v82; // eax
  __int64 v83; // rdi
  __int64 v84; // rax
  int v85; // eax
  _BOOL4 v86; // eax
  _QWORD *v87; // rdx
  __int64 v88; // rsi
  __int64 *v89; // rcx
  __int64 v90; // rcx
  __int64 v91; // r13
  int v92; // eax
  __int64 v93; // r11
  FILE *v94; // rbx
  int v95; // eax
  __int64 v96; // rax
  _QWORD **v97; // rdi
  _QWORD **v98; // rax
  _QWORD *v99; // rax
  __int64 v100; // rax
  __int64 v101; // rsi
  __int64 v102; // rax
  __int64 v103; // rdx
  _BOOL4 v104; // eax
  char v105; // cl
  __int64 v106; // rdx
  __int64 v107; // rdx
  char v108; // dl
  unsigned __int8 v109; // r13
  __int64 v110; // rax
  __int64 v111; // rcx
  __int64 v112; // rax
  __int64 v113; // rdx
  _DWORD *v114; // rsi
  __int64 v115; // rax
  _DWORD *v116; // [rsp+8h] [rbp-E8h]
  __int64 v117; // [rsp+10h] [rbp-E0h]
  unsigned int v118; // [rsp+1Ch] [rbp-D4h]
  __int64 v119; // [rsp+20h] [rbp-D0h]
  _DWORD *v120; // [rsp+28h] [rbp-C8h]
  __int64 v121; // [rsp+28h] [rbp-C8h]
  unsigned __int8 v122; // [rsp+30h] [rbp-C0h]
  _DWORD *v123; // [rsp+30h] [rbp-C0h]
  __int64 v124; // [rsp+30h] [rbp-C0h]
  __int64 v125; // [rsp+30h] [rbp-C0h]
  _DWORD *v126; // [rsp+30h] [rbp-C0h]
  _DWORD *v127; // [rsp+30h] [rbp-C0h]
  bool v128; // [rsp+38h] [rbp-B8h]
  unsigned int v129; // [rsp+38h] [rbp-B8h]
  unsigned int v130; // [rsp+38h] [rbp-B8h]
  unsigned int v131; // [rsp+38h] [rbp-B8h]
  unsigned int v132; // [rsp+38h] [rbp-B8h]
  __int64 v133; // [rsp+38h] [rbp-B8h]
  unsigned int v134; // [rsp+38h] [rbp-B8h]
  __int64 v135; // [rsp+38h] [rbp-B8h]
  unsigned int v136; // [rsp+38h] [rbp-B8h]
  unsigned int v137; // [rsp+38h] [rbp-B8h]
  __int64 v138; // [rsp+38h] [rbp-B8h]
  __int64 v139; // [rsp+38h] [rbp-B8h]
  __int64 v140; // [rsp+40h] [rbp-B0h]
  __int64 v141; // [rsp+40h] [rbp-B0h]
  __int64 v142; // [rsp+40h] [rbp-B0h]
  unsigned int v143; // [rsp+40h] [rbp-B0h]
  __int64 v144; // [rsp+40h] [rbp-B0h]
  unsigned int v145; // [rsp+40h] [rbp-B0h]
  __int64 v146; // [rsp+40h] [rbp-B0h]
  __int64 v147; // [rsp+40h] [rbp-B0h]
  unsigned int v148; // [rsp+40h] [rbp-B0h]
  bool v149; // [rsp+48h] [rbp-A8h]
  unsigned int *v150; // [rsp+48h] [rbp-A8h]
  unsigned __int8 v151; // [rsp+48h] [rbp-A8h]
  __int64 v152; // [rsp+48h] [rbp-A8h]
  __int64 v153; // [rsp+48h] [rbp-A8h]
  __int64 v154; // [rsp+48h] [rbp-A8h]
  __int64 v155; // [rsp+48h] [rbp-A8h]
  __int64 v156; // [rsp+48h] [rbp-A8h]
  __int64 v157; // [rsp+48h] [rbp-A8h]
  __int64 v158; // [rsp+48h] [rbp-A8h]
  __int64 v159; // [rsp+48h] [rbp-A8h]
  __int64 v160; // [rsp+48h] [rbp-A8h]
  __int64 v161; // [rsp+50h] [rbp-A0h]
  __int64 v162; // [rsp+58h] [rbp-98h]
  __int64 v163; // [rsp+60h] [rbp-90h]
  __int64 v165; // [rsp+68h] [rbp-88h]
  __int64 v166; // [rsp+68h] [rbp-88h]
  __int64 v167; // [rsp+68h] [rbp-88h]
  __int64 v168; // [rsp+68h] [rbp-88h]
  __int64 v169; // [rsp+68h] [rbp-88h]
  __int64 v170; // [rsp+78h] [rbp-78h] BYREF
  __m128i v171; // [rsp+80h] [rbp-70h] BYREF
  char v172; // [rsp+90h] [rbp-60h]
  char v173; // [rsp+91h] [rbp-5Fh]
  _BYTE *v174; // [rsp+98h] [rbp-58h]

  v3 = *(_BYTE *)(a1 + 81);
  if ( (v3 & 0x20) == 0 )
  {
    v4 = a1;
    v5 = *(_QWORD *)a1;
    v170 = 0;
    v163 = v5;
    if ( a2 == -1 )
    {
      if ( (v3 & 0x10) == 0 && !unk_4D04968 )
        goto LABEL_69;
      goto LABEL_114;
    }
    v7 = 776LL * a2;
    v8 = qword_4F04C68[0] + v7;
    v161 = v7;
    if ( (*(_BYTE *)(qword_4F04C68[0] + v7 + 4) & 0xFB) != 0 )
      goto LABEL_30;
    v9 = *(_QWORD *)(v8 + 24);
    if ( !v9 )
      v9 = v8 + 32;
    if ( (*(_BYTE *)(v9 + 144) & 1) == 0 )
    {
LABEL_30:
      v12 = *(_QWORD *)(v5 + 24);
      v25 = 776LL * dword_4F04C64;
      v10 = *(_DWORD *)(qword_4F04C68[0] + v25);
      if ( a2 == dword_4F04C64 )
      {
        v11 = 0;
LABEL_8:
        v13 = *(_BYTE *)(v4 + 80);
        v14 = 0;
        if ( v13 == 7 )
        {
          if ( dword_4F04C58 == -1 )
          {
            v31 = dword_4F04BA0;
            v32 = (unsigned int)dword_4F04BA0[7];
            if ( dword_4F077C4 != 2 )
            {
              if ( !v12 )
                goto LABEL_26;
              goto LABEL_44;
            }
            goto LABEL_144;
          }
          v162 = v4;
          sub_878710(v4, &v171);
          if ( (v173 & 0x40) == 0 )
          {
            v172 &= ~0x80u;
            v174 = 0;
          }
          v18 = sub_7D5DD0(&v171, 0x80000008, v15, v16, v17);
          v4 = v162;
          v14 = v18;
          if ( v18 )
          {
            v19 = *(_BYTE *)(v18 + 80);
            if ( (unsigned __int8)(v19 - 7) > 2u )
              goto LABEL_283;
            v20 = *(_QWORD *)(v162 + 88);
            v21 = 0;
            if ( v20 )
            {
              v22 = *(_QWORD *)(v20 + 40);
              if ( v22 )
                v21 = *(_BYTE *)(v22 + 28) == 0 || *(_BYTE *)(v22 + 28) == 3;
            }
            if ( v19 == 7
              && (v106 = *(_QWORD *)(v14 + 88)) != 0
              && (v107 = *(_QWORD *)(v106 + 40)) != 0
              && ((v108 = *(_BYTE *)(v107 + 28)) == 0 || v108 == 3)
              && v21 )
            {
LABEL_283:
              v14 = 0;
            }
            else if ( (v174[82] & 8) == 0 )
            {
              if ( v174[80] == 16 )
              {
                if ( (v174[96] & 4) != 0 )
                  v14 = (__int64)v174;
              }
              else
              {
                v14 = (__int64)v174;
              }
            }
          }
          v13 = *(_BYTE *)(v162 + 80);
        }
LABEL_42:
        v31 = dword_4F04BA0;
        v32 = (unsigned int)dword_4F04BA0[v13];
        if ( dword_4F077C4 != 2 )
          goto LABEL_43;
        v50 = dword_4F04C58;
        if ( dword_4F04C58 == -1 )
          goto LABEL_144;
        v151 = v13 - 4;
        v51 = qword_4F04C68[0];
        v52 = qword_4F04C68[0] + v161;
        v53 = *(_BYTE *)(qword_4F04C68[0] + v161 + 4);
        if ( v53 == 2 && (*(_WORD *)(v52 - 772) & 0x40FF) == 0xF )
        {
          v73 = *(__int64 **)(v52 - 752);
          if ( !v73 )
            v73 = (__int64 *)(v52 - 744);
          v74 = *v73;
          if ( v74 )
          {
            while ( v163 != *(_QWORD *)v74
                 || v151 <= 2u != (unsigned __int8)(*(_BYTE *)(v74 + 80) - 4) <= 2u && !dword_4D04964 )
            {
              v74 = *(_QWORD *)(v74 + 16);
              if ( !v74 )
              {
                v50 = dword_4F04C58;
                goto LABEL_127;
              }
            }
            if ( a3 )
              goto LABEL_26;
            v75 = (_DWORD *)(v4 + 48);
            v76 = 704;
            v132 = v32;
            v142 = v14;
            v77 = *(_QWORD *)(*(_QWORD *)v4 + 8LL);
            v153 = v4;
            goto LABEL_199;
          }
        }
LABEL_127:
        if ( !unk_4D04350 && (!dword_4F077BC || (_DWORD)qword_4F077B4 || qword_4F077A8 > 0x9EFBu) )
        {
          if ( v53 == 2 )
          {
            if ( (*(_BYTE *)(v52 + 5) & 0x20) == 0 )
              goto LABEL_131;
            v96 = v52 - 776;
            if ( *(_BYTE *)(v52 - 1548) == 15 )
              v96 = v52 - 1552;
            if ( (*(_BYTE *)(v96 + 8) & 8) == 0 || (*(_BYTE *)(v96 + 5) & 0x40) != 0 )
            {
LABEL_129:
              v54 = *(_QWORD *)(v163 + 24);
              if ( v54 && *(_BYTE *)(v54 + 80) == 7 && *(char *)(*(_QWORD *)(v54 + 88) + 175LL) < 0 )
              {
                v111 = *(int *)(v52 + 552);
                if ( (_DWORD)v111 == -1 )
                  BUG();
                if ( *(_DWORD *)(v54 + 40) == *(_DWORD *)(qword_4F04C68[0] + 776 * v111) && (v151 > 2u || dword_4D04964) )
                {
                  if ( a3 )
                    goto LABEL_144;
                  v137 = v32;
                  v147 = v14;
                  v158 = v4;
                  sub_6849F0(7u, 0xAF1u, (_DWORD *)(v4 + 48), *(_QWORD *)(*(_QWORD *)v4 + 8LL));
                  v4 = v158;
                  v14 = v147;
                  v32 = v137;
                  v51 = qword_4F04C68[0];
                  v31 = dword_4F04BA0;
                  if ( (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 8) & 1) == 0 )
                    goto LABEL_144;
                  v50 = dword_4F04C58;
                  if ( dword_4F04C58 == -1 )
                    goto LABEL_144;
                  v56 = 0;
                  goto LABEL_202;
                }
              }
LABEL_131:
              if ( a3 || (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 8) & 1) == 0 )
                goto LABEL_144;
              v55 = *(_QWORD *)v4;
              v56 = 0;
LABEL_134:
              v57 = 776LL * v50 + v51;
              v58 = *(__int64 **)(v57 + 24);
              v59 = (__int64 *)(v57 + 32);
              if ( !v58 )
                v58 = v59;
              v60 = *v58;
              if ( v60 )
              {
                while ( *(_QWORD *)v60 != v55
                     || *(_BYTE *)(v60 + 80) != 7
                     || *(char *)(*(_QWORD *)(v60 + 88) + 169LL) >= 0 )
                {
                  v60 = *(_QWORD *)(v60 + 16);
                  if ( !v60 )
                    goto LABEL_143;
                }
                v124 = v14;
                v130 = v32;
                v152 = v4;
                sub_6849F0(unk_4F07471, 0x4B3u, (_DWORD *)(v4 + 48), *(_QWORD *)(v55 + 8));
                v4 = v152;
                v31 = dword_4F04BA0;
                v32 = v130;
                v14 = v124;
              }
LABEL_143:
              if ( v56 )
                goto LABEL_26;
LABEL_144:
              v61 = unk_4F04C48;
              if ( (unk_4F04C48 == -1 || (*(_BYTE *)(qword_4F04C68[0] + 776LL * unk_4F04C48 + 10) & 1) == 0)
                && dword_4F04C44 == -1 )
              {
                goto LABEL_43;
              }
              if ( (_DWORD)v32 != 2 )
                goto LABEL_43;
              v62 = *(_BYTE *)(v4 + 80);
              if ( a3 || v62 == 13 || v62 == 3 && *(_BYTE *)(v4 + 104) )
                goto LABEL_43;
              v63 = 8;
              v64 = *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 5) & 0x10;
              if ( !v64 )
                v63 = byte_4F07472[0];
              if ( unk_4F04C48 < dword_4F04C44 )
              {
                v61 = (int)dword_4F04C44;
                v63 = 8;
              }
              if ( (_DWORD)qword_4F077B4 )
              {
                if ( !v64 )
                  v63 = 5;
              }
              else if ( dword_4F077BC && v62 == 18 )
              {
                v63 = 5;
              }
              if ( (_DWORD)v61 == -1 )
                goto LABEL_43;
              v65 = qword_4F04C68[0] + 776 * v61;
              if ( !v65 )
                goto LABEL_43;
              while ( 1 )
              {
                if ( (unsigned __int8)(*(_BYTE *)(v65 + 4) - 8) <= 1u )
                {
                  v87 = **(_QWORD ***)(v65 + 408);
                  if ( v87 )
                  {
                    v88 = *(_QWORD *)v4;
                    while ( 1 )
                    {
                      v89 = (__int64 *)v87[1];
                      v87 = (_QWORD *)*v87;
                      v90 = *v89;
                      if ( !v87 )
                        break;
                      if ( v90 == v88 )
                        goto LABEL_355;
                    }
                    if ( v90 == v88 )
                    {
LABEL_355:
                      v113 = *(_QWORD *)(v90 + 8);
                      v114 = (_DWORD *)(v4 + 48);
                      if ( v63 == 8 )
                      {
                        v169 = v4;
                        sub_6851A0(0x1F0u, v114, v113);
                        v4 = v169;
                        goto LABEL_26;
                      }
                      v138 = v4;
                      v159 = v14;
                      sub_684B10(0x1F1u, v114, v113);
                      v14 = v159;
                      v32 = 2;
                      v4 = v138;
                      v31 = dword_4F04BA0;
LABEL_43:
                      if ( !v12 )
                        goto LABEL_22;
LABEL_44:
                      v33 = 1;
                      while ( 2 )
                      {
                        if ( *(_DWORD *)(v12 + 40) != v10 )
                        {
                          if ( !v11 )
                          {
                            if ( v14 )
                              goto LABEL_23;
                            goto LABEL_27;
                          }
                          goto LABEL_46;
                        }
                        v34 = *(unsigned __int8 *)(v12 + 80);
                        if ( v31[v34] != (_DWORD)v32 )
                          goto LABEL_46;
                        v35 = &v170;
                        if ( !v33 )
                          v35 = 0;
                        if ( (_BYTE)v34 == 13 )
                          goto LABEL_83;
                        if ( HIDWORD(qword_4D0495C) || dword_4F077C4 == 1 || dword_4F077C0 && qword_4F077A8 <= 0x76BFu )
                        {
                          if ( (_BYTE)v34 != 7 )
                          {
LABEL_57:
                            if ( (_BYTE)v34 == 8 && *(_BYTE *)(v4 + 80) == 8 )
                            {
                              if ( (*(_BYTE *)(*(_QWORD *)(v12 + 88) + 145LL) & 2) != 0 )
                                goto LABEL_117;
                              if ( !dword_4F077C0 )
                                goto LABEL_223;
                              if ( !*(_QWORD *)(v12 + 96) && !a3 )
                                goto LABEL_66;
                              if ( (*(_BYTE *)(v4 + 83) & 0x40) == 0 )
                              {
                                v126 = v31;
                                v136 = v32;
                                v146 = v14;
                                v157 = v4;
                                sub_685490(0x4C5u, (FILE *)(v4 + 48), v12);
                                v31 = v126;
                                v32 = v136;
                                v14 = v146;
                                v4 = v157;
                              }
                              *(_BYTE *)(v4 + 83) |= 0x40u;
                              if ( !v35 )
                              {
LABEL_83:
                                v33 = 0;
                                goto LABEL_46;
                              }
LABEL_113:
                              *v35 = v12;
                              v33 = 0;
LABEL_46:
                              v12 = *(_QWORD *)(v12 + 8);
                              if ( !v12 )
                                goto LABEL_22;
                              continue;
                            }
LABEL_58:
                            if ( !HIDWORD(qword_4F077B4) || qword_4F077A8 > 0x9E97u )
                              goto LABEL_84;
                            if ( (_BYTE)v34 == 3 )
                            {
                              v40 = *(_BYTE *)(v4 + 80);
                              v41 = v40;
                              if ( v40 == 3 )
                                goto LABEL_206;
                              if ( dword_4F077C4 != 2 )
                                goto LABEL_62;
                            }
                            else
                            {
                              if ( dword_4F077C4 != 2 )
                                goto LABEL_62;
                              v40 = *(_BYTE *)(v4 + 80);
                              v41 = v40;
                              if ( (unsigned __int8)(v34 - 4) > 2u )
                                goto LABEL_86;
                              if ( v40 == 3 )
                              {
LABEL_206:
                                v123 = v31;
                                v133 = v4;
                                v143 = v32;
                                v154 = v14;
                                v78 = sub_8D2E30(*(_QWORD *)(v12 + 88));
                                v14 = v154;
                                v32 = v143;
                                v4 = v133;
                                v31 = v123;
                                if ( v78 )
                                {
                                  v79 = *(_QWORD *)(v133 + 88);
                                  v134 = v143;
                                  v144 = v154;
                                  v155 = v4;
                                  v80 = sub_8D2E30(v79);
                                  v4 = v155;
                                  v14 = v144;
                                  v32 = v134;
                                  v31 = v123;
                                  if ( v80 )
                                  {
                                    v135 = v155;
                                    v145 = v32;
                                    v156 = v14;
                                    v81 = sub_8D46C0(*(_QWORD *)(v12 + 88));
                                    v82 = sub_8D2310(v81);
                                    v14 = v156;
                                    v32 = v145;
                                    v4 = v135;
                                    v31 = v123;
                                    if ( v82 )
                                    {
                                      v83 = *(_QWORD *)(v135 + 88);
                                      v129 = v145;
                                      v140 = v156;
                                      v150 = (unsigned int *)v4;
                                      v84 = sub_8D46C0(v83);
                                      v85 = sub_8D2310(v84);
                                      v4 = (__int64)v150;
                                      v14 = v140;
                                      v32 = v129;
                                      v31 = v123;
                                      if ( v85 )
                                      {
                                        v86 = sub_729F80(v150[12]);
                                        v4 = (__int64)v150;
                                        v14 = v140;
                                        v32 = v129;
                                        v31 = v123;
                                        if ( v86 )
                                        {
                                          if ( a3 )
                                            goto LABEL_83;
                                          v47 = v150 + 12;
                                          v48 = 256;
                                          v49 = *(_QWORD *)(*(_QWORD *)v150 + 8LL);
                                          goto LABEL_124;
                                        }
                                      }
                                    }
                                  }
                                }
LABEL_84:
                                if ( dword_4F077C4 != 2 )
                                  goto LABEL_62;
                                v40 = *(_BYTE *)(v4 + 80);
                                LOBYTE(v34) = *(_BYTE *)(v12 + 80);
                                v41 = v40;
LABEL_86:
                                v149 = v40 == 23;
                                if ( ((_BYTE)v34 == 23 || v40 == 23) && !dword_4D044B8 )
                                {
LABEL_62:
                                  if ( a3 )
                                    goto LABEL_22;
                                  v36 = *(_BYTE *)(v4 + 80);
                                  if ( v36 == 3 )
                                  {
                                    v37 = *(_BYTE *)(v12 + 80);
                                    if ( v37 != 3 )
                                    {
                                      if ( dword_4F077C4 != 2 )
                                      {
LABEL_66:
                                        v166 = v4;
                                        sub_6851A0(0x65u, (_DWORD *)(v4 + 48), *(_QWORD *)(*(_QWORD *)v4 + 8LL));
                                        v4 = v166;
                                        goto LABEL_26;
                                      }
LABEL_228:
                                      if ( (unsigned __int8)(v37 - 4) > 2u )
                                        goto LABEL_66;
                                    }
                                  }
                                  else
                                  {
LABEL_225:
                                    if ( dword_4F077C4 != 2 || (unsigned __int8)(v36 - 4) > 2u )
                                      goto LABEL_66;
                                    v37 = *(_BYTE *)(v12 + 80);
                                    if ( v37 != 3 )
                                      goto LABEL_228;
                                  }
                                  v91 = *(_QWORD *)(v12 + 88);
                                  v167 = v4;
                                  v92 = sub_8D23B0(v91);
                                  v93 = v167;
                                  v94 = (FILE *)(v167 + 48);
                                  if ( !v92 && (v95 = sub_8D3A70(v91), v93 = v167, v95) )
                                  {
                                    sub_685920(v94, (FILE *)v12, 8u);
                                    v4 = v167;
                                  }
                                  else
                                  {
                                    v168 = v93;
                                    sub_6854C0(0x100u, v94, v12);
                                    v4 = v168;
                                  }
                                  goto LABEL_26;
                                }
                                if ( (*(_DWORD *)(qword_4F04C68[0] + v161 + 4) & 0x200FF) == 0x20009 )
                                  goto LABEL_83;
                                if ( (_BYTE)v34 == 3 )
                                {
                                  if ( *(_BYTE *)(v12 + 104) )
                                    goto LABEL_83;
                                  if ( v40 != 24 )
                                  {
                                    v42 = 3;
                                    goto LABEL_93;
                                  }
                                  v67 = *(_QWORD *)(v4 + 88);
                                  v41 = *(_BYTE *)(v67 + 80);
LABEL_285:
                                  v105 = v34;
                                  v68 = v12;
                                  goto LABEL_273;
                                }
                                if ( v40 == 24 )
                                {
                                  v67 = v4;
                                  goto LABEL_271;
                                }
                                if ( (_BYTE)v34 != 24 )
                                {
LABEL_262:
                                  v42 = v34;
                                  goto LABEL_184;
                                }
                                if ( v40 != 16 )
                                {
                                  v67 = v4;
                                  v68 = v12;
                                  goto LABEL_175;
                                }
                                v67 = **(_QWORD **)(v4 + 88);
                                v41 = *(_BYTE *)(v67 + 80);
                                if ( v41 != 24 )
                                {
                                  v68 = v12;
                                  goto LABEL_175;
                                }
LABEL_271:
                                v67 = *(_QWORD *)(v67 + 88);
                                v41 = *(_BYTE *)(v67 + 80);
                                if ( (_BYTE)v34 != 16 )
                                  goto LABEL_285;
                                v68 = **(_QWORD **)(v12 + 88);
                                v105 = *(_BYTE *)(v68 + 80);
LABEL_273:
                                if ( v105 == 24 )
LABEL_175:
                                  v68 = *(_QWORD *)(v68 + 88);
                                if ( v41 == 3 )
                                {
                                  v69 = *(unsigned __int8 *)(v68 + 80);
                                  if ( (_BYTE)v69 != 3 )
                                  {
                                    v69 = (unsigned int)(v69 - 4);
                                    if ( (unsigned __int8)v69 > 2u )
                                      goto LABEL_262;
                                  }
                                }
                                else if ( (unsigned __int8)(v41 - 4) > 2u
                                       || (v69 = *(unsigned __int8 *)(v68 + 80), (_BYTE)v69 != 3)
                                       && (v69 = (unsigned int)(v69 - 4), (unsigned __int8)v69 > 2u) )
                                {
                                  v42 = v34;
                                  if ( v41 == 19 && *(_BYTE *)(v68 + 80) == 19 )
                                  {
                                    v101 = *(_QWORD *)(*(_QWORD *)(v67 + 88) + 104LL);
                                    v102 = *(_QWORD *)(*(_QWORD *)(v68 + 88) + 104LL);
                                    if ( v101 == v102 )
                                      goto LABEL_83;
                                    if ( !v102 || !v101 )
                                      goto LABEL_262;
                                    if ( dword_4F07588 )
                                    {
                                      v103 = *(_QWORD *)(v101 + 32);
                                      if ( *(_QWORD *)(v102 + 32) == v103 && v103 )
                                        goto LABEL_83;
                                      goto LABEL_262;
                                    }
                                  }
LABEL_184:
                                  if ( dword_4F077BC && v42 == 7 && *(char *)(*(_QWORD *)(v12 + 88) + 171LL) < 0 )
                                  {
                                    if ( a3 )
                                      goto LABEL_83;
                                    v123 = v31;
                                    v47 = (_DWORD *)(v4 + 48);
                                    v48 = 1210;
                                    v129 = v32;
                                    v140 = v14;
                                    v49 = *(_QWORD *)(*(_QWORD *)v4 + 8LL);
                                    v150 = (unsigned int *)v4;
                                    goto LABEL_124;
                                  }
                                  v40 = *(_BYTE *)(v4 + 80);
LABEL_93:
                                  v43 = v40;
                                  v44 = v4;
                                  if ( v40 == 16 )
                                  {
                                    v44 = **(_QWORD **)(v4 + 88);
                                    v43 = *(_BYTE *)(v44 + 80);
                                  }
                                  if ( v43 == 24 )
                                  {
                                    v44 = *(_QWORD *)(v44 + 88);
                                    v43 = *(_BYTE *)(v44 + 80);
                                  }
                                  v45 = v12;
                                  if ( v42 == 16 )
                                  {
                                    v45 = **(_QWORD **)(v12 + 88);
                                    v42 = *(_BYTE *)(v45 + 80);
                                  }
                                  if ( v42 == 24 )
                                    v45 = *(_QWORD *)(v45 + 88);
                                  v128 = 0;
                                  if ( v40 == 24 )
                                    v128 = (*(_BYTE *)(v4 + 96) & 4) != 0;
                                  if ( dword_4D04964 )
                                  {
                                    if ( (unsigned __int8)(v43 - 19) <= 3u )
                                      goto LABEL_62;
                                    v46 = *(_BYTE *)(v45 + 80);
                                    if ( (unsigned __int8)(v46 - 19) <= 3u )
                                      goto LABEL_62;
                                    if ( v46 != 17 )
                                    {
LABEL_107:
                                      v122 = v43 - 4;
                                      if ( (unsigned __int8)(v43 - 4) > 2u )
                                      {
                                        if ( v43 != 3 )
                                        {
                                          if ( (unsigned __int8)(v46 - 4) <= 2u )
                                          {
LABEL_110:
                                            if ( v43 != 19 )
                                            {
                                              if ( !v35 || !v149 )
                                                goto LABEL_83;
                                              goto LABEL_113;
                                            }
                                            goto LABEL_297;
                                          }
LABEL_334:
                                          if ( v46 != 3 )
                                          {
                                            if ( !dword_4F077BC || !v128 )
                                              goto LABEL_297;
                                            goto LABEL_337;
                                          }
LABEL_346:
                                          if ( !*(_BYTE *)(v45 + 104) )
                                          {
LABEL_347:
                                            if ( !dword_4F077BC || !v128 )
                                              goto LABEL_297;
                                            if ( v122 > 2u )
                                            {
LABEL_337:
                                              if ( v43 != 3 )
                                                goto LABEL_297;
                                              goto LABEL_338;
                                            }
                                            goto LABEL_339;
                                          }
LABEL_359:
                                          if ( v43 != 3 )
                                          {
                                            if ( dword_4F077C4 != 2 || v122 > 2u )
                                              goto LABEL_110;
                                            if ( dword_4F077BC && v128 )
                                              goto LABEL_339;
LABEL_297:
                                            if ( dword_4D04964 )
                                            {
                                              if ( (*(_DWORD *)(qword_4F04C68[0] + v161 + 4) & 0x200FF) != 0x20006 )
                                                goto LABEL_62;
                                              v109 = *(_BYTE *)(v12 + 80);
                                              if ( v109 != 16 || (*(_BYTE *)(v12 + 96) & 4) == 0 )
                                                goto LABEL_300;
                                            }
                                            else
                                            {
                                              v109 = *(_BYTE *)(v12 + 80);
                                              if ( v109 != 16 )
                                                goto LABEL_299;
                                              if ( (*(_BYTE *)(v12 + 96) & 4) == 0 )
                                              {
                                                v109 = *(_BYTE *)(v4 + 80);
                                                if ( v109 != 16 )
                                                {
                                                  v127 = v31;
                                                  v139 = v4;
                                                  v148 = v32;
                                                  v160 = v14;
                                                  sub_881DB0(v12);
                                                  v14 = v160;
                                                  v32 = v148;
                                                  v4 = v139;
                                                  v31 = v127;
                                                  v33 = 0;
                                                  goto LABEL_46;
                                                }
LABEL_299:
                                                if ( (*(_DWORD *)(qword_4F04C68[0] + v161 + 4) & 0x200FF) != 0x20006 )
                                                  goto LABEL_62;
LABEL_300:
                                                if ( *(_BYTE *)(v4 + 80) != 16 )
                                                  goto LABEL_62;
                                                if ( (*(_BYTE *)(v4 + 96) & 4) == 0
                                                  || v43 != 2
                                                  || (v110 = *(_QWORD *)(v44 + 88)) == 0
                                                  || *(_BYTE *)(v110 + 173) != 12
                                                  || v109 > 0x14u
                                                  || ((0x120C00uLL >> v109) & 1) == 0 )
                                                {
LABEL_223:
                                                  if ( !a3 )
                                                  {
                                                    v36 = *(_BYTE *)(v4 + 80);
                                                    goto LABEL_225;
                                                  }
LABEL_22:
                                                  if ( v14 )
                                                  {
LABEL_23:
                                                    v23 = 2349;
                                                    if ( *(_BYTE *)(v14 + 80) == 7 )
                                                      v23 = (*(_BYTE *)(*(_QWORD *)(v14 + 88) + 89LL) & 1) == 0
                                                          ? 2349
                                                          : 1348;
                                                    v165 = v4;
                                                    sub_685460(v23, (FILE *)(v4 + 48), v14);
                                                    v4 = v165;
                                                  }
LABEL_26:
                                                  if ( !v11 )
                                                  {
LABEL_27:
                                                    v24 = v170;
                                                    if ( v170 )
                                                    {
                                                      *(_QWORD *)(v4 + 8) = *(_QWORD *)(v170 + 8);
                                                      *(_QWORD *)(v24 + 8) = v4;
                                                      return;
                                                    }
LABEL_69:
                                                    *(_QWORD *)(v4 + 8) = *(_QWORD *)(v163 + 24);
                                                    *(_QWORD *)(v163 + 24) = v4;
                                                    return;
                                                  }
LABEL_114:
                                                  sub_879210((_QWORD *)v4);
                                                  return;
                                                }
                                                *(_BYTE *)(v12 + 84) |= 4u;
LABEL_308:
                                                if ( v35 )
                                                  goto LABEL_113;
                                                v33 = 0;
                                                goto LABEL_46;
                                              }
                                              if ( (*(_DWORD *)(qword_4F04C68[0] + v161 + 4) & 0x200FF) != 0x20006 )
                                                goto LABEL_62;
                                            }
                                            if ( v46 != 2 )
                                              goto LABEL_62;
                                            v112 = *(_QWORD *)(v45 + 88);
                                            if ( !v112 )
                                              goto LABEL_62;
                                            if ( *(_BYTE *)(v112 + 173) != 12 )
                                              goto LABEL_62;
                                            if ( v43 > 0x14u )
                                              goto LABEL_62;
                                            if ( ((0x120C00uLL >> v43) & 1) == 0 )
                                            {
                                              if ( v43 != 2 )
                                                goto LABEL_62;
                                              v115 = *(_QWORD *)(v44 + 88);
                                              if ( !v115 || *(_BYTE *)(v115 + 173) != 12 )
                                                goto LABEL_62;
                                            }
                                            *(_BYTE *)(v4 + 84) |= 4u;
                                            v33 = 0;
                                            goto LABEL_46;
                                          }
LABEL_295:
                                          if ( !dword_4F077BC || !v128 )
                                            goto LABEL_297;
LABEL_338:
                                          if ( !*(_BYTE *)(v44 + 104) )
                                            goto LABEL_297;
LABEL_339:
                                          if ( v46 != 3 )
                                            goto LABEL_297;
                                          goto LABEL_308;
                                        }
                                        if ( !*(_BYTE *)(v44 + 104) )
                                        {
                                          if ( (unsigned __int8)(v46 - 4) > 2u )
                                            goto LABEL_334;
                                          goto LABEL_295;
                                        }
                                      }
                                      if ( v46 == 3 )
                                        goto LABEL_346;
                                      if ( dword_4F077C4 == 2 && (unsigned __int8)(v46 - 4) <= 2u )
                                        goto LABEL_359;
                                      if ( v46 == 19 )
                                        goto LABEL_347;
                                      if ( (_BYTE)v34 == 23 )
                                      {
                                        v33 = 0;
                                        goto LABEL_46;
                                      }
                                      goto LABEL_308;
                                    }
                                    v117 = v4;
                                    v119 = v14;
                                    v116 = v31;
                                    v118 = v32;
                                    v121 = v44;
                                    v104 = sub_8780F0(v45);
                                    v14 = v119;
                                    v4 = v117;
                                    if ( v104 )
                                      goto LABEL_62;
                                    v44 = v121;
                                    v31 = v116;
                                    v32 = v118;
                                    v43 = *(_BYTE *)(v121 + 80);
                                  }
                                  if ( v43 == 22 || (v46 = *(_BYTE *)(v45 + 80), v46 == 22) )
                                  {
                                    if ( !dword_4F077BC || (_DWORD)qword_4F077B4 )
                                      goto LABEL_62;
                                    v46 = *(_BYTE *)(v45 + 80);
                                  }
                                  goto LABEL_107;
                                }
                                v70 = *(_QWORD *)(v67 + 88);
                                v71 = *(_QWORD *)(v68 + 88);
                                if ( !v70 || !v71 )
                                  goto LABEL_262;
                                if ( v70 == v71 )
                                  goto LABEL_83;
                                v120 = v31;
                                v125 = v4;
                                v131 = v32;
                                v141 = v14;
                                v72 = sub_8D97D0(v70, v71, 0, v69, v32);
                                v14 = v141;
                                v32 = v131;
                                v4 = v125;
                                v31 = v120;
                                if ( v72 )
                                  goto LABEL_83;
                                v42 = *(_BYTE *)(v12 + 80);
                                goto LABEL_184;
                              }
                            }
                            if ( (unsigned __int8)(v40 - 4) <= 2u )
                              goto LABEL_206;
                            goto LABEL_86;
                          }
                          v39 = *(_QWORD *)(v12 + 88);
                          v38 = *(_BYTE *)(v4 + 80);
                          if ( *(char *)(v39 + 169) < 0 )
                          {
                            if ( v38 != 7 || (v100 = *(_QWORD *)(v4 + 88)) == 0 || *(char *)(v100 + 169) >= 0 )
                            {
                              if ( a3 )
                                goto LABEL_83;
                              v123 = v31;
                              v47 = (_DWORD *)(v4 + 48);
                              v48 = 460;
                              v129 = v32;
                              v140 = v14;
                              v49 = *(_QWORD *)(*(_QWORD *)v4 + 8LL);
                              v150 = (unsigned int *)v4;
LABEL_124:
                              sub_684B10(v48, v47, v49);
                              v4 = (__int64)v150;
                              v33 = 0;
                              v14 = v140;
                              v32 = v129;
                              v31 = v123;
                              goto LABEL_46;
                            }
LABEL_74:
                            if ( (*(_QWORD *)(v39 + 168) & 0x4000000000008000LL) == 0x4000000000008000LL )
                            {
LABEL_117:
                              if ( v35 )
                                *v35 = v12;
                              *(_BYTE *)(v4 + 83) |= 0x40u;
                              v33 = 0;
                              goto LABEL_46;
                            }
                            goto LABEL_58;
                          }
                        }
                        else
                        {
                          if ( (_BYTE)v34 != 7 )
                            goto LABEL_57;
                          v38 = *(_BYTE *)(v4 + 80);
                        }
                        break;
                      }
                      if ( v38 != 7 )
                        goto LABEL_58;
                      v39 = *(_QWORD *)(v12 + 88);
                      goto LABEL_74;
                    }
                  }
                  v63 = byte_4F07472[0];
                }
                v66 = *(int *)(v65 + 552);
                if ( (_DWORD)v66 != -1 )
                {
                  v65 = qword_4F04C68[0] + 776 * v66;
                  if ( v65 )
                    continue;
                }
                goto LABEL_43;
              }
            }
          }
          else
          {
            if ( v53 != 15 )
              goto LABEL_128;
            if ( (*(_BYTE *)(v52 - 768) & 8) == 0 )
              goto LABEL_128;
            v96 = v52 - 776;
            if ( (*(_BYTE *)(v52 - 771) & 0x40) != 0 )
              goto LABEL_128;
          }
          v97 = *(_QWORD ***)(v96 + 24);
          v98 = (_QWORD **)(v96 + 32);
          if ( !v97 )
            v97 = v98;
          v99 = *v97;
          if ( *v97 )
          {
            while ( v163 != *v99 )
            {
              v99 = (_QWORD *)v99[2];
              if ( !v99 )
                goto LABEL_128;
            }
            if ( a3 )
              goto LABEL_26;
            v75 = (_DWORD *)(v4 + 48);
            v76 = 779;
            v132 = v32;
            v142 = v14;
            v77 = *(_QWORD *)(*(_QWORD *)v4 + 8LL);
            v153 = v4;
LABEL_199:
            sub_6851A0(v76, v75, v77);
            v4 = v153;
            v14 = v142;
            v32 = v132;
            v51 = qword_4F04C68[0];
            v31 = dword_4F04BA0;
            if ( (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 8) & 1) == 0 )
              goto LABEL_26;
            v50 = dword_4F04C58;
            if ( dword_4F04C58 == -1 )
              goto LABEL_26;
            v56 = 1;
LABEL_202:
            v55 = *(_QWORD *)v4;
            goto LABEL_134;
          }
        }
LABEL_128:
        if ( (*(_BYTE *)(v52 + 5) & 0x20) == 0 )
          goto LABEL_131;
        goto LABEL_129;
      }
      v26 = qword_4F04C68[0] + v25 - 776;
      v27 = qword_4F04C68[0] + v25 - 1552;
      v28 = 0;
      v29 = 0;
      v30 = v27 - 776LL * (unsigned int)(dword_4F04C64 - 1 - a2);
      do
      {
        if ( v12 )
        {
          while ( 1 )
          {
            if ( *(_DWORD *)(v12 + 40) != v10 )
              goto LABEL_37;
            v28 = v12;
            v29 = 1;
            if ( !*(_QWORD *)(v12 + 8) )
              break;
            v12 = *(_QWORD *)(v12 + 8);
          }
          v12 = 0;
        }
LABEL_37:
        v26 -= 776;
        v10 = *(_DWORD *)(v26 + 776);
      }
      while ( v26 != v30 );
      if ( v29 )
        v170 = v28;
      v11 = 0;
    }
    else
    {
      v10 = *(_DWORD *)v8;
      v11 = 1;
      v12 = *(_QWORD *)(v5 + 32);
      if ( a2 == dword_4F04C64 )
        goto LABEL_8;
    }
    v13 = *(_BYTE *)(v4 + 80);
    v14 = 0;
    goto LABEL_42;
  }
}
