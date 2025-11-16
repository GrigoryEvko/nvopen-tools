// Function: sub_115ACB0
// Address: 0x115acb0
//
unsigned __int8 *__fastcall sub_115ACB0(const __m128i *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v7; // r15
  __int64 v8; // r14
  bool v9; // al
  unsigned __int8 *v10; // r9
  __int64 v12; // rcx
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rax
  _BYTE *v18; // rdx
  char v19; // al
  int v20; // ebx
  int v21; // eax
  char v22; // dl
  int v23; // ebx
  __int64 v24; // rax
  _BYTE *v25; // rdx
  __int64 v26; // rax
  unsigned int **v27; // rdi
  _BYTE *v28; // rsi
  int v29; // edi
  _BYTE *v30; // rdx
  __int64 v31; // rbx
  __int64 v32; // rax
  _BYTE *v33; // rcx
  __int64 v34; // rax
  _BYTE *v35; // rdx
  __int64 v36; // rdi
  int v37; // ebx
  unsigned int **v38; // rdi
  __int64 v39; // rax
  __int64 v40; // rax
  __int64 v41; // rax
  unsigned int **v42; // r14
  __int64 v43; // rax
  __int64 v44; // r15
  __int64 v45; // r14
  __int64 v46; // rax
  __int64 v47; // rax
  _BYTE *v48; // rdx
  __int64 v49; // rax
  __int64 v50; // rax
  _BYTE *v51; // rsi
  int v52; // edi
  __int64 v53; // rax
  __int64 v54; // rax
  __int64 v55; // rax
  char v56; // al
  __int64 v57; // rax
  __int64 v58; // rax
  unsigned int **v59; // r13
  int v60; // eax
  _BYTE *v61; // rdx
  _BYTE *v62; // rsi
  __int64 v63; // rcx
  __int64 v64; // rax
  __int64 v65; // rax
  __int64 v66; // rax
  __int64 v67; // rdi
  __int64 v68; // r15
  int v69; // eax
  __int64 v70; // rdi
  unsigned __int8 *v71; // rax
  bool v72; // zf
  __int64 v73; // r14
  __int64 v74; // r15
  int v75; // eax
  _BYTE *v76; // rcx
  _BYTE *v77; // rdx
  __int64 v78; // rdi
  __int64 v79; // r8
  __int64 v80; // rdx
  unsigned int **v81; // r13
  __int64 v82; // rax
  _BYTE *v83; // rdx
  __int64 v84; // rax
  __int64 v85; // rdx
  __int64 v86; // rcx
  __int64 v87; // r8
  bool v88; // al
  __int64 v89; // rsi
  __int64 v90; // rdx
  __int64 v91; // rcx
  __int64 v92; // r8
  _BYTE *v93; // rdx
  __int64 v94; // rcx
  unsigned int **v95; // rdi
  __int64 v96; // rax
  __int64 v97; // rax
  __int64 v98; // rax
  char v99; // al
  unsigned int **v100; // r14
  __int64 v101; // r14
  __int64 v102; // rax
  _BYTE *v103; // rax
  char v104; // al
  unsigned int **v105; // r14
  __int64 v106; // r14
  __int64 v107; // rax
  __int64 v108; // rdx
  __int64 v109; // rax
  __int64 v110; // r15
  __int64 v111; // r15
  __int64 v112; // rdx
  unsigned int v113; // esi
  int v114; // eax
  __int64 v115; // rcx
  __int64 v116; // rax
  __int64 v117; // rdx
  __int64 v118; // rcx
  __int64 v119; // r8
  unsigned int **v120; // r13
  __int64 v121; // rax
  __int64 v122; // rax
  _BYTE *v123; // rax
  char v124; // al
  unsigned int **v125; // r14
  __int64 v126; // r14
  __int64 v127; // rdx
  __int64 v128; // rcx
  __int64 v129; // rax
  unsigned int **v130; // r14
  __int64 v131; // r14
  bool v132; // al
  __int64 v133; // [rsp+8h] [rbp-D8h]
  _BYTE *v134; // [rsp+10h] [rbp-D0h]
  _BYTE *v135; // [rsp+10h] [rbp-D0h]
  char v136; // [rsp+18h] [rbp-C8h]
  _BYTE *v137; // [rsp+18h] [rbp-C8h]
  _BYTE *v138; // [rsp+18h] [rbp-C8h]
  _BYTE *v139; // [rsp+18h] [rbp-C8h]
  __int64 v140; // [rsp+18h] [rbp-C8h]
  int v141; // [rsp+18h] [rbp-C8h]
  unsigned __int8 *v142; // [rsp+18h] [rbp-C8h]
  _BYTE *v143; // [rsp+18h] [rbp-C8h]
  _BYTE *v144; // [rsp+18h] [rbp-C8h]
  __int64 v145; // [rsp+18h] [rbp-C8h]
  __int64 v146; // [rsp+18h] [rbp-C8h]
  __int64 v147; // [rsp+18h] [rbp-C8h]
  __int64 v148; // [rsp+18h] [rbp-C8h]
  __int64 v149; // [rsp+18h] [rbp-C8h]
  __int64 v150; // [rsp+18h] [rbp-C8h]
  _BYTE *v151; // [rsp+18h] [rbp-C8h]
  _BYTE *v152; // [rsp+28h] [rbp-B8h] BYREF
  _BYTE *v153; // [rsp+30h] [rbp-B0h] BYREF
  _BYTE *v154; // [rsp+38h] [rbp-A8h] BYREF
  __int64 v155; // [rsp+40h] [rbp-A0h]
  __int64 v156; // [rsp+48h] [rbp-98h]
  __int64 v157; // [rsp+50h] [rbp-90h] BYREF
  int v158; // [rsp+58h] [rbp-88h]
  _BYTE **v159; // [rsp+60h] [rbp-80h]
  __int16 v160; // [rsp+70h] [rbp-70h]
  __int64 v161; // [rsp+80h] [rbp-60h] BYREF
  _QWORD *v162; // [rsp+88h] [rbp-58h] BYREF
  _BYTE **v163; // [rsp+90h] [rbp-50h]
  _QWORD *v164; // [rsp+98h] [rbp-48h]
  __int64 v165; // [rsp+A0h] [rbp-40h]

  v7 = *(_QWORD *)(a2 - 32);
  v8 = *(_QWORD *)(a2 - 64);
  if ( *(_BYTE *)v7 > 0x15u )
    goto LABEL_3;
  if ( !(unsigned __int8)sub_AD7E60(*(_QWORD *)(a2 - 32), a2, a3, a4, a5) )
    goto LABEL_3;
  v19 = sub_920620(v8);
  if ( !v8 )
    goto LABEL_3;
  if ( !v19 )
    goto LABEL_3;
  if ( (*(_BYTE *)(v8 + 1) & 2) == 0 )
    goto LABEL_3;
  v136 = *(_BYTE *)v8;
  if ( (unsigned __int8)(*(_BYTE *)v8 - 42) > 0x11u )
    goto LABEL_3;
  v20 = sub_B45210(v8);
  v21 = sub_B45210(a2);
  v22 = v136;
  v23 = v21 & v20;
  v24 = *(_QWORD *)(v8 + 16);
  if ( v24 && !*(_QWORD *)(v24 + 8) )
  {
    if ( v136 != 50 )
      goto LABEL_33;
    v83 = *(_BYTE **)(v8 - 64);
    if ( *v83 > 0x15u || !*(_QWORD *)(v8 - 32) )
      goto LABEL_124;
    v115 = a1[5].m128i_i64[1];
    v152 = *(_BYTE **)(v8 - 32);
    v116 = sub_96E6C0(0x12u, v7, v83, v115);
    if ( v116 )
    {
      v28 = (_BYTE *)v116;
      if ( sub_AD7F90(v116, v7, v117, v118, v119) )
      {
        LOWORD(v165) = 257;
        v30 = v152;
        v29 = 21;
        goto LABEL_59;
      }
    }
    v22 = *(_BYTE *)v8;
  }
  if ( v22 == 50 )
  {
    v83 = *(_BYTE **)(v8 - 64);
    if ( v83 )
    {
LABEL_124:
      v152 = v83;
      if ( **(_BYTE **)(v8 - 32) <= 0x15u )
      {
        v143 = *(_BYTE **)(v8 - 32);
        v84 = sub_96E6C0(0x15u, v7, v143, a1[5].m128i_i64[1]);
        v87 = (__int64)v143;
        if ( v84 )
        {
          v135 = v143;
          v144 = (_BYTE *)v84;
          v88 = sub_AD7F90(v84, v7, v85, v86, v87);
          v30 = v144;
          v87 = (__int64)v135;
          if ( v88 )
          {
            LOWORD(v165) = 257;
            v28 = v152;
            v29 = 18;
            goto LABEL_59;
          }
        }
        v89 = v87;
        v90 = sub_96E6C0(0x15u, v87, (_BYTE *)v7, a1[5].m128i_i64[1]);
        if ( v90 )
        {
          v24 = *(_QWORD *)(v8 + 16);
          if ( !v24 )
            goto LABEL_3;
          if ( *(_QWORD *)(v24 + 8) )
            goto LABEL_130;
          v151 = (_BYTE *)v90;
          v132 = sub_AD7F90(v90, v89, v90, v91, v92);
          v30 = v151;
          if ( v132 )
          {
            LOWORD(v165) = 257;
            v28 = v152;
            v29 = 21;
            goto LABEL_59;
          }
        }
      }
    }
  }
  v24 = *(_QWORD *)(v8 + 16);
LABEL_33:
  if ( v24 )
  {
    if ( *(_QWORD *)(v24 + 8) )
      goto LABEL_130;
    if ( *(_BYTE *)v8 != 43 )
      goto LABEL_130;
    if ( !*(_QWORD *)(v8 - 64) )
      goto LABEL_130;
    v152 = *(_BYTE **)(v8 - 64);
    v25 = *(_BYTE **)(v8 - 32);
    if ( *v25 > 0x15u )
      goto LABEL_130;
    v26 = sub_96E6C0(0x12u, v7, v25, a1[5].m128i_i64[1]);
    if ( v26 )
    {
      LODWORD(v157) = v23;
      v27 = (unsigned int **)a1[2].m128i_i64[0];
      BYTE4(v157) = 1;
      v137 = (_BYTE *)v26;
      LOWORD(v165) = 257;
      v28 = (_BYTE *)sub_A826E0(v27, v152, (_BYTE *)v7, v157, (__int64)&v161, 0);
      v29 = 14;
      LOWORD(v165) = 257;
      v30 = v137;
LABEL_59:
      v140 = sub_B504D0(v29, (__int64)v28, (__int64)v30, (__int64)&v161, 0, 0);
      sub_B45150(v140, v23);
      return (unsigned __int8 *)v140;
    }
    v24 = *(_QWORD *)(v8 + 16);
    if ( v24 )
    {
LABEL_130:
      if ( !*(_QWORD *)(v24 + 8) && *(_BYTE *)v8 == 45 )
      {
        v93 = *(_BYTE **)(v8 - 64);
        if ( *v93 <= 0x15u )
        {
          if ( *(_QWORD *)(v8 - 32) )
          {
            v94 = a1[5].m128i_i64[1];
            v152 = *(_BYTE **)(v8 - 32);
            v145 = sub_96E6C0(0x12u, v7, v93, v94);
            if ( v145 )
            {
              LODWORD(v157) = v23;
              v95 = (unsigned int **)a1[2].m128i_i64[0];
              BYTE4(v157) = 1;
              LOWORD(v165) = 257;
              v96 = sub_A826E0(v95, v152, (_BYTE *)v7, v157, (__int64)&v161, 0);
              LOWORD(v165) = 257;
              v30 = (_BYTE *)v96;
              v29 = 16;
              v28 = (_BYTE *)v145;
              goto LABEL_59;
            }
          }
        }
      }
    }
  }
LABEL_3:
  if ( *(_BYTE *)a2 != 47 )
    goto LABEL_4;
  v31 = *(_QWORD *)(a2 - 64);
  if ( v31 )
  {
    if ( (unsigned __int8)sub_920620(*(_QWORD *)(a2 - 64))
      && (*(_BYTE *)(v31 + 1) & 2) != 0
      && (v32 = *(_QWORD *)(v31 + 16)) != 0
      && !*(_QWORD *)(v32 + 8)
      && *(_BYTE *)v31 == 50 )
    {
      v33 = *(_BYTE **)(a2 - 32);
      if ( *(_QWORD *)(v31 - 64) )
      {
        v152 = *(_BYTE **)(v31 - 64);
        v35 = v33;
        if ( *(_QWORD *)(v31 - 32) )
        {
          v153 = *(_BYTE **)(v31 - 32);
          if ( v33 )
          {
            v154 = v33;
            goto LABEL_55;
          }
LABEL_163:
          sub_920620(0);
          goto LABEL_4;
        }
      }
    }
    else
    {
      v33 = *(_BYTE **)(a2 - 32);
    }
  }
  else
  {
    sub_920620(0);
    v33 = *(_BYTE **)(a2 - 32);
  }
  if ( !v33 )
    goto LABEL_163;
  v138 = v33;
  if ( !(unsigned __int8)sub_920620((__int64)v33) )
    goto LABEL_4;
  if ( (v138[1] & 2) == 0 )
    goto LABEL_4;
  v34 = *((_QWORD *)v138 + 2);
  if ( !v34 )
    goto LABEL_4;
  if ( *(_QWORD *)(v34 + 8) )
    goto LABEL_4;
  if ( *v138 != 50 )
    goto LABEL_4;
  if ( !*((_QWORD *)v138 - 8) )
    goto LABEL_4;
  v152 = (_BYTE *)*((_QWORD *)v138 - 8);
  if ( !*((_QWORD *)v138 - 4) )
    goto LABEL_4;
  v35 = *(_BYTE **)(a2 - 64);
  v153 = (_BYTE *)*((_QWORD *)v138 - 4);
  if ( !v35 )
    goto LABEL_4;
  v154 = v35;
LABEL_55:
  v36 = v8;
  v139 = v35;
  if ( v35 == (_BYTE *)v8 )
    v36 = v7;
  v37 = sub_B45210(v36);
  v23 = sub_B45210(a2) & v37;
  if ( (v23 & 1) != 0 )
  {
    v38 = (unsigned int **)a1[2].m128i_i64[0];
    LODWORD(v157) = v23;
    LOWORD(v165) = 257;
    BYTE4(v157) = 1;
    v39 = sub_A826E0(v38, v152, v139, v157, (__int64)&v161, 0);
    LOWORD(v165) = 257;
    v30 = v153;
    v28 = (_BYTE *)v39;
    v29 = 21;
    goto LABEL_59;
  }
LABEL_4:
  if ( sub_B451C0(a2) )
  {
    LODWORD(v157) = 335;
    v158 = 0;
    v159 = &v152;
    v40 = *(_QWORD *)(v8 + 16);
    if ( v40 )
    {
      if ( !*(_QWORD *)(v40 + 8) )
      {
        if ( (unsigned __int8)sub_10E25C0((__int64)&v157, v8) )
        {
          LODWORD(v161) = 335;
          LODWORD(v162) = 0;
          v163 = &v153;
          v41 = *(_QWORD *)(v7 + 16);
          if ( v41 )
          {
            if ( !*(_QWORD *)(v41 + 8) && (unsigned __int8)sub_10E25C0((__int64)&v161, v7) )
            {
              v42 = (unsigned int **)a1[2].m128i_i64[0];
              LOWORD(v165) = 257;
              LODWORD(v157) = sub_B45210(a2);
              BYTE4(v157) = 1;
              v43 = sub_A826E0(v42, v152, v153, v157, (__int64)&v161, 0);
              v44 = a1[2].m128i_i64[0];
              v45 = v43;
              LOWORD(v165) = 257;
              LODWORD(v157) = sub_B45210(a2);
              BYTE4(v157) = 1;
              v46 = sub_B33BC0(v44, 0x14Fu, v45, v157, (__int64)&v161);
              return sub_F162A0((__int64)a1, a2, v46);
            }
          }
        }
      }
    }
  }
  if ( sub_B451E0(a2) )
  {
    v162 = &v153;
    v161 = 0x3FF0000000000000LL;
    if ( *(_BYTE *)v8 == 50 )
    {
      if ( (unsigned __int8)sub_1009690((double *)&v161, *(_QWORD *)(v8 - 64)) )
      {
        v47 = *(_QWORD *)(v8 - 32);
        if ( v47 )
        {
          *v162 = v47;
          v48 = v153;
          if ( *v153 == 85 )
          {
            v49 = *((_QWORD *)v153 - 4);
            if ( v49 )
            {
              if ( !*(_BYTE *)v49 && *(_QWORD *)(v49 + 24) == *((_QWORD *)v153 + 10) && *(_DWORD *)(v49 + 36) == 335 )
              {
                v50 = *(_QWORD *)&v153[-32 * (*((_DWORD *)v153 + 1) & 0x7FFFFFF)];
                if ( v50 )
                {
                  v152 = *(_BYTE **)&v153[-32 * (*((_DWORD *)v153 + 1) & 0x7FFFFFF)];
                  if ( v7 == v50 )
                  {
                    v51 = (_BYTE *)v7;
                    LOWORD(v165) = 257;
                    v52 = 21;
LABEL_122:
                    v142 = (unsigned __int8 *)sub_B504D0(v52, (__int64)v51, (__int64)v48, (__int64)&v161, 0, 0);
                    sub_B45260(v142, a2, 1);
                    return v142;
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  if ( sub_B451E0(a2) )
  {
    v162 = &v153;
    v161 = 0x3FF0000000000000LL;
    if ( *(_BYTE *)v7 == 50 )
    {
      if ( (unsigned __int8)sub_1009690((double *)&v161, *(_QWORD *)(v7 - 64)) )
      {
        v53 = *(_QWORD *)(v7 - 32);
        if ( v53 )
        {
          *v162 = v53;
          v48 = v153;
          if ( *v153 == 85 )
          {
            v54 = *((_QWORD *)v153 - 4);
            if ( v54 )
            {
              if ( !*(_BYTE *)v54 && *(_QWORD *)(v54 + 24) == *((_QWORD *)v153 + 10) && *(_DWORD *)(v54 + 36) == 335 )
              {
                v55 = *(_QWORD *)&v153[-32 * (*((_DWORD *)v153 + 1) & 0x7FFFFFF)];
                if ( v55 )
                {
                  v152 = *(_BYTE **)&v153[-32 * (*((_DWORD *)v153 + 1) & 0x7FFFFFF)];
                  if ( v8 == v55 )
                  {
                    v51 = (_BYTE *)v8;
                    LOWORD(v165) = 257;
                    v52 = 21;
                    goto LABEL_122;
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  if ( sub_B451C0(a2) )
  {
    v9 = sub_B451E0(a2);
    if ( v7 == v8 && v9 && (unsigned __int8)sub_BD3610(v8, 2) )
    {
      LODWORD(v162) = 335;
      v161 = (__int64)&v152;
      LODWORD(v163) = 0;
      v164 = &v153;
      if ( *(_BYTE *)v8 == 50 )
      {
        if ( *(_QWORD *)(v8 - 64) )
        {
          v152 = *(_BYTE **)(v8 - 64);
          if ( (unsigned __int8)sub_10E25C0((__int64)&v162, *(_QWORD *)(v8 - 32)) )
          {
            v120 = (unsigned int **)a1[2].m128i_i64[0];
            LOWORD(v165) = 257;
            sub_10A0170((__int64)&v157, a2);
            v121 = sub_A826E0(v120, v152, v152, v157, (__int64)&v161, 0);
            LOWORD(v165) = 257;
            v48 = v153;
            v51 = (_BYTE *)v121;
            v52 = 21;
            goto LABEL_122;
          }
        }
      }
      LODWORD(v161) = 335;
      LODWORD(v162) = 0;
      v163 = &v153;
      v164 = &v152;
      if ( *(_BYTE *)v8 == 50 )
      {
        if ( (unsigned __int8)sub_10E25C0((__int64)&v161, *(_QWORD *)(v8 - 64)) )
        {
          v80 = *(_QWORD *)(v8 - 32);
          if ( v80 )
          {
            *v164 = v80;
            v81 = (unsigned int **)a1[2].m128i_i64[0];
            LOWORD(v165) = 257;
            sub_10A0170((__int64)&v157, a2);
            v82 = sub_A826E0(v81, v152, v152, v157, (__int64)&v161, 0);
            v51 = v153;
            LOWORD(v165) = 257;
            v48 = (_BYTE *)v82;
            v52 = 21;
            goto LABEL_122;
          }
        }
      }
    }
  }
  if ( *(_BYTE *)a2 == 47 )
  {
    v12 = *(_QWORD *)(a2 - 64);
    v13 = *(_QWORD *)(a2 - 32);
    v14 = *(_QWORD *)(v12 + 16);
    if ( v14 )
    {
      if ( !*(_QWORD *)(v14 + 8) && *(_BYTE *)v12 == 85 )
      {
        v65 = *(_QWORD *)(v12 - 32);
        if ( v65 )
        {
          if ( !*(_BYTE *)v65 && *(_QWORD *)(v65 + 24) == *(_QWORD *)(v12 + 80) && *(_DWORD *)(v65 + 36) == 284 )
          {
            v66 = *(_QWORD *)(v12 - 32LL * (*(_DWORD *)(v12 + 4) & 0x7FFFFFF));
            if ( v66 )
            {
              v152 = *(_BYTE **)(v12 - 32LL * (*(_DWORD *)(v12 + 4) & 0x7FFFFFF));
              if ( *(_BYTE *)v12 == 85 )
              {
                v67 = *(_DWORD *)(v12 + 4) & 0x7FFFFFF;
                if ( *(_QWORD *)(v12 + 32 * (1 - v67)) )
                {
                  v153 = *(_BYTE **)(v12 + 32 * (1 - v67));
                  if ( v13 == v66 )
                    goto LABEL_112;
                }
              }
            }
          }
        }
      }
    }
    v15 = *(_QWORD *)(v13 + 16);
    if ( v15 )
    {
      if ( !*(_QWORD *)(v15 + 8) && *(_BYTE *)v13 == 85 )
      {
        v16 = *(_QWORD *)(v13 - 32);
        if ( v16 )
        {
          if ( !*(_BYTE *)v16 && *(_QWORD *)(v16 + 24) == *(_QWORD *)(v13 + 80) && *(_DWORD *)(v16 + 36) == 284 )
          {
            v17 = *(_QWORD *)(v13 - 32LL * (*(_DWORD *)(v13 + 4) & 0x7FFFFFF));
            if ( v17 )
            {
              v152 = *(_BYTE **)(v13 - 32LL * (*(_DWORD *)(v13 + 4) & 0x7FFFFFF));
              if ( *(_BYTE *)v13 == 85 )
              {
                v18 = *(_BYTE **)(v13 + 32 * (1LL - (*(_DWORD *)(v13 + 4) & 0x7FFFFFF)));
                if ( v18 )
                {
                  v153 = v18;
                  if ( v12 == v17 )
                  {
LABEL_112:
                    v68 = a1[2].m128i_i64[0];
                    v160 = 257;
                    v69 = sub_B45210(a2);
                    v70 = *(_QWORD *)(a2 + 8);
                    LODWORD(v155) = v69;
                    v71 = sub_AD8DD0(v70, 1.0);
                    BYTE4(v155) = 1;
                    v72 = *(_BYTE *)(v68 + 108) == 0;
                    v156 = v155;
                    if ( v72 )
                    {
                      v133 = (__int64)v71;
                      v141 = v155;
                      v134 = v153;
                      v73 = (*(__int64 (__fastcall **)(_QWORD, __int64, _BYTE *, unsigned __int8 *, _QWORD))(**(_QWORD **)(v68 + 80) + 40LL))(
                              *(_QWORD *)(v68 + 80),
                              14,
                              v153,
                              v71,
                              (unsigned int)v155);
                      if ( !v73 )
                      {
                        LOWORD(v165) = 257;
                        v107 = sub_B504D0(14, (__int64)v134, v133, (__int64)&v161, 0, 0);
                        v108 = *(_QWORD *)(v68 + 96);
                        v73 = v107;
                        if ( v108 )
                          sub_B99FD0(v107, 3u, v108);
                        sub_B45150(v73, v141);
                        (*(void (__fastcall **)(_QWORD, __int64, __int64 *, _QWORD, _QWORD))(**(_QWORD **)(v68 + 88)
                                                                                           + 16LL))(
                          *(_QWORD *)(v68 + 88),
                          v73,
                          &v157,
                          *(_QWORD *)(v68 + 56),
                          *(_QWORD *)(v68 + 64));
                        v109 = *(_QWORD *)v68;
                        v110 = 16LL * *(unsigned int *)(v68 + 8);
                        v148 = v109 + v110;
                        if ( v109 != v109 + v110 )
                        {
                          v111 = v109;
                          do
                          {
                            v112 = *(_QWORD *)(v111 + 8);
                            v113 = *(_DWORD *)v111;
                            v111 += 16;
                            sub_B99FD0(v73, v113, v112);
                          }
                          while ( v148 != v111 );
                        }
                      }
                    }
                    else
                    {
                      v73 = sub_B35400(v68, 0x66u, (__int64)v153, (__int64)v71, v155, (__int64)&v157, 0, 0, 0);
                    }
                    v74 = a1[2].m128i_i64[0];
                    LOWORD(v165) = 257;
                    v75 = sub_B45210(a2);
                    v76 = (_BYTE *)v73;
                    LODWORD(v157) = v75;
                    v77 = v152;
                    v78 = v74;
                    BYTE4(v157) = 1;
                    v79 = v157;
                    goto LABEL_115;
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  v10 = sub_115A4C0(a1, (unsigned __int8 *)a2);
  if ( !v10 )
  {
    v56 = sub_B44680(a2);
    v10 = 0;
    if ( !v56 )
      goto LABEL_94;
    if ( *(_BYTE *)v8 == 85 )
    {
      v97 = *(_QWORD *)(v8 - 32);
      if ( v97 )
      {
        if ( !*(_BYTE *)v97 && *(_QWORD *)(v97 + 24) == *(_QWORD *)(v8 + 80) && *(_DWORD *)(v97 + 36) == 284 )
        {
          v127 = *(_QWORD *)(v8 - 32LL * (*(_DWORD *)(v8 + 4) & 0x7FFFFFF));
          if ( v127 )
          {
            v152 = *(_BYTE **)(v8 - 32LL * (*(_DWORD *)(v8 + 4) & 0x7FFFFFF));
            if ( *(_BYTE *)v8 != 85 )
              goto LABEL_92;
            v128 = *(_DWORD *)(v8 + 4) & 0x7FFFFFF;
            if ( *(_QWORD *)(v8 + 32 * (1 - v128)) )
            {
              v153 = *(_BYTE **)(v8 + 32 * (1 - v128));
              if ( *(_BYTE *)v7 == 85 )
              {
                v129 = *(_QWORD *)(v7 - 32);
                if ( v129 )
                {
                  if ( !*(_BYTE *)v129
                    && *(_QWORD *)(v129 + 24) == *(_QWORD *)(v7 + 80)
                    && *(_DWORD *)(v129 + 36) == 284
                    && v127 == *(_QWORD *)(v7 - 32LL * (*(_DWORD *)(v7 + 4) & 0x7FFFFFF))
                    && *(_QWORD *)(v7 + 32 * (1LL - (*(_DWORD *)(v7 + 4) & 0x7FFFFFF))) )
                  {
                    v130 = (unsigned int **)a1[2].m128i_i64[0];
                    v154 = *(_BYTE **)(v7 + 32 * (1LL - (*(_DWORD *)(v7 + 4) & 0x7FFFFFF)));
                    LOWORD(v165) = 257;
                    sub_10A0170((__int64)&v157, a2);
                    v131 = sub_92A220(v130, v153, v154, v157, (__int64)&v161, 0);
                    v150 = a1[2].m128i_i64[0];
                    LOWORD(v165) = 257;
                    sub_10A0170((__int64)&v157, a2);
                    v76 = (_BYTE *)v131;
                    v79 = v157;
                    v77 = v152;
                    v78 = v150;
                    goto LABEL_115;
                  }
                }
              }
              if ( *(_BYTE *)v8 != 85 )
                goto LABEL_92;
            }
            v97 = *(_QWORD *)(v8 - 32);
          }
        }
        if ( v97 )
        {
          if ( !*(_BYTE *)v97 && *(_QWORD *)(v97 + 24) == *(_QWORD *)(v8 + 80) && *(_DWORD *)(v97 + 36) == 284 )
          {
            if ( *(_QWORD *)(v8 - 32LL * (*(_DWORD *)(v8 + 4) & 0x7FFFFFF)) )
            {
              v152 = *(_BYTE **)(v8 - 32LL * (*(_DWORD *)(v8 + 4) & 0x7FFFFFF));
              if ( *(_BYTE *)v8 == 85 )
              {
                v98 = *(_QWORD *)(v8 + 32 * (1LL - (*(_DWORD *)(v8 + 4) & 0x7FFFFFF)));
                if ( v98 )
                {
                  v153 = *(_BYTE **)(v8 + 32 * (1LL - (*(_DWORD *)(v8 + 4) & 0x7FFFFFF)));
                  LODWORD(v161) = 284;
                  LODWORD(v162) = 0;
                  v163 = &v154;
                  LODWORD(v164) = 1;
                  v165 = v98;
                  v99 = sub_10E25C0((__int64)&v161, v7);
                  v10 = 0;
                  if ( v99 )
                  {
                    if ( *(_BYTE *)v7 == 85
                      && *(_QWORD *)(v7 + 32
                                        * ((unsigned int)v164 - (unsigned __int64)(*(_DWORD *)(v7 + 4) & 0x7FFFFFF))) == v165 )
                    {
                      v100 = (unsigned int **)a1[2].m128i_i64[0];
                      LOWORD(v165) = 257;
                      sub_10A0170((__int64)&v157, a2);
                      v101 = sub_A826E0(v100, v152, v154, v157, (__int64)&v161, 0);
                      v146 = a1[2].m128i_i64[0];
                      LOWORD(v165) = 257;
                      sub_10A0170((__int64)&v157, a2);
                      v77 = (_BYTE *)v101;
                      v79 = v157;
                      v76 = v153;
                      v78 = v146;
LABEL_115:
                      v46 = sub_B33C40(v78, 0x11Cu, (__int64)v77, (__int64)v76, v79, (__int64)&v161);
                      return sub_F162A0((__int64)a1, a2, v46);
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
LABEL_92:
    if ( *(_BYTE *)v8 == 85 )
    {
      v122 = *(_QWORD *)(v8 - 32);
      if ( v122 )
      {
        if ( !*(_BYTE *)v122 && *(_QWORD *)(v122 + 24) == *(_QWORD *)(v8 + 80) && *(_DWORD *)(v122 + 36) == 88 )
        {
          v123 = *(_BYTE **)(v8 - 32LL * (*(_DWORD *)(v8 + 4) & 0x7FFFFFF));
          if ( v123 )
          {
            v163 = &v153;
            v152 = v123;
            LODWORD(v161) = 88;
            LODWORD(v162) = 0;
            v124 = sub_10E25C0((__int64)&v161, v7);
            v10 = 0;
            if ( v124 )
            {
              v125 = (unsigned int **)a1[2].m128i_i64[0];
              LOWORD(v165) = 257;
              sub_10A0170((__int64)&v157, a2);
              v126 = sub_92A220(v125, v152, v153, v157, (__int64)&v161, 0);
              v149 = a1[2].m128i_i64[0];
              LOWORD(v165) = 257;
              sub_10A0170((__int64)&v157, a2);
              v46 = sub_B33BC0(v149, 0x58u, v126, v157, (__int64)&v161);
              return sub_F162A0((__int64)a1, a2, v46);
            }
          }
        }
      }
    }
    if ( *(_BYTE *)v8 == 85 )
    {
      v102 = *(_QWORD *)(v8 - 32);
      if ( v102 )
      {
        if ( !*(_BYTE *)v102 && *(_QWORD *)(v102 + 24) == *(_QWORD *)(v8 + 80) && *(_DWORD *)(v102 + 36) == 90 )
        {
          v103 = *(_BYTE **)(v8 - 32LL * (*(_DWORD *)(v8 + 4) & 0x7FFFFFF));
          if ( v103 )
          {
            v163 = &v153;
            v152 = v103;
            LODWORD(v161) = 90;
            LODWORD(v162) = 0;
            v104 = sub_10E25C0((__int64)&v161, v7);
            v10 = 0;
            if ( v104 )
            {
              v105 = (unsigned int **)a1[2].m128i_i64[0];
              LOWORD(v165) = 257;
              sub_10A0170((__int64)&v157, a2);
              v106 = sub_92A220(v105, v152, v153, v157, (__int64)&v161, 0);
              v147 = a1[2].m128i_i64[0];
              LOWORD(v165) = 257;
              sub_10A0170((__int64)&v157, a2);
              v46 = sub_B33BC0(v147, 0x5Au, v106, v157, (__int64)&v161);
              return sub_F162A0((__int64)a1, a2, v46);
            }
          }
        }
      }
    }
LABEL_94:
    v161 = v7;
    v162 = &v153;
    v57 = *(_QWORD *)(v8 + 16);
    if ( v57
      && !*(_QWORD *)(v57 + 8)
      && *(_BYTE *)v8 == 47
      && (unsigned __int8)sub_1155120((__int64)&v161, v8)
      && v153 != (_BYTE *)v7 )
    {
      v59 = (unsigned int **)a1[2].m128i_i64[0];
      LOWORD(v165) = 257;
      v114 = sub_B45210(a2);
      v61 = (_BYTE *)v7;
      LODWORD(v157) = v114;
      v62 = (_BYTE *)v7;
      BYTE4(v157) = 1;
      v63 = v157;
    }
    else
    {
      v161 = v8;
      v162 = &v153;
      v58 = *(_QWORD *)(v7 + 16);
      if ( !v58
        || *(_QWORD *)(v58 + 8)
        || *(_BYTE *)v7 != 47
        || !(unsigned __int8)sub_1155120((__int64)&v161, v7)
        || v153 == (_BYTE *)v8 )
      {
        return v10;
      }
      v59 = (unsigned int **)a1[2].m128i_i64[0];
      LOWORD(v165) = 257;
      v60 = sub_B45210(a2);
      v61 = (_BYTE *)v8;
      LODWORD(v157) = v60;
      v62 = (_BYTE *)v8;
      BYTE4(v157) = 1;
      v63 = v157;
    }
    v64 = sub_A826E0(v59, v62, v61, v63, (__int64)&v161, 0);
    LOWORD(v165) = 257;
    v51 = (_BYTE *)v64;
    v52 = 18;
    v48 = v153;
    goto LABEL_122;
  }
  return v10;
}
