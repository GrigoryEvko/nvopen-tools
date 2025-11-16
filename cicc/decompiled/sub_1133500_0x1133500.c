// Function: sub_1133500
// Address: 0x1133500
//
unsigned __int8 *__fastcall sub_1133500(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r14
  unsigned int v7; // r13d
  bool v8; // al
  unsigned int v9; // r13d
  unsigned __int16 v10; // r15
  unsigned __int8 *v11; // r9
  _BYTE *v12; // r11
  int v13; // eax
  __int64 v14; // rdx
  _BYTE *v15; // rax
  __int64 v16; // r8
  char v17; // al
  __int64 v18; // r9
  unsigned int v19; // edx
  __int64 v20; // rsi
  __int64 v21; // rax
  __int64 v22; // rax
  _QWORD *v23; // r12
  bool v25; // zf
  __int64 v26; // r14
  _QWORD *v27; // rax
  _BYTE *v28; // rsi
  _BYTE *v29; // rsi
  int v30; // r14d
  __int64 v31; // rax
  __int64 v32; // r13
  _QWORD *v33; // rax
  bool v34; // al
  __int64 v35; // rax
  __int64 v36; // rsi
  __int64 *v37; // r14
  __int64 v38; // rdx
  __int64 v39; // rcx
  __int64 v40; // r8
  int v41; // eax
  __int64 v42; // r14
  int v43; // ecx
  __int64 *v44; // rax
  __int64 v45; // rdi
  __int64 v46; // r15
  _QWORD *v47; // rax
  __int64 v48; // r12
  __int64 *v49; // rdx
  __int64 *v50; // rdi
  __int64 **v51; // r10
  unsigned int v52; // eax
  _BYTE *v53; // rcx
  int v54; // eax
  unsigned int v55; // eax
  __int64 v56; // r8
  __int64 v57; // r9
  __int64 v58; // r11
  _QWORD *v59; // rax
  unsigned int **v60; // r14
  _BYTE *v61; // rax
  __int64 v62; // r14
  _QWORD *v63; // rax
  __int64 v64; // rcx
  int v65; // eax
  char v66; // al
  unsigned __int8 *v67; // rdx
  unsigned __int8 *v68; // rdi
  int v69; // eax
  int v70; // eax
  __int64 v71; // rdx
  unsigned __int8 *v72; // rdi
  int v73; // eax
  int v74; // eax
  _BYTE *v75; // rax
  __int64 v76; // rax
  __int64 v77; // r14
  unsigned int **v78; // r12
  _BYTE *v79; // rax
  __int64 v80; // rax
  __int64 v81; // r8
  __int64 v82; // rdx
  __int64 v83; // rcx
  __int64 v84; // r9
  _BYTE *v85; // r8
  __int64 v86; // rax
  __int64 v87; // r13
  _QWORD *v88; // rax
  __int64 v89; // rax
  __int64 v90; // r8
  __int64 v91; // rax
  __int64 v92; // rdx
  _BYTE *v93; // rax
  __int64 v94; // rdx
  _BYTE *v95; // rax
  int v96; // eax
  __int16 v97; // ax
  __int64 v98; // rbx
  unsigned int **v99; // r13
  __int64 v100; // rax
  __int64 v101; // r14
  _QWORD *v102; // r15
  __int64 v103; // rdx
  int v104; // ecx
  int v105; // eax
  _QWORD *v106; // rdi
  __int64 *v107; // rax
  __int64 v108; // rax
  unsigned int *v109; // rbx
  __int64 v110; // r14
  __int64 v111; // rdx
  unsigned int v112; // esi
  __int64 v113; // rdx
  int v114; // r14d
  __int64 v115; // r14
  unsigned int *v116; // r15
  __int64 v117; // rdx
  __int64 v118; // [rsp+18h] [rbp-1A8h]
  _BYTE *v119; // [rsp+28h] [rbp-198h]
  _BYTE *v120; // [rsp+28h] [rbp-198h]
  __int64 v121; // [rsp+28h] [rbp-198h]
  __int64 **v122; // [rsp+28h] [rbp-198h]
  unsigned int v123; // [rsp+28h] [rbp-198h]
  _BYTE *v124; // [rsp+30h] [rbp-190h]
  __int64 v125; // [rsp+30h] [rbp-190h]
  unsigned __int8 *v126; // [rsp+30h] [rbp-190h]
  unsigned __int8 *v127; // [rsp+30h] [rbp-190h]
  __int64 v128; // [rsp+30h] [rbp-190h]
  int v129; // [rsp+30h] [rbp-190h]
  unsigned int v130; // [rsp+30h] [rbp-190h]
  _BYTE *v131; // [rsp+38h] [rbp-188h]
  __int64 v132; // [rsp+38h] [rbp-188h]
  unsigned __int8 *v133; // [rsp+38h] [rbp-188h]
  __int64 *v134; // [rsp+38h] [rbp-188h]
  __int64 v135; // [rsp+38h] [rbp-188h]
  __int64 v136; // [rsp+38h] [rbp-188h]
  __int64 v137; // [rsp+38h] [rbp-188h]
  _BYTE *v138; // [rsp+38h] [rbp-188h]
  __int64 v139; // [rsp+38h] [rbp-188h]
  __int64 v140; // [rsp+38h] [rbp-188h]
  int v141; // [rsp+38h] [rbp-188h]
  unsigned __int8 *v142; // [rsp+40h] [rbp-180h]
  unsigned __int8 *v143; // [rsp+40h] [rbp-180h]
  int v144; // [rsp+40h] [rbp-180h]
  __int64 v145; // [rsp+40h] [rbp-180h]
  unsigned int **v146; // [rsp+40h] [rbp-180h]
  __int64 v147; // [rsp+40h] [rbp-180h]
  __int64 v148; // [rsp+40h] [rbp-180h]
  _BYTE *v149; // [rsp+40h] [rbp-180h]
  __int64 v150; // [rsp+40h] [rbp-180h]
  __int64 v151; // [rsp+40h] [rbp-180h]
  __int64 v152; // [rsp+40h] [rbp-180h]
  __int64 v154; // [rsp+48h] [rbp-178h]
  char v155; // [rsp+57h] [rbp-169h] BYREF
  __int64 v156; // [rsp+58h] [rbp-168h] BYREF
  __int64 v157; // [rsp+60h] [rbp-160h] BYREF
  __int64 v158; // [rsp+68h] [rbp-158h]
  _BYTE v159[32]; // [rsp+70h] [rbp-150h] BYREF
  __int16 v160; // [rsp+90h] [rbp-130h]
  __int64 *v161; // [rsp+A0h] [rbp-120h] BYREF
  __int64 **v162; // [rsp+A8h] [rbp-118h]
  __int16 v163; // [rsp+C0h] [rbp-100h]
  __int64 *v164; // [rsp+D0h] [rbp-F0h] BYREF
  __int64 v165; // [rsp+D8h] [rbp-E8h]
  _BYTE v166[32]; // [rsp+E0h] [rbp-E0h] BYREF
  __int64 *v167; // [rsp+100h] [rbp-C0h] BYREF
  __int64 v168; // [rsp+108h] [rbp-B8h] BYREF
  _QWORD v169[2]; // [rsp+110h] [rbp-B0h] BYREF
  __int64 v170; // [rsp+120h] [rbp-A0h]

  v4 = a2;
  v7 = *(_DWORD *)(a4 + 8);
  if ( v7 <= 0x40 )
    v8 = *(_QWORD *)a4 == 1;
  else
    v8 = v7 - 1 == (unsigned int)sub_C444A0(a4);
  v9 = *(_WORD *)(a2 + 2) & 0x3F;
  v10 = *(_WORD *)(a2 + 2) & 0x3F;
  if ( v8 && v10 == 40 )
  {
    v54 = sub_BCB060(*(_QWORD *)(a3 + 8));
    if ( !v54 )
    {
      v11 = *(unsigned __int8 **)(a3 - 32);
      v12 = *(_BYTE **)(a3 - 64);
      v13 = 40;
      goto LABEL_6;
    }
    v64 = (unsigned int)(v54 - 1);
    v25 = *(_BYTE *)a3 == 58;
    v169[0] = 0;
    v167 = (__int64 *)&v164;
    v12 = *(_BYTE **)(a3 - 64);
    v168 = v64;
    v169[1] = &v164;
    v170 = v64;
    if ( !v25 )
    {
      v11 = *(unsigned __int8 **)(a3 - 32);
      v13 = 40;
      goto LABEL_6;
    }
    if ( *v12 != 56 || !*((_QWORD *)v12 - 8) )
      goto LABEL_98;
    v164 = (__int64 *)*((_QWORD *)v12 - 8);
    v85 = (_BYTE *)*((_QWORD *)v12 - 4);
    if ( !v85 )
      goto LABEL_150;
    if ( *v85 != 17 )
    {
      v92 = (unsigned int)*(unsigned __int8 *)(*((_QWORD *)v85 + 1) + 8LL) - 17;
      if ( (unsigned int)v92 > 1 )
        goto LABEL_98;
      if ( *v85 > 0x15u )
        goto LABEL_98;
      v93 = sub_AD7630(*((_QWORD *)v12 - 4), 0, v92);
      v85 = v93;
      if ( !v93 || *v93 != 17 )
        goto LABEL_98;
      v64 = v168;
    }
    if ( *((_DWORD *)v85 + 8) > 0x40u )
    {
      v129 = *((_DWORD *)v85 + 8);
      v140 = v64;
      v149 = v85;
      v96 = sub_C444A0((__int64)(v85 + 24));
      v64 = v140;
      if ( (unsigned int)(v129 - v96) > 0x40 )
        goto LABEL_98;
      v86 = **((_QWORD **)v149 + 3);
    }
    else
    {
      v86 = *((_QWORD *)v85 + 3);
    }
    v67 = *(unsigned __int8 **)(a3 - 32);
    if ( v86 != v64 )
      goto LABEL_99;
    if ( sub_11327B0((__int64)v169, 26, v67) )
      goto LABEL_124;
LABEL_98:
    v67 = *(unsigned __int8 **)(a3 - 32);
LABEL_99:
    v11 = v67;
    if ( *v67 != 56 )
      goto LABEL_100;
    v89 = *((_QWORD *)v67 - 8);
    if ( !v89 )
      goto LABEL_100;
    *v167 = v89;
    v90 = *((_QWORD *)v67 - 4);
    if ( v90 )
    {
      if ( *(_BYTE *)v90 != 17 )
      {
        v94 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v90 + 8) + 8LL) - 17;
        if ( (unsigned int)v94 > 1 )
          goto LABEL_128;
        if ( *(_BYTE *)v90 > 0x15u )
          goto LABEL_128;
        v95 = sub_AD7630(v90, 0, v94);
        v90 = (__int64)v95;
        if ( !v95 || *v95 != 17 )
          goto LABEL_128;
      }
      if ( *(_DWORD *)(v90 + 32) > 0x40u )
      {
        v141 = *(_DWORD *)(v90 + 32);
        v150 = v90;
        if ( v141 - (unsigned int)sub_C444A0(v90 + 24) > 0x40 )
        {
          v12 = *(_BYTE **)(a3 - 64);
          goto LABEL_135;
        }
        v91 = **(_QWORD **)(v150 + 24);
      }
      else
      {
        v91 = *(_QWORD *)(v90 + 24);
      }
      v12 = *(_BYTE **)(a3 - 64);
      if ( v168 == v91 )
      {
        if ( sub_11327B0((__int64)v169, 26, *(unsigned __int8 **)(a3 - 64)) )
        {
LABEL_124:
          if ( v164 )
          {
            v154 = (__int64)v164;
            v87 = sub_AD64C0(v164[1], 1, 0);
            LOWORD(v170) = 257;
            v88 = sub_BD2C40(72, unk_3F10FD0);
            v23 = v88;
            if ( v88 )
              sub_1113300((__int64)v88, 40, v154, v87, (__int64)&v167);
            return (unsigned __int8 *)v23;
          }
        }
LABEL_128:
        v11 = *(unsigned __int8 **)(a3 - 32);
LABEL_100:
        v12 = *(_BYTE **)(a3 - 64);
        v13 = *(_WORD *)(a2 + 2) & 0x3F;
        goto LABEL_6;
      }
LABEL_135:
      v11 = *(unsigned __int8 **)(a3 - 32);
      v13 = *(_WORD *)(a2 + 2) & 0x3F;
      goto LABEL_6;
    }
LABEL_150:
    BUG();
  }
  v11 = *(unsigned __int8 **)(a3 - 32);
  v12 = *(_BYTE **)(a3 - 64);
  v13 = *(_WORD *)(a2 + 2) & 0x3F;
LABEL_6:
  v14 = *v11;
  if ( (unsigned int)(v13 - 32) > 1 )
    goto LABEL_17;
  if ( (unsigned __int8)v14 > 0x15u )
  {
LABEL_8:
    if ( (unsigned int)*(unsigned __int8 *)(*((_QWORD *)v11 + 1) + 8LL) - 17 > 1 || (unsigned __int8)v14 > 0x15u )
      goto LABEL_20;
    goto LABEL_10;
  }
  if ( (_BYTE)v14 == 5 )
  {
    if ( (unsigned int)*(unsigned __int8 *)(*((_QWORD *)v11 + 1) + 8LL) - 17 > 1 )
      goto LABEL_20;
LABEL_10:
    v131 = v12;
    v142 = v11;
    v15 = sub_AD7630((__int64)v11, 0, v14);
    if ( !v15 || *v15 != 17 )
      goto LABEL_20;
    v11 = v142;
    v12 = v131;
    v16 = (__int64)(v15 + 24);
LABEL_19:
    if ( (*(_WORD *)(a2 + 2) & 0x3Fu) - 32 > 1 )
      goto LABEL_20;
    if ( *(_DWORD *)(v16 + 8) <= 0x40u )
    {
      if ( *(_QWORD *)v16 != *(_QWORD *)a4 )
        goto LABEL_51;
    }
    else
    {
      v124 = v12;
      v133 = v11;
      v145 = v16;
      v34 = sub_C43C50(v16, (const void **)a4);
      v16 = v145;
      v11 = v133;
      v12 = v124;
      if ( !v34 )
      {
LABEL_51:
        v35 = *(_QWORD *)(a3 + 16);
        if ( v35 && !*(_QWORD *)(v35 + 8) )
        {
          v36 = v16;
          v134 = (__int64 *)v16;
          LOWORD(v170) = 257;
          v37 = *(__int64 **)(a1 + 32);
          v125 = (__int64)v12;
          sub_9865C0((__int64)&v161, v16);
          sub_987160((__int64)&v161, v36, v38, v39, v40);
          v41 = (int)v162;
          LODWORD(v162) = 0;
          LODWORD(v165) = v41;
          v164 = v161;
          v42 = sub_10BC480(v37, v125, (__int64)&v164, (__int64)&v167);
          sub_969240((__int64 *)&v164);
          sub_969240((__int64 *)&v161);
          sub_9865C0((__int64)&v164, a4);
          v43 = v165;
          if ( (unsigned int)v165 > 0x40 )
          {
            sub_C43C10(&v164, v134);
            v43 = v165;
            v44 = v164;
          }
          else
          {
            v44 = (__int64 *)(*v134 ^ (unsigned __int64)v164);
            v164 = v44;
          }
          v45 = *(_QWORD *)(a3 + 8);
          LODWORD(v168) = v43;
          v167 = v44;
          LODWORD(v165) = 0;
          v46 = sub_AD8D80(v45, (__int64)&v167);
          sub_969240((__int64 *)&v167);
          sub_969240((__int64 *)&v164);
          LOWORD(v170) = 257;
          v47 = sub_BD2C40(72, unk_3F10FD0);
          v23 = v47;
          if ( v47 )
            sub_1113300((__int64)v47, v9, v42, v46, (__int64)&v167);
          return (unsigned __int8 *)v23;
        }
LABEL_20:
        if ( sub_9893F0(v9, a4, &v155) )
        {
          v25 = *(_BYTE *)a3 == 58;
          v168 = 0;
          v167 = &v156;
          v169[0] = &v156;
          if ( v25 )
          {
            v28 = *(_BYTE **)(a3 - 64);
            if ( *v28 == 42 )
            {
              v66 = sub_1111CE0(&v167, (__int64)v28);
              v29 = *(_BYTE **)(a3 - 32);
              if ( v66 && v29 == *(_BYTE **)v169[0] )
              {
LABEL_47:
                v30 = v155 == 0 ? 38 : 40;
                v31 = sub_AD64C0(*(_QWORD *)(v156 + 8), v155 != 0, 0);
                LOWORD(v170) = 257;
                v32 = v31;
                v33 = sub_BD2C40(72, unk_3F10FD0);
                v23 = v33;
                if ( v33 )
                  sub_1113300((__int64)v33, v30, v156, v32, (__int64)&v167);
                return (unsigned __int8 *)v23;
              }
            }
            else
            {
              v29 = *(_BYTE **)(a3 - 32);
            }
            if ( *v29 == 42
              && (unsigned __int8)sub_1111CE0(&v167, (__int64)v29)
              && *(_QWORD *)(a3 - 64) == *(_QWORD *)v169[0] )
            {
              goto LABEL_47;
            }
          }
        }
        v19 = *(_DWORD *)(a4 + 8);
        v20 = *(_QWORD *)a4;
        v21 = 1LL << ((unsigned __int8)v19 - 1);
        if ( v19 > 0x40 )
        {
          v20 = *(_QWORD *)(v20 + 8LL * ((v19 - 1) >> 6));
          if ( (v20 & v21) != 0 )
            goto LABEL_23;
        }
        else if ( (v20 & v21) != 0 )
        {
LABEL_23:
          if ( (*(_WORD *)(v4 + 2) & 0x3Fu) - 32 > 1 )
            return 0;
          if ( *(_DWORD *)(a4 + 8) <= 0x40u )
          {
            if ( *(_QWORD *)a4 )
              return 0;
            v22 = *(_QWORD *)(a3 + 16);
            if ( !v22 )
              return 0;
          }
          else
          {
            v144 = *(_DWORD *)(a4 + 8);
            if ( v144 != (unsigned int)sub_C444A0(a4) )
              return 0;
            v22 = *(_QWORD *)(a3 + 16);
            if ( !v22 )
              return 0;
          }
          v48 = *(_QWORD *)(v22 + 8);
          if ( v48 )
            return 0;
          if ( *(_BYTE *)a3 != 58 )
            goto LABEL_61;
          v68 = *(unsigned __int8 **)(a3 - 64);
          v69 = *v68;
          if ( (unsigned __int8)v69 > 0x1Cu )
          {
            v70 = v69 - 29;
          }
          else
          {
            if ( (_BYTE)v69 != 5 )
              goto LABEL_61;
            v70 = *((unsigned __int16 *)v68 + 1);
          }
          if ( v70 == 47 )
          {
            v71 = *(_QWORD *)sub_986520((__int64)v68);
            if ( v71 )
            {
              v72 = *(unsigned __int8 **)(a3 - 32);
              v73 = *v72;
              if ( (unsigned __int8)v73 > 0x1Cu )
              {
                v74 = v73 - 29;
                goto LABEL_109;
              }
              if ( (_BYTE)v73 == 5 )
              {
                v74 = *((unsigned __int16 *)v72 + 1);
LABEL_109:
                v146 = *(unsigned int ***)(a1 + 32);
                if ( v74 == 47 )
                {
                  v139 = v71;
                  v121 = *(_QWORD *)sub_986520((__int64)v72);
                  if ( v121 )
                  {
                    LOWORD(v170) = 257;
                    v75 = (_BYTE *)sub_AD6530(*(_QWORD *)(v139 + 8), v20);
                    v76 = sub_92B530(v146, v9, v139, v75, (__int64)&v167);
                    LOWORD(v170) = 257;
                    v77 = v76;
                    v78 = *(unsigned int ***)(a1 + 32);
                    v79 = (_BYTE *)sub_AD6530(*(_QWORD *)(v121 + 8), v9);
                    v80 = sub_92B530(v78, v9, v121, v79, (__int64)&v167);
                    LOWORD(v170) = 257;
                    return (unsigned __int8 *)sub_B504D0((unsigned int)(v10 != 32) + 28, v77, v80, (__int64)&v167, 0, 0);
                  }
                }
LABEL_62:
                v169[0] = a3;
                v164 = (__int64 *)v166;
                v49 = (__int64 *)&v164;
                v50 = v169;
                v167 = v169;
                v51 = &v167;
                v165 = 0x200000000LL;
                v168 = 0x1000000001LL;
                v52 = 1;
                while ( 1 )
                {
                  v161 = (__int64 *)&v164;
                  v162 = v51;
                  v53 = (_BYTE *)v50[v52 - 1];
                  LODWORD(v168) = v52 - 1;
                  if ( *v53 != 58 )
                    break;
                  v81 = *((_QWORD *)v53 - 8);
                  v122 = v51;
                  v128 = v81;
                  if ( !v81 )
                    break;
                  v20 = *((_QWORD *)v53 - 4);
                  if ( !v20 )
                    break;
                  sub_1111320((__int64 *)&v161, v20, (__int64)v49, (__int64)v53, v81, v18);
                  v20 = v128;
                  sub_1111320((__int64 *)&v161, v128, v82, v83, v128, v84);
                  v52 = v168;
                  if ( !(_DWORD)v168 )
                  {
                    v97 = *(_WORD *)(v4 + 2);
                    v163 = 257;
                    v20 = v97 & 0x3F;
                    v130 = v97 & 0x3F;
                    v123 = ((v97 & 0x3F) != 32) + 28;
                    v48 = sub_92B530(
                            v146,
                            v20,
                            v164[2 * (unsigned int)v165 - 2],
                            (_BYTE *)v164[2 * (unsigned int)v165 - 1],
                            (__int64)&v161);
                    v98 = (__int64)&v164[2 * (unsigned int)v165 - 2];
                    if ( v164 == (__int64 *)v98 )
                    {
                      v50 = v167;
                    }
                    else
                    {
                      v99 = v146;
                      v118 = v4;
                      do
                      {
                        v160 = 257;
                        v101 = *(_QWORD *)(v98 - 16);
                        v151 = *(_QWORD *)(v98 - 8);
                        v102 = (_QWORD *)(*(__int64 (__fastcall **)(unsigned int *, _QWORD, __int64, __int64))(*(_QWORD *)v99[10] + 56LL))(
                                           v99[10],
                                           v130,
                                           v101,
                                           v151);
                        if ( !v102 )
                        {
                          v163 = 257;
                          v102 = sub_BD2C40(72, unk_3F10FD0);
                          if ( v102 )
                          {
                            v103 = *(_QWORD *)(v101 + 8);
                            v104 = *(unsigned __int8 *)(v103 + 8);
                            if ( (unsigned int)(v104 - 17) > 1 )
                            {
                              v108 = sub_BCB2A0(*(_QWORD **)v103);
                            }
                            else
                            {
                              v105 = *(_DWORD *)(v103 + 32);
                              v106 = *(_QWORD **)v103;
                              BYTE4(v158) = (_BYTE)v104 == 18;
                              LODWORD(v158) = v105;
                              v107 = (__int64 *)sub_BCB2A0(v106);
                              v108 = sub_BCE1B0(v107, v158);
                            }
                            sub_B523C0((__int64)v102, v108, 53, v130, v101, v151, (__int64)&v161, 0, 0, 0);
                          }
                          (*(void (__fastcall **)(unsigned int *, _QWORD *, _BYTE *, unsigned int *, unsigned int *))(*(_QWORD *)v99[11] + 16LL))(
                            v99[11],
                            v102,
                            v159,
                            v99[7],
                            v99[8]);
                          if ( *v99 != &(*v99)[4 * *((unsigned int *)v99 + 2)] )
                          {
                            v152 = v98;
                            v109 = *v99;
                            v110 = (__int64)&(*v99)[4 * *((unsigned int *)v99 + 2)];
                            do
                            {
                              v111 = *((_QWORD *)v109 + 1);
                              v112 = *v109;
                              v109 += 4;
                              sub_B99FD0((__int64)v102, v112, v111);
                            }
                            while ( (unsigned int *)v110 != v109 );
                            v98 = v152;
                          }
                        }
                        v20 = v123;
                        v160 = 257;
                        v100 = (*(__int64 (__fastcall **)(unsigned int *, _QWORD, __int64, _QWORD *))(*(_QWORD *)v99[10] + 16LL))(
                                 v99[10],
                                 v123,
                                 v48,
                                 v102);
                        if ( v100 )
                        {
                          v48 = v100;
                        }
                        else
                        {
                          v163 = 257;
                          v48 = sub_B504D0(v123, v48, (__int64)v102, (__int64)&v161, 0, 0);
                          if ( (unsigned __int8)sub_920620(v48) )
                          {
                            v113 = (__int64)v99[12];
                            v114 = *((_DWORD *)v99 + 26);
                            if ( v113 )
                              sub_B99FD0(v48, 3u, v113);
                            sub_B45150(v48, v114);
                          }
                          v20 = v48;
                          (*(void (__fastcall **)(unsigned int *, __int64, _BYTE *, unsigned int *, unsigned int *))(*(_QWORD *)v99[11] + 16LL))(
                            v99[11],
                            v48,
                            v159,
                            v99[7],
                            v99[8]);
                          v115 = (__int64)&(*v99)[4 * *((unsigned int *)v99 + 2)];
                          if ( *v99 != (unsigned int *)v115 )
                          {
                            v116 = *v99;
                            do
                            {
                              v117 = *((_QWORD *)v116 + 1);
                              v20 = *v116;
                              v116 += 4;
                              sub_B99FD0(v48, v20, v117);
                            }
                            while ( (unsigned int *)v115 != v116 );
                          }
                        }
                        v98 -= 16;
                      }
                      while ( v164 != (__int64 *)v98 );
                      v4 = v118;
                      v50 = v167;
                    }
                    break;
                  }
                  v50 = v167;
                  v51 = v122;
                }
                if ( v50 != v169 )
                  _libc_free(v50, v20);
                if ( v164 != (__int64 *)v166 )
                  _libc_free(v164, v20);
                if ( v48 )
                  return sub_F162A0(a1, v4, v48);
                return 0;
              }
            }
          }
LABEL_61:
          v146 = *(unsigned int ***)(a1 + 32);
          goto LABEL_62;
        }
        v25 = *(_BYTE *)a3 == 58;
        LOBYTE(v169[0]) = 0;
        v167 = &v156;
        v168 = (__int64)&v157;
        if ( !v25 )
          goto LABEL_23;
        if ( !*(_QWORD *)(a3 - 64) )
          goto LABEL_23;
        v20 = *(_QWORD *)(a3 - 32);
        v156 = *(_QWORD *)(a3 - 64);
        if ( !(unsigned __int8)sub_991580((__int64)&v168, v20) )
          goto LABEL_23;
        if ( v10 > 0x28u )
        {
          if ( v10 != 41 )
            goto LABEL_23;
        }
        else
        {
          if ( v10 > 0x26u )
          {
            v20 = a4;
            if ( (int)sub_C4C880(v157, a4) < 0 )
              goto LABEL_23;
LABEL_38:
            v26 = sub_AD6530(*(_QWORD *)(v156 + 8), v20);
            LOWORD(v170) = 257;
            v27 = sub_BD2C40(72, unk_3F10FD0);
            v23 = v27;
            if ( v27 )
              sub_1113300((__int64)v27, v9, v156, v26, (__int64)&v167);
            return (unsigned __int8 *)v23;
          }
          if ( v10 != 38 )
            goto LABEL_23;
        }
        v20 = a4;
        if ( (int)sub_C4C880(v157, a4) <= 0 )
          goto LABEL_23;
        LOWORD(v9) = sub_B53250(v9);
        goto LABEL_38;
      }
    }
    LODWORD(v165) = *(_DWORD *)(a4 + 8);
    if ( (unsigned int)v165 > 0x40 )
    {
      v120 = v12;
      v127 = v11;
      v137 = v16;
      sub_C43780((__int64)&v164, (const void **)a4);
      v12 = v120;
      v11 = v127;
      v16 = v137;
    }
    else
    {
      v164 = *(__int64 **)a4;
    }
    v119 = v12;
    v126 = v11;
    v135 = v16;
    sub_C46A40((__int64)&v164, 1);
    v55 = v165;
    LODWORD(v165) = 0;
    v56 = v135;
    LODWORD(v168) = v55;
    v57 = (__int64)v126;
    v167 = v164;
    v58 = (__int64)v119;
    if ( v55 > 0x40 )
    {
      v65 = sub_C44630((__int64)&v167);
      v56 = v135;
      v57 = (__int64)v126;
      v58 = (__int64)v119;
      if ( v65 == 1 )
        goto LABEL_82;
    }
    else if ( v164 && ((unsigned __int64)v164 & ((unsigned __int64)v164 - 1)) == 0 )
    {
LABEL_82:
      v136 = v58;
      v147 = v57;
      sub_969240((__int64 *)&v167);
      sub_969240((__int64 *)&v164);
      LOWORD(v170) = 257;
      v59 = sub_BD2C40(72, unk_3F10FD0);
      v23 = v59;
      if ( v59 )
        sub_1113300((__int64)v59, 3 * (v10 == 32) + 34, v136, v147, (__int64)&v167);
      return (unsigned __int8 *)v23;
    }
    v138 = (_BYTE *)v58;
    v148 = v56;
    sub_969240((__int64 *)&v167);
    sub_969240((__int64 *)&v164);
    v16 = v148;
    v12 = v138;
    goto LABEL_51;
  }
  v132 = (__int64)v12;
  v143 = v11;
  v17 = sub_AD6CA0((__int64)v11);
  v11 = v143;
  v12 = (_BYTE *)v132;
  if ( v17 || (*(_BYTE *)(a3 + 1) & 2) == 0 )
  {
    v14 = *v143;
LABEL_17:
    if ( (_BYTE)v14 == 17 )
    {
      v16 = (__int64)(v11 + 24);
      goto LABEL_19;
    }
    goto LABEL_8;
  }
  v60 = *(unsigned int ***)(a1 + 32);
  LOWORD(v170) = 257;
  v61 = (_BYTE *)sub_AD8D80(*((_QWORD *)v143 + 1), a4);
  v62 = sub_A825B0(v60, v143, v61, (__int64)&v167);
  LOWORD(v170) = 257;
  v63 = sub_BD2C40(72, unk_3F10FD0);
  v23 = v63;
  if ( v63 )
    sub_1113300((__int64)v63, v9, v132, v62, (__int64)&v167);
  return (unsigned __int8 *)v23;
}
