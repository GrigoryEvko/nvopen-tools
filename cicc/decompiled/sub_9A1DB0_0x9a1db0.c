// Function: sub_9A1DB0
// Address: 0x9a1db0
//
__int64 __fastcall sub_9A1DB0(unsigned __int8 *a1, __int64 a2, int a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r12
  char *v6; // r15
  const __m128i *v8; // r13
  unsigned __int8 v9; // dl
  __int64 v10; // rbx
  __int64 v11; // rax
  __int64 v12; // rdi
  __int64 v13; // r15
  __int64 v14; // r15
  unsigned int v15; // ebx
  __int64 v16; // r14
  int v17; // eax
  __int64 v18; // rax
  __int64 v19; // rdx
  unsigned int v20; // r9d
  __int64 v21; // rax
  __int64 *v22; // rbx
  __int64 v23; // rax
  __int64 v24; // r14
  _BYTE *v25; // r9
  __int64 v26; // rax
  char v27; // al
  _BYTE *v28; // r9
  char v29; // al
  __int64 v30; // rax
  char v31; // al
  unsigned int v32; // ebx
  int *v33; // rax
  int v34; // eax
  __int64 v35; // rbx
  __int64 v36; // rax
  __int64 v37; // rdx
  unsigned int v38; // ebx
  bool v39; // al
  __int64 v40; // rdx
  _QWORD *v41; // rdx
  char *v42; // r15
  char v43; // al
  __int64 v44; // rax
  __int64 v45; // rbx
  __int64 v46; // rax
  __int64 v47; // rdx
  unsigned int v48; // ebx
  __int64 v49; // rbx
  __int64 v50; // rax
  __int64 v51; // rcx
  unsigned int v52; // r8d
  __int64 v53; // rdx
  int v54; // ebx
  unsigned int v55; // r13d
  __int64 v56; // rax
  __int64 v57; // rax
  unsigned int v58; // eax
  unsigned __int8 **v59; // rax
  __int64 v60; // rax
  _QWORD *v61; // rax
  __int64 v62; // rax
  _QWORD *v63; // rax
  __int64 v64; // rax
  unsigned __int8 **v65; // rax
  __int64 v66; // rax
  __int64 v67; // rax
  _QWORD *v68; // rax
  _BYTE *v69; // rdx
  __int64 *v70; // rax
  _BYTE *v71; // rdx
  _QWORD *v72; // rax
  __int64 v73; // rax
  __m128i *v74; // rsi
  __int64 v75; // rcx
  __int64 v76; // rdx
  __int64 v77; // rax
  unsigned int v78; // edx
  bool v79; // zf
  _BYTE *v80; // rax
  __m128i v81; // xmm1
  __m128i v82; // xmm2
  unsigned __int64 v83; // xmm3_8
  __int64 v84; // rax
  __int64 v85; // r9
  _QWORD *v86; // r8
  _QWORD *v87; // r9
  __m128i *v88; // rcx
  __int64 v89; // rdx
  unsigned __int64 v90; // rax
  int v91; // edx
  unsigned __int64 v92; // rax
  char v93; // al
  unsigned int v94; // r14d
  __int64 v95; // rcx
  char *v96; // rdx
  __int64 v97; // rax
  int v98; // edx
  unsigned __int64 v99; // rax
  __int64 v100; // rdx
  unsigned __int64 v101; // rax
  int v102; // edx
  unsigned __int64 v103; // rax
  char v104; // al
  char *v105; // r8
  __int64 v106; // rdi
  __int64 v107; // rdx
  unsigned __int64 v108; // rax
  int v109; // edx
  unsigned __int64 v110; // rax
  char v111; // al
  __int64 v112; // rdi
  __int64 v113; // rdx
  unsigned __int64 v114; // rax
  int v115; // edx
  unsigned __int64 v116; // rax
  char v117; // al
  __int64 v118; // rdx
  unsigned __int64 v119; // rax
  __int64 v120; // rax
  __int64 v121; // rax
  __int64 v122; // rax
  __int64 v123; // rax
  unsigned int v124; // r10d
  __int64 v125; // rcx
  int v126; // r10d
  int v127; // r14d
  bool v128; // r15
  unsigned int i; // ebx
  unsigned __int8 *v130; // rax
  unsigned int v131; // r15d
  int v132; // eax
  __int64 v133; // rdx
  unsigned __int64 v134; // rax
  int v135; // edx
  unsigned __int64 v136; // rax
  __int64 v137; // r8
  __int64 v138; // rax
  int v139; // eax
  __int64 v140; // rdx
  __int64 v141; // rsi
  __int64 v142; // rax
  __int64 v143; // rcx
  unsigned __int8 v144; // cl
  int v145; // eax
  int v146; // r14d
  unsigned int v147; // ebx
  __int64 *v148; // rax
  __int64 v149; // rcx
  unsigned int v150; // r8d
  __int64 v151; // rsi
  int v152; // eax
  __int64 v153; // rax
  unsigned __int8 **v154; // rax
  __int64 v155; // rax
  __int64 v156; // rax
  _QWORD *v157; // rax
  int v158; // r14d
  char v159; // al
  _BYTE *v160; // r15
  unsigned int v161; // ebx
  __int64 v162; // rax
  __int64 v163; // rdi
  __int64 v164; // rdi
  signed __int64 v165; // rax
  __int64 v166; // rdx
  unsigned __int64 v167; // rax
  unsigned __int64 v168; // rdx
  char *v169; // rbx
  unsigned int v170; // ebx
  bool v171; // al
  __int64 v172; // rdx
  unsigned __int64 v173; // rax
  unsigned __int64 v174; // rdx
  __int64 v175; // rdx
  unsigned __int64 v176; // rax
  unsigned __int64 v177; // rdx
  _BYTE *v178; // rax
  __int64 v179; // rbx
  _BYTE *v180; // rax
  bool v181; // dl
  bool v182; // al
  bool v183; // al
  __int64 v184; // rax
  unsigned int v185; // r8d
  __int64 v186; // rdx
  __int64 v187; // rax
  __int64 v188; // rdx
  __int64 v189; // rax
  __int64 v190; // rax
  __int64 v191; // r8
  __int64 v192; // rax
  int v193; // r14d
  unsigned int j; // ebx
  _BYTE *v195; // rax
  bool v196; // al
  int v197; // r14d
  unsigned __int8 *v198; // rax
  int v199; // ecx
  __int64 v200; // rax
  __int64 v201; // rsi
  int v202; // r14d
  __int64 v203; // rax
  unsigned int v204; // r8d
  unsigned __int8 v205; // al
  char v206; // cl
  unsigned int v207; // edx
  int v208; // r14d
  __int64 v209; // rax
  unsigned int v210; // edx
  __int64 v211; // rax
  int v212; // [rsp+10h] [rbp-F0h]
  _QWORD *v213; // [rsp+10h] [rbp-F0h]
  unsigned __int8 v214; // [rsp+10h] [rbp-F0h]
  unsigned __int8 v215; // [rsp+10h] [rbp-F0h]
  char v216; // [rsp+10h] [rbp-F0h]
  __int64 *v217; // [rsp+18h] [rbp-E8h]
  unsigned int v218; // [rsp+18h] [rbp-E8h]
  _QWORD *v219; // [rsp+18h] [rbp-E8h]
  char *v220; // [rsp+18h] [rbp-E8h]
  __int64 v221; // [rsp+18h] [rbp-E8h]
  __int64 v222; // [rsp+18h] [rbp-E8h]
  __int64 v223; // [rsp+18h] [rbp-E8h]
  bool v224; // [rsp+18h] [rbp-E8h]
  int v225; // [rsp+18h] [rbp-E8h]
  unsigned int v226; // [rsp+18h] [rbp-E8h]
  __int64 v227; // [rsp+18h] [rbp-E8h]
  unsigned __int8 v228; // [rsp+20h] [rbp-E0h]
  __int64 v229; // [rsp+28h] [rbp-D8h]
  _BYTE *v230; // [rsp+28h] [rbp-D8h]
  unsigned __int8 v231; // [rsp+28h] [rbp-D8h]
  __int64 v232; // [rsp+28h] [rbp-D8h]
  __int64 v233; // [rsp+28h] [rbp-D8h]
  __m128i *v234; // [rsp+28h] [rbp-D8h]
  char *v235; // [rsp+28h] [rbp-D8h]
  __int64 v236; // [rsp+28h] [rbp-D8h]
  __int64 v237; // [rsp+28h] [rbp-D8h]
  __int64 v238; // [rsp+28h] [rbp-D8h]
  __int64 v239; // [rsp+28h] [rbp-D8h]
  __int64 v240; // [rsp+28h] [rbp-D8h]
  __int64 v241; // [rsp+28h] [rbp-D8h]
  __int64 v242; // [rsp+28h] [rbp-D8h]
  __int64 v243; // [rsp+28h] [rbp-D8h]
  __int64 v244; // [rsp+28h] [rbp-D8h]
  __int64 v245; // [rsp+28h] [rbp-D8h]
  __int64 v246; // [rsp+28h] [rbp-D8h]
  __int64 v247; // [rsp+28h] [rbp-D8h]
  unsigned int v248; // [rsp+28h] [rbp-D8h]
  __int64 v249; // [rsp+30h] [rbp-D0h] BYREF
  unsigned int v250; // [rsp+38h] [rbp-C8h]
  __int64 v251; // [rsp+40h] [rbp-C0h] BYREF
  unsigned int v252; // [rsp+48h] [rbp-B8h]
  __int64 v253; // [rsp+50h] [rbp-B0h] BYREF
  unsigned int v254; // [rsp+58h] [rbp-A8h]
  _QWORD *v255; // [rsp+60h] [rbp-A0h] BYREF
  __int64 v256; // [rsp+68h] [rbp-98h]
  __int64 v257[2]; // [rsp+70h] [rbp-90h] BYREF
  __m128i v258; // [rsp+80h] [rbp-80h] BYREF
  __m128i v259; // [rsp+90h] [rbp-70h] BYREF
  __m128i v260; // [rsp+A0h] [rbp-60h]
  __int128 v261; // [rsp+B0h] [rbp-50h]
  __int64 v262; // [rsp+C0h] [rbp-40h]

  while ( 2 )
  {
    v5 = (__int64)a1;
    LODWORD(v6) = *a1;
    v228 = a2;
    v212 = a3;
    if ( (unsigned __int8)v6 <= 0x15u )
    {
      if ( (_BYTE)a2 )
      {
        v258.m128i_i64[0] = 0;
        LODWORD(v6) = sub_993DE0(&v258, (__int64)a1);
        return (unsigned int)v6;
      }
      if ( (_BYTE)v6 == 17 )
      {
        if ( *((_DWORD *)a1 + 8) > 0x40u )
        {
          LOBYTE(v6) = (unsigned int)sub_C44630(a1 + 24) == 1;
          return (unsigned int)v6;
        }
        v30 = *((_QWORD *)a1 + 3);
        if ( !v30 )
          goto LABEL_38;
LABEL_52:
        LOBYTE(v6) = (v30 & (v30 - 1)) == 0;
        return (unsigned int)v6;
      }
      v35 = *((_QWORD *)a1 + 1);
      if ( (unsigned int)*(unsigned __int8 *)(v35 + 8) - 17 <= 1 )
      {
        v36 = sub_AD7630(a1, 0);
        if ( v36 && *(_BYTE *)v36 == 17 )
        {
          if ( *(_DWORD *)(v36 + 32) > 0x40u )
          {
            LOBYTE(v6) = (unsigned int)sub_C44630(v36 + 24) == 1;
            return (unsigned int)v6;
          }
          v30 = *(_QWORD *)(v36 + 24);
          if ( v30 )
            goto LABEL_52;
        }
        else if ( *(_BYTE *)(v35 + 8) == 17 )
        {
          v54 = *(_DWORD *)(v35 + 32);
          if ( v54 )
          {
            LODWORD(v6) = (unsigned __int8)a2;
            v55 = 0;
            while ( 1 )
            {
              v56 = sub_AD69F0(a1, v55);
              if ( !v56 )
                break;
              if ( *(_BYTE *)v56 != 13 )
              {
                if ( *(_BYTE *)v56 != 17 )
                  break;
                if ( *(_DWORD *)(v56 + 32) > 0x40u )
                {
                  if ( (unsigned int)sub_C44630(v56 + 24) != 1 )
                    break;
                }
                else
                {
                  v57 = *(_QWORD *)(v56 + 24);
                  if ( !v57 || (v57 & (v57 - 1)) != 0 )
                    break;
                }
                LODWORD(v6) = 1;
              }
              if ( v54 == ++v55 )
                return (unsigned int)v6;
            }
          }
        }
      }
LABEL_38:
      LODWORD(v6) = v228;
      return (unsigned int)v6;
    }
    v8 = (const __m128i *)a4;
    v9 = a2;
    if ( (_BYTE)a2 )
    {
      v17 = sub_BCB060(*((_QWORD *)a1 + 1));
      v9 = a2;
      if ( v17 == 1 )
        goto LABEL_19;
      v10 = v8[2].m128i_i64[0];
      if ( !v10 )
        goto LABEL_22;
    }
    else
    {
      v10 = *(_QWORD *)(a4 + 32);
      if ( !v10 )
        goto LABEL_22;
    }
    if ( !v8[2].m128i_i64[1] )
    {
      if ( !v8[3].m128i_i64[0] )
        goto LABEL_62;
      goto LABEL_40;
    }
    if ( !*(_BYTE *)(v10 + 192) )
    {
      v231 = v9;
      sub_CFDFC0(v10);
      v9 = v231;
    }
    v11 = *(unsigned int *)(v10 + 184);
    if ( (_DWORD)v11 )
    {
      v12 = *(_QWORD *)(v10 + 168);
      a2 = ((_DWORD)v11 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
      a4 = v12 + 88 * a2;
      a5 = *(_QWORD *)(a4 + 24);
      if ( v5 == a5 )
      {
LABEL_12:
        a2 = 5 * v11;
        if ( a4 != v12 + 88 * v11 )
        {
          a5 = *(_QWORD *)(a4 + 40);
          v13 = 32LL * *(unsigned int *)(a4 + 48);
          v229 = a5 + v13;
          if ( a5 + v13 != a5 )
          {
            v14 = *(_QWORD *)(a4 + 40);
            v15 = v9;
            do
            {
              v16 = *(_QWORD *)(v14 + 16);
              if ( v16 )
              {
                a2 = v15;
                if ( (unsigned __int8)sub_9858D0(
                                        v5,
                                        v15,
                                        *(_BYTE **)(v16 - 32LL * (*(_DWORD *)(v16 + 4) & 0x7FFFFFF)),
                                        1) )
                {
                  a2 = v8[2].m128i_i64[1];
                  if ( (unsigned __int8)sub_98CF40(v16, a2, v8[1].m128i_i64[1], 0) )
                    goto LABEL_19;
                }
              }
              v14 += 32;
            }
            while ( v229 != v14 );
          }
        }
      }
      else
      {
        a4 = 1;
        while ( a5 != -4096 )
        {
          v124 = a4 + 1;
          v125 = ((_DWORD)v11 - 1) & (unsigned int)(a2 + a4);
          a2 = (unsigned int)v125;
          a4 = v12 + 88 * v125;
          a5 = *(_QWORD *)(a4 + 24);
          if ( v5 == a5 )
            goto LABEL_12;
          a4 = v124;
        }
      }
    }
LABEL_22:
    v18 = v8[3].m128i_i64[0];
    if ( v18 )
    {
      if ( v8[2].m128i_i64[1] )
      {
        if ( v8[1].m128i_i64[1] )
        {
          v19 = *(unsigned int *)(v18 + 24);
          a4 = *(_QWORD *)(v18 + 8);
          if ( (_DWORD)v19 )
          {
            v20 = (v19 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
            v21 = a4 + 32LL * v20;
            a2 = *(_QWORD *)v21;
            if ( v5 == *(_QWORD *)v21 )
            {
LABEL_27:
              if ( v21 != a4 + 32 * v19 )
              {
                v22 = *(__int64 **)(v21 + 8);
                v217 = &v22[*(unsigned int *)(v21 + 16)];
                if ( v217 != v22 )
                {
                  while ( 1 )
                  {
                    v24 = *v22;
                    v25 = *(_BYTE **)(*v22 - 96);
                    v26 = *(_QWORD *)(*v22 - 32);
                    v255 = *(_QWORD **)(*v22 + 40);
                    v230 = v25;
                    v256 = v26;
                    v27 = sub_9858D0(v5, v228, v25, 1);
                    v28 = v230;
                    if ( v27 )
                    {
                      v29 = sub_B19C20(v8[1].m128i_i64[1], &v255, *(_QWORD *)(v8[2].m128i_i64[1] + 40));
                      v28 = v230;
                      if ( v29 )
                        break;
                    }
                    v23 = *(_QWORD *)(v24 - 64);
                    a2 = v228;
                    v258.m128i_i64[0] = *(_QWORD *)(v24 + 40);
                    v258.m128i_i64[1] = v23;
                    if ( (unsigned __int8)sub_9858D0(v5, v228, v28, 0) )
                    {
                      a2 = (__int64)&v258;
                      if ( (unsigned __int8)sub_B19C20(v8[1].m128i_i64[1], &v258, *(_QWORD *)(v8[2].m128i_i64[1] + 40)) )
                        break;
                    }
                    if ( v217 == ++v22 )
                      goto LABEL_61;
                  }
LABEL_19:
                  LODWORD(v6) = 1;
                  return (unsigned int)v6;
                }
                goto LABEL_61;
              }
            }
            else
            {
              LODWORD(a5) = 1;
              while ( a2 != -4096 )
              {
                v126 = a5 + 1;
                LODWORD(a5) = (v19 - 1) & (v20 + a5);
                v20 = a5;
                v21 = a4 + 32LL * (unsigned int)a5;
                a2 = *(_QWORD *)v21;
                if ( v5 == *(_QWORD *)v21 )
                  goto LABEL_27;
                LODWORD(a5) = v126;
              }
            }
          }
        }
        LOBYTE(v6) = *(_BYTE *)v5;
        if ( *(_BYTE *)v5 <= 0x1Cu )
          goto LABEL_58;
        break;
      }
      LOBYTE(v6) = *(_BYTE *)v5;
LABEL_40:
      v31 = (char)v6;
      if ( (unsigned __int8)v6 <= 0x1Cu )
        goto LABEL_58;
      goto LABEL_41;
    }
LABEL_61:
    LOBYTE(v6) = *(_BYTE *)v5;
LABEL_62:
    v31 = (char)v6;
    if ( (unsigned __int8)v6 <= 0x1Cu )
      goto LABEL_58;
    if ( !v8[2].m128i_i64[1] )
    {
LABEL_41:
      if ( v31 != 54 )
      {
LABEL_42:
        if ( v31 != 55 )
          goto LABEL_43;
        v40 = *(_QWORD *)(v5 - 64);
        if ( *(_BYTE *)v40 == 17 )
        {
          LODWORD(v6) = sub_986B30((__int64 *)(v40 + 24), a2, v40, a4, a5);
          goto LABEL_73;
        }
        v49 = *(_QWORD *)(v40 + 8);
        if ( (unsigned int)*(unsigned __int8 *)(v49 + 8) - 17 <= 1 && *(_BYTE *)v40 <= 0x15u )
        {
          v233 = *(_QWORD *)(v5 - 64);
          v50 = sub_AD7630(v233, 0);
          v53 = v233;
          if ( !v50 || *(_BYTE *)v50 != 17 )
          {
            if ( *(_BYTE *)(v49 + 8) == 17 )
            {
              v146 = *(_DWORD *)(v49 + 32);
              if ( v146 )
              {
                LODWORD(v6) = 0;
                v147 = 0;
                while ( 1 )
                {
                  v239 = v53;
                  v148 = (__int64 *)sub_AD69F0(v53, v147);
                  if ( !v148 )
                    break;
                  v151 = *(unsigned __int8 *)v148;
                  v53 = v239;
                  if ( (_BYTE)v151 != 13 )
                  {
                    if ( (_BYTE)v151 != 17 )
                      break;
                    v152 = sub_986B30(v148 + 3, v151, v239, v149, v150);
                    v53 = v239;
                    LODWORD(v6) = v152;
                    if ( !(_BYTE)v152 )
                      break;
                  }
                  if ( v146 == ++v147 )
                    goto LABEL_73;
                }
              }
            }
            goto LABEL_43;
          }
          LODWORD(v6) = sub_986B30((__int64 *)(v50 + 24), 0, v233, v51, v52);
LABEL_73:
          if ( (_BYTE)v6 )
            return (unsigned int)v6;
        }
LABEL_43:
        v32 = v212 + 1;
        v33 = (int *)sub_C94E20(qword_4F862D0);
        if ( v33 )
          v34 = *v33;
        else
          v34 = qword_4F862D0[2];
        if ( v212 == v34 )
          goto LABEL_58;
        switch ( *(_BYTE *)v5 )
        {
          case '"':
          case 'U':
            if ( !sub_988010(v5) )
              goto LABEL_58;
            v58 = sub_987FE0(v5);
            if ( v58 > 0xB5 )
            {
              if ( v58 > 0x14A )
              {
                if ( v58 - 365 > 1 )
                  goto LABEL_58;
              }
              else if ( v58 <= 0x148 )
              {
                goto LABEL_58;
              }
              if ( !(unsigned __int8)sub_9A1DB0(
                                       *(_QWORD *)(v5 + 32 * (1LL - (*(_DWORD *)(v5 + 4) & 0x7FFFFFF))),
                                       v228,
                                       v32,
                                       v8) )
                goto LABEL_58;
              a4 = (__int64)v8;
              a3 = v212 + 1;
              a2 = v228;
              a1 = *(unsigned __int8 **)(v5 - 32LL * (*(_DWORD *)(v5 + 4) & 0x7FFFFFF));
            }
            else if ( v58 > 0xB3 )
            {
              a1 = *(unsigned __int8 **)(v5 - 32LL * (*(_DWORD *)(v5 + 4) & 0x7FFFFFF));
              if ( *(unsigned __int8 **)(v5 + 32 * (1LL - (*(_DWORD *)(v5 + 4) & 0x7FFFFFF))) != a1 )
                goto LABEL_58;
              a2 = v228;
              a4 = (__int64)v8;
              a3 = v212 + 1;
            }
            else
            {
              if ( v58 - 14 > 1 )
                goto LABEL_58;
              a2 = v228;
              a4 = (__int64)v8;
              a3 = v212 + 1;
              a1 = *(unsigned __int8 **)(v5 - 32LL * (*(_DWORD *)(v5 + 4) & 0x7FFFFFF));
            }
            continue;
          case '*':
            if ( !v228 && (!v8[4].m128i_i8[0] || (*(_BYTE *)(v5 + 1) & 2) == 0 && ((*(_BYTE *)(v5 + 1) >> 1) & 2) == 0) )
              goto LABEL_58;
            v68 = (_QWORD *)sub_986520(v5);
            v69 = (_BYTE *)*v68;
            if ( *(_BYTE *)*v68 == 57 )
            {
              v164 = v68[4];
              if ( v164 == *((_QWORD *)v69 - 8) || v164 == *((_QWORD *)v69 - 4) )
              {
                if ( (unsigned __int8)sub_9A1DB0(v164, v228, v32, v8) )
                  goto LABEL_19;
              }
            }
            v70 = (__int64 *)sub_986520(v5);
            v71 = (_BYTE *)v70[4];
            if ( *v71 == 57 )
            {
              v163 = *v70;
              if ( *v70 == *((_QWORD *)v71 - 8) || v163 == *((_QWORD *)v71 - 4) )
              {
                if ( (unsigned __int8)sub_9A1DB0(v163, v228, v32, v8) )
                  goto LABEL_19;
              }
            }
            v218 = sub_BCB060(*(_QWORD *)(v5 + 8));
            sub_9878D0((__int64)&v255, v218);
            v72 = (_QWORD *)sub_986520(v5);
            sub_9AC0E0(*v72, &v255, v32, v8);
            sub_9878D0((__int64)&v258, v218);
            v73 = sub_986520(v5);
            sub_9AC0E0(*(_QWORD *)(v73 + 32), &v258, v32, v8);
            v74 = (__m128i *)&v255;
            sub_9865C0((__int64)&v249, (__int64)&v255);
            v76 = v250;
            if ( v250 > 0x40 )
            {
              v74 = &v258;
              sub_C43B90(&v249, &v258);
              v76 = v250;
              v77 = v249;
            }
            else
            {
              v77 = v258.m128i_i64[0] & v249;
              v249 &= v258.m128i_i64[0];
            }
            v252 = v76;
            v251 = v77;
            v250 = 0;
            sub_987160((__int64)&v251, (__int64)v74, v76, v75, (__int64)&v251);
            v78 = v252;
            v252 = 0;
            v254 = v78;
            v253 = v251;
            if ( v78 > 0x40 )
            {
              v158 = sub_C44630(&v253);
              sub_969240(&v253);
              sub_969240(&v251);
              sub_969240(&v249);
              if ( v158 == 1 )
              {
LABEL_147:
                if ( v228 || !sub_9867B0((__int64)&v259) || !sub_9867B0((__int64)v257) )
                {
                  sub_969240(v259.m128i_i64);
                  sub_969240(v258.m128i_i64);
                  sub_969240(v257);
                  sub_969240((__int64 *)&v255);
                  goto LABEL_19;
                }
                sub_969240(v259.m128i_i64);
                sub_969240(v258.m128i_i64);
                sub_969240(v257);
                sub_969240((__int64 *)&v255);
                goto LABEL_151;
              }
            }
            else
            {
              if ( v251 && (v251 & (v251 - 1)) == 0 )
              {
                sub_969240(&v253);
                sub_969240(&v251);
                sub_969240(&v249);
                goto LABEL_147;
              }
              sub_969240(&v253);
              sub_969240(&v251);
              sub_969240(&v249);
            }
            sub_969240(v259.m128i_i64);
            sub_969240(v258.m128i_i64);
            sub_969240(v257);
            sub_969240((__int64 *)&v255);
            if ( v228 )
              goto LABEL_153;
LABEL_151:
            if ( !v8[4].m128i_i8[0] || (*(_BYTE *)(v5 + 1) & 2) == 0 )
              goto LABEL_58;
LABEL_153:
            v79 = *(_BYTE *)v5 == 42;
            v258.m128i_i64[0] = 0;
            v259.m128i_i64[0] = 0;
            if ( v79 )
            {
              v80 = *(_BYTE **)(v5 - 64);
              if ( *v80 == 55 )
              {
                if ( (unsigned __int8)sub_995B10(&v258, *((_QWORD *)v80 - 8)) )
                {
                  LODWORD(v6) = sub_993A50(&v259, *(_QWORD *)(v5 - 32));
                  if ( (_BYTE)v6 )
                    return (unsigned int)v6;
                }
              }
            }
LABEL_58:
            LODWORD(v6) = 0;
            return (unsigned int)v6;
          case '.':
            v60 = sub_986520(v5);
            if ( !(unsigned __int8)sub_9A1DB0(*(_QWORD *)(v60 + 32), v228, v32, v8) )
              goto LABEL_58;
            v61 = (_QWORD *)sub_986520(v5);
            if ( !(unsigned __int8)sub_9A1DB0(*v61, v228, v32, v8) )
              goto LABEL_58;
            if ( v228 )
              goto LABEL_19;
            return sub_9B6260(v5, v8, v32);
          case '0':
            goto LABEL_115;
          case '6':
            if ( v228 || v8[4].m128i_i8[0] && ((unsigned __int8)sub_B448F0(v5) || (unsigned __int8)sub_B44900(v5)) )
              goto LABEL_113;
            goto LABEL_58;
          case '7':
            if ( v228 )
              goto LABEL_113;
LABEL_115:
            if ( !v8[4].m128i_i8[0] || (*(_BYTE *)(v5 + 1) & 2) == 0 )
              goto LABEL_58;
            goto LABEL_113;
          case '9':
            if ( v228 )
            {
              v62 = sub_986520(v5);
              if ( (unsigned __int8)sub_9A1DB0(*(_QWORD *)(v62 + 32), 1, v32, v8) )
                goto LABEL_19;
              v63 = (_QWORD *)sub_986520(v5);
              if ( (unsigned __int8)sub_9A1DB0(*v63, 1, v32, v8) )
                goto LABEL_19;
              v64 = *(_QWORD *)(sub_986520(v5) + 32);
              v255 = 0;
              v256 = v64;
              v65 = (unsigned __int8 **)sub_986520(v5);
              if ( sub_99C280((__int64)&v255, 15, *v65) )
                goto LABEL_38;
              v66 = *(_QWORD *)sub_986520(v5);
              v258.m128i_i64[0] = 0;
              v258.m128i_i64[1] = v66;
              v67 = sub_986520(v5);
              if ( sub_99C280((__int64)&v258, 15, *(unsigned __int8 **)(v67 + 32)) )
                goto LABEL_38;
            }
            else
            {
              v153 = *(_QWORD *)(sub_986520(v5) + 32);
              v255 = 0;
              v256 = v153;
              v154 = (unsigned __int8 **)sub_986520(v5);
              if ( sub_99C280((__int64)&v255, 15, *v154)
                || (v155 = *(_QWORD *)sub_986520(v5),
                    v258.m128i_i64[0] = 0,
                    v258.m128i_i64[1] = v155,
                    v156 = sub_986520(v5),
                    sub_99C280((__int64)&v258, 15, *(unsigned __int8 **)(v156 + 32))) )
              {
                v157 = (_QWORD *)sub_986520(v5);
                return sub_9B6260(*v157, v8, v32);
              }
            }
            goto LABEL_58;
          case 'C':
            if ( !v228 )
              goto LABEL_58;
            v59 = (unsigned __int8 **)sub_986520(v5);
            a4 = (__int64)v8;
            a3 = v212 + 1;
            a2 = 1;
            goto LABEL_114;
          case 'D':
LABEL_113:
            v59 = (unsigned __int8 **)sub_986520(v5);
            a2 = v228;
            a4 = (__int64)v8;
            a3 = v212 + 1;
LABEL_114:
            a1 = *v59;
            continue;
          case 'T':
            v81 = _mm_loadu_si128(v8 + 1);
            v82 = _mm_loadu_si128(v8 + 2);
            v83 = _mm_loadu_si128(v8 + 3).m128i_u64[0];
            v84 = v8[4].m128i_i64[0];
            v258 = _mm_loadu_si128(v8);
            v261 = v83;
            v262 = v84;
            v249 = 0;
            v251 = 0;
            v253 = 0;
            v259 = v81;
            v260 = v82;
            LODWORD(v6) = sub_990E50(v5, &v249, &v251, &v253);
            if ( !(_BYTE)v6 )
              goto LABEL_170;
            v85 = 4LL * (*(_DWORD *)(v5 + 4) & 0x7FFFFFF);
            if ( (*(_BYTE *)(v5 + 7) & 0x40) != 0 )
            {
              v86 = *(_QWORD **)(v5 - 8);
              v87 = &v86[v85];
            }
            else
            {
              v86 = (_QWORD *)(v5 - v85 * 8);
              v87 = (_QWORD *)v5;
            }
            v88 = &v258;
            if ( v86 == v87 )
              goto LABEL_239;
            do
            {
              if ( *v86 == v251 )
              {
                v89 = *(_QWORD *)(*(_QWORD *)(v5 - 8)
                                + 32LL * *(unsigned int *)(v5 + 72)
                                + 8LL * (unsigned int)(((__int64)v86 - *(_QWORD *)(v5 - 8)) >> 5));
                v90 = *(_QWORD *)(v89 + 48) & 0xFFFFFFFFFFFFFFF8LL;
                if ( v90 == v89 + 48 )
                {
                  v92 = 0;
                }
                else
                {
                  if ( !v90 )
                    goto LABEL_452;
                  v91 = *(unsigned __int8 *)(v90 - 24);
                  v92 = v90 - 24;
                  if ( (unsigned int)(v91 - 30) >= 0xB )
                    v92 = 0;
                }
                v213 = v87;
                v219 = v86;
                v234 = v88;
                v260.m128i_i64[1] = v92;
                v93 = sub_9A1DB0(v251, v228, v32, v88);
                v88 = v234;
                v86 = v219;
                v87 = v213;
                if ( !v93 )
                  goto LABEL_170;
              }
              v86 += 4;
            }
            while ( v87 != v86 );
LABEL_239:
            if ( *(_BYTE *)v249 != 46 && *(_QWORD *)(v249 - 32) != v253 )
              goto LABEL_170;
            v133 = *(_QWORD *)(v249 + 40);
            v134 = *(_QWORD *)(v133 + 48) & 0xFFFFFFFFFFFFFFF8LL;
            if ( v134 == v133 + 48 )
            {
              v136 = 0;
            }
            else
            {
              if ( !v134 )
                goto LABEL_452;
              v135 = *(unsigned __int8 *)(v134 - 24);
              v136 = v134 - 24;
              if ( (unsigned int)(v135 - 30) >= 0xB )
                v136 = 0;
            }
            break;
          case 'V':
            v120 = sub_986520(v5);
            if ( !(unsigned __int8)sub_9A1DB0(*(_QWORD *)(v120 + 32), v228, v32, v8) )
              goto LABEL_58;
            v121 = sub_986520(v5);
            a4 = (__int64)v8;
            a3 = v212 + 1;
            a2 = v228;
            a1 = *(unsigned __int8 **)(v121 + 64);
            continue;
          default:
            goto LABEL_58;
        }
        v260.m128i_i64[1] = v136;
        switch ( *(_BYTE *)v249 )
        {
          case '.':
            if ( !v228 )
            {
              if ( !(_BYTE)v262 )
                goto LABEL_170;
              v237 = v249;
              if ( !(unsigned __int8)sub_B448F0(v249) && !(unsigned __int8)sub_B44900(v237) )
                goto LABEL_170;
            }
            if ( !(unsigned __int8)sub_9A1DB0(v253, v228, v32, &v258) )
              goto LABEL_170;
            return (unsigned int)v6;
          case '0':
            goto LABEL_263;
          case '1':
            v140 = v251;
            v141 = *(unsigned __int8 *)v251;
            if ( (_BYTE)v141 == 17 )
            {
              if ( *(_DWORD *)(v251 + 32) <= 0x40u )
              {
                v142 = *(_QWORD *)(v251 + 24);
                if ( !v142 )
                  goto LABEL_170;
                v143 = v142 - 1;
                if ( (v142 & (v142 - 1)) != 0 )
                  goto LABEL_170;
                goto LABEL_261;
              }
              v183 = (unsigned int)sub_C44630(v251 + 24) == 1;
              goto LABEL_375;
            }
            v222 = *(_QWORD *)(v251 + 8);
            if ( (unsigned int)*(unsigned __int8 *)(v222 + 8) - 17 > 1 || (unsigned __int8)v141 > 0x15u )
              goto LABEL_170;
            v141 = 0;
            v242 = v251;
            v187 = sub_AD7630(v251, 0);
            v188 = v242;
            v143 = v222;
            if ( v187 && *(_BYTE *)v187 == 17 )
            {
              if ( *(_DWORD *)(v187 + 32) > 0x40u )
              {
                v183 = (unsigned int)sub_C44630(v187 + 24) == 1;
LABEL_375:
                if ( !v183 )
                  goto LABEL_170;
                goto LABEL_376;
              }
              v189 = *(_QWORD *)(v187 + 24);
              if ( !v189 || (v189 & (v189 - 1)) != 0 )
                goto LABEL_170;
            }
            else
            {
              if ( *(_BYTE *)(v222 + 8) != 17 )
                goto LABEL_170;
              LODWORD(v86) = 0;
              v143 = 0;
              v197 = *(_DWORD *)(v222 + 32);
              while ( v197 != (_DWORD)v143 )
              {
                v214 = (unsigned __int8)v86;
                v225 = v143;
                v245 = v188;
                v198 = (unsigned __int8 *)sub_AD69F0(v188, (unsigned int)v143);
                if ( !v198 )
                  goto LABEL_170;
                v141 = *v198;
                v188 = v245;
                v199 = v225;
                LODWORD(v86) = v214;
                if ( (_BYTE)v141 != 13 )
                {
                  if ( (_BYTE)v141 != 17 )
                    goto LABEL_170;
                  if ( *((_DWORD *)v198 + 8) > 0x40u )
                  {
                    if ( (unsigned int)sub_C44630(v198 + 24) != 1 )
                      goto LABEL_170;
                    v188 = v245;
                    v199 = v225;
                    LODWORD(v86) = (_DWORD)v6;
                  }
                  else
                  {
                    v200 = *((_QWORD *)v198 + 3);
                    if ( !v200 )
                      goto LABEL_170;
                    v141 = v200 - 1;
                    if ( (v200 & (v200 - 1)) != 0 )
                      goto LABEL_170;
                    LODWORD(v86) = (_DWORD)v6;
                  }
                }
                v143 = (unsigned int)(v199 + 1);
              }
              if ( !(_BYTE)v86 )
                goto LABEL_170;
            }
LABEL_376:
            v140 = v251;
            if ( *(_BYTE *)v251 == 17 )
            {
LABEL_261:
              v144 = sub_986B30((__int64 *)(v140 + 24), v141, v140, v143, (unsigned int)v86);
              goto LABEL_262;
            }
            v221 = *(_QWORD *)(v251 + 8);
            if ( (unsigned int)*(unsigned __int8 *)(v221 + 8) - 17 <= 1 && *(_BYTE *)v251 <= 0x15u )
            {
              v241 = v251;
              v184 = sub_AD7630(v251, 0);
              v186 = v241;
              if ( v184 && *(_BYTE *)v184 == 17 )
              {
                v144 = sub_986B30((__int64 *)(v184 + 24), 0, v241, v221, v185);
                goto LABEL_262;
              }
              if ( *(_BYTE *)(v221 + 8) == 17 )
              {
                v201 = 0;
                v144 = 0;
                v202 = *(_DWORD *)(v221 + 32);
                while ( v202 != (_DWORD)v201 )
                {
                  v215 = v144;
                  v246 = v186;
                  v203 = sub_AD69F0(v186, v201);
                  if ( !v203 )
                    goto LABEL_263;
                  v186 = v246;
                  v144 = v215;
                  if ( *(_BYTE *)v203 != 13 )
                  {
                    if ( *(_BYTE *)v203 != 17 )
                      goto LABEL_263;
                    v205 = sub_986B30((__int64 *)(v203 + 24), (unsigned int)v201, v246, v215, v204);
                    v186 = v246;
                    v144 = v205;
                    if ( !v205 )
                      goto LABEL_263;
                  }
                  v201 = (unsigned int)(v201 + 1);
                }
LABEL_262:
                if ( v144 )
                  goto LABEL_170;
              }
            }
LABEL_263:
            if ( v228
              || (_BYTE)v262
              && ((v145 = *(unsigned __int8 *)v249, (unsigned int)(v145 - 48) <= 1) || (unsigned __int8)(v145 - 55) <= 1u)
              && (*(_BYTE *)(v249 + 1) & 2) != 0 )
            {
              if ( (unsigned __int8)sub_9A1DB0(v253, 0, v32, &v258) )
                return (unsigned int)v6;
            }
            goto LABEL_170;
          case '6':
            if ( !v228 )
            {
              if ( !(_BYTE)v262 )
                goto LABEL_170;
              v238 = v249;
              if ( !(unsigned __int8)sub_B448F0(v249) && !(unsigned __int8)sub_B44900(v238) )
                goto LABEL_170;
            }
            return (unsigned int)v6;
          case '7':
            goto LABEL_251;
          case '8':
            v137 = v251;
            if ( *(_BYTE *)v251 == 17 )
            {
              if ( *(_DWORD *)(v251 + 32) <= 0x40u )
              {
                v138 = *(_QWORD *)(v251 + 24);
                if ( !v138 || (v138 & (v138 - 1)) != 0 )
                  goto LABEL_170;
                goto LABEL_250;
              }
              v182 = (unsigned int)sub_C44630(v251 + 24) == 1;
            }
            else
            {
              v223 = *(_QWORD *)(v251 + 8);
              if ( (unsigned int)*(unsigned __int8 *)(v223 + 8) - 17 > 1 || *(_BYTE *)v251 > 0x15u )
                goto LABEL_170;
              v243 = v251;
              v190 = sub_AD7630(v251, 0);
              v191 = v243;
              if ( !v190 || *(_BYTE *)v190 != 17 )
              {
                if ( *(_BYTE *)(v223 + 8) != 17 )
                  goto LABEL_170;
                v206 = 0;
                v207 = 0;
                v208 = *(_DWORD *)(v223 + 32);
                while ( v208 != v207 )
                {
                  v216 = v206;
                  v226 = v207;
                  v247 = v191;
                  v209 = sub_AD69F0(v191, v207);
                  if ( !v209 )
                    goto LABEL_170;
                  v191 = v247;
                  v210 = v226;
                  v206 = v216;
                  if ( *(_BYTE *)v209 != 13 )
                  {
                    if ( *(_BYTE *)v209 != 17 )
                      goto LABEL_170;
                    if ( *(_DWORD *)(v209 + 32) > 0x40u )
                    {
                      v227 = v247;
                      v248 = v210;
                      if ( (unsigned int)sub_C44630(v209 + 24) != 1 )
                        goto LABEL_170;
                      v210 = v248;
                      v191 = v227;
                      v206 = (char)v6;
                    }
                    else
                    {
                      v211 = *(_QWORD *)(v209 + 24);
                      if ( !v211 || (v211 & (v211 - 1)) != 0 )
                        goto LABEL_170;
                      v206 = (char)v6;
                    }
                  }
                  v207 = v210 + 1;
                }
                v137 = v251;
                if ( !v206 )
                  goto LABEL_170;
                goto LABEL_250;
              }
              if ( *(_DWORD *)(v190 + 32) <= 0x40u )
              {
                v192 = *(_QWORD *)(v190 + 24);
                if ( !v192 || (v192 & (v192 - 1)) != 0 )
                  goto LABEL_170;
                goto LABEL_373;
              }
              v182 = (unsigned int)sub_C44630(v190 + 24) == 1;
            }
            if ( !v182 )
              goto LABEL_170;
LABEL_373:
            v137 = v251;
LABEL_250:
            v255 = 0;
            if ( !(unsigned __int8)sub_993BE0(&v255, v137) )
            {
LABEL_251:
              if ( v228 )
                return (unsigned int)v6;
              if ( (_BYTE)v262 )
              {
                v139 = *(unsigned __int8 *)v249;
                if ( ((unsigned int)(v139 - 48) <= 1 || (unsigned __int8)(v139 - 55) <= 1u)
                  && (*(_BYTE *)(v249 + 1) & 2) != 0 )
                {
                  return (unsigned int)v6;
                }
              }
            }
LABEL_170:
            v94 = sub_993A20((__int64)qword_4F862D0) - 1;
            if ( v94 < v32 )
              v94 = v32;
            v95 = sub_986550(v5);
            v220 = v96;
            v6 = (char *)v95;
            v97 = (__int64)&v96[-v95] >> 7;
            if ( v97 > 0 )
            {
              v235 = (char *)(v95 + (v97 << 7));
              while ( 1 )
              {
                if ( v5 != *(_QWORD *)v6 )
                {
                  v118 = *(_QWORD *)(*(_QWORD *)(v5 - 8)
                                   + 32LL * *(unsigned int *)(v5 + 72)
                                   + 8LL * (unsigned int)((__int64)&v6[-*(_QWORD *)(v5 - 8)] >> 5));
                  v119 = *(_QWORD *)(v118 + 48) & 0xFFFFFFFFFFFFFFF8LL;
                  if ( v119 == v118 + 48 )
                  {
                    v99 = 0;
                  }
                  else
                  {
                    if ( !v119 )
                      goto LABEL_452;
                    v98 = *(unsigned __int8 *)(v119 - 24);
                    v99 = v119 - 24;
                    if ( (unsigned int)(v98 - 30) >= 0xB )
                      v99 = 0;
                  }
                  v260.m128i_i64[1] = v99;
                  if ( !(unsigned __int8)sub_9A1DB0(*(_QWORD *)v6, v228, v94, &v258) )
                    break;
                }
                if ( v5 != *((_QWORD *)v6 + 4) )
                {
                  v100 = *(_QWORD *)(*(_QWORD *)(v5 - 8)
                                   + 32LL * *(unsigned int *)(v5 + 72)
                                   + 8LL * (unsigned int)((__int64)&v6[-*(_QWORD *)(v5 - 8) + 32] >> 5));
                  v101 = *(_QWORD *)(v100 + 48) & 0xFFFFFFFFFFFFFFF8LL;
                  if ( v101 == v100 + 48 )
                  {
                    v103 = 0;
                  }
                  else
                  {
                    if ( !v101 )
                      goto LABEL_452;
                    v102 = *(unsigned __int8 *)(v101 - 24);
                    v103 = v101 - 24;
                    if ( (unsigned int)(v102 - 30) >= 0xB )
                      v103 = 0;
                  }
                  v260.m128i_i64[1] = v103;
                  v104 = sub_9A1DB0(*((_QWORD *)v6 + 4), v228, v94, &v258);
                  v105 = v6 + 32;
                  if ( !v104 )
                    goto LABEL_294;
                }
                v106 = *((_QWORD *)v6 + 8);
                if ( v5 != v106 )
                {
                  v107 = *(_QWORD *)(*(_QWORD *)(v5 - 8)
                                   + 32LL * *(unsigned int *)(v5 + 72)
                                   + 8LL * (unsigned int)((__int64)&v6[-*(_QWORD *)(v5 - 8) + 64] >> 5));
                  v108 = *(_QWORD *)(v107 + 48) & 0xFFFFFFFFFFFFFFF8LL;
                  if ( v108 == v107 + 48 )
                  {
                    v110 = 0;
                  }
                  else
                  {
                    if ( !v108 )
                      goto LABEL_452;
                    v109 = *(unsigned __int8 *)(v108 - 24);
                    v110 = v108 - 24;
                    if ( (unsigned int)(v109 - 30) >= 0xB )
                      v110 = 0;
                  }
                  v260.m128i_i64[1] = v110;
                  v111 = sub_9A1DB0(v106, v228, v94, &v258);
                  v105 = v6 + 64;
                  if ( !v111 )
                    goto LABEL_294;
                }
                v112 = *((_QWORD *)v6 + 12);
                if ( v5 != v112 )
                {
                  v113 = *(_QWORD *)(*(_QWORD *)(v5 - 8)
                                   + 32LL * *(unsigned int *)(v5 + 72)
                                   + 8LL * (unsigned int)((__int64)&v6[-*(_QWORD *)(v5 - 8) + 96] >> 5));
                  v114 = *(_QWORD *)(v113 + 48) & 0xFFFFFFFFFFFFFFF8LL;
                  if ( v114 == v113 + 48 )
                  {
                    v116 = 0;
                  }
                  else
                  {
                    if ( !v114 )
                      goto LABEL_452;
                    v115 = *(unsigned __int8 *)(v114 - 24);
                    v116 = v114 - 24;
                    if ( (unsigned int)(v115 - 30) >= 0xB )
                      v116 = 0;
                  }
                  v260.m128i_i64[1] = v116;
                  v117 = sub_9A1DB0(v112, v228, v94, &v258);
                  v105 = v6 + 96;
                  if ( !v117 )
                    goto LABEL_294;
                }
                v6 += 128;
                if ( v6 == v235 )
                  goto LABEL_327;
              }
              v105 = v6;
              goto LABEL_294;
            }
            v235 = (char *)v95;
LABEL_327:
            v105 = v220;
            v165 = v220 - v235;
            if ( v220 - v235 != 64 )
            {
              if ( v165 != 96 )
              {
                if ( v165 != 32 )
                {
LABEL_294:
                  LOBYTE(v6) = v220 == v105;
                  return (unsigned int)v6;
                }
                goto LABEL_330;
              }
              if ( v5 != *(_QWORD *)v235 )
              {
                v172 = *(_QWORD *)(*(_QWORD *)(v5 - 8)
                                 + 32LL * *(unsigned int *)(v5 + 72)
                                 + 8LL * (unsigned int)((__int64)&v235[-*(_QWORD *)(v5 - 8)] >> 5));
                v173 = *(_QWORD *)(v172 + 48) & 0xFFFFFFFFFFFFFFF8LL;
                if ( v173 == v172 + 48 )
                {
                  v174 = 0;
                }
                else
                {
                  if ( !v173 )
                    goto LABEL_452;
                  v174 = v173 - 24;
                  if ( (unsigned int)*(unsigned __int8 *)(v173 - 24) - 30 >= 0xB )
                    v174 = 0;
                }
                v169 = v235;
                v260.m128i_i64[1] = v174;
                if ( !(unsigned __int8)sub_9A1DB0(*(_QWORD *)v235, v228, v94, &v258) )
                {
LABEL_337:
                  v105 = v169;
                  goto LABEL_294;
                }
              }
              v235 += 32;
            }
            if ( v5 == *(_QWORD *)v235 )
              goto LABEL_356;
            v175 = *(_QWORD *)(*(_QWORD *)(v5 - 8)
                             + 32LL * *(unsigned int *)(v5 + 72)
                             + 8LL * (unsigned int)((__int64)&v235[-*(_QWORD *)(v5 - 8)] >> 5));
            v176 = *(_QWORD *)(v175 + 48) & 0xFFFFFFFFFFFFFFF8LL;
            if ( v176 == v175 + 48 )
            {
              v177 = 0;
            }
            else
            {
              if ( !v176 )
                goto LABEL_452;
              v177 = v176 - 24;
              if ( (unsigned int)*(unsigned __int8 *)(v176 - 24) - 30 >= 0xB )
                v177 = 0;
            }
            v169 = v235;
            v260.m128i_i64[1] = v177;
            if ( (unsigned __int8)sub_9A1DB0(*(_QWORD *)v235, v228, v94, &v258) )
            {
LABEL_356:
              v235 += 32;
LABEL_330:
              if ( v5 == *(_QWORD *)v235 )
              {
                v105 = v220;
                goto LABEL_294;
              }
              v166 = *(_QWORD *)(*(_QWORD *)(v5 - 8)
                               + 32LL * *(unsigned int *)(v5 + 72)
                               + 8LL * (unsigned int)((__int64)&v235[-*(_QWORD *)(v5 - 8)] >> 5));
              v167 = *(_QWORD *)(v166 + 48) & 0xFFFFFFFFFFFFFFF8LL;
              if ( v167 == v166 + 48 )
              {
                v168 = 0;
LABEL_335:
                v169 = v235;
                v260.m128i_i64[1] = v168;
                if ( (unsigned __int8)sub_9A1DB0(*(_QWORD *)v235, v228, v94, &v258) )
                  v169 = v220;
                goto LABEL_337;
              }
              if ( v167 )
              {
                v168 = v167 - 24;
                if ( (unsigned int)*(unsigned __int8 *)(v167 - 24) - 30 >= 0xB )
                  v168 = 0;
                goto LABEL_335;
              }
LABEL_452:
              BUG();
            }
            goto LABEL_337;
          default:
            goto LABEL_170;
        }
      }
      v37 = *(_QWORD *)(v5 - 64);
      if ( *(_BYTE *)v37 == 17 )
      {
        v38 = *(_DWORD *)(v37 + 32);
        if ( v38 <= 0x40 )
          v39 = *(_QWORD *)(v37 + 24) == 1;
        else
          v39 = v38 - 1 == (unsigned int)sub_C444A0(v37 + 24);
      }
      else
      {
        v45 = *(_QWORD *)(v37 + 8);
        if ( (unsigned int)*(unsigned __int8 *)(v45 + 8) - 17 > 1 || *(_BYTE *)v37 > 0x15u )
          goto LABEL_43;
        a2 = 0;
        v232 = *(_QWORD *)(v5 - 64);
        v46 = sub_AD7630(v232, 0);
        v47 = v232;
        if ( !v46 || *(_BYTE *)v46 != 17 )
        {
          if ( *(_BYTE *)(v45 + 8) == 17 )
          {
            v127 = *(_DWORD *)(v45 + 32);
            if ( v127 )
            {
              v128 = 0;
              for ( i = 0; i != v127; ++i )
              {
                a2 = i;
                v236 = v47;
                v130 = (unsigned __int8 *)sub_AD69F0(v47, i);
                v47 = v236;
                if ( !v130 )
                  goto LABEL_69;
                a2 = *v130;
                if ( (_BYTE)a2 != 13 )
                {
                  if ( (_BYTE)a2 != 17 )
                    goto LABEL_69;
                  v131 = *((_DWORD *)v130 + 8);
                  if ( v131 <= 0x40 )
                  {
                    v128 = *((_QWORD *)v130 + 3) == 1;
                  }
                  else
                  {
                    v132 = sub_C444A0(v130 + 24);
                    v47 = v236;
                    v128 = v131 - 1 == v132;
                  }
                  if ( !v128 )
                    goto LABEL_69;
                }
              }
              if ( v128 )
                goto LABEL_19;
            }
          }
          goto LABEL_69;
        }
        v48 = *(_DWORD *)(v46 + 32);
        if ( v48 <= 0x40 )
          v39 = *(_QWORD *)(v46 + 24) == 1;
        else
          v39 = v48 - 1 == (unsigned int)sub_C444A0(v46 + 24);
      }
      if ( v39 )
        goto LABEL_19;
LABEL_69:
      v31 = *(_BYTE *)v5;
      goto LABEL_42;
    }
    break;
  }
  if ( (_BYTE)v6 != 85 )
  {
    if ( (_BYTE)v6 != 76 )
    {
LABEL_56:
      v31 = (char)v6;
      goto LABEL_41;
    }
    if ( (*(_BYTE *)(v5 + 7) & 0x40) != 0 )
      v41 = *(_QWORD **)(v5 - 8);
    else
      v41 = (_QWORD *)(v5 - 32LL * (*(_DWORD *)(v5 + 4) & 0x7FFFFFF));
    v42 = (char *)*v41;
    if ( !*v41 )
      goto LABEL_43;
    v43 = *v42;
    if ( (unsigned __int8)*v42 <= 0x1Cu )
    {
      if ( v43 != 5 || *((_WORD *)v42 + 1) != 34 )
        goto LABEL_43;
    }
    else if ( v43 != 63 )
    {
      goto LABEL_43;
    }
    v44 = sub_BB5290(*v41, a2, v41);
    if ( *(_BYTE *)(v44 + 8) != 18 )
      goto LABEL_83;
    if ( (*((_DWORD *)v42 + 1) & 0x7FFFFFF) != 2 )
      goto LABEL_83;
    a2 = 8;
    if ( !(unsigned __int8)sub_BCAC40(*(_QWORD *)(v44 + 24), 8) )
      goto LABEL_83;
    a4 = *(_QWORD *)&v42[-32 * (*((_DWORD *)v42 + 1) & 0x7FFFFFF)];
    if ( *(_BYTE *)a4 > 0x15u )
      goto LABEL_83;
    v240 = *(_QWORD *)&v42[-32 * (*((_DWORD *)v42 + 1) & 0x7FFFFFF)];
    v159 = sub_AC30F0(v240);
    a4 = v240;
    if ( !v159 )
    {
      if ( *(_BYTE *)v240 == 17 )
      {
        v170 = *(_DWORD *)(v240 + 32);
        if ( v170 <= 0x40 )
          v171 = *(_QWORD *)(v240 + 24) == 0;
        else
          v171 = v170 == (unsigned int)sub_C444A0(v240 + 24);
        if ( !v171 )
          goto LABEL_83;
      }
      else
      {
        v179 = *(_QWORD *)(v240 + 8);
        if ( (unsigned int)*(unsigned __int8 *)(v179 + 8) - 17 > 1 )
          goto LABEL_83;
        a2 = 0;
        v180 = (_BYTE *)sub_AD7630(v240, 0);
        a4 = v240;
        v181 = 0;
        if ( v180 && *v180 == 17 )
        {
          v181 = sub_9867B0((__int64)(v180 + 24));
        }
        else
        {
          if ( *(_BYTE *)(v179 + 8) != 17 )
            goto LABEL_83;
          v193 = *(_DWORD *)(v179 + 32);
          for ( j = 0; v193 != j; ++j )
          {
            a2 = j;
            v224 = v181;
            v244 = a4;
            v195 = (_BYTE *)sub_AD69F0(a4, j);
            if ( !v195 )
              goto LABEL_83;
            a4 = v244;
            v181 = v224;
            if ( *v195 != 13 )
            {
              if ( *v195 != 17 )
                goto LABEL_83;
              v196 = sub_9867B0((__int64)(v195 + 24));
              a4 = v244;
              v181 = v196;
              if ( !v196 )
                goto LABEL_83;
            }
          }
        }
        if ( !v181 )
          goto LABEL_83;
      }
    }
    v160 = *(_BYTE **)&v42[32 * (1LL - (*((_DWORD *)v42 + 1) & 0x7FFFFFF))];
    if ( *v160 != 17 )
    {
      if ( (unsigned int)*(unsigned __int8 *)(*((_QWORD *)v160 + 1) + 8LL) - 17 > 1 )
        goto LABEL_83;
      if ( *v160 > 0x15u )
        goto LABEL_83;
      a2 = 0;
      v178 = (_BYTE *)sub_AD7630(v160, 0);
      v160 = v178;
      if ( !v178 || *v178 != 17 )
        goto LABEL_83;
    }
    v161 = *((_DWORD *)v160 + 8);
    if ( v161 <= 0x40 )
    {
      v162 = *((_QWORD *)v160 + 3);
      goto LABEL_305;
    }
    if ( v161 - (unsigned int)sub_C444A0(v160 + 24) <= 0x40 )
    {
      v162 = **((_QWORD **)v160 + 3);
LABEL_305:
      if ( v162 == 1 )
        goto LABEL_206;
    }
LABEL_83:
    v31 = *(_BYTE *)v5;
    goto LABEL_41;
  }
  v122 = *(_QWORD *)(v5 - 32);
  if ( !v122 || *(_BYTE *)v122 || *(_QWORD *)(v122 + 24) != *(_QWORD *)(v5 + 80) || *(_DWORD *)(v122 + 36) != 493 )
    goto LABEL_56;
LABEL_206:
  v123 = sub_B43CB0(v8[2].m128i_i64[1]);
  return sub_B2D610(v123, 96);
}
