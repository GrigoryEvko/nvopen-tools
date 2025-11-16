// Function: sub_33A7C00
// Address: 0x33a7c00
//
void __fastcall sub_33A7C00(
        __int64 a1,
        unsigned __int8 *a2,
        __int64 a3,
        __int64 a4,
        __int8 a5,
        unsigned __int8 a6,
        __int64 a7,
        __int64 *a8)
{
  __int64 v12; // rax
  int v13; // edx
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rbx
  int v19; // ebx
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rax
  __int64 v23; // rsi
  int v24; // edx
  __int64 *v25; // rbx
  __int64 v26; // r13
  __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // r14
  int v30; // r14d
  __int64 v31; // rax
  __int64 v32; // rdx
  __int64 v33; // r13
  __int64 v34; // rax
  __int64 v35; // rdx
  __int64 (*v36)(); // rax
  __int8 v37; // cl
  __int64 v38; // rax
  __int64 v39; // rdx
  __int64 v40; // rbx
  __int64 v41; // rbx
  __int64 v42; // r13
  __int64 v43; // rbx
  __int64 v44; // rax
  unsigned int *v45; // rax
  int v46; // edx
  __int64 v47; // rax
  __int64 v48; // rbx
  __int64 v49; // rax
  __int64 v50; // rdx
  __int64 v51; // rax
  __int64 (*v52)(); // rax
  bool v53; // zf
  __int64 v54; // r8
  __int64 v55; // r9
  char v56; // al
  __int64 v57; // rax
  __int64 v58; // rdx
  __int64 v59; // rbx
  char v60; // dl
  __int64 v61; // rax
  __int64 v62; // rbx
  __int64 v63; // r13
  __int64 v64; // rbx
  __int64 v65; // rax
  unsigned int *v66; // rax
  __int64 v67; // rbx
  __int64 v68; // r13
  __int64 (*v69)(); // rax
  __int64 v70; // rax
  __int64 v71; // rdx
  __int64 v72; // rbx
  __int64 v73; // rbx
  __int64 v74; // r13
  __int64 v75; // rbx
  __int64 v76; // rax
  unsigned int *v77; // rax
  int v78; // edx
  __int64 v79; // rax
  __int64 m128i_i64; // rdx
  _BYTE *v81; // rcx
  __m128i *v82; // rax
  __int64 v83; // rax
  __int64 v84; // rdx
  char v85; // al
  __int64 v86; // rdx
  char v87; // al
  char v88; // al
  __int64 v89; // rdx
  char v90; // al
  __int64 v91; // rdx
  char v92; // al
  unsigned __int64 v93; // rdi
  const __m128i *v94; // rax
  __m128i *v95; // rax
  const __m128i *v96; // rax
  char v97; // al
  __int64 v98; // r8
  __int64 v99; // r9
  bool v100; // dl
  bool v101; // sf
  __int64 v102; // rax
  __int64 v103; // rdx
  __int64 v104; // rbx
  __int64 v105; // rbx
  __int64 v106; // r14
  int v107; // r13d
  __int64 v108; // rbx
  __int64 v109; // rax
  bool v110; // al
  __int64 (*v111)(); // rax
  __int64 v112; // rdx
  int v113; // eax
  __int64 v114; // rax
  __int64 v115; // rdx
  _QWORD *v116; // rax
  __int64 (*v117)(); // rax
  __int64 v118; // rdx
  __int64 (__fastcall *v119)(__int64, __int64, unsigned int); // rax
  int v120; // edx
  __int16 v121; // ax
  __int64 v122; // rcx
  unsigned int v123; // eax
  __int64 v124; // rax
  __int64 v125; // rdx
  __int64 v126; // rbx
  _DWORD *v127; // rax
  __int64 v128; // rdx
  __int64 (*v129)(); // rax
  char v130; // al
  __int8 v131; // cl
  char v132; // al
  __int8 v133; // cl
  __int8 v134; // cl
  __int64 v135; // rdi
  unsigned __int64 v136; // rax
  __int64 v137; // r10
  __int64 v138; // rbx
  __int64 v139; // rax
  __int64 v140; // rsi
  __int64 v141; // r14
  __int64 v142; // r15
  __int64 v143; // rax
  unsigned __int16 *v144; // rbx
  __int128 v145; // rax
  __int64 v146; // rax
  __int64 v147; // rdx
  __int64 v148; // r12
  __int64 v149; // rbx
  __int64 v150; // r14
  __int128 v151; // [rsp-30h] [rbp-1300h]
  __int64 v154; // [rsp+38h] [rbp-1298h]
  __int64 v155; // [rsp+40h] [rbp-1290h]
  __int64 v156; // [rsp+40h] [rbp-1290h]
  __int64 v157; // [rsp+48h] [rbp-1288h]
  __int64 v158; // [rsp+50h] [rbp-1280h]
  __int64 v159; // [rsp+50h] [rbp-1280h]
  __int64 v160; // [rsp+58h] [rbp-1278h]
  int v161; // [rsp+58h] [rbp-1278h]
  __int8 v162; // [rsp+6Fh] [rbp-1261h]
  __int64 v163; // [rsp+70h] [rbp-1260h]
  __int128 v164; // [rsp+70h] [rbp-1260h]
  __m128i v165; // [rsp+80h] [rbp-1250h] BYREF
  __int64 v166; // [rsp+90h] [rbp-1240h]
  __int64 v167; // [rsp+98h] [rbp-1238h]
  __int64 v168; // [rsp+A0h] [rbp-1230h]
  __int64 v169; // [rsp+A8h] [rbp-1228h]
  __m128i v170; // [rsp+B0h] [rbp-1220h]
  __int64 v171; // [rsp+C0h] [rbp-1210h]
  __int64 v172; // [rsp+C8h] [rbp-1208h]
  __int64 v173; // [rsp+D0h] [rbp-1200h]
  __int64 v174; // [rsp+D8h] [rbp-11F8h]
  __int64 v175; // [rsp+E0h] [rbp-11F0h]
  __int64 v176; // [rsp+E8h] [rbp-11E8h]
  __int64 v177; // [rsp+F0h] [rbp-11E0h]
  __int64 v178; // [rsp+F8h] [rbp-11D8h]
  __int64 v179; // [rsp+100h] [rbp-11D0h]
  __int64 v180; // [rsp+108h] [rbp-11C8h]
  __int64 v181; // [rsp+110h] [rbp-11C0h]
  __int64 v182; // [rsp+118h] [rbp-11B8h]
  unsigned __int8 *v183; // [rsp+128h] [rbp-11A8h] BYREF
  const __m128i *v184; // [rsp+130h] [rbp-11A0h] BYREF
  __m128i *v185; // [rsp+138h] [rbp-1198h]
  const __m128i *v186; // [rsp+140h] [rbp-1190h]
  __m128i v187; // [rsp+150h] [rbp-1180h] BYREF
  __int64 v188; // [rsp+160h] [rbp-1170h]
  __int64 v189; // [rsp+168h] [rbp-1168h]
  __m128i v190; // [rsp+170h] [rbp-1160h] BYREF
  __m128i v191; // [rsp+180h] [rbp-1150h] BYREF
  __m128i v192; // [rsp+190h] [rbp-1140h] BYREF
  __int64 v193; // [rsp+1A0h] [rbp-1130h]
  const __m128i *v194; // [rsp+1A8h] [rbp-1128h]
  __m128i *v195; // [rsp+1B0h] [rbp-1120h]
  const __m128i *v196; // [rsp+1B8h] [rbp-1118h]
  __int64 v197; // [rsp+1C0h] [rbp-1110h]
  __int64 v198; // [rsp+1C8h] [rbp-1108h] BYREF
  __int32 v199; // [rsp+1D0h] [rbp-1100h]
  unsigned __int8 *v200; // [rsp+1D8h] [rbp-10F8h]
  _BYTE *v201; // [rsp+1E0h] [rbp-10F0h]
  __int64 v202; // [rsp+1E8h] [rbp-10E8h]
  _BYTE v203[1792]; // [rsp+1F0h] [rbp-10E0h] BYREF
  _BYTE *v204; // [rsp+8F0h] [rbp-9E0h]
  __int64 v205; // [rsp+8F8h] [rbp-9D8h]
  _BYTE v206[512]; // [rsp+900h] [rbp-9D0h] BYREF
  _BYTE *v207; // [rsp+B00h] [rbp-7D0h]
  __int64 v208; // [rsp+B08h] [rbp-7C8h]
  _BYTE v209[1792]; // [rsp+B10h] [rbp-7C0h] BYREF
  _BYTE *v210; // [rsp+1210h] [rbp-C0h]
  __int64 v211; // [rsp+1218h] [rbp-B8h]
  _BYTE v212[64]; // [rsp+1220h] [rbp-B0h] BYREF
  __int64 v213; // [rsp+1260h] [rbp-70h]
  __int64 v214; // [rsp+1268h] [rbp-68h]
  int v215; // [rsp+1270h] [rbp-60h]
  __int64 v216; // [rsp+1278h] [rbp-58h]
  __int64 v217; // [rsp+1280h] [rbp-50h]
  int v218; // [rsp+1288h] [rbp-48h]
  char v219; // [rsp+1290h] [rbp-40h]

  v162 = a5;
  v12 = sub_2E79000(*(__int64 **)(*(_QWORD *)(a1 + 864) + 40LL));
  v13 = *a2;
  v184 = 0;
  v155 = v12;
  v14 = *((_QWORD *)a2 + 10);
  v185 = 0;
  v157 = v14;
  v15 = *((_QWORD *)a2 + 1);
  v186 = 0;
  v154 = v15;
  if ( v13 == 40 )
  {
    v165.m128i_i64[0] = 32LL * (unsigned int)sub_B491D0((__int64)a2);
  }
  else
  {
    v165.m128i_i64[0] = 0;
    if ( v13 != 85 )
    {
      v165.m128i_i64[0] = 64;
      if ( v13 != 34 )
LABEL_194:
        BUG();
    }
  }
  if ( (a2[7] & 0x80u) == 0 )
    goto LABEL_10;
  v16 = sub_BD2BC0((__int64)a2);
  v18 = v16 + v17;
  if ( (a2[7] & 0x80u) == 0 )
  {
    if ( (unsigned int)(v18 >> 4) )
LABEL_196:
      BUG();
LABEL_10:
    v22 = 0;
    goto LABEL_11;
  }
  if ( !(unsigned int)((v18 - sub_BD2BC0((__int64)a2)) >> 4) )
    goto LABEL_10;
  if ( (a2[7] & 0x80u) == 0 )
    goto LABEL_196;
  v19 = *(_DWORD *)(sub_BD2BC0((__int64)a2) + 8);
  if ( (a2[7] & 0x80u) == 0 )
    BUG();
  v20 = sub_BD2BC0((__int64)a2);
  v22 = 32LL * (unsigned int)(*(_DWORD *)(v20 + v21 - 4) - v19);
LABEL_11:
  v23 = (unsigned int)((32LL * (*((_DWORD *)a2 + 1) & 0x7FFFFFF) - 32 - v165.m128i_i64[0] - v22) >> 5);
  sub_3375F60(&v184, v23);
  v163 = *(_QWORD *)(*(_QWORD *)(a1 + 864) + 16LL);
  if ( a5 )
  {
    v23 = (__int64)"disable-tail-calls";
    v126 = *(_QWORD *)(*((_QWORD *)a2 + 5) + 72LL);
    v190.m128i_i64[0] = sub_B2D7E0(v126, "disable-tail-calls", 0x12u);
    v127 = (_DWORD *)sub_A72240(v190.m128i_i64);
    if ( v128 == 4 )
    {
      v134 = v162;
      if ( !(a6 | (*v127 != 1702195828)) )
        v134 = 0;
      v162 = v134;
    }
    v129 = *(__int64 (**)())(*(_QWORD *)v163 + 2216LL);
    if ( v129 != sub_302E1B0 && ((unsigned __int8 (__fastcall *)(__int64))v129)(v163) )
    {
      v23 = 74;
      v190.m128i_i64[0] = *(_QWORD *)(v126 + 120);
      v130 = sub_A74390(v190.m128i_i64, 74, 0);
      v131 = v162;
      if ( v130 )
        v131 = 0;
      v162 = v131;
    }
  }
  v24 = *a2;
  v25 = (__int64 *)&a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
  if ( v24 == 40 )
  {
    v26 = -32 - 32LL * (unsigned int)sub_B491D0((__int64)a2);
  }
  else
  {
    v26 = -32;
    if ( v24 != 85 )
    {
      v26 = -96;
      if ( v24 != 34 )
        goto LABEL_194;
    }
  }
  if ( (a2[7] & 0x80u) == 0 )
  {
    v165.m128i_i64[0] = (__int64)&a2[v26];
    if ( v25 == (__int64 *)&a2[v26] )
    {
      v158 = 0;
      goto LABEL_52;
    }
    goto LABEL_26;
  }
  v27 = sub_BD2BC0((__int64)a2);
  v29 = v27 + v28;
  if ( (a2[7] & 0x80u) == 0 )
  {
    if ( !(unsigned int)(v29 >> 4) )
      goto LABEL_25;
LABEL_193:
    BUG();
  }
  if ( !(unsigned int)((v29 - sub_BD2BC0((__int64)a2)) >> 4) )
    goto LABEL_25;
  if ( (a2[7] & 0x80u) == 0 )
    goto LABEL_193;
  v30 = *(_DWORD *)(sub_BD2BC0((__int64)a2) + 8);
  if ( (a2[7] & 0x80u) == 0 )
    BUG();
  v31 = sub_BD2BC0((__int64)a2);
  v26 -= 32LL * (unsigned int)(*(_DWORD *)(v31 + v32 - 4) - v30);
LABEL_25:
  v165.m128i_i64[0] = (__int64)&a2[v26];
  if ( v25 == (__int64 *)&a2[v26] )
  {
    v158 = 0;
    goto LABEL_39;
  }
LABEL_26:
  v158 = 0;
  do
  {
    v190 = 0u;
    v33 = *v25;
    v191 = 0u;
    v192 = 0u;
    if ( !(unsigned __int8)sub_BCADB0(*(_QWORD *)(v33 + 8)) )
    {
      v34 = sub_338B750(a1, v33);
      v182 = v35;
      v181 = v34;
      v190.m128i_i64[1] = v34;
      v191.m128i_i32[0] = v35;
      v191.m128i_i64[1] = *(_QWORD *)(v33 + 8);
      sub_34470B0(&v190, a2, ((char *)v25 - (char *)&a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)]) >> 5);
      if ( (v192.m128i_i8[1] & 0x20) != 0 )
      {
        v36 = *(__int64 (**)())(*(_QWORD *)v163 + 2216LL);
        if ( v36 != sub_302E1B0 )
        {
          if ( ((unsigned __int8 (__fastcall *)(__int64))v36)(v163) )
          {
            v159 = *(_QWORD *)(a1 + 864);
            v119 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)v163 + 32LL);
            if ( v119 == sub_2D42F30 )
            {
              v120 = sub_AE2980(v155, 0)[1];
              v121 = 2;
              if ( v120 != 1 )
              {
                v121 = 3;
                if ( v120 != 2 )
                {
                  v121 = 4;
                  if ( v120 != 4 )
                  {
                    v121 = 5;
                    if ( v120 != 8 )
                    {
                      v121 = 6;
                      if ( v120 != 16 )
                      {
                        v121 = 7;
                        if ( v120 != 32 )
                        {
                          v121 = 8;
                          if ( v120 != 64 )
                            v121 = 9 * (v120 == 128);
                        }
                      }
                    }
                  }
                }
              }
            }
            else
            {
              v121 = v119(v163, v155, 0);
            }
            v122 = v160;
            LOWORD(v122) = v121;
            v160 = v122;
            v123 = sub_35D5F90(*(_QWORD *)(a1 + 968), a2, *(_QWORD *)(*(_QWORD *)(a1 + 960) + 744LL), v33);
            v124 = sub_33F0B60(v159, v123, (unsigned int)v160, 0);
            v158 = v33;
            v179 = v124;
            v180 = v125;
            v190.m128i_i64[1] = v124;
            v191.m128i_i32[0] = v125;
          }
        }
      }
      v23 = (__int64)v185;
      if ( v185 == v186 )
      {
        sub_332CDC0((unsigned __int64 *)&v184, v185, &v190);
      }
      else
      {
        if ( v185 )
        {
          *v185 = _mm_loadu_si128(&v190);
          *(__m128i *)(v23 + 16) = _mm_loadu_si128(&v191);
          *(__m128i *)(v23 + 32) = _mm_loadu_si128(&v192);
          v23 = (__int64)v185;
        }
        v23 += 48;
        v185 = (__m128i *)v23;
      }
      if ( (v192.m128i_i8[0] & 0x10) != 0 )
      {
        v37 = v162;
        if ( *(_BYTE *)v33 >= 0x1Du )
          v37 = 0;
        v162 = v37;
      }
    }
    v25 += 4;
  }
  while ( v25 != (__int64 *)v165.m128i_i64[0] );
LABEL_39:
  if ( (a2[7] & 0x80u) != 0 )
  {
    v38 = sub_BD2BC0((__int64)a2);
    v40 = v38 + v39;
    if ( (a2[7] & 0x80u) != 0 )
      v40 -= sub_BD2BC0((__int64)a2);
    v41 = v40 >> 4;
    if ( (_DWORD)v41 )
    {
      v42 = 0;
      v43 = 16LL * (unsigned int)v41;
      while ( 1 )
      {
        v44 = 0;
        if ( (a2[7] & 0x80u) != 0 )
          v44 = sub_BD2BC0((__int64)a2);
        v45 = (unsigned int *)(v42 + v44);
        if ( *(_DWORD *)(*(_QWORD *)v45 + 8LL) == 3 )
          break;
        v42 += 16;
        if ( v42 == v43 )
          goto LABEL_52;
      }
      v46 = *((_DWORD *)a2 + 1);
      v47 = v45[2];
      v190 = 0u;
      v191 = 0u;
      v192 = 0u;
      v48 = *(_QWORD *)&a2[32 * (v47 - (v46 & 0x7FFFFFF))];
      v49 = sub_338B750(a1, v48);
      v23 = (__int64)v185;
      v177 = v49;
      v178 = v50;
      v190.m128i_i64[1] = v49;
      v191.m128i_i32[0] = v50;
      v51 = *(_QWORD *)(v48 + 8);
      v192.m128i_i8[1] |= 0x40u;
      v191.m128i_i64[1] = v51;
      if ( v185 == v186 )
      {
        sub_332CDC0((unsigned __int64 *)&v184, v185, &v190);
      }
      else
      {
        if ( v185 )
        {
          *v185 = _mm_loadu_si128(&v190);
          *(__m128i *)(v23 + 16) = _mm_loadu_si128(&v191);
          *(__m128i *)(v23 + 32) = _mm_loadu_si128(&v192);
          v23 = (__int64)v185;
        }
        v23 += 48;
        v185 = (__m128i *)v23;
      }
    }
  }
LABEL_52:
  if ( v162 )
  {
    v23 = **(_QWORD **)(a1 + 864);
    v162 = sub_34B9AF0(a2, v23, 0);
  }
  v52 = *(__int64 (**)())(*(_QWORD *)v163 + 2216LL);
  if ( v52 != sub_302E1B0 )
  {
    v132 = ((__int64 (__fastcall *)(__int64))v52)(v163);
    if ( v158 )
    {
      v133 = v162;
      if ( v132 )
        v133 = 0;
      v162 = v133;
    }
  }
  v53 = !sub_B491E0((__int64)a2);
  v56 = a2[7];
  if ( v53 || v56 >= 0 )
    goto LABEL_68;
  v57 = sub_BD2BC0((__int64)a2);
  v59 = v57 + v58;
  v60 = a2[7];
  v56 = v60;
  if ( v60 >= 0 )
  {
    v62 = v59 >> 4;
  }
  else
  {
    v61 = sub_BD2BC0((__int64)a2);
    v60 = a2[7];
    v62 = (v59 - v61) >> 4;
    v56 = v60;
  }
  if ( !(_DWORD)v62 )
  {
LABEL_68:
    v156 = 0;
  }
  else
  {
    v63 = 0;
    v64 = 16LL * (unsigned int)v62;
    while ( 1 )
    {
      v65 = 0;
      if ( v60 < 0 )
        v65 = sub_BD2BC0((__int64)a2);
      v66 = (unsigned int *)(v63 + v65);
      if ( *(_DWORD *)(*(_QWORD *)v66 + 8LL) == 8 )
        break;
      v63 += 16;
      v60 = a2[7];
      if ( v63 == v64 )
      {
        v56 = a2[7];
        goto LABEL_68;
      }
    }
    v67 = v66[2];
    v68 = *((_DWORD *)a2 + 1) & 0x7FFFFFF;
    v69 = *(__int64 (**)())(*(_QWORD *)v163 + 2232LL);
    if ( v69 == sub_302E1D0 || !((unsigned __int8 (__fastcall *)(__int64))v69)(v163) )
      sub_C64ED0("Target doesn't support calls with kcfi operand bundles.", 1u);
    v156 = *(_QWORD *)&a2[32 * (v67 - v68)];
    v56 = a2[7];
  }
  if ( v56 >= 0 )
    goto LABEL_80;
  v70 = sub_BD2BC0((__int64)a2);
  v72 = v70 + v71;
  if ( (a2[7] & 0x80u) != 0 )
    v72 -= sub_BD2BC0((__int64)a2);
  v73 = v72 >> 4;
  if ( (_DWORD)v73 )
  {
    v74 = 0;
    v75 = 16LL * (unsigned int)v73;
    while ( 1 )
    {
      v76 = 0;
      if ( (a2[7] & 0x80u) != 0 )
        v76 = sub_BD2BC0((__int64)a2);
      v77 = (unsigned int *)(v74 + v76);
      if ( *(_DWORD *)(*(_QWORD *)v77 + 8LL) == 9 )
        break;
      v74 += 16;
      if ( v75 == v74 )
        goto LABEL_80;
    }
    v23 = *(_QWORD *)&a2[32 * (v77[2] - (unsigned __int64)(*((_DWORD *)a2 + 1) & 0x7FFFFFF))];
    v165.m128i_i64[0] = sub_338B750(a1, v23);
    v161 = v78;
  }
  else
  {
LABEL_80:
    v161 = 0;
    v165.m128i_i64[0] = 0;
  }
  v79 = *(_QWORD *)(a1 + 864);
  v219 = 0;
  v191.m128i_i64[1] = 0xFFFFFFFF00000020LL;
  m128i_i64 = *(unsigned int *)(a1 + 848);
  v197 = v79;
  v201 = v203;
  v202 = 0x2000000000LL;
  v205 = 0x2000000000LL;
  v208 = 0x2000000000LL;
  v204 = v206;
  v81 = v209;
  v210 = v212;
  v190 = 0u;
  v191.m128i_i64[0] = 0;
  v192 = 0u;
  v193 = 0;
  v194 = 0;
  v195 = 0;
  v196 = 0;
  v198 = 0;
  v199 = 0;
  v200 = 0;
  v207 = v209;
  v211 = 0x400000000LL;
  v213 = 0;
  v214 = 0;
  v215 = 0;
  v82 = *(__m128i **)a1;
  v187.m128i_i64[0] = 0;
  v187.m128i_i32[2] = m128i_i64;
  if ( v82 )
  {
    m128i_i64 = (__int64)v82[3].m128i_i64;
    if ( &v187 != &v82[3] )
    {
      v23 = v82[3].m128i_i64[0];
      v187.m128i_i64[0] = v23;
      if ( v23 )
      {
        sub_B96E90((__int64)&v187, v23, 1);
        if ( v198 )
          sub_B91220((__int64)&v198, v198);
        v23 = v187.m128i_i64[0];
        v198 = v187.m128i_i64[0];
        if ( v187.m128i_i64[0] )
          sub_B96E90((__int64)&v198, v187.m128i_i64[0], 1);
      }
    }
  }
  v199 = v187.m128i_i32[2];
  v83 = sub_33738B0(a1, v23, m128i_i64, (__int64)v81, v54, v55);
  v176 = v84;
  v175 = v83;
  v190.m128i_i64[0] = v83;
  v190.m128i_i32[2] = v84;
  v191.m128i_i64[0] = v154;
  v85 = sub_A74710((_QWORD *)a2 + 9, 0, 15);
  if ( !v85 )
  {
    v86 = *((_QWORD *)a2 - 4);
    if ( v86 )
    {
      if ( !*(_BYTE *)v86 && *(_QWORD *)(v86 + 24) == *((_QWORD *)a2 + 10) )
      {
        v183 = *(unsigned __int8 **)(v86 + 120);
        v85 = sub_A74710(&v183, 0, 15);
      }
    }
  }
  v191.m128i_i8[8] = v191.m128i_i8[8] & 0xF7 | (8 * (v85 & 1));
  if ( (unsigned __int8)sub_A73ED0((_QWORD *)a2 + 9, 36) || (v87 = sub_B49560((__int64)a2, 36)) != 0 )
  {
    v87 = 1;
  }
  else if ( *a2 != 34 )
  {
    v118 = *((_QWORD *)a2 + 4);
    if ( v118 == *((_QWORD *)a2 + 5) + 48LL || !v118 )
      BUG();
    v87 = *(_BYTE *)(v118 - 24) == 36;
  }
  v53 = *((_QWORD *)a2 + 2) == 0;
  v191.m128i_i8[8] = (16 * (v87 & 1)) | v191.m128i_i8[8] & 0xEF;
  v191.m128i_i8[8] = (4 * (*(_DWORD *)(v157 + 8) >> 8 != 0)) | (32 * !v53) | v191.m128i_i8[8] & 0xDB;
  v88 = sub_A74710((_QWORD *)a2 + 9, 0, 54);
  if ( !v88 )
  {
    v89 = *((_QWORD *)a2 - 4);
    if ( v89 )
    {
      if ( !*(_BYTE *)v89 && *(_QWORD *)(v89 + 24) == *((_QWORD *)a2 + 10) )
      {
        v183 = *(unsigned __int8 **)(v89 + 120);
        v88 = sub_A74710(&v183, 0, 54);
      }
    }
  }
  v191.m128i_i8[8] = v191.m128i_i8[8] & 0xFE | v88 & 1;
  v90 = sub_A74710((_QWORD *)a2 + 9, 0, 79);
  if ( !v90 )
  {
    v91 = *((_QWORD *)a2 - 4);
    if ( v91 )
    {
      if ( !*(_BYTE *)v91 && *(_QWORD *)(v91 + 24) == *((_QWORD *)a2 + 10) )
      {
        v183 = *(unsigned __int8 **)(v91 + 120);
        v90 = sub_A74710(&v183, 0, 79);
      }
    }
  }
  v191.m128i_i8[8] = v191.m128i_i8[8] & 0xFD | (2 * (v90 & 1));
  v92 = sub_A73ED0((_QWORD *)a2 + 9, 32);
  if ( !v92 )
    v92 = sub_B49560((__int64)a2, 32);
  v93 = (unsigned __int64)v194;
  v173 = a3;
  v191.m128i_i8[9] = v191.m128i_i8[9] & 0xFD | (2 * (v92 & 1));
  v192.m128i_i64[1] = a3;
  v174 = a4;
  LODWORD(v193) = a4;
  v192.m128i_i32[0] = (*((_WORD *)a2 + 1) >> 2) & 0x3FF;
  v191.m128i_i32[3] = *(_DWORD *)(v157 + 12) - 1;
  v94 = v184;
  v184 = 0;
  v194 = v94;
  v95 = v185;
  v185 = 0;
  v195 = v95;
  v96 = v186;
  v186 = 0;
  v196 = v96;
  if ( v93 )
    j_j___libc_free_0(v93);
  v200 = a2;
  v191.m128i_i8[10] = v162;
  v97 = sub_A73ED0((_QWORD *)a2 + 9, 6);
  if ( !v97 )
    v97 = sub_B49560((__int64)a2, 6);
  v100 = 0;
  v101 = (a2[7] & 0x80u) != 0;
  v191.m128i_i8[8] = v191.m128i_i8[8] & 0xBF | ((v97 & 1) << 6);
  if ( v101 )
  {
    v102 = sub_BD2BC0((__int64)a2);
    v104 = v102 + v103;
    if ( (a2[7] & 0x80u) != 0 )
      v104 -= sub_BD2BC0((__int64)a2);
    v105 = v104 >> 4;
    if ( (_DWORD)v105 )
    {
      v106 = 0;
      v107 = 0;
      v108 = 16LL * (unsigned int)v105;
      do
      {
        v109 = 0;
        if ( (a2[7] & 0x80u) != 0 )
          v109 = sub_BD2BC0((__int64)a2);
        v110 = *(_DWORD *)(*(_QWORD *)(v109 + v106) + 8LL) == 4;
        v106 += 16;
        v107 += v110;
      }
      while ( v108 != v106 );
      v100 = v107 != 0;
    }
    else
    {
      v100 = 0;
    }
  }
  v191.m128i_i8[9] = v100 | v191.m128i_i8[9] & 0xFE;
  v213 = v156;
  v214 = v165.m128i_i64[0];
  v215 = v161;
  if ( v187.m128i_i64[0] )
    sub_B91220((__int64)&v187, v187.m128i_i64[0]);
  if ( a8 )
  {
    v111 = *(__int64 (**)())(*(_QWORD *)v163 + 2240LL);
    if ( v111 == sub_302E1E0 || !((unsigned __int8 (__fastcall *)(__int64))v111)(v163) )
      sub_C64ED0("This target doesn't support calls with ptrauth operand bundles.", 1u);
    v112 = a8[1];
    v113 = *((_DWORD *)a8 + 4);
    v216 = *a8;
    v217 = v112;
    v218 = v113;
    if ( !v219 )
      v219 = 1;
  }
  sub_3384280(&v187, a1, (__int64)&v190, a7, v98, v99);
  if ( v187.m128i_i64[0] )
  {
    v114 = sub_3375A10(a1, *(_QWORD *)(a1 + 864), a2, v187.m128i_u64[0], v187.m128i_i64[1]);
    v183 = a2;
    v172 = v115;
    v187.m128i_i64[0] = v114;
    v171 = v114;
    v187.m128i_i32[2] = v115;
    v165 = _mm_loadu_si128(&v187);
    v116 = sub_337DC20(a1 + 8, (__int64 *)&v183);
    v170 = _mm_load_si128(&v165);
    *v116 = v170.m128i_i64[0];
    *((_DWORD *)v116 + 2) = v170.m128i_i32[2];
  }
  if ( v158 )
  {
    v117 = *(__int64 (**)())(*(_QWORD *)v163 + 2216LL);
    if ( v117 != sub_302E1B0 )
    {
      if ( ((unsigned __int8 (__fastcall *)(__int64))v117)(v163) )
      {
        v135 = *(_QWORD *)(a1 + 968);
        v136 = (unsigned __int64)&v210[16 * (unsigned int)v211 - 16];
        v137 = *(_QWORD *)v136;
        v138 = *(unsigned int *)(v136 + 8);
        v139 = *(_QWORD *)(a1 + 960);
        v165.m128i_i64[0] = v137;
        v140 = (unsigned int)sub_35D5BC0(v135, a2, *(_QWORD *)(v139 + 744), v158);
        v141 = v188;
        *(_QWORD *)&v164 = v165.m128i_i64[0];
        v142 = v189;
        v143 = v138;
        v144 = (unsigned __int16 *)(*(_QWORD *)(v165.m128i_i64[0] + 48) + 16 * v138);
        v165.m128i_i64[0] = v197;
        *((_QWORD *)&v164 + 1) = v143;
        *(_QWORD *)&v145 = sub_33F0B60(v197, v140, *v144, *((_QWORD *)v144 + 1));
        *((_QWORD *)&v151 + 1) = v142;
        *(_QWORD *)&v151 = v141;
        v146 = sub_340F900(v165.m128i_i32[0], 49, (unsigned int)&v198, 1, 0, DWORD2(v164), v151, v145, v164);
        v148 = *(_QWORD *)(a1 + 864);
        v149 = v146;
        v150 = v147;
        if ( v146 )
        {
          nullsub_1875(v146, v148, 0);
          v169 = v150;
          v168 = v149;
          *(_QWORD *)(v148 + 384) = v149;
          *(_DWORD *)(v148 + 392) = v169;
          sub_33E2B60(v148, 0);
        }
        else
        {
          v167 = v147;
          v166 = 0;
          *(_QWORD *)(v148 + 384) = 0;
          *(_DWORD *)(v148 + 392) = v167;
        }
      }
    }
  }
  if ( v210 != v212 )
    _libc_free((unsigned __int64)v210);
  if ( v207 != v209 )
    _libc_free((unsigned __int64)v207);
  if ( v204 != v206 )
    _libc_free((unsigned __int64)v204);
  if ( v201 != v203 )
    _libc_free((unsigned __int64)v201);
  if ( v198 )
    sub_B91220((__int64)&v198, v198);
  if ( v194 )
    j_j___libc_free_0((unsigned __int64)v194);
  if ( v184 )
    j_j___libc_free_0((unsigned __int64)v184);
}
