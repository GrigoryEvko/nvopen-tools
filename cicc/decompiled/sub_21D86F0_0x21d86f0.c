// Function: sub_21D86F0
// Address: 0x21d86f0
//
__int64 __fastcall sub_21D86F0(
        __int64 a1,
        __int64 a2,
        __m128i a3,
        double a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        _QWORD *a8,
        __int64 a9,
        const __m128i *a10,
        __int64 a11)
{
  __int64 v11; // rbx
  __int64 v12; // rax
  unsigned int v13; // eax
  _DWORD *v14; // rdi
  __int64 (*v15)(void); // rax
  __m128i v16; // xmm2
  bool v18; // zf
  char v19; // al
  _QWORD *v20; // rbx
  _QWORD *v21; // r12
  _BYTE *v22; // rsi
  __int64 v23; // rax
  _BYTE *v24; // rsi
  size_t v25; // rdx
  char *v26; // r12
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // rdi
  __int64 *v30; // r13
  __int64 v32; // rbx
  __int64 *v33; // rax
  int v34; // esi
  __int64 v35; // rdx
  __int64 v36; // rcx
  __int64 v37; // rdi
  __int64 v38; // rax
  __int64 v39; // rdx
  unsigned int v40; // r14d
  int v41; // edx
  int v42; // eax
  __int64 v43; // rdx
  __int64 v44; // rax
  __int64 v45; // rdx
  _QWORD *v46; // rax
  __int64 *v47; // rax
  __int64 **v48; // rax
  __int64 v49; // rdx
  __int64 v50; // rcx
  _BYTE *v51; // rbx
  _BOOL4 v52; // ebx
  _QWORD *v53; // rax
  unsigned int v54; // ebx
  __int64 v55; // rax
  int v56; // r9d
  int v57; // eax
  __int64 v58; // rax
  unsigned int v59; // edx
  __int64 v60; // rbx
  int v61; // r8d
  int v62; // r9d
  unsigned int v63; // edx
  __int64 v64; // r14
  __int64 v65; // rax
  __int64 *v66; // rax
  __int64 v67; // rdi
  __int64 v68; // rax
  __int64 v69; // r9
  int v70; // r14d
  __int64 v71; // rdx
  _QWORD *v72; // r8
  __int64 v73; // rax
  _QWORD *v74; // rax
  unsigned int v75; // eax
  int v76; // r9d
  __int64 v77; // rax
  _QWORD *v78; // rdi
  unsigned __int64 v79; // rdx
  __int64 *v80; // r14
  int v81; // r12d
  int v82; // eax
  unsigned int v83; // ebx
  __int64 v84; // rax
  unsigned __int32 v85; // r13d
  __int64 v86; // rax
  __int64 v87; // r10
  __int64 v88; // rsi
  unsigned int v89; // edi
  __int128 v90; // rax
  __int64 *v91; // rax
  __int64 v92; // rdx
  __int64 v93; // r13
  __int64 v94; // r12
  _QWORD *v95; // rax
  __int64 *v96; // rax
  __int64 **v97; // rax
  __int64 v98; // rdx
  __int64 v99; // rcx
  __int64 v100; // rax
  __int64 v101; // r10
  __int64 v102; // rcx
  int v103; // edx
  int v104; // eax
  __int64 v105; // rax
  __int64 v106; // rax
  unsigned __int64 v107; // rdx
  __int64 v108; // rax
  __int64 v109; // rbx
  int v110; // r8d
  int v111; // r9d
  __int64 v112; // r12
  unsigned int v113; // edx
  unsigned __int64 v114; // r13
  __int64 v115; // rcx
  char v116; // dl
  __int64 v117; // rsi
  __int64 v118; // rax
  __int64 *v119; // rax
  __int128 v120; // rax
  unsigned __int64 v121; // rdx
  unsigned int v122; // edx
  unsigned int v123; // eax
  _BYTE *v124; // rcx
  char v125; // dl
  unsigned int v126; // eax
  __int64 v127; // rax
  unsigned int v128; // edx
  __int64 v129; // r9
  int v130; // eax
  __int64 v131; // rdx
  __int64 v132; // r9
  __int64 *v133; // rax
  int v134; // r8d
  int v135; // r9d
  __int64 *v136; // rdx
  __int64 *v137; // r15
  __int64 *v138; // r14
  __int64 v139; // rdx
  __int64 **v140; // rdx
  unsigned int v141; // eax
  unsigned int v142; // r14d
  const void **v143; // rdx
  const void **v144; // rbx
  __int128 v145; // rax
  __int64 v146; // rax
  __int64 v147; // r14
  __int64 v148; // rdx
  __int64 v149; // r15
  int v150; // r8d
  int v151; // r9d
  __int64 v152; // rax
  __int64 *v153; // rax
  char v154; // al
  __int64 v155; // rax
  unsigned __int64 v156; // rdx
  __int64 *v157; // rax
  __int64 *v158; // rdx
  _QWORD *v159; // rax
  __int64 v160; // rax
  unsigned int v161; // eax
  __int64 v162; // rdx
  __int64 v163; // r15
  unsigned int v164; // ebx
  __int64 v165; // rax
  int v166; // eax
  __int64 v167; // r9
  unsigned int v168; // r15d
  __int64 v169; // rax
  __int64 v170; // rdx
  _QWORD *v171; // r8
  __int64 v172; // rax
  _QWORD *v173; // rax
  _QWORD *v174; // rax
  __int64 v175; // rax
  int v176; // edx
  int v177; // eax
  __int64 v178; // rax
  unsigned int v179; // ebx
  unsigned int v180; // edx
  _QWORD *v181; // rax
  __int64 v182; // rax
  int v183; // r8d
  int v184; // r9d
  __int64 v185; // r14
  __int64 v186; // rdx
  __int64 v187; // r15
  __int64 v188; // rdx
  __int64 *v189; // rdx
  __int128 v190; // [rsp-20h] [rbp-490h]
  __int128 v191; // [rsp-10h] [rbp-480h]
  __int128 v192; // [rsp-10h] [rbp-480h]
  __int128 v193; // [rsp-10h] [rbp-480h]
  int v194; // [rsp-10h] [rbp-480h]
  int v195; // [rsp-8h] [rbp-478h]
  unsigned int v196; // [rsp+0h] [rbp-470h]
  unsigned int v197; // [rsp+8h] [rbp-468h]
  __int64 v198; // [rsp+18h] [rbp-458h]
  __int64 v199; // [rsp+28h] [rbp-448h]
  __int64 v200; // [rsp+50h] [rbp-420h]
  unsigned __int64 v201; // [rsp+58h] [rbp-418h]
  int v202; // [rsp+64h] [rbp-40Ch]
  unsigned int v204; // [rsp+90h] [rbp-3E0h]
  int v205; // [rsp+98h] [rbp-3D8h]
  __int64 v206; // [rsp+A0h] [rbp-3D0h]
  __int64 v207; // [rsp+A8h] [rbp-3C8h]
  unsigned int v209; // [rsp+B8h] [rbp-3B8h]
  __int64 v210; // [rsp+C8h] [rbp-3A8h]
  unsigned __int64 v211; // [rsp+D0h] [rbp-3A0h]
  __int64 v212; // [rsp+D8h] [rbp-398h]
  int v213; // [rsp+E0h] [rbp-390h]
  int v214; // [rsp+E4h] [rbp-38Ch]
  _BYTE *v215; // [rsp+E8h] [rbp-388h]
  unsigned int v216; // [rsp+F0h] [rbp-380h]
  unsigned __int8 v217; // [rsp+F9h] [rbp-377h]
  unsigned __int8 v218; // [rsp+FAh] [rbp-376h]
  char v219; // [rsp+FBh] [rbp-375h]
  unsigned int v220; // [rsp+FCh] [rbp-374h]
  __int64 v221; // [rsp+100h] [rbp-370h]
  __int64 v222; // [rsp+100h] [rbp-370h]
  __int64 v223; // [rsp+108h] [rbp-368h]
  unsigned int v224; // [rsp+110h] [rbp-360h]
  __int64 v225; // [rsp+110h] [rbp-360h]
  __int64 v226; // [rsp+110h] [rbp-360h]
  __int64 v227; // [rsp+110h] [rbp-360h]
  _QWORD *v228; // [rsp+110h] [rbp-360h]
  __int64 v229; // [rsp+110h] [rbp-360h]
  __int64 v230; // [rsp+118h] [rbp-358h]
  __int64 v231; // [rsp+120h] [rbp-350h]
  __int64 v232; // [rsp+120h] [rbp-350h]
  __int64 v234; // [rsp+130h] [rbp-340h]
  _QWORD *v235; // [rsp+130h] [rbp-340h]
  __int64 v236; // [rsp+138h] [rbp-338h]
  __int64 v237; // [rsp+140h] [rbp-330h]
  int v238; // [rsp+140h] [rbp-330h]
  _QWORD *v239; // [rsp+140h] [rbp-330h]
  unsigned int v240; // [rsp+140h] [rbp-330h]
  __int64 v241; // [rsp+140h] [rbp-330h]
  unsigned __int64 v242; // [rsp+148h] [rbp-328h]
  int v243; // [rsp+1A4h] [rbp-2CCh] BYREF
  __int64 v244; // [rsp+1A8h] [rbp-2C8h] BYREF
  __m128i v245; // [rsp+1B0h] [rbp-2C0h] BYREF
  __m128i v246; // [rsp+1C0h] [rbp-2B0h] BYREF
  __int64 v247; // [rsp+1D0h] [rbp-2A0h] BYREF
  _BYTE *v248; // [rsp+1D8h] [rbp-298h]
  _BYTE *v249; // [rsp+1E0h] [rbp-290h]
  __int64 v250; // [rsp+1F0h] [rbp-280h] BYREF
  _BYTE *v251; // [rsp+1F8h] [rbp-278h]
  _BYTE *v252; // [rsp+200h] [rbp-270h]
  __int128 v253; // [rsp+210h] [rbp-260h]
  __int64 v254; // [rsp+220h] [rbp-250h]
  _QWORD v255[4]; // [rsp+230h] [rbp-240h] BYREF
  _QWORD *v256; // [rsp+250h] [rbp-220h] BYREF
  __int64 v257; // [rsp+258h] [rbp-218h]
  _QWORD v258[8]; // [rsp+260h] [rbp-210h] BYREF
  __int128 v259; // [rsp+2A0h] [rbp-1D0h] BYREF
  __int64 v260[16]; // [rsp+2B0h] [rbp-1C0h] BYREF
  _QWORD *v261; // [rsp+330h] [rbp-140h] BYREF
  __int64 v262; // [rsp+338h] [rbp-138h]
  _QWORD v263[38]; // [rsp+340h] [rbp-130h] BYREF
  __int64 v264; // [rsp+490h] [rbp+20h]

  v11 = a10[2].m128i_i64[0];
  v206 = sub_1E0A0C0(v11);
  v12 = sub_1E0A0C0(a10[2].m128i_i64[0]);
  v13 = 8 * sub_15A9520(v12, 0);
  if ( v13 == 32 )
  {
    v218 = 5;
  }
  else if ( v13 > 0x20 )
  {
    v218 = 6;
    if ( v13 != 64 )
    {
      v18 = v13 == 128;
      v19 = 7;
      if ( !v18 )
        v19 = 0;
      v218 = v19;
    }
  }
  else
  {
    v218 = 3;
    if ( v13 != 8 )
      v218 = 4 * (v13 == 16);
  }
  v210 = *(_QWORD *)v11;
  v244 = *(_QWORD *)(*(_QWORD *)v11 + 112LL);
  v14 = *(_DWORD **)(a1 + 81552);
  v15 = *(__int64 (**)(void))(*(_QWORD *)v14 + 56LL);
  if ( (char *)v15 == (char *)sub_214ABA0 )
  {
    v199 = (__int64)(v14 + 174);
  }
  else
  {
    v199 = v15();
    v14 = *(_DWORD **)(a1 + 81552);
  }
  v16 = _mm_loadu_si128(a10 + 11);
  if ( v14[63] > 0x13u )
  {
    v247 = 0;
    v248 = 0;
    v249 = 0;
    v250 = 0;
    v251 = 0;
    v252 = 0;
    if ( (*(_BYTE *)(v210 + 18) & 1) != 0 )
    {
      sub_15E08E0(v210, 0);
      v20 = *(_QWORD **)(v210 + 88);
      v21 = &v20[5 * *(_QWORD *)(v210 + 96)];
      if ( (*(_BYTE *)(v210 + 18) & 1) != 0 )
      {
        sub_15E08E0(v210, v210);
        v20 = *(_QWORD **)(v210 + 88);
      }
    }
    else
    {
      v20 = *(_QWORD **)(v210 + 88);
      v21 = &v20[5 * *(_QWORD *)(v210 + 96)];
    }
    for ( ; v21 != v20; v248 = v24 + 8 )
    {
      while ( 1 )
      {
        v261 = v20;
        v22 = v251;
        if ( v251 == v252 )
        {
          sub_21D8560((__int64)&v250, v251, &v261);
        }
        else
        {
          if ( v251 )
          {
            *(_QWORD *)v251 = v20;
            v22 = v251;
          }
          v251 = v22 + 8;
        }
        v23 = *v20;
        v24 = v248;
        v261 = (_QWORD *)*v20;
        if ( v248 != v249 )
          break;
        v20 += 5;
        sub_1278040((__int64)&v247, v248, &v261);
        if ( v21 == v20 )
          goto LABEL_27;
      }
      if ( v248 )
      {
        *(_QWORD *)v248 = v23;
        v24 = v248;
      }
      v20 += 5;
    }
LABEL_27:
    v25 = 0;
    v26 = off_4CD4950[0];
    if ( off_4CD4950[0] )
      v25 = strlen(off_4CD4950[0]);
    v27 = sub_1626CE0(v210, v26, v25);
    LODWORD(v198) = 0;
    if ( v27 )
    {
      v28 = *(_QWORD *)(*(_QWORD *)(v27 - 8LL * *(unsigned int *)(v27 + 8)) + 136LL);
      if ( *(_DWORD *)(v28 + 32) <= 0x40u )
        v198 = *(_QWORD *)(v28 + 24);
      else
        v198 = **(_QWORD **)(v28 + 24);
    }
    v29 = v250;
    v213 = (__int64)&v251[-v250] >> 3;
    if ( !v213 )
    {
LABEL_70:
      if ( v29 )
        j_j___libc_free_0(v29, &v252[-v29]);
      if ( v247 )
        j_j___libc_free_0(v247, &v249[-v247]);
      return a2;
    }
    v220 = 0;
    v30 = (__int64 *)a10;
    v212 = 1;
    v211 = 0;
    while ( 1 )
    {
      v32 = *(_QWORD *)(v247 + 8 * v211);
      v33 = *(__int64 **)(v29 + 8 * v211);
      v34 = v211;
      v35 = v33[3];
      if ( v35 )
      {
        v36 = *v33;
        if ( *(_QWORD *)(v35 + 40) )
        {
          if ( *(_BYTE *)(v36 + 8) == 15 )
          {
            v37 = *(_QWORD *)(v36 + 24);
            if ( *(_BYTE *)(v37 + 8) == 13 && (*(_BYTE *)(v37 + 9) & 4) == 0 )
            {
              v38 = sub_1643640(v37);
              v34 = v211;
              if ( v39 == 17
                && (!(*(_QWORD *)v38 ^ 0x5F2E746375727473LL | *(_QWORD *)(v38 + 8) ^ 0x5F64326567616D69LL)
                 && *(_BYTE *)(v38 + 16) == 116
                 || !(*(_QWORD *)v38 ^ 0x5F2E746375727473LL | *(_QWORD *)(v38 + 8) ^ 0x5F64336567616D69LL)
                 && *(_BYTE *)(v38 + 16) == 116
                 || !(*(_QWORD *)v38 ^ 0x5F2E746375727473LL | *(_QWORD *)(v38 + 8) ^ 0x5F72656C706D6173LL)
                 && *(_BYTE *)(v38 + 16) == 116) )
              {
                v214 = v212;
                v185 = sub_1D38BB0((__int64)v30, v212, a9, 5, 0, 0, a3, a4, v16, 0);
                v187 = v186;
                v188 = *(unsigned int *)(a11 + 8);
                if ( (unsigned int)v188 >= *(_DWORD *)(a11 + 12) )
                {
                  sub_16CD150(a11, (const void *)(a11 + 16), 0, 16, v183, v184);
                  v188 = *(unsigned int *)(a11 + 8);
                }
                v189 = (__int64 *)(*(_QWORD *)a11 + 16 * v188);
                ++v220;
                *v189 = v185;
                v189[1] = v187;
                ++*(_DWORD *)(a11 + 8);
                goto LABEL_69;
              }
              v33 = *(__int64 **)(v250 + 8 * v211);
            }
          }
        }
      }
      v214 = v212;
      v40 = v220 + 1;
      if ( !v33[1] )
        break;
      v217 = sub_1560290(&v244, v34, 6);
      if ( !v217 )
      {
        v41 = *(unsigned __int8 *)(v32 + 8);
        if ( (unsigned int)(v41 - 13) <= 1 || (_BYTE)v41 == 16 )
          goto LABEL_75;
        if ( !sub_1642F90(v32, 128) )
        {
          LOBYTE(v42) = sub_204D4D0(a1, v206, v32);
          LODWORD(v256) = v42;
          v257 = v43;
          v44 = sub_21D74F0(a1, v30, v211, v218, 0);
          v223 = v45;
          v221 = v44;
          v46 = (_QWORD *)sub_15E0530(v210);
          v47 = (__int64 *)sub_1F58E60((__int64)&v256, v46);
          v48 = (__int64 **)sub_1646BA0(v47, 101);
          v237 = sub_15A06D0(v48, 101, v49, v50);
          if ( (_BYTE)v256 )
            v224 = sub_1F3E310(&v256);
          else
            v224 = sub_1F58D40((__int64)&v256);
          v51 = (_BYTE *)(48LL * v220 + *a8);
          v231 = 48LL * v220;
          if ( (unsigned int)sub_1F3E310(v51 + 8) <= v224 )
          {
            v261 = 0;
            v262 = 0;
            v263[0] = 0;
            v174 = (_QWORD *)sub_15E0530(v210);
            v175 = sub_1F58E60((__int64)&v256, v174);
            v176 = sub_15A9FE0(v206, v175);
            LOBYTE(v260[0]) = 0;
            v177 = 0;
            v259 = (unsigned __int64)v237;
            if ( v237 )
            {
              v178 = *(_QWORD *)v237;
              if ( *(_BYTE *)(*(_QWORD *)v237 + 8LL) == 16 )
                v178 = **(_QWORD **)(v178 + 16);
              v177 = *(_DWORD *)(v178 + 8) >> 8;
            }
            HIDWORD(v260[0]) = v177;
            v179 = v196;
            LOBYTE(v179) = *(_BYTE *)(*a8 + v231 + 8);
            v60 = sub_1D2B730(
                    v30,
                    v179,
                    0,
                    a9,
                    v16.m128i_i64[0],
                    v16.m128i_i64[1],
                    v221,
                    v223,
                    v259,
                    v260[0],
                    v176,
                    0,
                    (__int64)&v261,
                    0);
            v64 = v180;
          }
          else
          {
            v18 = (*v51 & 2) == 0;
            v261 = 0;
            v262 = 0;
            v263[0] = 0;
            v52 = v18;
            v53 = (_QWORD *)sub_15E0530(v210);
            v54 = v52 + 2;
            v55 = sub_1F58E60((__int64)&v256, v53);
            v56 = sub_15A9FE0(v206, v55);
            LOBYTE(v260[0]) = 0;
            v57 = 0;
            v259 = (unsigned __int64)v237;
            if ( v237 )
            {
              v58 = *(_QWORD *)v237;
              if ( *(_BYTE *)(*(_QWORD *)v237 + 8LL) == 16 )
                v58 = **(_QWORD **)(v58 + 16);
              v57 = *(_DWORD *)(v58 + 8) >> 8;
            }
            HIDWORD(v260[0]) = v57;
            v59 = v197;
            LOBYTE(v59) = *(_BYTE *)(*a8 + v231 + 8);
            v60 = sub_1D2B810(
                    v30,
                    v54,
                    a9,
                    v59,
                    0,
                    v56,
                    *(_OWORD *)&v16,
                    v221,
                    v223,
                    v259,
                    v260[0],
                    (__int64)v256,
                    v257,
                    0,
                    (__int64)&v261);
            v64 = v63;
          }
          if ( v60 )
            *(_DWORD *)(v60 + 64) = v212;
          if ( v211 <= 0x40 && ((1 << v211) & (unsigned int)v198) != 0 )
            *(_BYTE *)(*(_QWORD *)(v60 + 104) + 76LL) = 1;
          v65 = *(unsigned int *)(a11 + 8);
          if ( (unsigned int)v65 >= *(_DWORD *)(a11 + 12) )
          {
            sub_16CD150(a11, (const void *)(a11 + 16), 0, 16, v61, v62);
            v65 = *(unsigned int *)(a11 + 8);
          }
          v66 = (__int64 *)(*(_QWORD *)a11 + 16 * v65);
          *v66 = v60;
          v66[1] = v64;
          ++*(_DWORD *)(a11 + 8);
          ++v220;
          goto LABEL_69;
        }
        LOBYTE(v41) = *(_BYTE *)(v32 + 8);
LABEL_75:
        if ( (_BYTE)v41 == 13 )
          v217 = (*(_DWORD *)(v32 + 8) & 0x200) != 0;
        *(_QWORD *)&v259 = v260;
        v261 = v263;
        v262 = 0x1000000000LL;
        *((_QWORD *)&v259 + 1) = 0x1000000000LL;
        sub_21CAE40(a1, v206, v32, (__int64)&v261, (__int64)&v259, 0);
        if ( !(_DWORD)v262 )
        {
          v256 = v258;
          sub_21CA7A0((__int64 *)&v256, "Empty parameter types are not supported", (__int64)"");
          sub_1C3EF50((__int64)&v256);
          if ( v256 != v258 )
            j_j___libc_free_0(v256, v258[0] + 1LL);
        }
        v75 = sub_15A9FE0(v206, v32);
        sub_21CB650(&v256, (__int64 *)&v261, (__int64 *)&v259, v75, 0, v76);
        v77 = sub_21D74F0(a1, v30, v211, v218, 0);
        v78 = v256;
        v200 = v77;
        v201 = v79;
        v202 = v262;
        if ( (_DWORD)v262 )
        {
          v264 = a11;
          v80 = v30;
          v234 = 0;
          v81 = -1;
          do
          {
            v82 = *((_DWORD *)v78 + v234);
            if ( (v82 & 1) != 0 )
              v81 = v234;
            if ( (v82 & 2) != 0 )
            {
              v83 = v234 + 1 - v81;
              a3 = _mm_loadu_si128((const __m128i *)&v261[2 * v234]);
              v246 = a3;
              v245 = a3;
              if ( a3.m128i_i8[0] == 2 )
              {
                v246.m128i_i8[0] = 3;
                v246.m128i_i64[1] = 0;
              }
              else if ( a3.m128i_i8[0] == 86 )
              {
                v246.m128i_i8[0] = 5;
                v246.m128i_i64[1] = 0;
              }
              v84 = sub_15E0530(v210);
              v85 = v246.m128i_i32[0];
              v239 = (_QWORD *)v84;
              v225 = v246.m128i_i64[1];
              LOBYTE(v86) = sub_1D15020(v246.m128i_i8[0], v83);
              v87 = 0;
              if ( !(_BYTE)v86 )
              {
                v86 = sub_1F593D0(v239, v85, v225, v83);
                v207 = v86;
                v87 = v131;
              }
              v88 = v207;
              v89 = v204;
              v226 = v87;
              LOBYTE(v88) = v86;
              v207 = v88;
              LOBYTE(v89) = v218;
              *(_QWORD *)&v90 = sub_1D38BB0((__int64)v80, *(_QWORD *)(v259 + 8LL * v81), a9, v89, 0, 0, a3, a4, v16, 0);
              LODWORD(v88) = v205;
              LOBYTE(v88) = v218;
              v91 = sub_1D332F0(v80, 52, a9, (unsigned int)v88, 0, 0, *(double *)a3.m128i_i64, a4, v16, v200, v201, v90);
              v93 = v92;
              v94 = (__int64)v91;
              v95 = (_QWORD *)sub_15E0530(v210);
              v96 = (__int64 *)sub_1F58E60((__int64)&v245, v95);
              v97 = (__int64 **)sub_1646BA0(v96, 101);
              v100 = sub_15A06D0(v97, 101, v98, v99);
              v101 = v226;
              v102 = v100;
              if ( v83 == 1 )
              {
                v129 = 8 * v234;
                if ( v217 )
                {
                  v243 = 1;
                  v130 = 1;
                }
                else
                {
                  v229 = v100;
                  v241 = v101;
                  if ( *(_QWORD *)(v259 + 8 * v234) )
                  {
                    v159 = (_QWORD *)sub_15E0530(v210);
                    v160 = sub_1F58E60((__int64)&v245, v159);
                    v130 = sub_15A9FE0(v206, v160);
                    v129 = 8 * v234;
                    v102 = v229;
                    v243 = v130;
                    v101 = v241;
                  }
                  else
                  {
                    v154 = sub_1C2FF50(v210, v212, &v243);
                    v101 = v241;
                    v102 = v229;
                    v129 = 8 * v234;
                    if ( v154 )
                    {
                      v130 = v243;
                    }
                    else
                    {
                      v130 = sub_15603A0(&v244, v212);
                      v101 = v241;
                      v102 = v229;
                      v243 = v130;
                      v129 = 8 * v234;
                      if ( !v130 )
                      {
                        v181 = (_QWORD *)sub_15E0530(v210);
                        v182 = sub_1F58E60((__int64)&v245, v181);
                        v130 = sub_15A9FE0(v206, v182);
                        v101 = v241;
                        v102 = v229;
                        v243 = v130;
                        v129 = 8 * v234;
                      }
                    }
                  }
                }
                v243 = -(*(_DWORD *)(v259 + v129) | v130) & (*(_DWORD *)(v259 + v129) | v130);
                v103 = v243;
              }
              else
              {
                v103 = v217;
                v243 = v217;
              }
              memset(v255, 0, 24);
              v104 = 0;
              v253 = (unsigned __int64)v102;
              LOBYTE(v254) = 0;
              if ( v102 )
              {
                v105 = *(_QWORD *)v102;
                if ( *(_BYTE *)(*(_QWORD *)v102 + 8LL) == 16 )
                  v105 = **(_QWORD **)(v105 + 16);
                v104 = *(_DWORD *)(v105 + 8) >> 8;
              }
              HIDWORD(v254) = v104;
              v106 = sub_1D2B730(
                       v80,
                       (unsigned int)v207,
                       v101,
                       a9,
                       v16.m128i_i64[0],
                       v16.m128i_i64[1],
                       v94,
                       v93,
                       v253,
                       v254,
                       v103,
                       0x30u,
                       (__int64)v255,
                       0);
              v242 = v107;
              v227 = v106;
              if ( v106 )
                *(_DWORD *)(v106 + 64) = v212;
              if ( v83 )
              {
                v232 = 48LL * ((unsigned int)v234 + v220);
                v108 = v83;
                v109 = 0;
                v222 = v108;
                do
                {
                  *(_QWORD *)&v120 = sub_1D38E70((__int64)v80, v109, a9, 0, a3, a4, v16);
                  v112 = (__int64)sub_1D332F0(
                                    v80,
                                    106,
                                    a9,
                                    v246.m128i_u32[0],
                                    (const void **)v246.m128i_i64[1],
                                    0,
                                    *(double *)a3.m128i_i64,
                                    a4,
                                    v16,
                                    v227,
                                    v242,
                                    v120);
                  v114 = v121;
                  if ( v245.m128i_i8[0] == 2 )
                  {
                    *((_QWORD *)&v192 + 1) = v121;
                    *(_QWORD *)&v192 = v112;
                    v112 = sub_1D309E0(
                             v80,
                             145,
                             a9,
                             2,
                             0,
                             0,
                             *(double *)a3.m128i_i64,
                             a4,
                             *(double *)v16.m128i_i64,
                             v192);
                    v114 = v122 | v114 & 0xFFFFFFFF00000000LL;
                  }
                  else if ( v245.m128i_i8[0] == 86 )
                  {
                    *((_QWORD *)&v191 + 1) = v121;
                    *(_QWORD *)&v191 = v112;
                    v112 = sub_1D309E0(
                             v80,
                             158,
                             a9,
                             86,
                             0,
                             0,
                             *(double *)a3.m128i_i64,
                             a4,
                             *(double *)v16.m128i_i64,
                             v191);
                    v114 = v113 | v114 & 0xFFFFFFFF00000000LL;
                  }
                  v115 = *a8 + v232;
                  v116 = *(_BYTE *)(v115 + 8);
                  if ( (unsigned __int8)(v116 - 2) > 5u && (unsigned __int8)(v116 - 14) > 0x47u )
                    goto LABEL_103;
                  v219 = *(_BYTE *)(v115 + 8);
                  v215 = (_BYTE *)(*a8 + v232);
                  v216 = sub_1F3E310((_BYTE *)(v115 + 8));
                  if ( v246.m128i_i8[0] )
                  {
                    v123 = sub_1F3E310(&v246);
                    v124 = v215;
                    v125 = v219;
                  }
                  else
                  {
                    v123 = sub_1F58D40((__int64)&v246);
                    v125 = v219;
                    v124 = v215;
                  }
                  if ( v216 <= v123 )
                  {
LABEL_103:
                    v117 = v264;
                    v118 = *(unsigned int *)(v264 + 8);
                    if ( (unsigned int)v118 >= *(_DWORD *)(v264 + 12) )
                      goto LABEL_111;
                  }
                  else
                  {
                    v126 = v209;
                    *((_QWORD *)&v193 + 1) = v114;
                    *(_QWORD *)&v193 = v112;
                    LOBYTE(v126) = v125;
                    v127 = sub_1D309E0(
                             v80,
                             (unsigned int)((*v124 & 2) == 0) + 142,
                             a9,
                             v126,
                             0,
                             0,
                             *(double *)a3.m128i_i64,
                             a4,
                             *(double *)v16.m128i_i64,
                             v193);
                    v110 = v194;
                    v111 = v195;
                    v112 = v127;
                    v117 = v264;
                    v114 = v128 | v114 & 0xFFFFFFFF00000000LL;
                    v118 = *(unsigned int *)(v264 + 8);
                    if ( (unsigned int)v118 >= *(_DWORD *)(v264 + 12) )
                    {
LABEL_111:
                      sub_16CD150(v117, (const void *)(v117 + 16), 0, 16, v110, v111);
                      v118 = *(unsigned int *)(v264 + 8);
                    }
                  }
                  ++v109;
                  v119 = (__int64 *)(*(_QWORD *)v264 + 16 * v118);
                  *v119 = v112;
                  v119[1] = v114;
                  ++*(_DWORD *)(v264 + 8);
                }
                while ( v109 != v222 );
              }
              v78 = v256;
              v81 = -1;
            }
            ++v234;
          }
          while ( v202 != v234 );
          v30 = v80;
          v220 += v202 + ((_DWORD)v262 == 0);
          a11 = v264;
        }
        else
        {
          ++v220;
        }
        if ( v78 != v258 )
          _libc_free((unsigned __int64)v78);
        if ( (__int64 *)v259 != v260 )
          _libc_free(v259);
LABEL_67:
        if ( v261 != v263 )
          _libc_free((unsigned __int64)v261);
        goto LABEL_69;
      }
      LOBYTE(v141) = sub_204D4D0(a1, v206, v32);
      v142 = v141;
      v144 = v143;
      v240 = v141;
      *(_QWORD *)&v145 = sub_21D74F0(a1, v30, v211, v218, 0);
      v146 = sub_1D309E0(v30, 287, a9, v142, v144, 0, *(double *)a3.m128i_i64, a4, *(double *)v16.m128i_i64, v145);
      v147 = v146;
      v149 = v148;
      if ( v146 )
        *(_DWORD *)(v146 + 64) = v212;
      if ( (unsigned __int8)sub_1C2F070(v210) )
      {
        v152 = *(unsigned int *)(a11 + 8);
        if ( (unsigned int)v152 >= *(_DWORD *)(a11 + 12) )
        {
          sub_16CD150(a11, (const void *)(a11 + 16), 0, 16, v150, v151);
          v152 = *(unsigned int *)(a11 + 8);
        }
        v153 = (__int64 *)(*(_QWORD *)a11 + 16 * v152);
        *v153 = v147;
        v153[1] = v149;
        ++*(_DWORD *)(a11 + 8);
        goto LABEL_127;
      }
      v155 = sub_1D38BB0((__int64)v30, 4241, a9, 5, 0, 0, a3, a4, v16, 0);
      *((_QWORD *)&v190 + 1) = v149;
      *(_QWORD *)&v190 = v147;
      v157 = sub_1D332F0(v30, 43, a9, v240, v144, 0, *(double *)a3.m128i_i64, a4, v16, v155, v156, v190);
      v137 = v158;
      v138 = v157;
      v139 = *(unsigned int *)(a11 + 8);
      if ( (unsigned int)v139 >= *(_DWORD *)(a11 + 12) )
        goto LABEL_145;
LABEL_126:
      v140 = (__int64 **)(*(_QWORD *)a11 + 16 * v139);
      *v140 = v138;
      v140[1] = v137;
      ++*(_DWORD *)(a11 + 8);
LABEL_127:
      ++v220;
LABEL_69:
      ++v211;
      v29 = v250;
      ++v212;
      if ( v213 == v214 )
        goto LABEL_70;
    }
    if ( (unsigned int)*(unsigned __int8 *)(v32 + 8) - 13 <= 1 || sub_1642F90(v32, 128) )
    {
      v67 = v30[4];
      v261 = v263;
      v262 = 0x1000000000LL;
      v68 = sub_1E0A0C0(v67);
      sub_21CAE40(a1, v68, v32, (__int64)&v261, 0, 0);
      v238 = v262;
      if ( (_DWORD)v262 )
        goto LABEL_62;
      *(_QWORD *)&v259 = v260;
      sub_21CA7A0((__int64 *)&v259, "Empty parameter types are not supported", (__int64)"");
      sub_1C3EF50((__int64)&v259);
      if ( (__int64 *)v259 != v260 )
        j_j___libc_free_0(v259, v260[0] + 1);
      v238 = v262;
      if ( (_DWORD)v262 )
      {
LABEL_62:
        v70 = 0;
        do
        {
          v72 = sub_1D2B300(v30, 0x30u, a9, *(unsigned __int8 *)(*a8 + 48LL * (v70 + v220) + 8), 0, v69);
          v69 = v71;
          v73 = *(unsigned int *)(a11 + 8);
          if ( (unsigned int)v73 >= *(_DWORD *)(a11 + 12) )
          {
            v230 = v71;
            v228 = v72;
            sub_16CD150(a11, (const void *)(a11 + 16), 0, 16, (int)v72, v71);
            v73 = *(unsigned int *)(a11 + 8);
            v72 = v228;
            v69 = v230;
          }
          v74 = (_QWORD *)(*(_QWORD *)a11 + 16 * v73);
          ++v70;
          *v74 = v72;
          v74[1] = v69;
          ++*(_DWORD *)(a11 + 8);
        }
        while ( v70 != v238 );
        v220 += v70 + ((_DWORD)v262 == 0);
      }
      else
      {
        ++v220;
      }
      goto LABEL_67;
    }
    if ( *(_BYTE *)(v32 + 8) == 16 )
    {
      LOBYTE(v161) = sub_204D4D0(a1, v206, v32);
      v163 = v162;
      v164 = v161;
      v165 = sub_15E0530(v210);
      v166 = sub_1FDDD20(v199, v165, v164, v163);
      if ( v166 )
      {
        v168 = v220 + 1;
        v40 = v166 + v220;
        v169 = v220;
        while ( 1 )
        {
          v171 = sub_1D2B300(v30, 0x30u, a9, *(unsigned __int8 *)(*a8 + 48 * v169 + 8), 0, v167);
          v167 = v170;
          v172 = *(unsigned int *)(a11 + 8);
          if ( (unsigned int)v172 >= *(_DWORD *)(a11 + 12) )
          {
            v236 = v170;
            v235 = v171;
            sub_16CD150(a11, (const void *)(a11 + 16), 0, 16, (int)v171, v170);
            v172 = *(unsigned int *)(a11 + 8);
            v171 = v235;
            v167 = v236;
          }
          v173 = (_QWORD *)(*(_QWORD *)a11 + 16 * v172);
          *v173 = v171;
          v173[1] = v167;
          v169 = v168;
          ++*(_DWORD *)(a11 + 8);
          if ( v40 == v168 )
            break;
          ++v168;
        }
      }
      v220 = v40;
      goto LABEL_69;
    }
    v133 = sub_1D2B300(v30, 0x30u, a9, *(unsigned __int8 *)(*a8 + 48LL * v220 + 8), 0, v132);
    v137 = v136;
    v138 = v133;
    v139 = *(unsigned int *)(a11 + 8);
    if ( (unsigned int)v139 < *(_DWORD *)(a11 + 12) )
      goto LABEL_126;
LABEL_145:
    sub_16CD150(a11, (const void *)(a11 + 16), 0, 16, v134, v135);
    v139 = *(unsigned int *)(a11 + 8);
    goto LABEL_126;
  }
  return a2;
}
