// Function: sub_1A5F590
// Address: 0x1a5f590
//
__int64 __fastcall sub_1A5F590(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 **a5,
        char a6,
        __m128 a7,
        double a8,
        double a9,
        double a10,
        double a11,
        double a12,
        double a13,
        __m128 a14,
        void (__fastcall *a15)(__int64, __int64, char *, _QWORD),
        __int64 a16,
        __int64 a17)
{
  __int64 v18; // rdx
  __int64 v19; // r12
  __int64 v20; // rbx
  double v21; // xmm4_8
  double v22; // xmm5_8
  unsigned __int64 v23; // r12
  char v24; // al
  __int64 *v25; // rax
  __int64 v26; // rax
  __int64 v27; // r14
  __int64 v28; // rbx
  __int64 v29; // rdx
  __int64 v30; // r13
  __int64 v31; // rax
  __int64 v32; // r15
  int v33; // r8d
  int v34; // r9d
  __int64 v35; // rax
  unsigned __int64 v36; // rax
  int v37; // r8d
  int v38; // r9d
  __int64 v39; // rax
  __int64 v40; // rbx
  int v41; // r8d
  int v42; // r9d
  __int64 v43; // rdx
  int v44; // r8d
  int v45; // r9d
  unsigned __int64 v46; // rbx
  __int64 v47; // rax
  unsigned __int64 v48; // r13
  unsigned __int64 *v49; // rax
  __int64 v50; // rbx
  __int64 v51; // r14
  __int64 v52; // rax
  __int64 v53; // r15
  _QWORD *v54; // rax
  __int64 v55; // rdx
  _BYTE *v56; // rbx
  int v57; // r8d
  __int64 v58; // rcx
  unsigned __int64 v59; // rdi
  __int64 v60; // r14
  int v61; // ecx
  int v62; // ecx
  __int64 v63; // r10
  unsigned int v64; // esi
  __int64 *v65; // rdx
  __int64 v66; // r9
  _QWORD *v67; // rcx
  _QWORD *v68; // rdx
  __int64 v69; // r9
  __int64 v70; // rax
  _QWORD *v71; // rax
  __int64 v72; // r10
  unsigned int v73; // eax
  __int64 v74; // r14
  __int64 v75; // rax
  __int64 v76; // r9
  __int64 v77; // r8
  unsigned int v78; // eax
  unsigned __int64 v79; // rdx
  __int64 v80; // rcx
  __int64 v81; // rsi
  unsigned int v82; // eax
  __int64 v83; // rcx
  _QWORD *v84; // r14
  unsigned __int64 v85; // r15
  __int64 v86; // r12
  unsigned __int64 *v87; // rcx
  __int64 v88; // rax
  unsigned __int64 *v89; // rax
  _QWORD *v90; // rax
  __int64 v91; // rdx
  _QWORD *v92; // rbx
  char v93; // al
  __int64 *v94; // rbx
  __int64 *v95; // r12
  __int64 *v96; // r15
  __int64 v97; // rdx
  __int64 v98; // rdx
  __int64 *v99; // r15
  __int64 v100; // rdx
  __int64 v101; // r9
  __int64 *v102; // r10
  __int64 v103; // r14
  __int64 v104; // r15
  unsigned __int64 v105; // r14
  unsigned __int64 *v106; // rdx
  __int64 v107; // rax
  unsigned __int64 *v108; // rax
  unsigned __int64 v109; // rax
  int v110; // edx
  __int64 v111; // rbx
  _QWORD *v112; // rax
  int v113; // r13d
  __int64 v114; // rax
  __int64 v115; // r14
  __int64 v116; // r13
  __int64 v117; // rbx
  __int64 v118; // rsi
  __int64 v119; // rbx
  double v120; // xmm4_8
  double v121; // xmm5_8
  char v122; // al
  _QWORD *v123; // rdx
  unsigned int v124; // eax
  unsigned int v125; // esi
  __int64 v126; // rax
  __int64 v127; // rbx
  __int64 v128; // r13
  __int64 v129; // rax
  __int64 v130; // rsi
  __int64 v131; // r13
  unsigned int v132; // eax
  __int64 v133; // rbx
  __int64 v134; // rcx
  char v135; // al
  __int64 v136; // rcx
  _QWORD *v137; // r12
  unsigned int v138; // eax
  unsigned int v139; // esi
  unsigned int v140; // ecx
  __int64 v141; // rsi
  __int64 v142; // rax
  double v143; // xmm4_8
  double v144; // xmm5_8
  unsigned __int64 v145; // rax
  __int64 v146; // r12
  __int64 v147; // r15
  __int64 v148; // rbx
  __int64 v149; // r13
  __int64 v150; // rdx
  __int64 v151; // rsi
  __int64 v152; // r13
  unsigned int v153; // eax
  __int64 v154; // r14
  __int64 v155; // rbx
  _QWORD *v156; // rdi
  _QWORD *v157; // rax
  _BYTE *v158; // r12
  __int64 v159; // r8
  __int64 v160; // r9
  unsigned int v161; // eax
  __int64 *v162; // rbx
  __int64 v163; // rdx
  __int64 v164; // r14
  unsigned __int64 v165; // r13
  __int64 v166; // rax
  unsigned __int64 v167; // rbx
  __int64 v168; // rax
  __int64 v169; // r13
  __int64 v170; // rax
  __int64 *v171; // r14
  char v172; // al
  __int64 v173; // rsi
  __int64 v174; // r13
  __int64 v175; // rax
  __int64 v176; // rsi
  __int64 v177; // r15
  _QWORD *v178; // rax
  __int64 *v179; // rsi
  unsigned __int64 v180; // rdx
  double v181; // xmm4_8
  double v182; // xmm5_8
  __int64 v183; // rdx
  __int64 v184; // rcx
  int v185; // r8d
  int v186; // r9d
  __int64 v187; // rdx
  __int64 v188; // rcx
  int v189; // r8d
  int v190; // r9d
  __int64 *v191; // rax
  __int64 v192; // rbx
  unsigned __int64 v193; // rdx
  __int64 *v194; // r12
  __int64 *i; // r15
  __int64 *v196; // r14
  _QWORD *v197; // rax
  __int64 v198; // rdx
  unsigned __int64 v199; // rax
  __int64 v200; // rax
  __int64 *v201; // rdx
  _QWORD *v202; // rax
  __int64 v203; // rdi
  __int64 v204; // rdx
  unsigned __int64 *v205; // rcx
  __int64 v206; // rsi
  unsigned __int64 v207; // rsi
  __int64 v208; // rcx
  _QWORD *v209; // rdi
  int v210; // r11d
  __int64 v211; // rdx
  __int64 v212; // rcx
  int v213; // r8d
  _QWORD *v214; // rax
  __int64 v215; // rsi
  unsigned __int64 v216; // rdi
  __int64 v217; // rsi
  __int64 v218; // rcx
  _QWORD *v219; // rax
  __int64 v220; // rax
  _QWORD *v221; // rax
  int v222; // edx
  __int64 v223; // rax
  unsigned int v224; // esi
  unsigned __int64 *v225; // [rsp+8h] [rbp-298h]
  int v226; // [rsp+10h] [rbp-290h]
  __int64 v227; // [rsp+10h] [rbp-290h]
  __int64 v228; // [rsp+20h] [rbp-280h]
  __int64 v229; // [rsp+28h] [rbp-278h]
  int v230; // [rsp+28h] [rbp-278h]
  unsigned __int64 v231; // [rsp+28h] [rbp-278h]
  char v232; // [rsp+40h] [rbp-260h]
  __int64 v233; // [rsp+40h] [rbp-260h]
  __int64 v234; // [rsp+40h] [rbp-260h]
  __int64 v237; // [rsp+58h] [rbp-248h]
  char v238; // [rsp+78h] [rbp-228h]
  __int64 v239; // [rsp+78h] [rbp-228h]
  __int64 *v240; // [rsp+78h] [rbp-228h]
  __int64 *v241; // [rsp+78h] [rbp-228h]
  __int64 v242; // [rsp+78h] [rbp-228h]
  unsigned __int8 v245; // [rsp+97h] [rbp-209h]
  char v247; // [rsp+A0h] [rbp-200h]
  unsigned __int64 v248; // [rsp+A0h] [rbp-200h]
  __int64 v249; // [rsp+A0h] [rbp-200h]
  __int64 v250; // [rsp+A0h] [rbp-200h]
  __int64 v251; // [rsp+A0h] [rbp-200h]
  __int64 *v252; // [rsp+A0h] [rbp-200h]
  __int64 v253; // [rsp+A8h] [rbp-1F8h]
  __int64 v255; // [rsp+B8h] [rbp-1E8h]
  __int64 v256; // [rsp+B8h] [rbp-1E8h]
  __int64 v257; // [rsp+C0h] [rbp-1E0h] BYREF
  __int64 v258; // [rsp+C8h] [rbp-1D8h] BYREF
  _BYTE *v259; // [rsp+D0h] [rbp-1D0h] BYREF
  __int64 v260; // [rsp+D8h] [rbp-1C8h]
  _BYTE v261[16]; // [rsp+E0h] [rbp-1C0h] BYREF
  __int64 v262; // [rsp+F0h] [rbp-1B0h] BYREF
  __int64 v263; // [rsp+F8h] [rbp-1A8h]
  __int64 v264; // [rsp+100h] [rbp-1A0h] BYREF
  unsigned int v265; // [rsp+108h] [rbp-198h]
  __int64 v266; // [rsp+110h] [rbp-190h]
  __int64 v267; // [rsp+120h] [rbp-180h] BYREF
  _BYTE *v268; // [rsp+128h] [rbp-178h] BYREF
  _BYTE *v269; // [rsp+130h] [rbp-170h]
  __int64 v270; // [rsp+138h] [rbp-168h]
  int v271; // [rsp+140h] [rbp-160h]
  _BYTE v272[24]; // [rsp+148h] [rbp-158h] BYREF
  __m128i v273; // [rsp+160h] [rbp-140h] BYREF
  _BYTE v274[64]; // [rsp+170h] [rbp-130h] BYREF
  __int64 v275; // [rsp+1B0h] [rbp-F0h] BYREF
  __int64 v276; // [rsp+1B8h] [rbp-E8h]
  _BYTE v277[16]; // [rsp+1C0h] [rbp-E0h] BYREF
  char v278; // [rsp+1D0h] [rbp-D0h]
  __int64 v279; // [rsp+200h] [rbp-A0h] BYREF
  _BYTE *v280; // [rsp+208h] [rbp-98h]
  _BYTE *v281; // [rsp+210h] [rbp-90h]
  __int64 v282; // [rsp+218h] [rbp-88h]
  int v283; // [rsp+220h] [rbp-80h]
  _BYTE v284[120]; // [rsp+228h] [rbp-78h] BYREF

  v245 = sub_13FCBF0(a1);
  if ( !v245 )
    return 0;
  v18 = **(_QWORD **)(a1 + 32);
  v283 = 0;
  v280 = v284;
  v281 = v284;
  v253 = v18;
  v279 = 0;
  v282 = 8;
  sub_1953970((__int64)&v275, (__int64)&v279, v18);
  v238 = 0;
LABEL_5:
  v19 = *(_QWORD *)(v253 + 48);
  v20 = v253 + 40;
  if ( v253 + 40 != v19 )
  {
    if ( !v19 )
      goto LABEL_26;
LABEL_7:
    if ( (unsigned __int8)sub_15F3040(v19 - 24) || sub_15F3330(v19 - 24) )
    {
      if ( v20 != v19 )
        goto LABEL_28;
    }
    else
    {
      while ( 1 )
      {
        v19 = *(_QWORD *)(v19 + 8);
        if ( v20 == v19 )
          break;
        if ( v19 )
          goto LABEL_7;
LABEL_26:
        if ( (unsigned __int8)sub_15F3040(0) || sub_15F3330(0) )
          goto LABEL_28;
      }
    }
  }
  v23 = sub_157EBA0(v253);
  v24 = *(_BYTE *)(v23 + 16);
  if ( v24 != 27 )
  {
    if ( v24 != 26 || (*(_DWORD *)(v23 + 20) & 0xFFFFFFF) != 3 || *(_BYTE *)(*(_QWORD *)(v23 - 72) + 16LL) <= 0x10u )
      goto LABEL_28;
    v267 = 0;
    v232 = sub_13FC1A0(a1, *(_QWORD *)(v23 - 72));
    if ( v232 )
    {
      v164 = *(_QWORD *)(v23 - 72);
      v165 = v267 & 0xFFFFFFFFFFFFFFF8LL;
      if ( (v267 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
      {
        if ( (v267 & 4) == 0 )
        {
          v166 = sub_22077B0(48);
          if ( v166 )
          {
            *(_QWORD *)v166 = v166 + 16;
            *(_QWORD *)(v166 + 8) = 0x400000000LL;
          }
          v167 = v166 & 0xFFFFFFFFFFFFFFF8LL;
          v267 = v166 | 4;
          v168 = *(unsigned int *)((v166 & 0xFFFFFFFFFFFFFFF8LL) + 8);
          if ( (unsigned int)v168 >= *(_DWORD *)(v167 + 12) )
          {
            sub_16CD150(v167, (const void *)(v167 + 16), 0, 8, v41, v42);
            v168 = *(unsigned int *)(v167 + 8);
          }
          *(_QWORD *)(*(_QWORD *)v167 + 8 * v168) = v165;
          v169 = v267;
          ++*(_DWORD *)(v167 + 8);
          v165 = v169 & 0xFFFFFFFFFFFFFFF8LL;
        }
        v170 = *(unsigned int *)(v165 + 8);
        if ( (unsigned int)v170 >= *(_DWORD *)(v165 + 12) )
        {
          sub_16CD150(v165, (const void *)(v165 + 16), 0, 8, v41, v42);
          v170 = *(unsigned int *)(v165 + 8);
        }
        *(_QWORD *)(*(_QWORD *)v165 + 8 * v170) = v164;
        ++*(_DWORD *)(v165 + 8);
      }
      else
      {
        v267 = *(_QWORD *)(v23 - 72);
      }
      goto LABEL_64;
    }
    v43 = *(_QWORD *)(v23 - 72);
    if ( *(_BYTE *)(v43 + 16) <= 0x17u )
    {
LABEL_61:
      if ( (v267 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
        goto LABEL_163;
      if ( (v267 & 4) != 0 )
      {
        if ( !*(_DWORD *)((v267 & 0xFFFFFFFFFFFFFFF8LL) + 8) )
          goto LABEL_163;
      }
      else
      {
        v232 = 0;
      }
LABEL_64:
      v50 = *(_QWORD *)(v23 - 24);
      v237 = a1 + 56;
      if ( sub_1377F70(a1 + 56, v50) )
      {
        v50 = *(_QWORD *)(v23 - 48);
        if ( sub_1377F70(a1 + 56, v50) )
          goto LABEL_163;
        v247 = 0;
        v51 = -24;
        v230 = 1;
      }
      else
      {
        v51 = -48;
        v230 = 0;
        v247 = v245;
      }
      v171 = (__int64 *)(v23 + v51);
      v228 = *v171;
      v256 = *(_QWORD *)(v23 + 40);
      if ( !sub_1A4FD20(a1, v256, v50) )
      {
LABEL_163:
        v32 = a1;
        sub_1A517A0((unsigned __int64 **)&v267);
        goto LABEL_29;
      }
      if ( v232 )
      {
        if ( !a17 )
        {
          v174 = sub_13FC520(a1);
          v242 = sub_1AA91E0(v174, **(_QWORD **)(a1 + 32), a2, a3);
          goto LABEL_313;
        }
      }
      else
      {
        v172 = *(_BYTE *)(*(_QWORD *)(v23 - 72) + 16LL);
        if ( v247 )
        {
          if ( v172 != 51 )
            goto LABEL_163;
        }
        else if ( v172 != 50 )
        {
          goto LABEL_163;
        }
        if ( !a17 )
        {
          v174 = sub_13FC520(a1);
          v242 = sub_1AA91E0(v174, **(_QWORD **)(a1 + 32), a2, a3);
LABEL_277:
          v175 = *(_QWORD *)(v50 + 48);
          if ( v175 )
            v176 = v175 - 24;
          else
            v176 = 0;
          v177 = sub_1AA8CA0(v50, v176, a2, a3);
          v178 = (_QWORD *)sub_157EBA0(v174);
          sub_15F20C0(v178);
          v179 = 0;
          v180 = v267 & 0xFFFFFFFFFFFFFFF8LL;
          if ( (v267 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
          {
            if ( (v267 & 4) != 0 )
            {
              v179 = *(__int64 **)v180;
              v180 = *(unsigned int *)(v180 + 8);
            }
            else
            {
              v179 = &v267;
              v180 = 1;
            }
          }
          sub_1A4F120(v174, (__int64)v179, v180, v247, v177, v242, *(double *)a7.m128_u64, a8, a9);
          goto LABEL_283;
        }
      }
      v173 = sub_13AE450(a3, v50);
      if ( v173 )
        sub_1465150(a17, v173);
      else
        sub_1465DB0(a17, (_QWORD *)a1);
      v174 = sub_13FC520(a1);
      v242 = sub_1AA91E0(v174, **(_QWORD **)(a1 + 32), a2, a3);
      if ( !v232 )
        goto LABEL_277;
LABEL_313:
      if ( sub_157F120(v50) )
      {
        v177 = v50;
        v202 = (_QWORD *)sub_157EBA0(v174);
        sub_15F20C0(v202);
      }
      else
      {
        v220 = *(_QWORD *)(v50 + 48);
        if ( v220 )
          v177 = sub_1AA8CA0(v50, v220 - 24, a2, a3);
        else
          v177 = sub_1AA8CA0(v50, 0, a2, a3);
        v221 = (_QWORD *)sub_157EBA0(v174);
        sub_15F20C0(v221);
      }
      v203 = v174 + 40;
      v204 = v23 + 24;
      v205 = *(unsigned __int64 **)(v23 + 32);
      if ( v174 + 40 != v23 + 24 && (unsigned __int64 *)v203 != v205 )
      {
        v206 = *(_QWORD *)(v23 + 40) + 40LL;
        if ( v203 != v206 )
        {
          v225 = *(unsigned __int64 **)(v23 + 32);
          sub_157EA80(v203, v206, v204, (__int64)v205);
          v205 = v225;
          v204 = v23 + 24;
          v203 = v174 + 40;
        }
        if ( (unsigned __int64 *)v203 != v205 && (unsigned __int64 *)v204 != v205 )
        {
          v207 = *v205 & 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)((*(_QWORD *)(v23 + 24) & 0xFFFFFFFFFFFFFFF8LL) + 8) = v205;
          *v205 = *v205 & 7 | *(_QWORD *)(v23 + 24) & 0xFFFFFFFFFFFFFFF8LL;
          v208 = *(_QWORD *)(v174 + 40);
          *(_QWORD *)(v207 + 8) = v203;
          v208 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v23 + 24) = v208 | *(_QWORD *)(v23 + 24) & 7LL;
          *(_QWORD *)(v208 + 8) = v204;
          *(_QWORD *)(v174 + 40) = v207 | *(_QWORD *)(v174 + 40) & 7LL;
        }
      }
      sub_1593B40((_QWORD *)(v23 - 24LL * v230 - 24), v177);
      sub_1593B40(v171, v242);
      v209 = sub_1648A60(56, 1u);
      if ( v209 )
        sub_15F8590((__int64)v209, v228, v256);
LABEL_283:
      if ( v177 == v50 )
        sub_1A4F070(v177, v174);
      else
        sub_1A4FE90(v50, v177, v256, v174, v232, a7, a8, a9, a10, v181, v182, a13, a14);
      v273.m128i_i64[0] = v174;
      v276 = 0x200000000LL;
      v275 = (__int64)v277;
      v273.m128i_i64[1] = v177 & 0xFFFFFFFFFFFFFFFBLL;
      sub_1A51800((__int64)&v275, &v273, v183, v184, v185, v186);
      if ( v232 )
      {
        v273.m128i_i64[1] = v50 | 4;
        v273.m128i_i64[0] = v256;
        sub_1A51800((__int64)&v275, &v273, v187, v188, v189, v190);
      }
      sub_15DC140(a2, (__int64 *)v275, (unsigned int)v276);
      v191 = (__int64 *)sub_16498A0(v23);
      if ( v247 )
        v192 = sub_159C540(v191);
      else
        v192 = sub_159C4F0(v191);
      v193 = v267 & 0xFFFFFFFFFFFFFFF8LL;
      if ( (v267 & 4) != 0 )
      {
        v194 = *(__int64 **)v193;
        v252 = (__int64 *)(*(_QWORD *)v193 + 8LL * *(unsigned int *)(v193 + 8));
        if ( *(__int64 **)v193 == v252 )
          goto LABEL_305;
      }
      else
      {
        if ( !v193 )
          goto LABEL_305;
        v194 = &v267;
        v252 = (__int64 *)&v268;
      }
      do
      {
        for ( i = *(__int64 **)(*v194 + 8); i; *(_QWORD *)(v192 + 8) = v196 )
        {
          while ( 1 )
          {
            v196 = i;
            i = (__int64 *)i[1];
            v197 = sub_1648700((__int64)v196);
            if ( *((_BYTE *)v197 + 16) > 0x17u && sub_1377F70(v237, v197[5]) )
            {
              if ( *v196 )
              {
                v198 = v196[1];
                v199 = v196[2] & 0xFFFFFFFFFFFFFFFCLL;
                *(_QWORD *)v199 = v198;
                if ( v198 )
                  *(_QWORD *)(v198 + 16) = *(_QWORD *)(v198 + 16) & 3LL | v199;
              }
              *v196 = v192;
              if ( v192 )
                break;
            }
            if ( !i )
              goto LABEL_304;
          }
          v200 = *(_QWORD *)(v192 + 8);
          v196[1] = v200;
          if ( v200 )
            *(_QWORD *)(v200 + 16) = (unsigned __int64)(v196 + 1) | *(_QWORD *)(v200 + 16) & 3LL;
          v196[2] = (v192 + 8) | v196[2] & 3;
        }
LABEL_304:
        ++v194;
      }
      while ( v252 != v194 );
LABEL_305:
      if ( v232 )
        sub_1A56890(a1, v242, a2, a3);
      if ( (_BYTE *)v275 != v277 )
        _libc_free(v275);
      sub_1A517A0((unsigned __int64 **)&v267);
      v109 = sub_157EBA0(v253);
      goto LABEL_151;
    }
    sub_1A511F0(&v275, a1, v43);
    v46 = v275 & 0xFFFFFFFFFFFFFFF8LL;
    if ( (v275 & 0xFFFFFFFFFFFFFFF8LL) == 0 || (v47 = (v275 >> 2) & 1, (_DWORD)v47) && !*(_DWORD *)(v46 + 8) )
    {
      if ( (v267 & 4) != 0 )
      {
        if ( (v267 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
          *(_DWORD *)((v267 & 0xFFFFFFFFFFFFFFF8LL) + 8) = 0;
      }
      else
      {
        v267 = 0;
      }
      goto LABEL_60;
    }
    if ( (v267 & 4) != 0 )
    {
      v48 = v267 & 0xFFFFFFFFFFFFFFF8LL;
      if ( (v267 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
      {
        if ( !(_BYTE)v47 )
        {
          v222 = *(_DWORD *)(v48 + 12);
          *(_DWORD *)(v48 + 8) = 0;
          v223 = 0;
          if ( !v222 )
          {
            sub_16CD150(v48, (const void *)(v48 + 16), 0, 8, v44, v45);
            v223 = 8LL * *(unsigned int *)(v48 + 8);
          }
          *(_QWORD *)(*(_QWORD *)v48 + v223) = v46;
          ++*(_DWORD *)(v48 + 8);
          v275 = 0;
          goto LABEL_60;
        }
        if ( *(_QWORD *)v48 != v48 + 16 )
          _libc_free(*(_QWORD *)v48);
        j_j___libc_free_0(v48, 48);
      }
    }
    v49 = (unsigned __int64 *)v275;
    v275 = 0;
    v267 = (__int64)v49;
LABEL_60:
    sub_1A517A0((unsigned __int64 **)&v275);
    goto LABEL_61;
  }
  if ( (*(_BYTE *)(v23 + 23) & 0x40) != 0 )
    v25 = *(__int64 **)(v23 - 8);
  else
    v25 = (__int64 *)(v23 - 24LL * (*(_DWORD *)(v23 + 20) & 0xFFFFFFF));
  v26 = *v25;
  v229 = v26;
  if ( *(_BYTE *)(v26 + 16) <= 0x10u || !sub_13FC1A0(a1, v26) )
  {
LABEL_28:
    v32 = a1;
    goto LABEL_29;
  }
  v27 = 0;
  v255 = *(_QWORD *)(v23 + 40);
  v259 = v261;
  v260 = 0x400000000LL;
  v28 = ((*(_DWORD *)(v23 + 20) & 0xFFFFFFFu) >> 1) - 1;
  v237 = a1 + 56;
  if ( (*(_DWORD *)(v23 + 20) & 0xFFFFFFFu) >> 1 != 1 )
  {
    do
    {
      while ( 1 )
      {
        v31 = 24;
        if ( (_DWORD)v27 != -2 )
          v31 = 24LL * (unsigned int)(2 * v27 + 3);
        v29 = (*(_BYTE *)(v23 + 23) & 0x40) != 0
            ? *(_QWORD *)(v23 - 8)
            : v23 - 24LL * (*(_DWORD *)(v23 + 20) & 0xFFFFFFF);
        v30 = *(_QWORD *)(v29 + v31);
        if ( !sub_1377F70(a1 + 56, v30) && sub_1A4FD20(a1, v255, v30) )
          break;
        if ( v28 == ++v27 )
          goto LABEL_39;
      }
      v35 = (unsigned int)v260;
      if ( (unsigned int)v260 >= HIDWORD(v260) )
      {
        sub_16CD150((__int64)&v259, v261, 0, 4, v33, v34);
        v35 = (unsigned int)v260;
      }
      *(_DWORD *)&v259[4 * v35] = v27++;
      LODWORD(v260) = v260 + 1;
    }
    while ( v28 != v27 );
  }
LABEL_39:
  v257 = 0;
  if ( (*(_BYTE *)(v23 + 23) & 0x40) != 0 )
    v36 = *(_QWORD *)(v23 - 8);
  else
    v36 = v23 - 24LL * (*(_DWORD *)(v23 + 20) & 0xFFFFFFF);
  if ( !sub_1377F70(v237, *(_QWORD *)(v36 + 24)) )
  {
    v39 = sub_13CF970(v23);
    if ( sub_1A4FD20(a1, v255, *(_QWORD *)(v39 + 24)) )
    {
      v40 = *(_QWORD *)(sub_13CF970(v23) + 24);
      if ( *(_BYTE *)(sub_157EBA0(v40) + 16) != 31 )
      {
        v257 = v40;
        goto LABEL_71;
      }
    }
  }
  if ( (_DWORD)v260 )
  {
    v40 = v257;
LABEL_71:
    if ( v40 )
    {
      v52 = sub_13CF970(v23);
      sub_1593B40((_QWORD *)(v52 + 24), 0);
      v53 = sub_13AE450(a3, v257);
      if ( v53 )
      {
        v54 = (_QWORD *)a1;
        if ( a1 != v53 )
        {
          while ( 1 )
          {
            v54 = (_QWORD *)*v54;
            if ( (_QWORD *)v53 == v54 )
              break;
            if ( !v54 )
              goto LABEL_161;
          }
        }
      }
    }
    else
    {
LABEL_161:
      v53 = a1;
    }
    v55 = (unsigned int)v260;
    v273.m128i_i64[0] = (__int64)v274;
    v273.m128i_i64[1] = 0x400000000LL;
    if ( (unsigned int)v260 > 4 )
    {
      sub_16CD150((__int64)&v273, v274, (unsigned int)v260, 16, v37, v38);
      v55 = (unsigned int)v260;
    }
    v56 = &v259[4 * v55];
    if ( v259 != v56 )
    {
      v248 = (unsigned __int64)v259;
      do
      {
        v57 = *((_DWORD *)v56 - 1);
        v58 = 24;
        if ( v57 != -2 )
          v58 = 24LL * (unsigned int)(2 * v57 + 3);
        if ( (*(_BYTE *)(v23 + 23) & 0x40) != 0 )
          v59 = *(_QWORD *)(v23 - 8);
        else
          v59 = v23 - 24LL * (*(_DWORD *)(v23 + 20) & 0xFFFFFFF);
        v60 = *(_QWORD *)(v59 + v58);
        v61 = *(_DWORD *)(a3 + 24);
        if ( v61 )
        {
          v62 = v61 - 1;
          v63 = *(_QWORD *)(a3 + 8);
          v64 = v62 & (((unsigned int)v60 >> 9) ^ ((unsigned int)v60 >> 4));
          v65 = (__int64 *)(v63 + 16LL * v64);
          v66 = *v65;
          if ( v60 == *v65 )
          {
LABEL_87:
            v67 = (_QWORD *)v65[1];
            if ( v67 )
            {
              if ( v67 != (_QWORD *)v53 && v53 )
              {
                v68 = (_QWORD *)v53;
                while ( 1 )
                {
                  v68 = (_QWORD *)*v68;
                  if ( v67 == v68 )
                    break;
                  if ( !v68 )
                    goto LABEL_94;
                }
                v53 = (__int64)v67;
              }
              goto LABEL_94;
            }
          }
          else
          {
            v110 = 1;
            while ( v66 != -8 )
            {
              v210 = v110 + 1;
              v64 = v62 & (v110 + v64);
              v65 = (__int64 *)(v63 + 16LL * v64);
              v66 = *v65;
              if ( v60 == *v65 )
                goto LABEL_87;
              v110 = v210;
            }
          }
        }
        v53 = 0;
LABEL_94:
        v69 = *(_QWORD *)(v59 + 24LL * (unsigned int)(2 * v57 + 2));
        v70 = v273.m128i_u32[2];
        if ( v273.m128i_i32[2] >= (unsigned __int32)v273.m128i_i32[3] )
        {
          v226 = *((_DWORD *)v56 - 1);
          v233 = *(_QWORD *)(v59 + 24LL * (unsigned int)(2 * v57 + 2));
          sub_16CD150((__int64)&v273, v274, 0, 16, v57, v69);
          v70 = v273.m128i_u32[2];
          v57 = v226;
          v69 = v233;
        }
        v71 = (_QWORD *)(v273.m128i_i64[0] + 16 * v70);
        v56 -= 4;
        *v71 = v69;
        v71[1] = v60;
        ++v273.m128i_i32[2];
        sub_15FFDB0(v23, v23, v57);
      }
      while ( (_BYTE *)v248 != v56 );
    }
    if ( a17 )
    {
      if ( v53 )
        sub_1465150(a17, v53);
      else
        sub_1465DB0(a17, (_QWORD *)a1);
    }
    v72 = *(_DWORD *)(v23 + 20) & 0xFFFFFFF;
    v73 = (*(_DWORD *)(v23 + 20) & 0xFFFFFFFu) >> 1;
    if ( v73 == 1 )
    {
      if ( !v257 )
      {
        v234 = *(_QWORD *)(sub_13CF970(v23) + 24);
        goto LABEL_166;
      }
LABEL_165:
      v234 = 0;
      goto LABEL_166;
    }
    v74 = v73 - 1;
    v75 = v74 - 1;
    if ( (v74 - 1) >> 2 )
    {
      v76 = 2;
      v77 = 1;
      v78 = 5;
      while ( 1 )
      {
        if ( (*(_BYTE *)(v23 + 23) & 0x40) != 0 )
        {
          v79 = *(_QWORD *)(v23 - 8);
          v80 = *(_QWORD *)(v79 + 24LL * v78);
        }
        else
        {
          v80 = *(_QWORD *)(v23 + 24 * (v78 - v72));
          v79 = v23 - 24 * v72;
        }
        if ( *(_QWORD *)(v79 + 72) != v80 )
          break;
        v81 = v76;
        if ( v80 != *(_QWORD *)(v79 + 24LL * (v78 + 2)) )
          goto LABEL_246;
        v81 = v77 + 2;
        if ( v80 != *(_QWORD *)(v79 + 24LL * (v78 + 4)) )
          goto LABEL_246;
        v81 = v77 + 3;
        if ( v80 != *(_QWORD *)(v79 + 24LL * (v78 + 6)) )
          goto LABEL_246;
        v77 += 4;
        v78 += 8;
        v76 += 4;
        if ( v77 == 4 * ((v74 - 1) >> 2) + 1 )
        {
          v75 = v74 - v77;
          goto LABEL_241;
        }
      }
LABEL_245:
      v81 = v77;
LABEL_246:
      v234 = 0;
      if ( v74 != v81 )
      {
LABEL_247:
        if ( v257 || *(_QWORD *)(sub_13CF970(v23) + 24) == v234 )
        {
LABEL_166:
          v111 = sub_13FC520(a1);
          v249 = v111;
          v227 = sub_1AA91E0(v111, **(_QWORD **)(a1 + 32), a2, a3);
          v112 = (_QWORD *)sub_157EBA0(v111);
          sub_15F20C0(v112);
          v113 = v273.m128i_i32[2];
          v114 = sub_1648B60(64);
          v115 = v114;
          if ( v114 )
            sub_15FFB20(v114, v229, v227, v113, v111);
          v116 = v257;
          v267 = 0;
          v268 = v272;
          v269 = v272;
          v270 = 2;
          v271 = 0;
          v262 = 0;
          v263 = 1;
          v264 = -8;
          v266 = -8;
          if ( !v257 )
          {
            v239 = v273.m128i_i64[0];
            v127 = v273.m128i_i64[0] + 16LL * v273.m128i_u32[2];
            if ( v273.m128i_i64[0] == v127 )
              goto LABEL_237;
            goto LABEL_193;
          }
          v117 = *(_QWORD *)(v257 + 8);
          if ( !v117 )
          {
LABEL_191:
            sub_1953970((__int64)&v275, (__int64)&v267, v116);
            sub_1A4F070(v257, v249);
LABEL_182:
            v239 = v273.m128i_i64[0];
            v127 = v273.m128i_i64[0] + 16LL * v273.m128i_u32[2];
            if ( v273.m128i_i64[0] == v127 )
              goto LABEL_183;
LABEL_193:
            v231 = v23;
            while ( 1 )
            {
              v146 = *(_QWORD *)(v127 - 8);
              v127 -= 16;
              v147 = *(_QWORD *)(v146 + 8);
              v258 = v146;
              if ( !v147 )
              {
LABEL_210:
                sub_1953970((__int64)&v275, (__int64)&v267, v146);
                if ( v278 )
                  sub_1A4F070(v258, v249);
                goto LABEL_206;
              }
              while ( (unsigned __int8)(*((_BYTE *)sub_1648700(v147) + 16) - 25) > 9u )
              {
                v147 = *(_QWORD *)(v147 + 8);
                if ( !v147 )
                  goto LABEL_210;
              }
              v135 = sub_1A54210((__int64)&v262, &v258, &v275);
              v137 = (_QWORD *)v275;
              if ( !v135 )
                break;
              v145 = *(_QWORD *)(v275 + 8);
              if ( !v145 )
                goto LABEL_202;
LABEL_205:
              *(_QWORD *)(v127 + 8) = v145;
LABEL_206:
              if ( v127 == v239 )
              {
                v148 = v273.m128i_i64[0];
                v23 = v231;
                v149 = v273.m128i_i64[0] + 16LL * v273.m128i_u32[2];
                if ( v273.m128i_i64[0] != v149 )
                {
                  do
                  {
                    v150 = *(_QWORD *)(v149 - 8);
                    v151 = *(_QWORD *)(v149 - 16);
                    v149 -= 16;
                    sub_15FFFB0(v115, v151, v150, v136, v159, v160);
                  }
                  while ( v148 != v149 );
                }
LABEL_183:
                v128 = v257;
                if ( v257 )
                {
                  v129 = sub_13CF970(v115);
                  v130 = v128;
                  v131 = 0;
                  sub_1593B40((_QWORD *)(v129 + 24), v130);
                  v132 = (*(_DWORD *)(v23 + 20) & 0xFFFFFFFu) >> 1;
                  v133 = v132 - 1;
                  if ( v132 != 1 )
                  {
                    do
                    {
                      ++v131;
                      if ( (*(_BYTE *)(v23 + 23) & 0x40) != 0 )
                        v134 = *(_QWORD *)(v23 - 8);
                      else
                        v134 = v23 - 24LL * (*(_DWORD *)(v23 + 20) & 0xFFFFFFF);
                      sub_15FFFB0(v115, *(_QWORD *)(v134 + 24LL * (unsigned int)(2 * v131)), v227, v134, v159, v160);
                    }
                    while ( v133 != v131 );
                  }
                  if ( v234 )
                  {
                    v152 = *(_QWORD *)(v23 + 40);
                    v153 = (*(_DWORD *)(v23 + 20) & 0xFFFFFFFu) >> 1;
                    v154 = v153 - 1;
                    if ( v153 == 1 )
                      goto LABEL_226;
                    v155 = 1;
                    if ( v257 )
                    {
                      if ( v154 != 1 )
                        goto LABEL_224;
                      goto LABEL_226;
                    }
LABEL_225:
                    while ( 1 )
                    {
                      sub_157F2D0(v234, v152, 1);
                      if ( v155 == v154 )
                        break;
LABEL_224:
                      ++v155;
                    }
LABEL_226:
                    sub_15F20C0((_QWORD *)v23);
                    v156 = sub_1648A60(56, 1u);
                    if ( v156 )
                      sub_15F8590((__int64)v156, v234, v152);
                  }
                  else if ( v257 )
                  {
                    v211 = sub_1A4F510(v23, ((*(_DWORD *)(v23 + 20) & 0xFFFFFFFu) >> 1) - 2);
                    if ( (*(_BYTE *)(v23 + 23) & 0x40) != 0 )
                      v214 = *(_QWORD **)(v23 - 8);
                    else
                      v214 = (_QWORD *)(v23 - 24 * v212);
                    if ( v214[3] )
                    {
                      v215 = v214[4];
                      v216 = v214[5] & 0xFFFFFFFFFFFFFFFCLL;
                      *(_QWORD *)v216 = v215;
                      if ( v215 )
                        *(_QWORD *)(v215 + 16) = v216 | *(_QWORD *)(v215 + 16) & 3LL;
                    }
                    v214[3] = v211;
                    if ( v211 )
                    {
                      v217 = *(_QWORD *)(v211 + 8);
                      v214[4] = v217;
                      if ( v217 )
                        *(_QWORD *)(v217 + 16) = (unsigned __int64)(v214 + 4) | *(_QWORD *)(v217 + 16) & 3LL;
                      v218 = v214[5];
                      v219 = v214 + 3;
                      v219[2] = (v211 + 8) | v218 & 3;
                      *(_QWORD *)(v211 + 8) = v219;
                    }
                    sub_15FFDB0(v23, v23, v213);
                  }
                  goto LABEL_228;
                }
LABEL_237:
                if ( v234 )
                {
                  v152 = *(_QWORD *)(v23 + 40);
                  v161 = (*(_DWORD *)(v23 + 20) & 0xFFFFFFFu) >> 1;
                  v154 = v161 - 1;
                  if ( v161 == 1 )
                    goto LABEL_226;
                  v155 = 1;
                  goto LABEL_225;
                }
LABEL_228:
                v276 = 0x400000000LL;
                v157 = v269;
                v275 = (__int64)v277;
                if ( v269 == v268 )
                  v158 = &v269[8 * HIDWORD(v270)];
                else
                  v158 = &v269[8 * (unsigned int)v270];
                if ( v269 != v158 )
                {
                  while ( 1 )
                  {
                    v91 = *v157;
                    v92 = v157;
                    if ( *v157 < 0xFFFFFFFFFFFFFFFELL )
                      break;
                    if ( v158 == (_BYTE *)++v157 )
                      goto LABEL_120;
                  }
                  if ( v158 != (_BYTE *)v157 )
                  {
                    v82 = 4;
                    v83 = 0;
                    v84 = v158;
                    while ( 1 )
                    {
                      v85 = v91 & 0xFFFFFFFFFFFFFFFBLL;
                      v86 = v91 | 4;
                      if ( (unsigned int)v83 >= v82 )
                      {
                        sub_16CD150((__int64)&v275, v277, 0, 16, v159, v160);
                        v83 = (unsigned int)v276;
                      }
                      v87 = (unsigned __int64 *)(v275 + 16 * v83);
                      v87[1] = v86;
                      *v87 = v255;
                      v88 = (unsigned int)(v276 + 1);
                      LODWORD(v276) = v88;
                      if ( HIDWORD(v276) <= (unsigned int)v88 )
                      {
                        sub_16CD150((__int64)&v275, v277, 0, 16, v159, v160);
                        v88 = (unsigned int)v276;
                      }
                      v89 = (unsigned __int64 *)(v275 + 16 * v88);
                      v89[1] = v85;
                      *v89 = v249;
                      v83 = (unsigned int)(v276 + 1);
                      v90 = v92 + 1;
                      LODWORD(v276) = v276 + 1;
                      if ( v92 + 1 == v84 )
                        break;
                      while ( 1 )
                      {
                        v91 = *v90;
                        v92 = v90;
                        if ( *v90 < 0xFFFFFFFFFFFFFFFELL )
                          break;
                        if ( v84 == ++v90 )
                          goto LABEL_120;
                      }
                      if ( v84 == v90 )
                        break;
                      v82 = HIDWORD(v276);
                    }
                  }
                }
LABEL_120:
                v93 = v263 & 1;
                if ( !((unsigned int)v263 >> 1) )
                {
                  if ( v93 )
                  {
                    v163 = 4;
                    v162 = &v264;
                  }
                  else
                  {
                    v162 = (__int64 *)v264;
                    v163 = 2LL * v265;
                  }
                  v94 = &v162[v163];
                  v95 = v94;
                  goto LABEL_126;
                }
                if ( v93 )
                {
                  v94 = &v267;
                  v95 = &v264;
LABEL_123:
                  while ( *v95 == -8 || *v95 == -16 )
                  {
                    v95 += 2;
                    if ( v94 == v95 )
                      goto LABEL_126;
                  }
                  v201 = v94;
                  v94 = v95;
                  v95 = v201;
LABEL_126:
                  if ( !v93 )
                  {
                    v96 = (__int64 *)v264;
                    v97 = v265;
                    goto LABEL_128;
                  }
                  v98 = 4;
                  v96 = &v264;
                }
                else
                {
                  v97 = v265;
                  v96 = (__int64 *)v264;
                  v95 = (__int64 *)v264;
                  v94 = (__int64 *)(v264 + 16LL * v265);
                  if ( (__int64 *)v264 != v94 )
                    goto LABEL_123;
LABEL_128:
                  v98 = 2 * v97;
                }
                v99 = &v96[v98];
                v100 = (unsigned int)v276;
                if ( v99 != v94 )
                {
                  v101 = v249;
                  v102 = v99;
                  do
                  {
                    v103 = v94[1];
                    v104 = v103 | 4;
                    v105 = v103 & 0xFFFFFFFFFFFFFFFBLL;
                    if ( (unsigned int)v100 >= HIDWORD(v276) )
                    {
                      v241 = v102;
                      v251 = v101;
                      sub_16CD150((__int64)&v275, v277, 0, 16, v159, v101);
                      v100 = (unsigned int)v276;
                      v102 = v241;
                      v101 = v251;
                    }
                    v106 = (unsigned __int64 *)(v275 + 16 * v100);
                    *v106 = v255;
                    v106[1] = v104;
                    v107 = (unsigned int)(v276 + 1);
                    LODWORD(v276) = v107;
                    if ( HIDWORD(v276) <= (unsigned int)v107 )
                    {
                      v240 = v102;
                      v250 = v101;
                      sub_16CD150((__int64)&v275, v277, 0, 16, v159, v101);
                      v107 = (unsigned int)v276;
                      v102 = v240;
                      v101 = v250;
                    }
                    v108 = (unsigned __int64 *)(v275 + 16 * v107);
                    *v108 = v101;
                    v108[1] = v105;
                    v100 = (unsigned int)(v276 + 1);
                    LODWORD(v276) = v276 + 1;
                    do
                      v94 += 2;
                    while ( v94 != v95 && (*v94 == -16 || *v94 == -8) );
                  }
                  while ( v94 != v102 );
                }
                sub_15DC140(a2, (__int64 *)v275, v100);
                sub_1A56890(a1, v227, a2, a3);
                if ( (_BYTE *)v275 != v277 )
                  _libc_free(v275);
                if ( (v263 & 1) == 0 )
                  j___libc_free_0(v264);
                if ( v269 != v268 )
                  _libc_free((unsigned __int64)v269);
                if ( (_BYTE *)v273.m128i_i64[0] != v274 )
                  _libc_free(v273.m128i_u64[0]);
                if ( v259 != v261 )
                  _libc_free((unsigned __int64)v259);
                v109 = sub_157EBA0(v253);
                if ( *(_BYTE *)(v109 + 16) != 26 )
                {
LABEL_154:
                  if ( v280 != v281 )
                    _libc_free((unsigned __int64)v281);
                  goto LABEL_156;
                }
LABEL_151:
                if ( (*(_DWORD *)(v109 + 20) & 0xFFFFFFF) == 3 )
                  goto LABEL_154;
                v253 = *(_QWORD *)(v109 - 24);
                if ( !sub_1377F70(v237, v253) )
                  goto LABEL_154;
                sub_1953970((__int64)&v275, (__int64)&v279, v253);
                v238 = v278;
                if ( !v278 )
                  goto LABEL_154;
                goto LABEL_5;
              }
            }
            ++v262;
            v138 = ((unsigned int)v263 >> 1) + 1;
            if ( (v263 & 1) != 0 )
            {
              v140 = 6;
              v139 = 2;
            }
            else
            {
              v139 = v265;
              v140 = 3 * v265;
            }
            if ( v140 <= 4 * v138 )
            {
              v139 *= 2;
            }
            else if ( v139 - (v138 + HIDWORD(v263)) > v139 >> 3 )
            {
LABEL_199:
              LODWORD(v263) = v263 & 1 | (2 * v138);
              if ( *v137 != -8 )
                --HIDWORD(v263);
              v137[1] = 0;
              *v137 = v258;
LABEL_202:
              v141 = *(_QWORD *)(v258 + 48);
              if ( v141 )
                v141 -= 24;
              v142 = sub_1AA8CA0(v258, v141, a2, a3);
              v137[1] = v142;
              sub_1A4FE90(v258, v142, v255, v249, 1, a7, a8, a9, a10, v143, v144, a13, a14);
              v145 = v137[1];
              goto LABEL_205;
            }
            sub_1A54810((__int64)&v262, v139);
            sub_1A54210((__int64)&v262, &v258, &v275);
            v137 = (_QWORD *)v275;
            v138 = ((unsigned int)v263 >> 1) + 1;
            goto LABEL_199;
          }
          while ( (unsigned __int8)(*((_BYTE *)sub_1648700(v117) + 16) - 25) > 9u )
          {
            v117 = *(_QWORD *)(v117 + 8);
            if ( !v117 )
              goto LABEL_191;
          }
          v118 = *(_QWORD *)(v116 + 48);
          if ( v118 )
            v118 -= 24;
          v119 = sub_1AA8CA0(v116, v118, a2, a3);
          sub_1A4FE90(v257, v119, v255, v249, 1, a7, a8, a9, a10, v120, v121, a13, a14);
          v122 = sub_1A54210((__int64)&v262, &v257, &v275);
          v123 = (_QWORD *)v275;
          if ( v122 )
          {
LABEL_181:
            v123[1] = v119;
            v257 = v119;
            goto LABEL_182;
          }
          ++v262;
          v124 = ((unsigned int)v263 >> 1) + 1;
          if ( (v263 & 1) != 0 )
          {
            v125 = 6;
            LODWORD(v159) = 2;
          }
          else
          {
            LODWORD(v159) = v265;
            v125 = 3 * v265;
          }
          if ( v125 <= 4 * v124 )
          {
            v224 = 2 * v159;
          }
          else
          {
            if ( (unsigned int)v159 - (v124 + HIDWORD(v263)) > (unsigned int)v159 >> 3 )
            {
LABEL_178:
              LODWORD(v263) = v263 & 1 | (2 * v124);
              if ( *v123 != -8 )
                --HIDWORD(v263);
              v126 = v257;
              v123[1] = 0;
              *v123 = v126;
              goto LABEL_181;
            }
            v224 = v159;
          }
          sub_1A54810((__int64)&v262, v224);
          sub_1A54210((__int64)&v262, &v257, &v275);
          v123 = (_QWORD *)v275;
          v124 = ((unsigned int)v263 >> 1) + 1;
          goto LABEL_178;
        }
        goto LABEL_165;
      }
LABEL_333:
      v234 = sub_1A4F510(v23, 0);
      goto LABEL_247;
    }
    v77 = 1;
LABEL_241:
    if ( v75 != 2 )
    {
      if ( v75 != 3 )
      {
        if ( v75 != 1 )
          goto LABEL_333;
        goto LABEL_244;
      }
      v275 = v23;
      v276 = v77;
      if ( !sub_1A4FE10(v23, &v275) )
        goto LABEL_245;
      ++v77;
    }
    v275 = v23;
    v276 = v77;
    if ( !sub_1A4FE10(v23, &v275) )
      goto LABEL_245;
    ++v77;
LABEL_244:
    v275 = v23;
    v276 = v77;
    if ( sub_1A4FE10(v23, &v275) )
      goto LABEL_333;
    goto LABEL_245;
  }
  v32 = a1;
  if ( v259 != v261 )
    _libc_free((unsigned __int64)v259);
LABEL_29:
  if ( v281 != v280 )
    _libc_free((unsigned __int64)v281);
  if ( v238 )
  {
LABEL_156:
    a15(a16, 1, 0, 0);
    return v245;
  }
  if ( !a6 && !byte_4FB43A0 )
    return 0;
  return sub_1A5E350(v32, a2, a3, a4, a5, a17, a7, a8, a9, a10, v21, v22, a13, a14, a15, a16);
}
