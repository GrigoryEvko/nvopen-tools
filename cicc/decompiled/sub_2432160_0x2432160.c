// Function: sub_2432160
// Address: 0x2432160
//
_QWORD *__fastcall sub_2432160(_QWORD *a1, __int16 *a2, __int64 **a3, __int64 a4)
{
  __int64 v7; // rdx
  __int16 v8; // ax
  _BYTE *v9; // rsi
  _BYTE *v10; // rsi
  __int64 v11; // rdx
  __int64 v12; // rax
  __m128i v13; // xmm1
  __m128i v14; // xmm2
  __int64 v15; // rax
  __m128i v16; // xmm0
  void (__fastcall *v17)(_QWORD, _QWORD, _QWORD); // rax
  const char *v18; // rsi
  __m128i *v19; // rdi
  unsigned __int64 v20; // rdx
  __int64 **v21; // r15
  __int64 **v22; // r14
  __int64 *v23; // r10
  __int64 **v24; // r15
  __int64 v25; // rbx
  __int64 v26; // r8
  __int64 v27; // rbx
  __int64 i; // r12
  __int64 *v29; // r15
  __int64 v30; // rsi
  unsigned __int64 v31; // r12
  unsigned __int64 v32; // r12
  __int64 v33; // r14
  _QWORD *v34; // r13
  _QWORD *v35; // r12
  __int64 *v36; // r14
  __int64 *v37; // rax
  unsigned __int64 v38; // rax
  __int64 v39; // r11
  __int64 v40; // r10
  __int64 v41; // r8
  __int64 v42; // rax
  __int64 v43; // rdx
  bool v44; // zf
  __int64 v45; // rax
  __int64 v46; // rax
  __int64 v47; // rsi
  unsigned __int8 *v48; // rsi
  __int64 v49; // rax
  bool v50; // al
  __int64 v51; // r9
  __int64 v52; // rax
  unsigned __int64 v53; // rdx
  unsigned __int64 v54; // rax
  __int64 *v55; // rdi
  int v56; // eax
  __int64 v57; // rdx
  unsigned int v58; // esi
  unsigned int v59; // edx
  _QWORD *v60; // rcx
  _QWORD *v61; // rdi
  unsigned __int64 v62; // rax
  unsigned __int64 v63; // r13
  __int64 *v64; // r14
  _QWORD *v65; // r15
  _QWORD *v66; // r13
  __int64 *v67; // rax
  unsigned __int64 v68; // r14
  unsigned __int64 v69; // rax
  int v70; // edx
  __int64 v71; // r10
  unsigned __int64 v72; // rax
  int v73; // edx
  __int64 v74; // r14
  __int64 v75; // rsi
  unsigned __int8 *v76; // rsi
  unsigned int v77; // ecx
  int v78; // edx
  _QWORD *v79; // rax
  __int64 v80; // r11
  int v81; // edi
  _QWORD *v82; // rsi
  __int64 v83; // rsi
  unsigned __int8 *v84; // rsi
  void *v85; // r14
  _QWORD *v86; // rbx
  void *v87; // rax
  _QWORD *v88; // r15
  _QWORD *v89; // rax
  void (__fastcall *v90)(_QWORD, _QWORD, _QWORD); // rax
  _QWORD *v91; // rdi
  _QWORD *v92; // rbx
  _QWORD *v93; // r14
  _QWORD *v94; // rdi
  __int64 v95; // r9
  void *v96; // rax
  _QWORD *v97; // r12
  _QWORD *v98; // rbx
  _QWORD *v99; // rax
  _QWORD *v100; // r14
  void (__fastcall *v101)(_QWORD, _QWORD, _QWORD); // rax
  _QWORD *v102; // rdi
  _QWORD *v103; // rbx
  _QWORD *v104; // r12
  _QWORD *v105; // rdi
  __int64 v106; // rcx
  unsigned __int64 v107; // r8
  __int64 v108; // r12
  __int64 v109; // rbx
  _QWORD *v110; // rdi
  _QWORD *v111; // rbx
  _QWORD *v112; // r12
  _QWORD *v113; // rdi
  _QWORD *v114; // rbx
  _QWORD *v115; // r12
  _QWORD *v116; // rdi
  unsigned __int64 v117; // r13
  unsigned __int64 v118; // rdi
  __int64 v119; // rax
  __int64 v120; // rbx
  __int64 v121; // r15
  _QWORD *v122; // r12
  unsigned __int64 v123; // rdi
  __int64 v124; // r14
  unsigned __int64 v125; // rdi
  unsigned __int64 v126; // rdi
  unsigned __int64 v127; // rdi
  __int64 v128; // rax
  __int64 v129; // rbx
  __int64 v130; // r15
  _QWORD *v131; // r12
  unsigned __int64 v132; // rdi
  __int64 v133; // r14
  unsigned __int64 v134; // rdi
  unsigned __int64 v135; // rdi
  unsigned __int64 v136; // r12
  int v137; // edx
  __int64 v138; // rbx
  unsigned __int64 v139; // rdi
  __int64 v140; // rbx
  __int64 v141; // r13
  _QWORD *v142; // r14
  unsigned __int64 v143; // rdi
  __int64 v144; // r15
  unsigned __int64 v145; // rdi
  unsigned __int64 v146; // rdi
  __int64 v148; // rax
  unsigned __int64 v149; // rdx
  int v150; // r11d
  _QWORD *v151; // rdi
  unsigned int v152; // r14d
  int v153; // ecx
  _QWORD *v154; // rsi
  __int64 v155; // [rsp-10h] [rbp-3E0h]
  __int64 v156; // [rsp-8h] [rbp-3D8h]
  __int64 v157; // [rsp+20h] [rbp-3B0h]
  __int64 v159; // [rsp+60h] [rbp-370h]
  unsigned __int64 v160; // [rsp+68h] [rbp-368h]
  __int64 *v161; // [rsp+68h] [rbp-368h]
  __int64 v162; // [rsp+70h] [rbp-360h]
  __int64 *v163; // [rsp+78h] [rbp-358h]
  __int64 *v164; // [rsp+80h] [rbp-350h]
  __int64 *v165; // [rsp+80h] [rbp-350h]
  unsigned __int64 v166; // [rsp+88h] [rbp-348h]
  __int64 *v167; // [rsp+90h] [rbp-340h]
  unsigned __int64 v168; // [rsp+90h] [rbp-340h]
  _BYTE *v169; // [rsp+98h] [rbp-338h]
  __int64 v170; // [rsp+A0h] [rbp-330h]
  __int64 v171; // [rsp+A0h] [rbp-330h]
  __int64 *v172; // [rsp+A8h] [rbp-328h]
  char v173; // [rsp+A8h] [rbp-328h]
  _BYTE *v174; // [rsp+A8h] [rbp-328h]
  __int64 v175; // [rsp+B0h] [rbp-320h] BYREF
  __int64 v176; // [rsp+B8h] [rbp-318h] BYREF
  unsigned __int8 *v177; // [rsp+C0h] [rbp-310h] BYREF
  unsigned __int8 *v178; // [rsp+C8h] [rbp-308h] BYREF
  __m128i v179; // [rsp+D0h] [rbp-300h] BYREF
  void (__fastcall *v180)(__m128i *, __m128i *, __int64, __int64, __int64, __int64); // [rsp+E0h] [rbp-2F0h]
  __int64 (__fastcall *v181)(__int64 *, __int64); // [rsp+E8h] [rbp-2E8h]
  __int64 *v182; // [rsp+F0h] [rbp-2E0h] BYREF
  __int64 v183; // [rsp+F8h] [rbp-2D8h]
  _BYTE v184[16]; // [rsp+100h] [rbp-2D0h] BYREF
  __int64 *v185; // [rsp+110h] [rbp-2C0h] BYREF
  __int64 v186; // [rsp+118h] [rbp-2B8h]
  _BYTE v187[16]; // [rsp+120h] [rbp-2B0h] BYREF
  __int64 v188[4]; // [rsp+130h] [rbp-2A0h] BYREF
  __int16 v189; // [rsp+150h] [rbp-280h]
  void *v190[2]; // [rsp+160h] [rbp-270h] BYREF
  void (__fastcall *v191)(_QWORD, _QWORD, _QWORD); // [rsp+170h] [rbp-260h] BYREF
  __int64 (__fastcall *v192)(__int64 *, __int64); // [rsp+178h] [rbp-258h]
  __int64 v193; // [rsp+190h] [rbp-240h]
  __int64 v194; // [rsp+198h] [rbp-238h]
  __int16 v195; // [rsp+1A0h] [rbp-230h]
  _QWORD *v196; // [rsp+1A8h] [rbp-228h]
  void **v197; // [rsp+1B0h] [rbp-220h]
  void **v198; // [rsp+1B8h] [rbp-218h]
  __int64 v199; // [rsp+1C0h] [rbp-210h]
  int v200; // [rsp+1C8h] [rbp-208h]
  __int16 v201; // [rsp+1CCh] [rbp-204h]
  char v202; // [rsp+1CEh] [rbp-202h]
  __int64 v203; // [rsp+1D0h] [rbp-200h]
  __int64 v204; // [rsp+1D8h] [rbp-1F8h]
  void *v205; // [rsp+1E0h] [rbp-1F0h] BYREF
  void *v206; // [rsp+1E8h] [rbp-1E8h] BYREF
  __int16 v207; // [rsp+1F0h] [rbp-1E0h] BYREF
  int v208; // [rsp+1F2h] [rbp-1DEh]
  __int16 v209; // [rsp+1F6h] [rbp-1DAh]
  __int64 v210[2]; // [rsp+1F8h] [rbp-1D8h] BYREF
  _QWORD v211[2]; // [rsp+208h] [rbp-1C8h] BYREF
  __int64 v212[2]; // [rsp+218h] [rbp-1B8h] BYREF
  _QWORD v213[4]; // [rsp+228h] [rbp-1A8h] BYREF
  int v214; // [rsp+248h] [rbp-188h]
  _BYTE *v215; // [rsp+250h] [rbp-180h]
  __int64 v216; // [rsp+258h] [rbp-178h]
  _BYTE v217[16]; // [rsp+260h] [rbp-170h] BYREF
  __int64 **v218; // [rsp+270h] [rbp-160h]
  __m128i v219; // [rsp+278h] [rbp-158h] BYREF
  __int64 (__fastcall *v220)(_QWORD *, _QWORD *, int); // [rsp+288h] [rbp-148h]
  __int64 (__fastcall *v221)(__int64 *, __int64); // [rsp+290h] [rbp-140h]
  __int64 *v222; // [rsp+298h] [rbp-138h]
  _BYTE *v223; // [rsp+2A0h] [rbp-130h]
  __int64 v224; // [rsp+2A8h] [rbp-128h]
  _BYTE v225[128]; // [rsp+2B0h] [rbp-120h] BYREF
  void *v226; // [rsp+330h] [rbp-A0h]
  _QWORD *v227; // [rsp+338h] [rbp-98h]
  void (__fastcall *v228)(_QWORD, _QWORD, _QWORD); // [rsp+340h] [rbp-90h]
  void *v229; // [rsp+348h] [rbp-88h]
  _QWORD *v230; // [rsp+350h] [rbp-80h]
  void (__fastcall *v231)(_QWORD, _QWORD, _QWORD); // [rsp+358h] [rbp-78h]
  __int64 v232; // [rsp+360h] [rbp-70h] BYREF
  __int64 v233; // [rsp+368h] [rbp-68h]
  __int64 v234; // [rsp+370h] [rbp-60h]
  __int64 v235; // [rsp+378h] [rbp-58h]
  unsigned __int64 v236; // [rsp+380h] [rbp-50h]
  __int64 v237; // [rsp+388h] [rbp-48h]
  __int64 v238; // [rsp+390h] [rbp-40h]

  v7 = *((_QWORD *)a2 + 2);
  v207 = *a2;
  v208 = *(_DWORD *)(a2 + 1);
  v8 = a2[3];
  v9 = (_BYTE *)*((_QWORD *)a2 + 1);
  v209 = v8;
  v210[0] = (__int64)v211;
  sub_2425700(v210, v9, (__int64)&v9[v7]);
  v10 = (_BYTE *)*((_QWORD *)a2 + 5);
  v11 = (__int64)&v10[*((_QWORD *)a2 + 6)];
  v212[0] = (__int64)v213;
  sub_2425700(v212, v10, v11);
  v215 = v217;
  v216 = 0x400000000LL;
  v223 = v225;
  v214 = 0;
  v218 = 0;
  v220 = 0;
  v222 = 0;
  v224 = 0x1000000000LL;
  v226 = 0;
  v227 = 0;
  v228 = 0;
  v229 = 0;
  v230 = 0;
  v231 = 0;
  v232 = 0;
  v233 = 0;
  v234 = 0;
  v235 = 0;
  v236 = 0;
  v237 = 0;
  v238 = 0x1000000000LL;
  v12 = sub_BC0510(a4, &unk_4F82418, (__int64)a3);
  v13 = _mm_loadu_si128((const __m128i *)v190);
  v14 = _mm_loadu_si128(&v219);
  v15 = *(_QWORD *)(v12 + 8);
  v218 = a3;
  v180 = 0;
  v179.m128i_i64[0] = v15;
  v16 = _mm_loadu_si128(&v179);
  v175 = v15;
  v176 = v15;
  v179 = v13;
  v181 = v192;
  v17 = (void (__fastcall *)(_QWORD, _QWORD, _QWORD))v220;
  v220 = sub_2425290;
  v191 = v17;
  v192 = v221;
  v221 = sub_2425160;
  *(__m128i *)v190 = v14;
  v219 = v16;
  if ( v17 )
    v17(v190, v190, 3);
  v18 = "llvm.dbg.cu";
  v19 = (__m128i *)a3;
  v222 = *a3;
  v157 = sub_BA8DC0((__int64)a3, (__int64)"llvm.dbg.cu", 11);
  if ( v157 && v207 )
  {
    v182 = (__int64 *)v184;
    v183 = 0x200000000LL;
    v186 = 0x200000000LL;
    v21 = (__int64 **)v218[4];
    v185 = (__int64 *)v187;
    if ( v218 + 3 == v21 )
    {
      v173 = (_DWORD)v183 != 0;
    }
    else
    {
      v22 = v21;
      v23 = 0;
      v24 = v218 + 3;
      do
      {
        v25 = (__int64)(v22 - 7);
        if ( !v22 )
          v25 = 0;
        if ( !v23 )
        {
          if ( !v220 )
            sub_4263D6(v19, v18, v20);
          v19 = &v219;
          v18 = (const char *)v25;
          v23 = (__int64 *)v221(v219.m128i_i64, v25);
        }
        v26 = v25 + 72;
        v27 = *(_QWORD *)(v25 + 80);
        if ( v26 == v27 )
        {
          i = 0;
        }
        else
        {
          while ( 1 )
          {
            if ( !v27 )
LABEL_262:
              BUG();
            i = *(_QWORD *)(v27 + 32);
            if ( i != v27 + 24 )
              break;
            v27 = *(_QWORD *)(v27 + 8);
            if ( v26 == v27 )
              goto LABEL_14;
          }
        }
        while ( v26 != v27 )
        {
          if ( !i )
            BUG();
          if ( *(_BYTE *)(i - 24) == 85 )
          {
            v18 = *(const char **)(i - 56);
            if ( v18 )
            {
              if ( !*v18 && *((_QWORD *)v18 + 3) == *(_QWORD *)(i + 56) )
              {
                v19 = (__m128i *)*v23;
                v170 = v26;
                v172 = v23;
                v50 = sub_981210(*v23, (__int64)v18, (unsigned int *)v190);
                v23 = v172;
                v26 = v170;
                if ( v50 )
                {
                  v51 = i - 24;
                  if ( LODWORD(v190[0]) == 281 )
                  {
                    v148 = (unsigned int)v183;
                    v149 = (unsigned int)v183 + 1LL;
                    if ( v149 > HIDWORD(v183) )
                    {
                      v18 = v184;
                      v19 = (__m128i *)&v182;
                      sub_C8D5F0((__int64)&v182, v184, v149, 8u, v170, v51);
                      v148 = (unsigned int)v183;
                      v26 = v170;
                      v23 = v172;
                      v51 = i - 24;
                    }
                    v20 = (unsigned __int64)v182;
                    v182[v148] = v51;
                    LODWORD(v183) = v183 + 1;
                  }
                  else if ( (unsigned int)(LODWORD(v190[0]) - 219) <= 7 )
                  {
                    v52 = (unsigned int)v186;
                    v53 = (unsigned int)v186 + 1LL;
                    if ( v53 > HIDWORD(v186) )
                    {
                      v18 = v187;
                      v19 = (__m128i *)&v185;
                      sub_C8D5F0((__int64)&v185, v187, v53, 8u, v170, v51);
                      v52 = (unsigned int)v186;
                      v26 = v170;
                      v23 = v172;
                      v51 = i - 24;
                    }
                    v20 = (unsigned __int64)v185;
                    v185[v52] = v51;
                    LODWORD(v186) = v186 + 1;
                  }
                }
              }
            }
          }
          for ( i = *(_QWORD *)(i + 8); ; i = *(_QWORD *)(v27 + 32) )
          {
            v49 = v27 - 24;
            if ( !v27 )
              v49 = 0;
            if ( i != v49 + 48 )
              break;
            v27 = *(_QWORD *)(v27 + 8);
            if ( v26 == v27 )
              goto LABEL_14;
            if ( !v27 )
              goto LABEL_262;
          }
        }
LABEL_14:
        v22 = (__int64 **)v22[1];
      }
      while ( v24 != v22 );
      v29 = v182;
      v164 = v23;
      v163 = &v182[(unsigned int)v183];
      if ( v182 != v163 )
      {
        while ( 1 )
        {
          v34 = (_QWORD *)*v29;
          v196 = (_QWORD *)sub_BD5C60(*v29);
          v195 = 0;
          v197 = &v205;
          v190[0] = &v191;
          v198 = &v206;
          v201 = 512;
          v205 = &unk_49DA100;
          v190[1] = (void *)0x200000000LL;
          v206 = &unk_49DA0B0;
          v199 = 0;
          v200 = 0;
          v202 = 7;
          v203 = 0;
          v204 = 0;
          v193 = 0;
          v194 = 0;
          sub_D5F1F0((__int64)v190, (__int64)v34);
          v35 = (_QWORD *)v34[5];
          v36 = (__int64 *)v34[4];
          v37 = (__int64 *)sub_BCB2D0(v196);
          v38 = sub_BCF480(v37, 0, 0, 0);
          v39 = (__int64)v218;
          v188[0] = 0;
          v40 = v38;
          if ( *(_BYTE *)(*v164 + 171) || (v41 = 0, *(_BYTE *)(*v164 + 169)) )
          {
            v159 = (__int64)v218;
            v160 = v38;
            v54 = sub_A7A090(v188, v222, 0, 54);
            v39 = v159;
            v40 = v160;
            v41 = v54;
          }
          v42 = sub_BA8C10(v39, (__int64)"__gcov_fork", 0xBu, v40, v41);
          v44 = *(v34 - 4) == 0;
          v34[10] = v42;
          if ( !v44 )
          {
            v45 = *(v34 - 3);
            *(_QWORD *)*(v34 - 2) = v45;
            if ( v45 )
              *(_QWORD *)(v45 + 16) = *(v34 - 2);
          }
          *(v34 - 4) = v43;
          if ( v43 )
          {
            v46 = *(_QWORD *)(v43 + 16);
            *(v34 - 3) = v46;
            if ( v46 )
              *(_QWORD *)(v46 + 16) = v34 - 3;
            *(v34 - 2) = v43 + 16;
            *(_QWORD *)(v43 + 16) = v34 - 4;
          }
          v189 = 257;
          sub_AA8550(v35, v36, 0, (__int64)v188, 0);
          v30 = v34[6];
          v178 = (unsigned __int8 *)v30;
          if ( !v30 )
            break;
          sub_B96E90((__int64)&v178, v30, 1);
          v30 = (__int64)v178;
          v31 = v35[6] & 0xFFFFFFFFFFFFFFF8LL;
          if ( v31 )
            goto LABEL_18;
          v32 = 0;
LABEL_19:
          v188[0] = v30;
          v33 = v32 + 48;
          if ( v30 )
          {
            sub_B96E90((__int64)v188, v30, 1);
            if ( (__int64 *)v33 == v188 )
            {
              if ( v188[0] )
                sub_B91220((__int64)v188, v188[0]);
              goto LABEL_23;
            }
            v47 = *(_QWORD *)(v32 + 48);
            if ( v47 )
LABEL_42:
              sub_B91220(v33, v47);
            v48 = (unsigned __int8 *)v188[0];
            *(_QWORD *)(v32 + 48) = v188[0];
            if ( v48 )
              sub_B976B0((__int64)v188, v48, v33);
            goto LABEL_23;
          }
LABEL_40:
          if ( (__int64 *)v33 != v188 )
          {
            v47 = *(_QWORD *)(v32 + 48);
            if ( v47 )
              goto LABEL_42;
            *(_QWORD *)(v32 + 48) = v188[0];
          }
LABEL_23:
          if ( v178 )
            sub_B91220((__int64)&v178, (__int64)v178);
          nullsub_61();
          v205 = &unk_49DA100;
          nullsub_63();
          if ( v190[0] != &v191 )
            _libc_free((unsigned __int64)v190[0]);
          if ( v163 == ++v29 )
          {
            v55 = v185;
            v56 = v186;
            v161 = &v185[(unsigned int)v186];
            goto LABEL_72;
          }
        }
        v31 = v35[6] & 0xFFFFFFFFFFFFFFF8LL;
        if ( !v31 )
        {
          v188[0] = 0;
          v33 = 48;
          v32 = 0;
          goto LABEL_40;
        }
LABEL_18:
        v32 = v31 - 24;
        goto LABEL_19;
      }
      v55 = v185;
      v56 = v186;
      v161 = &v185[(unsigned int)v186];
LABEL_72:
      if ( v161 != v55 )
      {
        v167 = v55;
        while ( 1 )
        {
          v65 = (_QWORD *)*v167;
          v196 = (_QWORD *)sub_BD5C60(*v167);
          v190[0] = &v191;
          v197 = &v205;
          v190[1] = (void *)0x200000000LL;
          v198 = &v206;
          v201 = 512;
          v195 = 0;
          v199 = 0;
          v200 = 0;
          v205 = &unk_49DA100;
          v202 = 7;
          v203 = 0;
          v206 = &unk_49DA0B0;
          v204 = 0;
          v193 = 0;
          v194 = 0;
          sub_D5F1F0((__int64)v190, (__int64)v65);
          v66 = (_QWORD *)v65[5];
          v165 = (__int64 *)v65[4];
          v67 = (__int64 *)sub_BCB120(v196);
          v68 = sub_BCF480(v67, 0, 0, 0);
          v69 = sub_BA8CA0((__int64)v218, (__int64)"llvm_writeout_files", 0x13u, v68);
          v189 = 257;
          sub_921880((unsigned int **)v190, v69, v70, 0, 0, (__int64)v188, 0);
          v71 = (__int64)v165;
          v177 = (unsigned __int8 *)v65[6];
          if ( v177 )
          {
            sub_B96E90((__int64)&v177, (__int64)v177, 1);
            v71 = (__int64)v165;
          }
          if ( v71 )
            v71 -= 24;
          sub_D5F1F0((__int64)v190, v71);
          v72 = sub_BA8CA0((__int64)v218, (__int64)"llvm_reset_counters", 0x13u, v68);
          v189 = 257;
          v74 = sub_921880((unsigned int **)v190, v72, v73, 0, 0, (__int64)v188, 0);
          v178 = v177;
          if ( !v177 )
            break;
          sub_B96E90((__int64)&v178, (__int64)v177, 1);
          v57 = v74 + 48;
          if ( (unsigned __int8 **)(v74 + 48) == &v178 )
          {
            if ( v178 )
              sub_B91220((__int64)&v178, (__int64)v178);
LABEL_77:
            v58 = v235;
            if ( !(_DWORD)v235 )
              goto LABEL_100;
            goto LABEL_78;
          }
          v75 = *(_QWORD *)(v74 + 48);
          if ( v75 )
            goto LABEL_97;
LABEL_98:
          v76 = v178;
          *(_QWORD *)(v74 + 48) = v178;
          if ( !v76 )
            goto LABEL_77;
          sub_B976B0((__int64)&v178, v76, v57);
          v58 = v235;
          if ( !(_DWORD)v235 )
          {
LABEL_100:
            ++v232;
            goto LABEL_101;
          }
LABEL_78:
          v59 = (v58 - 1) & (((unsigned int)v66 >> 9) ^ ((unsigned int)v66 >> 4));
          v60 = (_QWORD *)(v233 + 8LL * v59);
          v61 = (_QWORD *)*v60;
          if ( v66 == (_QWORD *)*v60 )
            goto LABEL_79;
          v150 = 1;
          v79 = 0;
          while ( v61 != (_QWORD *)-4096LL )
          {
            if ( v61 == (_QWORD *)-8192LL && !v79 )
              v79 = v60;
            v59 = (v58 - 1) & (v150 + v59);
            v60 = (_QWORD *)(v233 + 8LL * v59);
            v61 = (_QWORD *)*v60;
            if ( v66 == (_QWORD *)*v60 )
              goto LABEL_79;
            ++v150;
          }
          if ( !v79 )
            v79 = v60;
          ++v232;
          v78 = v234 + 1;
          if ( 4 * ((int)v234 + 1) < 3 * v58 )
          {
            if ( v58 - HIDWORD(v234) - v78 <= v58 >> 3 )
            {
              sub_E3B4A0((__int64)&v232, v58);
              if ( !(_DWORD)v235 )
              {
LABEL_263:
                LODWORD(v234) = v234 + 1;
                BUG();
              }
              v151 = 0;
              v152 = (v235 - 1) & (((unsigned int)v66 >> 9) ^ ((unsigned int)v66 >> 4));
              v78 = v234 + 1;
              v153 = 1;
              v79 = (_QWORD *)(v233 + 8LL * v152);
              v154 = (_QWORD *)*v79;
              if ( v66 != (_QWORD *)*v79 )
              {
                while ( v154 != (_QWORD *)-4096LL )
                {
                  if ( !v151 && v154 == (_QWORD *)-8192LL )
                    v151 = v79;
                  v152 = (v235 - 1) & (v153 + v152);
                  v79 = (_QWORD *)(v233 + 8LL * v152);
                  v154 = (_QWORD *)*v79;
                  if ( v66 == (_QWORD *)*v79 )
                    goto LABEL_235;
                  ++v153;
                }
                if ( v151 )
                  v79 = v151;
              }
            }
            goto LABEL_235;
          }
LABEL_101:
          sub_E3B4A0((__int64)&v232, 2 * v58);
          if ( !(_DWORD)v235 )
            goto LABEL_263;
          v77 = (v235 - 1) & (((unsigned int)v66 >> 9) ^ ((unsigned int)v66 >> 4));
          v78 = v234 + 1;
          v79 = (_QWORD *)(v233 + 8LL * v77);
          v80 = *v79;
          if ( v66 != (_QWORD *)*v79 )
          {
            v81 = 1;
            v82 = 0;
            while ( v80 != -4096 )
            {
              if ( v80 == -8192 && !v82 )
                v82 = v79;
              v77 = (v235 - 1) & (v81 + v77);
              v79 = (_QWORD *)(v233 + 8LL * v77);
              v80 = *v79;
              if ( v66 == (_QWORD *)*v79 )
                goto LABEL_235;
              ++v81;
            }
            if ( v82 )
              v79 = v82;
          }
LABEL_235:
          LODWORD(v234) = v78;
          if ( *v79 != -4096 )
            --HIDWORD(v234);
          *v79 = v66;
LABEL_79:
          v189 = 257;
          sub_AA8550(v66, v165, 0, (__int64)v188, 0);
          v62 = v66[6] & 0xFFFFFFFFFFFFFFF8LL;
          v188[0] = (__int64)v177;
          v63 = v62 - 24;
          if ( !v62 )
            v63 = 0;
          v64 = (__int64 *)(v63 + 48);
          if ( v177 )
          {
            sub_B96E90((__int64)v188, (__int64)v177, 1);
            if ( v64 != v188 )
            {
              v83 = *(_QWORD *)(v63 + 48);
              if ( v83 )
LABEL_110:
                sub_B91220(v63 + 48, v83);
              v84 = (unsigned __int8 *)v188[0];
              *(_QWORD *)(v63 + 48) = v188[0];
              if ( v84 )
                sub_B976B0((__int64)v188, v84, v63 + 48);
              goto LABEL_85;
            }
            if ( v188[0] )
              sub_B91220((__int64)v188, v188[0]);
          }
          else if ( v64 != v188 )
          {
            v83 = *(_QWORD *)(v63 + 48);
            if ( v83 )
              goto LABEL_110;
          }
LABEL_85:
          if ( v177 )
            sub_B91220((__int64)&v177, (__int64)v177);
          nullsub_61();
          v205 = &unk_49DA100;
          nullsub_63();
          if ( v190[0] != &v191 )
            _libc_free((unsigned __int64)v190[0]);
          if ( v161 == ++v167 )
          {
            v56 = v186;
            v55 = v185;
            goto LABEL_118;
          }
        }
        v57 = v74 + 48;
        if ( (unsigned __int8 **)(v74 + 48) == &v178 )
          goto LABEL_77;
        v75 = *(_QWORD *)(v74 + 48);
        if ( !v75 )
          goto LABEL_77;
LABEL_97:
        v162 = v57;
        sub_B91220(v57, v75);
        v57 = v162;
        goto LABEL_98;
      }
LABEL_118:
      v173 = ((unsigned int)v183 | v56) != 0;
      if ( v55 != (__int64 *)v187 )
        _libc_free((unsigned __int64)v55);
    }
    if ( v182 != (__int64 *)v184 )
      _libc_free((unsigned __int64)v182);
    sub_242E160((unsigned __int64 *)v190, (__int64)&v207, v210[0], v210[1]);
    v85 = v226;
    v86 = v227;
    v87 = v190[0];
    v88 = v226;
    v190[0] = 0;
    v226 = v87;
    v89 = v190[1];
    v190[1] = 0;
    v227 = v89;
    v90 = v191;
    v191 = 0;
    v228 = v90;
    while ( v86 != v88 )
    {
      v91 = v88;
      v88 += 2;
      sub_C88FF0(v91);
    }
    if ( v85 )
      j_j___libc_free_0((unsigned __int64)v85);
    v92 = v190[1];
    v93 = v190[0];
    if ( v190[1] != v190[0] )
    {
      do
      {
        v94 = v93;
        v93 += 2;
        sub_C88FF0(v94);
      }
      while ( v92 != v93 );
      v93 = v190[0];
    }
    if ( v93 )
      j_j___libc_free_0((unsigned __int64)v93);
    sub_242E160((unsigned __int64 *)v190, (__int64)&v207, v212[0], v212[1]);
    v96 = v190[0];
    v97 = v229;
    v190[0] = 0;
    v98 = v230;
    v229 = v96;
    v99 = v190[1];
    v100 = v97;
    v190[1] = 0;
    v230 = v99;
    v101 = v191;
    v191 = 0;
    v231 = v101;
    while ( v98 != v100 )
    {
      v102 = v100;
      v100 += 2;
      sub_C88FF0(v102);
    }
    if ( v97 )
      j_j___libc_free_0((unsigned __int64)v97);
    v103 = v190[1];
    v104 = v190[0];
    if ( v190[1] != v190[0] )
    {
      do
      {
        v105 = v104;
        v104 += 2;
        sub_C88FF0(v105);
      }
      while ( v103 != v104 );
      v104 = v190[0];
    }
    if ( v104 )
      j_j___libc_free_0((unsigned __int64)v104);
    sub_242ECA0((__int64)&v207, v157, v173, (__int64)sub_2425120, &v175, v95, (int)sub_2425140, &v176);
    if ( v180 )
      v180(&v179, &v179, 3, v106, v155, v156);
    memset(a1, 0, 0x60u);
    *((_DWORD *)a1 + 4) = 2;
    a1[1] = a1 + 4;
    *((_BYTE *)a1 + 28) = 1;
    a1[7] = a1 + 10;
    *((_DWORD *)a1 + 16) = 2;
    *((_BYTE *)a1 + 76) = 1;
  }
  else
  {
    if ( v180 )
      ((void (__fastcall *)(__m128i *, __m128i *, __int64))v180)(&v179, &v179, 3);
    a1[6] = 0;
    a1[1] = a1 + 4;
    a1[7] = a1 + 10;
    a1[2] = 0x100000002LL;
    a1[8] = 2;
    *((_DWORD *)a1 + 18) = 0;
    *((_BYTE *)a1 + 76) = 1;
    *((_DWORD *)a1 + 6) = 0;
    *((_BYTE *)a1 + 28) = 1;
    a1[4] = &qword_4F82400;
    *a1 = 1;
  }
  v107 = v236;
  if ( HIDWORD(v237) && (_DWORD)v237 )
  {
    v108 = 8LL * (unsigned int)v237;
    v109 = 0;
    do
    {
      v110 = *(_QWORD **)(v107 + v109);
      if ( v110 != (_QWORD *)-8LL && v110 )
      {
        sub_C7D6A0((__int64)v110, *v110 + 17LL, 8);
        v107 = v236;
      }
      v109 += 8;
    }
    while ( v108 != v109 );
  }
  _libc_free(v107);
  sub_C7D6A0(v233, 8LL * (unsigned int)v235, 8);
  v111 = v230;
  v112 = v229;
  if ( v230 != v229 )
  {
    do
    {
      v113 = v112;
      v112 += 2;
      sub_C88FF0(v113);
    }
    while ( v111 != v112 );
    v112 = v229;
  }
  if ( v112 )
    j_j___libc_free_0((unsigned __int64)v112);
  v114 = v227;
  v115 = v226;
  if ( v227 != v226 )
  {
    do
    {
      v116 = v115;
      v115 += 2;
      sub_C88FF0(v116);
    }
    while ( v114 != v115 );
    v115 = v226;
  }
  if ( v115 )
    j_j___libc_free_0((unsigned __int64)v115);
  v169 = v223;
  v174 = &v223[8 * (unsigned int)v224];
  if ( v223 != v174 )
  {
    do
    {
      v174 -= 8;
      v117 = *(_QWORD *)v174;
      if ( *(_QWORD *)v174 )
      {
        v118 = *(_QWORD *)(v117 + 296);
        if ( *(_DWORD *)(v117 + 308) )
        {
          v119 = *(unsigned int *)(v117 + 304);
          if ( (_DWORD)v119 )
          {
            v120 = 8 * v119;
            v121 = 0;
            do
            {
              v122 = *(_QWORD **)(v118 + v121);
              if ( v122 != (_QWORD *)-8LL && v122 )
              {
                v123 = v122[6];
                v124 = *v122 + 193LL;
                if ( (_QWORD *)v123 != v122 + 8 )
                  _libc_free(v123);
                v125 = v122[2];
                if ( (_QWORD *)v125 != v122 + 4 )
                  j_j___libc_free_0(v125);
                sub_C7D6A0((__int64)v122, v124, 8);
                v118 = *(_QWORD *)(v117 + 296);
              }
              v121 += 8;
            }
            while ( v120 != v121 );
          }
        }
        _libc_free(v118);
        v126 = *(_QWORD *)(v117 + 216);
        if ( v126 != v117 + 232 )
          _libc_free(v126);
        v127 = *(_QWORD *)(v117 + 176);
        v168 = v117 + 80;
        if ( *(_DWORD *)(v117 + 188) )
        {
          v128 = *(unsigned int *)(v117 + 184);
          if ( (_DWORD)v128 )
          {
            v129 = 8 * v128;
            v130 = 0;
            do
            {
              v131 = *(_QWORD **)(v127 + v130);
              if ( v131 != (_QWORD *)-8LL && v131 )
              {
                v132 = v131[6];
                v133 = *v131 + 193LL;
                if ( (_QWORD *)v132 != v131 + 8 )
                  _libc_free(v132);
                v134 = v131[2];
                if ( (_QWORD *)v134 != v131 + 4 )
                  j_j___libc_free_0(v134);
                sub_C7D6A0((__int64)v131, v133, 8);
                v127 = *(_QWORD *)(v117 + 176);
              }
              v130 += 8;
            }
            while ( v129 != v130 );
          }
        }
        _libc_free(v127);
        v135 = *(_QWORD *)(v117 + 96);
        if ( v135 != v117 + 112 )
          _libc_free(v135);
        v171 = *(_QWORD *)(v117 + 64);
        v136 = v171 + ((unsigned __int64)*(unsigned int *)(v117 + 72) << 7);
        if ( v171 != v136 )
        {
          v166 = v117;
          do
          {
            v137 = *(_DWORD *)(v136 - 12);
            v136 -= 128LL;
            if ( v137 && (v138 = *(unsigned int *)(v136 + 112), (_DWORD)v138) )
            {
              v139 = *(_QWORD *)(v136 + 104);
              v140 = 8 * v138;
              v141 = 0;
              do
              {
                v142 = *(_QWORD **)(v139 + v141);
                if ( v142 != (_QWORD *)-8LL && v142 )
                {
                  v143 = v142[6];
                  v144 = *v142 + 193LL;
                  if ( (_QWORD *)v143 != v142 + 8 )
                    _libc_free(v143);
                  v145 = v142[2];
                  if ( (_QWORD *)v145 != v142 + 4 )
                    j_j___libc_free_0(v145);
                  sub_C7D6A0((__int64)v142, v144, 8);
                  v139 = *(_QWORD *)(v136 + 104);
                }
                v141 += 8;
              }
              while ( v140 != v141 );
            }
            else
            {
              v139 = *(_QWORD *)(v136 + 104);
            }
            _libc_free(v139);
            v146 = *(_QWORD *)(v136 + 24);
            if ( v146 != v136 + 40 )
              _libc_free(v146);
          }
          while ( v171 != v136 );
          v117 = v166;
          v136 = *(_QWORD *)(v166 + 64);
        }
        if ( v136 != v168 )
          _libc_free(v136);
        sub_C7D6A0(*(_QWORD *)(v117 + 40), 16LL * *(unsigned int *)(v117 + 56), 8);
        j_j___libc_free_0(v117);
      }
    }
    while ( v169 != v174 );
    v174 = v223;
  }
  if ( v174 != v225 )
    _libc_free((unsigned __int64)v174);
  if ( v220 )
    v220(&v219, &v219, 3);
  if ( v215 != v217 )
    _libc_free((unsigned __int64)v215);
  if ( (_QWORD *)v212[0] != v213 )
    j_j___libc_free_0(v212[0]);
  if ( (_QWORD *)v210[0] != v211 )
    j_j___libc_free_0(v210[0]);
  return a1;
}
