// Function: sub_167DAB0
// Address: 0x167dab0
//
__int64 __fastcall sub_167DAB0(
        __int64 *a1,
        __int64 *a2,
        int a3,
        __m128i *a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        double a11,
        __m128 a12)
{
  __m128 v13; // xmm0
  void (__fastcall *v14)(__m128 *, __m128 *, __int64); // r8
  __m128i v15; // xmm1
  void (__fastcall *v16)(__m128 *, __int64, unsigned __int64 *); // rdx
  __int64 v17; // rax
  __int64 v18; // rbx
  int v19; // ecx
  _QWORD *v20; // rsi
  __int64 *v21; // rax
  __int64 v22; // rdx
  __int64 *v23; // r15
  int *v24; // rax
  __int64 *v25; // r14
  _QWORD *v26; // r12
  int *v27; // rsi
  __int64 v28; // rcx
  __int64 v29; // rdx
  unsigned int v30; // ebx
  __int64 v31; // r13
  __int64 v32; // rax
  __int64 v33; // rdx
  int v34; // eax
  __int64 v35; // rdx
  __int64 v36; // rax
  unsigned int v37; // r13d
  int *v38; // rax
  int *v39; // r13
  __int64 v40; // rcx
  __int64 v41; // rdx
  __int64 v42; // rax
  int *v43; // rax
  __int64 v44; // rdx
  _BOOL8 v45; // rdi
  __int64 v46; // rax
  __int64 v47; // rdx
  int v48; // eax
  __int64 v49; // rdx
  _QWORD *v50; // rax
  __int64 v51; // rbx
  unsigned int v52; // edx
  __int64 *v53; // rcx
  __int64 v54; // rdi
  __int64 v55; // rax
  __int64 *v56; // rdx
  __int64 v57; // rax
  __int64 v58; // rbx
  __int64 v59; // rdi
  __int64 v60; // rbx
  __int64 v61; // rdi
  __int64 v62; // rbx
  __int64 v63; // rdi
  __int64 *v64; // rax
  __int64 *v65; // rbx
  __int64 *v66; // r13
  __int64 v67; // r12
  unsigned int v68; // ecx
  _QWORD *v69; // rax
  __int64 v70; // rdx
  _QWORD *v71; // rdx
  _BYTE *v72; // rsi
  __int64 *v73; // rbx
  __int64 *v74; // r13
  __int64 v75; // r12
  unsigned int v76; // ecx
  _QWORD *v77; // rax
  __int64 v78; // rdx
  _BYTE *v79; // rsi
  _BYTE *v80; // rcx
  __int64 *v81; // rbx
  __int64 *v82; // r13
  __int64 v83; // rax
  __int64 v84; // r12
  unsigned int v85; // r15d
  unsigned int v86; // ecx
  __int64 *v87; // rax
  __int64 v88; // rdx
  _QWORD *v89; // rdx
  _BYTE *v90; // rsi
  __int64 *v91; // r12
  __int64 *v92; // r13
  char *v93; // rsi
  unsigned int v94; // r12d
  _QWORD *v95; // rbx
  _QWORD *v96; // r13
  __int64 v97; // rdi
  unsigned __int64 v98; // r8
  __int64 v99; // r13
  __int64 v100; // rbx
  unsigned __int64 v101; // rdi
  __int64 *v102; // r13
  const char *v104; // rax
  __int64 v105; // r13
  __int64 v106; // rbx
  __int64 v107; // rbx
  unsigned int v108; // eax
  __int64 *v109; // r12
  __int64 *v110; // r13
  char *v111; // rsi
  int v112; // r10d
  _QWORD *v113; // r9
  int v114; // edx
  int v115; // r10d
  _QWORD *v116; // r9
  int v117; // edx
  __int64 v118; // r12
  __int64 *i; // r13
  char **v120; // rax
  __int64 v121; // rdx
  _QWORD *v122; // rax
  __int64 *v123; // r13
  __int64 *v124; // r14
  __int64 v125; // rdi
  __int64 v126; // r12
  const char *v127; // rax
  __int64 v128; // rdx
  __int64 v129; // rax
  unsigned int v130; // eax
  char *v131; // rsi
  int v132; // r10d
  __int64 *v133; // r9
  int v134; // edx
  int v135; // r9d
  __int64 *v136; // rax
  int v137; // edx
  _QWORD *v138; // r9
  __int64 v139; // r15
  int v140; // ecx
  __int64 v141; // rsi
  unsigned int v142; // ecx
  __int64 v143; // r9
  int v144; // edi
  _QWORD *v145; // rsi
  _QWORD *v146; // r9
  __int64 v147; // r15
  int v148; // ecx
  __int64 v149; // rsi
  unsigned int v150; // ecx
  __int64 v151; // r9
  int v152; // edi
  _QWORD *v153; // rsi
  __int64 v154; // rbx
  __int64 v155; // rsi
  __int64 v156; // rax
  __int64 v157; // rcx
  unsigned __int64 v158; // rsi
  unsigned __int64 v159; // rdi
  __int64 v160; // rdi
  __int64 v161; // rax
  unsigned int v162; // ecx
  __int64 v163; // r8
  int v164; // edi
  __int64 *v165; // rsi
  __int64 *v166; // r12
  __int64 *v167; // r14
  __int64 v168; // rdi
  const char *v169; // rax
  size_t v170; // rdx
  __int64 *v171; // r14
  __int64 v172; // rdx
  __int64 *v173; // r15
  unsigned __int64 v174; // rdi
  __int64 **v175; // r13
  __int64 v176; // rdx
  int v177; // esi
  unsigned int v178; // r12d
  __int64 *v179; // rcx
  __int64 v180; // rdi
  __int64 v181; // rcx
  __int64 v182; // r8
  int v183; // edi
  __int64 *v184; // rsi
  __int64 *v185; // r9
  __int64 v186; // r15
  int v187; // ecx
  __int64 v188; // rsi
  __int64 v189; // [rsp-10h] [rbp-230h]
  __int64 v190; // [rsp+8h] [rbp-218h]
  unsigned int v191; // [rsp+10h] [rbp-210h]
  unsigned int v192; // [rsp+14h] [rbp-20Ch]
  __int64 v193; // [rsp+18h] [rbp-208h]
  __int64 v194; // [rsp+18h] [rbp-208h]
  __int64 v195; // [rsp+18h] [rbp-208h]
  __int64 v196; // [rsp+20h] [rbp-200h]
  __int64 v197; // [rsp+28h] [rbp-1F8h]
  char v198; // [rsp+30h] [rbp-1F0h]
  __int64 v199; // [rsp+38h] [rbp-1E8h]
  __int64 v200; // [rsp+38h] [rbp-1E8h]
  __int64 v201; // [rsp+38h] [rbp-1E8h]
  __int64 *v202; // [rsp+40h] [rbp-1E0h]
  int *v203; // [rsp+40h] [rbp-1E0h]
  __int64 v204; // [rsp+40h] [rbp-1E0h]
  int *v205; // [rsp+40h] [rbp-1E0h]
  __int64 *v206; // [rsp+48h] [rbp-1D8h]
  unsigned int v207; // [rsp+48h] [rbp-1D8h]
  __int64 **v208; // [rsp+48h] [rbp-1D8h]
  unsigned __int8 v209; // [rsp+57h] [rbp-1C9h] BYREF
  __int64 v210; // [rsp+58h] [rbp-1C8h] BYREF
  __int64 v211; // [rsp+60h] [rbp-1C0h] BYREF
  __int64 v212; // [rsp+68h] [rbp-1B8h] BYREF
  __int64 v213; // [rsp+70h] [rbp-1B0h] BYREF
  __int64 v214; // [rsp+78h] [rbp-1A8h] BYREF
  __int64 *v215; // [rsp+80h] [rbp-1A0h] BYREF
  __int64 v216; // [rsp+88h] [rbp-198h]
  unsigned __int64 v217; // [rsp+90h] [rbp-190h] BYREF
  __int64 **v218; // [rsp+98h] [rbp-188h]
  __int16 v219; // [rsp+A0h] [rbp-180h]
  __int64 *v220[2]; // [rsp+B0h] [rbp-170h] BYREF
  __int16 v221; // [rsp+C0h] [rbp-160h]
  __int64 v222; // [rsp+D0h] [rbp-150h] BYREF
  __int64 v223; // [rsp+D8h] [rbp-148h]
  __int64 v224; // [rsp+E0h] [rbp-140h]
  __int64 v225; // [rsp+E8h] [rbp-138h]
  __m128i v226; // [rsp+F0h] [rbp-130h] BYREF
  __int64 (__fastcall *v227)(__m128i *, __m128i *, int); // [rsp+100h] [rbp-120h]
  __int64 (__fastcall *v228)(__m128i *, __int64, __int64); // [rsp+108h] [rbp-118h]
  __int64 *v229; // [rsp+110h] [rbp-110h] BYREF
  __int64 *v230; // [rsp+118h] [rbp-108h]
  __int64 v231; // [rsp+120h] [rbp-100h] BYREF
  __int64 v232; // [rsp+128h] [rbp-F8h]
  __int64 v233; // [rsp+130h] [rbp-F0h]
  __int64 v234; // [rsp+138h] [rbp-E8h]
  char **v235; // [rsp+140h] [rbp-E0h]
  char **v236; // [rsp+148h] [rbp-D8h]
  __int64 v237; // [rsp+150h] [rbp-D0h]
  int v238; // [rsp+158h] [rbp-C8h]
  unsigned __int64 v239; // [rsp+160h] [rbp-C0h] BYREF
  __int64 v240; // [rsp+168h] [rbp-B8h]
  __int64 v241; // [rsp+170h] [rbp-B0h]
  __m128 v242; // [rsp+180h] [rbp-A0h] BYREF
  void (__fastcall *v243)(__m128 *, __m128 *, __int64); // [rsp+190h] [rbp-90h]
  void (__fastcall *v244)(__m128 *, __int64, unsigned __int64 *); // [rsp+198h] [rbp-88h]
  __int64 v245; // [rsp+1A0h] [rbp-80h] BYREF
  int v246; // [rsp+1A8h] [rbp-78h] BYREF
  int *v247; // [rsp+1B0h] [rbp-70h]
  int *v248; // [rsp+1B8h] [rbp-68h]
  int *v249; // [rsp+1C0h] [rbp-60h]
  __int64 v250; // [rsp+1C8h] [rbp-58h]
  __int64 v251; // [rsp+1D0h] [rbp-50h] BYREF
  _QWORD *v252; // [rsp+1D8h] [rbp-48h]
  __int64 v253; // [rsp+1E0h] [rbp-40h]
  unsigned int v254; // [rsp+1E8h] [rbp-38h]

  v13 = (__m128)_mm_loadu_si128(a4);
  v14 = (void (__fastcall *)(__m128 *, __m128 *, __int64))a4[1].m128i_i64[0];
  a4[1].m128i_i64[0] = 0;
  v15 = _mm_loadu_si128(&v226);
  v16 = (void (__fastcall *)(__m128 *, __int64, unsigned __int64 *))a4[1].m128i_i64[1];
  a4[1].m128i_i64[1] = 0;
  v241 = 0x1000000000LL;
  *a4 = v15;
  v17 = *a2;
  v229 = a1;
  *a2 = 0;
  v230 = (__int64 *)v17;
  v231 = 0;
  v232 = 0;
  v233 = 0;
  v234 = 0;
  v235 = 0;
  v236 = 0;
  v237 = 0;
  v238 = a3;
  v239 = 0;
  v240 = 0;
  v243 = v14;
  v226 = (__m128i)v13;
  v242 = v13;
  v244 = v16;
  v18 = *a1;
  v246 = 0;
  v247 = 0;
  v248 = &v246;
  v249 = &v246;
  v250 = 0;
  v251 = 0;
  v252 = 0;
  v253 = 0;
  v254 = 0;
  v19 = *(_DWORD *)(v17 + 136);
  v197 = v18;
  v222 = 0;
  v223 = 0;
  v224 = 0;
  v225 = 0;
  if ( v19 )
  {
    v20 = *(_QWORD **)(v17 + 128);
    if ( *v20 && *v20 != -8 )
    {
      v23 = *(__int64 **)(v17 + 128);
    }
    else
    {
      v21 = v20 + 1;
      do
      {
        do
        {
          v22 = *v21;
          v23 = v21++;
        }
        while ( v22 == -8 );
      }
      while ( !v22 );
    }
    v206 = &v20[v19];
    if ( v206 != v23 )
    {
      v24 = 0;
      v196 = v18 + 128;
      v25 = v23;
      while ( 1 )
      {
        v26 = (_QWORD *)(*v25 + 8);
        if ( v24 )
        {
          v27 = &v246;
          do
          {
            while ( 1 )
            {
              v28 = *((_QWORD *)v24 + 2);
              v29 = *((_QWORD *)v24 + 3);
              if ( *((_QWORD *)v24 + 4) >= (unsigned __int64)v26 )
                break;
              v24 = (int *)*((_QWORD *)v24 + 3);
              if ( !v29 )
                goto LABEL_14;
            }
            v27 = v24;
            v24 = (int *)*((_QWORD *)v24 + 2);
          }
          while ( v28 );
LABEL_14:
          if ( v27 != &v246 && *((_QWORD *)v27 + 4) <= (unsigned __int64)v26 )
            goto LABEL_39;
        }
        v30 = *(_DWORD *)(*v25 + 16);
        v31 = *v229;
        v32 = sub_1580C70((_QWORD *)(*v25 + 8));
        v199 = v33;
        v202 = (__int64 *)v32;
        v34 = sub_16D1B30(v31 + 128, v32, v33);
        if ( v34 == -1
          || (v35 = *(_QWORD *)(v31 + 128), v36 = v35 + 8LL * v34, v36 == v35 + 8LL * *(unsigned int *)(v31 + 136)) )
        {
          v198 = 1;
          goto LABEL_22;
        }
        v37 = *(_DWORD *)(*(_QWORD *)v36 + 16LL);
        v215 = v202;
        v216 = v199;
        if ( ((v37 | v30) & 0xFFFFFFFD) != 0 )
        {
          if ( v30 != v37 )
          {
            v217 = (unsigned __int64)"Linking COMDATs named '";
            v219 = 1283;
            v218 = &v215;
            v220[0] = (__int64 *)&v217;
            v104 = "': invalid selection kinds!";
            goto LABEL_121;
          }
          if ( v37 == 3 )
          {
            v217 = (unsigned __int64)"Linking COMDATs named '";
            v218 = &v215;
            v220[0] = (__int64 *)&v217;
            v104 = "': noduplicates has been violated!";
            v219 = 1283;
            goto LABEL_121;
          }
          if ( v37 > 3 )
          {
            if ( v30 != 4 )
              goto LABEL_22;
          }
          else if ( !v37 )
          {
            goto LABEL_21;
          }
        }
        else
        {
          if ( v37 != 2 && v30 != 2 )
          {
LABEL_21:
            v198 = 0;
            v30 = 0;
            goto LABEL_22;
          }
          v37 = 2;
        }
        v106 = *v229;
        if ( (unsigned __int8)sub_167BC20((__int64)&v229, *v229, (__int64)v215, v216, &v213) )
          goto LABEL_97;
        v198 = sub_167BC20((__int64)&v229, (__int64)v230, (__int64)v215, v216, &v214);
        if ( v198 )
          goto LABEL_97;
        v200 = sub_1632FA0(v106);
        v204 = sub_1632FA0((__int64)v230);
        v193 = v200;
        v107 = *(_QWORD *)(v213 + 24);
        v108 = sub_15A9FE0(v200, v107);
        v201 = 1;
        v191 = v108;
        while ( 2 )
        {
          switch ( *(_BYTE *)(v107 + 8) )
          {
            case 0:
            case 8:
            case 0xA:
            case 0xC:
            case 0x10:
              v161 = v201 * *(_QWORD *)(v107 + 32);
              v107 = *(_QWORD *)(v107 + 24);
              v201 = v161;
              continue;
            case 1:
              v154 = 16;
              break;
            case 2:
              v154 = 32;
              break;
            case 3:
            case 9:
              v154 = 64;
              break;
            case 4:
              v154 = 80;
              break;
            case 5:
            case 6:
              v154 = 128;
              break;
            case 7:
              v154 = 8 * (unsigned int)sub_15A9520(v193, 0);
              break;
            case 0xB:
              v154 = *(_DWORD *)(v107 + 8) >> 8;
              break;
            case 0xD:
              v154 = 8LL * *(_QWORD *)sub_15A9930(v193, v107);
              break;
            case 0xE:
              v160 = v193;
              v195 = *(_QWORD *)(v107 + 32);
              v154 = 8 * sub_12BE0A0(v160, *(_QWORD *)(v107 + 24)) * v195;
              break;
            case 0xF:
              v154 = 8 * (unsigned int)sub_15A9520(v193, *(_DWORD *)(v107 + 8) >> 8);
              break;
          }
          break;
        }
        v155 = *(_QWORD *)(v214 + 24);
        v194 = 1;
        v192 = sub_15A9FE0(v204, v155);
        while ( 2 )
        {
          switch ( *(_BYTE *)(v155 + 8) )
          {
            case 1:
              v157 = 16;
              goto LABEL_242;
            case 2:
              v157 = 32;
              goto LABEL_242;
            case 3:
            case 9:
              v157 = 64;
              goto LABEL_242;
            case 4:
              v157 = 80;
              goto LABEL_242;
            case 5:
            case 6:
              v157 = 128;
              goto LABEL_242;
            case 7:
              v157 = 8 * (unsigned int)sub_15A9520(v204, 0);
              goto LABEL_242;
            case 0xB:
              v157 = *(_DWORD *)(v155 + 8) >> 8;
              goto LABEL_242;
            case 0xD:
              v157 = 8LL * *(_QWORD *)sub_15A9930(v204, v155);
              goto LABEL_242;
            case 0xE:
              v190 = *(_QWORD *)(v155 + 32);
              v157 = 8 * sub_12BE0A0(v204, *(_QWORD *)(v155 + 24)) * v190;
              goto LABEL_242;
            case 0xF:
              v157 = 8 * (unsigned int)sub_15A9520(v204, *(_DWORD *)(v155 + 8) >> 8);
LABEL_242:
              if ( v37 == 1 )
              {
                if ( *(_QWORD *)(v214 - 24) != *(_QWORD *)(v213 - 24) )
                {
                  v217 = (unsigned __int64)"Linking COMDATs named '";
                  v218 = &v215;
                  v220[0] = (__int64 *)&v217;
                  v104 = "': ExactMatch violated!";
                  v219 = 1283;
LABEL_121:
                  v220[1] = (__int64 *)v104;
                  v221 = 770;
                  v105 = *v230;
                  sub_1670450((__int64)&v226, 0, (__int64)v220);
                  sub_16027F0(v105, (__int64)&v226);
LABEL_97:
                  v94 = 1;
                  goto LABEL_98;
                }
              }
              else
              {
                v158 = (v191 + ((unsigned __int64)(v154 * v201 + 7) >> 3) - 1) / v191 * v191;
                v159 = (v192 + ((unsigned __int64)(v157 * v194 + 7) >> 3) - 1) / v192 * v192;
                if ( v37 == 2 )
                {
                  v198 = v159 > v158;
                }
                else if ( v159 != v158 )
                {
                  v217 = (unsigned __int64)"Linking COMDATs named '";
                  v218 = &v215;
                  v220[0] = (__int64 *)&v217;
                  v104 = "': SameSize violated!";
                  v219 = 1283;
                  goto LABEL_121;
                }
              }
              v30 = v37;
              break;
            case 0x10:
              v156 = v194 * *(_QWORD *)(v155 + 32);
              v155 = *(_QWORD *)(v155 + 24);
              v194 = v156;
              continue;
            default:
              goto LABEL_400;
          }
          break;
        }
LABEL_22:
        v38 = v247;
        v39 = &v246;
        if ( !v247 )
          goto LABEL_29;
        do
        {
          while ( 1 )
          {
            v40 = *((_QWORD *)v38 + 2);
            v41 = *((_QWORD *)v38 + 3);
            if ( *((_QWORD *)v38 + 4) >= (unsigned __int64)v26 )
              break;
            v38 = (int *)*((_QWORD *)v38 + 3);
            if ( !v41 )
              goto LABEL_27;
          }
          v39 = v38;
          v38 = (int *)*((_QWORD *)v38 + 2);
        }
        while ( v40 );
LABEL_27:
        if ( v39 == &v246 || *((_QWORD *)v39 + 4) > (unsigned __int64)v26 )
        {
LABEL_29:
          v203 = v39;
          v42 = sub_22077B0(48);
          *(_QWORD *)(v42 + 32) = v26;
          v39 = (int *)v42;
          *(_DWORD *)(v42 + 40) = 0;
          *(_BYTE *)(v42 + 44) = 0;
          v43 = (int *)sub_167C8F0(&v245, v203, (unsigned __int64 *)(v42 + 32));
          if ( v44 )
          {
            v45 = v43 || &v246 == (int *)v44 || (unsigned __int64)v26 < *(_QWORD *)(v44 + 32);
            sub_220F040(v45, v39, v44, &v246);
            ++v250;
          }
          else
          {
            v205 = v43;
            j_j___libc_free_0(v39, 48);
            v39 = v205;
          }
        }
        v39[10] = v30;
        *((_BYTE *)v39 + 44) = v198;
        if ( v198 )
        {
          v46 = sub_1580C70(v26);
          v48 = sub_16D1B30(v196, v46, v47);
          if ( v48 != -1 )
          {
            v49 = *(_QWORD *)(v197 + 128);
            v50 = (_QWORD *)(v49 + 8LL * v48);
            if ( v50 != (_QWORD *)(v49 + 8LL * *(unsigned int *)(v197 + 136)) )
            {
              v51 = *v50 + 8LL;
              if ( (_DWORD)v225 )
              {
                v52 = (v225 - 1) & (((unsigned int)v51 >> 9) ^ ((unsigned int)v51 >> 4));
                v53 = (__int64 *)(v223 + 8LL * v52);
                v54 = *v53;
                if ( v51 == *v53 )
                  goto LABEL_39;
                v135 = 1;
                v136 = 0;
                while ( v54 != -8 )
                {
                  if ( !v136 && v54 == -16 )
                    v136 = v53;
                  v52 = (v225 - 1) & (v135 + v52);
                  v53 = (__int64 *)(v223 + 8LL * v52);
                  v54 = *v53;
                  if ( v51 == *v53 )
                    goto LABEL_39;
                  ++v135;
                }
                if ( !v136 )
                  v136 = v53;
                ++v222;
                v137 = v224 + 1;
                if ( 4 * ((int)v224 + 1) < (unsigned int)(3 * v225) )
                {
                  if ( (int)v225 - HIDWORD(v224) - v137 <= (unsigned int)v225 >> 3 )
                  {
                    sub_15564C0((__int64)&v222, v225);
                    if ( !(_DWORD)v225 )
                    {
LABEL_401:
                      LODWORD(v224) = v224 + 1;
                      BUG();
                    }
                    v177 = 1;
                    v178 = (v225 - 1) & (((unsigned int)v51 >> 9) ^ ((unsigned int)v51 >> 4));
                    v137 = v224 + 1;
                    v179 = 0;
                    v136 = (__int64 *)(v223 + 8LL * v178);
                    v180 = *v136;
                    if ( v51 != *v136 )
                    {
                      while ( v180 != -8 )
                      {
                        if ( v180 == -16 && !v179 )
                          v179 = v136;
                        v178 = (v225 - 1) & (v177 + v178);
                        v136 = (__int64 *)(v223 + 8LL * v178);
                        v180 = *v136;
                        if ( v51 == *v136 )
                          goto LABEL_205;
                        ++v177;
                      }
                      if ( v179 )
                        v136 = v179;
                    }
                  }
                  goto LABEL_205;
                }
              }
              else
              {
                ++v222;
              }
              sub_15564C0((__int64)&v222, 2 * v225);
              if ( !(_DWORD)v225 )
                goto LABEL_401;
              v162 = (v225 - 1) & (((unsigned int)v51 >> 9) ^ ((unsigned int)v51 >> 4));
              v137 = v224 + 1;
              v136 = (__int64 *)(v223 + 8LL * v162);
              v163 = *v136;
              if ( v51 != *v136 )
              {
                v164 = 1;
                v165 = 0;
                while ( v163 != -8 )
                {
                  if ( !v165 && v163 == -16 )
                    v165 = v136;
                  v162 = (v225 - 1) & (v164 + v162);
                  v136 = (__int64 *)(v223 + 8LL * v162);
                  v163 = *v136;
                  if ( v51 == *v136 )
                    goto LABEL_205;
                  ++v164;
                }
                if ( v165 )
                  v136 = v165;
              }
LABEL_205:
              LODWORD(v224) = v137;
              if ( *v136 != -8 )
                --HIDWORD(v224);
              *v136 = v51;
            }
          }
        }
LABEL_39:
        v55 = v25[1];
        if ( v55 != -8 && v55 )
        {
          ++v25;
        }
        else
        {
          v56 = v25 + 2;
          do
          {
            do
            {
              v57 = *v56;
              v25 = v56++;
            }
            while ( v57 == -8 );
          }
          while ( !v57 );
        }
        if ( v206 == v25 )
          break;
        v24 = v247;
      }
    }
  }
  v58 = *(_QWORD *)(v197 + 48);
  while ( v197 + 40 != v58 )
  {
    v59 = v58;
    v58 = *(_QWORD *)(v58 + 8);
    sub_167BD50(v59 - 48, (__int64)&v222, v13, *(double *)v15.m128i_i64, a7, a8, a9, a10, a11, a12);
  }
  v60 = *(_QWORD *)(v197 + 16);
  while ( v197 + 8 != v60 )
  {
    v61 = v60;
    v60 = *(_QWORD *)(v60 + 8);
    sub_167BD50(v61 - 56, (__int64)&v222, v13, *(double *)v15.m128i_i64, a7, a8, a9, a10, a11, a12);
  }
  v62 = *(_QWORD *)(v197 + 32);
  while ( v197 + 24 != v62 )
  {
    v63 = v62;
    v62 = *(_QWORD *)(v62 + 8);
    sub_167BD50(v63 - 56, (__int64)&v222, v13, *(double *)v15.m128i_i64, a7, a8, a9, a10, a11, a12);
  }
  v64 = v230;
  v65 = (__int64 *)v230[2];
  v66 = v230 + 1;
  if ( v230 + 1 != v65 )
  {
    while ( 1 )
    {
      if ( !v65 )
        BUG();
      if ( (*(_BYTE *)(v65 - 3) & 0xFu) - 2 > 1 )
        goto LABEL_54;
      v67 = *(v65 - 1);
      if ( !v67 )
        goto LABEL_54;
      if ( !v254 )
        break;
      v68 = (v254 - 1) & (((unsigned int)v67 >> 9) ^ ((unsigned int)v67 >> 4));
      v69 = &v252[4 * v68];
      v70 = *v69;
      if ( v67 != *v69 )
      {
        v112 = 1;
        v113 = 0;
        while ( v70 != -8 )
        {
          if ( v70 == -16 && !v113 )
            v113 = v69;
          v68 = (v254 - 1) & (v112 + v68);
          v69 = &v252[4 * v68];
          v70 = *v69;
          if ( v67 == *v69 )
            goto LABEL_60;
          ++v112;
        }
        if ( v113 )
          v69 = v113;
        ++v251;
        v114 = v253 + 1;
        if ( 4 * ((int)v253 + 1) < 3 * v254 )
        {
          if ( v254 - HIDWORD(v253) - v114 <= v254 >> 3 )
          {
            sub_167C9F0((__int64)&v251, v254);
            if ( !v254 )
              goto LABEL_399;
            v146 = 0;
            LODWORD(v147) = (v254 - 1) & (((unsigned int)v67 >> 9) ^ ((unsigned int)v67 >> 4));
            v114 = v253 + 1;
            v148 = 1;
            v69 = &v252[4 * (unsigned int)v147];
            v149 = *v69;
            if ( v67 != *v69 )
            {
              while ( v149 != -8 )
              {
                if ( v149 == -16 && !v146 )
                  v146 = v69;
                v147 = (v254 - 1) & ((_DWORD)v147 + v148);
                v69 = &v252[4 * v147];
                v149 = *v69;
                if ( v67 == *v69 )
                  goto LABEL_151;
                ++v148;
              }
              if ( v146 )
                v69 = v146;
            }
          }
LABEL_151:
          LODWORD(v253) = v114;
          if ( *v69 != -8 )
            --HIDWORD(v253);
          *v69 = v67;
          v72 = 0;
          v69[1] = 0;
          v69[2] = 0;
          v69[3] = 0;
          v226.m128i_i64[0] = (__int64)(v65 - 7);
          goto LABEL_154;
        }
LABEL_229:
        sub_167C9F0((__int64)&v251, 2 * v254);
        if ( !v254 )
        {
LABEL_399:
          LODWORD(v253) = v253 + 1;
          BUG();
        }
        v150 = (v254 - 1) & (((unsigned int)v67 >> 9) ^ ((unsigned int)v67 >> 4));
        v114 = v253 + 1;
        v69 = &v252[4 * v150];
        v151 = *v69;
        if ( v67 != *v69 )
        {
          v152 = 1;
          v153 = 0;
          while ( v151 != -8 )
          {
            if ( v151 == -16 && !v153 )
              v153 = v69;
            v150 = (v254 - 1) & (v152 + v150);
            v69 = &v252[4 * v150];
            v151 = *v69;
            if ( v67 == *v69 )
              goto LABEL_151;
            ++v152;
          }
          if ( v153 )
            v69 = v153;
        }
        goto LABEL_151;
      }
LABEL_60:
      v71 = (_QWORD *)v69[2];
      v72 = (_BYTE *)v69[3];
      v226.m128i_i64[0] = (__int64)(v65 - 7);
      if ( v71 == (_QWORD *)v72 )
      {
LABEL_154:
        sub_167C6C0((__int64)(v69 + 1), v72, &v226);
LABEL_54:
        v65 = (__int64 *)v65[1];
        if ( v66 == v65 )
          goto LABEL_64;
      }
      else
      {
        if ( v71 )
        {
          *v71 = v65 - 7;
          v71 = (_QWORD *)v69[2];
        }
        v69[2] = v71 + 1;
        v65 = (__int64 *)v65[1];
        if ( v66 == v65 )
        {
LABEL_64:
          v64 = v230;
          goto LABEL_65;
        }
      }
    }
    ++v251;
    goto LABEL_229;
  }
LABEL_65:
  v73 = (__int64 *)v64[4];
  v74 = v64 + 3;
  if ( v73 != v64 + 3 )
  {
    while ( 1 )
    {
      if ( !v73 )
        BUG();
      if ( (*(_BYTE *)(v73 - 3) & 0xFu) - 2 > 1 )
        goto LABEL_67;
      v75 = *(v73 - 1);
      if ( !v75 )
        goto LABEL_67;
      if ( !v254 )
        break;
      v76 = (v254 - 1) & (((unsigned int)v75 >> 9) ^ ((unsigned int)v75 >> 4));
      v77 = &v252[4 * v76];
      v78 = *v77;
      if ( v75 != *v77 )
      {
        v115 = 1;
        v116 = 0;
        while ( v78 != -8 )
        {
          if ( !v116 && v78 == -16 )
            v116 = v77;
          v76 = (v254 - 1) & (v115 + v76);
          v77 = &v252[4 * v76];
          v78 = *v77;
          if ( v75 == *v77 )
            goto LABEL_73;
          ++v115;
        }
        if ( v116 )
          v77 = v116;
        ++v251;
        v117 = v253 + 1;
        if ( 4 * ((int)v253 + 1) < 3 * v254 )
        {
          if ( v254 - HIDWORD(v253) - v117 <= v254 >> 3 )
          {
            sub_167C9F0((__int64)&v251, v254);
            if ( !v254 )
              goto LABEL_400;
            v138 = 0;
            LODWORD(v139) = (v254 - 1) & (((unsigned int)v75 >> 9) ^ ((unsigned int)v75 >> 4));
            v117 = v253 + 1;
            v140 = 1;
            v77 = &v252[4 * (unsigned int)v139];
            v141 = *v77;
            if ( v75 != *v77 )
            {
              while ( v141 != -8 )
              {
                if ( v141 == -16 && !v138 )
                  v138 = v77;
                v139 = (v254 - 1) & ((_DWORD)v139 + v140);
                v77 = &v252[4 * v139];
                v141 = *v77;
                if ( v75 == *v77 )
                  goto LABEL_161;
                ++v140;
              }
              if ( v138 )
                v77 = v138;
            }
          }
LABEL_161:
          LODWORD(v253) = v117;
          if ( *v77 != -8 )
            --HIDWORD(v253);
          *v77 = v75;
          v79 = 0;
          v77[1] = 0;
          v77[2] = 0;
          v77[3] = 0;
          v226.m128i_i64[0] = (__int64)(v73 - 7);
          goto LABEL_164;
        }
LABEL_215:
        sub_167C9F0((__int64)&v251, 2 * v254);
        if ( !v254 )
        {
LABEL_400:
          LODWORD(v253) = v253 + 1;
          BUG();
        }
        v142 = (v254 - 1) & (((unsigned int)v75 >> 9) ^ ((unsigned int)v75 >> 4));
        v117 = v253 + 1;
        v77 = &v252[4 * v142];
        v143 = *v77;
        if ( v75 != *v77 )
        {
          v144 = 1;
          v145 = 0;
          while ( v143 != -8 )
          {
            if ( v143 == -16 && !v145 )
              v145 = v77;
            v142 = (v254 - 1) & (v144 + v142);
            v77 = &v252[4 * v142];
            v143 = *v77;
            if ( v75 == *v77 )
              goto LABEL_161;
            ++v144;
          }
          if ( v145 )
            v77 = v145;
        }
        goto LABEL_161;
      }
LABEL_73:
      v79 = (_BYTE *)v77[2];
      v80 = (_BYTE *)v77[3];
      v226.m128i_i64[0] = (__int64)(v73 - 7);
      if ( v80 == v79 )
      {
LABEL_164:
        sub_167C6C0((__int64)(v77 + 1), v79, &v226);
LABEL_67:
        v73 = (__int64 *)v73[1];
        if ( v74 == v73 )
          goto LABEL_77;
      }
      else
      {
        if ( v79 )
        {
          *(_QWORD *)v79 = v73 - 7;
          v79 = (_BYTE *)v77[2];
        }
        v77[2] = v79 + 8;
        v73 = (__int64 *)v73[1];
        if ( v74 == v73 )
        {
LABEL_77:
          v64 = v230;
          goto LABEL_78;
        }
      }
    }
    ++v251;
    goto LABEL_215;
  }
LABEL_78:
  v81 = (__int64 *)v64[6];
  v82 = v64 + 5;
  if ( v64 + 5 == v81 )
    goto LABEL_91;
  do
  {
    while ( 1 )
    {
      if ( !v81 )
        BUG();
      if ( (*(_BYTE *)(v81 - 2) & 0xFu) - 2 > 1 )
        goto LABEL_80;
      v83 = sub_15E4F10((__int64)(v81 - 6));
      v84 = v83;
      if ( !v83 )
        goto LABEL_80;
      if ( !v254 )
      {
        ++v251;
        goto LABEL_304;
      }
      v85 = ((unsigned int)v83 >> 9) ^ ((unsigned int)v83 >> 4);
      v86 = (v254 - 1) & v85;
      v87 = &v252[4 * v86];
      v88 = *v87;
      if ( v84 != *v87 )
      {
        v132 = 1;
        v133 = 0;
        while ( v88 != -8 )
        {
          if ( v88 == -16 && !v133 )
            v133 = v87;
          v86 = (v254 - 1) & (v132 + v86);
          v87 = &v252[4 * v86];
          v88 = *v87;
          if ( v84 == *v87 )
            goto LABEL_86;
          ++v132;
        }
        if ( v133 )
          v87 = v133;
        ++v251;
        v134 = v253 + 1;
        if ( 4 * ((int)v253 + 1) < 3 * v254 )
        {
          if ( v254 - HIDWORD(v253) - v134 <= v254 >> 3 )
          {
            sub_167C9F0((__int64)&v251, v254);
            if ( !v254 )
            {
LABEL_405:
              LODWORD(v253) = v253 + 1;
              BUG();
            }
            v185 = 0;
            LODWORD(v186) = (v254 - 1) & v85;
            v134 = v253 + 1;
            v187 = 1;
            v87 = &v252[4 * (unsigned int)v186];
            v188 = *v87;
            if ( v84 != *v87 )
            {
              while ( v188 != -8 )
              {
                if ( v188 == -16 && !v185 )
                  v185 = v87;
                v186 = (v254 - 1) & ((_DWORD)v186 + v187);
                v87 = &v252[4 * v186];
                v188 = *v87;
                if ( v84 == *v87 )
                  goto LABEL_195;
                ++v187;
              }
              if ( v185 )
                v87 = v185;
            }
          }
          goto LABEL_195;
        }
LABEL_304:
        sub_167C9F0((__int64)&v251, 2 * v254);
        if ( !v254 )
          goto LABEL_405;
        LODWORD(v181) = (v254 - 1) & (((unsigned int)v84 >> 9) ^ ((unsigned int)v84 >> 4));
        v134 = v253 + 1;
        v87 = &v252[4 * (unsigned int)v181];
        v182 = *v87;
        if ( v84 != *v87 )
        {
          v183 = 1;
          v184 = 0;
          while ( v182 != -8 )
          {
            if ( v182 == -16 && !v184 )
              v184 = v87;
            v181 = (v254 - 1) & ((_DWORD)v181 + v183);
            v87 = &v252[4 * v181];
            v182 = *v87;
            if ( v84 == *v87 )
              goto LABEL_195;
            ++v183;
          }
          if ( v184 )
            v87 = v184;
        }
LABEL_195:
        LODWORD(v253) = v134;
        if ( *v87 != -8 )
          --HIDWORD(v253);
        *v87 = v84;
        v90 = 0;
        v87[1] = 0;
        v87[2] = 0;
        v87[3] = 0;
        v226.m128i_i64[0] = (__int64)(v81 - 6);
        goto LABEL_198;
      }
LABEL_86:
      v89 = (_QWORD *)v87[2];
      v90 = (_BYTE *)v87[3];
      v226.m128i_i64[0] = (__int64)(v81 - 6);
      if ( v89 != (_QWORD *)v90 )
        break;
LABEL_198:
      sub_167C6C0((__int64)(v87 + 1), v90, &v226);
LABEL_80:
      v81 = (__int64 *)v81[1];
      if ( v82 == v81 )
        goto LABEL_90;
    }
    if ( v89 )
    {
      *v89 = v81 - 6;
      v89 = (_QWORD *)v87[2];
    }
    v87[2] = (__int64)(v89 + 1);
    v81 = (__int64 *)v81[1];
  }
  while ( v82 != v81 );
LABEL_90:
  v64 = v230;
LABEL_91:
  v91 = (__int64 *)v64[2];
  v92 = v64 + 1;
  if ( v64 + 1 != v91 )
  {
    do
    {
      v93 = (char *)(v91 - 7);
      if ( !v91 )
        v93 = 0;
      if ( (unsigned __int8)sub_167D520((__int64)&v229, v93) )
        goto LABEL_97;
      v91 = (__int64 *)v91[1];
    }
    while ( v92 != v91 );
    v64 = v230;
  }
  v109 = (__int64 *)v64[4];
  v110 = v64 + 3;
  if ( v109 != v64 + 3 )
  {
    do
    {
      v111 = (char *)(v109 - 7);
      if ( !v109 )
        v111 = 0;
      if ( (unsigned __int8)sub_167D520((__int64)&v229, v111) )
        goto LABEL_97;
      v109 = (__int64 *)v109[1];
    }
    while ( v110 != v109 );
    v64 = v230;
  }
  v118 = v64[6];
  for ( i = v64 + 5; i != (__int64 *)v118; v118 = *(_QWORD *)(v118 + 8) )
  {
    v131 = (char *)(v118 - 48);
    if ( !v118 )
      v131 = 0;
    if ( (unsigned __int8)sub_167D520((__int64)&v229, v131) )
      goto LABEL_97;
  }
  v120 = v235;
  v121 = 0;
  v207 = 0;
  if ( v236 == v235 )
    goto LABEL_279;
  while ( 2 )
  {
    v220[0] = (__int64 *)sub_15E4F10((__int64)v120[v121]);
    if ( v220[0] )
    {
      v122 = sub_167CC10((__int64)&v251, (__int64 *)v220);
      v123 = (__int64 *)v122[2];
      v124 = (__int64 *)v122[1];
      if ( v124 != v123 )
      {
        while ( 1 )
        {
          v125 = *v124;
          v226.m128i_i64[0] = v125;
          if ( (*(_BYTE *)(v125 + 23) & 0x20) == 0 )
            break;
          if ( ((*(_BYTE *)(v125 + 32) + 9) & 0xFu) <= 1 )
            break;
          v126 = *v229;
          v127 = sub_1649960(v125);
          v129 = sub_1632000(v126, (__int64)v127, v128);
          if ( !v129 || (*(_BYTE *)(v129 + 32) & 0xFu) - 7 <= 1 )
            break;
          LOBYTE(v217) = 1;
          if ( (v238 & 1) != 0 )
            goto LABEL_174;
          v130 = sub_167C240((__int64)&v229, (bool *)&v217, v129, v226.m128i_i64[0]);
          if ( (_BYTE)v130 )
          {
            v94 = v130;
            goto LABEL_98;
          }
          if ( (_BYTE)v217 )
            goto LABEL_174;
LABEL_175:
          if ( v123 == ++v124 )
            goto LABEL_169;
        }
        LOBYTE(v217) = 1;
LABEL_174:
        sub_167D2C0((__int64)&v231, (char **)&v226);
        goto LABEL_175;
      }
    }
LABEL_169:
    v120 = v235;
    v121 = ++v207;
    if ( v207 < (unsigned __int64)(v236 - v235) )
      continue;
    break;
  }
  if ( v243 && v236 != v235 )
  {
    v166 = (__int64 *)v235;
    v167 = (__int64 *)v236;
    do
    {
      v168 = *v166++;
      v169 = sub_1649960(v168);
      sub_167C570((__int64)&v239, v169, v170);
    }
    while ( v167 != v166 );
  }
LABEL_279:
  v226.m128i_i64[0] = (__int64)&v229;
  v220[0] = v230;
  v228 = sub_167CE60;
  v209 = 0;
  v227 = (__int64 (__fastcall *)(__m128i *, __m128i *, int))sub_167BA20;
  v230 = 0;
  sub_16786A0(
    &v210,
    (__int64)v229,
    v220,
    v235,
    v236 - v235,
    &v226,
    *(double *)v13.m128_u64,
    *(double *)v15.m128i_i64,
    a7,
    a8,
    a9,
    a10,
    a11,
    a12,
    0,
    (v238 & 4) != 0);
  v171 = v220[0];
  v172 = v189;
  if ( v220[0] )
  {
    sub_1633490((_QWORD **)v220[0]);
    j_j___libc_free_0(v171, 736);
  }
  if ( v227 )
    v227(&v226, &v226, 3);
  v173 = (__int64 *)(v210 & 0xFFFFFFFFFFFFFFFELL);
  if ( (v210 & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    v174 = v210 & 0xFFFFFFFFFFFFFFFELL;
    v210 = 0;
    v211 = 0;
    v226.m128i_i64[0] = v197;
    v226.m128i_i64[1] = (__int64)&v209;
    v212 = 0;
    if ( (*(unsigned __int8 (__fastcall **)(unsigned __int64, void *, __int64))(*v173 + 48))(v174, &unk_4FA032A, v172) )
    {
      v213 = 1;
      v208 = (__int64 **)v173[2];
      if ( (__int64 **)v173[1] != v208 )
      {
        v175 = (__int64 **)v173[1];
        do
        {
          v215 = *v175;
          *v175 = 0;
          sub_167BF30((__int64 *)&v217, &v215, (__int64)&v226);
          v176 = v213;
          v213 = 0;
          v214 = v176 | 1;
          sub_12BEC00((unsigned __int64 *)v220, (unsigned __int64 *)&v214, &v217);
          if ( (v213 & 1) != 0 || (v213 & 0xFFFFFFFFFFFFFFFELL) != 0 )
            sub_16BCAE0(&v213);
          v213 |= (unsigned __int64)v220[0] | 1;
          if ( (v214 & 1) != 0 || (v214 & 0xFFFFFFFFFFFFFFFELL) != 0 )
            sub_16BCAE0(&v214);
          if ( (v217 & 1) != 0 || (v217 & 0xFFFFFFFFFFFFFFFELL) != 0 )
            sub_16BCAE0(&v217);
          if ( v215 )
            (*(void (__fastcall **)(__int64 *))(*v215 + 8))(v215);
          ++v175;
        }
        while ( v208 != v175 );
      }
      v217 = v213 | 1;
      (*(void (__fastcall **)(__int64 *))(*v173 + 8))(v173);
    }
    else
    {
      v220[0] = v173;
      sub_167BF30((__int64 *)&v217, v220, (__int64)&v226);
      if ( v220[0] )
        (*(void (__fastcall **)(__int64 *))(*v220[0] + 8))(v220[0]);
    }
    if ( (v217 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    {
      v217 = v217 & 0xFFFFFFFFFFFFFFFELL | 1;
      sub_16BCAE0(&v217);
    }
    if ( (v212 & 1) != 0 || (v212 & 0xFFFFFFFFFFFFFFFELL) != 0 )
      sub_16BCAE0(&v212);
    if ( (v211 & 1) != 0 || (v211 & 0xFFFFFFFFFFFFFFFELL) != 0 )
      sub_16BCAE0(&v211);
    if ( (v210 & 1) != 0 || (v210 & 0xFFFFFFFFFFFFFFFELL) != 0 )
      sub_16BCAE0(&v210);
  }
  v94 = v209;
  if ( v209 )
    goto LABEL_97;
  if ( v243 )
    v244(&v242, v197, &v239);
LABEL_98:
  j___libc_free_0(v223);
  if ( v254 )
  {
    v95 = v252;
    v96 = &v252[4 * v254];
    do
    {
      if ( *v95 != -8 && *v95 != -16 )
      {
        v97 = v95[1];
        if ( v97 )
          j_j___libc_free_0(v97, v95[3] - v97);
      }
      v95 += 4;
    }
    while ( v96 != v95 );
  }
  j___libc_free_0(v252);
  sub_167BA50((__int64)v247);
  if ( v243 )
    v243(&v242, &v242, 3);
  v98 = v239;
  if ( HIDWORD(v240) && (_DWORD)v240 )
  {
    v99 = 8LL * (unsigned int)v240;
    v100 = 0;
    do
    {
      v101 = *(_QWORD *)(v98 + v100);
      if ( v101 && v101 != -8 )
      {
        _libc_free(v101);
        v98 = v239;
      }
      v100 += 8;
    }
    while ( v99 != v100 );
  }
  _libc_free(v98);
  if ( v235 )
    j_j___libc_free_0(v235, v237 - (_QWORD)v235);
  j___libc_free_0(v232);
  v102 = v230;
  if ( v230 )
  {
    sub_1633490((_QWORD **)v230);
    j_j___libc_free_0(v102, 736);
  }
  return v94;
}
