// Function: sub_CC0130
// Address: 0xcc0130
//
unsigned __int64 __fastcall sub_CC0130(__int64 a1, __int64 *a2, unsigned __int64 a3)
{
  __int64 v3; // rbx
  __int64 v4; // rax
  __int64 v5; // rdi
  unsigned __int64 result; // rax
  __m128i *v7; // r14
  char *v8; // r13
  unsigned __int64 v9; // rcx
  unsigned __int64 v10; // r12
  char v11; // al
  char *v12; // r15
  char *v13; // rsi
  __int64 v14; // rdx
  char v15; // cl
  _QWORD *v16; // rdx
  unsigned __int8 v17; // cl
  bool v18; // zf
  __int64 v19; // r9
  __m128i v20; // xmm6
  __m128i v21; // xmm7
  __m128i v22; // xmm3
  __m128i v23; // xmm4
  unsigned __int8 v24; // r8
  __m128i v25; // xmm1
  __m128i v26; // xmm0
  __m128i v27; // xmm6
  __m128i v28; // xmm7
  __m128i v29; // xmm2
  __m128i v30; // xmm3
  __m128i v31; // xmm4
  unsigned __int8 v32; // al
  __m128i v33; // xmm4
  unsigned __int8 v34; // r8
  __int64 v35; // r15
  __m128i v36; // xmm7
  __m128i v37; // xmm3
  __int64 v38; // r12
  __m128i v39; // xmm1
  __m128i v40; // xmm0
  __m128i v41; // xmm5
  __m128i v42; // xmm6
  __m128i v43; // xmm4
  __m128i v44; // xmm2
  __m128i v45; // xmm1
  __m128i v46; // xmm0
  __m128i v47; // xmm5
  __int32 v48; // eax
  char v49; // cl
  char v50; // edx^2
  __int64 v51; // rax
  unsigned __int64 v52; // rax
  int v53; // eax
  __int64 v54; // rdx
  __m128i v55; // xmm6
  __m128i v56; // xmm7
  __int64 v57; // r10
  __int64 v58; // rdx
  __m256i *v59; // r13
  unsigned __int64 v60; // rcx
  unsigned __int64 i; // r12
  unsigned __int8 v62; // r10
  __m128i v63; // xmm2
  __m128i v64; // xmm3
  __int64 v65; // rcx
  char v66; // al
  __int64 *v67; // rbx
  __int64 *v68; // rsi
  unsigned __int64 v69; // rcx
  unsigned __int64 v70; // rax
  __m128i *v71; // rdx
  __m128i v72; // xmm5
  __m128i v73; // xmm1
  __m128i v74; // xmm0
  __m128i v75; // xmm6
  __m128i v76; // xmm7
  __m128i v77; // xmm5
  __m128i v78; // xmm2
  __m128i v79; // xmm6
  __m128i v80; // xmm7
  __m128i v81; // xmm1
  __m128i v82; // xmm0
  unsigned __int8 v83; // al
  __m256i *v84; // r14
  __m128i v85; // xmm3
  unsigned __int8 v86; // r8
  __int64 v87; // r13
  __m128i v88; // xmm1
  __m128i v89; // xmm0
  __int64 v90; // r15
  __m128i v91; // xmm4
  __m128i v92; // xmm5
  __m128i v93; // xmm6
  __m128i v94; // xmm7
  __m128i v95; // xmm2
  __m128i v96; // xmm3
  __m128i v97; // xmm4
  __m128i v98; // xmm5
  __m128i v99; // xmm6
  __int16 v100; // edx^2
  __int8 v101; // al
  int v102; // eax
  __int64 v103; // rdx
  unsigned __int8 v104; // cl
  __int64 *v105; // r15
  char v106; // al
  unsigned __int64 v107; // r13
  __int64 *v108; // r14
  __int64 *v109; // rsi
  __int64 v110; // rax
  int v111; // edx
  char v112; // cl
  __int64 *v113; // r15
  unsigned __int64 v114; // r13
  __m128i v115; // xmm0
  unsigned __int8 v116; // r8
  __int64 v117; // r14
  __m128i v118; // xmm6
  __m128i v119; // xmm7
  __int64 v120; // r12
  __m128i v121; // xmm2
  __m128i v122; // xmm3
  __m128i v123; // xmm4
  __m128i v124; // xmm5
  __m128i v125; // xmm1
  __m128i v126; // xmm0
  __m128i v127; // xmm2
  __m128i v128; // xmm3
  __m128i v129; // xmm4
  __int32 v130; // eax
  char v131; // cl
  char v132; // edx^2
  __int64 v133; // rax
  unsigned __int64 v134; // rax
  __int64 v135; // r14
  __m128i *v136; // r11
  unsigned __int64 v137; // rbx
  __m256i *v138; // rcx
  __m128i *v139; // rdx
  unsigned __int64 v140; // r15
  __int64 v141; // rcx
  unsigned __int64 v142; // r15
  __m128i v143; // xmm5
  __m128i v144; // xmm6
  __m128i v145; // xmm7
  __int64 v146; // rdi
  unsigned __int8 v147; // al
  __m256i *v148; // r14
  __m128i v149; // xmm1
  unsigned __int8 v150; // r8
  __int64 v151; // r13
  __m128i v152; // xmm5
  __m128i v153; // xmm6
  __int64 v154; // r15
  __m128i v155; // xmm0
  __m128i v156; // xmm2
  __m128i v157; // xmm3
  __m128i v158; // xmm4
  __m128i v159; // xmm7
  __m128i v160; // xmm1
  __m128i v161; // xmm0
  __m128i v162; // xmm2
  __m128i v163; // xmm3
  __int16 v164; // edx^2
  __int8 v165; // al
  __m128i *v166; // rax
  unsigned __int64 v167; // rdi
  unsigned __int8 v168; // dl
  __m256i *v169; // r14
  __m128i v170; // xmm5
  unsigned __int8 v171; // r8
  __int64 v172; // r13
  __m128i v173; // xmm2
  __m128i v174; // xmm3
  __int64 v175; // r15
  __m128i v176; // xmm6
  __m128i v177; // xmm7
  __m128i v178; // xmm1
  __m128i v179; // xmm0
  __m128i v180; // xmm4
  __m128i v181; // xmm5
  __m128i v182; // xmm6
  __m128i v183; // xmm7
  __m128i v184; // xmm1
  __int32 v185; // edx
  char v186; // al
  unsigned __int64 v187; // rdx
  __m128i v188; // xmm7
  __m128i *v189; // rax
  __m128i *v190; // rdx
  const __m128i *v191; // rsi
  unsigned __int64 v192; // rsi
  __int8 *v193; // rdx
  char *v194; // rdi
  unsigned int v195; // edx
  char *v196; // r14
  unsigned int v197; // edx
  unsigned int v198; // ecx
  __int64 v199; // rdi
  unsigned __int64 v200; // rsi
  signed __int64 v201; // r8
  unsigned int v202; // edx
  __int64 v203; // rdi
  unsigned __int64 v204; // rsi
  signed __int64 v205; // r14
  unsigned int v206; // edx
  __int64 v207; // rdi
  size_t v208; // r13
  unsigned __int64 v209; // rdx
  __int64 *v210; // rax
  int v211; // edx
  unsigned __int32 v212; // [rsp+1Ch] [rbp-484h]
  unsigned __int32 v213; // [rsp+1Ch] [rbp-484h]
  unsigned __int32 v214; // [rsp+20h] [rbp-480h]
  unsigned __int32 v215; // [rsp+20h] [rbp-480h]
  unsigned __int32 v216; // [rsp+24h] [rbp-47Ch]
  unsigned __int32 v217; // [rsp+24h] [rbp-47Ch]
  __int64 v218; // [rsp+28h] [rbp-478h]
  __int32 v219; // [rsp+28h] [rbp-478h]
  __int64 v220; // [rsp+30h] [rbp-470h]
  __int64 v221; // [rsp+30h] [rbp-470h]
  __int64 v222; // [rsp+38h] [rbp-468h]
  __int32 v223; // [rsp+40h] [rbp-460h]
  __int64 *v224; // [rsp+48h] [rbp-458h]
  __int8 v225; // [rsp+50h] [rbp-450h]
  __int64 v226; // [rsp+50h] [rbp-450h]
  __int128 v227; // [rsp+50h] [rbp-450h]
  int v228; // [rsp+50h] [rbp-450h]
  unsigned __int64 v229; // [rsp+58h] [rbp-448h]
  __int64 v230; // [rsp+58h] [rbp-448h]
  unsigned __int64 v231; // [rsp+60h] [rbp-440h]
  __int64 v232; // [rsp+68h] [rbp-438h]
  __int64 *v233; // [rsp+68h] [rbp-438h]
  __int8 v234; // [rsp+68h] [rbp-438h]
  unsigned __int64 v235; // [rsp+68h] [rbp-438h]
  unsigned __int8 v236; // [rsp+68h] [rbp-438h]
  unsigned __int64 v237; // [rsp+70h] [rbp-430h]
  unsigned __int64 v238; // [rsp+70h] [rbp-430h]
  unsigned __int64 v239; // [rsp+70h] [rbp-430h]
  unsigned __int8 v240; // [rsp+70h] [rbp-430h]
  __m128i *v241; // [rsp+70h] [rbp-430h]
  unsigned __int64 v242; // [rsp+70h] [rbp-430h]
  __m128i v243; // [rsp+70h] [rbp-430h]
  unsigned __int64 v244; // [rsp+70h] [rbp-430h]
  __m128i *v245; // [rsp+70h] [rbp-430h]
  __m128i v246; // [rsp+80h] [rbp-420h] BYREF
  __m128i v247; // [rsp+90h] [rbp-410h] BYREF
  __int64 v248; // [rsp+A0h] [rbp-400h]
  __m128i v249; // [rsp+A8h] [rbp-3F8h] BYREF
  __m128i v250; // [rsp+B8h] [rbp-3E8h] BYREF
  __m128i v251; // [rsp+C8h] [rbp-3D8h] BYREF
  __m128i v252; // [rsp+D8h] [rbp-3C8h] BYREF
  __int16 v253; // [rsp+E8h] [rbp-3B8h]
  unsigned __int8 v254; // [rsp+EAh] [rbp-3B6h]
  __m128i v255; // [rsp+F0h] [rbp-3B0h] BYREF
  __m128i v256; // [rsp+100h] [rbp-3A0h] BYREF
  __m128i v257; // [rsp+110h] [rbp-390h] BYREF
  __m128i v258; // [rsp+120h] [rbp-380h] BYREF
  __m128i v259; // [rsp+130h] [rbp-370h]
  __m128i v260; // [rsp+140h] [rbp-360h]
  __m128i v261; // [rsp+150h] [rbp-350h]
  _BYTE v262[8]; // [rsp+168h] [rbp-338h]
  __m128i v263; // [rsp+170h] [rbp-330h] BYREF
  __m128i v264; // [rsp+180h] [rbp-320h]
  __m128i v265; // [rsp+190h] [rbp-310h] BYREF
  __m128i v266; // [rsp+1A0h] [rbp-300h]
  __m128i v267; // [rsp+1B0h] [rbp-2F0h]
  __m128i v268; // [rsp+1C0h] [rbp-2E0h]
  __m128i v269; // [rsp+1D0h] [rbp-2D0h]
  __m256i v270; // [rsp+270h] [rbp-230h] BYREF
  _BYTE v271[24]; // [rsp+290h] [rbp-210h] BYREF
  __m128i v272; // [rsp+2A8h] [rbp-1F8h] BYREF
  __m128i v273; // [rsp+2B8h] [rbp-1E8h] BYREF
  __m128i v274; // [rsp+2C8h] [rbp-1D8h] BYREF
  char v275; // [rsp+2D8h] [rbp-1C8h]
  unsigned __int8 v276; // [rsp+2D9h] [rbp-1C7h]

  v3 = a1;
  v4 = *(unsigned __int8 *)(a1 + 137);
  v5 = *(unsigned __int8 *)(a1 + 136);
  v224 = a2;
  v231 = a3;
  result = v5 + (v4 << 6);
  if ( result )
  {
    v7 = (__m128i *)(v3 + 32);
    v8 = (char *)a2;
    v9 = 1024 - result;
    v232 = v3 + 72;
    if ( 1024 - result > a3 )
      v9 = a3;
    v229 = v9;
    v10 = v9;
    if ( (_BYTE)v5 )
    {
      v208 = 64 - v5;
      if ( v9 <= 64 - v5 )
        v208 = v9;
      v12 = (char *)a2 + v208;
      memcpy((void *)(v3 + 72 + v5), a2, v208);
      LOBYTE(v14) = v208 + *(_BYTE *)(v3 + 136);
      *(_BYTE *)(v3 + 136) = v14;
      v10 -= v208;
      if ( !v10 )
      {
        v14 = (unsigned __int8)v14;
        v15 = 0;
LABEL_11:
        v16 = (_QWORD *)(v232 + v14);
        result = (unsigned int)v10;
        if ( (unsigned int)v10 >= 8 )
        {
          *v16 = *(_QWORD *)v12;
          *(_QWORD *)((char *)v16 + (unsigned int)v10 - 8) = *(_QWORD *)&v12[(unsigned int)v10 - 8];
          v200 = (unsigned __int64)(v16 + 1) & 0xFFFFFFFFFFFFFFF8LL;
          v201 = v12 - ((char *)v16 - v200);
          result = ((_DWORD)v10 + (_DWORD)v16 - (_DWORD)v200) & 0xFFFFFFF8;
          if ( (unsigned int)result >= 8 )
          {
            result = ((_DWORD)v10 + (_DWORD)v16 - (_DWORD)v200) & 0xFFFFFFF8;
            v202 = 0;
            do
            {
              v203 = v202;
              v202 += 8;
              *(_QWORD *)(v200 + v203) = *(_QWORD *)(v201 + v203);
            }
            while ( v202 < (unsigned int)result );
          }
        }
        else if ( (v10 & 4) != 0 )
        {
          *(_DWORD *)v16 = *(_DWORD *)v12;
          *(_DWORD *)((char *)v16 + (unsigned int)v10 - 4) = *(_DWORD *)&v12[(unsigned int)v10 - 4];
        }
        else if ( (_DWORD)v10 )
        {
          *(_BYTE *)v16 = *v12;
          if ( (v10 & 2) != 0 )
            *(_WORD *)((char *)v16 + (unsigned int)v10 - 2) = *(_WORD *)&v12[(unsigned int)v10 - 2];
        }
        v17 = *(_BYTE *)(v3 + 136) + v15;
        v18 = v231 == v229;
        v231 -= v229;
        *(_BYTE *)(v3 + 136) = v17;
        if ( v18 )
          return result;
        v18 = *(_BYTE *)(v3 + 137) == 0;
        v19 = *(_QWORD *)(v3 + 64);
        v275 = v17;
        v20 = _mm_loadu_si128((const __m128i *)(v3 + 72));
        v21 = _mm_loadu_si128((const __m128i *)(v3 + 88));
        v22 = _mm_loadu_si128((const __m128i *)(v3 + 104));
        v23 = _mm_loadu_si128((const __m128i *)(v3 + 120));
        v24 = *(_BYTE *)(v3 + 138) | v18;
        v25 = _mm_loadu_si128((const __m128i *)(v3 + 32));
        *(_QWORD *)v271 = v19;
        v26 = _mm_loadu_si128((const __m128i *)(v3 + 48));
        *(__m128i *)&v271[8] = v20;
        v27 = _mm_loadu_si128((const __m128i *)v271);
        v276 = v24 | 2;
        v272 = v21;
        v28 = _mm_loadu_si128((const __m128i *)&v271[16]);
        v273 = v22;
        v29 = _mm_loadu_si128((const __m128i *)&v272.m128i_u64[1]);
        v274 = v23;
        v30 = _mm_loadu_si128((const __m128i *)&v273.m128i_u64[1]);
        v31 = _mm_loadu_si128((const __m128i *)&v274.m128i_u64[1]);
        *(__m128i *)v270.m256i_i8 = v25;
        *(__m128i *)&v270.m256i_u64[2] = v26;
        v255 = v25;
        v256 = v26;
        v257 = v27;
        v258 = v28;
        v259 = v29;
        v260 = v30;
        v261 = v31;
        sub_CC2280(&v270, &v257.m128i_u64[1], v17, v19, v24 | 2u);
        v225 = v270.m256i_i8[0];
        v216 = (unsigned __int32)v270.m256i_i32[0] >> 8;
        v212 = HIBYTE(v270.m256i_i32[0]);
        v214 = HIWORD(v270.m256i_i32[0]);
        v220 = *(__int64 *)((char *)v270.m256i_i64 + 4);
        v218 = *(__int64 *)((char *)&v270.m256i_i64[1] + 4);
        v222 = *(__int64 *)((char *)&v270.m256i_i64[2] + 4);
        v223 = v270.m256i_i32[7];
        v238 = (int)sub_39FAC40(*(_QWORD *)(v3 + 64));
        v32 = *(_BYTE *)(v3 + 144);
        if ( v238 < v32 )
        {
          do
          {
            v33 = _mm_loadu_si128((const __m128i *)(v3 + 16));
            v34 = *(_BYTE *)(v3 + 138) | 4;
            *(__m128i *)v270.m256i_i8 = _mm_loadu_si128((const __m128i *)v3);
            v35 = 32 * (v32 - 2);
            *(__m128i *)&v270.m256i_u64[2] = v33;
            v36 = _mm_loadu_si128((const __m128i *)&v270);
            v37 = _mm_loadu_si128((const __m128i *)&v270.m256i_u64[2]);
            v38 = v3 + (int)v35 + 145;
            v276 = v34;
            v39 = _mm_loadu_si128((const __m128i *)v38);
            v40 = _mm_loadu_si128((const __m128i *)(v38 + 16));
            v275 = 64;
            v41 = _mm_loadu_si128((const __m128i *)(v38 + 32));
            v42 = _mm_loadu_si128((const __m128i *)(v38 + 48));
            *(_QWORD *)v271 = 0;
            *(__m128i *)&v271[8] = v39;
            v43 = _mm_loadu_si128((const __m128i *)v271);
            v272 = v40;
            v44 = _mm_loadu_si128((const __m128i *)&v271[16]);
            v273 = v41;
            v45 = _mm_loadu_si128((const __m128i *)&v272.m128i_u64[1]);
            v274 = v42;
            v46 = _mm_loadu_si128((const __m128i *)&v273.m128i_u64[1]);
            v47 = _mm_loadu_si128((const __m128i *)&v274.m128i_u64[1]);
            v263 = v36;
            v264 = v37;
            v265 = v43;
            v266 = v44;
            v267 = v45;
            v268 = v46;
            v269 = v47;
            sub_CC2280(&v270, &v265.m128i_u64[1], 64, 0, v34);
            v48 = v270.m256i_i32[0];
            *(_BYTE *)(v3 + v35 + 145) = v270.m256i_i8[0];
            v49 = BYTE1(v48);
            v50 = BYTE2(v48);
            *(_BYTE *)(v38 + 3) = HIBYTE(v48);
            v51 = *(__int64 *)((char *)v270.m256i_i64 + 4);
            *(_BYTE *)(v38 + 2) = v50;
            *(_QWORD *)(v38 + 4) = v51;
            v52 = *(unsigned __int64 *)((char *)&v270.m256i_u64[1] + 4);
            *(_BYTE *)(v38 + 1) = v49;
            *(_OWORD *)(v38 + 12) = __PAIR128__(*(unsigned __int64 *)((char *)&v270.m256i_u64[2] + 4), v52);
            *(_DWORD *)(v38 + 28) = v270.m256i_i32[7];
            v32 = *(_BYTE *)(v3 + 144) - 1;
            *(_BYTE *)(v3 + 144) = v32;
          }
          while ( v238 < v32 );
          v7 = (__m128i *)(v3 + 32);
        }
        v53 = 32 * v32;
        v224 = (__int64 *)((char *)v224 + v229);
        *(_BYTE *)(v3 + v53 + 145) = v225;
        v54 = v3 + v53 + 145;
        *(_BYTE *)(v54 + 1) = v216;
        *(_BYTE *)(v54 + 2) = v214;
        *(_BYTE *)(v54 + 3) = v212;
        *(_QWORD *)(v54 + 4) = v220;
        *(_QWORD *)(v54 + 12) = v218;
        *(_QWORD *)(v54 + 20) = v222;
        *(_DWORD *)(v54 + 28) = v223;
        v55 = _mm_loadu_si128((const __m128i *)v3);
        v56 = _mm_loadu_si128((const __m128i *)(v3 + 16));
        v57 = *(_QWORD *)(v3 + 64) + 1LL;
        ++*(_BYTE *)(v3 + 144);
        *(__m128i *)(v3 + 32) = v55;
        v7[1] = v56;
        *(_QWORD *)(v3 + 64) = v57;
        *(_OWORD *)(v3 + 72) = 0;
        *(_OWORD *)(v3 + 88) = 0;
        *(_OWORD *)(v3 + 104) = 0;
        *(_OWORD *)(v3 + 120) = 0;
        *(_WORD *)(v3 + 136) = 0;
        if ( v231 <= 0x400 )
          goto LABEL_47;
        goto LABEL_22;
      }
      sub_CC2280(
        v3 + 32,
        v232,
        64,
        *(_QWORD *)(v3 + 64),
        (unsigned __int8)(*(_BYTE *)(v3 + 138) | (*(_BYTE *)(v3 + 137) == 0)));
      ++*(_BYTE *)(v3 + 137);
      *(_BYTE *)(v3 + 136) = 0;
      *(_OWORD *)(v3 + 72) = 0;
      *(_OWORD *)(v3 + 88) = 0;
      *(_OWORD *)(v3 + 104) = 0;
      *(_OWORD *)(v3 + 120) = 0;
      v8 = (char *)a2 + v208;
    }
    if ( v10 <= 0x40 )
    {
      v14 = *(unsigned __int8 *)(v3 + 136);
      v12 = v8;
      if ( v10 > 64 - v14 )
        LODWORD(v10) = 64 - v14;
      v15 = v10;
    }
    else
    {
      v11 = *(_BYTE *)(v3 + 137);
      v237 = (v10 - 65) >> 6;
      v12 = &v8[64 * v237 + 64];
      do
      {
        v13 = v8;
        v8 += 64;
        sub_CC2280(v3 + 32, v13, 64, *(_QWORD *)(v3 + 64), (unsigned __int8)(*(_BYTE *)(v3 + 138) | (v11 == 0)));
        v11 = *(_BYTE *)(v3 + 137) + 1;
        *(_BYTE *)(v3 + 137) = v11;
      }
      while ( v8 != v12 );
      v14 = *(unsigned __int8 *)(v3 + 136);
      v10 = v10 - (v237 << 6) - 64;
      if ( v10 > 64 - v14 )
        LODWORD(v10) = 64 - v14;
      v15 = v10;
    }
    goto LABEL_11;
  }
  if ( a3 <= 0x400 )
    goto LABEL_44;
  v57 = *(_QWORD *)(v3 + 64);
LABEL_22:
  v58 = v57;
  v59 = &v270;
  do
  {
    _BitScanReverse64(&v60, v231 | 1);
    for ( i = 1LL << v60; ((v58 << 10) & (i - 1)) != 0; i >>= 1 )
      ;
    v62 = *(_BYTE *)(v3 + 138);
    if ( i <= 0x400 )
    {
      v63 = _mm_loadu_si128((const __m128i *)v3);
      v64 = _mm_loadu_si128((const __m128i *)(v3 + 16));
      v253 = 0;
      v254 = v62;
      v248 = v58;
      v246 = v63;
      v247 = v64;
      v249 = 0;
      v250 = 0;
      v251 = 0;
      v252 = 0;
      if ( i <= 0x40 )
      {
        v71 = &v249;
        v69 = i;
        v233 = v224;
        v70 = 64;
      }
      else
      {
        v226 = v3;
        v65 = v58;
        v66 = 0;
        v67 = v224;
        v233 = &v224[8 * ((i - 65) >> 6) + 8];
        while ( 1 )
        {
          v68 = v67;
          v67 += 8;
          sub_CC2280(&v246, v68, 64, v65, (unsigned __int8)(v62 | (v66 == 0)));
          v66 = ++HIBYTE(v253);
          if ( v67 == &v224[8 * ((i - 65) >> 6) + 8] )
            break;
          v62 = v254;
          v65 = v248;
        }
        v3 = v226;
        v69 = i - ((i - 65) >> 6 << 6) - 64;
        v70 = 64LL - (unsigned __int8)v253;
        v71 = (__m128i *)((char *)&v249 + (unsigned __int8)v253);
      }
      if ( v69 <= v70 )
        LODWORD(v70) = v69;
      if ( (unsigned int)v70 >= 8 )
      {
        v71->m128i_i64[0] = *v233;
        *(__int64 *)((char *)&v71->m128i_i64[-1] + (unsigned int)v70) = *(__int64 *)((char *)v233 + (unsigned int)v70 - 8);
        v192 = (unsigned __int64)&v71->m128i_u64[1] & 0xFFFFFFFFFFFFFFF8LL;
        v193 = &v71->m128i_i8[-v192];
        v194 = (char *)((char *)v233 - v193);
        v195 = (v70 + (_DWORD)v193) & 0xFFFFFFF8;
        v196 = v194;
        if ( v195 >= 8 )
        {
          v197 = v195 & 0xFFFFFFF8;
          v198 = 0;
          do
          {
            v199 = v198;
            v198 += 8;
            *(_QWORD *)(v192 + v199) = *(_QWORD *)&v196[v199];
          }
          while ( v198 < v197 );
        }
      }
      else if ( (v70 & 4) != 0 )
      {
        v71->m128i_i32[0] = *(_DWORD *)v233;
        *(__int32 *)((char *)&v71->m128i_i32[-1] + (unsigned int)v70) = *(_DWORD *)((char *)v233 + (unsigned int)v70 - 4);
      }
      else if ( (_DWORD)v70 )
      {
        v71->m128i_i8[0] = *(_BYTE *)v233;
        if ( (v70 & 2) != 0 )
          *(__int16 *)((char *)&v71->m128i_i16[-1] + (unsigned int)v70) = *(_WORD *)((char *)v233 + (unsigned int)v70 - 2);
      }
      v72 = _mm_loadu_si128(&v249);
      v73 = _mm_loadu_si128(&v246);
      v275 = v253 + v70;
      v74 = _mm_loadu_si128(&v247);
      *(__m128i *)&v271[8] = v72;
      v75 = _mm_loadu_si128(&v250);
      v76 = _mm_loadu_si128(&v251);
      *(__m128i *)v270.m256i_i8 = v73;
      v77 = _mm_loadu_si128(&v252);
      *(_QWORD *)v271 = v248;
      v272 = v75;
      v274 = v77;
      v276 = v254 | (HIBYTE(v253) == 0) | 2;
      v78 = _mm_loadu_si128((const __m128i *)&v274.m128i_u64[1]);
      v273 = v76;
      v79 = _mm_loadu_si128((const __m128i *)&v272.m128i_u64[1]);
      v80 = _mm_loadu_si128((const __m128i *)&v273.m128i_u64[1]);
      *(__m128i *)&v270.m256i_u64[2] = v74;
      v255 = v73;
      v81 = _mm_loadu_si128((const __m128i *)v271);
      v256 = v74;
      v82 = _mm_loadu_si128((const __m128i *)&v271[16]);
      LOBYTE(v253) = v253 + v70;
      v257 = v81;
      v258 = v82;
      v259 = v79;
      v260 = v80;
      v261 = v78;
      sub_CC2280(v59, &v257.m128i_u64[1], (unsigned __int8)v253, v248, v276);
      v234 = v270.m256i_i8[0];
      v217 = (unsigned __int32)v270.m256i_i32[0] >> 8;
      v213 = HIBYTE(v270.m256i_i32[0]);
      v215 = HIWORD(v270.m256i_i32[0]);
      v227 = *(_OWORD *)((char *)v270.m256i_i64 + 4);
      v221 = *(__int64 *)((char *)&v270.m256i_i64[2] + 4);
      v219 = v270.m256i_i32[7];
      v239 = (int)sub_39FAC40(v248);
      v83 = *(_BYTE *)(v3 + 144);
      if ( v239 < v83 )
      {
        v84 = v59;
        do
        {
          v85 = _mm_loadu_si128((const __m128i *)(v3 + 16));
          *(__m128i *)v84->m256i_i8 = _mm_loadu_si128((const __m128i *)v3);
          v86 = *(_BYTE *)(v3 + 138);
          *(__m128i *)&v84->m256i_u64[2] = v85;
          v87 = 32 * (v83 - 2);
          v88 = _mm_loadu_si128((const __m128i *)&v270);
          v275 = 64;
          v89 = _mm_loadu_si128((const __m128i *)&v270.m256i_u64[2]);
          v90 = v3 + (int)v87 + 145;
          v276 = v86 | 4;
          v91 = _mm_loadu_si128((const __m128i *)v90);
          v92 = _mm_loadu_si128((const __m128i *)(v90 + 16));
          *(_QWORD *)v271 = 0;
          v93 = _mm_loadu_si128((const __m128i *)(v90 + 32));
          v94 = _mm_loadu_si128((const __m128i *)(v90 + 48));
          v263 = v88;
          *(__m128i *)&v271[8] = v91;
          v95 = _mm_loadu_si128((const __m128i *)v271);
          v272 = v92;
          v96 = _mm_loadu_si128((const __m128i *)&v271[16]);
          v273 = v93;
          v97 = _mm_loadu_si128((const __m128i *)&v272.m128i_u64[1]);
          v274 = v94;
          v98 = _mm_loadu_si128((const __m128i *)&v273.m128i_u64[1]);
          v99 = _mm_loadu_si128((const __m128i *)&v274.m128i_u64[1]);
          v264 = v89;
          v265 = v95;
          v266 = v96;
          v267 = v97;
          v268 = v98;
          v269 = v99;
          sub_CC2280(v84, &v265.m128i_u64[1], 64, 0, v86 | 4u);
          v100 = v270.m256i_i16[1];
          v101 = v270.m256i_i8[1];
          *(_BYTE *)(v3 + v87 + 145) = v270.m256i_i8[0];
          *(_BYTE *)(v90 + 1) = v101;
          *(_WORD *)(v90 + 2) = v100;
          *(_OWORD *)(v90 + 4) = *(_OWORD *)((char *)v270.m256i_i64 + 4);
          *(_QWORD *)(v90 + 20) = *(__int64 *)((char *)&v270.m256i_i64[2] + 4);
          *(_DWORD *)(v90 + 28) = v270.m256i_i32[7];
          v83 = *(_BYTE *)(v3 + 144) - 1;
          *(_BYTE *)(v3 + 144) = v83;
        }
        while ( v239 < v83 );
        v59 = v84;
      }
      v102 = 32 * v83;
      *(_BYTE *)(v3 + v102 + 145) = v234;
      v103 = v3 + v102 + 145;
      *(_BYTE *)(v103 + 1) = v217;
      *(_BYTE *)(v103 + 2) = v215;
      *(_BYTE *)(v103 + 3) = v213;
      *(_OWORD *)(v103 + 4) = v227;
      *(_QWORD *)(v103 + 20) = v221;
      *(_DWORD *)(v103 + 28) = v219;
      ++*(_BYTE *)(v3 + 144);
      goto LABEL_43;
    }
    v240 = *(_BYTE *)(v3 + 138);
    v135 = sub_CBFA60((__int64)v224, i, (const __m128i *)v3, v58, v62, (__int64)v59);
    v228 = v240 | 4;
    if ( (unsigned __int64)(v135 - 3) > 0xD )
      goto LABEL_70;
    v136 = &v255;
    v230 = v3;
    v137 = v135;
    do
    {
      if ( v137 == 1 )
      {
        v142 = -2;
        v245 = v136;
        sub_CC2330((_DWORD)v136, 0, 1, v230, 0, 0, v228, 0, 0, (__int64)&v263);
        v190 = &v263;
        v191 = (const __m128i *)v59;
        v136 = v245;
        LODWORD(v141) = 32;
      }
      else
      {
        v138 = v59;
        v139 = v136;
        v140 = (v137 - 2) >> 1;
        do
        {
          v139->m128i_i64[0] = (__int64)v138;
          v139 = (__m128i *)((char *)v139 + 8);
          v138 += 2;
        }
        while ( &v255.m128i_u64[v140 + 1] != (unsigned __int64 *)v139 );
        v235 = v140 + 1;
        v241 = v136;
        sub_CC2330((_DWORD)v136, v140 + 1, 1, v230, 0, 0, v228, 0, 0, (__int64)&v263);
        v136 = v241;
        v141 = 32 * (v140 + 1);
        if ( v137 <= ((v137 - 2) & 0xFFFFFFFFFFFFFFFELL) + 2 )
        {
          v142 = v140 - 2;
          v137 = v235;
          goto LABEL_68;
        }
        v190 = (__m128i *)((char *)&v263 + v141);
        v137 = v140 + 2;
        LODWORD(v141) = v141 + 32;
        v191 = (const __m128i *)&v59[2 * v235];
        v142 = v140 - 1;
      }
      *v190 = _mm_loadu_si128(v191);
      v190[1] = _mm_loadu_si128(v191 + 1);
LABEL_68:
      *(__int64 *)((char *)&v59->m256i_i64[-1] + (unsigned int)v141) = *(_QWORD *)&v262[(unsigned int)v141];
      qmemcpy(v59, &v263, 8LL * ((unsigned int)(v141 - 1) >> 3));
    }
    while ( v142 <= 0xD );
    v3 = v230;
LABEL_70:
    v143 = _mm_loadu_si128((const __m128i *)&v270.m256i_u64[2]);
    v144 = _mm_loadu_si128((const __m128i *)v271);
    v145 = _mm_loadu_si128((const __m128i *)&v271[16]);
    v146 = *(_QWORD *)(v3 + 64);
    v255 = _mm_loadu_si128((const __m128i *)&v270);
    v256 = v143;
    v257 = v144;
    v258 = v145;
    v242 = (int)sub_39FAC40(v146);
    v147 = *(_BYTE *)(v3 + 144);
    if ( v242 < v147 )
    {
      v148 = v59;
      do
      {
        v149 = _mm_loadu_si128((const __m128i *)(v3 + 16));
        *(__m128i *)v148->m256i_i8 = _mm_loadu_si128((const __m128i *)v3);
        v150 = *(_BYTE *)(v3 + 138);
        *(__m128i *)&v148->m256i_u64[2] = v149;
        v151 = 32 * (v147 - 2);
        v152 = _mm_loadu_si128((const __m128i *)&v270);
        v275 = 64;
        v153 = _mm_loadu_si128((const __m128i *)&v270.m256i_u64[2]);
        v154 = v3 + (int)v151 + 145;
        v276 = v150 | 4;
        v155 = _mm_loadu_si128((const __m128i *)v154);
        v156 = _mm_loadu_si128((const __m128i *)(v154 + 16));
        *(_QWORD *)v271 = 0;
        v157 = _mm_loadu_si128((const __m128i *)(v154 + 32));
        v158 = _mm_loadu_si128((const __m128i *)(v154 + 48));
        v263 = v152;
        *(__m128i *)&v271[8] = v155;
        v159 = _mm_loadu_si128((const __m128i *)v271);
        v272 = v156;
        v160 = _mm_loadu_si128((const __m128i *)&v271[16]);
        v273 = v157;
        v161 = _mm_loadu_si128((const __m128i *)&v272.m128i_u64[1]);
        v274 = v158;
        v162 = _mm_loadu_si128((const __m128i *)&v273.m128i_u64[1]);
        v163 = _mm_loadu_si128((const __m128i *)&v274.m128i_u64[1]);
        v264 = v153;
        v265 = v159;
        v266 = v160;
        v267 = v161;
        v268 = v162;
        v269 = v163;
        sub_CC2280(v148, &v265.m128i_u64[1], 64, 0, v150 | 4u);
        v164 = v270.m256i_i16[1];
        v165 = v270.m256i_i8[1];
        *(_BYTE *)(v3 + v151 + 145) = v270.m256i_i8[0];
        *(_BYTE *)(v154 + 1) = v165;
        *(_WORD *)(v154 + 2) = v164;
        *(_OWORD *)(v154 + 4) = *(_OWORD *)((char *)v270.m256i_i64 + 4);
        *(_QWORD *)(v154 + 20) = *(__int64 *)((char *)&v270.m256i_i64[2] + 4);
        *(_DWORD *)(v154 + 28) = v270.m256i_i32[7];
        v147 = *(_BYTE *)(v3 + 144) - 1;
        *(_BYTE *)(v3 + 144) = v147;
      }
      while ( v242 < v147 );
      v59 = v148;
    }
    v166 = (__m128i *)(v3 + 32 * v147 + 145);
    v243 = _mm_loadu_si128(&v256);
    *v166 = _mm_loadu_si128(&v255);
    v166[1] = v243;
    LOBYTE(v166) = *(_BYTE *)(v3 + 144);
    v167 = *(_QWORD *)(v3 + 64) + (i >> 11);
    *(_BYTE *)(v3 + 144) = (_BYTE)v166 + 1;
    v236 = (_BYTE)v166 + 1;
    v244 = (int)sub_39FAC40(v167);
    v168 = v236;
    if ( v244 < v236 )
    {
      v169 = v59;
      do
      {
        v170 = _mm_loadu_si128((const __m128i *)(v3 + 16));
        *(__m128i *)v169->m256i_i8 = _mm_loadu_si128((const __m128i *)v3);
        v171 = *(_BYTE *)(v3 + 138);
        *(__m128i *)&v169->m256i_u64[2] = v170;
        v172 = 32 * (v168 - 2);
        v173 = _mm_loadu_si128((const __m128i *)&v270);
        v275 = 64;
        v174 = _mm_loadu_si128((const __m128i *)&v270.m256i_u64[2]);
        v175 = v3 + (int)v172 + 145;
        v276 = v171 | 4;
        v176 = _mm_loadu_si128((const __m128i *)v175);
        v177 = _mm_loadu_si128((const __m128i *)(v175 + 16));
        *(_QWORD *)v271 = 0;
        v178 = _mm_loadu_si128((const __m128i *)(v175 + 32));
        v179 = _mm_loadu_si128((const __m128i *)(v175 + 48));
        v263 = v173;
        *(__m128i *)&v271[8] = v176;
        v180 = _mm_loadu_si128((const __m128i *)v271);
        v272 = v177;
        v181 = _mm_loadu_si128((const __m128i *)&v271[16]);
        v273 = v178;
        v182 = _mm_loadu_si128((const __m128i *)&v272.m128i_u64[1]);
        v274 = v179;
        v183 = _mm_loadu_si128((const __m128i *)&v273.m128i_u64[1]);
        v184 = _mm_loadu_si128((const __m128i *)&v274.m128i_u64[1]);
        v264 = v174;
        v265 = v180;
        v266 = v181;
        v267 = v182;
        v268 = v183;
        v269 = v184;
        sub_CC2280(v169, &v265.m128i_u64[1], 64, 0, v171 | 4u);
        v185 = v270.m256i_i32[0];
        *(_BYTE *)(v3 + v172 + 145) = v270.m256i_i8[0];
        v186 = BYTE1(v185);
        *(_BYTE *)(v175 + 3) = HIBYTE(v185);
        *(_BYTE *)(v175 + 2) = BYTE2(v185);
        v187 = *(unsigned __int64 *)((char *)v270.m256i_u64 + 4);
        *(_BYTE *)(v175 + 1) = v186;
        *(_OWORD *)(v175 + 4) = __PAIR128__(*(unsigned __int64 *)((char *)&v270.m256i_u64[1] + 4), v187);
        *(_QWORD *)(v175 + 20) = *(__int64 *)((char *)&v270.m256i_i64[2] + 4);
        *(_DWORD *)(v175 + 28) = v270.m256i_i32[7];
        v168 = *(_BYTE *)(v3 + 144) - 1;
        *(_BYTE *)(v3 + 144) = v168;
      }
      while ( v244 < v168 );
      v59 = v169;
    }
    v188 = _mm_loadu_si128(&v258);
    v189 = (__m128i *)(v3 + 32 * v168 + 145);
    *v189 = _mm_loadu_si128(&v257);
    v189[1] = v188;
    ++*(_BYTE *)(v3 + 144);
LABEL_43:
    v231 -= i;
    result = v231;
    v58 = *(_QWORD *)(v3 + 64) + (i >> 10);
    v224 = (__int64 *)((char *)v224 + i);
    *(_QWORD *)(v3 + 64) = v58;
  }
  while ( v231 > 0x400 );
LABEL_44:
  if ( !v231 )
    return result;
  v104 = *(_BYTE *)(v3 + 136);
  v232 = v3 + 72;
  if ( !v104 )
  {
LABEL_47:
    if ( v231 <= 0x40 )
    {
      v110 = *(unsigned __int8 *)(v3 + 136);
      v108 = v224;
      v211 = 64 - v110;
      if ( v231 <= 64 - v110 )
        v211 = v231;
      LODWORD(v231) = v211;
      v112 = v211;
    }
    else
    {
      v105 = v224;
      v106 = *(_BYTE *)(v3 + 137);
      v107 = (v231 - 65) >> 6;
      v108 = &v224[8 * v107 + 8];
      do
      {
        v109 = v105;
        v105 += 8;
        sub_CC2280(v3 + 32, v109, 64, *(_QWORD *)(v3 + 64), (unsigned __int8)(*(_BYTE *)(v3 + 138) | (v106 == 0)));
        v106 = *(_BYTE *)(v3 + 137) + 1;
        *(_BYTE *)(v3 + 137) = v106;
      }
      while ( v105 != v108 );
      v110 = *(unsigned __int8 *)(v3 + 136);
      v111 = 64 - v110;
      if ( v231 - (v107 << 6) - 64 <= 64 - v110 )
        v111 = v231 - ((_DWORD)v107 << 6) - 64;
      LODWORD(v231) = v111;
      v112 = v111;
    }
    goto LABEL_53;
  }
  v209 = 64LL - v104;
  if ( v209 > v231 )
    v209 = v231;
  v210 = (__int64 *)(v232 + v104);
  if ( (unsigned int)v209 >= 8 )
  {
    *v210 = *v224;
    *(__int64 *)((char *)v210 + (unsigned int)v209 - 8) = *(__int64 *)((char *)v224 + (unsigned int)v209 - 8);
    qmemcpy(
      (void *)((unsigned __int64)(v210 + 1) & 0xFFFFFFFFFFFFFFF8LL),
      (const void *)((char *)v224 - ((char *)v210 - ((unsigned __int64)(v210 + 1) & 0xFFFFFFFFFFFFFFF8LL))),
      8LL * (((unsigned int)v209 + (_DWORD)v210 - (((_DWORD)v210 + 8) & 0xFFFFFFF8)) >> 3));
LABEL_110:
    v104 = *(_BYTE *)(v3 + 136);
    goto LABEL_107;
  }
  if ( (v209 & 4) != 0 )
  {
    *(_DWORD *)v210 = *(_DWORD *)v224;
    *(_DWORD *)((char *)v210 + (unsigned int)v209 - 4) = *(_DWORD *)((char *)v224 + (unsigned int)v209 - 4);
    v104 = *(_BYTE *)(v3 + 136);
    goto LABEL_107;
  }
  if ( (_DWORD)v209 )
  {
    *(_BYTE *)v210 = *(_BYTE *)v224;
    if ( (v209 & 2) != 0 )
    {
      *(_WORD *)((char *)v210 + (unsigned int)v209 - 2) = *(_WORD *)((char *)v224 + (unsigned int)v209 - 2);
      v104 = *(_BYTE *)(v3 + 136);
      goto LABEL_107;
    }
    goto LABEL_110;
  }
LABEL_107:
  LOBYTE(v110) = v104 + v209;
  *(_BYTE *)(v3 + 136) = v104 + v209;
  v108 = (__int64 *)((char *)v224 + v209);
  v231 -= v209;
  if ( v231 )
  {
    sub_CC2280(
      v3 + 32,
      v232,
      64,
      *(_QWORD *)(v3 + 64),
      (unsigned __int8)(*(_BYTE *)(v3 + 138) | (*(_BYTE *)(v3 + 137) == 0)));
    ++*(_BYTE *)(v3 + 137);
    *(_BYTE *)(v3 + 136) = 0;
    *(_OWORD *)(v3 + 72) = 0;
    v224 = v108;
    *(_OWORD *)(v3 + 88) = 0;
    *(_OWORD *)(v3 + 104) = 0;
    *(_OWORD *)(v3 + 120) = 0;
    goto LABEL_47;
  }
  v110 = (unsigned __int8)v110;
  v112 = 0;
LABEL_53:
  v113 = (__int64 *)(v110 + v232);
  if ( (unsigned int)v231 >= 8 )
  {
    v204 = (unsigned __int64)(v113 + 1) & 0xFFFFFFFFFFFFFFF8LL;
    *v113 = *v108;
    *(__int64 *)((char *)v113 + (unsigned int)v231 - 8) = *(__int64 *)((char *)v108 + (unsigned int)v231 - 8);
    v205 = (char *)v108 - ((char *)v113 - v204);
    if ( (((_DWORD)v231 + (_DWORD)v113 - (_DWORD)v204) & 0xFFFFFFF8) >= 8 )
    {
      v206 = 0;
      do
      {
        v207 = v206;
        v206 += 8;
        *(_QWORD *)(v204 + v207) = *(_QWORD *)(v205 + v207);
      }
      while ( v206 < (((_DWORD)v231 + (_DWORD)v113 - (_DWORD)v204) & 0xFFFFFFF8) );
    }
  }
  else if ( (v231 & 4) != 0 )
  {
    *(_DWORD *)v113 = *(_DWORD *)v108;
    *(_DWORD *)((char *)v113 + (unsigned int)v231 - 4) = *(_DWORD *)((char *)v108 + (unsigned int)v231 - 4);
  }
  else if ( (_DWORD)v231 )
  {
    *(_BYTE *)v113 = *(_BYTE *)v108;
    if ( (v231 & 2) != 0 )
      *(_WORD *)((char *)v113 + (unsigned int)v231 - 2) = *(_WORD *)((char *)v108 + (unsigned int)v231 - 2);
  }
  *(_BYTE *)(v3 + 136) += v112;
  v114 = (int)sub_39FAC40(*(_QWORD *)(v3 + 64));
  result = *(unsigned __int8 *)(v3 + 144);
  if ( result > v114 )
  {
    do
    {
      v115 = _mm_loadu_si128((const __m128i *)(v3 + 16));
      v116 = *(_BYTE *)(v3 + 138);
      *(__m128i *)v270.m256i_i8 = _mm_loadu_si128((const __m128i *)v3);
      *(__m128i *)&v270.m256i_u64[2] = v115;
      v117 = 32 * ((unsigned __int8)result - 2);
      v118 = _mm_loadu_si128((const __m128i *)&v270);
      v275 = 64;
      v119 = _mm_loadu_si128((const __m128i *)&v270.m256i_u64[2]);
      v120 = v3 + (int)v117 + 145;
      v276 = v116 | 4;
      v121 = _mm_loadu_si128((const __m128i *)v120);
      v122 = _mm_loadu_si128((const __m128i *)(v120 + 16));
      *(_QWORD *)v271 = 0;
      v123 = _mm_loadu_si128((const __m128i *)(v120 + 32));
      v124 = _mm_loadu_si128((const __m128i *)(v120 + 48));
      v263 = v118;
      *(__m128i *)&v271[8] = v121;
      v125 = _mm_loadu_si128((const __m128i *)v271);
      v272 = v122;
      v126 = _mm_loadu_si128((const __m128i *)&v271[16]);
      v273 = v123;
      v127 = _mm_loadu_si128((const __m128i *)&v272.m128i_u64[1]);
      v274 = v124;
      v128 = _mm_loadu_si128((const __m128i *)&v273.m128i_u64[1]);
      v129 = _mm_loadu_si128((const __m128i *)&v274.m128i_u64[1]);
      v264 = v119;
      v265 = v125;
      v266 = v126;
      v267 = v127;
      v268 = v128;
      v269 = v129;
      sub_CC2280(&v270, &v265.m128i_u64[1], 64, 0, v116 | 4u);
      v130 = v270.m256i_i32[0];
      *(_BYTE *)(v3 + v117 + 145) = v270.m256i_i8[0];
      v131 = BYTE1(v130);
      v132 = BYTE2(v130);
      *(_BYTE *)(v120 + 3) = HIBYTE(v130);
      v133 = *(__int64 *)((char *)v270.m256i_i64 + 4);
      *(_BYTE *)(v120 + 2) = v132;
      *(_QWORD *)(v120 + 4) = v133;
      v134 = *(unsigned __int64 *)((char *)&v270.m256i_u64[1] + 4);
      *(_BYTE *)(v120 + 1) = v131;
      *(_OWORD *)(v120 + 12) = __PAIR128__(*(unsigned __int64 *)((char *)&v270.m256i_u64[2] + 4), v134);
      *(_DWORD *)(v120 + 28) = v270.m256i_i32[7];
      result = (unsigned int)(unsigned __int8)(*(_BYTE *)(v3 + 144))-- - 1;
    }
    while ( v114 < (unsigned __int8)result );
  }
  return result;
}
