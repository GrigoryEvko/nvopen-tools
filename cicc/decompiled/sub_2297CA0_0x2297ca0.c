// Function: sub_2297CA0
// Address: 0x2297ca0
//
__int64 *__fastcall sub_2297CA0(__int64 *a1, __int64 a2, __int64 a3, _BYTE *a4)
{
  __int64 v5; // r14
  _BYTE *v7; // r12
  __int64 v9; // rbx
  __int64 v10; // rdi
  __int64 v11; // rax
  __m128i v12; // xmm6
  __m128i v13; // xmm7
  __m128i v14; // xmm0
  __m128i v15; // xmm1
  __m128i *v16; // rax
  __int64 *v17; // rax
  __int64 v18; // rdi
  __int64 *v19; // rbx
  __int64 v20; // r8
  __int64 v21; // r9
  _QWORD *v22; // rax
  unsigned __int8 *v23; // r13
  unsigned __int64 v24; // r12
  unsigned __int64 v25; // r14
  unsigned __int64 v26; // r14
  unsigned __int64 v27; // r14
  _QWORD *v28; // rax
  unsigned __int8 *v29; // rax
  unsigned __int8 *v30; // r9
  __int64 *v31; // rcx
  unsigned __int8 *v32; // rax
  __int64 v33; // r12
  __int64 v34; // r15
  unsigned __int8 *v35; // rdi
  __int64 v36; // rbx
  __int64 v37; // rcx
  __int64 v38; // r8
  __int64 v39; // r9
  __int64 v40; // rcx
  __int64 v41; // r8
  __int64 v42; // r9
  __int64 v43; // rax
  int v44; // edx
  __int64 *v45; // rsi
  __int64 v46; // r10
  __int64 v47; // rdi
  int v48; // edx
  unsigned int v49; // ecx
  __int64 *v50; // rax
  __int64 v51; // r11
  _QWORD *v52; // r8
  __int64 v53; // r11
  __int64 v54; // rdi
  unsigned int v55; // ecx
  __int64 *v56; // rax
  __int64 v57; // r13
  _QWORD *v58; // rdx
  int v59; // eax
  __int64 v60; // rdx
  __int64 v61; // rcx
  __int64 v62; // r8
  __int64 v63; // r9
  unsigned __int8 *v64; // r13
  unsigned __int64 v65; // r15
  __int64 v66; // rax
  __int64 v67; // rax
  __int64 v68; // r8
  __int64 v69; // r9
  __int64 v70; // r8
  __int64 v71; // r9
  __int64 v72; // r14
  __int64 v73; // r12
  unsigned __int8 *v74; // rax
  unsigned __int64 *v75; // rbx
  int v76; // edx
  __int64 v77; // rcx
  __int64 v78; // rbx
  __int64 v79; // rdx
  __int64 v80; // r15
  unsigned __int64 v81; // rcx
  __int64 v82; // rdx
  bool v83; // al
  __int64 v84; // r14
  __int64 v85; // r8
  __int64 v86; // r9
  _QWORD *v87; // rax
  __int64 v88; // rax
  __int64 v89; // rdx
  __int64 v90; // rcx
  __int64 v91; // r8
  __int64 v92; // r9
  _QWORD **v93; // rax
  unsigned __int8 *v94; // rbx
  _QWORD **v95; // rax
  int v96; // eax
  __int64 v97; // r8
  unsigned __int8 *v98; // rax
  unsigned int v99; // edx
  int v100; // eax
  int v101; // eax
  __int64 v102; // r9
  int v103; // eax
  unsigned __int64 v104; // r8
  __m128i *v105; // rax
  __int32 v106; // ebx
  __int64 v107; // rcx
  __int64 v108; // rdx
  unsigned int v109; // ebx
  __int64 v110; // rdx
  int v111; // eax
  const __m128i *v112; // r13
  _QWORD *v113; // r12
  unsigned __int8 *v114; // rax
  __int64 v115; // r15
  unsigned int v116; // esi
  __int64 v117; // r8
  __int64 v118; // r9
  __int64 v119; // rax
  __int64 v120; // r8
  unsigned __int64 v121; // rdx
  unsigned __int64 *v122; // r10
  unsigned __int8 *v123; // rax
  __int64 v124; // rbx
  __int64 v125; // rcx
  __int64 v126; // r8
  __int64 v127; // r9
  __int64 v128; // rdx
  __int64 v129; // rax
  __int64 v130; // r12
  __int64 v131; // r15
  unsigned __int8 *v132; // rsi
  unsigned int v133; // edx
  unsigned __int64 v134; // rax
  unsigned int v135; // eax
  __int64 v136; // r15
  unsigned __int64 v137; // r11
  unsigned int v138; // r12d
  char v139; // al
  unsigned int v140; // r12d
  __int64 v141; // r13
  _QWORD *v142; // rax
  unsigned int v143; // eax
  __int64 *v144; // r10
  int v145; // ebx
  __int64 v146; // rax
  unsigned __int64 v147; // rdx
  _QWORD *v148; // rax
  _QWORD *v149; // rcx
  __int64 v150; // r8
  unsigned int v151; // r15d
  __int64 *v152; // rax
  int v153; // ebx
  _QWORD *v154; // rax
  __int64 v155; // rdx
  __int64 v156; // rcx
  __int64 v157; // r8
  __int64 v158; // r9
  _QWORD *v159; // rbx
  int v160; // r15d
  int v161; // r8d
  __int64 *v162; // [rsp+8h] [rbp-4F8h]
  __int64 v163; // [rsp+10h] [rbp-4F0h]
  _BYTE *v164; // [rsp+20h] [rbp-4E0h]
  _BYTE *v165; // [rsp+20h] [rbp-4E0h]
  _BYTE *v166; // [rsp+28h] [rbp-4D8h]
  _BYTE *v167; // [rsp+28h] [rbp-4D8h]
  __int64 v168; // [rsp+30h] [rbp-4D0h]
  __int64 v169; // [rsp+48h] [rbp-4B8h]
  __int64 v170; // [rsp+50h] [rbp-4B0h]
  __int64 *v171; // [rsp+50h] [rbp-4B0h]
  __int64 *v172; // [rsp+58h] [rbp-4A8h]
  unsigned __int8 *v173; // [rsp+60h] [rbp-4A0h]
  unsigned __int8 *v174; // [rsp+60h] [rbp-4A0h]
  unsigned __int8 *v175; // [rsp+60h] [rbp-4A0h]
  __int64 v176; // [rsp+60h] [rbp-4A0h]
  __int64 v177; // [rsp+60h] [rbp-4A0h]
  __int64 v178; // [rsp+70h] [rbp-490h]
  _BYTE *v179; // [rsp+70h] [rbp-490h]
  bool v180; // [rsp+70h] [rbp-490h]
  _QWORD *v181; // [rsp+70h] [rbp-490h]
  __int64 v182; // [rsp+70h] [rbp-490h]
  __int64 v183; // [rsp+78h] [rbp-488h]
  _BYTE *v184; // [rsp+78h] [rbp-488h]
  unsigned int v185; // [rsp+78h] [rbp-488h]
  __int64 v186; // [rsp+88h] [rbp-478h]
  unsigned int v187; // [rsp+88h] [rbp-478h]
  __int64 v188; // [rsp+88h] [rbp-478h]
  char v189; // [rsp+88h] [rbp-478h]
  const __m128i *v190; // [rsp+88h] [rbp-478h]
  unsigned __int8 *v191; // [rsp+90h] [rbp-470h]
  char v192; // [rsp+90h] [rbp-470h]
  __int64 *v193; // [rsp+90h] [rbp-470h]
  unsigned int v194; // [rsp+90h] [rbp-470h]
  _BYTE *v195; // [rsp+90h] [rbp-470h]
  bool v196; // [rsp+98h] [rbp-468h]
  __int64 *v197; // [rsp+98h] [rbp-468h]
  unsigned __int64 v198; // [rsp+98h] [rbp-468h]
  unsigned int v199; // [rsp+ACh] [rbp-454h] BYREF
  unsigned __int64 *v200; // [rsp+B0h] [rbp-450h] BYREF
  unsigned __int64 *v201; // [rsp+B8h] [rbp-448h] BYREF
  unsigned __int64 v202; // [rsp+C0h] [rbp-440h] BYREF
  unsigned __int64 *v203; // [rsp+C8h] [rbp-438h] BYREF
  unsigned __int64 *v204; // [rsp+D0h] [rbp-430h] BYREF
  unsigned __int64 *v205; // [rsp+D8h] [rbp-428h] BYREF
  unsigned __int64 **i; // [rsp+E0h] [rbp-420h] BYREF
  int v207; // [rsp+E8h] [rbp-418h]
  unsigned __int64 **v208; // [rsp+F0h] [rbp-410h] BYREF
  int v209; // [rsp+F8h] [rbp-408h]
  unsigned __int64 **j; // [rsp+100h] [rbp-400h] BYREF
  unsigned int v211; // [rsp+108h] [rbp-3F8h]
  __m128i v212; // [rsp+110h] [rbp-3F0h] BYREF
  __m128i v213; // [rsp+120h] [rbp-3E0h] BYREF
  __m128i v214; // [rsp+130h] [rbp-3D0h] BYREF
  __m128i v215; // [rsp+140h] [rbp-3C0h] BYREF
  __m128i v216; // [rsp+150h] [rbp-3B0h] BYREF
  __m128i v217; // [rsp+160h] [rbp-3A0h] BYREF
  void *v218; // [rsp+170h] [rbp-390h] BYREF
  __int64 v219; // [rsp+178h] [rbp-388h]
  __m128i v220; // [rsp+180h] [rbp-380h]
  __m128i v221; // [rsp+190h] [rbp-370h] BYREF
  unsigned __int64 v222; // [rsp+1A0h] [rbp-360h]
  unsigned __int8 *v223; // [rsp+1B0h] [rbp-350h] BYREF
  __int64 v224; // [rsp+1B8h] [rbp-348h]
  __m128i v225; // [rsp+1C0h] [rbp-340h] BYREF
  __m128i v226; // [rsp+1D0h] [rbp-330h]
  __int64 v227; // [rsp+1E0h] [rbp-320h]
  __int64 v228; // [rsp+1E8h] [rbp-318h]
  __m128i v229; // [rsp+220h] [rbp-2E0h] BYREF
  __m128i v230; // [rsp+230h] [rbp-2D0h] BYREF
  __m128i v231; // [rsp+240h] [rbp-2C0h] BYREF
  _QWORD v232[2]; // [rsp+380h] [rbp-180h] BYREF
  char v233; // [rsp+390h] [rbp-170h]
  _BYTE *v234; // [rsp+398h] [rbp-168h]
  __int64 v235; // [rsp+3A0h] [rbp-160h]
  _BYTE v236[136]; // [rsp+3A8h] [rbp-158h] BYREF
  _QWORD v237[2]; // [rsp+430h] [rbp-D0h] BYREF
  __int64 v238; // [rsp+440h] [rbp-C0h]
  __int64 v239; // [rsp+448h] [rbp-B8h] BYREF
  unsigned int v240; // [rsp+450h] [rbp-B0h]
  char v241; // [rsp+4C8h] [rbp-38h] BYREF

  v5 = a2;
  v7 = (_BYTE *)a3;
  if ( !(unsigned __int8)sub_B46420(a3) && !(unsigned __int8)sub_B46490((__int64)v7)
    || !(unsigned __int8)sub_B46420((__int64)a4) && !(unsigned __int8)sub_B46490((__int64)a4) )
  {
LABEL_5:
    *a1 = 0;
    return a1;
  }
  if ( !sub_228AA50((__int64)v7) )
    goto LABEL_49;
  v196 = sub_228AA50((__int64)a4);
  if ( !v196 )
    goto LABEL_49;
  v183 = sub_228AED0(v7);
  v178 = sub_228AED0(a4);
  sub_D66840(&v229, v7);
  v215 = _mm_loadu_si128(&v229);
  v216 = _mm_loadu_si128(&v230);
  v191 = (unsigned __int8 *)v229.m128i_i64[0];
  v217 = _mm_loadu_si128(&v231);
  sub_D66840(&v229, a4);
  v9 = v229.m128i_i64[0];
  v10 = *(_QWORD *)(a2 + 24);
  v173 = (unsigned __int8 *)v229.m128i_i64[0];
  v212 = _mm_loadu_si128(&v229);
  v213 = _mm_loadu_si128(&v230);
  v214 = _mm_loadu_si128(&v231);
  sub_B2BEC0(v10);
  v218 = (void *)v9;
  v11 = *(_QWORD *)a2;
  v12 = _mm_loadu_si128(&v213);
  v219 = -1;
  v13 = _mm_loadu_si128(&v214);
  v14 = _mm_loadu_si128(&v216);
  v224 = -1;
  v15 = _mm_loadu_si128(&v217);
  v223 = v191;
  v230.m128i_i64[0] = 0;
  v230.m128i_i64[1] = 1;
  v229.m128i_i64[0] = v11;
  v229.m128i_i64[1] = v11;
  v16 = &v231;
  v220 = v12;
  v221 = v13;
  v225 = v14;
  v226 = v15;
  do
  {
    v16->m128i_i64[0] = -4;
    v16 = (__m128i *)((char *)v16 + 40);
    v16[-2].m128i_i64[0] = -3;
    v16[-2].m128i_i64[1] = -4;
    v16[-1].m128i_i64[0] = -3;
  }
  while ( v16 != (__m128i *)v232 );
  v232[1] = 0;
  v234 = v236;
  v235 = 0x400000000LL;
  v232[0] = v237;
  v233 = 0;
  v236[129] = 1;
  v237[1] = 0;
  v238 = 1;
  v237[0] = &unk_49DDBE8;
  v17 = &v239;
  do
  {
    *v17 = -4096;
    v17 += 2;
  }
  while ( v17 != (__int64 *)&v241 );
  v236[128] = 1;
  if ( !(unsigned __int8)sub_CF4D50(v229.m128i_i64[0], (__int64)&v218, (__int64)&v223, (__int64)&v229.m128i_i64[1], 0) )
    goto LABEL_14;
  v174 = sub_98ACB0(v173, 6u);
  v29 = sub_98ACB0(v191, 6u);
  v30 = v174;
  v192 = 3;
  v175 = v29;
  if ( v30 != v29 )
  {
    if ( (unsigned __int8)sub_CF7060(v30) && (unsigned __int8)sub_CF7060(v175) )
    {
LABEL_14:
      v192 = 0;
      goto LABEL_15;
    }
    v192 = 1;
  }
LABEL_15:
  v237[0] = &unk_49DDBE8;
  if ( (v238 & 1) == 0 )
    sub_C7D6A0(v239, 16LL * v240, 8);
  nullsub_184();
  if ( v234 != v236 )
    _libc_free((unsigned __int64)v234);
  if ( (v230.m128i_i8[8] & 1) == 0 )
    sub_C7D6A0(v231.m128i_i64[0], 40LL * v231.m128i_u32[2], 8);
  if ( !v192 )
    goto LABEL_5;
  if ( v192 == 3 )
  {
    sub_228D4A0(a2, (__int64)v7, (__int64)a4);
    sub_228CD10((__int64)&v218, (__int64)v7, (__int64)a4, v7 != a4, *(_DWORD *)(a2 + 32));
    v18 = *(_QWORD *)(a2 + 8);
    v223 = (unsigned __int8 *)&v225;
    v226.m128i_i64[1] = 1;
    v227 = 1;
    v228 = 1;
    v224 = 0x200000001LL;
    v225 = 0u;
    v226.m128i_i32[0] = 0;
    v193 = sub_DD8400(v18, v183);
    v19 = sub_DD8400(*(_QWORD *)(a2 + 8), v178);
    v186 = sub_D97190(*(_QWORD *)(a2 + 8), (__int64)v193);
    if ( v186 != sub_D97190(*(_QWORD *)(a2 + 8), (__int64)v19) )
    {
      v22 = (_QWORD *)sub_22077B0(0x28u);
      if ( v22 )
      {
        v22[1] = v7;
        v22[2] = a4;
        v22[3] = 0;
        *v22 = &unk_4A08DD0;
        v22[4] = 0;
      }
      *a1 = (__int64)v22;
      goto LABEL_27;
    }
    v31 = v193;
    v194 = 1;
    *(_QWORD *)v223 = v31;
    v32 = v223;
    *((_QWORD *)v223 + 1) = v19;
    if ( !(_BYTE)qword_4FDB5A8 )
      goto LABEL_56;
    if ( !(unsigned __int8)sub_2295EA0(a2, (__int64)v7, (__int64)a4, (__int64)&v223) || (v194 = v224) != 0 )
    {
      v32 = v223;
LABEL_56:
      v179 = v7;
      v172 = a1;
      v33 = 0;
      v184 = a4;
      while ( 1 )
      {
        v187 = v33;
        v36 = 48 * v33;
        sub_228BF90(
          (unsigned __int64 *)&v32[48 * v33 + 24],
          *(_DWORD *)(v5 + 40) + 1,
          0,
          *(unsigned int *)(v5 + 40),
          v20,
          v21);
        sub_228BF90((unsigned __int64 *)&v223[48 * v33 + 32], *(_DWORD *)(v5 + 40) + 1, 0, v37, v38, v39);
        sub_228BF90((unsigned __int64 *)&v223[48 * v33 + 40], v194, 0, v40, v41, v42);
        sub_228D9F0(v5, (__int64 *)&v223[48 * v33]);
        v43 = *(_QWORD *)(v5 + 16);
        v44 = *(_DWORD *)(v43 + 24);
        v45 = (__int64 *)&v223[48 * v33];
        v46 = *(_QWORD *)(v43 + 8);
        v47 = *((_QWORD *)v184 + 5);
        if ( v44 )
        {
          v48 = v44 - 1;
          v49 = v48 & (((unsigned int)v47 >> 9) ^ ((unsigned int)v47 >> 4));
          v50 = (__int64 *)(v46 + 16LL * v49);
          v51 = *v50;
          if ( v47 == *v50 )
          {
LABEL_63:
            v52 = (_QWORD *)v50[1];
          }
          else
          {
            v100 = 1;
            while ( v51 != -4096 )
            {
              v161 = v100 + 1;
              v49 = v48 & (v100 + v49);
              v50 = (__int64 *)(v46 + 16LL * v49);
              v51 = *v50;
              if ( v47 == *v50 )
                goto LABEL_63;
              v100 = v161;
            }
            v52 = 0;
          }
          v53 = v45[1];
          v54 = *((_QWORD *)v179 + 5);
          v55 = v48 & (((unsigned int)v54 >> 9) ^ ((unsigned int)v54 >> 4));
          v56 = (__int64 *)(v46 + 16LL * v55);
          v57 = *v56;
          if ( v54 == *v56 )
          {
LABEL_65:
            v58 = (_QWORD *)v56[1];
          }
          else
          {
            v101 = 1;
            while ( v57 != -4096 )
            {
              v160 = v101 + 1;
              v55 = v48 & (v101 + v55);
              v56 = (__int64 *)(v46 + 16LL * v55);
              v57 = *v56;
              if ( v54 == *v56 )
                goto LABEL_65;
              v101 = v160;
            }
            v58 = 0;
          }
        }
        else
        {
          v53 = v45[1];
          v52 = 0;
          v58 = 0;
        }
        v59 = sub_228DB90(v5, *v45, v58, v53, v52, (__int64 *)&v223[48 * v33 + 24]);
        v64 = &v223[v36];
        v65 = *(_QWORD *)&v223[v36 + 32];
        *(_DWORD *)&v223[v36 + 16] = v59;
        if ( (v65 & 1) != 0 )
        {
          v34 = *((_QWORD *)v64 + 3);
          if ( (v34 & 1) != 0 )
          {
            *((_QWORD *)v64 + 4) = v34;
            v35 = v64;
          }
          else
          {
            v67 = sub_22077B0(0x48u);
            if ( v67 )
            {
              *(_QWORD *)v67 = v67 + 16;
              *(_QWORD *)(v67 + 8) = 0x600000000LL;
              if ( *(_DWORD *)(v34 + 8) )
              {
                v177 = v67;
                sub_228AB60(v67, v34, v67 + 16, 0x600000000LL, v68, v69);
                v67 = v177;
              }
              *(_DWORD *)(v67 + 64) = *(_DWORD *)(v34 + 64);
            }
            *((_QWORD *)v64 + 4) = v67;
            v35 = &v223[v36];
          }
        }
        else
        {
          v66 = *((_QWORD *)v64 + 3);
          if ( (v66 & 1) != 0 )
          {
            v35 = v64;
            if ( v65 )
            {
              if ( *(_QWORD *)v65 != v65 + 16 )
                _libc_free(*(_QWORD *)v65);
              j_j___libc_free_0(v65);
              v66 = *((_QWORD *)v64 + 3);
              v35 = &v223[v36];
            }
            *((_QWORD *)v64 + 4) = v66;
          }
          else
          {
            v176 = *((_QWORD *)v64 + 3);
            sub_228AB60(v65, v176, v60, v61, v62, v63);
            *(_DWORD *)(v65 + 64) = *(_DWORD *)(v176 + 64);
            v35 = &v223[v36];
          }
        }
        ++v33;
        sub_228AC40((__int64 *)v35 + 5, v187);
        if ( v194 <= (unsigned int)v33 )
          break;
        v32 = v223;
      }
      sub_B48880((__int64 *)&v200, v194, 0);
      sub_B48880((__int64 *)&v201, v194, 0);
      v168 = v5;
      v72 = 1;
      v166 = v179;
      v73 = 0;
      v164 = v184;
      while ( 1 )
      {
        v185 = v72 - 1;
        v74 = v223;
        v75 = (unsigned __int64 *)&v223[v73];
        v76 = *(_DWORD *)&v223[v73 + 16];
        if ( v76 == 4 )
        {
          v93 = (_QWORD **)sub_D48290(*(_QWORD *)(v168 + 16), *((_QWORD *)v166 + 5));
          sub_228D790(v168, *v75, v93, v75 + 3);
          v94 = &v223[v73];
          v95 = (_QWORD **)sub_D48290(*(_QWORD *)(v168 + 16), *((_QWORD *)v164 + 5));
          sub_228D790(v168, *((_QWORD *)v94 + 1), v95, (unsigned __int64 *)v94 + 3);
          v221.m128i_i8[11] = 0;
          v188 = v73 + 48;
        }
        else if ( v76 )
        {
          if ( v194 <= (unsigned int)v72 )
          {
            v188 = v73 + 48;
          }
          else
          {
            v77 = v196;
            v78 = v73 + 48;
            v170 = v72;
            v188 = v73 + 48;
            v180 = v196;
            v79 = v72 + v194 - 1 - (unsigned int)v72;
            v80 = 48 * v79;
            while ( 1 )
            {
              v229.m128i_i64[0] = 1;
              v84 = *(_QWORD *)&v74[v73 + 32];
              if ( (v84 & 1) != 0 )
              {
                v229.m128i_i64[0] = *(_QWORD *)&v74[v73 + 32];
              }
              else
              {
                v88 = sub_22077B0(0x48u);
                if ( v88 )
                {
                  *(_QWORD *)v88 = v88 + 16;
                  v77 = 0x600000000LL;
                  *(_QWORD *)(v88 + 8) = 0x600000000LL;
                  if ( *(_DWORD *)(v84 + 8) )
                  {
                    v169 = v88;
                    sub_228AB60(v88, v84, v79, 0x600000000LL, v70, v71);
                    v88 = v169;
                  }
                  v79 = *(unsigned int *)(v84 + 64);
                  *(_DWORD *)(v88 + 64) = v79;
                }
                v229.m128i_i64[0] = v88;
                v74 = v223;
              }
              sub_228C5E0((unsigned __int64 *)&v229, (unsigned __int64 *)&v74[v78 + 32], v79, v77, v70, v71);
              if ( (v229.m128i_i8[0] & 1) != 0 )
              {
                v81 = (unsigned __int64)v229.m128i_i64[0] >> 58;
                v82 = ~(-1LL << ((unsigned __int64)v229.m128i_i64[0] >> 58));
                v83 = (((unsigned __int64)v229.m128i_i64[0] >> 1) & v82) != 0;
              }
              else
              {
                v87 = sub_228AAA0(
                        *(_QWORD **)v229.m128i_i64[0],
                        *(_QWORD *)v229.m128i_i64[0] + 8LL * *(unsigned int *)(v229.m128i_i64[0] + 8));
                v83 = v81 != (_QWORD)v87;
              }
              if ( v83 )
              {
                sub_228C270(
                  (unsigned __int64 *)&v223[v78 + 32],
                  (unsigned __int64 *)&v223[v73 + 32],
                  v82,
                  v81,
                  v85,
                  v86);
                sub_228C270(
                  (unsigned __int64 *)&v223[v78 + 40],
                  (unsigned __int64 *)&v223[v73 + 40],
                  v89,
                  v90,
                  v91,
                  v92);
                v180 = 0;
              }
              sub_228BF40((unsigned __int64 **)&v229);
              if ( v80 == v78 )
                break;
              v74 = v223;
              v78 += 48;
            }
            v72 = v170;
            if ( !v180 )
              goto LABEL_82;
            v75 = (unsigned __int64 *)&v223[v73];
          }
          if ( sub_228ACA0(v75[5]) == 1 )
            sub_228AC40((__int64 *)&v200, v185);
          else
            sub_228AC40((__int64 *)&v201, v185);
        }
        else
        {
          sub_228AC40((__int64 *)&v200, v185);
          v188 = v73 + 48;
        }
LABEL_82:
        v73 = v188;
        if ( v194 - 1 + 2LL == ++v72 )
        {
          a1 = v172;
          v5 = v168;
          v7 = v166;
          a4 = v164;
          goto LABEL_107;
        }
      }
    }
    sub_B48880((__int64 *)&v200, 0, 0);
    sub_B48880((__int64 *)&v201, 0, 0);
LABEL_107:
    sub_228CEF0((__int64)&v212, *(_QWORD *)(v5 + 8));
    v96 = sub_228AE30((unsigned __int64)v200);
    v207 = v96;
    for ( i = &v200; v207 != -1; v96 = v207 )
    {
      v98 = &v223[48 * v96];
      v99 = *((_DWORD *)v98 + 4);
      if ( v99 == 2 )
      {
        if ( sub_2295B30(v5, *(_QWORD *)v98, *((_QWORD *)v98 + 1), (__int64)&v218, v97) )
          goto LABEL_115;
      }
      else if ( v99 > 2 )
      {
        if ( v99 != 3 )
LABEL_254:
          BUG();
        if ( (unsigned __int8)sub_2291E30(
                                v5,
                                *(_QWORD *)v98,
                                *((_QWORD *)v98 + 1),
                                (unsigned __int64 *)v98 + 3,
                                (__int64)&v218) )
        {
LABEL_115:
          *a1 = 0;
          goto LABEL_116;
        }
      }
      else if ( v99 )
      {
        v229.m128i_i64[0] = 0;
        if ( sub_2294A30(
               v5,
               *(_QWORD *)v98,
               *((_QWORD *)v98 + 1),
               v215.m128i_i32,
               (__int64)&v218,
               (__int64)&v212,
               v229.m128i_i64) )
        {
          goto LABEL_115;
        }
      }
      else if ( sub_228EC60(v5, *(_QWORD *)v98, *((_QWORD *)v98 + 1), (__int64)&v218) )
      {
        goto LABEL_115;
      }
      sub_2293510((__int64)&i);
    }
    if ( sub_228ACA0((unsigned __int64)v201) )
    {
      v103 = *(_DWORD *)(v5 + 40);
      v229.m128i_i64[1] = 0x400000000LL;
      v104 = (unsigned int)(v103 + 1);
      v105 = &v230;
      v106 = v104;
      v229.m128i_i64[0] = (__int64)&v230;
      if ( v104 )
      {
        if ( v104 > 4 )
        {
          v198 = v104;
          sub_C8D5F0((__int64)&v229, &v230, v104, 0x30u, v104, v102);
          v104 = v198;
        }
        v105 = (__m128i *)v229.m128i_i64[0];
        v107 = v229.m128i_i64[0] + 48 * v104;
        v108 = v229.m128i_i64[0] + 48LL * v229.m128i_u32[2];
        if ( v108 != v107 )
        {
          do
          {
            if ( v108 )
            {
              *(_DWORD *)v108 = 0;
              *(_QWORD *)(v108 + 8) = 0;
              *(_QWORD *)(v108 + 16) = 0;
              *(_QWORD *)(v108 + 24) = 0;
              *(_QWORD *)(v108 + 32) = 0;
              *(_QWORD *)(v108 + 40) = 0;
            }
            v108 += 48;
          }
          while ( v107 != v108 );
          v105 = (__m128i *)v229.m128i_i64[0];
        }
        v229.m128i_i32[2] = v106;
      }
      v109 = 0;
      while ( 1 )
      {
        v110 = v109++;
        sub_228CEF0((__int64)v105[3 * v110].m128i_i64, *(_QWORD *)(v5 + 8));
        if ( *(_DWORD *)(v5 + 40) < v109 )
          break;
        v105 = (__m128i *)v229.m128i_i64[0];
      }
      v111 = sub_228AE30((unsigned __int64)v201);
      v209 = v111;
      v208 = &v201;
      if ( v111 != -1 )
      {
        v171 = a1;
        v165 = a4;
        v112 = &v212;
        v167 = v7;
        v113 = &j;
LABEL_143:
        v202 = 1;
        v114 = &v223[48 * v111];
        v115 = *((_QWORD *)v114 + 5);
        if ( (v115 & 1) != 0 )
        {
          v202 = *((_QWORD *)v114 + 5);
        }
        else
        {
          v154 = (_QWORD *)sub_22077B0(0x48u);
          v159 = v154;
          if ( v154 )
          {
            *v154 = v154 + 2;
            v154[1] = 0x600000000LL;
            if ( *(_DWORD *)(v115 + 8) )
              sub_228AB60((__int64)v154, v115, v155, v156, v157, v158);
            *((_DWORD *)v159 + 16) = *(_DWORD *)(v115 + 64);
          }
          v202 = (unsigned __int64)v159;
        }
        sub_B48880((__int64 *)&v203, v194, 0);
        sub_B48880((__int64 *)&v204, v194, 0);
        sub_B48880((__int64 *)&v205, *(_DWORD *)(v5 + 40) + 1, 0);
        v215.m128i_i64[0] = (__int64)&v216;
        v215.m128i_i64[1] = 0x400000000LL;
        v116 = sub_228AE30(v202);
        v211 = v116;
        for ( j = (unsigned __int64 **)&v202; v211 != -1; v116 = v211 )
        {
          if ( *(_DWORD *)&v223[48 * v116 + 16] == 1 )
            sub_228AC40((__int64 *)&v203, v116);
          else
            sub_228AC40((__int64 *)&v204, v116);
          v119 = v215.m128i_u32[2];
          v120 = (__int64)&v223[v117];
          v121 = v215.m128i_u32[2] + 1LL;
          if ( v121 > v215.m128i_u32[3] )
          {
            v182 = v120;
            sub_C8D5F0((__int64)&v215, &v216, v121, 8u, v120, v118);
            v119 = v215.m128i_u32[2];
            v120 = v182;
          }
          *(_QWORD *)(v215.m128i_i64[0] + 8 * v119) = v120;
          ++v215.m128i_i32[2];
          sub_2293510((__int64)v113);
        }
        v136 = (__int64)v113;
        sub_228D890(v5, (__int64 **)v215.m128i_i64[0], v215.m128i_u32[2]);
LABEL_190:
        v137 = (unsigned __int64)v203;
        do
        {
          if ( (v137 & 1) != 0 )
          {
            if ( ((v137 >> 1) & ~(-1LL << (v137 >> 58))) == 0 )
              goto LABEL_228;
          }
          else
          {
            v148 = sub_228AAA0(*(_QWORD **)v137, *(_QWORD *)v137 + 8LL * *(unsigned int *)(v137 + 8));
            if ( v149 == v148 )
            {
LABEL_228:
              v113 = (_QWORD *)v136;
              v151 = sub_228AE30((unsigned __int64)v204);
              v211 = v151;
              j = &v204;
              if ( v151 != -1 )
              {
                do
                {
                  v152 = (__int64 *)&v223[48 * v151];
                  if ( *((_DWORD *)v152 + 4) == 2 )
                  {
                    if ( sub_2295B30(v5, *v152, v152[1], (__int64)&v218, v150) )
                      goto LABEL_212;
                    sub_228AD70((__int64 *)&v204, v151);
                  }
                  sub_2293510((__int64)v113);
                  v151 = v211;
                }
                while ( v211 != -1 );
                v122 = v204;
              }
              v211 = sub_228AE30((unsigned __int64)v122);
              j = &v204;
              while ( v211 != -1 )
              {
                v123 = &v223[48 * v211];
                if ( *((_DWORD *)v123 + 4) != 3 )
                  BUG();
                if ( (unsigned __int8)sub_2291E30(
                                        v5,
                                        *(_QWORD *)v123,
                                        *((_QWORD *)v123 + 1),
                                        (unsigned __int64 *)v123 + 3,
                                        (__int64)&v218) )
                  goto LABEL_212;
                sub_2293510((__int64)v113);
              }
              v211 = sub_228AE30((unsigned __int64)v205);
              j = &v205;
              while ( v211 != -1 && *(_DWORD *)(v5 + 32) >= v211 )
              {
                v124 = 16LL * (v211 - 1);
                sub_2292C70(v5, v124 + v222, (_DWORD *)(v229.m128i_i64[0] + 48LL * v211));
                if ( (*(_BYTE *)(v222 + v124) & 7) == 0 )
                  goto LABEL_212;
                sub_2293510((__int64)v113);
              }
              if ( (__m128i *)v215.m128i_i64[0] != &v216 )
                _libc_free(v215.m128i_u64[0]);
              sub_228BF40(&v205);
              sub_228BF40(&v204);
              sub_228BF40(&v203);
              sub_228BF40((unsigned __int64 **)&v202);
              sub_2293510((__int64)&v208);
              v111 = v209;
              if ( v209 == -1 )
              {
                a1 = v171;
                v7 = v167;
                a4 = v165;
                goto LABEL_175;
              }
              goto LABEL_143;
            }
          }
          v138 = sub_228AE30(v137);
          v211 = v138;
          j = &v203;
        }
        while ( v138 == -1 );
        v189 = 0;
        while ( 1 )
        {
          i = 0;
          if ( sub_2294A30(
                 v5,
                 *(_QWORD *)&v223[48 * v138],
                 *(_QWORD *)&v223[48 * v138 + 8],
                 (int *)&v199,
                 (__int64)&v218,
                 (__int64)v112,
                 (__int64 *)&i) )
          {
            break;
          }
          sub_228AC40((__int64 *)&v205, v199);
          v139 = sub_228E3E0(v5, (__m128i *)(v229.m128i_i64[0] + 48LL * v199), v112);
          if ( v139 )
          {
            if ( !*(_DWORD *)(v229.m128i_i64[0] + 48LL * v199) )
              break;
            v189 = v139;
            sub_228AD70((__int64 *)&v203, v138);
            sub_2293510(v136);
            v138 = v211;
            if ( v211 == -1 )
              goto LABEL_202;
          }
          else
          {
            sub_228AD70((__int64 *)&v203, v138);
            sub_2293510(v136);
            v138 = v211;
            if ( v211 == -1 )
            {
              if ( !v189 )
                goto LABEL_190;
LABEL_202:
              v190 = v112;
              v211 = sub_228AE30((unsigned __int64)v204);
              j = &v204;
              while ( 1 )
              {
                v140 = v211;
                if ( v211 == -1 )
                  break;
                v141 = 48LL * v211;
                if ( (unsigned __int8)sub_22929F0(
                                        v5,
                                        (__int64 *)&v223[v141],
                                        (__int64 *)&v223[v141 + 8],
                                        (unsigned __int64 *)&v223[v141 + 24],
                                        &v229,
                                        &v221.m128i_i8[11]) )
                {
                  v162 = (__int64 *)&v223[v141];
                  v163 = *(_QWORD *)(v5 + 16);
                  v181 = (_QWORD *)sub_D48290(v163, *((_QWORD *)v165 + 5));
                  v142 = (_QWORD *)sub_D48290(v163, *((_QWORD *)v167 + 5));
                  v143 = sub_228DB90(v5, *v162, v142, v162[1], v181, v162 + 3);
                  v144 = (__int64 *)&v223[v141];
                  *(_DWORD *)&v223[v141 + 16] = v143;
                  if ( v143 == 1 )
                  {
                    sub_228AC40((__int64 *)&v203, v140);
                    sub_228AD70((__int64 *)&v204, v140);
                  }
                  else if ( v143 <= 1 )
                  {
                    if ( sub_228EC60(v5, *v144, v144[1], (__int64)&v218) )
                      goto LABEL_212;
                    sub_228AD70((__int64 *)&v204, v140);
                  }
                  else if ( v143 - 2 > 1 )
                  {
                    goto LABEL_254;
                  }
                }
                sub_2293510(v136);
              }
              v112 = v190;
              goto LABEL_190;
            }
          }
        }
LABEL_212:
        a1 = v171;
        *v171 = 0;
        if ( (__m128i *)v215.m128i_i64[0] != &v216 )
          _libc_free(v215.m128i_u64[0]);
        sub_228BF40(&v205);
        sub_228BF40(&v204);
        sub_228BF40(&v203);
        sub_228BF40((unsigned __int64 **)&v202);
        if ( (__m128i *)v229.m128i_i64[0] != &v230 )
          _libc_free(v229.m128i_u64[0]);
        goto LABEL_116;
      }
LABEL_175:
      if ( (__m128i *)v229.m128i_i64[0] != &v230 )
        _libc_free(v229.m128i_u64[0]);
    }
    sub_B48880(v229.m128i_i64, *(_DWORD *)(v5 + 40) + 1, 0);
    v128 = v194;
    if ( v194 )
    {
      v129 = v194;
      v195 = v7;
      v197 = a1;
      v130 = 0;
      v131 = 48 * v129;
      do
      {
        v132 = &v223[v130];
        v130 += 48;
        sub_228C270((unsigned __int64 *)&v229, (unsigned __int64 *)v132 + 3, v128, v125, v126, v127);
      }
      while ( v131 != v130 );
      a1 = v197;
      v7 = v195;
    }
    if ( *(_DWORD *)(v5 + 32) )
    {
      v133 = 1;
      do
      {
        if ( (v229.m128i_i8[0] & 1) != 0 )
          v134 = ((((unsigned __int64)v229.m128i_i64[0] >> 1) & ~(-1LL << ((unsigned __int64)v229.m128i_i64[0] >> 58))) >> v133)
               & 1;
        else
          v134 = (*(_QWORD *)(*(_QWORD *)v229.m128i_i64[0] + 8LL * (v133 >> 6)) >> v133) & 1LL;
        if ( (_BYTE)v134 )
          *(_BYTE *)(v222 + 16LL * (v133 - 1)) &= ~8u;
        v135 = *(_DWORD *)(v5 + 32);
        ++v133;
      }
      while ( v135 >= v133 );
      if ( v7 != a4 )
      {
        if ( v135 )
        {
          v153 = 1;
          while ( (sub_228A590((__int64)&v218, v153) & 2) != 0 )
          {
            if ( *(_DWORD *)(v5 + 32) < (unsigned int)++v153 )
              goto LABEL_222;
          }
          v221.m128i_i8[10] = 0;
        }
        goto LABEL_222;
      }
      if ( v135 )
      {
        v145 = 1;
        while ( (unsigned int)sub_228A590((__int64)&v218, v145) == 2 )
        {
          if ( *(_DWORD *)(v5 + 32) < (unsigned int)++v145 )
            goto LABEL_249;
        }
        goto LABEL_222;
      }
    }
    else if ( v7 != a4 )
    {
LABEL_222:
      v146 = sub_22077B0(0x38u);
      if ( v146 )
      {
        *(_QWORD *)v146 = &unk_4A08E50;
        *(_QWORD *)(v146 + 8) = v219;
        *(__m128i *)(v146 + 16) = v220;
        *(_QWORD *)(v146 + 32) = v221.m128i_i64[0];
        *(_DWORD *)(v146 + 40) = v221.m128i_i32[2];
        v147 = v222;
        v222 = 0;
        *(_QWORD *)(v146 + 48) = v147;
      }
      *a1 = v146;
      goto LABEL_225;
    }
LABEL_249:
    *a1 = 0;
LABEL_225:
    sub_228BF40((unsigned __int64 **)&v229);
LABEL_116:
    sub_228BF40(&v201);
    sub_228BF40(&v200);
LABEL_27:
    v23 = v223;
    v24 = (unsigned __int64)&v223[48 * (unsigned int)v224];
    if ( v223 != (unsigned __int8 *)v24 )
    {
      do
      {
        v25 = *(_QWORD *)(v24 - 8);
        v24 -= 48LL;
        if ( (v25 & 1) == 0 && v25 )
        {
          if ( *(_QWORD *)v25 != v25 + 16 )
            _libc_free(*(_QWORD *)v25);
          j_j___libc_free_0(v25);
        }
        v26 = *(_QWORD *)(v24 + 32);
        if ( (v26 & 1) == 0 && v26 )
        {
          if ( *(_QWORD *)v26 != v26 + 16 )
            _libc_free(*(_QWORD *)v26);
          j_j___libc_free_0(v26);
        }
        v27 = *(_QWORD *)(v24 + 24);
        if ( (v27 & 1) == 0 && v27 )
        {
          if ( *(_QWORD *)v27 != v27 + 16 )
            _libc_free(*(_QWORD *)v27);
          j_j___libc_free_0(v27);
        }
      }
      while ( v23 != (unsigned __int8 *)v24 );
      v24 = (unsigned __int64)v223;
    }
    if ( (__m128i *)v24 != &v225 )
      _libc_free(v24);
    v218 = &unk_4A08E50;
    if ( v222 )
      j_j___libc_free_0_0(v222);
    return a1;
  }
LABEL_49:
  v28 = (_QWORD *)sub_22077B0(0x28u);
  if ( v28 )
  {
    v28[1] = v7;
    v28[2] = a4;
    v28[3] = 0;
    *v28 = &unk_4A08DD0;
    v28[4] = 0;
  }
  *a1 = (__int64)v28;
  return a1;
}
