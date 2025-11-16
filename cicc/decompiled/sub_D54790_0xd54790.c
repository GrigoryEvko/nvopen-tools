// Function: sub_D54790
// Address: 0xd54790
//
__int64 __fastcall sub_D54790(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rbx
  __int64 v7; // rdx
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  __m128i v12; // xmm0
  __m128i v13; // xmm1
  __m128i v14; // xmm2
  __m128i v15; // xmm3
  __m128i v16; // xmm4
  __int64 v17; // rdx
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  __m128i v22; // xmm5
  __m128i v23; // xmm6
  __m128i v24; // xmm7
  __m128i v25; // xmm0
  __m128i v26; // xmm1
  __int64 v27; // rdx
  __int64 v28; // rdx
  __int64 v29; // rcx
  __int64 v30; // r8
  __int64 v31; // r9
  __m128i v32; // xmm2
  __m128i v33; // xmm3
  __m128i v34; // xmm4
  __m128i v35; // xmm5
  __m128i v36; // xmm6
  __int64 v37; // rdx
  __int64 v38; // rdx
  __int64 v39; // rcx
  __int64 v40; // r8
  __int64 v41; // r9
  __m128i v42; // xmm7
  __m128i v43; // xmm0
  __m128i v44; // xmm1
  __m128i v45; // xmm2
  __m128i v46; // xmm3
  __int64 v47; // rdx
  __int64 v48; // rcx
  __int64 v49; // r8
  __int64 v50; // r9
  __int64 v51; // rdx
  __int64 v52; // rcx
  __int64 v53; // r8
  __int64 v54; // r9
  __int64 v55; // rdx
  __int64 v56; // rcx
  __int64 v57; // r8
  __int64 v58; // r9
  __int64 v59; // r8
  __int64 v60; // r9
  _QWORD *v61; // rsi
  __int64 v62; // rcx
  __int64 v63; // rdx
  __int64 *v64; // rsi
  __int64 result; // rax
  __int64 v66; // rdi
  __int64 *v67; // rbx
  unsigned __int64 v68; // r12
  __int64 v69; // rdi
  __int64 v70; // rdi
  __int64 *v71; // rbx
  unsigned __int64 v72; // r12
  __int64 v73; // rdi
  __int64 v74; // rdi
  __int64 *v75; // rbx
  unsigned __int64 v76; // r12
  __int64 v77; // rdi
  __int64 v78; // rdi
  __int64 *v79; // rbx
  unsigned __int64 v80; // r12
  __int64 v81; // rdi
  __int64 v82; // rdi
  __int64 *v83; // rbx
  unsigned __int64 v84; // r12
  __int64 v85; // rdi
  __int64 v86; // rdi
  __int64 *v87; // rbx
  unsigned __int64 v88; // r12
  __int64 v89; // rdi
  __int64 v90; // rdi
  __int64 *v91; // rbx
  unsigned __int64 v92; // r12
  __int64 v93; // rdi
  __int64 v94; // rdi
  __int64 *v95; // rbx
  unsigned __int64 v96; // r12
  __int64 v97; // rdi
  __int64 v98; // rdi
  __int64 *v99; // rbx
  unsigned __int64 v100; // r12
  __int64 v101; // rdi
  __int64 v102; // rdi
  __int64 *v103; // rbx
  unsigned __int64 v104; // r12
  __int64 v105; // rdi
  __int64 v106; // rdi
  __int64 *v107; // rbx
  unsigned __int64 v108; // r12
  __int64 v109; // rdi
  __int64 v110; // rdi
  __int64 *v111; // rbx
  unsigned __int64 v112; // r12
  __int64 v113; // rdi
  __int64 v114[4]; // [rsp+20h] [rbp-990h] BYREF
  __int64 v115[4]; // [rsp+40h] [rbp-970h] BYREF
  __int64 v116[4]; // [rsp+60h] [rbp-950h] BYREF
  _BYTE v117[8]; // [rsp+80h] [rbp-930h] BYREF
  __int64 v118; // [rsp+88h] [rbp-928h]
  char v119; // [rsp+9Ch] [rbp-914h]
  _BYTE v120[64]; // [rsp+A0h] [rbp-910h] BYREF
  __m128i v121; // [rsp+E0h] [rbp-8D0h] BYREF
  __m128i v122; // [rsp+F0h] [rbp-8C0h] BYREF
  __m128i v123; // [rsp+100h] [rbp-8B0h] BYREF
  __m128i v124; // [rsp+110h] [rbp-8A0h] BYREF
  __m128i v125; // [rsp+120h] [rbp-890h] BYREF
  int v126; // [rsp+130h] [rbp-880h]
  _BYTE v127[8]; // [rsp+140h] [rbp-870h] BYREF
  __int64 v128; // [rsp+148h] [rbp-868h]
  char v129; // [rsp+15Ch] [rbp-854h]
  _BYTE v130[64]; // [rsp+160h] [rbp-850h] BYREF
  __m128i v131; // [rsp+1A0h] [rbp-810h] BYREF
  __m128i v132; // [rsp+1B0h] [rbp-800h]
  __m128i v133; // [rsp+1C0h] [rbp-7F0h]
  __m128i v134; // [rsp+1D0h] [rbp-7E0h]
  __m128i v135; // [rsp+1E0h] [rbp-7D0h]
  int v136; // [rsp+1F0h] [rbp-7C0h]
  _BYTE v137[8]; // [rsp+200h] [rbp-7B0h] BYREF
  __int64 v138; // [rsp+208h] [rbp-7A8h]
  char v139; // [rsp+21Ch] [rbp-794h]
  _BYTE v140[64]; // [rsp+220h] [rbp-790h] BYREF
  __m128i v141; // [rsp+260h] [rbp-750h] BYREF
  __m128i v142; // [rsp+270h] [rbp-740h] BYREF
  __m128i v143; // [rsp+280h] [rbp-730h] BYREF
  __m128i v144; // [rsp+290h] [rbp-720h] BYREF
  __m128i v145; // [rsp+2A0h] [rbp-710h] BYREF
  int v146; // [rsp+2B0h] [rbp-700h]
  _BYTE v147[8]; // [rsp+2C0h] [rbp-6F0h] BYREF
  __int64 v148; // [rsp+2C8h] [rbp-6E8h]
  char v149; // [rsp+2DCh] [rbp-6D4h]
  _BYTE v150[64]; // [rsp+2E0h] [rbp-6D0h] BYREF
  __m128i v151; // [rsp+320h] [rbp-690h] BYREF
  __m128i v152; // [rsp+330h] [rbp-680h]
  __m128i v153; // [rsp+340h] [rbp-670h]
  __m128i v154; // [rsp+350h] [rbp-660h]
  __m128i v155; // [rsp+360h] [rbp-650h]
  int v156; // [rsp+370h] [rbp-640h]
  _BYTE v157[8]; // [rsp+380h] [rbp-630h] BYREF
  __int64 v158; // [rsp+388h] [rbp-628h]
  char v159; // [rsp+39Ch] [rbp-614h]
  _BYTE v160[64]; // [rsp+3A0h] [rbp-610h] BYREF
  __m128i v161; // [rsp+3E0h] [rbp-5D0h] BYREF
  __m128i v162; // [rsp+3F0h] [rbp-5C0h]
  __m128i v163; // [rsp+400h] [rbp-5B0h]
  __m128i v164; // [rsp+410h] [rbp-5A0h]
  __m128i v165; // [rsp+420h] [rbp-590h]
  int v166; // [rsp+430h] [rbp-580h]
  _BYTE v167[8]; // [rsp+440h] [rbp-570h] BYREF
  __int64 v168; // [rsp+448h] [rbp-568h]
  char v169; // [rsp+45Ch] [rbp-554h]
  _BYTE v170[64]; // [rsp+460h] [rbp-550h] BYREF
  __m128i v171; // [rsp+4A0h] [rbp-510h] BYREF
  __m128i v172; // [rsp+4B0h] [rbp-500h] BYREF
  __m128i v173; // [rsp+4C0h] [rbp-4F0h] BYREF
  __m128i v174; // [rsp+4D0h] [rbp-4E0h] BYREF
  __m128i v175; // [rsp+4E0h] [rbp-4D0h] BYREF
  int v176; // [rsp+4F0h] [rbp-4C0h]
  _BYTE v177[8]; // [rsp+500h] [rbp-4B0h] BYREF
  __int64 v178; // [rsp+508h] [rbp-4A8h]
  char v179; // [rsp+51Ch] [rbp-494h]
  _BYTE v180[64]; // [rsp+520h] [rbp-490h] BYREF
  __m128i v181; // [rsp+560h] [rbp-450h] BYREF
  __m128i v182; // [rsp+570h] [rbp-440h]
  __m128i v183; // [rsp+580h] [rbp-430h]
  __m128i v184; // [rsp+590h] [rbp-420h]
  __m128i v185; // [rsp+5A0h] [rbp-410h]
  int v186; // [rsp+5B0h] [rbp-400h]
  _BYTE v187[8]; // [rsp+5C0h] [rbp-3F0h] BYREF
  __int64 v188; // [rsp+5C8h] [rbp-3E8h]
  char v189; // [rsp+5DCh] [rbp-3D4h]
  _BYTE v190[64]; // [rsp+5E0h] [rbp-3D0h] BYREF
  __m128i v191; // [rsp+620h] [rbp-390h] BYREF
  __m128i v192; // [rsp+630h] [rbp-380h] BYREF
  __m128i v193; // [rsp+640h] [rbp-370h] BYREF
  __m128i v194; // [rsp+650h] [rbp-360h] BYREF
  __m128i v195; // [rsp+660h] [rbp-350h] BYREF
  int v196; // [rsp+670h] [rbp-340h]
  _BYTE v197[8]; // [rsp+680h] [rbp-330h] BYREF
  __int64 v198; // [rsp+688h] [rbp-328h]
  char v199; // [rsp+69Ch] [rbp-314h]
  __int64 v200; // [rsp+6E0h] [rbp-2D0h]
  __int64 v201; // [rsp+6E8h] [rbp-2C8h]
  unsigned __int64 v202; // [rsp+708h] [rbp-2A8h]
  __int64 v203; // [rsp+728h] [rbp-288h]
  _BYTE v204[8]; // [rsp+740h] [rbp-270h] BYREF
  __int64 v205; // [rsp+748h] [rbp-268h]
  char v206; // [rsp+75Ch] [rbp-254h]
  __int64 v207; // [rsp+7A0h] [rbp-210h]
  __int64 v208; // [rsp+7A8h] [rbp-208h]
  unsigned __int64 v209; // [rsp+7C8h] [rbp-1E8h]
  __int64 v210; // [rsp+7E8h] [rbp-1C8h]
  _BYTE v211[8]; // [rsp+800h] [rbp-1B0h] BYREF
  __int64 v212; // [rsp+808h] [rbp-1A8h]
  char v213; // [rsp+81Ch] [rbp-194h]
  __int64 v214; // [rsp+860h] [rbp-150h]
  __int64 v215; // [rsp+868h] [rbp-148h]
  __int64 v216; // [rsp+870h] [rbp-140h]
  __int64 v217; // [rsp+878h] [rbp-138h]
  __int64 v218; // [rsp+880h] [rbp-130h]
  unsigned __int64 v219; // [rsp+888h] [rbp-128h]
  __int64 v220; // [rsp+890h] [rbp-120h]
  __int64 v221; // [rsp+898h] [rbp-118h]
  __int64 v222; // [rsp+8A8h] [rbp-108h]
  _BYTE v223[8]; // [rsp+8C0h] [rbp-F0h] BYREF
  __int64 v224; // [rsp+8C8h] [rbp-E8h]
  char v225; // [rsp+8DCh] [rbp-D4h]
  __int64 v226; // [rsp+920h] [rbp-90h]
  __int64 v227; // [rsp+928h] [rbp-88h]
  __int64 v228; // [rsp+930h] [rbp-80h]
  __int64 v229; // [rsp+938h] [rbp-78h]
  __int64 v230; // [rsp+940h] [rbp-70h]
  unsigned __int64 v231; // [rsp+948h] [rbp-68h]
  __int64 v232; // [rsp+950h] [rbp-60h]
  __int64 v233; // [rsp+958h] [rbp-58h]
  __int64 v234; // [rsp+960h] [rbp-50h]
  __int64 v235; // [rsp+968h] [rbp-48h]

  v6 = a3;
  sub_D53210((__int64)v137, a2, a3, a4, a5, a6);
  sub_C8CF70((__int64)v147, v150, 8, (__int64)v140, (__int64)v137);
  v151 = 0u;
  v152 = 0u;
  v153 = 0u;
  v154 = 0u;
  v155 = 0u;
  sub_D53100(v151.m128i_i64, 0, v7);
  if ( v141.m128i_i64[0] )
  {
    v11 = v152.m128i_i64[1];
    v10 = v153.m128i_i64[0];
    v9 = v154.m128i_i64[1];
    v8 = v155.m128i_i64[0];
    v12 = _mm_loadu_si128(&v141);
    v13 = _mm_loadu_si128(&v142);
    v141 = v151;
    v14 = _mm_loadu_si128(&v143);
    v15 = _mm_loadu_si128(&v144);
    v16 = _mm_loadu_si128(&v145);
    v142 = v152;
    v143 = v153;
    v144 = v154;
    v145 = v155;
    v151 = v12;
    v152 = v13;
    v153 = v14;
    v154 = v15;
    v155 = v16;
  }
  v156 = v146;
  sub_D53210((__int64)v117, a1, v8, v9, v10, v11);
  sub_C8CF70((__int64)v127, v130, 8, (__int64)v120, (__int64)v117);
  v131 = 0u;
  v132 = 0u;
  v133 = 0u;
  v134 = 0u;
  v135 = 0u;
  sub_D53100(v131.m128i_i64, 0, v17);
  if ( v121.m128i_i64[0] )
  {
    v21 = v132.m128i_i64[1];
    v20 = v133.m128i_i64[0];
    v19 = v134.m128i_i64[1];
    v18 = v135.m128i_i64[0];
    v22 = _mm_loadu_si128(&v121);
    v23 = _mm_loadu_si128(&v122);
    v121 = v131;
    v24 = _mm_loadu_si128(&v123);
    v25 = _mm_loadu_si128(&v124);
    v26 = _mm_loadu_si128(&v125);
    v122 = v132;
    v123 = v133;
    v124 = v134;
    v125 = v135;
    v131 = v22;
    v132 = v23;
    v133 = v24;
    v134 = v25;
    v135 = v26;
  }
  v136 = v126;
  sub_D53210((__int64)v167, (__int64)v147, v18, v19, v20, v21);
  sub_C8CF70((__int64)v157, v160, 8, (__int64)v170, (__int64)v167);
  v161 = 0u;
  v162 = 0u;
  v163 = 0u;
  v164 = 0u;
  v165 = 0u;
  sub_D53100(v161.m128i_i64, 0, v27);
  if ( v171.m128i_i64[0] )
  {
    v31 = v162.m128i_i64[1];
    v30 = v163.m128i_i64[0];
    v29 = v164.m128i_i64[1];
    v28 = v165.m128i_i64[0];
    v32 = _mm_loadu_si128(&v171);
    v33 = _mm_loadu_si128(&v172);
    v171 = v161;
    v34 = _mm_loadu_si128(&v173);
    v35 = _mm_loadu_si128(&v174);
    v36 = _mm_loadu_si128(&v175);
    v172 = v162;
    v173 = v163;
    v174 = v164;
    v175 = v165;
    v161 = v32;
    v162 = v33;
    v163 = v34;
    v164 = v35;
    v165 = v36;
  }
  v166 = v176;
  sub_D53210((__int64)v187, (__int64)v127, v28, v29, v30, v31);
  sub_C8CF70((__int64)v177, v180, 8, (__int64)v190, (__int64)v187);
  v181 = 0u;
  v182 = 0u;
  v183 = 0u;
  v184 = 0u;
  v185 = 0u;
  sub_D53100(v181.m128i_i64, 0, v37);
  if ( v191.m128i_i64[0] )
  {
    v41 = v182.m128i_i64[1];
    v40 = v183.m128i_i64[0];
    v39 = v184.m128i_i64[1];
    v38 = v185.m128i_i64[0];
    v42 = _mm_loadu_si128(&v191);
    v43 = _mm_loadu_si128(&v192);
    v191 = v181;
    v44 = _mm_loadu_si128(&v193);
    v45 = _mm_loadu_si128(&v194);
    v46 = _mm_loadu_si128(&v195);
    v192 = v182;
    v193 = v183;
    v194 = v184;
    v195 = v185;
    v181 = v42;
    v182 = v43;
    v183 = v44;
    v184 = v45;
    v185 = v46;
  }
  v186 = v196;
  sub_D53210((__int64)v197, (__int64)v157, v38, v39, v40, v41);
  sub_D53210((__int64)v204, (__int64)v177, v47, v48, v49, v50);
  sub_D53210((__int64)v211, (__int64)v197, v51, v52, v53, v54);
  sub_D53210((__int64)v223, (__int64)v204, v55, v56, v57, v58);
  while ( 1 )
  {
    v59 = v218;
    v60 = v216;
    v61 = (_QWORD *)v228;
    v62 = ((v220 - v221) >> 5) + 16 * (((__int64)(v222 - v219) >> 3) - 1) + ((v218 - v216) >> 5);
    v63 = (v230 - v228) >> 5;
    if ( v62 != v63 + 16 * (((__int64)(v235 - v231) >> 3) - 1) + ((v232 - v233) >> 5) )
      goto LABEL_10;
    v114[0] = v228;
    v116[3] = v219;
    v115[2] = v234;
    v64 = v115;
    v115[0] = v232;
    v114[1] = v229;
    v115[1] = v233;
    v115[3] = v235;
    v114[2] = v230;
    v114[3] = v231;
    v116[0] = v216;
    v116[1] = v217;
    v116[2] = v218;
    result = sub_D542B0(v114, v115, v116);
    if ( (_BYTE)result )
      break;
    v61 = (_QWORD *)v228;
LABEL_10:
    v6 += 8;
    *(_QWORD *)(v6 - 8) = *v61;
    sub_D53E10((__int64)v223, (__int64)v61, (__int64 *)v63, v62, v59, v60);
  }
  v66 = v226;
  if ( v226 )
  {
    v67 = (__int64 *)v231;
    v68 = v235 + 8;
    if ( v235 + 8 > v231 )
    {
      do
      {
        v69 = *v67++;
        j_j___libc_free_0(v69, 512);
      }
      while ( v68 > (unsigned __int64)v67 );
      v66 = v226;
    }
    v64 = (__int64 *)(8 * v227);
    result = j_j___libc_free_0(v66, 8 * v227);
  }
  if ( !v225 )
    result = _libc_free(v224, v64);
  v70 = v214;
  if ( v214 )
  {
    v71 = (__int64 *)v219;
    v72 = v222 + 8;
    if ( v222 + 8 > v219 )
    {
      do
      {
        v73 = *v71++;
        j_j___libc_free_0(v73, 512);
      }
      while ( v72 > (unsigned __int64)v71 );
      v70 = v214;
    }
    v64 = (__int64 *)(8 * v215);
    result = j_j___libc_free_0(v70, 8 * v215);
  }
  if ( !v213 )
    result = _libc_free(v212, v64);
  v74 = v207;
  if ( v207 )
  {
    v75 = (__int64 *)v209;
    v76 = v210 + 8;
    if ( v210 + 8 > v209 )
    {
      do
      {
        v77 = *v75++;
        j_j___libc_free_0(v77, 512);
      }
      while ( v76 > (unsigned __int64)v75 );
      v74 = v207;
    }
    v64 = (__int64 *)(8 * v208);
    result = j_j___libc_free_0(v74, 8 * v208);
  }
  if ( !v206 )
    result = _libc_free(v205, v64);
  v78 = v200;
  if ( v200 )
  {
    v79 = (__int64 *)v202;
    v80 = v203 + 8;
    if ( v203 + 8 > v202 )
    {
      do
      {
        v81 = *v79++;
        j_j___libc_free_0(v81, 512);
      }
      while ( v80 > (unsigned __int64)v79 );
      v78 = v200;
    }
    v64 = (__int64 *)(8 * v201);
    result = j_j___libc_free_0(v78, 8 * v201);
  }
  if ( !v199 )
    result = _libc_free(v198, v64);
  v82 = v181.m128i_i64[0];
  if ( v181.m128i_i64[0] )
  {
    v83 = (__int64 *)v183.m128i_i64[1];
    v84 = v185.m128i_i64[1] + 8;
    if ( (unsigned __int64)(v185.m128i_i64[1] + 8) > v183.m128i_i64[1] )
    {
      do
      {
        v85 = *v83++;
        j_j___libc_free_0(v85, 512);
      }
      while ( v84 > (unsigned __int64)v83 );
      v82 = v181.m128i_i64[0];
    }
    v64 = (__int64 *)(8 * v181.m128i_i64[1]);
    result = j_j___libc_free_0(v82, 8 * v181.m128i_i64[1]);
  }
  if ( !v179 )
    result = _libc_free(v178, v64);
  v86 = v191.m128i_i64[0];
  if ( v191.m128i_i64[0] )
  {
    v87 = (__int64 *)v193.m128i_i64[1];
    v88 = v195.m128i_i64[1] + 8;
    if ( (unsigned __int64)(v195.m128i_i64[1] + 8) > v193.m128i_i64[1] )
    {
      do
      {
        v89 = *v87++;
        j_j___libc_free_0(v89, 512);
      }
      while ( v88 > (unsigned __int64)v87 );
      v86 = v191.m128i_i64[0];
    }
    v64 = (__int64 *)(8 * v191.m128i_i64[1]);
    result = j_j___libc_free_0(v86, 8 * v191.m128i_i64[1]);
  }
  if ( !v189 )
    result = _libc_free(v188, v64);
  v90 = v161.m128i_i64[0];
  if ( v161.m128i_i64[0] )
  {
    v91 = (__int64 *)v163.m128i_i64[1];
    v92 = v165.m128i_i64[1] + 8;
    if ( (unsigned __int64)(v165.m128i_i64[1] + 8) > v163.m128i_i64[1] )
    {
      do
      {
        v93 = *v91++;
        j_j___libc_free_0(v93, 512);
      }
      while ( v92 > (unsigned __int64)v91 );
      v90 = v161.m128i_i64[0];
    }
    v64 = (__int64 *)(8 * v161.m128i_i64[1]);
    result = j_j___libc_free_0(v90, 8 * v161.m128i_i64[1]);
  }
  if ( !v159 )
    result = _libc_free(v158, v64);
  v94 = v171.m128i_i64[0];
  if ( v171.m128i_i64[0] )
  {
    v95 = (__int64 *)v173.m128i_i64[1];
    v96 = v175.m128i_i64[1] + 8;
    if ( (unsigned __int64)(v175.m128i_i64[1] + 8) > v173.m128i_i64[1] )
    {
      do
      {
        v97 = *v95++;
        j_j___libc_free_0(v97, 512);
      }
      while ( v96 > (unsigned __int64)v95 );
      v94 = v171.m128i_i64[0];
    }
    v64 = (__int64 *)(8 * v171.m128i_i64[1]);
    result = j_j___libc_free_0(v94, 8 * v171.m128i_i64[1]);
  }
  if ( !v169 )
    result = _libc_free(v168, v64);
  v98 = v131.m128i_i64[0];
  if ( v131.m128i_i64[0] )
  {
    v99 = (__int64 *)v133.m128i_i64[1];
    v100 = v135.m128i_i64[1] + 8;
    if ( (unsigned __int64)(v135.m128i_i64[1] + 8) > v133.m128i_i64[1] )
    {
      do
      {
        v101 = *v99++;
        j_j___libc_free_0(v101, 512);
      }
      while ( v100 > (unsigned __int64)v99 );
      v98 = v131.m128i_i64[0];
    }
    v64 = (__int64 *)(8 * v131.m128i_i64[1]);
    result = j_j___libc_free_0(v98, 8 * v131.m128i_i64[1]);
  }
  if ( !v129 )
    result = _libc_free(v128, v64);
  v102 = v121.m128i_i64[0];
  if ( v121.m128i_i64[0] )
  {
    v103 = (__int64 *)v123.m128i_i64[1];
    v104 = v125.m128i_i64[1] + 8;
    if ( (unsigned __int64)(v125.m128i_i64[1] + 8) > v123.m128i_i64[1] )
    {
      do
      {
        v105 = *v103++;
        j_j___libc_free_0(v105, 512);
      }
      while ( v104 > (unsigned __int64)v103 );
      v102 = v121.m128i_i64[0];
    }
    v64 = (__int64 *)(8 * v121.m128i_i64[1]);
    result = j_j___libc_free_0(v102, 8 * v121.m128i_i64[1]);
  }
  if ( !v119 )
    result = _libc_free(v118, v64);
  v106 = v151.m128i_i64[0];
  if ( v151.m128i_i64[0] )
  {
    v107 = (__int64 *)v153.m128i_i64[1];
    v108 = v155.m128i_i64[1] + 8;
    if ( (unsigned __int64)(v155.m128i_i64[1] + 8) > v153.m128i_i64[1] )
    {
      do
      {
        v109 = *v107++;
        j_j___libc_free_0(v109, 512);
      }
      while ( v108 > (unsigned __int64)v107 );
      v106 = v151.m128i_i64[0];
    }
    v64 = (__int64 *)(8 * v151.m128i_i64[1]);
    result = j_j___libc_free_0(v106, 8 * v151.m128i_i64[1]);
  }
  if ( !v149 )
    result = _libc_free(v148, v64);
  v110 = v141.m128i_i64[0];
  if ( v141.m128i_i64[0] )
  {
    v111 = (__int64 *)v143.m128i_i64[1];
    v112 = v145.m128i_i64[1] + 8;
    if ( (unsigned __int64)(v145.m128i_i64[1] + 8) > v143.m128i_i64[1] )
    {
      do
      {
        v113 = *v111++;
        j_j___libc_free_0(v113, 512);
      }
      while ( v112 > (unsigned __int64)v111 );
      v110 = v141.m128i_i64[0];
    }
    v64 = (__int64 *)(8 * v141.m128i_i64[1]);
    result = j_j___libc_free_0(v110, 8 * v141.m128i_i64[1]);
  }
  if ( !v139 )
    return _libc_free(v138, v64);
  return result;
}
