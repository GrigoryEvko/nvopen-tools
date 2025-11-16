// Function: sub_25D26F0
// Address: 0x25d26f0
//
void __fastcall sub_25D26F0(__int64 a1, __int64 a2, void *a3, size_t a4, __int64 a5)
{
  _QWORD *v6; // r14
  int v8; // eax
  int v9; // eax
  __int64 v10; // r8
  __int64 v11; // rdx
  __int64 *v12; // rax
  __int64 v13; // rdi
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // rax
  _QWORD *v17; // r13
  __int64 v18; // rdx
  _QWORD *v19; // rsi
  _QWORD *i; // rbx
  __int64 v21; // rcx
  __int64 *v22; // rdx
  __int64 v23; // rdi
  __int64 v24; // r9
  unsigned int v25; // eax
  __int64 *v26; // rsi
  __int64 v27; // r10
  __m128i v28; // xmm6
  __m128i v29; // xmm1
  __m128i v30; // xmm0
  __m128i v31; // xmm2
  __m128i v32; // xmm7
  __m128i v33; // xmm6
  __m128i v34; // xmm1
  __m128i v35; // xmm0
  __m128i v36; // xmm4
  __m128i v37; // xmm3
  __m128i v38; // xmm5
  __m128i v39; // xmm1
  __m128i v40; // xmm0
  __m128i v41; // xmm13
  __m128i v42; // xmm12
  __m128i v43; // xmm11
  __m128i v44; // xmm7
  __m128i v45; // xmm10
  __m128i v46; // xmm6
  __m128i v47; // xmm9
  __m128i v48; // xmm5
  __m128i v49; // xmm4
  __m128i v50; // xmm3
  __m128i v51; // xmm8
  __m128i v52; // xmm2
  __m128i v53; // xmm1
  __int64 v54; // r9
  _QWORD *v55; // rax
  __int64 j; // rcx
  __int64 *v57; // r14
  _QWORD *v58; // r12
  __int64 v59; // rdx
  __int64 v60; // rdx
  int v61; // eax
  int v62; // eax
  __int64 *v63; // r9
  __int64 v64; // r10
  __int64 *v65; // r12
  __int64 v66; // rdx
  int v67; // eax
  int v68; // eax
  const void *v69; // rdi
  size_t v70; // rax
  int v71; // eax
  _QWORD *v72; // rax
  __int64 *v73; // r12
  unsigned __int32 k; // eax
  __int64 v75; // rsi
  __int64 v76; // rax
  unsigned int v77; // r10d
  int v78; // esi
  int v79; // eax
  int v80; // esi
  int v81; // eax
  char *v82; // rdi
  __int64 v83; // r9
  size_t v84; // rdx
  int v85; // r8d
  int v86; // r10d
  unsigned int m; // r11d
  __int64 v88; // rcx
  const void *v89; // rsi
  bool v90; // al
  unsigned int v91; // r11d
  int v92; // eax
  __int64 m128i_i64; // rsi
  __m128i *v94; // rax
  __m128i v95; // xmm7
  int v96; // [rsp+Ch] [rbp-1314h]
  __int64 *v98; // [rsp+18h] [rbp-1308h]
  __int64 v99; // [rsp+18h] [rbp-1308h]
  unsigned int v100; // [rsp+20h] [rbp-1300h]
  void *s2; // [rsp+40h] [rbp-12E0h]
  int s2a; // [rsp+40h] [rbp-12E0h]
  unsigned __int8 v106; // [rsp+48h] [rbp-12D8h]
  __int64 *v107; // [rsp+48h] [rbp-12D8h]
  __int64 v108; // [rsp+48h] [rbp-12D8h]
  unsigned __int8 v109; // [rsp+50h] [rbp-12D0h]
  char v110; // [rsp+50h] [rbp-12D0h]
  size_t v111; // [rsp+50h] [rbp-12D0h]
  _QWORD *v112; // [rsp+58h] [rbp-12C8h]
  size_t n; // [rsp+60h] [rbp-12C0h]
  __int64 v114; // [rsp+68h] [rbp-12B8h]
  int v115; // [rsp+68h] [rbp-12B8h]
  __int64 v116; // [rsp+70h] [rbp-12B0h]
  unsigned __int8 v117; // [rsp+70h] [rbp-12B0h]
  __int64 v118; // [rsp+70h] [rbp-12B0h]
  __int64 v119; // [rsp+70h] [rbp-12B0h]
  __int64 v120; // [rsp+78h] [rbp-12A8h]
  __int64 v121; // [rsp+78h] [rbp-12A8h]
  _QWORD v122[6]; // [rsp+B0h] [rbp-1270h] BYREF
  __m128i v123; // [rsp+E0h] [rbp-1240h] BYREF
  __m128i v124; // [rsp+F0h] [rbp-1230h] BYREF
  __m128i v125; // [rsp+100h] [rbp-1220h] BYREF
  __int64 v126; // [rsp+110h] [rbp-1210h]
  __m128i v127; // [rsp+120h] [rbp-1200h] BYREF
  __m128i v128; // [rsp+130h] [rbp-11F0h] BYREF
  __m128i v129; // [rsp+140h] [rbp-11E0h] BYREF
  __int64 v130; // [rsp+150h] [rbp-11D0h]
  __m128i v131; // [rsp+160h] [rbp-11C0h] BYREF
  __m128i v132; // [rsp+170h] [rbp-11B0h] BYREF
  __m128i v133; // [rsp+180h] [rbp-11A0h] BYREF
  __int64 v134; // [rsp+190h] [rbp-1190h]
  __m128i v135; // [rsp+198h] [rbp-1188h] BYREF
  __m128i v136; // [rsp+1A8h] [rbp-1178h] BYREF
  __m128i v137; // [rsp+1B8h] [rbp-1168h] BYREF
  __int64 v138; // [rsp+1C8h] [rbp-1158h]
  __m128i v139; // [rsp+1D0h] [rbp-1150h]
  __m128i v140; // [rsp+1E0h] [rbp-1140h]
  __m128i v141; // [rsp+1F0h] [rbp-1130h]
  __m128i v142; // [rsp+200h] [rbp-1120h]
  __m128i v143; // [rsp+210h] [rbp-1110h]
  __m128i v144; // [rsp+220h] [rbp-1100h]
  __m128i v145; // [rsp+230h] [rbp-10F0h]
  __int64 v146; // [rsp+240h] [rbp-10E0h]
  __m128i v147; // [rsp+250h] [rbp-10D0h]
  __m128i v148; // [rsp+260h] [rbp-10C0h]
  __m128i v149; // [rsp+270h] [rbp-10B0h]
  __m128i v150; // [rsp+280h] [rbp-10A0h]
  __m128i v151; // [rsp+290h] [rbp-1090h]
  __m128i v152; // [rsp+2A0h] [rbp-1080h]
  __m128i v153; // [rsp+2B0h] [rbp-1070h]
  __int64 v154; // [rsp+2C0h] [rbp-1060h]
  __m128i v155; // [rsp+2D0h] [rbp-1050h] BYREF
  __m128i v156; // [rsp+2E0h] [rbp-1040h] BYREF
  __m128i v157; // [rsp+2F0h] [rbp-1030h] BYREF
  _BYTE v158[24]; // [rsp+300h] [rbp-1020h] BYREF
  __m128i v159; // [rsp+318h] [rbp-1008h] BYREF
  __m128i v160; // [rsp+328h] [rbp-FF8h] BYREF
  __int64 v161; // [rsp+338h] [rbp-FE8h]
  __int64 v162; // [rsp+340h] [rbp-FE0h]
  __m128i v163; // [rsp+350h] [rbp-FD0h] BYREF
  __m128i v164; // [rsp+360h] [rbp-FC0h] BYREF
  __m128i v165; // [rsp+370h] [rbp-FB0h] BYREF
  _BYTE v166[24]; // [rsp+380h] [rbp-FA0h] BYREF
  __m128i v167; // [rsp+398h] [rbp-F88h] BYREF
  __m128i v168; // [rsp+3A8h] [rbp-F78h] BYREF
  __int64 v169; // [rsp+3B8h] [rbp-F68h]
  __int64 v170; // [rsp+3C0h] [rbp-F60h]
  __m128i v171; // [rsp+3D0h] [rbp-F50h]
  __m128i v172; // [rsp+3E0h] [rbp-F40h]
  __m128i v173; // [rsp+3F0h] [rbp-F30h]
  __m128i v174; // [rsp+400h] [rbp-F20h]
  __m128i v175; // [rsp+410h] [rbp-F10h]
  __m128i v176; // [rsp+420h] [rbp-F00h]
  __m128i v177; // [rsp+430h] [rbp-EF0h]
  __m128i v178[8]; // [rsp+440h] [rbp-EE0h] BYREF
  __m128i v179; // [rsp+4C0h] [rbp-E60h] BYREF
  __m128i v180; // [rsp+4D0h] [rbp-E50h]
  __m128i v181; // [rsp+4E0h] [rbp-E40h]
  __m128i v182; // [rsp+4F0h] [rbp-E30h]
  __m128i v183; // [rsp+500h] [rbp-E20h]
  __m128i v184; // [rsp+510h] [rbp-E10h]
  __m128i v185; // [rsp+520h] [rbp-E00h]
  __m128i v186; // [rsp+530h] [rbp-DF0h]
  __m128i v187; // [rsp+540h] [rbp-DE0h]
  __m128i v188; // [rsp+550h] [rbp-DD0h]
  __m128i v189; // [rsp+560h] [rbp-DC0h]
  __m128i v190; // [rsp+570h] [rbp-DB0h]
  __m128i v191; // [rsp+580h] [rbp-DA0h]
  __m128i v192; // [rsp+590h] [rbp-D90h]
  __m128i v193; // [rsp+5A0h] [rbp-D80h]
  __m128i v194; // [rsp+5B0h] [rbp-D70h]
  __m128i v195; // [rsp+5C0h] [rbp-D60h]
  __m128i v196; // [rsp+5D0h] [rbp-D50h]
  __m128i v197; // [rsp+5E0h] [rbp-D40h]
  __m128i v198; // [rsp+5F0h] [rbp-D30h]
  __m128i v199; // [rsp+600h] [rbp-D20h]
  __m128i v200; // [rsp+610h] [rbp-D10h]
  __m128i v201; // [rsp+620h] [rbp-D00h]
  __m128i v202; // [rsp+630h] [rbp-CF0h]
  __m128i v203; // [rsp+640h] [rbp-CE0h]
  size_t v204[2]; // [rsp+650h] [rbp-CD0h] BYREF
  __m128i v205; // [rsp+660h] [rbp-CC0h]
  __m128i v206; // [rsp+670h] [rbp-CB0h]
  __m128i v207; // [rsp+680h] [rbp-CA0h]
  __m128i v208; // [rsp+690h] [rbp-C90h]
  __m128i v209; // [rsp+6A0h] [rbp-C80h]
  __m128i v210; // [rsp+6B0h] [rbp-C70h]
  __m128i v211; // [rsp+6C0h] [rbp-C60h]
  __m128i v212; // [rsp+6D0h] [rbp-C50h] BYREF
  __m128i v213; // [rsp+6E0h] [rbp-C40h] BYREF
  __m128i v214; // [rsp+6F0h] [rbp-C30h]
  __m128i v215; // [rsp+700h] [rbp-C20h]
  __m128i v216; // [rsp+710h] [rbp-C10h]
  __m128i v217; // [rsp+720h] [rbp-C00h]
  __m128i v218; // [rsp+730h] [rbp-BF0h]
  __int64 v219; // [rsp+740h] [rbp-BE0h]
  __m128i v220; // [rsp+748h] [rbp-BD8h]
  __m128i v221; // [rsp+758h] [rbp-BC8h]
  __m128i v222; // [rsp+768h] [rbp-BB8h]
  __m128i v223; // [rsp+778h] [rbp-BA8h]
  __m128i v224; // [rsp+788h] [rbp-B98h]
  __m128i v225; // [rsp+798h] [rbp-B88h]
  __m128i v226; // [rsp+7A8h] [rbp-B78h]
  __int64 v227; // [rsp+7B8h] [rbp-B68h]
  char *v228; // [rsp+AE0h] [rbp-840h]
  __int64 v229; // [rsp+AE8h] [rbp-838h]
  char v230; // [rsp+AF0h] [rbp-830h] BYREF

  v6 = (_QWORD *)a1;
  v8 = sub_C92610();
  v9 = sub_C92860((__int64 *)(a1 + 40), a3, a4, v8);
  if ( v9 == -1
    || (v11 = *(_QWORD *)(a1 + 40),
        v12 = (__int64 *)(v11 + 8LL * v9),
        v12 == (__int64 *)(v11 + 8LL * *(unsigned int *)(a1 + 48))) )
  {
    sub_25D1D80(a1, a2, (unsigned __int8 *)a3, a4, a5);
    return;
  }
  v13 = *(_QWORD *)(a1 + 24);
  v14 = v6[4];
  v15 = v6[2];
  v122[2] = v6[1];
  v122[0] = v13;
  v122[4] = a5;
  v122[5] = v14;
  v122[3] = v15;
  v122[1] = a2;
  v16 = *v12;
  v228 = &v230;
  v17 = *(_QWORD **)(v16 + 16);
  v18 = *(unsigned int *)(v16 + 32);
  v229 = 0x8000000000LL;
  v19 = &v17[v18];
  v112 = v19;
  if ( *(_DWORD *)(v16 + 24) && v17 != v19 )
  {
    while ( (*v17 & 0xFFFFFFFFFFFFFFF0LL) == 0xFFFFFFFFFFFFFFF0LL )
    {
      if ( v19 == ++v17 )
        return;
    }
    if ( v17 != v19 )
    {
LABEL_10:
      for ( i = v17 + 1; v112 != i; ++i )
      {
        if ( (*i & 0xFFFFFFFFFFFFFFF0LL) != 0xFFFFFFFFFFFFFFF0LL )
          break;
      }
      v21 = *(unsigned int *)(a2 + 24);
      v22 = (__int64 *)(*v17 & 0xFFFFFFFFFFFFFFF8LL);
      v23 = *(_QWORD *)(a2 + 8);
      v24 = *v22;
      if ( (_DWORD)v21 )
      {
        v25 = (v21 - 1) & (((0xBF58476D1CE4E5B9LL * v24) >> 31) ^ (484763065 * v24));
        v26 = (__int64 *)(v23 + 16LL * v25);
        v27 = *v26;
        if ( v24 == *v26 )
        {
LABEL_15:
          v21 = v23 + 16 * v21;
          if ( v26 != (__int64 *)v21 )
          {
            if ( ((unsigned __int8 (__fastcall *)(_QWORD, __int64, __int64))v6[1])(v6[2], v24, v26[1]) )
              goto LABEL_59;
            v22 = (__int64 *)(*v17 & 0xFFFFFFFFFFFFFFF8LL);
          }
        }
        else
        {
          v80 = 1;
          while ( v27 != -1 )
          {
            v10 = (unsigned int)(v80 + 1);
            v25 = (v21 - 1) & (v80 + v25);
            v26 = (__int64 *)(v23 + 16LL * v25);
            v27 = *v26;
            if ( v24 == *v26 )
              goto LABEL_15;
            v80 = v10;
          }
        }
      }
      sub_25CD050(
        (__int64)&v131,
        v22[3],
        (v22[4] - v22[3]) >> 3,
        v21,
        v10,
        v24,
        v6[3],
        v22[3],
        (v22[4] - v22[3]) >> 3,
        (__int64)a3,
        a4);
      v28 = _mm_loadu_si128(&v131);
      v29 = _mm_loadu_si128(&v136);
      v30 = _mm_loadu_si128(&v137);
      v31 = _mm_loadu_si128(&v135);
      v32 = _mm_loadu_si128(&v132);
      v126 = v134;
      v123 = v28;
      v33 = _mm_loadu_si128(&v133);
      v124 = v32;
      v125 = v33;
      v130 = v138;
      v191.m128i_i64[0] = v138;
      v183.m128i_i64[0] = v138;
      v199.m128i_i64[0] = v138;
      v207.m128i_i64[0] = v138;
      v127 = v31;
      v128 = v29;
      v129 = v30;
      v188 = v31;
      v189 = v29;
      v190 = v30;
      v180 = v31;
      v181 = v29;
      v182 = v30;
      v196 = v31;
      v197 = v29;
      v198 = v30;
      *(__m128i *)v204 = v31;
      v205 = v29;
      v206 = v30;
      v212 = v31;
      v215.m128i_i64[0] = v138;
      *(_QWORD *)v158 = v138;
      v161 = v138;
      v213 = v29;
      v214 = v30;
      v155 = v31;
      v156 = v29;
      v157 = v30;
      *(__m128i *)&v158[8] = v31;
      v159 = v29;
      v160 = v30;
      sub_25CD250((__int64)&v155);
      v34 = _mm_loadu_si128(&v124);
      v35 = _mm_loadu_si128(&v125);
      v36 = _mm_loadu_si128(&v128);
      v37 = _mm_loadu_si128(&v129);
      v188 = _mm_loadu_si128(&v123);
      v38 = _mm_loadu_si128(&v127);
      v191.m128i_i64[0] = v126;
      v199.m128i_i64[0] = v126;
      v183.m128i_i64[0] = v130;
      v207.m128i_i64[0] = v130;
      v215.m128i_i64[0] = v126;
      *(_QWORD *)v166 = v126;
      v189 = v34;
      v190 = v35;
      v180 = v38;
      v181 = v36;
      v182 = v37;
      v196 = v188;
      v197 = v34;
      v198 = v35;
      *(__m128i *)v204 = v38;
      v205 = v36;
      v206 = v37;
      v212 = v188;
      v213 = v34;
      v214 = v35;
      v163 = v188;
      v164 = v34;
      v165 = v35;
      *(__m128i *)&v166[8] = v38;
      v169 = v130;
      v167 = v36;
      v168 = v37;
      sub_25CD250((__int64)&v163);
      v39 = _mm_loadu_si128(&v165);
      v40 = _mm_loadu_si128(&v155);
      v41 = _mm_loadu_si128(&v164);
      v180 = _mm_loadu_si128(&v163);
      v42 = _mm_loadu_si128((const __m128i *)v166);
      v219 = v170;
      v43 = _mm_loadu_si128((const __m128i *)&v166[16]);
      v44 = _mm_loadu_si128(&v157);
      v182 = v39;
      v45 = _mm_loadu_si128((const __m128i *)&v167.m128i_u64[1]);
      v46 = _mm_loadu_si128((const __m128i *)v158);
      v212 = v180;
      v47 = _mm_loadu_si128((const __m128i *)&v168.m128i_u64[1]);
      v48 = _mm_loadu_si128((const __m128i *)&v158[16]);
      v214 = v39;
      v49 = _mm_loadu_si128((const __m128i *)&v159.m128i_u64[1]);
      v50 = _mm_loadu_si128((const __m128i *)&v160.m128i_u64[1]);
      v181 = v41;
      v51 = _mm_loadu_si128(&v156);
      v183 = v42;
      v184 = v43;
      v185 = v45;
      v186 = v47;
      v213 = v41;
      v215 = v42;
      v216 = v43;
      v217 = v45;
      v218 = v47;
      v220 = v40;
      v221 = v51;
      v227 = v162;
      v146 = v162;
      v178[0].m128i_i64[0] = v162;
      v154 = v170;
      v147 = v180;
      v149 = v39;
      v98 = (__int64 *)v180.m128i_i64[0];
      *(__m128i *)v204 = v180;
      v222 = v44;
      v223 = v46;
      v224 = v48;
      v225 = v49;
      v226 = v50;
      v139 = v40;
      v140 = v51;
      v141 = v44;
      v142 = v46;
      v143 = v48;
      v144 = v49;
      v145 = v50;
      v171 = v40;
      v172 = v51;
      v173 = v44;
      v174 = v46;
      v175 = v48;
      v176 = v49;
      v177 = v50;
      v148 = v41;
      v150 = v42;
      v151 = v43;
      v152 = v45;
      v153 = v47;
      v205 = v41;
      v179.m128i_i64[0] = v170;
      v179.m128i_i16[4] = 256;
      v178[0].m128i_i16[4] = 256;
      s2 = (void *)v39.m128i_i64[0];
      v206 = v39;
      v178[1] = v180;
      v52 = _mm_loadu_si128(&v179);
      v178[3] = v39;
      v53 = _mm_loadu_si128(v178);
      v207 = v42;
      v208 = v43;
      v209 = v45;
      v210 = v47;
      v178[2] = v41;
      v178[4] = v42;
      v178[5] = v43;
      v178[6] = v45;
      v178[7] = v47;
      v187 = v52;
      v188 = v40;
      v189 = v51;
      v190 = v44;
      v191 = v46;
      v192 = v48;
      v193 = v49;
      v194 = v50;
      v195 = v53;
      v211 = v52;
      v196 = v40;
      v197 = v51;
      v198 = v44;
      v199 = v46;
      v54 = v42.m128i_i64[1];
      v200 = v48;
      v116 = v180.m128i_i64[1];
      n = v206.m128i_u64[1];
      v201 = v49;
      v202 = v50;
      v203 = v53;
      if ( v40.m128i_i64[0] == v180.m128i_i64[0] )
        goto LABEL_59;
      v55 = v6;
      j = *(unsigned __int8 *)(v204[1] + 336);
      v57 = (__int64 *)v180.m128i_i64[0];
      v58 = v55;
      while ( 1 )
      {
        v59 = *v57;
        if ( (!(_BYTE)j || *(char *)(v59 + 12) < 0) && (*(_BYTE *)(v59 + 12) & 0xFu) > 0xA )
          goto LABEL_62;
        if ( ((unsigned __int8 (__fastcall *)(_QWORD, _QWORD, __int64, __int64, __int64, __int64))v58[1])(
               v58[2],
               *(_QWORD *)(*v17 & 0xFFFFFFFFFFFFFFF8LL),
               v59,
               j,
               v10,
               v54) )
        {
          break;
        }
        ++v57;
        for ( j = *(unsigned __int8 *)(v116 + 336); v57 != (__int64 *)v42.m128i_i64[1]; ++v57 )
        {
          v60 = *v57;
          if ( !(_BYTE)j || *(char *)(v60 + 12) < 0 )
          {
            switch ( *(_BYTE *)(v60 + 12) & 0xF )
            {
              case 0:
              case 1:
              case 3:
              case 5:
              case 6:
              case 7:
              case 8:
                v61 = *(_DWORD *)(v60 + 8);
                if ( !v61 )
                {
                  v60 = *(_QWORD *)(v60 + 64);
                  v61 = *(_DWORD *)(v60 + 8);
                }
                if ( v61 == 1 )
                {
                  v10 = *(unsigned __int8 *)(v60 + 12);
                  if ( v41.m128i_i64[1] == 1
                    || (unsigned int)(v10 & 0xF) - 7 > 1
                    || *(_QWORD *)(v60 + 32) == n
                    && (!n
                     || (v106 = j,
                         v109 = *(_BYTE *)(v60 + 12),
                         v62 = memcmp(*(const void **)(v60 + 24), s2, n),
                         v10 = v109,
                         j = v106,
                         !v62)) )
                  {
                    v10 &= 0x40u;
                    if ( !(_DWORD)v10 )
                      goto LABEL_38;
                  }
                }
                continue;
              case 2:
              case 4:
              case 9:
              case 0xA:
                continue;
              default:
                goto LABEL_62;
            }
          }
        }
LABEL_38:
        if ( v57 == (__int64 *)v40.m128i_i64[0] )
        {
          v63 = (__int64 *)v42.m128i_i64[1];
          v6 = v58;
          goto LABEL_40;
        }
      }
      v72 = v58;
      v73 = v57;
      v63 = (__int64 *)v42.m128i_i64[1];
      v6 = v72;
      LOBYTE(j) = *(_BYTE *)(v116 + 336);
      if ( (__int64 *)v40.m128i_i64[0] == v73 )
      {
LABEL_40:
        v64 = *v98;
        if ( (!(_BYTE)j || *(char *)(v64 + 12) < 0) && (*(_BYTE *)(v64 + 12) & 0xFu) > 0xA )
          BUG();
        v65 = v98;
        while ( ++v65 != v63 )
        {
          v66 = *v65;
          if ( !(_BYTE)j || *(char *)(v66 + 12) < 0 )
          {
            switch ( *(_BYTE *)(v66 + 12) & 0xF )
            {
              case 0:
              case 1:
              case 3:
              case 5:
              case 6:
              case 7:
              case 8:
                v67 = *(_DWORD *)(v66 + 8);
                if ( !v67 )
                {
                  v66 = *(_QWORD *)(v66 + 64);
                  v67 = *(_DWORD *)(v66 + 8);
                }
                if ( v67 == 1 )
                {
                  v10 = *(unsigned __int8 *)(v66 + 12);
                  if ( v41.m128i_i64[1] == 1 )
                    goto LABEL_55;
                  if ( (unsigned int)(v10 & 0xF) - 7 > 1 )
                    goto LABEL_55;
                  if ( n == *(_QWORD *)(v66 + 32) )
                  {
                    if ( !n )
                      goto LABEL_55;
                    v107 = v63;
                    v110 = j;
                    v114 = v64;
                    v117 = *(_BYTE *)(v66 + 12);
                    v68 = memcmp(*(const void **)(v66 + 24), s2, n);
                    v10 = v117;
                    v64 = v114;
                    LOBYTE(j) = v110;
                    v63 = v107;
                    if ( !v68 )
                    {
LABEL_55:
                      v10 &= 0x40u;
                      if ( !(_DWORD)v10 )
                        goto LABEL_56;
                    }
                  }
                }
                continue;
              case 2:
              case 4:
              case 9:
              case 0xA:
                continue;
              default:
                goto LABEL_62;
            }
          }
        }
      }
      else
      {
        v64 = *v73;
        if ( (!(_BYTE)j || *(char *)(v64 + 12) < 0) && (*(_BYTE *)(v64 + 12) & 0xFu) > 0xA )
LABEL_62:
          BUG();
      }
LABEL_56:
      v69 = *(const void **)(v64 + 24);
      v70 = *(_QWORD *)(v64 + 32);
      v204[0] = (size_t)v69;
      v204[1] = v70;
      if ( v70 == a4 )
      {
        v120 = v64;
        if ( !a4 )
          goto LABEL_59;
        v71 = memcmp(v69, a3, a4);
        v64 = v120;
        if ( !v71 )
          goto LABEL_59;
      }
      v118 = v64;
      sub_25D0260(a5, v204[0], v204[1], *(_QWORD *)(*v17 & 0xFFFFFFFFFFFFFFF8LL));
      v212.m128i_i64[0] = (__int64)&v213;
      v212.m128i_i64[1] = 0x8000000000LL;
      sub_25D06E0((__int64)v122, v118, (__int64)&v212);
      for ( k = v212.m128i_u32[2]; v212.m128i_i32[2]; k = v212.m128i_u32[2] )
      {
        v75 = *(_QWORD *)(v212.m128i_i64[0] + 8LL * k - 8);
        v212.m128i_i32[2] = k - 1;
        sub_25D06E0((__int64)v122, v75, (__int64)&v212);
      }
      if ( (__m128i *)v212.m128i_i64[0] != &v213 )
        _libc_free(v212.m128i_u64[0]);
      v76 = v6[4];
      v121 = v76;
      if ( !v76 )
        goto LABEL_59;
      v77 = *(_DWORD *)(v76 + 24);
      if ( !v77 )
      {
        v212.m128i_i64[0] = 0;
        ++*(_QWORD *)v76;
        goto LABEL_75;
      }
      v115 = *(_DWORD *)(v76 + 24);
      v119 = *(_QWORD *)(v76 + 8);
      v81 = sub_C94890((_QWORD *)v204[0], v204[1]);
      v82 = (char *)v204[0];
      v83 = 0;
      v84 = v204[1];
      v85 = v115 - 1;
      v86 = 1;
      for ( m = (v115 - 1) & v81; ; m = v85 & v91 )
      {
        v88 = v119 + 48LL * m;
        v89 = *(const void **)v88;
        if ( *(_QWORD *)v88 == -1 )
        {
          if ( v82 == (char *)-1LL )
          {
LABEL_93:
            m128i_i64 = v88 + 16;
            goto LABEL_94;
          }
LABEL_97:
          if ( v83 )
            v88 = v83;
          v77 = *(_DWORD *)(v121 + 24);
          v212.m128i_i64[0] = v88;
          ++*(_QWORD *)v121;
          v79 = *(_DWORD *)(v121 + 16) + 1;
          if ( 4 * v79 < 3 * v77 )
          {
            if ( v77 - (v79 + *(_DWORD *)(v121 + 20)) > v77 >> 3 )
              goto LABEL_101;
            v78 = v77;
LABEL_76:
            sub_25CE770(v121, v78);
            sub_25CE0A0(v121, (__int64)v204, &v212);
            v79 = *(_DWORD *)(v121 + 16) + 1;
LABEL_101:
            *(_DWORD *)(v121 + 16) = v79;
            v94 = (__m128i *)v212.m128i_i64[0];
            if ( *(_QWORD *)v212.m128i_i64[0] != -1 )
              --*(_DWORD *)(v121 + 20);
            v95 = _mm_loadu_si128((const __m128i *)v204);
            m128i_i64 = (__int64)v94[1].m128i_i64;
            v94[1].m128i_i64[0] = 0;
            v94[1].m128i_i64[1] = 0;
            v94[2].m128i_i64[0] = 0;
            v94[2].m128i_i32[2] = 0;
            *v94 = v95;
LABEL_94:
            sub_25CF280((__int64)&v212, m128i_i64, v17);
LABEL_59:
            if ( v112 != i )
            {
              v17 = i;
              goto LABEL_10;
            }
            return;
          }
LABEL_75:
          v78 = 2 * v77;
          goto LABEL_76;
        }
        v90 = v82 + 2 == 0;
        if ( v89 != (const void *)-2LL )
        {
          if ( v84 != *(_QWORD *)(v88 + 8) )
            goto LABEL_87;
          v96 = v86;
          v99 = v83;
          v100 = m;
          s2a = v85;
          if ( !v84 )
            goto LABEL_93;
          v108 = v119 + 48LL * m;
          v111 = v84;
          v92 = memcmp(v82, v89, v84);
          v84 = v111;
          v88 = v108;
          v85 = s2a;
          m = v100;
          v90 = v92 == 0;
          v83 = v99;
          v86 = v96;
        }
        if ( v90 )
          goto LABEL_93;
        if ( v89 == (const void *)-1LL )
          goto LABEL_97;
LABEL_87:
        if ( v89 != (const void *)-2LL || v83 )
          v88 = v83;
        v91 = v86 + m;
        v83 = v88;
        ++v86;
      }
    }
  }
}
