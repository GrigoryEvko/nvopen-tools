// Function: sub_6040F0
// Address: 0x6040f0
//
__int64 __fastcall sub_6040F0(
        __int64 a1,
        __int64 a2,
        int a3,
        int a4,
        _QWORD *a5,
        _DWORD *a6,
        _QWORD *a7,
        _QWORD *a8,
        __int64 a9,
        __m128i *a10)
{
  __m128i v12; // xmm6
  __m128i v13; // xmm4
  __int64 v14; // rax
  _QWORD *v15; // r15
  bool v16; // al
  __int64 v17; // rcx
  __int64 v18; // rdi
  __int16 v19; // bx
  __int64 v20; // rsi
  char v21; // bl
  __m128i v22; // xmm1
  __m128i v23; // xmm2
  __m128i v24; // xmm3
  __m128i v25; // xmm4
  __int64 v26; // rax
  __int64 v28; // rax
  _QWORD *v29; // r13
  _QWORD *v30; // r15
  __int64 v31; // rcx
  __int64 v32; // rcx
  __int64 v33; // rdx
  __int64 v34; // r8
  __int64 v35; // r9
  __int64 v36; // rcx
  __int64 v37; // rax
  __int64 v38; // rdx
  _QWORD *v39; // rsi
  __int64 v40; // rbx
  char v41; // al
  char v42; // al
  unsigned __int16 v43; // dx
  __m128i v44; // xmm6
  __m128i v45; // xmm2
  __m128i v46; // xmm4
  __int64 v47; // rax
  _QWORD *v48; // r14
  __int64 v49; // rbx
  __int64 v50; // rax
  __int64 v51; // rax
  unsigned int v52; // r15d
  __int64 v53; // r13
  _BOOL8 v54; // rbx
  __int64 v55; // rax
  __int64 v56; // rax
  __int64 v57; // rax
  __int64 v58; // r8
  __int64 v59; // rdi
  __int64 v60; // rdi
  __int64 v61; // rsi
  __int64 v62; // rcx
  __int64 v63; // r14
  __int64 v64; // rax
  __int8 v65; // al
  __int64 v66; // rdx
  __int64 v67; // r12
  char v68; // al
  char v69; // bl
  __int64 v70; // rdx
  __int64 v71; // rcx
  __int64 v72; // rsi
  __int64 v73; // rdi
  _QWORD *v74; // r14
  __int64 v75; // rbx
  __int64 j; // r14
  __int64 v77; // rax
  char v78; // al
  __int64 v79; // rbx
  __int64 v80; // rax
  __int64 v81; // rax
  __int64 v82; // r11
  _BOOL4 v83; // eax
  __int64 v84; // rdx
  __int64 k; // rbx
  __int64 v86; // rdx
  __int64 v87; // rdx
  __m128i *v88; // rsi
  __int64 v89; // rdx
  __int64 v90; // rcx
  __int64 v91; // rax
  __int64 v92; // rax
  bool v93; // r14
  bool v94; // bl
  __int64 v95; // rax
  __m128i v96; // xmm1
  __m128i v97; // xmm2
  __m128i v98; // xmm4
  __int64 v99; // rax
  int v100; // eax
  __int64 v101; // rcx
  int v102; // r8d
  __int64 v103; // rax
  __int64 v104; // rdx
  __int16 v105; // ax
  __int64 v106; // rdx
  int v107; // ebx
  int v108; // r12d
  __int64 v109; // rdx
  __int64 v110; // rcx
  _QWORD *v111; // r14
  _BYTE *v112; // rax
  __int64 v113; // rdx
  __int64 v114; // rcx
  char v115; // al
  __int64 v116; // rsi
  __int64 *v117; // rdx
  char v118; // r13
  __int64 v119; // r11
  __int16 v120; // r12
  __int64 v121; // rax
  __int64 v122; // rdx
  __int64 v123; // rcx
  __int64 v124; // rax
  int v125; // eax
  __int64 v126; // rdi
  __int64 v127; // rax
  __int64 v128; // rdx
  __int64 v129; // rcx
  __int64 v130; // rax
  unsigned int v131; // eax
  int v132; // eax
  __int64 v133; // rdx
  int i; // eax
  __int64 v135; // rax
  __int64 v136; // rax
  __int64 v137; // r14
  __int64 *v138; // rbx
  __int64 v139; // r14
  char v140; // dl
  __int64 v141; // rax
  int v142; // eax
  __int64 v143; // rax
  __int64 v144; // rax
  char v145; // al
  __int64 v146; // rax
  __int64 v147; // r13
  __int64 v148; // rax
  __int64 v149; // rax
  __int64 v150; // rdx
  __int64 v151; // rcx
  int v152; // eax
  int v153; // eax
  __int64 v154; // rax
  char v155; // ah
  int v156; // r8d
  unsigned __int8 v157; // al
  char *v158; // rcx
  __int64 v159; // rdi
  __int64 v160; // rax
  __int64 v161; // rdx
  __int64 v162; // rax
  int v163; // eax
  int v164; // eax
  bool v165; // cl
  __int64 v166; // r13
  __int64 v167; // rax
  __int64 v168; // rdx
  int v169; // eax
  __int64 v170; // r14
  __int64 v171; // rdi
  __int64 v172; // rbx
  __int64 v173; // rdx
  __int64 v174; // rcx
  __int64 v175; // r12
  __int64 v176; // rax
  __int64 v177; // r14
  __int64 v178; // rax
  __int64 v179; // rcx
  int v180; // eax
  __int64 v181; // r10
  unsigned int v182; // r15d
  __int64 v183; // rcx
  int v184; // r8d
  unsigned int v185; // r9d
  char v186; // al
  __int64 v187; // rax
  __int64 v188; // rax
  int v189; // eax
  __int64 v190; // rdx
  __int64 v191; // rcx
  __int64 v192; // rdx
  __int64 v193; // rcx
  int v194; // eax
  __int64 v195; // [rsp-10h] [rbp-450h]
  __int64 v196; // [rsp-8h] [rbp-448h]
  int v199; // [rsp+1Ch] [rbp-424h]
  __int64 v200; // [rsp+20h] [rbp-420h]
  __int64 v201; // [rsp+28h] [rbp-418h]
  _BYTE *v203; // [rsp+38h] [rbp-408h]
  __int64 v204; // [rsp+40h] [rbp-400h]
  __int64 v205; // [rsp+48h] [rbp-3F8h]
  char v206; // [rsp+53h] [rbp-3EDh]
  int v207; // [rsp+54h] [rbp-3ECh]
  _QWORD *v208; // [rsp+58h] [rbp-3E8h]
  __int64 v209; // [rsp+60h] [rbp-3E0h]
  __int16 v210; // [rsp+68h] [rbp-3D8h]
  char v211; // [rsp+6Bh] [rbp-3D5h]
  int v212; // [rsp+6Ch] [rbp-3D4h]
  __int64 v213; // [rsp+70h] [rbp-3D0h]
  char v214; // [rsp+83h] [rbp-3BDh]
  __int64 v216; // [rsp+88h] [rbp-3B8h]
  __int16 v217; // [rsp+90h] [rbp-3B0h]
  __int64 v219; // [rsp+A0h] [rbp-3A0h]
  char v220; // [rsp+A0h] [rbp-3A0h]
  _BOOL4 v221; // [rsp+A8h] [rbp-398h]
  __int64 v222; // [rsp+A8h] [rbp-398h]
  __int64 v223; // [rsp+A8h] [rbp-398h]
  __int64 v224; // [rsp+B0h] [rbp-390h]
  _BOOL4 v225; // [rsp+B0h] [rbp-390h]
  __int64 v226; // [rsp+B0h] [rbp-390h]
  __int64 v227; // [rsp+B0h] [rbp-390h]
  __int64 v228; // [rsp+B0h] [rbp-390h]
  __int64 v229; // [rsp+B0h] [rbp-390h]
  __int64 v230; // [rsp+B0h] [rbp-390h]
  __int64 v231; // [rsp+B0h] [rbp-390h]
  __int64 v232; // [rsp+B0h] [rbp-390h]
  __int64 v233; // [rsp+B0h] [rbp-390h]
  __int64 v234; // [rsp+B0h] [rbp-390h]
  __int64 v235; // [rsp+B0h] [rbp-390h]
  __int64 v236; // [rsp+B0h] [rbp-390h]
  __int64 v237; // [rsp+B0h] [rbp-390h]
  __int64 v238; // [rsp+B0h] [rbp-390h]
  __int64 v239; // [rsp+B0h] [rbp-390h]
  __int64 v240; // [rsp+B0h] [rbp-390h]
  __int64 v241; // [rsp+B0h] [rbp-390h]
  __int64 v242; // [rsp+B0h] [rbp-390h]
  __int64 v243; // [rsp+B0h] [rbp-390h]
  __int64 v244; // [rsp+B0h] [rbp-390h]
  __int64 v245; // [rsp+B0h] [rbp-390h]
  __int64 v246; // [rsp+B0h] [rbp-390h]
  __int64 v247; // [rsp+B0h] [rbp-390h]
  __int64 v248; // [rsp+B0h] [rbp-390h]
  __int64 v249; // [rsp+B0h] [rbp-390h]
  unsigned int v250; // [rsp+B0h] [rbp-390h]
  __int64 v251; // [rsp+B0h] [rbp-390h]
  __int64 v252; // [rsp+B0h] [rbp-390h]
  __int64 v253; // [rsp+B0h] [rbp-390h]
  __int64 v254; // [rsp+B0h] [rbp-390h]
  __int64 v255; // [rsp+B0h] [rbp-390h]
  __int64 v256; // [rsp+B0h] [rbp-390h]
  __int64 v257; // [rsp+B0h] [rbp-390h]
  __int64 v258; // [rsp+B0h] [rbp-390h]
  _QWORD *v259; // [rsp+C0h] [rbp-380h]
  _BOOL4 v260; // [rsp+C0h] [rbp-380h]
  unsigned int v261; // [rsp+C0h] [rbp-380h]
  __int64 v262; // [rsp+C0h] [rbp-380h]
  int v263; // [rsp+C0h] [rbp-380h]
  __int64 v264; // [rsp+C0h] [rbp-380h]
  bool v265; // [rsp+C0h] [rbp-380h]
  __int64 v266; // [rsp+C0h] [rbp-380h]
  unsigned int v267; // [rsp+E0h] [rbp-360h] BYREF
  unsigned int v268; // [rsp+E4h] [rbp-35Ch] BYREF
  __int64 v269; // [rsp+E8h] [rbp-358h] BYREF
  __m128i v270; // [rsp+F0h] [rbp-350h] BYREF
  __m128i v271; // [rsp+100h] [rbp-340h] BYREF
  __m128i v272; // [rsp+110h] [rbp-330h] BYREF
  __m128i v273; // [rsp+120h] [rbp-320h]
  __m128i v274; // [rsp+130h] [rbp-310h]
  __m128i v275; // [rsp+140h] [rbp-300h]
  __m128i v276; // [rsp+150h] [rbp-2F0h] BYREF
  __m128i v277; // [rsp+160h] [rbp-2E0h] BYREF
  __m128i v278; // [rsp+170h] [rbp-2D0h] BYREF
  __m128i v279; // [rsp+180h] [rbp-2C0h] BYREF
  __m128i v280; // [rsp+190h] [rbp-2B0h] BYREF
  __m128i v281; // [rsp+1A0h] [rbp-2A0h] BYREF
  __int64 v282; // [rsp+1B0h] [rbp-290h]
  _QWORD v283[59]; // [rsp+1C0h] [rbp-280h] BYREF
  __m128i v284; // [rsp+398h] [rbp-A8h] BYREF
  __m128i v285; // [rsp+3A8h] [rbp-98h] BYREF
  __m128i v286; // [rsp+3B8h] [rbp-88h] BYREF
  __m128i v287; // [rsp+3C8h] [rbp-78h] BYREF
  __m128i v288; // [rsp+3D8h] [rbp-68h] BYREF
  __int64 v289; // [rsp+3E8h] [rbp-58h]
  char v290; // [rsp+3F0h] [rbp-50h]
  char v291; // [rsp+3F1h] [rbp-4Fh]

  v213 = *(_QWORD *)a1;
  sub_854430();
  v12 = _mm_loadu_si128(&xmmword_4F06660[3]);
  v13 = _mm_loadu_si128(&xmmword_4F06660[2]);
  v272.m128i_i64[0] = _mm_loadu_si128(xmmword_4F06660).m128i_u64[0];
  v273 = _mm_loadu_si128(&xmmword_4F06660[1]);
  v14 = unk_4F077C8;
  *a6 = 0;
  v15 = v283;
  v274 = v13;
  v272.m128i_i64[1] = v14;
  v275 = v12;
  sub_5E4C60((__int64)v283, &dword_4F063F8);
  v203 = 0;
  if ( a2 )
  {
    v203 = *(_BYTE **)a2;
    qmemcpy(v283, *(const void **)a2, sizeof(v283));
    v283[19] = v283;
    v283[54] = 0;
    *(_QWORD *)a2 = v283;
  }
  v206 = *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 4);
  BYTE2(v283[15]) = (8 * (v206 == 9)) | BYTE2(v283[15]) & 0xF7;
  v16 = 0;
  if ( *(_BYTE *)(v213 + 140) == 9 )
    v16 = (*(_BYTE *)(*(_QWORD *)(v213 + 168) + 109LL) & 0x20) != 0;
  *(_QWORD *)((char *)&v283[15] + 5) = *(_QWORD *)((char *)&v283[15] + 5) & 0x7FF7FFFFFFFFFFBFLL
                                     | ((unsigned __int64)v16 << 51)
                                     | 0x8000000000000040LL;
  if ( a4 )
    BYTE6(v283[15]) |= 4u;
  if ( (v283[16] & 0x8000000LL) != 0 )
  {
    v18 = (__int64)&dword_4F063F8;
    v20 = 0;
    v283[23] = sub_5E64E0();
    v28 = sub_72B6D0(&dword_4F063F8, 0);
    v290 &= 0xF5u;
    v21 = BYTE4(v283[33]);
    v283[34] = v28;
    v283[36] = v28;
    v204 = 0;
    v217 = 0;
    BYTE1(v283[15]) |= 0x40u;
    v283[4] = *(_QWORD *)&dword_4F063F8;
    v283[3] = *(_QWORD *)&dword_4F063F8;
    v201 = 0;
    v209 = 0;
    v205 = 0;
    v283[6] = *(_QWORD *)&dword_4F063F8;
    v283[5] = *(_QWORD *)&dword_4F063F8;
    goto LABEL_39;
  }
  if ( dword_4F077C4 == 2 && !(a3 | (v206 == 9)) )
  {
    sub_643C90(v283);
    v283[23] = sub_5CC190(1);
    if ( dword_4F077C4 == 2 )
    {
LABEL_35:
      v18 = 2231;
      goto LABEL_11;
    }
LABEL_10:
    v18 = 2054;
    goto LABEL_11;
  }
  v283[23] = sub_5CC190(1);
  if ( dword_4F077C4 != 2 )
    goto LABEL_10;
  if ( !a3 )
    goto LABEL_35;
  v291 |= 2u;
  v18 = 2743;
LABEL_11:
  if ( dword_4D043F8 )
    v18 |= 0x8000000uLL;
  if ( dword_4D043E0 )
    v18 |= 0x400000uLL;
  if ( word_4F06418[0] == 187 )
  {
    BYTE6(v283[15]) |= 4u;
    sub_7B8B50(v18 | 0x40000, v213, &dword_4D043E0, v17);
    v18 |= 0x40000uLL;
  }
  if ( dword_4F077B8 )
    v18 |= 0x10uLL;
  ++*(_BYTE *)(qword_4F061C8 + 63LL);
  sub_672A20(v18, v283, &v284);
  v19 = v283[1];
  v217 = v283[1];
  --*(_BYTE *)(qword_4F061C8 + 63LL);
  v201 = v19 & 0x100;
  v209 = v19 & 1;
  v205 = v19 & 8;
  if ( (v19 & 8) != 0 )
    *(_BYTE *)(a1 + 8) |= 0x10u;
  v20 = v217 & 0x1000;
  v204 = v20;
  v21 = BYTE4(v283[33]);
  v290 = v290 & 0xF5 | (2 * ((v217 & 0x400) != 0)) | (8 * ((v217 & 0x800) != 0));
  if ( dword_4F077C4 == 2 && (v217 & 0x20) != 0 )
  {
    v18 = (__int64)v283;
    sub_643D30(v283);
  }
  if ( (v217 & 0x80) != 0 )
  {
    *(_QWORD *)dword_4F07508 = *(_QWORD *)&dword_4F063F8;
    if ( (v217 & 0x10) == 0 )
      sub_6851C0(40, dword_4F07508);
    sub_6851C0(65, dword_4F07508);
    sub_854B40();
    *a6 = 1;
    goto LABEL_26;
  }
LABEL_39:
  if ( word_4F06418[0] != 75 )
  {
    v199 = 0;
    v200 = v283[36];
    goto LABEL_41;
  }
  v118 = BYTE5(v283[33]);
  v119 = v283[34];
  v120 = v283[1];
  if ( BYTE5(v283[33]) || (v18 = v283[34], v230 = v283[34], v131 = sub_8D23B0(v283[34]), v119 = v230, v131) )
  {
LABEL_381:
    if ( dword_4F077C4 != 2 )
    {
      if ( dword_4F077C4 == 1 )
        goto LABEL_389;
      if ( (v120 & 0x20) != 0 )
      {
        v18 = 5;
        if ( dword_4D04964 )
          v18 = byte_4F07472[0];
        v20 = 40;
        v228 = v119;
        sub_684AA0(v18, 40, &dword_4F063F8);
        v119 = v228;
        goto LABEL_389;
      }
      if ( dword_4F077C0 )
      {
        if ( (v120 & 0x100) != 0 )
        {
          v121 = v283[23];
          if ( v283[23] )
          {
            v226 = v119;
            goto LABEL_388;
          }
        }
      }
      v18 = 5;
      if ( dword_4D04964 )
        v18 = byte_4F07472[0];
      goto LABEL_472;
    }
LABEL_446:
    if ( (v120 & 0x1000) != 0 )
    {
      v20 = (__int64)&v283[4];
      v18 = 719;
      v236 = v119;
      sub_6851C0(719, &v283[4]);
      v119 = v236;
    }
    if ( (v120 & 8) == 0 )
    {
      if ( (v291 & 2) != 0 )
      {
        if ( !v283[0] )
        {
          v20 = (__int64)&v283[4];
          v18 = 787;
          v229 = v119;
          sub_6851C0(787, &v283[4]);
          v291 |= 0x20u;
          v119 = v229;
        }
        goto LABEL_389;
      }
      if ( (v120 & 0x10) != 0 )
      {
        if ( v118 == 4 )
        {
          v18 = 5;
          if ( dword_4D04964 )
            v18 = byte_4F07472[0];
          v20 = 375;
          v240 = v119;
          sub_684AA0(v18, 375, &dword_4F063F8);
          v119 = v240;
        }
        else if ( v118 )
        {
          v20 = 80;
          v233 = v119;
          v18 = qword_4D0495C == 0 ? 7 : 5;
          sub_684AA0(v18, 80, &v283[4]);
          v119 = v233;
        }
        if ( (v120 & 4) != 0 )
        {
          v20 = (__int64)&v283[4];
          v18 = 377;
          v243 = v119;
          sub_6851C0(377, &v283[4]);
          v119 = v243;
        }
        if ( (v120 & 2) != 0 )
        {
          v20 = (__int64)&v283[4];
          v18 = 326;
          v242 = v119;
          sub_6851C0(326, &v283[4]);
          v119 = v242;
        }
        if ( (v120 & 0x2000) != 0 )
        {
          v20 = (__int64)&v283[4];
          v18 = 771;
          v241 = v119;
          sub_6851C0(771, &v283[4]);
          v119 = v241;
        }
        if ( (*(_BYTE *)(v119 + 140) & 0xFB) != 8 )
        {
          if ( (v120 & 0x30) == 0 )
            goto LABEL_390;
          goto LABEL_395;
        }
        v18 = v119;
        v227 = v119;
        v20 = dword_4F077C4 != 2;
        v152 = sub_8D4C10(v119, v20);
        v119 = v227;
        if ( !v152 )
          goto LABEL_389;
        v18 = 5;
        if ( dword_4D04964 )
          v18 = byte_4F07472[0];
      }
      else
      {
        if ( v118 == 4 )
        {
          v18 = 5;
          if ( dword_4D04964 )
            v18 = byte_4F07472[0];
          v20 = 375;
          v239 = v119;
          sub_684AA0(v18, 375, &dword_4F063F8);
          v119 = v239;
          goto LABEL_389;
        }
        if ( (v120 & 0x20) == 0 )
        {
          if ( dword_4F077BC )
          {
            if ( (v120 & 0x100) != 0 )
            {
              v121 = v283[23];
              v226 = v119;
              if ( !v283[23] )
              {
                v20 = (__int64)&v283[4];
                v18 = 64;
                sub_684B30(64, &v283[4]);
                v119 = v226;
                goto LABEL_389;
              }
LABEL_388:
              v20 = v121 + 56;
              v18 = 2274;
              sub_684B30(2274, v121 + 56);
              v119 = v226;
              goto LABEL_389;
            }
            v18 = (v120 & 0x200) == 0 ? 8 : 5;
          }
          else
          {
            v18 = 8;
          }
LABEL_472:
          v20 = 64;
          v232 = v119;
          sub_684AA0(v18, 64, &v283[4]);
          v119 = v232;
          goto LABEL_389;
        }
        v18 = 5;
        if ( dword_4D04964 )
          v18 = byte_4F07472[0];
        v20 = 64;
        v227 = v119;
        sub_684AA0(v18, 64, &v283[4]);
        v119 = v227;
        if ( (*(_BYTE *)(v227 + 140) & 0xFB) != 8 )
        {
          if ( (v120 & 0x30) == 0 )
            goto LABEL_390;
          goto LABEL_426;
        }
        v18 = v227;
        v20 = dword_4F077C4 != 2;
        v169 = sub_8D4C10(v227, v20);
        v119 = v227;
        if ( !v169 )
          goto LABEL_389;
        v18 = 5;
        if ( dword_4D04964 )
          v18 = byte_4F07472[0];
      }
      v20 = 21;
      sub_684AA0(v18, 21, &v283[4]);
      v119 = v227;
      goto LABEL_389;
    }
    v139 = v283[34];
    v140 = *(_BYTE *)(v283[34] + 140LL);
    if ( v140 == 12 )
    {
      v141 = v283[34];
      do
      {
        v141 = *(_QWORD *)(v141 + 160);
        v140 = *(_BYTE *)(v141 + 140);
      }
      while ( v140 == 12 );
    }
    if ( !v140 )
      goto LABEL_389;
    if ( (v283[1] & 1) != 0 )
    {
      v18 = v283[34];
      v234 = v119;
      v142 = sub_8D3A70(v283[34]);
      v119 = v234;
      if ( (v142 || (v18 = v139, v163 = sub_8D3D40(v139), v119 = v234, v163))
        && (*(_BYTE *)(v139 + 140) != 12 || (v18 = v139, v258 = v119, v194 = sub_8D4C10(v139, 1), v119 = v258, !v194)) )
      {
        if ( dword_4F04C44 == -1 )
        {
          if ( !unk_4F0776C )
          {
            v155 = BYTE1(v283[1]);
            if ( (v283[1] & 0x10000) == 0 )
            {
              if ( (v283[1] & 6) != 0 )
              {
                v255 = v119;
                sub_6851C0(277, &v283[3]);
                v119 = v255;
                goto LABEL_594;
              }
              v156 = *(_DWORD *)(v213 + 176) & 0x19000;
LABEL_593:
              if ( (v155 & 2) != 0 )
                goto LABEL_594;
              while ( 1 )
              {
                v157 = *(_BYTE *)(v139 + 140);
                if ( v157 != 12 )
                  break;
                v139 = *(_QWORD *)(v139 + 160);
              }
              if ( unk_4F0776C )
              {
LABEL_611:
                if ( v156 != 4096 )
                {
                  if ( *(_BYTE *)(v139 + 140) == 14 )
                  {
                    if ( !dword_4F07590 )
                      goto LABEL_594;
                    v247 = v119;
                    v160 = sub_7CFE40(v139);
                    v119 = v247;
                    v139 = v160;
                  }
                  if ( !dword_4F04C3C )
                  {
                    v264 = v119;
                    sub_8756F0(257, *(_QWORD *)v139, &qword_4D04A08, 0);
                    v161 = *(_QWORD *)(sub_86A2A0(v139) + 24);
                    *(_BYTE *)(v161 + 57) |= 5u;
                    v248 = v161;
                    v162 = sub_729420(*(_BYTE *)(v161 - 8) & 1, &v284);
                    v119 = v264;
                    *(_QWORD *)(v248 + 8) = v162;
                  }
                }
LABEL_594:
                v18 = v213;
                v20 = v139;
                v244 = v119;
                sub_5ED880(v213, v139, 0, (__int64)&v284);
                v119 = v244;
LABEL_595:
                if ( (v283[1] & 0x10200) != 0x200
                  && dword_4F077BC
                  && (dword_4F077C4 != 2 || unk_4F07778 <= 201102 && !dword_4F07774) )
                {
                  v18 = (__int64)&v283[3];
                  v20 = 2627;
                  v245 = v119;
                  sub_684B40(&v283[3], 2627);
                  v119 = v245;
                }
                goto LABEL_389;
              }
              if ( v157 == 11 )
              {
                v158 = "union";
                goto LABEL_608;
              }
              if ( v157 > 0xBu )
              {
                v158 = "class";
                if ( v157 == 14 )
                  goto LABEL_608;
              }
              else
              {
                if ( v157 == 9 )
                {
                  v158 = "class";
                  goto LABEL_608;
                }
                v158 = "struct";
                if ( v157 == 10 )
                {
LABEL_608:
                  v159 = 4;
                  if ( dword_4D04964 )
                    v159 = byte_4F07472[0];
                  v263 = v156;
                  v246 = v119;
                  sub_6849F0(v159, 451, &qword_4D04A08, v158);
                  v156 = v263;
                  v119 = v246;
                  goto LABEL_611;
                }
              }
              sub_721090(v18);
            }
LABEL_659:
            v20 = (__int64)&v283[3];
            v18 = 1011;
            v256 = v119;
            sub_6851C0(1011, &v283[3]);
            v119 = v256;
            goto LABEL_389;
          }
LABEL_624:
          v18 = v139;
          v249 = v119;
          v164 = sub_8D3A70(v139);
          v119 = v249;
          v165 = 1;
          if ( !v164 )
          {
            v18 = v139;
            v189 = sub_8D3D40(v139);
            v119 = v249;
            v165 = v189 != 0;
          }
          v155 = BYTE1(v283[1]);
          if ( (v283[1] & 0x10000) == 0 || unk_4F0776C )
          {
            if ( (v283[1] & 6) != 0 )
            {
              v20 = (__int64)&v283[3];
              v18 = 277;
              v265 = v165;
              v253 = v119;
              sub_6851C0(277, &v283[3]);
              v119 = v253;
              if ( !v265 )
                goto LABEL_595;
              goto LABEL_594;
            }
            v250 = *(_DWORD *)(v213 + 176);
            v20 = v250;
            v156 = v250 & 0x19000;
            if ( !v165 )
            {
              if ( v156 != 4096 && !dword_4F04C3C )
              {
                v251 = v119;
                v20 = 6;
                v18 = v139;
                v166 = sub_869D30();
                v167 = sub_86A1D0(v139, 6, 0);
                v168 = v283[3];
                v119 = v251;
                *(_BYTE *)(v167 + 57) |= 4u;
                *(_QWORD *)v167 = v168;
                *(_BYTE *)(v166 + 16) = 53;
                *(_QWORD *)(v166 + 24) = v167;
              }
              goto LABEL_595;
            }
            goto LABEL_593;
          }
          goto LABEL_659;
        }
      }
      else if ( unk_4F0776C && (v283[15] & 0x7F) == 0 && dword_4F04C44 == -1 )
      {
        goto LABEL_624;
      }
    }
    v20 = (__int64)&v283[3];
    v18 = 277;
    v235 = v119;
    sub_6851C0(277, &v283[3]);
    v119 = v235;
    goto LABEL_389;
  }
  if ( dword_4F077C4 == 2 )
  {
    v20 = 0;
    if ( *(_BYTE *)(v283[36] + 140LL) == 11 )
    {
      if ( !*(_QWORD *)(v283[36] + 8LL) && (v283[1] & 0x20) != 0 && (v283[1] & 0x18) == 0 )
      {
        v290 |= 0x40u;
        goto LABEL_519;
      }
      if ( (v290 & 0x40) != 0 )
        goto LABEL_519;
      goto LABEL_446;
    }
  }
  else
  {
    v20 = unk_4F07778 > 201111;
  }
  v18 = (unsigned int)qword_4D043AC | HIDWORD(qword_4D043AC);
  if ( !qword_4D043AC )
    goto LABEL_517;
  v18 = v283[36];
  v220 = v283[1];
  v222 = v230;
  v261 = v131;
  v231 = v283[36];
  v132 = sub_8D3A70(v283[36]);
  v133 = v231;
  v119 = v222;
  v20 = (unsigned int)v20;
  if ( !v132 )
    goto LABEL_517;
  i = dword_4F077C4;
  if ( dword_4F077C4 != 2 )
  {
    v18 = dword_4F077C0;
    if ( !dword_4F077C0 || (v18 = (__int64)&qword_4F077A8, qword_4F077A8 <= 0x765Bu) )
    {
      if ( *(_BYTE *)(v231 + 140) != 12 )
      {
LABEL_634:
        v290 |= 0x40u;
        goto LABEL_635;
      }
LABEL_465:
      v18 = v231;
      v136 = sub_8D21F0(v231);
      v133 = v231;
      v119 = v222;
      if ( *(_BYTE *)(v136 + 140) != 12 )
      {
        if ( dword_4F077B8 )
        {
          v137 = v231;
          for ( i = dword_4F077C4; *(_BYTE *)(v137 + 140) == 12; v137 = *(_QWORD *)(v137 + 160) )
            ;
          v20 = v261;
          goto LABEL_663;
        }
        goto LABEL_660;
      }
      if ( dword_4F077C4 == 2 )
      {
        v20 = 0;
        v145 = v290;
        v137 = v231;
        if ( *(_BYTE *)(v231 + 140) != 12 )
          goto LABEL_664;
      }
      else
      {
        if ( HIDWORD(qword_4D043AC) )
        {
          if ( dword_4F077B8 )
          {
            v18 = v231;
            v188 = sub_8D21C0(v231);
            v133 = v231;
            v119 = v222;
            v137 = v188;
LABEL_661:
            v145 = v290;
            if ( *(_BYTE *)(v137 + 140) != 12 )
            {
              v20 = 0;
              i = dword_4F077C4;
              goto LABEL_663;
            }
            goto LABEL_518;
          }
LABEL_660:
          v18 = v231;
          v178 = sub_8D21C0(v231);
          v119 = v222;
          v133 = v231;
          v137 = v178;
          goto LABEL_661;
        }
        v145 = v290;
        if ( *(_BYTE *)(v231 + 140) != 12 )
        {
          v290 |= 0xC0u;
          if ( !dword_4D04964 )
            goto LABEL_651;
          v20 = 619;
LABEL_639:
          v252 = v119;
          v18 = byte_4F07472[0];
          sub_684AA0(byte_4F07472[0], v20, &dword_4F063F8);
          v145 = v290;
          v119 = v252;
        }
      }
LABEL_518:
      if ( (v145 & 0x40) != 0 )
        goto LABEL_519;
      goto LABEL_381;
    }
  }
  if ( (v220 & 0x18) != 0 )
    goto LABEL_517;
  if ( *(_BYTE *)(v231 + 140) == 12 )
  {
    v135 = v231;
    do
      v135 = *(_QWORD *)(v135 + 160);
    while ( *(_BYTE *)(v135 + 140) == 12 );
    if ( (*(_BYTE *)(v135 + 177) & 4) == 0 )
      goto LABEL_517;
    goto LABEL_465;
  }
  if ( (*(_BYTE *)(v231 + 177) & 4) == 0 )
    goto LABEL_517;
  v137 = v231;
  v20 = (unsigned int)v20;
LABEL_663:
  if ( i != 2 )
    goto LABEL_634;
LABEL_664:
  v179 = *(_QWORD *)(*(_QWORD *)v137 + 96LL);
  if ( (*(_BYTE *)(v179 + 178) & 0x40) == 0 && !dword_4F077B4 )
  {
    if ( !dword_4F077BC )
      goto LABEL_669;
    if ( qword_4F077A8 > 0x76BFu )
    {
      v18 = v137;
      v223 = *(_QWORD *)(*(_QWORD *)v137 + 96LL);
      v266 = v133;
      v257 = v119;
      v180 = sub_8E3AD0(v137);
      v119 = v257;
      v133 = v266;
      v179 = v223;
      v20 = (unsigned int)v20;
      if ( !v180 )
        goto LABEL_669;
    }
    else if ( **(_QWORD **)(v137 + 168) )
    {
      goto LABEL_669;
    }
  }
  v181 = *(_QWORD *)v179;
  if ( !*(_QWORD *)v179 )
  {
LABEL_669:
    if ( (v290 & 0x40) != 0 )
      goto LABEL_636;
    goto LABEL_381;
  }
  v182 = v20;
  v20 = v179;
  v290 |= 0x40u;
  v183 = v133;
  v184 = dword_4F077C4;
  v185 = dword_4F07588;
  do
  {
    v186 = *(_BYTE *)(v181 + 80);
    if ( v186 != 8 && *(_QWORD *)(v20 + 16) != v181 )
    {
      if ( v186 != 3 && (v184 != 2 || (unsigned __int8)(v186 - 4) > 2u)
        || v183 != v137 && (!v185 || (v187 = *(_QWORD *)(v137 + 32), *(_QWORD *)(v183 + 32) != v187) || !v187) )
      {
        v18 = v181;
        if ( !(unsigned int)sub_5E4A50(v181) )
        {
          v290 &= ~0x40u;
          v15 = v283;
          goto LABEL_381;
        }
      }
    }
    v181 = *(_QWORD *)(v181 + 16);
  }
  while ( v181 );
  v20 = v182;
  v15 = v283;
LABEL_635:
  if ( (v290 & 0x40) != 0 )
  {
LABEL_636:
    v145 = v290 | 0x80;
    v290 |= 0x80u;
    if ( (_DWORD)v20 )
      goto LABEL_518;
    if ( dword_4D04964 )
    {
      v20 = (unsigned int)(dword_4F077C4 == 2) + 619;
      goto LABEL_639;
    }
LABEL_651:
    if ( dword_4F077B8 && unk_4D04320 )
    {
      v20 = (__int64)&dword_4F063F8;
      v254 = v119;
      v18 = (unsigned int)(dword_4F077C4 == 2) + 619;
      sub_684B30(v18, &dword_4F063F8);
      v145 = v290;
      v119 = v254;
      goto LABEL_518;
    }
LABEL_517:
    v145 = v290;
    goto LABEL_518;
  }
LABEL_519:
  v18 = v119;
  v237 = v119;
  v146 = sub_8D21F0(v119);
  v119 = v237;
  v147 = v146;
  if ( v290 < 0 )
  {
    if ( *(_QWORD *)(v146 + 8) )
    {
      if ( (v120 & 0x20) == 0 )
      {
        v18 = 1;
        sub_8756F0(1, *(_QWORD *)v146, &v283[4], 0);
        v119 = v237;
      }
    }
    else
    {
      *(_BYTE *)(v146 + 177) |= 8u;
    }
  }
  *(_BYTE *)(v147 + 88) |= 4u;
  v20 = (unsigned int)dword_4D04964;
  if ( dword_4D04964 && dword_4F077C4 == 2 && *(_BYTE *)(v147 + 140) == 11 && !*(_QWORD *)(v147 + 160) )
  {
    v20 = 2458;
    v238 = v119;
    v18 = unk_4F07471;
    sub_684AA0(unk_4F07471, 2458, &v283[4]);
    v119 = v238;
  }
LABEL_389:
  if ( (v120 & 0x30) != 0 )
  {
    while ( *(_BYTE *)(v119 + 140) == 12 )
      v119 = *(_QWORD *)(v119 + 160);
    goto LABEL_395;
  }
LABEL_390:
  if ( (v290 & 0x40) == 0 )
    goto LABEL_391;
  if ( *(_BYTE *)(v119 + 140) != 12 )
  {
LABEL_395:
    if ( (v120 & 0x20) == 0 )
    {
      if ( !dword_4F04C3C )
      {
        v144 = sub_86A2A0(v119);
        if ( v144 )
        {
          if ( *(_BYTE *)(v144 + 16) == 53 )
            *(_BYTE *)(*(_QWORD *)(v144 + 24) + 57LL) |= 1u;
        }
      }
LABEL_397:
      v18 = (__int64)v283;
      sub_650B20(v283);
      if ( (v290 & 0x40) == 0 )
      {
LABEL_391:
        sub_854AB0();
        if ( !a3 )
        {
          sub_7B8B50(v18, v20, v122, v123);
          *a6 = 1;
        }
        goto LABEL_26;
      }
      goto LABEL_398;
    }
LABEL_426:
    *(_BYTE *)(v119 + 143) |= 8u;
    goto LABEL_397;
  }
LABEL_398:
  sub_643D30(v283);
  *(_BYTE *)(*(_QWORD *)(v213 + 168) + 110LL) |= 4u;
  if ( !dword_4F077B8 )
    goto LABEL_405;
  v199 = 1;
  v200 = v283[36];
  if ( *(_BYTE *)(v283[36] + 140LL) == 12 && !*(_QWORD *)(v283[36] + 8LL) )
  {
    if ( qword_4F077A8 <= 0x76BFu )
    {
      sub_684B30(1565, &v283[4]);
      v200 = v283[36];
    }
    else
    {
      v124 = v283[36];
      do
        v124 = *(_QWORD *)(v124 + 160);
      while ( *(_BYTE *)(v124 + 140) == 12 );
      v283[36] = v124;
      v283[34] = v124;
      sub_684B30(1566, &v283[4]);
LABEL_405:
      v199 = 1;
      v200 = v283[36];
    }
  }
LABEL_41:
  v29 = v283;
  v212 = v21 == 4;
  v30 = &qword_4F061C8;
  v211 = v283[15] & 0x7F;
  v207 = v283[9];
  v210 = WORD2(v283[9]);
  while ( 2 )
  {
    v269 = *(_QWORD *)&dword_4F063F8;
    ++*(_BYTE *)(*v30 + 75LL);
    if ( (v283[15] & 0x40000000000LL) != 0 && (unsigned int)sub_67B2C0() )
    {
      BYTE5(v283[15]) |= 0x10u;
      v74 = v30;
      v15 = v29;
      if ( a2 )
        sub_899160(a2, v213, a10);
      else
        sub_65BC40(v29, &v276, &v272, a10);
      goto LABEL_183;
    }
    if ( (v283[16] & 0x8000000LL) == 0 || word_4F06418[0] == 27 )
    {
      v224 = *(_QWORD *)a1;
      v37 = *v30;
      ++*(_BYTE *)(v37 + 63);
      ++*(_BYTE *)(v37 + 171);
      sub_87E3B0(&v276);
      v38 = v283[1];
      v39 = qword_4F04C68;
      v40 = v283[1] & 0x100;
      v216 = v283[1] & 8;
      v41 = *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 4);
      v283[0] = 0;
      v214 = v41;
      v42 = v290;
      v290 &= ~0x20u;
      if ( (v42 & 1) != 0
        || (BYTE1(v38) = BYTE1(v283[1]) & 0xF3, v290 = v42 & 0xD5, v283[1] = v38, (v283[16] & 0x8000000LL) != 0) )
      {
LABEL_79:
        v43 = word_4F06418[0];
        if ( word_4F06418[0] == 55 && !v40 )
          goto LABEL_192;
        goto LABEL_80;
      }
      if ( word_4F06418[0] == 37 )
      {
LABEL_78:
        v290 |= 8u;
        v283[35] = sub_72CBA0();
        v283[36] = v283[35];
        goto LABEL_79;
      }
      v39 = &dword_4F077C4;
      if ( dword_4F077C4 == 2 )
      {
        if ( word_4F06418[0] != 1 || (unk_4D04A11 & 2) == 0 )
        {
          v39 = 0;
          if ( (unsigned int)sub_7C0F00(0, 0) )
          {
            if ( (unk_4D04A10 & 0x20) != 0 )
              goto LABEL_78;
            if ( word_4F06418[0] != 1 )
            {
LABEL_235:
              if ( v40 )
              {
                v283[3] = *(_QWORD *)&dword_4F063F8;
                v283[35] = sub_72BA30(5);
                v283[36] = v283[35];
                v43 = word_4F06418[0];
              }
              else
              {
                v43 = word_4F06418[0];
                if ( word_4F06418[0] == 55 )
                {
LABEL_192:
                  v290 |= 0x20u;
                  v43 = 55;
                }
              }
LABEL_80:
              *(_QWORD *)dword_4F07508 = *(_QWORD *)&dword_4F063F8;
              if ( (v290 & 0x60) != 0 )
              {
                v44 = _mm_loadu_si128(&xmmword_4F06660[3]);
                v45 = _mm_loadu_si128(&xmmword_4F06660[1]);
                v46 = _mm_loadu_si128(&xmmword_4F06660[2]);
                v272.m128i_i64[0] = _mm_loadu_si128(xmmword_4F06660).m128i_u64[0];
                v273 = v45;
                v272.m128i_i64[1] = *(_QWORD *)&dword_4F063F8;
                v273.m128i_i8[1] = v45.m128i_i8[1] | 0x20;
                v274 = v46;
                v275 = v44;
                if ( (v283[15] & 0x2000000000LL) != 0 )
                  sub_6451E0(v29);
                HIBYTE(v283[15]) = (16 * (v216 == 0)) | HIBYTE(v283[15]) & 0xEF;
                v47 = *v30;
                --*(_BYTE *)(v47 + 63);
                --*(_BYTE *)(v47 + 171);
                goto LABEL_84;
              }
              if ( (v283[16] & 0x8000000LL) != 0 )
              {
                sub_62B8E0(v29, &v276, 0, &v284);
                LODWORD(v55) = v283[36];
                v54 = *(_BYTE *)(v283[36] + 140LL) == 7;
LABEL_135:
                LOBYTE(v55) = v216 == 0;
                v33 = (unsigned int)(16 * v55);
                HIBYTE(v283[15]) = (16 * (v216 == 0)) | HIBYTE(v283[15]) & 0xEF;
                v56 = *v30;
                --*(_BYTE *)(v56 + 63);
                --*(_BYTE *)(v56 + 171);
                if ( dword_4F077C4 != 2 )
                {
                  sub_643D30(v29);
                  if ( dword_4F077C4 != 2 || !v54 )
                    goto LABEL_62;
                  goto LABEL_138;
                }
                if ( v54 )
                  goto LABEL_59;
LABEL_84:
                sub_643D30(v29);
                if ( (v217 & 0xC) == 0 )
                  goto LABEL_63;
LABEL_85:
                v48 = v30;
                v15 = v29;
                if ( v205 )
                  sub_6851C0(277, &v283[4]);
                if ( (v217 & 4) != 0 )
                  sub_6851C0(377, &v283[4]);
                goto LABEL_368;
              }
              if ( v40 && (v290 & 0xA) == 0 )
              {
                if ( v43 == 1 )
                {
                  if ( dword_4F077C4 != 2 )
                    goto LABEL_117;
                  if ( (unk_4D04A11 & 2) != 0 )
                  {
                    if ( (unk_4D04A12 & 1) != 0 )
                      goto LABEL_362;
                    goto LABEL_102;
                  }
                  v39 = 0;
                  if ( (unsigned int)sub_7C0F00(0, 0) && (unk_4D04A12 & 1) != 0 )
                    goto LABEL_362;
                }
                else if ( v43 != 34 && v43 != 27 )
                {
                  if ( dword_4F077C4 != 2
                    || v43 != 33
                    && (!unk_4D04474 || v43 != 52)
                    && ((v39 = (_QWORD *)dword_4D0485C, !dword_4D0485C) || v43 != 25)
                    && v43 != 156 )
                  {
LABEL_362:
                    v111 = v30;
                    v15 = v29;
                    v112 = (_BYTE *)*v111;
                    --v112[75];
                    --v112[63];
                    --v112[171];
                    sub_6851D0(169);
                    if ( word_4F06418[0] == 75 )
                      sub_7B8B50(169, v39, v113, v114);
                    sub_854B40();
                    *a6 = 1;
                    if ( v206 == 9 )
                      a8[14] = sub_72C930();
                    goto LABEL_26;
                  }
                  goto LABEL_102;
                }
              }
              if ( dword_4F077C4 != 2 )
              {
LABEL_117:
                v49 = 17;
                goto LABEL_118;
              }
LABEL_102:
              if ( qword_4CF8008 )
              {
                if ( *(_QWORD *)(qword_4CF8008 + 128) || (*(_BYTE *)(qword_4CF8008 + 184) & 8) != 0 )
                {
                  sub_5E9580(qword_4CF8008);
                  qword_4CF8008 = sub_5E4B20(v224);
                }
              }
              else if ( v214 != 9 )
              {
                qword_4CF8008 = sub_5E4B20(v224);
              }
              v49 = 513;
              ++*(_BYTE *)(*v30 + 81LL);
              if ( (v283[1] & 1) == 0 )
                v49 = (v283[15] & 0x7F) == 0 ? 66049LL : 513LL;
              if ( (v290 & 2) != 0 )
                v49 |= 0x20uLL;
              if ( v216 )
                v49 |= 4uLL;
              v50 = v49;
              if ( (v291 & 2) != 0 )
              {
                LOBYTE(v50) = v49 | 0x80;
                v49 = v50;
              }
LABEL_118:
              if ( dword_4F077BC && *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 4) == 9 )
                v49 |= 4uLL;
              if ( dword_4D043E0 )
                sub_650E40(v29);
              if ( dword_4F077B8 )
              {
                if ( dword_4F077C0 )
                {
                  v51 = v49;
                  if ( dword_4F04C58 != -1 )
                  {
                    BYTE1(v51) = BYTE1(v49) | 0x40;
                    v49 = v51;
                  }
                }
              }
              v208 = v30;
              v259 = v29;
              while ( 1 )
              {
                v52 = dword_4F06650[0];
                v53 = v283[54];
                sub_626F50(v49, v259, v224, &v272, &v276, &v284);
                if ( dword_4F077C4 != 2 )
                {
                  v30 = v208;
                  LODWORD(v54) = 0;
                  v29 = v259;
                  if ( v212 )
                    goto LABEL_134;
LABEL_228:
                  v54 = (unsigned int)sub_8D2310(v283[36]) != 0;
                  goto LABEL_134;
                }
                if ( *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 4) != 8
                  || !v283[46]
                  || (v283[16] & 0x80000000000LL) != 0 )
                {
                  break;
                }
                sub_897F40(v52, v53, &v276, &v272);
              }
              v30 = v208;
              v29 = v259;
              --*(_BYTE *)(*v208 + 81LL);
              v75 = v283[36];
              if ( BYTE5(v283[33]) != 4 && qword_4D0495C && (unsigned int)sub_6454D0(v283[36], &v272.m128i_u64[1]) )
                v283[36] = sub_72C930();
              if ( (v283[1] & 0x20) != 0 && (v291 & 1) == 0 )
              {
                for ( j = v75; !(unsigned int)sub_8D2310(j); j = sub_8D4870(j) )
                {
                  while ( (unsigned int)sub_8D3320(j) )
                  {
                    j = sub_8D46C0(j);
                    if ( (unsigned int)sub_8D2310(j) )
                      goto LABEL_222;
                  }
                  if ( !(unsigned int)sub_8D3D10(j) )
                    goto LABEL_223;
                }
LABEL_222:
                sub_6851C0(305, &v283[3]);
                v291 |= 1u;
              }
LABEL_223:
              if ( *(char *)(a1 + 8) < 0 && (unsigned int)sub_8D2310(v75) )
                sub_8DCB20(v75);
              if ( v214 == 9 && (v283[33] & 0xFD0000000000LL) != 0 )
              {
                BYTE5(v283[33]) = 0;
                v212 = 0;
              }
              v77 = (v283[2] >> 5) & 1LL;
              if ( (v273.m128i_i8[0] & 0x20) != 0 )
                LOBYTE(v77) = 1;
              LODWORD(v54) = 0;
              v290 = (2 * ((v283[2] & 0x10) != 0)) | v290 & 0xF5 | (8 * v77);
              if ( !v212 )
                goto LABEL_228;
LABEL_134:
              sub_645270(v29);
              v55 = sub_650E40(v29);
              goto LABEL_135;
            }
          }
          else if ( word_4F06418[0] != 1 )
          {
            goto LABEL_235;
          }
LABEL_357:
          v39 = v29;
          if ( (unsigned int)sub_672080(v224, v29) )
          {
            v290 |= 2u;
            v283[35] = sub_72CBA0();
            v283[36] = v283[35];
            goto LABEL_79;
          }
          goto LABEL_235;
        }
      }
      else if ( word_4F06418[0] != 1 )
      {
        goto LABEL_235;
      }
      if ( (unk_4D04A10 & 0x20) != 0 )
        goto LABEL_78;
      goto LABEL_357;
    }
    sub_87E3B0(&v276);
    v283[36] = sub_732700(v283[34], 0, 0, 0, 0, 0, 0, 0);
    v33 = *(_QWORD *)(v283[36] + 168LL);
    *(_QWORD *)(v33 + 40) = v213;
    *(_DWORD *)(v33 + 18) = *(_DWORD *)(v33 + 18) & 0xFEFFFF80 | 0x1000001;
    BYTE5(v283[15]) |= 8u;
    v34 = v195;
    v283[35] = v283[36];
    v35 = v196;
    v281.m128i_i64[0] = v283[36];
    if ( dword_4F077C4 != 2 )
      goto LABEL_61;
LABEL_59:
    if ( (v283[16] & 0x8000000LL) != 0 || !v283[46] )
    {
LABEL_61:
      sub_643D30(v29);
      if ( dword_4F077C4 != 2 )
      {
LABEL_62:
        if ( (v217 & 0xC) != 0 )
          goto LABEL_85;
LABEL_63:
        if ( (v217 & 2) == 0 )
        {
          if ( (v290 & 8) == 0 )
            goto LABEL_65;
LABEL_190:
          sub_854B40();
LABEL_45:
          if ( a3 )
          {
LABEL_46:
            if ( v283[0] && *(_BYTE *)(v283[0] + 80LL) == 21 )
              goto LABEL_48;
            v48 = v30;
            v15 = v29;
            if ( (v291 & 0x20) == 0 )
            {
              sub_6851C0(787, &v283[4]);
              v291 |= 0x20u;
            }
LABEL_368:
            --*(_BYTE *)(*v48 + 75LL);
            sub_854B40();
            goto LABEL_26;
          }
LABEL_48:
          if ( (v290 & 1) == 0 )
            sub_65C1C0(v29);
          --*(_BYTE *)(*v30 + 75LL);
          v290 &= ~1u;
          if ( word_4F06418[0] == 67 )
          {
            if ( dword_4F077C4 == 2 )
              sub_65C470(v29);
            sub_643EB0(v29, 1);
            sub_65C040(v29);
            LOBYTE(v283[15]) = v211 | v283[15] & 0x80;
            LODWORD(v283[9]) = v207;
            WORD2(v283[9]) = v210;
          }
          if ( dword_4F077BC
            && qword_4F077A8 > 0x76BFu
            && word_4F06418[0] == 67
            && (unsigned __int16)sub_7BE840(0, 0) == 75 )
          {
            sub_684B30(228, &dword_4F063F8);
            sub_7B8B50(228, &dword_4F063F8, v89, v90);
          }
          if ( !(unsigned int)sub_7BE800(67) )
          {
            v15 = v29;
            goto LABEL_26;
          }
          continue;
        }
        if ( !dword_4D04820 && (!dword_4F077BC || !(unsigned int)sub_657F30(&v283[11])) || BYTE5(v283[33]) != 2 )
        {
          sub_6851C0(2860, &v283[11]);
          sub_5F7920((__int64)&v272, a1, v29, v31);
          goto LABEL_45;
        }
        if ( (v290 & 8) != 0 )
          goto LABEL_190;
LABEL_65:
        if ( v212 )
        {
          if ( (v283[2] & 4) != 0 )
          {
            if ( (unsigned int)sub_8D2310(v283[36]) )
            {
              sub_684B30(472, &v272.m128i_u64[1]);
            }
            else
            {
              sub_6851C0(283, &v272.m128i_u64[1]);
              v96 = _mm_loadu_si128(&xmmword_4F06660[3]);
              v97 = _mm_loadu_si128(&xmmword_4F06660[1]);
              v98 = _mm_loadu_si128(&xmmword_4F06660[2]);
              v272.m128i_i64[0] = _mm_loadu_si128(xmmword_4F06660).m128i_u64[0];
              v273 = v97;
              v274 = v98;
              v273.m128i_i8[1] = v97.m128i_i8[1] | 0x20;
              v272.m128i_i64[1] = *(_QWORD *)dword_4F07508;
              v275 = v96;
            }
          }
          if ( !v209 )
            sub_64E990(&v269, v283[36], 0, 0, 0, v201 == 0);
          sub_65E230(&v272, v29, v213, &v284);
          if ( qword_4CF8008 && (*(_QWORD *)(qword_4CF8008 + 128) || (*(_BYTE *)(qword_4CF8008 + 184) & 8) != 0) )
            *(_QWORD *)(qword_4CF8008 + 16) = v283[0];
          goto LABEL_45;
        }
        if ( a3 )
        {
          if ( v206 == 9 )
            goto LABEL_46;
        }
        else
        {
          if ( !unk_4D04424 && word_4F06418[0] == 56 && dword_4F077C4 == 2 )
          {
            if ( (v125 = sub_8D3350(v283[36]), v126 = v283[36], v125)
              && !v204
              && (*(_BYTE *)(v283[36] + 140LL) & 0xFB) == 8
              && (v153 = sub_8D4C10(v283[36], dword_4F077C4 != 2), v126 = v283[36], v153 == 1)
              || (unsigned int)sub_8D3D40(v126) )
            {
              if ( !(BYTE5(v283[33]) | v283[16] & 0x40) )
              {
                v212 = sub_696820();
                if ( v212 )
                {
                  sub_6851D0(1023);
                  v212 = 0;
                }
                else
                {
                  v170 = v283[36];
                  v171 = 5;
                  v172 = *(_QWORD *)a1;
                  if ( dword_4D04964 )
                    v171 = byte_4F07472[0];
                  sub_684AA0(v171, 382, &dword_4F063F8);
                  sub_7B8B50(v171, 382, v173, v174);
                  v175 = sub_724D50(0);
                  sub_6D6AC0(v170, v29, v175);
                  *(_BYTE *)(v175 + 170) |= 0x10u;
                  sub_73A770(v175);
                  v176 = sub_647630(2, &v272, unk_4F04C5C, 0);
                  *(_QWORD *)(v176 + 88) = v175;
                  v177 = v176;
                  sub_877D80(v175, v176);
                  sub_877E20(v177, v175, v172);
                  v283[0] = v177;
                  *(_BYTE *)(v175 + 88) = *(_BYTE *)(a1 + 12) & 3 | *(_BYTE *)(v175 + 88) & 0xFC;
                  sub_8756F0(3, v177, &v272.m128i_u64[1], v283[44]);
                  sub_729470(v175, &v284);
                  sub_854980(v177, 0);
                  sub_733310(v175, 0);
                }
                goto LABEL_48;
              }
            }
          }
          if ( v206 == 9 )
          {
            v212 = 0;
            goto LABEL_48;
          }
        }
        if ( v204 && (*(_BYTE *)(v283[36] + 140LL) & 0xFB) == 8 && (sub_8D4C10(v283[36], dword_4F077C4 != 2) & 1) != 0 )
          sub_6851C0(719, &v283[4]);
        if ( !v209 )
          sub_64E990(&v269, v283[36], 0, 0, 0, v201 == 0);
        if ( v199 || (v290 & 0x20) != 0 || (v283[2] & 2) != 0 || (v273.m128i_i8[1] & 0x20) == 0 )
        {
          if ( BYTE5(v283[33]) == 2 )
            sub_5F2920((__int64)&v272, (__int64 *)a1, a2, v29);
          else
            sub_5F7920((__int64)&v272, a1, v29, v36);
        }
        if ( dword_4F077C4 == 2 && unk_4F07778 <= 202001 && word_4F06418[0] == 56 )
        {
          *(_QWORD *)dword_4F07508 = *(_QWORD *)&dword_4F063F8;
          sub_6851D0(321);
        }
        goto LABEL_45;
      }
LABEL_138:
      if ( word_4F06418[0] != 75 )
      {
        if ( word_4F06418[0] == 73 || word_4F06418[0] == 163 || (v290 & 2) != 0 && word_4F06418[0] == 55 )
        {
          v280.m128i_i8[0] |= 4u;
        }
        else if ( word_4F06418[0] == 56 )
        {
          sub_7ADF70(&v270, 0);
          sub_7AE360(&v270);
          sub_7B8B50(&v270, 0, v128, v129);
          if ( dword_4D04464 && word_4F06418[0] == 152 )
          {
            v280.m128i_i8[0] |= 0x14u;
            if ( dword_4F077BC && (dword_4F077C4 != 2 || unk_4F07778 <= 201102 && !dword_4F07774) )
              sub_684B40(&dword_4F063F8, 2513);
          }
          else if ( unk_4D04468 && word_4F06418[0] == 83 )
          {
            v280.m128i_i8[0] |= 0xCu;
            if ( dword_4F077BC && (dword_4F077C4 != 2 || unk_4F07778 <= 201102 && !dword_4F07774) )
              sub_684B40(&dword_4F063F8, 2514);
          }
          else if ( word_4F06418[0] == 4 )
          {
            sub_7AE360(&v270);
            if ( (unsigned __int16)sub_7B8B50(&v270, 0, v150, v151) == 73 )
              v280.m128i_i8[0] |= 4u;
          }
          sub_7BC000(&v270);
        }
      }
      LOBYTE(v57) = 1;
      if ( (v280.m128i_i8[0] & 4) == 0 )
        v57 = (v283[1] >> 1) & 1LL;
      v280.m128i_i8[0] = (2 * v57) | v280.m128i_i8[0] & 0xFD;
      if ( (v280.m128i_i8[0] & 4) != 0 )
      {
        if ( (v290 & 1) == 0 )
          sub_6851C0(65, &dword_4F063F8);
        LODWORD(v282) = sub_7A7D00();
      }
      v225 = (v280.m128i_i8[0] & 4) != 0;
      if ( v204 )
        sub_6851C0(719, &v283[4]);
      if ( !v209
        && !(BYTE3(v283[16]) & 8 | v273.m128i_i8[0] & 0x10 | v290 & 0xA)
        && ((v273.m128i_i8[1] & 0x20) == 0 || !(unsigned int)sub_878690(&v272)) )
      {
        sub_64E990(&v269, v283[36], 1, (v280.m128i_i8[0] & 4) != 0, 0, v201 == 0);
      }
      sub_87E350(&v276);
      v59 = v283[36];
      if ( v283[36] == v200
        || v200
        && v283[36]
        && dword_4F07588
        && (v92 = *(_QWORD *)(v283[36] + 32LL), *(_QWORD *)(v200 + 32) == v92)
        && v92 )
      {
        v280.m128i_i8[0] |= 0x80u;
        v93 = 0;
        v280.m128i_i64[1] = v283[44];
        if ( !v205 )
          v93 = BYTE5(v283[33]) != 2;
        v94 = (v280.m128i_i8[0] & 4) != 0;
        if ( (v280.m128i_i8[0] & 4) == 0 )
          goto LABEL_302;
        sub_6851C0(93, &v272.m128i_u64[1]);
        v59 = v283[36];
        if ( *(_BYTE *)(v283[36] + 140LL) != 12 )
          goto LABEL_333;
        do
        {
          v59 = *(_QWORD *)(v59 + 160);
LABEL_302:
          ;
        }
        while ( *(_BYTE *)(v59 + 140) == 12 );
        if ( v93 || v94 )
        {
LABEL_333:
          v95 = sub_73EDA0(v59, 0);
          if ( !v93 )
            goto LABEL_307;
          *(_QWORD *)(*(_QWORD *)(v95 + 168) + 40LL) = v213;
          *(_BYTE *)(*(_QWORD *)(v95 + 168) + 21LL) |= 1u;
LABEL_309:
          v283[36] = v95;
        }
        else if ( (*(_BYTE *)(*(_QWORD *)(v59 + 168) + 17LL) & 0x70) != 0x20 )
        {
          v95 = sub_73EDA0(v59, 0);
LABEL_307:
          if ( qword_4D0495C )
          {
            *(_QWORD *)(*(_QWORD *)(v95 + 168) + 40LL) = 0;
            *(_BYTE *)(*(_QWORD *)(v95 + 168) + 21LL) &= ~1u;
            *(_BYTE *)(*(_QWORD *)(v95 + 168) + 18LL) &= 0x80u;
          }
          goto LABEL_309;
        }
      }
      if ( (v217 & 4) != 0 && (v273.m128i_i8[1] & 0x20) == 0 )
      {
        v115 = v290;
        if ( (v290 & 0x10) == 0 )
        {
          v116 = 377;
          if ( (v283[1] & 8) != 0 || (v290 & 2) != 0 )
          {
LABEL_377:
            v117 = &v283[3];
            if ( (v115 & 1) == 0 )
              v117 = &v272.m128i_i64[1];
            sub_684AA0(8, v116, v117);
            v290 |= 0x10u;
          }
          else
          {
            if ( (unsigned int)sub_8D3B10(v213) )
            {
              v115 = v290;
              v116 = 377;
              goto LABEL_377;
            }
            if ( BYTE5(v283[33]) == 2 || (v273.m128i_i8[0] & 8) != 0 && (unsigned __int8)(v275.m128i_i8[8] - 1) <= 3u )
            {
              v115 = v290;
              v116 = 314;
              goto LABEL_377;
            }
          }
        }
      }
      if ( v205 )
      {
        v60 = (__int64)&v272;
        v61 = *(_QWORD *)a1;
        v63 = sub_5F1000(&v272, *(_QWORD *)a1, (__int64)&v276, v29, v58);
      }
      else
      {
        v78 = v290;
        if ( (v290 & 0xA) != 0 && BYTE5(v283[33]) == 2 )
        {
          sub_6851C0(378, &v283[4]);
          BYTE5(v283[33]) = 0;
          v78 = v290;
        }
        if ( (v78 & 2) != 0 && (!unk_4D04414 || (v280.m128i_i8[0] & 0x18) == 0) || (v217 & 4) != 0 )
        {
          *(_BYTE *)(a1 + 8) |= 6u;
        }
        else if ( (v78 & 8) != 0 && (v280.m128i_i8[0] & 0x18) == 0 )
        {
          *(_BYTE *)(a1 + 8) |= 4u;
        }
        if ( v206 == 9 )
        {
          v74 = v30;
          v15 = v29;
          *a7 = v283[36];
          a8[14] = v281.m128i_i64[0];
          a8[8] = v276.m128i_i64[0];
          a8[13] = v276.m128i_i64[1];
          v280.m128i_i8[1] |= 8u;
          sub_643F80(v29, 0);
          goto LABEL_183;
        }
        if ( a3 )
        {
          v74 = v30;
          v15 = v29;
          if ( dword_4F077BC
            && qword_4F077A8 <= 0x9D07u
            && word_4F06418[0] == 56
            && (unsigned __int16)sub_7BE840(0, 0) == 4 )
          {
            sub_7B8B50(0, 0, v190, v191);
            sub_7B8B50(0, 0, v192, v193);
          }
          sub_5FA450((__int64)&v272, a5, a9, &v276, a1, (__int64)v29);
          if ( (v217 & 0x2000) != 0 )
          {
            if ( (v290 & 2) != 0 )
            {
              *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(v283[0] + 88LL) + 176LL) + 193LL) |= 0x80u;
            }
            else if ( (v273.m128i_i8[0] & 0x10) != 0 && unk_4D044AC )
            {
              *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(v283[0] + 88LL) + 176LL) + 194LL) |= 1u;
            }
          }
          if ( (v280.m128i_i8[0] & 4) != 0 )
          {
            *(_DWORD *)(a2 + 36) = 1;
            v203[122] |= 1u;
          }
          goto LABEL_183;
        }
        v61 = (__int64)&v276;
        v60 = (__int64)&v272;
        sub_5FBCD0(&v272, (__int64)&v276, a1, v29, 0);
        v63 = v283[0];
        if ( (*(_WORD *)(a1 + 8) & 0x180) != 0 && (*(_DWORD *)(*(_QWORD *)a1 + 176LL) & 0x44000) == 0 )
        {
          v99 = *(_QWORD *)(*(_QWORD *)(v283[0] + 96LL) + 56LL);
          if ( !*(_DWORD *)(v99 + 64) )
            *(_DWORD *)(v99 + 64) = dword_4F06650[0];
        }
        else
        {
          v79 = *(_QWORD *)(v283[0] + 96LL);
          if ( v79 )
          {
            v80 = v281.m128i_i64[0];
            *(_QWORD *)(v79 + 120) = v281.m128i_i64[0];
            if ( !v80 )
            {
              v60 = v283[36];
              v61 = (__int64)&v276;
              v80 = sub_624310(v283[36], &v276);
            }
            *(_QWORD *)(v79 + 112) = v80;
            *(_QWORD *)(v79 + 64) = v276.m128i_i64[0];
            *(_QWORD *)(v79 + 104) = v276.m128i_i64[1];
            v280.m128i_i8[1] |= 8u;
            v81 = *(_QWORD *)(v79 + 32);
            if ( v81 )
            {
              if ( *(_BYTE *)(v81 + 80) == 10 && *(_BYTE *)(v63 + 80) == 10 )
              {
                v130 = *(_QWORD *)(*(_QWORD *)(v81 + 88) + 104LL);
                if ( v130 )
                {
                  if ( (*(_BYTE *)(v130 + 11) & 1) != 0 )
                  {
                    v60 = *(_QWORD *)(*(_QWORD *)(v63 + 88) + 104LL);
                    sub_5CF700((__int64 *)v60);
                  }
                }
              }
            }
            if ( (v283[16] & 0x40000000000LL) != 0 )
            {
              *(_BYTE *)(*(_QWORD *)(v63 + 88) + 208LL) |= 4u;
              *(_BYTE *)(v79 + 80) |= 2u;
            }
          }
        }
        if ( (v217 & 0x2000) != 0 )
        {
          if ( (v290 & 2) != 0 )
          {
            *(_BYTE *)(*(_QWORD *)(v63 + 88) + 193LL) |= 0x80u;
            *(_BYTE *)(a1 + 8) |= 2u;
          }
          else if ( (v273.m128i_i8[0] & 0x10) != 0 && unk_4D044AC )
          {
            *(_BYTE *)(*(_QWORD *)(v63 + 88) + 194LL) |= 1u;
          }
        }
      }
      if ( (v280.m128i_i8[0] & 4) != 0 && (!dword_4F07590 || dword_4D047B0 || *(char *)(v213 + 177) >= 0) )
      {
        v64 = qword_4CF8008;
        if ( qword_4CF8008 )
          goto LABEL_168;
LABEL_287:
        if ( dword_4F04C64 == -1
          || (v91 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v91 + 7) & 1) == 0)
          || dword_4F04C44 == -1 && (*(_BYTE *)(v91 + 6) & 2) == 0 )
        {
          if ( (v280.m128i_i8[1] & 8) == 0 )
          {
            v60 = (__int64)&v276.m128i_i64[1];
            sub_87E280(&v276.m128i_u64[1]);
          }
        }
      }
      else
      {
        v60 = (__int64)&v276;
        sub_876830(&v276);
        v64 = qword_4CF8008;
        if ( !qword_4CF8008 )
          goto LABEL_287;
LABEL_168:
        *(_QWORD *)(v64 + 16) = v63;
        *(__m128i *)(v64 + 24) = _mm_loadu_si128(&v276);
        *(__m128i *)(v64 + 40) = _mm_loadu_si128(&v277);
        *(__m128i *)(v64 + 56) = _mm_loadu_si128(&v278);
        *(__m128i *)(v64 + 72) = _mm_loadu_si128(&v279);
        *(__m128i *)(v64 + 88) = _mm_loadu_si128(&v280);
        *(__m128i *)(v64 + 104) = _mm_loadu_si128(&v281);
        *(_QWORD *)(v64 + 120) = v282;
      }
      if ( word_4F06418[0] != 56 || (v61 = 0, v60 = 0, v105 = sub_7BE840(0, 0), v105 == 152) || v105 == 83 )
      {
LABEL_170:
        v65 = v280.m128i_i8[0];
        goto LABEL_171;
      }
      if ( (*(_BYTE *)(v63 + 81) & 0x10) == 0 || v213 != *(_QWORD *)(v63 + 64) )
        goto LABEL_349;
      v106 = *(_QWORD *)(v63 + 88);
      if ( *(_BYTE *)(v63 + 80) == 20 )
        v106 = *(_QWORD *)(v106 + 176);
      v62 = *(unsigned __int8 *)(v106 + 192);
      if ( (v62 & 2) == 0 )
      {
        v148 = v213;
        if ( (*(_BYTE *)(v213 + 177) & 0xC0) == 0 )
          goto LABEL_349;
        if ( *(_BYTE *)(v213 + 140) == 12 )
        {
          do
            v148 = *(_QWORD *)(v148 + 160);
          while ( *(_BYTE *)(v148 + 140) == 12 );
        }
        else
        {
          v148 = v213;
        }
        if ( (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)v148 + 96LL) + 180LL) & 2) == 0 )
        {
          v61 = (__int64)&dword_4F077BC;
          if ( dword_4F077BC )
          {
            v107 = 1;
LABEL_543:
            v62 &= 0x10u;
            if ( (_DWORD)v62 )
            {
              v108 = 1;
              v149 = sub_736C60(7, *(_QWORD *)(v106 + 104));
              v60 = 1849;
              v61 = v149 + 56;
              if ( !v149 )
                v61 = (__int64)&dword_4F063F8;
              sub_684B30(1849, v61);
            }
            else
            {
              v108 = 1;
              if ( (*(_BYTE *)(v213 + 176) & 1) != 0 )
              {
                v61 = (__int64)&dword_4F063F8;
                v60 = 1849;
                sub_684B30(1849, &dword_4F063F8);
              }
            }
LABEL_351:
            sub_7B8B50(v60, v61, v106, v62);
            if ( word_4F06418[0] == 4 )
            {
              v60 = (__int64)&unk_4F06300;
              if ( (unk_4F063A9 & 2) == 0 )
              {
                v61 = (__int64)&dword_4F077BC;
                if ( !dword_4F077BC )
                  goto LABEL_356;
                if ( !(unsigned int)sub_72A2A0() )
                {
                  v61 = (__int64)&dword_4F077BC;
                  goto LABEL_353;
                }
              }
            }
            else
            {
              v61 = (__int64)&dword_4F077BC;
LABEL_353:
              if ( !dword_4F077BC || qword_4F077A8 > 0x76BFu || word_4F06418[0] != 188 )
              {
LABEL_356:
                v60 = 320;
                *(_QWORD *)dword_4F07508 = *(_QWORD *)&dword_4F063F8;
                sub_6851D0(320);
                goto LABEL_170;
              }
            }
            if ( (v108 & (v107 ^ 1)) != 0 )
            {
              *(_BYTE *)(*(_QWORD *)(v63 + 88) + 192LL) |= 8u;
              *(_BYTE *)(v213 + 176) |= 0xA0u;
              v143 = *(_QWORD *)(*(_QWORD *)(v63 + 88) + 72LL);
              if ( v143 )
              {
                v110 = *(unsigned int *)(v143 + 32);
                if ( (_DWORD)v110 )
                {
                  v109 = *(_QWORD *)&dword_4F063F8;
                  *(_QWORD *)(v143 + 40) = *(_QWORD *)&dword_4F063F8;
                }
              }
            }
            sub_7B8B50(v60, v61, v109, v110);
            v65 = v280.m128i_i8[0];
            if ( (v280.m128i_i8[0] & 4) != 0 )
            {
              v61 = 2019;
              v60 = 7;
              sub_684AA0(7, 2019, &dword_4F063F8);
              v65 = v280.m128i_i8[0];
            }
LABEL_171:
            v66 = v225;
            v67 = v283[0];
            v68 = v65 & 0x18;
            v69 = v283[1];
            if ( v225 )
            {
              v66 = *v30;
              --*(_BYTE *)(*v30 + 75LL);
              if ( v68 )
                goto LABEL_173;
              v219 = *(_QWORD *)a1;
              v221 = (v290 & 2) != 0;
              sub_7ADF70(&v270, 1);
              v100 = sub_7C8F90((unsigned int)&v270, v221, 0, (unsigned int)&v267, (unsigned int)&v268, 0, 0);
              v82 = v219;
              v102 = v100;
              v103 = qword_4CF8008;
              *(__m128i *)(qword_4CF8008 + 152) = _mm_loadu_si128(&v270);
              *(__m128i *)(v103 + 168) = _mm_loadu_si128(&v271);
              v104 = v196;
              if ( v102 )
              {
                sub_7B8B50(&v270, v221, v196, v101);
                v82 = v219;
              }
              if ( word_4F06418[0] == 75 )
              {
                v262 = v82;
                sub_7B8B50(&v270, v221, v104, v101);
                v82 = v262;
              }
              if ( (*(_WORD *)(a1 + 8) & 0x180) == 0 )
              {
LABEL_484:
                v15 = v29;
                *a6 = 1;
                goto LABEL_26;
              }
              v83 = v225;
LABEL_268:
              if ( (*(_DWORD *)(v82 + 176) & 0x14000) == 0 && (v69 & 8) == 0 && (*(_BYTE *)(v82 + 89) & 1) == 0 )
              {
                v84 = *(_QWORD *)(v67 + 96);
                if ( v84 )
                {
                  for ( k = *(_QWORD *)(v84 + 56); *(_BYTE *)(v82 + 140) == 12; v82 = *(_QWORD *)(v82 + 160) )
                    ;
                  v86 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)v82 + 96LL) + 80LL);
                  if ( v86 )
                  {
                    v87 = *(_QWORD *)(v86 + 32);
                    v88 = &v270;
                    v260 = v83;
                    if ( !v83 )
                      v88 = 0;
                    sub_879080(k, v88, v87);
                    v83 = v260;
                  }
                  if ( v83 )
                  {
                    *(_QWORD *)(k + 80) = sub_888280(v67, k, v267, v268);
                    sub_898C50(k, &v270);
                  }
                }
              }
              goto LABEL_174;
            }
            if ( v68 )
            {
LABEL_173:
              sub_7B8B50(v60, v61, v66, v62);
              sub_7B8B50(v60, v61, v70, v71);
              sub_7BE280(75, 65, 0, 0);
              if ( (*(_WORD *)(a1 + 8) & 0x180) != 0 )
              {
                v82 = *(_QWORD *)a1;
                v83 = 0;
                goto LABEL_268;
              }
LABEL_174:
              if ( v225 )
                goto LABEL_484;
            }
            else if ( (*(_WORD *)(a1 + 8) & 0x180) != 0 )
            {
              v82 = *(_QWORD *)a1;
              v83 = 0;
              goto LABEL_268;
            }
            if ( v205 || (*(_BYTE *)(*(_QWORD *)(v63 + 88) + 192LL) & 0xA) != 2 )
              goto LABEL_45;
            if ( (*(_BYTE *)(a1 + 9) & 4) != 0 )
            {
              v72 = v63;
              v73 = 329;
              if ( !dword_4D04964 )
              {
                sub_6854B0(329, v63);
                goto LABEL_45;
              }
            }
            else
            {
              if ( *(_QWORD *)(a1 + 32) )
                goto LABEL_45;
              v127 = v213;
              if ( (*(_BYTE *)(v213 + 177) & 4) == 0 )
              {
                while ( (*(_BYTE *)(v127 + 89) & 4) != 0 )
                {
                  v127 = *(_QWORD *)(*(_QWORD *)(v127 + 40) + 32LL);
                  if ( (*(_BYTE *)(v127 + 177) & 4) != 0 )
                    goto LABEL_418;
                }
                goto LABEL_45;
              }
LABEL_418:
              v72 = v63;
              v73 = 782;
            }
            sub_6854E0(v73, v72);
            goto LABEL_45;
          }
LABEL_349:
          v107 = 0;
          v108 = 0;
          if ( (v290 & 0x10) == 0 )
          {
            v61 = (__int64)&dword_4F063F8;
            v60 = 319;
            sub_6851C0(319, &dword_4F063F8);
          }
          goto LABEL_351;
        }
      }
      v107 = 0;
      goto LABEL_543;
    }
    break;
  }
  v74 = v30;
  v15 = v29;
  v270.m128i_i16[0] = 75;
  if ( dword_4F04C44 != -1 || (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) & 2) != 0 )
  {
    sub_603EB0(unk_4CF7FF0, v283[8], &v283[46], v283[8]);
    BYTE5(v283[16]) |= 0x10u;
  }
  v138 = (__int64 *)qword_4CF8008;
  if ( qword_4CF8008 && !*(_QWORD *)(qword_4CF8008 + 16) )
  {
    if ( (*(_BYTE *)(qword_4CF8008 + 184) & 2) == 0 )
      sub_679050(*(_QWORD *)(qword_4CF8008 + 128));
    v154 = qword_4CF8000;
    v138[16] = 0;
    qword_4CF8008 = 0;
    *v138 = v154;
    qword_4CF8000 = (__int64)v138;
  }
  sub_8BE2A0(v29, &v270, v33, v32, v34, v35);
  if ( v270.m128i_i16[0] == 74 )
  {
    sub_7BE280(74, 67, 0, 0);
    *a6 = 1;
  }
LABEL_183:
  --*(_BYTE *)(*v74 + 75LL);
LABEL_26:
  if ( dword_4F077C4 == 2 )
  {
    sub_643D30(v15);
    sub_65C470(v15);
  }
  sub_643EB0(v15, 0);
  if ( a10 )
  {
    v22 = _mm_loadu_si128(&v285);
    v23 = _mm_loadu_si128(&v286);
    v24 = _mm_loadu_si128(&v287);
    v25 = _mm_loadu_si128(&v288);
    *a10 = _mm_loadu_si128(&v284);
    a10[1] = v22;
    a10[2] = v23;
    a10[3] = v24;
    a10[4] = v25;
    a10[5].m128i_i64[0] = v289;
  }
  if ( a2 )
  {
    v26 = *((_QWORD *)v203 + 54);
    qmemcpy(v203, v15, 0x1D8u);
    *((_QWORD *)v203 + 19) = v203;
    *((_QWORD *)v203 + 54) = v26;
    *(_QWORD *)a2 = v203;
  }
  return v283[0];
}
