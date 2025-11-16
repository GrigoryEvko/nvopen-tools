// Function: sub_2746CA0
// Address: 0x2746ca0
//
void __fastcall sub_2746CA0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v8; // rax
  _QWORD *v9; // rdx
  __int8 v10; // cl
  __int64 v11; // rax
  __int64 v12; // rbx
  __int64 v13; // r15
  bool v14; // al
  __int64 v15; // r8
  __int64 v16; // r9
  bool v17; // al
  __int64 v18; // r8
  __int64 v19; // r9
  __int64 v20; // r9
  _BYTE *v21; // r15
  _BYTE *v22; // rcx
  _BYTE *v23; // rbx
  __int64 v24; // rdx
  __int64 v25; // r12
  _BYTE **v26; // rax
  __int64 v27; // rsi
  unsigned __int64 v28; // r15
  __int64 v29; // rax
  const char *v30; // rax
  __int64 *v31; // rdx
  __int64 v32; // rax
  unsigned __int128 *v33; // rdx
  unsigned __int64 v34; // rcx
  char v35; // al
  __int64 v36; // rax
  void *v37; // rdx
  __int64 v38; // rcx
  __int64 v39; // r12
  int v40; // r13d
  __int64 v41; // rbx
  __int64 v42; // r15
  __m128i v43; // rax
  __int64 v44; // rdx
  __int64 v45; // rcx
  __int64 v46; // rbx
  __int64 v47; // rax
  __int64 v48; // rbx
  int v49; // esi
  __int64 v50; // rax
  const char *v51; // r15
  __int64 v52; // rdx
  unsigned __int64 *v53; // rdi
  __int64 *v54; // rdx
  _QWORD *v55; // r15
  _QWORD *v56; // rdx
  __int64 v57; // rdi
  __int64 v58; // rax
  __int64 *v59; // rax
  __int64 v60; // rax
  __int64 v61; // rbx
  __int64 v62; // r13
  _QWORD *v63; // rax
  __int64 v64; // r12
  __int64 v65; // r13
  __int64 v66; // rbx
  __int64 v67; // rdx
  unsigned int v68; // esi
  unsigned __int64 v69; // rax
  __int64 v70; // rdx
  __int64 v71; // rsi
  __int64 v72; // r8
  __int64 v73; // r9
  __int64 v74; // r12
  _QWORD *v75; // rax
  __int32 v76; // ecx
  _QWORD *v77; // rdx
  __int64 v78; // rbx
  __int64 v79; // r14
  char v80; // al
  __int64 v81; // r8
  __int64 v82; // r9
  unsigned __int8 *v83; // r15
  __int64 v84; // r13
  unsigned int v85; // r12d
  __int64 (__fastcall *v86)(__int64, unsigned int, _BYTE *, unsigned __int8 *); // rax
  _QWORD **v87; // rdx
  int v88; // ecx
  __int64 *v89; // rax
  __int64 v90; // rsi
  __int64 v91; // r13
  __int64 v92; // r12
  __int64 v93; // rdx
  unsigned int v94; // esi
  char v95; // al
  __int64 v96; // r8
  __int64 v97; // r9
  unsigned __int64 v98; // rdx
  __int64 v99; // rax
  __int64 v100; // rdx
  __int64 v101; // rdx
  __int64 v102; // rsi
  _QWORD *v103; // rbx
  _QWORD *v104; // r12
  __int64 v105; // rax
  __int64 v106; // rdx
  __int64 v107; // rax
  int v108; // r10d
  unsigned int v109; // eax
  _QWORD *v110; // rbx
  _QWORD *v111; // r12
  __int64 v112; // rsi
  _QWORD *v113; // rsi
  char v114; // al
  const char **v115; // rcx
  __m128i si128; // xmm0
  __m128i v117; // xmm1
  unsigned __int64 v118; // rsi
  __int64 v119; // [rsp+18h] [rbp-318h]
  __int64 v124; // [rsp+48h] [rbp-2E8h]
  __int64 v125; // [rsp+50h] [rbp-2E0h]
  __int64 v126; // [rsp+50h] [rbp-2E0h]
  __int64 v127; // [rsp+58h] [rbp-2D8h]
  __int64 v128; // [rsp+68h] [rbp-2C8h]
  __int64 v129; // [rsp+70h] [rbp-2C0h]
  unsigned __int8 *v130; // [rsp+70h] [rbp-2C0h]
  __int64 v131; // [rsp+78h] [rbp-2B8h]
  unsigned int v132; // [rsp+78h] [rbp-2B8h]
  __int64 v133; // [rsp+78h] [rbp-2B8h]
  _BYTE **v134; // [rsp+78h] [rbp-2B8h]
  __int64 v135; // [rsp+88h] [rbp-2A8h]
  __int64 v136[4]; // [rsp+90h] [rbp-2A0h] BYREF
  _QWORD *v137; // [rsp+B0h] [rbp-280h] BYREF
  __int64 *v138; // [rsp+B8h] [rbp-278h]
  __int64 j; // [rsp+C0h] [rbp-270h]
  __m128i *v140; // [rsp+C8h] [rbp-268h]
  __int16 v141; // [rsp+D0h] [rbp-260h]
  unsigned __int128 v142; // [rsp+E0h] [rbp-250h] BYREF
  __m128i v143; // [rsp+F0h] [rbp-240h] BYREF
  __int64 *v144; // [rsp+100h] [rbp-230h]
  const char *v145; // [rsp+110h] [rbp-220h] BYREF
  __int64 v146; // [rsp+118h] [rbp-218h]
  __int16 v147; // [rsp+130h] [rbp-200h]
  _BYTE *v148; // [rsp+140h] [rbp-1F0h] BYREF
  __int64 v149; // [rsp+148h] [rbp-1E8h]
  _BYTE v150[48]; // [rsp+150h] [rbp-1E0h] BYREF
  _BYTE *v151; // [rsp+180h] [rbp-1B0h] BYREF
  __int64 v152; // [rsp+188h] [rbp-1A8h]
  _BYTE v153[48]; // [rsp+190h] [rbp-1A0h] BYREF
  __int64 v154; // [rsp+1C0h] [rbp-170h] BYREF
  _QWORD *v155; // [rsp+1C8h] [rbp-168h]
  __int64 v156; // [rsp+1D0h] [rbp-160h]
  unsigned int v157; // [rsp+1D8h] [rbp-158h]
  _QWORD *v158; // [rsp+1E8h] [rbp-148h]
  unsigned int v159; // [rsp+1F8h] [rbp-138h]
  char v160; // [rsp+200h] [rbp-130h]
  void *v161; // [rsp+210h] [rbp-120h] BYREF
  unsigned __int64 v162; // [rsp+218h] [rbp-118h] BYREF
  __int64 v163; // [rsp+220h] [rbp-110h]
  __int64 v164; // [rsp+228h] [rbp-108h]
  __int64 v165; // [rsp+230h] [rbp-100h] BYREF
  __m128i v166; // [rsp+270h] [rbp-C0h] BYREF
  __m128i v167; // [rsp+280h] [rbp-B0h] BYREF
  __int64 *i; // [rsp+290h] [rbp-A0h]
  __int64 v169; // [rsp+2A0h] [rbp-90h]
  unsigned __int64 v170; // [rsp+2A8h] [rbp-88h]
  __int64 v171; // [rsp+2B0h] [rbp-80h]
  __int64 *v172; // [rsp+2B8h] [rbp-78h]
  void **v173; // [rsp+2C0h] [rbp-70h]
  void **v174; // [rsp+2C8h] [rbp-68h]
  __int64 v175; // [rsp+2D0h] [rbp-60h]
  int v176; // [rsp+2D8h] [rbp-58h]
  __int16 v177; // [rsp+2DCh] [rbp-54h]
  char v178; // [rsp+2DEh] [rbp-52h]
  __int64 v179; // [rsp+2E0h] [rbp-50h]
  __int64 v180; // [rsp+2E8h] [rbp-48h]
  void *v181; // [rsp+2F0h] [rbp-40h] BYREF
  void *v182; // [rsp+2F8h] [rbp-38h] BYREF

  if ( !a2 )
    return;
  v154 = 0;
  v119 = sub_BD5C60(a1);
  v157 = 128;
  v8 = (_QWORD *)sub_C7D670(0x2000, 8);
  v156 = 0;
  v155 = v8;
  v166.m128i_i64[1] = 2;
  v9 = v8 + 1024;
  v166.m128i_i64[0] = (__int64)&unk_49DD7B0;
  v167.m128i_i64[0] = 0;
  v167.m128i_i64[1] = -4096;
  for ( i = 0; v9 != v8; v8 += 8 )
  {
    if ( v8 )
    {
      v10 = v166.m128i_i8[8];
      v8[2] = 0;
      v8[3] = -4096;
      *v8 = &unk_49DD7B0;
      v8[1] = v10 & 6;
      v8[4] = i;
    }
  }
  v160 = 0;
  v148 = v150;
  v149 = 0x600000000LL;
  v136[3] = (__int64)&v148;
  v136[0] = a5;
  v136[2] = (__int64)&v154;
  v161 = 0;
  v11 = a3 + 24 * a4;
  v12 = a3;
  v162 = (unsigned __int64)&v165;
  v127 = v11;
  v13 = v11;
  v163 = 8;
  LODWORD(v164) = 0;
  BYTE4(v164) = 1;
  for ( v136[1] = (__int64)&v161; v13 != v12; v12 += 24 )
  {
    if ( *(_DWORD *)v12 != 42 )
    {
      v14 = sub_B532B0(*(_DWORD *)v12);
      v166 = *(__m128i *)(v12 + 8);
      sub_27468F0(v136, &v166, 2, v14, v15, v16);
    }
  }
  v17 = sub_B532B0(*(_WORD *)(a1 + 2) & 0x3F);
  v166.m128i_i64[0] = a1;
  sub_27468F0(v136, &v166, 1, v17, v18, v19);
  v151 = v153;
  v152 = 0x600000000LL;
  v21 = &v148[8 * (unsigned int)v149];
  if ( v21 == v148 )
  {
    v27 = (__int64)v153;
    v24 = 0;
  }
  else
  {
    v22 = v153;
    v23 = v148 + 8;
    v24 = 0;
    v25 = *(_QWORD *)(*(_QWORD *)v148 + 8LL);
    v26 = &v151;
    while ( 1 )
    {
      *(_QWORD *)&v22[8 * v24] = v25;
      v24 = (unsigned int)(v152 + 1);
      LODWORD(v152) = v152 + 1;
      if ( v21 == v23 )
        break;
      v25 = *(_QWORD *)(*(_QWORD *)v23 + 8LL);
      if ( v24 + 1 > (unsigned __int64)HIDWORD(v152) )
      {
        v134 = v26;
        sub_C8D5F0((__int64)v26, v153, v24 + 1, 8u, v24 + 1, v20);
        v24 = (unsigned int)v152;
        v26 = v134;
      }
      v22 = v151;
      v23 += 8;
    }
    v27 = (__int64)v151;
  }
  v28 = sub_BCF480(*(__int64 **)(a1 + 8), (const void *)v27, v24, 0);
  v147 = 259;
  v145 = "repro";
  v29 = sub_B43CB0(a1);
  v30 = sub_BD5D20(v29);
  v141 = 261;
  v138 = v31;
  v137 = v30;
  v32 = sub_B43CA0(a1);
  v33 = *(unsigned __int128 **)(v32 + 168);
  v34 = *(_QWORD *)(v32 + 176);
  v35 = v141;
  if ( !(_BYTE)v141 )
  {
    LOWORD(v144) = 256;
LABEL_19:
    LOWORD(i) = 256;
    goto LABEL_20;
  }
  if ( (_BYTE)v141 == 1 )
  {
    v142 = __PAIR128__(v34, (unsigned __int64)v33);
    LOWORD(v144) = 261;
    v114 = v147;
    if ( !(_BYTE)v147 )
      goto LABEL_19;
    if ( (_BYTE)v147 != 1 )
    {
      v129 = v34;
      v27 = 5;
      goto LABEL_167;
    }
LABEL_175:
    si128 = _mm_load_si128((const __m128i *)&v142);
    v117 = _mm_load_si128(&v143);
    i = v144;
    v166 = si128;
    v167 = v117;
    goto LABEL_20;
  }
  if ( HIBYTE(v141) == 1 )
  {
    v128 = (__int64)v138;
    v113 = v137;
  }
  else
  {
    v113 = &v137;
    v35 = 2;
  }
  v143.m128i_i64[0] = (__int64)v113;
  v27 = v128;
  BYTE1(v144) = v35;
  v114 = v147;
  v142 = __PAIR128__(v34, (unsigned __int64)v33);
  v143.m128i_i64[1] = v128;
  LOBYTE(v144) = 5;
  if ( !(_BYTE)v147 )
    goto LABEL_19;
  if ( (_BYTE)v147 == 1 )
    goto LABEL_175;
  v33 = &v142;
  v27 = 2;
LABEL_167:
  if ( HIBYTE(v147) == 1 )
  {
    v125 = v146;
    v115 = (const char **)v145;
  }
  else
  {
    v115 = &v145;
    v114 = 2;
  }
  v167.m128i_i64[0] = (__int64)v115;
  v166.m128i_i64[0] = (__int64)v33;
  v166.m128i_i64[1] = v129;
  v167.m128i_i64[1] = v125;
  LOBYTE(i) = v27;
  BYTE1(i) = v114;
LABEL_20:
  v36 = sub_BD2DA0(136);
  v39 = v36;
  if ( v36 )
  {
    v27 = v28;
    sub_B2C3B0(v36, v28, 0, 0xFFFFFFFF, (__int64)&v166, a2);
  }
  if ( (_DWORD)v149 )
  {
    v40 = 0;
    v41 = 0;
    while ( 1 )
    {
      if ( (*(_BYTE *)(v39 + 2) & 1) != 0 )
        sub_B2C6D0(v39, v27, (__int64)v37, v38);
      v131 = 8 * v41;
      v42 = 40 * v41;
      v130 = (unsigned __int8 *)(40 * v41 + *(_QWORD *)(v39 + 96));
      v43.m128i_i64[0] = (__int64)sub_BD5D20(*(_QWORD *)&v148[8 * v41]);
      LOWORD(i) = 261;
      v166 = v43;
      sub_BD6B50(v130, (const char **)&v166);
      if ( (*(_BYTE *)(v39 + 2) & 1) != 0 )
        sub_B2C6D0(v39, (__int64)&v166, v44, v45);
      v46 = *(_QWORD *)(v39 + 96);
      v47 = *(_QWORD *)&v148[v131];
      v166.m128i_i64[1] = 2;
      v167.m128i_i64[0] = 0;
      v48 = v42 + v46;
      v167.m128i_i64[1] = v47;
      if ( v47 != 0 && v47 != -4096 && v47 != -8192 )
        sub_BD73F0((__int64)&v166.m128i_i64[1]);
      v49 = v157;
      v166.m128i_i64[0] = (__int64)&unk_49DD7B0;
      i = &v154;
      if ( !v157 )
        break;
      v50 = v167.m128i_i64[1];
      v38 = (v157 - 1) & (((unsigned __int32)v167.m128i_i32[2] >> 9) ^ ((unsigned __int32)v167.m128i_i32[2] >> 4));
      v56 = &v155[8 * v38];
      v57 = v56[3];
      if ( v167.m128i_i64[1] != v57 )
      {
        v108 = 1;
        v51 = 0;
        while ( v57 != -4096 )
        {
          if ( v57 == -8192 && !v51 )
            v51 = (const char *)v56;
          v38 = (v157 - 1) & (v108 + (_DWORD)v38);
          v56 = &v155[8 * (unsigned __int64)(unsigned int)v38];
          v57 = v56[3];
          if ( v167.m128i_i64[1] == v57 )
            goto LABEL_46;
          ++v108;
        }
        if ( !v51 )
          v51 = (const char *)v56;
        ++v154;
        v38 = (unsigned int)(v156 + 1);
        v145 = v51;
        if ( 4 * (int)v38 < 3 * v157 )
        {
          if ( v157 - HIDWORD(v156) - (unsigned int)v38 <= v157 >> 3 )
          {
LABEL_34:
            sub_CF32C0((__int64)&v154, v49);
            sub_F9E960((__int64)&v154, (__int64)&v166, &v145);
            v50 = v167.m128i_i64[1];
            v51 = v145;
            v38 = (unsigned int)(v156 + 1);
          }
          v52 = *((_QWORD *)v51 + 3);
          LODWORD(v156) = v38;
          if ( v52 == -4096 )
          {
            v53 = (unsigned __int64 *)(v51 + 8);
            if ( v50 != -4096 )
              goto LABEL_40;
          }
          else
          {
            --HIDWORD(v156);
            if ( v52 != v50 )
            {
              v53 = (unsigned __int64 *)(v51 + 8);
              if ( v52 && v52 != -8192 )
              {
                sub_BD60C0(v53);
                v50 = v167.m128i_i64[1];
                v53 = (unsigned __int64 *)(v51 + 8);
              }
LABEL_40:
              *((_QWORD *)v51 + 3) = v50;
              if ( v50 != 0 && v50 != -4096 && v50 != -8192 )
                sub_BD6050(v53, v166.m128i_i64[1] & 0xFFFFFFFFFFFFFFF8LL);
              v50 = v167.m128i_i64[1];
            }
          }
          v54 = i;
          v55 = v51 + 40;
          *v55 = 6;
          v55[1] = 0;
          *(v55 - 1) = v54;
          v55[2] = 0;
          goto LABEL_47;
        }
LABEL_33:
        v49 = 2 * v157;
        goto LABEL_34;
      }
LABEL_46:
      v55 = v56 + 5;
LABEL_47:
      v27 = (__int64)&unk_49DB358;
      LOBYTE(v38) = v50 != -4096;
      v37 = &unk_49DB368;
      v166.m128i_i64[0] = (__int64)&unk_49DB368;
      LOBYTE(v37) = v50 != 0;
      if ( ((v50 != 0) & (unsigned __int8)v38) != 0 && v50 != -8192 )
        sub_BD60C0(&v166.m128i_i64[1]);
      v58 = v55[2];
      if ( v48 != v58 )
      {
        LOBYTE(v38) = v58 != -4096;
        if ( ((v58 != 0) & (unsigned __int8)v38) != 0 && v58 != -8192 )
          sub_BD60C0(v55);
        v55[2] = v48;
        LOBYTE(v37) = v48 != -4096;
        if ( ((v48 != 0) & (unsigned __int8)v37) != 0 && v48 != -8192 )
          sub_BD73F0((__int64)v55);
      }
      v41 = (unsigned int)(v40 + 1);
      v40 = v41;
      if ( (unsigned int)v41 >= (unsigned int)v149 )
        goto LABEL_58;
    }
    ++v154;
    v145 = 0;
    goto LABEL_33;
  }
LABEL_58:
  v166.m128i_i64[0] = (__int64)"entry";
  LOWORD(i) = 259;
  v126 = sub_22077B0(0x50u);
  if ( v126 )
    sub_AA4D50(v126, v119, (__int64)&v166, v39, 0);
  v59 = (__int64 *)sub_AA48A0(v126);
  v169 = v126;
  LOWORD(v171) = 0;
  v166.m128i_i64[0] = (__int64)&v167;
  v166.m128i_i64[1] = 0x200000000LL;
  v173 = &v181;
  v174 = &v182;
  v172 = v59;
  v177 = 512;
  v181 = &unk_49DA100;
  v175 = 0;
  v176 = 0;
  v182 = &unk_49DA0B0;
  v178 = 7;
  v179 = 0;
  v180 = 0;
  v124 = v126 + 48;
  v170 = v126 + 48;
  v60 = sub_ACD6D0(v59);
  v61 = (__int64)v172;
  v62 = v60;
  v147 = 257;
  v132 = v60 != 0;
  v63 = sub_BD2C40(72, v132);
  v64 = (__int64)v63;
  if ( v63 )
    sub_B4BB80((__int64)v63, v61, v62, v132, 0, 0);
  (*((void (__fastcall **)(void **, __int64, const char **, unsigned __int64, __int64))*v174 + 2))(
    v174,
    v64,
    &v145,
    v170,
    v171);
  v65 = v166.m128i_i64[0];
  v66 = v166.m128i_i64[0] + 16LL * v166.m128i_u32[2];
  if ( v166.m128i_i64[0] != v66 )
  {
    do
    {
      v67 = *(_QWORD *)(v65 + 8);
      v68 = *(_DWORD *)v65;
      v65 += 16;
      sub_B99FD0(v64, v68, v67);
    }
    while ( v66 != v65 );
  }
  v133 = *(_QWORD *)(v126 + 48);
  v69 = v133 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (v133 & 0xFFFFFFFFFFFFFFF8LL) == v124 )
    goto LABEL_188;
  if ( !v69 )
    BUG();
  if ( (unsigned int)*(unsigned __int8 *)(v69 - 24) - 30 > 0xA )
LABEL_188:
    BUG();
  v70 = *(_QWORD *)(v69 + 16);
  v170 = v133 & 0xFFFFFFFFFFFFFFF8LL;
  LOWORD(v171) = 0;
  v169 = v70;
  v71 = *(_QWORD *)sub_B46C60(v69 - 24);
  v145 = (const char *)v71;
  if ( !v71 || (sub_B96E90((__int64)&v145, v71, 1), (v74 = (__int64)v145) == 0) )
  {
    sub_93FB40((__int64)&v166, 0);
    v74 = (__int64)v145;
    goto LABEL_134;
  }
  v75 = (_QWORD *)v166.m128i_i64[0];
  v76 = v166.m128i_i32[2];
  v77 = (_QWORD *)(v166.m128i_i64[0] + 16LL * v166.m128i_u32[2]);
  if ( (_QWORD *)v166.m128i_i64[0] == v77 )
  {
LABEL_141:
    if ( v166.m128i_u32[2] >= (unsigned __int64)v166.m128i_u32[3] )
    {
      v118 = v166.m128i_u32[2] + 1LL;
      if ( v166.m128i_u32[3] < v118 )
      {
        sub_C8D5F0((__int64)&v166, &v167, v118, 0x10u, v72, v73);
        v77 = (_QWORD *)(v166.m128i_i64[0] + 16LL * v166.m128i_u32[2]);
      }
      *v77 = 0;
      v77[1] = v74;
      v74 = (__int64)v145;
      ++v166.m128i_i32[2];
    }
    else
    {
      if ( v77 )
      {
        *(_DWORD *)v77 = 0;
        v77[1] = v74;
        v76 = v166.m128i_i32[2];
        v74 = (__int64)v145;
      }
      v166.m128i_i32[2] = v76 + 1;
    }
LABEL_134:
    if ( !v74 )
      goto LABEL_75;
    goto LABEL_74;
  }
  while ( *(_DWORD *)v75 )
  {
    v75 += 2;
    if ( v77 == v75 )
      goto LABEL_141;
  }
  v75[1] = v145;
LABEL_74:
  sub_B91220((__int64)&v145, v74);
LABEL_75:
  v140 = &v166;
  v78 = a3;
  v137 = (_QWORD *)a5;
  v138 = &v154;
  for ( j = a6; v127 != v78; v78 += 24 )
  {
    if ( *(_DWORD *)v78 == 42 )
      continue;
    v80 = sub_B532B0(*(_DWORD *)v78);
    v145 = *(const char **)(v78 + 8);
    v146 = *(_QWORD *)(v78 + 16);
    sub_2745AB0(&v137, &v145, 2, v80, v81, v82);
    v83 = *(unsigned __int8 **)(v78 + 16);
    v84 = *(_QWORD *)(v78 + 8);
    v85 = *(_DWORD *)v78;
    LOWORD(v144) = 257;
    v86 = (__int64 (__fastcall *)(__int64, unsigned int, _BYTE *, unsigned __int8 *))*((_QWORD *)*v173 + 7);
    if ( v86 == sub_928890 )
    {
      if ( *(_BYTE *)v84 > 0x15u || *v83 > 0x15u )
      {
LABEL_85:
        v147 = 257;
        v79 = (__int64)sub_BD2C40(72, unk_3F10FD0);
        if ( v79 )
        {
          v87 = *(_QWORD ***)(v84 + 8);
          v88 = *((unsigned __int8 *)v87 + 8);
          if ( (unsigned int)(v88 - 17) > 1 )
          {
            v90 = sub_BCB2A0(*v87);
          }
          else
          {
            BYTE4(v135) = (_BYTE)v88 == 18;
            LODWORD(v135) = *((_DWORD *)v87 + 8);
            v89 = (__int64 *)sub_BCB2A0(*v87);
            v90 = sub_BCE1B0(v89, v135);
          }
          sub_B523C0(v79, v90, 53, v85, v84, (__int64)v83, (__int64)&v145, 0, 0, 0);
        }
        (*((void (__fastcall **)(void **, __int64, unsigned __int128 *, unsigned __int64, __int64))*v174 + 2))(
          v174,
          v79,
          &v142,
          v170,
          v171);
        v91 = v166.m128i_i64[0];
        v92 = v166.m128i_i64[0] + 16LL * v166.m128i_u32[2];
        if ( v166.m128i_i64[0] != v92 )
        {
          do
          {
            v93 = *(_QWORD *)(v91 + 8);
            v94 = *(_DWORD *)v91;
            v91 += 16;
            sub_B99FD0(v79, v94, v93);
          }
          while ( v92 != v91 );
        }
        goto LABEL_80;
      }
      v79 = sub_AAB310(v85, (unsigned __int8 *)v84, v83);
    }
    else
    {
      v79 = v86((__int64)v173, v85, (_BYTE *)v84, v83);
    }
    if ( !v79 )
      goto LABEL_85;
LABEL_80:
    sub_B33B40((__int64)&v166, v79, 0, 0);
  }
  v95 = sub_B532B0(*(_WORD *)(a1 + 2) & 0x3F);
  v145 = (const char *)a1;
  sub_2745AB0(&v137, &v145, 1, v95, v96, v97);
  v98 = *(_QWORD *)(v126 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v124 == v98 )
    goto LABEL_190;
  if ( !v98 )
    BUG();
  if ( (unsigned int)*(unsigned __int8 *)(v98 - 24) - 30 > 0xA )
LABEL_190:
    BUG();
  if ( (*(_BYTE *)(v98 - 17) & 0x40) != 0 )
    v99 = *(_QWORD *)(v98 - 32);
  else
    v99 = v98 - 24 - 32LL * (*(_DWORD *)(v98 - 20) & 0x7FFFFFF);
  if ( *(_QWORD *)v99 )
  {
    v100 = *(_QWORD *)(v99 + 8);
    **(_QWORD **)(v99 + 16) = v100;
    if ( v100 )
      *(_QWORD *)(v100 + 16) = *(_QWORD *)(v99 + 16);
  }
  *(_QWORD *)v99 = a1;
  v101 = *(_QWORD *)(a1 + 16);
  *(_QWORD *)(v99 + 8) = v101;
  if ( v101 )
    *(_QWORD *)(v101 + 16) = v99 + 8;
  *(_QWORD *)(v99 + 16) = a1 + 16;
  *(_QWORD *)(a1 + 16) = v99;
  v145 = (const char *)v126;
  sub_F45F60((__int64)&v145, 1, (__int64)&v154);
  nullsub_61();
  v181 = &unk_49DA100;
  nullsub_63();
  if ( (__m128i *)v166.m128i_i64[0] != &v167 )
    _libc_free(v166.m128i_u64[0]);
  if ( v151 != v153 )
    _libc_free((unsigned __int64)v151);
  if ( !BYTE4(v164) )
    _libc_free(v162);
  if ( v148 != v150 )
    _libc_free((unsigned __int64)v148);
  if ( v160 )
  {
    v109 = v159;
    v160 = 0;
    if ( v159 )
    {
      v110 = v158;
      v111 = &v158[2 * v159];
      do
      {
        if ( *v110 != -4096 && *v110 != -8192 )
        {
          v112 = v110[1];
          if ( v112 )
            sub_B91220((__int64)(v110 + 1), v112);
        }
        v110 += 2;
      }
      while ( v111 != v110 );
      v109 = v159;
    }
    sub_C7D6A0((__int64)v158, 16LL * v109, 8);
  }
  v102 = v157;
  if ( v157 )
  {
    v103 = v155;
    v162 = 2;
    v163 = 0;
    v104 = &v155[8 * (unsigned __int64)v157];
    v164 = -4096;
    v161 = &unk_49DD7B0;
    v166.m128i_i64[0] = (__int64)&unk_49DD7B0;
    v105 = -4096;
    v165 = 0;
    v166.m128i_i64[1] = 2;
    v167.m128i_i64[0] = 0;
    v167.m128i_i64[1] = -8192;
    i = 0;
    while ( 1 )
    {
      v106 = v103[3];
      if ( v106 != v105 )
      {
        v105 = v167.m128i_i64[1];
        if ( v106 != v167.m128i_i64[1] )
        {
          v107 = v103[7];
          if ( v107 != 0 && v107 != -4096 && v107 != -8192 )
          {
            sub_BD60C0(v103 + 5);
            v106 = v103[3];
          }
          v105 = v106;
        }
      }
      *v103 = &unk_49DB368;
      if ( v105 != 0 && v105 != -4096 && v105 != -8192 )
        sub_BD60C0(v103 + 1);
      v103 += 8;
      if ( v104 == v103 )
        break;
      v105 = v164;
    }
    v166.m128i_i64[0] = (__int64)&unk_49DB368;
    if ( v167.m128i_i64[1] != -4096 && v167.m128i_i64[1] != 0 && v167.m128i_i64[1] != -8192 )
      sub_BD60C0(&v166.m128i_i64[1]);
    v161 = &unk_49DB368;
    if ( v164 != 0 && v164 != -4096 && v164 != -8192 )
      sub_BD60C0(&v162);
    v102 = v157;
  }
  sub_C7D6A0((__int64)v155, v102 << 6, 8);
}
