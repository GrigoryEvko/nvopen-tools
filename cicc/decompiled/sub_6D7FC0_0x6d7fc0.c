// Function: sub_6D7FC0
// Address: 0x6d7fc0
//
__int64 __fastcall sub_6D7FC0(__m128i *a1, __int64 p_IO_read_end, __int64 a3, __int64 a4, _QWORD *a5, __m128i *a6)
{
  __int64 v7; // rbx
  __int16 v8; // r12
  __m128i *p_IO_backup_base; // r13
  unsigned __int16 *v10; // rcx
  __int64 v11; // rax
  __int64 IO_read_end_low; // rdi
  __int64 v13; // rcx
  __int64 v14; // rax
  _QWORD *i; // rdx
  char v16; // al
  __int64 v17; // r12
  char *v18; // xmm4_8
  __m128i v19; // xmm6
  __m128i v20; // xmm7
  int v21; // edx
  int v22; // ecx
  __int64 v23; // rax
  char kk; // dl
  _QWORD *v25; // r10
  __int16 v26; // ax
  char *v27; // rax
  char v28; // al
  __int64 v29; // rax
  char i2; // dl
  char j; // al
  __int64 *v33; // r15
  __int64 v34; // rax
  __int64 v35; // r11
  __int64 v36; // r15
  __int64 v37; // rax
  __int64 v38; // r11
  __int64 v39; // r10
  __int64 v40; // r8
  unsigned __int8 v41; // al
  __m128i v42; // xmm1
  __m128i v43; // xmm2
  __m128i v44; // xmm3
  int v45; // eax
  __int64 v46; // rax
  __int64 v47; // r10
  char **v48; // r8
  char v49; // al
  __m128i v50; // xmm1
  __m128i v51; // xmm2
  __m128i v52; // xmm3
  char v53; // dl
  __int64 v54; // rax
  __int64 v55; // rcx
  __int64 v56; // rdx
  __int64 v57; // rcx
  __int64 v58; // r8
  int v59; // r10d
  __m128i v60; // xmm2
  __m128i v61; // xmm3
  __m128i v62; // xmm4
  char *v63; // rax
  char *IO_read_ptr; // rax
  __int64 v65; // rax
  unsigned int v66; // ebx
  char *IO_write_ptr; // r11
  unsigned __int64 v68; // rcx
  char *v69; // rdx
  char **v70; // rax
  FILE *v71; // rsi
  __int64 v72; // rbx
  __int64 v73; // rcx
  __int64 v74; // r8
  char *v75; // r11
  __int64 v76; // rdx
  char v77; // al
  __int64 v78; // rax
  int v79; // eax
  __int64 v80; // rcx
  unsigned __int8 v81; // al
  int v82; // eax
  char v83; // dl
  __int64 v84; // rax
  __int64 v85; // rax
  __int64 v86; // rbx
  __int64 v87; // rax
  char v88; // dl
  int v89; // eax
  __int64 v90; // r10
  __m128i v91; // xmm1
  __m128i v92; // xmm2
  __m128i v93; // xmm3
  __int64 v94; // rax
  char k; // dl
  __m128i v96; // xmm5
  __m128i v97; // xmm6
  __m128i v98; // xmm7
  __int64 v99; // rax
  __int64 v100; // rax
  int v101; // edx
  __m128i v102; // xmm1
  __m128i v103; // xmm2
  __m128i v104; // xmm3
  __int64 v105; // rdx
  __int64 v106; // rcx
  __int64 v107; // r8
  unsigned int v108; // eax
  __int64 v109; // rcx
  __int64 v110; // r8
  __int64 v111; // rdx
  int v112; // r10d
  __int64 v113; // rcx
  __int64 v114; // r8
  __int64 jj; // rcx
  int v116; // eax
  unsigned int v117; // eax
  int v118; // eax
  __int64 v119; // r11
  char v120; // al
  bool v121; // zf
  __int64 v122; // rax
  __int64 v123; // rax
  __int64 v124; // r11
  __m128i v125; // xmm5
  __m128i v126; // xmm6
  __m128i v127; // xmm7
  char IO_backup_base; // al
  unsigned __int8 v129; // cl
  __int64 v130; // rax
  __int8 v131; // al
  __int64 v132; // r9
  __int64 mm; // rdi
  int v134; // eax
  __int64 v135; // rdx
  int v136; // ecx
  __int64 v137; // rax
  char *IO_write_end; // r15
  __int64 v139; // r8
  char v140; // dl
  __int64 v141; // rax
  __m128i v142; // xmm1
  __m128i v143; // xmm2
  __m128i v144; // xmm3
  __m128i v145; // xmm4
  __m128i v146; // xmm5
  __m128i v147; // xmm6
  __m128i v148; // xmm7
  __m128i v149; // xmm0
  __int8 v150; // al
  _QWORD *v151; // rax
  __int64 v152; // rcx
  _QWORD *v153; // r8
  int v154; // eax
  __int64 v155; // rax
  int v156; // eax
  __int64 v157; // rcx
  __int64 v158; // r8
  __int64 v159; // r15
  __int64 v160; // rdx
  __int64 v161; // rcx
  __int64 v162; // r8
  __int64 v163; // r9
  __int64 v164; // rax
  __int64 v165; // rax
  __int64 v166; // rdi
  int v167; // ecx
  __int64 v168; // rsi
  __int64 i1; // rax
  __int64 v170; // r15
  __int64 v171; // rdi
  __int64 v172; // rdx
  int v173; // eax
  char v174; // al
  __int64 v175; // r9
  __int64 v176; // rax
  char *v177; // rax
  __int64 v178; // rax
  int v179; // eax
  __int64 v180; // rax
  _DWORD *v181; // r10
  __int64 v182; // rdx
  __int64 v183; // rcx
  __int64 v184; // r8
  __int64 v185; // r15
  __int64 v186; // rax
  __int64 v187; // rax
  __int64 ii; // rax
  __int64 v189; // rcx
  __int64 v190; // r15
  __int64 v191; // rax
  int v192; // eax
  __int64 v193; // rax
  __int64 v194; // rax
  __int64 v195; // rdx
  __int64 v196; // rcx
  __int64 v197; // r8
  __int64 v198; // rdi
  int v199; // eax
  int v200; // ecx
  __int64 v201; // rax
  __int64 v202; // rcx
  __int64 v203; // rdx
  char v204; // cl
  __int64 n; // rdx
  __int64 v206; // rdx
  char **v207; // rdi
  __int64 v208; // rcx
  __m128i *v209; // rsi
  __int64 v210; // rax
  __int64 v211; // rdx
  char v212; // al
  __m128i v213; // xmm2
  __m128i v214; // xmm3
  __m128i v215; // xmm4
  __m128i v216; // xmm0
  __m128i v217; // xmm5
  __m128i v218; // xmm1
  __m128i v219; // xmm6
  __m128i v220; // xmm2
  __m128i v221; // xmm7
  __m128i v222; // xmm3
  __m128i v223; // xmm4
  __m128i v224; // xmm0
  __int8 v225; // al
  int v226; // eax
  int v227; // edx
  __int64 v228; // rax
  __int64 v229; // rax
  __int64 nn; // rax
  __int64 v231; // rax
  __int64 m; // rax
  int v233; // eax
  __int64 v234; // rax
  __m128i v235; // xmm7
  __m128i v236; // xmm0
  __m128i v237; // xmm1
  __int64 v238; // rax
  __int32 v239; // r15d
  _DWORD *v240; // r15
  __int64 v241; // rdx
  __int64 v242; // rcx
  __int64 v243; // r8
  __int64 v244; // r9
  __int64 v245; // rax
  __int64 v246; // rcx
  __int64 v247; // r8
  __int64 v248; // r9
  __int64 v249; // rdx
  __int64 v250; // rax
  int v251; // eax
  __int64 v252; // [rsp+20h] [rbp-3F0h]
  int v253; // [rsp+20h] [rbp-3F0h]
  __int64 v254; // [rsp+28h] [rbp-3E8h]
  __int64 v255; // [rsp+28h] [rbp-3E8h]
  unsigned int v256; // [rsp+28h] [rbp-3E8h]
  char *v257; // [rsp+28h] [rbp-3E8h]
  char **v258; // [rsp+30h] [rbp-3E0h]
  __int64 v259; // [rsp+30h] [rbp-3E0h]
  int v260; // [rsp+38h] [rbp-3D8h]
  __int64 v261; // [rsp+38h] [rbp-3D8h]
  int v262; // [rsp+38h] [rbp-3D8h]
  __int64 v263; // [rsp+38h] [rbp-3D8h]
  _BOOL4 v264; // [rsp+40h] [rbp-3D0h]
  int v265; // [rsp+40h] [rbp-3D0h]
  _BOOL4 v267; // [rsp+54h] [rbp-3BCh]
  int v268; // [rsp+60h] [rbp-3B0h]
  unsigned int v269; // [rsp+60h] [rbp-3B0h]
  __int64 v270; // [rsp+60h] [rbp-3B0h]
  __int64 v271; // [rsp+60h] [rbp-3B0h]
  int v272; // [rsp+60h] [rbp-3B0h]
  char *v273; // [rsp+60h] [rbp-3B0h]
  int v274; // [rsp+68h] [rbp-3A8h]
  __int64 v275; // [rsp+68h] [rbp-3A8h]
  __int64 v276; // [rsp+68h] [rbp-3A8h]
  __int64 v277; // [rsp+68h] [rbp-3A8h]
  _BOOL4 v278; // [rsp+68h] [rbp-3A8h]
  __int64 v279; // [rsp+68h] [rbp-3A8h]
  __int64 v280; // [rsp+68h] [rbp-3A8h]
  int v281; // [rsp+70h] [rbp-3A0h]
  char *v282; // [rsp+70h] [rbp-3A0h]
  char *v283; // [rsp+70h] [rbp-3A0h]
  char *v284; // [rsp+70h] [rbp-3A0h]
  char *v285; // [rsp+70h] [rbp-3A0h]
  int v286; // [rsp+78h] [rbp-398h]
  _BOOL4 v287; // [rsp+7Ch] [rbp-394h]
  bool v288; // [rsp+80h] [rbp-390h]
  char *v289; // [rsp+80h] [rbp-390h]
  __int64 v290; // [rsp+80h] [rbp-390h]
  char *v291; // [rsp+80h] [rbp-390h]
  __int64 v292; // [rsp+80h] [rbp-390h]
  char *v293; // [rsp+80h] [rbp-390h]
  char *v294; // [rsp+80h] [rbp-390h]
  __int64 v295; // [rsp+80h] [rbp-390h]
  __int64 v296; // [rsp+88h] [rbp-388h]
  _QWORD *v297; // [rsp+88h] [rbp-388h]
  int v298; // [rsp+88h] [rbp-388h]
  unsigned int v299; // [rsp+88h] [rbp-388h]
  int v300; // [rsp+88h] [rbp-388h]
  __int64 v301; // [rsp+88h] [rbp-388h]
  char *v302; // [rsp+88h] [rbp-388h]
  __int64 v303; // [rsp+88h] [rbp-388h]
  __int64 v304; // [rsp+88h] [rbp-388h]
  _QWORD *v305; // [rsp+88h] [rbp-388h]
  _QWORD *v306; // [rsp+88h] [rbp-388h]
  int v307; // [rsp+88h] [rbp-388h]
  int v308; // [rsp+88h] [rbp-388h]
  _DWORD *v309; // [rsp+88h] [rbp-388h]
  __int64 v310; // [rsp+88h] [rbp-388h]
  __int64 v311; // [rsp+88h] [rbp-388h]
  unsigned int v312; // [rsp+88h] [rbp-388h]
  __int64 v313; // [rsp+88h] [rbp-388h]
  __int8 v314; // [rsp+88h] [rbp-388h]
  int v315; // [rsp+9Ch] [rbp-374h] BYREF
  unsigned int v316; // [rsp+A0h] [rbp-370h] BYREF
  int v317; // [rsp+A4h] [rbp-36Ch] BYREF
  __int64 v318; // [rsp+A8h] [rbp-368h] BYREF
  FILE v319[2]; // [rsp+B0h] [rbp-360h] BYREF
  __m128i v320; // [rsp+260h] [rbp-1B0h] BYREF
  __m128i v321; // [rsp+270h] [rbp-1A0h]
  __m128i v322; // [rsp+280h] [rbp-190h]
  __m128i v323; // [rsp+290h] [rbp-180h]
  _OWORD v324[5]; // [rsp+2A0h] [rbp-170h] BYREF
  __m128i v325; // [rsp+2F0h] [rbp-120h]
  __m128i v326; // [rsp+300h] [rbp-110h]
  __m128i v327; // [rsp+310h] [rbp-100h]
  __m128i v328; // [rsp+320h] [rbp-F0h]
  __m128i v329; // [rsp+330h] [rbp-E0h]
  __m128i v330; // [rsp+340h] [rbp-D0h]
  __m128i v331; // [rsp+350h] [rbp-C0h]
  __m128i v332; // [rsp+360h] [rbp-B0h]
  __m128i v333; // [rsp+370h] [rbp-A0h]
  __m128i v334; // [rsp+380h] [rbp-90h]
  __m128i v335; // [rsp+390h] [rbp-80h]
  __m128i v336; // [rsp+3A0h] [rbp-70h]
  __m128i v337; // [rsp+3B0h] [rbp-60h]

  v7 = p_IO_read_end;
  v281 = a3;
  v286 = a4;
  v315 = 0;
  if ( p_IO_read_end )
  {
    v8 = *(_WORD *)(p_IO_read_end + 8);
    p_IO_backup_base = (__m128i *)&v319[0]._IO_backup_base;
    a1 = (__m128i *)p_IO_read_end;
    p_IO_read_end = (__int64)&v319[0]._IO_backup_base;
    sub_6F8DA0(a1, &v319[0]._IO_backup_base, a3, a4, &v318, &v316);
    v288 = v8 == 30;
    v267 = v8 == 30;
    v11 = qword_4D03C50;
    if ( (*(_BYTE *)(qword_4D03C50 + 19LL) & 0x40) == 0 )
    {
      v287 = v315;
      goto LABEL_4;
    }
  }
  else
  {
    v10 = word_4F06418;
    p_IO_backup_base = a1;
    v287 = 0;
    v288 = word_4F06418[0] == 30;
    v318 = *(_QWORD *)&dword_4F063F8;
    v316 = dword_4F06650[0];
    v267 = word_4F06418[0] == 30;
    v11 = qword_4D03C50;
    if ( (*(_BYTE *)(qword_4D03C50 + 19LL) & 0x40) == 0 )
      goto LABEL_4;
  }
  v16 = *(_BYTE *)(v11 + 16);
  if ( !v16 )
  {
    if ( (unsigned int)sub_6E5430(a1, p_IO_read_end, a3, v10, a5) )
    {
      p_IO_read_end = (__int64)&dword_4F063F8;
      sub_6851C0(0x3Au, &dword_4F063F8);
    }
    goto LABEL_192;
  }
  if ( (unsigned __int8)(v16 - 1) <= 1u )
  {
    if ( (unsigned int)sub_693A90() )
    {
      v287 = 1;
      goto LABEL_4;
    }
    if ( *(_BYTE *)(qword_4D03C50 + 16LL) == 1 )
    {
      if ( (unsigned int)sub_6E5430(a1, p_IO_read_end, v105, v106, v107) )
      {
        p_IO_read_end = (__int64)&dword_4F063F8;
        sub_6851C0(0x3Cu, &dword_4F063F8);
      }
    }
    else if ( (unsigned int)sub_6E5430(a1, p_IO_read_end, v105, v106, v107) )
    {
      p_IO_read_end = (__int64)&dword_4F063F8;
      sub_6851C0(0x211u, &dword_4F063F8);
    }
LABEL_192:
    v315 = 1;
    IO_read_end_low = (__int64)p_IO_backup_base;
    v287 = 0;
    sub_6E6450(p_IO_backup_base);
    if ( !v7 )
    {
      v261 = 0;
      v17 = 0;
      v274 = 0;
      goto LABEL_108;
    }
    v296 = 0;
    goto LABEL_20;
  }
  v287 = 0;
  if ( v16 != 3 || dword_4F077C4 != 2 )
  {
LABEL_4:
    if ( !v288 )
    {
LABEL_5:
      p_IO_read_end = 4;
      IO_read_end_low = (__int64)p_IO_backup_base;
      sub_6F69D0(p_IO_backup_base, 4);
      v288 = 0;
      goto LABEL_6;
    }
    goto LABEL_131;
  }
  if ( !v288 )
  {
    v287 = v315;
    goto LABEL_5;
  }
  if ( !(unsigned int)sub_693A90() )
    goto LABEL_500;
  v225 = p_IO_backup_base[1].m128i_i8[1];
  if ( v225 == 1 )
  {
    v287 = 1;
    if ( !(unsigned int)sub_6ED0A0(p_IO_backup_base) )
      goto LABEL_131;
    v225 = p_IO_backup_base[1].m128i_i8[1];
  }
  if ( v225 == 2 )
    v287 = p_IO_backup_base[1].m128i_i8[0] != 2;
  else
LABEL_500:
    v287 = 0;
LABEL_131:
  if ( dword_4F077C4 == 2 )
    sub_68E150((__int64)p_IO_backup_base, (__int64)&v318, v316, 1, 0);
  if ( v287 )
  {
    sub_6E25A0(&v319[0]._IO_read_end, &v320);
    sub_6F69D0(p_IO_backup_base, 0);
    IO_read_end_low = LOBYTE(v319[0]._IO_read_end);
    p_IO_read_end = v320.m128i_u32[0];
    sub_6E25E0(LOBYTE(v319[0]._IO_read_end), v320.m128i_u32[0]);
  }
  else
  {
    p_IO_read_end = 0;
    IO_read_end_low = (__int64)p_IO_backup_base;
    sub_6F69D0(p_IO_backup_base, 0);
  }
  v288 = 1;
LABEL_6:
  if ( !p_IO_backup_base[1].m128i_i8[0] )
    goto LABEL_48;
  v14 = p_IO_backup_base->m128i_i64[0];
  v296 = p_IO_backup_base->m128i_i64[0];
  for ( i = (_QWORD *)*(unsigned __int8 *)(p_IO_backup_base->m128i_i64[0] + 140);
        (_BYTE)i == 12;
        i = (_QWORD *)*(unsigned __int8 *)(v14 + 140) )
  {
    v14 = *(_QWORD *)(v14 + 160);
  }
  if ( !(_BYTE)i )
  {
LABEL_48:
    v315 = 1;
    IO_read_end_low = (__int64)p_IO_backup_base;
    sub_6E6870(p_IO_backup_base);
    v296 = 0;
    goto LABEL_49;
  }
  if ( v288 )
  {
    if ( dword_4F077C4 == 1 )
    {
      IO_read_end_low = p_IO_backup_base->m128i_i64[0];
      if ( (unsigned int)sub_8D2930(v296) )
        goto LABEL_105;
      v296 = p_IO_backup_base->m128i_i64[0];
    }
    IO_read_end_low = v296;
    if ( (unsigned int)sub_8DD3B0(v296) )
    {
      v296 = *(_QWORD *)&dword_4D03B80;
      goto LABEL_49;
    }
    p_IO_read_end = 3164;
    IO_read_end_low = (__int64)p_IO_backup_base;
    if ( (unsigned int)sub_6FB4D0(p_IO_backup_base, 3164) )
    {
      IO_read_end_low = p_IO_backup_base->m128i_i64[0];
      v296 = sub_8D46C0(p_IO_backup_base->m128i_i64[0]);
      goto LABEL_49;
    }
    v315 = 1;
LABEL_105:
    v296 = 0;
LABEL_106:
    if ( !v7 )
    {
      v17 = 0;
      v274 = 0;
      v261 = v296;
      goto LABEL_108;
    }
LABEL_20:
    v17 = 0;
    v315 = 1;
    v274 = 0;
    v18 = (char *)_mm_loadu_si128(xmmword_4F06660).m128i_u64[0];
    v268 = 0;
    v19 = _mm_loadu_si128(&xmmword_4F06660[2]);
    v20 = _mm_loadu_si128(&xmmword_4F06660[3]);
    *(__m128i *)&v319[0]._IO_write_base = _mm_loadu_si128(&xmmword_4F06660[1]);
    BYTE1(v319[0]._IO_write_base) |= 0x20u;
    v319[0]._IO_read_end = v18;
    v319[0]._IO_read_base = *(char **)dword_4F07508;
    *(__m128i *)&v319[0]._IO_write_end = v19;
    *(__m128i *)&v319[0]._IO_buf_end = v20;
LABEL_21:
    v319[0]._IO_read_ptr = *(char **)(*(_QWORD *)v7 + 44LL);
    goto LABEL_22;
  }
  if ( p_IO_backup_base[1].m128i_i8[1] == 1 )
  {
    IO_read_end_low = (__int64)p_IO_backup_base;
    if ( !(unsigned int)sub_6ED0A0(p_IO_backup_base) )
    {
      IO_read_end_low = (__int64)p_IO_backup_base;
      sub_6ECC10(p_IO_backup_base, p_IO_read_end, i);
    }
  }
LABEL_49:
  v260 = v315;
  if ( v315 )
    goto LABEL_106;
  v17 = v296;
  for ( j = *(_BYTE *)(v296 + 140); j == 12; j = *(_BYTE *)(v17 + 140) )
    v17 = *(_QWORD *)(v17 + 160);
  v33 = (__int64 *)v7;
  if ( j == 6 )
    goto LABEL_91;
  while ( 1 )
  {
    if ( j == 14 )
    {
      v65 = *(_QWORD *)v17;
      if ( *(_QWORD *)v17
        && *(char *)(v65 + 81) < 0
        && (v85 = *(_QWORD *)(v65 + 72), v86 = *(_QWORD *)(v85 + 88), (*(_BYTE *)(v86 + 141) & 0x20) == 0)
        && (IO_read_end_low = *(_QWORD *)(v85 + 88), (unsigned int)sub_85FB30(IO_read_end_low)) )
      {
        j = *(_BYTE *)(v86 + 140);
        v17 = v86;
      }
      else
      {
        v66 = 0;
        if ( (*(_BYTE *)(v296 + 140) & 0xFB) == 8 )
          v66 = sub_8D4C10(v296, dword_4F077C4 != 2);
        p_IO_read_end = v66;
        IO_read_end_low = sub_7CFE40(v17);
        v17 = IO_read_end_low;
        v296 = sub_73C570(IO_read_end_low, v66, -1);
        j = *(_BYTE *)(IO_read_end_low + 140);
      }
    }
    if ( (unsigned __int8)(j - 9) > 2u )
      goto LABEL_59;
    if ( dword_4F077C4 == 2 )
    {
      IO_read_end_low = v17;
      if ( (unsigned int)sub_8D23B0(v17) )
      {
        IO_read_end_low = v17;
        if ( (unsigned int)sub_8D3A70(v17) )
        {
          p_IO_read_end = 0;
          IO_read_end_low = v17;
          sub_8AD220(v17, 0);
        }
      }
    }
    if ( (*(_BYTE *)(v17 + 141) & 0x20) == 0
      || (v34 = *(_QWORD *)(*(_QWORD *)(v17 + 168) + 152LL)) != 0 && (*(_BYTE *)(v34 + 29) & 0x20) == 0
      || dword_4F04C44 == -1
      && (i = qword_4F04C68, v99 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v99 + 6) & 6) == 0)
      && *(_BYTE *)(v99 + 4) != 12 )
    {
LABEL_59:
      v7 = (__int64)v33;
      if ( !v33 )
      {
        v274 = 1;
        v261 = v296;
        goto LABEL_108;
      }
      v35 = *v33;
LABEL_61:
      v36 = *(_QWORD *)(v35 + 72);
      if ( !v281 || *(_BYTE *)(v35 + 24) != 1 || (unsigned __int8)(*(_BYTE *)(v35 + 56) - 105) > 4u )
        v36 = *(_QWORD *)(v36 + 16);
      if ( v36 )
      {
        p_IO_read_end = (__int64)&v320;
        IO_read_end_low = v36;
        v275 = v35;
        v37 = sub_6E3DA0(v36, &v320);
        v38 = v275;
        v39 = v37;
        v40 = v37 + 68;
        v264 = (*(_BYTE *)(v37 + 18) & 0x40) != 0;
        v268 = *(_BYTE *)(v37 + 19) >> 7;
      }
      else
      {
        IO_read_end_low = v35;
        p_IO_read_end = (__int64)&v320;
        v279 = v35;
        v100 = sub_6E3DA0(v35, &v320);
        v38 = v279;
        v40 = 0;
        v268 = 0;
        v264 = 0;
        v39 = v100;
      }
      v41 = *(_BYTE *)(v38 + 56);
      if ( v41 <= 0x5Fu )
      {
        if ( v41 <= 0x5Du )
        {
          if ( (unsigned __int8)(v41 - 22) > 1u )
            goto LABEL_75;
          IO_read_end_low = v17;
          v276 = v39;
          v42 = _mm_loadu_si128(&xmmword_4F06660[1]);
          v43 = _mm_loadu_si128(&xmmword_4F06660[2]);
          v44 = _mm_loadu_si128(&xmmword_4F06660[3]);
          v319[0]._IO_read_end = (char *)_mm_loadu_si128(xmmword_4F06660).m128i_u64[0];
          *(__m128i *)&v319[0]._IO_write_base = v42;
          LODWORD(v319[0]._IO_write_base) = v42.m128i_i32[0] | 0x20400;
          v319[0]._IO_buf_base = (char *)v43.m128i_i64[1];
          v319[0]._IO_read_base = *(char **)&dword_4F077C8;
          v319[0]._IO_write_end = (char *)v17;
          *(__m128i *)&v319[0]._IO_buf_end = v44;
          if ( (unsigned int)sub_8D3A70(v17) )
          {
            if ( v36 )
            {
LABEL_71:
              v319[0]._IO_read_base = *(char **)(v36 + 36);
LABEL_72:
              v45 = v315;
LABEL_84:
              if ( !v45 && v319[0]._IO_buf_base )
              {
                IO_read_end_low = (__int64)&v319[0]._IO_read_end;
                p_IO_read_end = v7;
                sub_691040((__int64)&v319[0]._IO_read_end, v7, 0);
                v274 = 1;
                goto LABEL_21;
              }
LABEL_176:
              v274 = 1;
              goto LABEL_21;
            }
          }
          else
          {
            BYTE1(v319[0]._IO_write_base) |= 8u;
            if ( v36 )
              goto LABEL_71;
          }
          v45 = v315;
          v319[0]._IO_read_base = *(char **)(v276 + 68);
          goto LABEL_84;
        }
        if ( !*(_BYTE *)(v39 + 16) )
          goto LABEL_181;
        v94 = *(_QWORD *)v39;
        for ( k = *(_BYTE *)(*(_QWORD *)v39 + 140LL); k == 12; k = *(_BYTE *)(v94 + 140) )
          v94 = *(_QWORD *)(v94 + 160);
        if ( !k )
        {
LABEL_181:
          *(_BYTE *)(v7 + 56) = 1;
          v315 = 1;
          v96 = _mm_loadu_si128(&xmmword_4F06660[1]);
          v97 = _mm_loadu_si128(&xmmword_4F06660[2]);
          v98 = _mm_loadu_si128(&xmmword_4F06660[3]);
          *(__m128i *)&v319[0]._IO_read_end = _mm_loadu_si128(xmmword_4F06660);
          *(__m128i *)&v319[0]._IO_write_base = v96;
          *(__m128i *)&v319[0]._IO_write_end = v97;
          *(__m128i *)&v319[0]._IO_buf_end = v98;
          if ( !v36 )
          {
            v274 = 1;
            v319[0]._IO_read_base = *(char **)(v39 + 68);
            goto LABEL_21;
          }
          v319[0]._IO_read_base = *(char **)(v36 + 36);
          goto LABEL_176;
        }
        v280 = v39;
        p_IO_read_end = (__int64)&v319[0]._IO_read_end;
        IO_read_end_low = sub_8A1CE0(
                            *(_QWORD *)(v39 + 136),
                            *(_QWORD *)(*(_QWORD *)(v39 + 136) + 64LL),
                            *(_QWORD *)(v7 + 24),
                            *(_QWORD *)(v7 + 32),
                            v40,
                            0,
                            0,
                            *(_DWORD *)(v7 + 40),
                            (__int64)&v315,
                            *(_QWORD *)(v7 + 48));
        sub_878710(IO_read_end_low, &v319[0]._IO_read_end);
        if ( v36 )
        {
          v319[0]._IO_read_base = *(char **)(v36 + 36);
          v45 = v315;
        }
        else
        {
          v45 = v315;
          v319[0]._IO_read_base = *(char **)(v280 + 68);
        }
LABEL_206:
        if ( v45 )
          goto LABEL_176;
        if ( !v264 )
          goto LABEL_84;
        IO_read_end_low = (__int64)&v319[0]._IO_read_end;
        p_IO_read_end = (__int64)&v319[0]._IO_read_base;
        if ( !(unsigned int)sub_688C20((__int64)&v319[0]._IO_read_end, (FILE *)&v319[0]._IO_read_base, v17) )
        {
          v315 = 1;
          v274 = 1;
          goto LABEL_21;
        }
        goto LABEL_72;
      }
      if ( v41 > 0x65u )
      {
        if ( (unsigned __int8)(v41 - 106) > 1u )
          goto LABEL_75;
      }
      else if ( v41 <= 0x63u )
      {
        goto LABEL_75;
      }
      p_IO_read_end = 0;
      IO_read_end_low = v36;
      v254 = v38;
      v258 = (char **)v40;
      v277 = v39;
      v46 = sub_6E4240(v36, 0);
      v47 = v277;
      v48 = v258;
      v36 = v46;
      v49 = *(_BYTE *)(v46 + 24);
      if ( v49 != 2 )
      {
        if ( v49 != 3 && v49 != 20 )
          goto LABEL_75;
        IO_read_end_low = **(_QWORD **)(v36 + 56);
        if ( IO_read_end_low )
        {
          p_IO_read_end = (__int64)&v319[0]._IO_read_end;
          sub_878710(IO_read_end_low, &v319[0]._IO_read_end);
          LOBYTE(v319[0]._IO_write_base) = v264 | (__int64)v319[0]._IO_write_base & 0xFE;
          v45 = v315;
          v319[0]._IO_read_base = *(char **)(v36 + 36);
          goto LABEL_206;
        }
        goto LABEL_81;
      }
      v87 = *(_QWORD *)(v36 + 56);
      v259 = v87;
      if ( *(_BYTE *)(v87 + 173) == 12 )
      {
        v88 = *(_BYTE *)(v87 + 176);
        if ( v88 == 2 || v88 == 13 )
        {
          v177 = *(char **)(v87 + 8);
          if ( v177 )
          {
            if ( (*(_BYTE *)(v259 + 89) & 0x40) != 0 )
              BUG();
            if ( (*(_BYTE *)(v259 + 89) & 8) != 0 )
              v177 = *(char **)(v259 + 24);
            v200 = *v177;
            v253 = 0;
            v201 = *(_QWORD *)(v36 + 56);
            if ( v200 == 126 )
            {
              if ( (unsigned int)sub_8D3A70(v17) )
              {
                for ( m = v17; *(_BYTE *)(m + 140) == 12; m = *(_QWORD *)(m + 160) )
                  ;
                IO_read_end_low = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)m + 96LL) + 24LL);
                if ( IO_read_end_low )
                {
                  p_IO_read_end = (__int64)&v319[0]._IO_read_end;
                  sub_878710(IO_read_end_low, &v319[0]._IO_read_end);
                  LOBYTE(v319[0]._IO_write_base) = v264 | (__int64)v319[0]._IO_write_base & 0xFE;
                  v45 = v315;
                  v319[0]._IO_read_base = *(char **)(v36 + 36);
                  goto LABEL_84;
                }
              }
              IO_read_end_low = v17;
              v235 = _mm_loadu_si128(&xmmword_4F06660[1]);
              v236 = _mm_loadu_si128(&xmmword_4F06660[2]);
              v237 = _mm_loadu_si128(&xmmword_4F06660[3]);
              v319[0]._IO_read_end = (char *)_mm_loadu_si128(xmmword_4F06660).m128i_u64[0];
              *(__m128i *)&v319[0]._IO_write_base = v235;
              LODWORD(v319[0]._IO_write_base) = v235.m128i_i32[0] | 0x20400;
              v319[0]._IO_buf_base = (char *)v236.m128i_i64[1];
              v319[0]._IO_read_base = *(char **)&dword_4F077C8;
              v319[0]._IO_write_end = (char *)v17;
              *(__m128i *)&v319[0]._IO_buf_end = v237;
              if ( !(unsigned int)sub_8D3A70(v17) )
                BYTE1(v319[0]._IO_write_base) |= 8u;
              goto LABEL_71;
            }
          }
          else
          {
            v253 = 0;
            v201 = *(_QWORD *)(v36 + 56);
          }
LABEL_466:
          v202 = *(_QWORD *)(*(_QWORD *)(v201 + 40) + 32LL);
          if ( v88 != 3 )
          {
            v278 = 0;
            goto LABEL_468;
          }
LABEL_519:
          v227 = 1;
          if ( (*(_BYTE *)(v201 + 177) & 1) == 0 )
            v227 = v264;
          p_IO_read_end = *(_QWORD *)(v201 + 184) != 0;
          v264 = v227;
          v278 = *(_QWORD *)(v201 + 184) != 0;
LABEL_468:
          v203 = *(_QWORD *)(*(_QWORD *)(v202 + 168) + 256LL);
          if ( !v203 )
            v203 = v202;
          if ( (!v264 || *(_QWORD *)&dword_4D03B80 == v203) && !v278 )
          {
            v204 = *(_BYTE *)(v17 + 140);
            for ( n = v17; v204 == 12; v204 = *(_BYTE *)(n + 140) )
              n = *(_QWORD *)(n + 160);
            if ( (unsigned __int8)(v204 - 9) > 2u
              || (*(_BYTE *)(n + 141) & 0x20) != 0
              && ((v206 = *(_QWORD *)(*(_QWORD *)(n + 168) + 152LL)) == 0 || (*(_BYTE *)(v206 + 29) & 0x20) != 0)
              || !*(_QWORD *)v201 )
            {
LABEL_81:
              v278 = 1;
LABEL_82:
              *(_BYTE *)(v7 + 56) = 1;
              v45 = 1;
              v50 = _mm_loadu_si128(&xmmword_4F06660[1]);
              v51 = _mm_loadu_si128(&xmmword_4F06660[2]);
              v315 = 1;
              v52 = _mm_loadu_si128(&xmmword_4F06660[3]);
              *(__m128i *)&v319[0]._IO_read_end = _mm_loadu_si128(xmmword_4F06660);
              *(__m128i *)&v319[0]._IO_write_base = v50;
              *(__m128i *)&v319[0]._IO_write_end = v51;
              *(__m128i *)&v319[0]._IO_buf_end = v52;
              goto LABEL_83;
            }
            v207 = &v319[0]._IO_read_end;
            v208 = 16;
            v209 = xmmword_4F06660;
            while ( v208 )
            {
              *(_DWORD *)v207 = v209->m128i_i32[0];
              v209 = (__m128i *)((char *)v209 + 4);
              v207 = (char **)((char *)v207 + 4);
              --v208;
            }
            IO_read_end_low = (__int64)&v319[0]._IO_read_end;
            p_IO_read_end = v17;
            v319[0]._IO_read_base = *v48;
            v319[0]._IO_read_end = **(char ***)v201;
            v210 = sub_7D2AC0(&v319[0]._IO_read_end, v17, &loc_1400000);
            v211 = v210;
            if ( !v210 )
              goto LABEL_82;
            IO_read_end_low = (__int64)v319[0]._IO_write_ptr;
            if ( (v319[0]._IO_write_ptr[82] & 4) != 0 )
              goto LABEL_82;
            v212 = *(_BYTE *)(v210 + 80);
            if ( v212 == 19 )
            {
              if ( !v253 )
              {
                v278 = 0;
                goto LABEL_82;
              }
            }
            else
            {
              if ( v212 != 20 )
              {
                if ( v212 != 17 || (IO_read_end_low = v211, v263 = v211, v251 = sub_8780F0(v211), v211 = v263, !v251) )
                {
                  if ( *(_BYTE *)(v211 + 80) != 2
                    || (v250 = *(_QWORD *)(v211 + 88), *(_BYTE *)(v250 + 173) != 12)
                    || *(_BYTE *)(v250 + 176) != 2 )
                  {
                    if ( v253 )
                      goto LABEL_82;
                    IO_read_end_low = (__int64)v319[0]._IO_write_ptr;
                    goto LABEL_491;
                  }
                }
                IO_read_end_low = (__int64)v319[0]._IO_write_ptr;
              }
              if ( v253 )
              {
                v257 = *(char **)(v259 + 192);
                goto LABEL_492;
              }
            }
LABEL_491:
            v253 = 0;
            v257 = 0;
LABEL_492:
            v260 = v253;
            goto LABEL_430;
          }
LABEL_174:
          IO_read_end_low = v17;
          v252 = v254;
          v255 = v47;
          v89 = sub_8D3A70(v17);
          v90 = v255;
          if ( !v89 )
          {
            *(_BYTE *)(v7 + 56) = 1;
            v315 = 1;
            v91 = _mm_loadu_si128(&xmmword_4F06660[1]);
            v92 = _mm_loadu_si128(&xmmword_4F06660[2]);
            v93 = _mm_loadu_si128(&xmmword_4F06660[3]);
            *(__m128i *)&v319[0]._IO_read_end = _mm_loadu_si128(xmmword_4F06660);
            *(__m128i *)&v319[0]._IO_write_base = v91;
            *(__m128i *)&v319[0]._IO_write_end = v92;
            *(__m128i *)&v319[0]._IO_buf_end = v93;
            v319[0]._IO_read_base = *(char **)(v36 + 36);
            goto LABEL_176;
          }
          v174 = *(_BYTE *)(v259 + 176);
          if ( v174 == 4 )
          {
            v257 = 0;
            v259 = *(_QWORD *)(v259 + 184);
          }
          else
          {
            v257 = 0;
            if ( v174 == 11 )
            {
              v260 = 1;
              v257 = *(char **)(v259 + 192);
              v259 = *(_QWORD *)(v259 + 184);
            }
          }
          v175 = *(unsigned int *)(v7 + 40);
          if ( *(char *)(v90 + 19) < 0 )
            v175 = (unsigned int)v175 | 0x20;
          p_IO_read_end = *(_QWORD *)(v7 + 24);
          v176 = sub_730C30(v259, p_IO_read_end, *(_QWORD *)(v7 + 32), v90 + 68, *(_QWORD *)(v7 + 48), v175);
          IO_read_end_low = v176;
          if ( v176 && v278 )
          {
            v278 = 1;
            if ( (*(_BYTE *)(v176 + 84) & 1) == 0 )
              goto LABEL_372;
            if ( *(_BYTE *)(v176 + 80) != 2 )
              goto LABEL_372;
            v234 = *(_QWORD *)(v176 + 88);
            if ( *(_BYTE *)(v234 + 173) != 12 || *(_BYTE *)(v234 + 176) != 3 )
              goto LABEL_372;
            p_IO_read_end = *(_QWORD *)(v234 + 184);
            IO_read_end_low = sub_7D3640(v17, p_IO_read_end, v252 + 28);
          }
          else
          {
            v278 = 1;
          }
LABEL_430:
          if ( !IO_read_end_low )
            goto LABEL_82;
LABEL_372:
          p_IO_read_end = (__int64)&v319[0]._IO_read_end;
          sub_878710(IO_read_end_low, &v319[0]._IO_read_end);
          LOBYTE(v319[0]._IO_write_base) = v264 | (__int64)v319[0]._IO_write_base & 0xFE;
          if ( v260 )
          {
            BYTE2(v319[0]._IO_write_base) |= 1u;
            v319[0]._IO_buf_base = v257;
          }
          v45 = v315;
LABEL_83:
          v319[0]._IO_read_base = *(char **)(v36 + 36);
          if ( !v278 )
            goto LABEL_84;
          goto LABEL_206;
        }
        if ( v88 != 11 )
        {
          v278 = 0;
          if ( v88 != 3 || (*(_BYTE *)(v87 + 89) & 4) == 0 )
            goto LABEL_174;
          v253 = 0;
          v202 = *(_QWORD *)(*(_QWORD *)(v87 + 40) + 32LL);
          v201 = *(_QWORD *)(v36 + 56);
          goto LABEL_519;
        }
        v201 = *(_QWORD *)(v87 + 184);
        if ( (*(_BYTE *)(v201 + 89) & 4) != 0 )
        {
          v253 = 1;
          v88 = *(_BYTE *)(v201 + 176);
          goto LABEL_466;
        }
      }
      v278 = 0;
      goto LABEL_174;
    }
    if ( v33 )
    {
      v35 = *v33;
      v7 = (__int64)v33;
      goto LABEL_61;
    }
    i = (_QWORD *)((unsigned int)qword_4F077B4 | dword_4F077BC);
    if ( !((unsigned int)qword_4F077B4 | dword_4F077BC) )
      break;
    v17 = *(_QWORD *)&dword_4D03B80;
    j = *(_BYTE *)(*(_QWORD *)&dword_4D03B80 + 140LL);
    if ( j == 6 )
    {
LABEL_91:
      if ( (*(_BYTE *)(v17 + 168) & 1) == 0 )
        goto LABEL_59;
      do
      {
        v17 = *(_QWORD *)(v17 + 160);
        j = *(_BYTE *)(v17 + 140);
      }
      while ( j == 12 );
    }
  }
  v274 = 1;
  v7 = 0;
  v261 = v296;
LABEL_108:
  sub_7B8B50(IO_read_end_low, p_IO_read_end, i, v13);
  if ( dword_4F077C4 != 2 )
  {
    v269 = 16898;
    goto LABEL_110;
  }
  v101 = 16898;
  v269 = 16930;
  if ( word_4F06418[0] == 160 )
  {
    if ( unk_4F07778 > 201102
      || dword_4F07774
      || unk_4F04C48 != -1 && (*(_BYTE *)(qword_4F04C68[0] + 776LL * unk_4F04C48 + 10) & 1) != 0 )
    {
      v269 = 148002;
      v172 = 147970;
    }
    else
    {
      v172 = 147970;
      v269 = 148002;
      if ( dword_4F04C44 == -1 )
      {
        IO_read_end_low = 5;
        if ( dword_4D04964 )
          IO_read_end_low = unk_4F07471;
        p_IO_read_end = 1139;
        sub_6E5C80(IO_read_end_low, 1139, &dword_4F063F8);
        v172 = 16898;
        v269 = 16930;
      }
    }
    v307 = v172;
    sub_7B8B50(IO_read_end_low, p_IO_read_end, v172, v55);
    v101 = v307;
  }
  if ( v315 || (v308 = v101, v173 = sub_8D3A70(v17), v101 = v308, !v173) )
    v269 = v101 | 0x60;
  if ( dword_4F077C4 == 2 )
  {
    if ( (word_4F06418[0] != 1 || (unk_4D04A11 & 2) == 0) && !(unsigned int)sub_7C0F00(v269, v17) )
    {
LABEL_201:
      if ( !dword_4D041A8 || word_4F06418[0] != 25 || (unsigned __int16)sub_7BE840(0, 0) != 55 )
      {
        IO_read_end_low = 1;
        v102 = _mm_loadu_si128(&xmmword_4F06660[1]);
        p_IO_read_end = (unsigned int)(dword_4F077C4 != 2) + 133;
        v103 = _mm_loadu_si128(&xmmword_4F06660[2]);
        v104 = _mm_loadu_si128(&xmmword_4F06660[3]);
        v319[0]._IO_read_end = (char *)_mm_loadu_si128(xmmword_4F06660).m128i_u64[0];
        *(__m128i *)&v319[0]._IO_write_base = v102;
        *(__m128i *)&v319[0]._IO_write_end = v103;
        *(__m128i *)&v319[0]._IO_buf_end = v104;
        v319[0]._IO_read_base = *(char **)&dword_4F063F8;
        *(_QWORD *)&dword_4F061D8 = qword_4F063F0;
        sub_7BE280(1, p_IO_read_end, 0, 0);
        v315 = 1;
        v63 = *(char **)&dword_4F061D8;
        goto LABEL_117;
      }
      p_IO_read_end = (__int64)&v320;
      sub_6E2E50(0, &v320);
      sub_6D7930((__int64)&v320);
      IO_read_end_low = (__int64)&v320;
      if ( !(unsigned int)sub_696840((__int64)&v320) )
      {
        if ( v321.m128i_i8[0] != 2 )
        {
          if ( v321.m128i_i8[0] != 1 )
            goto LABEL_414;
          v245 = sub_6F6F40(&v320, 0);
          v249 = *(unsigned __int8 *)(v245 + 24);
          if ( (_BYTE)v249 != 20 && (_BYTE)v249 != 4 && (_BYTE)v249 != 3 )
          {
            p_IO_read_end = (__int64)v324 + 4;
            IO_read_end_low = 3386;
            sub_69A8C0(3386, (_DWORD *)v324 + 1, v249, v246, v247, v248);
            v315 = 1;
            v63 = *(char **)&dword_4F061D8;
            goto LABEL_117;
          }
          p_IO_read_end = (__int64)&v319[0]._IO_read_end;
          IO_read_end_low = **(_QWORD **)(v245 + 56);
LABEL_585:
          sub_878710(IO_read_end_low, &v319[0]._IO_read_end);
          v319[0]._IO_read_base = *(char **)((char *)v324 + 4);
          v63 = *(char **)&dword_4F061D8;
          goto LABEL_117;
        }
        if ( v336.m128i_i8[0] == 8 || v336.m128i_i8[0] == 11 )
        {
          IO_read_end_low = *(_QWORD *)v336.m128i_i64[1];
          p_IO_read_end = (__int64)&v319[0]._IO_read_end;
          goto LABEL_585;
        }
        v273 = (char *)v336.m128i_i64[1];
        v239 = v337.m128i_i32[0];
        v314 = v336.m128i_i8[0];
        if ( (unsigned int)sub_6E5430(&v320, &v320, v336.m128i_i64[1], v336.m128i_u8[0], v184) )
        {
          v319[0]._IO_read_base = v273;
          p_IO_read_end = (__int64)v324 + 4;
          LOBYTE(v319[0]._IO_read_end) = v314;
          LODWORD(v319[0]._IO_write_base) = v239;
          v240 = sub_67D610(0xD28u, (_DWORD *)v324 + 1, 8u);
          sub_67F190(
            (__int64)v240,
            (__int64)v324 + 4,
            v241,
            v242,
            v243,
            v244,
            (char)v319[0]._IO_read_end,
            (__int64)v319[0]._IO_read_base,
            (int)v319[0]._IO_write_base);
          IO_read_end_low = (__int64)v240;
          sub_685910((__int64)v240, (FILE *)((char *)v324 + 4));
        }
LABEL_414:
        v315 = 1;
        v63 = *(char **)&dword_4F061D8;
LABEL_117:
        v319[0]._IO_read_ptr = v63;
LABEL_118:
        v268 = word_4F06418[0] == 27;
LABEL_119:
        v296 = v261;
        goto LABEL_22;
      }
      IO_read_end_low = (__int64)&v320;
      p_IO_read_end = 0;
      v185 = sub_6F6F40(&v320, 0);
      v319[0]._IO_read_ptr = *(char **)&dword_4F061D8;
      if ( !v185 )
        goto LABEL_118;
      v186 = sub_6F6F40(p_IO_backup_base, 0);
      *(_QWORD *)(v186 + 16) = v185;
      v187 = sub_73DBF0((unsigned int)v288 + 94, *(_QWORD *)&dword_4D03B80, v186);
      sub_6E70E0(v187, a5);
      return sub_6E3BA0(a5, &v318, v316, 0);
    }
  }
  else
  {
LABEL_110:
    if ( word_4F06418[0] != 1 )
      goto LABEL_201;
  }
  IO_read_end_low = v269;
  p_IO_read_end = 11;
  v59 = sub_7C8410(v269, 11, &v317);
  v315 |= v317;
  *(_QWORD *)&v319[0]._flags = qword_4D04A08;
  v319[0]._IO_read_ptr = *(char **)&dword_4F063F8;
  if ( v286 && (unk_4D04A10 & 0x20) != 0 )
  {
    if ( (unsigned int)sub_6E5430(v269, 11, v56, v57, v58) )
    {
      p_IO_read_end = (__int64)&dword_4F063F8;
      IO_read_end_low = 40;
      sub_6851C0(0x28u, &dword_4F063F8);
    }
LABEL_115:
    v315 = 1;
LABEL_116:
    v60 = _mm_loadu_si128((const __m128i *)&unk_4D04A10);
    v61 = _mm_loadu_si128(&xmmword_4D04A20);
    v62 = _mm_loadu_si128((const __m128i *)&unk_4D04A30);
    *(__m128i *)&v319[0]._IO_read_end = _mm_loadu_si128((const __m128i *)&qword_4D04A00);
    *(__m128i *)&v319[0]._IO_write_base = v60;
    *(_QWORD *)&dword_4F061D8 = qword_4F063F0;
    *(__m128i *)&v319[0]._IO_write_end = v61;
    *(__m128i *)&v319[0]._IO_buf_end = v62;
    sub_7B8B50(IO_read_end_low, p_IO_read_end, v56, v57);
    v63 = *(char **)&dword_4F061D8;
    goto LABEL_117;
  }
  if ( (unk_4D04A11 & 4) != 0 )
  {
    if ( (unk_4D04A11 & 0x20) != 0 || (unk_4D04A12 & 2) == 0 || !xmmword_4D04A20.m128i_i64[0] )
      goto LABEL_115;
    goto LABEL_116;
  }
  if ( !v17
    || (unsigned __int8)(*(_BYTE *)(v17 + 140) - 9) > 2u
    || (v300 = v59, v116 = sub_8D23B0(v17), v59 = v300, v116) )
  {
    p_IO_read_end = v17;
    IO_read_end_low = (__int64)p_IO_backup_base;
    v298 = v59;
    v108 = sub_68A290((__int64)p_IO_backup_base, (_BYTE *)v17, v267);
    v59 = v298;
    v111 = v108;
    if ( !v108 )
    {
LABEL_218:
      v112 = 1;
      goto LABEL_219;
    }
  }
  if ( v59 )
  {
    v109 = 0;
    goto LABEL_388;
  }
  if ( (unk_4D04A10 & 0x20) != 0 )
  {
    IO_read_end_low = v17;
    v301 = unk_4D04A38;
    v117 = sub_8DBE70(v17);
    v112 = 0;
    v111 = v117;
    if ( v301 )
    {
      IO_read_end_low = v301;
      v256 = v117;
      v118 = sub_8DBE70(v301);
      v112 = 0;
      v111 = v256;
      if ( v118 )
      {
        v119 = v301;
        if ( !v256 )
        {
          v120 = *(_BYTE *)(v301 + 140);
          if ( v120 == 14 )
          {
            IO_read_end_low = v301;
            v238 = sub_7CFE40(v301);
            v112 = 0;
            v119 = v238;
            v120 = *(_BYTE *)(v238 + 140);
          }
          v121 = v120 == 12;
          v122 = v119;
          if ( v121 )
          {
            do
              v122 = *(_QWORD *)(v122 + 160);
            while ( *(_BYTE *)(v122 + 140) == 12 );
          }
          v123 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)v122 + 96LL) + 24LL);
          if ( !v123 )
          {
            p_IO_read_end = 2;
            IO_read_end_low = v119;
            v123 = sub_7D0130(v119, 2, 0, &qword_4D04A00);
            v112 = 0;
          }
          goto LABEL_252;
        }
LABEL_555:
        p_IO_read_end = 2;
        IO_read_end_low = v17;
        v123 = sub_7D0130(v17, 2, 0, &qword_4D04A00);
        v112 = 0;
        goto LABEL_252;
      }
    }
    if ( (_DWORD)v111 )
      goto LABEL_555;
    for ( ii = v17; *(_BYTE *)(ii + 140) == 12; ii = *(_QWORD *)(ii + 160) )
      ;
    v123 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)ii + 96LL) + 24LL);
LABEL_252:
    unk_4D04A18 = v123;
  }
  else
  {
    IO_read_end_low = (__int64)&qword_4D04A00;
    p_IO_read_end = v17;
    v123 = sub_7D2AC0(&qword_4D04A00, v17, &loc_1400000);
    v112 = 0;
  }
  if ( !v123 )
  {
    if ( (unk_4D04A10 & 0x20) == 0 )
    {
      v111 = 1;
      goto LABEL_218;
    }
    unk_4D04A10 |= 0x20400u;
    v111 = 1;
    xmmword_4D04A20.m128i_i64[0] = v17;
LABEL_219:
    if ( dword_4F077C4 != 1 && !unk_4D0436C )
      goto LABEL_223;
    if ( !v288 )
    {
      if ( p_IO_backup_base[1].m128i_i8[1] != 1 )
      {
LABEL_223:
        v299 = v111;
        if ( v112 )
        {
          v315 = 1;
          sub_6E6000(IO_read_end_low, p_IO_read_end, v111, v109, v110, &qword_4D04A00);
          if ( v299 && (unk_4D04A11 & 0x20) == 0 )
          {
            if ( (unsigned int)sub_6E5430(IO_read_end_low, p_IO_read_end, v299, v113, v114) )
              sub_686A10((dword_4F077C4 != 2) + 135, dword_4F07508, *(_QWORD *)(qword_4D04A00 + 8), *(_QWORD *)v17);
            v228 = sub_87F050(&qword_4D04A00);
            sub_8767A0(68, v228, dword_4F07508, 0);
          }
          p_IO_read_end = (__int64)&qword_4D04A00;
          IO_read_end_low = (__int64)&v319[0]._IO_read_end;
          for ( jj = 16; jj; --jj )
          {
            *(_DWORD *)IO_read_end_low = *(_DWORD *)p_IO_read_end;
            p_IO_read_end += 4;
            IO_read_end_low += 4;
          }
          *(_QWORD *)&dword_4F061D8 = qword_4F063F0;
          sub_7B8B50(IO_read_end_low, p_IO_read_end, &dword_4F061D8, 0);
          v319[0]._IO_read_ptr = *(char **)&dword_4F061D8;
          v268 = word_4F06418[0] == 27;
          v296 = v261;
          goto LABEL_22;
        }
        v296 = 0;
        v124 = v17;
        goto LABEL_255;
      }
      IO_read_end_low = (__int64)p_IO_backup_base;
      v272 = v112;
      v312 = v111;
      if ( (unsigned int)sub_6ED0A0(p_IO_backup_base) )
      {
        v111 = v312;
        v112 = v272;
        goto LABEL_223;
      }
    }
    v189 = *(_QWORD *)(qword_4D04A00 + 32);
    if ( v189 )
    {
      v190 = 0;
      do
      {
        if ( *(_BYTE *)(v189 + 80) == 8 )
        {
          if ( v190 )
          {
            v191 = *(_QWORD *)(v189 + 88);
            v111 = *(_QWORD *)(v190 + 88);
            p_IO_read_end = *(_QWORD *)(v191 + 128);
            if ( *(_QWORD *)(v111 + 128) != p_IO_read_end || *(_BYTE *)(v111 + 136) != *(_BYTE *)(v191 + 136) )
            {
              v271 = v189;
              v192 = sub_6E5430(IO_read_end_low, p_IO_read_end, v111, v189, v110);
              v109 = v271;
              if ( v192 )
              {
                p_IO_read_end = (__int64)&dword_4F063F8;
                IO_read_end_low = 1576;
                sub_686C60(0x628u, (FILE *)&dword_4F063F8, v190, v271);
              }
              goto LABEL_441;
            }
          }
          else
          {
            v190 = v189;
          }
        }
        v189 = *(_QWORD *)(v189 + 8);
      }
      while ( v189 );
      if ( !v190 )
        goto LABEL_503;
      sub_878710(v190, &qword_4D04A00);
      qword_4D04A08 = *(_QWORD *)&v319[0]._flags;
      if ( v288 )
      {
        if ( (unsigned int)sub_6E53E0(5, 234, v319) )
          sub_684B30(0xEAu, v319);
      }
      else
      {
        if ( (unsigned int)sub_6E53E0(5, 233, v319) )
          sub_684B30(0xE9u, v319);
        sub_6FF5E0(p_IO_backup_base, 0);
      }
      v296 = *(_QWORD *)(v190 + 64);
      p_IO_read_end = (__int64)p_IO_backup_base;
      IO_read_end_low = sub_72D2E0(v296, 0);
      sub_6FC3F0(IO_read_end_low, p_IO_backup_base, 0);
      v124 = v296;
      *(_BYTE *)(v296 + 88) |= 4u;
    }
    else
    {
LABEL_503:
      if ( (unsigned int)sub_6E5430(IO_read_end_low, p_IO_read_end, v111, v189, v110) )
      {
        IO_read_end_low = 1577;
        p_IO_read_end = *(_QWORD *)(qword_4D04A00 + 8);
        sub_6851F0(0x629u, p_IO_read_end);
      }
LABEL_441:
      v315 = 1;
      v124 = v17;
      v296 = 0;
    }
    goto LABEL_255;
  }
  v124 = v17;
  v296 = 0;
  if ( *(_BYTE *)(v123 + 80) != 19 )
    goto LABEL_255;
  if ( dword_4F077C4 != 2 )
  {
    if ( word_4F06418[0] == 1 )
      goto LABEL_461;
LABEL_75:
    sub_721090(IO_read_end_low);
  }
  if ( word_4F06418[0] != 1 || (unk_4D04A11 & 2) == 0 )
  {
    IO_read_end_low = v269;
    if ( !(unsigned int)sub_7C0F00(v269, 0) )
      goto LABEL_75;
  }
LABEL_461:
  IO_read_end_low = v269;
  p_IO_read_end = 11;
  v109 = sub_7BF130(v269, 11, &v317);
  if ( v317 )
  {
    v315 = 1;
    v112 = 1;
    goto LABEL_393;
  }
  if ( (unk_4D04A10 & 1) != 0 )
  {
LABEL_388:
    IO_read_end_low = (__int64)&qword_4D04A00;
    p_IO_read_end = (__int64)&v319[0]._IO_read_ptr;
    v270 = v109;
    v179 = sub_688C20((__int64)&qword_4D04A00, (FILE *)&v319[0]._IO_read_ptr, v17);
    v109 = v270;
    v112 = v179;
    if ( v179 )
    {
      v180 = unk_4D04A18;
      v111 = *(unsigned __int8 *)(unk_4D04A18 + 80LL);
      if ( (_BYTE)v111 == 16 )
      {
        v180 = **(_QWORD **)(unk_4D04A18 + 88LL);
        v111 = *(unsigned __int8 *)(v180 + 80);
      }
      v296 = 0;
      v124 = v17;
      if ( (_BYTE)v111 != 24 )
        goto LABEL_255;
      v109 = *(_QWORD *)(v180 + 88);
      v112 = 0;
    }
    else
    {
      v315 = 1;
    }
    goto LABEL_393;
  }
  v112 = 1;
LABEL_393:
  v296 = 0;
  v124 = v17;
  v111 = 1;
  if ( !v109 )
    goto LABEL_219;
LABEL_255:
  if ( word_4D04898 )
  {
    if ( !dword_4D04880 && (unk_4D04A10 & 0x420) == 0x20 )
    {
      IO_read_end_low = v124;
      if ( !(unsigned int)sub_8DBE70(v124) )
      {
        p_IO_read_end = (__int64)&dword_4F063F8;
        IO_read_end_low = 28;
        sub_6E91E0(28, &dword_4F063F8);
      }
    }
  }
  v125 = _mm_loadu_si128((const __m128i *)&unk_4D04A10);
  v126 = _mm_loadu_si128(&xmmword_4D04A20);
  v127 = _mm_loadu_si128((const __m128i *)&unk_4D04A30);
  *(__m128i *)&v319[0]._IO_read_end = _mm_loadu_si128((const __m128i *)&qword_4D04A00);
  *(__m128i *)&v319[0]._IO_write_base = v125;
  *(_QWORD *)&dword_4F061D8 = qword_4F063F0;
  *(__m128i *)&v319[0]._IO_write_end = v126;
  *(__m128i *)&v319[0]._IO_buf_end = v127;
  sub_7B8B50(IO_read_end_low, p_IO_read_end, v111, v109);
  v319[0]._IO_read_ptr = *(char **)&dword_4F061D8;
  v268 = word_4F06418[0] == 27;
  if ( !v296 )
    goto LABEL_119;
  if ( *(_BYTE *)(v296 + 140) == 12 )
  {
    v17 = v296;
    do
      v17 = *(_QWORD *)(v17 + 160);
    while ( *(_BYTE *)(v17 + 140) == 12 );
  }
  else
  {
    v17 = v296;
  }
LABEL_22:
  if ( (BYTE1(v319[0]._IO_write_base) & 4) != 0 )
  {
    v21 = v315;
    v22 = 1;
    goto LABEL_24;
  }
  if ( v274 )
  {
    v53 = *(_BYTE *)(v17 + 140);
    if ( (unsigned __int8)(v53 - 9) > 2u )
      goto LABEL_96;
    if ( (*(_BYTE *)(v17 + 141) & 0x20) != 0 )
    {
      p_IO_read_end = v17;
      IO_read_end_low = (__int64)p_IO_backup_base;
      if ( !(unsigned int)sub_68A290((__int64)p_IO_backup_base, (_BYTE *)v17, v267) )
      {
        v53 = *(_BYTE *)(v17 + 140);
LABEL_96:
        if ( v53 == 12 )
        {
          v54 = v17;
          do
          {
            v54 = *(_QWORD *)(v54 + 160);
            v53 = *(_BYTE *)(v54 + 140);
          }
          while ( v53 == 12 );
        }
        if ( v53 )
        {
          v170 = p_IO_backup_base->m128i_i64[0];
          if ( (unsigned int)sub_8D23B0(v17) && (unsigned int)sub_8D3A70(v17) )
          {
            v170 = v17;
            v171 = !v288 ? 70 : 833;
          }
          else if ( dword_4F077C4 == 2 )
          {
            v171 = !v288 ? 153 : 131;
          }
          else
          {
            v171 = !v288 ? 154 : 132;
          }
          sub_6E6930(v171, p_IO_backup_base, v170);
        }
        v315 = 1;
        *(_QWORD *)dword_4F07508 = *(__int64 *)((char *)p_IO_backup_base[4].m128i_i64 + 4);
LABEL_29:
        sub_6E6260(a5);
        sub_6E6450(p_IO_backup_base);
        v25 = dword_4F07508;
        goto LABEL_30;
      }
    }
  }
  v21 = v315;
  v22 = 0;
LABEL_24:
  *(_QWORD *)dword_4F07508 = *(__int64 *)((char *)p_IO_backup_base[4].m128i_i64 + 4);
  if ( v21 || !p_IO_backup_base[1].m128i_i8[0] )
    goto LABEL_29;
  v23 = p_IO_backup_base->m128i_i64[0];
  for ( kk = *(_BYTE *)(p_IO_backup_base->m128i_i64[0] + 140); kk == 12; kk = *(_BYTE *)(v23 + 140) )
    v23 = *(_QWORD *)(v23 + 160);
  if ( !kk )
    goto LABEL_29;
  if ( !v22 )
  {
    IO_write_ptr = v319[0]._IO_write_ptr;
    v68 = *((unsigned __int8 *)v319[0]._IO_write_ptr + 80);
    v69 = v319[0]._IO_write_ptr;
    if ( (_BYTE)v68 == 16 )
    {
      v70 = (char **)*((_QWORD *)v319[0]._IO_write_ptr + 11);
      v69 = *v70;
      v68 = (unsigned __int8)(*v70)[80];
    }
    if ( (_BYTE)v68 == 24 )
    {
      v69 = (char *)*((_QWORD *)v69 + 11);
      v68 = (unsigned __int8)v69[80];
    }
    v71 = (FILE *)qword_4F04C68;
    *(_QWORD *)&v319[0]._flags = v319[0]._IO_read_base;
    if ( *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 4) != 1 )
    {
LABEL_144:
      if ( (_BYTE)v68 != 17 )
        goto LABEL_145;
      v262 = 0;
      v72 = 0;
      v265 = 0;
LABEL_147:
      IO_read_end_low = (__int64)&v319[0]._IO_read_end;
      v282 = v69;
      v289 = IO_write_ptr;
      sub_6E62A0(&v319[0]._IO_read_end);
      v75 = v289;
      v76 = (__int64)v282;
      if ( (BYTE1(v319[0]._IO_write_base) & 0x20) != 0 )
      {
        sub_6E6260(a5);
        sub_6E5940(p_IO_backup_base);
        sub_6E5930(v72);
        v25 = dword_4F07508;
        goto LABEL_30;
      }
      if ( !v286 )
        goto LABEL_153;
      v77 = v282[80];
      if ( v77 == 2 )
      {
        v137 = *((_QWORD *)v282 + 11);
        if ( v137 && *(_BYTE *)(v137 + 173) == 12 )
        {
LABEL_303:
          sub_6F35D0(v76, a5);
          sub_6E46C0(a5, &v319[0]._IO_read_end);
          *(_QWORD *)((char *)a5 + 68) = *(_QWORD *)&v319[0]._flags;
          *(_QWORD *)((char *)a5 + 76) = v319[0]._IO_read_ptr;
LABEL_288:
          *((_BYTE *)a5 + 19) = ((_BYTE)v268 << 7) | *((_BYTE *)a5 + 19) & 0x7F;
          sub_82F8F0(p_IO_backup_base, v267, a5);
          v25 = dword_4F07508;
          LOBYTE(v268) = 0;
          goto LABEL_30;
        }
      }
      else if ( v77 == 8 )
      {
        v78 = *((_QWORD *)v282 + 11);
        v283 = v289;
        IO_read_end_low = *(_QWORD *)(v78 + 120);
        v290 = v76;
        v79 = sub_8D32E0(IO_read_end_low);
        v76 = v290;
        v75 = v283;
        if ( v79 )
        {
          if ( (unsigned int)sub_6E5430(IO_read_end_low, v71, v290, v80, v74) )
            sub_6851C0(0x773u, v319);
        }
        else
        {
          if ( (*(_BYTE *)(*(_QWORD *)(v290 + 88) + 144LL) & 4) == 0 )
          {
LABEL_153:
            v81 = *(_BYTE *)(v76 + 80);
            switch ( v81 )
            {
              case 2u:
                goto LABEL_303;
              case 3u:
              case 4u:
              case 5u:
              case 6u:
                if ( (unsigned int)sub_6E5430(IO_read_end_low, v71, v76, v81, v74) )
                  sub_6851C0(0xFEu, v319);
                goto LABEL_274;
              case 7u:
              case 9u:
                if ( v81 != 9 && v81 != 7 )
                  BUG();
                v132 = *(_QWORD *)(v76 + 88);
                if ( (*(_DWORD *)(v132 + 168) & 0xD01000) != 0x100000
                  || (v295 = v76,
                      v313 = *(_QWORD *)(v76 + 88),
                      v233 = sub_8D2600(*(_QWORD *)(v132 + 120)),
                      v132 = v313,
                      !v233) )
                {
                  sub_6F8E70(v132, v319, &v319[0]._IO_read_ptr, a5, v72);
                  sub_6E46C0(a5, &v319[0]._IO_read_end);
                  goto LABEL_288;
                }
                sub_686890(0xD5Bu, &v319[0]._IO_read_base, v295, *(_QWORD *)(v313 + 120));
                sub_6E6260(a5);
                v25 = dword_4F07508;
                goto LABEL_30;
              case 8u:
                sub_82FD20(
                  (_DWORD)p_IO_backup_base,
                  v267,
                  v76,
                  (_DWORD)v75,
                  BYTE1(v319[0]._IO_write_base) & 1,
                  1,
                  (__int64)v319);
                sub_68D540(
                  p_IO_backup_base,
                  v296,
                  v267,
                  0,
                  (__int64)&v319[0]._IO_read_end,
                  (__int64)v319,
                  (__int64)&v319[0]._IO_read_ptr,
                  v72,
                  (const __m128i *)a5);
                v25 = dword_4F07508;
                goto LABEL_30;
              case 0xAu:
                for ( mm = *(_QWORD *)(*(_QWORD *)(v76 + 88) + 152LL);
                      *(_BYTE *)(mm + 140) == 12;
                      mm = *(_QWORD *)(mm + 160) )
                {
                  ;
                }
                if ( v265 )
                  goto LABEL_523;
                if ( !*(_QWORD *)(*(_QWORD *)(mm + 168) + 40LL) )
                {
                  sub_6EAB60(
                    (_DWORD)v75,
                    (__int64)v319[0]._IO_write_base & 1,
                    0,
                    (unsigned int)v319,
                    (unsigned int)&v319[0]._IO_read_ptr,
                    v72,
                    (__int64)a5);
                  sub_6E4620(a5);
                  goto LABEL_288;
                }
                v302 = v75;
                v292 = v76;
                v134 = sub_7328C0();
                v75 = v302;
                if ( v134 )
                {
LABEL_523:
                  if ( *(_BYTE *)(qword_4D03C50 + 16LL) == 3 && !word_4D04898 )
                    goto LABEL_525;
                }
                else
                {
                  v135 = v292;
                  if ( *(_BYTE *)(qword_4D03C50 + 16LL) == 3 && !word_4D04898 )
                  {
LABEL_525:
                    sub_6E6890(28, a5);
                    v25 = dword_4F07508;
                    goto LABEL_30;
                  }
                  v81 = *(_BYTE *)(v292 + 80);
                  if ( v81 == 10 )
                  {
                    v136 = (int)v302;
                    v293 = v302;
                    v303 = v135;
                    sub_82FD20(
                      (_DWORD)p_IO_backup_base,
                      v267,
                      v135,
                      v136,
                      BYTE1(v319[0]._IO_write_base) & 1,
                      1,
                      (__int64)v319);
                    v75 = v293;
                    v81 = *(_BYTE *)(v303 + 80);
                  }
LABEL_277:
                  if ( v81 != 17 && v81 != 20 )
                  {
                    sub_6EAB60(
                      (_DWORD)v75,
                      (__int64)v319[0]._IO_write_base & 1,
                      0,
                      (unsigned int)v319,
                      (unsigned int)&v319[0]._IO_read_ptr,
                      v72,
                      (__int64)a5);
                    sub_6E4620(a5);
                    goto LABEL_280;
                  }
                }
LABEL_423:
                sub_6E7190(v75, &v319[0]._IO_read_end, a5);
                if ( v262 )
                {
                  sub_6F3BA0(a5, 0);
                  goto LABEL_288;
                }
LABEL_280:
                *a6 = _mm_loadu_si128(p_IO_backup_base);
                a6[1] = _mm_loadu_si128(p_IO_backup_base + 1);
                a6[2] = _mm_loadu_si128(p_IO_backup_base + 2);
                a6[3] = _mm_loadu_si128(p_IO_backup_base + 3);
                a6[4] = _mm_loadu_si128(p_IO_backup_base + 4);
                a6[5] = _mm_loadu_si128(p_IO_backup_base + 5);
                a6[6] = _mm_loadu_si128(p_IO_backup_base + 6);
                a6[7] = _mm_loadu_si128(p_IO_backup_base + 7);
                a6[8] = _mm_loadu_si128(p_IO_backup_base + 8);
                v131 = p_IO_backup_base[1].m128i_i8[0];
                if ( v131 == 2 )
                {
                  a6[9] = _mm_loadu_si128(p_IO_backup_base + 9);
                  a6[10] = _mm_loadu_si128(p_IO_backup_base + 10);
                  a6[11] = _mm_loadu_si128(p_IO_backup_base + 11);
                  a6[12] = _mm_loadu_si128(p_IO_backup_base + 12);
                  a6[13] = _mm_loadu_si128(p_IO_backup_base + 13);
                  a6[14] = _mm_loadu_si128(p_IO_backup_base + 14);
                  a6[15] = _mm_loadu_si128(p_IO_backup_base + 15);
                  a6[16] = _mm_loadu_si128(p_IO_backup_base + 16);
                  a6[17] = _mm_loadu_si128(p_IO_backup_base + 17);
                  a6[18] = _mm_loadu_si128(p_IO_backup_base + 18);
                  a6[19] = _mm_loadu_si128(p_IO_backup_base + 19);
                  a6[20] = _mm_loadu_si128(p_IO_backup_base + 20);
                  a6[21] = _mm_loadu_si128(p_IO_backup_base + 21);
                }
                else if ( v131 == 5 || v131 == 1 )
                {
                  a6[9].m128i_i64[0] = p_IO_backup_base[9].m128i_i64[0];
                }
                sub_82F1E0(a6, v267, a5);
                v25 = dword_4F07508;
                break;
              case 0x11u:
              case 0x14u:
                if ( *(_BYTE *)(qword_4D03C50 + 16LL) == 3 && !word_4D04898 )
                  goto LABEL_525;
                if ( !v265 )
                  goto LABEL_277;
                goto LABEL_423;
              case 0x13u:
                sub_6F3AD0(v75, 1, v319[0]._IO_buf_base, (__int64)v319[0]._IO_write_base & 1, a5);
                sub_6F7FE0(a5, 0);
                sub_82F8F0(p_IO_backup_base, v267, a5);
                v25 = dword_4F07508;
                goto LABEL_30;
              case 0x15u:
                v304 = v76;
                if ( (unsigned int)sub_6E5430(IO_read_end_low, v71, v76, v81, v74) )
                  sub_6854C0(0x1B9u, v319, v304);
LABEL_274:
                sub_6E6450(p_IO_backup_base);
                sub_6E6260(a5);
                v25 = dword_4F07508;
                goto LABEL_30;
              default:
                goto LABEL_75;
            }
            goto LABEL_30;
          }
          if ( (unsigned int)sub_6E5430(IO_read_end_low, v71, v290, v80, v74) )
            sub_6851C0(0x591u, v319);
        }
LABEL_397:
        sub_6E6260(a5);
        v25 = dword_4F07508;
        goto LABEL_30;
      }
      if ( (unsigned int)sub_6E5430(&v319[0]._IO_read_end, v71, v282, v73, v74) )
        sub_6851C0(0x590u, v319);
      goto LABEL_397;
    }
    if ( (unsigned __int8)v68 <= 0x14u )
    {
      v178 = 1182720;
      if ( _bittest64(&v178, v68) )
      {
        if ( !v17
          || (unsigned __int8)(*(_BYTE *)(v17 + 140) - 9) > 2u
          || *(char *)(v17 + 177) >= 0
          || ((__int64)v319[0]._IO_write_base & 0x20) != 0 )
        {
          goto LABEL_144;
        }
        v262 = 1;
        v72 = 0;
        v265 = 1;
        goto LABEL_147;
      }
LABEL_145:
      if ( (v69[84] & 4) != 0 )
        goto LABEL_146;
      if ( (_BYTE)v68 == 16 )
      {
        v71 = (FILE *)**((_QWORD **)v69 + 11);
        IO_backup_base = (char)v71->_IO_backup_base;
        goto LABEL_266;
      }
    }
    else if ( (v69[84] & 4) != 0 )
    {
      goto LABEL_146;
    }
    IO_backup_base = v68;
    v71 = (FILE *)v69;
LABEL_266:
    if ( IO_backup_base == 24 )
      IO_backup_base = v71->_IO_save_end[80];
    if ( (unsigned __int8)(IO_backup_base - 10) <= 1u )
    {
      v285 = v69;
      v294 = v319[0]._IO_write_ptr;
      v226 = sub_82EE30(IO_read_end_low, v71, v69);
      IO_write_ptr = v294;
      v69 = v285;
      if ( v226 )
        goto LABEL_146;
      LOBYTE(v68) = v285[80];
    }
    v129 = v68 - 10;
    if ( v7 )
    {
      if ( v129 > 1u )
      {
LABEL_271:
        v71 = v319;
        v284 = IO_write_ptr;
        v291 = v69;
        v130 = sub_6E50B0(v69, v319);
        v69 = v291;
        v262 = 0;
        v265 = 0;
        IO_write_ptr = v284;
        v72 = v130;
        goto LABEL_147;
      }
      v229 = *((_QWORD *)v69 + 11);
      if ( (*(_BYTE *)(v229 + 206) & 0x10) != 0 )
      {
        *(_BYTE *)(v7 + 56) = 1;
        v72 = 0;
        v262 = 0;
        v265 = 0;
        goto LABEL_147;
      }
    }
    else
    {
      if ( v129 > 1u )
        goto LABEL_271;
      v229 = *((_QWORD *)v69 + 11);
    }
    for ( nn = *(_QWORD *)(v229 + 152); *(_BYTE *)(nn + 140) == 12; nn = *(_QWORD *)(nn + 160) )
      ;
    v231 = **(_QWORD **)(nn + 168);
    if ( !v231 || (*(_BYTE *)(v231 + 35) & 1) == 0 )
      goto LABEL_271;
LABEL_146:
    v262 = 0;
    v72 = 0;
    v265 = 1;
    goto LABEL_147;
  }
  IO_write_end = v319[0]._IO_write_end;
  if ( (unsigned int)sub_8D3A70(v319[0]._IO_write_end) )
  {
    while ( IO_write_end[140] == 12 )
      IO_write_end = (char *)*((_QWORD *)IO_write_end + 20);
    v156 = sub_8D23B0(v17);
    if ( (char *)v17 != IO_write_end
      && !v156
      && !(unsigned int)sub_8D97D0(v17, IO_write_end, 0, v157, v158)
      && (*(_BYTE *)(v17 + 177) & 0x20) == 0
      && (IO_write_end[177] & 0x20) == 0 )
    {
      v199 = sub_8D5CE0(v17, IO_write_end);
      sub_6F7270((_DWORD)p_IO_backup_base, v199, 0, 1, 0, 1, 0, 1);
    }
    goto LABEL_324;
  }
  if ( v288 )
  {
    p_IO_read_end = 0;
    v193 = sub_72D2E0(IO_write_end, 0);
    v139 = p_IO_backup_base->m128i_i64[0];
    IO_write_end = (char *)v193;
    v140 = *(_BYTE *)(p_IO_backup_base->m128i_i64[0] + 140);
    if ( v140 == 6 )
    {
      v194 = sub_8D46C0(p_IO_backup_base->m128i_i64[0]);
      v140 = *(_BYTE *)(v194 + 140);
      v139 = v194;
    }
  }
  else
  {
    v139 = p_IO_backup_base->m128i_i64[0];
    v140 = *(_BYTE *)(p_IO_backup_base->m128i_i64[0] + 140);
  }
  if ( v140 == 12 )
  {
    v141 = v139;
    do
    {
      v141 = *(_QWORD *)(v141 + 160);
      v140 = *(_BYTE *)(v141 + 140);
    }
    while ( v140 == 12 );
  }
  if ( v140 )
  {
    v310 = v139;
    if ( !(unsigned int)sub_8D3350(v139) && !(unsigned int)sub_8D2B80(v310) )
    {
      v197 = v310;
      v198 = dword_4F077BC;
      if ( !dword_4F077BC || (_DWORD)qword_4F077B4 || qword_4F077A8 > 0x9D08u )
      {
LABEL_451:
        v311 = v197;
        if ( (unsigned int)sub_6E5430(v198, p_IO_read_end, v195, v196, v197) )
          sub_685360(0xB06u, &p_IO_backup_base[4].m128i_i32[1], v311);
        goto LABEL_311;
      }
      v198 = v310;
      if ( !(unsigned int)sub_8D3410(v310) )
      {
        v197 = v310;
        goto LABEL_451;
      }
    }
  }
LABEL_311:
  v142 = _mm_loadu_si128(p_IO_backup_base + 1);
  v143 = _mm_loadu_si128(p_IO_backup_base + 2);
  v144 = _mm_loadu_si128(p_IO_backup_base + 3);
  v145 = _mm_loadu_si128(p_IO_backup_base + 4);
  v146 = _mm_loadu_si128(p_IO_backup_base + 5);
  v320 = _mm_loadu_si128(p_IO_backup_base);
  v147 = _mm_loadu_si128(p_IO_backup_base + 6);
  v148 = _mm_loadu_si128(p_IO_backup_base + 7);
  v321 = v142;
  v149 = _mm_loadu_si128(p_IO_backup_base + 8);
  v150 = p_IO_backup_base[1].m128i_i8[0];
  v322 = v143;
  v323 = v144;
  v324[0] = v145;
  v324[1] = v146;
  v324[2] = v147;
  v324[3] = v148;
  v324[4] = v149;
  if ( v150 == 2 )
  {
    v213 = _mm_loadu_si128(p_IO_backup_base + 10);
    v214 = _mm_loadu_si128(p_IO_backup_base + 11);
    v215 = _mm_loadu_si128(p_IO_backup_base + 12);
    v216 = _mm_loadu_si128(p_IO_backup_base + 13);
    v325 = _mm_loadu_si128(p_IO_backup_base + 9);
    v217 = _mm_loadu_si128(p_IO_backup_base + 18);
    v218 = _mm_loadu_si128(p_IO_backup_base + 14);
    v326 = v213;
    v219 = _mm_loadu_si128(p_IO_backup_base + 19);
    v220 = _mm_loadu_si128(p_IO_backup_base + 15);
    v327 = v214;
    v221 = _mm_loadu_si128(p_IO_backup_base + 20);
    v222 = _mm_loadu_si128(p_IO_backup_base + 16);
    v328 = v215;
    v223 = _mm_loadu_si128(p_IO_backup_base + 17);
    v329 = v216;
    v224 = _mm_loadu_si128(p_IO_backup_base + 21);
    v330 = v218;
    v331 = v220;
    v332 = v222;
    v333 = v223;
    v334 = v217;
    v335 = v219;
    v336 = v221;
    v337 = v224;
  }
  else if ( v150 == 5 || v150 == 1 )
  {
    v325.m128i_i64[0] = p_IO_backup_base[9].m128i_i64[0];
  }
  v151 = (_QWORD *)sub_6F6F40(p_IO_backup_base, 0);
  v153 = v151;
  if ( (char *)*v151 == IO_write_end
    || (v305 = v151, v154 = sub_8D97D0(*v151, IO_write_end, 0, v152, v151), v153 = v305, v154) )
  {
    *v153 = IO_write_end;
  }
  else
  {
    if ( (*((_BYTE *)v305 + 25) & 3) != 0 )
      v155 = sub_73DC50(v305, IO_write_end);
    else
      v155 = sub_73E110(v305, IO_write_end);
    v153 = (_QWORD *)v155;
  }
  sub_6E7170(v153, p_IO_backup_base);
  sub_6E4EE0(p_IO_backup_base, &v320);
LABEL_324:
  v159 = sub_6F6F40(p_IO_backup_base, 0);
  v164 = sub_72CBE0(p_IO_backup_base, 0, v160, v161, v162, v163);
  v165 = sub_73DBF0((unsigned int)v288 + 22, v164, v159);
  sub_6E70E0(v165, a5);
  sub_6E46C0(a5, &v319[0]._IO_read_end);
  v25 = dword_4F07508;
  if ( v7 )
  {
    if ( v281 || *(_BYTE *)(*(_QWORD *)v7 + 24LL) == 1 && (unsigned __int8)(*(_BYTE *)(*(_QWORD *)v7 + 56LL) - 22) <= 1u )
      goto LABEL_30;
  }
  else
  {
    if ( v268 )
      goto LABEL_30;
    if ( !dword_4D04964 && word_4F06418[0] == 28 )
    {
      sub_7ADF70(&v320, 0);
      v181 = dword_4F07508;
      do
      {
        v309 = v181;
        sub_7AE360(&v320);
        sub_7B8B50(&v320, 0, v182, v183);
        v181 = v309;
      }
      while ( word_4F06418[0] == 28 );
      if ( word_4F06418[0] == 27 )
      {
        sub_7BC000(&v320);
        v25 = v309;
        LOBYTE(v268) = 0;
        goto LABEL_30;
      }
      sub_7BC000(&v320);
      v25 = v309;
    }
  }
  v166 = 8;
  if ( dword_4F077BC )
  {
    v167 = qword_4F077B4;
    if ( !(_DWORD)qword_4F077B4 )
    {
      if ( qword_4F077A8 <= 0x112FFu && v7 )
      {
        if ( dword_4F04C64 >= 0 )
        {
          v168 = qword_4F04C68[0];
          for ( i1 = qword_4F04C68[0] + 776LL; ; i1 += 776 )
          {
            if ( *(_BYTE *)(v168 + 4) == 14 )
              v167 -= ((*(_BYTE *)(v168 + 12) & 0x10) == 0) - 1;
            v168 = i1;
            if ( i1 == qword_4F04C68[0] + 776LL + 776LL * dword_4F04C64 )
              break;
          }
          v166 = 3 * (unsigned int)(v167 < 2) + 5;
        }
      }
      else
      {
        v166 = 8;
      }
    }
  }
  v306 = v25;
  sub_6E5C80(v166, 3006, &v319[0]._IO_read_base);
  v25 = v306;
LABEL_30:
  if ( (*((_BYTE *)a5 + 18) & 1) != 0 )
  {
    *((_DWORD *)a5 + 17) = v319[0]._flags;
    *((_WORD *)a5 + 36) = *((_WORD *)&v319[0]._flags + 2);
    *v25 = *(_QWORD *)((char *)a5 + 68);
    IO_read_ptr = v319[0]._IO_read_ptr;
    *(_QWORD *)((char *)a5 + 76) = v319[0]._IO_read_ptr;
    *(_QWORD *)&dword_4F061D8 = IO_read_ptr;
    sub_6E3280(a5, 0);
    *((_BYTE *)a5 + 19) = ((_BYTE)v268 << 7) | *((_BYTE *)a5 + 19) & 0x7F;
    goto LABEL_36;
  }
  if ( dword_4F077BC )
  {
    if ( !(_DWORD)qword_4F077B4 && qword_4F077A8 > 0x11237u )
      goto LABEL_35;
LABEL_33:
    if ( v17 )
    {
      if ( (unsigned __int8)(*(_BYTE *)(v17 + 140) - 9) <= 2u && *(char *)(v17 + 177) < 0 )
      {
        v297 = v25;
        v82 = sub_8DBE70(*a5);
        v25 = v297;
        if ( !v82 )
        {
          v83 = *((_BYTE *)a5 + 16);
          v84 = *(_QWORD *)&dword_4D03B80;
          *a5 = *(_QWORD *)&dword_4D03B80;
          if ( v83 == 1 )
          {
            *(_QWORD *)a5[18] = v84;
          }
          else if ( v83 == 2 )
          {
            a5[34] = v84;
          }
        }
      }
    }
    goto LABEL_35;
  }
  if ( (_DWORD)qword_4F077B4 )
    goto LABEL_33;
LABEL_35:
  v26 = p_IO_backup_base[4].m128i_i16[4];
  *((_DWORD *)a5 + 17) = p_IO_backup_base[4].m128i_i32[1];
  *((_WORD *)a5 + 36) = v26;
  *v25 = *(_QWORD *)((char *)a5 + 68);
  v27 = v319[0]._IO_read_ptr;
  *(_QWORD *)((char *)a5 + 76) = v319[0]._IO_read_ptr;
  *(_QWORD *)&dword_4F061D8 = v27;
  sub_6E3280(a5, &v318);
LABEL_36:
  if ( !v287 || word_4D04898 )
  {
    sub_6E26D0(1, a5);
  }
  else if ( (unsigned __int8)(*(_BYTE *)(qword_4D03C50 + 16LL) - 2) > 1u
         || *((_BYTE *)a5 + 17) != 1
         || (unsigned int)sub_6ED0A0(a5) )
  {
    sub_6FA3A0(a5);
    v28 = *((_BYTE *)a5 + 16);
    if ( v28 == 2 )
    {
      if ( (unsigned int)sub_8D2930(*a5) )
        return sub_6E3BA0(a5, &v318, v316, 0);
      v28 = *((_BYTE *)a5 + 16);
    }
    if ( v28 )
    {
      v29 = *a5;
      for ( i2 = *(_BYTE *)(*a5 + 140LL); i2 == 12; i2 = *(_BYTE *)(v29 + 140) )
        v29 = *(_QWORD *)(v29 + 160);
      if ( i2 )
        sub_6E68E0(28, a5);
    }
  }
  return sub_6E3BA0(a5, &v318, v316, 0);
}
