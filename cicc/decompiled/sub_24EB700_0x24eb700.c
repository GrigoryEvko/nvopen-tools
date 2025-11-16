// Function: sub_24EB700
// Address: 0x24eb700
//
void __fastcall sub_24EB700(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v6; // rbx
  unsigned __int64 v7; // r12
  __int64 v8; // rax
  __int64 v9; // r8
  __int64 v10; // r9
  unsigned __int64 v11; // rdx
  unsigned __int8 **v12; // rax
  __int64 v13; // r13
  unsigned __int8 *v14; // rbx
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 v17; // rax
  __int64 v18; // rax
  _QWORD *v19; // r12
  __int64 v20; // r14
  unsigned __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // rdx
  int v24; // eax
  int v25; // eax
  unsigned int v26; // edx
  __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // rdx
  int v30; // edx
  __int64 v31; // rax
  __int64 v32; // rdx
  __int64 *v33; // r15
  __int64 *v34; // rdx
  unsigned __int8 *v35; // rbx
  _QWORD *v36; // r13
  unsigned __int8 *v37; // r12
  int v38; // eax
  unsigned int v39; // esi
  __int64 v40; // rax
  __int64 v41; // rsi
  __int64 v42; // rsi
  __int64 v43; // r14
  __int64 v44; // rbx
  int v45; // eax
  __int64 *v46; // r12
  __int64 v47; // r14
  __int64 v48; // rbx
  __int64 v49; // r15
  bool v50; // zf
  _QWORD *v51; // rax
  void *v52; // rcx
  __int64 v53; // rdx
  char v54; // cl
  __int64 v55; // rsi
  _QWORD *v56; // r13
  _QWORD *v57; // rbx
  __int64 v58; // rax
  __int64 v59; // rdx
  __int64 v60; // rax
  __int64 v61; // rdx
  unsigned int v62; // eax
  _QWORD *v63; // rbx
  _QWORD *v64; // r15
  __int64 v65; // rsi
  __int64 v66; // r13
  __int64 v67; // rax
  int v68; // edx
  __int64 v69; // rax
  __int64 v70; // rdx
  __int64 *v71; // r12
  __int64 v72; // r14
  int v73; // ebx
  __int64 v74; // rax
  __int64 v75; // r13
  __int64 v76; // rbx
  int v77; // edx
  unsigned int v78; // ecx
  unsigned __int8 v79; // al
  int v80; // r14d
  __int64 v81; // r8
  __int64 v82; // r9
  __int64 v83; // r14
  __int64 v84; // rbx
  __int64 v85; // rdx
  unsigned int v86; // esi
  __int64 v87; // rax
  unsigned __int64 v88; // rdx
  int v89; // eax
  __int64 **v90; // r15
  __int64 **v91; // rcx
  _BYTE *v92; // r14
  __int64 v93; // rsi
  __int64 v94; // r14
  unsigned int *v95; // r15
  unsigned int *v96; // rbx
  __int64 v97; // rdx
  __int64 *v98; // rax
  __int64 v99; // rax
  unsigned __int64 v100; // rax
  __int64 v101; // rdx
  __int64 v102; // r12
  __int64 v103; // rax
  __int64 v104; // rax
  unsigned __int64 v105; // rdx
  __int64 v106; // rax
  __int64 v107; // rax
  __int64 v108; // rsi
  __int64 *v109; // r14
  int v110; // r15d
  _BYTE *v111; // rdx
  __int64 v112; // rax
  __int64 v113; // [rsp+10h] [rbp-380h]
  _QWORD *v114; // [rsp+18h] [rbp-378h]
  unsigned __int8 *v115; // [rsp+20h] [rbp-370h]
  _QWORD *v116; // [rsp+20h] [rbp-370h]
  unsigned int v117; // [rsp+2Ch] [rbp-364h]
  __int64 *v118; // [rsp+38h] [rbp-358h]
  _QWORD *v119; // [rsp+38h] [rbp-358h]
  __int64 *v120; // [rsp+38h] [rbp-358h]
  unsigned __int8 *v121; // [rsp+40h] [rbp-350h]
  __int64 *v123; // [rsp+58h] [rbp-338h]
  unsigned __int8 **v124; // [rsp+60h] [rbp-330h]
  __int64 *v125; // [rsp+68h] [rbp-328h]
  __int64 v126; // [rsp+70h] [rbp-320h]
  __int64 v127; // [rsp+70h] [rbp-320h]
  __int64 v128; // [rsp+78h] [rbp-318h]
  __int64 v132; // [rsp+A0h] [rbp-2F0h]
  __int64 v133; // [rsp+A8h] [rbp-2E8h]
  __int64 v134; // [rsp+A8h] [rbp-2E8h]
  unsigned __int8 **v135; // [rsp+B0h] [rbp-2E0h]
  unsigned __int64 v136; // [rsp+B8h] [rbp-2D8h]
  __int64 *v137; // [rsp+B8h] [rbp-2D8h]
  __int64 v138; // [rsp+C0h] [rbp-2D0h] BYREF
  __int64 v139; // [rsp+C8h] [rbp-2C8h] BYREF
  _QWORD v140[4]; // [rsp+D0h] [rbp-2C0h] BYREF
  __int16 v141; // [rsp+F0h] [rbp-2A0h]
  void *v142; // [rsp+100h] [rbp-290h]
  _QWORD v143[2]; // [rsp+108h] [rbp-288h] BYREF
  __int64 v144; // [rsp+118h] [rbp-278h]
  __int64 v145; // [rsp+120h] [rbp-270h]
  void *v146; // [rsp+130h] [rbp-260h] BYREF
  __int64 v147; // [rsp+138h] [rbp-258h] BYREF
  __int64 v148; // [rsp+140h] [rbp-250h]
  __int64 v149; // [rsp+148h] [rbp-248h]
  void *i; // [rsp+150h] [rbp-240h]
  __int64 *v151; // [rsp+160h] [rbp-230h] BYREF
  __int64 v152; // [rsp+168h] [rbp-228h]
  _BYTE v153[32]; // [rsp+170h] [rbp-220h] BYREF
  const char *v154[3]; // [rsp+190h] [rbp-200h] BYREF
  char v155; // [rsp+1ACh] [rbp-1E4h]
  __int16 v156; // [rsp+1B0h] [rbp-1E0h]
  __m128i v157; // [rsp+230h] [rbp-160h] BYREF
  _QWORD v158[2]; // [rsp+240h] [rbp-150h] BYREF
  int v159; // [rsp+250h] [rbp-140h]
  __int64 *v160; // [rsp+258h] [rbp-138h]
  __int64 v161; // [rsp+260h] [rbp-130h]
  __int64 v162; // [rsp+268h] [rbp-128h] BYREF
  __int64 v163; // [rsp+270h] [rbp-120h]
  _QWORD *v164; // [rsp+278h] [rbp-118h]
  _QWORD **v165; // [rsp+280h] [rbp-110h]
  void **v166; // [rsp+288h] [rbp-108h]
  __int64 v167; // [rsp+290h] [rbp-100h]
  int v168; // [rsp+298h] [rbp-F8h]
  __int16 v169; // [rsp+29Ch] [rbp-F4h]
  char v170; // [rsp+29Eh] [rbp-F2h]
  __int64 v171; // [rsp+2A0h] [rbp-F0h]
  void **v172; // [rsp+2A8h] [rbp-E8h]
  _QWORD *v173; // [rsp+2B0h] [rbp-E0h] BYREF
  void *v174; // [rsp+2B8h] [rbp-D8h] BYREF
  int v175; // [rsp+2C0h] [rbp-D0h]
  __int16 v176; // [rsp+2C4h] [rbp-CCh]
  char v177; // [rsp+2C6h] [rbp-CAh]
  __int64 v178; // [rsp+2C8h] [rbp-C8h]
  __int64 v179; // [rsp+2D0h] [rbp-C0h]
  void *v180; // [rsp+2D8h] [rbp-B8h] BYREF
  _QWORD v181[4]; // [rsp+2E0h] [rbp-B0h] BYREF
  _QWORD *v182; // [rsp+300h] [rbp-90h]
  __int64 v183; // [rsp+308h] [rbp-88h]
  unsigned int v184; // [rsp+310h] [rbp-80h]
  _QWORD *v185; // [rsp+320h] [rbp-70h]
  unsigned int v186; // [rsp+330h] [rbp-60h]
  char v187; // [rsp+338h] [rbp-58h]
  __int64 v188; // [rsp+348h] [rbp-48h]
  __int64 v189; // [rsp+350h] [rbp-40h]
  __int64 v190; // [rsp+358h] [rbp-38h]

  sub_B2D470(a2, 36);
  sub_B2D520(a2, 22);
  sub_B2D520(a2, 43);
  v6 = *(_QWORD *)(*(_QWORD *)a3 - 32LL * (*(_DWORD *)(*(_QWORD *)a3 + 4LL) & 0x7FFFFFF));
  if ( *(_BYTE *)(a3 + 360) )
  {
    v7 = *(_QWORD *)(v6 + 32 * (2LL - (*(_DWORD *)(v6 + 4) & 0x7FFFFFF)));
  }
  else
  {
    v164 = (_QWORD *)sub_BD5C60(*(_QWORD *)(*(_QWORD *)a3 - 32LL * (*(_DWORD *)(*(_QWORD *)a3 + 4LL) & 0x7FFFFFF)));
    v157.m128i_i64[0] = (__int64)v158;
    v165 = &v173;
    v166 = &v174;
    v173 = &unk_49DA100;
    LOWORD(v163) = 0;
    v169 = 512;
    v157.m128i_i64[1] = 0x200000000LL;
    v174 = &unk_49DA0B0;
    v167 = 0;
    v168 = 0;
    v170 = 7;
    v171 = 0;
    v172 = 0;
    v161 = 0;
    v162 = 0;
    sub_D5F1F0((__int64)&v157, v6);
    v99 = sub_B2BEC0(a2);
    v100 = sub_BDB740(v99, *(_QWORD *)(a3 + 288));
    v152 = v101;
    v151 = (__int64 *)v100;
    v102 = sub_CA1930(&v151);
    v103 = sub_BCB2E0(v164);
    v104 = sub_ACD640(v103, v102, 0);
    v105 = sub_24F4BB0(a3, &v157, v104, 0);
    v106 = *(_QWORD *)a3;
    v156 = 257;
    v7 = sub_24E55A0(v157.m128i_i64, 0x31u, v105, *(__int64 ***)(v106 + 8), (__int64)v154, 0, (int)v146, 0);
    sub_2463EC0(v157.m128i_i64, v7, *(_QWORD *)(v6 + 32 * (2LL - (*(_DWORD *)(v6 + 4) & 0x7FFFFFF))), 0, 0);
    sub_F94A20(&v157, v7);
  }
  v157 = (__m128i)6uLL;
  v8 = *(_QWORD *)(a3 + 312);
  if ( v8 )
  {
    v158[0] = *(_QWORD *)(a3 + 312);
    if ( v8 != -4096 && v8 != -8192 )
      sub_BD73F0((__int64)&v157);
  }
  else
  {
    v158[0] = 0;
  }
  sub_BD84D0(*(_QWORD *)a3, v7);
  *(_QWORD *)(a3 + 312) = v158[0];
  sub_D68D70(&v157);
  v11 = *(unsigned int *)(a3 + 128);
  v151 = (__int64 *)v153;
  v152 = 0x400000000LL;
  v123 = *(__int64 **)(a2 + 64);
  if ( *(_DWORD *)(a4 + 12) < (unsigned int)v11 )
  {
    sub_C8D5F0(a4, (const void *)(a4 + 16), v11, 8u, v9, v10);
    v11 = *(unsigned int *)(a3 + 128);
  }
  v12 = *(unsigned __int8 ***)(a3 + 120);
  v135 = v12;
  v124 = &v12[v11];
  if ( v124 != v12 )
  {
    v13 = 0;
    v128 = 0;
    v132 = 0;
    do
    {
      v138 = v128;
      v14 = *v135;
      LOWORD(v159) = 2819;
      v157.m128i_i64[0] = (__int64)".resume.";
      v158[0] = &v138;
      v133 = sub_24E5090(a2, a3, &v157, v123, 0);
      v17 = *(unsigned int *)(a4 + 8);
      if ( v17 + 1 > (unsigned __int64)*(unsigned int *)(a4 + 12) )
      {
        sub_C8D5F0(a4, (const void *)(a4 + 16), v17 + 1, 8u, v15, v16);
        v17 = *(unsigned int *)(a4 + 8);
      }
      *(_QWORD *)(*(_QWORD *)a4 + 8 * v17) = v133;
      v18 = v126;
      ++*(_DWORD *)(a4 + 8);
      v19 = (_QWORD *)*((_QWORD *)v14 + 5);
      LOWORD(v18) = 0;
      LOWORD(v159) = 257;
      v126 = v18;
      v20 = sub_AA8550(v19, (__int64 *)v14 + 3, v18, (__int64)&v157, 0);
      v21 = v19[6] & 0xFFFFFFFFFFFFFFF8LL;
      if ( (_QWORD *)v21 == v19 + 6 )
        goto LABEL_105;
      if ( !v21 )
        BUG();
      v136 = v21 - 24;
      if ( (unsigned int)*(unsigned __int8 *)(v21 - 24) - 30 > 0xA )
      {
LABEL_105:
        v22 = -32;
        v136 = 0;
        if ( v132 )
        {
LABEL_17:
          if ( !*(_QWORD *)(v136 - 32) || (v23 = *(_QWORD *)(v136 - 24), (**(_QWORD **)(v136 - 16) = v23) == 0) )
          {
            *(_QWORD *)(v136 - 32) = v132;
            goto LABEL_92;
          }
LABEL_19:
          *(_QWORD *)(v23 + 16) = *(_QWORD *)(v136 - 16);
          goto LABEL_20;
        }
      }
      else
      {
        v22 = v21 - 56;
        if ( v132 )
          goto LABEL_17;
      }
      v157.m128i_i64[0] = (__int64)"coro.return";
      LOWORD(v159) = 259;
      v66 = sub_B2BE50(a2);
      v67 = sub_22077B0(0x50u);
      v132 = v67;
      if ( v67 )
        sub_AA4D50(v67, v66, (__int64)&v157, a2, v20);
      *(_QWORD *)(a3 + 352) = v132;
      v164 = (_QWORD *)sub_AA48A0(v132);
      v165 = &v173;
      v166 = &v174;
      v157.m128i_i64[0] = (__int64)v158;
      v157.m128i_i64[1] = 0x200000000LL;
      v173 = &unk_49DA100;
      LOWORD(v163) = 0;
      v68 = *(_DWORD *)(a3 + 128);
      v169 = 512;
      v174 = &unk_49DA0B0;
      v162 = v132 + 48;
      v167 = 0;
      v168 = 0;
      v170 = 7;
      v171 = 0;
      v172 = 0;
      v161 = v132;
      v156 = 257;
      v13 = sub_D5C860(v157.m128i_i64, *(_QWORD *)(v133 + 8), v68, (__int64)v154);
      v69 = **(_QWORD **)(*(_QWORD *)(sub_B43CB0(*(_QWORD *)a3) + 24) + 16LL);
      if ( *(_BYTE *)(v69 + 8) != 15
        || (v70 = *(_QWORD *)(v69 + 16),
            v118 = (__int64 *)(v70 + 8LL * *(unsigned int *)(v69 + 12)),
            v118 == (__int64 *)(v70 + 8)) )
      {
        v89 = v152;
      }
      else
      {
        v115 = v14;
        v113 = v13;
        v114 = v19;
        v71 = (__int64 *)(v70 + 8);
        do
        {
          v72 = *v71;
          LOWORD(i) = 257;
          v73 = *(_DWORD *)(a3 + 128);
          v156 = 257;
          v74 = sub_BD2DA0(80);
          v75 = v74;
          if ( v74 )
          {
            sub_B44260(v74, v72, 55, 0x8000000u, 0, 0);
            *(_DWORD *)(v75 + 72) = v73;
            sub_BD6B50((unsigned __int8 *)v75, v154);
            sub_BD2A10(v75, *(_DWORD *)(v75 + 72), 1);
          }
          if ( *(_BYTE *)v75 > 0x1Cu )
          {
            switch ( *(_BYTE *)v75 )
            {
              case ')':
              case '+':
              case '-':
              case '/':
              case '2':
              case '5':
              case 'J':
              case 'K':
              case 'S':
                goto LABEL_120;
              case 'T':
              case 'U':
              case 'V':
                v76 = *(_QWORD *)(v75 + 8);
                v77 = *(unsigned __int8 *)(v76 + 8);
                v78 = v77 - 17;
                v79 = *(_BYTE *)(v76 + 8);
                if ( (unsigned int)(v77 - 17) <= 1 )
                  v79 = *(_BYTE *)(**(_QWORD **)(v76 + 16) + 8LL);
                if ( v79 <= 3u || v79 == 5 || (v79 & 0xFD) == 4 )
                  goto LABEL_120;
                if ( (_BYTE)v77 == 15 )
                {
                  if ( (*(_BYTE *)(v76 + 9) & 4) == 0 || !sub_BCB420(*(_QWORD *)(v75 + 8)) )
                    break;
                  v98 = *(__int64 **)(v76 + 16);
                  v76 = *v98;
                  v77 = *(unsigned __int8 *)(*v98 + 8);
                  v78 = v77 - 17;
                }
                else if ( (_BYTE)v77 == 16 )
                {
                  do
                  {
                    v76 = *(_QWORD *)(v76 + 24);
                    LOBYTE(v77) = *(_BYTE *)(v76 + 8);
                  }
                  while ( (_BYTE)v77 == 16 );
                  v78 = (unsigned __int8)v77 - 17;
                }
                if ( v78 <= 1 )
                  LOBYTE(v77) = *(_BYTE *)(**(_QWORD **)(v76 + 16) + 8LL);
                if ( (unsigned __int8)v77 <= 3u || (_BYTE)v77 == 5 || (v77 & 0xFD) == 4 )
                {
LABEL_120:
                  v80 = v168;
                  if ( v167 )
                    sub_B99FD0(v75, 3u, v167);
                  sub_B45150(v75, v80);
                }
                break;
              default:
                break;
            }
          }
          (*((void (__fastcall **)(void **, __int64, void **, __int64, __int64))*v166 + 2))(
            v166,
            v75,
            &v146,
            v162,
            v163);
          v83 = v157.m128i_i64[0];
          v84 = v157.m128i_i64[0] + 16LL * v157.m128i_u32[2];
          if ( v157.m128i_i64[0] != v84 )
          {
            do
            {
              v85 = *(_QWORD *)(v83 + 8);
              v86 = *(_DWORD *)v83;
              v83 += 16;
              sub_B99FD0(v75, v86, v85);
            }
            while ( v84 != v83 );
          }
          v87 = (unsigned int)v152;
          v88 = (unsigned int)v152 + 1LL;
          if ( v88 > HIDWORD(v152) )
          {
            sub_C8D5F0((__int64)&v151, v153, v88, 8u, v81, v82);
            v87 = (unsigned int)v152;
          }
          ++v71;
          v151[v87] = v75;
          v89 = v152 + 1;
          LODWORD(v152) = v152 + 1;
        }
        while ( v118 != v71 );
        v14 = v115;
        v19 = v114;
        v13 = v113;
      }
      v90 = **(__int64 ****)(*(_QWORD *)(a2 + 24) + 16LL);
      v91 = v90;
      if ( v89 )
        v91 = (__int64 **)*v90[2];
      v156 = 257;
      v92 = (_BYTE *)sub_24E55A0(v157.m128i_i64, 0x31u, v13, v91, (__int64)v154, 0, (int)v146, 0);
      if ( (_DWORD)v152 )
      {
        v107 = sub_ACADE0(v90);
        v156 = 257;
        LODWORD(v146) = 0;
        v92 = (_BYTE *)sub_2466140(v157.m128i_i64, v107, v92, &v146, 1, (__int64)v154);
        v120 = &v151[(unsigned int)v152];
        if ( v120 != v151 )
        {
          v108 = (__int64)v92;
          v109 = v151;
          v110 = 1;
          do
          {
            v111 = (_BYTE *)*v109;
            v156 = 257;
            ++v109;
            LODWORD(v146) = v110++;
            v112 = sub_2466140(v157.m128i_i64, v108, v111, &v146, 1, (__int64)v154);
            v108 = v112;
          }
          while ( v120 != v109 );
          v92 = (_BYTE *)v112;
        }
      }
      v156 = 257;
      v116 = v164;
      v117 = (v92 != 0) | v117 & 0xE0000000;
      v119 = sub_BD2C40(72, v92 != 0);
      if ( v119 )
        sub_B4BB80((__int64)v119, (__int64)v116, (__int64)v92, v117, 0, 0);
      v93 = (__int64)v119;
      (*((void (__fastcall **)(void **, _QWORD *, const char **, __int64, __int64))*v166 + 2))(
        v166,
        v119,
        v154,
        v162,
        v163);
      v94 = 16LL * v157.m128i_u32[2];
      v95 = (unsigned int *)(v157.m128i_i64[0] + v94);
      if ( v157.m128i_i64[0] != v157.m128i_i64[0] + v94 )
      {
        v121 = v14;
        v96 = (unsigned int *)v157.m128i_i64[0];
        do
        {
          v97 = *((_QWORD *)v96 + 1);
          v93 = *v96;
          v96 += 4;
          sub_B99FD0((__int64)v119, v93, v97);
        }
        while ( v95 != v96 );
        v14 = v121;
      }
      sub_F94A20(&v157, v93);
      v22 = v136 - 32;
      if ( *(_QWORD *)(v136 - 32) )
      {
        v23 = *(_QWORD *)(v136 - 24);
        **(_QWORD **)(v136 - 16) = v23;
        if ( v23 )
          goto LABEL_19;
      }
LABEL_20:
      *(_QWORD *)(v136 - 32) = v132;
      if ( !v132 )
      {
        v24 = *(_DWORD *)(v13 + 4) & 0x7FFFFFF;
        if ( v24 != *(_DWORD *)(v13 + 72) )
          goto LABEL_22;
        goto LABEL_95;
      }
LABEL_92:
      v61 = *(_QWORD *)(v132 + 16);
      *(_QWORD *)(v136 - 24) = v61;
      if ( v61 )
        *(_QWORD *)(v61 + 16) = v136 - 24;
      *(_QWORD *)(v136 - 16) = v132 + 16;
      *(_QWORD *)(v132 + 16) = v22;
      v24 = *(_DWORD *)(v13 + 4) & 0x7FFFFFF;
      if ( v24 != *(_DWORD *)(v13 + 72) )
        goto LABEL_22;
LABEL_95:
      sub_B48D90(v13);
      v24 = *(_DWORD *)(v13 + 4) & 0x7FFFFFF;
LABEL_22:
      v25 = (v24 + 1) & 0x7FFFFFF;
      v26 = v25 | *(_DWORD *)(v13 + 4) & 0xF8000000;
      v27 = *(_QWORD *)(v13 - 8) + 32LL * (unsigned int)(v25 - 1);
      *(_DWORD *)(v13 + 4) = v26;
      if ( *(_QWORD *)v27 )
      {
        v28 = *(_QWORD *)(v27 + 8);
        **(_QWORD **)(v27 + 16) = v28;
        if ( v28 )
          *(_QWORD *)(v28 + 16) = *(_QWORD *)(v27 + 16);
      }
      *(_QWORD *)v27 = v133;
      if ( v133 )
      {
        v29 = *(_QWORD *)(v133 + 16);
        *(_QWORD *)(v27 + 8) = v29;
        if ( v29 )
          *(_QWORD *)(v29 + 16) = v27 + 8;
        *(_QWORD *)(v27 + 16) = v133 + 16;
        *(_QWORD *)(v133 + 16) = v27;
      }
      *(_QWORD *)(*(_QWORD *)(v13 - 8)
                + 32LL * *(unsigned int *)(v13 + 72)
                + 8LL * ((*(_DWORD *)(v13 + 4) & 0x7FFFFFFu) - 1)) = v19;
      v30 = *v14;
      if ( v30 == 40 )
      {
        sub_B491D0((__int64)v14);
      }
      else if ( v30 != 85 && v30 != 34 )
      {
        BUG();
      }
      if ( (v14[7] & 0x80u) != 0 )
      {
        v31 = sub_BD2BC0((__int64)v14);
        if ( (v14[7] & 0x80u) != 0 )
        {
          if ( (unsigned int)((v31 + v32 - sub_BD2BC0((__int64)v14)) >> 4) )
          {
            if ( (v14[7] & 0x80u) != 0 )
            {
              sub_BD2BC0((__int64)v14);
              if ( (v14[7] & 0x80u) != 0 )
                sub_BD2BC0((__int64)v14);
            }
          }
        }
      }
      v33 = v151;
      v34 = &v151[(unsigned int)v152];
      v35 = &v14[-32 * (*((_DWORD *)v14 + 1) & 0x7FFFFFF)];
      if ( v34 != v151 )
      {
        v134 = v13;
        v36 = v19;
        v37 = v35;
        do
        {
          v43 = *v33;
          v44 = *(_QWORD *)v37;
          v45 = *(_DWORD *)(*v33 + 4) & 0x7FFFFFF;
          if ( v45 == *(_DWORD *)(*v33 + 72) )
          {
            v137 = v34;
            sub_B48D90(*v33);
            v34 = v137;
            v45 = *(_DWORD *)(v43 + 4) & 0x7FFFFFF;
          }
          v38 = (v45 + 1) & 0x7FFFFFF;
          v39 = v38 | *(_DWORD *)(v43 + 4) & 0xF8000000;
          v40 = *(_QWORD *)(v43 - 8) + 32LL * (unsigned int)(v38 - 1);
          *(_DWORD *)(v43 + 4) = v39;
          if ( *(_QWORD *)v40 )
          {
            v41 = *(_QWORD *)(v40 + 8);
            **(_QWORD **)(v40 + 16) = v41;
            if ( v41 )
              *(_QWORD *)(v41 + 16) = *(_QWORD *)(v40 + 16);
          }
          *(_QWORD *)v40 = v44;
          if ( v44 )
          {
            v42 = *(_QWORD *)(v44 + 16);
            *(_QWORD *)(v40 + 8) = v42;
            if ( v42 )
              *(_QWORD *)(v42 + 16) = v40 + 8;
            *(_QWORD *)(v40 + 16) = v44 + 16;
            *(_QWORD *)(v44 + 16) = v40;
          }
          ++v33;
          v37 += 32;
          *(_QWORD *)(*(_QWORD *)(v43 - 8)
                    + 32LL * *(unsigned int *)(v43 + 72)
                    + 8LL * ((*(_DWORD *)(v43 + 4) & 0x7FFFFFFu) - 1)) = v36;
        }
        while ( v34 != v33 );
        v13 = v134;
      }
      ++v135;
      ++v128;
    }
    while ( v124 != v135 );
  }
  sub_24E3E60((__int64)v154, a2);
  v46 = *(__int64 **)(a3 + 120);
  v125 = &v46[*(unsigned int *)(a3 + 128)];
  if ( v125 != v46 )
  {
    v47 = 0;
    do
    {
      v139 = v47;
      v48 = *v46;
      v49 = *(_QWORD *)(*(_QWORD *)a4 + 8 * v47);
      v140[0] = "resume.";
      v140[2] = &v139;
      v141 = 2819;
      v127 = sub_C996C0("BaseCloner", 10, 0, 0);
      v157.m128i_i64[1] = a2;
      v157.m128i_i64[0] = (__int64)&unk_4A16A60;
      v158[0] = v140;
      v50 = *(_DWORD *)(a3 + 280) == 3;
      v158[1] = a3;
      v159 = v50 + 3;
      v171 = sub_B2BE50(a2);
      v160 = &v162;
      v172 = &v180;
      v161 = 0x200000000LL;
      v173 = v181;
      v176 = 512;
      v180 = &unk_49DA100;
      LOWORD(v168) = 0;
      v181[0] = &unk_49DA0B0;
      v174 = 0;
      v181[1] = a5;
      v175 = 0;
      v177 = 7;
      v178 = 0;
      v179 = 0;
      v166 = 0;
      v167 = 0;
      v181[2] = v154;
      v181[3] = 0;
      v184 = 128;
      v51 = (_QWORD *)sub_C7D670(0x2000, 8);
      v52 = &unk_49DD7A0;
      v183 = 0;
      v182 = v51;
      v147 = 2;
      v53 = (__int64)&v51[8 * (unsigned __int64)v184];
      v146 = &unk_49DD7B0;
      v148 = 0;
      v149 = -4096;
      for ( i = 0; (_QWORD *)v53 != v51; v51 += 8 )
      {
        if ( v51 )
        {
          v54 = v147;
          v51[2] = 0;
          v51[3] = -4096;
          *v51 = &unk_49DD7B0;
          v51[1] = v54 & 6;
          v52 = i;
          v51[4] = i;
        }
      }
      v187 = 0;
      v188 = v49;
      v189 = 0;
      v190 = v48;
      sub_24EA1C0((__int64)&v157, (__int64)&unk_49DD7B0, v53, (__int64)v52);
      v157.m128i_i64[0] = (__int64)&unk_4A16A60;
      if ( v187 )
      {
        v62 = v186;
        v187 = 0;
        if ( v186 )
        {
          v63 = v185;
          v64 = &v185[2 * v186];
          do
          {
            if ( *v63 != -8192 && *v63 != -4096 )
            {
              v65 = v63[1];
              if ( v65 )
                sub_B91220((__int64)(v63 + 1), v65);
            }
            v63 += 2;
          }
          while ( v64 != v63 );
          v62 = v186;
        }
        sub_C7D6A0((__int64)v185, 16LL * v62, 8);
      }
      v55 = v184;
      if ( v184 )
      {
        v56 = v182;
        v143[0] = 2;
        v143[1] = 0;
        v57 = &v182[8 * (unsigned __int64)v184];
        v144 = -4096;
        v142 = &unk_49DD7B0;
        v146 = &unk_49DD7B0;
        v58 = -4096;
        v145 = 0;
        v147 = 2;
        v148 = 0;
        v149 = -8192;
        i = 0;
        while ( 1 )
        {
          v59 = v56[3];
          if ( v59 != v58 )
          {
            v58 = v149;
            if ( v59 != v149 )
            {
              v60 = v56[7];
              if ( v60 != 0 && v60 != -4096 && v60 != -8192 )
              {
                sub_BD60C0(v56 + 5);
                v59 = v56[3];
              }
              v58 = v59;
            }
          }
          *v56 = &unk_49DB368;
          if ( v58 != 0 && v58 != -4096 && v58 != -8192 )
            sub_BD60C0(v56 + 1);
          v56 += 8;
          if ( v57 == v56 )
            break;
          v58 = v144;
        }
        v146 = &unk_49DB368;
        if ( v149 != -4096 && v149 != 0 && v149 != -8192 )
          sub_BD60C0(&v147);
        v142 = &unk_49DB368;
        if ( v144 != -4096 && v144 != 0 && v144 != -8192 )
          sub_BD60C0(v143);
        v55 = v184;
      }
      sub_C7D6A0((__int64)v182, v55 << 6, 8);
      nullsub_61();
      v180 = &unk_49DA100;
      nullsub_63();
      if ( v160 != &v162 )
        _libc_free((unsigned __int64)v160);
      if ( v127 )
        sub_C9AF60(v127);
      ++v47;
      ++v46;
    }
    while ( v125 != v46 );
  }
  if ( !v155 )
    _libc_free((unsigned __int64)v154[1]);
  if ( v151 != (__int64 *)v153 )
    _libc_free((unsigned __int64)v151);
}
