// Function: sub_303E7C0
// Address: 0x303e7c0
//
__int64 __fastcall sub_303E7C0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        _QWORD *a6,
        _QWORD *a7,
        __int64 a8,
        __int64 a9)
{
  int v9; // r12d
  __int64 v11; // r14
  __int64 v12; // r15
  __int64 *v13; // rdi
  __int64 v14; // rbx
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 v17; // r14
  int v18; // ebx
  __int64 v19; // rax
  __int16 v20; // si
  __int64 v21; // rax
  unsigned __int64 v22; // r13
  __int64 v23; // rax
  unsigned __int64 v24; // rdx
  __int64 *v25; // rax
  unsigned __int16 *v26; // rdi
  __int64 *v27; // rax
  __int64 v28; // r12
  __int64 v29; // r13
  __int64 v30; // r15
  __int64 v31; // rdx
  __int16 v32; // ax
  bool v33; // al
  bool v34; // al
  _BYTE *v35; // rax
  bool v36; // al
  bool v37; // al
  int v38; // edx
  char v39; // cl
  __int64 v40; // r9
  __int64 v41; // r13
  __int64 v42; // rbx
  __int64 v43; // rax
  unsigned __int64 v44; // rdx
  int v45; // edx
  __int64 v46; // rax
  __int64 v47; // r10
  __int64 v48; // r11
  int v49; // eax
  int v50; // edx
  __int64 v51; // rax
  __int64 v52; // rdx
  __int64 v53; // r12
  __int64 v54; // rax
  __int64 v55; // rcx
  __int64 v56; // rax
  __int64 v57; // rax
  __int64 v58; // r8
  __int64 v59; // rcx
  __int64 v60; // rdx
  int v61; // eax
  __int64 v62; // rax
  __int64 v63; // r15
  char v64; // al
  unsigned __int64 v65; // rax
  __int64 v66; // rax
  unsigned __int64 v67; // rdx
  __int64 v68; // rax
  __int64 v69; // rdx
  unsigned __int64 v70; // rdx
  __int64 *v71; // rax
  __int64 v72; // rax
  unsigned __int16 v73; // dx
  __int64 v74; // rax
  __int64 v75; // rdx
  __int64 v76; // rdx
  char v77; // al
  __int64 v78; // rdx
  __int64 v79; // r12
  __int64 v81; // rax
  __int64 v82; // r12
  __m128i v83; // xmm5
  unsigned __int16 v84; // ax
  __int64 v85; // rdx
  __int64 v86; // rax
  __int64 v87; // rdx
  unsigned __int64 v88; // rax
  __int64 v89; // r10
  __int64 v90; // rbx
  __int32 v91; // r11d
  __int64 v92; // r15
  __int128 v93; // rax
  int v94; // r9d
  __int64 v95; // rax
  __int64 v96; // rdx
  __int64 v97; // r13
  __int64 v98; // r12
  __int64 v99; // rax
  __int64 v100; // rdx
  __int64 v101; // rcx
  int v102; // edx
  __int64 v103; // rax
  int v104; // eax
  __int64 v105; // rax
  __int64 v106; // rdx
  char v107; // bl
  __int64 v108; // rax
  __int64 v109; // rdx
  int v110; // edx
  __int128 v111; // [rsp-10h] [rbp-5B0h]
  __int64 v112; // [rsp-10h] [rbp-5B0h]
  __int64 v113; // [rsp-8h] [rbp-5A8h]
  __int64 v114; // [rsp+8h] [rbp-598h]
  __int64 v116; // [rsp+30h] [rbp-570h]
  __int64 v117; // [rsp+40h] [rbp-560h]
  __int128 v118; // [rsp+40h] [rbp-560h]
  __int64 v119; // [rsp+58h] [rbp-548h]
  __int64 v120; // [rsp+60h] [rbp-540h]
  __m128i v121; // [rsp+68h] [rbp-538h]
  unsigned int v122; // [rsp+70h] [rbp-530h]
  __int64 v123; // [rsp+70h] [rbp-530h]
  __int64 v124; // [rsp+70h] [rbp-530h]
  unsigned __int8 v125; // [rsp+70h] [rbp-530h]
  __int64 v126; // [rsp+70h] [rbp-530h]
  __int64 v127; // [rsp+78h] [rbp-528h]
  __int64 v128; // [rsp+78h] [rbp-528h]
  unsigned __int16 *v129; // [rsp+80h] [rbp-520h]
  unsigned int v130; // [rsp+80h] [rbp-520h]
  __int64 v131; // [rsp+80h] [rbp-520h]
  __int64 v132; // [rsp+80h] [rbp-520h]
  __int64 v133; // [rsp+88h] [rbp-518h]
  __int64 v134; // [rsp+A0h] [rbp-500h]
  __int64 v136; // [rsp+B8h] [rbp-4E8h]
  bool v137; // [rsp+B8h] [rbp-4E8h]
  __int64 v138; // [rsp+C0h] [rbp-4E0h]
  __int64 v139; // [rsp+C8h] [rbp-4D8h]
  __m128i v140; // [rsp+D0h] [rbp-4D0h] BYREF
  __m128i v141; // [rsp+E0h] [rbp-4C0h] BYREF
  __int64 v142; // [rsp+F0h] [rbp-4B0h]
  __int64 v143; // [rsp+F8h] [rbp-4A8h]
  __int64 v144; // [rsp+100h] [rbp-4A0h]
  __int64 v145; // [rsp+108h] [rbp-498h]
  __m128i v146; // [rsp+110h] [rbp-490h]
  __int64 v147; // [rsp+120h] [rbp-480h]
  __int64 v148; // [rsp+128h] [rbp-478h]
  __int64 v149; // [rsp+130h] [rbp-470h]
  __int64 v150; // [rsp+138h] [rbp-468h]
  __int64 v151; // [rsp+140h] [rbp-460h]
  __int64 v152; // [rsp+148h] [rbp-458h]
  __m128i v153; // [rsp+150h] [rbp-450h] BYREF
  __int64 v154; // [rsp+160h] [rbp-440h]
  __int64 v155; // [rsp+168h] [rbp-438h]
  __m128i v156; // [rsp+170h] [rbp-430h] BYREF
  __int64 v157; // [rsp+180h] [rbp-420h]
  __int64 v158; // [rsp+188h] [rbp-418h]
  __int64 v159; // [rsp+190h] [rbp-410h]
  __int64 v160; // [rsp+198h] [rbp-408h]
  __int64 v161; // [rsp+1A0h] [rbp-400h]
  __int64 v162; // [rsp+1B0h] [rbp-3F0h] BYREF
  __int64 v163; // [rsp+1B8h] [rbp-3E8h]
  __int64 v164; // [rsp+1C0h] [rbp-3E0h]
  __int64 v165; // [rsp+1C8h] [rbp-3D8h]
  __int64 v166; // [rsp+1D0h] [rbp-3D0h] BYREF
  __int64 v167; // [rsp+1D8h] [rbp-3C8h]
  __int64 v168; // [rsp+1E0h] [rbp-3C0h]
  __int64 v169; // [rsp+1E8h] [rbp-3B8h]
  __int64 v170; // [rsp+1F0h] [rbp-3B0h]
  __int64 v171; // [rsp+1F8h] [rbp-3A8h]
  unsigned __int64 v172[2]; // [rsp+200h] [rbp-3A0h] BYREF
  char v173; // [rsp+210h] [rbp-390h] BYREF
  _BYTE *v174; // [rsp+250h] [rbp-350h] BYREF
  __int64 v175; // [rsp+258h] [rbp-348h]
  _BYTE v176[96]; // [rsp+260h] [rbp-340h] BYREF
  unsigned __int64 v177[2]; // [rsp+2C0h] [rbp-2E0h] BYREF
  _BYTE v178[128]; // [rsp+2D0h] [rbp-2D0h] BYREF
  _BYTE *v179; // [rsp+350h] [rbp-250h] BYREF
  __int64 v180; // [rsp+358h] [rbp-248h]
  _BYTE v181[256]; // [rsp+360h] [rbp-240h] BYREF
  _BYTE *v182; // [rsp+460h] [rbp-140h] BYREF
  __int64 v183; // [rsp+468h] [rbp-138h]
  _BYTE v184[304]; // [rsp+470h] [rbp-130h] BYREF

  v11 = a9;
  v140.m128i_i64[0] = a2;
  v12 = a8;
  v13 = *(__int64 **)(a9 + 40);
  v140.m128i_i64[1] = a3;
  v117 = *v13;
  v14 = **(_QWORD **)(*(_QWORD *)(*v13 + 24) + 16LL);
  v133 = v14;
  v182 = v184;
  v119 = sub_2E79000(v13);
  v179 = v181;
  v177[0] = (unsigned __int64)v178;
  v180 = 0x1000000000LL;
  v183 = 0x1000000000LL;
  v177[1] = 0x1000000000LL;
  sub_30351C0(a1, v119, v14, (__int64)&v182, (__int64)v177, 0);
  if ( !(_DWORD)v183 )
    goto LABEL_17;
  v136 = 16LL * (unsigned int)v183;
  v17 = 0;
  HIWORD(v18) = HIWORD(v9);
  v141.m128i_i64[0] = 0;
  do
  {
    v26 = (unsigned __int16 *)&v182[v17];
    v27 = (__int64 *)(v17 + *a7);
    v28 = *v27;
    v29 = v27[1];
    v30 = *v27;
    v31 = *((unsigned int *)v27 + 2);
    LOWORD(v172[0]) = 0;
    v32 = *(_WORD *)&v182[v17];
    if ( v32 )
    {
      if ( (unsigned __int16)(v32 - 2) > 7u )
        goto LABEL_4;
    }
    else
    {
      v122 = v31;
      v129 = (unsigned __int16 *)&v182[v17];
      v33 = sub_30070A0((__int64)v26);
      v26 = v129;
      v31 = v122;
      if ( !v33 )
        goto LABEL_4;
    }
    v130 = v31;
    v34 = sub_3031480(v26, v172);
    v31 = v130;
    if ( v34 )
    {
      v35 = &v182[v17];
      *(_WORD *)v35 = v172[0];
      *((_QWORD *)v35 + 1) = 0;
    }
LABEL_4:
    v19 = *(_QWORD *)(v30 + 48) + 16 * v31;
    v20 = *(_WORD *)v19;
    v21 = *(_QWORD *)(v19 + 8);
    LOWORD(v174) = v20;
    v175 = v21;
    if ( v20 )
    {
      if ( (unsigned __int16)(v20 - 2) > 7u )
        goto LABEL_6;
    }
    else
    {
      v123 = v31;
      v36 = sub_30070A0((__int64)&v174);
      v31 = v123;
      if ( !v36 )
        goto LABEL_6;
    }
    v131 = v31;
    v37 = sub_3031480((unsigned __int16 *)&v174, v172);
    v31 = v131;
    if ( v37 )
    {
      LOWORD(v18) = v172[0];
      *((_QWORD *)&v111 + 1) = v29;
      *(_QWORD *)&v111 = v28;
      v30 = sub_33FAF80(a9, (unsigned int)((*(_BYTE *)(*a6 + v141.m128i_i64[0]) & 2) == 0) + 213, a8, v18, 0, v16, v111);
      v31 = (unsigned int)v31;
    }
LABEL_6:
    v22 = v31 | v29 & 0xFFFFFFFF00000000LL;
    v23 = (unsigned int)v180;
    v24 = (unsigned int)v180 + 1LL;
    if ( v24 > HIDWORD(v180) )
    {
      sub_C8D5F0((__int64)&v179, v181, v24, 0x10u, v15, v16);
      v23 = (unsigned int)v180;
    }
    v25 = (__int64 *)&v179[16 * v23];
    v141.m128i_i64[0] += 56;
    v17 += 16;
    *v25 = v30;
    v25[1] = v22;
    LODWORD(v180) = v180 + 1;
  }
  while ( v136 != v17 );
  v12 = a8;
  v11 = a9;
LABEL_17:
  v38 = *(unsigned __int8 *)(v133 + 8);
  if ( (_BYTE)v38 == 12
    || (unsigned __int8)v38 <= 3u
    || (_BYTE)v38 == 5
    || (v38 & 0xFB) == 0xA
    || (v38 & 0xFD) == 4
    || ((unsigned __int8)(*(_BYTE *)(v133 + 8) - 15) <= 3u || v38 == 20) && (unsigned __int8)sub_BCEBA0(v133, 0) )
  {
    v39 = sub_303E610(a1, v117, v133, v119);
  }
  else
  {
    v39 = 0;
  }
  sub_3031850(v172, (__int64)&v182, v177, v39, 0);
  v137 = 0;
  if ( *(_BYTE *)(v133 + 8) == 12 )
  {
    v107 = sub_AE5020(v119, v133);
    v108 = sub_9208B0(v119, v133);
    v175 = v109;
    v174 = (_BYTE *)(8 * (((1LL << v107) + ((unsigned __int64)(v108 + 7) >> 3) - 1) >> v107 << v107));
    v137 = (unsigned __int64)sub_CA1930(&v174) <= 0x1F;
  }
  v41 = v12;
  v174 = v176;
  v132 = 4LL * (unsigned int)v183;
  v42 = 0;
  v175 = 0x600000000LL;
  if ( (_DWORD)v183 )
  {
    while ( 1 )
    {
      v53 = 4 * v42;
      v54 = 4 * v42 + *a7;
      v141 = _mm_loadu_si128((const __m128i *)&v179[4 * v42]);
      v55 = *(_QWORD *)v54;
      v56 = *(unsigned int *)(v54 + 8);
      if ( v137 )
      {
        v57 = sub_33FAF80(
                v11,
                (unsigned int)((*(_BYTE *)(*a6 + 14 * v42) & 2) == 0) + 213,
                v41,
                7,
                0,
                v40,
                *(_OWORD *)&v141);
        v59 = v112;
        v151 = v57;
        v141.m128i_i64[0] = v57;
        v152 = v60;
        v60 = (unsigned int)v60;
        v141.m128i_i64[1] = (unsigned int)v60 | v141.m128i_i64[1] & 0xFFFFFFFF00000000LL;
      }
      else
      {
        v72 = *(_QWORD *)(v55 + 48) + 16 * v56;
        v73 = *(_WORD *)v72;
        v74 = *(_QWORD *)(v72 + 8);
        LOWORD(v166) = v73;
        v167 = v74;
        if ( v73 )
        {
          if ( v73 == 1 || (unsigned __int16)(v73 - 504) <= 7u )
LABEL_102:
            BUG();
          v81 = 16LL * (v73 - 1);
          v76 = *(_QWORD *)&byte_444C4A0[v81];
          v77 = byte_444C4A0[v81 + 8];
        }
        else
        {
          v154 = sub_3007260((__int64)&v166);
          v155 = v75;
          v76 = v154;
          v77 = v155;
        }
        v166 = v76;
        LOBYTE(v167) = v77;
        if ( (unsigned __int64)sub_CA1930(&v166) <= 0xF )
        {
          v149 = sub_33FAF80(v11, 215, v41, 6, 0, v40, *(_OWORD *)&v141);
          v141.m128i_i64[0] = v149;
          v150 = v78;
          v141.m128i_i64[1] = (unsigned int)v78 | v141.m128i_i64[1] & 0xFFFFFFFF00000000LL;
          v60 = v113;
        }
      }
      v61 = *(_DWORD *)(v172[0] + v42);
      if ( v61 != 3 )
      {
        if ( (v61 & 1) != 0 )
          goto LABEL_44;
        goto LABEL_25;
      }
      if ( (unsigned int)*(unsigned __int8 *)(v133 + 8) - 15 > 1 )
        goto LABEL_44;
      if ( v137 )
        v153 = (__m128i)7uLL;
      else
        v153 = _mm_loadu_si128((const __m128i *)&v182[v53]);
      v62 = sub_3007410((__int64)&v153, *(__int64 **)v133, v60, v59, v58, v40);
      v125 = sub_AE5020(v119, v62);
      v63 = *(_QWORD *)(v177[0] + 2 * v42);
      v64 = sub_AE5020(v119, v133);
      v65 = (v63 | (1LL << v64)) & -(v63 | (1LL << v64));
      if ( !v65 || (_BitScanReverse64(&v65, v65), v125 <= (unsigned __int8)(63 - (v65 ^ 0x3F))) )
      {
        if ( (*(_DWORD *)(v172[0] + v42) & 1) != 0 )
        {
LABEL_44:
          v66 = (unsigned int)v175;
          v67 = (unsigned int)v175 + 1LL;
          if ( v67 > HIDWORD(v175) )
          {
            sub_C8D5F0((__int64)&v174, v176, v67, 0x10u, v58, v40);
            v66 = (unsigned int)v175;
          }
          *(__m128i *)&v174[16 * v66] = _mm_load_si128(&v140);
          LODWORD(v175) = v175 + 1;
          v58 = sub_3400BD0(v11, *(_QWORD *)(v177[0] + 2 * v42), v41, 7, 0, 0, 0);
          v68 = (unsigned int)v175;
          v40 = v69;
          v70 = (unsigned int)v175 + 1LL;
          if ( v70 > HIDWORD(v175) )
          {
            v126 = v58;
            v128 = v40;
            sub_C8D5F0((__int64)&v174, v176, v70, 0x10u, v58, v40);
            v68 = (unsigned int)v175;
            v58 = v126;
            v40 = v128;
          }
          v71 = (__int64 *)&v174[16 * v68];
          *v71 = v58;
          v71[1] = v40;
          v43 = (unsigned int)(v175 + 1);
          v44 = v43 + 1;
          LODWORD(v175) = v175 + 1;
          if ( v43 + 1 > (unsigned __int64)HIDWORD(v175) )
          {
LABEL_49:
            sub_C8D5F0((__int64)&v174, v176, v44, 0x10u, v58, v40);
            v43 = (unsigned int)v175;
          }
LABEL_26:
          *(__m128i *)&v174[16 * v43] = _mm_load_si128(&v141);
          v45 = v175;
          v46 = (unsigned int)(v175 + 1);
          LODWORD(v175) = v175 + 1;
          if ( (*(_BYTE *)(v172[0] + v42) & 2) != 0 )
          {
            if ( v45 == 3 )
            {
              v141.m128i_i32[0] = 577;
            }
            else
            {
              v141.m128i_i32[0] = 578;
              if ( v45 != 5 )
              {
                if ( v45 != 2 )
                  goto LABEL_102;
                v141.m128i_i32[0] = 576;
              }
            }
            if ( v137 )
            {
              v47 = 7;
              v48 = 0;
            }
            else
            {
              v47 = *(_QWORD *)&v182[4 * v42];
              v48 = *(_QWORD *)&v182[v53 + 8];
            }
            v124 = v47;
            v127 = v48;
            v138 = (__int64)v174;
            v166 = 0;
            v167 = 0;
            v168 = 0;
            v169 = 0;
            v162 = 0;
            v163 = 0;
            LODWORD(v164) = 0;
            BYTE4(v164) = 0;
            v139 = v46;
            v49 = sub_33ED250(v11, 1, 0, (unsigned int)(v45 - 1));
            v51 = sub_33EB1C0(
                    v11,
                    v141.m128i_i32[0],
                    v41,
                    v49,
                    v50,
                    0,
                    v138,
                    v139,
                    v124,
                    v127,
                    v162,
                    v163,
                    v164,
                    2,
                    0,
                    (__int64)&v166);
            LODWORD(v175) = 0;
            v142 = v51;
            v140.m128i_i64[0] = v51;
            v143 = v52;
            v140.m128i_i64[1] = (unsigned int)v52 | v140.m128i_i64[1] & 0xFFFFFFFF00000000LL;
          }
          goto LABEL_34;
        }
LABEL_25:
        v43 = (unsigned int)v175;
        v44 = (unsigned int)v175 + 1LL;
        if ( v44 > HIDWORD(v175) )
          goto LABEL_49;
        goto LABEL_26;
      }
      v82 = *(_QWORD *)(v177[0] + 2 * v42);
      v83 = _mm_load_si128(&v141);
      v121 = v140;
      v84 = v153.m128i_i16[0];
      v156 = _mm_load_si128(&v153);
      v118 = (__int128)v83;
      if ( v153.m128i_i16[0] == 13 )
      {
        v156.m128i_i64[1] = 0;
        v156.m128i_i16[0] = 8;
        goto LABEL_84;
      }
      if ( v153.m128i_i16[0] > 0xDu )
        break;
      if ( v153.m128i_i16[0] == 12 )
        goto LABEL_95;
      if ( v153.m128i_i16[0] <= 9u )
        goto LABEL_85;
      v156.m128i_i64[1] = 0;
      v156.m128i_i16[0] = 6;
LABEL_84:
      v147 = sub_33FAF80(v11, 234, v41, v156.m128i_i32[0], v156.m128i_i32[2], v40, *(_OWORD *)&v141);
      *(_QWORD *)&v118 = v147;
      v148 = v85;
      *((_QWORD *)&v118 + 1) = (unsigned int)v85 | *((_QWORD *)&v118 + 1) & 0xFFFFFFFF00000000LL;
      v84 = v156.m128i_i16[0];
LABEL_85:
      if ( v84 )
      {
        v110 = v84;
        if ( v84 == 1 )
          goto LABEL_102;
        goto LABEL_98;
      }
      v86 = sub_3007260((__int64)&v156);
      v157 = v86;
      v158 = v87;
LABEL_87:
      v166 = v86;
      LOBYTE(v167) = v87;
      v88 = (unsigned __int64)sub_CA1930(&v166) >> 3;
      if ( (_DWORD)v88 )
      {
        v89 = v82 + 1;
        v114 = v42;
        v141.m128i_i32[0] = 0;
        v90 = v41;
        v91 = v82;
        v116 = v82 + 1 + (unsigned int)(v88 - 1);
        v92 = v121.m128i_i64[0];
        while ( 1 )
        {
          v120 = v89;
          v121.m128i_i32[0] = v91;
          *(_QWORD *)&v93 = sub_3400BD0(v11, v141.m128i_i32[0], v90, 7, 0, 0, 0);
          v95 = sub_3406EB0(v11, 192, v90, v156.m128i_i32[0], v156.m128i_i32[2], v94, v118, v93);
          v97 = v96;
          v98 = v95;
          v166 = v92;
          v167 = v121.m128i_i64[1];
          v99 = sub_3400BD0(v11, v121.m128i_i32[0], v90, 7, 0, 0, 0);
          v170 = v98;
          v169 = v100;
          v171 = v97;
          v168 = v99;
          v162 = 0;
          v163 = 0;
          v164 = 0;
          v165 = 0;
          LODWORD(v97) = sub_33ED250(v11, 1, 0, v101);
          LODWORD(v98) = v102;
          v103 = v134;
          LOWORD(v103) = 5;
          v134 = v103;
          v159 = 0;
          v160 = 0;
          LODWORD(v161) = 0;
          BYTE4(v161) = 0;
          v104 = sub_33CC4A0(v11, v103, 0);
          v105 = sub_33EB1C0(
                   v11,
                   576,
                   v90,
                   v97,
                   v98,
                   v104,
                   (__int64)&v166,
                   3,
                   v134,
                   0,
                   v159,
                   v160,
                   v161,
                   2,
                   0,
                   (__int64)&v162);
          v141.m128i_i32[0] += 8;
          v144 = v105;
          v92 = v105;
          v145 = v106;
          v91 = v120;
          v121.m128i_i64[1] = (unsigned int)v106 | v121.m128i_i64[1] & 0xFFFFFFFF00000000LL;
          if ( v120 == v116 )
            break;
          v89 = v120 + 1;
        }
        v121.m128i_i64[0] = v105;
        v41 = v90;
        v42 = v114;
      }
      v146 = v121;
      v140.m128i_i64[0] = v121.m128i_i64[0];
      v140.m128i_i64[1] = v121.m128i_u32[2] | v140.m128i_i64[1] & 0xFFFFFFFF00000000LL;
LABEL_34:
      v42 += 4;
      if ( v42 == v132 )
      {
        LODWORD(v12) = v41;
        goto LABEL_64;
      }
    }
    if ( v153.m128i_i16[0] != 127 && v153.m128i_i16[0] != 138 )
    {
      v110 = v153.m128i_u16[0];
LABEL_98:
      if ( (unsigned __int16)(v84 - 504) <= 7u )
        goto LABEL_102;
      v87 = 16LL * (v110 - 1);
      v86 = *(_QWORD *)&byte_444C4A0[v87];
      LOBYTE(v87) = byte_444C4A0[v87 + 8];
      goto LABEL_87;
    }
LABEL_95:
    v156.m128i_i64[1] = 0;
    v156.m128i_i16[0] = 7;
    goto LABEL_84;
  }
LABEL_64:
  v79 = sub_33FAF80(v11, 503, v12, 1, 0, v40, *(_OWORD *)&v140);
  if ( v174 != v176 )
    _libc_free((unsigned __int64)v174);
  if ( (char *)v172[0] != &v173 )
    _libc_free(v172[0]);
  if ( (_BYTE *)v177[0] != v178 )
    _libc_free(v177[0]);
  if ( v182 != v184 )
    _libc_free((unsigned __int64)v182);
  if ( v179 != v181 )
    _libc_free((unsigned __int64)v179);
  return v79;
}
