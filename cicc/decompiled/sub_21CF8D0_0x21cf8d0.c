// Function: sub_21CF8D0
// Address: 0x21cf8d0
//
__m128i *__fastcall sub_21CF8D0(
        __m128i *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        _QWORD *a5,
        _QWORD *a6,
        unsigned int a7,
        __int64 *a8,
        __int64 a9,
        __int64 a10)
{
  unsigned int v12; // edx
  char v13; // al
  __int64 v15; // rax
  __int64 v16; // rax
  __m128i *v17; // rdx
  __m128i si128; // xmm0
  _BYTE *v19; // rdx
  unsigned int v20; // ebx
  unsigned int v21; // r14d
  char v22; // al
  __int64 v23; // r12
  unsigned __int8 v24; // dl
  _DWORD *v25; // rax
  __int64 v26; // rsi
  unsigned __int64 v27; // r13
  __int64 v28; // rcx
  unsigned __int8 v29; // al
  unsigned int v30; // eax
  unsigned int v31; // ebx
  _QWORD *v32; // rdx
  void **v33; // rdi
  __int64 v34; // rsi
  __int64 v35; // rax
  _WORD *v36; // rdx
  __int64 v37; // rax
  _BYTE *v38; // rdx
  unsigned __int64 v39; // r13
  void **v40; // rdi
  __int64 v41; // rax
  __int64 v42; // rdx
  _BYTE *v43; // rax
  void **v44; // rdi
  __int64 v45; // rdi
  _BYTE *v46; // rax
  __int64 v47; // rax
  int v48; // eax
  unsigned int v49; // eax
  __int64 v50; // rsi
  __int64 v51; // rdx
  unsigned __int64 v52; // r9
  _QWORD *v53; // rax
  int v54; // eax
  unsigned int v55; // r13d
  unsigned int v56; // eax
  __int64 v57; // rsi
  unsigned int v58; // r13d
  __int64 v59; // rcx
  unsigned __int64 v60; // r12
  __int64 v61; // rax
  _BYTE *v62; // rdx
  unsigned __int64 v63; // r12
  void **v64; // rdi
  __int64 v65; // rax
  __int64 v66; // rdx
  _BYTE *v67; // rax
  _BYTE *v68; // rax
  void **v69; // rdi
  __int64 v70; // rdi
  _BYTE *v71; // rax
  __int64 v72; // rax
  int v73; // eax
  unsigned int v74; // eax
  __int64 v75; // rsi
  __int64 v76; // rdx
  unsigned __int64 v77; // r9
  _QWORD *v78; // rax
  int v79; // eax
  char v80; // al
  __int64 v81; // r13
  _QWORD *v82; // rdx
  void **v83; // rdi
  __int64 v84; // rdi
  _BYTE *v85; // rax
  unsigned __int64 v86; // rbx
  __int64 v87; // rax
  __int64 v88; // rdx
  __int16 v89; // ax
  _BYTE *v90; // rax
  __int64 v91; // rax
  _BYTE *v92; // rdx
  _QWORD *v93; // rcx
  __int64 v94; // rdx
  _QWORD *v95; // rax
  __int64 v96; // rsi
  void **v97; // rdi
  _BYTE *v98; // rdx
  void **v99; // r12
  _BYTE *v100; // rdx
  __int64 v101; // rax
  __int64 v102; // rax
  _BYTE *v103; // rdx
  __int64 v104; // r13
  void **v105; // rdi
  __int64 v106; // rax
  __int64 v107; // rdx
  __int64 v108; // rbx
  __int64 v109; // rcx
  int v110; // edx
  __int64 v111; // rax
  __int64 v112; // rax
  __int64 v113; // rax
  __int64 v114; // rax
  unsigned int v115; // esi
  int v116; // eax
  unsigned __int64 v117; // rax
  _QWORD *v118; // rax
  unsigned int v119; // esi
  int v120; // eax
  _QWORD *v121; // rax
  __int64 v122; // rax
  unsigned __int64 v123; // rax
  __int64 v124; // rdi
  _BYTE *v125; // rax
  unsigned __int64 v126; // rax
  unsigned __int64 v127; // rcx
  __int64 v128; // rax
  _QWORD *v129; // rax
  _QWORD *v130; // rdx
  void **v131; // r12
  __int64 v132; // [rsp+0h] [rbp-230h]
  __int64 v133; // [rsp+8h] [rbp-228h]
  __int64 v134; // [rsp+8h] [rbp-228h]
  __int64 v135; // [rsp+8h] [rbp-228h]
  unsigned __int64 v136; // [rsp+8h] [rbp-228h]
  __int64 v137; // [rsp+10h] [rbp-220h]
  __int64 v138; // [rsp+10h] [rbp-220h]
  unsigned __int64 v139; // [rsp+10h] [rbp-220h]
  __int64 v140; // [rsp+10h] [rbp-220h]
  unsigned __int64 v141; // [rsp+10h] [rbp-220h]
  unsigned __int64 v142; // [rsp+10h] [rbp-220h]
  __int64 v143; // [rsp+10h] [rbp-220h]
  __int64 v144; // [rsp+18h] [rbp-218h]
  __int64 v145; // [rsp+18h] [rbp-218h]
  unsigned __int64 v146; // [rsp+18h] [rbp-218h]
  __int64 v147; // [rsp+18h] [rbp-218h]
  unsigned __int64 v148; // [rsp+18h] [rbp-218h]
  __int64 v149; // [rsp+18h] [rbp-218h]
  __int64 v150; // [rsp+18h] [rbp-218h]
  __int64 v151; // [rsp+18h] [rbp-218h]
  __int64 v152; // [rsp+20h] [rbp-210h]
  __int64 v153; // [rsp+20h] [rbp-210h]
  __int64 v154; // [rsp+20h] [rbp-210h]
  __int64 v155; // [rsp+20h] [rbp-210h]
  __int64 v156; // [rsp+20h] [rbp-210h]
  __int64 v157; // [rsp+20h] [rbp-210h]
  __int64 v158; // [rsp+20h] [rbp-210h]
  __int64 v159; // [rsp+20h] [rbp-210h]
  unsigned int v161; // [rsp+38h] [rbp-1F8h]
  __int64 v162; // [rsp+38h] [rbp-1F8h]
  __int64 v163; // [rsp+38h] [rbp-1F8h]
  __int64 v164; // [rsp+38h] [rbp-1F8h]
  __int64 v165; // [rsp+38h] [rbp-1F8h]
  __int64 v166; // [rsp+38h] [rbp-1F8h]
  int v167; // [rsp+40h] [rbp-1F0h]
  __int64 v168; // [rsp+40h] [rbp-1F0h]
  __int64 v172; // [rsp+68h] [rbp-1C8h]
  __int64 v173; // [rsp+68h] [rbp-1C8h]
  __int64 v174; // [rsp+68h] [rbp-1C8h]
  char v175; // [rsp+7Fh] [rbp-1B1h] BYREF
  __m128i *v176; // [rsp+80h] [rbp-1B0h] BYREF
  __int64 v177; // [rsp+88h] [rbp-1A8h]
  __m128i v178; // [rsp+90h] [rbp-1A0h] BYREF
  _QWORD v179[2]; // [rsp+A0h] [rbp-190h] BYREF
  __m128i v180; // [rsp+B0h] [rbp-180h] BYREF
  void *v181; // [rsp+C0h] [rbp-170h] BYREF
  _BYTE *v182; // [rsp+C8h] [rbp-168h]
  _BYTE *v183; // [rsp+D0h] [rbp-160h]
  _BYTE *v184; // [rsp+D8h] [rbp-158h]
  int v185; // [rsp+E0h] [rbp-150h]
  __m128i **v186; // [rsp+E8h] [rbp-148h]
  __m128i *v187; // [rsp+F0h] [rbp-140h] BYREF
  size_t v188; // [rsp+F8h] [rbp-138h]
  __m128i v189; // [rsp+100h] [rbp-130h] BYREF
  char v190; // [rsp+110h] [rbp-120h]

  v12 = 8 * sub_15A9520(a3, 0);
  if ( v12 == 32 )
  {
    v13 = 5;
  }
  else if ( v12 > 0x20 )
  {
    v13 = 6;
    if ( v12 != 64 )
    {
      v13 = 0;
      if ( v12 == 128 )
        v13 = 7;
    }
  }
  else
  {
    v13 = 3;
    if ( v12 != 8 )
      v13 = 4 * (v12 == 16);
  }
  v175 = v13;
  if ( *(_DWORD *)(*(_QWORD *)(a2 + 81552) + 252LL) <= 0x13u )
  {
    a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
    sub_21CA7A0(a1->m128i_i64, byte_3F871B3, (__int64)byte_3F871B3);
    return a1;
  }
  v177 = 0;
  v186 = &v176;
  v181 = &unk_49EFBE0;
  v176 = &v178;
  v178.m128i_i8[0] = 0;
  v185 = 1;
  v184 = 0;
  v183 = 0;
  v182 = 0;
  v15 = sub_16E7EE0((__int64)&v181, "prototype_", 0xAu);
  v16 = sub_16E7A90(v15, *(unsigned int *)(a2 + 81560));
  v17 = *(__m128i **)(v16 + 24);
  if ( *(_QWORD *)(v16 + 16) - (_QWORD)v17 <= 0x11u )
  {
    sub_16E7EE0(v16, " : .callprototype ", 0x12u);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_435DC10);
    v17[1].m128i_i16[0] = 8293;
    *v17 = si128;
    *(_QWORD *)(v16 + 24) += 18LL;
  }
  if ( *(_BYTE *)(a4 + 8) )
  {
    if ( v183 == v184 )
      sub_16E7EE0((__int64)&v181, "(", 1u);
    else
      *v184++ = 40;
    v29 = *(_BYTE *)(a4 + 8);
    if ( (unsigned __int8)(v29 - 1) <= 5u )
      goto LABEL_32;
    if ( v29 == 11 )
    {
      if ( !sub_1642F90(a4, 128) )
      {
        if ( *(_BYTE *)(a4 + 8) == 11 )
        {
          v30 = *(_DWORD *)(a4 + 8) >> 8;
          goto LABEL_33;
        }
LABEL_32:
        v30 = sub_1643030(a4);
LABEL_33:
        v31 = 32;
        v32 = v184;
        if ( v30 >= 0x20 )
          v31 = v30;
        if ( (unsigned __int64)(v183 - v184) <= 8 )
        {
          v33 = (void **)sub_16E7EE0((__int64)&v181, ".param .b", 9u);
        }
        else
        {
          v184[8] = 98;
          v33 = &v181;
          *v32 = 0x2E206D617261702ELL;
          v184 += 9;
        }
        v34 = v31;
        goto LABEL_38;
      }
      v29 = *(_BYTE *)(a4 + 8);
    }
    if ( v29 != 15 )
    {
      if ( (unsigned int)v29 - 13 > 1 )
      {
LABEL_172:
        if ( v29 != 16 )
          sub_1642F90(a4, 128);
      }
      v101 = sub_15F2050(a10 & 0xFFFFFFFFFFFFFFF8LL);
      v102 = sub_1632FA0(v101);
      v103 = v184;
      v104 = v102;
      if ( (unsigned __int64)(v183 - v184) <= 0xD )
      {
        v105 = (void **)sub_16E7EE0((__int64)&v181, ".param .align ", 0xEu);
      }
      else
      {
        *((_DWORD *)v184 + 2) = 1734962273;
        v105 = &v181;
        *(_QWORD *)v103 = 0x2E206D617261702ELL;
        *((_WORD *)v103 + 6) = 8302;
        v184 += 14;
      }
      v106 = sub_16E7A90((__int64)v105, a7);
      v107 = *(_QWORD *)(v106 + 24);
      v108 = v106;
      if ( (unsigned __int64)(*(_QWORD *)(v106 + 16) - v107) <= 6 )
      {
        v108 = sub_16E7EE0(v106, " .b8 _[", 7u);
      }
      else
      {
        *(_DWORD *)v107 = 945958432;
        *(_WORD *)(v107 + 4) = 24352;
        *(_BYTE *)(v107 + 6) = 91;
        *(_QWORD *)(v106 + 24) += 7LL;
      }
      v109 = (unsigned int)sub_15A9FE0(v104, a4);
      v29 = *(_BYTE *)(a4 + 8);
      switch ( v110 )
      {
        case 0:
          v174 = v109;
          v129 = (_QWORD *)sub_15A9930(v104, a4);
          v127 = v174;
          v128 = 8LL * *v129;
          break;
        case 1:
          v168 = v109;
          v173 = *(_QWORD *)(a4 + 32);
          v126 = sub_12BE0A0(v104, *(_QWORD *)(a4 + 24));
          v127 = v168;
          v128 = 8 * v173 * v126;
          break;
        default:
          goto LABEL_172;
      }
      v124 = sub_16E7A90(v108, v127 * ((v127 + ((unsigned __int64)(v128 + 7) >> 3) - 1) / v127));
      v125 = *(_BYTE **)(v124 + 24);
      if ( *(_BYTE **)(v124 + 16) == v125 )
      {
        sub_16E7EE0(v124, "]", 1u);
      }
      else
      {
        *v125 = 93;
        ++*(_QWORD *)(v124 + 24);
      }
      goto LABEL_40;
    }
    v130 = v184;
    if ( (unsigned __int64)(v183 - v184) <= 8 )
    {
      v131 = (void **)sub_16E7EE0((__int64)&v181, ".param .b", 9u);
    }
    else
    {
      v184[8] = 98;
      v131 = &v181;
      *v130 = 0x2E206D617261702ELL;
      v184 += 9;
    }
    v33 = v131;
    v34 = (unsigned int)sub_1F3E310(&v175);
LABEL_38:
    v35 = sub_16E7A90((__int64)v33, v34);
    v36 = *(_WORD **)(v35 + 24);
    if ( *(_QWORD *)(v35 + 16) - (_QWORD)v36 <= 1u )
    {
      sub_16E7EE0(v35, " _", 2u);
    }
    else
    {
      *v36 = 24352;
      *(_QWORD *)(v35 + 24) += 2LL;
    }
LABEL_40:
    if ( (unsigned __int64)(v183 - v184) <= 1 )
    {
      sub_16E7EE0((__int64)&v181, ") ", 2u);
      v19 = v184;
    }
    else
    {
      *(_WORD *)v184 = 8233;
      v19 = v184 + 2;
      v184 += 2;
    }
    goto LABEL_17;
  }
  if ( (unsigned __int64)(v183 - v184) <= 1 )
  {
    sub_16E7EE0((__int64)&v181, "()", 2u);
    v19 = v184;
  }
  else
  {
    *(_WORD *)v184 = 10536;
    v19 = v184 + 2;
    v184 += 2;
  }
LABEL_17:
  if ( (unsigned __int64)(v183 - v19) <= 2 )
  {
    sub_16E7EE0((__int64)&v181, "_ (", 3u);
  }
  else
  {
    v19[2] = 40;
    *(_WORD *)v19 = 8287;
    v184 += 3;
  }
  v167 = -858993459 * ((__int64)(a5[1] - *a5) >> 3);
  if ( v167 )
  {
    v20 = 1;
    v21 = 0;
    v22 = 1;
    v172 = 0;
    while ( !*(_BYTE *)(a9 + 16) || *(_DWORD *)a9 > v20 - 1 )
    {
      v23 = *(_QWORD *)(*a5 + v172 + 24);
      if ( !v22 )
      {
        if ( (unsigned __int64)(v183 - v184) <= 1 )
        {
          sub_16E7EE0((__int64)&v181, ", ", 2u);
        }
        else
        {
          *(_WORD *)v184 = 8236;
          v184 += 2;
        }
      }
      v24 = *(_BYTE *)(v23 + 8);
      v25 = (_DWORD *)(*a6 + 48LL * v21);
      if ( (*(_BYTE *)v25 & 0x10) != 0 )
      {
        if ( v24 != 15 )
LABEL_234:
          BUG();
        v162 = *(_QWORD *)(v23 + 24);
        v55 = 1 << ((*v25 >> 15) & 0xF);
        v56 = sub_15A9FE0(a3, v162);
        v57 = v162;
        v58 = v55 >> 1;
        v59 = 1;
        v60 = v56;
        while ( 2 )
        {
          switch ( *(_BYTE *)(v57 + 8) )
          {
            case 1:
              v61 = 16;
              goto LABEL_75;
            case 2:
              v61 = 32;
              goto LABEL_75;
            case 3:
            case 9:
              v61 = 64;
              goto LABEL_75;
            case 4:
              v61 = 80;
              goto LABEL_75;
            case 5:
            case 6:
              v61 = 128;
              goto LABEL_75;
            case 7:
              v166 = v59;
              v79 = sub_15A9520(a3, 0);
              v59 = v166;
              v61 = (unsigned int)(8 * v79);
              goto LABEL_75;
            case 0xB:
              v61 = *(_DWORD *)(v57 + 8) >> 8;
              goto LABEL_75;
            case 0xD:
              v165 = v59;
              v78 = (_QWORD *)sub_15A9930(a3, v57);
              v59 = v165;
              v61 = 8LL * *v78;
              goto LABEL_75;
            case 0xE:
              v145 = v59;
              v156 = *(_QWORD *)(v57 + 24);
              v164 = *(_QWORD *)(v57 + 32);
              v74 = sub_15A9FE0(a3, v156);
              v75 = v156;
              v76 = 1;
              v59 = v145;
              v77 = v74;
              while ( 2 )
              {
                switch ( *(_BYTE *)(v75 + 8) )
                {
                  case 1:
                    v112 = 16;
                    goto LABEL_189;
                  case 2:
                    v112 = 32;
                    goto LABEL_189;
                  case 3:
                  case 9:
                    v112 = 64;
                    goto LABEL_189;
                  case 4:
                    v112 = 80;
                    goto LABEL_189;
                  case 5:
                  case 6:
                    v112 = 128;
                    goto LABEL_189;
                  case 7:
                    v138 = v145;
                    v115 = 0;
                    v146 = v77;
                    v157 = v76;
                    goto LABEL_198;
                  case 0xB:
                    v112 = *(_DWORD *)(v75 + 8) >> 8;
                    goto LABEL_189;
                  case 0xD:
                    v140 = v145;
                    v148 = v77;
                    v159 = v76;
                    v118 = (_QWORD *)sub_15A9930(a3, v75);
                    v76 = v159;
                    v77 = v148;
                    v59 = v140;
                    v112 = 8LL * *v118;
                    goto LABEL_189;
                  case 0xE:
                    v133 = v145;
                    v139 = v77;
                    v147 = v76;
                    v158 = *(_QWORD *)(v75 + 32);
                    v117 = sub_12BE0A0(a3, *(_QWORD *)(v75 + 24));
                    v76 = v147;
                    v77 = v139;
                    v59 = v133;
                    v112 = 8 * v158 * v117;
                    goto LABEL_189;
                  case 0xF:
                    v138 = v145;
                    v146 = v77;
                    v157 = v76;
                    v115 = *(_DWORD *)(v75 + 8) >> 8;
LABEL_198:
                    v116 = sub_15A9520(a3, v115);
                    v76 = v157;
                    v77 = v146;
                    v59 = v138;
                    v112 = (unsigned int)(8 * v116);
LABEL_189:
                    v61 = 8 * v77 * v164 * ((v77 + ((unsigned __int64)(v112 * v76 + 7) >> 3) - 1) / v77);
                    goto LABEL_75;
                  case 0x10:
                    v114 = *(_QWORD *)(v75 + 32);
                    v75 = *(_QWORD *)(v75 + 24);
                    v76 *= v114;
                    continue;
                  default:
                    goto LABEL_234;
                }
              }
            case 0xF:
              v163 = v59;
              v73 = sub_15A9520(a3, *(_DWORD *)(v57 + 8) >> 8);
              v59 = v163;
              v61 = (unsigned int)(8 * v73);
LABEL_75:
              v62 = v184;
              v63 = (v60 + ((unsigned __int64)(v59 * v61 + 7) >> 3) - 1) / v60 * v60;
              if ( (unsigned __int64)(v183 - v184) <= 0xD )
              {
                v64 = (void **)sub_16E7EE0((__int64)&v181, ".param .align ", 0xEu);
              }
              else
              {
                *((_DWORD *)v184 + 2) = 1734962273;
                v64 = &v181;
                *(_QWORD *)v62 = 0x2E206D617261702ELL;
                *((_WORD *)v62 + 6) = 8302;
                v184 += 14;
              }
              v65 = sub_16E7A90((__int64)v64, v58);
              v66 = *(_QWORD *)(v65 + 24);
              if ( (unsigned __int64)(*(_QWORD *)(v65 + 16) - v66) <= 4 )
              {
                sub_16E7EE0(v65, " .b8 ", 5u);
                v67 = v184;
                if ( v183 != v184 )
                {
LABEL_79:
                  *v67 = 95;
                  v68 = ++v184;
                  goto LABEL_80;
                }
              }
              else
              {
                *(_DWORD *)v66 = 945958432;
                *(_BYTE *)(v66 + 4) = 32;
                *(_QWORD *)(v65 + 24) += 5LL;
                v67 = v184;
                if ( v183 != v184 )
                  goto LABEL_79;
              }
              sub_16E7EE0((__int64)&v181, "_", 1u);
              v68 = v184;
LABEL_80:
              if ( v183 == v68 )
              {
                v69 = (void **)sub_16E7EE0((__int64)&v181, "[", 1u);
              }
              else
              {
                *v68 = 91;
                v69 = &v181;
                ++v184;
              }
              v70 = sub_16E7A90((__int64)v69, (unsigned int)v63);
              v71 = *(_BYTE **)(v70 + 24);
              if ( *(_BYTE **)(v70 + 16) == v71 )
              {
                sub_16E7EE0(v70, "]", 1u);
              }
              else
              {
                *v71 = 93;
                ++*(_QWORD *)(v70 + 24);
              }
              break;
            case 0x10:
              v72 = *(_QWORD *)(v57 + 32);
              v57 = *(_QWORD *)(v57 + 24);
              v59 *= v72;
              continue;
            default:
              goto LABEL_234;
          }
          break;
        }
      }
      else if ( (unsigned int)v24 - 13 <= 1 || v24 == 16 || sub_1642F90(v23, 128) )
      {
        v161 = sub_21CF790(a2, 0, 0, a10, v23, v20, a3);
        v26 = v23;
        v27 = (unsigned int)sub_15A9FE0(a3, v23);
        v28 = 1;
        while ( 2 )
        {
          switch ( *(_BYTE *)(v26 + 8) )
          {
            case 1:
              v37 = 16;
              goto LABEL_43;
            case 2:
              v37 = 32;
              goto LABEL_43;
            case 3:
            case 9:
              v37 = 64;
              goto LABEL_43;
            case 4:
              v37 = 80;
              goto LABEL_43;
            case 5:
            case 6:
              v37 = 128;
              goto LABEL_43;
            case 7:
              v155 = v28;
              v54 = sub_15A9520(a3, 0);
              v28 = v155;
              v37 = (unsigned int)(8 * v54);
              goto LABEL_43;
            case 0xB:
              v37 = *(_DWORD *)(v26 + 8) >> 8;
              goto LABEL_43;
            case 0xD:
              v154 = v28;
              v53 = (_QWORD *)sub_15A9930(a3, v26);
              v28 = v154;
              v37 = 8LL * *v53;
              goto LABEL_43;
            case 0xE:
              v137 = v28;
              v144 = *(_QWORD *)(v26 + 24);
              v153 = *(_QWORD *)(v26 + 32);
              v49 = sub_15A9FE0(a3, v144);
              v50 = v144;
              v28 = v137;
              v51 = 1;
              v52 = v49;
              while ( 2 )
              {
                switch ( *(_BYTE *)(v50 + 8) )
                {
                  case 1:
                    v113 = 16;
                    goto LABEL_192;
                  case 2:
                    v113 = 32;
                    goto LABEL_192;
                  case 3:
                  case 9:
                    v113 = 64;
                    goto LABEL_192;
                  case 4:
                    v113 = 80;
                    goto LABEL_192;
                  case 5:
                  case 6:
                    v113 = 128;
                    goto LABEL_192;
                  case 7:
                    v134 = v137;
                    v119 = 0;
                    v141 = v52;
                    v149 = v51;
                    goto LABEL_207;
                  case 0xB:
                    v113 = *(_DWORD *)(v50 + 8) >> 8;
                    goto LABEL_192;
                  case 0xD:
                    v135 = v137;
                    v142 = v52;
                    v150 = v51;
                    v121 = (_QWORD *)sub_15A9930(a3, v50);
                    v51 = v150;
                    v52 = v142;
                    v28 = v135;
                    v113 = 8LL * *v121;
                    goto LABEL_192;
                  case 0xE:
                    v132 = v137;
                    v136 = v52;
                    v143 = v51;
                    v151 = *(_QWORD *)(v50 + 32);
                    v123 = sub_12BE0A0(a3, *(_QWORD *)(v50 + 24));
                    v51 = v143;
                    v52 = v136;
                    v28 = v132;
                    v113 = 8 * v151 * v123;
                    goto LABEL_192;
                  case 0xF:
                    v134 = v137;
                    v141 = v52;
                    v149 = v51;
                    v119 = *(_DWORD *)(v50 + 8) >> 8;
LABEL_207:
                    v120 = sub_15A9520(a3, v119);
                    v51 = v149;
                    v52 = v141;
                    v28 = v134;
                    v113 = (unsigned int)(8 * v120);
LABEL_192:
                    v37 = 8 * v52 * v153 * ((v52 + ((unsigned __int64)(v113 * v51 + 7) >> 3) - 1) / v52);
                    goto LABEL_43;
                  case 0x10:
                    v122 = *(_QWORD *)(v50 + 32);
                    v50 = *(_QWORD *)(v50 + 24);
                    v51 *= v122;
                    continue;
                  default:
                    goto LABEL_234;
                }
              }
            case 0xF:
              v152 = v28;
              v48 = sub_15A9520(a3, *(_DWORD *)(v26 + 8) >> 8);
              v28 = v152;
              v37 = (unsigned int)(8 * v48);
LABEL_43:
              v38 = v184;
              v39 = (v27 + ((unsigned __int64)(v37 * v28 + 7) >> 3) - 1) / v27 * v27;
              if ( (unsigned __int64)(v183 - v184) <= 0xD )
              {
                v40 = (void **)sub_16E7EE0((__int64)&v181, ".param .align ", 0xEu);
              }
              else
              {
                *((_DWORD *)v184 + 2) = 1734962273;
                *((_WORD *)v38 + 6) = 8302;
                v40 = &v181;
                *(_QWORD *)v38 = 0x2E206D617261702ELL;
                v184 += 14;
              }
              v41 = sub_16E7A90((__int64)v40, v161);
              v42 = *(_QWORD *)(v41 + 24);
              if ( (unsigned __int64)(*(_QWORD *)(v41 + 16) - v42) <= 4 )
              {
                sub_16E7EE0(v41, " .b8 ", 5u);
              }
              else
              {
                *(_DWORD *)v42 = 945958432;
                *(_BYTE *)(v42 + 4) = 32;
                *(_QWORD *)(v41 + 24) += 5LL;
              }
              if ( v183 == v184 )
              {
                sub_16E7EE0((__int64)&v181, "_", 1u);
                v43 = v184;
                if ( v183 != v184 )
                {
LABEL_49:
                  *v43 = 91;
                  v44 = &v181;
                  ++v184;
                  goto LABEL_50;
                }
              }
              else
              {
                *v184 = 95;
                v43 = v184 + 1;
                v184 = v43;
                if ( v183 != v43 )
                  goto LABEL_49;
              }
              v44 = (void **)sub_16E7EE0((__int64)&v181, "[", 1u);
LABEL_50:
              v45 = sub_16E7A90((__int64)v44, (unsigned int)v39);
              v46 = *(_BYTE **)(v45 + 24);
              if ( *(_BYTE **)(v45 + 16) == v46 )
              {
                sub_16E7EE0(v45, "]", 1u);
              }
              else
              {
                *v46 = 93;
                ++*(_QWORD *)(v45 + 24);
              }
              v188 = 0x1000000000LL;
              v187 = &v189;
              sub_20C7CE0(a2, a3, v23, (__int64)&v187, 0, 0);
              if ( (_DWORD)v188 )
                v21 = v21 + v188 - 1;
              if ( v187 != &v189 )
                _libc_free((unsigned __int64)v187);
              break;
            case 0x10:
              v47 = *(_QWORD *)(v26 + 32);
              v26 = *(_QWORD *)(v26 + 24);
              v28 *= v47;
              continue;
            default:
              goto LABEL_234;
          }
          break;
        }
      }
      else
      {
        v80 = *(_BYTE *)(v23 + 8);
        if ( v80 == 11 )
        {
          v81 = *(_DWORD *)(v23 + 8) >> 8;
          if ( *(_DWORD *)(v23 + 8) <= 0x1FFFu )
            v81 = 32;
        }
        else if ( v80 == 15 )
        {
          v81 = (unsigned int)sub_1F3E310(&v175);
        }
        else
        {
          v81 = 32;
          if ( v80 != 1 )
            v81 = (unsigned int)sub_1643030(v23);
        }
        v82 = v184;
        if ( (unsigned __int64)(v183 - v184) <= 8 )
        {
          v83 = (void **)sub_16E7EE0((__int64)&v181, ".param .b", 9u);
        }
        else
        {
          v184[8] = 98;
          v83 = &v181;
          *v82 = 0x2E206D617261702ELL;
          v184 += 9;
        }
        v84 = sub_16E7A90((__int64)v83, v81);
        v85 = *(_BYTE **)(v84 + 24);
        if ( *(_BYTE **)(v84 + 16) == v85 )
        {
          sub_16E7EE0(v84, " ", 1u);
        }
        else
        {
          *v85 = 32;
          ++*(_QWORD *)(v84 + 24);
        }
        if ( v183 == v184 )
          sub_16E7EE0((__int64)&v181, "_", 1u);
        else
          *v184++ = 95;
      }
      v172 += 40;
      ++v21;
      v22 = 0;
      if ( v167 == v20 )
      {
        if ( !*(_BYTE *)(a9 + 16) )
          goto LABEL_120;
        goto LABEL_180;
      }
      ++v20;
    }
    if ( v22 )
      goto LABEL_156;
LABEL_180:
    if ( v184 == v183 )
    {
      v111 = sub_16E7EE0((__int64)&v181, ",", 1u);
      v98 = *(_BYTE **)(v111 + 24);
      v99 = (void **)v111;
    }
    else
    {
      *v184 = 44;
      v99 = &v181;
      v98 = ++v184;
    }
    goto LABEL_157;
  }
  if ( *(_BYTE *)(a9 + 16) )
  {
LABEL_156:
    v98 = v184;
    v99 = &v181;
LABEL_157:
    if ( (unsigned __int64)((_BYTE *)v99[2] - v98) <= 0xE )
    {
      v99 = (void **)sub_16E7EE0((__int64)v99, " .param .align ", 0xFu);
    }
    else
    {
      qmemcpy(v98, " .param .align ", 15);
      v99[3] = (char *)v99[3] + 15;
    }
    sub_16A95F0(*(_QWORD *)(a9 + 8), (__int64)v99, 1);
    v100 = v99[3];
    if ( (unsigned __int64)((_BYTE *)v99[2] - v100) <= 8 )
    {
      sub_16E7EE0((__int64)v99, " .b8 _[]\n", 9u);
    }
    else
    {
      v100[8] = 10;
      *(_QWORD *)v100 = 0x5D5B5F2038622E20LL;
      v99[3] = (char *)v99[3] + 9;
    }
  }
LABEL_120:
  if ( v183 == v184 )
    sub_16E7EE0((__int64)&v181, ")", 1u);
  else
    *v184++ = 41;
  v86 = a10 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (unsigned __int8)sub_1560260((_QWORD *)((a10 & 0xFFFFFFFFFFFFFFF8LL) + 56), -1, 29)
    || (v87 = *(_QWORD *)(v86 - 24), !*(_BYTE *)(v87 + 16))
    && (v187 = *(__m128i **)(v87 + 112), (unsigned __int8)sub_1560260(&v187, -1, 29)) )
  {
    if ( !*(_BYTE *)(**(_QWORD **)(*(_QWORD *)(v86 + 64) + 16LL) + 8LL) )
    {
      v92 = v184;
      if ( (unsigned __int64)(v183 - v184) <= 9 )
      {
        sub_16E7EE0((__int64)&v181, ".noreturn ", 0xAu);
      }
      else
      {
        *(_QWORD *)v184 = 0x72757465726F6E2ELL;
        *((_WORD *)v92 + 4) = 8302;
        v184 += 10;
      }
    }
  }
  v88 = *a8;
  v89 = *(_WORD *)(*a8 + 24);
  if ( v89 >= 0 )
  {
LABEL_128:
    if ( v89 == 43 )
    {
      v93 = *(_QWORD **)(v88 + 32);
      v94 = *(_QWORD *)(*v93 + 88LL);
      v95 = *(_QWORD **)(v94 + 24);
      if ( *(_DWORD *)(v94 + 32) > 0x40u )
        v95 = (_QWORD *)*v95;
      if ( (_DWORD)v95 == 3760 )
      {
        v96 = *(_QWORD *)(v93[10] + 88LL);
        if ( v96 )
        {
          sub_3937240(v179, v96, 0);
          v190 = 1;
          v187 = &v189;
          if ( (__m128i *)v179[0] == &v180 )
          {
            v189 = _mm_load_si128(&v180);
          }
          else
          {
            v187 = (__m128i *)v179[0];
            v189.m128i_i64[0] = v180.m128i_i64[0];
          }
          v188 = v179[1];
          if ( v183 == v184 )
          {
            v97 = (void **)sub_16E7EE0((__int64)&v181, " ", 1u);
          }
          else
          {
            *v184 = 32;
            v97 = &v181;
            ++v184;
          }
          sub_16E7EE0((__int64)v97, v187->m128i_i8, v188);
          if ( v190 )
          {
            if ( v187 != &v189 )
              j_j___libc_free_0(v187, v189.m128i_i64[0] + 1);
          }
        }
      }
    }
  }
  else
  {
    while ( v89 == -16 )
    {
      v88 = **(_QWORD **)(v88 + 32);
      v89 = *(_WORD *)(v88 + 24);
      if ( v89 >= 0 )
        goto LABEL_128;
    }
  }
  if ( v183 == v184 )
  {
    sub_16E7EE0((__int64)&v181, ";", 1u);
    v90 = v184;
  }
  else
  {
    *v184 = 59;
    v90 = ++v184;
  }
  if ( v182 != v90 )
    sub_16E7BA0((__int64 *)&v181);
  a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
  if ( v176 == &v178 )
  {
    a1[1] = _mm_load_si128(&v178);
  }
  else
  {
    a1->m128i_i64[0] = (__int64)v176;
    a1[1].m128i_i64[0] = v178.m128i_i64[0];
  }
  v91 = v177;
  v177 = 0;
  v178.m128i_i8[0] = 0;
  a1->m128i_i64[1] = v91;
  v176 = &v178;
  sub_16E7BC0((__int64 *)&v181);
  if ( v176 != &v178 )
    j_j___libc_free_0(v176, v178.m128i_i64[0] + 1);
  return a1;
}
