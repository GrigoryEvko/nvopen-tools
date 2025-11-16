// Function: sub_2A59A40
// Address: 0x2a59a40
//
void __fastcall sub_2A59A40(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rcx
  __int64 v7; // r12
  __int64 v9; // r8
  char v10; // di
  __int64 v11; // r15
  __int64 v12; // rbx
  __int64 v13; // rdx
  __int64 v14; // rsi
  _QWORD *v15; // rax
  _QWORD *v16; // r13
  _QWORD *v17; // rbx
  __int64 *v18; // rdi
  signed __int64 v19; // r15
  int v20; // eax
  unsigned int v21; // eax
  __int64 v22; // rdx
  __int64 v23; // r13
  _QWORD *v24; // rax
  __int64 v25; // rdx
  __int64 v26; // r13
  __int64 v27; // r14
  __int64 v28; // rax
  const char *v29; // rsi
  __int64 v30; // r8
  const char *v31; // r14
  unsigned __int64 v32; // rax
  int v33; // ecx
  __int64 *v34; // rdx
  __int64 v35; // r15
  __int64 v36; // rax
  __int64 v37; // r14
  char v38; // dl
  __int64 v39; // rax
  __int64 v40; // rdx
  __int64 *v41; // r12
  __int64 *v42; // r15
  __int64 v43; // rbx
  __int64 *v44; // rax
  __int64 *v45; // rdx
  __int64 v46; // rax
  unsigned __int64 v47; // rdx
  __int64 v48; // rbx
  int v49; // edx
  unsigned int v50; // ecx
  unsigned __int8 v51; // al
  __int64 *v52; // rax
  int v53; // r15d
  unsigned __int64 v54; // r15
  __int64 *v55; // rbx
  __int64 v56; // rdx
  unsigned int v57; // esi
  unsigned int v58; // esi
  __int64 v59; // r9
  __int64 v60; // r8
  int v61; // r10d
  _QWORD *v62; // rdx
  unsigned int v63; // edi
  _QWORD *v64; // rax
  __int64 v65; // rcx
  __int64 *v66; // rax
  __int64 v67; // rax
  unsigned __int64 v68; // rdx
  __int64 v69; // rax
  __int64 v70; // rbx
  __int64 *v71; // r13
  __int64 v72; // r15
  int v73; // eax
  __int64 v74; // rax
  __int64 v75; // rbx
  __int64 v76; // rdx
  int v77; // eax
  __int64 v78; // r8
  __int64 **v79; // rbx
  char v80; // al
  __int64 **v81; // r15
  __int64 v82; // rax
  __int64 v83; // rsi
  __int64 v84; // rax
  __int64 v85; // rdi
  __int64 *v86; // r13
  __int64 **v87; // rax
  _QWORD *v88; // rdx
  _QWORD *v89; // rax
  __int64 v90; // rdx
  _QWORD *v91; // rax
  int v92; // eax
  int v93; // eax
  int v94; // esi
  int v95; // esi
  unsigned int v96; // ecx
  __int64 v97; // rdi
  int v98; // ebx
  int v99; // ecx
  int v100; // ecx
  _QWORD *v101; // rdi
  unsigned int v102; // ebx
  int v103; // r11d
  __int64 v104; // rsi
  __int64 *v105; // rax
  __int64 *v106; // r13
  __int64 v107; // rsi
  __int64 *v108; // rbx
  char v109; // di
  __int64 *v110; // rax
  __int64 *v111; // rdx
  unsigned __int64 v112; // r9
  unsigned __int64 v113; // rbx
  __int64 v114; // [rsp+8h] [rbp-658h]
  __int64 v115; // [rsp+30h] [rbp-630h]
  __int64 v116; // [rsp+40h] [rbp-620h]
  __int64 *v117; // [rsp+70h] [rbp-5F0h]
  __int64 v118; // [rsp+78h] [rbp-5E8h]
  _BYTE *v120; // [rsp+90h] [rbp-5D0h]
  __int64 v121; // [rsp+A0h] [rbp-5C0h]
  unsigned __int64 v122; // [rsp+A8h] [rbp-5B8h]
  __int64 v123; // [rsp+A8h] [rbp-5B8h]
  __int64 *v125; // [rsp+B8h] [rbp-5A8h]
  __int64 v126; // [rsp+B8h] [rbp-5A8h]
  __int64 *v127; // [rsp+B8h] [rbp-5A8h]
  _QWORD v128[2]; // [rsp+D0h] [rbp-590h] BYREF
  char v129; // [rsp+E0h] [rbp-580h]
  __int64 *v130; // [rsp+E8h] [rbp-578h]
  __int64 *v131; // [rsp+F0h] [rbp-570h]
  _QWORD v132[4]; // [rsp+100h] [rbp-560h] BYREF
  __int16 v133; // [rsp+120h] [rbp-540h]
  const char *v134[4]; // [rsp+130h] [rbp-530h] BYREF
  __int16 v135; // [rsp+150h] [rbp-510h]
  __int64 v136; // [rsp+160h] [rbp-500h] BYREF
  __int64 *v137; // [rsp+168h] [rbp-4F8h]
  __int64 v138; // [rsp+170h] [rbp-4F0h]
  int v139; // [rsp+178h] [rbp-4E8h]
  char v140; // [rsp+17Ch] [rbp-4E4h]
  char v141; // [rsp+180h] [rbp-4E0h] BYREF
  __int64 v142; // [rsp+190h] [rbp-4D0h] BYREF
  _BYTE *v143; // [rsp+198h] [rbp-4C8h]
  __int64 v144; // [rsp+1A0h] [rbp-4C0h]
  int v145; // [rsp+1A8h] [rbp-4B8h]
  char v146; // [rsp+1ACh] [rbp-4B4h]
  _BYTE v147[16]; // [rsp+1B0h] [rbp-4B0h] BYREF
  _BYTE *v148; // [rsp+1C0h] [rbp-4A0h] BYREF
  __int64 v149; // [rsp+1C8h] [rbp-498h]
  _BYTE v150[32]; // [rsp+1D0h] [rbp-490h] BYREF
  __int64 *v151; // [rsp+1F0h] [rbp-470h] BYREF
  __int64 v152; // [rsp+1F8h] [rbp-468h]
  _BYTE v153[256]; // [rsp+200h] [rbp-460h] BYREF
  __int64 v154; // [rsp+300h] [rbp-360h] BYREF
  _BYTE *v155; // [rsp+308h] [rbp-358h]
  __int64 v156; // [rsp+310h] [rbp-350h]
  int v157; // [rsp+318h] [rbp-348h]
  char v158; // [rsp+31Ch] [rbp-344h]
  _BYTE v159[256]; // [rsp+320h] [rbp-340h] BYREF
  __int64 *v160; // [rsp+420h] [rbp-240h] BYREF
  unsigned __int64 v161; // [rsp+428h] [rbp-238h]
  __int64 v162; // [rsp+430h] [rbp-230h] BYREF
  int v163; // [rsp+438h] [rbp-228h]
  char v164; // [rsp+43Ch] [rbp-224h]
  char v165; // [rsp+440h] [rbp-220h] BYREF
  __int64 v166; // [rsp+450h] [rbp-210h]
  __int64 v167; // [rsp+458h] [rbp-208h]
  __int64 v168; // [rsp+460h] [rbp-200h]
  __int64 v169; // [rsp+468h] [rbp-1F8h]
  void **v170; // [rsp+470h] [rbp-1F0h]
  void **v171; // [rsp+478h] [rbp-1E8h]
  __int64 v172; // [rsp+480h] [rbp-1E0h]
  int v173; // [rsp+488h] [rbp-1D8h]
  __int16 v174; // [rsp+48Ch] [rbp-1D4h]
  char v175; // [rsp+48Eh] [rbp-1D2h]
  __int64 v176; // [rsp+490h] [rbp-1D0h]
  __int64 v177; // [rsp+498h] [rbp-1C8h]
  void *v178; // [rsp+4A0h] [rbp-1C0h] BYREF
  void *v179; // [rsp+4A8h] [rbp-1B8h] BYREF

  v6 = *(_QWORD *)a1 + 104LL * *(unsigned int *)(a1 + 8);
  v115 = v6;
  if ( *(_QWORD *)a1 == v6 )
    return;
  v7 = *(_QWORD *)a1;
  v118 = a1 + 432;
  do
  {
    v128[0] = a2;
    v128[1] = 0;
    v129 = 0;
    v136 = 0;
    v137 = (__int64 *)&v141;
    v138 = 2;
    v139 = 0;
    v140 = 1;
    v9 = *(unsigned int *)(v7 + 16);
    if ( (_DWORD)v9 )
    {
      v105 = *(__int64 **)(v7 + 8);
      v106 = &v105[2 * *(unsigned int *)(v7 + 24)];
      if ( v105 != v106 )
      {
        while ( 1 )
        {
          v107 = *v105;
          v108 = v105;
          if ( *v105 != -8192 && v107 != -4096 )
            break;
          v105 += 2;
          if ( v106 == v105 )
            goto LABEL_4;
        }
        if ( v106 != v105 )
        {
          v109 = 1;
LABEL_218:
          v110 = v137;
          v6 = HIDWORD(v138);
          v111 = &v137[HIDWORD(v138)];
          if ( v137 == v111 )
          {
LABEL_230:
            if ( HIDWORD(v138) < (unsigned int)v138 )
            {
              v6 = (unsigned int)++HIDWORD(v138);
              *v111 = v107;
              v109 = v140;
              ++v136;
              goto LABEL_222;
            }
            goto LABEL_229;
          }
          while ( *v110 != v107 )
          {
            if ( v111 == ++v110 )
              goto LABEL_230;
          }
LABEL_222:
          while ( 1 )
          {
            v108 += 2;
            if ( v108 == v106 )
              break;
            while ( 1 )
            {
              v107 = *v108;
              if ( *v108 != -4096 && v107 != -8192 )
                break;
              v108 += 2;
              if ( v106 == v108 )
                goto LABEL_4;
            }
            if ( v106 == v108 )
              break;
            if ( v109 )
              goto LABEL_218;
LABEL_229:
            sub_C8CC70((__int64)&v136, v107, (__int64)v111, v6, v9, a6);
            v109 = v140;
          }
        }
      }
    }
LABEL_4:
    v10 = 1;
    v142 = 0;
    v144 = 2;
    v131 = &v136;
    v145 = 0;
    v143 = v147;
    v146 = 1;
    v11 = *(_QWORD *)(v7 + 32);
    v12 = v11 + 8LL * *(unsigned int *)(v7 + 40);
    if ( v12 == v11 )
    {
      v154 = 0;
      v158 = 1;
      v16 = v147;
      v151 = (__int64 *)v153;
      v152 = 0x2000000000LL;
      v156 = 32;
      v155 = v159;
      v157 = 0;
      goto LABEL_196;
    }
    do
    {
      while ( 1 )
      {
        v13 = *(_QWORD *)(*(_QWORD *)v11 + 24LL);
        if ( *(_BYTE *)v13 == 84 )
        {
          v6 = *(_QWORD *)(v13 - 8);
          v13 = 32LL * *(unsigned int *)(v13 + 72);
          v14 = *(_QWORD *)(v6 + v13 + 8LL * (unsigned int)((*(_QWORD *)v11 - v6) >> 5));
        }
        else
        {
          v14 = *(_QWORD *)(v13 + 40);
        }
        if ( !v10 )
          goto LABEL_62;
        v6 = (__int64)v143;
        v13 = (__int64)&v143[8 * HIDWORD(v144)];
        v9 = HIDWORD(v144);
        if ( v143 != (_BYTE *)v13 )
          break;
LABEL_67:
        if ( HIDWORD(v144) >= (unsigned int)v144 )
        {
LABEL_62:
          v11 += 8;
          sub_C8CC70((__int64)&v142, v14, v13, v6, v9, a6);
          v10 = v146;
          v6 = (__int64)v143;
          if ( v12 == v11 )
            goto LABEL_13;
        }
        else
        {
          v9 = (unsigned int)(HIDWORD(v144) + 1);
          v11 += 8;
          ++HIDWORD(v144);
          *(_QWORD *)v13 = v14;
          v6 = (__int64)v143;
          ++v142;
          v10 = v146;
          if ( v12 == v11 )
            goto LABEL_13;
        }
      }
      v15 = v143;
      while ( v14 != *v15 )
      {
        if ( (_QWORD *)v13 == ++v15 )
          goto LABEL_67;
      }
      v11 += 8;
    }
    while ( v12 != v11 );
LABEL_13:
    v16 = (_QWORD *)v6;
    v154 = 0;
    v156 = 32;
    v151 = (__int64 *)v153;
    v152 = 0x2000000000LL;
    v157 = 0;
    v155 = v159;
    v158 = 1;
    if ( !v10 )
    {
      v17 = (_QWORD *)(v6 + 8LL * (unsigned int)v144);
      goto LABEL_15;
    }
LABEL_196:
    v17 = &v16[HIDWORD(v144)];
LABEL_15:
    if ( v16 == v17 )
    {
LABEL_18:
      HIDWORD(v161) = 64;
      v160 = &v162;
LABEL_19:
      v18 = &v162;
      LODWORD(v19) = 0;
      v20 = 0;
      goto LABEL_20;
    }
    while ( *v16 >= 0xFFFFFFFFFFFFFFFELL )
    {
      if ( ++v16 == v17 )
        goto LABEL_18;
    }
    v160 = &v162;
    v161 = 0x4000000000LL;
    if ( v16 == v17 )
      goto LABEL_19;
    v88 = v16;
    v19 = 0;
    while ( 1 )
    {
      v89 = v88 + 1;
      if ( v88 + 1 == v17 )
        break;
      while ( 1 )
      {
        v88 = v89;
        if ( *v89 < 0xFFFFFFFFFFFFFFFELL )
          break;
        if ( v17 == ++v89 )
          goto LABEL_170;
      }
      ++v19;
      if ( v89 == v17 )
        goto LABEL_171;
    }
LABEL_170:
    ++v19;
LABEL_171:
    v6 = (__int64)&v162;
    if ( v19 > 64 )
    {
      sub_C8D5F0((__int64)&v160, &v162, v19, 8u, v9, a6);
      v6 = (__int64)&v160[(unsigned int)v161];
    }
    v90 = *v16;
    do
    {
      v91 = v16 + 1;
      *(_QWORD *)v6 = v90;
      v6 += 8;
      if ( v16 + 1 == v17 )
        break;
      while ( 1 )
      {
        v90 = *v91;
        v16 = v91;
        if ( *v91 < 0xFFFFFFFFFFFFFFFELL )
          break;
        if ( v17 == ++v91 )
          goto LABEL_177;
      }
    }
    while ( v91 != v17 );
LABEL_177:
    v20 = v161;
    v18 = v160;
LABEL_20:
    v21 = v20 + v19;
    LODWORD(v161) = v21;
    if ( !v21 )
      goto LABEL_28;
    while ( 2 )
    {
      v22 = v21;
      v23 = v18[v21 - 1];
      LODWORD(v161) = v21 - 1;
      if ( !v158 )
        goto LABEL_47;
      v24 = v155;
      v6 = HIDWORD(v156);
      v22 = (__int64)&v155[8 * HIDWORD(v156)];
      if ( v155 != (_BYTE *)v22 )
      {
        while ( v23 != *v24 )
        {
          if ( (_QWORD *)v22 == ++v24 )
            goto LABEL_65;
        }
LABEL_26:
        v21 = v161;
        v18 = v160;
        goto LABEL_27;
      }
LABEL_65:
      if ( HIDWORD(v156) < (unsigned int)v156 )
      {
        ++HIDWORD(v156);
        *(_QWORD *)v22 = v23;
        ++v154;
      }
      else
      {
LABEL_47:
        sub_C8CC70((__int64)&v154, v23, v22, v6, v9, a6);
        if ( !v38 )
          goto LABEL_26;
      }
      v39 = sub_102DBD0(v118, v23);
      if ( v39 == v39 + 8 * v40 )
        goto LABEL_26;
      v126 = v7;
      v41 = (__int64 *)(v39 + 8 * v40);
      v42 = (__int64 *)v39;
      while ( 2 )
      {
        while ( 2 )
        {
          v43 = *v42;
          if ( v140 )
          {
            v44 = v137;
            v45 = &v137[HIDWORD(v138)];
            if ( v137 == v45 )
              break;
            while ( v43 != *v44 )
            {
              if ( v45 == ++v44 )
                goto LABEL_58;
            }
            goto LABEL_55;
          }
          if ( sub_C8CA60((__int64)&v136, v43) )
          {
LABEL_55:
            if ( v41 == ++v42 )
              goto LABEL_56;
            continue;
          }
          break;
        }
LABEL_58:
        v46 = (unsigned int)v161;
        v6 = HIDWORD(v161);
        v47 = (unsigned int)v161 + 1LL;
        if ( v47 > HIDWORD(v161) )
        {
          sub_C8D5F0((__int64)&v160, &v162, v47, 8u, v9, a6);
          v46 = (unsigned int)v161;
        }
        ++v42;
        v160[v46] = v43;
        LODWORD(v161) = v161 + 1;
        if ( v41 != v42 )
          continue;
        break;
      }
LABEL_56:
      v7 = v126;
      v21 = v161;
      v18 = v160;
LABEL_27:
      if ( v21 )
        continue;
      break;
    }
LABEL_28:
    if ( v18 != &v162 )
      _libc_free((unsigned __int64)v18);
    v129 = 1;
    v130 = &v154;
    sub_D6A180((__int64)v128, (__int64)&v151);
    v25 = (unsigned int)v152;
    v148 = v150;
    v149 = 0x400000000LL;
    v117 = &v151[(unsigned int)v152];
    if ( v117 != v151 )
    {
      v125 = v151;
      v116 = a2;
      while ( 1 )
      {
        v26 = *v125;
        v27 = *(_QWORD *)(*v125 + 56);
        v28 = sub_AA48A0(*v125);
        v172 = 0;
        v169 = v28;
        v160 = &v162;
        v170 = &v178;
        v161 = 0x200000000LL;
        v171 = &v179;
        v173 = 0;
        v178 = &unk_49DA100;
        v174 = 512;
        v175 = 7;
        v179 = &unk_49DA0B0;
        LOWORD(v168) = 1;
        v176 = 0;
        v177 = 0;
        v166 = v26;
        v167 = v27;
        if ( v27 == v26 + 48 )
          goto LABEL_43;
        if ( v27 )
          v27 -= 24;
        v29 = *(const char **)sub_B46C60(v27);
        v134[0] = v29;
        if ( v29 && (sub_B96E90((__int64)v134, (__int64)v29, 1), (v31 = v134[0]) != 0) )
        {
          v32 = (unsigned __int64)v160;
          v33 = v161;
          v34 = &v160[2 * (unsigned int)v161];
          if ( v160 != v34 )
          {
            while ( *(_DWORD *)v32 )
            {
              v32 += 16LL;
              if ( v34 == (__int64 *)v32 )
                goto LABEL_144;
            }
            *(const char **)(v32 + 8) = v134[0];
LABEL_42:
            sub_B91220((__int64)v134, (__int64)v31);
            goto LABEL_43;
          }
LABEL_144:
          if ( (unsigned int)v161 >= (unsigned __int64)HIDWORD(v161) )
          {
            v112 = (unsigned int)v161 + 1LL;
            v113 = v114 & 0xFFFFFFFF00000000LL;
            v114 &= 0xFFFFFFFF00000000LL;
            if ( HIDWORD(v161) < v112 )
            {
              sub_C8D5F0((__int64)&v160, &v162, v112, 0x10u, v30, v112);
              v34 = &v160[2 * (unsigned int)v161];
            }
            *v34 = v113;
            v34[1] = (__int64)v31;
            v31 = v134[0];
            LODWORD(v161) = v161 + 1;
          }
          else
          {
            if ( v34 )
            {
              *(_DWORD *)v34 = 0;
              v34[1] = (__int64)v31;
              v33 = v161;
              v31 = v134[0];
            }
            LODWORD(v161) = v33 + 1;
          }
        }
        else
        {
          sub_93FB40((__int64)&v160, 0);
          v31 = v134[0];
        }
        if ( v31 )
          goto LABEL_42;
LABEL_43:
        v133 = 261;
        v132[0] = *(_QWORD *)(v7 + 80);
        v132[1] = *(_QWORD *)(v7 + 88);
        v35 = *(_QWORD *)(v7 + 96);
        v135 = 257;
        v36 = sub_BD2DA0(80);
        v37 = v36;
        if ( v36 )
        {
          sub_B44260(v36, v35, 55, 0x8000000u, 0, 0);
          *(_DWORD *)(v37 + 72) = 0;
          sub_BD6B50((unsigned __int8 *)v37, v134);
          sub_BD2A10(v37, *(_DWORD *)(v37 + 72), 1);
        }
        if ( *(_BYTE *)v37 > 0x1Cu )
        {
          switch ( *(_BYTE *)v37 )
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
              goto LABEL_84;
            case 'T':
            case 'U':
            case 'V':
              v48 = *(_QWORD *)(v37 + 8);
              v49 = *(unsigned __int8 *)(v48 + 8);
              v50 = v49 - 17;
              v51 = *(_BYTE *)(v48 + 8);
              if ( (unsigned int)(v49 - 17) <= 1 )
                v51 = *(_BYTE *)(**(_QWORD **)(v48 + 16) + 8LL);
              if ( v51 <= 3u || v51 == 5 || (v51 & 0xFD) == 4 )
                goto LABEL_84;
              if ( (_BYTE)v49 == 15 )
              {
                if ( (*(_BYTE *)(v48 + 9) & 4) == 0 || !sub_BCB420(*(_QWORD *)(v37 + 8)) )
                  break;
                v52 = *(__int64 **)(v48 + 16);
                v48 = *v52;
                v49 = *(unsigned __int8 *)(*v52 + 8);
                v50 = v49 - 17;
              }
              else if ( (_BYTE)v49 == 16 )
              {
                do
                {
                  v48 = *(_QWORD *)(v48 + 24);
                  LOBYTE(v49) = *(_BYTE *)(v48 + 8);
                }
                while ( (_BYTE)v49 == 16 );
                v50 = (unsigned __int8)v49 - 17;
              }
              if ( v50 <= 1 )
                LOBYTE(v49) = *(_BYTE *)(**(_QWORD **)(v48 + 16) + 8LL);
              if ( (unsigned __int8)v49 <= 3u || (_BYTE)v49 == 5 || (v49 & 0xFD) == 4 )
              {
LABEL_84:
                v53 = v173;
                if ( v172 )
                  sub_B99FD0(v37, 3u, v172);
                sub_B45150(v37, v53);
              }
              break;
            default:
              break;
          }
        }
        (*((void (__fastcall **)(void **, __int64, _QWORD *, __int64, __int64))*v171 + 2))(v171, v37, v132, v167, v168);
        v54 = (unsigned __int64)v160;
        v55 = &v160[2 * (unsigned int)v161];
        if ( v160 != v55 )
        {
          do
          {
            v56 = *(_QWORD *)(v54 + 8);
            v57 = *(_DWORD *)v54;
            v54 += 16LL;
            sub_B99FD0(v37, v57, v56);
          }
          while ( v55 != (__int64 *)v54 );
        }
        v58 = *(_DWORD *)(v7 + 24);
        if ( !v58 )
        {
          ++*(_QWORD *)v7;
          goto LABEL_198;
        }
        v59 = v58 - 1;
        v60 = *(_QWORD *)(v7 + 8);
        v61 = 1;
        v62 = 0;
        v63 = v59 & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
        v64 = (_QWORD *)(v60 + 16LL * v63);
        v65 = *v64;
        if ( v26 != *v64 )
        {
          while ( v65 != -4096 )
          {
            if ( !v62 && v65 == -8192 )
              v62 = v64;
            v63 = v59 & (v61 + v63);
            v64 = (_QWORD *)(v60 + 16LL * v63);
            v65 = *v64;
            if ( v26 == *v64 )
              goto LABEL_91;
            ++v61;
          }
          if ( !v62 )
            v62 = v64;
          v92 = *(_DWORD *)(v7 + 16);
          ++*(_QWORD *)v7;
          v93 = v92 + 1;
          if ( 4 * v93 >= 3 * v58 )
          {
LABEL_198:
            sub_116E750(v7, 2 * v58);
            v94 = *(_DWORD *)(v7 + 24);
            if ( !v94 )
              goto LABEL_248;
            v95 = v94 - 1;
            v59 = *(_QWORD *)(v7 + 8);
            v96 = v95 & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
            v93 = *(_DWORD *)(v7 + 16) + 1;
            v62 = (_QWORD *)(v59 + 16LL * v96);
            v97 = *v62;
            if ( v26 != *v62 )
            {
              v98 = 1;
              v60 = 0;
              while ( v97 != -4096 )
              {
                if ( !v60 && v97 == -8192 )
                  v60 = (__int64)v62;
                v96 = v95 & (v98 + v96);
                v62 = (_QWORD *)(v59 + 16LL * v96);
                v97 = *v62;
                if ( v26 == *v62 )
                  goto LABEL_192;
                ++v98;
              }
              if ( v60 )
                v62 = (_QWORD *)v60;
            }
          }
          else if ( v58 - *(_DWORD *)(v7 + 20) - v93 <= v58 >> 3 )
          {
            sub_116E750(v7, v58);
            v99 = *(_DWORD *)(v7 + 24);
            if ( !v99 )
            {
LABEL_248:
              ++*(_DWORD *)(v7 + 16);
              BUG();
            }
            v100 = v99 - 1;
            v60 = *(_QWORD *)(v7 + 8);
            v101 = 0;
            v102 = v100 & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
            v103 = 1;
            v93 = *(_DWORD *)(v7 + 16) + 1;
            v62 = (_QWORD *)(v60 + 16LL * v102);
            v104 = *v62;
            if ( v26 != *v62 )
            {
              while ( v104 != -4096 )
              {
                if ( !v101 && v104 == -8192 )
                  v101 = v62;
                v59 = (unsigned int)(v103 + 1);
                v102 = v100 & (v103 + v102);
                v62 = (_QWORD *)(v60 + 16LL * v102);
                v104 = *v62;
                if ( v26 == *v62 )
                  goto LABEL_192;
                ++v103;
              }
              if ( v101 )
                v62 = v101;
            }
          }
LABEL_192:
          *(_DWORD *)(v7 + 16) = v93;
          if ( *v62 != -4096 )
            --*(_DWORD *)(v7 + 20);
          *v62 = v26;
          v66 = v62 + 1;
          v62[1] = 0;
          goto LABEL_92;
        }
LABEL_91:
        v66 = v64 + 1;
LABEL_92:
        *v66 = v37;
        v67 = (unsigned int)v149;
        v68 = (unsigned int)v149 + 1LL;
        if ( v68 > HIDWORD(v149) )
        {
          sub_C8D5F0((__int64)&v148, v150, v68, 8u, v60, v59);
          v67 = (unsigned int)v149;
        }
        *(_QWORD *)&v148[8 * v67] = v37;
        LODWORD(v149) = v149 + 1;
        if ( a3 )
        {
          v69 = *(unsigned int *)(a3 + 8);
          if ( v69 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
          {
            sub_C8D5F0(a3, (const void *)(a3 + 16), v69 + 1, 8u, v60, v59);
            v69 = *(unsigned int *)(a3 + 8);
          }
          *(_QWORD *)(*(_QWORD *)a3 + 8 * v69) = v37;
          ++*(_DWORD *)(a3 + 8);
        }
        nullsub_61();
        v178 = &unk_49DA100;
        nullsub_63();
        if ( v160 != &v162 )
          _libc_free((unsigned __int64)v160);
        if ( v117 == ++v125 )
        {
          a2 = v116;
          v120 = &v148[8 * (unsigned int)v149];
          if ( v120 != v148 )
          {
            v122 = (unsigned __int64)v148;
            do
            {
              v70 = *(_QWORD *)v122;
              a6 = sub_102DBD0(v118, *(_QWORD *)(*(_QWORD *)v122 + 40LL));
              v127 = (__int64 *)(a6 + 8 * v25);
              if ( (__int64 *)a6 != v127 )
              {
                v71 = (__int64 *)a6;
                v72 = v70;
                do
                {
                  v75 = *v71;
                  v76 = sub_2A59460(a1, *v71, v7, v116);
                  v77 = *(_DWORD *)(v72 + 4) & 0x7FFFFFF;
                  if ( v77 == *(_DWORD *)(v72 + 72) )
                  {
                    v121 = v76;
                    sub_B48D90(v72);
                    v76 = v121;
                    v77 = *(_DWORD *)(v72 + 4) & 0x7FFFFFF;
                  }
                  v73 = (v77 + 1) & 0x7FFFFFF;
                  v6 = v73 | *(_DWORD *)(v72 + 4) & 0xF8000000;
                  v74 = *(_QWORD *)(v72 - 8) + 32LL * (unsigned int)(v73 - 1);
                  *(_DWORD *)(v72 + 4) = v6;
                  if ( *(_QWORD *)v74 )
                  {
                    v6 = *(_QWORD *)(v74 + 8);
                    **(_QWORD **)(v74 + 16) = v6;
                    if ( v6 )
                      *(_QWORD *)(v6 + 16) = *(_QWORD *)(v74 + 16);
                  }
                  *(_QWORD *)v74 = v76;
                  if ( v76 )
                  {
                    v6 = *(_QWORD *)(v76 + 16);
                    *(_QWORD *)(v74 + 8) = v6;
                    if ( v6 )
                    {
                      a6 = v74 + 8;
                      *(_QWORD *)(v6 + 16) = v74 + 8;
                    }
                    *(_QWORD *)(v74 + 16) = v76 + 16;
                    *(_QWORD *)(v76 + 16) = v74;
                  }
                  ++v71;
                  v25 = (*(_DWORD *)(v72 + 4) & 0x7FFFFFFu) - 1;
                  *(_QWORD *)(*(_QWORD *)(v72 - 8) + 32LL * *(unsigned int *)(v72 + 72) + 8 * v25) = v75;
                }
                while ( v127 != v71 );
              }
              v122 += 8LL;
            }
            while ( v120 != (_BYTE *)v122 );
          }
          break;
        }
      }
    }
    v160 = 0;
    v161 = (unsigned __int64)&v165;
    v162 = 4;
    v163 = 0;
    v164 = 1;
    v78 = *(_QWORD *)(v7 + 32);
    v79 = (__int64 **)(v78 + 8LL * *(unsigned int *)(v7 + 40));
    if ( v79 == (__int64 **)v78 )
      goto LABEL_153;
    v80 = 1;
    v81 = *(__int64 ***)(v7 + 32);
    while ( 2 )
    {
      v86 = *v81;
      if ( !v80 )
        goto LABEL_118;
      v6 = v161;
      v25 = v161 + 8LL * HIDWORD(v162);
      v87 = (__int64 **)v161;
      if ( v161 == v25 )
      {
LABEL_138:
        if ( HIDWORD(v162) < (unsigned int)v162 )
        {
          ++HIDWORD(v162);
          *(_QWORD *)v25 = v86;
          v160 = (__int64 *)((char *)v160 + 1);
          v82 = v86[3];
          if ( *(_BYTE *)v82 != 84 )
          {
LABEL_140:
            v83 = *(_QWORD *)(v82 + 40);
LABEL_121:
            v84 = sub_2A59460(a1, v83, v7, a2);
            v85 = *v86;
            if ( v84 != *v86 )
            {
              if ( (*(_BYTE *)(v85 + 1) & 1) == 0 )
              {
LABEL_123:
                v6 = v86[2];
                v25 = v86[1];
                *(_QWORD *)v6 = v25;
                if ( v25 )
                {
                  v6 = v86[2];
                  *(_QWORD *)(v25 + 16) = v6;
                }
                goto LABEL_125;
              }
              v123 = v84;
              sub_BD7FF0(v85, v84);
              v85 = *v86;
              v84 = v123;
            }
            if ( v85 )
              goto LABEL_123;
LABEL_125:
            *v86 = v84;
            if ( v84 )
            {
              v25 = *(_QWORD *)(v84 + 16);
              v6 = v84 + 16;
              v86[1] = v25;
              if ( v25 )
                *(_QWORD *)(v25 + 16) = v86 + 1;
              v86[2] = v6;
              *(_QWORD *)(v84 + 16) = v86;
            }
            v80 = v164;
LABEL_130:
            if ( v79 == ++v81 )
            {
              if ( !v80 )
                _libc_free(v161);
              goto LABEL_153;
            }
            continue;
          }
LABEL_120:
          v83 = *(_QWORD *)(*(_QWORD *)(v82 - 8)
                          + 32LL * *(unsigned int *)(v82 + 72)
                          + 8LL * (unsigned int)(((__int64)v86 - *(_QWORD *)(v82 - 8)) >> 5));
          goto LABEL_121;
        }
LABEL_118:
        sub_C8CC70((__int64)&v160, (__int64)v86, v25, v6, v78, a6);
        v80 = v164;
        if ( !(_BYTE)v25 )
          goto LABEL_130;
        v82 = v86[3];
        if ( *(_BYTE *)v82 != 84 )
          goto LABEL_140;
        goto LABEL_120;
      }
      break;
    }
    while ( 1 )
    {
      while ( *v87 != v86 )
      {
        if ( (__int64 **)v25 == ++v87 )
          goto LABEL_138;
      }
      if ( v79 == ++v81 )
        break;
      v86 = *v81;
      v87 = (__int64 **)v161;
      if ( v161 == v25 )
        goto LABEL_138;
    }
LABEL_153:
    if ( v148 != v150 )
      _libc_free((unsigned __int64)v148);
    if ( !v158 )
      _libc_free((unsigned __int64)v155);
    if ( v151 != (__int64 *)v153 )
      _libc_free((unsigned __int64)v151);
    if ( !v146 )
      _libc_free((unsigned __int64)v143);
    if ( !v140 )
      _libc_free((unsigned __int64)v137);
    v7 += 104;
  }
  while ( v115 != v7 );
}
