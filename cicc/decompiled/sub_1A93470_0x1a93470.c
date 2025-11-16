// Function: sub_1A93470
// Address: 0x1a93470
//
__int64 __fastcall sub_1A93470(__int64 a1, unsigned __int64 a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // r14
  unsigned __int8 v6; // al
  int v7; // r8d
  int v8; // r9d
  __int64 v9; // rax
  __int64 v10; // rbx
  __int64 v11; // r13
  __int64 v12; // rax
  _QWORD *v13; // r12
  _QWORD *v14; // rax
  _QWORD *v15; // r15
  _QWORD *v16; // rax
  char v17; // r12
  __int64 v18; // rbx
  __int64 v19; // rax
  __int64 v20; // r12
  _QWORD *v21; // rdi
  int v22; // r11d
  unsigned int v23; // ecx
  _QWORD *v24; // rdx
  __int64 v25; // rsi
  __int64 v26; // rax
  __int64 v27; // rdx
  __int64 v28; // rdi
  __int64 v29; // rdi
  _QWORD *v30; // r13
  _QWORD *i; // r15
  __int64 v32; // rdi
  char *v33; // rsi
  unsigned __int64 v34; // rax
  __int64 v35; // rbx
  unsigned __int8 v36; // dl
  _QWORD *v37; // rbx
  _QWORD *v38; // r12
  __int64 v39; // rax
  char *v40; // rbx
  char *v41; // r12
  __int64 v42; // rax
  _QWORD *v43; // rdx
  char *v44; // rcx
  _QWORD *v45; // rax
  _QWORD *v46; // rbx
  _QWORD *v47; // r12
  __int64 v48; // r13
  __int64 v49; // rdi
  __int64 v51; // rcx
  bool v52; // cf
  __int64 v53; // rax
  unsigned __int64 v54; // rax
  __int64 v55; // r13
  __int64 v56; // r15
  __int64 v57; // r13
  char *v58; // rcx
  _QWORD *v59; // rdx
  _QWORD *v60; // rax
  _QWORD *v61; // rcx
  __int64 v62; // rax
  __int64 v63; // rbx
  int v64; // eax
  __int64 v65; // rdi
  __int64 v66; // rbx
  int v67; // edx
  unsigned __int64 v68; // rdx
  __int64 v69; // rax
  char v70; // r12
  __int64 v71; // rax
  char v72; // r12
  __int64 v73; // rcx
  __int64 v74; // rax
  __int64 v75; // rax
  int v76; // r8d
  int v77; // r9d
  char *v78; // rcx
  __int64 v79; // r15
  __int64 v80; // r12
  __int64 v81; // rbx
  unsigned __int64 v82; // rax
  char *v83; // r13
  __int64 v84; // rbx
  int v85; // eax
  __int64 v86; // rax
  const char *v87; // r15
  size_t v88; // rdx
  size_t v89; // r14
  size_t v90; // rdx
  const char *v91; // rdi
  size_t v92; // r12
  __int64 *v93; // r12
  __int64 *v94; // rdx
  __int64 v95; // rcx
  __int64 *v96; // rax
  __int64 *v97; // r14
  __int64 v98; // rax
  __int64 v99; // rbx
  unsigned int v100; // r12d
  __int64 v101; // rdx
  __int64 *v102; // r8
  int v103; // r11d
  __int64 v104; // rax
  __int64 *v105; // rcx
  __int64 v106; // rdi
  int v107; // ecx
  _BYTE *v108; // rsi
  _BYTE *v109; // r12
  _QWORD *v110; // r15
  __int64 v111; // rax
  unsigned __int64 v112; // r8
  int v113; // r9d
  __int64 v114; // rax
  __int64 *v115; // rdx
  __int64 *v116; // rax
  __int64 v117; // rdx
  int v118; // edx
  __int64 v119; // rdi
  __int64 v120; // rax
  int v121; // r9d
  __int64 *v122; // rsi
  int v123; // r9d
  __int64 v124; // rax
  unsigned __int8 v125; // [rsp+17h] [rbp-369h]
  __int64 v126; // [rsp+20h] [rbp-360h]
  unsigned __int64 v127; // [rsp+28h] [rbp-358h]
  __int64 v128; // [rsp+28h] [rbp-358h]
  __int64 *v129; // [rsp+30h] [rbp-350h]
  __int64 v130; // [rsp+40h] [rbp-340h]
  __int64 v131; // [rsp+40h] [rbp-340h]
  char *v132; // [rsp+50h] [rbp-330h]
  _QWORD *v133; // [rsp+58h] [rbp-328h]
  __int64 *v134; // [rsp+58h] [rbp-328h]
  unsigned __int64 v135; // [rsp+58h] [rbp-328h]
  char *v137; // [rsp+60h] [rbp-320h]
  __int64 *dest; // [rsp+68h] [rbp-318h]
  char *desta; // [rsp+68h] [rbp-318h]
  __int64 *destb; // [rsp+68h] [rbp-318h]
  char *v141; // [rsp+70h] [rbp-310h] BYREF
  char *v142; // [rsp+78h] [rbp-308h]
  char *v143; // [rsp+80h] [rbp-300h]
  _QWORD *v144; // [rsp+90h] [rbp-2F0h] BYREF
  _QWORD *v145; // [rsp+98h] [rbp-2E8h]
  __int64 v146; // [rsp+A0h] [rbp-2E0h]
  __int64 v147; // [rsp+B0h] [rbp-2D0h] BYREF
  __int64 v148; // [rsp+B8h] [rbp-2C8h]
  __int64 v149; // [rsp+C0h] [rbp-2C0h]
  __int64 v150; // [rsp+D0h] [rbp-2B0h] BYREF
  __int64 v151; // [rsp+D8h] [rbp-2A8h]
  __int64 v152; // [rsp+E0h] [rbp-2A0h]
  __int64 v153; // [rsp+E8h] [rbp-298h]
  unsigned __int64 v154[2]; // [rsp+F0h] [rbp-290h] BYREF
  char v155; // [rsp+100h] [rbp-280h] BYREF
  __int64 v156; // [rsp+108h] [rbp-278h]
  _QWORD *v157; // [rsp+110h] [rbp-270h]
  __int64 v158; // [rsp+118h] [rbp-268h]
  unsigned int v159; // [rsp+120h] [rbp-260h]
  unsigned __int64 v160; // [rsp+130h] [rbp-250h]
  char v161; // [rsp+138h] [rbp-248h]
  int v162; // [rsp+13Ch] [rbp-244h]
  __int64 *v163; // [rsp+140h] [rbp-240h] BYREF
  __int64 v164; // [rsp+148h] [rbp-238h]
  _BYTE v165[128]; // [rsp+150h] [rbp-230h] BYREF
  __int64 v166; // [rsp+1D0h] [rbp-1B0h] BYREF
  __int64 v167; // [rsp+1D8h] [rbp-1A8h]
  __int64 v168; // [rsp+1E0h] [rbp-1A0h]
  __int64 v169; // [rsp+1E8h] [rbp-198h]
  _BYTE *v170; // [rsp+1F0h] [rbp-190h] BYREF
  _BYTE *v171; // [rsp+1F8h] [rbp-188h]
  __int64 v172; // [rsp+200h] [rbp-180h]
  _BYTE v173[32]; // [rsp+208h] [rbp-178h] BYREF
  _BYTE *v174; // [rsp+228h] [rbp-158h]
  __int64 v175; // [rsp+230h] [rbp-150h]
  _BYTE v176[192]; // [rsp+238h] [rbp-148h] BYREF
  _BYTE *v177; // [rsp+2F8h] [rbp-88h]
  __int64 v178; // [rsp+300h] [rbp-80h]
  _BYTE v179[120]; // [rsp+308h] [rbp-78h] BYREF

  v2 = *(__int64 **)(a1 + 8);
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_251:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_4F9B6E8 )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_251;
  }
  v5 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(*(_QWORD *)(v3 + 8), &unk_4F9B6E8)
     + 360;
  v6 = sub_1AF0CE0(a2, 0, 0);
  v156 = 0;
  v125 = v6;
  v154[0] = (unsigned __int64)&v155;
  v154[1] = 0x100000000LL;
  v157 = 0;
  v158 = 0;
  v159 = 0;
  v161 = 0;
  v162 = 0;
  v160 = a2;
  sub_15D3930((__int64)v154);
  v163 = (__int64 *)v165;
  v164 = 0x1000000000LL;
  if ( byte_4FB58A0 )
    goto LABEL_6;
  sub_1611B20(&v150, *(_QWORD *)(a2 + 40));
  v70 = byte_4FB5980;
  v71 = sub_22077B0(224);
  v134 = (__int64 *)v71;
  v72 = v70 ^ 1;
  if ( v71 )
  {
    *(_QWORD *)(v71 + 8) = 0;
    *(_QWORD *)(v71 + 16) = &unk_4FB57F1;
    v73 = v71;
    v74 = v71 + 64;
    *(_DWORD *)(v74 - 40) = 3;
    *(_QWORD *)(v74 - 32) = 0;
    *(_QWORD *)(v74 - 24) = 0;
    *(_QWORD *)(v74 - 16) = 0;
    *(_DWORD *)v74 = 0;
    *(_QWORD *)(v74 + 8) = 0;
    *(_QWORD *)(v73 + 80) = v74;
    *(_QWORD *)(v73 + 88) = v74;
    *(_QWORD *)(v73 + 128) = v73 + 112;
    *(_QWORD *)(v73 + 136) = v73 + 112;
    *(_QWORD *)(v73 + 96) = 0;
    *(_DWORD *)(v73 + 112) = 0;
    *(_QWORD *)(v73 + 120) = 0;
    *(_QWORD *)(v73 + 144) = 0;
    *(_BYTE *)(v73 + 152) = 0;
    *(_QWORD *)v73 = off_49F5D48;
    *(_QWORD *)(v73 + 160) = 0;
    *(_QWORD *)(v73 + 168) = 0;
    *(_QWORD *)(v73 + 176) = 0;
    *(_BYTE *)(v73 + 184) = v72;
    *(_QWORD *)(v73 + 192) = 0;
    *(_QWORD *)(v73 + 200) = 0;
    *(_QWORD *)(v73 + 208) = 0;
    *(_QWORD *)(v73 + 216) = 0;
    v75 = sub_163A1D0();
    sub_1A91D30(v75);
  }
  sub_1618EA0((__int64)&v150, v134, 0);
  sub_161A730((__int64)&v150, a2);
  v160 = a2;
  sub_15D3930((__int64)v154);
  v78 = (char *)v134[21];
  v79 = v134[20];
  v132 = v78;
  if ( v78 == (char *)v79 )
  {
    v93 = (__int64 *)v134[20];
LABEL_154:
    destb = v93;
    goto LABEL_155;
  }
  v80 = (__int64)&v78[-v79];
  v81 = v134[21];
  _BitScanReverse64(&v82, (__int64)&v78[-v79] >> 3);
  sub_1A91840((char *)v79, v78, 2LL * (int)(63 - (v82 ^ 0x3F)));
  if ( v80 > 128 )
  {
    sub_1A915D0((_QWORD *)v79, (_QWORD *)(v79 + 128));
    desta = (char *)(v79 + 128);
    if ( v81 == v79 + 128 )
      goto LABEL_147;
    v130 = v5;
    while ( 1 )
    {
      v83 = desta;
      v84 = *(_QWORD *)desta;
      while ( 1 )
      {
        v87 = sub_1649960(*(_QWORD *)(*((_QWORD *)v83 - 1) + 40LL));
        v89 = v88;
        v91 = sub_1649960(*(_QWORD *)(v84 + 40));
        v92 = v90;
        if ( v89 < v90 )
          break;
        if ( v90 )
        {
          v85 = memcmp(v91, v87, v90);
          if ( v85 )
            goto LABEL_144;
        }
        if ( v89 == v92 )
          goto LABEL_145;
LABEL_139:
        if ( v89 <= v92 )
          goto LABEL_145;
LABEL_140:
        v86 = *((_QWORD *)v83 - 1);
        v83 -= 8;
        *((_QWORD *)v83 + 1) = v86;
      }
      if ( !v89 )
        goto LABEL_145;
      v85 = memcmp(v91, v87, v89);
      if ( !v85 )
        goto LABEL_139;
LABEL_144:
      if ( v85 < 0 )
        goto LABEL_140;
LABEL_145:
      desta += 8;
      *(_QWORD *)v83 = v84;
      if ( v132 == desta )
      {
        v5 = v130;
        goto LABEL_147;
      }
    }
  }
  sub_1A915D0((_QWORD *)v79, v132);
LABEL_147:
  v93 = (__int64 *)v134[21];
  v79 = v134[20];
  if ( (__int64 *)v79 == v93 )
    goto LABEL_154;
  v94 = (__int64 *)v134[20];
  do
  {
    v96 = v94++;
    if ( v94 == v93 )
      goto LABEL_154;
    v95 = *(v94 - 1);
  }
  while ( v95 != *v94 );
  destb = v94;
  if ( v96 == v93 )
    goto LABEL_154;
  v115 = v96 + 2;
  if ( v93 == v96 + 2 )
    goto LABEL_201;
  while ( 1 )
  {
    if ( *v115 != v95 )
    {
      v96[1] = *v115;
      ++v96;
    }
    if ( v93 == ++v115 )
      break;
    v95 = *v96;
  }
  destb = v96 + 1;
  v116 = (__int64 *)v134[21];
  if ( destb != v93 )
  {
    if ( v116 != v93 )
    {
      memmove(destb, v93, (char *)v116 - (char *)v93);
      v79 = v134[20];
      destb = (__int64 *)((char *)destb + v134[21] - (_QWORD)v93);
      if ( destb == (__int64 *)v134[21] )
        goto LABEL_155;
      goto LABEL_197;
    }
LABEL_201:
    v79 = v134[20];
LABEL_197:
    v134[21] = (__int64)destb;
    goto LABEL_155;
  }
  destb = (__int64 *)v134[21];
  v79 = v134[20];
LABEL_155:
  if ( destb == (__int64 *)v79 )
    goto LABEL_186;
  v131 = v5;
  v97 = (__int64 *)v79;
LABEL_160:
  while ( 2 )
  {
    v99 = *v97;
    if ( byte_4FB5B40 )
    {
      v166 = 0;
      v100 = 0;
      v167 = 0;
      v168 = 0;
      v169 = 0;
      v170 = 0;
      v171 = 0;
      v172 = 0;
      while ( 1 )
      {
        if ( (unsigned int)sub_15F4D60(v99) <= v100 )
        {
          v109 = v171;
          v110 = v170;
          if ( v170 != v171 )
          {
            do
            {
              v111 = sub_1AA91E0(*(_QWORD *)(v99 + 40), *v110, v154, 0);
              v112 = sub_157EBA0(v111);
              v114 = (unsigned int)v164;
              if ( (unsigned int)v164 >= HIDWORD(v164) )
              {
                v135 = v112;
                sub_16CD150((__int64)&v163, v165, 0, 8, v112, v113);
                v114 = (unsigned int)v164;
                v112 = v135;
              }
              ++v110;
              v163[v114] = v112;
              LODWORD(v164) = v164 + 1;
            }
            while ( v109 != (_BYTE *)v110 );
          }
          j___libc_free_0(0);
          if ( v170 )
            j_j___libc_free_0(v170, v172 - (_QWORD)v170);
          ++v97;
          j___libc_free_0(v167);
          if ( destb == v97 )
            goto LABEL_185;
          goto LABEL_160;
        }
        v147 = sub_15F4DF0(v99, v100);
        if ( sub_15CC8F0((__int64)v154, v147, *(_QWORD *)(v99 + 40)) )
          break;
LABEL_162:
        ++v100;
      }
      if ( (_DWORD)v169 )
      {
        v101 = v147;
        v102 = 0;
        v103 = 1;
        LODWORD(v104) = (v169 - 1) & (((unsigned int)v147 >> 9) ^ ((unsigned int)v147 >> 4));
        v105 = (__int64 *)(v167 + 8LL * (unsigned int)v104);
        v106 = *v105;
        if ( v147 == *v105 )
          goto LABEL_162;
        while ( v106 != -8 )
        {
          if ( v106 == -16 && !v102 )
            v102 = v105;
          v104 = ((_DWORD)v169 - 1) & (unsigned int)(v104 + v103);
          v105 = (__int64 *)(v167 + 8 * v104);
          v106 = *v105;
          if ( v147 == *v105 )
            goto LABEL_162;
          ++v103;
        }
        if ( !v102 )
          v102 = v105;
        ++v166;
        v107 = v168 + 1;
        if ( 4 * ((int)v168 + 1) < (unsigned int)(3 * v169) )
        {
          if ( (int)v169 - HIDWORD(v168) - v107 > (unsigned int)v169 >> 3 )
            goto LABEL_172;
          sub_13B3D40((__int64)&v166, v169);
          if ( !(_DWORD)v169 )
          {
LABEL_250:
            LODWORD(v168) = v168 + 1;
            goto LABEL_251;
          }
          v119 = v147;
          v122 = 0;
          v123 = 1;
          v107 = v168 + 1;
          LODWORD(v124) = (v169 - 1) & (((unsigned int)v147 >> 9) ^ ((unsigned int)v147 >> 4));
          v102 = (__int64 *)(v167 + 8LL * (unsigned int)v124);
          v101 = *v102;
          if ( v147 == *v102 )
            goto LABEL_172;
          while ( v101 != -8 )
          {
            if ( !v122 && v101 == -16 )
              v122 = v102;
            v124 = ((_DWORD)v169 - 1) & (unsigned int)(v124 + v123);
            v102 = (__int64 *)(v167 + 8 * v124);
            v101 = *v102;
            if ( v147 == *v102 )
              goto LABEL_172;
            ++v123;
          }
          goto LABEL_222;
        }
      }
      else
      {
        ++v166;
      }
      sub_13B3D40((__int64)&v166, 2 * v169);
      if ( !(_DWORD)v169 )
        goto LABEL_250;
      v119 = v147;
      v107 = v168 + 1;
      LODWORD(v120) = (v169 - 1) & (((unsigned int)v147 >> 9) ^ ((unsigned int)v147 >> 4));
      v102 = (__int64 *)(v167 + 8LL * (unsigned int)v120);
      v101 = *v102;
      if ( v147 == *v102 )
        goto LABEL_172;
      v121 = 1;
      v122 = 0;
      while ( v101 != -8 )
      {
        if ( !v122 && v101 == -16 )
          v122 = v102;
        v120 = ((_DWORD)v169 - 1) & (unsigned int)(v120 + v121);
        v102 = (__int64 *)(v167 + 8 * v120);
        v101 = *v102;
        if ( v147 == *v102 )
          goto LABEL_172;
        ++v121;
      }
LABEL_222:
      v101 = v119;
      if ( v122 )
        v102 = v122;
LABEL_172:
      LODWORD(v168) = v107;
      if ( *v102 != -8 )
        --HIDWORD(v168);
      *v102 = v101;
      v108 = v171;
      if ( v171 == (_BYTE *)v172 )
      {
        sub_1292090((__int64)&v170, v171, &v147);
      }
      else
      {
        if ( v171 )
        {
          *(_QWORD *)v171 = v147;
          v108 = v171;
        }
        v171 = v108 + 8;
      }
      goto LABEL_162;
    }
    v98 = (unsigned int)v164;
    if ( (unsigned int)v164 >= HIDWORD(v164) )
    {
      sub_16CD150((__int64)&v163, v165, 0, 8, v76, v77);
      v98 = (unsigned int)v164;
    }
    ++v97;
    v163[v98] = v99;
    LODWORD(v164) = v164 + 1;
    if ( destb != v97 )
      continue;
    break;
  }
LABEL_185:
  v125 = 1;
  v5 = v131;
LABEL_186:
  sub_160FDE0(&v150);
LABEL_6:
  if ( byte_4FB5A60 )
  {
    v9 = (unsigned int)v164;
    goto LABEL_8;
  }
  v62 = *(_QWORD *)(a2 + 80);
  if ( !v62 )
    BUG();
  v63 = *(_QWORD *)(v62 + 24);
  if ( v63 )
    v63 -= 24;
LABEL_113:
  v64 = *(unsigned __int8 *)(v63 + 16);
  if ( (unsigned int)(v64 - 25) > 9 )
  {
    do
    {
      if ( (unsigned __int8)v64 <= 0x17u )
      {
LABEL_120:
        v66 = *(_QWORD *)(v63 + 32);
        if ( !v66 )
          goto LABEL_200;
        goto LABEL_121;
      }
      if ( (_BYTE)v64 == 78 )
        goto LABEL_124;
LABEL_118:
      if ( (_BYTE)v64 == 29 )
      {
        v68 = v63 & 0xFFFFFFFFFFFFFFF8LL;
        if ( (v63 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
          goto LABEL_199;
LABEL_125:
        if ( *(_BYTE *)(v68 + 16) != 78 )
          goto LABEL_126;
        v117 = *(_QWORD *)(v68 - 24);
        if ( *(_BYTE *)(v117 + 16) )
          goto LABEL_126;
        if ( (*(_BYTE *)(v117 + 33) & 0x20) == 0 )
          goto LABEL_126;
        v118 = *(_DWORD *)(v117 + 36);
        if ( v118 == 78 || (unsigned int)(v118 - 80) <= 1 )
          goto LABEL_126;
      }
LABEL_119:
      if ( (unsigned int)(v64 - 25) > 9 )
        goto LABEL_120;
LABEL_199:
      v66 = *(_QWORD *)(sub_157F210(*(_QWORD *)(v63 + 40)) + 48);
      if ( !v66 )
      {
LABEL_200:
        v63 = 0;
        goto LABEL_113;
      }
LABEL_121:
      v67 = *(unsigned __int8 *)(v66 - 8);
      v63 = v66 - 24;
      v64 = v67;
    }
    while ( (unsigned int)(v67 - 25) > 9 );
  }
  v65 = sub_157F210(*(_QWORD *)(v63 + 40));
  if ( v65 && sub_157F120(v65) )
  {
    v64 = *(unsigned __int8 *)(v63 + 16);
    if ( (unsigned __int8)v64 > 0x17u )
    {
      if ( (_BYTE)v64 == 78 )
      {
LABEL_124:
        v68 = v63 & 0xFFFFFFFFFFFFFFF8LL;
        if ( (v63 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
          goto LABEL_125;
        goto LABEL_120;
      }
      goto LABEL_118;
    }
    goto LABEL_119;
  }
LABEL_126:
  v69 = (unsigned int)v164;
  if ( (unsigned int)v164 >= HIDWORD(v164) )
  {
    sub_16CD150((__int64)&v163, v165, 0, 8, v7, v8);
    v69 = (unsigned int)v164;
  }
  v125 = 1;
  v163[v69] = v63;
  v9 = (unsigned int)(v164 + 1);
  LODWORD(v164) = v164 + 1;
LABEL_8:
  v129 = &v163[v9];
  if ( v163 == v129 )
    goto LABEL_70;
  dest = v163;
  v126 = 0;
  v133 = 0;
  v137 = 0;
  while ( 2 )
  {
    v10 = *dest;
    v141 = 0;
    v142 = 0;
    v11 = *(_QWORD *)(v10 + 40);
    v143 = 0;
    v12 = sub_15F2050(v10);
    v13 = (_QWORD *)sub_16321A0(v12, (__int64)"gc.safepoint_poll", 17);
    LOWORD(v168) = 257;
    v14 = sub_1648A60(72, 1u);
    v15 = v14;
    if ( v14 )
    {
      sub_15F5ED0((__int64)v14, v13, (__int64)&v166, v10);
      v16 = v15 + 3;
    }
    else
    {
      v16 = 0;
    }
    if ( *(_QWORD **)(v11 + 48) == v16 )
    {
      v127 = (unsigned __int64)v16;
      v17 = 1;
    }
    else
    {
      v17 = 0;
      v127 = *v16 & 0xFFFFFFFFFFFFFFF8LL;
    }
    v18 = v16[1];
    v166 = 0;
    v171 = v173;
    v172 = 0x400000000LL;
    v177 = v179;
    v174 = v176;
    v167 = 0;
    v168 = 0;
    v169 = 0;
    v170 = 0;
    v175 = 0x800000000LL;
    v178 = 0x800000000LL;
    sub_1AE1260(v15, &v166, 0, 1);
    v144 = 0;
    v145 = 0;
    v146 = 0;
    v150 = 0;
    v151 = 0;
    v152 = 0;
    v153 = 0;
    if ( v17 )
      v19 = *(_QWORD *)(v11 + 48);
    else
      v19 = *(_QWORD *)(v127 + 8);
    if ( v18 )
      v18 -= 24;
    if ( !v19 )
    {
      v147 = 0;
      v148 = 0;
      v149 = 0;
      BUG();
    }
    v147 = 0;
    v128 = v19 - 24;
    v148 = 0;
    v149 = 0;
    v20 = *(_QWORD *)(v19 + 16);
    v150 = 1;
    sub_13B3D40((__int64)&v150, 0);
    if ( !(_DWORD)v153 )
    {
      LODWORD(v152) = v152 + 1;
      BUG();
    }
    v21 = 0;
    v22 = 1;
    v23 = (v153 - 1) & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
    v24 = (_QWORD *)(v151 + 8LL * v23);
    v25 = *v24;
    if ( *v24 != v20 )
    {
      while ( v25 != -8 )
      {
        if ( !v21 && v25 == -16 )
          v21 = v24;
        v23 = (v153 - 1) & (v22 + v23);
        v24 = (_QWORD *)(v151 + 8LL * v23);
        v25 = *v24;
        if ( v20 == *v24 )
          goto LABEL_21;
        ++v22;
      }
      if ( v21 )
        v24 = v21;
    }
LABEL_21:
    LODWORD(v152) = v152 + 1;
    if ( *v24 != -8 )
      --HIDWORD(v152);
    *v24 = v20;
    sub_1A930A0(v128, v18, (__int64)&v144, (__int64)&v150, (__int64)&v147);
    v26 = v148;
    if ( v147 == v148 )
    {
      v29 = v148;
    }
    else
    {
      do
      {
        v27 = *(_QWORD *)(v26 - 8);
        v148 = v26 - 8;
        v28 = *(_QWORD *)(v27 + 48);
        if ( v28 )
          v28 -= 24;
        sub_1A930A0(v28, v18, (__int64)&v144, (__int64)&v150, (__int64)&v147);
        v26 = v148;
        v29 = v147;
      }
      while ( v148 != v147 );
    }
    if ( v29 )
      j_j___libc_free_0(v29, v149 - v29);
    v30 = v145;
    for ( i = v144; v30 != i; ++i )
    {
      v32 = 0;
      v34 = *i & 0xFFFFFFFFFFFFFFF8LL;
      v35 = *i | 4LL;
      v147 = v35;
      v36 = *(_BYTE *)(v34 + 16);
      if ( v36 > 0x17u )
      {
        if ( v36 == 78 )
        {
          v32 = v34 | 4;
        }
        else
        {
          v32 = 0;
          if ( v36 == 29 )
            v32 = v34;
        }
      }
      if ( !(unsigned __int8)sub_1AEC650(v32, v5) && (unsigned __int8)sub_1A91760(&v147) )
      {
        v147 = v35;
        v33 = v142;
        if ( v142 == v143 )
        {
          sub_1A92F20(&v141, v142, &v147);
        }
        else
        {
          if ( v142 )
          {
            *(_QWORD *)v142 = v35;
            v33 = v142;
          }
          v142 = v33 + 8;
        }
      }
    }
    j___libc_free_0(v151);
    if ( v144 )
      j_j___libc_free_0(v144, v146 - (_QWORD)v144);
    if ( v177 != v179 )
      _libc_free((unsigned __int64)v177);
    v37 = v174;
    v38 = &v174[24 * (unsigned int)v175];
    if ( v174 != (_BYTE *)v38 )
    {
      do
      {
        v39 = *(v38 - 1);
        v38 -= 3;
        if ( v39 != -8 && v39 != 0 && v39 != -16 )
          sub_1649B30(v38);
      }
      while ( v37 != v38 );
      v38 = v174;
    }
    if ( v38 != (_QWORD *)v176 )
      _libc_free((unsigned __int64)v38);
    if ( v171 != v173 )
      _libc_free((unsigned __int64)v171);
    v40 = v142;
    v41 = v141;
    if ( v142 != v141 )
    {
      v42 = v142 - v141;
      if ( v126 - (__int64)v133 >= (unsigned __int64)(v142 - v141) )
      {
        v43 = (_QWORD *)((char *)v133 + v42);
        v44 = v141;
        v45 = v133;
        do
        {
          if ( v45 )
            *v45 = *(_QWORD *)v44;
          ++v45;
          v44 += 8;
        }
        while ( v45 != v43 );
        v133 = v45;
        goto LABEL_64;
      }
      v51 = ((char *)v133 - v137) >> 3;
      if ( v42 >> 3 > (unsigned __int64)(0xFFFFFFFFFFFFFFFLL - v51) )
        sub_4262D8((__int64)"vector::_M_range_insert");
      v52 = (char *)v133 - v137 < (unsigned __int64)v42;
      v53 = v42 >> 3;
      if ( !v52 )
        v53 = ((char *)v133 - v137) >> 3;
      v52 = __CFADD__(v51, v53);
      v54 = v51 + v53;
      if ( v52 )
      {
        v55 = 0x7FFFFFFFFFFFFFF8LL;
        goto LABEL_95;
      }
      if ( v54 )
      {
        if ( v54 > 0xFFFFFFFFFFFFFFFLL )
          v54 = 0xFFFFFFFFFFFFFFFLL;
        v55 = 8 * v54;
LABEL_95:
        v56 = sub_22077B0(v55);
        v57 = v56 + v55;
      }
      else
      {
        v57 = 0;
        v56 = 0;
      }
      if ( v133 == (_QWORD *)v137 )
      {
        v60 = (_QWORD *)v56;
      }
      else
      {
        v58 = v137;
        v59 = (_QWORD *)v56;
        v60 = (_QWORD *)(v56 + (char *)v133 - v137);
        do
        {
          if ( v59 )
            *v59 = *(_QWORD *)v58;
          ++v59;
          v58 += 8;
        }
        while ( v59 != v60 );
      }
      v61 = (_QWORD *)((char *)v60 + v40 - v41);
      v133 = v61;
      do
      {
        if ( v60 )
          *v60 = *(_QWORD *)v41;
        ++v60;
        v41 += 8;
      }
      while ( v60 != v61 );
      if ( v137 )
        j_j___libc_free_0(v137, v126 - (_QWORD)v137);
      v126 = v57;
      v41 = v141;
      v137 = (char *)v56;
    }
LABEL_64:
    if ( v41 )
      j_j___libc_free_0(v41, v143 - v41);
    if ( v129 != ++dest )
      continue;
    break;
  }
  if ( v137 )
    j_j___libc_free_0(v137, v126 - (_QWORD)v137);
  v129 = v163;
LABEL_70:
  if ( v129 != (__int64 *)v165 )
    _libc_free((unsigned __int64)v129);
  if ( v159 )
  {
    v46 = v157;
    v47 = &v157[2 * v159];
    do
    {
      if ( *v46 != -8 && *v46 != -16 )
      {
        v48 = v46[1];
        if ( v48 )
        {
          v49 = *(_QWORD *)(v48 + 24);
          if ( v49 )
            j_j___libc_free_0(v49, *(_QWORD *)(v48 + 40) - v49);
          j_j___libc_free_0(v48, 56);
        }
      }
      v46 += 2;
    }
    while ( v47 != v46 );
  }
  j___libc_free_0(v157);
  if ( (char *)v154[0] != &v155 )
    _libc_free(v154[0]);
  return v125;
}
