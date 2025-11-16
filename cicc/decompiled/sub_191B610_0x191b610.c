// Function: sub_191B610
// Address: 0x191b610
//
__int64 __fastcall sub_191B610(
        __int64 a1,
        _QWORD *a2,
        __int64 a3,
        __m128 a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        __m128 a11)
{
  __int64 v11; // r14
  __int64 v14; // rdx
  unsigned int v15; // r15d
  unsigned int v17; // eax
  char v18; // al
  __int64 v19; // rax
  unsigned int v20; // eax
  _QWORD *v21; // r13
  __int64 v22; // rbx
  _QWORD *v23; // rcx
  __int64 v24; // r15
  __int64 v25; // rsi
  unsigned int v26; // r8d
  __int64 *v27; // rcx
  __int64 v28; // rdi
  unsigned int v29; // esi
  __int64 v30; // r11
  unsigned int v31; // ecx
  __int64 v32; // rdx
  unsigned int v33; // eax
  __int64 *v34; // rdi
  __int64 v35; // r9
  unsigned int v36; // r10d
  unsigned int v37; // r9d
  unsigned __int8 *v38; // rax
  __int64 v39; // r8
  int v40; // eax
  __int64 v41; // rsi
  _DWORD *v42; // rax
  int v43; // r8d
  int v44; // r9d
  __int64 v45; // rdx
  _QWORD *v46; // rdx
  __int64 v47; // rax
  __int64 v48; // rdx
  unsigned __int8 *v49; // rdi
  int v50; // eax
  int v51; // eax
  __int64 v52; // rdx
  __int64 v53; // rdx
  _QWORD *v54; // rax
  _QWORD *v55; // rsi
  signed __int64 v56; // rdx
  _QWORD *v57; // rcx
  __int64 v58; // rdx
  __int64 v59; // rdx
  __int64 v60; // rdx
  unsigned int v61; // ebx
  __int64 v62; // r9
  const char *v63; // rax
  int v64; // ebx
  __int64 v65; // r15
  __int64 v66; // rdx
  __int64 v67; // rax
  __int64 v68; // rcx
  __int64 v69; // r8
  __int64 v70; // r9
  __int64 v71; // r10
  __int64 v72; // rdx
  __int64 v73; // r13
  __int64 v74; // r15
  __int64 v75; // r14
  __int64 v76; // rdi
  _QWORD *v77; // rax
  __int64 v78; // r8
  unsigned __int64 v79; // rdi
  __int64 v80; // r8
  __int64 v81; // rdx
  __int64 v82; // rax
  __int64 v83; // rdi
  __int64 v84; // rbx
  __int64 v85; // rdx
  __int64 v86; // r8
  int v87; // eax
  __int64 v88; // rax
  int v89; // edi
  __int64 v90; // rax
  _QWORD *v91; // rax
  int v92; // eax
  __int64 v93; // rax
  int v94; // edx
  __int64 v95; // rdx
  __int64 *v96; // rax
  __int64 v97; // rsi
  unsigned __int64 v98; // rdx
  __int64 v99; // rdx
  __int64 v100; // rdx
  __int64 v101; // rax
  int v102; // ecx
  int v103; // r10d
  int v104; // r10d
  unsigned __int8 *v105; // r8
  int v106; // eax
  int v107; // edx
  __int64 v108; // rcx
  __int64 v109; // rdx
  double v110; // xmm4_8
  double v111; // xmm5_8
  unsigned __int8 *v112; // rsi
  __int64 v113; // r10
  unsigned __int8 **v114; // r15
  char v115; // al
  int *v116; // rax
  _QWORD *v117; // rcx
  _QWORD *j; // rax
  _QWORD *v119; // rdx
  __int64 v120; // rcx
  _QWORD *v121; // rbx
  int v122; // eax
  __int64 v123; // rdi
  int v124; // eax
  int v125; // r8d
  unsigned int v126; // esi
  _QWORD *v127; // r15
  _QWORD *v128; // rdx
  __int64 v129; // r8
  unsigned __int64 v130; // rbx
  unsigned __int64 v131; // rax
  __int64 v132; // rdx
  __int64 v133; // rcx
  double v134; // xmm4_8
  double v135; // xmm5_8
  __int64 v136; // rax
  unsigned int v137; // edx
  __int64 *v138; // rax
  __int64 v139; // rax
  __int64 v140; // rdx
  __int64 v141; // rax
  __int64 v142; // rsi
  _QWORD *v143; // rsi
  __int64 v144; // rdx
  __int64 v145; // rsi
  int v146; // edi
  unsigned int i; // ecx
  _QWORD *v148; // rax
  int v149; // r8d
  int v150; // r9d
  unsigned __int64 v151; // r12
  unsigned __int64 *v152; // rax
  unsigned int v153; // ecx
  __int64 v154; // [rsp+0h] [rbp-160h]
  _QWORD *v155; // [rsp+8h] [rbp-158h]
  int v156; // [rsp+18h] [rbp-148h]
  __int64 v157; // [rsp+18h] [rbp-148h]
  _DWORD *v158; // [rsp+18h] [rbp-148h]
  __int64 v159; // [rsp+18h] [rbp-148h]
  unsigned int v160; // [rsp+28h] [rbp-138h]
  __int64 v161; // [rsp+28h] [rbp-138h]
  __int64 v162; // [rsp+28h] [rbp-138h]
  __int64 v163; // [rsp+30h] [rbp-130h]
  unsigned __int8 v164; // [rsp+38h] [rbp-128h]
  unsigned int v165; // [rsp+44h] [rbp-11Ch]
  int v166; // [rsp+48h] [rbp-118h]
  __int64 v167; // [rsp+48h] [rbp-118h]
  __int64 v168; // [rsp+48h] [rbp-118h]
  __int64 v169; // [rsp+48h] [rbp-118h]
  __int64 v170; // [rsp+48h] [rbp-118h]
  __int64 v171; // [rsp+48h] [rbp-118h]
  __int64 v172; // [rsp+48h] [rbp-118h]
  __int64 *v173; // [rsp+48h] [rbp-118h]
  __int64 v174; // [rsp+50h] [rbp-110h]
  __int64 v175; // [rsp+50h] [rbp-110h]
  unsigned int v176; // [rsp+58h] [rbp-108h]
  __int64 v177; // [rsp+58h] [rbp-108h]
  __int64 v178; // [rsp+58h] [rbp-108h]
  __int64 v179; // [rsp+68h] [rbp-F8h] BYREF
  __int64 v180[2]; // [rsp+70h] [rbp-F0h] BYREF
  unsigned __int8 *v181[2]; // [rsp+80h] [rbp-E0h] BYREF
  __int16 v182; // [rsp+90h] [rbp-D0h]
  _BYTE *v183; // [rsp+A0h] [rbp-C0h] BYREF
  __int64 v184; // [rsp+A8h] [rbp-B8h]
  _BYTE v185[176]; // [rsp+B0h] [rbp-B0h] BYREF

  v11 = a1;
  if ( (unsigned __int8)(*((_BYTE *)a2 + 16) - 25) <= 0x34u )
  {
    v14 = 0x100000100003FFLL;
    if ( _bittest64(&v14, (unsigned int)*((unsigned __int8 *)a2 + 16) - 25) )
      return 0;
  }
  if ( !*(_BYTE *)(*a2 + 8LL) )
    return 0;
  if ( (unsigned __int8)sub_15F2ED0((__int64)a2) )
    return 0;
  if ( (unsigned __int8)sub_15F3040((__int64)a2) )
    return 0;
  LOBYTE(v17) = sub_15F3330((__int64)a2);
  v15 = v17;
  if ( (_BYTE)v17 )
    return 0;
  v18 = *((_BYTE *)a2 + 16);
  if ( v18 == 78 )
  {
    v109 = *(a2 - 3);
    if ( (*(_BYTE *)(v109 + 16) || (*(_BYTE *)(v109 + 33) & 0x20) == 0
                                || (unsigned int)(*(_DWORD *)(v109 + 36) - 35) > 3)
      && *(_BYTE *)(v109 + 16) != 20 )
    {
      goto LABEL_14;
    }
    return 0;
  }
  if ( !a3 || v18 != 35 )
  {
    if ( v18 != 75 && v18 != 76 )
      goto LABEL_14;
    return 0;
  }
  v19 = *(a2 - 6);
  if ( v19 )
  {
    if ( a3 == v19 )
    {
      v136 = *(a2 - 3);
      if ( *(_BYTE *)(v136 + 16) == 13 )
      {
        v137 = *(_DWORD *)(v136 + 32);
        v138 = *(__int64 **)(v136 + 24);
        v139 = v137 > 0x40
             ? *v138
             : (__int64)((_QWORD)v138 << (64 - (unsigned __int8)v137)) >> (64 - (unsigned __int8)v137);
        if ( v139 == 1 && (*(_DWORD *)(a3 + 20) & 0xFFFFFFF) != 0 )
        {
          v140 = 0;
          v141 = 24LL * (*(_DWORD *)(a3 + 20) & 0xFFFFFFF);
          do
          {
            v142 = a3 - v141;
            if ( (*(_BYTE *)(a3 + 23) & 0x40) != 0 )
              v142 = *(_QWORD *)(a3 - 8);
            v143 = *(_QWORD **)(v142 + v140);
            if ( a2 == v143 )
            {
              if ( v143 )
                return 0;
            }
            v140 += 24;
          }
          while ( v140 != v141 );
        }
      }
    }
  }
LABEL_14:
  v174 = a1 + 152;
  v20 = sub_190AC30(a1 + 152, (__int64)a2, 1);
  v21 = (_QWORD *)a2[5];
  v176 = v20;
  if ( *(_BYTE *)(a1 + 784) )
    sub_191A350(a1, v21[7]);
  v183 = v185;
  v184 = 0x800000000LL;
  v22 = v21[1];
  if ( !v22 )
    goto LABEL_36;
  while ( 1 )
  {
    v23 = sub_1648700(v22);
    if ( (unsigned __int8)(*((_BYTE *)v23 + 16) - 25) <= 9u )
      break;
    v22 = *(_QWORD *)(v22 + 8);
    if ( !v22 )
      goto LABEL_36;
  }
  v163 = 0;
  v160 = 0;
  v164 = v15;
  v165 = ((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4);
  v166 = 0;
LABEL_34:
  v47 = *(_QWORD *)(v11 + 24);
  v48 = *(unsigned int *)(v47 + 48);
  if ( !(_DWORD)v48 )
    goto LABEL_35;
  v24 = v23[5];
  v25 = *(_QWORD *)(v47 + 32);
  v26 = (v48 - 1) & (((unsigned int)v24 >> 4) ^ ((unsigned int)v24 >> 9));
  v27 = (__int64 *)(v25 + 16LL * v26);
  v28 = *v27;
  if ( v24 != *v27 )
  {
    v102 = 1;
    while ( v28 != -8 )
    {
      v103 = v102 + 1;
      v26 = (v48 - 1) & (v102 + v26);
      v27 = (__int64 *)(v25 + 16LL * v26);
      v28 = *v27;
      if ( v24 == *v27 )
        goto LABEL_20;
      v102 = v103;
    }
    goto LABEL_35;
  }
LABEL_20:
  if ( v27 == (__int64 *)(v25 + 16 * v48) || !v27[1] )
  {
LABEL_35:
    v15 = v164;
    goto LABEL_36;
  }
  v29 = *(_DWORD *)(v11 + 776);
  v179 = v24;
  v30 = v11 + 752;
  if ( !v29 )
  {
    ++*(_QWORD *)(v11 + 752);
LABEL_141:
    v29 *= 2;
    goto LABEL_142;
  }
  v31 = v29 - 1;
  v32 = *(_QWORD *)(v11 + 760);
  v33 = (v29 - 1) & (((unsigned int)v24 >> 4) ^ ((unsigned int)v24 >> 9));
  v34 = (__int64 *)(v32 + 16LL * v33);
  v35 = *v34;
  if ( v24 == *v34 )
  {
LABEL_24:
    v36 = *((_DWORD *)v34 + 2);
    v180[0] = (__int64)v21;
    goto LABEL_25;
  }
  v104 = 1;
  v105 = 0;
  while ( v35 != -8 )
  {
    if ( v35 == -16 && !v105 )
      v105 = (unsigned __int8 *)v34;
    v33 = v31 & (v104 + v33);
    v34 = (__int64 *)(v32 + 16LL * v33);
    v35 = *v34;
    if ( v24 == *v34 )
      goto LABEL_24;
    ++v104;
  }
  v106 = *(_DWORD *)(v11 + 768);
  if ( !v105 )
    v105 = (unsigned __int8 *)v34;
  ++*(_QWORD *)(v11 + 752);
  v107 = v106 + 1;
  if ( 4 * (v106 + 1) >= 3 * v29 )
    goto LABEL_141;
  v108 = v24;
  if ( v29 - *(_DWORD *)(v11 + 772) - v107 <= v29 >> 3 )
  {
LABEL_142:
    sub_1917AE0(v11 + 752, v29);
    sub_190F2D0(v11 + 752, &v179, v181);
    v105 = v181[0];
    v108 = v179;
    v30 = v11 + 752;
    v107 = *(_DWORD *)(v11 + 768) + 1;
  }
  *(_DWORD *)(v11 + 768) = v107;
  if ( *(_QWORD *)v105 != -8 )
    --*(_DWORD *)(v11 + 772);
  *(_QWORD *)v105 = v108;
  *((_DWORD *)v105 + 2) = 0;
  v29 = *(_DWORD *)(v11 + 776);
  v180[0] = (__int64)v21;
  v32 = *(_QWORD *)(v11 + 760);
  if ( !v29 )
  {
    ++*(_QWORD *)(v11 + 752);
    goto LABEL_138;
  }
  v36 = 0;
  v31 = v29 - 1;
LABEL_25:
  v37 = v31 & v165;
  v38 = (unsigned __int8 *)(v32 + 16LL * (v31 & v165));
  v39 = *(_QWORD *)v38;
  if ( v21 != *(_QWORD **)v38 )
  {
    v156 = 1;
    v49 = 0;
    while ( v39 != -8 )
    {
      if ( !v49 && v39 == -16 )
        v49 = v38;
      v37 = v31 & (v156 + v37);
      v38 = (unsigned __int8 *)(v32 + 16LL * v37);
      v39 = *(_QWORD *)v38;
      if ( v21 == *(_QWORD **)v38 )
        goto LABEL_26;
      ++v156;
    }
    if ( !v49 )
      v49 = v38;
    v50 = *(_DWORD *)(v11 + 768);
    ++*(_QWORD *)(v11 + 752);
    v51 = v50 + 1;
    if ( 4 * v51 < 3 * v29 )
    {
      v52 = (__int64)v21;
      if ( v29 - (v51 + *(_DWORD *)(v11 + 772)) > v29 >> 3 )
      {
LABEL_44:
        *(_DWORD *)(v11 + 768) = v51;
        if ( *(_QWORD *)v49 != -8 )
          --*(_DWORD *)(v11 + 772);
        *(_QWORD *)v49 = v52;
        *((_DWORD *)v49 + 2) = 0;
        goto LABEL_47;
      }
LABEL_139:
      v159 = v30;
      sub_1917AE0(v30, v29);
      sub_190F2D0(v159, v180, v181);
      v49 = v181[0];
      v52 = v180[0];
      v51 = *(_DWORD *)(v11 + 768) + 1;
      goto LABEL_44;
    }
LABEL_138:
    v29 *= 2;
    goto LABEL_139;
  }
LABEL_26:
  if ( *((_DWORD *)v38 + 2) > v36 )
    goto LABEL_27;
LABEL_47:
  v53 = 24LL * (*((_DWORD *)a2 + 5) & 0xFFFFFFF);
  if ( (*((_BYTE *)a2 + 23) & 0x40) != 0 )
  {
    v54 = (_QWORD *)*(a2 - 1);
    v55 = &v54[(unsigned __int64)v53 / 8];
  }
  else
  {
    v55 = a2;
    v54 = &a2[v53 / 0xFFFFFFFFFFFFFFF8LL];
  }
  v56 = 0xAAAAAAAAAAAAAAABLL * (v53 >> 3);
  if ( !(v56 >> 2) )
  {
LABEL_61:
    if ( v56 != 2 )
    {
      if ( v56 != 3 )
      {
        if ( v56 != 1 )
          goto LABEL_27;
        goto LABEL_64;
      }
      if ( *(_BYTE *)(*v54 + 16LL) > 0x17u && v21 == *(_QWORD **)(*v54 + 40LL) )
        goto LABEL_66;
      v54 += 3;
    }
    if ( *(_BYTE *)(*v54 + 16LL) > 0x17u && v21 == *(_QWORD **)(*v54 + 40LL) )
      goto LABEL_66;
    v54 += 3;
LABEL_64:
    if ( *(_BYTE *)(*v54 + 16LL) <= 0x17u || v21 != *(_QWORD **)(*v54 + 40LL) )
      goto LABEL_27;
    goto LABEL_66;
  }
  v57 = &v54[12 * (v56 >> 2)];
  while ( *(_BYTE *)(*v54 + 16LL) <= 0x17u || v21 != *(_QWORD **)(*v54 + 40LL) )
  {
    v58 = v54[3];
    if ( *(_BYTE *)(v58 + 16) > 0x17u && v21 == *(_QWORD **)(v58 + 40) )
    {
      v54 += 3;
      break;
    }
    v59 = v54[6];
    if ( *(_BYTE *)(v59 + 16) > 0x17u && v21 == *(_QWORD **)(v59 + 40) )
    {
      v54 += 6;
      break;
    }
    v60 = v54[9];
    if ( *(_BYTE *)(v60 + 16) > 0x17u && v21 == *(_QWORD **)(v60 + 40) )
    {
      v54 += 9;
      break;
    }
    v54 += 12;
    if ( v54 == v57 )
    {
      v56 = 0xAAAAAAAAAAAAAAABLL * (v55 - v54);
      goto LABEL_61;
    }
  }
LABEL_66:
  if ( v55 != v54 )
    goto LABEL_35;
LABEL_27:
  v40 = sub_19170B0(v174, v24, (__int64)v21, v176);
  v41 = v24;
  v42 = sub_1910330(v11, v24, v40);
  if ( v42 )
  {
    if ( v42 != (_DWORD *)a2 )
    {
      v45 = (unsigned int)v184;
      if ( (unsigned int)v184 >= HIDWORD(v184) )
      {
        v41 = (__int64)v185;
        v158 = v42;
        sub_16CD150((__int64)&v183, v185, 0, 16, v43, v44);
        v45 = (unsigned int)v184;
        v42 = v158;
      }
      v46 = &v183[16 * v45];
      ++v166;
      *v46 = v42;
      v46[1] = v24;
      LODWORD(v184) = v184 + 1;
      goto LABEL_32;
    }
    goto LABEL_35;
  }
  v90 = (unsigned int)v184;
  if ( (unsigned int)v184 >= HIDWORD(v184) )
  {
    v41 = (__int64)v185;
    sub_16CD150((__int64)&v183, v185, 0, 16, v43, v44);
    v90 = (unsigned int)v184;
  }
  v91 = &v183[16 * v90];
  ++v160;
  *v91 = 0;
  v91[1] = v24;
  v163 = v24;
  LODWORD(v184) = v184 + 1;
LABEL_32:
  while ( 1 )
  {
    v22 = *(_QWORD *)(v22 + 8);
    if ( !v22 )
      break;
    v23 = sub_1648700(v22);
    if ( (unsigned __int8)(*((_BYTE *)v23 + 16) - 25) <= 9u )
      goto LABEL_34;
  }
  v61 = v160;
  v15 = v164;
  if ( v166 == 0 || v160 > 1 )
    goto LABEL_36;
  v161 = 0;
  if ( !v61 )
    goto LABEL_73;
  if ( !(unsigned __int8)sub_14AF470((__int64)a2, 0, 0, 0) )
  {
    v144 = *(unsigned int *)(v11 + 136);
    if ( (_DWORD)v144 )
    {
      v145 = *(_QWORD *)(v11 + 120);
      v146 = 1;
      for ( i = (v144 - 1) & v165; ; i = (v144 - 1) & v153 )
      {
        v148 = (_QWORD *)(v145 + 16LL * i);
        if ( v21 == (_QWORD *)*v148 )
          break;
        if ( *v148 == -8 )
          goto LABEL_186;
        v153 = v146 + i;
        ++v146;
      }
      if ( v148 != (_QWORD *)(v145 + 16 * v144) && (unsigned __int8)sub_1B29870(*(_QWORD *)(v11 + 144), v148[1], a2) )
      {
LABEL_212:
        v15 = 0;
        goto LABEL_36;
      }
    }
  }
LABEL_186:
  if ( *(_BYTE *)(sub_157EBA0(v163) + 16) == 28 )
    goto LABEL_36;
  v130 = (unsigned int)sub_137DFF0(v163, (__int64)v21);
  v131 = sub_157EBA0(v163);
  if ( (unsigned __int8)sub_137E040(v131, v130, 0) )
  {
    v151 = sub_157EBA0(v163);
    if ( *(_DWORD *)(v11 + 800) >= *(_DWORD *)(v11 + 804) )
      sub_16CD150(v11 + 792, (const void *)(v11 + 808), 0, 16, v149, v150);
    v152 = (unsigned __int64 *)(*(_QWORD *)(v11 + 792) + 16LL * *(unsigned int *)(v11 + 800));
    *v152 = v151;
    v152[1] = v130;
    ++*(_DWORD *)(v11 + 800);
    goto LABEL_212;
  }
  v41 = sub_15F4880((__int64)a2);
  v161 = v41;
  v15 = sub_1917860(v11, v41, v163, (__int64)v21);
  if ( !(_BYTE)v15 )
  {
    sub_164BEC0(v41, v41, v132, v133, a4, a5, a6, a7, v134, v135, a10, a11);
    goto LABEL_36;
  }
  if ( *(_QWORD *)v11 )
    sub_14139C0(*(_QWORD *)v11, v41);
LABEL_73:
  v62 = v21[6];
  if ( v62 )
    v62 -= 24;
  v167 = v62;
  v63 = sub_1649960((__int64)a2);
  v64 = v184;
  v65 = *a2;
  v180[0] = (__int64)v63;
  v182 = 773;
  v181[0] = (unsigned __int8 *)v180;
  v180[1] = v66;
  v181[1] = ".pre-phi";
  v67 = sub_1648B60(64);
  v70 = v167;
  v71 = v67;
  if ( v67 )
  {
    v168 = v67;
    sub_15F1EA0(v67, v65, 53, 0, 0, v70);
    *(_DWORD *)(v168 + 56) = v64;
    sub_164B780(v168, (__int64 *)v181);
    v41 = *(unsigned int *)(v168 + 56);
    sub_1648880(v168, v41, 1);
    v71 = v168;
  }
  v72 = 16LL * (unsigned int)v184;
  v169 = v72;
  v157 = v161 + 8;
  if ( (_DWORD)v184 )
  {
    v155 = v21;
    v73 = 0;
    v74 = v161;
    v154 = v11;
    v75 = v71;
    do
    {
      v84 = *(_QWORD *)&v183[v73];
      if ( v84 )
      {
        sub_1909530((__int64)a2, *(_QWORD *)&v183[v73]);
        v41 = *(_QWORD *)&v183[v73 + 8];
        v87 = *(_DWORD *)(v75 + 20) & 0xFFFFFFF;
        if ( v87 == *(_DWORD *)(v75 + 56) )
        {
          v162 = *(_QWORD *)&v183[v73 + 8];
          sub_15F55D0(v75, v41, v85, v68, v86, v70);
          v41 = v162;
          v87 = *(_DWORD *)(v75 + 20) & 0xFFFFFFF;
        }
        v88 = (v87 + 1) & 0xFFFFFFF;
        v89 = v88 | *(_DWORD *)(v75 + 20) & 0xF0000000;
        *(_DWORD *)(v75 + 20) = v89;
        if ( (v89 & 0x40000000) != 0 )
          v76 = *(_QWORD *)(v75 - 8);
        else
          v76 = v75 - 24 * v88;
        v77 = (_QWORD *)(v76 + 24LL * (unsigned int)(v88 - 1));
        if ( *v77 )
        {
          v78 = v77[1];
          v79 = v77[2] & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v79 = v78;
          if ( v78 )
          {
            v70 = *(_QWORD *)(v78 + 16) & 3LL;
            *(_QWORD *)(v78 + 16) = v70 | v79;
          }
        }
        *v77 = v84;
        v80 = *(_QWORD *)(v84 + 8);
        v77[1] = v80;
        if ( v80 )
        {
          v70 = (unsigned __int64)(v77 + 1) | *(_QWORD *)(v80 + 16) & 3LL;
          *(_QWORD *)(v80 + 16) = v70;
        }
        v69 = v77[2] & 3LL;
        v77[2] = v69 | (v84 + 8);
        *(_QWORD *)(v84 + 8) = v77;
        v81 = *(_DWORD *)(v75 + 20) & 0xFFFFFFF;
        v82 = (unsigned int)(v81 - 1);
        if ( (*(_BYTE *)(v75 + 23) & 0x40) != 0 )
          v83 = *(_QWORD *)(v75 - 8);
        else
          v83 = v75 - 24 * v81;
        v72 = 3LL * *(unsigned int *)(v75 + 56);
        *(_QWORD *)(v83 + 8 * v82 + 24LL * *(unsigned int *)(v75 + 56) + 8) = v41;
      }
      else
      {
        v92 = *(_DWORD *)(v75 + 20) & 0xFFFFFFF;
        if ( v92 == *(_DWORD *)(v75 + 56) )
        {
          sub_15F55D0(v75, v41, v72, v68, v69, v70);
          v92 = *(_DWORD *)(v75 + 20) & 0xFFFFFFF;
        }
        v93 = (v92 + 1) & 0xFFFFFFF;
        v94 = v93 | *(_DWORD *)(v75 + 20) & 0xF0000000;
        *(_DWORD *)(v75 + 20) = v94;
        if ( (v94 & 0x40000000) != 0 )
          v95 = *(_QWORD *)(v75 - 8);
        else
          v95 = v75 - 24 * v93;
        v96 = (__int64 *)(v95 + 24LL * (unsigned int)(v93 - 1));
        if ( *v96 )
        {
          v97 = v96[1];
          v98 = v96[2] & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v98 = v97;
          if ( v97 )
            *(_QWORD *)(v97 + 16) = *(_QWORD *)(v97 + 16) & 3LL | v98;
        }
        *v96 = v74;
        if ( v74 )
        {
          v99 = *(_QWORD *)(v74 + 8);
          v96[1] = v99;
          if ( v99 )
            *(_QWORD *)(v99 + 16) = (unsigned __int64)(v96 + 1) | *(_QWORD *)(v99 + 16) & 3LL;
          v96[2] = v157 | v96[2] & 3;
          *(_QWORD *)(v74 + 8) = v96;
        }
        v100 = *(_DWORD *)(v75 + 20) & 0xFFFFFFF;
        v101 = (unsigned int)(v100 - 1);
        if ( (*(_BYTE *)(v75 + 23) & 0x40) != 0 )
          v41 = *(_QWORD *)(v75 - 8);
        else
          v41 = v75 - 24 * v100;
        v72 = 3LL * *(unsigned int *)(v75 + 56);
        *(_QWORD *)(v41 + 8 * v101 + 24LL * *(unsigned int *)(v75 + 56) + 8) = v163;
      }
      v73 += 16;
    }
    while ( v169 != v73 );
    v71 = v75;
    v21 = v155;
    v11 = v154;
  }
  v170 = v71;
  sub_19110A0(v174, v71, v176);
  sub_190B100(v174, v176, (__int64)v21);
  sub_1910810(v11, v176, v170, (__int64)v21);
  v112 = (unsigned __int8 *)a2[6];
  v113 = v170;
  v181[0] = v112;
  v114 = (unsigned __int8 **)(v170 + 48);
  if ( v112 )
  {
    sub_1623A60((__int64)v181, (__int64)v112, 2);
    v113 = v170;
    if ( v114 == v181 )
      goto LABEL_156;
  }
  else if ( v114 == v181 )
  {
    goto LABEL_158;
  }
  v171 = v113;
  sub_19094F0((__int64 *)v114, v181);
  v113 = v171;
LABEL_156:
  if ( v181[0] )
  {
    v172 = v113;
    sub_161E7C0((__int64)v181, (__int64)v181[0]);
    v113 = v172;
  }
LABEL_158:
  v173 = (__int64 *)v113;
  sub_164D160((__int64)a2, v113, a4, a5, a6, a7, v110, v111, a10, a11);
  if ( *(_QWORD *)v11 )
  {
    v115 = *(_BYTE *)(*v173 + 8);
    if ( v115 == 16 )
      v115 = *(_BYTE *)(**(_QWORD **)(*v173 + 16) + 8LL);
    if ( v115 == 15 )
      sub_14134C0(*(_QWORD *)v11, v173);
  }
  sub_190ACD0(v174, (__int64)a2);
  LODWORD(v181[0]) = v176;
  v116 = sub_190FE40(v11 + 376, (int *)v181);
  v117 = 0;
  for ( j = v116 + 2; a2 != (_QWORD *)*j || v21 != (_QWORD *)j[1]; j = (_QWORD *)j[2] )
  {
    v117 = j;
    if ( !j[2] )
      goto LABEL_170;
  }
  v119 = (_QWORD *)j[2];
  if ( v117 )
  {
    v117[2] = v119;
  }
  else if ( v119 )
  {
    *j = *v119;
    j[1] = v119[1];
    j[2] = v119[2];
    j[3] = v119[3];
  }
  else
  {
    *j = 0;
    j[1] = 0;
  }
LABEL_170:
  if ( *(_QWORD *)v11 )
    sub_14191F0(*(_QWORD *)v11, (__int64)a2);
  v180[0] = a2[5];
  if ( (unsigned __int8)sub_190CC30(v11 + 112, v180, v181) )
  {
    v120 = *(_QWORD *)(v11 + 144);
    v121 = (_QWORD *)*((_QWORD *)v181[0] + 1);
    v122 = *(_DWORD *)(v120 + 24);
    if ( !v122 )
    {
LABEL_181:
      v15 = 1;
      sub_15F20C0(a2);
      if ( a2 == v121 )
        sub_1918240(v11, (__int64)v21);
      goto LABEL_36;
    }
    v123 = *(_QWORD *)(v120 + 8);
LABEL_175:
    v124 = v122 - 1;
    v125 = 1;
    v126 = v124 & v165;
    v127 = (_QWORD *)(v123 + 16LL * (v124 & v165));
    v128 = (_QWORD *)*v127;
    if ( v21 == (_QWORD *)*v127 )
    {
LABEL_176:
      v129 = v127[1];
      if ( v129 )
      {
        if ( (*(_BYTE *)(v129 + 8) & 1) == 0 )
        {
          v175 = v120;
          v177 = v127[1];
          j___libc_free_0(*(_QWORD *)(v129 + 16));
          v120 = v175;
          v129 = v177;
        }
        v178 = v120;
        j_j___libc_free_0(v129, 552);
        v120 = v178;
      }
      *v127 = -16;
      --*(_DWORD *)(v120 + 16);
      ++*(_DWORD *)(v120 + 20);
    }
    else
    {
      while ( v128 != (_QWORD *)-8LL )
      {
        v126 = v124 & (v125 + v126);
        v127 = (_QWORD *)(v123 + 16LL * v126);
        v128 = (_QWORD *)*v127;
        if ( v21 == (_QWORD *)*v127 )
          goto LABEL_176;
        ++v125;
      }
    }
    goto LABEL_181;
  }
  v120 = *(_QWORD *)(v11 + 144);
  v122 = *(_DWORD *)(v120 + 24);
  if ( v122 )
  {
    v123 = *(_QWORD *)(v120 + 8);
    v121 = 0;
    goto LABEL_175;
  }
  v15 = 1;
  sub_15F20C0(a2);
LABEL_36:
  if ( v183 != v185 )
    _libc_free((unsigned __int64)v183);
  return v15;
}
