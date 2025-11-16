// Function: sub_19A2ED0
// Address: 0x19a2ed0
//
__int64 __fastcall sub_19A2ED0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __m128i a7,
        __m128i a8,
        __int64 a9)
{
  __int64 v9; // r12
  __int64 v12; // r12
  _QWORD *v13; // rdx
  _QWORD *v14; // rax
  _QWORD *v15; // rbx
  __int64 v16; // rax
  __int64 v17; // rax
  unsigned __int64 v18; // rbx
  __int64 v19; // rax
  __int64 *v20; // rax
  __int64 v21; // rdx
  __int64 *v22; // r12
  __int64 v23; // rdi
  __int64 *v24; // rbx
  __int64 v25; // rbx
  unsigned __int64 v26; // rbx
  __int64 *v27; // r14
  __int64 v28; // r15
  __int64 v29; // r12
  __int64 v30; // rax
  int v31; // r8d
  __int64 v32; // rcx
  unsigned int v33; // edx
  __int64 *v34; // rax
  __int64 v35; // rdi
  _QWORD **v36; // r11
  unsigned int i; // esi
  _QWORD *v38; // rax
  __int64 v39; // rdx
  __int64 v40; // rax
  __int64 v41; // r10
  unsigned int v42; // edi
  __int64 *v43; // rdx
  __int64 v44; // r14
  __int64 *v45; // rcx
  int v46; // r10d
  __int64 v47; // rdi
  unsigned int v48; // edx
  __int64 *v49; // rax
  __int64 v50; // rbx
  _QWORD **v51; // rbx
  _QWORD *v52; // rax
  unsigned int j; // edx
  unsigned int v54; // ecx
  __int64 v55; // rdx
  __int64 v56; // r8
  int v57; // eax
  int v58; // edi
  unsigned int v59; // ecx
  __int64 v60; // rsi
  __int64 v61; // r12
  __int64 v62; // r10
  unsigned int v63; // esi
  int v64; // r11d
  __int64 v65; // rsi
  __int64 v66; // rsi
  unsigned __int8 *v67; // rsi
  __int64 v68; // rdi
  __int64 v69; // r12
  __int64 v70; // rbx
  __int64 v71; // r8
  __int64 *v72; // r12
  __int64 v73; // rax
  __int64 v74; // rbx
  __int64 v75; // rax
  int v76; // r8d
  __int64 v77; // r9
  __int64 v78; // rax
  __int64 v79; // rsi
  __int64 v80; // rbx
  __int64 v81; // rsi
  __int64 v82; // rdi
  __int64 v83; // rsi
  __int64 v84; // rax
  int v85; // eax
  __int64 *v86; // rax
  __int64 v87; // rdi
  unsigned int v88; // eax
  __int64 *v89; // r14
  __int64 v90; // rsi
  __int64 v91; // rdx
  __int64 v92; // rdx
  __int64 v93; // r10
  __int64 v94; // rdx
  int v95; // ebx
  unsigned int v96; // ecx
  __int64 *v97; // r8
  __int64 v98; // r12
  __int64 *v99; // rcx
  unsigned int v100; // r8d
  __int64 *v101; // rdx
  __int64 v102; // r12
  __int64 v103; // rdx
  __int64 v104; // rcx
  int v105; // r8d
  unsigned __int64 v106; // r14
  __int64 v107; // rax
  int v108; // edx
  __int64 v109; // rax
  __int64 *v110; // rax
  int v111; // r8d
  __int64 v112; // rdx
  __int64 v113; // rcx
  __int64 v114; // rbx
  __int64 v115; // r14
  __int64 v116; // r15
  unsigned int v117; // eax
  _QWORD *v118; // rdi
  __int64 v119; // r15
  int v120; // eax
  __int64 v121; // rdx
  unsigned __int64 v122; // rcx
  __int64 **v123; // rdx
  __int64 *v124; // rax
  __int64 v125; // rax
  __int64 v126; // rdi
  __int64 v127; // rsi
  int v128; // eax
  __int64 v129; // r14
  __int64 v130; // rax
  __int64 v131; // r14
  __int64 v132; // rax
  __int64 v133; // rax
  __int64 v134; // rax
  __int64 v135; // rax
  __int64 *v136; // rax
  __int64 v137; // rax
  __int64 v138; // rdi
  __int64 v139; // rax
  __int64 ***v140; // rax
  __int64 v141; // r13
  int v142; // eax
  __int64 v143; // rdx
  unsigned __int64 v144; // rcx
  __int64 v145; // rdx
  int v146; // r12d
  __int64 v147; // r14
  __int64 v148; // rax
  int v149; // edx
  int v150; // ebx
  int v151; // eax
  _QWORD *v152; // rdx
  int v153; // r10d
  __int64 *v154; // rax
  __int64 v155; // rax
  __int64 v156; // rdi
  __int64 v157; // [rsp+8h] [rbp-128h]
  __int64 v158; // [rsp+10h] [rbp-120h]
  __int64 v159; // [rsp+20h] [rbp-110h]
  int v160; // [rsp+20h] [rbp-110h]
  int v161; // [rsp+20h] [rbp-110h]
  __int64 **v162; // [rsp+28h] [rbp-108h]
  __int64 v163; // [rsp+30h] [rbp-100h]
  __int64 v164; // [rsp+30h] [rbp-100h]
  __int64 v165; // [rsp+30h] [rbp-100h]
  __int64 v167; // [rsp+40h] [rbp-F0h]
  __int64 v168; // [rsp+40h] [rbp-F0h]
  __int64 *v169; // [rsp+40h] [rbp-F0h]
  char *v172; // [rsp+58h] [rbp-D8h]
  __int64 *v174; // [rsp+60h] [rbp-D0h]
  __int64 ***v175; // [rsp+60h] [rbp-D0h]
  __int64 v176; // [rsp+60h] [rbp-D0h]
  __int64 v177; // [rsp+68h] [rbp-C8h]
  __int64 v178; // [rsp+78h] [rbp-B8h] BYREF
  char *v179; // [rsp+80h] [rbp-B0h] BYREF
  __int64 v180; // [rsp+88h] [rbp-A8h]
  _QWORD v181[4]; // [rsp+90h] [rbp-A0h] BYREF
  unsigned __int8 *v182; // [rsp+B0h] [rbp-80h] BYREF
  __int64 v183; // [rsp+B8h] [rbp-78h]
  _BYTE v184[112]; // [rsp+C0h] [rbp-70h] BYREF

  v9 = *(_QWORD *)(a3 + 8);
  v177 = a6;
  if ( *(_BYTE *)(a2 + 729) )
    return v9;
  v179 = (char *)v181;
  v180 = 0x400000000LL;
  if ( *(_BYTE *)(v9 + 16) > 0x17u )
  {
    v181[0] = v9;
    LODWORD(v180) = 1;
  }
  if ( *(_DWORD *)(a2 + 32) == 3 )
  {
    v114 = *(_QWORD *)(*(_QWORD *)a3 - 24LL);
    if ( !v114 )
      BUG();
    if ( *(_BYTE *)(v114 + 16) > 0x17u )
    {
      v181[(unsigned int)v180] = v114;
      LODWORD(v180) = v180 + 1;
    }
  }
  v12 = *(_QWORD *)(a1 + 40);
  v13 = *(_QWORD **)(a3 + 32);
  v14 = *(_QWORD **)(a3 + 24);
  v163 = a3 + 16;
  if ( v13 == v14 )
  {
    v15 = &v14[*(unsigned int *)(a3 + 44)];
    if ( v14 == v15 )
    {
      v152 = *(_QWORD **)(a3 + 24);
    }
    else
    {
      do
      {
        if ( v12 == *v14 )
          break;
        ++v14;
      }
      while ( v15 != v14 );
      v152 = v15;
    }
  }
  else
  {
    v15 = &v13[*(unsigned int *)(a3 + 40)];
    v14 = sub_16CC9F0(v163, v12);
    if ( v12 == *v14 )
    {
      v112 = *(_QWORD *)(a3 + 32);
      if ( v112 == *(_QWORD *)(a3 + 24) )
        v113 = *(unsigned int *)(a3 + 44);
      else
        v113 = *(unsigned int *)(a3 + 40);
      v152 = (_QWORD *)(v112 + 8 * v113);
    }
    else
    {
      v16 = *(_QWORD *)(a3 + 32);
      if ( v16 != *(_QWORD *)(a3 + 24) )
      {
        v14 = (_QWORD *)(v16 + 8LL * *(unsigned int *)(a3 + 40));
        goto LABEL_10;
      }
      v14 = (_QWORD *)(v16 + 8LL * *(unsigned int *)(a3 + 44));
      v152 = v14;
    }
  }
  while ( v152 != v14 && *v14 >= 0xFFFFFFFFFFFFFFFELL )
    ++v14;
LABEL_10:
  if ( v14 != v15 )
  {
    if ( sub_19A2CE0((_QWORD *)a3, *(_QWORD *)(a1 + 40)) )
    {
      v17 = sub_13FCB50(*(_QWORD *)(a1 + 40));
      v18 = sub_157EBA0(v17);
      v19 = (unsigned int)v180;
      if ( (unsigned int)v180 >= HIDWORD(v180) )
      {
        sub_16CD150((__int64)&v179, v181, 0, 8, (int)&v179, a6);
        v19 = (unsigned int)v180;
      }
      *(_QWORD *)&v179[8 * v19] = v18;
      LODWORD(v180) = v180 + 1;
    }
    else
    {
      v109 = (unsigned int)v180;
      if ( (unsigned int)v180 >= HIDWORD(v180) )
      {
        sub_16CD150((__int64)&v179, v181, 0, 8, (int)&v179, a6);
        v109 = (unsigned int)v180;
      }
      *(_QWORD *)&v179[8 * v109] = *(_QWORD *)(a1 + 56);
      LODWORD(v180) = v180 + 1;
    }
  }
  v20 = *(__int64 **)(a3 + 32);
  if ( v20 == *(__int64 **)(a3 + 24) )
    v21 = *(unsigned int *)(a3 + 44);
  else
    v21 = *(unsigned int *)(a3 + 40);
  v22 = &v20[v21];
  if ( v20 == v22 )
    goto LABEL_20;
  while ( 1 )
  {
    v23 = *v20;
    v24 = v20;
    if ( (unsigned __int64)*v20 < 0xFFFFFFFFFFFFFFFELL )
      break;
    if ( v22 == ++v20 )
      goto LABEL_20;
  }
  if ( v22 == v20 )
    goto LABEL_20;
  while ( 2 )
  {
    if ( *(_QWORD *)(a1 + 40) == v23 )
      goto LABEL_140;
    v182 = v184;
    v183 = 0x400000000LL;
    sub_13F9CA0(v23, (__int64)&v182);
    LODWORD(a6) = v183;
    if ( !(_DWORD)v183 )
      goto LABEL_138;
    v87 = *(_QWORD *)v182;
    if ( (_DWORD)v183 == 1 )
      goto LABEL_166;
    v169 = v24;
    v88 = 1;
    v89 = v22;
    while ( 1 )
    {
      while ( 1 )
      {
LABEL_148:
        v90 = *(_QWORD *)&v182[8 * v88];
        v91 = *(_QWORD *)(*(_QWORD *)(v87 + 56) + 80LL);
        if ( v91 )
          v91 -= 24;
        if ( v91 != v87 && v90 != v91 )
          break;
        ++v88;
        v87 = v91;
        if ( (_DWORD)v183 == v88 )
          goto LABEL_165;
      }
      v92 = *(_QWORD *)(a1 + 16);
      v93 = *(_QWORD *)(v92 + 32);
      v94 = *(unsigned int *)(v92 + 48);
      if ( !(_DWORD)v94 )
        goto LABEL_175;
      v95 = v94 - 1;
      v96 = (v94 - 1) & (((unsigned int)v87 >> 9) ^ ((unsigned int)v87 >> 4));
      v97 = (__int64 *)(v93 + 16LL * v96);
      v98 = *v97;
      if ( v87 == *v97 )
      {
LABEL_154:
        v99 = (__int64 *)(v93 + 16 * v94);
        if ( v97 != v99 )
        {
          v87 = v97[1];
          goto LABEL_156;
        }
      }
      else
      {
        v111 = 1;
        while ( v98 != -8 )
        {
          v96 = v95 & (v111 + v96);
          v161 = v111 + 1;
          v97 = (__int64 *)(v93 + 16LL * v96);
          v98 = *v97;
          if ( *v97 == v87 )
            goto LABEL_154;
          v111 = v161;
        }
        v99 = (__int64 *)(v93 + 16LL * (unsigned int)v94);
      }
      v87 = 0;
LABEL_156:
      v100 = v95 & (((unsigned int)v90 >> 9) ^ ((unsigned int)v90 >> 4));
      v101 = (__int64 *)(v93 + 16LL * v100);
      v102 = *v101;
      if ( v90 != *v101 )
      {
        v108 = 1;
        while ( v102 != -8 )
        {
          v100 = v95 & (v108 + v100);
          v160 = v108 + 1;
          v101 = (__int64 *)(v93 + 16LL * v100);
          v102 = *v101;
          if ( v90 == *v101 )
            goto LABEL_157;
          v108 = v160;
        }
        goto LABEL_175;
      }
LABEL_157:
      if ( v101 != v99 )
      {
        v103 = v101[1];
        if ( v87 )
        {
          if ( v103 )
            break;
        }
      }
LABEL_175:
      ++v88;
      v87 = 0;
      if ( (_DWORD)v183 == v88 )
        goto LABEL_165;
    }
    while ( v87 != v103 )
    {
      if ( *(_DWORD *)(v87 + 16) < *(_DWORD *)(v103 + 16) )
      {
        v104 = v87;
        v87 = v103;
        v103 = v104;
      }
      v87 = *(_QWORD *)(v87 + 8);
      if ( !v87 )
      {
        if ( (_DWORD)v183 != ++v88 )
          goto LABEL_148;
        goto LABEL_165;
      }
    }
    ++v88;
    v87 = *(_QWORD *)v87;
    if ( (_DWORD)v183 != v88 )
      goto LABEL_148;
LABEL_165:
    v24 = v169;
    v22 = v89;
LABEL_166:
    v106 = sub_157EBA0(v87);
    v107 = (unsigned int)v180;
    if ( (unsigned int)v180 >= HIDWORD(v180) )
    {
      sub_16CD150((__int64)&v179, v181, 0, 8, v105, a6);
      v107 = (unsigned int)v180;
    }
    *(_QWORD *)&v179[8 * v107] = v106;
    LODWORD(v180) = v180 + 1;
LABEL_138:
    if ( v182 != v184 )
      _libc_free((unsigned __int64)v182);
LABEL_140:
    v86 = v24 + 1;
    if ( v24 + 1 != v22 )
    {
      v23 = *v86;
      ++v24;
      if ( (unsigned __int64)*v86 < 0xFFFFFFFFFFFFFFFELL )
      {
LABEL_144:
        if ( v22 == v24 )
          break;
        continue;
      }
      while ( v22 != ++v86 )
      {
        v23 = *v86;
        v24 = v86;
        if ( (unsigned __int64)*v86 < 0xFFFFFFFFFFFFFFFELL )
          goto LABEL_144;
      }
    }
    break;
  }
LABEL_20:
  if ( !a5 )
LABEL_262:
    BUG();
  v25 = a5;
  if ( *(_BYTE *)(a5 - 8) == 34 )
    goto LABEL_57;
  v167 = a5;
  v26 = a5 - 24;
LABEL_23:
  v27 = (__int64 *)v179;
  v172 = &v179[8 * (unsigned int)v180];
  if ( v179 == v172 )
    goto LABEL_195;
  v28 = 0;
  do
  {
    while ( 1 )
    {
      v29 = *v27;
      if ( *v27 == v26 || !sub_15CCEE0(*(_QWORD *)(a1 + 16), *v27, v26) )
        goto LABEL_55;
      if ( *(_QWORD *)(v26 + 40) == *(_QWORD *)(v29 + 40) && (!v28 || !sub_15CCEE0(*(_QWORD *)(a1 + 16), v29, v28)) )
        break;
      if ( v172 == (char *)++v27 )
        goto LABEL_34;
    }
    v28 = *(_QWORD *)(v29 + 32);
    if ( v28 )
      v28 -= 24;
    ++v27;
  }
  while ( v172 != (char *)v27 );
LABEL_34:
  v167 = v28 + 24;
  if ( !v28 )
LABEL_195:
    v167 = v26 + 24;
  v30 = *(_QWORD *)(a1 + 24);
  v31 = *(_DWORD *)(v30 + 24);
  v32 = *(_QWORD *)(v167 + 16);
  a6 = *(_QWORD *)(v30 + 8);
  if ( v31 )
  {
    v33 = (v31 - 1) & (((unsigned int)v32 >> 9) ^ ((unsigned int)v32 >> 4));
    v34 = (__int64 *)(a6 + 16LL * v33);
    v35 = *v34;
    if ( *v34 == v32 )
    {
LABEL_37:
      v36 = (_QWORD **)v34[1];
      i = 0;
      if ( v36 )
      {
        v38 = *v36;
        for ( i = 1; v38; ++i )
          v38 = (_QWORD *)*v38;
      }
    }
    else
    {
      v151 = 1;
      while ( v35 != -8 )
      {
        v153 = v151 + 1;
        v33 = (v31 - 1) & (v151 + v33);
        v34 = (__int64 *)(a6 + 16LL * v33);
        v35 = *v34;
        if ( v32 == *v34 )
          goto LABEL_37;
        v151 = v153;
      }
      v36 = 0;
      i = 0;
    }
  }
  else
  {
    i = 0;
    v36 = 0;
  }
  v39 = *(_QWORD *)(a1 + 16);
  v40 = *(unsigned int *)(v39 + 48);
  if ( (_DWORD)v40 )
  {
    v41 = *(_QWORD *)(v39 + 32);
    v42 = (v40 - 1) & (((unsigned int)v32 >> 9) ^ ((unsigned int)v32 >> 4));
    v43 = (__int64 *)(v41 + 16LL * v42);
    v44 = *v43;
    if ( *v43 == v32 )
    {
LABEL_42:
      if ( v43 != (__int64 *)(v41 + 16 * v40) )
      {
        v45 = (__int64 *)v43[1];
        if ( v45 )
        {
          v46 = v31 - 1;
          while ( 1 )
          {
            v45 = (__int64 *)v45[1];
            if ( !v45 )
              goto LABEL_55;
            v47 = *v45;
            if ( !v31 )
              goto LABEL_133;
            v48 = v46 & (((unsigned int)v47 >> 9) ^ ((unsigned int)v47 >> 4));
            v49 = (__int64 *)(a6 + 16LL * v48);
            v50 = *v49;
            if ( v47 != *v49 )
              break;
LABEL_48:
            v51 = (_QWORD **)v49[1];
            if ( v51 )
            {
              v52 = *v51;
              for ( j = 1; v52; ++j )
                v52 = (_QWORD *)*v52;
              if ( i >= j )
                goto LABEL_52;
            }
            else
            {
              j = 0;
LABEL_52:
              if ( i != j || v36 == v51 )
              {
                v26 = sub_157EBA0(v47);
                if ( *(_BYTE *)(v26 + 16) != 34 )
                  goto LABEL_23;
                goto LABEL_55;
              }
            }
          }
          v85 = 1;
          while ( v50 != -8 )
          {
            v146 = v85 + 1;
            v48 = v46 & (v85 + v48);
            v49 = (__int64 *)(a6 + 16LL * v48);
            v50 = *v49;
            if ( v47 == *v49 )
              goto LABEL_48;
            v85 = v146;
          }
LABEL_133:
          j = 0;
          v51 = 0;
          goto LABEL_52;
        }
      }
    }
    else
    {
      v149 = 1;
      while ( v44 != -8 )
      {
        v150 = v149 + 1;
        v42 = (v40 - 1) & (v149 + v42);
        v43 = (__int64 *)(v41 + 16LL * v42);
        v44 = *v43;
        if ( v32 == *v43 )
          goto LABEL_42;
        v149 = v150;
      }
    }
  }
LABEL_55:
  v25 = v167;
  while ( *(_BYTE *)(v25 - 8) == 77 )
  {
    v25 = *(_QWORD *)(v25 + 8);
    if ( !v25 )
      goto LABEL_262;
LABEL_57:
    ;
  }
  while ( 1 )
  {
    v54 = *(unsigned __int8 *)(v25 - 8) - 34;
    if ( v54 > 0x36 || ((1LL << v54) & 0x40018000000001LL) == 0 )
      break;
    v25 = *(_QWORD *)(v25 + 8);
    if ( !v25 )
      goto LABEL_262;
  }
  while ( *(_BYTE *)(v25 - 8) == 78 )
  {
    v84 = *(_QWORD *)(v25 - 48);
    if ( *(_BYTE *)(v84 + 16) || (*(_BYTE *)(v84 + 33) & 0x20) == 0 || (unsigned int)(*(_DWORD *)(v84 + 36) - 35) > 3 )
      break;
    v25 = *(_QWORD *)(v25 + 8);
    if ( !v25 )
      goto LABEL_262;
  }
  v55 = 0;
  v56 = *(_QWORD *)(v177 + 64);
  v57 = *(_DWORD *)(v177 + 80);
  v58 = v57 - 1;
  while ( 2 )
  {
    v61 = v25 - 24;
    if ( !v25 )
      v61 = 0;
    if ( v57 )
    {
      LODWORD(a6) = 1;
      v59 = v58 & (((unsigned int)v61 >> 9) ^ ((unsigned int)v61 >> 4));
      v60 = *(_QWORD *)(v56 + 8LL * v59);
      if ( v61 != v60 )
      {
        while ( v60 != -8 )
        {
          v59 = v58 & (a6 + v59);
          v60 = *(_QWORD *)(v56 + 8LL * v59);
          if ( v61 == v60 )
            goto LABEL_63;
          LODWORD(a6) = a6 + 1;
        }
        break;
      }
LABEL_63:
      if ( a5 == v25 )
        goto LABEL_71;
      v25 = *(_QWORD *)(v25 + 8);
      continue;
    }
    break;
  }
  v59 = *(_DWORD *)(v177 + 112);
  if ( !v59 )
    goto LABEL_71;
  v62 = *(_QWORD *)(v177 + 96);
  v63 = v59 - 1;
  v64 = 1;
  v59 = (v59 - 1) & (((unsigned int)v61 >> 9) ^ ((unsigned int)v61 >> 4));
  a6 = *(_QWORD *)(v62 + 8LL * v59);
  if ( v61 == a6 )
    goto LABEL_63;
  while ( a6 != -8 )
  {
    v59 = v63 & (v64 + v59);
    a6 = *(_QWORD *)(v62 + 8LL * v59);
    if ( v61 == a6 )
      goto LABEL_63;
    ++v64;
  }
LABEL_71:
  if ( v179 != (char *)v181 )
    _libc_free((unsigned __int64)v179);
  *(_QWORD *)(v177 + 272) = *(_QWORD *)(v61 + 40);
  *(_QWORD *)(v177 + 280) = v61 + 24;
  v65 = *(_QWORD *)(v61 + 48);
  v182 = (unsigned __int8 *)v65;
  if ( v65 )
  {
    sub_1623A60((__int64)&v182, v65, 2);
    v66 = *(_QWORD *)(v177 + 264);
    if ( v66 )
      goto LABEL_75;
LABEL_76:
    v67 = v182;
    *(_QWORD *)(v177 + 264) = v182;
    if ( v67 )
      sub_1623210((__int64)&v182, v67, v177 + 264);
  }
  else
  {
    v66 = *(_QWORD *)(v177 + 264);
    if ( v66 )
    {
LABEL_75:
      sub_161E7C0(v177 + 264, v66);
      goto LABEL_76;
    }
  }
  v168 = v177 + 152;
  if ( v177 + 152 != v163 )
    sub_16CCD50(v168, v163, v55, v59, v56, a6);
  v162 = **(__int64 ****)(a3 + 8);
  if ( *(_DWORD *)(a4 + 40) )
  {
    v69 = sub_1456040(**(_QWORD **)(a4 + 32));
LABEL_83:
    if ( !v69 )
      goto LABEL_184;
  }
  else
  {
    v68 = *(_QWORD *)(a4 + 80);
    if ( v68 )
    {
      v69 = sub_1456040(v68);
      goto LABEL_83;
    }
    v110 = *(__int64 **)a4;
    if ( !*(_QWORD *)a4 || (v69 = *v110) == 0 )
    {
LABEL_184:
      v69 = (__int64)v162;
      goto LABEL_86;
    }
  }
  v70 = sub_1456E10(*(_QWORD *)(a1 + 8), v69);
  if ( v70 == sub_1456E10(*(_QWORD *)(a1 + 8), (__int64)v162) )
    v69 = (__int64)v162;
LABEL_86:
  v159 = sub_1456E10(*(_QWORD *)(a1 + 8), v69);
  v182 = v184;
  v183 = 0x800000000LL;
  v71 = *(_QWORD *)(a4 + 32);
  if ( v71 != v71 + 8LL * *(unsigned int *)(a4 + 40) )
  {
    v174 = (__int64 *)(v71 + 8LL * *(unsigned int *)(a4 + 40));
    v158 = v69;
    v72 = *(__int64 **)(a4 + 32);
    do
    {
      v73 = sub_1499A20(*v72, v163, *(_QWORD **)(a1 + 8), a7, a8);
      v74 = *(_QWORD *)(a1 + 8);
      v75 = sub_38761C0(v177, v73, 0);
      v77 = sub_145DC80(v74, v75);
      v78 = (unsigned int)v183;
      if ( (unsigned int)v183 >= HIDWORD(v183) )
      {
        v157 = v77;
        sub_16CD150((__int64)&v182, v184, 0, 8, v76, v77);
        v78 = (unsigned int)v183;
        v77 = v157;
      }
      ++v72;
      *(_QWORD *)&v182[8 * v78] = v77;
      LODWORD(v183) = v183 + 1;
    }
    while ( v174 != v72 );
    v69 = v158;
  }
  if ( *(_QWORD *)(a4 + 24) )
  {
    v178 = sub_1499A20(*(_QWORD *)(a4 + 80), v163, *(_QWORD **)(a1 + 8), a7, a8);
    v127 = v178;
    v128 = *(_DWORD *)(a2 + 32);
    if ( v128 != 3 )
    {
      if ( (_DWORD)v183 && v128 == 2 )
      {
        if ( sub_19937D0(*(__int64 **)(a1 + 32), a2, a4) )
        {
          v154 = sub_147DD40(*(_QWORD *)(a1 + 8), (__int64 *)&v182, 0, 0, a7, a8);
          v155 = sub_38761C0(v177, v154, 0);
          v156 = *(_QWORD *)(a1 + 8);
          LODWORD(v183) = 0;
          v179 = (char *)sub_145DC80(v156, v155);
          sub_1458920((__int64)&v182, &v179);
        }
        v127 = v178;
      }
      v129 = *(_QWORD *)(a1 + 8);
      v130 = sub_38761C0(v177, v127, 0);
      v178 = sub_145DC80(v129, v130);
      v176 = *(_QWORD *)(a4 + 24);
      if ( v176 != 1 )
      {
        v131 = *(_QWORD *)(a1 + 8);
        v132 = sub_1456040(v178);
        v133 = sub_145CF80(v131, v132, v176, 0);
        v178 = sub_13A5B60(v131, v178, v133, 0, 0);
      }
      sub_1458920((__int64)&v182, &v178);
      goto LABEL_93;
    }
    if ( *(_QWORD *)(a4 + 24) == 1 )
    {
      v147 = *(_QWORD *)(a1 + 8);
      v148 = sub_38761C0(v177, v178, 0);
      v179 = (char *)sub_145DC80(v147, v148);
      sub_1458920((__int64)&v182, &v179);
      v175 = 0;
    }
    else
    {
      v175 = (__int64 ***)sub_38761C0(v177, v178, 0);
    }
  }
  else
  {
LABEL_93:
    v175 = 0;
  }
  v79 = *(_QWORD *)a4;
  if ( *(_QWORD *)a4 )
  {
    if ( (_DWORD)v183 )
    {
      v136 = sub_147DD40(*(_QWORD *)(a1 + 8), (__int64 *)&v182, 0, 0, a7, a8);
      v137 = sub_38761C0(v177, v136, v69);
      v138 = *(_QWORD *)(a1 + 8);
      LODWORD(v183) = 0;
      v179 = (char *)sub_145DC80(v138, v137);
      sub_1458920((__int64)&v182, &v179);
      v79 = *(_QWORD *)a4;
    }
    v179 = (char *)sub_145DC80(*(_QWORD *)(a1 + 8), v79);
    sub_1458920((__int64)&v182, &v179);
  }
  if ( (_DWORD)v183 )
  {
    v124 = sub_147DD40(*(_QWORD *)(a1 + 8), (__int64 *)&v182, 0, 0, a7, a8);
    v125 = sub_38761C0(v177, v124, v69);
    v126 = *(_QWORD *)(a1 + 8);
    LODWORD(v183) = 0;
    v179 = (char *)sub_145DC80(v126, v125);
    sub_1458920((__int64)&v182, &v179);
  }
  v80 = *(_QWORD *)(a4 + 8) + *(_QWORD *)(a3 + 72);
  if ( v80 )
  {
    if ( *(_DWORD *)(a2 + 32) == 3 )
    {
      if ( v175 )
      {
        v179 = (char *)sub_145DC80(*(_QWORD *)(a1 + 8), (__int64)v175);
        sub_1458920((__int64)&v182, &v179);
        v175 = (__int64 ***)sub_15A0680(v159, v80, 0);
      }
      else
      {
        v175 = (__int64 ***)sub_15A0680(v159, -v80, 0);
      }
    }
    else
    {
      v165 = *(_QWORD *)(a1 + 8);
      v135 = sub_15A0930(v159, v80);
      v179 = (char *)sub_145DC80(v165, v135);
      sub_1458920((__int64)&v182, &v179);
    }
  }
  v81 = *(_QWORD *)(a4 + 88);
  if ( v81 )
  {
    v164 = *(_QWORD *)(a1 + 8);
    v134 = sub_15A0930(v159, v81);
    v179 = (char *)sub_145DC80(v164, v134);
    sub_1458920((__int64)&v182, &v179);
  }
  v82 = *(_QWORD *)(a1 + 8);
  if ( (_DWORD)v183 )
    v83 = (__int64)sub_147DD40(v82, (__int64 *)&v182, 0, 0, a7, a8);
  else
    v83 = sub_145CF80(v82, v159, 0, 0);
  v9 = sub_38761C0(v177, v83, v69);
  sub_18CE100(v168);
  sub_1940B30(v177 + 88);
  if ( *(_DWORD *)(a2 + 32) == 3 )
  {
    v115 = *(_QWORD *)a3;
    v116 = *(_QWORD *)(*(_QWORD *)a3 - 24LL);
    v117 = *(_DWORD *)(a9 + 8);
    if ( v117 >= *(_DWORD *)(a9 + 12) )
    {
      sub_170B450(a9, 0);
      v117 = *(_DWORD *)(a9 + 8);
    }
    v118 = (_QWORD *)(*(_QWORD *)a9 + 24LL * v117);
    if ( v118 )
    {
      *v118 = 6;
      v118[1] = 0;
      v118[2] = v116;
      if ( v116 != -8 && v116 != 0 && v116 != -16 )
        sub_164C220((__int64)v118);
      v117 = *(_DWORD *)(a9 + 8);
    }
    v119 = v115 - 24;
    *(_DWORD *)(a9 + 8) = v117 + 1;
    if ( *(_QWORD *)(a4 + 24) == -1 )
    {
      if ( v162 == *v175 )
      {
        if ( !*(_QWORD *)(v115 - 24) )
        {
          *(_QWORD *)(v115 - 24) = v175;
LABEL_212:
          v123 = v175[1];
          *(_QWORD *)(v115 - 16) = v123;
          if ( v123 )
            v123[2] = (__int64 *)((v115 - 16) | (unsigned __int64)v123[2] & 3);
          *(_QWORD *)(v115 - 8) = (unsigned __int64)(v175 + 1) | *(_QWORD *)(v115 - 8) & 3LL;
          v175[1] = (__int64 **)v119;
          goto LABEL_109;
        }
      }
      else
      {
        v179 = "tmp";
        LOWORD(v181[0]) = 259;
        v120 = sub_15FBEB0(v175, 0, (__int64)v162, 0);
        v175 = (__int64 ***)sub_15FDBD0(v120, (__int64)v175, (__int64)v162, (__int64)&v179, v115);
        if ( !*(_QWORD *)(v115 - 24) )
          goto LABEL_211;
      }
      v121 = *(_QWORD *)(v115 - 16);
      v122 = *(_QWORD *)(v115 - 8) & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v122 = v121;
      if ( v121 )
        *(_QWORD *)(v121 + 16) = v122 | *(_QWORD *)(v121 + 16) & 3LL;
LABEL_211:
      *(_QWORD *)(v115 - 24) = v175;
      if ( !v175 )
        goto LABEL_109;
      goto LABEL_212;
    }
    v139 = sub_1456E10(*(_QWORD *)(a1 + 8), (__int64)v162);
    v140 = (__int64 ***)sub_15A0930(v139, -v80);
    v141 = (__int64)v140;
    if ( v162 == *v140 )
    {
      if ( !*(_QWORD *)(v115 - 24) )
      {
        *(_QWORD *)(v115 - 24) = v140;
LABEL_229:
        v145 = *(_QWORD *)(v141 + 8);
        *(_QWORD *)(v115 - 16) = v145;
        if ( v145 )
          *(_QWORD *)(v145 + 16) = (v115 - 16) | *(_QWORD *)(v145 + 16) & 3LL;
        *(_QWORD *)(v115 - 8) = (v141 + 8) | *(_QWORD *)(v115 - 8) & 3LL;
        *(_QWORD *)(v141 + 8) = v119;
        goto LABEL_109;
      }
    }
    else
    {
      v142 = sub_15FBEB0(v140, 0, (__int64)v162, 0);
      v141 = sub_15A46C0(v142, (__int64 ***)v141, v162, 0);
      if ( !*(_QWORD *)(v115 - 24) )
        goto LABEL_228;
    }
    v143 = *(_QWORD *)(v115 - 16);
    v144 = *(_QWORD *)(v115 - 8) & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v144 = v143;
    if ( v143 )
      *(_QWORD *)(v143 + 16) = v144 | *(_QWORD *)(v143 + 16) & 3LL;
LABEL_228:
    *(_QWORD *)(v115 - 24) = v141;
    if ( !v141 )
      goto LABEL_109;
    goto LABEL_229;
  }
LABEL_109:
  if ( v182 != v184 )
    _libc_free((unsigned __int64)v182);
  return v9;
}
