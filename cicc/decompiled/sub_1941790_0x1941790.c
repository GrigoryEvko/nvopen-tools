// Function: sub_1941790
// Address: 0x1941790
//
void __fastcall sub_1941790(
        __int64 a1,
        __int64 a2,
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
  __int64 v11; // r15
  __int64 v12; // r14
  unsigned __int64 v13; // r15
  __int64 v14; // rax
  __int64 v15; // r15
  __int64 v16; // r13
  __int64 v17; // r12
  __int64 i; // rbx
  __int64 v19; // rcx
  char v20; // r8
  __int64 v21; // rax
  unsigned int v22; // edi
  __int64 v23; // rdx
  __int64 v24; // rax
  __int64 v25; // rsi
  __int64 v26; // rsi
  __int64 v27; // r8
  _BYTE *v28; // rax
  _BYTE *v29; // rdx
  __int64 v30; // rdi
  _BYTE *v31; // rbx
  __int64 v32; // r14
  __int64 v33; // r14
  __int64 v34; // rdx
  __int64 *v35; // rax
  unsigned __int64 v36; // r13
  __int64 v37; // rdi
  unsigned __int64 v38; // rsi
  __int64 v39; // rsi
  double v40; // xmm4_8
  double v41; // xmm5_8
  __int64 v42; // r15
  __int64 v43; // r12
  __int64 v44; // rbx
  __int64 v45; // r14
  __int64 v46; // rax
  __int64 v47; // r13
  __int64 v48; // rcx
  int v49; // edx
  int v50; // edx
  __int64 v51; // rdi
  __int64 v52; // rsi
  unsigned int v53; // ecx
  __int64 *v54; // rax
  __int64 v55; // r9
  int v56; // r12d
  _QWORD *v57; // rax
  __int64 v58; // r8
  unsigned __int8 v59; // al
  bool v60; // dl
  unsigned __int8 v61; // al
  __int64 v62; // r9
  unsigned int v63; // eax
  unsigned __int64 *v64; // rdi
  unsigned __int64 v65; // rax
  bool v66; // zf
  __int64 *v67; // rax
  int v68; // edx
  _BYTE *v69; // rax
  __int64 *v70; // rax
  __int64 v71; // rax
  unsigned int v72; // eax
  unsigned __int64 *v73; // rdi
  unsigned __int64 v74; // rax
  __int64 v75; // rdx
  __int64 v76; // rsi
  int v77; // edi
  __int64 v78; // r8
  int v79; // edi
  unsigned int v80; // r9d
  __int64 *v81; // rax
  __int64 v82; // r11
  _QWORD *v83; // rdx
  unsigned int v84; // r9d
  __int64 *v85; // rax
  __int64 v86; // r11
  _QWORD *v87; // rax
  int v88; // eax
  unsigned int v89; // eax
  unsigned __int64 *v90; // rdi
  unsigned __int64 v91; // rax
  unsigned int v92; // r12d
  __int64 v93; // r14
  unsigned int v94; // ebx
  _QWORD *v95; // r13
  int v96; // r15d
  bool v97; // cf
  __int64 v98; // rax
  __int64 v99; // r13
  __int64 v100; // rax
  __int64 v101; // rax
  __int64 v102; // r15
  int v103; // r13d
  int v104; // eax
  __int64 v105; // r14
  __int64 v106; // r13
  __int64 v107; // rbx
  __int64 v108; // r12
  int v109; // r8d
  int v110; // eax
  int v111; // ecx
  __int64 v112; // rax
  int v113; // eax
  int v114; // ecx
  __int64 v115; // rax
  __int64 v116; // [rsp+0h] [rbp-2B0h]
  __int64 v117; // [rsp+8h] [rbp-2A8h]
  __int64 v118; // [rsp+10h] [rbp-2A0h]
  __int64 v119; // [rsp+18h] [rbp-298h]
  __int64 v120; // [rsp+18h] [rbp-298h]
  int v121; // [rsp+30h] [rbp-280h]
  int v122; // [rsp+38h] [rbp-278h]
  char v123; // [rsp+3Ch] [rbp-274h]
  int v124; // [rsp+3Ch] [rbp-274h]
  __int64 v126; // [rsp+48h] [rbp-268h]
  __int64 v127; // [rsp+48h] [rbp-268h]
  __int64 v128; // [rsp+50h] [rbp-260h]
  __int64 v129; // [rsp+58h] [rbp-258h]
  __int64 v130; // [rsp+58h] [rbp-258h]
  __int64 v131; // [rsp+58h] [rbp-258h]
  __int64 v132; // [rsp+58h] [rbp-258h]
  _BYTE *v133; // [rsp+60h] [rbp-250h]
  unsigned __int64 v134; // [rsp+68h] [rbp-248h]
  char v136; // [rsp+70h] [rbp-240h]
  int v137; // [rsp+78h] [rbp-238h]
  _BYTE *v138; // [rsp+78h] [rbp-238h]
  _BYTE *v139; // [rsp+80h] [rbp-230h] BYREF
  __int64 v140; // [rsp+88h] [rbp-228h]
  _BYTE v141[32]; // [rsp+90h] [rbp-220h] BYREF
  _BYTE *v142; // [rsp+B0h] [rbp-200h] BYREF
  __int64 v143; // [rsp+B8h] [rbp-1F8h]
  _BYTE v144[64]; // [rsp+C0h] [rbp-1F0h] BYREF
  unsigned __int64 v145; // [rsp+100h] [rbp-1B0h] BYREF
  __int64 v146; // [rsp+108h] [rbp-1A8h]
  unsigned __int64 v147[2]; // [rsp+110h] [rbp-1A0h] BYREF
  int v148; // [rsp+120h] [rbp-190h]
  _BYTE v149[72]; // [rsp+128h] [rbp-188h] BYREF
  _BYTE *v150; // [rsp+170h] [rbp-140h] BYREF
  __int64 v151; // [rsp+178h] [rbp-138h]
  _BYTE v152[304]; // [rsp+180h] [rbp-130h] BYREF

  v11 = a1;
  v142 = v144;
  v143 = 0x800000000LL;
  sub_13FA0E0(a2, (__int64)&v142);
  v151 = 0x800000000LL;
  v150 = v152;
  v133 = &v142[8 * (unsigned int)v143];
  if ( v142 != v133 )
  {
    v12 = a1;
    v13 = (unsigned __int64)v142;
    do
    {
      v14 = *(_QWORD *)(*(_QWORD *)v13 + 48LL);
      if ( !v14 )
        BUG();
      if ( *(_BYTE *)(v14 - 8) == 77 )
      {
        v134 = v13;
        v15 = v14 - 24;
        v16 = *(_QWORD *)(v14 + 8);
        v17 = v12;
        v137 = *(_DWORD *)(v14 - 4) & 0xFFFFFFF;
        if ( !*(_QWORD *)(v14 - 24 + 8) )
          goto LABEL_7;
LABEL_6:
        if ( sub_1456C80(*(_QWORD *)(v17 + 8), *(_QWORD *)v15) )
        {
          sub_1464C80(*(_QWORD *)(v17 + 8), v15);
          if ( v137 )
          {
            v128 = v16;
            v44 = 0;
            v45 = v17;
            while ( 1 )
            {
              while ( 1 )
              {
                v56 = v44;
                if ( (*(_BYTE *)(v15 + 23) & 0x40) != 0 )
                  v46 = *(_QWORD *)(v15 - 8);
                else
                  v46 = v15 - 24LL * (*(_DWORD *)(v15 + 20) & 0xFFFFFFF);
                v47 = *(_QWORD *)(v46 + 24 * v44);
                if ( *(_BYTE *)(v47 + 16) <= 0x17u )
                  goto LABEL_68;
                v48 = 0;
                v49 = *(_DWORD *)(*(_QWORD *)v45 + 24LL);
                if ( v49 )
                {
                  v50 = v49 - 1;
                  v51 = *(_QWORD *)(*(_QWORD *)v45 + 8LL);
                  v52 = *(_QWORD *)(v46 + 8 * v44 + 24LL * *(unsigned int *)(v15 + 56) + 8);
                  v53 = v50 & (((unsigned int)v52 >> 9) ^ ((unsigned int)v52 >> 4));
                  v54 = (__int64 *)(v51 + 16LL * v53);
                  v55 = *v54;
                  if ( v52 == *v54 )
                  {
LABEL_66:
                    v48 = v54[1];
                  }
                  else
                  {
                    v88 = 1;
                    while ( v55 != -8 )
                    {
                      v109 = v88 + 1;
                      v53 = v50 & (v88 + v53);
                      v54 = (__int64 *)(v51 + 16LL * v53);
                      v55 = *v54;
                      if ( v52 == *v54 )
                        goto LABEL_66;
                      v88 = v109;
                    }
                    v48 = 0;
                  }
                }
                if ( a2 != v48 )
                  goto LABEL_68;
                if ( !sub_1377F70(a2 + 56, *(_QWORD *)(v47 + 40)) )
                  goto LABEL_68;
                v126 = sub_1472610(*(_QWORD *)(v45 + 8), v47, *(_QWORD **)a2);
                if ( !sub_146CEE0(*(_QWORD *)(v45 + 8), v126, a2)
                  || !(unsigned __int8)sub_3870AF0(v126, *(_QWORD *)(v45 + 8)) )
                {
                  goto LABEL_68;
                }
                if ( *(_WORD *)(v126 + 24) <= 4u )
                  break;
                v124 = 0;
                v121 = 0;
                if ( !*(_QWORD *)(v47 + 8) )
                  break;
                v122 = v44;
                v92 = 0;
                v116 = v45;
                v93 = *(_QWORD *)(v47 + 8);
                v119 = v15;
                v118 = v47;
                v117 = v44;
                while ( 1 )
                {
                  v94 = v92 + 1;
                  v95 = sub_1648700(v93);
                  v96 = *((unsigned __int8 *)v95 + 16) - 24;
                  if ( sub_1377F70(a2 + 56, v95[5]) )
                  {
                    if ( v96 == 54 || (++v92, v96 == 1) )
                    {
                      ++v121;
                      v92 = v94;
                    }
LABEL_168:
                    v93 = *(_QWORD *)(v93 + 8);
                    if ( !v93 )
                      break;
                    goto LABEL_169;
                  }
                  if ( v96 == 53 )
                  {
                    v102 = v95[1];
                    if ( v102 )
                    {
                      v103 = v124;
                      while ( 1 )
                      {
                        v104 = *((unsigned __int8 *)sub_1648700(v102) + 16);
                        if ( v104 != 78 )
                          v103 -= (v104 == 25) - 1;
                        v102 = *(_QWORD *)(v102 + 8);
                        v92 = v94;
                        if ( !v102 || v94 > 6 )
                          break;
                        ++v94;
                      }
                      v124 = v103;
                    }
                    goto LABEL_168;
                  }
                  if ( v96 == 54 || v96 == 1 )
                  {
                    ++v92;
                    goto LABEL_168;
                  }
                  v93 = *(_QWORD *)(v93 + 8);
                  ++v124;
                  ++v92;
                  if ( !v93 )
                    break;
LABEL_169:
                  if ( v92 > 6 )
                  {
                    v15 = v119;
                    v47 = v118;
                    v56 = v122;
                    v44 = v117;
                    v45 = v116;
                    goto LABEL_75;
                  }
                }
                v15 = v119;
                v97 = v92 < 6;
                v66 = v92 == 6;
                v47 = v118;
                v56 = v122;
                v44 = v117;
                v45 = v116;
                if ( v121 == 0 || !v97 && !v66 || v124 )
                  break;
LABEL_68:
                if ( v137 == (_DWORD)++v44 )
                  goto LABEL_101;
              }
LABEL_75:
              v145 = 0;
              v146 = (__int64)v149;
              v147[0] = (unsigned __int64)v149;
              v147[1] = 8;
              v148 = 0;
              v123 = sub_3872990(a3, v126, a2, v47, &v145, 1, v116);
              if ( v147[0] != v146 )
                _libc_free(v147[0]);
              v129 = *(_QWORD *)v15;
              v57 = (_QWORD *)sub_3872950(a3, v126, v47, a2);
              v58 = (__int64)v57;
              if ( !v57 || v129 != *v57 )
                v58 = sub_38767A0(a3, v126, v129, v47);
              v59 = *(_BYTE *)(v47 + 16);
              if ( v59 <= 0x17u )
              {
                v60 = 0;
                if ( v59 != 5 || *(_WORD *)(v47 + 18) != 32 )
                  goto LABEL_82;
              }
              else
              {
                v60 = 0;
                if ( v59 != 56 )
                  goto LABEL_82;
              }
              if ( (*(_BYTE *)(v47 + 23) & 0x40) != 0 )
                v70 = *(__int64 **)(v47 - 8);
              else
                v70 = (__int64 *)(v47 - 24LL * (*(_DWORD *)(v47 + 20) & 0xFFFFFFF));
              v71 = *v70;
              v66 = v47 == v71;
              v47 = v71;
              v60 = !v66;
LABEL_82:
              v61 = *(_BYTE *)(v58 + 16);
              if ( v61 <= 0x17u )
              {
                v62 = v58;
                if ( v61 == 5 && *(_WORD *)(v58 + 18) == 32 )
                {
LABEL_104:
                  if ( (*(_BYTE *)(v58 + 23) & 0x40) != 0 )
                    v67 = *(__int64 **)(v58 - 8);
                  else
                    v67 = (__int64 *)(v58 - 24LL * (*(_DWORD *)(v58 + 20) & 0xFFFFFFF));
                  v62 = *v67;
                  if ( *v67 == v58 && !v60 )
                  {
LABEL_107:
                    v68 = v151;
                    if ( (unsigned int)v151 >= HIDWORD(v151) )
                    {
                      v132 = v58;
                      sub_16CD150((__int64)&v150, v152, 0, 32, v58, v62);
                      v68 = v151;
                      v58 = v132;
                    }
                    v69 = &v150[32 * v68];
                    if ( v69 )
                    {
                      *((_QWORD *)v69 + 2) = v58;
                      *(_QWORD *)v69 = v15;
                      v69[24] = v123;
                      *((_DWORD *)v69 + 2) = v56;
                      v68 = v151;
                    }
                    LODWORD(v151) = v68 + 1;
                    goto LABEL_68;
                  }
                  goto LABEL_86;
                }
              }
              else
              {
                if ( v61 == 56 )
                  goto LABEL_104;
                v62 = v58;
              }
              if ( !v60 )
                goto LABEL_107;
LABEL_86:
              if ( v47 == v62 )
                goto LABEL_107;
              if ( *(_BYTE *)(*(_QWORD *)v47 + 8LL) == 15 && *(_BYTE *)(*(_QWORD *)v62 + 8LL) == 15 )
              {
                v120 = v58;
                v127 = v62;
                v130 = *(_QWORD *)(v45 + 8);
                v98 = sub_146F1B0(v130, v47);
                v99 = sub_1456F20(v130, v98);
                v131 = *(_QWORD *)(v45 + 8);
                v100 = sub_146F1B0(v131, v127);
                v101 = sub_1456F20(v131, v100);
                v58 = v120;
                if ( v99 == v101 )
                  goto LABEL_107;
              }
              v145 = 6;
              v146 = 0;
              v147[0] = v58;
              if ( v58 != -8 && v58 != -16 )
                sub_164C220((__int64)&v145);
              v63 = *(_DWORD *)(v45 + 56);
              if ( v63 >= *(_DWORD *)(v45 + 60) )
              {
                sub_170B450(v45 + 48, 0);
                v63 = *(_DWORD *)(v45 + 56);
              }
              v64 = (unsigned __int64 *)(*(_QWORD *)(v45 + 48) + 24LL * v63);
              if ( v64 )
              {
                *v64 = 6;
                v64[1] = 0;
                v65 = v147[0];
                v66 = v147[0] == -8;
                v64[2] = v147[0];
                if ( v65 != 0 && !v66 && v65 != -16 )
                  sub_1649AC0(v64, v145 & 0xFFFFFFFFFFFFFFF8LL);
                v63 = *(_DWORD *)(v45 + 56);
              }
              *(_DWORD *)(v45 + 56) = v63 + 1;
              if ( v147[0] == 0 || v147[0] == -8 || v147[0] == -16 )
                goto LABEL_68;
              ++v44;
              sub_1649B30(&v145);
              if ( v137 == (_DWORD)v44 )
              {
LABEL_101:
                v16 = v128;
                v17 = v45;
                break;
              }
            }
          }
        }
LABEL_7:
        while ( 1 )
        {
          v15 = v16 - 24;
          if ( *(_BYTE *)(v16 - 8) != 77 )
            break;
          v16 = *(_QWORD *)(v16 + 8);
          if ( *(_QWORD *)(v15 + 8) )
            goto LABEL_6;
        }
        v13 = v134;
        v12 = v17;
      }
      v13 += 8LL;
    }
    while ( v133 != (_BYTE *)v13 );
    v11 = v12;
  }
  if ( !sub_13FC520(a2) )
  {
    v136 = 0;
    goto LABEL_43;
  }
  v139 = v141;
  v140 = 0x400000000LL;
  sub_13F9CA0(a2, (__int64)&v139);
  v145 = (unsigned __int64)v147;
  v146 = 0x800000000LL;
  sub_13FA0E0(a2, (__int64)&v145);
  if ( (unsigned int)v146 > 1 || (unsigned int)v140 > 1 )
  {
LABEL_38:
    v136 = 0;
    goto LABEL_39;
  }
  for ( i = *(_QWORD *)(*(_QWORD *)v145 + 48LL); ; i = *(_QWORD *)(i + 8) )
  {
    if ( !i )
      BUG();
    if ( *(_BYTE *)(i - 8) != 77 )
      break;
    v19 = i - 24;
    v20 = *(_BYTE *)(i - 1) & 0x40;
    v21 = 0x17FFFFFFE8LL;
    v22 = *(_DWORD *)(i - 4) & 0xFFFFFFF;
    if ( v22 )
    {
      v23 = 24LL * *(unsigned int *)(i + 32) + 8;
      v24 = 0;
      do
      {
        v25 = v19 - 24LL * v22;
        if ( v20 )
          v25 = *(_QWORD *)(i - 32);
        if ( *(_QWORD *)v139 == *(_QWORD *)(v25 + v23) )
        {
          v21 = 24 * v24;
          goto LABEL_26;
        }
        ++v24;
        v23 += 8;
      }
      while ( v22 != (_DWORD)v24 );
      v21 = 0x17FFFFFFE8LL;
      if ( !v20 )
      {
LABEL_27:
        v26 = v19 - 24LL * v22;
        goto LABEL_28;
      }
    }
    else
    {
LABEL_26:
      if ( !v20 )
        goto LABEL_27;
    }
    v26 = *(_QWORD *)(i - 32);
LABEL_28:
    v27 = *(_QWORD *)(v26 + v21);
    v28 = v150;
    v29 = &v150[32 * (unsigned int)v151];
    if ( v150 == v29 )
    {
LABEL_218:
      if ( *(_BYTE *)(v27 + 16) > 0x17u && !sub_13FC1D0(a2, v27) )
        goto LABEL_38;
    }
    else
    {
      while ( 1 )
      {
        if ( v19 == *(_QWORD *)v28 )
        {
          v30 = (*(_BYTE *)(v19 + 23) & 0x40) != 0
              ? *(_QWORD *)(i - 32)
              : v19 - 24LL * (*(_DWORD *)(v19 + 20) & 0xFFFFFFF);
          if ( v27 == *(_QWORD *)(v30 + 24LL * *((unsigned int *)v28 + 2)) )
            break;
        }
        v28 += 32;
        if ( v29 == v28 )
          goto LABEL_218;
      }
    }
  }
  v105 = *(_QWORD *)(a2 + 32);
  if ( v105 != *(_QWORD *)(a2 + 40) )
  {
    v106 = *(_QWORD *)(a2 + 40);
    while ( 2 )
    {
      while ( 1 )
      {
        v107 = *(_QWORD *)(*(_QWORD *)v105 + 48LL);
        v108 = *(_QWORD *)v105 + 40LL;
        if ( v108 != v107 )
          break;
LABEL_198:
        v105 += 8;
        if ( v106 == v105 )
          goto LABEL_205;
      }
      do
      {
        if ( v107 )
        {
          if ( (unsigned __int8)sub_15F3040(v107 - 24) || sub_15F3330(v107 - 24) )
          {
            if ( v107 != v108 )
              goto LABEL_38;
            goto LABEL_198;
          }
        }
        else if ( (unsigned __int8)sub_15F3040(0) || sub_15F3330(0) )
        {
          goto LABEL_38;
        }
        v107 = *(_QWORD *)(v107 + 8);
      }
      while ( v108 != v107 );
      v105 += 8;
      if ( v106 != v105 )
        continue;
      break;
    }
  }
LABEL_205:
  v136 = 1;
LABEL_39:
  if ( (unsigned __int64 *)v145 != v147 )
    _libc_free(v145);
  if ( v139 != v141 )
    _libc_free((unsigned __int64)v139);
LABEL_43:
  v31 = v150;
  v32 = 32LL * (unsigned int)v151;
  v138 = &v150[v32];
  if ( v150 == &v150[v32] )
    goto LABEL_141;
  v33 = v11;
  while ( 2 )
  {
    while ( 2 )
    {
      v42 = *(_QWORD *)v31;
      v43 = *((_QWORD *)v31 + 2);
      if ( dword_4FAF860 == 1 && !v136 && v31[24] )
      {
        v147[0] = *((_QWORD *)v31 + 2);
        v145 = 6;
        v146 = 0;
        if ( v43 != 0 && v43 != -8 && v43 != -16 )
          sub_164C220((__int64)&v145);
        v89 = *(_DWORD *)(v33 + 56);
        if ( v89 >= *(_DWORD *)(v33 + 60) )
        {
          sub_170B450(v33 + 48, 0);
          v89 = *(_DWORD *)(v33 + 56);
        }
        v90 = (unsigned __int64 *)(*(_QWORD *)(v33 + 48) + 24LL * v89);
        if ( v90 )
        {
          *v90 = 6;
          v90[1] = 0;
          v91 = v147[0];
          v66 = v147[0] == -8;
          v90[2] = v147[0];
          if ( v91 != 0 && !v66 && v91 != -16 )
            sub_1649AC0(v90, v145 & 0xFFFFFFFFFFFFFFF8LL);
          v89 = *(_DWORD *)(v33 + 56);
        }
        *(_DWORD *)(v33 + 56) = v89 + 1;
        if ( v147[0] != 0 && v147[0] != -8 && v147[0] != -16 )
          sub_1649B30(&v145);
LABEL_55:
        v31 += 32;
        if ( v138 == v31 )
          goto LABEL_140;
        continue;
      }
      break;
    }
    *(_BYTE *)(v33 + 448) = 1;
    if ( (*(_BYTE *)(v42 + 23) & 0x40) != 0 )
      v34 = *(_QWORD *)(v42 - 8);
    else
      v34 = v42 - 24LL * (*(_DWORD *)(v42 + 20) & 0xFFFFFFF);
    v35 = (__int64 *)(v34 + 24LL * *((unsigned int *)v31 + 2));
    v36 = *v35;
    if ( *v35 )
    {
      v37 = v35[1];
      v38 = v35[2] & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v38 = v37;
      if ( v37 )
        *(_QWORD *)(v37 + 16) = *(_QWORD *)(v37 + 16) & 3LL | v38;
    }
    *v35 = v43;
    if ( v43 )
    {
      v39 = *(_QWORD *)(v43 + 8);
      v35[1] = v39;
      if ( v39 )
        *(_QWORD *)(v39 + 16) = (unsigned __int64)(v35 + 1) | *(_QWORD *)(v39 + 16) & 3LL;
      v35[2] = (v43 + 8) | v35[2] & 3;
      *(_QWORD *)(v43 + 8) = v35;
    }
    if ( !(unsigned __int8)sub_1AE9990(v36, *(_QWORD *)(v33 + 32)) )
      goto LABEL_54;
    v147[0] = v36;
    v145 = 6;
    v146 = 0;
    if ( v36 != 0 && v36 != -8 && v36 != -16 )
      sub_164C220((__int64)&v145);
    v72 = *(_DWORD *)(v33 + 56);
    if ( v72 >= *(_DWORD *)(v33 + 60) )
    {
      sub_170B450(v33 + 48, 0);
      v72 = *(_DWORD *)(v33 + 56);
    }
    v73 = (unsigned __int64 *)(*(_QWORD *)(v33 + 48) + 24LL * v72);
    if ( v73 )
    {
      *v73 = 6;
      v73[1] = 0;
      v74 = v147[0];
      v66 = v147[0] == -8;
      v73[2] = v147[0];
      if ( v74 != 0 && !v66 && v74 != -16 )
        sub_1649AC0(v73, v145 & 0xFFFFFFFFFFFFFFF8LL);
      v72 = *(_DWORD *)(v33 + 56);
    }
    *(_DWORD *)(v33 + 56) = v72 + 1;
    if ( v147[0] != 0 && v147[0] != -8 && v147[0] != -16 )
    {
      sub_1649B30(&v145);
      if ( (*(_DWORD *)(v42 + 20) & 0xFFFFFFF) != 1 )
        goto LABEL_55;
    }
    else
    {
LABEL_54:
      if ( (*(_DWORD *)(v42 + 20) & 0xFFFFFFF) != 1 )
        goto LABEL_55;
    }
    if ( *(_BYTE *)(v43 + 16) <= 0x17u )
      goto LABEL_139;
    v75 = *(_QWORD *)(v43 + 40);
    v76 = *(_QWORD *)(v42 + 40);
    if ( v75 == v76 )
      goto LABEL_139;
    v77 = *(_DWORD *)(*(_QWORD *)v33 + 24LL);
    if ( !v77 )
      goto LABEL_139;
    v78 = *(_QWORD *)(*(_QWORD *)v33 + 8LL);
    v79 = v77 - 1;
    v80 = v79 & (((unsigned int)v75 >> 9) ^ ((unsigned int)v75 >> 4));
    v81 = (__int64 *)(v78 + 16LL * v80);
    v82 = *v81;
    if ( v75 == *v81 )
    {
LABEL_134:
      v83 = (_QWORD *)v81[1];
      if ( !v83 )
        goto LABEL_139;
      v84 = v79 & (((unsigned int)v76 >> 9) ^ ((unsigned int)v76 >> 4));
      v85 = (__int64 *)(v78 + 16LL * v84);
      v86 = *v85;
      if ( v76 == *v85 )
      {
LABEL_136:
        v87 = (_QWORD *)v85[1];
        if ( v83 == v87 )
          goto LABEL_139;
        while ( v87 )
        {
          v87 = (_QWORD *)*v87;
          if ( v83 == v87 )
            goto LABEL_139;
        }
      }
      else
      {
        v110 = 1;
        while ( v86 != -8 )
        {
          v111 = v110 + 1;
          v112 = v79 & (v84 + v110);
          v84 = v112;
          v85 = (__int64 *)(v78 + 16 * v112);
          v86 = *v85;
          if ( v76 == *v85 )
            goto LABEL_136;
          v110 = v111;
        }
      }
      goto LABEL_55;
    }
    v113 = 1;
    while ( v82 != -8 )
    {
      v114 = v113 + 1;
      v115 = v79 & (v80 + v113);
      v80 = v115;
      v81 = (__int64 *)(v78 + 16 * v115);
      v82 = *v81;
      if ( v75 == *v81 )
        goto LABEL_134;
      v113 = v114;
    }
LABEL_139:
    v31 += 32;
    sub_164D160(v42, v43, a4, a5, a6, a7, v40, v41, a10, a11);
    sub_15F20C0((_QWORD *)v42);
    if ( v138 != v31 )
      continue;
    break;
  }
LABEL_140:
  v138 = v150;
LABEL_141:
  *(_QWORD *)(a3 + 272) = 0;
  *(_QWORD *)(a3 + 280) = 0;
  if ( v138 != v152 )
    _libc_free((unsigned __int64)v138);
  if ( v142 != v144 )
    _libc_free((unsigned __int64)v142);
}
