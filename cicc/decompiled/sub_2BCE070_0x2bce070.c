// Function: sub_2BCE070
// Address: 0x2bce070
//
__int64 __fastcall sub_2BCE070(__int64 a1, __int64 *a2, unsigned __int64 a3, __int64 a4, char a5, __m128i a6)
{
  __int64 v8; // rdx
  __int64 *v9; // r15
  __int64 *v10; // r12
  __int64 v11; // r14
  unsigned int v12; // r12d
  __int64 v13; // rdi
  int v14; // edx
  unsigned int v15; // r14d
  unsigned int v16; // eax
  _QWORD *v17; // r15
  _QWORD *v18; // r14
  unsigned __int64 v19; // rsi
  _QWORD *v20; // rax
  _QWORD *v21; // rdi
  __int64 v22; // rcx
  __int64 v23; // rdx
  __int64 v24; // rax
  _QWORD *v25; // rdi
  __int64 v26; // rcx
  __int64 v27; // rdx
  unsigned int v28; // eax
  __int64 v29; // r12
  unsigned int v30; // ebx
  unsigned int v31; // r14d
  __int64 v32; // r11
  unsigned int v33; // r12d
  char v34; // al
  __int64 v35; // r8
  __int64 v36; // r9
  unsigned int v37; // eax
  __int64 v38; // r15
  __int64 v39; // rax
  unsigned int v40; // r15d
  unsigned int v41; // r15d
  unsigned int v42; // eax
  __int64 v43; // rcx
  __int64 *v44; // rdx
  __int64 *v45; // r8
  __int64 *v46; // rdx
  unsigned int v47; // edi
  int v48; // esi
  __int64 v49; // r9
  int v50; // esi
  unsigned int v51; // ecx
  _BYTE *v52; // r10
  _BYTE *v53; // rax
  __int64 v54; // rcx
  _QWORD *v55; // r9
  __int64 *v56; // r13
  __int8 *v57; // rsi
  size_t v58; // rdx
  __int64 v59; // rdx
  __int64 v60; // rcx
  __int64 v61; // r8
  __int64 v62; // r9
  __int64 *v64; // r15
  __int64 v65; // r12
  __int64 v66; // rax
  __int64 v67; // rdx
  __int64 v68; // r8
  __int64 v69; // r9
  __m128i v70; // xmm0
  __m128i v71; // xmm1
  __m128i v72; // xmm2
  unsigned __int64 *v73; // r12
  __int64 v74; // r8
  unsigned __int64 *v75; // r14
  unsigned __int64 v76; // rdi
  _BOOL8 v77; // rsi
  __int64 v78; // rdx
  __int64 v79; // rcx
  char v80; // al
  char v81; // al
  __int64 v82; // rdx
  __int64 v83; // rcx
  unsigned __int64 v84; // r8
  __int64 v85; // r9
  __int64 v86; // r11
  __m128i *v87; // rax
  __int64 v88; // r11
  __int64 v89; // rcx
  __int64 v90; // r8
  __int64 v91; // r9
  signed __int64 v92; // rax
  __int64 v93; // rdx
  bool v94; // sf
  bool v95; // of
  __int64 v96; // r15
  __int64 v97; // r12
  __int64 v98; // rax
  __int64 v99; // r11
  unsigned __int64 *v100; // r12
  unsigned __int64 *v101; // r15
  unsigned __int64 *v102; // rbx
  __int64 v103; // r15
  unsigned __int64 v104; // rdi
  __int64 v105; // rdx
  __int64 v106; // rcx
  unsigned __int64 v107; // r8
  __int64 v108; // r9
  __int64 *v109; // r14
  __int64 v110; // r13
  __int64 v111; // rax
  __int64 v112; // r13
  __int64 v113; // rax
  __int64 v114; // rcx
  __int64 v115; // r8
  __int64 v116; // r9
  __m128i v117; // xmm3
  __m128i v118; // xmm5
  __int64 v119; // rdx
  __int64 v120; // rax
  __int64 v121; // rax
  __int64 v122; // rax
  __int64 v123; // rax
  __int64 v124; // [rsp+8h] [rbp-4B8h]
  unsigned int v125; // [rsp+14h] [rbp-4ACh]
  char v126; // [rsp+18h] [rbp-4A8h]
  __int64 v127; // [rsp+18h] [rbp-4A8h]
  __int64 *v128; // [rsp+20h] [rbp-4A0h]
  unsigned int v129; // [rsp+28h] [rbp-498h]
  unsigned int v130; // [rsp+2Ch] [rbp-494h]
  __int64 v132; // [rsp+38h] [rbp-488h]
  __int64 v133; // [rsp+40h] [rbp-480h]
  int v134; // [rsp+40h] [rbp-480h]
  __int64 v135; // [rsp+40h] [rbp-480h]
  __int64 v136; // [rsp+40h] [rbp-480h]
  __int64 v137; // [rsp+40h] [rbp-480h]
  __int64 v138; // [rsp+40h] [rbp-480h]
  __int64 v139; // [rsp+40h] [rbp-480h]
  __int64 v140; // [rsp+40h] [rbp-480h]
  unsigned int v141; // [rsp+40h] [rbp-480h]
  __int64 v142; // [rsp+40h] [rbp-480h]
  __int64 v143; // [rsp+40h] [rbp-480h]
  __int64 *v144; // [rsp+48h] [rbp-478h]
  unsigned __int8 v146; // [rsp+59h] [rbp-467h]
  char v148; // [rsp+5Bh] [rbp-465h]
  unsigned int v149; // [rsp+5Ch] [rbp-464h]
  __int64 v150; // [rsp+60h] [rbp-460h]
  __int64 v151; // [rsp+60h] [rbp-460h]
  __int64 v152; // [rsp+60h] [rbp-460h]
  unsigned int v153; // [rsp+68h] [rbp-458h]
  void **v154; // [rsp+70h] [rbp-450h] BYREF
  __int64 v155; // [rsp+78h] [rbp-448h]
  _BYTE v156[16]; // [rsp+80h] [rbp-440h] BYREF
  void *s; // [rsp+90h] [rbp-430h] BYREF
  size_t v158; // [rsp+98h] [rbp-428h]
  _QWORD v159[2]; // [rsp+A0h] [rbp-420h] BYREF
  unsigned __int64 v160[6]; // [rsp+B0h] [rbp-410h] BYREF
  unsigned __int64 v161[2]; // [rsp+E0h] [rbp-3E0h] BYREF
  _QWORD v162[2]; // [rsp+F0h] [rbp-3D0h] BYREF
  unsigned __int64 v163[2]; // [rsp+100h] [rbp-3C0h] BYREF
  void ***v164; // [rsp+110h] [rbp-3B0h] BYREF
  __int64 *v165; // [rsp+130h] [rbp-390h] BYREF
  int v166; // [rsp+138h] [rbp-388h]
  char v167; // [rsp+13Ch] [rbp-384h]
  __int64 v168; // [rsp+140h] [rbp-380h] BYREF
  __m128i v169; // [rsp+148h] [rbp-378h]
  __int64 v170; // [rsp+158h] [rbp-368h]
  __m128i v171; // [rsp+160h] [rbp-360h] BYREF
  __m128i v172; // [rsp+170h] [rbp-350h]
  _BYTE *v173; // [rsp+180h] [rbp-340h] BYREF
  __int64 v174; // [rsp+188h] [rbp-338h]
  _BYTE v175[320]; // [rsp+190h] [rbp-330h] BYREF
  char v176; // [rsp+2D0h] [rbp-1F0h]
  int v177; // [rsp+2D4h] [rbp-1ECh]
  __int64 v178; // [rsp+2D8h] [rbp-1E8h]
  void *v179; // [rsp+2E0h] [rbp-1E0h] BYREF
  __int64 v180; // [rsp+2E8h] [rbp-1D8h]
  __int64 v181; // [rsp+2F0h] [rbp-1D0h] BYREF
  __m128i v182; // [rsp+2F8h] [rbp-1C8h] BYREF
  __int64 v183; // [rsp+308h] [rbp-1B8h]
  __m128i v184; // [rsp+310h] [rbp-1B0h] BYREF
  __m128i v185; // [rsp+320h] [rbp-1A0h] BYREF
  unsigned __int64 *v186; // [rsp+330h] [rbp-190h] BYREF
  unsigned int v187; // [rsp+338h] [rbp-188h]
  _BYTE v188[324]; // [rsp+340h] [rbp-180h] BYREF
  int v189; // [rsp+484h] [rbp-3Ch]
  __int64 v190; // [rsp+488h] [rbp-38h]

  if ( a3 <= 1 )
    return 0;
  v124 = sub_2B5F980(a2, a3, *(__int64 **)(a1 + 16));
  if ( !v124 || !v8 )
    return 0;
  v9 = &a2[a3];
  v144 = v9;
  if ( v9 == a2 )
  {
LABEL_9:
    v132 = sub_2B08520((char *)*a2);
    v12 = sub_2B49BC0(a4, (unsigned __int8 *)v124);
    v153 = a3;
    v13 = *(_QWORD *)(a1 + 8);
    v14 = 2;
    if ( *(_DWORD *)(a4 + 3364) / v12 >= 2 )
      v14 = *(_DWORD *)(a4 + 3364) / v12;
    v15 = v14;
    v130 = v14;
    v16 = sub_2B1E190(v13, v132, a3);
    if ( v16 < v15 )
      v16 = v15;
    v129 = v16;
    v17 = sub_C52410();
    v18 = v17 + 1;
    v19 = sub_C959E0();
    v20 = (_QWORD *)v17[2];
    if ( v20 )
    {
      v21 = v17 + 1;
      do
      {
        while ( 1 )
        {
          v22 = v20[2];
          v23 = v20[3];
          if ( v19 <= v20[4] )
            break;
          v20 = (_QWORD *)v20[3];
          if ( !v23 )
            goto LABEL_18;
        }
        v21 = v20;
        v20 = (_QWORD *)v20[2];
      }
      while ( v22 );
LABEL_18:
      if ( v21 != v18 && v19 >= v21[4] )
        v18 = v21;
    }
    if ( v18 == (_QWORD *)((char *)sub_C52410() + 8) )
      goto LABEL_96;
    v24 = v18[7];
    if ( !v24 )
      goto LABEL_96;
    v25 = v18 + 6;
    do
    {
      while ( 1 )
      {
        v26 = *(_QWORD *)(v24 + 16);
        v27 = *(_QWORD *)(v24 + 24);
        if ( *(_DWORD *)(v24 + 32) >= dword_500FE68 )
          break;
        v24 = *(_QWORD *)(v24 + 24);
        if ( !v27 )
          goto LABEL_27;
      }
      v25 = (_QWORD *)v24;
      v24 = *(_QWORD *)(v24 + 16);
    }
    while ( v26 );
LABEL_27:
    if ( v25 == v18 + 6 || dword_500FE68 < *((_DWORD *)v25 + 8) || !*((_DWORD *)v25 + 9) )
LABEL_96:
      v28 = sub_DFB340(*(_QWORD *)(a4 + 3296));
    else
      v28 = qword_500FEE8;
    if ( v28 && v28 <= v129 && (v129 = v28, v28 == 1) )
    {
      v56 = *(__int64 **)(a4 + 3352);
      if ( !(unsigned __int8)sub_2B0D8D0(*v56) )
        return 0;
      sub_B176B0((__int64)&v179, (__int64)"slp-vectorizer", (__int64)"SmallVF", 7, v124);
      sub_B18290((__int64)&v179, "Cannot SLP vectorize list: vectorization factor ", 0x30u);
      v57 = "less than 2 is not supported";
      v58 = 28;
    }
    else
    {
      v29 = (int)qword_5010428;
      if ( (unsigned int)a3 > 1 )
      {
        v126 = 0;
        v30 = 0;
        v31 = 0;
        v146 = 0;
        v149 = v129;
        while ( v149 >= v130 )
        {
          sub_2B08680(v132, v149);
          if ( (unsigned int)sub_DFDB60(*(_QWORD *)(a1 + 8)) == v149 || v153 <= v31 )
            goto LABEL_69;
          v125 = v31;
          v32 = v29;
          while ( 2 )
          {
            v150 = v32;
            v33 = v153 - v31;
            if ( v153 - v31 > v149 )
              v33 = v149;
            v34 = sub_2B1F720(*(_QWORD *)(a1 + 8), v132, v33);
            v32 = v150;
            v148 = v34;
            if ( !v34 )
            {
LABEL_101:
              ++v31;
              goto LABEL_102;
            }
            if ( v129 > v33 && a5 )
              goto LABEL_103;
            v37 = v130;
            if ( v130 < v33 )
              v37 = v33;
            if ( v149 > v37 || v149 == v130 && v33 <= 1 )
            {
LABEL_103:
              v29 = v32;
              goto LABEL_68;
            }
            s = v159;
            v158 = 0x600000000LL;
            if ( v33 > 6 )
            {
              sub_C8D5F0((__int64)&s, v159, v33, 8u, v35, v36);
              memset(s, 0, 8LL * v33);
              v55 = s;
              LODWORD(v158) = v33;
              v32 = v150;
              v44 = &a2[v31];
              v45 = (__int64 *)s;
              if ( v144 == v44 )
              {
LABEL_65:
                v29 = v32;
                goto LABEL_66;
              }
LABEL_57:
              v46 = v44 + 1;
              v47 = 0;
LABEL_62:
              v53 = (_BYTE *)*(v46 - 1);
              if ( *v53 > 0x1Cu )
              {
                v48 = *(_DWORD *)(a4 + 2000);
                v49 = *(_QWORD *)(a4 + 1984);
                if ( v48 )
                {
                  v50 = v48 - 1;
                  v51 = v50 & (((unsigned int)v53 >> 9) ^ ((unsigned int)v53 >> 4));
                  v52 = *(_BYTE **)(v49 + 8LL * v51);
                  if ( v53 == v52 )
                  {
LABEL_60:
                    if ( v144 == v46 )
                    {
                      if ( v33 != v47 )
                        goto LABEL_147;
LABEL_98:
                      v133 = v32;
                      sub_2BAAD20(a4, v45, (unsigned int)v158);
                      v77 = 0;
                      v80 = sub_2B2DB00(a4, 0, v78, v79);
                      v32 = v133;
                      if ( v80 )
                      {
                        if ( s != v159 )
                        {
                          _libc_free((unsigned __int64)s);
                          v32 = v133;
                        }
                        goto LABEL_101;
                      }
                      v81 = sub_2B2A740((__int64 **)a4);
                      v86 = v133;
                      if ( v81 )
                      {
                        sub_2BBDBE0((__int64 **)a4, a6, 0, v82, v83, v84, v85);
                        v77 = **(_BYTE **)s != 91;
                        sub_2BBFB60(a4, v77, a6, v105, v106, v107, v108);
                        v86 = v133;
                      }
                      v135 = v86;
                      sub_2BB0460(a4, v77, v82, v83, v84, v85);
                      v179 = 0;
                      v87 = (__m128i *)&v181;
                      v180 = 1;
                      do
                      {
                        v87->m128i_i64[0] = -4096;
                        v87 = (__m128i *)((char *)v87 + 8);
                      }
                      while ( v87 != &v184 );
                      sub_2B4F3D0(a4, (__int64)&v179);
                      v88 = v135;
                      if ( (v180 & 1) == 0 )
                      {
                        sub_C7D6A0(v181, 8LL * v182.m128i_u32[0], 8);
                        v88 = v135;
                      }
                      v136 = v88;
                      sub_2BB3590(a4);
                      v92 = sub_2B94A80(a4, 0, 0, v89, v90, v91);
                      v32 = v136;
                      v95 = __OFSUB__((_DWORD)v93, v30);
                      v94 = (int)(v93 - v30) < 0;
                      v96 = v93;
                      if ( (_DWORD)v93 == v30 )
                      {
                        v95 = __OFSUB__(v92, v136);
                        v94 = v92 - v136 < 0;
                      }
                      if ( v94 != v95 )
                      {
                        v30 = v93;
                        v32 = v92;
                      }
                      if ( (_DWORD)v93 )
                      {
                        if ( (int)v93 >= 0 )
                          goto LABEL_120;
LABEL_126:
                        v127 = v32;
                        v128 = *(__int64 **)(a4 + 3352);
                        v137 = v92;
                        sub_B174A0(
                          (__int64)&v179,
                          (__int64)"slp-vectorizer",
                          (__int64)"VectorizedList",
                          14,
                          *(_QWORD *)s);
                        sub_B18290((__int64)&v179, "SLP vectorized with cost ", 0x19u);
                        sub_B16D50((__int64)v161, "Cost", 4, v137, v96);
                        v97 = sub_23FD640((__int64)&v179, (__int64)v161);
                        sub_B18290(v97, " and with tree size ", 0x14u);
                        sub_B169E0((__int64 *)&v165, "TreeSize", 8, *(_DWORD *)(a4 + 8));
                        v98 = sub_23FD640(v97, (__int64)&v165);
                        sub_1049740(v128, v98);
                        v99 = v127;
                        if ( (__m128i *)v169.m128i_i64[1] != &v171 )
                        {
                          j_j___libc_free_0(v169.m128i_u64[1]);
                          v99 = v127;
                        }
                        if ( v165 != &v168 )
                        {
                          v138 = v99;
                          j_j___libc_free_0((unsigned __int64)v165);
                          v99 = v138;
                        }
                        if ( (void ****)v163[0] != &v164 )
                        {
                          v139 = v99;
                          j_j___libc_free_0(v163[0]);
                          v99 = v139;
                        }
                        if ( (_QWORD *)v161[0] != v162 )
                        {
                          v140 = v99;
                          j_j___libc_free_0(v161[0]);
                          v99 = v140;
                        }
                        v100 = v186;
                        v179 = &unk_49D9D40;
                        v101 = &v186[10 * v187];
                        if ( v186 != v101 )
                        {
                          v141 = v30;
                          v102 = &v186[10 * v187];
                          v103 = v99;
                          do
                          {
                            v102 -= 10;
                            v104 = v102[4];
                            if ( (unsigned __int64 *)v104 != v102 + 6 )
                              j_j___libc_free_0(v104);
                            if ( (unsigned __int64 *)*v102 != v102 + 2 )
                              j_j___libc_free_0(*v102);
                          }
                          while ( v100 != v102 );
                          v99 = v103;
                          v30 = v141;
                          v101 = v186;
                        }
                        if ( v101 != (unsigned __int64 *)v188 )
                        {
                          v142 = v99;
                          _libc_free((unsigned __int64)v101);
                          v99 = v142;
                        }
                        v143 = v99;
                        sub_2BCA0A0(a4);
                        v31 += v149;
                        v125 = v31;
                        v32 = v143;
                        v146 = v148;
                      }
                      else
                      {
                        if ( -(int)qword_5010428 > v92 )
                          goto LABEL_126;
LABEL_120:
                        ++v31;
                      }
                      if ( s != v159 )
                      {
                        v151 = v32;
                        _libc_free((unsigned __int64)s);
                        v32 = v151;
                      }
                      v126 = v148;
LABEL_102:
                      if ( v153 <= v31 )
                        goto LABEL_103;
                      continue;
                    }
                    goto LABEL_61;
                  }
                  v134 = 1;
                  while ( v52 != (_BYTE *)-4096LL )
                  {
                    v51 = v50 & (v134 + v51);
                    ++v134;
                    v52 = *(_BYTE **)(v49 + 8LL * v51);
                    if ( v53 == v52 )
                      goto LABEL_60;
                  }
                }
              }
              v54 = v47++;
              v45[v54] = (__int64)v53;
              if ( v47 == v33 )
              {
                v45 = (__int64 *)s;
                goto LABEL_98;
              }
              v45 = (__int64 *)s;
              v55 = s;
              if ( v144 == v46 )
                goto LABEL_65;
LABEL_61:
              ++v46;
              goto LABEL_62;
            }
            break;
          }
          v38 = 8LL * v33;
          if ( v38 )
          {
            v39 = (unsigned int)v38;
            v40 = v38 - 1;
            *(_QWORD *)((char *)&v159[-1] + v39) = 0;
            if ( v40 >= 8 )
            {
              v41 = v40 & 0xFFFFFFF8;
              v42 = 0;
              do
              {
                v43 = v42;
                v42 += 8;
                *(_QWORD *)((char *)v159 + v43) = 0;
              }
              while ( v42 < v41 );
            }
          }
          LODWORD(v158) = v33;
          v44 = &a2[v31];
          if ( v144 != v44 )
          {
            v45 = v159;
            goto LABEL_57;
          }
LABEL_147:
          v55 = s;
          v29 = v32;
LABEL_66:
          if ( v55 != v159 )
            _libc_free((unsigned __int64)v55);
LABEL_68:
          v31 = v125;
LABEL_69:
          v149 = sub_2B1E190(*(_QWORD *)(a1 + 8), *(_QWORD *)(v124 + 8), v149 - 1);
          if ( v153 <= v31 + 1 )
            break;
        }
        if ( v146 == 1 )
          return v146;
        if ( v126 )
        {
          v109 = *(__int64 **)(a4 + 3352);
          v110 = *v109;
          v111 = sub_B2BE50(*v109);
          if ( sub_B6EA50(v111)
            || (v122 = sub_B2BE50(v110),
                v123 = sub_B6F970(v122),
                (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v123 + 48LL))(v123)) )
          {
            sub_B176B0((__int64)&v179, (__int64)"slp-vectorizer", (__int64)"NotBeneficial", 13, v124);
            sub_B18290((__int64)&v179, "List vectorization was possible but not beneficial with cost ", 0x3Du);
            sub_B16D50((__int64)v161, "Cost", 4, v29, v30);
            v112 = sub_2445430((__int64)&v179, (__int64)v161);
            sub_B18290(v112, " >= ", 4u);
            sub_B16530((__int64 *)&s, "Treshold", 8, -(int)qword_5010428);
            v113 = sub_2445430(v112, (__int64)&s);
            v166 = *(_DWORD *)(v113 + 8);
            v167 = *(_BYTE *)(v113 + 12);
            v168 = *(_QWORD *)(v113 + 16);
            v117 = _mm_loadu_si128((const __m128i *)(v113 + 24));
            v165 = (__int64 *)&unk_49D9D40;
            v169 = v117;
            v170 = *(_QWORD *)(v113 + 40);
            v171 = _mm_loadu_si128((const __m128i *)(v113 + 48));
            v118 = _mm_loadu_si128((const __m128i *)(v113 + 64));
            v173 = v175;
            v174 = 0x400000000LL;
            v172 = v118;
            v119 = *(unsigned int *)(v113 + 88);
            if ( (_DWORD)v119 )
            {
              v152 = v113;
              sub_2B44C50((__int64)&v173, v113 + 80, v119, v114, v115, v116);
              v113 = v152;
            }
            v176 = *(_BYTE *)(v113 + 416);
            v177 = *(_DWORD *)(v113 + 420);
            v178 = *(_QWORD *)(v113 + 424);
            v165 = (__int64 *)&unk_49D9DB0;
            sub_2240A30(v160);
            sub_2240A30((unsigned __int64 *)&s);
            sub_2240A30(v163);
            sub_2240A30(v161);
            v179 = &unk_49D9D40;
            sub_23FD590((__int64)&v186);
            sub_1049740(v109, (__int64)&v165);
            v165 = (__int64 *)&unk_49D9D40;
            sub_23FD590((__int64)&v173);
          }
          return 0;
        }
        if ( v146 )
          return v146;
      }
      v56 = *(__int64 **)(a4 + 3352);
      if ( !(unsigned __int8)sub_2B0D8D0(*v56) )
        return 0;
      sub_B176B0((__int64)&v179, (__int64)"slp-vectorizer", (__int64)"NotPossible", 11, v124);
      sub_B18290((__int64)&v179, "Cannot SLP vectorize list: vectorization was impossible", 0x37u);
      v57 = " with available vectorization factors";
      v58 = 37;
    }
    sub_B18290((__int64)&v179, v57, v58);
    sub_23FE290((__int64)&v165, (__int64)&v179, v59, v60, v61, v62);
    v178 = v190;
    v165 = (__int64 *)&unk_49D9DB0;
    v179 = &unk_49D9D40;
    sub_23FD590((__int64)&v186);
    sub_1049740(v56, (__int64)&v165);
    v165 = (__int64 *)&unk_49D9D40;
    sub_23FD590((__int64)&v173);
    return 0;
  }
  v10 = a2;
  while ( 1 )
  {
    v11 = *(_QWORD *)(*v10 + 8);
    if ( *(_BYTE *)*v10 != 91 && !sub_2B08630(*(_QWORD *)(*v10 + 8)) )
      break;
    if ( v9 == ++v10 )
      goto LABEL_9;
  }
  v64 = *(__int64 **)(a4 + 3352);
  v146 = 0;
  v65 = *v64;
  v66 = sub_B2BE50(*v64);
  if ( sub_B6EA50(v66)
    || (v120 = sub_B2BE50(v65),
        v121 = sub_B6F970(v120),
        (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v121 + 48LL))(v121)) )
  {
    v154 = (void **)v156;
    v163[1] = 0x100000000LL;
    v164 = &v154;
    v161[0] = (unsigned __int64)&unk_49DD210;
    v155 = 0;
    v156[0] = 0;
    v161[1] = 0;
    v162[0] = 0;
    v162[1] = 0;
    v163[0] = 0;
    sub_CB5980((__int64)v161, 0, 0, 0);
    sub_A587F0(v11, (__int64)v161, 0, 0);
    sub_B176B0((__int64)&v179, (__int64)"slp-vectorizer", (__int64)"UnsupportedType", 15, v124);
    sub_B18290((__int64)&v179, "Cannot SLP vectorize list: type ", 0x20u);
    s = v159;
    sub_2B0E310((__int64 *)&s, v154, (__int64)v154 + v155);
    if ( 0x3FFFFFFFFFFFFFFFLL - v158 <= 0x1C )
      sub_4262D8((__int64)"basic_string::append");
    sub_2241490((unsigned __int64 *)&s, " is unsupported by vectorizer", 0x1Du);
    sub_B18290((__int64)&v179, (__int8 *)s, v158);
    v70 = _mm_loadu_si128(&v182);
    v71 = _mm_loadu_si128(&v184);
    v166 = v180;
    v72 = _mm_loadu_si128(&v185);
    v169 = v70;
    v167 = BYTE4(v180);
    v171 = v71;
    v168 = v181;
    v165 = (__int64 *)&unk_49D9D40;
    v172 = v72;
    v170 = v183;
    v173 = v175;
    v174 = 0x400000000LL;
    if ( v187 )
      sub_2B44C50((__int64)&v173, (__int64)&v186, v67, v187, v68, v69);
    v176 = v188[320];
    v177 = v189;
    v178 = v190;
    v165 = (__int64 *)&unk_49D9DB0;
    if ( s != v159 )
      j_j___libc_free_0((unsigned __int64)s);
    v73 = v186;
    v179 = &unk_49D9D40;
    v74 = 10LL * v187;
    v75 = &v186[v74];
    if ( v186 != &v186[v74] )
    {
      do
      {
        v75 -= 10;
        v76 = v75[4];
        if ( (unsigned __int64 *)v76 != v75 + 6 )
          j_j___libc_free_0(v76);
        if ( (unsigned __int64 *)*v75 != v75 + 2 )
          j_j___libc_free_0(*v75);
      }
      while ( v73 != v75 );
      v75 = v186;
    }
    if ( v75 != (unsigned __int64 *)v188 )
      _libc_free((unsigned __int64)v75);
    v161[0] = (unsigned __int64)&unk_49DD210;
    sub_CB5840((__int64)v161);
    if ( v154 != (void **)v156 )
      j_j___libc_free_0((unsigned __int64)v154);
    sub_1049740(v64, (__int64)&v165);
    v165 = (__int64 *)&unk_49D9D40;
    sub_23FD590((__int64)&v173);
  }
  return v146;
}
