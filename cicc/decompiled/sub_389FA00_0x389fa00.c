// Function: sub_389FA00
// Address: 0x389fa00
//
__int64 __fastcall sub_389FA00(__int64 a1, __int64 *a2, unsigned __int8 a3)
{
  unsigned __int64 v6; // r14
  unsigned int v7; // r14d
  int v9; // eax
  const char *v10; // rax
  unsigned int v11; // eax
  __int64 v12; // r9
  unsigned __int64 v13; // rax
  int v14; // eax
  int v15; // eax
  unsigned __int64 v16; // rsi
  __int64 v17; // r9
  __int64 v18; // r9
  __m128i *v19; // r12
  __m128i *v20; // rbx
  unsigned __int64 v21; // rdi
  char v22; // al
  bool v23; // al
  int v24; // r8d
  __int64 v25; // r9
  __int64 v26; // rbx
  __m128i *v27; // r14
  __int8 *v28; // r14
  __int64 v29; // rcx
  __int8 *v30; // rdx
  unsigned __int64 v31; // r14
  __int64 v32; // r13
  __int64 v33; // rax
  __int64 *v34; // r14
  __int64 v35; // rax
  size_t v36; // rdx
  __int64 v37; // r9
  __int64 v38; // r13
  __int64 v39; // rdi
  _QWORD *v40; // rax
  unsigned __int64 v41; // rsi
  int *v42; // rax
  __int64 v43; // r9
  unsigned __int64 v44; // rdi
  unsigned __int64 v45; // r13
  __int64 v46; // rax
  __int64 v47; // r8
  unsigned __int64 *v48; // rdi
  unsigned __int64 *v49; // rsi
  unsigned __int64 *v50; // rdx
  unsigned __int64 v51; // r13
  __int64 v52; // rcx
  char v53; // dl
  char v54; // si
  __int64 v55; // rcx
  char v56; // al
  unsigned int v57; // esi
  __int64 v58; // rdx
  __int64 v59; // rsi
  __int64 v60; // r9
  unsigned __int64 *v61; // rax
  __int64 v62; // r12
  __int64 v63; // r9
  __int64 v64; // rbx
  __int64 v65; // r12
  __int8 *v66; // rax
  const char *v67; // rax
  __int64 v68; // rdx
  char v69; // cl
  __int16 *v70; // r14
  __int64 v71; // r9
  __int64 v72; // rax
  __int64 v73; // rax
  __int64 v74; // r14
  _BYTE *v75; // rsi
  _QWORD *v76; // rsi
  __int64 v77; // rax
  __int64 v78; // rdx
  __int64 v79; // rdi
  __int64 v80; // r8
  __int64 v81; // r8
  __int64 v82; // rax
  const char *v83; // rsi
  __int64 v84; // rax
  _QWORD *v85; // rax
  __int64 v86; // rax
  int *v87; // rax
  char v88; // al
  char v89; // al
  char v90; // al
  char v91; // al
  unsigned __int8 v92; // [rsp+14h] [rbp-5BCh]
  __int64 v93; // [rsp+18h] [rbp-5B8h]
  int s2; // [rsp+30h] [rbp-5A0h]
  _QWORD *s2c; // [rsp+30h] [rbp-5A0h]
  int *s2a; // [rsp+30h] [rbp-5A0h]
  void *s2b; // [rsp+30h] [rbp-5A0h]
  __int64 v98; // [rsp+40h] [rbp-590h]
  __int64 v99; // [rsp+48h] [rbp-588h]
  __int64 v100; // [rsp+48h] [rbp-588h]
  __int64 v101; // [rsp+48h] [rbp-588h]
  __int64 v102; // [rsp+48h] [rbp-588h]
  __int64 v103; // [rsp+48h] [rbp-588h]
  size_t n; // [rsp+68h] [rbp-568h]
  __int64 v105; // [rsp+70h] [rbp-560h]
  __int64 v106; // [rsp+70h] [rbp-560h]
  unsigned __int8 v107; // [rsp+70h] [rbp-560h]
  unsigned __int64 v108; // [rsp+80h] [rbp-550h]
  int v109; // [rsp+80h] [rbp-550h]
  __int64 v110; // [rsp+80h] [rbp-550h]
  __int64 v111; // [rsp+80h] [rbp-550h]
  unsigned __int64 v112; // [rsp+88h] [rbp-548h]
  __int64 v113; // [rsp+88h] [rbp-548h]
  __int64 v114; // [rsp+88h] [rbp-548h]
  __int64 v115; // [rsp+88h] [rbp-548h]
  __int64 v116; // [rsp+88h] [rbp-548h]
  __int64 v117; // [rsp+88h] [rbp-548h]
  __int64 v118; // [rsp+88h] [rbp-548h]
  __int64 v119; // [rsp+88h] [rbp-548h]
  __int64 v120; // [rsp+88h] [rbp-548h]
  __int64 v121; // [rsp+88h] [rbp-548h]
  unsigned __int8 v122; // [rsp+98h] [rbp-538h]
  __int64 v123; // [rsp+98h] [rbp-538h]
  char v124; // [rsp+A5h] [rbp-52Bh] BYREF
  char v125; // [rsp+A6h] [rbp-52Ah] BYREF
  unsigned __int8 v126; // [rsp+A7h] [rbp-529h] BYREF
  int v127; // [rsp+A8h] [rbp-528h] BYREF
  int v128; // [rsp+ACh] [rbp-524h] BYREF
  int v129; // [rsp+B0h] [rbp-520h] BYREF
  int v130; // [rsp+B4h] [rbp-51Ch] BYREF
  unsigned int v131; // [rsp+B8h] [rbp-518h] BYREF
  int v132; // [rsp+BCh] [rbp-514h] BYREF
  __int64 *v133; // [rsp+C0h] [rbp-510h] BYREF
  unsigned __int64 v134; // [rsp+C8h] [rbp-508h] BYREF
  __int64 v135; // [rsp+D0h] [rbp-500h] BYREF
  __int64 v136; // [rsp+D8h] [rbp-4F8h] BYREF
  __int64 v137; // [rsp+E0h] [rbp-4F0h] BYREF
  __int64 v138; // [rsp+E8h] [rbp-4E8h] BYREF
  __int64 v139; // [rsp+F0h] [rbp-4E0h] BYREF
  __int64 v140; // [rsp+F8h] [rbp-4D8h] BYREF
  char *v141[4]; // [rsp+100h] [rbp-4D0h] BYREF
  _QWORD *v142; // [rsp+120h] [rbp-4B0h] BYREF
  _BYTE *v143; // [rsp+128h] [rbp-4A8h]
  _BYTE *v144; // [rsp+130h] [rbp-4A0h]
  __m128i v145; // [rsp+140h] [rbp-490h] BYREF
  __int16 v146; // [rsp+150h] [rbp-480h]
  __m128i v147; // [rsp+160h] [rbp-470h] BYREF
  char v148; // [rsp+170h] [rbp-460h]
  char v149; // [rsp+171h] [rbp-45Fh]
  __m128i v150; // [rsp+180h] [rbp-450h] BYREF
  __int16 v151; // [rsp+190h] [rbp-440h]
  _BYTE *v152; // [rsp+1A0h] [rbp-430h] BYREF
  size_t v153; // [rsp+1A8h] [rbp-428h]
  _BYTE v154[16]; // [rsp+1B0h] [rbp-420h] BYREF
  _QWORD *v155; // [rsp+1C0h] [rbp-410h] BYREF
  size_t v156; // [rsp+1C8h] [rbp-408h]
  _BYTE v157[16]; // [rsp+1D0h] [rbp-400h] BYREF
  _QWORD *v158; // [rsp+1E0h] [rbp-3F0h] BYREF
  __int64 v159; // [rsp+1E8h] [rbp-3E8h]
  _BYTE v160[16]; // [rsp+1F0h] [rbp-3E0h] BYREF
  __m128i v161; // [rsp+200h] [rbp-3D0h] BYREF
  char v162; // [rsp+210h] [rbp-3C0h]
  char v163; // [rsp+211h] [rbp-3BFh]
  _BYTE *v164; // [rsp+220h] [rbp-3B0h] BYREF
  __int64 v165; // [rsp+228h] [rbp-3A8h]
  _BYTE v166[64]; // [rsp+230h] [rbp-3A0h] BYREF
  __m128i v167; // [rsp+270h] [rbp-360h] BYREF
  int v168; // [rsp+280h] [rbp-350h] BYREF
  _QWORD *v169; // [rsp+288h] [rbp-348h]
  int *v170; // [rsp+290h] [rbp-340h]
  int *v171; // [rsp+298h] [rbp-338h]
  __int64 v172; // [rsp+2A0h] [rbp-330h]
  __int64 v173; // [rsp+2A8h] [rbp-328h]
  __int64 v174; // [rsp+2B0h] [rbp-320h]
  __int64 v175; // [rsp+2B8h] [rbp-318h]
  __int64 v176; // [rsp+2C0h] [rbp-310h]
  __int64 v177; // [rsp+2C8h] [rbp-308h]
  __m128i v178; // [rsp+2D0h] [rbp-300h] BYREF
  int v179; // [rsp+2E0h] [rbp-2F0h] BYREF
  _QWORD *v180; // [rsp+2E8h] [rbp-2E8h]
  int *v181; // [rsp+2F0h] [rbp-2E0h]
  int *v182; // [rsp+2F8h] [rbp-2D8h]
  __int64 v183; // [rsp+300h] [rbp-2D0h]
  __int64 v184; // [rsp+308h] [rbp-2C8h]
  __int64 v185; // [rsp+310h] [rbp-2C0h]
  __int64 v186; // [rsp+318h] [rbp-2B8h]
  __int64 v187; // [rsp+320h] [rbp-2B0h]
  __int64 v188; // [rsp+328h] [rbp-2A8h]
  __m128i v189; // [rsp+330h] [rbp-2A0h] BYREF
  int v190; // [rsp+340h] [rbp-290h]
  __int64 v191; // [rsp+348h] [rbp-288h]
  unsigned __int64 v192[2]; // [rsp+350h] [rbp-280h] BYREF
  char v193; // [rsp+360h] [rbp-270h] BYREF
  char *v194; // [rsp+370h] [rbp-260h]
  __int64 v195; // [rsp+378h] [rbp-258h]
  char v196; // [rsp+380h] [rbp-250h] BYREF
  __int64 v197; // [rsp+390h] [rbp-240h]
  int v198; // [rsp+398h] [rbp-238h]
  char v199; // [rsp+39Ch] [rbp-234h]
  _QWORD v200[5]; // [rsp+3A8h] [rbp-228h] BYREF
  __m128i *v201; // [rsp+3D0h] [rbp-200h] BYREF
  __int64 v202; // [rsp+3D8h] [rbp-1F8h]
  _WORD v203[248]; // [rsp+3E0h] [rbp-1F0h] BYREF

  v122 = a3;
  v6 = *(_QWORD *)(a1 + 56);
  v167.m128i_i64[0] = 0;
  v168 = 0;
  v169 = 0;
  v170 = &v168;
  v171 = &v168;
  v172 = 0;
  v173 = 0;
  v174 = 0;
  v175 = 0;
  v176 = 0;
  v177 = 0;
  v133 = 0;
  if ( (unsigned __int8)sub_388C1F0(a1, (__int64)&v127, &v125, &v128, &v129, &v124)
    || (unsigned __int8)sub_388C2C0(a1, &v130)
    || (unsigned __int8)sub_388C990(a1, &v167)
    || (v112 = *(_QWORD *)(a1 + 56),
        v201 = (__m128i *)"expected type",
        v203[0] = 259,
        (unsigned __int8)sub_3891B00(a1, (__int64 *)&v133, (__int64)&v201, 1)) )
  {
    v7 = 1;
    goto LABEL_3;
  }
  switch ( v127 )
  {
    case 1:
    case 2:
    case 3:
    case 4:
    case 5:
    case 7:
    case 8:
      if ( !a3 )
      {
        HIBYTE(v203[0]) = 1;
        v10 = "invalid linkage for function declaration";
        goto LABEL_12;
      }
      v9 = v128;
      if ( v127 != 7 )
        goto LABEL_16;
      goto LABEL_10;
    case 6:
    case 10:
      HIBYTE(v203[0]) = 1;
      v10 = "invalid function linkage type";
      goto LABEL_12;
    case 9:
      if ( !a3 )
        goto LABEL_15;
      HIBYTE(v203[0]) = 1;
      v10 = "invalid linkage for function definition";
      goto LABEL_12;
    default:
LABEL_15:
      v9 = v128;
LABEL_16:
      if ( v127 != 8 )
        goto LABEL_17;
LABEL_10:
      if ( !v9 )
      {
LABEL_17:
        v11 = sub_1643460((__int64)v133);
        v12 = a1 + 8;
        v7 = v11;
        if ( !(_BYTE)v11 )
        {
          v201 = (__m128i *)"invalid function return type";
          v203[0] = 259;
          v7 = sub_38814C0(a1 + 8, v112, (__int64)&v201);
          goto LABEL_3;
        }
        v13 = *(_QWORD *)(a1 + 56);
        v154[0] = 0;
        v153 = 0;
        v108 = v13;
        v152 = v154;
        v14 = *(_DWORD *)(a1 + 64);
        if ( v14 == 373 )
        {
          sub_2240AE0((unsigned __int64 *)&v152, (unsigned __int64 *)(a1 + 72));
          v12 = a1 + 8;
        }
        else
        {
          if ( v14 != 368 )
          {
            v201 = (__m128i *)"expected function name";
            v203[0] = 259;
            goto LABEL_22;
          }
          if ( *(_DWORD *)(a1 + 104) != (__int64)(*(_QWORD *)(a1 + 1008) - *(_QWORD *)(a1 + 1000)) >> 3 )
          {
            v178.m128i_i64[0] = (__int64)(*(_QWORD *)(a1 + 1008) - *(_QWORD *)(a1 + 1000)) >> 3;
            v189.m128i_i64[0] = (__int64)"function expected to be numbered '%";
            v189.m128i_i64[1] = (__int64)&v178;
            LOWORD(v190) = 2819;
            v201 = &v189;
            v202 = (__int64)"'";
            v203[0] = 770;
LABEL_22:
            v7 = sub_38814C0(v12, v108, (__int64)&v201);
            goto LABEL_23;
          }
        }
        v105 = v12;
        v15 = sub_3887100(v12);
        *(_DWORD *)(a1 + 64) = v15;
        if ( v15 != 12 )
        {
          v16 = *(_QWORD *)(a1 + 56);
          v201 = (__m128i *)"expected '(' in function argument list";
          v203[0] = 259;
          v7 = sub_38814C0(v105, v16, (__int64)&v201);
LABEL_23:
          if ( v152 != v154 )
            j_j___libc_free_0((unsigned __int64)v152);
          goto LABEL_3;
        }
        v201 = (__m128i *)v203;
        v202 = 0x800000000LL;
        v181 = &v179;
        v182 = &v179;
        v155 = v157;
        v178.m128i_i64[0] = 0;
        v179 = 0;
        v180 = 0;
        v183 = 0;
        v184 = 0;
        v185 = 0;
        v186 = 0;
        v187 = 0;
        v188 = 0;
        memset(v141, 0, 24);
        v134 = 0;
        v156 = 0;
        v157[0] = 0;
        v158 = v160;
        v159 = 0;
        v160[0] = 0;
        v132 = 0;
        v135 = 0;
        v136 = 0;
        v137 = 0;
        if ( (unsigned __int8)sub_3892DD0(a1, (__int64)&v201, &v126) )
          goto LABEL_44;
        if ( (unsigned __int8)sub_388ADB0(a1, &v132) )
          goto LABEL_44;
        if ( (unsigned __int8)sub_388FCA0(a1, &v178, (__int64)v141, 0, &v134) )
          goto LABEL_44;
        v17 = v105;
        if ( *(_DWORD *)(a1 + 64) == 90 )
        {
          *(_DWORD *)(a1 + 64) = sub_3887100(v105);
          v22 = sub_388B0A0(a1, (unsigned __int64 *)&v155);
          v17 = v105;
          if ( v22 )
            goto LABEL_44;
        }
        v106 = v17;
        if ( (unsigned __int8)sub_3899C10(a1, v152, v153, &v138) )
          goto LABEL_44;
        v99 = v106;
        v107 = sub_388C5A0(a1, &v131);
        if ( v107 )
          goto LABEL_44;
        v18 = v99;
        if ( *(_DWORD *)(a1 + 64) == 98 )
        {
          *(_DWORD *)(a1 + 64) = sub_3887100(v99);
          v88 = sub_388B0A0(a1, (unsigned __int64 *)&v158);
          v18 = v99;
          if ( v88 )
            goto LABEL_44;
        }
        if ( *(_DWORD *)(a1 + 64) == 99 )
        {
          v103 = v18;
          *(_DWORD *)(a1 + 64) = sub_3887100(v18);
          v91 = sub_389C3E0((__int64 **)a1, &v135);
          v18 = v103;
          if ( v91 )
            goto LABEL_44;
        }
        if ( *(_DWORD *)(a1 + 64) == 100 )
        {
          v102 = v18;
          *(_DWORD *)(a1 + 64) = sub_3887100(v18);
          v90 = sub_389C3E0((__int64 **)a1, &v136);
          v18 = v102;
          if ( v90 )
            goto LABEL_44;
        }
        if ( *(_DWORD *)(a1 + 64) == 271 )
        {
          v101 = v18;
          *(_DWORD *)(a1 + 64) = sub_3887100(v18);
          v89 = sub_389C3E0((__int64 **)a1, &v137);
          v18 = v101;
          if ( v89 )
          {
LABEL_44:
            if ( v158 != (_QWORD *)v160 )
              j_j___libc_free_0((unsigned __int64)v158);
            if ( v155 != (_QWORD *)v157 )
              j_j___libc_free_0((unsigned __int64)v155);
            if ( v141[0] )
              j_j___libc_free_0((unsigned __int64)v141[0]);
            sub_3887AD0(v180);
            v19 = v201;
            v20 = (__m128i *)((char *)v201 + 56 * (unsigned int)v202);
            if ( v201 != v20 )
            {
              do
              {
                v20 = (__m128i *)((char *)v20 - 56);
                v21 = v20[1].m128i_u64[1];
                if ( (unsigned __int64 *)v21 != &v20[2].m128i_u64[1] )
                  j_j___libc_free_0(v21);
              }
              while ( v19 != v20 );
              v19 = v201;
            }
            if ( v19 != (__m128i *)v203 )
              _libc_free((unsigned __int64)v19);
            goto LABEL_23;
          }
        }
        v100 = v178.m128i_i8[0] & 0x20;
        if ( (v178.m128i_i8[0] & 0x20) != 0 )
        {
          v189.m128i_i64[0] = (__int64)"'builtin' attribute not valid on function";
          LOWORD(v190) = 259;
          v7 = sub_38814C0(v18, v134, (__int64)&v189);
          goto LABEL_44;
        }
        v98 = v18;
        v23 = sub_1560E20((__int64)&v178);
        v25 = v98;
        if ( v23 )
        {
          v131 = v184;
          sub_1560700(&v178, 1);
          v25 = v98;
        }
        v92 = a3;
        v164 = v166;
        v165 = 0x800000000LL;
        v142 = 0;
        s2 = v202;
        v143 = 0;
        v26 = 0;
        v144 = 0;
        v93 = v25;
        while ( s2 != (_DWORD)v26 )
        {
          v29 = 56 * v26;
          v27 = v201;
          v30 = &v201->m128i_i8[56 * v26];
          if ( v143 == v144 )
          {
            sub_1277EB0((__int64)&v142, v143, (_QWORD *)v30 + 1);
            v27 = v201;
            v29 = 56 * v26;
          }
          else
          {
            if ( v143 )
            {
              *(_QWORD *)v143 = *((_QWORD *)v30 + 1);
              v27 = v201;
            }
            v143 += 8;
          }
          v28 = &v27->m128i_i8[v29];
          if ( (unsigned int)v165 >= HIDWORD(v165) )
            sub_16CD150((__int64)&v164, v166, 0, 8, v24, v25);
          ++v26;
          *(_QWORD *)&v164[8 * (unsigned int)v165] = *((_QWORD *)v28 + 2);
          LODWORD(v165) = v165 + 1;
        }
        v31 = (unsigned int)v165;
        s2c = v164;
        v32 = sub_1560BF0(*(__int64 **)a1, &v167);
        v33 = sub_1560BF0(*(__int64 **)a1, &v178);
        v139 = sub_155FDB0(*(__int64 **)a1, v33, v32, s2c, v31);
        if ( (unsigned __int8)sub_1560260(&v139, 1, 53) && *((_BYTE *)v133 + 8) )
        {
          v189.m128i_i64[0] = (__int64)"functions with 'sret' argument must return void";
          LOWORD(v190) = 259;
          v107 = sub_38814C0(v93, v112, (__int64)&v189);
        }
        else
        {
          v34 = (__int64 *)sub_1644EA0(v133, v142, (v143 - (_BYTE *)v142) >> 3, v126);
          v35 = sub_1646BA0(v34, 0);
          v36 = v153;
          *a2 = 0;
          v37 = v93;
          v38 = v35;
          if ( v36 )
          {
            v113 = v36;
            s2a = (int *)sub_38902E0(a1 + 904, (__int64)&v152);
            v39 = *(_QWORD *)(a1 + 176);
            if ( (int *)(a1 + 912) == s2a )
            {
              v82 = sub_16321A0(v39, (__int64)v152, v113);
              *a2 = v82;
              if ( v82 )
              {
                v123 = v93;
                v83 = "invalid redefinition of function '";
              }
              else
              {
                v84 = sub_1632000(*(_QWORD *)(a1 + 176), (__int64)v152, v153);
                v37 = v93;
                if ( !v84 )
                  goto LABEL_85;
                v123 = v93;
                v83 = "redefinition of function '@";
              }
              sub_8FD6D0((__int64)&v161, v83, &v152);
              sub_94F930(&v189, (__int64)&v161, "'");
              v150.m128i_i64[0] = (__int64)&v189;
              v41 = v108;
              v151 = 260;
              goto LABEL_78;
            }
            v40 = (_QWORD *)sub_16321A0(v39, (__int64)v152, v113);
            *a2 = (__int64)v40;
            if ( v40 )
            {
              if ( v38 != *v40 )
              {
                v123 = v93;
                sub_8FD6D0((__int64)&v161, "invalid forward reference to function '", &v152);
                sub_94F930(&v189, (__int64)&v161, "' with wrong type!");
                v151 = 260;
                v150.m128i_i64[0] = (__int64)&v189;
                v41 = *((_QWORD *)s2a + 9);
LABEL_78:
                v107 = sub_38814C0(v123, v41, (__int64)&v150);
                sub_2240A30((unsigned __int64 *)&v189);
                sub_2240A30((unsigned __int64 *)&v161);
                goto LABEL_79;
              }
              v42 = sub_220F330(s2a, (_QWORD *)(a1 + 912));
              v43 = v93;
              v44 = *((_QWORD *)v42 + 4);
              v45 = (unsigned __int64)v42;
              if ( (int *)v44 != v42 + 12 )
              {
                j_j___libc_free_0(v44);
                v43 = v93;
              }
              v114 = v43;
              j_j___libc_free_0(v45);
              --*(_QWORD *)(a1 + 944);
              v37 = v114;
LABEL_85:
              v46 = *a2;
              v47 = *(_QWORD *)(a1 + 176);
              if ( *a2 )
              {
                v48 = (unsigned __int64 *)(v47 + 24);
                v49 = (unsigned __int64 *)(v46 + 56);
                v50 = *(unsigned __int64 **)(v46 + 64);
                if ( v47 + 24 != v46 + 56 && v48 != v50 && v49 != v50 )
                {
                  v51 = *v50 & 0xFFFFFFFFFFFFFFF8LL;
                  *(_QWORD *)((*(_QWORD *)(v46 + 56) & 0xFFFFFFFFFFFFFFF8LL) + 8) = v50;
                  *v50 = *v50 & 7 | *(_QWORD *)(v46 + 56) & 0xFFFFFFFFFFFFFFF8LL;
                  v52 = *(_QWORD *)(v47 + 24);
                  *(_QWORD *)(v51 + 8) = v48;
                  v52 &= 0xFFFFFFFFFFFFFFF8LL;
                  *(_QWORD *)(v46 + 56) = v52 | *(_QWORD *)(v46 + 56) & 7LL;
                  *(_QWORD *)(v52 + 8) = v49;
                  *(_QWORD *)(v47 + 24) = *(_QWORD *)(v47 + 24) & 7LL | v51;
                  v46 = *a2;
                }
                goto LABEL_90;
              }
LABEL_138:
              v110 = v47;
              v120 = v37;
              LOWORD(v190) = 260;
              v189.m128i_i64[0] = (__int64)&v152;
              v46 = sub_1648B60(120);
              v37 = v120;
              v81 = v110;
              if ( v46 )
              {
                v111 = v120;
                v121 = v46;
                sub_15E2490(v46, (__int64)v34, 0, (__int64)&v189, v81);
                v37 = v111;
                v46 = v121;
              }
              *a2 = v46;
LABEL_90:
              if ( !v153 )
              {
                v189.m128i_i64[0] = v46;
                v75 = *(_BYTE **)(a1 + 1008);
                if ( v75 == *(_BYTE **)(a1 + 1016) )
                {
                  v119 = v37;
                  sub_167C6C0(a1 + 1000, v75, &v189);
                  v46 = *a2;
                  v37 = v119;
                }
                else
                {
                  if ( v75 )
                    *(_QWORD *)v75 = v46;
                  *(_QWORD *)(a1 + 1008) += 8LL;
                  v46 = *a2;
                }
              }
              v53 = v127;
              v54 = v127 & 0xF;
              if ( (unsigned int)(v127 - 7) > 1 )
              {
                v69 = v54 | *(_BYTE *)(v46 + 32) & 0xF0;
                *(_BYTE *)(v46 + 32) = v69;
                if ( (v53 & 0xFu) - 7 > 1 && ((v69 & 0x30) == 0 || v54 == 9) )
                {
LABEL_94:
                  if ( v124 )
                    *(_BYTE *)(*a2 + 33) |= 0x40u;
                  v55 = *a2;
                  v56 = (16 * (v128 & 3)) | *(_BYTE *)(*a2 + 32) & 0xCF;
                  *(_BYTE *)(*a2 + 32) = v56;
                  if ( (v56 & 0xFu) - 7 <= 1 || (v56 & 0x30) != 0 && (v56 & 0xF) != 9 )
                    *(_BYTE *)(v55 + 33) |= 0x40u;
                  v115 = v37;
                  *(_BYTE *)(*a2 + 33) = v129 & 3 | *(_BYTE *)(*a2 + 33) & 0xFC;
                  v57 = v131;
                  v58 = v139;
                  *(_WORD *)(*a2 + 18) = (16 * v130) | *(_WORD *)(*a2 + 18) & 0xC00F;
                  *(_QWORD *)(*a2 + 112) = v58;
                  *(_BYTE *)(*a2 + 32) = ((_BYTE)v132 << 6) | *(_BYTE *)(*a2 + 32) & 0x3F;
                  sub_15E4CC0(*a2, v57);
                  sub_15E5D20(*a2, v155, v156);
                  v59 = v137;
                  *(_QWORD *)(*a2 + 48) = v138;
                  sub_15E3D80(*a2, v59);
                  v60 = v115;
                  if ( v159 )
                  {
                    v74 = *a2;
                    sub_2241BD0(v189.m128i_i64, (__int64)&v158);
                    sub_15E4280(v74, &v189);
                    sub_2240A30((unsigned __int64 *)&v189);
                    v60 = v115;
                  }
                  v116 = v60;
                  sub_15E3F20(*a2, v135);
                  sub_15E40D0(*a2, v136);
                  v189.m128i_i64[0] = *a2;
                  v61 = sub_3898320((_QWORD *)(a1 + 1128), (unsigned __int64 *)&v189);
                  sub_3887600((__int64)v61, v141);
                  v62 = *a2;
                  v63 = v116;
                  if ( (*(_BYTE *)(v62 + 18) & 1) != 0 )
                  {
                    sub_15E08E0(v62, (__int64)v141);
                    v63 = v116;
                  }
                  v117 = v63;
                  v109 = v202;
                  v64 = *(_QWORD *)(v62 + 88);
                  while ( v109 != (_DWORD)v100 )
                  {
                    v65 = 7 * v100;
                    v66 = &v201->m128i_i8[56 * v100];
                    if ( *((_QWORD *)v66 + 4) )
                    {
                      LOWORD(v190) = 260;
                      v189.m128i_i64[0] = (__int64)(v66 + 24);
                      sub_164B780(v64, v189.m128i_i64);
                      s2b = (void *)v201[1].m128i_i64[v65 + 1];
                      n = v201[2].m128i_u64[v65];
                      v67 = sub_1649960(v64);
                      if ( n != v68 || n && memcmp(v67, s2b, n) )
                      {
                        sub_8FD6D0((__int64)&v161, "redefinition of argument '%", &v201[1].m128i_i64[v65 + 1]);
                        sub_94F930(&v189, (__int64)&v161, "'");
                        v151 = 260;
                        v150.m128i_i64[0] = (__int64)&v189;
                        v107 = sub_38814C0(v117, v201->m128i_u64[7 * v100], (__int64)&v150);
                        sub_2240A30((unsigned __int64 *)&v189);
                        sub_2240A30((unsigned __int64 *)&v161);
                        goto LABEL_79;
                      }
                    }
                    ++v100;
                    v64 += 40;
                  }
                  if ( !v92 )
                  {
                    v189.m128i_i32[0] = 0;
                    v192[0] = (unsigned __int64)&v193;
                    v189.m128i_i64[1] = 0;
                    v191 = 0;
                    v192[1] = 0;
                    v193 = 0;
                    v194 = &v196;
                    v195 = 0;
                    v196 = 0;
                    v198 = 1;
                    v197 = 0;
                    v199 = 0;
                    v70 = (__int16 *)sub_1698280();
                    sub_169D3F0((__int64)&v161, 0.0);
                    sub_169E320(v200, v161.m128i_i64, v70);
                    sub_1698460((__int64)&v161);
                    v200[4] = 0;
                    v71 = v117;
                    if ( v153 )
                    {
                      v189.m128i_i32[0] = 3;
                      sub_2240AE0(v192, (unsigned __int64 *)&v152);
                      v71 = v117;
                    }
                    else
                    {
                      v72 = *(_QWORD *)(a1 + 1008) - *(_QWORD *)(a1 + 1000);
                      v189.m128i_i32[0] = 1;
                      v190 = (v72 >> 3) - 1;
                    }
                    v118 = v71;
                    v73 = sub_38911E0(a1 + 1072, (__int64)&v189);
                    if ( v73 != a1 + 1080 )
                    {
                      v163 = 1;
                      v161.m128i_i64[0] = (__int64)"cannot take blockaddress inside a declaration";
                      v162 = 3;
                      v122 = sub_38814C0(v118, *(_QWORD *)(v73 + 40), (__int64)&v161);
                    }
                    sub_388AE20((__int64)&v189);
                    v107 = v122;
                  }
                  goto LABEL_79;
                }
              }
              else
              {
                *(_BYTE *)(v46 + 32) = v54 | *(_BYTE *)(v46 + 32) & 0xC0;
              }
              *(_BYTE *)(v46 + 33) |= 0x40u;
              goto LABEL_94;
            }
            v189.m128i_i64[0] = (__int64)"invalid forward reference to function as global value!";
            LOWORD(v190) = 259;
            v107 = sub_38814C0(v93, *((_QWORD *)s2a + 9), (__int64)&v189);
          }
          else
          {
            v76 = (_QWORD *)(a1 + 960);
            v77 = (__int64)(*(_QWORD *)(a1 + 1008) - *(_QWORD *)(a1 + 1000)) >> 3;
            v78 = *(_QWORD *)(a1 + 968);
            v79 = a1 + 960;
            while ( v78 )
            {
              v80 = *(_QWORD *)(v78 + 24);
              if ( (unsigned int)((__int64)(*(_QWORD *)(a1 + 1008) - *(_QWORD *)(a1 + 1000)) >> 3) <= *(_DWORD *)(v78 + 32) )
              {
                v80 = *(_QWORD *)(v78 + 16);
                v79 = v78;
              }
              v78 = v80;
            }
            if ( v76 == (_QWORD *)v79 || (unsigned int)v77 < *(_DWORD *)(v79 + 32) )
            {
              v47 = *(_QWORD *)(a1 + 176);
              goto LABEL_138;
            }
            v85 = *(_QWORD **)(v79 + 40);
            *a2 = (__int64)v85;
            if ( v38 == *v85 )
            {
              v87 = sub_220F330((int *)v79, v76);
              j_j___libc_free_0((unsigned __int64)v87);
              --*(_QWORD *)(a1 + 992);
              v37 = v93;
              goto LABEL_85;
            }
            v161.m128i_i64[0] = (__int64)"' disagree";
            v86 = *(_QWORD *)(a1 + 1008) - *(_QWORD *)(a1 + 1000);
            v163 = 1;
            v162 = 3;
            v140 = v86 >> 3;
            v145.m128i_i64[0] = (__int64)&v140;
            v147.m128i_i64[0] = (__int64)"type of definition and forward reference of '@";
            v146 = 267;
            v149 = 1;
            v148 = 3;
            sub_14EC200(&v150, &v147, &v145);
            sub_14EC200(&v189, &v150, &v161);
            v107 = sub_38814C0(v93, v108, (__int64)&v189);
          }
        }
LABEL_79:
        if ( v164 != v166 )
          _libc_free((unsigned __int64)v164);
        sub_388FA10((unsigned __int64 *)&v142);
        v7 = v107;
        goto LABEL_44;
      }
      HIBYTE(v203[0]) = 1;
      v10 = "symbol with local linkage must have default visibility";
LABEL_12:
      v201 = (__m128i *)v10;
      LOBYTE(v203[0]) = 3;
      v7 = sub_38814C0(a1 + 8, v6, (__int64)&v201);
LABEL_3:
      sub_3887AD0(v169);
      return v7;
  }
}
