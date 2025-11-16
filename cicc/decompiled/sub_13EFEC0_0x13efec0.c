// Function: sub_13EFEC0
// Address: 0x13efec0
//
void __fastcall sub_13EFEC0(__int64 a1)
{
  unsigned __int64 v2; // rbx
  const void *v3; // r13
  __int64 v4; // r12
  __int64 v5; // rax
  __int64 *v6; // rax
  __int64 v7; // rbx
  __int64 v8; // r12
  unsigned __int8 v9; // al
  __int64 v10; // rax
  __int64 v11; // r14
  __int64 v12; // rbx
  __int64 v13; // rax
  char v14; // al
  char v15; // dl
  __int64 v16; // rax
  bool v17; // zf
  unsigned int v18; // edx
  int v19; // ecx
  int v20; // ecx
  int v21; // r9d
  unsigned int v22; // r8d
  unsigned __int64 v23; // rdi
  unsigned __int64 v24; // rdi
  unsigned int i; // edx
  _QWORD *v26; // rdi
  unsigned int v27; // edx
  __int64 v28; // r13
  __int64 v29; // r14
  __int64 v30; // rax
  __int64 v31; // r10
  __int64 v32; // rsi
  unsigned __int8 v33; // al
  __int64 v34; // rdi
  char v35; // dl
  __int64 v36; // r13
  char v37; // al
  __int64 v38; // r8
  __int64 v39; // r8
  __int64 v40; // rsi
  _BYTE *v41; // rdi
  __int64 v42; // rax
  unsigned int v43; // eax
  __int64 v44; // r13
  char v45; // al
  int v46; // eax
  _BYTE *v47; // rsi
  unsigned __int64 v48; // rdx
  char v49; // al
  __int64 v50; // rax
  char v51; // dl
  unsigned int v52; // eax
  int v53; // eax
  __int64 v54; // rdx
  __int64 v55; // rdi
  unsigned int v56; // ecx
  _QWORD *v57; // rax
  _QWORD *k; // rdx
  __int64 v59; // rax
  char v60; // al
  __int64 v61; // rdi
  unsigned int v62; // eax
  __int64 v63; // rax
  int v64; // edx
  __int64 v65; // rax
  __int64 v66; // rcx
  unsigned int v67; // eax
  __int64 v68; // rax
  __int64 v69; // rdi
  __int64 v70; // rsi
  int v71; // r13d
  int v72; // eax
  __int64 v73; // rsi
  __int64 v74; // rax
  __int64 v75; // rdx
  __int64 v76; // rcx
  int v77; // eax
  __int64 v78; // rax
  char v79; // si
  __int64 v80; // rsi
  __int64 v81; // rax
  __int64 v82; // rax
  __int32 v83; // eax
  __int64 v84; // rax
  __int64 v85; // rsi
  unsigned __int64 *v86; // rdi
  __int64 v87; // rax
  char v88; // si
  __int64 v89; // rsi
  __int64 v90; // rax
  __int64 v91; // rax
  __int32 v92; // eax
  __int64 v93; // rax
  __int64 v94; // rsi
  _QWORD *v95; // rdi
  unsigned int v96; // eax
  __int64 v97; // rax
  unsigned __int64 v98; // rax
  unsigned __int64 v99; // rax
  int v100; // ebx
  __int64 v101; // r12
  _QWORD *v102; // rax
  __int64 v103; // rdx
  _QWORD *j; // rdx
  __int64 v105; // rsi
  __int64 v106; // rax
  __int64 v107; // rdi
  __int64 v108; // rsi
  int v109; // r13d
  __int64 v110; // rax
  int v111; // eax
  __int64 v112; // rsi
  __int64 v113; // rax
  __int64 v114; // r14
  _QWORD *v115; // rax
  __int64 v116; // rax
  __int64 v117; // r14
  __int64 v118; // [rsp+8h] [rbp-218h]
  __int64 v119; // [rsp+8h] [rbp-218h]
  char v120; // [rsp+13h] [rbp-20Dh]
  int v121; // [rsp+14h] [rbp-20Ch]
  __int64 v122; // [rsp+18h] [rbp-208h]
  __int64 v123; // [rsp+18h] [rbp-208h]
  __int64 v124; // [rsp+20h] [rbp-200h]
  char v125; // [rsp+20h] [rbp-200h]
  _QWORD *v126; // [rsp+20h] [rbp-200h]
  _QWORD *v127; // [rsp+20h] [rbp-200h]
  __int64 v128; // [rsp+20h] [rbp-200h]
  __int64 v129; // [rsp+20h] [rbp-200h]
  __int64 v130; // [rsp+20h] [rbp-200h]
  __int64 v131; // [rsp+20h] [rbp-200h]
  char v132; // [rsp+28h] [rbp-1F8h]
  __int64 v133; // [rsp+28h] [rbp-1F8h]
  char v134; // [rsp+28h] [rbp-1F8h]
  char v135; // [rsp+28h] [rbp-1F8h]
  char v136; // [rsp+28h] [rbp-1F8h]
  char v137; // [rsp+28h] [rbp-1F8h]
  char v138; // [rsp+28h] [rbp-1F8h]
  __int64 v139; // [rsp+28h] [rbp-1F8h]
  char v140; // [rsp+28h] [rbp-1F8h]
  __int64 v141; // [rsp+28h] [rbp-1F8h]
  __int64 v142; // [rsp+28h] [rbp-1F8h]
  unsigned __int64 v143; // [rsp+28h] [rbp-1F8h]
  unsigned __int64 v144; // [rsp+28h] [rbp-1F8h]
  __int64 v145; // [rsp+30h] [rbp-1F0h] BYREF
  _DWORD v146[6]; // [rsp+38h] [rbp-1E8h] BYREF
  __int64 v147; // [rsp+50h] [rbp-1D0h] BYREF
  unsigned int v148; // [rsp+58h] [rbp-1C8h]
  __int64 v149; // [rsp+60h] [rbp-1C0h] BYREF
  unsigned int v150; // [rsp+68h] [rbp-1B8h]
  int v151; // [rsp+70h] [rbp-1B0h] BYREF
  __int64 v152; // [rsp+78h] [rbp-1A8h]
  unsigned int v153; // [rsp+80h] [rbp-1A0h]
  __int64 v154; // [rsp+88h] [rbp-198h]
  unsigned int v155; // [rsp+90h] [rbp-190h]
  unsigned __int64 v156; // [rsp+A0h] [rbp-180h] BYREF
  __int64 v157; // [rsp+A8h] [rbp-178h] BYREF
  unsigned __int64 v158; // [rsp+B0h] [rbp-170h] BYREF
  __int64 v159; // [rsp+B8h] [rbp-168h]
  unsigned int v160; // [rsp+C0h] [rbp-160h]
  unsigned __int64 v161; // [rsp+D0h] [rbp-150h] BYREF
  __int64 v162; // [rsp+D8h] [rbp-148h] BYREF
  unsigned __int64 v163; // [rsp+E0h] [rbp-140h] BYREF
  __int64 v164; // [rsp+E8h] [rbp-138h]
  unsigned int v165; // [rsp+F0h] [rbp-130h]
  __int64 v166; // [rsp+100h] [rbp-120h] BYREF
  __int64 v167; // [rsp+108h] [rbp-118h]
  __int64 v168; // [rsp+110h] [rbp-110h] BYREF
  __int64 v169; // [rsp+118h] [rbp-108h]
  unsigned int v170; // [rsp+120h] [rbp-100h]
  __m128i v171; // [rsp+130h] [rbp-F0h] BYREF
  unsigned int v172; // [rsp+140h] [rbp-E0h]
  unsigned __int64 v173; // [rsp+148h] [rbp-D8h] BYREF
  unsigned int v174; // [rsp+150h] [rbp-D0h]
  _BYTE *v175; // [rsp+160h] [rbp-C0h] BYREF
  __int64 v176; // [rsp+168h] [rbp-B8h]
  _BYTE v177[176]; // [rsp+170h] [rbp-B0h] BYREF

  v2 = *(unsigned int *)(a1 + 104);
  v3 = *(const void **)(a1 + 96);
  v175 = v177;
  v176 = 0x800000000LL;
  v4 = 16 * v2;
  if ( v2 > 8 )
  {
    sub_16CD150(&v175, v177, v2, 16);
    v41 = &v175[16 * (unsigned int)v176];
  }
  else
  {
    if ( !v4 )
      goto LABEL_3;
    v41 = v177;
  }
  memcpy(v41, v3, 16 * v2);
  LODWORD(v4) = v176;
LABEL_3:
  v5 = *(unsigned int *)(a1 + 104);
  v121 = 500;
  LODWORD(v176) = v4 + v2;
  if ( !(_DWORD)v5 )
    goto LABEL_61;
  do
  {
    v6 = (__int64 *)(*(_QWORD *)(a1 + 96) + 16 * v5 - 16);
    v7 = v6[1];
    v8 = *v6;
    if ( *(_BYTE *)(v7 + 16) <= 0x10u || (v120 = sub_13E8A40(a1, v6[1], *v6)) != 0 )
    {
LABEL_42:
      v18 = *(_DWORD *)(a1 + 104);
LABEL_43:
      v19 = *(_DWORD *)(a1 + 264);
      v5 = v18 - 1;
      *(_DWORD *)(a1 + 104) = v5;
      if ( v19 )
      {
        v20 = v19 - 1;
        v21 = 1;
        v22 = (unsigned int)v7 >> 9;
        v23 = (((v22 ^ ((unsigned int)v7 >> 4)
               | ((unsigned __int64)(((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4)) << 32))
              - 1
              - ((unsigned __int64)(v22 ^ ((unsigned int)v7 >> 4)) << 32)) >> 22)
            ^ ((v22 ^ ((unsigned int)v7 >> 4)
              | ((unsigned __int64)(((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4)) << 32))
             - 1
             - ((unsigned __int64)(v22 ^ ((unsigned int)v7 >> 4)) << 32));
        v24 = ((9 * (((v23 - 1 - (v23 << 13)) >> 8) ^ (v23 - 1 - (v23 << 13)))) >> 15)
            ^ (9 * (((v23 - 1 - (v23 << 13)) >> 8) ^ (v23 - 1 - (v23 << 13))));
        for ( i = v20 & (((v24 - 1 - (v24 << 27)) >> 31) ^ (v24 - 1 - ((_DWORD)v24 << 27))); ; i = v20 & v27 )
        {
          v26 = (_QWORD *)(*(_QWORD *)(a1 + 248) + 16LL * i);
          if ( v8 == *v26 && v7 == v26[1] )
            break;
          if ( *v26 == -8 && v26[1] == -8 )
            goto LABEL_30;
          v27 = v21 + i;
          ++v21;
        }
        *v26 = -16;
        v26[1] = -16;
        v5 = *(unsigned int *)(a1 + 104);
        --*(_DWORD *)(a1 + 256);
        ++*(_DWORD *)(a1 + 260);
      }
LABEL_30:
      if ( !(_DWORD)v5 )
        goto LABEL_61;
      goto LABEL_31;
    }
    v151 = 0;
    v9 = *(_BYTE *)(v7 + 16);
    if ( v9 <= 0x17u || *(_QWORD *)(v7 + 40) != v8 )
    {
      LODWORD(v161) = 0;
      v10 = *(_QWORD *)(*(_QWORD *)(v8 + 56) + 80LL);
      if ( v10 && v8 == v10 - 24 )
      {
        v28 = *(_QWORD *)v7;
        if ( *(_BYTE *)(*(_QWORD *)v7 + 8LL) == 15
          && ((unsigned __int8)sub_14BFF20(v7, *(_QWORD *)(a1 + 280), 0, 0, 0, 0)
           || (unsigned __int8)sub_13E7760(v7, v8)
           && !(unsigned __int8)sub_15E4690(*(_QWORD *)(v8 + 56), *(_DWORD *)(v28 + 8) >> 8)) )
        {
          v42 = sub_1599A20(v28);
          v171.m128i_i32[0] = 0;
          if ( *(_BYTE *)(v42 + 16) != 9 )
            sub_13EA940(v171.m128i_i32, v42);
        }
        else
        {
          v171.m128i_i32[0] = 4;
        }
        sub_13E8810((int *)&v161, (unsigned int *)&v171);
        if ( v171.m128i_i32[0] == 3 )
        {
          if ( v174 > 0x40 && v173 )
            j_j___libc_free_0_0(v173);
          if ( v172 > 0x40 && v171.m128i_i64[1] )
            j_j___libc_free_0_0(v171.m128i_i64[1]);
        }
      }
      else
      {
        v11 = *(_QWORD *)(v8 + 8);
        if ( v11 )
        {
          while ( (unsigned __int8)(*(_BYTE *)(sub_1648700(v11) + 16) - 25) > 9u )
          {
            v11 = *(_QWORD *)(v11 + 8);
            if ( !v11 )
              goto LABEL_25;
          }
          v124 = v7;
          v12 = v11;
LABEL_15:
          LODWORD(v166) = 0;
          v13 = sub_1648700(v12);
          v14 = sub_13EFC20(a1, v124, *(_QWORD *)(v13 + 40), v8, (int *)&v166, 0);
          if ( v14 )
          {
            v132 = v14;
            sub_13EACF0((int *)&v161, (unsigned int *)&v166);
            if ( (_DWORD)v161 != 4 )
            {
              if ( (_DWORD)v166 != 3 )
                goto LABEL_495;
              if ( v170 > 0x40 && v169 )
                j_j___libc_free_0_0(v169);
              if ( (unsigned int)v168 > 0x40 && v167 )
              {
                j_j___libc_free_0_0(v167);
                v12 = *(_QWORD *)(v12 + 8);
                if ( v12 )
                  goto LABEL_14;
              }
              else
              {
LABEL_495:
                while ( 1 )
                {
                  v12 = *(_QWORD *)(v12 + 8);
                  if ( !v12 )
                    break;
LABEL_14:
                  if ( (unsigned __int8)(*(_BYTE *)(sub_1648700(v12) + 16) - 25) <= 9u )
                    goto LABEL_15;
                }
              }
              v7 = v124;
              goto LABEL_25;
            }
            v7 = v124;
            v35 = v132;
            v36 = *(_QWORD *)v124;
            if ( *(_BYTE *)(*(_QWORD *)v124 + 8LL) == 15 )
            {
              v37 = sub_13E7760(v124, v8);
              v35 = v132;
              if ( v37 )
              {
                v49 = sub_15E4690(*(_QWORD *)(v8 + 56), *(_DWORD *)(v36 + 8) >> 8);
                v35 = v132;
                if ( !v49 )
                {
                  v50 = sub_1599A20(v36);
                  v51 = v132;
                  v171.m128i_i32[0] = 0;
                  if ( *(_BYTE *)(v50 + 16) != 9 )
                  {
                    sub_13EA940(v171.m128i_i32, v50);
                    v51 = v132;
                  }
                  v125 = v51;
                  sub_13E8810((int *)&v161, (unsigned int *)&v171);
                  sub_13EA000((__int64)&v171);
                  v35 = v125;
                }
              }
            }
            v138 = v35;
            sub_13E8810(&v151, (unsigned int *)&v161);
            v15 = v138;
          }
          else
          {
            v7 = v124;
            v15 = 0;
          }
LABEL_105:
          if ( (_DWORD)v166 == 3 )
          {
            if ( v170 > 0x40 && v169 )
            {
              v136 = v15;
              j_j___libc_free_0_0(v169);
              v15 = v136;
            }
            if ( (unsigned int)v168 > 0x40 && v167 )
            {
              v137 = v15;
              j_j___libc_free_0_0(v167);
              v15 = v137;
            }
          }
LABEL_26:
          if ( (_DWORD)v161 == 3 )
          {
            if ( v165 > 0x40 && v164 )
            {
              v134 = v15;
              j_j___libc_free_0_0(v164);
              v15 = v134;
            }
            if ( (unsigned int)v163 > 0x40 )
            {
              v34 = v162;
              if ( v162 )
                goto LABEL_103;
            }
          }
          goto LABEL_27;
        }
      }
LABEL_25:
      sub_13E8810(&v151, (unsigned int *)&v161);
      v15 = 1;
      goto LABEL_26;
    }
    if ( v9 == 77 )
    {
      LODWORD(v161) = 0;
      if ( (*(_DWORD *)(v7 + 20) & 0xFFFFFFF) == 0 )
        goto LABEL_25;
      v29 = 0;
      v133 = 8LL * (*(_DWORD *)(v7 + 20) & 0xFFFFFFF);
      while ( 1 )
      {
        v30 = (*(_BYTE *)(v7 + 23) & 0x40) != 0 ? *(_QWORD *)(v7 - 8) : v7 - 24LL * (*(_DWORD *)(v7 + 20) & 0xFFFFFFF);
        v31 = *(_QWORD *)(v30 + v29 + 24LL * *(unsigned int *)(v7 + 56) + 8);
        v32 = *(_QWORD *)(v30 + 3 * v29);
        LODWORD(v166) = 0;
        v33 = *(_BYTE *)(v32 + 16);
        if ( v33 <= 0x10u )
          break;
        if ( !(unsigned __int8)sub_13EEDA0(a1, (unsigned __int8 *)v32, v31, v8, (int *)&v166, v7) )
        {
          v15 = 0;
          goto LABEL_105;
        }
LABEL_94:
        sub_13EACF0((int *)&v161, (unsigned int *)&v166);
        if ( (_DWORD)v161 == 4 )
        {
          sub_13E8810(&v151, (unsigned int *)&v161);
          v15 = 1;
          goto LABEL_105;
        }
        if ( (_DWORD)v166 == 3 )
        {
          if ( v170 > 0x40 && v169 )
            j_j___libc_free_0_0(v169);
          if ( (unsigned int)v168 > 0x40 && v167 )
            j_j___libc_free_0_0(v167);
        }
        v29 += 8;
        if ( v29 == v133 )
          goto LABEL_25;
      }
      v171.m128i_i32[0] = 0;
      if ( v33 != 9 )
      {
        if ( v33 == 13 )
        {
          v148 = *(_DWORD *)(v32 + 32);
          if ( v148 > 0x40 )
            sub_16A4FD0(&v147, v32 + 24);
          else
            v147 = *(_QWORD *)(v32 + 24);
          sub_1589870(&v156, &v147);
          if ( v171.m128i_i32[0] == 3 )
          {
            if ( !(unsigned __int8)sub_158A120(&v156) )
            {
              if ( v172 > 0x40 && v171.m128i_i64[1] )
                j_j___libc_free_0_0(v171.m128i_i64[1]);
              v171.m128i_i64[1] = v156;
              v52 = v157;
              LODWORD(v157) = 0;
              v172 = v52;
              if ( v174 <= 0x40 || !v173 )
              {
                v173 = v158;
                v174 = v159;
                goto LABEL_90;
              }
              j_j___libc_free_0_0(v173);
              v43 = v157;
              v173 = v158;
              v174 = v159;
LABEL_176:
              if ( v43 > 0x40 && v156 )
                j_j___libc_free_0_0(v156);
              goto LABEL_90;
            }
          }
          else if ( !(unsigned __int8)sub_158A120(&v156) )
          {
            v171.m128i_i32[0] = 3;
            v172 = v157;
            v171.m128i_i64[1] = v156;
            v174 = v159;
            v173 = v158;
LABEL_90:
            if ( v148 > 0x40 && v147 )
              j_j___libc_free_0_0(v147);
            goto LABEL_93;
          }
          if ( v171.m128i_i32[0] != 4 )
          {
            if ( (unsigned int)(v171.m128i_i32[0] - 1) > 1 )
            {
              if ( v171.m128i_i32[0] == 3 )
              {
                if ( v174 > 0x40 && v173 )
                  j_j___libc_free_0_0(v173);
                if ( v172 > 0x40 && v171.m128i_i64[1] )
                  j_j___libc_free_0_0(v171.m128i_i64[1]);
              }
            }
            else
            {
              v171.m128i_i64[1] = 0;
            }
            v171.m128i_i32[0] = 4;
          }
          if ( (unsigned int)v159 > 0x40 && v158 )
            j_j___libc_free_0_0(v158);
          v43 = v157;
          goto LABEL_176;
        }
        v171.m128i_i32[0] = 1;
        v171.m128i_i64[1] = v32;
      }
LABEL_93:
      sub_13E8810((int *)&v166, (unsigned int *)&v171);
      if ( v171.m128i_i32[0] == 3 )
      {
        if ( v174 > 0x40 && v173 )
          j_j___libc_free_0_0(v173);
        if ( v172 > 0x40 && v171.m128i_i64[1] )
          j_j___libc_free_0_0(v171.m128i_i64[1]);
      }
      goto LABEL_94;
    }
    if ( v9 != 79 )
    {
      v44 = *(_QWORD *)v7;
      v45 = *(_BYTE *)(*(_QWORD *)v7 + 8LL);
      if ( v45 == 15 )
      {
        if ( (unsigned __int8)sub_14BFF20(v7, *(_QWORD *)(a1 + 280), 0, 0, 0, 0) )
        {
          v74 = sub_1599A20(v44);
          v171.m128i_i32[0] = 0;
          if ( *(_BYTE *)(v74 + 16) != 9 )
            sub_13EA940(v171.m128i_i32, v74);
          sub_13E8810(&v151, (unsigned int *)&v171);
          sub_13EA000((__int64)&v171);
          goto LABEL_51;
        }
        v45 = *(_BYTE *)(*(_QWORD *)v7 + 8LL);
      }
      if ( v45 != 11 )
        goto LABEL_215;
      v46 = *(unsigned __int8 *)(v7 + 16);
      if ( (unsigned int)(v46 - 60) > 0xC )
      {
        v62 = v46 - 35;
        if ( v62 <= 0x11 && *(_BYTE *)(*(_QWORD *)(v7 - 24) + 16LL) == 13 )
        {
          switch ( v62 )
          {
            case 0u:
            case 2u:
            case 4u:
            case 6u:
            case 0xCu:
            case 0xDu:
            case 0xEu:
            case 0xFu:
            case 0x10u:
              v105 = *(_QWORD *)(v7 - 48);
              if ( *(_BYTE *)(v105 + 16) <= 0x10u )
                goto LABEL_401;
              if ( (unsigned __int8)sub_13E8A40(a1, v105, v8)
                || (v106 = *(_QWORD *)(v7 - 48),
                    v171.m128i_i64[0] = v8,
                    v171.m128i_i64[1] = v106,
                    !(unsigned __int8)sub_13ED650(a1, &v171)) )
              {
                v105 = *(_QWORD *)(v7 - 48);
LABEL_401:
                v107 = *(_QWORD *)(a1 + 280);
                v108 = *(_QWORD *)v105;
                v109 = 1;
                while ( 2 )
                {
                  switch ( *(_BYTE *)(v108 + 8) )
                  {
                    case 1:
                      v111 = 16;
                      goto LABEL_409;
                    case 2:
                      v111 = 32;
                      goto LABEL_409;
                    case 3:
                    case 9:
                      v111 = 64;
                      goto LABEL_409;
                    case 4:
                      v111 = 80;
                      goto LABEL_409;
                    case 5:
                    case 6:
                      v111 = 128;
                      goto LABEL_409;
                    case 7:
                      v111 = 8 * sub_15A9520(v107, 0);
                      goto LABEL_409;
                    case 0xB:
                      v111 = *(_DWORD *)(v108 + 8) >> 8;
                      goto LABEL_409;
                    case 0xD:
                      v111 = 8 * *(_QWORD *)sub_15A9930(v107, v108);
                      goto LABEL_409;
                    case 0xE:
                      v114 = *(_QWORD *)(v108 + 32);
                      v129 = *(_QWORD *)(a1 + 280);
                      v122 = *(_QWORD *)(v108 + 24);
                      v143 = (unsigned int)sub_15A9FE0(v107, v122);
                      v111 = 8
                           * v114
                           * v143
                           * ((v143 + ((unsigned __int64)(sub_127FA20(v129, v122) + 7) >> 3) - 1)
                            / v143);
                      goto LABEL_409;
                    case 0xF:
                      v111 = 8 * sub_15A9520(v107, *(_DWORD *)(v108 + 8) >> 8);
LABEL_409:
                      sub_15897D0(&v156, (unsigned int)(v109 * v111), 1);
                      v112 = *(_QWORD *)(v7 - 48);
                      if ( *(_BYTE *)(v112 + 16) <= 0x10u )
                        goto LABEL_412;
                      if ( !(unsigned __int8)sub_13E8A40(a1, v112, v8) )
                        goto LABEL_413;
                      v112 = *(_QWORD *)(v7 - 48);
LABEL_412:
                      sub_13E9630(v171.m128i_i32, a1, v112, v8);
                      sub_13EE9C0(a1, *(_QWORD *)(v7 - 48), v171.m128i_i32, v7);
                      if ( v171.m128i_i32[0] != 3 )
                        goto LABEL_413;
                      if ( (unsigned int)v157 <= 0x40 && v172 <= 0x40 )
                      {
                        LODWORD(v157) = v172;
                        v156 = v171.m128i_i64[1] & (0xFFFFFFFFFFFFFFFFLL >> -(char)v172);
                      }
                      else
                      {
                        sub_16A51C0(&v156, &v171.m128i_u64[1]);
                      }
                      if ( (unsigned int)v159 <= 0x40 && v174 <= 0x40 )
                      {
                        LODWORD(v159) = v174;
                        v158 = v173 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v174);
                        if ( v171.m128i_i32[0] != 3 )
                          goto LABEL_413;
                      }
                      else
                      {
                        sub_16A51C0(&v158, &v173);
                        if ( v171.m128i_i32[0] != 3 )
                          goto LABEL_413;
                        if ( v174 > 0x40 && v173 )
                          j_j___libc_free_0_0(v173);
                      }
                      if ( v172 > 0x40 && v171.m128i_i64[1] )
                        j_j___libc_free_0_0(v171.m128i_i64[1]);
LABEL_413:
                      v113 = *(_QWORD *)(v7 - 24);
                      v171.m128i_i32[2] = *(_DWORD *)(v113 + 32);
                      if ( v171.m128i_i32[2] > 0x40u )
                        sub_16A4FD0(&v171, v113 + 24);
                      else
                        v171.m128i_i64[0] = *(_QWORD *)(v113 + 24);
                      sub_1589870(&v161, &v171);
                      if ( v171.m128i_i32[2] > 0x40u && v171.m128i_i64[0] )
                        j_j___libc_free_0_0(v171.m128i_i64[0]);
                      sub_1590D80(&v166, &v156, (unsigned int)*(unsigned __int8 *)(v7 + 16) - 24, &v161);
                      sub_13EA060(v171.m128i_i32, &v166);
                      sub_13E8810(&v151, (unsigned int *)&v171);
                      if ( v171.m128i_i32[0] == 3 )
                      {
                        if ( v174 > 0x40 && v173 )
                          j_j___libc_free_0_0(v173);
                        if ( v172 > 0x40 && v171.m128i_i64[1] )
                          j_j___libc_free_0_0(v171.m128i_i64[1]);
                      }
                      if ( (unsigned int)v169 > 0x40 && v168 )
                        j_j___libc_free_0_0(v168);
                      if ( (unsigned int)v167 > 0x40 && v166 )
                        j_j___libc_free_0_0(v166);
                      if ( (unsigned int)v164 > 0x40 && v163 )
                        j_j___libc_free_0_0(v163);
                      if ( (unsigned int)v162 > 0x40 && v161 )
                        j_j___libc_free_0_0(v161);
                      if ( (unsigned int)v159 > 0x40 && v158 )
                        j_j___libc_free_0_0(v158);
                      if ( (unsigned int)v157 <= 0x40 )
                        goto LABEL_51;
                      v55 = v156;
                      if ( !v156 )
                        goto LABEL_51;
                      goto LABEL_222;
                    case 0x10:
                      v110 = *(_QWORD *)(v108 + 32);
                      v108 = *(_QWORD *)(v108 + 24);
                      v109 *= (_DWORD)v110;
                      continue;
                    default:
LABEL_492:
                      BUG();
                  }
                }
              }
              return;
            default:
              goto LABEL_189;
          }
          goto LABEL_28;
        }
LABEL_215:
        sub_13EA4E0(v171.m128i_i32, v7);
LABEL_216:
        sub_13E8810(&v151, (unsigned int *)&v171);
        if ( v171.m128i_i32[0] == 3 )
        {
          if ( v174 > 0x40 && v173 )
            j_j___libc_free_0_0(v173);
          if ( v172 > 0x40 )
          {
            v55 = v171.m128i_i64[1];
            if ( v171.m128i_i64[1] )
LABEL_222:
              j_j___libc_free_0_0(v55);
          }
        }
LABEL_51:
        sub_13EC960(a1, v7, v8, (unsigned int *)&v151);
        if ( v151 != 3 )
          goto LABEL_42;
        v120 = 1;
        goto LABEL_53;
      }
      v47 = *(_BYTE **)(v7 - 24);
      v48 = *(unsigned __int8 *)(*(_QWORD *)v47 + 8LL);
      if ( (unsigned __int8)v48 <= 0xFu && (v66 = 35454, _bittest64(&v66, v48)) )
      {
        v67 = v46 - 24;
        if ( v67 <= 0x26 )
          goto LABEL_283;
      }
      else
      {
        if ( (unsigned int)(v48 - 13) > 1 && (_DWORD)v48 != 16 || !(unsigned __int8)sub_16435F0(*(_QWORD *)v47, 0) )
        {
LABEL_189:
          v171.m128i_i32[0] = 4;
          goto LABEL_216;
        }
        v67 = *(unsigned __int8 *)(v7 + 16) - 24;
        if ( v67 <= 0x26 )
        {
          if ( v67 <= 0x23 )
            goto LABEL_189;
          goto LABEL_309;
        }
      }
      if ( v67 != 47 )
        goto LABEL_189;
LABEL_309:
      v47 = *(_BYTE **)(v7 - 24);
LABEL_283:
      if ( v47[16] <= 0x10u )
        goto LABEL_287;
      if ( (unsigned __int8)sub_13E8A40(a1, (__int64)v47, v8)
        || (v68 = *(_QWORD *)(v7 - 24),
            v171.m128i_i64[0] = v8,
            v171.m128i_i64[1] = v68,
            !(unsigned __int8)sub_13ED650(a1, &v171)) )
      {
        v47 = *(_BYTE **)(v7 - 24);
LABEL_287:
        v69 = *(_QWORD *)(a1 + 280);
        v70 = *(_QWORD *)v47;
        v71 = 1;
        while ( 2 )
        {
          switch ( *(_BYTE *)(v70 + 8) )
          {
            case 1:
              v72 = 16;
              goto LABEL_290;
            case 2:
              v72 = 32;
              goto LABEL_290;
            case 3:
            case 9:
              v72 = 64;
              goto LABEL_290;
            case 4:
              v72 = 80;
              goto LABEL_290;
            case 5:
            case 6:
              v72 = 128;
              goto LABEL_290;
            case 7:
              v72 = 8 * sub_15A9520(v69, 0);
              goto LABEL_290;
            case 0xB:
              v72 = *(_DWORD *)(v70 + 8) >> 8;
              goto LABEL_290;
            case 0xD:
              v72 = 8 * *(_QWORD *)sub_15A9930(v69, v70);
              goto LABEL_290;
            case 0xE:
              v117 = *(_QWORD *)(v70 + 32);
              v130 = *(_QWORD *)(a1 + 280);
              v123 = *(_QWORD *)(v70 + 24);
              v144 = (unsigned int)sub_15A9FE0(v69, v123);
              v72 = 8 * v117 * v144 * ((v144 + ((unsigned __int64)(sub_127FA20(v130, v123) + 7) >> 3) - 1) / v144);
              goto LABEL_290;
            case 0xF:
              v72 = 8 * sub_15A9520(v69, *(_DWORD *)(v70 + 8) >> 8);
LABEL_290:
              sub_15897D0(&v161, (unsigned int)(v71 * v72), 1);
              v73 = *(_QWORD *)(v7 - 24);
              if ( *(_BYTE *)(v73 + 16) <= 0x10u )
                goto LABEL_293;
              if ( !(unsigned __int8)sub_13E8A40(a1, v73, v8) )
                goto LABEL_294;
              v73 = *(_QWORD *)(v7 - 24);
LABEL_293:
              sub_13E9630(v171.m128i_i32, a1, v73, v8);
              sub_13EE9C0(a1, *(_QWORD *)(v7 - 24), v171.m128i_i32, v7);
              if ( v171.m128i_i32[0] != 3 )
                goto LABEL_294;
              if ( (unsigned int)v162 <= 0x40 && v172 <= 0x40 )
              {
                LODWORD(v162) = v172;
                v161 = v171.m128i_i64[1] & (0xFFFFFFFFFFFFFFFFLL >> -(char)v172);
              }
              else
              {
                sub_16A51C0(&v161, &v171.m128i_u64[1]);
              }
              if ( (unsigned int)v164 <= 0x40 && v174 <= 0x40 )
              {
                LODWORD(v164) = v174;
                v163 = v173 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v174);
                if ( v171.m128i_i32[0] != 3 )
                  goto LABEL_294;
              }
              else
              {
                sub_16A51C0(&v163, &v173);
                if ( v171.m128i_i32[0] != 3 )
                  goto LABEL_294;
                if ( v174 > 0x40 && v173 )
                  j_j___libc_free_0_0(v173);
              }
              if ( v172 > 0x40 && v171.m128i_i64[1] )
                j_j___libc_free_0_0(v171.m128i_i64[1]);
LABEL_294:
              sub_158DBC0(
                &v166,
                &v161,
                (unsigned int)*(unsigned __int8 *)(v7 + 16) - 24,
                *(_DWORD *)(*(_QWORD *)v7 + 8LL) >> 8);
              sub_13EA060(v171.m128i_i32, &v166);
              sub_13E8810(&v151, (unsigned int *)&v171);
              if ( v171.m128i_i32[0] == 3 )
              {
                if ( v174 > 0x40 && v173 )
                  j_j___libc_free_0_0(v173);
                if ( v172 > 0x40 && v171.m128i_i64[1] )
                  j_j___libc_free_0_0(v171.m128i_i64[1]);
              }
              if ( (unsigned int)v169 > 0x40 && v168 )
                j_j___libc_free_0_0(v168);
              if ( (unsigned int)v167 > 0x40 && v166 )
                j_j___libc_free_0_0(v166);
              if ( (unsigned int)v164 > 0x40 && v163 )
                j_j___libc_free_0_0(v163);
              if ( (unsigned int)v162 <= 0x40 )
                goto LABEL_51;
              v55 = v161;
              if ( !v161 )
                goto LABEL_51;
              goto LABEL_222;
            case 0x10:
              v116 = *(_QWORD *)(v70 + 32);
              v70 = *(_QWORD *)(v70 + 24);
              v71 *= (_DWORD)v116;
              continue;
            default:
              goto LABEL_492;
          }
        }
      }
      goto LABEL_28;
    }
    v38 = *(_QWORD *)(v7 - 48);
    if ( *(_BYTE *)(v38 + 16) > 0x10u )
    {
      if ( !(unsigned __int8)sub_13E8A40(a1, *(_QWORD *)(v7 - 48), v8) )
      {
        v65 = *(_QWORD *)(v7 - 48);
        v171.m128i_i64[0] = v8;
        v171.m128i_i64[1] = v65;
        if ( (unsigned __int8)sub_13ED650(a1, &v171) )
          goto LABEL_28;
        v171.m128i_i32[0] = 4;
        goto LABEL_216;
      }
      v38 = *(_QWORD *)(v7 - 48);
    }
    sub_13E9630((int *)&v156, a1, v38, v8);
    if ( (_DWORD)v156 == 4 )
    {
      v171.m128i_i32[0] = 4;
    }
    else
    {
      v39 = *(_QWORD *)(v7 - 24);
      if ( *(_BYTE *)(v39 + 16) <= 0x10u )
        goto LABEL_123;
      if ( (unsigned __int8)sub_13E8A40(a1, *(_QWORD *)(v7 - 24), v8) )
      {
        v39 = *(_QWORD *)(v7 - 24);
LABEL_123:
        sub_13E9630((int *)&v161, a1, v39, v8);
        if ( (_DWORD)v161 == 4 )
        {
          v171.m128i_i32[0] = 4;
        }
        else
        {
          if ( (_DWORD)v156 == 3
            && (_DWORD)v161 == 3
            && (v145 = 0,
                *(_QWORD *)v146 = 0,
                v63 = sub_14B2890(v7, &v145, v146, 0, 0),
                v146[5] = v64,
                *(_QWORD *)&v146[3] = v63,
                (unsigned int)(v63 - 7) > 1) )
          {
            v40 = *(_QWORD *)(v7 - 48);
            if ( (_DWORD)v63 && v40 == v145 && *(_QWORD *)(v7 - 24) == *(_QWORD *)v146 )
            {
              if ( (_DWORD)v63 == 3 )
              {
                sub_158F0F0(&v147, &v157, &v162);
              }
              else if ( (unsigned int)v63 > 3 )
              {
                sub_158F360(&v147, &v157, &v162);
              }
              else if ( (_DWORD)v63 == 1 )
              {
                sub_158F5D0(&v147, &v157, &v162);
              }
              else
              {
                sub_158F840(&v147, &v157, &v162);
              }
              LODWORD(v167) = v148;
              if ( v148 > 0x40 )
                sub_16A4FD0(&v166, &v147);
              else
                v166 = v147;
              LODWORD(v169) = v150;
              if ( v150 > 0x40 )
                sub_16A4FD0(&v168, &v149);
              else
                v168 = v149;
              sub_13EA060(v171.m128i_i32, &v166);
              sub_13E8810(&v151, (unsigned int *)&v171);
              sub_13EA000((__int64)&v171);
              sub_135E100(&v168);
              sub_135E100(&v166);
              sub_135E100(&v149);
              sub_135E100(&v147);
              goto LABEL_137;
            }
          }
          else
          {
            v40 = *(_QWORD *)(v7 - 48);
          }
          v139 = *(_QWORD *)(v7 - 72);
          sub_13EE900(v171.m128i_i32, v40, v139, 1);
          sub_13EA210((int *)&v166, (__int64)&v156, (__int64)&v171);
          sub_13E8810((int *)&v156, (unsigned int *)&v166);
          if ( (_DWORD)v166 == 3 )
          {
            if ( v170 > 0x40 && v169 )
              j_j___libc_free_0_0(v169);
            if ( (unsigned int)v168 > 0x40 && v167 )
              j_j___libc_free_0_0(v167);
          }
          if ( v171.m128i_i32[0] == 3 )
          {
            if ( v174 > 0x40 && v173 )
              j_j___libc_free_0_0(v173);
            if ( v172 > 0x40 && v171.m128i_i64[1] )
              j_j___libc_free_0_0(v171.m128i_i64[1]);
          }
          sub_13EE900(v171.m128i_i32, *(_QWORD *)(v7 - 24), v139, 0);
          sub_13EA210((int *)&v166, (__int64)&v161, (__int64)&v171);
          sub_13E8810((int *)&v161, (unsigned int *)&v166);
          if ( (_DWORD)v166 == 3 )
          {
            if ( v170 > 0x40 && v169 )
              j_j___libc_free_0_0(v169);
            if ( (unsigned int)v168 > 0x40 && v167 )
              j_j___libc_free_0_0(v167);
          }
          if ( v171.m128i_i32[0] == 3 )
          {
            if ( v174 > 0x40 && v173 )
              j_j___libc_free_0_0(v173);
            if ( v172 > 0x40 && v171.m128i_i64[1] )
              j_j___libc_free_0_0(v171.m128i_i64[1]);
          }
          if ( *(_BYTE *)(v139 + 16) == 75 )
          {
            v75 = *(_QWORD *)(v139 - 24);
            if ( *(_BYTE *)(v75 + 16) == 13 )
            {
              v76 = *(_QWORD *)(v139 - 48);
              v77 = *(unsigned __int16 *)(v139 + 18);
              BYTE1(v77) &= ~0x80u;
              if ( v77 == 32 )
              {
                v87 = *(_QWORD *)(v7 - 24);
                v88 = *(_BYTE *)(v87 + 16);
                if ( v88 == 35 )
                {
                  if ( v76 != *(_QWORD *)(v87 - 48) )
                    goto LABEL_132;
                  v90 = *(_QWORD *)(v87 - 24);
                  if ( *(_BYTE *)(v90 + 16) != 13 )
                    goto LABEL_132;
                }
                else
                {
                  if ( v88 != 5 )
                    goto LABEL_132;
                  if ( *(_WORD *)(v87 + 18) != 11 )
                    goto LABEL_132;
                  v89 = *(_DWORD *)(v87 + 20) & 0xFFFFFFF;
                  if ( v76 != *(_QWORD *)(v87 - 24 * v89) )
                    goto LABEL_132;
                  v90 = *(_QWORD *)(v87 + 24 * (1 - v89));
                  if ( *(_BYTE *)(v90 + 16) != 13 )
                    goto LABEL_132;
                }
                v91 = v90 + 24;
                v171.m128i_i32[2] = *(_DWORD *)(v75 + 32);
                if ( v171.m128i_i32[2] > 0x40u )
                {
                  v119 = v91;
                  v131 = v75;
                  sub_16A4FD0(&v171, v75 + 24);
                  v91 = v119;
                  v75 = v131;
                }
                else
                {
                  v171.m128i_i64[0] = *(_QWORD *)(v75 + 24);
                }
                v127 = (_QWORD *)v75;
                sub_16A7200(&v171, v91);
                v92 = v171.m128i_i32[2];
                v171.m128i_i32[2] = 0;
                LODWORD(v167) = v92;
                v166 = v171.m128i_i64[0];
                v93 = sub_15A1070(*v127, &v166);
                v94 = v93;
                if ( (unsigned int)v167 > 0x40 && v166 )
                {
                  v142 = v93;
                  j_j___libc_free_0_0(v166);
                  v94 = v142;
                }
                if ( v171.m128i_i32[2] > 0x40u && v171.m128i_i64[0] )
                  j_j___libc_free_0_0(v171.m128i_i64[0]);
                v171.m128i_i32[0] = 0;
                if ( *(_BYTE *)(v94 + 16) != 9 )
                  sub_13EA940(v171.m128i_i32, v94);
                sub_13EA210((int *)&v166, (__int64)&v161, (__int64)&v171);
                v86 = &v161;
LABEL_348:
                sub_13E8810((int *)v86, (unsigned int *)&v166);
                sub_13EA000((__int64)&v166);
                sub_13EA000((__int64)&v171);
              }
              else if ( v77 == 33 )
              {
                v78 = *(_QWORD *)(v7 - 48);
                v79 = *(_BYTE *)(v78 + 16);
                if ( v79 == 35 )
                {
                  if ( v76 == *(_QWORD *)(v78 - 48) )
                  {
                    v81 = *(_QWORD *)(v78 - 24);
                    if ( *(_BYTE *)(v81 + 16) == 13 )
                      goto LABEL_337;
                  }
                }
                else if ( v79 == 5 && *(_WORD *)(v78 + 18) == 11 )
                {
                  v80 = *(_DWORD *)(v78 + 20) & 0xFFFFFFF;
                  if ( v76 == *(_QWORD *)(v78 - 24 * v80) )
                  {
                    v81 = *(_QWORD *)(v78 + 24 * (1 - v80));
                    if ( *(_BYTE *)(v81 + 16) == 13 )
                    {
LABEL_337:
                      v82 = v81 + 24;
                      v171.m128i_i32[2] = *(_DWORD *)(v75 + 32);
                      if ( v171.m128i_i32[2] > 0x40u )
                      {
                        v118 = v82;
                        v128 = v75;
                        sub_16A4FD0(&v171, v75 + 24);
                        v82 = v118;
                        v75 = v128;
                      }
                      else
                      {
                        v171.m128i_i64[0] = *(_QWORD *)(v75 + 24);
                      }
                      v126 = (_QWORD *)v75;
                      sub_16A7200(&v171, v82);
                      v83 = v171.m128i_i32[2];
                      v171.m128i_i32[2] = 0;
                      LODWORD(v167) = v83;
                      v166 = v171.m128i_i64[0];
                      v84 = sub_15A1070(*v126, &v166);
                      v85 = v84;
                      if ( (unsigned int)v167 > 0x40 && v166 )
                      {
                        v141 = v84;
                        j_j___libc_free_0_0(v166);
                        v85 = v141;
                      }
                      if ( v171.m128i_i32[2] > 0x40u && v171.m128i_i64[0] )
                        j_j___libc_free_0_0(v171.m128i_i64[0]);
                      v171.m128i_i32[0] = 0;
                      if ( *(_BYTE *)(v85 + 16) != 9 )
                        sub_13EA940(v171.m128i_i32, v85);
                      sub_13EA210((int *)&v166, (__int64)&v156, (__int64)&v171);
                      v86 = &v156;
                      goto LABEL_348;
                    }
                  }
                }
              }
            }
          }
LABEL_132:
          v171.m128i_i32[0] = 0;
          if ( (_DWORD)v156 )
          {
            if ( (_DWORD)v156 == 4 )
              v171.m128i_i32[0] = 4;
            else
              sub_13E8810(v171.m128i_i32, (unsigned int *)&v156);
          }
          sub_13EACF0(v171.m128i_i32, (unsigned int *)&v161);
        }
        sub_13E8810(&v151, (unsigned int *)&v171);
        if ( v171.m128i_i32[0] == 3 )
        {
          if ( v174 > 0x40 && v173 )
            j_j___libc_free_0_0(v173);
          if ( v172 > 0x40 && v171.m128i_i64[1] )
            j_j___libc_free_0_0(v171.m128i_i64[1]);
        }
LABEL_137:
        if ( (_DWORD)v161 != 3 )
          goto LABEL_138;
        if ( v165 > 0x40 && v164 )
          j_j___libc_free_0_0(v164);
        if ( (unsigned int)v163 <= 0x40 )
          goto LABEL_138;
        v61 = v162;
        if ( !v162 )
          goto LABEL_138;
        goto LABEL_237;
      }
      v59 = *(_QWORD *)(v7 - 24);
      v171.m128i_i64[0] = v8;
      v171.m128i_i64[1] = v59;
      v60 = sub_13ED650(a1, &v171);
      v15 = 0;
      if ( v60 )
        goto LABEL_139;
      v171.m128i_i32[0] = 4;
    }
    sub_13E8810(&v151, (unsigned int *)&v171);
    if ( v171.m128i_i32[0] != 3 )
      goto LABEL_138;
    if ( v174 > 0x40 && v173 )
      j_j___libc_free_0_0(v173);
    if ( v172 <= 0x40 )
      goto LABEL_138;
    v61 = v171.m128i_i64[1];
    if ( !v171.m128i_i64[1] )
      goto LABEL_138;
LABEL_237:
    j_j___libc_free_0_0(v61);
LABEL_138:
    v15 = 1;
LABEL_139:
    if ( (_DWORD)v156 == 3 )
    {
      if ( v160 > 0x40 && v159 )
      {
        v140 = v15;
        j_j___libc_free_0_0(v159);
        v15 = v140;
      }
      if ( (unsigned int)v158 > 0x40 )
      {
        v34 = v157;
        if ( v157 )
        {
LABEL_103:
          v135 = v15;
          j_j___libc_free_0_0(v34);
          v15 = v135;
        }
      }
    }
LABEL_27:
    if ( v15 )
      goto LABEL_51;
LABEL_28:
    if ( v151 != 3 )
    {
      v5 = *(unsigned int *)(a1 + 104);
      goto LABEL_30;
    }
LABEL_53:
    if ( v155 > 0x40 && v154 )
      j_j___libc_free_0_0(v154);
    if ( v153 > 0x40 && v152 )
      j_j___libc_free_0_0(v152);
    v18 = *(_DWORD *)(a1 + 104);
    v5 = v18;
    if ( v120 )
      goto LABEL_43;
    if ( !v18 )
      goto LABEL_61;
LABEL_31:
    --v121;
  }
  while ( v121 );
  v16 = (unsigned int)v176;
  if ( (_DWORD)v176 )
  {
    do
    {
      v171.m128i_i32[0] = 4;
      sub_13EC960(a1, *(_QWORD *)&v175[16 * v16 - 8], *(_QWORD *)&v175[16 * v16 - 16], (unsigned int *)&v171);
      if ( v171.m128i_i32[0] == 3 )
      {
        if ( v174 > 0x40 && v173 )
          j_j___libc_free_0_0(v173);
        if ( v172 > 0x40 && v171.m128i_i64[1] )
          j_j___libc_free_0_0(v171.m128i_i64[1]);
      }
      v17 = (_DWORD)v176 == 1;
      v16 = (unsigned int)(v176 - 1);
      LODWORD(v176) = v176 - 1;
    }
    while ( !v17 );
  }
  v53 = *(_DWORD *)(a1 + 256);
  ++*(_QWORD *)(a1 + 240);
  if ( v53 )
  {
    v56 = 4 * v53;
    v54 = *(unsigned int *)(a1 + 264);
    if ( (unsigned int)(4 * v53) < 0x40 )
      v56 = 64;
    if ( v56 >= (unsigned int)v54 )
      goto LABEL_226;
    v95 = *(_QWORD **)(a1 + 248);
    v96 = v53 - 1;
    if ( v96 )
    {
      _BitScanReverse(&v96, v96);
      v97 = (unsigned int)(1 << (33 - (v96 ^ 0x1F)));
      if ( (int)v97 < 64 )
        v97 = 64;
      if ( (_DWORD)v97 == (_DWORD)v54 )
      {
        *(_QWORD *)(a1 + 256) = 0;
        v115 = &v95[2 * v97];
        do
        {
          if ( v95 )
          {
            *v95 = -8;
            v95[1] = -8;
          }
          v95 += 2;
        }
        while ( v115 != v95 );
        goto LABEL_213;
      }
      v98 = (4 * (int)v97 / 3u + 1) | ((unsigned __int64)(4 * (int)v97 / 3u + 1) >> 1);
      v99 = ((v98 | (v98 >> 2)) >> 4) | v98 | (v98 >> 2) | ((((v98 | (v98 >> 2)) >> 4) | v98 | (v98 >> 2)) >> 8);
      v100 = (v99 | (v99 >> 16)) + 1;
      v101 = 16 * ((v99 | (v99 >> 16)) + 1);
    }
    else
    {
      v101 = 2048;
      v100 = 128;
    }
    j___libc_free_0(v95);
    *(_DWORD *)(a1 + 264) = v100;
    v102 = (_QWORD *)sub_22077B0(v101);
    v103 = *(unsigned int *)(a1 + 264);
    *(_QWORD *)(a1 + 256) = 0;
    *(_QWORD *)(a1 + 248) = v102;
    for ( j = &v102[2 * v103]; j != v102; v102 += 2 )
    {
      if ( v102 )
      {
        *v102 = -8;
        v102[1] = -8;
      }
    }
LABEL_213:
    *(_DWORD *)(a1 + 104) = 0;
    goto LABEL_61;
  }
  if ( !*(_DWORD *)(a1 + 260) )
    goto LABEL_213;
  v54 = *(unsigned int *)(a1 + 264);
  if ( (unsigned int)v54 > 0x40 )
  {
    j___libc_free_0(*(_QWORD *)(a1 + 248));
    *(_QWORD *)(a1 + 248) = 0;
    *(_QWORD *)(a1 + 256) = 0;
    *(_DWORD *)(a1 + 264) = 0;
    goto LABEL_213;
  }
LABEL_226:
  v57 = *(_QWORD **)(a1 + 248);
  for ( k = &v57[2 * v54]; k != v57; *(v57 - 1) = -8 )
  {
    *v57 = -8;
    v57 += 2;
  }
  *(_QWORD *)(a1 + 256) = 0;
  *(_DWORD *)(a1 + 104) = 0;
LABEL_61:
  if ( v175 != v177 )
    _libc_free((unsigned __int64)v175);
}
