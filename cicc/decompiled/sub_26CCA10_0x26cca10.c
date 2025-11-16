// Function: sub_26CCA10
// Address: 0x26cca10
//
__int64 __fastcall sub_26CCA10(__int64 a1, __int64 a2)
{
  __int64 v2; // r15
  __int64 result; // rax
  __m128i *v4; // rax
  bool v5; // zf
  __int64 v6; // rax
  unsigned __int64 v7; // r12
  int v8; // eax
  __int64 v9; // r12
  __int64 v10; // rsi
  unsigned int v11; // eax
  unsigned int v12; // ebx
  __int64 v13; // r13
  __m128i *v14; // rax
  __m128i *v15; // rcx
  unsigned int v16; // esi
  __m128i *v17; // rdx
  __int64 v18; // r10
  int v19; // r12d
  _QWORD *v20; // rdi
  unsigned int i; // eax
  __m128i **v22; // r8
  __m128i *v23; // r9
  unsigned int v24; // eax
  unsigned __int64 v25; // r8
  unsigned __int64 v26; // r9
  int v27; // r12d
  __int64 v28; // rax
  unsigned __int64 v29; // rdx
  int v30; // edx
  unsigned __int64 v31; // rax
  unsigned __int64 v32; // r12
  __int64 *v33; // rax
  __int64 v34; // rcx
  __int64 v35; // rax
  __int64 v36; // r12
  char v37; // al
  __m128i *v38; // r13
  __int8 v39; // al
  __int64 v40; // r14
  __int64 v41; // rdx
  __int64 v42; // r12
  __int64 v43; // rdx
  __int64 v44; // rbx
  _QWORD *v45; // rdi
  __int64 v46; // rax
  unsigned int *v47; // rsi
  const __m128i *v48; // rax
  __int64 v49; // rdx
  __m128i v50; // xmm2
  __int64 v51; // rdx
  _QWORD *v52; // rbx
  unsigned __int64 v53; // rdi
  unsigned __int64 v54; // r15
  char v55; // di
  __int64 v56; // r9
  __int64 v57; // rax
  unsigned __int64 v58; // r14
  __m128i v59; // xmm1
  __int64 v60; // rbx
  unsigned __int64 v61; // r13
  _QWORD *v62; // rax
  __int64 *v63; // rdx
  __int64 v64; // r8
  __int64 *v65; // r13
  int *v66; // rbx
  unsigned int *v67; // r14
  __int64 v68; // rax
  unsigned int *v69; // r10
  int v70; // edx
  unsigned int **v71; // rax
  _QWORD *v72; // rax
  __int64 v73; // r12
  unsigned int **v74; // rax
  __int64 *v75; // r13
  __int64 v76; // r12
  __int64 v77; // rax
  __int64 v78; // rdx
  __int64 v79; // r9
  __m128i v80; // xmm7
  __m128i v81; // xmm6
  unsigned int v82; // r8d
  __m128i v83; // xmm7
  __m128i *v84; // rax
  int v85; // eax
  _QWORD *v86; // rdi
  __int64 v87; // rax
  __int64 *v88; // rdx
  __int64 v89; // rax
  __int64 v90; // rsi
  unsigned int v91; // ecx
  __int64 v92; // rbx
  __int64 v93; // r15
  __int64 j; // rbx
  _QWORD *v95; // r15
  unsigned __int64 v96; // rbx
  __int64 v97; // rax
  float v98; // xmm0_4
  float v99; // xmm0_4
  int v100; // eax
  unsigned int v101; // eax
  __int64 v102; // rsi
  __int64 v103; // r14
  __m128i *v104; // rbx
  __int64 v105; // r12
  __int64 v106; // rdx
  __int64 v107; // rax
  __int64 v108; // rax
  __int64 v109; // rdx
  __int64 *v110; // rax
  __m128i v111; // xmm5
  __m128i v112; // xmm6
  unsigned __int64 v113; // rbx
  void *v114; // rax
  size_t v115; // rdx
  unsigned int v116; // ebx
  __int64 v117; // rdx
  __int64 v118; // rax
  __int64 *v119; // rax
  __int64 *v120; // [rsp+8h] [rbp-4C8h]
  __int64 v121; // [rsp+10h] [rbp-4C0h]
  __int64 v122; // [rsp+18h] [rbp-4B8h]
  __m128i *v123; // [rsp+18h] [rbp-4B8h]
  __int64 v124; // [rsp+18h] [rbp-4B8h]
  unsigned __int64 v125; // [rsp+20h] [rbp-4B0h]
  __int64 v126; // [rsp+20h] [rbp-4B0h]
  __m128i *v127; // [rsp+20h] [rbp-4B0h]
  __int64 v128; // [rsp+20h] [rbp-4B0h]
  __int64 *v129; // [rsp+28h] [rbp-4A8h]
  __int64 v130; // [rsp+28h] [rbp-4A8h]
  float v131; // [rsp+30h] [rbp-4A0h]
  unsigned __int64 v132; // [rsp+38h] [rbp-498h]
  unsigned __int64 v133; // [rsp+38h] [rbp-498h]
  unsigned int v134; // [rsp+38h] [rbp-498h]
  unsigned int v135; // [rsp+38h] [rbp-498h]
  unsigned __int64 v136; // [rsp+40h] [rbp-490h]
  unsigned __int64 v137; // [rsp+40h] [rbp-490h]
  __int64 v138; // [rsp+48h] [rbp-488h]
  __int64 v139; // [rsp+58h] [rbp-478h]
  __int64 v140; // [rsp+60h] [rbp-470h]
  unsigned __int64 v141; // [rsp+68h] [rbp-468h]
  __int64 v142; // [rsp+68h] [rbp-468h]
  __int64 v143; // [rsp+70h] [rbp-460h]
  __m128i *v144; // [rsp+78h] [rbp-458h]
  unsigned int v145; // [rsp+78h] [rbp-458h]
  __m128i *v146; // [rsp+80h] [rbp-450h] BYREF
  __int64 v147; // [rsp+88h] [rbp-448h] BYREF
  char v148[8]; // [rsp+90h] [rbp-440h] BYREF
  __int64 v149; // [rsp+98h] [rbp-438h] BYREF
  unsigned int *v150; // [rsp+A0h] [rbp-430h] BYREF
  __int64 v151; // [rsp+A8h] [rbp-428h]
  _BYTE v152[16]; // [rsp+B0h] [rbp-420h] BYREF
  __int64 *v153; // [rsp+C0h] [rbp-410h] BYREF
  __int64 v154; // [rsp+C8h] [rbp-408h]
  __int64 v155; // [rsp+D0h] [rbp-400h] BYREF
  unsigned int v156; // [rsp+D8h] [rbp-3F8h]
  __m128i *v157; // [rsp+F0h] [rbp-3E0h] BYREF
  __int64 v158; // [rsp+F8h] [rbp-3D8h] BYREF
  unsigned __int64 v159; // [rsp+100h] [rbp-3D0h]
  __int64 *v160; // [rsp+108h] [rbp-3C8h]
  __m128i v161; // [rsp+110h] [rbp-3C0h] BYREF
  _QWORD v162[2]; // [rsp+120h] [rbp-3B0h] BYREF
  __m128i v163; // [rsp+130h] [rbp-3A0h] BYREF
  void *s; // [rsp+140h] [rbp-390h] BYREF
  unsigned __int64 v165; // [rsp+148h] [rbp-388h]
  _QWORD *v166; // [rsp+150h] [rbp-380h] BYREF
  _BYTE v167[24]; // [rsp+158h] [rbp-378h] BYREF
  __m128i v168; // [rsp+170h] [rbp-360h] BYREF
  __m128i v169; // [rsp+180h] [rbp-350h]
  __m128i *v170; // [rsp+190h] [rbp-340h] BYREF
  __int64 v171; // [rsp+198h] [rbp-338h]
  _BYTE v172[324]; // [rsp+1A0h] [rbp-330h] BYREF
  int v173; // [rsp+2E4h] [rbp-1ECh]
  __int64 v174; // [rsp+2E8h] [rbp-1E8h]
  __m128i *v175; // [rsp+2F0h] [rbp-1E0h] BYREF
  unsigned __int64 v176; // [rsp+2F8h] [rbp-1D8h]
  _QWORD *v177; // [rsp+300h] [rbp-1D0h]
  __int64 *v178; // [rsp+308h] [rbp-1C8h] BYREF
  __m128i v179; // [rsp+310h] [rbp-1C0h] BYREF
  __m128i v180; // [rsp+320h] [rbp-1B0h] BYREF
  __m128i v181; // [rsp+330h] [rbp-1A0h] BYREF
  __int64 v182; // [rsp+340h] [rbp-190h] BYREF
  unsigned int v183; // [rsp+348h] [rbp-188h]
  char v184; // [rsp+490h] [rbp-40h]
  int v185; // [rsp+494h] [rbp-3Ch]
  __int64 v186; // [rsp+498h] [rbp-38h]

  v2 = a1;
  sub_B2BE50(a2);
  result = *(_QWORD *)(a2 + 80);
  v138 = a2 + 72;
  v139 = a1 + 40;
  v143 = result;
  if ( result != a2 + 72 )
  {
    while ( 1 )
    {
      v4 = (__m128i *)(v143 - 24);
      if ( !v143 )
        v4 = 0;
      v146 = v4;
      v5 = *sub_26CC460(v139, (__int64 *)&v146) == 0;
      v6 = (__int64)v146;
      if ( !v5 )
        break;
      v144 = v146 + 3;
      if ( byte_4FF61E8 || (_BYTE)qword_4FF7E28 )
      {
        v36 = v146[3].m128i_i64[1];
        if ( (__m128i *)v36 != v144 )
        {
          while ( 1 )
          {
            if ( !v36 )
              BUG();
            v37 = *(_BYTE *)(v36 - 24);
            if ( v37 != 85 && v37 != 34 )
              goto LABEL_64;
            if ( sub_B491E0(v36 - 24) )
            {
              sub_B99FD0(v36 - 24, 2u, 0);
              v36 = *(_QWORD *)(v36 + 8);
              if ( (__m128i *)v36 == v144 )
              {
LABEL_69:
                v6 = (__int64)v146;
                v144 = v146 + 3;
                break;
              }
            }
            else
            {
              LODWORD(v175) = 0;
              sub_BC8EC0(v36 - 24, (unsigned int *)&v175, 1, 0);
LABEL_64:
              v36 = *(_QWORD *)(v36 + 8);
              if ( (__m128i *)v36 == v144 )
                goto LABEL_69;
            }
          }
        }
      }
LABEL_7:
      v7 = *(_QWORD *)(v6 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( (__m128i *)v7 == v144 )
      {
        v9 = 0;
      }
      else
      {
        if ( !v7 )
          BUG();
        v8 = *(unsigned __int8 *)(v7 - 24);
        v9 = v7 - 24;
        if ( (unsigned int)(v8 - 30) >= 0xB )
          v9 = 0;
      }
      if ( (unsigned int)sub_B46E30(v9) != 1 && (unsigned __int8)(*(_BYTE *)v9 - 31) <= 2u )
      {
        v10 = *(_QWORD *)(v9 + 48);
        v147 = v10;
        if ( v10 )
          sub_B96E90((__int64)&v147, v10, 1);
        v153 = 0;
        v150 = (unsigned int *)v152;
        v151 = 0x400000000LL;
        v154 = 0;
        v155 = 0;
        v156 = 0;
        if ( LOBYTE(qword_500BA28[8]) )
        {
          v101 = sub_B46E30(v9);
          if ( v101 )
          {
            v113 = 8LL * v101;
            v114 = (void *)sub_22077B0(v113);
            v115 = v113;
            v116 = 0;
            v141 = (unsigned __int64)v114;
            memset(v114, 0, v115);
            while ( 1 )
            {
              v11 = sub_B46E30(v9);
              if ( v11 <= v116 )
                break;
              v175 = (__m128i *)sub_B46EC0(v9, v116);
              v117 = *sub_26CC460((__int64)&v153, (__int64 *)&v175);
              v118 = v116++;
              *(_QWORD *)(v141 + 8 * v118) = v117;
              v119 = sub_26CC460((__int64)&v153, (__int64 *)&v175);
              ++*v119;
            }
          }
          else
          {
            v11 = 0;
            v141 = 0;
          }
        }
        else
        {
          v11 = sub_B46E30(v9);
          v141 = 0;
        }
        v12 = 0;
        v13 = v9;
        v145 = 0;
        v122 = v2 + 72;
        while ( 1 )
        {
          if ( v12 >= v11 )
          {
            sub_2A3E730(v13, v150, (unsigned int)v151, 0);
            if ( v145 )
            {
              if ( !(unsigned __int8)sub_B92100(v13, v148) || byte_4FF61E8 )
              {
                sub_BC8EC0(v13, v150, (unsigned int)v151, 0);
                v75 = *(__int64 **)(v2 + 1288);
                v76 = *v75;
                v77 = sub_B2BE50(*v75);
                if ( sub_B6EA50(v77)
                  || (v107 = sub_B2BE50(v76),
                      v108 = sub_B6F970(v107),
                      (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v108 + 48LL))(v108)) )
                {
                  sub_B174A0((__int64)&v175, (__int64)"sample-profile", (__int64)"PopularDest", 11, v121);
                  sub_B18290((__int64)&v175, "most popular destination for conditional branches at ", 0x35u);
                  v149 = v147;
                  if ( v147 )
                    sub_B96E90((__int64)&v149, v147, 1);
                  sub_B16E20((__int64)&v157, "CondBranchesLoc", 15, &v149);
                  s = &v166;
                  sub_26BA410((__int64 *)&s, v157, (__int64)v157->m128i_i64 + v158);
                  *(_QWORD *)&v167[8] = &v168;
                  sub_26BA410((__int64 *)&v167[8], v161.m128i_i64[0], v161.m128i_i64[0] + v161.m128i_i64[1]);
                  v169 = _mm_load_si128(&v163);
                  sub_B180C0((__int64)&v175, (unsigned __int64)&s);
                  sub_2240A30((unsigned __int64 *)&v167[8]);
                  sub_2240A30((unsigned __int64 *)&s);
                  v80 = _mm_loadu_si128((const __m128i *)&v178);
                  v81 = _mm_load_si128(&v180);
                  v82 = v183;
                  v171 = 0x400000000LL;
                  LODWORD(v165) = v176;
                  *(__m128i *)v167 = v80;
                  v83 = _mm_load_si128(&v181);
                  BYTE4(v165) = BYTE4(v176);
                  v168 = v81;
                  v166 = v177;
                  v169 = v83;
                  s = &unk_49D9D40;
                  *(_QWORD *)&v167[16] = v179.m128i_i64[1];
                  v84 = (__m128i *)v172;
                  v170 = (__m128i *)v172;
                  if ( v183 )
                  {
                    v102 = v183;
                    if ( v183 > 4 )
                    {
                      v134 = v183;
                      sub_11F02D0((__int64)&v170, v183, v78, 0x400000000LL, v183, v79);
                      v84 = v170;
                      v102 = v183;
                      v82 = v134;
                    }
                    v103 = v182 + 80 * v102;
                    if ( v182 != v103 )
                    {
                      v135 = v82;
                      v104 = v84;
                      v105 = v182;
                      do
                      {
                        if ( v104 )
                        {
                          v104->m128i_i64[0] = (__int64)v104[1].m128i_i64;
                          sub_26BA410(v104->m128i_i64, *(_BYTE **)v105, *(_QWORD *)v105 + *(_QWORD *)(v105 + 8));
                          v104[2].m128i_i64[0] = (__int64)v104[3].m128i_i64;
                          sub_26BA410(
                            v104[2].m128i_i64,
                            *(_BYTE **)(v105 + 32),
                            *(_QWORD *)(v105 + 32) + *(_QWORD *)(v105 + 40));
                          v104[4] = _mm_loadu_si128((const __m128i *)(v105 + 64));
                        }
                        v105 += 80;
                        v104 += 5;
                      }
                      while ( v103 != v105 );
                      v82 = v135;
                    }
                    LODWORD(v171) = v82;
                  }
                  v172[320] = v184;
                  v173 = v185;
                  v174 = v186;
                  s = &unk_49D9D78;
                  sub_2240A30((unsigned __int64 *)&v161);
                  sub_2240A30((unsigned __int64 *)&v157);
                  if ( v149 )
                    sub_B91220((__int64)&v149, v149);
                  v175 = (__m128i *)&unk_49D9D40;
                  sub_23FD590((__int64)&v182);
                  sub_1049740(v75, (__int64)&s);
                  s = &unk_49D9D40;
                  sub_23FD590((__int64)&v170);
                }
              }
            }
            else if ( byte_4FF61E8 )
            {
              sub_B99FD0(v13, 2u, 0);
            }
            if ( v141 )
              j_j___libc_free_0(v141);
            sub_C7D6A0(v154, 16LL * v156, 8);
            if ( v150 != (unsigned int *)v152 )
              _libc_free((unsigned __int64)v150);
            if ( v147 )
              sub_B91220((__int64)&v147, v147);
            break;
          }
          v14 = (__m128i *)sub_B46EC0(v13, v12);
          v15 = v146;
          v16 = *(_DWORD *)(v2 + 96);
          v157 = v14;
          v17 = v14;
          v175 = v146;
          v176 = (unsigned __int64)v14;
          if ( !v16 )
          {
            ++*(_QWORD *)(v2 + 72);
            s = 0;
LABEL_50:
            v16 *= 2;
            goto LABEL_51;
          }
          v18 = *(_QWORD *)(v2 + 80);
          v19 = 1;
          v20 = 0;
          for ( i = (v16 - 1)
                  & (((0xBF58476D1CE4E5B9LL
                     * (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4)
                      | ((unsigned __int64)(((unsigned int)v146 >> 9) ^ ((unsigned int)v146 >> 4)) << 32))) >> 31)
                   ^ (484763065 * (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4)))); ; i = (v16 - 1) & v24 )
          {
            v22 = (__m128i **)(v18 + 24LL * i);
            v23 = *v22;
            if ( v146 == *v22 && v17 == v22[1] )
            {
              v25 = (unsigned __int64)v22[2];
              v26 = 0xFFFFFFFFLL;
              if ( v25 <= 0xFFFFFFFF )
                v26 = v25;
              if ( LOBYTE(qword_500BA28[8]) )
              {
                v125 = v25;
                v132 = v26;
                v32 = v26 / *sub_26CC460((__int64)&v153, (__int64 *)&v157);
                v136 = *(_QWORD *)(v141 + 8LL * v12);
                v33 = sub_26CC460((__int64)&v153, (__int64 *)&v157);
                v26 = v132;
                v25 = v125;
                v27 = (v136 < v132 % *v33) + (_DWORD)v32;
              }
              else
              {
                v27 = -1;
                if ( v25 <= 0xFFFFFFFE )
                  goto LABEL_56;
              }
              goto LABEL_44;
            }
            if ( v23 == (__m128i *)-4096LL )
              break;
            if ( v23 == (__m128i *)-8192LL && v22[1] == (__m128i *)-8192LL && !v20 )
              v20 = (_QWORD *)(v18 + 24LL * i);
LABEL_27:
            v24 = v19 + i;
            ++v19;
          }
          if ( v22[1] != (__m128i *)-4096LL )
            goto LABEL_27;
          v100 = *(_DWORD *)(v2 + 88);
          if ( !v20 )
            v20 = v22;
          ++*(_QWORD *)(v2 + 72);
          v30 = v100 + 1;
          s = v20;
          if ( 4 * (v100 + 1) >= 3 * v16 )
            goto LABEL_50;
          if ( v16 - *(_DWORD *)(v2 + 92) - v30 > v16 >> 3 )
            goto LABEL_52;
LABEL_51:
          sub_26CC5A0(v122, v16);
          sub_26C3690(v122, (__int64 *)&v175, (__int64 **)&s);
          v15 = v175;
          v20 = s;
          v30 = *(_DWORD *)(v2 + 88) + 1;
LABEL_52:
          *(_DWORD *)(v2 + 88) = v30;
          if ( *v20 != -4096 || v20[1] != -4096 )
            --*(_DWORD *)(v2 + 92);
          *v20 = v15;
          v31 = v176;
          v20[2] = 0;
          v20[1] = v31;
          if ( LOBYTE(qword_500BA28[8]) )
          {
            sub_26CC460((__int64)&v153, (__int64 *)&v157);
            v27 = 0;
            sub_26CC460((__int64)&v153, (__int64 *)&v157);
            v26 = 0;
            v25 = 0;
          }
          else
          {
            v26 = 0;
            v25 = 0;
LABEL_56:
            v27 = v26 + 1;
          }
LABEL_44:
          v28 = (unsigned int)v151;
          v29 = (unsigned int)v151 + 1LL;
          if ( v29 > HIDWORD(v151) )
          {
            v133 = v25;
            v137 = v26;
            sub_C8D5F0((__int64)&v150, v152, v29, 4u, v25, v26);
            v28 = (unsigned int)v151;
            v25 = v133;
            v26 = v137;
          }
          v150[v28] = v27;
          LODWORD(v151) = v151 + 1;
          if ( v25 && v145 < v26 )
          {
            v145 = v26;
            v34 = sub_AA50C0((__int64)v157, 1);
            v35 = v34 - 24;
            if ( !v34 )
              v35 = 0;
            v121 = v35;
          }
          ++v12;
          v11 = sub_B46E30(v13);
        }
      }
      result = *(_QWORD *)(v143 + 8);
      v143 = result;
      if ( v138 == result )
        return result;
    }
    v38 = (__m128i *)v146[3].m128i_i64[1];
    v144 = v146 + 3;
    if ( v38 == &v146[3] )
      goto LABEL_7;
    v140 = v2;
LABEL_73:
    if ( !v38 )
      BUG();
    v39 = v38[-2].m128i_i8[8];
    v40 = (__int64)&v38[-2].m128i_i64[1];
    if ( v39 != 85 && v39 != 34 )
      goto LABEL_92;
    v41 = v38[-4].m128i_i64[1];
    if ( v41 && !*(_BYTE *)v41 && v38[3].m128i_i64[1] == *(_QWORD *)(v41 + 24) )
    {
      if ( v39 != 85 || (*(_BYTE *)(v41 + 33) & 0x20) == 0 )
      {
        LODWORD(v175) = *sub_26CC460(v139, (__int64 *)&v146);
        sub_BC8EC0((__int64)&v38[-2].m128i_i64[1], (unsigned int *)&v175, 1, 0);
      }
      goto LABEL_92;
    }
    if ( !v38[1].m128i_i64[1] )
      goto LABEL_92;
    v42 = sub_B10CD0((__int64)&v38[1].m128i_i64[1]);
    v44 = sub_26CAC90(v140, (__int64)&v38[-2].m128i_i64[1], v43);
    if ( !v44 )
      goto LABEL_92;
    v149 = sub_C1B090(v42, 0);
    v45 = *(_QWORD **)(v44 + 168);
    if ( v45 && (v46 = sub_C1BA30(v45, (__int64)&v149)) != 0 )
      v47 = (unsigned int *)(v46 + 16);
    else
      v47 = (unsigned int *)&v149;
    v48 = (const __m128i *)sub_26C2A80(v44 + 72, v47);
    if ( v48 == (const __m128i *)(v44 + 80) )
      goto LABEL_92;
    v168.m128i_i8[8] &= ~1u;
    s = 0;
    v49 = v48[3].m128i_i64[1];
    v166 = 0;
    v165 = v49;
    *(_QWORD *)v167 = v48[4].m128i_i64[1];
    v50 = _mm_loadu_si128(v48 + 5);
    v168.m128i_i64[0] = 0;
    *(__m128i *)&v167[8] = v50;
    sub_26BAC00(&s, (__int64)v48[3].m128i_i64);
    if ( (v168.m128i_i8[8] & 1) != 0 )
      goto LABEL_92;
    if ( !*(_QWORD *)v167 )
    {
      v52 = v166;
      while ( v52 )
      {
        v53 = (unsigned __int64)v52;
        v52 = (_QWORD *)*v52;
        j_j___libc_free_0(v53);
      }
      memset(s, 0, 8 * v165);
      *(_QWORD *)v167 = 0;
      v166 = 0;
LABEL_90:
      if ( s != &v168 )
        j_j___libc_free_0((unsigned __int64)s);
      goto LABEL_92;
    }
    if ( unk_4F838D4 )
    {
      sub_3143F80(&v153, &v38[-2].m128i_u64[1], v51);
      if ( BYTE4(v155) )
      {
        v131 = *(float *)&v155;
        if ( *(float *)&v155 < 1.0 )
        {
          v95 = v166;
          v158 = 1;
          v157 = (__m128i *)v162;
          v159 = 0;
          v160 = 0;
          v161.m128i_i32[0] = 1065353216;
          v161.m128i_i64[1] = 0;
          v162[0] = 0;
          if ( v166 )
          {
            v128 = v44;
            do
            {
              v97 = v95[3];
              if ( v97 < 0 )
              {
                v106 = v95[3] & 1LL | (v95[3] >> 1);
                v98 = (float)(int)v106 + (float)(int)v106;
              }
              else
              {
                v98 = (float)(int)v97;
              }
              v99 = v98 * v131;
              if ( v99 < 9.223372e18 )
                v96 = (unsigned int)(int)v99;
              else
                v96 = (unsigned int)(int)(float)(v99 - 9.223372e18) ^ 0x8000000000000000LL;
              *sub_C1CD30(&v157, (const __m128i *)(v95 + 1)) = v96;
              v95 = (_QWORD *)*v95;
            }
            while ( v95 );
            v44 = v128;
            v109 = v158;
            v110 = v160;
          }
          else
          {
            v110 = 0;
            v109 = 1;
          }
          v111 = _mm_load_si128(&v161);
          v175 = 0;
          v180.m128i_i8[8] &= ~1u;
          v176 = v109;
          v177 = 0;
          v178 = v110;
          v180.m128i_i64[0] = 0;
          v179 = v111;
          sub_26BAC00(&v175, (__int64)&v157);
          if ( (v168.m128i_i8[8] & 1) == 0 )
          {
            sub_26C2A20((__int64)&s);
            if ( s != &v168 )
              j_j___libc_free_0((unsigned __int64)s);
          }
          if ( (v180.m128i_i8[8] & 1) != 0 )
          {
            v168.m128i_i8[8] |= 1u;
            v165 = v176;
            LODWORD(s) = (_DWORD)v175;
          }
          else
          {
            v165 = v176;
            v112 = _mm_load_si128(&v179);
            v168.m128i_i64[0] = 0;
            *(_QWORD *)v167 = v178;
            v168.m128i_i8[8] &= ~1u;
            s = v175;
            v166 = v177;
            *(__m128i *)&v167[8] = v112;
            if ( v175 == &v180 )
            {
              s = &v168;
              v168.m128i_i64[0] = v180.m128i_i64[0];
            }
            if ( v177 )
              *((_QWORD *)s + v177[4] % v176) = &v166;
          }
          sub_26C2A20((__int64)&v157);
          if ( v157 != (__m128i *)v162 )
            j_j___libc_free_0((unsigned __int64)v157);
        }
      }
    }
    v54 = (unsigned __int64)v166;
    LODWORD(v158) = 0;
    v153 = &v155;
    v154 = 0x200000000LL;
    v159 = 0;
    v160 = &v158;
    v161 = (__m128i)(unsigned __int64)&v158;
    if ( v166 )
    {
      v129 = &v38[-2].m128i_i64[1];
      v126 = v44;
      v123 = v38;
      do
      {
        while ( 1 )
        {
          v57 = sub_22077B0(0x38u);
          v58 = *(_QWORD *)(v54 + 24);
          v59 = _mm_loadu_si128((const __m128i *)(v54 + 8));
          v60 = v57 + 32;
          v61 = v57;
          *(_QWORD *)(v57 + 48) = v58;
          *(__m128i *)(v57 + 32) = v59;
          v62 = sub_C1C4A0((__int64)&v157, v57 + 32);
          if ( v63 )
            break;
          j_j___libc_free_0(v61);
          v54 = *(_QWORD *)v54;
          if ( !v54 )
            goto LABEL_106;
        }
        if ( v62 || v63 == &v158 )
        {
          v55 = 1;
        }
        else if ( v58 == v63[6] )
        {
          v120 = v63;
          v85 = sub_C1F8C0(v60, (__int64)(v63 + 4));
          v63 = v120;
          v55 = v85 < 0;
        }
        else
        {
          v55 = v58 > v63[6];
        }
        sub_220F040(v55, v61, v63, &v158);
        ++v161.m128i_i64[1];
        v54 = *(_QWORD *)v54;
      }
      while ( v54 );
LABEL_106:
      v64 = (__int64)v160;
      v40 = (__int64)v129;
      v44 = v126;
      v38 = v123;
      if ( v160 != &v158 )
      {
        v142 = (__int64)v129;
        v130 = v126;
        v127 = v123;
        v65 = v160;
        do
        {
          v66 = (int *)v65[4];
          v67 = (unsigned int *)v65[5];
          if ( v66 )
          {
            sub_C7D030(&v175);
            sub_C7D280((int *)&v175, v66, (size_t)v67);
            sub_C7D290(&v175, &v150);
            v67 = v150;
          }
          v68 = (unsigned int)v154;
          v69 = (unsigned int *)v65[6];
          v70 = v154;
          if ( (unsigned int)v154 >= (unsigned __int64)HIDWORD(v154) )
          {
            if ( HIDWORD(v154) < (unsigned __int64)(unsigned int)v154 + 1 )
            {
              v124 = v65[6];
              sub_C8D5F0((__int64)&v153, &v155, (unsigned int)v154 + 1LL, 0x10u, v64, v56);
              v68 = (unsigned int)v154;
              v69 = (unsigned int *)v124;
            }
            v74 = (unsigned int **)&v153[2 * v68];
            *v74 = v67;
            v74[1] = v69;
            LODWORD(v154) = v154 + 1;
          }
          else
          {
            v71 = (unsigned int **)&v153[2 * (unsigned int)v154];
            if ( v71 )
            {
              *v71 = v67;
              v71[1] = v69;
              v70 = v154;
            }
            LODWORD(v154) = v70 + 1;
          }
          v65 = (__int64 *)sub_220EF30((__int64)v65);
        }
        while ( v65 != &v158 );
        v40 = v142;
        v44 = v130;
        v38 = v127;
      }
      v54 = v159;
    }
    sub_26BB830(v54);
    v72 = v166;
    if ( v166 )
    {
      v73 = 0;
      do
      {
        v73 += v72[3];
        v72 = (_QWORD *)*v72;
      }
      while ( v72 );
      if ( unk_4F838D3 )
        goto LABEL_121;
    }
    else
    {
      v73 = 0;
      if ( unk_4F838D3 )
        goto LABEL_122;
    }
    v86 = *(_QWORD **)(v44 + 168);
    if ( v86 && (v87 = sub_C1BA30(v86, (__int64)&v149)) != 0 )
      v88 = (__int64 *)(v87 + 16);
    else
      v88 = &v149;
    v89 = *(_QWORD *)(v44 + 136);
    v90 = v44 + 128;
    if ( !v89 )
      goto LABEL_121;
    v91 = *(_DWORD *)v88;
    v92 = v44 + 128;
    while ( 1 )
    {
      while ( *(_DWORD *)(v89 + 32) < v91 )
      {
        v89 = *(_QWORD *)(v89 + 24);
LABEL_158:
        if ( !v89 )
        {
LABEL_159:
          if ( v90 != v92
            && *(_DWORD *)(v92 + 32) <= v91
            && (*(_DWORD *)(v92 + 32) != v91 || *((_DWORD *)v88 + 1) >= *(_DWORD *)(v92 + 36)) )
          {
            v93 = *(_QWORD *)(v92 + 64);
            for ( j = v92 + 48; j != v93; v93 = sub_220EF30(v93) )
              v73 += sub_EF9210((_QWORD *)(v93 + 48));
          }
LABEL_121:
          if ( v73 )
          {
            sub_26CB7A0(v40, (__int64)&v153, v73);
          }
          else
          {
LABEL_122:
            if ( byte_4FF61E8 )
              sub_B99FD0(v40, 2u, 0);
          }
          if ( v153 != &v155 )
            _libc_free((unsigned __int64)v153);
          if ( (v168.m128i_i8[8] & 1) == 0 )
          {
            sub_26C2A20((__int64)&s);
            goto LABEL_90;
          }
LABEL_92:
          v38 = (__m128i *)v38->m128i_i64[1];
          if ( v144 == v38 )
          {
            v6 = (__int64)v146;
            v2 = v140;
            v144 = v146 + 3;
            goto LABEL_7;
          }
          goto LABEL_73;
        }
      }
      if ( *(_DWORD *)(v89 + 32) == v91 && *(_DWORD *)(v89 + 36) < *((_DWORD *)v88 + 1) )
      {
        v89 = *(_QWORD *)(v89 + 24);
        goto LABEL_158;
      }
      v92 = v89;
      v89 = *(_QWORD *)(v89 + 16);
      if ( !v89 )
        goto LABEL_159;
    }
  }
  return result;
}
