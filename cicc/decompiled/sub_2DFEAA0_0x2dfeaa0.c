// Function: sub_2DFEAA0
// Address: 0x2dfeaa0
//
__int64 __fastcall sub_2DFEAA0(__int64 a1, __int64 i, unsigned __int64 a3, __int64 a4, __int64 a5, unsigned __int64 a6)
{
  unsigned __int32 v6; // r14d
  __int64 v7; // r15
  unsigned int *v8; // r14
  unsigned int v9; // r12d
  unsigned int v10; // eax
  __int64 v11; // rbx
  __int64 **v12; // r13
  __int64 v13; // rbx
  __int64 v14; // rdx
  signed __int64 v15; // r8
  unsigned int v16; // edi
  __int64 v17; // rax
  __int64 v18; // r12
  unsigned __int64 v19; // rbx
  __int64 *v20; // rax
  __int64 v21; // r11
  signed __int64 *v22; // r12
  unsigned int v23; // eax
  __int64 v24; // r9
  __int64 v25; // rdx
  __int64 v26; // rdi
  __int64 v27; // rdx
  __int64 v28; // r13
  __int64 *v29; // rax
  __int64 v30; // rax
  _DWORD *v31; // rsi
  __int64 *v32; // rax
  bool v33; // al
  __int64 v34; // rax
  _QWORD *v35; // rax
  __int64 v36; // rcx
  __int64 v37; // rdx
  __int64 *v38; // r9
  unsigned __int64 v39; // rax
  __int64 v40; // r8
  __int64 *v41; // rdi
  _DWORD *v42; // rax
  __int64 v43; // r11
  __int64 v44; // r11
  __int64 *v45; // rax
  __int64 *v46; // rax
  __int64 v47; // rax
  __int64 v48; // rax
  __int64 v49; // r8
  __int64 v50; // r10
  unsigned int v51; // r10d
  _QWORD *v52; // rbx
  __int64 v53; // rdx
  __int64 v54; // rcx
  __int64 v55; // r8
  __int64 v56; // r9
  __int64 v57; // rax
  __int64 v58; // r9
  __int64 v59; // rdx
  __int64 v60; // rdi
  __int64 v61; // rax
  __int64 v62; // rcx
  __int64 v63; // rdx
  void *v64; // r8
  __int64 v65; // rax
  unsigned __int64 v66; // rdi
  __int64 *v67; // r13
  __int64 v68; // rax
  __int64 v69; // rdx
  __int64 v70; // rax
  __int64 v71; // r9
  unsigned __int32 *v72; // rcx
  __int64 v73; // rdx
  unsigned __int32 *v74; // r15
  __int64 v75; // rsi
  __int64 v76; // rdx
  unsigned __int32 *v77; // rax
  unsigned __int32 *v78; // rdx
  __int64 v79; // rax
  int v80; // edx
  __int64 v82; // r11
  __int64 v83; // rdx
  __int64 v84; // rax
  __int64 v85; // rax
  _DWORD *v86; // rsi
  _BYTE *v87; // r9
  int v88; // r10d
  __int64 v89; // r11
  int v90; // r10d
  unsigned int v91; // eax
  __int64 v92; // rcx
  __int64 *v93; // rbx
  __int64 v94; // rax
  __int64 v95; // r13
  __int64 v96; // r9
  _QWORD *v97; // rax
  _QWORD *v98; // rsi
  __int64 v99; // rax
  __int64 v100; // rax
  unsigned int v101; // esi
  unsigned int v102; // esi
  __int64 *v103; // rsi
  unsigned __int64 v104; // rdx
  __int64 v105; // rax
  unsigned __int32 *v106; // rbx
  unsigned __int32 *v107; // r13
  unsigned int v108; // r15d
  int *v109; // rsi
  __int64 v110; // rdx
  void **v111; // rax
  void *v112; // rdi
  _BYTE *v113; // rcx
  char v114; // dl
  char v115; // al
  char v116; // al
  bool v117; // al
  __int64 *v118; // rax
  __int64 v119; // rcx
  __int64 v120; // rax
  unsigned __int64 v121; // rdi
  size_t v122; // rax
  unsigned __int64 v123; // rcx
  int v124; // eax
  __int64 *v125; // rax
  __int64 v126; // rcx
  __int64 *v127; // rsi
  __int64 v128; // rsi
  unsigned int v129; // esi
  __int32 v130; // eax
  __int64 v131; // rax
  __int64 v132; // [rsp+0h] [rbp-220h]
  __int64 *v133; // [rsp+8h] [rbp-218h]
  int v134; // [rsp+18h] [rbp-208h]
  __int64 v135; // [rsp+20h] [rbp-200h]
  __int64 v136; // [rsp+28h] [rbp-1F8h]
  __int64 v137; // [rsp+30h] [rbp-1F0h]
  unsigned int v138; // [rsp+38h] [rbp-1E8h]
  __int64 v139; // [rsp+38h] [rbp-1E8h]
  __int64 v140; // [rsp+38h] [rbp-1E8h]
  __int64 **v141; // [rsp+40h] [rbp-1E0h]
  __int64 v142; // [rsp+48h] [rbp-1D8h]
  __int64 v143; // [rsp+50h] [rbp-1D0h]
  const void *v144; // [rsp+58h] [rbp-1C8h]
  int v145; // [rsp+60h] [rbp-1C0h]
  unsigned __int8 v146; // [rsp+67h] [rbp-1B9h]
  signed __int64 *v147; // [rsp+70h] [rbp-1B0h]
  __int64 v148; // [rsp+70h] [rbp-1B0h]
  __int32 v149; // [rsp+78h] [rbp-1A8h]
  __int64 v150; // [rsp+78h] [rbp-1A8h]
  __int64 v152; // [rsp+88h] [rbp-198h]
  _QWORD *v153; // [rsp+88h] [rbp-198h]
  unsigned __int64 *v154; // [rsp+88h] [rbp-198h]
  _BYTE *v155; // [rsp+88h] [rbp-198h]
  unsigned int *v156; // [rsp+98h] [rbp-188h]
  int v157; // [rsp+A4h] [rbp-17Ch] BYREF
  __int64 v158; // [rsp+A8h] [rbp-178h] BYREF
  void *src; // [rsp+B0h] [rbp-170h] BYREF
  unsigned __int8 v160; // [rsp+B8h] [rbp-168h]
  __int64 v161; // [rsp+C0h] [rbp-160h]
  __int64 v162; // [rsp+D0h] [rbp-150h] BYREF
  _BYTE *v163; // [rsp+D8h] [rbp-148h] BYREF
  __int64 v164; // [rsp+E0h] [rbp-140h]
  _BYTE v165[72]; // [rsp+E8h] [rbp-138h] BYREF
  __int64 *v166; // [rsp+130h] [rbp-F0h] BYREF
  __int64 v167; // [rsp+138h] [rbp-E8h] BYREF
  __int64 v168; // [rsp+140h] [rbp-E0h] BYREF
  _BYTE v169[72]; // [rsp+148h] [rbp-D8h] BYREF
  __m128i v170; // [rsp+190h] [rbp-90h] BYREF
  __int64 v171; // [rsp+1A0h] [rbp-80h] BYREF
  _QWORD v172[15]; // [rsp+1A8h] [rbp-78h] BYREF

  v6 = i;
  v163 = v165;
  v164 = 0x400000000LL;
  v152 = a1 + 232;
  v162 = a1 + 232;
  v156 = (unsigned int *)(a3 + 4 * a4);
  if ( v156 != (unsigned int *)a3 )
  {
    v146 = 0;
    v7 = a5;
    v8 = (unsigned int *)a3;
    v149 = i;
    v142 = 40LL * (unsigned int)i;
    v144 = (const void *)(a5 + 168);
    v143 = a5 + 152;
    while ( 1 )
    {
      v9 = *v8;
      a3 = *(unsigned int *)(v7 + 160);
      v10 = *v8 & 0x7FFFFFFF;
      v11 = 8LL * v10;
      if ( v10 >= (unsigned int)a3 )
        break;
      i = *(_QWORD *)(v7 + 152);
      v12 = *(__int64 ***)(i + 8LL * v10);
      if ( !v12 )
        break;
LABEL_5:
      a4 = *((unsigned int *)v12 + 2);
      if ( (_DWORD)a4 )
      {
        v13 = v162;
        v14 = *(unsigned int *)(v162 + 160);
        v15 = **v12;
        if ( (_DWORD)v14 )
        {
          i = **v12;
          sub_2DF6390((__int64)&v162, i, v14, a4, v15, a6);
          a5 = (unsigned int)v164;
        }
        else
        {
          i = *(unsigned int *)(v162 + 164);
          if ( (_DWORD)i )
          {
            a4 = v162 + 8;
            v16 = *(_DWORD *)((v15 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (v15 >> 1) & 3;
            do
            {
              v15 = *(_QWORD *)a4 & 0xFFFFFFFFFFFFFFF8LL;
              if ( (*(_DWORD *)(v15 + 24) | (unsigned int)(*(__int64 *)a4 >> 1) & 3) > v16 )
                break;
              v14 = (unsigned int)(v14 + 1);
              a4 += 16;
            }
            while ( (_DWORD)i != (_DWORD)v14 );
          }
          v17 = 0;
          LODWORD(v164) = 0;
          v18 = (v14 << 32) | (unsigned int)i;
          if ( !HIDWORD(v164) )
          {
            i = (__int64)v165;
            sub_C8D5F0((__int64)&v163, v165, 1u, 0x10u, v15, a6);
            v17 = 16LL * (unsigned int)v164;
          }
          a3 = (unsigned __int64)v163;
          *(_QWORD *)&v163[v17] = v13;
          *(_QWORD *)(a3 + v17 + 8) = v18;
          a5 = (unsigned int)(v164 + 1);
          LODWORD(v164) = v164 + 1;
        }
        if ( (_DWORD)a5 )
        {
          v19 = (unsigned __int64)v163;
          if ( *((_DWORD *)v163 + 3) < *((_DWORD *)v163 + 2) )
          {
            v20 = (__int64 *)sub_2DF4990((__int64)&v162);
            i = *((unsigned int *)v12 + 2);
            a4 = (__int64)*v12;
            v22 = *v12;
            v23 = *(_DWORD *)((*v20 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (*v20 >> 1) & 3;
            v24 = (*v12)[3 * i - 2];
            v25 = v24 >> 1;
            a6 = v24 & 0xFFFFFFFFFFFFFFF8LL;
            a3 = *(_DWORD *)(a6 + 24) | (unsigned int)(v25 & 3);
            if ( v23 < (unsigned int)a3 )
            {
              v147 = (signed __int64 *)(a4 + 24 * i);
              v26 = *(_QWORD *)(a4 + 8);
              a3 = v26 & 0xFFFFFFFFFFFFFFF8LL;
              if ( (*(_DWORD *)((v26 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v26 >> 1) & 3) <= v23 )
              {
                do
                {
                  v27 = v22[4];
                  v22 += 3;
                  a3 = *(_DWORD *)((v27 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v27 >> 1) & 3;
                }
                while ( v23 >= (unsigned int)a3 );
              }
              if ( v22 != v147 )
              {
                v145 = -1;
                v141 = v12;
                v28 = v21;
                while ( 1 )
                {
                  v29 = (__int64 *)sub_2DF4990(v28);
                  i *= 24;
                  a3 = *(_DWORD *)((*v29 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v29 >> 1) & 3;
                  if ( (unsigned int)a3 < (*(_DWORD *)((*(_QWORD *)(a4 + i - 16) & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                         | (unsigned int)(*(__int64 *)(a4 + i - 16) >> 1) & 3) )
                  {
                    v99 = v22[1];
                    a4 = v99 & 0xFFFFFFFFFFFFFFF8LL;
                    if ( (*(_DWORD *)((v99 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v99 >> 1) & 3) <= (unsigned int)a3 )
                    {
                      do
                      {
                        v100 = v22[4];
                        v22 += 3;
                        a4 = v100 & 0xFFFFFFFFFFFFFFF8LL;
                      }
                      while ( (unsigned int)a3 >= (*(_DWORD *)((v100 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                 | (unsigned int)(v100 >> 1) & 3) );
                    }
                  }
                  else
                  {
                    v22 = (signed __int64 *)(a4 + i);
                  }
                  if ( v22 == v147 )
                    goto LABEL_63;
                  v30 = sub_2DF49D0(v28);
                  v170.m128i_i32[0] = v149;
                  v31 = (_DWORD *)(*(_QWORD *)v30 + 4LL * (*(_BYTE *)(v30 + 8) & 0x3F));
                  if ( v31 != sub_2DF4D60(*(_DWORD **)v30, (__int64)v31, v170.m128i_i32) )
                    break;
LABEL_43:
                  a4 = v162;
                  a3 = *(unsigned int *)(v162 + 160);
                  if ( (_DWORD)a3 )
                    goto LABEL_44;
LABEL_113:
                  a3 = 0;
                  v47 = *(_QWORD *)(*(_QWORD *)(v19 + 16LL * (unsigned int)v164 - 16)
                                  + 16LL * *(unsigned int *)(v19 + 16LL * (unsigned int)v164 - 16 + 12)
                                  + 8);
LABEL_45:
                  i = *(_DWORD *)((v22[1] & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v22[1] >> 1) & 3;
                  if ( (unsigned int)i >= (*(_DWORD *)((v47 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v47 >> 1) & 3) )
                  {
                    sub_2DF51A0(v28);
                    a5 = (unsigned int)v164;
                    if ( !(_DWORD)v164 )
                      goto LABEL_63;
                    v19 = (unsigned __int64)v163;
                    a6 = *((unsigned int *)v163 + 3);
                    if ( (unsigned int)a6 >= *((_DWORD *)v163 + 2) )
                      goto LABEL_63;
                    v118 = (__int64 *)sub_2DF4990(v28);
                    i = *((unsigned int *)v141 + 2);
                    a4 = (__int64)*v141;
                    a3 = *(_DWORD *)((*v118 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v118 >> 1) & 3;
                    v48 = (__int64)&(*v141)[3 * i];
                    if ( (unsigned int)a3 < (*(_DWORD *)((*(_QWORD *)(v48 - 16) & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                           | (unsigned int)(*(__int64 *)(v48 - 16) >> 1) & 3) )
                    {
                      v48 = (__int64)v22;
                      if ( (*(_DWORD *)((v22[1] & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v22[1] >> 1) & 3) > (unsigned int)a3 )
                        goto LABEL_61;
                      do
                      {
                        v119 = *(_QWORD *)(v48 + 32);
                        v48 += 24;
                        i = v119 & 0xFFFFFFFFFFFFFFF8LL;
                        a4 = *(_DWORD *)((v119 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v119 >> 1) & 3;
                      }
                      while ( (unsigned int)a3 >= (unsigned int)a4 );
                    }
                  }
                  else
                  {
                    v48 = (__int64)(v22 + 3);
                    if ( v22 + 3 == v147 )
                      goto LABEL_63;
                    a5 = (unsigned int)v164;
                    if ( !(_DWORD)v164 )
                      goto LABEL_63;
                    i = *(unsigned int *)(v19 + 8);
                    if ( *(_DWORD *)(v19 + 12) < (unsigned int)i )
                    {
                      if ( (_DWORD)a3 )
                      {
                        i = v22[3];
                        sub_2DF7A30(v28, i);
                        a5 = (unsigned int)v164;
                        v48 = (__int64)(v22 + 3);
                      }
                      else
                      {
                        a6 = *(unsigned int *)(a4 + 164);
                        v49 = v19 + 16LL * (unsigned int)v164 - 16;
                        for ( i = *(unsigned int *)(v49 + 12); (_DWORD)a6 != (_DWORD)i; i = (unsigned int)(i + 1) )
                        {
                          v50 = *(_QWORD *)(a4 + 16LL * (unsigned int)i + 8);
                          a3 = *(_DWORD *)((v50 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v50 >> 1) & 3;
                          if ( (unsigned int)a3 > (*(_DWORD *)((v22[3] & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                 | (unsigned int)(v22[3] >> 1) & 3) )
                            break;
                        }
                        *(_DWORD *)(v49 + 12) = i;
                        a5 = (unsigned int)v164;
                      }
                      if ( !(_DWORD)a5 )
                        goto LABEL_63;
                      v19 = (unsigned __int64)v163;
                    }
                    a6 = *(unsigned int *)(v19 + 12);
                    v51 = *(_DWORD *)(v19 + 8);
                  }
                  if ( (signed __int64 *)v48 == v147 || (unsigned int)a6 >= v51 )
                    goto LABEL_63;
                  i = *((unsigned int *)v141 + 2);
LABEL_61:
                  v22 = (signed __int64 *)v48;
                }
                v138 = *(_DWORD *)(v162 + 160);
                v32 = (__int64 *)(*(_QWORD *)(v19 + 16LL * (unsigned int)a5 - 16)
                                + 16LL * *(unsigned int *)(v19 + 16LL * (unsigned int)a5 - 16 + 12)
                                + 8);
                if ( v138 )
                {
                  v137 = v162;
                  v117 = sub_2DF8300(v22, *v32);
                  a4 = v137;
                  a3 = v138;
                  if ( !v117 )
                  {
LABEL_44:
                    v47 = *(_QWORD *)(*(_QWORD *)(v19 + 16LL * (unsigned int)v164 - 16)
                                    + 16LL * *(unsigned int *)(v19 + 16LL * (unsigned int)v164 - 16 + 12)
                                    + 8);
                    goto LABEL_45;
                  }
                }
                else
                {
                  v139 = v162;
                  v33 = sub_2DF8300(v22, *v32);
                  a4 = v139;
                  if ( !v33 )
                    goto LABEL_113;
                }
                if ( v145 == -1 )
                {
                  v130 = *((_DWORD *)v141 + 28);
                  v171 = 0;
                  v170.m128i_i64[0] = 0;
                  v170.m128i_i32[2] = v130;
                  v131 = *(_QWORD *)(a1 + 56);
                  v172[0] = 0;
                  v172[1] = 0;
                  v170.m128i_i32[0] = *(_DWORD *)(v131 + v142) & 0xFFF00;
                  v146 = 1;
                  v145 = sub_2DF49F0(a1, &v170, a3, a4, a5, a6);
                }
                v158 = *(_QWORD *)sub_2DF4990(v28);
                v140 = *(_QWORD *)sub_2DF49B0(v28);
                v34 = sub_2DF49D0(v28);
                sub_2DF52D0((__int64)&src, v34);
                if ( sub_2DF8300(&v158, *v22) )
                {
                  v35 = (_QWORD *)sub_2DF4990(v28);
                  *v35 = v36;
                }
                v37 = v22[1];
                v136 = (v140 >> 1) & 3;
                if ( ((unsigned int)v136 | *(_DWORD *)((v140 & 0xFFFFFFFFFFFFFFF8LL) + 24)) > (*(_DWORD *)((v37 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                             | (unsigned int)(v37 >> 1)
                                                                                             & 3) )
                {
                  *(_QWORD *)(*(_QWORD *)&v163[16 * (unsigned int)v164 - 16]
                            + 16LL * *(unsigned int *)&v163[16 * (unsigned int)v164 - 4]
                            + 8) = v37;
                  v128 = (unsigned int)(v164 - 1);
                  if ( *(_DWORD *)&v163[16 * v128 + 12] == *(_DWORD *)&v163[16 * v128 + 8] - 1 )
                    sub_2DF4670(v28, v128, v37);
                }
                v38 = (__int64 *)src;
                v170.m128i_i64[0] = (__int64)&v171;
                v157 = v149;
                v170.m128i_i64[1] = 0xC00000000LL;
                v39 = v160 & 0x3F;
                v40 = 4 * v39;
                if ( v39 > 0xC )
                {
                  v132 = 4 * v39;
                  v133 = (__int64 *)src;
                  v134 = v160 & 0x3F;
                  sub_C8D5F0((__int64)&v170, &v171, v160 & 0x3F, 4u, v40, (__int64)src);
                  LODWORD(v39) = v134;
                  v38 = v133;
                  v40 = v132;
                  v127 = (__int64 *)(v170.m128i_i64[0] + 4LL * v170.m128i_u32[2]);
                }
                else
                {
                  v41 = &v171;
                  if ( !v40 )
                    goto LABEL_34;
                  v127 = &v171;
                }
                if ( (unsigned int)v40 >= 8 )
                {
                  *v127 = *v38;
                  *(__int64 *)((char *)v127 + v40 - 8) = *(__int64 *)((char *)v38 + v40 - 8);
                  qmemcpy(
                    (void *)((unsigned __int64)(v127 + 1) & 0xFFFFFFFFFFFFFFF8LL),
                    (const void *)((char *)v38 - ((char *)v127 - ((unsigned __int64)(v127 + 1) & 0xFFFFFFFFFFFFFFF8LL))),
                    8LL * (((unsigned int)v40 + (_DWORD)v127 - (((_DWORD)v127 + 8) & 0xFFFFFFF8)) >> 3));
                }
                else if ( (v40 & 4) != 0 )
                {
                  *(_DWORD *)v127 = *(_DWORD *)v38;
                  *(_DWORD *)((char *)v127 + (unsigned int)v40 - 4) = *(_DWORD *)((char *)v38 + (unsigned int)v40 - 4);
                }
                else if ( (_DWORD)v40 )
                {
                  *(_BYTE *)v127 = *(_BYTE *)v38;
                  if ( (v40 & 2) != 0 )
                    *(_WORD *)((char *)v127 + (unsigned int)v40 - 2) = *(_WORD *)((char *)v38 + (unsigned int)v40 - 2);
                }
                v41 = (__int64 *)v170.m128i_i64[0];
                LODWORD(v40) = v170.m128i_i32[2];
LABEL_34:
                v170.m128i_i32[2] = v40 + v39;
                v42 = sub_2DF4E20(v41, (__int64)v41 + 4 * (unsigned int)(v40 + v39), &v157);
                v135 = v43;
                *v42 = v145;
                sub_2DF5BF0(v43, (int *)v170.m128i_i64[0], v170.m128i_u32[2], (v160 & 0x40) != 0, v160 >> 7, v161);
                v44 = v135;
                if ( (__int64 *)v170.m128i_i64[0] != &v171 )
                {
                  _libc_free(v170.m128i_u64[0]);
                  v44 = v135;
                }
                sub_2DF7680(v28, v44);
                if ( v166 )
                  j_j___libc_free_0_0((unsigned __int64)v166);
                v45 = (__int64 *)sub_2DF4990(v28);
                if ( sub_2DF8300(&v158, *v45) )
                {
                  sub_2DF52D0((__int64)&v170, (__int64)&src);
                  v125 = (__int64 *)sub_2DF4990(v28);
                  sub_2DFCEE0(v28, v158, *v125, v126);
                  if ( v170.m128i_i64[0] )
                    j_j___libc_free_0_0(v170.m128i_u64[0]);
                  sub_2DF51A0(v28);
                }
                v46 = (__int64 *)sub_2DF49B0(v28);
                if ( (*(_DWORD *)((v140 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)v136) <= (*(_DWORD *)((*v46 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                              | (unsigned int)(*v46 >> 1)
                                                                                              & 3) )
                {
LABEL_40:
                  if ( src )
                    j_j___libc_free_0_0((unsigned __int64)src);
                  v19 = (unsigned __int64)v163;
                  goto LABEL_43;
                }
                sub_2DF51A0(v28);
                sub_2DF52D0((__int64)&v170, (__int64)&src);
                sub_2DFCEE0(v28, v22[1], v140, (__int64)&v170);
                if ( v170.m128i_i64[0] )
                  j_j___libc_free_0_0(v170.m128i_u64[0]);
                v123 = (unsigned __int64)&v163[16 * (unsigned int)v164 - 16];
                v124 = *(_DWORD *)(v123 + 12);
                if ( v124 )
                {
                  if ( (_DWORD)v164 && *((_DWORD *)v163 + 3) < *((_DWORD *)v163 + 2)
                    || (v129 = *(_DWORD *)(v162 + 160)) == 0 )
                  {
                    *(_DWORD *)(v123 + 12) = v124 - 1;
                    goto LABEL_40;
                  }
                }
                else
                {
                  v129 = *(_DWORD *)(v162 + 160);
                }
                sub_F03AD0((unsigned int *)&v163, v129);
                goto LABEL_40;
              }
            }
          }
        }
      }
LABEL_63:
      if ( v156 == ++v8 )
      {
        v6 = v149;
        goto LABEL_65;
      }
    }
    v91 = v10 + 1;
    if ( (unsigned int)a3 < v91 && v91 != a3 )
    {
      if ( v91 >= a3 )
      {
        v95 = *(_QWORD *)(v7 + 168);
        v96 = v91 - a3;
        if ( v91 > (unsigned __int64)*(unsigned int *)(v7 + 164) )
        {
          v148 = v91 - a3;
          sub_C8D5F0(v143, v144, v91, 8u, v91, v96);
          a3 = *(unsigned int *)(v7 + 160);
          v96 = v148;
        }
        v92 = *(_QWORD *)(v7 + 152);
        v97 = (_QWORD *)(v92 + 8 * a3);
        v98 = &v97[v96];
        if ( v97 != v98 )
        {
          do
            *v97++ = v95;
          while ( v98 != v97 );
          LODWORD(a3) = *(_DWORD *)(v7 + 160);
          v92 = *(_QWORD *)(v7 + 152);
        }
        *(_DWORD *)(v7 + 160) = v96 + a3;
        goto LABEL_99;
      }
      *(_DWORD *)(v7 + 160) = v91;
    }
    v92 = *(_QWORD *)(v7 + 152);
LABEL_99:
    v93 = (__int64 *)(v92 + v11);
    v94 = sub_2E10F30(v9);
    *v93 = v94;
    i = v94;
    v12 = (__int64 **)v94;
    sub_2E11E80(v7, v94);
    goto LABEL_5;
  }
  v146 = 0;
LABEL_65:
  v52 = v172;
  v170.m128i_i64[1] = (__int64)v172;
  v170.m128i_i64[0] = v152;
  v171 = 0x400000000LL;
  sub_2DF64A0(v170.m128i_i64, i, a3, a4, a5, a6);
  v167 = (__int64)v169;
  v168 = 0x400000000LL;
  v166 = (__int64 *)v170.m128i_i64[0];
  if ( (_DWORD)v171 )
    sub_2DF4EE0((__int64)&v167, (char **)&v170.m128i_i64[1], v53, v54, v55, v56);
  if ( (_QWORD *)v170.m128i_i64[1] != v172 )
    _libc_free(v170.m128i_u64[1]);
  v57 = (unsigned int)v168;
  v58 = v167;
  if ( !(_DWORD)v168 )
  {
LABEL_71:
    if ( (_BYTE *)v58 != v169 )
      _libc_free(v58);
    v59 = *(_QWORD *)(a1 + 56);
    v60 = v59 + 40LL * v6;
    v61 = *(unsigned int *)(a1 + 64);
    v62 = 5 * v61;
    v63 = v59 + 40 * v61;
    if ( v63 != v60 + 40 )
    {
      memmove((void *)v60, (const void *)(v60 + 40), v63 - (v60 + 40));
      LODWORD(v61) = *(_DWORD *)(a1 + 64);
    }
    v170.m128i_i64[1] = (__int64)v172;
    *(_DWORD *)(a1 + 64) = v61 - 1;
    v171 = 0x400000000LL;
    v170.m128i_i64[0] = v152;
    sub_2DF64A0(v170.m128i_i64, a1, v63, v62, v55, v58);
    v65 = (unsigned int)v171;
    v66 = v170.m128i_u64[1];
    if ( !(_DWORD)v171 )
    {
LABEL_88:
      if ( (_QWORD *)v66 != v52 )
        _libc_free(v66);
      goto LABEL_90;
    }
    v67 = &v168;
    while ( 1 )
    {
      if ( *(_DWORD *)(v66 + 12) >= *(_DWORD *)(v66 + 8) )
        goto LABEL_88;
      v68 = v66 + 16 * v65 - 16;
      v69 = *(unsigned int *)(v68 + 12);
      v70 = *(_QWORD *)v68;
      v71 = v70 + 24 * v69 + 64;
      v72 = *(unsigned __int32 **)v71;
      v73 = 4LL * (*(_BYTE *)(v70 + 24 * v69 + 72) & 0x3F);
      v74 = (unsigned __int32 *)(*(_QWORD *)v71 + v73);
      v75 = v73 >> 2;
      v76 = v73 >> 4;
      if ( v76 )
      {
        v77 = *(unsigned __int32 **)v71;
        v78 = &v72[4 * v76];
        while ( v6 >= *v77 )
        {
          if ( v6 < v77[1] )
          {
            ++v77;
            break;
          }
          if ( v6 < v77[2] )
          {
            v77 += 2;
            break;
          }
          if ( v6 < v77[3] )
          {
            v77 += 3;
            break;
          }
          v77 += 4;
          if ( v78 == v77 )
          {
            v75 = v74 - v77;
            goto LABEL_151;
          }
        }
LABEL_85:
        if ( v74 != v77 )
        {
          v166 = v67;
          v167 = 0x400000000LL;
          if ( v72 == v74 )
          {
            v110 = 0;
            v109 = (int *)v67;
          }
          else
          {
            v153 = v52;
            v103 = v67;
            v104 = 4;
            v105 = 0;
            v106 = v72;
            v107 = v74;
            while ( 1 )
            {
              v108 = *v106;
              if ( *v106 != -1 )
                v108 = (__PAIR64__(v108, v6) - v108) >> 32;
              if ( v105 + 1 > v104 )
              {
                v150 = v71;
                sub_C8D5F0((__int64)&v166, v103, v105 + 1, 4u, (__int64)v64, v71);
                v105 = (unsigned int)v167;
                v71 = v150;
              }
              ++v106;
              *((_DWORD *)v166 + v105) = v108;
              v105 = (unsigned int)(v167 + 1);
              LODWORD(v167) = v167 + 1;
              if ( v107 == v106 )
                break;
              v104 = HIDWORD(v167);
            }
            v67 = v103;
            v52 = v153;
            v109 = (int *)v166;
            v110 = (unsigned int)v105;
          }
          sub_2DF5BF0(
            (__int64)&src,
            v109,
            v110,
            (*(_BYTE *)(v71 + 8) & 0x40) != 0,
            *(_BYTE *)(v71 + 8) >> 7,
            *(_QWORD *)(v71 + 16));
          if ( v166 != v67 )
            _libc_free((unsigned __int64)v166);
          v111 = (void **)sub_2DF49D0((__int64)&v170);
          v112 = src;
          v113 = v111;
          if ( v111 != &src )
          {
            if ( (v160 & 0x3F) != 0 )
            {
              v154 = (unsigned __int64 *)v111;
              v120 = sub_2207820(4LL * (v160 & 0x3F));
              v113 = v154;
              v64 = (void *)v120;
              v121 = *v154;
              *v154 = v120;
              if ( v121 )
              {
                j_j___libc_free_0_0(v121);
                v113 = v154;
                v64 = (void *)*v154;
              }
              v112 = src;
              v114 = v160 & 0x3F;
              v122 = 4LL * (v160 & 0x3F);
              if ( v122 )
              {
                v155 = v113;
                memmove(v64, src, v122);
                v112 = src;
                v113 = v155;
                v114 = v160 & 0x3F;
              }
            }
            else
            {
              *v111 = 0;
              v112 = src;
              v114 = v160 & 0x3F;
            }
            v115 = v114 | v113[8] & 0xC0;
            v113[8] = v115;
            v116 = v160 & 0x40 | v115 & 0xBF;
            v113[8] = v116;
            v113[8] = v160 & 0x80 | v116 & 0x7F;
            *((_QWORD *)v113 + 2) = v161;
          }
          if ( v112 )
            j_j___libc_free_0_0((unsigned __int64)v112);
          v66 = v170.m128i_u64[1];
        }
        goto LABEL_86;
      }
      v77 = *(unsigned __int32 **)v71;
LABEL_151:
      if ( v75 != 2 )
      {
        if ( v75 != 3 )
        {
          if ( v75 != 1 )
            goto LABEL_86;
          goto LABEL_154;
        }
        if ( v6 < *v77 )
          goto LABEL_85;
        ++v77;
      }
      if ( v6 < *v77 )
        goto LABEL_85;
      ++v77;
LABEL_154:
      if ( v6 < *v77 )
        goto LABEL_85;
LABEL_86:
      v79 = v66 + 16LL * (unsigned int)v171 - 16;
      v80 = *(_DWORD *)(v79 + 12) + 1;
      *(_DWORD *)(v79 + 12) = v80;
      v65 = (unsigned int)v171;
      v66 = v170.m128i_u64[1];
      if ( v80 == *(_DWORD *)(v170.m128i_i64[1] + 16LL * (unsigned int)v171 - 8) )
      {
        v102 = *(_DWORD *)(v170.m128i_i64[0] + 160);
        if ( v102 )
        {
          sub_F03D40(&v170.m128i_i64[1], v102);
          v65 = (unsigned int)v171;
          v66 = v170.m128i_u64[1];
        }
      }
      if ( !(_DWORD)v65 )
        goto LABEL_88;
    }
  }
  while ( 1 )
  {
    if ( *(_DWORD *)(v58 + 12) >= *(_DWORD *)(v58 + 8) )
      goto LABEL_71;
    v82 = v58 + 16 * v57 - 16;
    v83 = 3LL * *(unsigned int *)(v82 + 12);
    v84 = *(_QWORD *)v82;
    v170.m128i_i32[0] = v6;
    v85 = v84 + 8 * v83 + 64;
    v86 = (_DWORD *)(*(_QWORD *)v85 + 4LL * (*(_BYTE *)(v85 + 8) & 0x3F));
    if ( v86 != sub_2DF4D60(*(_DWORD **)v85, (__int64)v86, v170.m128i_i32) )
      break;
    v90 = v88 + 1;
    *(_DWORD *)(v89 + 12) = v90;
    v57 = (unsigned int)v168;
    v58 = v167;
    if ( v90 == *(_DWORD *)(v167 + 16LL * (unsigned int)v168 - 8) )
    {
      v101 = *(_DWORD *)(v55 + 160);
      if ( v101 )
      {
        sub_F03D40(&v167, v101);
        v57 = (unsigned int)v168;
        v58 = v167;
      }
    }
    if ( !(_DWORD)v57 )
      goto LABEL_71;
  }
  if ( v87 != v169 )
    _libc_free((unsigned __int64)v87);
LABEL_90:
  if ( v163 != v165 )
    _libc_free((unsigned __int64)v163);
  return v146;
}
