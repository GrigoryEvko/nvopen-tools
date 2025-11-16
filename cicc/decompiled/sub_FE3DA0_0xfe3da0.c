// Function: sub_FE3DA0
// Address: 0xfe3da0
//
__int64 __fastcall sub_FE3DA0(__int64 a1)
{
  __int64 result; // rax
  __int64 v2; // rdi
  __int64 v3; // rdx
  __int64 v4; // rax
  __int64 v5; // rbx
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // r12
  __int64 v9; // r14
  unsigned __int64 v10; // rbx
  int v11; // r10d
  __int64 *v12; // rdx
  unsigned int v13; // edi
  _QWORD *v14; // rax
  __int64 v15; // rcx
  unsigned __int64 *v16; // rax
  __int16 v17; // cx
  __int16 v18; // dx
  __int16 v19; // dx
  unsigned __int64 v20; // rax
  __int64 v21; // r12
  __int64 v22; // rax
  int v23; // ecx
  __int64 v24; // r8
  _QWORD *v25; // rbx
  _QWORD *v26; // r13
  __int64 v27; // rdi
  double v28; // xmm0_8
  __int64 v29; // rsi
  __int64 v30; // r15
  char *v31; // rax
  __int64 v32; // r9
  char *v33; // rdx
  char *v34; // r14
  unsigned __int64 v35; // rax
  char *v36; // r12
  _QWORD *v37; // rax
  _QWORD *v38; // r13
  _QWORD *v39; // r14
  __int64 v40; // rdi
  char *v41; // r14
  int v42; // r8d
  unsigned int v43; // r15d
  size_t v44; // rdx
  __int64 *v45; // r13
  __int64 v46; // rax
  unsigned __int64 i; // rax
  unsigned __int64 v48; // rax
  __int64 *v49; // rax
  unsigned int v50; // ecx
  __int64 *v51; // r8
  __int64 v52; // r15
  int v53; // r9d
  __int64 v54; // rsi
  __int64 v55; // rdi
  __int64 *v56; // r13
  unsigned __int64 v57; // rbx
  __int64 v58; // rdi
  char *k; // rbx
  __int64 v60; // rbx
  __int64 v61; // r15
  __int64 v62; // rbx
  unsigned int v63; // esi
  __int64 *v64; // rdx
  __int64 v65; // r10
  __int64 v66; // r9
  __int64 v67; // rax
  __int64 v68; // rsi
  __int64 v69; // rax
  __int64 v70; // r14
  __int64 v71; // rax
  __int64 v72; // rcx
  __int64 v73; // rdx
  __int64 *v74; // rax
  __int64 v75; // rbx
  __int64 v76; // r15
  __int64 v77; // rax
  unsigned __int64 v78; // rdi
  unsigned __int16 v79; // cx
  unsigned __int64 v80; // rsi
  __int16 v81; // dx
  unsigned __int64 v82; // rax
  __int16 v83; // cx
  unsigned __int64 v84; // rdx
  unsigned __int16 v85; // dx
  unsigned __int64 v86; // rdi
  __int16 v87; // si
  __int16 v88; // dx
  unsigned __int64 v89; // rdi
  __int64 v90; // rax
  __int64 v91; // rax
  __int16 v92; // r9
  unsigned __int64 v93; // rsi
  unsigned __int16 v94; // dx
  unsigned __int64 v95; // rax
  __int64 *v96; // rax
  unsigned int v97; // ecx
  unsigned __int64 **v98; // rdx
  unsigned __int64 *v99; // r13
  unsigned __int64 *v100; // rbx
  unsigned __int64 v101; // rdx
  unsigned int v102; // eax
  char v103; // cl
  __int64 v104; // rax
  __int64 v105; // rdi
  __int64 *v106; // rdi
  __int64 v107; // rdx
  int v108; // edx
  int v109; // edi
  int v110; // r10d
  __int64 *v111; // r9
  unsigned __int64 v112; // [rsp+8h] [rbp-228h]
  char *v113; // [rsp+18h] [rbp-218h]
  __int16 v115; // [rsp+38h] [rbp-1F8h]
  __int64 v116; // [rsp+40h] [rbp-1F0h]
  __int64 v117; // [rsp+40h] [rbp-1F0h]
  unsigned __int16 v118; // [rsp+40h] [rbp-1F0h]
  __int64 v119; // [rsp+58h] [rbp-1D8h]
  unsigned __int64 v120; // [rsp+58h] [rbp-1D8h]
  __int64 v121; // [rsp+60h] [rbp-1D0h]
  unsigned __int64 v122; // [rsp+60h] [rbp-1D0h]
  __int64 j; // [rsp+60h] [rbp-1D0h]
  __int64 v124; // [rsp+80h] [rbp-1B0h]
  __m128i v125; // [rsp+90h] [rbp-1A0h]
  unsigned __int16 v126; // [rsp+ACh] [rbp-184h] BYREF
  unsigned __int16 v127; // [rsp+AEh] [rbp-182h] BYREF
  __int64 v128; // [rsp+B0h] [rbp-180h] BYREF
  unsigned __int64 v129; // [rsp+B8h] [rbp-178h] BYREF
  __int128 v130; // [rsp+C0h] [rbp-170h] BYREF
  __int128 v131; // [rsp+D0h] [rbp-160h] BYREF
  unsigned __int64 v132; // [rsp+E0h] [rbp-150h] BYREF
  unsigned __int16 v133; // [rsp+E8h] [rbp-148h]
  unsigned __int64 v134; // [rsp+F0h] [rbp-140h] BYREF
  unsigned __int16 v135; // [rsp+F8h] [rbp-138h]
  __int64 v136; // [rsp+100h] [rbp-130h] BYREF
  __int64 v137; // [rsp+108h] [rbp-128h]
  __int64 v138; // [rsp+110h] [rbp-120h]
  _QWORD *v139; // [rsp+120h] [rbp-110h] BYREF
  _QWORD *v140; // [rsp+128h] [rbp-108h]
  __int64 v141; // [rsp+130h] [rbp-100h]
  __int64 v142; // [rsp+140h] [rbp-F0h] BYREF
  __int64 v143; // [rsp+148h] [rbp-E8h]
  __int64 v144; // [rsp+150h] [rbp-E0h]
  unsigned int v145; // [rsp+158h] [rbp-D8h]
  void *v146; // [rsp+160h] [rbp-D0h] BYREF
  __int64 v147; // [rsp+168h] [rbp-C8h]
  _BYTE v148[48]; // [rsp+170h] [rbp-C0h] BYREF
  int v149; // [rsp+1A0h] [rbp-90h]
  __int64 v150; // [rsp+1B0h] [rbp-80h] BYREF
  __int64 v151; // [rsp+1B8h] [rbp-78h]
  __int64 *v152; // [rsp+1C0h] [rbp-70h]
  __int64 *v153; // [rsp+1C8h] [rbp-68h]
  __int64 v154; // [rsp+1D0h] [rbp-60h]
  __int64 *v155; // [rsp+1D8h] [rbp-58h]
  __int64 *v156; // [rsp+1E0h] [rbp-50h]
  __int64 v157; // [rsp+1E8h] [rbp-48h]
  __int64 v158; // [rsp+1F0h] [rbp-40h]
  __int64 *v159; // [rsp+1F8h] [rbp-38h]

  v136 = 0;
  v137 = 0;
  v138 = 0;
  result = (__int64)sub_FE3390(a1, (__int64)&v136);
  v2 = v137;
  v3 = v136;
  if ( v137 == v136 )
    goto LABEL_34;
  v142 = 0;
  v143 = 0;
  v144 = 0;
  v4 = (v137 - v136) >> 3;
  v145 = 0;
  if ( (unsigned __int64)(v137 - v136) > 0x3FFFFFFFFFFFFFF8LL )
    goto LABEL_184;
  v5 = 16 * v4;
  if ( !v4 )
  {
    v8 = 0;
    v119 = 0;
    v130 = 0;
LABEL_9:
    v116 = v8;
    v9 = v8;
    v10 = 0;
    while ( 1 )
    {
      v21 = *(_QWORD *)(v3 + 8 * v10);
      if ( !v145 )
        break;
      v11 = 1;
      v12 = 0;
      v13 = (v145 - 1) & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
      v14 = (_QWORD *)(v143 + 16LL * v13);
      v15 = *v14;
      if ( v21 != *v14 )
      {
        while ( v15 != -4096 )
        {
          if ( !v12 && v15 == -8192 )
            v12 = v14;
          v13 = (v145 - 1) & (v11 + v13);
          v14 = (_QWORD *)(v143 + 16LL * v13);
          v15 = *v14;
          if ( v21 == *v14 )
            goto LABEL_11;
          ++v11;
        }
        if ( !v12 )
          v12 = v14;
        ++v142;
        v23 = v144 + 1;
        if ( 4 * ((int)v144 + 1) < 3 * v145 )
        {
          if ( v145 - HIDWORD(v144) - v23 <= v145 >> 3 )
          {
            sub_FE19E0((__int64)&v142, v145);
            if ( !v145 )
            {
LABEL_190:
              LODWORD(v144) = v144 + 1;
              BUG();
            }
            v51 = 0;
            LODWORD(v52) = (v145 - 1) & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
            v53 = 1;
            v23 = v144 + 1;
            v12 = (__int64 *)(v143 + 16LL * (unsigned int)v52);
            v54 = *v12;
            if ( v21 != *v12 )
            {
              while ( v54 != -4096 )
              {
                if ( !v51 && v54 == -8192 )
                  v51 = v12;
                v52 = (v145 - 1) & ((_DWORD)v52 + v53);
                v12 = (__int64 *)(v143 + 16 * v52);
                v54 = *v12;
                if ( v21 == *v12 )
                  goto LABEL_21;
                ++v53;
              }
              if ( v51 )
                v12 = v51;
            }
          }
          goto LABEL_21;
        }
LABEL_19:
        sub_FE19E0((__int64)&v142, 2 * v145);
        if ( !v145 )
          goto LABEL_190;
        LODWORD(v22) = (v145 - 1) & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
        v23 = v144 + 1;
        v12 = (__int64 *)(v143 + 16LL * (unsigned int)v22);
        v24 = *v12;
        if ( v21 != *v12 )
        {
          v110 = 1;
          v111 = 0;
          while ( v24 != -4096 )
          {
            if ( v24 == -8192 && !v111 )
              v111 = v12;
            v22 = (v145 - 1) & ((_DWORD)v22 + v110);
            v12 = (__int64 *)(v143 + 16 * v22);
            v24 = *v12;
            if ( v21 == *v12 )
              goto LABEL_21;
            ++v110;
          }
          if ( v111 )
            v12 = v111;
        }
LABEL_21:
        LODWORD(v144) = v23;
        if ( *v12 != -4096 )
          --HIDWORD(v144);
        *v12 = v21;
        v16 = (unsigned __int64 *)(v12 + 1);
        v12[1] = 0;
        goto LABEL_12;
      }
LABEL_11:
      v16 = v14 + 1;
LABEL_12:
      *v16 = v10;
      LODWORD(v150) = sub_FDD0F0(a1, v21);
      v124 = sub_FE8AC0(a1, &v150);
      *(_QWORD *)v9 = v124;
      v150 = v124;
      v146 = (void *)v130;
      v17 = WORD4(v130);
      *(_WORD *)(v9 + 8) = v18;
      LOWORD(v134) = v17;
      LOWORD(v139) = v18;
      v19 = sub_FDCA70(
              (unsigned __int64 *)&v146,
              (unsigned __int16 *)&v134,
              (unsigned __int64 *)&v150,
              (unsigned __int16 *)&v139);
      v20 = (unsigned __int64)v146 + v150;
      if ( __CFADD__(v146, v150) )
      {
        ++v19;
        v20 = (v20 >> 1) | 0x8000000000000000LL;
      }
      *(_QWORD *)&v130 = v20;
      WORD4(v130) = v19;
      if ( v19 > 0x3FFF )
      {
        *(_QWORD *)&v130 = -1;
        WORD4(v130) = 0x3FFF;
      }
      v3 = v136;
      ++v10;
      v9 += 16;
      if ( (v137 - v136) >> 3 <= v10 )
      {
        v8 = v116;
        goto LABEL_38;
      }
    }
    ++v142;
    goto LABEL_19;
  }
  v6 = sub_22077B0(16 * v4);
  v7 = v6 + v5;
  v8 = v6;
  v119 = v6 + v5;
  do
  {
    if ( v6 )
    {
      *(_QWORD *)v6 = 0;
      *(_WORD *)(v6 + 8) = 0;
    }
    v6 += 16;
  }
  while ( v6 != v7 );
  v3 = v136;
  v130 = 0;
  if ( v137 != v136 )
    goto LABEL_9;
LABEL_38:
  if ( v8 != v119 )
  {
    v121 = v8;
    do
    {
      v27 = v8;
      v8 += 16;
      sub_FDE760(v27, (__int64)&v130);
    }
    while ( v119 != v8 );
    v8 = v121;
  }
  v139 = 0;
  v140 = 0;
  v141 = 0;
  sub_FDFC80(a1, &v136, (__int64)&v142, &v139);
  v28 = 1.0 / *(double *)&qword_4F8E288[8];
  if ( 1.0 / *(double *)&qword_4F8E288[8] >= 9.223372036854776e18 )
  {
    v146 = (void *)(unsigned int)(int)(v28 - 9.223372036854776e18);
    v146 = (void *)((unsigned __int64)v146 ^ 0x8000000000000000LL);
  }
  else
  {
    v146 = (void *)(unsigned int)(int)v28;
  }
  v29 = (__int64)&v146;
  LOWORD(v147) = 0;
  v150 = 1;
  LOWORD(v151) = 0;
  v125 = _mm_loadu_si128((const __m128i *)sub_FDE760((__int64)&v150, (__int64)&v146));
  v112 = v119 - v8;
  v146 = (void *)v125.m128i_i64[0];
  v122 = (v119 - v8) >> 4;
  LOWORD(v147) = v125.m128i_i16[4];
  v120 = v122 * LODWORD(qword_4F8E368[8]);
  if ( v112 > 0x5555555555555550LL )
LABEL_184:
    sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
  v30 = 24 * v122;
  if ( !v122 )
  {
    v41 = 0;
    v42 = 0;
    v43 = 0;
    v146 = v148;
    HIDWORD(v147) = 6;
    v113 = 0;
LABEL_63:
    LODWORD(v147) = v43;
    goto LABEL_64;
  }
  v31 = (char *)sub_22077B0(24 * v122);
  v33 = &v31[v30];
  v34 = v31;
  v113 = &v31[v30];
  do
  {
    if ( v31 )
    {
      *(_QWORD *)v31 = 0;
      *((_QWORD *)v31 + 1) = 0;
      *((_QWORD *)v31 + 2) = 0;
    }
    v31 += 24;
  }
  while ( v31 != v33 );
  v35 = 0;
  v150 = 0;
  v117 = v8;
  v36 = v34;
  do
  {
    v37 = &v139[3 * v35];
    v38 = (_QWORD *)v37[1];
    v39 = (_QWORD *)*v37;
    if ( (_QWORD *)*v37 != v38 )
    {
      do
      {
        while ( 1 )
        {
          v40 = (__int64)&v36[24 * *v39];
          v29 = *(_QWORD *)(v40 + 8);
          if ( v29 != *(_QWORD *)(v40 + 16) )
            break;
          v39 += 3;
          sub_9CA200(v40, (_BYTE *)v29, &v150);
          if ( v38 == v39 )
            goto LABEL_58;
        }
        if ( v29 )
        {
          *(_QWORD *)v29 = v150;
          v29 = *(_QWORD *)(v40 + 8);
        }
        v29 += 8;
        v39 += 3;
        *(_QWORD *)(v40 + 8) = v29;
      }
      while ( v38 != v39 );
    }
LABEL_58:
    v35 = v150 + 1;
    v150 = v35;
  }
  while ( v122 > v35 );
  v41 = v36;
  v8 = v117;
  v42 = v122;
  v43 = (unsigned int)(v122 + 63) >> 6;
  v146 = v148;
  v147 = 0x600000000LL;
  if ( v43 <= 6 )
  {
    if ( v43 )
    {
      v44 = 8LL * v43;
      if ( v44 )
      {
        v29 = 0;
        memset(v148, 0, v44);
        v42 = v122;
      }
    }
    goto LABEL_63;
  }
  sub_C8D5F0((__int64)&v146, v148, v43, 8u, (unsigned int)v122, v32);
  v29 = 0;
  memset(v146, 0, 8LL * v43);
  LODWORD(v147) = (unsigned int)(v122 + 63) >> 6;
  v42 = v122;
LABEL_64:
  v149 = v42;
  v150 = 0;
  v152 = 0;
  v153 = 0;
  v154 = 0;
  v155 = 0;
  v156 = 0;
  v157 = 0;
  v158 = 0;
  v159 = 0;
  v151 = 8;
  v150 = sub_22077B0(64);
  v45 = (__int64 *)(v150 + ((4 * v151 - 4) & 0xFFFFFFFFFFFFFFF8LL));
  v46 = sub_22077B0(512);
  v155 = v45;
  *v45 = v46;
  v153 = (__int64 *)v46;
  v154 = v46 + 512;
  v159 = v45;
  v157 = v46;
  v158 = v46 + 512;
  v152 = (__int64 *)v46;
  v156 = (__int64 *)v46;
  v134 = 0;
  if ( v122 )
  {
    for ( i = 0; i < v122; v134 = i )
    {
      v48 = v8 + 16 * i;
      v29 = (unsigned int)*(__int16 *)(v48 + 8);
      if ( (int)sub_D788E0(*(_QWORD *)v48, *(_WORD *)(v48 + 8), 0, 0) > 0 )
      {
        v49 = v156;
        if ( v156 == (__int64 *)(v158 - 8) )
        {
          v29 = (__int64)&v134;
          sub_FE0450(&v150, &v134);
          v50 = v134;
        }
        else
        {
          v50 = v134;
          if ( v156 )
          {
            *v156 = v134;
            v49 = v156;
          }
          v156 = v49 + 1;
        }
        *((_QWORD *)v146 + (v50 >> 6)) |= 1LL << v50;
      }
      i = v134 + 1;
    }
    if ( v120 )
    {
      for ( j = 1; ; ++j )
      {
        if ( v156 == v152 )
          goto LABEL_90;
        v72 = *v152;
        v128 = *v152;
        if ( v152 == (__int64 *)(v154 - 8) )
        {
          j_j___libc_free_0(v153, 512);
          LODWORD(v72) = v128;
          v107 = *++v155 + 512;
          v153 = (__int64 *)*v155;
          v154 = v107;
          v152 = v153;
        }
        else
        {
          ++v152;
        }
        *((_QWORD *)v146 + ((unsigned int)v72 >> 6)) &= ~(1LL << v72);
        v73 = v128;
        v133 = 0;
        v131 = 0;
        v132 = 1;
        v74 = &v139[3 * v128];
        v75 = v74[1];
        if ( *v74 != v75 )
        {
          v76 = *v74;
          while ( 1 )
          {
            if ( v73 == *(_QWORD *)v76 )
            {
              v83 = *(_WORD *)(v76 + 16);
              v84 = *(_QWORD *)(v76 + 8);
              v76 += 24;
              v132 = sub_FDCB20(v132, v133, v84, v83);
              v133 = v85;
              if ( v75 == v76 )
                goto LABEL_133;
            }
            else
            {
              v77 = v8 + 16LL * *(_QWORD *)v76;
              v78 = *(_QWORD *)v77;
              v79 = *(_WORD *)(v77 + 8);
              v134 = v78;
              v135 = v79;
              if ( v78 )
              {
                v80 = *(_QWORD *)(v76 + 8);
                if ( v80 )
                {
                  v92 = *(_WORD *)(v76 + 16);
                  if ( v78 > 0xFFFFFFFF || v80 > 0xFFFFFFFF )
                  {
                    v115 = *(_WORD *)(v76 + 16);
                    v118 = v79;
                    v95 = sub_F04140(v78, v80);
                    v92 = v115;
                    v79 = v118;
                    v93 = v95;
                  }
                  else
                  {
                    v93 = v78 * v80;
                    v94 = 0;
                  }
                  v134 = v93;
                  v135 = v94;
                  sub_D78C90((__int64)&v134, (__int16)(v79 + v92));
                  v79 = v135;
                  v78 = v134;
                }
                else
                {
                  v134 = 0;
                  v79 = *(_WORD *)(v76 + 16);
                  v78 = 0;
                  v135 = v79;
                }
              }
              v134 = v78;
              v127 = v79;
              v129 = v131;
              v126 = WORD4(v131);
              v81 = sub_FDCA70(&v129, &v126, &v134, &v127);
              v82 = v129 + v134;
              if ( __CFADD__(v129, v134) )
              {
                ++v81;
                v82 = (v82 >> 1) | 0x8000000000000000LL;
              }
              *(_QWORD *)&v131 = v82;
              WORD4(v131) = v81;
              if ( v81 > 0x3FFF )
              {
                *(_QWORD *)&v131 = -1;
                WORD4(v131) = 0x3FFF;
              }
              v76 += 24;
              if ( v75 == v76 )
              {
LABEL_133:
                v86 = v132;
                v87 = v133;
                goto LABEL_134;
              }
            }
            v73 = v128;
          }
        }
        v87 = 0;
        v86 = 1;
LABEL_134:
        if ( (unsigned int)sub_D788E0(v86, v87, 1u, 0) )
          sub_FDE760((__int64)&v131, (__int64)&v132);
        if ( (int)sub_D788E0(*(_QWORD *)(v8 + 16 * v128), *(_WORD *)(v8 + 16 * v128 + 8), v131, SWORD4(v131)) < 0 )
          v89 = sub_FDCB20(v131, WORD4(v131), *(_QWORD *)(v8 + 16 * v128), *(_WORD *)(v8 + 16 * v128 + 8));
        else
          v89 = sub_FDCB20(*(_QWORD *)(v8 + 16 * v128), *(_WORD *)(v8 + 16 * v128 + 8), v131, SWORD4(v131));
        v29 = (unsigned int)v88;
        if ( (int)sub_D788E0(v89, v88, v125.m128i_u64[0], v125.m128i_i16[4]) <= 0 )
          goto LABEL_139;
        v96 = v156;
        if ( v156 == (__int64 *)(v158 - 8) )
        {
          v29 = (__int64)&v128;
          sub_FE0450(&v150, &v128);
          v97 = v128;
        }
        else
        {
          v97 = v128;
          if ( v156 )
          {
            *v156 = v128;
            v96 = v156;
          }
          v156 = v96 + 1;
        }
        *((_QWORD *)v146 + (v97 >> 6)) |= 1LL << v97;
        v90 = v128;
        v98 = (unsigned __int64 **)&v41[24 * v128];
        v99 = *v98;
        v100 = v98[1];
        if ( *v98 != v100 )
          break;
LABEL_140:
        v91 = v8 + 16 * v90;
        *(_QWORD *)v91 = v131;
        *(_WORD *)(v91 + 8) = WORD4(v131);
        if ( v120 == j )
          goto LABEL_90;
      }
      do
      {
        v101 = *v99;
        v102 = *v99;
        v134 = v101;
        v103 = v101 & 0x3F;
        v104 = 8LL * (v102 >> 6);
        v29 = (__int64)v146 + v104;
        v105 = *(_QWORD *)((char *)v146 + v104);
        if ( !_bittest64(&v105, v101) )
        {
          v106 = v156;
          if ( v156 == (__int64 *)(v158 - 8) )
          {
            sub_FE0450(&v150, &v134);
            v103 = v134 & 0x3F;
            v29 = (__int64)v146 + 8 * ((unsigned int)v134 >> 6);
          }
          else
          {
            if ( v156 )
            {
              *v156 = v101;
              v106 = v156;
              v29 = (__int64)v146 + v104;
            }
            v156 = v106 + 1;
          }
          *(_QWORD *)v29 |= 1LL << v103;
        }
        ++v99;
      }
      while ( v100 != v99 );
LABEL_139:
      v90 = v128;
      goto LABEL_140;
    }
  }
LABEL_90:
  v55 = v150;
  if ( v150 )
  {
    v56 = v155;
    v57 = (unsigned __int64)(v159 + 1);
    if ( v159 + 1 > v155 )
    {
      do
      {
        v58 = *v56++;
        j_j___libc_free_0(v58, 512);
      }
      while ( v57 > (unsigned __int64)v56 );
      v55 = v150;
    }
    v29 = 8 * v151;
    j_j___libc_free_0(v55, 8 * v151);
  }
  if ( v146 != v148 )
    _libc_free(v146, v29);
  for ( k = v41; v113 != k; k += 24 )
  {
    if ( *(_QWORD *)k )
      j_j___libc_free_0(*(_QWORD *)k, *((_QWORD *)k + 2) - *(_QWORD *)k);
  }
  if ( v41 )
    j_j___libc_free_0(v41, v113 - v41);
  v60 = *(_QWORD *)(a1 + 128);
  v61 = *(_QWORD *)(v60 + 80);
  v62 = v60 + 72;
  if ( v62 != v61 )
  {
    while ( 1 )
    {
      v70 = v61 - 24;
      if ( !v61 )
        v70 = 0;
      LODWORD(v67) = sub_FDD0F0(a1, v70);
      if ( (_DWORD)v67 == -1 )
        goto LABEL_108;
      if ( !v145 )
        goto LABEL_113;
      v63 = (v145 - 1) & (((unsigned int)v70 >> 9) ^ ((unsigned int)v70 >> 4));
      v64 = (__int64 *)(v143 + 16LL * v63);
      v65 = *v64;
      if ( v70 != *v64 )
        break;
LABEL_106:
      v66 = *(_QWORD *)(a1 + 8);
      v67 = (unsigned int)v67;
      v68 = v66 + 24LL * (unsigned int)v67;
      if ( v64 == (__int64 *)(v143 + 16LL * v145) )
        goto LABEL_114;
      v69 = v8 + 16 * v64[1];
      *(_QWORD *)v68 = *(_QWORD *)v69;
      *(_WORD *)(v68 + 8) = *(_WORD *)(v69 + 8);
LABEL_108:
      v61 = *(_QWORD *)(v61 + 8);
      if ( v62 == v61 )
        goto LABEL_24;
    }
    v108 = 1;
    while ( v65 != -4096 )
    {
      v109 = v108 + 1;
      v63 = (v145 - 1) & (v108 + v63);
      v64 = (__int64 *)(v143 + 16LL * v63);
      v65 = *v64;
      if ( v70 == *v64 )
        goto LABEL_106;
      v108 = v109;
    }
LABEL_113:
    v66 = *(_QWORD *)(a1 + 8);
    v67 = (unsigned int)v67;
LABEL_114:
    v71 = v66 + 24 * v67;
    *(_QWORD *)v71 = 0;
    *(_WORD *)(v71 + 8) = 0;
    goto LABEL_108;
  }
LABEL_24:
  v25 = v140;
  v26 = v139;
  if ( v140 != v139 )
  {
    do
    {
      if ( *v26 )
        j_j___libc_free_0(*v26, v26[2] - *v26);
      v26 += 3;
    }
    while ( v25 != v26 );
    v26 = v139;
  }
  if ( v26 )
    j_j___libc_free_0(v26, v141 - (_QWORD)v26);
  if ( v8 )
    j_j___libc_free_0(v8, v112);
  result = sub_C7D6A0(v143, 16LL * v145, 8);
  v2 = v136;
LABEL_34:
  if ( v2 )
    return j_j___libc_free_0(v2, v138 - v2);
  return result;
}
