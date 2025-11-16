// Function: sub_1F69540
// Address: 0x1f69540
//
__int64 __fastcall sub_1F69540(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        __m128i a4,
        __m128i a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  unsigned int v10; // r13d
  __int64 v12; // rax
  int v13; // eax
  __int64 *v14; // rcx
  double v15; // xmm4_8
  double v16; // xmm5_8
  __int64 v17; // rax
  _QWORD *v18; // rbx
  _QWORD *v19; // r13
  __int64 v20; // rax
  unsigned __int64 *v21; // rax
  unsigned __int64 *v22; // r12
  __int64 v23; // rax
  double v24; // xmm4_8
  double v25; // xmm5_8
  __int64 v26; // rbx
  unsigned int v27; // esi
  __int64 v28; // rdi
  unsigned int v29; // r13d
  unsigned int v30; // ecx
  _QWORD *v31; // r12
  __int64 v32; // rdx
  unsigned __int64 v33; // rdx
  __int64 *v34; // rcx
  __int64 v35; // r12
  __int64 *v36; // r15
  _BYTE *v37; // rsi
  __int64 v38; // rax
  double v39; // xmm4_8
  double v40; // xmm5_8
  char v41; // dl
  int v42; // r12d
  _QWORD *v43; // rbx
  unsigned int v44; // eax
  __int64 v45; // rdx
  _QWORD *v46; // r13
  __int64 v47; // rax
  unsigned __int64 *v48; // rax
  unsigned __int64 *v49; // r12
  int v50; // eax
  __int64 v51; // rdx
  _QWORD *v52; // rax
  _QWORD *k; // rdx
  __int64 v54; // r12
  __int64 v55; // r14
  __int64 v56; // rbx
  __int64 v57; // rdi
  __int64 v58; // r14
  unsigned __int64 *v59; // r14
  __int64 *v60; // rcx
  int v61; // edx
  __int64 v62; // r14
  __int64 v63; // rbx
  unsigned __int8 v64; // al
  unsigned __int64 v65; // rdi
  char v66; // al
  __int64 v67; // rbx
  __int64 v68; // r12
  double v69; // xmm4_8
  double v70; // xmm5_8
  __int64 v71; // r12
  unsigned __int64 v72; // r14
  char v73; // al
  int v74; // r12d
  __int64 v75; // rax
  __int64 v76; // rdx
  __int64 v77; // r13
  __int64 v78; // r13
  __int64 v79; // r15
  __int64 v80; // r13
  __int64 v81; // rax
  unsigned int *v82; // rax
  __int64 v83; // rax
  __int64 *v84; // rax
  __int64 v85; // rax
  double v86; // xmm4_8
  double v87; // xmm5_8
  char v88; // al
  __int64 v89; // rax
  __int64 v90; // rdx
  __int64 v91; // r13
  __int64 v92; // r13
  __int64 v93; // r15
  __int64 v94; // r13
  __int64 v95; // rax
  __int64 v96; // rax
  unsigned __int64 v97; // rax
  double v98; // xmm4_8
  double v99; // xmm5_8
  __int64 v100; // rdi
  _QWORD *v101; // rdi
  __int64 v102; // rdx
  __int64 v103; // rax
  int v104; // r10d
  _QWORD *v105; // rax
  int v106; // ecx
  unsigned int v107; // ecx
  _QWORD *v108; // rdi
  unsigned int v109; // eax
  int v110; // eax
  unsigned __int64 v111; // rax
  unsigned __int64 v112; // rax
  int v113; // ebx
  __int64 v114; // r12
  _QWORD *v115; // rax
  __int64 v116; // rdx
  _QWORD *j; // rdx
  int v118; // esi
  int v119; // esi
  __int64 v120; // r9
  unsigned int v121; // edx
  __int64 v122; // r11
  int v123; // r8d
  _QWORD *v124; // rdi
  int v125; // edx
  int v126; // edx
  __int64 v127; // r9
  int v128; // edi
  _QWORD *v129; // rsi
  unsigned int v130; // r13d
  __int64 v131; // r8
  int v132; // edx
  int v133; // ebx
  unsigned int v134; // eax
  _QWORD *v135; // rdi
  unsigned __int64 v136; // rax
  unsigned __int64 v137; // rax
  __int64 v138; // rax
  _QWORD *v139; // rax
  __int64 v140; // rdx
  _QWORD *i; // rdx
  _QWORD *v142; // rax
  _QWORD *v143; // rax
  __int64 *v144; // [rsp+0h] [rbp-D0h]
  __int64 v146; // [rsp+10h] [rbp-C0h]
  __int64 *v147; // [rsp+20h] [rbp-B0h]
  __int64 v148; // [rsp+28h] [rbp-A8h]
  __int64 v149; // [rsp+30h] [rbp-A0h]
  __int64 *v150; // [rsp+38h] [rbp-98h]
  __int64 v151; // [rsp+40h] [rbp-90h]
  __int64 v153; // [rsp+50h] [rbp-80h]
  __int64 v154; // [rsp+58h] [rbp-78h]
  __int64 v155; // [rsp+60h] [rbp-70h]
  __int64 v156; // [rsp+60h] [rbp-70h]
  __int64 v157; // [rsp+68h] [rbp-68h]
  __int64 *v158; // [rsp+68h] [rbp-68h]
  __int64 v159; // [rsp+78h] [rbp-58h] BYREF
  __int64 v160; // [rsp+80h] [rbp-50h] BYREF
  __int64 v161; // [rsp+88h] [rbp-48h]
  __int64 v162; // [rsp+90h] [rbp-40h]
  int v163; // [rsp+98h] [rbp-38h]

  v10 = 0;
  if ( (*(_BYTE *)(a2 + 18) & 8) == 0 )
    return v10;
  v12 = sub_15E38F0(a2);
  v13 = sub_14DD7D0(v12);
  *(_DWORD *)(a1 + 156) = v13;
  if ( v13 > 10 )
  {
    if ( v13 != 12 )
      return v10;
  }
  else if ( v13 <= 6 )
  {
    return v10;
  }
  *(_QWORD *)(a1 + 160) = sub_1632FA0(*(_QWORD *)(a2 + 40));
  sub_1AF0CE0(a2, 0, 0, v14, a3, *(double *)a4.m128i_i64, *(double *)a5.m128i_i64, a6, v15, v16, a9, a10);
  sub_14DDFC0((__int64)&v160, a2);
  v17 = *(unsigned int *)(a1 + 192);
  v155 = a1 + 168;
  if ( (_DWORD)v17 )
  {
    v18 = *(_QWORD **)(a1 + 176);
    v19 = &v18[2 * v17];
    do
    {
      if ( *v18 != -8 && *v18 != -16 )
      {
        v20 = v18[1];
        if ( (v20 & 4) != 0 )
        {
          v21 = (unsigned __int64 *)(v20 & 0xFFFFFFFFFFFFFFF8LL);
          v22 = v21;
          if ( v21 )
          {
            if ( (unsigned __int64 *)*v21 != v21 + 2 )
              _libc_free(*v21);
            j_j___libc_free_0(v22, 48);
          }
        }
      }
      v18 += 2;
    }
    while ( v19 != v18 );
  }
  j___libc_free_0(*(_QWORD *)(a1 + 176));
  v23 = v161;
  ++v160;
  ++*(_QWORD *)(a1 + 168);
  *(_QWORD *)(a1 + 176) = v23;
  v161 = 0;
  *(_QWORD *)(a1 + 184) = v162;
  v162 = 0;
  *(_DWORD *)(a1 + 192) = v163;
  v163 = 0;
  j___libc_free_0(0);
  v146 = a2 + 72;
  v157 = *(_QWORD *)(a2 + 80);
  if ( v157 != a2 + 72 )
  {
    while ( 1 )
    {
      v26 = v157 - 24;
      if ( !v157 )
        v26 = 0;
      v27 = *(_DWORD *)(a1 + 192);
      if ( !v27 )
        break;
      v28 = *(_QWORD *)(a1 + 176);
      v29 = ((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4);
      v30 = (v27 - 1) & v29;
      v31 = (_QWORD *)(v28 + 16LL * v30);
      v32 = *v31;
      if ( v26 != *v31 )
      {
        v104 = 1;
        v105 = 0;
        while ( v32 != -8 )
        {
          if ( !v105 && v32 == -16 )
            v105 = v31;
          v30 = (v27 - 1) & (v104 + v30);
          v31 = (_QWORD *)(v28 + 16LL * v30);
          v32 = *v31;
          if ( v26 == *v31 )
            goto LABEL_20;
          ++v104;
        }
        if ( !v105 )
          v105 = v31;
        ++*(_QWORD *)(a1 + 168);
        v106 = *(_DWORD *)(a1 + 184) + 1;
        if ( 4 * v106 < 3 * v27 )
        {
          if ( v27 - *(_DWORD *)(a1 + 188) - v106 <= v27 >> 3 )
          {
            sub_14DDDA0(v155, v27);
            v125 = *(_DWORD *)(a1 + 192);
            if ( !v125 )
            {
LABEL_235:
              ++*(_DWORD *)(a1 + 184);
              BUG();
            }
            v126 = v125 - 1;
            v127 = *(_QWORD *)(a1 + 176);
            v128 = 1;
            v129 = 0;
            v130 = v126 & v29;
            v106 = *(_DWORD *)(a1 + 184) + 1;
            v105 = (_QWORD *)(v127 + 16LL * v130);
            v131 = *v105;
            if ( v26 != *v105 )
            {
              while ( v131 != -8 )
              {
                if ( v131 == -16 && !v129 )
                  v129 = v105;
                v130 = v126 & (v128 + v130);
                v105 = (_QWORD *)(v127 + 16LL * v130);
                v131 = *v105;
                if ( v26 == *v105 )
                  goto LABEL_163;
                ++v128;
              }
              if ( v129 )
                v105 = v129;
            }
          }
          goto LABEL_163;
        }
LABEL_180:
        sub_14DDDA0(v155, 2 * v27);
        v118 = *(_DWORD *)(a1 + 192);
        if ( !v118 )
          goto LABEL_235;
        v119 = v118 - 1;
        v120 = *(_QWORD *)(a1 + 176);
        v121 = v119 & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
        v106 = *(_DWORD *)(a1 + 184) + 1;
        v105 = (_QWORD *)(v120 + 16LL * v121);
        v122 = *v105;
        if ( v26 != *v105 )
        {
          v123 = 1;
          v124 = 0;
          while ( v122 != -8 )
          {
            if ( !v124 && v122 == -16 )
              v124 = v105;
            v121 = v119 & (v123 + v121);
            v105 = (_QWORD *)(v120 + 16LL * v121);
            v122 = *v105;
            if ( v26 == *v105 )
              goto LABEL_163;
            ++v123;
          }
          if ( v124 )
            v105 = v124;
        }
LABEL_163:
        *(_DWORD *)(a1 + 184) = v106;
        if ( *v105 != -8 )
          --*(_DWORD *)(a1 + 188);
        *v105 = v26;
        v105[1] = 0;
        goto LABEL_29;
      }
LABEL_20:
      v33 = v31[1] & 0xFFFFFFFFFFFFFFF8LL;
      if ( (v31[1] & 4) != 0 )
      {
        v34 = *(__int64 **)v33;
        v35 = *(_QWORD *)v33 + 8LL * *(unsigned int *)(v33 + 8);
LABEL_22:
        if ( (__int64 *)v35 != v34 )
        {
          v36 = v34;
          do
          {
            while ( 1 )
            {
              v159 = *v36;
              v38 = sub_1F65D50(a1 + 200, &v159);
              v160 = v26;
              v37 = *(_BYTE **)(v38 + 8);
              if ( v37 != *(_BYTE **)(v38 + 16) )
                break;
              ++v36;
              sub_15D0700(v38, v37, &v160);
              if ( (__int64 *)v35 == v36 )
                goto LABEL_29;
            }
            if ( v37 )
            {
              *(_QWORD *)v37 = v26;
              v37 = *(_BYTE **)(v38 + 8);
            }
            ++v36;
            *(_QWORD *)(v38 + 8) = v37 + 8;
          }
          while ( (__int64 *)v35 != v36 );
        }
        goto LABEL_29;
      }
      v34 = v31 + 1;
      v35 = (__int64)(v31 + 2);
      if ( v33 )
        goto LABEL_22;
LABEL_29:
      v157 = *(_QWORD *)(v157 + 8);
      if ( v146 == v157 )
        goto LABEL_30;
    }
    ++*(_QWORD *)(a1 + 168);
    goto LABEL_180;
  }
LABEL_30:
  sub_1F67100(a1, a2, a3, *(double *)a4.m128i_i64, *(double *)a5.m128i_i64, a6, v24, v25, a9, a10);
  if ( !byte_4FCE6A0 )
  {
    v41 = 1;
    if ( !*(_BYTE *)(a1 + 153) )
      v41 = byte_4FCE4E0;
    sub_1F66030(a1, a2, v41, a3, *(double *)a4.m128i_i64, *(double *)a5.m128i_i64, a6, v39, v40, a9, a10);
  }
  if ( !byte_4FCE5C0 )
  {
    v60 = *(__int64 **)(a1 + 240);
    v144 = v60;
    v147 = *(__int64 **)(a1 + 232);
    if ( v147 == v60 )
      goto LABEL_94;
    while ( 1 )
    {
      v153 = sub_157ED20(*v147);
      v61 = *(unsigned __int8 *)(v153 + 16);
      if ( (unsigned int)(v61 - 73) > 1 )
      {
        v148 = 0;
        v153 = 0;
        v149 = 0;
      }
      else if ( (_BYTE)v61 == 74 )
      {
        v148 = v153;
        v149 = 0;
      }
      else
      {
        v103 = 0;
        if ( (_BYTE)v61 == 73 )
          v103 = v153;
        v148 = 0;
        v149 = v103;
      }
      v60 = (__int64 *)v147[2];
      v150 = v60;
      v158 = (__int64 *)v147[1];
      if ( v158 == v60 )
        goto LABEL_93;
      do
      {
        v62 = *v158;
        v63 = *(_QWORD *)(*v158 + 48);
        v154 = *v158 + 40;
        if ( v63 == v154 )
          goto LABEL_87;
        v151 = *v158;
        while ( 1 )
        {
          if ( !v63 )
            BUG();
          v156 = v63 - 24;
          v64 = *(_BYTE *)(v63 - 8);
          if ( v64 <= 0x17u )
            goto LABEL_85;
          if ( v64 == 78 )
          {
            v71 = v156 | 4;
          }
          else
          {
            if ( v64 != 29 )
              goto LABEL_85;
            v71 = v156 & 0xFFFFFFFFFFFFFFFBLL;
          }
          v72 = v71 & 0xFFFFFFFFFFFFFFF8LL;
          if ( (v71 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
            goto LABEL_85;
          v73 = *(_BYTE *)(v72 + 23);
          v74 = (v71 >> 2) & 1;
          if ( v74 )
          {
            if ( v73 < 0 )
            {
              v75 = sub_1648A40(v72);
              v77 = v75 + v76;
              if ( *(char *)(v72 + 23) < 0 )
                v77 -= sub_1648A40(v72);
              v78 = v77 >> 4;
              if ( (_DWORD)v78 )
              {
                v79 = 0;
                v80 = 16LL * (unsigned int)v78;
                while ( 1 )
                {
                  v81 = 0;
                  if ( *(char *)(v72 + 23) < 0 )
                    v81 = sub_1648A40(v72);
                  v82 = (unsigned int *)(v79 + v81);
                  if ( *(_DWORD *)(*(_QWORD *)v82 + 8LL) == 1 )
                    break;
                  v79 += 16;
                  if ( v80 == v79 )
                    goto LABEL_120;
                }
LABEL_109:
                v83 = *(_QWORD *)(v72 + 24 * (v82[2] - (unsigned __int64)(*(_DWORD *)(v72 + 20) & 0xFFFFFFF)));
                goto LABEL_110;
              }
            }
          }
          else if ( v73 < 0 )
          {
            v89 = sub_1648A40(v72);
            v91 = v89 + v90;
            if ( *(char *)(v72 + 23) < 0 )
              v91 -= sub_1648A40(v72);
            v92 = v91 >> 4;
            if ( (_DWORD)v92 )
            {
              v93 = 0;
              v94 = 16LL * (unsigned int)v92;
              do
              {
                v95 = 0;
                if ( *(char *)(v72 + 23) < 0 )
                  v95 = sub_1648A40(v72);
                v82 = (unsigned int *)(v93 + v95);
                if ( *(_DWORD *)(*(_QWORD *)v82 + 8LL) == 1 )
                  goto LABEL_109;
                v93 += 16;
              }
              while ( v94 != v93 );
            }
          }
LABEL_120:
          v83 = 0;
LABEL_110:
          if ( v153 == v83 )
            goto LABEL_85;
          v84 = (__int64 *)(v72 - 72);
          if ( (_BYTE)v74 )
            v84 = (__int64 *)(v72 - 24);
          v85 = sub_1649C60(*v84);
          if ( *(_BYTE *)(v85 + 16) )
            break;
          if ( (*(_BYTE *)(v85 + 33) & 0x20) == 0 )
          {
            if ( !(_BYTE)v74 )
              goto LABEL_135;
LABEL_116:
            v88 = *(_BYTE *)(*(_QWORD *)(v72 - 24) + 16LL);
            goto LABEL_117;
          }
          v101 = (_QWORD *)(v72 + 56);
          if ( !(_BYTE)v74 )
          {
            if ( !(unsigned __int8)sub_1560260(v101, -1, 30) )
            {
              v96 = *(_QWORD *)(v72 - 72);
              if ( *(_BYTE *)(v96 + 16) || (v160 = *(_QWORD *)(v96 + 112), !(unsigned __int8)sub_1560260(&v160, -1, 30)) )
              {
LABEL_135:
                v62 = v151;
                goto LABEL_136;
              }
            }
            goto LABEL_85;
          }
          if ( (unsigned __int8)sub_1560260(v101, -1, 30) )
            goto LABEL_85;
          v102 = *(_QWORD *)(v72 - 24);
          v88 = *(_BYTE *)(v102 + 16);
          if ( !v88 )
          {
            v160 = *(_QWORD *)(v102 + 112);
            if ( (unsigned __int8)sub_1560260(&v160, -1, 30) )
              goto LABEL_85;
            goto LABEL_116;
          }
LABEL_117:
          if ( v88 != 20 )
          {
            v62 = v151;
LABEL_119:
            sub_1AEE6A0(v156, 0, 0, 0, a3, *(double *)a4.m128i_i64, *(double *)a5.m128i_i64, a6, v86, v87, a9, a10);
            goto LABEL_87;
          }
LABEL_85:
          v63 = *(_QWORD *)(v63 + 8);
          if ( v154 == v63 )
          {
            v62 = v151;
            goto LABEL_87;
          }
        }
        v62 = v151;
        if ( (_BYTE)v74 )
          goto LABEL_119;
LABEL_136:
        sub_1AF0970(v62, 0, a3, *(double *)a4.m128i_i64, *(double *)a5.m128i_i64, a6, v86, v87, a9, a10);
        v97 = *(_QWORD *)(sub_157EBA0(v62) + 24) & 0xFFFFFFFFFFFFFFF8LL;
        v100 = v97 - 24;
        if ( !v97 )
          v100 = 0;
        sub_1AEE6A0(v100, 0, 0, 0, a3, *(double *)a4.m128i_i64, *(double *)a5.m128i_i64, a6, v98, v99, a9, a10);
LABEL_87:
        v65 = sub_157EBA0(v62);
        v66 = *(_BYTE *)(v65 + 16);
        if ( v66 == 25 && v153 )
          goto LABEL_91;
        if ( v66 != 33 )
        {
          if ( v66 != 32 )
          {
            if ( v66 == 29 && *(_DWORD *)(a1 + 156) == 9 && v149 )
              sub_1AF0970(v62, 0, a3, *(double *)a4.m128i_i64, *(double *)a5.m128i_i64, a6, v39, v40, a9, a10);
            goto LABEL_92;
          }
          if ( *(_QWORD *)(v65 - 24LL * (*(_DWORD *)(v65 + 20) & 0xFFFFFFF)) == v149 )
            goto LABEL_92;
LABEL_91:
          sub_1AEE6A0(v65, 0, 0, 0, a3, *(double *)a4.m128i_i64, *(double *)a5.m128i_i64, a6, v39, v40, a9, a10);
          goto LABEL_92;
        }
        if ( *(_QWORD *)(v65 - 48) != v148 )
          goto LABEL_91;
LABEL_92:
        ++v158;
      }
      while ( v150 != v158 );
LABEL_93:
      v147 += 4;
      if ( v144 == v147 )
      {
LABEL_94:
        v67 = *(_QWORD *)(a2 + 80);
        while ( v146 != v67 )
        {
          v68 = v67;
          v67 = *(_QWORD *)(v67 + 8);
          v68 -= 24;
          sub_1AF47C0(v68, 0, a3, *(double *)a4.m128i_i64, *(double *)a5.m128i_i64, a6, v39, v40, a9, a10);
          sub_1AEE9C0(v68, 1u, 0, 0);
          sub_1AA7EA0(v68, 0, 0, 0, 0, a3, a4, a5, a6, v69, v70, a9, a10);
        }
        sub_1AF0CE0(a2, 0, 0, v60, a3, *(double *)a4.m128i_i64, *(double *)a5.m128i_i64, a6, v39, v40, a9, a10);
        break;
      }
    }
  }
  v42 = *(_DWORD *)(a1 + 184);
  ++*(_QWORD *)(a1 + 168);
  if ( v42 || *(_DWORD *)(a1 + 188) )
  {
    v43 = *(_QWORD **)(a1 + 176);
    v44 = 4 * v42;
    v45 = *(unsigned int *)(a1 + 192);
    v46 = &v43[2 * v45];
    if ( (unsigned int)(4 * v42) < 0x40 )
      v44 = 64;
    if ( v44 >= (unsigned int)v45 )
    {
      for ( ; v46 != v43; v43 += 2 )
      {
        if ( *v43 != -8 )
        {
          if ( *v43 != -16 )
          {
            v47 = v43[1];
            if ( (v47 & 4) != 0 )
            {
              v48 = (unsigned __int64 *)(v47 & 0xFFFFFFFFFFFFFFF8LL);
              v49 = v48;
              if ( v48 )
              {
                if ( (unsigned __int64 *)*v48 != v48 + 2 )
                  _libc_free(*v48);
                j_j___libc_free_0(v49, 48);
              }
            }
          }
          *v43 = -8;
        }
      }
      goto LABEL_50;
    }
    do
    {
      if ( *v43 != -8 && *v43 != -16 )
      {
        v58 = v43[1];
        if ( (v58 & 4) != 0 )
        {
          v59 = (unsigned __int64 *)(v58 & 0xFFFFFFFFFFFFFFF8LL);
          if ( v59 )
          {
            if ( (unsigned __int64 *)*v59 != v59 + 2 )
              _libc_free(*v59);
            j_j___libc_free_0(v59, 48);
          }
        }
      }
      v43 += 2;
    }
    while ( v46 != v43 );
    v132 = *(_DWORD *)(a1 + 192);
    if ( v42 )
    {
      v133 = 64;
      if ( v42 != 1 )
      {
        _BitScanReverse(&v134, v42 - 1);
        v133 = 1 << (33 - (v134 ^ 0x1F));
        if ( v133 < 64 )
          v133 = 64;
      }
      v135 = *(_QWORD **)(a1 + 176);
      if ( v133 == v132 )
      {
        *(_QWORD *)(a1 + 184) = 0;
        v142 = &v135[2 * (unsigned int)v133];
        do
        {
          if ( v135 )
            *v135 = -8;
          v135 += 2;
        }
        while ( v142 != v135 );
      }
      else
      {
        j___libc_free_0(v135);
        v136 = (4 * v133 / 3u + 1) | ((unsigned __int64)(4 * v133 / 3u + 1) >> 1);
        v137 = (((v136 >> 2) | v136) >> 4) | (v136 >> 2) | v136;
        v138 = ((((v137 >> 8) | v137) >> 16) | (v137 >> 8) | v137) + 1;
        *(_DWORD *)(a1 + 192) = v138;
        v139 = (_QWORD *)sub_22077B0(16 * v138);
        v140 = *(unsigned int *)(a1 + 192);
        *(_QWORD *)(a1 + 184) = 0;
        *(_QWORD *)(a1 + 176) = v139;
        for ( i = &v139[2 * v140]; i != v139; v139 += 2 )
        {
          if ( v139 )
            *v139 = -8;
        }
      }
    }
    else
    {
      if ( !v132 )
      {
LABEL_50:
        *(_QWORD *)(a1 + 184) = 0;
        goto LABEL_51;
      }
      j___libc_free_0(*(_QWORD *)(a1 + 176));
      *(_QWORD *)(a1 + 176) = 0;
      *(_QWORD *)(a1 + 184) = 0;
      *(_DWORD *)(a1 + 192) = 0;
    }
  }
LABEL_51:
  v50 = *(_DWORD *)(a1 + 216);
  ++*(_QWORD *)(a1 + 200);
  if ( v50 )
  {
    v51 = *(unsigned int *)(a1 + 224);
    v107 = 4 * v50;
    if ( (unsigned int)(4 * v50) < 0x40 )
      v107 = 64;
    if ( (unsigned int)v51 <= v107 )
      goto LABEL_54;
    v108 = *(_QWORD **)(a1 + 208);
    v109 = v50 - 1;
    if ( v109 )
    {
      _BitScanReverse(&v109, v109);
      v110 = 1 << (33 - (v109 ^ 0x1F));
      if ( v110 < 64 )
        v110 = 64;
      if ( (_DWORD)v51 == v110 )
      {
        *(_QWORD *)(a1 + 216) = 0;
        v143 = &v108[2 * (unsigned int)v51];
        do
        {
          if ( v108 )
            *v108 = -8;
          v108 += 2;
        }
        while ( v143 != v108 );
        goto LABEL_57;
      }
      v111 = (4 * v110 / 3u + 1) | ((unsigned __int64)(4 * v110 / 3u + 1) >> 1);
      v112 = ((v111 | (v111 >> 2)) >> 4)
           | v111
           | (v111 >> 2)
           | ((((v111 | (v111 >> 2)) >> 4) | v111 | (v111 >> 2)) >> 8);
      v113 = (v112 | (v112 >> 16)) + 1;
      v114 = 16 * ((v112 | (v112 >> 16)) + 1);
    }
    else
    {
      v114 = 2048;
      v113 = 128;
    }
    j___libc_free_0(v108);
    *(_DWORD *)(a1 + 224) = v113;
    v115 = (_QWORD *)sub_22077B0(v114);
    v116 = *(unsigned int *)(a1 + 224);
    *(_QWORD *)(a1 + 216) = 0;
    *(_QWORD *)(a1 + 208) = v115;
    for ( j = &v115[2 * v116]; j != v115; v115 += 2 )
    {
      if ( v115 )
        *v115 = -8;
    }
  }
  else if ( *(_DWORD *)(a1 + 220) )
  {
    v51 = *(unsigned int *)(a1 + 224);
    if ( (unsigned int)v51 <= 0x40 )
    {
LABEL_54:
      v52 = *(_QWORD **)(a1 + 208);
      for ( k = &v52[2 * v51]; k != v52; v52 += 2 )
        *v52 = -8;
      *(_QWORD *)(a1 + 216) = 0;
      goto LABEL_57;
    }
    j___libc_free_0(*(_QWORD *)(a1 + 208));
    *(_QWORD *)(a1 + 208) = 0;
    *(_QWORD *)(a1 + 216) = 0;
    *(_DWORD *)(a1 + 224) = 0;
  }
LABEL_57:
  v10 = 1;
  v54 = *(_QWORD *)(a1 + 232);
  v55 = *(_QWORD *)(a1 + 240);
  if ( v54 != v55 )
  {
    v56 = *(_QWORD *)(a1 + 232);
    do
    {
      v57 = *(_QWORD *)(v56 + 8);
      if ( v57 )
        j_j___libc_free_0(v57, *(_QWORD *)(v56 + 24) - v57);
      v56 += 32;
    }
    while ( v55 != v56 );
    v10 = 1;
    *(_QWORD *)(a1 + 240) = v54;
  }
  return v10;
}
