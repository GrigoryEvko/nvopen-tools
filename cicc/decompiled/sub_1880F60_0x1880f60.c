// Function: sub_1880F60
// Address: 0x1880f60
//
__int64 *__fastcall sub_1880F60(
        __int64 a1,
        __int64 *a2,
        __int64 a3,
        __int64 ***a4,
        __int64 a5,
        __m128 a6,
        double a7,
        double a8,
        double a9,
        double a10,
        double a11,
        double a12,
        __m128 a13)
{
  __int64 *result; // rax
  __int64 v15; // rbx
  __int64 v16; // r8
  __int64 v17; // rdi
  __int64 *v18; // rax
  __int64 v19; // rdi
  __int64 v20; // rax
  __int64 v21; // rdi
  __int64 v22; // rax
  __int64 v23; // rdi
  __int64 v24; // rax
  unsigned __int64 v25; // r13
  __int64 v26; // r12
  __int64 v27; // r14
  _QWORD *v28; // rdx
  __int64 v29; // rsi
  __int64 v30; // rax
  __int64 v31; // rsi
  __int64 v32; // rdx
  __int64 v33; // rax
  __int64 v34; // rdx
  __int64 v35; // rax
  _QWORD *v36; // r15
  _QWORD *v37; // rdi
  unsigned int v38; // esi
  __int64 v39; // r13
  __int64 v40; // rdi
  unsigned int v41; // r9d
  __int64 v42; // r8
  unsigned int v43; // eax
  __int64 v44; // rcx
  __int64 v45; // rax
  __int64 *v46; // rax
  __int64 v47; // rdx
  __int64 v48; // rdi
  __int64 *v49; // rdx
  __int64 *v50; // rsi
  __int64 v51; // rax
  __int64 v52; // r14
  _QWORD *v53; // r13
  __int64 **v54; // r12
  __int64 v55; // r15
  unsigned __int64 v56; // rbx
  size_t v57; // rdx
  int v58; // eax
  __int64 v59; // rbx
  size_t v60; // r15
  int v61; // eax
  __int64 v62; // r14
  __int64 *v63; // r13
  __int64 *j; // r12
  __int64 v65; // rsi
  double v66; // xmm4_8
  double v67; // xmm5_8
  _QWORD *v68; // r15
  _QWORD *v69; // r10
  int v70; // r11d
  _BYTE *v71; // rdx
  int v72; // eax
  int v73; // eax
  int *v74; // r12
  int *v75; // rdi
  __int64 *v76; // rdx
  __int64 *v77; // r15
  __int64 v78; // rax
  __int64 *v79; // r12
  __int64 v80; // rbx
  __int64 v81; // r13
  __int64 v82; // r10
  __int64 v83; // rdx
  __int64 v84; // rax
  _QWORD *v85; // r14
  char *v86; // r14
  __int64 v87; // rax
  __int64 v88; // rdi
  __int64 v89; // r12
  __int64 v90; // rdi
  __int64 v91; // rsi
  __int64 v92; // rax
  _QWORD *v93; // rax
  unsigned __int64 v94; // rdi
  _QWORD *v95; // rax
  _QWORD *v96; // rax
  int v97; // edx
  unsigned __int64 v98; // rax
  __int64 v99; // rdx
  unsigned __int64 v100; // rax
  unsigned __int64 v101; // rcx
  bool v102; // cf
  unsigned __int64 v103; // rax
  __int64 v104; // rax
  __int64 v105; // r15
  __int64 i; // r14
  __int64 v107; // rax
  __int64 v108; // rdx
  int v109; // esi
  __int64 v110; // r13
  __int64 v111; // rdi
  __int64 v112; // rax
  __int64 v113; // rax
  __int64 v114; // rax
  _QWORD *v115; // rax
  int v116; // r14d
  unsigned int v117; // r11d
  __int64 v118; // rax
  __int64 v119; // r14
  __int64 v120; // [rsp+8h] [rbp-1D8h]
  __int64 v121; // [rsp+8h] [rbp-1D8h]
  __int64 v122; // [rsp+10h] [rbp-1D0h]
  __int64 v123; // [rsp+10h] [rbp-1D0h]
  __int64 v124; // [rsp+18h] [rbp-1C8h]
  __int64 v125; // [rsp+20h] [rbp-1C0h]
  __int64 *v126; // [rsp+28h] [rbp-1B8h]
  __int64 v127; // [rsp+30h] [rbp-1B0h]
  __int64 *v129; // [rsp+58h] [rbp-188h]
  _QWORD *v130; // [rsp+70h] [rbp-170h]
  __int64 v131; // [rsp+70h] [rbp-170h]
  __int64 v132; // [rsp+70h] [rbp-170h]
  _QWORD *v133; // [rsp+78h] [rbp-168h]
  __int64 v134; // [rsp+78h] [rbp-168h]
  __int64 v135; // [rsp+78h] [rbp-168h]
  __int64 v136; // [rsp+80h] [rbp-160h] BYREF
  __m128i *p_s2; // [rsp+88h] [rbp-158h] BYREF
  __int64 *v138; // [rsp+90h] [rbp-150h] BYREF
  __int64 v139; // [rsp+98h] [rbp-148h]
  void *s2; // [rsp+A0h] [rbp-140h] BYREF
  __int64 **v141; // [rsp+A8h] [rbp-138h]
  _QWORD v142[2]; // [rsp+B0h] [rbp-130h] BYREF
  char v143[8]; // [rsp+C0h] [rbp-120h] BYREF
  char v144; // [rsp+C8h] [rbp-118h] BYREF
  int *v145; // [rsp+D0h] [rbp-110h]
  char *v146; // [rsp+D8h] [rbp-108h]
  __int64 v147; // [rsp+E8h] [rbp-F8h]
  __int64 v148; // [rsp+F0h] [rbp-F0h]
  unsigned __int64 v149; // [rsp+F8h] [rbp-E8h]
  unsigned int v150; // [rsp+100h] [rbp-E0h]
  unsigned __int64 *v151; // [rsp+110h] [rbp-D0h] BYREF
  __int64 v152; // [rsp+118h] [rbp-C8h]
  unsigned __int64 v153; // [rsp+120h] [rbp-C0h] BYREF
  unsigned __int64 v154; // [rsp+128h] [rbp-B8h]
  _QWORD *v155; // [rsp+130h] [rbp-B0h]
  __int64 v156; // [rsp+138h] [rbp-A8h]
  unsigned __int64 v157; // [rsp+140h] [rbp-A0h]
  unsigned __int64 v158; // [rsp+1A0h] [rbp-40h]
  char *v159; // [rsp+1A8h] [rbp-38h]

  v127 = sub_15A4510(a4, *(__int64 ***)(a1 + 56), 0);
  result = &a2[a3];
  v126 = result;
  if ( result == a2 )
    return result;
  v129 = a2;
  v15 = a1;
  do
  {
    v158 = -1;
    v159 = 0;
    v16 = *v129;
    v151 = &v153;
    v152 = 0x1000000000LL;
    v136 = v16;
    if ( *(_DWORD *)(a5 + 16) )
    {
      v76 = *(__int64 **)(a5 + 8);
      v77 = &v76[2 * *(unsigned int *)(a5 + 24)];
      if ( v76 != v77 )
      {
        while ( 1 )
        {
          v78 = *v76;
          v79 = v76;
          if ( *v76 != -16 && v78 != -8 )
            break;
          v76 += 2;
          if ( v77 == v76 )
            goto LABEL_4;
        }
        if ( v77 != v76 )
        {
          v132 = v15;
          v80 = v16;
          do
          {
            v81 = v78 + 24;
            v82 = v78 + 24 + 8LL * *(_QWORD *)(v78 + 8);
            if ( v78 + 24 != v82 )
            {
              do
              {
                while ( 1 )
                {
                  v83 = *(unsigned int *)(*(_QWORD *)v81 + 8LL);
                  if ( v80 == *(_QWORD *)(*(_QWORD *)v81 + 8 * (1 - v83)) )
                    break;
                  v81 += 8;
                  if ( v82 == v81 )
                    goto LABEL_98;
                }
                v84 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)v81 - 8 * v83) + 136LL);
                v85 = *(_QWORD **)(v84 + 24);
                if ( *(_DWORD *)(v84 + 32) > 0x40u )
                  v85 = (_QWORD *)*v85;
                v86 = (char *)v85 + v79[1];
                if ( (unsigned __int64)v86 < v158 )
                  v158 = (unsigned __int64)v86;
                if ( v86 > v159 )
                  v159 = v86;
                v87 = (unsigned int)v152;
                if ( (unsigned int)v152 >= HIDWORD(v152) )
                {
                  v135 = v82;
                  sub_16CD150((__int64)&v151, &v153, 0, 8, v16, 1);
                  v87 = (unsigned int)v152;
                  v82 = v135;
                }
                v81 += 8;
                v151[v87] = (unsigned __int64)v86;
                LODWORD(v152) = v152 + 1;
              }
              while ( v82 != v81 );
            }
LABEL_98:
            v79 += 2;
            if ( v79 == v77 )
              break;
            while ( 1 )
            {
              v78 = *v79;
              if ( *v79 != -8 && v78 != -16 )
                break;
              v79 += 2;
              if ( v77 == v79 )
                goto LABEL_102;
            }
          }
          while ( v77 != v79 );
LABEL_102:
          v15 = v132;
        }
      }
    }
LABEL_4:
    sub_18807D0((__int64)v143, (__int64)&v151);
    if ( v151 != &v153 )
      _libc_free((unsigned __int64)v151);
    v17 = *(_QWORD *)(v15 + 96);
    LODWORD(v151) = 0;
    v18 = (__int64 *)sub_159C470(v17, v148, 0);
    v19 = *(_QWORD *)(v15 + 48);
    BYTE4(s2) = 0;
    v138 = v18;
    v20 = sub_15A2E80(v19, v127, &v138, 1u, 0, (__int64)&s2, 0);
    v21 = *(_QWORD *)(v15 + 48);
    v152 = v20;
    v22 = sub_159C470(v21, v150, 0);
    v23 = *(_QWORD *)(v15 + 96);
    v153 = v22;
    v24 = sub_159C470(v23, v149 - 1, 0);
    v25 = v149;
    v154 = v24;
    if ( v149 != v147 )
    {
      if ( v149 <= 0x40 )
      {
        v88 = (__int64)v146;
        LODWORD(v151) = 2;
        if ( v146 == &v144 )
          goto LABEL_115;
        v89 = 0;
        do
        {
          v89 |= 1LL << *(_QWORD *)(v88 + 32);
          v88 = sub_220EF30(v88);
        }
        while ( (char *)v88 != &v144 );
        if ( !v89 )
        {
LABEL_115:
          LODWORD(v151) = 0;
          v26 = 0;
        }
        else
        {
          if ( v25 > 0x20 )
            v90 = *(_QWORD *)(v15 + 88);
          else
            v90 = *(_QWORD *)(v15 + 72);
          v91 = v89;
          v26 = 0;
          v157 = sub_159C470(v90, v91, 0);
        }
        goto LABEL_29;
      }
      LODWORD(v151) = 1;
      LOWORD(v142[0]) = 257;
      v133 = sub_1648A60(88, 1u);
      if ( v133 )
        sub_15E51E0((__int64)v133, *(_QWORD *)v15, *(_QWORD *)(v15 + 48), 1, 8, 0, (__int64)&s2, 0, 0, 0, 0);
      LOWORD(v142[0]) = 257;
      v130 = sub_1648A60(88, 1u);
      if ( v130 )
        sub_15E51E0((__int64)v130, *(_QWORD *)v15, *(_QWORD *)(v15 + 48), 1, 8, 0, (__int64)&s2, 0, 0, 0, 0);
      v26 = *(_QWORD *)(v15 + 152);
      if ( v26 != *(_QWORD *)(v15 + 160) )
      {
        if ( v26 )
        {
          a6 = 0;
          *(_QWORD *)(v26 + 40) = 0;
          *(_OWORD *)(v26 + 16) = 0;
          *(_QWORD *)(v26 + 32) = v26 + 8;
          *(_QWORD *)(v26 + 24) = v26 + 8;
          *(_OWORD *)v26 = 0;
          *(_OWORD *)(v26 + 48) = 0;
          *(_OWORD *)(v26 + 64) = 0;
          v26 = *(_QWORD *)(v15 + 152);
        }
        v27 = v26 + 80;
        *(_QWORD *)(v15 + 152) = v26 + 80;
LABEL_16:
        if ( (char *)(v27 - 80) != v143 )
        {
          v28 = *(_QWORD **)(v27 - 64);
          s2 = v28;
          v29 = *(_QWORD *)(v27 - 48);
          v142[0] = v27 - 80;
          v141 = (__int64 **)v29;
          if ( v28 )
          {
            v28[1] = 0;
            if ( *(_QWORD *)(v29 + 16) )
              v141 = *(__int64 ***)(v29 + 16);
          }
          else
          {
            v141 = 0;
          }
          *(_QWORD *)(v27 - 64) = 0;
          *(_QWORD *)(v27 - 56) = v27 - 72;
          *(_QWORD *)(v27 - 48) = v27 - 72;
          *(_QWORD *)(v27 - 40) = 0;
          if ( v145 )
          {
            v30 = sub_18734C0(v145, v27 - 72, &s2);
            v31 = v30;
            do
            {
              v32 = v30;
              v30 = *(_QWORD *)(v30 + 16);
            }
            while ( v30 );
            *(_QWORD *)(v27 - 56) = v32;
            v33 = v31;
            do
            {
              v34 = v33;
              v33 = *(_QWORD *)(v33 + 24);
            }
            while ( v33 );
            *(_QWORD *)(v27 - 48) = v34;
            v35 = v147;
            *(_QWORD *)(v27 - 64) = v31;
            *(_QWORD *)(v27 - 40) = v35;
          }
          v36 = s2;
          if ( s2 )
          {
            do
            {
              sub_1876060(v36[3]);
              v37 = v36;
              v36 = (_QWORD *)v36[2];
              j_j___libc_free_0(v37, 40);
            }
            while ( v36 );
          }
        }
        *(_QWORD *)(v27 - 32) = v149;
        *(_QWORD *)(v27 - 16) = v130;
        *(_QWORD *)(v27 - 24) = v133;
        v155 = v133;
        v156 = *(_QWORD *)(v27 - 16);
LABEL_29:
        v38 = *(_DWORD *)(v15 + 136);
        v39 = v15 + 112;
        if ( !v38 )
          goto LABEL_75;
        goto LABEL_30;
      }
      v99 = v26 - *(_QWORD *)(v15 + 144);
      v124 = *(_QWORD *)(v15 + 144);
      v100 = 0xCCCCCCCCCCCCCCCDLL * (v99 >> 4);
      if ( v100 == 0x199999999999999LL )
        sub_4262D8((__int64)"vector::_M_realloc_insert");
      v101 = 1;
      if ( v100 )
        v101 = 0xCCCCCCCCCCCCCCCDLL * (v99 >> 4);
      v102 = __CFADD__(v101, v100);
      v103 = v101 - 0x3333333333333333LL * (v99 >> 4);
      if ( v102 )
      {
        v119 = 0x7FFFFFFFFFFFFFD0LL;
      }
      else
      {
        if ( !v103 )
        {
          v123 = 0;
          v27 = 80;
          v104 = v26 - *(_QWORD *)(v15 + 144);
          v125 = 0;
          if ( !v99 )
          {
LABEL_139:
            v105 = v124;
            if ( v26 == v124 )
            {
              v26 = v125;
            }
            else
            {
              for ( i = v125; ; i += 80 )
              {
                if ( i )
                {
                  v107 = *(_QWORD *)(v105 + 16);
                  v108 = i + 8;
                  if ( v107 )
                  {
                    v109 = *(_DWORD *)(v105 + 8);
                    *(_QWORD *)(i + 16) = v107;
                    *(_DWORD *)(i + 8) = v109;
                    *(_QWORD *)(i + 24) = *(_QWORD *)(v105 + 24);
                    *(_QWORD *)(i + 32) = *(_QWORD *)(v105 + 32);
                    *(_QWORD *)(v107 + 8) = v108;
                    *(_QWORD *)(i + 40) = *(_QWORD *)(v105 + 40);
                    *(_QWORD *)(v105 + 16) = 0;
                    *(_QWORD *)(v105 + 24) = v105 + 8;
                    *(_QWORD *)(v105 + 32) = v105 + 8;
                    *(_QWORD *)(v105 + 40) = 0;
                  }
                  else
                  {
                    *(_DWORD *)(i + 8) = 0;
                    *(_QWORD *)(i + 16) = 0;
                    *(_QWORD *)(i + 24) = v108;
                    *(_QWORD *)(i + 32) = v108;
                    *(_QWORD *)(i + 40) = 0;
                  }
                  *(_QWORD *)(i + 48) = *(_QWORD *)(v105 + 48);
                  *(_QWORD *)(i + 56) = *(_QWORD *)(v105 + 56);
                  *(_QWORD *)(i + 64) = *(_QWORD *)(v105 + 64);
                  *(_QWORD *)(i + 72) = *(_QWORD *)(v105 + 72);
                }
                v110 = *(_QWORD *)(v105 + 16);
                while ( v110 )
                {
                  sub_1876060(*(_QWORD *)(v110 + 24));
                  v111 = v110;
                  v110 = *(_QWORD *)(v110 + 16);
                  j_j___libc_free_0(v111, 40);
                }
                v105 += 80;
                v112 = i + 80;
                if ( v26 == v105 )
                  break;
              }
              v27 = i + 160;
              v26 = v112;
            }
            if ( v124 )
              j_j___libc_free_0(v124, *(_QWORD *)(v15 + 160) - v124);
            *(_QWORD *)(v15 + 152) = v27;
            *(_QWORD *)(v15 + 144) = v125;
            *(_QWORD *)(v15 + 160) = v123;
            goto LABEL_16;
          }
LABEL_138:
          a6 = 0;
          *(_QWORD *)(v104 + 40) = 0;
          *(_OWORD *)(v104 + 16) = 0;
          *(_QWORD *)(v104 + 32) = v104 + 8;
          *(_QWORD *)(v104 + 24) = v104 + 8;
          *(_OWORD *)v104 = 0;
          *(_OWORD *)(v104 + 48) = 0;
          *(_OWORD *)(v104 + 64) = 0;
          goto LABEL_139;
        }
        if ( v103 > 0x199999999999999LL )
          v103 = 0x199999999999999LL;
        v119 = 80 * v103;
      }
      v121 = v26 - *(_QWORD *)(v15 + 144);
      v125 = sub_22077B0(v119);
      v123 = v125 + v119;
      v27 = v125 + 80;
      v104 = v125 + v121;
      if ( !(v125 + v121) )
        goto LABEL_139;
      goto LABEL_138;
    }
    v38 = *(_DWORD *)(v15 + 136);
    v39 = v15 + 112;
    v26 = 0;
    LODWORD(v151) = (v149 != 1) + 3;
    if ( !v38 )
    {
LABEL_75:
      ++*(_QWORD *)(v15 + 112);
      goto LABEL_76;
    }
LABEL_30:
    v40 = v136;
    v41 = v38 - 1;
    v42 = *(_QWORD *)(v15 + 120);
    v43 = (v38 - 1) & (((unsigned int)v136 >> 9) ^ ((unsigned int)v136 >> 4));
    v44 = *(_QWORD *)(v42 + 40LL * v43);
    v134 = v42 + 40LL * v43;
    if ( v136 == v44 )
      goto LABEL_31;
    v69 = (_QWORD *)(v42 + 40LL * ((v38 - 1) & (((unsigned int)v136 >> 9) ^ ((unsigned int)v136 >> 4))));
    v70 = 1;
    v71 = 0;
    while ( 1 )
    {
      if ( v44 == -4 )
      {
        v72 = *(_DWORD *)(v15 + 128);
        if ( !v71 )
          v71 = v69;
        ++*(_QWORD *)(v15 + 112);
        v73 = v72 + 1;
        if ( 4 * v73 < 3 * v38 )
        {
          if ( v38 - *(_DWORD *)(v15 + 132) - v73 > v38 >> 3 )
          {
LABEL_67:
            *(_DWORD *)(v15 + 128) = v73;
            if ( *(_QWORD *)v71 != -4 )
              --*(_DWORD *)(v15 + 132);
            *(_QWORD *)v71 = v40;
            v71[32] = 0;
            *((_QWORD *)v71 + 1) = 0;
            *((_QWORD *)v71 + 2) = 0;
            *((_QWORD *)v71 + 3) = 0;
            goto LABEL_70;
          }
LABEL_77:
          sub_1874910(v39, v38);
          sub_1872280(v39, &v136, &s2);
          v71 = s2;
          v40 = v136;
          v73 = *(_DWORD *)(v15 + 128) + 1;
          goto LABEL_67;
        }
LABEL_76:
        v38 *= 2;
        goto LABEL_77;
      }
      if ( !v71 && v44 == -8 )
        v71 = v69;
      v116 = v70 + 1;
      v117 = v43 + v70;
      v43 = v41 & v117;
      v69 = (_QWORD *)(v42 + 40LL * (v41 & v117));
      v44 = *v69;
      if ( v136 == *v69 )
        break;
      v70 = v116;
    }
    v134 = v42 + 40LL * (v41 & v117);
LABEL_31:
    v45 = v134;
    if ( !*(_BYTE *)(v134 + 32) )
      goto LABEL_55;
    v46 = (__int64 *)sub_161E970(v136);
    v48 = v47;
    v138 = v46;
    v49 = v46;
    v50 = v46;
    v51 = *(_QWORD *)(v15 + 8);
    v139 = v48;
    v131 = v51;
    s2 = v142;
    if ( v49 )
    {
      sub_18736F0((__int64 *)&s2, v50, (__int64)v50 + v139);
    }
    else
    {
      v141 = 0;
      LOBYTE(v142[0]) = 0;
    }
    v52 = v131 + 88;
    if ( !*(_QWORD *)(v131 + 96) )
    {
      v52 = v131 + 88;
LABEL_113:
      p_s2 = (__m128i *)&s2;
      v92 = sub_1880E00((_QWORD *)(v131 + 80), (_QWORD *)v52, &p_s2);
      v53 = s2;
      v52 = v92;
      goto LABEL_48;
    }
    v122 = v26;
    v120 = v15;
    v53 = s2;
    v54 = v141;
    v55 = *(_QWORD *)(v131 + 96);
    while ( 2 )
    {
      while ( 2 )
      {
        v56 = *(_QWORD *)(v55 + 40);
        v57 = (size_t)v54;
        if ( v56 <= (unsigned __int64)v54 )
          v57 = *(_QWORD *)(v55 + 40);
        if ( !v57 || (v58 = memcmp(*(const void **)(v55 + 32), v53, v57)) == 0 )
        {
          v59 = v56 - (_QWORD)v54;
          if ( v59 >= 0x80000000LL )
            goto LABEL_45;
          if ( v59 > (__int64)0xFFFFFFFF7FFFFFFFLL )
          {
            v58 = v59;
            break;
          }
LABEL_36:
          v55 = *(_QWORD *)(v55 + 24);
          if ( !v55 )
            goto LABEL_46;
          continue;
        }
        break;
      }
      if ( v58 < 0 )
        goto LABEL_36;
LABEL_45:
      v52 = v55;
      v55 = *(_QWORD *)(v55 + 16);
      if ( v55 )
        continue;
      break;
    }
LABEL_46:
    v60 = (size_t)v54;
    v15 = v120;
    v26 = v122;
    if ( v52 == v131 + 88 || sub_1872D20(v53, v60, *(const void **)(v52 + 32), *(_QWORD *)(v52 + 40)) < 0 )
      goto LABEL_113;
LABEL_48:
    if ( v53 != v142 )
      j_j___libc_free_0(v53, v142[0] + 1LL);
    v61 = (int)v151;
    *(_DWORD *)(v52 + 64) = (_DWORD)v151;
    s2 = (void *)v15;
    v141 = &v138;
    if ( !v61
      || (sub_1874510((__int64)&s2, (__int64)"global_addr", 11, v152), (unsigned int)((_DWORD)v151 - 1) > 1)
      && (_DWORD)v151 != 4 )
    {
LABEL_51:
      v62 = 0;
      goto LABEL_52;
    }
    if ( (unsigned int)(*(_DWORD *)(v15 + 24) - 31) <= 1 && *(_DWORD *)(v15 + 32) == 2 )
    {
      v113 = sub_15A3BA0(v153, *(__int64 ***)(v15 + 56), 0);
      sub_1874510((__int64)&s2, (__int64)"align", 5, v113);
    }
    else
    {
      v93 = *(_QWORD **)(v153 + 24);
      if ( *(_DWORD *)(v153 + 32) > 0x40u )
        v93 = (_QWORD *)*v93;
      *(_QWORD *)(v52 + 72) = v93;
    }
    v94 = v154;
    if ( (unsigned int)(*(_DWORD *)(v15 + 24) - 31) <= 1 && *(_DWORD *)(v15 + 32) == 2 )
    {
      v114 = sub_15A3BA0(v154, *(__int64 ***)(v15 + 56), 0);
      sub_1874510((__int64)&s2, (__int64)"size_m1", 7, v114);
      v94 = v154;
    }
    else
    {
      v95 = *(_QWORD **)(v154 + 24);
      if ( *(_DWORD *)(v154 + 32) > 0x40u )
        v95 = (_QWORD *)*v95;
      *(_QWORD *)(v52 + 80) = v95;
    }
    v96 = *(_QWORD **)(v94 + 24);
    if ( *(_DWORD *)(v94 + 32) > 0x40u )
      v96 = (_QWORD *)*v96;
    v97 = (int)v151;
    v98 = (unsigned __int64)v96 + 1;
    if ( (_DWORD)v151 == 2 )
    {
      *(_DWORD *)(v52 + 68) = (v98 > 0x20) + 5;
LABEL_165:
      if ( (unsigned int)(*(_DWORD *)(v15 + 24) - 31) <= 1 && *(_DWORD *)(v15 + 32) == 2 )
      {
        v118 = sub_15A3BA0(v157, *(__int64 ***)(v15 + 56), 0);
        sub_1874510((__int64)&s2, (__int64)"inline_bits", 11, v118);
      }
      else
      {
        v115 = *(_QWORD **)(v157 + 24);
        if ( *(_DWORD *)(v157 + 32) > 0x40u )
          v115 = (_QWORD *)*v115;
        *(_QWORD *)(v52 + 96) = v115;
      }
      goto LABEL_51;
    }
    *(_DWORD *)(v52 + 68) = v98 < 0x81 ? 7 : 32;
    if ( v97 != 1 )
      goto LABEL_51;
    sub_1874510((__int64)&s2, (__int64)"byte_array", 10, (__int64)v155);
    if ( (unsigned int)(*(_DWORD *)(v15 + 24) - 31) <= 1 && *(_DWORD *)(v15 + 32) == 2 )
    {
      sub_1874510((__int64)&s2, (__int64)"bit_mask", 8, v156);
      if ( (_DWORD)v151 != 2 )
        goto LABEL_51;
      goto LABEL_165;
    }
    v62 = v52 + 88;
LABEL_52:
    if ( v26 )
      *(_QWORD *)(v26 + 72) = v62;
    v45 = v134;
LABEL_55:
    v63 = *(__int64 **)(v45 + 16);
    for ( j = *(__int64 **)(v45 + 8); v63 != j; ++j )
    {
      v68 = (_QWORD *)*j;
      if ( (_DWORD)v151 )
        v65 = sub_187EA30((__int64 *)v15, v136, *j, (__int64)&v151, *(double *)a6.m128_u64, a7, a8);
      else
        v65 = sub_159C540(**(__int64 ***)v15);
      sub_164D160((__int64)v68, v65, a6, a7, a8, a9, v66, v67, a12, a13);
      sub_15F20C0(v68);
    }
LABEL_70:
    v74 = v145;
    while ( v74 )
    {
      sub_1876060(*((_QWORD *)v74 + 3));
      v75 = v74;
      v74 = (int *)*((_QWORD *)v74 + 2);
      j_j___libc_free_0(v75, 40);
    }
    result = ++v129;
  }
  while ( v126 != v129 );
  return result;
}
