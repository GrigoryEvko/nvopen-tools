// Function: sub_30C17F0
// Address: 0x30c17f0
//
void __fastcall sub_30C17F0(_QWORD *a1)
{
  __int64 v1; // rsi
  __int64 v2; // r15
  int v3; // r11d
  __int64 v4; // rbx
  __int64 v5; // rdi
  __int64 v6; // rsi
  __int64 v7; // rdx
  int v8; // r10d
  unsigned int v9; // r9d
  unsigned __int64 v10; // r8
  void *v11; // rcx
  __int64 v12; // rax
  unsigned __int64 v13; // r12
  __int64 v14; // r13
  __int64 v15; // r8
  unsigned __int64 v16; // rdx
  unsigned __int64 v17; // r12
  void *v18; // r15
  void *v19; // r14
  __int64 v20; // r9
  unsigned __int64 v21; // rax
  void *v22; // rbx
  int v23; // eax
  int v24; // eax
  __int64 v25; // rdx
  bool v26; // zf
  unsigned __int64 *v27; // rdx
  unsigned __int64 *v28; // r12
  void *v29; // rdi
  __int64 v30; // rbx
  __int64 v31; // rax
  __int64 v32; // rdx
  __int64 *v33; // r13
  __int64 v34; // rbx
  __int64 *v35; // r12
  unsigned __int64 v36; // rax
  __int64 *v37; // rbx
  __int64 *v38; // rdi
  __int64 (__fastcall *v39)(__int64, __int64); // rax
  __int64 v40; // rax
  __int64 v41; // rdx
  __int64 v42; // rcx
  __int64 v43; // r8
  __int64 v44; // r9
  __int64 v45; // rbx
  __int64 *v46; // rdx
  __int64 v47; // r8
  __int64 v48; // r9
  __int64 *v49; // r12
  __int64 v50; // rax
  __int64 *v51; // r13
  __int64 v52; // rcx
  __int64 v53; // rsi
  _QWORD *v54; // rax
  __int64 v55; // rax
  __int64 *v56; // r12
  __int64 v57; // r13
  _QWORD *v58; // rax
  _QWORD *v59; // rdx
  int v60; // eax
  __int64 v61; // rdx
  int v62; // eax
  __int64 v63; // rdx
  unsigned __int64 *v64; // rbx
  unsigned __int64 *v65; // r12
  __int64 *v66; // r15
  __int64 v67; // r12
  __int64 v68; // rax
  unsigned int v69; // ecx
  _QWORD *v70; // rax
  _QWORD *m; // rdx
  unsigned int v72; // ecx
  _QWORD *v73; // rax
  _QWORD *j; // rdx
  __int64 v75; // r8
  size_t v76; // r9
  __int64 v77; // r12
  _QWORD *v78; // rbx
  void *v79; // rdi
  __int64 v80; // rax
  unsigned __int64 *v81; // r12
  __int64 v82; // r14
  unsigned __int64 *v83; // r15
  void *v84; // rax
  unsigned int v85; // ebx
  const void *v86; // rsi
  size_t v87; // rdx
  unsigned __int64 *v88; // r14
  int v89; // r14d
  unsigned int v90; // eax
  int v91; // ebx
  _QWORD *v92; // rdi
  _QWORD *v93; // rax
  unsigned int v94; // kr00_4
  unsigned __int64 v95; // rdx
  unsigned __int64 v96; // rax
  _QWORD *v97; // rax
  __int64 v98; // rdx
  _QWORD *k; // rdx
  unsigned int v100; // eax
  int v101; // ebx
  _QWORD *v102; // rdi
  unsigned int v103; // kr04_4
  unsigned __int64 v104; // rdx
  unsigned __int64 v105; // rax
  _QWORD *v106; // rax
  __int64 v107; // rdx
  _QWORD *i; // rdx
  _QWORD *v109; // rax
  unsigned __int64 *v110; // [rsp+18h] [rbp-328h]
  unsigned __int8 v111; // [rsp+27h] [rbp-319h]
  __int64 v113; // [rsp+48h] [rbp-2F8h]
  size_t n; // [rsp+58h] [rbp-2E8h]
  signed __int64 na; // [rsp+58h] [rbp-2E8h]
  __int64 v116; // [rsp+60h] [rbp-2E0h]
  __int64 v117; // [rsp+60h] [rbp-2E0h]
  __int64 v118; // [rsp+60h] [rbp-2E0h]
  __int64 v119; // [rsp+60h] [rbp-2E0h]
  __int64 *v120; // [rsp+68h] [rbp-2D8h]
  _QWORD *v121; // [rsp+78h] [rbp-2C8h] BYREF
  _QWORD v122[4]; // [rsp+80h] [rbp-2C0h] BYREF
  unsigned int v123; // [rsp+A0h] [rbp-2A0h]
  unsigned __int64 v124; // [rsp+A8h] [rbp-298h]
  void *s1; // [rsp+C0h] [rbp-280h]
  void *v126; // [rsp+C8h] [rbp-278h]
  unsigned __int64 v127; // [rsp+D8h] [rbp-268h]
  __int64 v128; // [rsp+E0h] [rbp-260h]
  _QWORD *v129; // [rsp+F0h] [rbp-250h] BYREF
  unsigned __int64 *v130; // [rsp+F8h] [rbp-248h]
  _QWORD *v131; // [rsp+100h] [rbp-240h]
  __int64 v132; // [rsp+108h] [rbp-238h]
  __int64 v133; // [rsp+110h] [rbp-230h]
  unsigned __int64 v134; // [rsp+118h] [rbp-228h]
  __int64 v135; // [rsp+120h] [rbp-220h]
  __int64 v136; // [rsp+128h] [rbp-218h]
  void *s2; // [rsp+130h] [rbp-210h]
  __int64 v138; // [rsp+138h] [rbp-208h]
  __int64 v139; // [rsp+140h] [rbp-200h]
  unsigned __int64 v140; // [rsp+148h] [rbp-1F8h]
  __int64 v141; // [rsp+150h] [rbp-1F0h]
  __int64 v142; // [rsp+158h] [rbp-1E8h]
  unsigned __int64 *v143; // [rsp+160h] [rbp-1E0h] BYREF
  __int64 v144; // [rsp+168h] [rbp-1D8h]
  _BYTE v145[192]; // [rsp+170h] [rbp-1D0h] BYREF
  __int64 v146; // [rsp+230h] [rbp-110h] BYREF
  unsigned __int64 v147; // [rsp+238h] [rbp-108h]
  __int64 v148; // [rsp+240h] [rbp-100h]
  int v149; // [rsp+248h] [rbp-F8h]
  int v150; // [rsp+24Ch] [rbp-F4h]
  unsigned int v151; // [rsp+250h] [rbp-F0h] BYREF
  unsigned __int64 v152; // [rsp+258h] [rbp-E8h]
  __int64 v153; // [rsp+260h] [rbp-E0h]
  __int64 v154; // [rsp+268h] [rbp-D8h]
  unsigned __int64 v155; // [rsp+270h] [rbp-D0h]
  __int64 v156; // [rsp+278h] [rbp-C8h]
  __int64 v157; // [rsp+280h] [rbp-C0h]
  unsigned __int64 v158; // [rsp+288h] [rbp-B8h]
  __int64 v159; // [rsp+290h] [rbp-B0h]
  __int64 v160; // [rsp+298h] [rbp-A8h]
  _QWORD v161[2]; // [rsp+2A0h] [rbp-A0h] BYREF
  __int64 v162; // [rsp+2B0h] [rbp-90h]
  __int64 v163; // [rsp+2B8h] [rbp-88h]
  __int64 v164; // [rsp+2C0h] [rbp-80h]
  unsigned __int64 v165; // [rsp+2C8h] [rbp-78h]
  __int64 v166; // [rsp+2D0h] [rbp-70h]
  __int64 v167; // [rsp+2D8h] [rbp-68h]
  unsigned __int64 v168; // [rsp+2E0h] [rbp-60h]
  __int64 v169; // [rsp+2E8h] [rbp-58h]
  __int64 v170; // [rsp+2F0h] [rbp-50h]
  unsigned __int64 v171; // [rsp+2F8h] [rbp-48h]
  __int64 v172; // [rsp+300h] [rbp-40h]
  __int64 v173; // [rsp+308h] [rbp-38h]

  v111 = (*(__int64 (__fastcall **)(_QWORD *))(*a1 + 88LL))(a1);
  if ( !v111 )
    return;
  v143 = (unsigned __int64 *)v145;
  v144 = 0x400000000LL;
  v1 = *(_QWORD *)(a1[1] + 88LL);
  v129 = 0;
  v130 = 0;
  v131 = 0;
  v132 = 0;
  v133 = 0;
  v134 = 0;
  v135 = 0;
  v136 = 0;
  s2 = 0;
  v138 = 0;
  v139 = 0;
  v140 = 0;
  v141 = 0;
  v142 = 0;
  sub_30BD420((__int64)&v129, v1);
  sub_30BD840((__int64)&v129);
  v2 = v142;
  v3 = v132;
  v142 = 0;
  v4 = (__int64)v131;
  v5 = v135;
  v131 = 0;
  v6 = v136;
  v7 = v138;
  v136 = 0;
  v8 = HIDWORD(v132);
  v9 = v133;
  LODWORD(v146) = (_DWORD)v129;
  v10 = v134;
  v11 = s2;
  v132 = 0;
  v12 = v139;
  v13 = v140;
  LODWORD(v133) = 0;
  v14 = v141;
  v130 = (unsigned __int64 *)((char *)v130 + 1);
  v135 = 0;
  v134 = 0;
  v139 = 0;
  v138 = 0;
  s2 = 0;
  v141 = 0;
  v140 = 0;
  v147 = 1;
  v148 = v4;
  v149 = v3;
  v153 = v5;
  v154 = v6;
  v156 = v7;
  v150 = v8;
  v151 = v9;
  v152 = v10;
  v155 = (unsigned __int64)v11;
  v157 = v12;
  v158 = v13;
  v159 = v14;
  v160 = v2;
  v161[0] = 0;
  v161[1] = 1;
  v162 = 0;
  v163 = 0;
  v164 = 0;
  v165 = 0;
  v166 = 0;
  v167 = 0;
  v168 = 0;
  v169 = 0;
  v170 = 0;
  v171 = 0;
  v172 = 0;
  v173 = 0;
  sub_C7D6A0(0, 0, 8);
  sub_C7D6A0(0, 0, 8);
  if ( v140 )
    j_j___libc_free_0(v140);
  if ( s2 )
    j_j___libc_free_0((unsigned __int64)s2);
  if ( v134 )
    j_j___libc_free_0(v134);
  sub_C7D6A0((__int64)v131, 16LL * (unsigned int)v133, 8);
  sub_C7D6A0(0, 0, 8);
  sub_30B96B0((__int64)v122, (int *)&v146);
  sub_30B96B0((__int64)&v129, (int *)v161);
  while ( 1 )
  {
    v16 = v127;
    v17 = v140;
    v18 = v126;
    v19 = s1;
    v20 = (_BYTE *)v126 - (_BYTE *)s1;
    if ( v128 - v127 != v141 - v140 )
      goto LABEL_10;
    if ( v127 != v128 )
    {
      v21 = v140;
      while ( *(_QWORD *)v16 == *(_QWORD *)v21
           && *(_QWORD *)(v16 + 8) == *(_QWORD *)(v21 + 8)
           && *(_DWORD *)(v16 + 24) == *(_DWORD *)(v21 + 24) )
      {
        v16 += 32LL;
        v21 += 32LL;
        if ( v128 == v16 )
          goto LABEL_19;
      }
      goto LABEL_10;
    }
LABEL_19:
    v22 = s2;
    if ( v138 - (_QWORD)s2 != v20 )
      goto LABEL_10;
    if ( !v20 )
      break;
    v116 = (_BYTE *)v126 - (_BYTE *)s1;
    v23 = memcmp(s1, s2, (_BYTE *)v126 - (_BYTE *)s1);
    v20 = v116;
    if ( !v23 )
    {
      if ( !v17 )
        goto LABEL_35;
      goto LABEL_33;
    }
LABEL_10:
    if ( (unsigned __int64)v20 > 8 )
    {
      v24 = v144;
      if ( HIDWORD(v144) <= (unsigned int)v144 )
      {
        na = v20;
        v119 = sub_C8D7D0((__int64)&v143, (__int64)v145, 0, 0x30u, (unsigned __int64 *)&v121, v20);
        v77 = 6LL * (unsigned int)v144;
        v78 = (_QWORD *)(v77 * 8 + v119);
        if ( v77 * 8 + v119 )
        {
          v76 = na;
          v79 = v78 + 2;
          v78[1] = 0x400000000LL;
          LODWORD(v80) = 0;
          *v78 = v78 + 2;
          if ( (unsigned __int64)na > 0x20 )
          {
            sub_C8D5F0((__int64)v78, v78 + 2, na >> 3, 8u, v75, na);
            v80 = *((unsigned int *)v78 + 2);
            v76 = na;
            v79 = (void *)(*v78 + 8 * v80);
          }
          if ( v18 != v19 )
          {
            memmove(v79, v19, v76);
            LODWORD(v80) = *((_DWORD *)v78 + 2);
          }
          *((_DWORD *)v78 + 2) = (na >> 3) + v80;
          v77 = 6LL * (unsigned int)v144;
        }
        v81 = &v143[v77];
        if ( v143 != v81 )
        {
          v82 = v119;
          v83 = v143;
          do
          {
            if ( v82 )
            {
              v84 = (void *)(v82 + 16);
              *(_DWORD *)(v82 + 8) = 0;
              *(_QWORD *)v82 = v82 + 16;
              *(_DWORD *)(v82 + 12) = 4;
              v85 = *((_DWORD *)v83 + 2);
              if ( v85 )
              {
                if ( (unsigned __int64 *)v82 != v83 )
                {
                  v86 = v83 + 2;
                  if ( (unsigned __int64 *)*v83 == v83 + 2 )
                  {
                    v87 = 8LL * v85;
                    if ( v85 <= 4
                      || (sub_C8D5F0(v82, (const void *)(v82 + 16), v85, 8u, v85, v76),
                          v84 = *(void **)v82,
                          v86 = (const void *)*v83,
                          (v87 = 8LL * *((unsigned int *)v83 + 2)) != 0) )
                    {
                      memcpy(v84, v86, v87);
                    }
                    *(_DWORD *)(v82 + 8) = v85;
                    *((_DWORD *)v83 + 2) = 0;
                  }
                  else
                  {
                    *(_QWORD *)v82 = *v83;
                    *(_DWORD *)(v82 + 8) = *((_DWORD *)v83 + 2);
                    *(_DWORD *)(v82 + 12) = *((_DWORD *)v83 + 3);
                    *v83 = (unsigned __int64)v86;
                    *((_DWORD *)v83 + 3) = 0;
                    *((_DWORD *)v83 + 2) = 0;
                  }
                }
              }
            }
            v83 += 6;
            v82 += 48;
          }
          while ( v81 != v83 );
          v88 = v143;
          v81 = &v143[6 * (unsigned int)v144];
          if ( v81 != v143 )
          {
            do
            {
              v81 -= 6;
              if ( (unsigned __int64 *)*v81 != v81 + 2 )
                _libc_free(*v81);
            }
            while ( v88 != v81 );
            v81 = v143;
          }
        }
        v89 = (int)v121;
        if ( v81 != (unsigned __int64 *)v145 )
          _libc_free((unsigned __int64)v81);
        LODWORD(v144) = v144 + 1;
        HIDWORD(v144) = v89;
        v143 = (unsigned __int64 *)v119;
      }
      else
      {
        v25 = 6LL * (unsigned int)v144;
        v26 = &v143[v25] == 0;
        v27 = &v143[v25];
        v28 = v27;
        if ( !v26 )
        {
          v29 = v27 + 2;
          v27[1] = 0x400000000LL;
          v30 = v20 >> 3;
          LODWORD(v31) = 0;
          *v27 = (unsigned __int64)(v27 + 2);
          if ( (unsigned __int64)v20 > 0x20 )
          {
            v118 = v20;
            sub_C8D5F0((__int64)v27, v29, v20 >> 3, 8u, v15, v20);
            v31 = *((unsigned int *)v28 + 2);
            v20 = v118;
            v29 = (void *)(*v28 + 8 * v31);
          }
          if ( v18 != v19 )
          {
            memmove(v29, v19, v20);
            LODWORD(v31) = *((_DWORD *)v28 + 2);
          }
          *((_DWORD *)v28 + 2) = v30 + v31;
          v24 = v144;
        }
        LODWORD(v144) = v24 + 1;
      }
    }
    sub_30BD840((__int64)v122);
  }
  if ( !v140 )
    goto LABEL_34;
LABEL_33:
  j_j___libc_free_0(v17);
  v22 = s2;
LABEL_34:
  if ( !v22 )
    goto LABEL_36;
LABEL_35:
  j_j___libc_free_0((unsigned __int64)v22);
LABEL_36:
  if ( v134 )
    j_j___libc_free_0(v134);
  sub_C7D6A0((__int64)v131, 16LL * (unsigned int)v133, 8);
  if ( v127 )
    j_j___libc_free_0(v127);
  if ( s1 )
    j_j___libc_free_0((unsigned __int64)s1);
  if ( v124 )
    j_j___libc_free_0(v124);
  sub_C7D6A0(v122[2], 16LL * v123, 8);
  if ( v171 )
    j_j___libc_free_0(v171);
  if ( v168 )
    j_j___libc_free_0(v168);
  if ( v165 )
    j_j___libc_free_0(v165);
  sub_C7D6A0(v162, 16LL * (unsigned int)v164, 8);
  if ( v158 )
    j_j___libc_free_0(v158);
  if ( v155 )
    j_j___libc_free_0(v155);
  if ( v152 )
    j_j___libc_free_0(v152);
  sub_C7D6A0(v148, 16LL * v151, 8);
  v32 = 6LL * (unsigned int)v144;
  v110 = &v143[v32];
  if ( &v143[v32] == v143 )
    goto LABEL_86;
  v113 = (__int64)v143;
  while ( 2 )
  {
    v33 = *(__int64 **)v113;
    v34 = 8LL * *(unsigned int *)(v113 + 8);
    v35 = (__int64 *)(*(_QWORD *)v113 + v34);
    if ( *(__int64 **)v113 != v35 )
    {
      _BitScanReverse64(&v36, v34 >> 3);
      sub_30C0440(*(__int64 **)v113, (char *)(*(_QWORD *)v113 + v34), 2LL * (int)(63 - (v36 ^ 0x3F)), (__int64)a1);
      if ( (unsigned __int64)v34 <= 0x80 )
      {
        sub_30BCD00(v33, v35, (__int64)a1);
      }
      else
      {
        v37 = v33 + 16;
        sub_30BCD00(v33, v33 + 16, (__int64)a1);
        if ( v35 != v33 + 16 )
        {
          do
          {
            v38 = v37++;
            sub_30BDDA0(v38, (__int64)a1);
          }
          while ( v35 != v37 );
        }
      }
    }
    v39 = *(__int64 (__fastcall **)(__int64, __int64))(*a1 + 32LL);
    if ( v39 == sub_30B27D0 )
    {
      v40 = sub_22077B0(0x70u);
      v45 = v40;
      if ( v40 )
        sub_30B0D80(v40, v113, v41, v42, v43, v44);
      sub_30B2450(a1[1], v45);
    }
    else
    {
      v45 = v39((__int64)a1, v113);
    }
    v49 = *(__int64 **)v113;
    v50 = *(unsigned int *)(v113 + 8);
    LOBYTE(v150) = 1;
    v146 = 0;
    v148 = 4;
    v51 = &v49[v50];
    v149 = 0;
    v147 = (unsigned __int64)&v151;
    if ( v49 == v51 )
    {
      v68 = a1[1];
      v56 = *(__int64 **)(v68 + 96);
      n = (size_t)&v56[*(unsigned int *)(v68 + 104)];
      if ( v56 == (__int64 *)n )
        goto LABEL_85;
      LOBYTE(v52) = v111;
LABEL_75:
      v120 = v56;
      while ( 1 )
      {
        v57 = *v120;
        if ( v45 + 8 != *v120 + 8 )
          break;
LABEL_82:
        if ( (__int64 *)n == ++v120 )
          goto LABEL_83;
      }
      if ( (_BYTE)v52 )
      {
        v58 = (_QWORD *)v147;
        v59 = (_QWORD *)(v147 + 8LL * HIDWORD(v148));
        if ( (_QWORD *)v147 != v59 )
        {
          while ( v57 != *v58 )
          {
            if ( v59 == ++v58 )
              goto LABEL_107;
          }
          goto LABEL_82;
        }
      }
      else if ( sub_C8CA60((__int64)&v146, *v120) )
      {
LABEL_109:
        LOBYTE(v52) = v150;
        goto LABEL_82;
      }
LABEL_107:
      v122[0] = 0;
      v130 = (unsigned __int64 *)&v121;
      v121 = a1;
      v129 = v122;
      v131 = a1;
      v66 = *(__int64 **)v113;
      v117 = *(_QWORD *)v113 + 8LL * *(unsigned int *)(v113 + 8);
      if ( v117 != *(_QWORD *)v113 )
      {
        do
        {
          v67 = *v66++;
          sub_30BEB60((__int64 *)&v129, v57, v67, v45, 0);
          sub_30BEB60((__int64 *)&v129, v67, v57, v45, 1);
        }
        while ( (__int64 *)v117 != v66 );
      }
      goto LABEL_109;
    }
    v52 = v111;
    do
    {
      while ( 1 )
      {
        while ( 1 )
        {
          v53 = *v49;
          if ( (_BYTE)v52 )
            break;
LABEL_101:
          ++v49;
          sub_C8CC70((__int64)&v146, v53, (__int64)v46, v52, v47, v48);
          v52 = (unsigned __int8)v150;
          if ( v51 == v49 )
            goto LABEL_74;
        }
        v54 = (_QWORD *)v147;
        v46 = (__int64 *)(v147 + 8LL * HIDWORD(v148));
        if ( (__int64 *)v147 != v46 )
          break;
LABEL_103:
        if ( HIDWORD(v148) >= (unsigned int)v148 )
          goto LABEL_101;
        ++v49;
        ++HIDWORD(v148);
        *v46 = v53;
        v52 = (unsigned __int8)v150;
        ++v146;
        if ( v51 == v49 )
          goto LABEL_74;
      }
      while ( v53 != *v54 )
      {
        if ( v46 == ++v54 )
          goto LABEL_103;
      }
      ++v49;
    }
    while ( v51 != v49 );
LABEL_74:
    v55 = a1[1];
    v56 = *(__int64 **)(v55 + 96);
    n = (size_t)&v56[*(unsigned int *)(v55 + 104)];
    if ( (__int64 *)n != v56 )
      goto LABEL_75;
LABEL_83:
    if ( !(_BYTE)v52 )
      _libc_free(v147);
LABEL_85:
    v113 += 48;
    if ( v110 != (unsigned __int64 *)v113 )
      continue;
    break;
  }
LABEL_86:
  ++a1[8];
  v60 = *((_DWORD *)a1 + 20);
  if ( v60 )
  {
    v72 = 4 * v60;
    v61 = *((unsigned int *)a1 + 22);
    if ( (unsigned int)(4 * v60) < 0x40 )
      v72 = 64;
    if ( (unsigned int)v61 <= v72 )
      goto LABEL_123;
    v100 = v60 - 1;
    if ( v100 )
    {
      _BitScanReverse(&v100, v100);
      v101 = 1 << (33 - (v100 ^ 0x1F));
      v102 = (_QWORD *)a1[9];
      if ( v101 < 64 )
        v101 = 64;
      if ( (_DWORD)v61 == v101 )
      {
        a1[10] = 0;
        v109 = &v102[2 * (unsigned int)v61];
        do
        {
          if ( v102 )
            *v102 = -4096;
          v102 += 2;
        }
        while ( v109 != v102 );
        goto LABEL_90;
      }
    }
    else
    {
      v101 = 64;
      v102 = (_QWORD *)a1[9];
    }
    sub_C7D6A0((__int64)v102, 16LL * *((unsigned int *)a1 + 22), 8);
    v103 = 4 * v101;
    v104 = ((((((((v103 / 3 + 1) | ((unsigned __int64)(v103 / 3 + 1) >> 1)) >> 2)
              | (v103 / 3 + 1)
              | ((unsigned __int64)(v103 / 3 + 1) >> 1)) >> 4)
            | (((v103 / 3 + 1) | ((unsigned __int64)(v103 / 3 + 1) >> 1)) >> 2)
            | (v103 / 3 + 1)
            | ((unsigned __int64)(v103 / 3 + 1) >> 1)) >> 8)
          | (((((v103 / 3 + 1) | ((unsigned __int64)(v103 / 3 + 1) >> 1)) >> 2)
            | (v103 / 3 + 1)
            | ((unsigned __int64)(v103 / 3 + 1) >> 1)) >> 4)
          | (((v103 / 3 + 1) | ((unsigned __int64)(v103 / 3 + 1) >> 1)) >> 2)
          | (v103 / 3 + 1)
          | ((unsigned __int64)(v103 / 3 + 1) >> 1)) >> 16;
    v105 = (v104
          | (((((((v103 / 3 + 1) | ((unsigned __int64)(v103 / 3 + 1) >> 1)) >> 2)
              | (v103 / 3 + 1)
              | ((unsigned __int64)(v103 / 3 + 1) >> 1)) >> 4)
            | (((v103 / 3 + 1) | ((unsigned __int64)(v103 / 3 + 1) >> 1)) >> 2)
            | (v103 / 3 + 1)
            | ((unsigned __int64)(v103 / 3 + 1) >> 1)) >> 8)
          | (((((v103 / 3 + 1) | ((unsigned __int64)(v103 / 3 + 1) >> 1)) >> 2)
            | (v103 / 3 + 1)
            | ((unsigned __int64)(v103 / 3 + 1) >> 1)) >> 4)
          | (((v103 / 3 + 1) | ((unsigned __int64)(v103 / 3 + 1) >> 1)) >> 2)
          | (v103 / 3 + 1)
          | ((unsigned __int64)(v103 / 3 + 1) >> 1))
         + 1;
    *((_DWORD *)a1 + 22) = v105;
    v106 = (_QWORD *)sub_C7D670(16 * v105, 8);
    v107 = *((unsigned int *)a1 + 22);
    a1[10] = 0;
    a1[9] = v106;
    for ( i = &v106[2 * v107]; i != v106; v106 += 2 )
    {
      if ( v106 )
        *v106 = -4096;
    }
    goto LABEL_90;
  }
  if ( !*((_DWORD *)a1 + 21) )
    goto LABEL_90;
  v61 = *((unsigned int *)a1 + 22);
  if ( (unsigned int)v61 > 0x40 )
  {
    sub_C7D6A0(a1[9], 16LL * *((unsigned int *)a1 + 22), 8);
    a1[9] = 0;
    a1[10] = 0;
    *((_DWORD *)a1 + 22) = 0;
    goto LABEL_90;
  }
LABEL_123:
  v73 = (_QWORD *)a1[9];
  for ( j = &v73[2 * v61]; j != v73; v73 += 2 )
    *v73 = -4096;
  a1[10] = 0;
LABEL_90:
  v62 = *((_DWORD *)a1 + 28);
  ++a1[12];
  if ( v62 )
  {
    v69 = 4 * v62;
    v63 = *((unsigned int *)a1 + 30);
    if ( (unsigned int)(4 * v62) < 0x40 )
      v69 = 64;
    if ( v69 >= (unsigned int)v63 )
      goto LABEL_117;
    v90 = v62 - 1;
    if ( v90 )
    {
      _BitScanReverse(&v90, v90);
      v91 = 1 << (33 - (v90 ^ 0x1F));
      v92 = (_QWORD *)a1[13];
      if ( v91 < 64 )
        v91 = 64;
      if ( v91 == (_DWORD)v63 )
      {
        a1[14] = 0;
        v93 = &v92[2 * (unsigned int)v91];
        do
        {
          if ( v92 )
            *v92 = -4096;
          v92 += 2;
        }
        while ( v93 != v92 );
        goto LABEL_94;
      }
    }
    else
    {
      v91 = 64;
      v92 = (_QWORD *)a1[13];
    }
    sub_C7D6A0((__int64)v92, 16LL * *((unsigned int *)a1 + 30), 8);
    v94 = 4 * v91;
    v95 = ((((((((v94 / 3 + 1) | ((unsigned __int64)(v94 / 3 + 1) >> 1)) >> 2)
             | (v94 / 3 + 1)
             | ((unsigned __int64)(v94 / 3 + 1) >> 1)) >> 4)
           | (((v94 / 3 + 1) | ((unsigned __int64)(v94 / 3 + 1) >> 1)) >> 2)
           | (v94 / 3 + 1)
           | ((unsigned __int64)(v94 / 3 + 1) >> 1)) >> 8)
         | (((((v94 / 3 + 1) | ((unsigned __int64)(v94 / 3 + 1) >> 1)) >> 2)
           | (v94 / 3 + 1)
           | ((unsigned __int64)(v94 / 3 + 1) >> 1)) >> 4)
         | (((v94 / 3 + 1) | ((unsigned __int64)(v94 / 3 + 1) >> 1)) >> 2)
         | (v94 / 3 + 1)
         | ((unsigned __int64)(v94 / 3 + 1) >> 1)) >> 16;
    v96 = (v95
         | (((((((v94 / 3 + 1) | ((unsigned __int64)(v94 / 3 + 1) >> 1)) >> 2)
             | (v94 / 3 + 1)
             | ((unsigned __int64)(v94 / 3 + 1) >> 1)) >> 4)
           | (((v94 / 3 + 1) | ((unsigned __int64)(v94 / 3 + 1) >> 1)) >> 2)
           | (v94 / 3 + 1)
           | ((unsigned __int64)(v94 / 3 + 1) >> 1)) >> 8)
         | (((((v94 / 3 + 1) | ((unsigned __int64)(v94 / 3 + 1) >> 1)) >> 2)
           | (v94 / 3 + 1)
           | ((unsigned __int64)(v94 / 3 + 1) >> 1)) >> 4)
         | (((v94 / 3 + 1) | ((unsigned __int64)(v94 / 3 + 1) >> 1)) >> 2)
         | (v94 / 3 + 1)
         | ((unsigned __int64)(v94 / 3 + 1) >> 1))
        + 1;
    *((_DWORD *)a1 + 30) = v96;
    v97 = (_QWORD *)sub_C7D670(16 * v96, 8);
    v98 = *((unsigned int *)a1 + 30);
    a1[14] = 0;
    a1[13] = v97;
    for ( k = &v97[2 * v98]; k != v97; v97 += 2 )
    {
      if ( v97 )
        *v97 = -4096;
    }
  }
  else
  {
    if ( !*((_DWORD *)a1 + 29) )
      goto LABEL_94;
    v63 = *((unsigned int *)a1 + 30);
    if ( (unsigned int)v63 > 0x40 )
    {
      sub_C7D6A0(a1[13], 16 * v63, 8);
      a1[13] = 0;
      a1[14] = 0;
      *((_DWORD *)a1 + 30) = 0;
      goto LABEL_94;
    }
LABEL_117:
    v70 = (_QWORD *)a1[13];
    for ( m = &v70[2 * v63]; m != v70; v70 += 2 )
      *v70 = -4096;
    a1[14] = 0;
  }
LABEL_94:
  v64 = v143;
  v65 = &v143[6 * (unsigned int)v144];
  if ( v143 != v65 )
  {
    do
    {
      v65 -= 6;
      if ( (unsigned __int64 *)*v65 != v65 + 2 )
        _libc_free(*v65);
    }
    while ( v64 != v65 );
    v65 = v143;
  }
  if ( v65 != (unsigned __int64 *)v145 )
    _libc_free((unsigned __int64)v65);
}
