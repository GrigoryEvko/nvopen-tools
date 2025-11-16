// Function: sub_EBC8F0
// Address: 0xebc8f0
//
__int64 __fastcall sub_EBC8F0(__int64 a1, __int64 *a2, __int64 **a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *v8; // rcx
  __int64 *v9; // r14
  __int64 *v10; // rdx
  unsigned __int64 v11; // rcx
  unsigned __int64 v12; // rax
  unsigned __int64 v13; // rbx
  unsigned __int64 v14; // rax
  _QWORD *v15; // rax
  _QWORD *i; // rdx
  __int64 v17; // r13
  __int64 v18; // rax
  _DWORD *v19; // rax
  char v20; // r14
  __int64 v21; // rax
  bool v22; // zf
  __int64 *v23; // r15
  int v24; // eax
  __int64 *p_s2; // rsi
  __int64 result; // rax
  size_t v27; // r15
  unsigned int v28; // ebx
  __int64 v29; // r14
  __int64 *v30; // rdx
  __int64 *v31; // rdi
  unsigned __int64 v32; // rax
  __int64 v33; // r8
  __int64 v34; // r9
  __int64 v35; // rax
  __int64 v36; // rax
  __int64 v37; // r15
  __int64 v38; // r14
  __int64 v39; // rdi
  int v40; // ebx
  void *v41; // rcx
  __int64 *v42; // r12
  __int64 v43; // r14
  size_t v44; // r14
  __int64 v45; // rdx
  __int64 v46; // rax
  __int64 v47; // rbx
  __int64 v48; // r12
  unsigned int v49; // r13d
  __int64 v50; // rdi
  unsigned __int64 v51; // rdx
  _QWORD *v52; // rax
  _QWORD *j; // rdx
  unsigned __int64 v54; // rsi
  __int64 *v55; // r15
  __int64 *v56; // rbx
  __int64 v57; // r13
  __int64 v58; // r12
  __int64 v59; // rdi
  char v60; // al
  __int64 *v61; // rdx
  unsigned __int64 v62; // r14
  int v63; // edx
  unsigned int v64; // eax
  __int64 v65; // rdi
  __int64 v66; // rax
  unsigned int v67; // edx
  __int64 v68; // rdi
  __int64 v69; // rdx
  __int64 v70; // r14
  __int64 (*v71)(); // rax
  unsigned int v72; // eax
  __int64 v73; // rbx
  __int64 v74; // r11
  __int64 v75; // rdx
  unsigned __int8 v76; // r9
  __int64 v77; // rax
  __int64 v78; // rcx
  size_t v79; // rdx
  __int64 *v80; // r14
  __int64 *v81; // r15
  __int64 *v82; // rbx
  __int64 v83; // r14
  __int64 v84; // r13
  __int64 v85; // rdi
  __int64 v86; // rax
  __int64 v87; // [rsp+0h] [rbp-1C0h]
  unsigned int v88; // [rsp+10h] [rbp-1B0h]
  __int64 v89; // [rsp+18h] [rbp-1A8h]
  __int64 v90; // [rsp+18h] [rbp-1A8h]
  bool v92; // [rsp+2Fh] [rbp-191h]
  char v94; // [rsp+38h] [rbp-188h]
  __int64 v95; // [rsp+38h] [rbp-188h]
  __int64 *v96; // [rsp+40h] [rbp-180h]
  __int64 *v97; // [rsp+40h] [rbp-180h]
  char v98; // [rsp+48h] [rbp-178h]
  unsigned __int64 v99; // [rsp+48h] [rbp-178h]
  unsigned int v100; // [rsp+50h] [rbp-170h]
  unsigned int v101; // [rsp+58h] [rbp-168h]
  unsigned __int8 v102; // [rsp+58h] [rbp-168h]
  unsigned __int8 v103; // [rsp+58h] [rbp-168h]
  unsigned __int8 v104; // [rsp+58h] [rbp-168h]
  unsigned __int8 v105; // [rsp+58h] [rbp-168h]
  __int64 *v106; // [rsp+58h] [rbp-168h]
  unsigned __int8 v107; // [rsp+58h] [rbp-168h]
  unsigned __int64 v108; // [rsp+68h] [rbp-158h] BYREF
  _QWORD v109[2]; // [rsp+70h] [rbp-150h] BYREF
  void *v110; // [rsp+80h] [rbp-140h]
  size_t v111; // [rsp+88h] [rbp-138h]
  __int16 v112; // [rsp+90h] [rbp-130h]
  __int64 v113[2]; // [rsp+A0h] [rbp-120h] BYREF
  const char *v114; // [rsp+B0h] [rbp-110h]
  __int16 v115; // [rsp+C0h] [rbp-100h]
  _QWORD v116[2]; // [rsp+D0h] [rbp-F0h] BYREF
  __int64 v117; // [rsp+E0h] [rbp-E0h]
  __int64 v118; // [rsp+E8h] [rbp-D8h]
  __int16 v119; // [rsp+F0h] [rbp-D0h]
  const char *v120; // [rsp+100h] [rbp-C0h] BYREF
  __m128i v121; // [rsp+108h] [rbp-B8h] BYREF
  __int64 v122; // [rsp+118h] [rbp-A8h] BYREF
  unsigned int v123; // [rsp+120h] [rbp-A0h]
  _BYTE *v124; // [rsp+130h] [rbp-90h] BYREF
  __int64 v125; // [rsp+138h] [rbp-88h]
  _BYTE v126[32]; // [rsp+140h] [rbp-80h] BYREF
  void *s2; // [rsp+160h] [rbp-60h] BYREF
  size_t n; // [rsp+168h] [rbp-58h]
  __int64 v129; // [rsp+170h] [rbp-50h] BYREF
  __int64 v130; // [rsp+178h] [rbp-48h]
  __int64 v131; // [rsp+180h] [rbp-40h]
  __int16 v132; // [rsp+188h] [rbp-38h]

  v8 = a3[1];
  v9 = *a3;
  v10 = v8;
  if ( a2 )
  {
    v11 = 0xAAAAAAAAAAAAAAABLL * (v8 - v9);
    v12 = 0xAAAAAAAAAAAAAAABLL * ((a2[5] - a2[4]) >> 4);
    v124 = v126;
    v125 = 0x400000000LL;
    v13 = (unsigned int)v12;
    v100 = v12;
    v92 = (_DWORD)v12 != 0;
    if ( (unsigned int)v12 > v11 )
    {
      sub_EA8F90((__int64)a3, (unsigned int)v12 - v11);
      v14 = (unsigned int)v125;
      goto LABEL_45;
    }
    if ( (unsigned int)v12 >= v11 )
    {
      if ( !(_DWORD)v12 )
        goto LABEL_11;
      v14 = 0;
      if ( HIDWORD(v125) >= v13 )
        goto LABEL_6;
      goto LABEL_48;
    }
    v80 = &v9[3 * (unsigned int)v12];
    if ( v10 == v80 )
    {
LABEL_141:
      v14 = (unsigned int)v125;
LABEL_45:
      if ( v14 == v13 )
      {
LABEL_11:
        v94 = 0;
        if ( v100 )
          v94 = *(_BYTE *)(a2[5] - 7);
        goto LABEL_13;
      }
      if ( v14 > v13 )
      {
LABEL_10:
        LODWORD(v125) = v100;
        goto LABEL_11;
      }
      if ( HIDWORD(v125) >= v13 )
      {
LABEL_6:
        v15 = &v124[8 * v14];
        for ( i = &v124[8 * v13]; i != v15; ++v15 )
        {
          if ( v15 )
            *v15 = 0;
        }
        goto LABEL_10;
      }
LABEL_48:
      sub_C8D5F0((__int64)&v124, v126, v13, 8u, a5, a6);
      v14 = (unsigned int)v125;
      goto LABEL_6;
    }
    v99 = (unsigned int)v12;
    v81 = v10;
    v82 = v80;
    v106 = v80;
    do
    {
LABEL_143:
      v83 = v82[1];
      v84 = *v82;
      if ( v83 != *v82 )
      {
        do
        {
          if ( *(_DWORD *)(v84 + 32) > 0x40u )
          {
            v85 = *(_QWORD *)(v84 + 24);
            if ( v85 )
              j_j___libc_free_0_0(v85);
          }
          v84 += 40;
        }
        while ( v83 != v84 );
        v84 = *v82;
      }
      if ( v84 )
        j_j___libc_free_0(v84, v82[2] - v84);
      v82 += 3;
    }
    while ( v81 != v82 );
    v13 = v99;
    a3[1] = v106;
    goto LABEL_141;
  }
  v124 = v126;
  v125 = 0x400000000LL;
  if ( v9 != v8 )
  {
    v106 = v9;
    v81 = v8;
    v100 = 0;
    v92 = 0;
    v99 = 0;
    v82 = v9;
    goto LABEL_143;
  }
  v100 = 0;
  v92 = 0;
  v94 = 0;
LABEL_13:
  v101 = 0;
  v17 = a1 + 40;
  v98 = 0;
  while ( 1 )
  {
    v18 = sub_ECD690(v17);
    s2 = 0;
    v96 = (__int64 *)v18;
    v132 = 0;
    v19 = *(_DWORD **)(a1 + 48);
    n = 0;
    v129 = 0;
    v130 = 0;
    v131 = 0;
    if ( *v19 == 2 )
    {
      LODWORD(v120) = 0;
      v121 = 0u;
      v123 = 1;
      v122 = 0;
      sub_1095550(v17, &v120, 1, 1);
      v40 = (int)v120;
      if ( v123 > 0x40 && v122 )
        j_j___libc_free_0_0(v122);
      if ( v40 == 28 )
      {
        if ( (unsigned __int8)sub_EB61F0(a1, (__int64 *)&s2) )
        {
          p_s2 = v96;
          v120 = "invalid argument identifier for formal argument";
          LOWORD(v123) = 259;
          result = sub_ECDA70(a1, v96, &v120, 0, 0);
          goto LABEL_66;
        }
        if ( **(_DWORD **)(a1 + 48) != 28 )
        {
          p_s2 = (__int64 *)&v120;
          v120 = "expected '=' after formal parameter identifier";
          LOWORD(v123) = 259;
          result = sub_ECE0E0(a1, &v120, 0, 0);
          goto LABEL_66;
        }
        v20 = 0;
        sub_EABFE0(a1);
        if ( v94 )
        {
          v20 = v100 - 1 == v101;
          if ( !n )
          {
LABEL_57:
            v120 = "cannot mix positional and keyword arguments";
            LOWORD(v123) = 259;
LABEL_65:
            p_s2 = v96;
            result = sub_ECDA70(a1, v96, &v120, 0, 0);
            goto LABEL_66;
          }
        }
        else
        {
LABEL_18:
          if ( !n )
            goto LABEL_57;
        }
        v98 = 1;
        goto LABEL_20;
      }
    }
    v20 = 0;
    if ( v94 )
      v20 = v100 - 1 == v101;
    if ( v98 )
      goto LABEL_18;
LABEL_20:
    v21 = sub_ECD690(v17);
    v22 = *(_BYTE *)(a1 + 871) == 0;
    v108 = 0;
    v23 = (__int64 *)v21;
    if ( v22 )
      goto LABEL_23;
    v24 = **(_DWORD **)(a1 + 48);
    if ( v24 == 37 )
    {
      sub_EABFE0(a1);
      p_s2 = v113;
      if ( sub_EAC4D0(a1, v113, (__int64)&v108) )
        goto LABEL_156;
      v68 = *(_QWORD *)(a1 + 232);
      v69 = 0;
      v70 = v113[0];
      v71 = *(__int64 (**)())(*(_QWORD *)v68 + 80LL);
      if ( v71 != sub_C13ED0 )
        v69 = ((__int64 (__fastcall *)(__int64, __int64 *, _QWORD))v71)(v68, v113, 0);
      if ( !sub_E81930(v70, v116, v69) )
      {
        p_s2 = v23;
        v120 = "expected absolute expression";
        LOWORD(v123) = 259;
        result = sub_ECDA70(a1, v23, &v120, 0, 0);
        goto LABEL_66;
      }
      p_s2 = (__int64 *)v130;
      LODWORD(v120) = 4;
      v121.m128i_i64[0] = (__int64)v23;
      v123 = 64;
      v121.m128i_i64[1] = v108 - (_QWORD)v23;
      v122 = v116[0];
      if ( v130 != v131 )
      {
        if ( !v130 )
        {
LABEL_109:
          v130 = 40;
          goto LABEL_24;
        }
        *(_DWORD *)v130 = 4;
        *(__m128i *)(p_s2 + 1) = _mm_loadu_si128(&v121);
        v72 = v123;
        *((_DWORD *)p_s2 + 8) = v123;
        if ( v72 <= 0x40 )
        {
LABEL_117:
          p_s2[3] = v122;
          v66 = v130;
          v67 = v123;
LABEL_118:
          v130 = v66 + 40;
LABEL_119:
          if ( v67 > 0x40 && v122 )
            j_j___libc_free_0_0(v122);
          goto LABEL_24;
        }
LABEL_108:
        v65 = (__int64)(p_s2 + 3);
        p_s2 = &v122;
        sub_C43780(v65, (const void **)&v122);
        v66 = v130;
        v67 = v123;
        goto LABEL_118;
      }
LABEL_126:
      sub_EA8D20(&v129, (__int64)p_s2, (__int64)&v120);
      v67 = v123;
      goto LABEL_119;
    }
    if ( v24 == 39 )
    {
      v60 = *(_BYTE *)v23;
      v61 = v23;
      if ( *(_BYTE *)v23 == 10 || v60 == 62 )
      {
LABEL_127:
        if ( v60 == 62 )
        {
LABEL_105:
          v62 = (unsigned __int64)v61 + 1;
          v63 = *(_DWORD *)(a1 + 304);
          v108 = v62;
          sub_EA24B0(a1, v62, v63);
          sub_EABFE0(a1);
          v121.m128i_i64[0] = (__int64)v23;
          p_s2 = (__int64 *)v130;
          LODWORD(v120) = 3;
          v121.m128i_i64[1] = v62 - (_QWORD)v23;
          v123 = 64;
          v122 = 0;
          if ( v130 != v131 )
          {
            if ( !v130 )
              goto LABEL_109;
            *(_DWORD *)v130 = 3;
            *(__m128i *)(p_s2 + 1) = _mm_loadu_si128(&v121);
            v64 = v123;
            *((_DWORD *)p_s2 + 8) = v123;
            if ( v64 <= 0x40 )
              goto LABEL_117;
            goto LABEL_108;
          }
          goto LABEL_126;
        }
      }
      else
      {
        while ( v60 != 13 && v60 )
        {
          v61 = (__int64 *)((char *)v61 + (v60 == 33) + 1);
          v60 = *(_BYTE *)v61;
          if ( *(_BYTE *)v61 == 62 )
            goto LABEL_105;
          if ( v60 == 10 )
            goto LABEL_127;
        }
      }
    }
LABEL_23:
    p_s2 = &v129;
    result = sub_EBC400(a1, &v129, v20);
    if ( (_BYTE)result )
      goto LABEL_66;
LABEL_24:
    v27 = n;
    v28 = v101;
    if ( n )
    {
      v41 = s2;
      if ( !v100 )
      {
        v44 = n;
LABEL_64:
        v110 = v41;
        v113[0] = (__int64)v109;
        v45 = *a2;
        v46 = a2[1];
        v109[0] = "parameter named '";
        v111 = v44;
        v118 = v46;
        v114 = "' does not exist for macro '";
        v120 = (const char *)v116;
        v112 = 1283;
        v115 = 770;
        v116[0] = v113;
        v117 = v45;
        v119 = 1282;
        v121.m128i_i64[1] = (__int64)"'";
        LOWORD(v123) = 770;
        goto LABEL_65;
      }
      v28 = 0;
      v89 = a1;
      v42 = (__int64 *)s2;
      v43 = a2[4];
      while ( 1 )
      {
        if ( *(_QWORD *)(v43 + 8) == v27 )
        {
          p_s2 = v42;
          if ( !memcmp(*(const void **)v43, v42, v27) )
            break;
        }
        ++v28;
        v43 += 48;
        if ( v28 == v100 )
        {
          v41 = v42;
          a1 = v89;
          v44 = v27;
          goto LABEL_64;
        }
      }
      a1 = v89;
    }
    if ( v130 != v129 )
    {
      v29 = v28;
      v30 = *a3;
      v31 = a3[1];
      v32 = 0xAAAAAAAAAAAAAAABLL * (v31 - *a3);
      if ( v28 >= v32 )
      {
        v54 = v28 + 1;
        if ( v54 > v32 )
        {
          sub_EA8F90((__int64)a3, v54 - v32);
          v30 = *a3;
        }
        else if ( v54 < v32 )
        {
          v97 = &v30[3 * v54];
          if ( v31 != v97 )
          {
            v90 = v17;
            v55 = a3[1];
            v87 = a1;
            v88 = v28;
            v56 = &v30[3 * v54];
            do
            {
              v57 = v56[1];
              v58 = *v56;
              if ( v57 != *v56 )
              {
                do
                {
                  if ( *(_DWORD *)(v58 + 32) > 0x40u )
                  {
                    v59 = *(_QWORD *)(v58 + 24);
                    if ( v59 )
                      j_j___libc_free_0_0(v59);
                  }
                  v58 += 40;
                }
                while ( v57 != v58 );
                v58 = *v56;
              }
              if ( v58 )
                j_j___libc_free_0(v58, v56[2] - v58);
              v56 += 3;
            }
            while ( v55 != v56 );
            v17 = v90;
            v28 = v88;
            a1 = v87;
            v30 = *a3;
            a3[1] = v97;
          }
        }
      }
      p_s2 = &v129;
      sub_EA31A0(&v30[3 * v29], &v129);
      v35 = (unsigned int)v125;
      if ( v28 >= (unsigned int)v125 )
      {
        v51 = v28 + 1;
        if ( v51 != (unsigned int)v125 )
        {
          if ( v51 >= (unsigned int)v125 )
          {
            if ( v51 > HIDWORD(v125) )
            {
              p_s2 = (__int64 *)v126;
              sub_C8D5F0((__int64)&v124, v126, v51, 8u, v33, v34);
              v35 = (unsigned int)v125;
              v51 = v28 + 1;
            }
            v52 = &v124[8 * v35];
            for ( j = &v124[8 * v51]; j != v52; ++v52 )
            {
              if ( v52 )
                *v52 = 0;
            }
          }
          LODWORD(v125) = v28 + 1;
        }
      }
      v36 = sub_ECD690(v17);
      *(_QWORD *)&v124[8 * v29] = v36;
    }
    if ( **(_DWORD **)(a1 + 48) == 9 )
      break;
    sub_ECE2A0(a1, 26);
    v37 = v130;
    v38 = v129;
    if ( v130 != v129 )
    {
      do
      {
        if ( *(_DWORD *)(v38 + 32) > 0x40u )
        {
          v39 = *(_QWORD *)(v38 + 24);
          if ( v39 )
            j_j___libc_free_0_0(v39);
        }
        v38 += 40;
      }
      while ( v37 != v38 );
      v38 = v129;
    }
    if ( v38 )
      j_j___libc_free_0(v38, v131 - v38);
    if ( ++v101 >= v100 && v92 )
    {
      p_s2 = (__int64 *)&s2;
      s2 = "too many positional arguments";
      LOWORD(v131) = 259;
      result = sub_ECE0E0(a1, &s2, 0, 0);
      goto LABEL_41;
    }
  }
  if ( v100 )
  {
    v73 = 0;
    result = 0;
    do
    {
      if ( (*a3)[(unsigned __int64)(3 * v73) / 8 + 1] == (*a3)[(unsigned __int64)(3 * v73) / 8] )
      {
        v74 = 6 * v73;
        v75 = 6 * v73 + a2[4];
        v76 = *(_BYTE *)(v75 + 40);
        if ( v76 )
        {
          v77 = a2[1];
          v78 = *a2;
          v112 = 1283;
          v109[0] = "missing value for required parameter '";
          v110 = *(void **)v75;
          v79 = *(_QWORD *)(v75 + 8);
          v118 = v77;
          v120 = (const char *)v116;
          v121.m128i_i64[1] = (__int64)"'";
          v111 = v79;
          v113[0] = (__int64)v109;
          v114 = "' in macro '";
          v115 = 770;
          v116[0] = v113;
          v117 = v78;
          v119 = 1282;
          LOWORD(v123) = 770;
          if ( *(_QWORD *)&v124[v73] )
          {
            p_s2 = *(__int64 **)&v124[v73];
          }
          else
          {
            v107 = v76;
            v86 = sub_ECD690(v17);
            v74 = 6 * v73;
            v76 = v107;
            p_s2 = (__int64 *)v86;
          }
          v95 = v74;
          v105 = v76;
          sub_ECDA70(a1, p_s2, &v120, 0, 0);
          v75 = v95 + a2[4];
          result = v105;
        }
        if ( *(_QWORD *)(v75 + 24) != *(_QWORD *)(v75 + 16) )
        {
          p_s2 = (__int64 *)(v75 + 16);
          v104 = result;
          sub_EA31A0(&(*a3)[(unsigned __int64)(3 * v73) / 8], (__int64 *)(v75 + 16));
          result = v104;
        }
      }
      v73 += 8;
    }
    while ( 8LL * v100 != v73 );
  }
  else
  {
LABEL_156:
    result = 0;
  }
LABEL_66:
  v47 = v130;
  v48 = v129;
  if ( v130 != v129 )
  {
    v49 = result;
    do
    {
      if ( *(_DWORD *)(v48 + 32) > 0x40u )
      {
        v50 = *(_QWORD *)(v48 + 24);
        if ( v50 )
          j_j___libc_free_0_0(v50);
      }
      v48 += 40;
    }
    while ( v47 != v48 );
    v48 = v129;
    result = v49;
  }
  if ( v48 )
  {
    v103 = result;
    p_s2 = (__int64 *)(v131 - v48);
    j_j___libc_free_0(v48, v131 - v48);
    result = v103;
  }
LABEL_41:
  if ( v124 != v126 )
  {
    v102 = result;
    _libc_free(v124, p_s2);
    return v102;
  }
  return result;
}
