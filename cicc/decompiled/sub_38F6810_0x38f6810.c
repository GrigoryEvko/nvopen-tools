// Function: sub_38F6810
// Address: 0x38f6810
//
__int64 __fastcall sub_38F6810(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, int a5, int a6)
{
  unsigned __int64 *v8; // rcx
  unsigned __int64 *v9; // r14
  unsigned __int64 *v10; // rdx
  unsigned int v11; // eax
  unsigned __int64 v12; // rcx
  unsigned __int64 v13; // r13
  bool v14; // cf
  unsigned __int64 v15; // rax
  _QWORD *v16; // rax
  _QWORD *i; // rdx
  __int64 v18; // r13
  unsigned int v19; // ebx
  __int64 v20; // rax
  _DWORD *v21; // rax
  char v22; // r14
  __int64 v23; // rax
  __int64 v24; // r8
  __int64 v25; // r9
  bool v26; // zf
  _BYTE *v27; // rcx
  int v28; // eax
  __int64 result; // rax
  __int64 v30; // r8
  __int64 v31; // r9
  unsigned int v32; // r15d
  __int64 v33; // r14
  __int64 v34; // rdx
  __int64 v35; // rdi
  unsigned __int64 v36; // rax
  int v37; // r9d
  __int64 v38; // rax
  unsigned __int64 v39; // r15
  int v40; // eax
  __int64 v41; // r15
  unsigned __int64 v42; // r14
  unsigned __int64 v43; // rdi
  __int64 v44; // rdx
  unsigned int v45; // ecx
  int v46; // r15d
  __int64 v47; // r8
  __int64 v48; // r9
  unsigned int v49; // r13d
  size_t v50; // r15
  __int64 v51; // r14
  void *v52; // rbx
  __int64 v53; // rbx
  unsigned __int64 v54; // r12
  unsigned int v55; // r13d
  unsigned __int64 v56; // rdi
  unsigned __int64 v57; // rdx
  int v58; // r8d
  unsigned __int64 v59; // rsi
  unsigned __int64 *v60; // r14
  unsigned __int64 *v61; // rbx
  unsigned __int64 v62; // r13
  unsigned __int64 v63; // r12
  unsigned __int64 v64; // rdi
  char v65; // al
  char *v66; // rdx
  unsigned __int64 v67; // r14
  int v68; // edx
  __int64 v69; // rsi
  unsigned int v70; // eax
  __int64 v71; // rax
  unsigned int v72; // edx
  _QWORD *v73; // rax
  _QWORD *v74; // rdx
  int v75; // edx
  __int64 v76; // rdi
  __int64 v77; // r14
  __int64 v78; // rcx
  __int64 (*v79)(); // rax
  unsigned int v80; // eax
  int v81; // eax
  __int64 v82; // rbx
  __int64 v83; // r13
  __int64 v84; // r11
  __int64 v85; // rdx
  unsigned __int8 v86; // r10
  __int64 v87; // rsi
  unsigned __int64 *v88; // r14
  unsigned __int64 *v89; // rbx
  unsigned __int64 *v90; // r12
  unsigned __int64 v91; // r14
  unsigned __int64 v92; // r15
  unsigned __int64 v93; // rdi
  __int64 v94; // rax
  __int64 v95; // [rsp+8h] [rbp-188h]
  __int64 v96; // [rsp+10h] [rbp-180h]
  __int64 v97; // [rsp+20h] [rbp-170h]
  unsigned int v98; // [rsp+20h] [rbp-170h]
  unsigned int v99; // [rsp+28h] [rbp-168h]
  __int64 v100; // [rsp+28h] [rbp-168h]
  _BYTE *v101; // [rsp+28h] [rbp-168h]
  __int64 v102; // [rsp+28h] [rbp-168h]
  __int64 v105; // [rsp+40h] [rbp-150h]
  __int64 v106; // [rsp+40h] [rbp-150h]
  __int64 v107; // [rsp+40h] [rbp-150h]
  char v108; // [rsp+48h] [rbp-148h]
  __int64 v109; // [rsp+48h] [rbp-148h]
  __int64 v110; // [rsp+48h] [rbp-148h]
  char v111; // [rsp+50h] [rbp-140h]
  unsigned __int64 *v112; // [rsp+50h] [rbp-140h]
  bool v113; // [rsp+5Bh] [rbp-135h]
  unsigned int v114; // [rsp+5Ch] [rbp-134h]
  unsigned __int8 v115; // [rsp+5Ch] [rbp-134h]
  unsigned __int8 v116; // [rsp+5Ch] [rbp-134h]
  unsigned __int8 v117; // [rsp+5Ch] [rbp-134h]
  unsigned __int8 v118; // [rsp+5Ch] [rbp-134h]
  unsigned __int8 v119; // [rsp+5Ch] [rbp-134h]
  unsigned __int64 v120; // [rsp+68h] [rbp-128h] BYREF
  const char *v121; // [rsp+70h] [rbp-120h] BYREF
  void **p_s2; // [rsp+78h] [rbp-118h]
  __int16 v123; // [rsp+80h] [rbp-110h]
  const char **v124; // [rsp+90h] [rbp-100h] BYREF
  const char *v125; // [rsp+98h] [rbp-F8h]
  __int16 v126; // [rsp+A0h] [rbp-F0h]
  const char ***v127; // [rsp+B0h] [rbp-E0h] BYREF
  __int64 v128; // [rsp+B8h] [rbp-D8h]
  __int16 v129; // [rsp+C0h] [rbp-D0h]
  const char *v130; // [rsp+D0h] [rbp-C0h] BYREF
  __m128i v131; // [rsp+D8h] [rbp-B8h] BYREF
  unsigned __int64 v132; // [rsp+E8h] [rbp-A8h] BYREF
  unsigned int v133; // [rsp+F0h] [rbp-A0h]
  _BYTE *v134; // [rsp+100h] [rbp-90h] BYREF
  __int64 v135; // [rsp+108h] [rbp-88h]
  _BYTE v136[32]; // [rsp+110h] [rbp-80h] BYREF
  void *s2; // [rsp+130h] [rbp-60h] BYREF
  size_t n; // [rsp+138h] [rbp-58h]
  unsigned __int64 v139; // [rsp+140h] [rbp-50h] BYREF
  __int64 v140; // [rsp+148h] [rbp-48h]
  __int64 v141; // [rsp+150h] [rbp-40h]
  __int16 v142; // [rsp+158h] [rbp-38h]

  v8 = (unsigned __int64 *)a3[1];
  v9 = (unsigned __int64 *)*a3;
  v10 = v8;
  if ( a2 )
  {
    v11 = -1431655765 * ((__int64)(*(_QWORD *)(a2 + 40) - *(_QWORD *)(a2 + 32)) >> 4);
    v134 = v136;
    v12 = 0xAAAAAAAAAAAAAAABLL * (v8 - v9);
    v135 = 0x400000000LL;
    v13 = v11;
    v114 = v11;
    v113 = v11 != 0;
    v14 = v11 < v12;
    if ( v11 > v12 )
    {
      sub_38E9830((__int64)a3, v11 - v12);
      goto LABEL_13;
    }
    v15 = 0;
    if ( !v14 )
    {
LABEL_4:
      if ( v13 <= v15 )
        goto LABEL_15;
      if ( HIDWORD(v135) < v13 )
        sub_16CD150((__int64)&v134, v136, v13, 8, a5, a6);
      v16 = &v134[8 * (unsigned int)v135];
      for ( i = &v134[8 * v13]; i != v16; ++v16 )
      {
        if ( v16 )
          *v16 = 0;
      }
LABEL_14:
      LODWORD(v135) = v114;
LABEL_15:
      v108 = 0;
      if ( v114 )
        v108 = *(_BYTE *)(*(_QWORD *)(a2 + 40) - 7LL);
      goto LABEL_17;
    }
    v88 = &v9[3 * v13];
    v112 = v88;
    if ( v10 == v88 )
    {
      v15 = (unsigned int)v135;
      goto LABEL_4;
    }
    v110 = a1;
    v89 = v10;
    v90 = v88;
    do
    {
LABEL_144:
      v91 = v90[1];
      v92 = *v90;
      if ( v91 != *v90 )
      {
        do
        {
          if ( *(_DWORD *)(v92 + 32) > 0x40u )
          {
            v93 = *(_QWORD *)(v92 + 24);
            if ( v93 )
              j_j___libc_free_0_0(v93);
          }
          v92 += 40LL;
        }
        while ( v91 != v92 );
        v92 = *v90;
      }
      if ( v92 )
        j_j___libc_free_0(v92);
      v90 += 3;
    }
    while ( v90 != v89 );
    a1 = v110;
    a3[1] = (__int64)v112;
LABEL_13:
    v15 = (unsigned int)v135;
    if ( (unsigned int)v135 > v13 )
      goto LABEL_14;
    goto LABEL_4;
  }
  v134 = v136;
  v135 = 0x400000000LL;
  if ( v9 != v8 )
  {
    v110 = a1;
    v13 = 0;
    v89 = v8;
    v90 = v9;
    v112 = v9;
    v114 = 0;
    v113 = 0;
    goto LABEL_144;
  }
  v113 = 0;
  v114 = 0;
  v108 = 0;
LABEL_17:
  v111 = 0;
  v18 = a1 + 144;
  v19 = 0;
  while ( 1 )
  {
    v20 = sub_3909290(v18);
    s2 = 0;
    v105 = v20;
    v142 = 0;
    v21 = *(_DWORD **)(a1 + 152);
    n = 0;
    v139 = 0;
    v140 = 0;
    v141 = 0;
    if ( *v21 == 2 )
    {
      v131 = 0u;
      v133 = 1;
      v132 = 0;
      sub_392A3E0(v18, &v130, 1, 1);
      v46 = (int)v130;
      if ( v133 > 0x40 && v132 )
        j_j___libc_free_0_0(v132);
      if ( v46 == 27 )
      {
        if ( (unsigned __int8)sub_38F0EE0(a1, (__int64 *)&s2, v44, v45) )
        {
          v130 = "invalid argument identifier for formal argument";
          v131.m128i_i16[4] = 259;
          result = sub_3909790(a1, v105, &v130, 0, 0);
          goto LABEL_66;
        }
        if ( **(_DWORD **)(a1 + 152) != 27 )
        {
          v130 = "expected '=' after formal parameter identifier";
          v131.m128i_i16[4] = 259;
          result = sub_3909CF0(a1, &v130, 0, 0, v47, v48);
          goto LABEL_66;
        }
        v22 = 0;
        sub_38EB180(a1);
        if ( v108 )
        {
          v22 = v114 - 1 == v19;
          if ( !n )
          {
LABEL_59:
            v130 = "cannot mix positional and keyword arguments";
            v131.m128i_i16[4] = 259;
LABEL_65:
            result = sub_3909790(a1, v105, &v130, 0, 0);
            goto LABEL_66;
          }
        }
        else
        {
LABEL_22:
          if ( !n )
            goto LABEL_59;
        }
        v111 = 1;
        goto LABEL_24;
      }
    }
    v22 = 0;
    if ( v108 )
      v22 = v114 - 1 == v19;
    if ( v111 )
      goto LABEL_22;
LABEL_24:
    v23 = sub_3909290(v18);
    v26 = *(_BYTE *)(a1 + 272) == 0;
    v120 = 0;
    v27 = (_BYTE *)v23;
    if ( v26 )
      goto LABEL_27;
    v28 = **(_DWORD **)(a1 + 152);
    if ( v28 == 36 )
    {
      v101 = v27;
      sub_38EB180(a1);
      if ( sub_38EB6A0(a1, (__int64 *)&v124, (__int64)&v120) )
        goto LABEL_158;
      v75 = 0;
      v76 = *(_QWORD *)(a1 + 328);
      v77 = (__int64)v124;
      v78 = (__int64)v101;
      v79 = *(__int64 (**)())(*(_QWORD *)v76 + 72LL);
      if ( v79 != sub_168DB40 )
      {
        v81 = ((__int64 (__fastcall *)(__int64, const char ***, _QWORD, _BYTE *))v79)(v76, &v124, 0, v101);
        v78 = (__int64)v101;
        v75 = v81;
      }
      v102 = v78;
      if ( !sub_38CF2B0(v77, &v127, v75) )
      {
        v130 = "expected absolute expression";
        v131.m128i_i16[4] = 259;
        result = sub_3909790(a1, v102, &v130, 0, 0);
        goto LABEL_66;
      }
      v69 = v140;
      LODWORD(v130) = 4;
      v131.m128i_i64[0] = v102;
      v133 = 64;
      v131.m128i_i64[1] = v120 - v102;
      v132 = (unsigned __int64)v127;
      if ( v140 != v141 )
      {
        if ( !v140 )
        {
LABEL_104:
          v140 = 40;
          goto LABEL_28;
        }
        *(_DWORD *)v140 = 4;
        *(__m128i *)(v69 + 8) = _mm_loadu_si128(&v131);
        v80 = v133;
        *(_DWORD *)(v69 + 32) = v133;
        if ( v80 <= 0x40 )
        {
LABEL_121:
          *(_QWORD *)(v69 + 24) = v132;
          v71 = v140;
          v72 = v133;
LABEL_122:
          v140 = v71 + 40;
LABEL_123:
          if ( v72 > 0x40 && v132 )
            j_j___libc_free_0_0(v132);
          goto LABEL_28;
        }
LABEL_103:
        sub_16A4FD0(v69 + 24, (const void **)&v132);
        v71 = v140;
        v72 = v133;
        goto LABEL_122;
      }
LABEL_129:
      sub_38E95C0(&v139, v69, (__int64)&v130);
      v72 = v133;
      goto LABEL_123;
    }
    if ( v28 == 38 )
    {
      v65 = *v27;
      v66 = v27;
      if ( *v27 == 10 || v65 == 62 )
      {
LABEL_130:
        if ( v65 == 62 )
        {
LABEL_100:
          v67 = (unsigned __int64)(v66 + 1);
          v68 = *(_DWORD *)(a1 + 376);
          v100 = (__int64)v27;
          v120 = v67;
          sub_38E2E70(a1, v67, v68);
          sub_38EB180(a1);
          v69 = v140;
          LODWORD(v130) = 3;
          v131.m128i_i64[0] = v100;
          v131.m128i_i64[1] = v67 - v100;
          v133 = 64;
          v132 = 0;
          if ( v140 != v141 )
          {
            if ( !v140 )
              goto LABEL_104;
            *(_DWORD *)v140 = 3;
            *(__m128i *)(v69 + 8) = _mm_loadu_si128(&v131);
            v70 = v133;
            *(_DWORD *)(v69 + 32) = v133;
            if ( v70 <= 0x40 )
              goto LABEL_121;
            goto LABEL_103;
          }
          goto LABEL_129;
        }
      }
      else
      {
        while ( v65 && v65 != 13 )
        {
          v66 += (v65 == 33) + 1;
          v65 = *v66;
          if ( *v66 == 62 )
            goto LABEL_100;
          if ( v65 == 10 )
            goto LABEL_130;
        }
      }
    }
LABEL_27:
    result = sub_38F6040(a1, &v139, v22, (__int64)v27, v24, v25);
    if ( (_BYTE)result )
      goto LABEL_66;
LABEL_28:
    v32 = v19;
    if ( n )
    {
      if ( !v114 )
      {
LABEL_64:
        v123 = 1283;
        v121 = "parameter named '";
        p_s2 = &s2;
        v124 = &v121;
        v125 = "' does not exist for macro '";
        v127 = &v124;
        v126 = 770;
        v128 = a2;
        v130 = (const char *)&v127;
        v129 = 1282;
        v131.m128i_i64[0] = (__int64)"'";
        v131.m128i_i16[4] = 770;
        goto LABEL_65;
      }
      v99 = v19;
      v97 = v18;
      v49 = 0;
      v50 = n;
      v51 = *(_QWORD *)(a2 + 32);
      v52 = s2;
      while ( v50 != *(_QWORD *)(v51 + 8) || memcmp(*(const void **)v51, v52, v50) )
      {
        ++v49;
        v51 += 48;
        if ( v49 == v114 )
          goto LABEL_64;
      }
      v32 = v49;
      v19 = v99;
      v18 = v97;
    }
    if ( v140 != v139 )
    {
      v33 = v32;
      v34 = *a3;
      v35 = a3[1];
      v36 = 0xAAAAAAAAAAAAAAABLL * ((v35 - *a3) >> 3);
      if ( v32 >= v36 )
      {
        v59 = v32 + 1;
        if ( v59 > v36 )
        {
          sub_38E9830((__int64)a3, v59 - v36);
          v34 = *a3;
        }
        else if ( v59 < v36 )
        {
          v106 = v34 + 24 * v59;
          if ( v35 != v106 )
          {
            v96 = v18;
            v95 = a1;
            v60 = (unsigned __int64 *)a3[1];
            v98 = v19;
            v61 = (unsigned __int64 *)(v34 + 24 * v59);
            do
            {
              v62 = v61[1];
              v63 = *v61;
              if ( v62 != *v61 )
              {
                do
                {
                  if ( *(_DWORD *)(v63 + 32) > 0x40u )
                  {
                    v64 = *(_QWORD *)(v63 + 24);
                    if ( v64 )
                      j_j___libc_free_0_0(v64);
                  }
                  v63 += 40LL;
                }
                while ( v62 != v63 );
                v63 = *v61;
              }
              if ( v63 )
                j_j___libc_free_0(v63);
              v61 += 3;
            }
            while ( v60 != v61 );
            v33 = v32;
            v19 = v98;
            v18 = v96;
            a1 = v95;
            a3[1] = v106;
            v34 = *a3;
          }
        }
      }
      sub_38E3D90((unsigned __int64 *)(v34 + 24 * v33), &v139, v34);
      v38 = (unsigned int)v135;
      if ( v32 < (unsigned int)v135 )
        goto LABEL_32;
      v57 = v32 + 1;
      v58 = v32 + 1;
      if ( v57 < (unsigned int)v135 )
      {
        LODWORD(v135) = v32 + 1;
        goto LABEL_32;
      }
      if ( v57 <= (unsigned int)v135 )
      {
LABEL_32:
        v39 = (unsigned __int64)v134;
      }
      else
      {
        if ( v57 > HIDWORD(v135) )
        {
          sub_16CD150((__int64)&v134, v136, v57, 8, v58, v37);
          v38 = (unsigned int)v135;
          v58 = v32 + 1;
          v57 = v32 + 1;
        }
        v39 = (unsigned __int64)v134;
        v73 = &v134[8 * v38];
        v74 = &v134[8 * v57];
        if ( v73 != v74 )
        {
          do
          {
            if ( v73 )
              *v73 = 0;
            ++v73;
          }
          while ( v74 != v73 );
          v39 = (unsigned __int64)v134;
        }
        LODWORD(v135) = v58;
      }
      *(_QWORD *)(v39 + 8 * v33) = sub_3909290(v18);
    }
    v40 = **(_DWORD **)(a1 + 152);
    if ( v40 == 9 )
      break;
    if ( v40 == 25 )
      sub_38EB180(a1);
    v41 = v140;
    v42 = v139;
    if ( v140 != v139 )
    {
      do
      {
        if ( *(_DWORD *)(v42 + 32) > 0x40u )
        {
          v43 = *(_QWORD *)(v42 + 24);
          if ( v43 )
            j_j___libc_free_0_0(v43);
        }
        v42 += 40LL;
      }
      while ( v41 != v42 );
      v42 = v139;
    }
    if ( v42 )
      j_j___libc_free_0(v42);
    if ( ++v19 >= v114 && v113 )
    {
      s2 = "too many positional arguments";
      LOWORD(v139) = 259;
      result = sub_3909CF0(a1, &s2, 0, 0, v30, v31);
      goto LABEL_48;
    }
  }
  if ( v114 )
  {
    v82 = 0;
    result = 0;
    v109 = v18;
    v83 = 8LL * v114;
    do
    {
      if ( *(_QWORD *)(3 * v82 + *a3 + 8) == *(_QWORD *)(3 * v82 + *a3) )
      {
        v84 = 6 * v82;
        v85 = 6 * v82 + *(_QWORD *)(a2 + 32);
        v86 = *(_BYTE *)(v85 + 40);
        if ( v86 )
        {
          v121 = "missing value for required parameter '";
          v123 = 1283;
          v124 = &v121;
          v125 = "' in macro '";
          v127 = &v124;
          v130 = (const char *)&v127;
          v131.m128i_i64[0] = (__int64)"'";
          v131.m128i_i16[4] = 770;
          p_s2 = (void **)v85;
          v126 = 770;
          v128 = a2;
          v129 = 1282;
          if ( *(_QWORD *)&v134[v82] )
          {
            v87 = *(_QWORD *)&v134[v82];
          }
          else
          {
            v119 = v86;
            v94 = sub_3909290(v109);
            v84 = 6 * v82;
            v86 = v119;
            v87 = v94;
          }
          v107 = v84;
          v118 = v86;
          sub_3909790(a1, v87, &v130, 0, 0);
          result = v118;
          v85 = v107 + *(_QWORD *)(a2 + 32);
        }
        if ( *(_QWORD *)(v85 + 24) != *(_QWORD *)(v85 + 16) )
        {
          v117 = result;
          sub_38E3D90((unsigned __int64 *)(3 * v82 + *a3), (unsigned __int64 *)(v85 + 16), v85);
          result = v117;
        }
      }
      v82 += 8;
    }
    while ( v82 != v83 );
  }
  else
  {
LABEL_158:
    result = 0;
  }
LABEL_66:
  v53 = v140;
  v54 = v139;
  if ( v140 != v139 )
  {
    v55 = result;
    do
    {
      if ( *(_DWORD *)(v54 + 32) > 0x40u )
      {
        v56 = *(_QWORD *)(v54 + 24);
        if ( v56 )
          j_j___libc_free_0_0(v56);
      }
      v54 += 40LL;
    }
    while ( v53 != v54 );
    v54 = v139;
    result = v55;
  }
  if ( v54 )
  {
    v116 = result;
    j_j___libc_free_0(v54);
    result = v116;
  }
LABEL_48:
  if ( v134 != v136 )
  {
    v115 = result;
    _libc_free((unsigned __int64)v134);
    return v115;
  }
  return result;
}
