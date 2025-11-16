// Function: sub_320C1A0
// Address: 0x320c1a0
//
__int64 __fastcall sub_320C1A0(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int64 *v4; // r12
  _QWORD *v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  unsigned __int64 v10; // r13
  int v11; // edi
  int v12; // esi
  size_t v13; // r10
  unsigned __int64 v14; // r8
  size_t *v15; // rdi
  size_t *v16; // rax
  _QWORD *v17; // rsi
  unsigned __int64 v18; // rdi
  __int64 v19; // r15
  unsigned __int64 v20; // r12
  unsigned __int64 v21; // r13
  unsigned __int64 v22; // r14
  unsigned __int64 v23; // rdi
  _BYTE *v24; // r14
  unsigned __int64 v25; // r12
  unsigned __int64 v26; // r15
  unsigned __int64 v27; // r13
  unsigned __int64 v28; // rdi
  _BYTE *v29; // r14
  unsigned __int64 v30; // r12
  unsigned __int64 v31; // r15
  unsigned __int64 v32; // r13
  unsigned __int64 v33; // rdi
  unsigned int v34; // edx
  __int64 v35; // r14
  unsigned __int8 v36; // al
  int v37; // eax
  __int64 v38; // r12
  void (__fastcall *v39)(__int64, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD); // r13
  unsigned __int8 v40; // al
  _BYTE **v41; // rcx
  _BYTE *v42; // rsi
  unsigned __int8 v43; // al
  _BYTE **v44; // rsi
  unsigned int v45; // eax
  __int64 v46; // r8
  __int64 v47; // r9
  __int64 v48; // r12
  int v49; // eax
  __int64 v50; // rcx
  _QWORD *v51; // rax
  __int64 *v52; // rsi
  __int64 v53; // rdi
  _QWORD *v54; // rdx
  __int64 v55; // rcx
  __int64 v56; // r8
  unsigned __int8 v57; // al
  bool v59; // cc
  unsigned __int64 v60; // rdi
  unsigned __int64 v61; // rdi
  unsigned __int64 v62; // rdi
  unsigned __int64 v63; // rsi
  char v64; // al
  unsigned __int64 v65; // rdx
  size_t v66; // r10
  unsigned __int64 v67; // r14
  __int64 v68; // r11
  _QWORD *v69; // rax
  __int64 v70; // rcx
  __int64 v71; // r12
  unsigned __int8 *v72; // rax
  __int64 v73; // r14
  unsigned int v74; // esi
  __int64 v75; // r8
  __int64 v76; // r15
  __int64 *v77; // rdi
  __int64 v78; // rcx
  __int64 *v79; // rdx
  __int64 v80; // r9
  int v81; // eax
  __int64 v82; // rax
  __int64 v83; // r12
  char *v84; // r15
  _QWORD *v85; // r11
  _QWORD *v86; // rsi
  unsigned __int64 v87; // r8
  _QWORD *v88; // rcx
  unsigned __int64 v89; // rdx
  char *v90; // rax
  __int64 v91; // rax
  __int64 *v92; // r12
  __int64 v93; // r9
  unsigned int v94; // eax
  __int64 *v95; // rdx
  __int64 v96; // rdi
  unsigned int v97; // esi
  int v98; // eax
  int v99; // r11d
  __int64 *v100; // r8
  int v101; // eax
  unsigned int v102; // [rsp+18h] [rbp-198h]
  unsigned int v103; // [rsp+20h] [rbp-190h]
  _QWORD *v104; // [rsp+28h] [rbp-188h]
  char v105; // [rsp+28h] [rbp-188h]
  unsigned int v106; // [rsp+28h] [rbp-188h]
  __int64 v107; // [rsp+28h] [rbp-188h]
  size_t v108; // [rsp+28h] [rbp-188h]
  size_t n; // [rsp+30h] [rbp-180h]
  size_t na; // [rsp+30h] [rbp-180h]
  __int64 nc; // [rsp+30h] [rbp-180h]
  size_t nd; // [rsp+30h] [rbp-180h]
  size_t nb; // [rsp+30h] [rbp-180h]
  __int64 v114; // [rsp+38h] [rbp-178h]
  unsigned __int64 v115; // [rsp+38h] [rbp-178h]
  __int64 v117; // [rsp+48h] [rbp-168h] BYREF
  _BYTE *v118; // [rsp+50h] [rbp-160h] BYREF
  __int64 v119; // [rsp+58h] [rbp-158h]
  _BYTE v120[88]; // [rsp+60h] [rbp-150h] BYREF
  char *v121; // [rsp+B8h] [rbp-F8h]
  __int64 v122; // [rsp+C0h] [rbp-F0h]
  char v123; // [rsp+C8h] [rbp-E8h] BYREF
  __int128 v124; // [rsp+D0h] [rbp-E0h]
  __int64 *v125; // [rsp+E0h] [rbp-D0h] BYREF
  _BYTE *v126; // [rsp+E8h] [rbp-C8h] BYREF
  __int64 v127; // [rsp+F0h] [rbp-C0h]
  _BYTE v128[88]; // [rsp+F8h] [rbp-B8h] BYREF
  char *v129; // [rsp+150h] [rbp-60h] BYREF
  __int64 v130; // [rsp+158h] [rbp-58h]
  char v131; // [rsp+160h] [rbp-50h] BYREF
  __int64 v132; // [rsp+168h] [rbp-48h]
  int v133; // [rsp+170h] [rbp-40h]

  v4 = *(unsigned __int64 **)(a1 + 792);
  v121 = &v123;
  v118 = v120;
  v126 = v128;
  v117 = a3;
  v119 = 0x100000000LL;
  v122 = 0x100000000LL;
  v125 = (__int64 *)a2;
  v127 = 0x100000000LL;
  v129 = &v131;
  v130 = 0x100000000LL;
  v132 = 0;
  v133 = 0;
  v124 = 0;
  v5 = (_QWORD *)sub_22077B0(0xA0u);
  v10 = (unsigned __int64)v5;
  if ( v5 )
    *v5 = 0;
  v11 = v127;
  v5[1] = v125;
  v114 = (__int64)(v5 + 2);
  v104 = v5 + 4;
  v5[2] = v5 + 4;
  v5[3] = 0x100000000LL;
  if ( v11 )
    sub_320B760(v114, (__int64)&v126, v6, v7, v8, v9);
  v12 = v130;
  *(_QWORD *)(v10 + 120) = v10 + 136;
  *(_QWORD *)(v10 + 128) = 0x100000000LL;
  if ( v12 )
    sub_31F4530(v10 + 120, &v129, v6, v7, v8, v9);
  v13 = *(_QWORD *)(v10 + 8);
  *(_QWORD *)(v10 + 144) = v132;
  *(_DWORD *)(v10 + 152) = v133;
  v14 = v4[1];
  v15 = *(size_t **)(*v4 + 8 * (v13 % v14));
  if ( !v15 )
    goto LABEL_91;
  v16 = (size_t *)*v15;
  if ( v13 != *(_QWORD *)(*v15 + 8) )
  {
    do
    {
      v17 = (_QWORD *)*v16;
      if ( !*v16 )
        goto LABEL_91;
      v15 = v16;
      if ( v13 % v14 != v17[1] % v14 )
        goto LABEL_91;
      v16 = (size_t *)*v16;
    }
    while ( v13 != v17[1] );
  }
  n = *v15;
  if ( !*v15 )
  {
LABEL_91:
    v63 = v4[1];
    na = v13;
    v107 = 8 * (v13 % v14);
    v64 = sub_222DA10((__int64)(v4 + 4), v14, v4[3], 1);
    v66 = na;
    v67 = v65;
    if ( !v64 )
    {
      v68 = v107;
      v69 = *(_QWORD **)(*v4 + v107);
      if ( v69 )
      {
LABEL_93:
        *(_QWORD *)v10 = *v69;
        **(_QWORD **)(*v4 + v68) = v10;
LABEL_94:
        ++v4[3];
        n = v10;
        v105 = 1;
        goto LABEL_30;
      }
LABEL_132:
      *(_QWORD *)v10 = v4[2];
      v4[2] = v10;
      if ( *(_QWORD *)v10 )
        *(_QWORD *)(*v4 + 8 * (*(_QWORD *)(*(_QWORD *)v10 + 8LL) % v4[1])) = v10;
      *(_QWORD *)(*v4 + v68) = v4 + 2;
      goto LABEL_94;
    }
    if ( v65 == 1 )
    {
      v4[6] = 0;
      v84 = (char *)(v4 + 6);
      v85 = v4 + 6;
    }
    else
    {
      if ( v65 > 0xFFFFFFFFFFFFFFFLL )
        sub_4261EA(v4 + 4, v63, v65);
      v108 = na;
      nc = 8 * v65;
      v84 = (char *)sub_22077B0(8 * v65);
      memset(v84, 0, nc);
      v66 = v108;
      v85 = v4 + 6;
    }
    v86 = (_QWORD *)v4[2];
    v4[2] = 0;
    if ( !v86 )
    {
LABEL_129:
      if ( v85 != (_QWORD *)*v4 )
      {
        nd = v66;
        j_j___libc_free_0(*v4);
        v66 = nd;
      }
      v4[1] = v67;
      *v4 = (unsigned __int64)v84;
      v68 = 8 * (v66 % v67);
      v69 = *(_QWORD **)&v84[v68];
      if ( v69 )
        goto LABEL_93;
      goto LABEL_132;
    }
    v87 = 0;
    while ( 1 )
    {
      while ( 1 )
      {
        v88 = v86;
        v86 = (_QWORD *)*v86;
        v89 = v88[1] % v67;
        v90 = &v84[8 * v89];
        if ( !*(_QWORD *)v90 )
          break;
        *v88 = **(_QWORD **)v90;
        **(_QWORD **)v90 = v88;
LABEL_125:
        if ( !v86 )
          goto LABEL_129;
      }
      *v88 = v4[2];
      v4[2] = (unsigned __int64)v88;
      *(_QWORD *)v90 = v4 + 2;
      if ( !*v88 )
      {
        v87 = v89;
        goto LABEL_125;
      }
      *(_QWORD *)&v84[8 * v87] = v88;
      v87 = v89;
      if ( !v86 )
        goto LABEL_129;
    }
  }
  v18 = *(_QWORD *)(v10 + 120);
  if ( v10 + 136 != v18 )
    _libc_free(v18);
  v19 = *(_QWORD *)(v10 + 16);
  v20 = v19 + 88LL * *(unsigned int *)(v10 + 24);
  if ( v19 != v20 )
  {
    v115 = v10;
    do
    {
      v20 -= 88LL;
      if ( *(_BYTE *)(v20 + 80) )
      {
        v59 = *(_DWORD *)(v20 + 72) <= 0x40u;
        *(_BYTE *)(v20 + 80) = 0;
        if ( !v59 )
        {
          v62 = *(_QWORD *)(v20 + 64);
          if ( v62 )
            j_j___libc_free_0_0(v62);
        }
      }
      v21 = *(_QWORD *)(v20 + 40);
      v22 = v21 + 40LL * *(unsigned int *)(v20 + 48);
      if ( v21 != v22 )
      {
        do
        {
          v22 -= 40LL;
          v23 = *(_QWORD *)(v22 + 8);
          if ( v23 != v22 + 24 )
            _libc_free(v23);
        }
        while ( v21 != v22 );
        v21 = *(_QWORD *)(v20 + 40);
      }
      if ( v21 != v20 + 56 )
        _libc_free(v21);
      sub_C7D6A0(*(_QWORD *)(v20 + 16), 12LL * *(unsigned int *)(v20 + 32), 4);
    }
    while ( v19 != v20 );
    v10 = v115;
    v20 = *(_QWORD *)(v115 + 16);
  }
  if ( v104 != (_QWORD *)v20 )
    _libc_free(v20);
  j_j___libc_free_0(v10);
  v105 = 0;
  v114 = n + 16;
LABEL_30:
  if ( v129 != &v131 )
    _libc_free((unsigned __int64)v129);
  v24 = v126;
  v25 = (unsigned __int64)&v126[88 * (unsigned int)v127];
  if ( v126 != (_BYTE *)v25 )
  {
    do
    {
      v25 -= 88LL;
      if ( *(_BYTE *)(v25 + 80) )
      {
        v59 = *(_DWORD *)(v25 + 72) <= 0x40u;
        *(_BYTE *)(v25 + 80) = 0;
        if ( !v59 )
        {
          v60 = *(_QWORD *)(v25 + 64);
          if ( v60 )
            j_j___libc_free_0_0(v60);
        }
      }
      v26 = *(_QWORD *)(v25 + 40);
      v27 = v26 + 40LL * *(unsigned int *)(v25 + 48);
      if ( v26 != v27 )
      {
        do
        {
          v27 -= 40LL;
          v28 = *(_QWORD *)(v27 + 8);
          if ( v28 != v27 + 24 )
            _libc_free(v28);
        }
        while ( v26 != v27 );
        v26 = *(_QWORD *)(v25 + 40);
      }
      if ( v26 != v25 + 56 )
        _libc_free(v26);
      sub_C7D6A0(*(_QWORD *)(v25 + 16), 12LL * *(unsigned int *)(v25 + 32), 4);
    }
    while ( v24 != (_BYTE *)v25 );
    v25 = (unsigned __int64)v126;
  }
  if ( (_BYTE *)v25 != v128 )
    _libc_free(v25);
  if ( v121 != &v123 )
    _libc_free((unsigned __int64)v121);
  v29 = v118;
  v30 = (unsigned __int64)&v118[88 * (unsigned int)v119];
  if ( v118 != (_BYTE *)v30 )
  {
    do
    {
      v30 -= 88LL;
      if ( *(_BYTE *)(v30 + 80) )
      {
        v59 = *(_DWORD *)(v30 + 72) <= 0x40u;
        *(_BYTE *)(v30 + 80) = 0;
        if ( !v59 )
        {
          v61 = *(_QWORD *)(v30 + 64);
          if ( v61 )
            j_j___libc_free_0_0(v61);
        }
      }
      v31 = *(_QWORD *)(v30 + 40);
      v32 = v31 + 40LL * *(unsigned int *)(v30 + 48);
      if ( v31 != v32 )
      {
        do
        {
          v32 -= 40LL;
          v33 = *(_QWORD *)(v32 + 8);
          if ( v33 != v32 + 24 )
            _libc_free(v33);
        }
        while ( v31 != v32 );
        v31 = *(_QWORD *)(v30 + 40);
      }
      if ( v31 != v30 + 56 )
        _libc_free(v31);
      sub_C7D6A0(*(_QWORD *)(v30 + 16), 12LL * *(unsigned int *)(v30 + 32), 4);
    }
    while ( v29 != (_BYTE *)v30 );
    v30 = (unsigned __int64)v118;
  }
  if ( (_BYTE *)v30 != v120 )
    _libc_free(v30);
  if ( v105 )
  {
    v34 = *(_DWORD *)(*(_QWORD *)(a1 + 792) + 456LL);
    v35 = a2 - 16;
    v36 = *(_BYTE *)(a2 - 16);
    if ( (v36 & 2) != 0 )
    {
      if ( *(_DWORD *)(a2 - 24) != 2 )
        goto LABEL_63;
      v70 = *(_QWORD *)(a2 - 32);
    }
    else
    {
      if ( ((*(_WORD *)(a2 - 16) >> 6) & 0xF) != 2 )
        goto LABEL_63;
      v70 = v35 - 8LL * ((v36 >> 2) & 0xF);
    }
    v71 = *(_QWORD *)(v70 + 8);
    if ( v71 )
    {
      v72 = sub_AF34D0(*(unsigned __int8 **)v70);
      v34 = *(_DWORD *)(sub_320C1A0(a1, v71, v72) + 136);
    }
LABEL_63:
    v37 = *(_DWORD *)(a1 + 1048);
    *(_DWORD *)(a1 + 1048) = v37 + 1;
    *(_DWORD *)(n + 152) = v37;
    v38 = *(_QWORD *)(a1 + 528);
    v39 = *(void (__fastcall **)(__int64, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD))(*(_QWORD *)v38 + 728LL);
    v40 = *(_BYTE *)(a2 - 16);
    if ( (v40 & 2) != 0 )
      v41 = *(_BYTE ***)(a2 - 32);
    else
      v41 = (_BYTE **)(v35 - 8LL * ((v40 >> 2) & 0xF));
    v42 = *v41;
    if ( **v41 != 16 )
    {
      v43 = *(v42 - 16);
      if ( (v43 & 2) != 0 )
        v44 = (_BYTE **)*((_QWORD *)v42 - 4);
      else
        v44 = (_BYTE **)&v42[-8 * ((v43 >> 2) & 0xF) - 16];
      v42 = *v44;
    }
    v102 = *(unsigned __int16 *)(a2 + 2);
    v103 = *(_DWORD *)(a2 + 4);
    v106 = v34;
    v45 = sub_31FF830(a1, (unsigned __int64)v42);
    v39(v38, *(unsigned int *)(n + 152), v106, v45, v103, v102, 0);
    v48 = v117;
    *(_QWORD *)(n + 144) = v117;
    v49 = *(_DWORD *)(a1 + 1152);
    if ( !v49 )
    {
      v50 = *(unsigned int *)(a1 + 1176);
      v51 = *(_QWORD **)(a1 + 1168);
      v52 = &v51[v50];
      v53 = (8 * v50) >> 3;
      if ( !((8 * v50) >> 5) )
        goto LABEL_137;
      v54 = &v51[4 * ((8 * v50) >> 5)];
      do
      {
        if ( v48 == *v51 )
          goto LABEL_77;
        if ( v48 == v51[1] )
        {
          ++v51;
          goto LABEL_77;
        }
        if ( v48 == v51[2] )
        {
          v51 += 2;
          goto LABEL_77;
        }
        if ( v48 == v51[3] )
        {
          v51 += 3;
          goto LABEL_77;
        }
        v51 += 4;
      }
      while ( v51 != v54 );
      v53 = v52 - v51;
LABEL_137:
      if ( v53 != 2 )
      {
        if ( v53 != 3 )
        {
          if ( v53 != 1 )
            goto LABEL_140;
LABEL_151:
          if ( v48 == *v51 )
          {
LABEL_77:
            if ( v52 == v51 )
              goto LABEL_140;
LABEL_78:
            LODWORD(v118) = sub_3207610(a1, v48);
            v57 = *(_BYTE *)(a2 - 16);
            if ( (v57 & 2) != 0 )
            {
              if ( *(_DWORD *)(a2 - 24) != 2 )
              {
LABEL_80:
                sub_3200960((__int64)&v125, *(_QWORD *)(a1 + 792) + 80LL, (unsigned int *)&v118, v55, v56);
                return v114;
              }
              v73 = *(_QWORD *)(a2 - 32);
            }
            else
            {
              v55 = a2;
              if ( ((*(_WORD *)(a2 - 16) >> 6) & 0xF) != 2 )
                goto LABEL_80;
              v73 = v35 - 8LL * ((v57 >> 2) & 0xF);
            }
            if ( *(_QWORD *)(v73 + 8) )
              return v114;
            goto LABEL_80;
          }
LABEL_140:
          if ( v50 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 1180) )
          {
            sub_C8D5F0(a1 + 1168, (const void *)(a1 + 1184), v50 + 1, 8u, v46, v47);
            v52 = (__int64 *)(*(_QWORD *)(a1 + 1168) + 8LL * *(unsigned int *)(a1 + 1176));
          }
          *v52 = v48;
          v91 = (unsigned int)(*(_DWORD *)(a1 + 1176) + 1);
          *(_DWORD *)(a1 + 1176) = v91;
          if ( (unsigned int)v91 <= 4 )
          {
LABEL_143:
            v48 = v117;
            goto LABEL_78;
          }
          v92 = *(__int64 **)(a1 + 1168);
          nb = (size_t)&v92[v91];
          while ( 1 )
          {
            v97 = *(_DWORD *)(a1 + 1160);
            if ( !v97 )
              break;
            v93 = *(_QWORD *)(a1 + 1144);
            v94 = (v97 - 1) & (((unsigned int)*v92 >> 9) ^ ((unsigned int)*v92 >> 4));
            v95 = (__int64 *)(v93 + 8LL * v94);
            v96 = *v95;
            if ( *v95 != *v92 )
            {
              v99 = 1;
              v100 = 0;
              while ( v96 != -4096 )
              {
                if ( v96 == -8192 && !v100 )
                  v100 = v95;
                v94 = (v97 - 1) & (v99 + v94);
                v95 = (__int64 *)(v93 + 8LL * v94);
                v96 = *v95;
                if ( *v92 == *v95 )
                  goto LABEL_155;
                ++v99;
              }
              v101 = *(_DWORD *)(a1 + 1152);
              if ( v100 )
                v95 = v100;
              ++*(_QWORD *)(a1 + 1136);
              v98 = v101 + 1;
              v125 = v95;
              if ( 4 * v98 < 3 * v97 )
              {
                if ( v97 - *(_DWORD *)(a1 + 1156) - v98 <= v97 >> 3 )
                {
LABEL_159:
                  sub_32026C0(a1 + 1136, v97);
                  sub_31FDB70(a1 + 1136, v92, &v125);
                  v95 = v125;
                  v98 = *(_DWORD *)(a1 + 1152) + 1;
                }
                *(_DWORD *)(a1 + 1152) = v98;
                if ( *v95 != -4096 )
                  --*(_DWORD *)(a1 + 1156);
                *v95 = *v92;
                goto LABEL_155;
              }
LABEL_158:
              v97 *= 2;
              goto LABEL_159;
            }
LABEL_155:
            if ( (__int64 *)nb == ++v92 )
              goto LABEL_143;
          }
          ++*(_QWORD *)(a1 + 1136);
          v125 = 0;
          goto LABEL_158;
        }
        if ( v48 == *v51 )
          goto LABEL_77;
        ++v51;
      }
      if ( v48 != *v51 )
      {
        ++v51;
        goto LABEL_151;
      }
      goto LABEL_77;
    }
    v74 = *(_DWORD *)(a1 + 1160);
    if ( v74 )
    {
      v75 = 1;
      v76 = *(_QWORD *)(a1 + 1144);
      v77 = 0;
      LODWORD(v78) = (v74 - 1) & (((unsigned int)v48 >> 9) ^ ((unsigned int)v48 >> 4));
      v79 = (__int64 *)(v76 + 8LL * (unsigned int)v78);
      v80 = *v79;
      if ( v48 == *v79 )
        goto LABEL_78;
      while ( v80 != -4096 )
      {
        if ( v80 == -8192 && !v77 )
          v77 = v79;
        v78 = (v74 - 1) & ((_DWORD)v78 + (_DWORD)v75);
        v79 = (__int64 *)(v76 + 8 * v78);
        v80 = *v79;
        if ( v48 == *v79 )
          goto LABEL_78;
        v75 = (unsigned int)(v75 + 1);
      }
      if ( v77 )
        v79 = v77;
      v81 = v49 + 1;
      ++*(_QWORD *)(a1 + 1136);
      v125 = v79;
      if ( 4 * v81 < 3 * v74 )
      {
        if ( v74 - *(_DWORD *)(a1 + 1156) - v81 > v74 >> 3 )
        {
LABEL_114:
          *(_DWORD *)(a1 + 1152) = v81;
          if ( *v79 != -4096 )
            --*(_DWORD *)(a1 + 1156);
          *v79 = v48;
          v82 = *(unsigned int *)(a1 + 1176);
          v83 = v117;
          if ( v82 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 1180) )
          {
            sub_C8D5F0(a1 + 1168, (const void *)(a1 + 1184), v82 + 1, 8u, v75, v80);
            v82 = *(unsigned int *)(a1 + 1176);
          }
          *(_QWORD *)(*(_QWORD *)(a1 + 1168) + 8 * v82) = v83;
          ++*(_DWORD *)(a1 + 1176);
          goto LABEL_143;
        }
LABEL_173:
        sub_32026C0(a1 + 1136, v74);
        sub_31FDB70(a1 + 1136, &v117, &v125);
        v48 = v117;
        v79 = v125;
        v81 = *(_DWORD *)(a1 + 1152) + 1;
        goto LABEL_114;
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 1136);
      v125 = 0;
    }
    v74 *= 2;
    goto LABEL_173;
  }
  return v114;
}
