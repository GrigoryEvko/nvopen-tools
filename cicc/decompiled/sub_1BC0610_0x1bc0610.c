// Function: sub_1BC0610
// Address: 0x1bc0610
//
void __fastcall sub_1BC0610(
        __int64 a1,
        __m128 a2,
        double a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        __m128 a9,
        __int64 a10,
        __int64 a11,
        __int64 a12,
        __int64 a13,
        __int64 a14)
{
  __int64 *v14; // r13
  __int64 v15; // r14
  __int64 v16; // rax
  int v17; // ecx
  __int64 v18; // rsi
  __int64 v19; // rdi
  int v20; // ecx
  unsigned int v21; // edx
  __int64 *v22; // rax
  __int64 v23; // rbx
  __int64 v24; // r12
  _QWORD *v25; // rax
  __int64 v26; // rcx
  _QWORD *v27; // rdi
  _QWORD *v28; // rax
  __int64 v29; // rsi
  _QWORD *v30; // r10
  char *v31; // r12
  char *v32; // rbx
  __int64 v33; // rcx
  __int64 v34; // rdx
  __int64 v35; // rax
  __int64 v36; // r8
  __int64 v37; // rdi
  int v38; // r9d
  unsigned int v39; // esi
  __int64 *v40; // rdx
  __int64 v41; // r10
  __int64 v42; // r15
  __int64 *v43; // r13
  __int64 v44; // rax
  __int64 *v45; // r15
  __int64 v46; // r14
  __int64 v47; // rbx
  char *v48; // rax
  char *v49; // r12
  int v50; // r8d
  int v51; // r9d
  __int64 v52; // rax
  __int64 v53; // r13
  __int64 v54; // rbx
  __int64 *v55; // r12
  __int64 v56; // rax
  __int64 *v57; // r15
  __int64 v58; // r14
  double v59; // xmm4_8
  double v60; // xmm5_8
  _QWORD *v61; // r12
  double v62; // xmm4_8
  double v63; // xmm5_8
  _QWORD *v64; // rax
  __int64 v65; // rcx
  unsigned __int64 v66; // rdx
  unsigned int v67; // eax
  __int64 *v68; // rdx
  int v69; // eax
  __int64 v70; // rdx
  _QWORD *v71; // rax
  _QWORD *i; // rdx
  __int64 v73; // rax
  int v74; // eax
  __int64 v75; // rdx
  _QWORD *v76; // rax
  _QWORD *k; // rdx
  __int64 v78; // rax
  _QWORD *v79; // rax
  _QWORD *v80; // rax
  __int64 v81; // rsi
  _QWORD *v82; // rcx
  __int64 v83; // rax
  unsigned __int64 v84; // rax
  _QWORD *v85; // rcx
  __int64 v86; // rsi
  int v87; // edx
  int v88; // r11d
  unsigned int v89; // ecx
  _QWORD *v90; // rdi
  unsigned int v91; // eax
  int v92; // eax
  unsigned __int64 v93; // rax
  unsigned __int64 v94; // rax
  int v95; // ebx
  __int64 v96; // r12
  _QWORD *v97; // rax
  __int64 v98; // rdx
  _QWORD *m; // rdx
  unsigned int v100; // ecx
  _QWORD *v101; // rdi
  unsigned int v102; // eax
  int v103; // eax
  unsigned __int64 v104; // rax
  unsigned __int64 v105; // rax
  int v106; // ebx
  __int64 v107; // r12
  _QWORD *v108; // rax
  __int64 v109; // rdx
  _QWORD *j; // rdx
  int v111; // eax
  _QWORD *v112; // rdx
  _QWORD *v113; // rax
  _QWORD *v114; // rax
  __int64 v115; // [rsp+18h] [rbp-138h]
  __int64 **v116; // [rsp+18h] [rbp-138h]
  _QWORD *v117; // [rsp+20h] [rbp-130h]
  __int64 **v118; // [rsp+20h] [rbp-130h]
  _QWORD *v119; // [rsp+20h] [rbp-130h]
  __int64 v120; // [rsp+28h] [rbp-128h]
  __int64 v121; // [rsp+28h] [rbp-128h]
  __int64 *v123; // [rsp+38h] [rbp-118h]
  __int64 v124; // [rsp+38h] [rbp-118h]
  void *src; // [rsp+40h] [rbp-110h] BYREF
  __int64 v126; // [rsp+48h] [rbp-108h]
  _BYTE v127[64]; // [rsp+50h] [rbp-100h] BYREF
  __int64 *v128; // [rsp+90h] [rbp-C0h] BYREF
  __int64 v129; // [rsp+98h] [rbp-B8h]
  _BYTE v130[176]; // [rsp+A0h] [rbp-B0h] BYREF

  v14 = *(__int64 **)(a1 + 1112);
  v123 = *(__int64 **)(a1 + 1120);
  if ( v14 != v123 )
  {
    while ( 1 )
    {
      v15 = *v14;
      if ( (unsigned __int8)(*(_BYTE *)(*v14 + 16) - 84) > 1u )
        goto LABEL_16;
      v16 = *(_QWORD *)(a1 + 1344);
      v17 = *(_DWORD *)(v16 + 24);
      if ( !v17 )
        goto LABEL_16;
      v18 = *(_QWORD *)(v15 + 40);
      v19 = *(_QWORD *)(v16 + 8);
      v20 = v17 - 1;
      v21 = v20 & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
      v22 = (__int64 *)(v19 + 16LL * v21);
      a13 = *v22;
      if ( v18 != *v22 )
      {
        v111 = 1;
        while ( a13 != -8 )
        {
          LODWORD(a14) = v111 + 1;
          v21 = v20 & (v111 + v21);
          v22 = (__int64 *)(v19 + 16LL * v21);
          a13 = *v22;
          if ( v18 == *v22 )
            goto LABEL_5;
          v111 = a14;
        }
        goto LABEL_16;
      }
LABEL_5:
      v23 = v22[1];
      if ( !v23 )
        goto LABEL_16;
      v24 = sub_13FC520(v22[1]);
      if ( !v24 )
        goto LABEL_16;
      if ( (*(_BYTE *)(v15 + 23) & 0x40) != 0 )
      {
        v25 = *(_QWORD **)(v15 - 8);
        v26 = *v25;
        a14 = v25[3];
        if ( *(_BYTE *)(*v25 + 16LL) > 0x17u )
          goto LABEL_9;
LABEL_80:
        if ( *(_BYTE *)(a14 + 16) <= 0x17u )
          goto LABEL_87;
        v27 = *(_QWORD **)(v23 + 72);
LABEL_82:
        v80 = *(_QWORD **)(v23 + 64);
        v81 = *(_QWORD *)(a14 + 40);
        if ( v27 == v80 )
        {
          v82 = &v80[*(unsigned int *)(v23 + 84)];
          if ( v80 == v82 )
          {
            v112 = *(_QWORD **)(v23 + 64);
          }
          else
          {
            do
            {
              if ( v81 == *v80 )
                break;
              ++v80;
            }
            while ( v82 != v80 );
            v112 = v82;
          }
        }
        else
        {
          v121 = *(_QWORD *)(a14 + 40);
          v119 = &v27[*(unsigned int *)(v23 + 80)];
          v80 = sub_16CC9F0(v23 + 56, v81);
          v82 = v119;
          if ( v121 == *v80 )
          {
            v86 = *(_QWORD *)(v23 + 72);
            if ( v86 == *(_QWORD *)(v23 + 64) )
              v112 = (_QWORD *)(v86 + 8LL * *(unsigned int *)(v23 + 84));
            else
              v112 = (_QWORD *)(v86 + 8LL * *(unsigned int *)(v23 + 80));
          }
          else
          {
            v83 = *(_QWORD *)(v23 + 72);
            if ( v83 != *(_QWORD *)(v23 + 64) )
            {
              v80 = (_QWORD *)(v83 + 8LL * *(unsigned int *)(v23 + 80));
              goto LABEL_86;
            }
            v80 = (_QWORD *)(v83 + 8LL * *(unsigned int *)(v23 + 84));
            v112 = v80;
          }
        }
        while ( v112 != v80 && *v80 >= 0xFFFFFFFFFFFFFFFELL )
          ++v80;
LABEL_86:
        if ( v80 == v82 )
          goto LABEL_87;
LABEL_16:
        if ( v123 == ++v14 )
          break;
      }
      else
      {
        v79 = (_QWORD *)(v15 - 24LL * (*(_DWORD *)(v15 + 20) & 0xFFFFFFF));
        v26 = *v79;
        a14 = v79[3];
        if ( *(_BYTE *)(*v79 + 16LL) <= 0x17u )
          goto LABEL_80;
LABEL_9:
        v27 = *(_QWORD **)(v23 + 72);
        v28 = *(_QWORD **)(v23 + 64);
        v29 = *(_QWORD *)(v26 + 40);
        if ( *(_BYTE *)(a14 + 16) <= 0x17u )
          a14 = 0;
        if ( v27 == v28 )
        {
          v85 = &v27[*(unsigned int *)(v23 + 84)];
          if ( v27 == v85 )
          {
            v30 = *(_QWORD **)(v23 + 72);
          }
          else
          {
            do
            {
              if ( v29 == *v28 )
                break;
              ++v28;
            }
            while ( v85 != v28 );
            v30 = &v27[*(unsigned int *)(v23 + 84)];
          }
LABEL_105:
          while ( v85 != v28 )
          {
            if ( *v28 < 0xFFFFFFFFFFFFFFFELL )
              goto LABEL_15;
            ++v28;
          }
          if ( v28 != v30 )
            goto LABEL_16;
        }
        else
        {
          v115 = a14;
          v117 = &v27[*(unsigned int *)(v23 + 80)];
          v28 = sub_16CC9F0(v23 + 56, v29);
          v30 = v117;
          a14 = v115;
          if ( v29 == *v28 )
          {
            v27 = *(_QWORD **)(v23 + 72);
            if ( v27 == *(_QWORD **)(v23 + 64) )
              v85 = &v27[*(unsigned int *)(v23 + 84)];
            else
              v85 = &v27[*(unsigned int *)(v23 + 80)];
            goto LABEL_105;
          }
          v27 = *(_QWORD **)(v23 + 72);
          if ( v27 == *(_QWORD **)(v23 + 64) )
          {
            v28 = &v27[*(unsigned int *)(v23 + 84)];
            v85 = v28;
            goto LABEL_105;
          }
          v28 = &v27[*(unsigned int *)(v23 + 80)];
LABEL_15:
          if ( v28 != v30 )
            goto LABEL_16;
        }
        if ( a14 )
          goto LABEL_82;
LABEL_87:
        ++v14;
        v84 = sub_157EBA0(v24);
        sub_15F22F0((_QWORD *)v15, v84);
        if ( v123 == v14 )
          break;
      }
    }
  }
  src = v127;
  v126 = 0x800000000LL;
  v31 = *(char **)(a1 + 1176);
  v32 = *(char **)(a1 + 1168);
  if ( (unsigned __int64)(v31 - v32) > 0x40 )
  {
    sub_16CD150((__int64)&src, v127, (v31 - v32) >> 3, 8, a13, a14);
    v33 = (unsigned int)v126;
    v31 = *(char **)(a1 + 1176);
    v32 = *(char **)(a1 + 1168);
    if ( v32 == v31 )
      goto LABEL_27;
  }
  else
  {
    v33 = 0;
    if ( v31 == v32 )
    {
      v43 = (__int64 *)v127;
      v45 = (__int64 *)v127;
      goto LABEL_151;
    }
  }
  do
  {
    v34 = *(_QWORD *)(a1 + 1352);
    v35 = *(unsigned int *)(v34 + 48);
    if ( (_DWORD)v35 )
    {
      v36 = *(_QWORD *)v32;
      v37 = *(_QWORD *)(v34 + 32);
      v38 = v35 - 1;
      v39 = (v35 - 1) & (((unsigned int)*(_QWORD *)v32 >> 9) ^ ((unsigned int)*(_QWORD *)v32 >> 4));
      v40 = (__int64 *)(v37 + 16LL * v39);
      v41 = *v40;
      if ( *(_QWORD *)v32 == *v40 )
      {
LABEL_21:
        if ( v40 != (__int64 *)(v37 + 16 * v35) )
        {
          v42 = v40[1];
          if ( v42 )
          {
            if ( (unsigned int)v33 >= HIDWORD(v126) )
            {
              sub_16CD150((__int64)&src, v127, 0, 8, v36, v38);
              v33 = (unsigned int)v126;
            }
            *((_QWORD *)src + v33) = v42;
            v33 = (unsigned int)(v126 + 1);
            LODWORD(v126) = v126 + 1;
          }
        }
      }
      else
      {
        v87 = 1;
        while ( v41 != -8 )
        {
          v88 = v87 + 1;
          v39 = v38 & (v87 + v39);
          v40 = (__int64 *)(v37 + 16LL * v39);
          v41 = *v40;
          if ( v36 == *v40 )
            goto LABEL_21;
          v87 = v88;
        }
      }
    }
    v32 += 8;
  }
  while ( v31 != v32 );
LABEL_27:
  v43 = (__int64 *)src;
  v44 = 8LL * (unsigned int)v33;
  v45 = (__int64 *)((char *)src + v44);
  v46 = v44 >> 3;
  if ( !v44 )
  {
LABEL_151:
    v47 = 0;
    v49 = 0;
    sub_1BBBBC0(v43, v45, a1);
    goto LABEL_30;
  }
  while ( 1 )
  {
    v47 = 8 * v46;
    v48 = (char *)sub_2207800(8 * v46, &unk_435FF63);
    v49 = v48;
    if ( v48 )
      break;
    v46 >>= 1;
    if ( !v46 )
      goto LABEL_151;
  }
  sub_1BBC940(v43, v45, v48, (char *)v46, a1);
LABEL_30:
  j_j___libc_free_0(v49, v47);
  v128 = (__int64 *)v130;
  v129 = 0x1000000000LL;
  v118 = (__int64 **)src;
  v116 = (__int64 **)((char *)src + 8 * (unsigned int)v126);
  if ( src != v116 )
  {
    while ( 1 )
    {
      v52 = **v118;
      v120 = v52 + 40;
      v124 = *(_QWORD *)(v52 + 48);
      if ( v52 + 40 != v124 )
        break;
LABEL_52:
      if ( ++v118 == v116 )
        goto LABEL_53;
    }
    while ( 1 )
    {
      while ( 1 )
      {
        v53 = v124;
        v124 = *(_QWORD *)(v124 + 8);
        if ( (unsigned __int8)(*(_BYTE *)(v53 - 8) - 83) <= 1u )
          break;
LABEL_33:
        if ( v120 == v124 )
          goto LABEL_52;
      }
      v54 = v53 - 24;
      v55 = &v128[(unsigned int)v129];
      v56 = (unsigned int)v129;
      if ( v128 == v55 )
      {
LABEL_75:
        if ( (unsigned int)v56 >= HIDWORD(v129) )
        {
          sub_16CD150((__int64)&v128, v130, 0, 8, v50, v51);
          v56 = (unsigned int)v129;
        }
        v128[v56] = v54;
        LODWORD(v129) = v129 + 1;
        goto LABEL_33;
      }
      v57 = v128;
      while ( 1 )
      {
        v58 = *v57;
        if ( sub_15F41F0(v53 - 24, *v57) )
        {
          if ( sub_15CC8F0(*(_QWORD *)(a1 + 1352), *(_QWORD *)(v58 + 40), *(_QWORD *)(v53 + 16)) )
            break;
        }
        if ( v55 == ++v57 )
        {
          v56 = (unsigned int)v129;
          goto LABEL_75;
        }
      }
      v61 = (_QWORD *)(v53 - 24);
      sub_164D160(v53 - 24, v58, a2, a3, a4, a5, v59, v60, a8, a9);
      sub_15F2070((_QWORD *)(v53 - 24));
      if ( (*(_BYTE *)(v53 - 1) & 0x40) != 0 )
      {
        v64 = *(_QWORD **)(v53 - 32);
        v61 = &v64[3 * (*(_DWORD *)(v53 - 4) & 0xFFFFFFF)];
      }
      else
      {
        v64 = (_QWORD *)(v54 - 24LL * (*(_DWORD *)(v53 - 4) & 0xFFFFFFF));
      }
      for ( ; v61 != v64; v64 += 3 )
      {
        if ( *v64 )
        {
          v65 = v64[1];
          v66 = v64[2] & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v66 = v65;
          if ( v65 )
            *(_QWORD *)(v65 + 16) = *(_QWORD *)(v65 + 16) & 3LL | v66;
        }
        *v64 = 0;
      }
      v67 = *(_DWORD *)(a1 + 312);
      if ( v67 >= *(_DWORD *)(a1 + 316) )
      {
        sub_1BC04A0(a1 + 304, 0, a2, a3, a4, a5, v62, v63, a8, a9);
        v67 = *(_DWORD *)(a1 + 312);
      }
      v68 = (__int64 *)(*(_QWORD *)(a1 + 304) + 8LL * v67);
      if ( v68 )
      {
        *v68 = v54;
        v67 = *(_DWORD *)(a1 + 312);
      }
      *(_DWORD *)(a1 + 312) = v67 + 1;
      if ( v120 == v124 )
        goto LABEL_52;
    }
  }
LABEL_53:
  ++*(_QWORD *)(a1 + 1136);
  v69 = *(_DWORD *)(a1 + 1152);
  if ( !v69 )
  {
    if ( !*(_DWORD *)(a1 + 1156) )
      goto LABEL_59;
    v70 = *(unsigned int *)(a1 + 1160);
    if ( (unsigned int)v70 > 0x40 )
    {
      j___libc_free_0(*(_QWORD *)(a1 + 1144));
      *(_QWORD *)(a1 + 1144) = 0;
      *(_QWORD *)(a1 + 1152) = 0;
      *(_DWORD *)(a1 + 1160) = 0;
      goto LABEL_59;
    }
    goto LABEL_56;
  }
  v70 = *(unsigned int *)(a1 + 1160);
  v100 = 4 * v69;
  if ( (unsigned int)(4 * v69) < 0x40 )
    v100 = 64;
  if ( (unsigned int)v70 <= v100 )
  {
LABEL_56:
    v71 = *(_QWORD **)(a1 + 1144);
    for ( i = &v71[v70]; i != v71; ++v71 )
      *v71 = -8;
    *(_QWORD *)(a1 + 1152) = 0;
    goto LABEL_59;
  }
  v101 = *(_QWORD **)(a1 + 1144);
  v102 = v69 - 1;
  if ( !v102 )
  {
    v107 = 1024;
    v106 = 128;
LABEL_138:
    j___libc_free_0(v101);
    *(_DWORD *)(a1 + 1160) = v106;
    v108 = (_QWORD *)sub_22077B0(v107);
    v109 = *(unsigned int *)(a1 + 1160);
    *(_QWORD *)(a1 + 1152) = 0;
    *(_QWORD *)(a1 + 1144) = v108;
    for ( j = &v108[v109]; j != v108; ++v108 )
    {
      if ( v108 )
        *v108 = -8;
    }
    goto LABEL_59;
  }
  _BitScanReverse(&v102, v102);
  v103 = 1 << (33 - (v102 ^ 0x1F));
  if ( v103 < 64 )
    v103 = 64;
  if ( (_DWORD)v70 != v103 )
  {
    v104 = (4 * v103 / 3u + 1) | ((unsigned __int64)(4 * v103 / 3u + 1) >> 1);
    v105 = ((v104 | (v104 >> 2)) >> 4) | v104 | (v104 >> 2) | ((((v104 | (v104 >> 2)) >> 4) | v104 | (v104 >> 2)) >> 8);
    v106 = (v105 | (v105 >> 16)) + 1;
    v107 = 8 * ((v105 | (v105 >> 16)) + 1);
    goto LABEL_138;
  }
  *(_QWORD *)(a1 + 1152) = 0;
  v113 = &v101[v70];
  do
  {
    if ( v101 )
      *v101 = -8;
    ++v101;
  }
  while ( v113 != v101 );
LABEL_59:
  v73 = *(_QWORD *)(a1 + 1168);
  if ( v73 != *(_QWORD *)(a1 + 1176) )
    *(_QWORD *)(a1 + 1176) = v73;
  v74 = *(_DWORD *)(a1 + 1096);
  ++*(_QWORD *)(a1 + 1080);
  if ( !v74 )
  {
    if ( !*(_DWORD *)(a1 + 1100) )
      goto LABEL_67;
    v75 = *(unsigned int *)(a1 + 1104);
    if ( (unsigned int)v75 > 0x40 )
    {
      j___libc_free_0(*(_QWORD *)(a1 + 1088));
      *(_QWORD *)(a1 + 1088) = 0;
      *(_QWORD *)(a1 + 1096) = 0;
      *(_DWORD *)(a1 + 1104) = 0;
      goto LABEL_67;
    }
    goto LABEL_64;
  }
  v75 = *(unsigned int *)(a1 + 1104);
  v89 = 4 * v74;
  if ( (unsigned int)(4 * v74) < 0x40 )
    v89 = 64;
  if ( (unsigned int)v75 <= v89 )
  {
LABEL_64:
    v76 = *(_QWORD **)(a1 + 1088);
    for ( k = &v76[v75]; k != v76; ++v76 )
      *v76 = -8;
    *(_QWORD *)(a1 + 1096) = 0;
    goto LABEL_67;
  }
  v90 = *(_QWORD **)(a1 + 1088);
  v91 = v74 - 1;
  if ( !v91 )
  {
    v96 = 1024;
    v95 = 128;
LABEL_125:
    j___libc_free_0(v90);
    *(_DWORD *)(a1 + 1104) = v95;
    v97 = (_QWORD *)sub_22077B0(v96);
    v98 = *(unsigned int *)(a1 + 1104);
    *(_QWORD *)(a1 + 1096) = 0;
    *(_QWORD *)(a1 + 1088) = v97;
    for ( m = &v97[v98]; m != v97; ++v97 )
    {
      if ( v97 )
        *v97 = -8;
    }
    goto LABEL_67;
  }
  _BitScanReverse(&v91, v91);
  v92 = 1 << (33 - (v91 ^ 0x1F));
  if ( v92 < 64 )
    v92 = 64;
  if ( (_DWORD)v75 != v92 )
  {
    v93 = (4 * v92 / 3u + 1) | ((unsigned __int64)(4 * v92 / 3u + 1) >> 1);
    v94 = ((v93 | (v93 >> 2)) >> 4) | v93 | (v93 >> 2) | ((((v93 | (v93 >> 2)) >> 4) | v93 | (v93 >> 2)) >> 8);
    v95 = (v94 | (v94 >> 16)) + 1;
    v96 = 8 * ((v94 | (v94 >> 16)) + 1);
    goto LABEL_125;
  }
  *(_QWORD *)(a1 + 1096) = 0;
  v114 = &v90[v75];
  do
  {
    if ( v90 )
      *v90 = -8;
    ++v90;
  }
  while ( v114 != v90 );
LABEL_67:
  v78 = *(_QWORD *)(a1 + 1112);
  if ( v78 != *(_QWORD *)(a1 + 1120) )
    *(_QWORD *)(a1 + 1120) = v78;
  if ( v128 != (__int64 *)v130 )
    _libc_free((unsigned __int64)v128);
  if ( src != v127 )
    _libc_free((unsigned __int64)src);
}
