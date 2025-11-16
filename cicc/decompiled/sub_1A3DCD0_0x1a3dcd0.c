// Function: sub_1A3DCD0
// Address: 0x1a3dcd0
//
__int64 __fastcall sub_1A3DCD0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __m128 a7,
        double a8,
        double a9,
        double a10,
        double a11,
        double a12,
        double a13,
        __m128 a14)
{
  __int64 v19; // rax
  double v20; // xmm4_8
  double v21; // xmm5_8
  __int64 v22; // rax
  __int64 v23; // r14
  unsigned __int64 i; // r12
  unsigned int v25; // eax
  int v26; // r8d
  int v27; // r9d
  __int64 v28; // rax
  void *v29; // rax
  __int64 v30; // rdx
  const void *v31; // rsi
  unsigned __int64 v32; // rdx
  unsigned __int64 v33; // rax
  int v34; // r12d
  __int64 v35; // r13
  __int64 v36; // rax
  __int64 v37; // rdx
  int v38; // eax
  __int64 v39; // rdx
  _QWORD *v40; // rax
  _QWORD *n; // rdx
  __int64 *v42; // rsi
  int v43; // edx
  int v44; // ecx
  __int64 v45; // r9
  unsigned int v46; // edx
  __int64 **v47; // rdi
  __int64 *v48; // r8
  char v49; // r13
  double v50; // xmm4_8
  double v51; // xmm5_8
  __int64 *v52; // rdi
  __int64 v53; // r13
  __int64 *v54; // rax
  __int64 *v55; // r15
  __int64 *j; // r14
  int v57; // eax
  int v58; // edx
  __int64 v59; // r8
  unsigned int v60; // eax
  __int64 *v61; // rsi
  __int64 v62; // rdi
  __int64 v63; // rax
  __int64 *v64; // rdi
  __int64 v65; // r13
  __int64 *v66; // rax
  __int64 *v67; // r15
  __int64 *k; // r14
  int v69; // eax
  int v70; // edx
  __int64 v71; // r8
  unsigned int v72; // eax
  __int64 *v73; // rsi
  __int64 v74; // rdi
  __int64 v75; // rax
  char *v76; // r14
  __int64 *v77; // r15
  __int64 v78; // r13
  __int64 v79; // rax
  __int64 *v80; // r13
  __int64 *m; // r13
  unsigned int v82; // eax
  unsigned int v83; // ecx
  _QWORD *v84; // rdi
  unsigned int v85; // eax
  __int64 v86; // rax
  unsigned __int64 v87; // r12
  unsigned int v88; // eax
  _QWORD *v89; // rax
  __int64 v90; // rdx
  _QWORD *ii; // rdx
  __int64 v92; // r13
  __int64 v93; // r12
  _BYTE *v95; // rax
  _BYTE *v96; // rdx
  _QWORD *v97; // rax
  _BYTE *v98; // rdx
  int v99; // eax
  int v100; // edi
  int v101; // r10d
  _QWORD *v102; // rax
  int v103; // esi
  int v104; // r9d
  int v105; // esi
  int v106; // r9d
  _BYTE *v107; // rcx
  _QWORD *v108; // rax
  _BYTE *v109; // rdx
  char v111; // [rsp+18h] [rbp-F8h]
  _QWORD *v112; // [rsp+18h] [rbp-F8h]
  __int64 v113; // [rsp+20h] [rbp-F0h] BYREF
  _BYTE *v114; // [rsp+28h] [rbp-E8h]
  void *s; // [rsp+30h] [rbp-E0h]
  _BYTE v116[12]; // [rsp+38h] [rbp-D8h]
  _BYTE v117[40]; // [rsp+48h] [rbp-C8h] BYREF
  __int64 v118; // [rsp+70h] [rbp-A0h] BYREF
  _BYTE *v119; // [rsp+78h] [rbp-98h]
  _BYTE *v120; // [rsp+80h] [rbp-90h]
  __int64 v121; // [rsp+88h] [rbp-88h]
  int v122; // [rsp+90h] [rbp-80h]
  _BYTE v123[16]; // [rsp+98h] [rbp-78h] BYREF
  __int64 v124; // [rsp+A8h] [rbp-68h] BYREF
  _BYTE *v125; // [rsp+B0h] [rbp-60h]
  _BYTE *v126; // [rsp+B8h] [rbp-58h]
  __int64 v127; // [rsp+C0h] [rbp-50h]
  int v128; // [rsp+C8h] [rbp-48h]
  _BYTE v129[64]; // [rsp+D0h] [rbp-40h] BYREF

  v19 = sub_15E0530(a3);
  *(_QWORD *)(a2 + 8) = a4;
  *(_QWORD *)a2 = v19;
  *(_QWORD *)(a2 + 24) = a5;
  *(_QWORD *)(a2 + 16) = a6;
  v22 = *(_QWORD *)(a3 + 80);
  if ( !v22 )
    BUG();
  v23 = *(_QWORD *)(v22 + 24);
  for ( i = *(_QWORD *)(v22 + 16) & 0xFFFFFFFFFFFFFFF8LL; v23 != i; v23 = *(_QWORD *)(v23 + 8) )
  {
    if ( !v23 )
      BUG();
    if ( *(_BYTE *)(v23 - 8) == 53 )
    {
      v118 = v23 - 24;
      sub_1A30BB0(a2 + 32, &v118);
    }
  }
  v113 = 0;
  *(_QWORD *)v116 = 4;
  *(_DWORD *)&v116[8] = 0;
  v111 = 0;
  v114 = v117;
  s = v117;
  v25 = *(_DWORD *)(a2 + 72);
  while ( v25 )
  {
LABEL_26:
    v42 = *(__int64 **)(*(_QWORD *)(a2 + 64) + 8LL * v25 - 8);
    v43 = *(_DWORD *)(a2 + 56);
    if ( v43 )
    {
      v44 = v43 - 1;
      v45 = *(_QWORD *)(a2 + 40);
      v46 = (v43 - 1) & (((unsigned int)v42 >> 9) ^ ((unsigned int)v42 >> 4));
      v47 = (__int64 **)(v45 + 8LL * v46);
      v48 = *v47;
      if ( v42 == *v47 )
      {
LABEL_28:
        *v47 = (__int64 *)-16LL;
        v25 = *(_DWORD *)(a2 + 72);
        --*(_DWORD *)(a2 + 48);
        ++*(_DWORD *)(a2 + 52);
      }
      else
      {
        v100 = 1;
        while ( v48 != (__int64 *)-8LL )
        {
          v101 = v100 + 1;
          v46 = v44 & (v100 + v46);
          v47 = (__int64 **)(v45 + 8LL * v46);
          v48 = *v47;
          if ( v42 == *v47 )
            goto LABEL_28;
          v100 = v101;
        }
      }
    }
    *(_DWORD *)(a2 + 72) = v25 - 1;
    v49 = sub_1A3B290(a2, v42, a7, a8, a9, a10, v20, v21, a13, a14);
    v111 |= sub_1A2EF40(a2, (__int64)&v113, a7, a8, a9, a10, v50, v51, a13, a14) | v49;
    if ( *(_DWORD *)&v116[4] == *(_DWORD *)&v116[8] )
    {
LABEL_70:
      v25 = *(_DWORD *)(a2 + 72);
    }
    else
    {
      v52 = *(__int64 **)(a2 + 64);
      v53 = (__int64)&v52[*(unsigned int *)(a2 + 72)];
      v54 = sub_1A26660(v52, v53, (__int64)&v113, a2 + 32);
      v55 = v54;
      if ( (__int64 *)v53 != v54 )
      {
        for ( j = v54 + 1; (__int64 *)v53 != j; ++*(_DWORD *)(a2 + 52) )
        {
          while ( !sub_1A26350((__int64)&v113, *j) )
          {
            *v55++ = *j;
LABEL_34:
            if ( (__int64 *)v53 == ++j )
              goto LABEL_39;
          }
          v57 = *(_DWORD *)(a2 + 56);
          if ( !v57 )
            goto LABEL_34;
          v58 = v57 - 1;
          v59 = *(_QWORD *)(a2 + 40);
          v60 = (v57 - 1) & (((unsigned int)*j >> 9) ^ ((unsigned int)*j >> 4));
          v61 = (__int64 *)(v59 + 8LL * v60);
          v62 = *v61;
          if ( *j != *v61 )
          {
            v103 = 1;
            while ( v62 != -8 )
            {
              v104 = v103 + 1;
              v60 = v58 & (v103 + v60);
              v61 = (__int64 *)(v59 + 8LL * v60);
              v62 = *v61;
              if ( *j == *v61 )
                goto LABEL_38;
              v103 = v104;
            }
            goto LABEL_34;
          }
LABEL_38:
          ++j;
          *v61 = -16;
          --*(_DWORD *)(a2 + 48);
        }
      }
LABEL_39:
      v63 = *(_QWORD *)(a2 + 64);
      if ( v55 != (__int64 *)(v63 + 8LL * *(unsigned int *)(a2 + 72)) )
        *(_DWORD *)(a2 + 72) = ((__int64)v55 - v63) >> 3;
      v64 = *(__int64 **)(a2 + 352);
      v65 = (__int64)&v64[*(unsigned int *)(a2 + 360)];
      v66 = sub_1A26660(v64, v65, (__int64)&v113, a2 + 320);
      v67 = v66;
      if ( (__int64 *)v65 != v66 )
      {
        for ( k = v66 + 1; (__int64 *)v65 != k; ++*(_DWORD *)(a2 + 340) )
        {
          while ( !sub_1A26350((__int64)&v113, *k) )
          {
            *v67++ = *k;
LABEL_45:
            if ( (__int64 *)v65 == ++k )
              goto LABEL_50;
          }
          v69 = *(_DWORD *)(a2 + 344);
          if ( !v69 )
            goto LABEL_45;
          v70 = v69 - 1;
          v71 = *(_QWORD *)(a2 + 328);
          v72 = (v69 - 1) & (((unsigned int)*k >> 9) ^ ((unsigned int)*k >> 4));
          v73 = (__int64 *)(v71 + 8LL * v72);
          v74 = *v73;
          if ( *k != *v73 )
          {
            v105 = 1;
            while ( v74 != -8 )
            {
              v106 = v105 + 1;
              v72 = v70 & (v105 + v72);
              v73 = (__int64 *)(v71 + 8LL * v72);
              v74 = *v73;
              if ( *k == *v73 )
                goto LABEL_49;
              v105 = v106;
            }
            goto LABEL_45;
          }
LABEL_49:
          ++k;
          *v73 = -16;
          --*(_DWORD *)(a2 + 336);
        }
      }
LABEL_50:
      v75 = *(_QWORD *)(a2 + 352);
      if ( v67 != (__int64 *)(v75 + 8LL * *(unsigned int *)(a2 + 360)) )
        *(_DWORD *)(a2 + 360) = ((__int64)v67 - v75) >> 3;
      v76 = *(char **)(a2 + 504);
      v77 = *(__int64 **)(a2 + 496);
      v78 = (v76 - (char *)v77) >> 5;
      v79 = (v76 - (char *)v77) >> 3;
      if ( v78 > 0 )
      {
        v80 = &v77[4 * v78];
        while ( !sub_1A26350((__int64)&v113, *v77) )
        {
          if ( sub_1A26350((__int64)&v113, v77[1]) )
          {
            ++v77;
            goto LABEL_59;
          }
          if ( sub_1A26350((__int64)&v113, v77[2]) )
          {
            v77 += 2;
            goto LABEL_59;
          }
          if ( sub_1A26350((__int64)&v113, v77[3]) )
          {
            v77 += 3;
            goto LABEL_59;
          }
          v77 += 4;
          if ( v80 == v77 )
          {
            v79 = (v76 - (char *)v77) >> 3;
            goto LABEL_112;
          }
        }
        goto LABEL_59;
      }
LABEL_112:
      if ( v79 == 2 )
      {
LABEL_128:
        if ( sub_1A26350((__int64)&v113, *v77) )
          goto LABEL_59;
        ++v77;
        goto LABEL_130;
      }
      if ( v79 == 3 )
      {
        if ( sub_1A26350((__int64)&v113, *v77) )
          goto LABEL_59;
        ++v77;
        goto LABEL_128;
      }
      if ( v79 != 1 )
        goto LABEL_115;
LABEL_130:
      if ( !sub_1A26350((__int64)&v113, *v77) )
      {
LABEL_115:
        v77 = (__int64 *)v76;
        goto LABEL_64;
      }
LABEL_59:
      if ( v76 != (char *)v77 )
      {
        for ( m = v77 + 1; v76 != (char *)m; ++m )
        {
          if ( !sub_1A26350((__int64)&v113, *m) )
            *v77++ = *m;
        }
      }
LABEL_64:
      sub_1A26970(a2 + 496, (char *)v77, v76);
      ++v113;
      if ( s == v114 )
        goto LABEL_69;
      v82 = 4 * (*(_DWORD *)&v116[4] - *(_DWORD *)&v116[8]);
      if ( v82 < 0x20 )
        v82 = 32;
      if ( *(_DWORD *)v116 <= v82 )
      {
        memset(s, -1, 8LL * *(unsigned int *)v116);
LABEL_69:
        *(_QWORD *)&v116[4] = 0;
        goto LABEL_70;
      }
      sub_16CC920((__int64)&v113);
      v25 = *(_DWORD *)(a2 + 72);
    }
  }
  v111 |= sub_1A22280(a2);
  j___libc_free_0(*(_QWORD *)(a2 + 40));
  v28 = *(unsigned int *)(a2 + 344);
  *(_DWORD *)(a2 + 56) = v28;
  if ( (_DWORD)v28 )
  {
    v29 = (void *)sub_22077B0(8 * v28);
    v30 = *(unsigned int *)(a2 + 56);
    v31 = *(const void **)(a2 + 328);
    *(_QWORD *)(a2 + 40) = v29;
    *(_QWORD *)(a2 + 48) = *(_QWORD *)(a2 + 336);
    memcpy(v29, v31, 8 * v30);
  }
  else
  {
    *(_QWORD *)(a2 + 40) = 0;
    *(_QWORD *)(a2 + 48) = 0;
  }
  v32 = *(unsigned int *)(a2 + 360);
  v33 = *(unsigned int *)(a2 + 72);
  v34 = *(_DWORD *)(a2 + 360);
  if ( v32 <= v33 )
  {
    if ( *(_DWORD *)(a2 + 360) )
      memmove(*(void **)(a2 + 64), *(const void **)(a2 + 352), 8 * v32);
  }
  else
  {
    if ( v32 > *(unsigned int *)(a2 + 76) )
    {
      v35 = 0;
      *(_DWORD *)(a2 + 72) = 0;
      sub_16CD150(a2 + 64, (const void *)(a2 + 80), v32, 8, v26, v27);
      v32 = *(unsigned int *)(a2 + 360);
    }
    else
    {
      v35 = 8 * v33;
      if ( *(_DWORD *)(a2 + 72) )
      {
        memmove(*(void **)(a2 + 64), *(const void **)(a2 + 352), 8 * v33);
        v32 = *(unsigned int *)(a2 + 360);
      }
    }
    v36 = *(_QWORD *)(a2 + 352);
    v37 = 8 * v32;
    if ( v36 + v35 != v37 + v36 )
      memcpy((void *)(v35 + *(_QWORD *)(a2 + 64)), (const void *)(v36 + v35), v37 - v35);
  }
  v38 = *(_DWORD *)(a2 + 336);
  ++*(_QWORD *)(a2 + 320);
  *(_DWORD *)(a2 + 72) = v34;
  if ( !v38 )
  {
    if ( !*(_DWORD *)(a2 + 340) )
      goto LABEL_25;
    v39 = *(unsigned int *)(a2 + 344);
    if ( (unsigned int)v39 <= 0x40 )
      goto LABEL_22;
    j___libc_free_0(*(_QWORD *)(a2 + 328));
    *(_DWORD *)(a2 + 344) = 0;
LABEL_120:
    *(_QWORD *)(a2 + 328) = 0;
LABEL_24:
    *(_QWORD *)(a2 + 336) = 0;
    goto LABEL_25;
  }
  v83 = 4 * v38;
  v39 = *(unsigned int *)(a2 + 344);
  if ( (unsigned int)(4 * v38) < 0x40 )
    v83 = 64;
  if ( v83 >= (unsigned int)v39 )
  {
LABEL_22:
    v40 = *(_QWORD **)(a2 + 328);
    for ( n = &v40[v39]; n != v40; ++v40 )
      *v40 = -8;
    goto LABEL_24;
  }
  v84 = *(_QWORD **)(a2 + 328);
  v85 = v38 - 1;
  if ( v85 )
  {
    _BitScanReverse(&v85, v85);
    v86 = (unsigned int)(1 << (33 - (v85 ^ 0x1F)));
    if ( (int)v86 < 64 )
      v86 = 64;
    if ( (_DWORD)v86 == (_DWORD)v39 )
    {
      *(_QWORD *)(a2 + 336) = 0;
      v102 = &v84[v86];
      do
      {
        if ( v84 )
          *v84 = -8;
        ++v84;
      }
      while ( v102 != v84 );
      goto LABEL_25;
    }
    v87 = 4 * (int)v86 / 3u + 1;
  }
  else
  {
    v87 = 86;
  }
  j___libc_free_0(v84);
  v88 = sub_1454B60(v87);
  *(_DWORD *)(a2 + 344) = v88;
  if ( !v88 )
    goto LABEL_120;
  v89 = (_QWORD *)sub_22077B0(8LL * v88);
  v90 = *(unsigned int *)(a2 + 344);
  *(_QWORD *)(a2 + 336) = 0;
  *(_QWORD *)(a2 + 328) = v89;
  for ( ii = &v89[v90]; ii != v89; ++v89 )
  {
    if ( v89 )
      *v89 = -8;
  }
LABEL_25:
  *(_DWORD *)(a2 + 360) = 0;
  v25 = *(_DWORD *)(a2 + 72);
  if ( v25 )
    goto LABEL_26;
  v92 = a1 + 40;
  v93 = a1 + 96;
  if ( v111 )
  {
    v118 = 0;
    v125 = v129;
    v119 = v123;
    v120 = v123;
    v121 = 2;
    v122 = 0;
    v124 = 0;
    v126 = v129;
    v127 = 2;
    v128 = 0;
    v95 = sub_15CC2D0((__int64)&v118, (__int64)&unk_4F9EE48);
    if ( v120 == v119 )
      v96 = &v120[8 * HIDWORD(v121)];
    else
      v96 = &v120[8 * (unsigned int)v121];
    for ( ; v96 != v95; v95 += 8 )
    {
      if ( *(_QWORD *)v95 < 0xFFFFFFFFFFFFFFFELL )
        break;
    }
    if ( v95 == v123 )
      sub_1412190((__int64)&v118, (__int64)&unk_4F9EE50);
    v97 = sub_15CC2D0((__int64)&v124, (__int64)&unk_4F98E60);
    if ( v126 == v125 )
      v98 = &v126[8 * HIDWORD(v127)];
    else
      v98 = &v126[8 * (unsigned int)v127];
    if ( v97 == (_QWORD *)v98 )
    {
      v99 = v128;
    }
    else
    {
      *v97 = -2;
      v99 = ++v128;
    }
    if ( HIDWORD(v127) != v99 )
      goto LABEL_106;
    if ( v120 == v119 )
      v107 = &v120[8 * HIDWORD(v121)];
    else
      v107 = &v120[8 * (unsigned int)v121];
    v112 = v107;
    v108 = sub_15CC2D0((__int64)&v118, (__int64)&unk_4F9EE48);
    if ( v120 == v119 )
      v109 = &v120[8 * HIDWORD(v121)];
    else
      v109 = &v120[8 * (unsigned int)v121];
    for ( ; v109 != (_BYTE *)v108; ++v108 )
    {
      if ( *v108 < 0xFFFFFFFFFFFFFFFELL )
        break;
    }
    if ( v108 == v112 )
LABEL_106:
      sub_1412190((__int64)&v118, (__int64)&unk_4F98E60);
    sub_16CCEE0((_QWORD *)a1, v92, 2, (__int64)&v118);
    sub_16CCEE0((_QWORD *)(a1 + 56), v93, 2, (__int64)&v124);
    if ( v126 != v125 )
      _libc_free((unsigned __int64)v126);
    if ( v120 != v119 )
      _libc_free((unsigned __int64)v120);
  }
  else
  {
    *(_QWORD *)(a1 + 8) = v92;
    *(_QWORD *)a1 = 0;
    *(_QWORD *)(a1 + 16) = v92;
    *(_QWORD *)(a1 + 24) = 2;
    *(_DWORD *)(a1 + 32) = 0;
    *(_QWORD *)(a1 + 56) = 0;
    *(_QWORD *)(a1 + 64) = v93;
    *(_QWORD *)(a1 + 72) = v93;
    *(_QWORD *)(a1 + 80) = 2;
    *(_DWORD *)(a1 + 88) = 0;
    sub_1412190(a1, (__int64)&unk_4F9EE48);
  }
  if ( s != v114 )
    _libc_free((unsigned __int64)s);
  return a1;
}
