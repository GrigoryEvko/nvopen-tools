// Function: sub_2D0D040
// Address: 0x2d0d040
//
__int64 __fastcall sub_2D0D040(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // r12
  __int64 v4; // rax
  __int64 v5; // rsi
  unsigned int v6; // edx
  __int64 v7; // r13
  unsigned int v8; // edx
  __int64 v9; // rcx
  unsigned int v10; // edi
  unsigned int v11; // r9d
  __int64 *v12; // rsi
  __int64 v13; // r11
  int *v14; // rsi
  float v15; // xmm1_4
  float v16; // xmm0_4
  __int64 v17; // rcx
  int v18; // esi
  __int64 v19; // rdi
  int *v20; // rsi
  float v21; // xmm1_4
  float v22; // xmm0_4
  __int64 *v23; // rsi
  __int64 v24; // rbx
  int v25; // eax
  __int64 v26; // r10
  int v27; // ecx
  unsigned int v28; // esi
  __int64 v29; // rax
  unsigned int i; // r14d
  int v33; // edx
  unsigned int v34; // eax
  unsigned int v35; // r9d
  unsigned int v36; // esi
  __int64 v37; // r10
  int v38; // ecx
  unsigned __int64 v39; // r8
  __int64 v40; // rdx
  unsigned __int64 v41; // r8
  __int64 *v44; // rdi
  int v45; // r12d
  unsigned int v46; // r13d
  __int64 v47; // rbx
  int v48; // eax
  __int64 v49; // rax
  __int64 v50; // rax
  __int64 v51; // r14
  signed int v52; // ebx
  __int64 *v53; // rdx
  __int64 v54; // rcx
  __int64 v55; // r8
  __int64 v56; // r9
  __int64 *v57; // rax
  __int64 v58; // r9
  __int64 v59; // r13
  __int64 v60; // rax
  unsigned int v61; // ebx
  void *v62; // rax
  _BYTE *v63; // rax
  char v65; // dl
  __int64 v66; // rax
  char v67; // dl
  int v68; // ecx
  __int64 v69; // rax
  __int64 v70; // rbx
  unsigned int v71; // esi
  __int64 v72; // rbx
  __int64 v73; // r11
  int v74; // r13d
  _QWORD *v75; // rdx
  __int64 v76; // rdi
  _QWORD *v77; // rcx
  _BYTE *v78; // r8
  _QWORD *v79; // rdx
  _BYTE *v80; // rsi
  size_t v81; // rdx
  int v82; // eax
  unsigned int v83; // r9d
  __int64 v84; // r10
  int v85; // esi
  _DWORD *v86; // rax
  int v87; // edi
  int v88; // ecx
  int v89; // r8d
  int v90; // r8d
  __int64 *v91; // [rsp+10h] [rbp-2A0h]
  int v92; // [rsp+1Ch] [rbp-294h]
  char v93; // [rsp+40h] [rbp-270h]
  __int64 v94; // [rsp+48h] [rbp-268h]
  unsigned int v95; // [rsp+48h] [rbp-268h]
  bool v96; // [rsp+50h] [rbp-260h]
  __int64 *v97; // [rsp+50h] [rbp-260h]
  bool v98; // [rsp+58h] [rbp-258h]
  __int64 v99; // [rsp+58h] [rbp-258h]
  _BYTE *v100; // [rsp+68h] [rbp-248h] BYREF
  __int64 v101; // [rsp+70h] [rbp-240h] BYREF
  __int64 v102; // [rsp+78h] [rbp-238h]
  __int64 v103; // [rsp+80h] [rbp-230h]
  unsigned int v104; // [rsp+88h] [rbp-228h]
  __int64 *v105; // [rsp+90h] [rbp-220h]
  __int64 v106; // [rsp+98h] [rbp-218h]
  _QWORD *v107; // [rsp+A0h] [rbp-210h] BYREF
  int v108; // [rsp+A8h] [rbp-208h]
  unsigned int v109; // [rsp+ACh] [rbp-204h]
  unsigned int v110; // [rsp+B0h] [rbp-200h]
  unsigned __int64 v111; // [rsp+C0h] [rbp-1F0h]
  char v112; // [rsp+D4h] [rbp-1DCh]
  __int64 v113; // [rsp+120h] [rbp-190h] BYREF
  __int64 *v114; // [rsp+128h] [rbp-188h]
  __int64 v115; // [rsp+130h] [rbp-180h]
  int v116; // [rsp+138h] [rbp-178h]
  char v117; // [rsp+13Ch] [rbp-174h]
  char v118; // [rsp+140h] [rbp-170h] BYREF
  char *v119; // [rsp+1C0h] [rbp-F0h] BYREF
  __int64 v120; // [rsp+1C8h] [rbp-E8h]
  __int64 v121; // [rsp+1D0h] [rbp-E0h]
  unsigned __int64 v122; // [rsp+1D8h] [rbp-D8h]
  unsigned __int64 v123; // [rsp+1E0h] [rbp-D0h]
  __int16 v124; // [rsp+1E8h] [rbp-C8h]
  _BYTE *v125; // [rsp+1F0h] [rbp-C0h]
  __int64 v126; // [rsp+1F8h] [rbp-B8h]
  _BYTE v127[64]; // [rsp+200h] [rbp-B0h] BYREF
  __int64 v128; // [rsp+240h] [rbp-70h] BYREF
  _BYTE *v129; // [rsp+248h] [rbp-68h]
  __int64 v130; // [rsp+250h] [rbp-60h]
  int v131; // [rsp+258h] [rbp-58h]
  char v132; // [rsp+25Ch] [rbp-54h]
  _BYTE v133[80]; // [rsp+260h] [rbp-50h] BYREF

  v105 = (__int64 *)&v107;
  v1 = *(_QWORD *)a1;
  v101 = 0;
  v2 = *(_QWORD *)(v1 + 80);
  v102 = 0;
  v103 = 0;
  v104 = 0;
  v106 = 0;
  v94 = v1 + 72;
  if ( v2 == v1 + 72 )
    return sub_C7D6A0(v102, 16LL * v104, 8);
  do
  {
    v17 = *(_QWORD *)(a1 + 24);
    if ( v2 )
    {
      v4 = v2 - 24;
      v5 = (unsigned int)(*(_DWORD *)(v2 + 20) + 1);
      v6 = *(_DWORD *)(v2 + 20) + 1;
    }
    else
    {
      v4 = 0;
      v5 = 0;
      v6 = 0;
    }
    if ( v6 >= *(_DWORD *)(v17 + 32) || !*(_QWORD *)(*(_QWORD *)(v17 + 24) + 8 * v5) )
      goto LABEL_10;
    v7 = *(_QWORD *)(a1 + 48);
    v8 = *(_DWORD *)(v7 + 136);
    v9 = *(_QWORD *)(v7 + 120);
    if ( v8 )
    {
      v10 = v8 - 1;
      v11 = (v8 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
      v12 = (__int64 *)(v9 + 16LL * v11);
      v13 = *v12;
      if ( *v12 == v4 )
      {
LABEL_8:
        v14 = (int *)v12[1];
        v15 = (float)v14[1];
        v16 = (float)v14[5];
        v98 = v15 > v16;
        v96 = (float)*v14 > (float)v14[4];
        if ( v15 <= v16 && !v96 )
          goto LABEL_10;
        goto LABEL_98;
      }
      v18 = 1;
      while ( v13 != -4096 )
      {
        v89 = v18 + 1;
        v11 = v10 & (v18 + v11);
        v12 = (__int64 *)(v9 + 16LL * v11);
        v13 = *v12;
        if ( *v12 == v4 )
          goto LABEL_8;
        v18 = v89;
      }
    }
    v19 = v8;
    v20 = *(int **)(v9 + 16LL * v8 + 8);
    v21 = (float)v20[1];
    v22 = (float)v20[5];
    v98 = v21 > v22;
    v96 = (float)*v20 > (float)v20[4];
    if ( v21 <= v22 && !v96 )
      goto LABEL_10;
    if ( !v8 )
      goto LABEL_18;
    v10 = v8 - 1;
LABEL_98:
    v83 = v10 & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
    v23 = (__int64 *)(v9 + 16LL * v83);
    v84 = *v23;
    if ( *v23 != v4 )
    {
      v85 = 1;
      while ( v84 != -4096 )
      {
        v90 = v85 + 1;
        v83 = v10 & (v85 + v83);
        v23 = (__int64 *)(v9 + 16LL * v83);
        v84 = *v23;
        if ( *v23 == v4 )
          goto LABEL_19;
        v85 = v90;
      }
      v19 = v8;
LABEL_18:
      v23 = (__int64 *)(v9 + 16 * v19);
    }
LABEL_19:
    v24 = v23[1];
    v25 = *(_DWORD *)(v24 + 88);
    if ( v25 )
    {
      v26 = *(_QWORD *)(v24 + 24);
      v27 = -v25;
      v28 = (unsigned int)(v25 - 1) >> 6;
      v29 = 0;
      while ( 1 )
      {
        _RDX = *(_QWORD *)(v26 + 8 * v29);
        if ( v28 == (_DWORD)v29 )
          _RDX = (0xFFFFFFFFFFFFFFFFLL >> v27) & *(_QWORD *)(v26 + 8 * v29);
        if ( _RDX )
          break;
        if ( ++v29 == v28 + 1 )
          goto LABEL_10;
      }
      __asm { tzcnt   rdx, rdx }
      for ( i = ((_DWORD)v29 << 6) + _RDX; i != -1; i = ((_DWORD)v40 << 6) + _RAX )
      {
        v119 = *(char **)(*(_QWORD *)(v7 + 88) + 8LL * i);
        if ( sub_2D05690((_BYTE *)a1, (__int64)v119)
          && (sub_BCAC40(*((_QWORD *)v119 + 1), 1) && v98 || !sub_BCAC40(*((_QWORD *)v119 + 1), 1) && v96) )
        {
          v86 = (_DWORD *)sub_2D0CEA0((__int64)&v101, (__int64 *)&v119);
          ++*v86;
        }
        v33 = *(_DWORD *)(v24 + 88);
        v34 = i + 1;
        if ( v33 == i + 1 )
          break;
        v35 = v34 >> 6;
        v36 = (unsigned int)(v33 - 1) >> 6;
        if ( v34 >> 6 > v36 )
          break;
        v37 = *(_QWORD *)(v24 + 24);
        v38 = 64 - (v34 & 0x3F);
        v39 = 0xFFFFFFFFFFFFFFFFLL >> v38;
        v40 = v35;
        if ( v38 == 64 )
          v39 = 0;
        v41 = ~v39;
        while ( 1 )
        {
          _RAX = *(_QWORD *)(v37 + 8 * v40);
          if ( v35 == (_DWORD)v40 )
            _RAX = v41 & *(_QWORD *)(v37 + 8 * v40);
          if ( (_DWORD)v40 == v36 )
            _RAX &= 0xFFFFFFFFFFFFFFFFLL >> -(char)*(_DWORD *)(v24 + 88);
          if ( _RAX )
            break;
          if ( v36 < (unsigned int)++v40 )
            goto LABEL_10;
        }
        __asm { tzcnt   rax, rax }
      }
    }
LABEL_10:
    v2 = *(_QWORD *)(v2 + 8);
  }
  while ( v94 != v2 );
  v44 = v105;
  v91 = &v105[2 * (unsigned int)v106];
  if ( v91 != v105 )
  {
    v97 = v105;
    while ( 2 )
    {
      v45 = 0;
      v46 = 0;
      v47 = *v97;
      v48 = *((_DWORD *)v97 + 2);
      v117 = 1;
      v113 = 0;
      v92 = v48;
      v115 = 16;
      v114 = (__int64 *)&v118;
      v49 = *(_QWORD *)(a1 + 48);
      v116 = 0;
      v50 = sub_FCD870(v47, *(_QWORD *)(*(_QWORD *)v49 + 40LL) + 312LL);
      v119 = (char *)v47;
      v120 = v50;
      v124 = 0;
      v121 = 0;
      v125 = v127;
      v126 = 0x800000000LL;
      v122 = 0;
      v123 = 0;
      v128 = 0;
      v129 = v133;
      v130 = 4;
      v131 = 0;
      v132 = 1;
      v51 = *(_QWORD *)(v47 + 16);
      v52 = 0;
      v95 = 0;
      v93 = 1;
      if ( v51 )
      {
        while ( 1 )
        {
          while ( 1 )
          {
            v54 = sub_2D05370(a1, v51, 1);
            if ( v54 )
              break;
LABEL_49:
            v51 = *(_QWORD *)(v51 + 8);
            if ( !v51 )
              goto LABEL_50;
          }
          if ( !v117 )
            goto LABEL_63;
          v57 = v114;
          v53 = &v114[HIDWORD(v115)];
          if ( v114 != v53 )
          {
            while ( v54 != *v57 )
            {
              if ( v53 == ++v57 )
                goto LABEL_80;
            }
            goto LABEL_49;
          }
LABEL_80:
          if ( HIDWORD(v115) < (unsigned int)v115 )
          {
            ++HIDWORD(v115);
            *v53 = v54;
            v66 = (__int64)v119;
            ++v113;
            v67 = *v119;
            if ( (unsigned __int8)*v119 > 0x1Cu )
              goto LABEL_65;
LABEL_82:
            if ( v67 != 22 )
              BUG();
            v69 = *(_QWORD *)(*(_QWORD *)(v66 + 24) + 80LL);
            if ( v69 && v54 == v69 - 24 )
              goto LABEL_85;
LABEL_66:
            sub_2D08230((__int64)&v107, a1, (__int64)&v119, v54, 1);
            v68 = HIDWORD(v107);
            if ( (int)v107 < 0 || SHIDWORD(v107) < 0 )
              goto LABEL_72;
            if ( !v107 )
            {
              v68 = 0;
LABEL_72:
              ++v95;
              goto LABEL_73;
            }
            if ( v108 > *(_DWORD *)(a1 + 64) || v109 > dword_50158A8 || dword_50157C8 < v110 )
              goto LABEL_72;
LABEL_73:
            v46 += v110 + v108 * v109;
            if ( v45 < (int)v107 )
              v45 = (int)v107;
            if ( v52 < v68 )
              v52 = v68;
            if ( v112 )
              goto LABEL_49;
            _libc_free(v111);
            v51 = *(_QWORD *)(v51 + 8);
            if ( !v51 )
              break;
          }
          else
          {
LABEL_63:
            v99 = v54;
            sub_C8CC70((__int64)&v113, v54, (__int64)v53, v54, v55, v56);
            v54 = v99;
            if ( !v65 )
              goto LABEL_49;
            v66 = (__int64)v119;
            v67 = *v119;
            if ( (unsigned __int8)*v119 <= 0x1Cu )
              goto LABEL_82;
LABEL_65:
            if ( v54 != *(_QWORD *)(v66 + 40) )
              goto LABEL_66;
LABEL_85:
            v51 = *(_QWORD *)(v51 + 8);
            v93 = 0;
            if ( !v51 )
              break;
          }
        }
      }
LABEL_50:
      v123 = __PAIR64__(v52, v45);
      LODWORD(v121) = v92;
      HIDWORD(v121) = HIDWORD(v115) - v116;
      v122 = __PAIR64__(v46, v95);
      LOBYTE(v124) = v93;
      v59 = sub_22077B0(0xD0u);
      v60 = (__int64)v119;
      *(_QWORD *)(v59 + 72) = 0x800000000LL;
      v61 = v126;
      *(_QWORD *)(v59 + 16) = v60;
      *(_QWORD *)(v59 + 24) = v120;
      *(_QWORD *)(v59 + 32) = v121;
      *(_QWORD *)(v59 + 40) = v122;
      *(_QWORD *)(v59 + 48) = v123;
      *(_WORD *)(v59 + 56) = v124;
      v62 = (void *)(v59 + 80);
      *(_QWORD *)(v59 + 64) = v59 + 80;
      if ( v61 )
      {
        if ( v125 == v127 )
        {
          v80 = v127;
          v81 = 8LL * v61;
          if ( v61 <= 8
            || (sub_C8D5F0(v59 + 64, (const void *)(v59 + 80), v61, 8u, v61, v58),
                v62 = *(void **)(v59 + 64),
                v80 = v125,
                (v81 = 8LL * (unsigned int)v126) != 0) )
          {
            memcpy(v62, v80, v81);
          }
          *(_DWORD *)(v59 + 72) = v61;
          LODWORD(v126) = 0;
        }
        else
        {
          v82 = HIDWORD(v126);
          *(_QWORD *)(v59 + 64) = v125;
          *(_DWORD *)(v59 + 72) = v61;
          *(_DWORD *)(v59 + 76) = v82;
          v126 = 0;
          v125 = v127;
        }
      }
      sub_C8CF70(v59 + 144, (void *)(v59 + 176), 4, (__int64)v133, (__int64)&v128);
      sub_2208C80((_QWORD *)v59, a1 + 104);
      v63 = v119;
      ++*(_QWORD *)(a1 + 120);
      if ( *v63 <= 0x1Cu )
        goto LABEL_52;
      v70 = *(_QWORD *)(a1 + 112);
      v71 = *(_DWORD *)(a1 + 96);
      v100 = v63;
      v72 = v70 + 16;
      if ( v71 )
      {
        v73 = *(_QWORD *)(a1 + 80);
        v74 = 1;
        v75 = 0;
        LODWORD(v76) = (v71 - 1) & (((unsigned int)v63 >> 9) ^ ((unsigned int)v63 >> 4));
        v77 = (_QWORD *)(v73 + 16LL * (unsigned int)v76);
        v78 = (_BYTE *)*v77;
        if ( v63 == (_BYTE *)*v77 )
        {
LABEL_89:
          v79 = v77 + 1;
          goto LABEL_90;
        }
        while ( v78 != (_BYTE *)-4096LL )
        {
          if ( !v75 && v78 == (_BYTE *)-8192LL )
            v75 = v77;
          v76 = (v71 - 1) & ((_DWORD)v76 + v74);
          v77 = (_QWORD *)(v73 + 16 * v76);
          v78 = (_BYTE *)*v77;
          if ( v63 == (_BYTE *)*v77 )
            goto LABEL_89;
          ++v74;
        }
        v87 = *(_DWORD *)(a1 + 88);
        if ( !v75 )
          v75 = v77;
        ++*(_QWORD *)(a1 + 72);
        v88 = v87 + 1;
        v107 = v75;
        if ( 4 * (v87 + 1) < 3 * v71 )
        {
          if ( v71 - *(_DWORD *)(a1 + 92) - v88 > v71 >> 3 )
          {
LABEL_120:
            *(_DWORD *)(a1 + 88) = v88;
            if ( *v75 != -4096 )
              --*(_DWORD *)(a1 + 92);
            *v75 = v63;
            v79 = v75 + 1;
            *v79 = 0;
LABEL_90:
            *v79 = v72;
LABEL_52:
            if ( !v132 )
              _libc_free((unsigned __int64)v129);
            if ( v125 != v127 )
              _libc_free((unsigned __int64)v125);
            if ( !v117 )
              _libc_free((unsigned __int64)v114);
            v97 += 2;
            if ( v91 == v97 )
            {
              v44 = v105;
              goto LABEL_60;
            }
            continue;
          }
LABEL_125:
          sub_2D08C60(a1 + 72, v71);
          sub_2D06590(a1 + 72, (__int64 *)&v100, &v107);
          v63 = v100;
          v75 = v107;
          v88 = *(_DWORD *)(a1 + 88) + 1;
          goto LABEL_120;
        }
      }
      else
      {
        ++*(_QWORD *)(a1 + 72);
        v107 = 0;
      }
      break;
    }
    v71 *= 2;
    goto LABEL_125;
  }
LABEL_60:
  if ( v44 != (__int64 *)&v107 )
    _libc_free((unsigned __int64)v44);
  return sub_C7D6A0(v102, 16LL * v104, 8);
}
