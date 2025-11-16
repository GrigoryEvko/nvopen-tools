// Function: sub_2B52070
// Address: 0x2b52070
//
__int64 __fastcall sub_2B52070(__int64 a1, __int64 *a2, unsigned __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v7; // rbx
  __int64 v8; // r13
  __int64 i; // r15
  __int64 v10; // rax
  unsigned __int64 v11; // rdx
  __int64 v12; // rax
  unsigned __int8 *v13; // rbx
  int v14; // eax
  __int64 v15; // rsi
  int v16; // eax
  unsigned int v17; // esi
  char v18; // al
  __int64 v19; // r8
  unsigned __int64 v20; // r9
  __int64 v21; // rax
  __int64 v22; // r9
  __int64 v23; // rdx
  __int64 v24; // rdx
  __int64 v25; // r8
  __int64 v26; // r9
  __int64 *v27; // r13
  __int64 v28; // rdx
  __int64 *v29; // r15
  __int64 v30; // rcx
  __int64 v31; // r8
  __int64 v32; // r9
  __int64 *v33; // r13
  __int64 *v34; // r15
  unsigned __int64 v35; // rdi
  __int64 v36; // rax
  unsigned int v37; // ecx
  int v38; // edi
  unsigned int v39; // esi
  bool v40; // r13
  __int64 v41; // r8
  __int64 v42; // r9
  int v43; // eax
  int v44; // r9d
  __int64 v45; // rax
  __int64 v46; // rax
  __int64 v47; // r8
  __int64 *v48; // r9
  __int64 **j; // rax
  __int64 **v50; // rdi
  int *v51; // rdx
  int *k; // rsi
  __int64 v53; // rax
  __int64 *v54; // rcx
  __int64 *v55; // rax
  __int64 v56; // rdi
  int *v57; // rdx
  int *v58; // rsi
  __int64 v59; // rax
  __int64 *v60; // rcx
  __int64 *v61; // rax
  __int64 v62; // rdi
  _DWORD *v63; // rcx
  __int64 v64; // rdx
  _DWORD *v65; // rdi
  __int64 v66; // rsi
  __int64 v67; // rdx
  _DWORD *v68; // rax
  _DWORD *v69; // rdx
  __int64 **v70; // rdi
  unsigned __int64 v71; // rax
  __int64 v72; // rdx
  __int64 **v73; // rcx
  __int64 *v74; // rdi
  __int64 *v75; // rbx
  __int64 *v76; // r12
  unsigned __int64 v77; // rdi
  unsigned __int64 *v78; // rbx
  unsigned __int64 *v79; // r12
  unsigned __int64 v80; // rdi
  __int64 *v82; // r8
  __int64 m; // rax
  __int64 *v84; // rax
  int *v85; // rdx
  int *ii; // r9
  __int64 v87; // rax
  __int64 *v88; // rcx
  __int64 *v89; // rax
  __int64 v90; // rsi
  __int64 v91; // rcx
  __int64 v92; // r8
  __int64 v93; // r9
  __int64 **v94; // r13
  __int64 **v95; // r15
  unsigned __int64 v96; // rdi
  char *v97; // rdi
  __int64 **v98; // r8
  __int64 **v99; // rax
  __int64 v100; // rax
  unsigned __int64 v101; // rdx
  size_t n; // [rsp+8h] [rbp-1D8h]
  int na; // [rsp+8h] [rbp-1D8h]
  size_t nb; // [rsp+8h] [rbp-1D8h]
  unsigned int v105; // [rsp+10h] [rbp-1D0h]
  char v106; // [rsp+10h] [rbp-1D0h]
  char v107; // [rsp+10h] [rbp-1D0h]
  __int64 v108; // [rsp+10h] [rbp-1D0h]
  __int64 v109; // [rsp+10h] [rbp-1D0h]
  __int64 v110; // [rsp+10h] [rbp-1D0h]
  __int64 v111; // [rsp+10h] [rbp-1D0h]
  __int64 v114; // [rsp+50h] [rbp-190h]
  __int64 v115; // [rsp+58h] [rbp-188h] BYREF
  __int64 v116; // [rsp+60h] [rbp-180h] BYREF
  __int64 v117; // [rsp+68h] [rbp-178h]
  __int64 v118; // [rsp+70h] [rbp-170h]
  unsigned int v119; // [rsp+78h] [rbp-168h]
  unsigned __int64 *v120; // [rsp+80h] [rbp-160h] BYREF
  __int64 v121; // [rsp+88h] [rbp-158h]
  int *v122; // [rsp+90h] [rbp-150h] BYREF
  __int64 v123; // [rsp+98h] [rbp-148h]
  _BYTE v124[48]; // [rsp+A0h] [rbp-140h] BYREF
  void *src; // [rsp+D0h] [rbp-110h] BYREF
  __int64 v126; // [rsp+D8h] [rbp-108h]
  _BYTE v127[48]; // [rsp+E0h] [rbp-100h] BYREF
  __int64 **v128; // [rsp+110h] [rbp-D0h] BYREF
  __int64 v129; // [rsp+118h] [rbp-C8h]
  __int64 *v130[6]; // [rsp+120h] [rbp-C0h] BYREF
  __int64 *v131; // [rsp+150h] [rbp-90h] BYREF
  __int64 v132; // [rsp+158h] [rbp-88h]
  _BYTE v133[128]; // [rsp+160h] [rbp-80h] BYREF

  v7 = a3;
  v120 = (unsigned __int64 *)&v122;
  v122 = (int *)v124;
  v116 = 0;
  v117 = 0;
  v118 = 0;
  v119 = 0;
  v121 = 0;
  v123 = 0xC00000000LL;
  if ( (int)a3 > 0 )
  {
    v8 = (unsigned int)(a3 - 1);
    for ( i = 0; ; i = v12 )
    {
      v13 = (unsigned __int8 *)a2[i];
      v14 = *v13;
      if ( (_BYTE)v14 != 90 )
        break;
      v15 = *(_QWORD *)(*((_QWORD *)v13 - 8) + 8LL);
      if ( *(_BYTE *)(v15 + 8) != 17 )
        goto LABEL_7;
      v16 = **((unsigned __int8 **)v13 - 4);
      if ( (_BYTE)v16 != 17 && (unsigned int)(v16 - 12) > 1 )
        goto LABEL_7;
      v131 = (__int64 *)sub_2B15730(a2[i]);
      if ( !BYTE4(v131) )
        goto LABEL_4;
      a6 = (unsigned int)v131;
      v17 = *(_DWORD *)(v15 + 32);
      if ( (unsigned int)v131 >= v17 )
        goto LABEL_4;
      v105 = (unsigned int)v131;
      sub_B48880((__int64 *)&src, v17, 1u);
      if ( ((unsigned __int8)src & 1) != 0 )
        src = (void *)(2
                     * (((unsigned __int64)src >> 58 << 57)
                      | ~(-1LL << ((unsigned __int64)src >> 58)) & ((unsigned __int64)src >> 1) & ~(1LL << v105))
                     + 1);
      else
        *(_QWORD *)(*(_QWORD *)src + 8LL * (v105 >> 6)) &= ~(1LL << v105);
      sub_2B25A00(&v128, *((char **)v13 - 8), (unsigned __int64 *)&src);
      v18 = sub_2B0D9E0((unsigned __int64)v128);
      if ( (v20 & 1) == 0 && v20 )
      {
        if ( *(_QWORD *)v20 != v20 + 16 )
        {
          n = v20;
          v106 = v18;
          _libc_free(*(_QWORD *)v20);
          v20 = n;
          v18 = v106;
        }
        v107 = v18;
        j_j___libc_free_0(v20);
        v18 = v107;
      }
      if ( v18 )
      {
        v100 = (unsigned int)v123;
        v101 = (unsigned int)v123 + 1LL;
        if ( v101 > HIDWORD(v123) )
        {
          sub_C8D5F0((__int64)&v122, v124, v101, 4u, v19, v20);
          v100 = (unsigned int)v123;
        }
        v122[v100] = i;
        LODWORD(v123) = v123 + 1;
        sub_228BF40((unsigned __int64 **)&src);
        goto LABEL_7;
      }
      v128 = (__int64 **)*((_QWORD *)v13 - 8);
      v21 = sub_2B51DE0((__int64)&v116, (__int64 *)&v128);
      v23 = *(unsigned int *)(v21 + 8);
      if ( v23 + 1 > (unsigned __int64)*(unsigned int *)(v21 + 12) )
      {
        v111 = v21;
        sub_C8D5F0(v21, (const void *)(v21 + 16), v23 + 1, 4u, v23 + 1, v22);
        v21 = v111;
        v23 = *(unsigned int *)(v111 + 8);
      }
      *(_DWORD *)(*(_QWORD *)v21 + 4 * v23) = i;
      ++*(_DWORD *)(v21 + 8);
      sub_228BF40((unsigned __int64 **)&src);
      v12 = i + 1;
      if ( v8 == i )
      {
LABEL_26:
        v7 = a3;
        sub_2B3FDA0((__int64)&v116);
        goto LABEL_27;
      }
LABEL_8:
      ;
    }
    if ( (unsigned int)(v14 - 12) <= 1 )
    {
LABEL_4:
      v10 = (unsigned int)v123;
      v11 = (unsigned int)v123 + 1LL;
      if ( v11 > HIDWORD(v123) )
      {
        sub_C8D5F0((__int64)&v122, v124, v11, 4u, a5, a6);
        v10 = (unsigned int)v123;
      }
      v122[v10] = i;
      LODWORD(v123) = v123 + 1;
    }
LABEL_7:
    v12 = i + 1;
    if ( v8 == i )
      goto LABEL_26;
    goto LABEL_8;
  }
  sub_2B3FDA0((__int64)&v116);
LABEL_27:
  v129 = 0;
  v128 = v130;
  if ( !(_DWORD)v121 )
  {
    v131 = (__int64 *)v133;
    v132 = 0x100000000LL;
    goto LABEL_29;
  }
  sub_2B443C0((__int64)&v128, (__int64)&v120, v24, (unsigned int)v121, v25, v26);
  v131 = (__int64 *)v133;
  v132 = 0x100000000LL;
  if ( (_DWORD)v129 )
  {
    sub_2B443C0((__int64)&v131, (__int64)&v128, (unsigned int)v129, v91, v92, v93);
    v94 = v128;
    v95 = &v128[9 * (unsigned int)v129];
    if ( v128 == v95 )
      goto LABEL_115;
    do
    {
      v95 -= 9;
      v96 = (unsigned __int64)v95[1];
      if ( (__int64 **)v96 != v95 + 3 )
        _libc_free(v96);
    }
    while ( v94 != v95 );
  }
  v94 = v128;
LABEL_115:
  if ( v94 != v130 )
    _libc_free((unsigned __int64)v94);
LABEL_29:
  v27 = v131;
  v28 = 9LL * (unsigned int)v132;
  v29 = &v131[v28];
  sub_2B48560((__int64 *)&v128, v131, 0x8E38E38E38E38E39LL * ((v28 * 8) >> 3));
  if ( v130[0] )
    sub_2B18B80(v27, (__int64)v29, v130[0], v129, v31);
  else
    sub_2B12AA0((__int64)v27, (__int64)v29, 0, v30, v31, v32);
  v33 = v130[0];
  v34 = &v130[0][9 * v129];
  if ( v130[0] != v34 )
  {
    do
    {
      v35 = v33[1];
      if ( (__int64 *)v35 != v33 + 3 )
        _libc_free(v35);
      v33 += 9;
    }
    while ( v34 != v33 );
    v34 = v130[0];
  }
  j_j___libc_free_0((unsigned __int64)v34);
  v36 = (unsigned int)v132;
  if ( (_DWORD)v132
    && ((v37 = v123 + *((_DWORD *)v131 + 4), v38 = v37, (_DWORD)v132 != 1)
      ? (v39 = *((_DWORD *)v131 + 22) + v37, v38 = v39 | v37)
      : (v39 = 0),
        v38) )
  {
    v40 = v37 != 0 && v39 <= v37;
  }
  else
  {
    if ( !(_DWORD)v123 )
    {
      BYTE4(v115) = 0;
      goto LABEL_78;
    }
    v40 = 0;
  }
  v41 = 8 * v7;
  src = v127;
  v126 = 0x600000000LL;
  v42 = (__int64)(8 * v7) >> 3;
  if ( 8 * v7 > 0x30 )
  {
    sub_C8D5F0((__int64)&src, v127, (__int64)(8 * v7) >> 3, 8u, v41, v42);
    v42 = (__int64)(8 * v7) >> 3;
    v41 = 8 * v7;
    v97 = (char *)src + 8 * (unsigned int)v126;
  }
  else
  {
    v43 = 0;
    if ( !v41 )
      goto LABEL_45;
    v97 = v127;
  }
  na = v42;
  v109 = v41;
  memcpy(v97, a2, v41);
  v43 = v126;
  LODWORD(v42) = na;
  v41 = v109;
LABEL_45:
  v44 = v43 + v42;
  v45 = *a2;
  v108 = v41;
  LODWORD(v126) = v44;
  v46 = sub_ACADE0(*(__int64 ***)(v45 + 8));
  v47 = v108;
  v48 = (__int64 *)v46;
  v128 = v130;
  v129 = 0x600000000LL;
  if ( v7 > 6 )
  {
    nb = v108;
    v110 = v46;
    sub_C8D5F0((__int64)&v128, v130, v7, 8u, v47, v46);
    v50 = v128;
    v98 = (__int64 **)((char *)v128 + nb);
    v99 = v128;
    if ( v128 != (__int64 **)((char *)v128 + nb) )
    {
      do
        *v99++ = (__int64 *)v110;
      while ( v98 != v99 );
      v50 = v128;
    }
    LODWORD(v129) = v7;
  }
  else
  {
    if ( v7 )
    {
      for ( j = v130; (__int64 **)((char *)v130 + v108) != j; ++j )
        *j = v48;
    }
    LODWORD(v129) = v7;
    v50 = v128;
  }
  if ( v40 )
  {
    v51 = (int *)v131[1];
    for ( k = &v51[*((unsigned int *)v131 + 4)]; k != v51; v50 = v128 )
    {
      v53 = *v51++;
      v53 *= 8;
      v54 = (__int64 *)((char *)a2 + v53);
      v55 = (__int64 *)((char *)v50 + v53);
      v56 = *v55;
      *v55 = *v54;
      *v54 = v56;
    }
  }
  else if ( (_DWORD)v132 )
  {
    v82 = &v115;
    v115 = 0x100000000LL;
    for ( m = 0; ; m = *(unsigned int *)v82 )
    {
      v84 = &v131[9 * m];
      v85 = (int *)v84[1];
      for ( ii = &v85[*((unsigned int *)v84 + 4)]; ii != v85; *v88 = v90 )
      {
        v87 = *v85++;
        v87 *= 8;
        v88 = (__int64 *)((char *)a2 + v87);
        v89 = (__int64 *)((char *)v50 + v87);
        v90 = *v89;
        *v89 = *v88;
        v50 = v128;
      }
      v82 = (__int64 *)((char *)v82 + 4);
      if ( &v116 == v82 )
        break;
    }
  }
  v57 = v122;
  v58 = &v122[(unsigned int)v123];
  if ( v58 != v122 )
  {
    do
    {
      v59 = *v57++;
      v59 *= 8;
      v60 = (__int64 *)((char *)a2 + v59);
      v61 = (__int64 *)((char *)v50 + v59);
      v62 = *v61;
      *v61 = *v60;
      *v60 = v62;
      v50 = v128;
    }
    while ( v58 != v57 );
  }
  v114 = sub_2B25EA0((_BYTE **)v50, (unsigned int)v129, a4);
  if ( !BYTE4(v114) )
    goto LABEL_100;
  v63 = *(_DWORD **)a4;
  v64 = 4LL * *(unsigned int *)(a4 + 8);
  v65 = (_DWORD *)(*(_QWORD *)a4 + v64);
  v66 = v64 >> 2;
  v67 = v64 >> 4;
  if ( v67 )
  {
    v68 = *(_DWORD **)a4;
    v69 = &v63[4 * v67];
    while ( *v68 == -1 )
    {
      if ( v68[1] != -1 )
      {
        ++v68;
        goto LABEL_63;
      }
      if ( v68[2] != -1 )
      {
        v68 += 2;
        goto LABEL_63;
      }
      if ( v68[3] != -1 )
      {
        v68 += 3;
        goto LABEL_63;
      }
      v68 += 4;
      if ( v69 == v68 )
      {
        v66 = v65 - v68;
        goto LABEL_97;
      }
    }
    goto LABEL_63;
  }
  v68 = *(_DWORD **)a4;
LABEL_97:
  if ( v66 == 2 )
    goto LABEL_135;
  if ( v66 == 3 )
  {
    if ( *v68 != -1 )
      goto LABEL_63;
    ++v68;
LABEL_135:
    if ( *v68 != -1 )
      goto LABEL_63;
    ++v68;
    goto LABEL_137;
  }
  if ( v66 != 1 )
    goto LABEL_100;
LABEL_137:
  if ( *v68 == -1 )
    goto LABEL_100;
LABEL_63:
  if ( v65 == v68 )
  {
LABEL_100:
    if ( 8LL * (unsigned int)v126 )
      memmove(a2, src, 8LL * (unsigned int)v126);
    BYTE4(v115) = 0;
    v70 = v128;
    goto LABEL_72;
  }
  v70 = v128;
  if ( (int)v129 > 0 )
  {
    v71 = 0;
    v72 = 4LL * (unsigned int)(v129 - 1);
    while ( 1 )
    {
      if ( v63[v71 / 4] == -1 )
      {
        v73 = &v70[v71 / 4];
        if ( *(_BYTE *)*v73 == 12 )
        {
          v74 = (__int64 *)a2[v71 / 4];
          a2[v71 / 4] = (__int64)*v73;
          *v73 = v74;
          v70 = v128;
        }
      }
      if ( v72 == v71 )
        break;
      v63 = *(_DWORD **)a4;
      v71 += 4LL;
    }
  }
  v115 = v114;
LABEL_72:
  if ( v70 != v130 )
    _libc_free((unsigned __int64)v70);
  if ( src != v127 )
    _libc_free((unsigned __int64)src);
  v36 = (unsigned int)v132;
LABEL_78:
  v75 = v131;
  v76 = &v131[9 * v36];
  if ( v131 != v76 )
  {
    do
    {
      v76 -= 9;
      v77 = v76[1];
      if ( (__int64 *)v77 != v76 + 3 )
        _libc_free(v77);
    }
    while ( v75 != v76 );
    v76 = v131;
  }
  if ( v76 != (__int64 *)v133 )
    _libc_free((unsigned __int64)v76);
  if ( v122 != (int *)v124 )
    _libc_free((unsigned __int64)v122);
  v78 = v120;
  v79 = &v120[9 * (unsigned int)v121];
  if ( v120 != v79 )
  {
    do
    {
      v79 -= 9;
      v80 = v79[1];
      if ( (unsigned __int64 *)v80 != v79 + 3 )
        _libc_free(v80);
    }
    while ( v78 != v79 );
    v79 = v120;
  }
  if ( v79 != (unsigned __int64 *)&v122 )
    _libc_free((unsigned __int64)v79);
  sub_C7D6A0(v117, 16LL * v119, 8);
  return v115;
}
