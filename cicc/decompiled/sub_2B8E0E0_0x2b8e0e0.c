// Function: sub_2B8E0E0
// Address: 0x2b8e0e0
//
_QWORD *__fastcall sub_2B8E0E0(
        _QWORD *a1,
        __int64 a2,
        __int64 a3,
        char *a4,
        unsigned __int64 a5,
        __int64 a6,
        __int64 a7,
        unsigned int a8,
        char a9)
{
  unsigned __int64 v9; // r14
  unsigned __int64 *v10; // r12
  _QWORD *v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rcx
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 v18; // rax
  unsigned int v19; // r12d
  unsigned int v20; // eax
  __int64 v21; // r14
  unsigned int v22; // r13d
  unsigned int v23; // r15d
  __int64 v24; // rdx
  unsigned int v25; // r9d
  char *v26; // r11
  int v27; // eax
  unsigned __int64 v28; // rcx
  _QWORD *v29; // rdx
  __int64 v30; // rax
  __int64 **v31; // r10
  _DWORD *v32; // rax
  __int64 v33; // r8
  __int64 v34; // r9
  __int64 **v35; // r10
  __int64 v36; // rax
  __int64 v37; // rcx
  unsigned __int64 v38; // rdx
  __int64 v39; // rdi
  __int64 v40; // rax
  bool v41; // al
  __int64 v42; // rdx
  __int64 v43; // rcx
  __int64 v44; // r8
  __int64 v45; // r10
  bool v46; // al
  __int64 v47; // r8
  __int64 v48; // r9
  unsigned __int64 v49; // r14
  unsigned __int64 *v50; // r13
  __int64 v51; // rcx
  __int64 v52; // rax
  unsigned __int64 v53; // rsi
  __int64 v54; // rax
  unsigned __int64 v55; // rsi
  __int64 v56; // rdx
  __int64 v57; // rax
  __int64 v58; // rdi
  __int64 *v59; // r15
  int v60; // eax
  _QWORD *v61; // rdi
  __int64 v62; // rax
  __int64 v63; // rcx
  unsigned __int64 v64; // rdx
  char *v65; // rdx
  bool v66; // zf
  char *v67; // rdi
  _QWORD *v68; // rax
  __int64 v69; // rdi
  _QWORD *v70; // rsi
  __int64 v71; // rcx
  __int64 v72; // rdi
  __int64 v73; // rcx
  __int64 v74; // rdx
  __int64 v75; // rcx
  __int64 v76; // r8
  _QWORD *v77; // rax
  int v78; // eax
  __int64 v79; // rdx
  int v80; // eax
  __int64 v81; // rax
  __int64 v82; // rcx
  __int64 v83; // rdx
  char *v84; // rax
  char *v85; // rsi
  __int64 v86; // r8
  __int64 v87; // rdx
  unsigned __int64 *v88; // r13
  unsigned __int64 *v89; // r12
  _BYTE *v90; // r9
  __int64 v91; // r9
  __int64 v92; // r10
  char *v93; // r11
  __int64 v94; // rdx
  __int64 v95; // rcx
  __int64 v96; // r8
  __int64 v97; // r9
  __int64 v98; // r13
  _QWORD *v99; // rdi
  int v100; // r15d
  _QWORD *v101; // [rsp+0h] [rbp-130h]
  __int64 v102; // [rsp+20h] [rbp-110h]
  int v103; // [rsp+20h] [rbp-110h]
  __int64 **v110; // [rsp+60h] [rbp-D0h]
  __int64 **v111; // [rsp+60h] [rbp-D0h]
  __int64 v112; // [rsp+60h] [rbp-D0h]
  int v113; // [rsp+6Ch] [rbp-C4h]
  _DWORD *v114; // [rsp+70h] [rbp-C0h]
  unsigned __int64 v115; // [rsp+78h] [rbp-B8h] BYREF
  char *v116; // [rsp+80h] [rbp-B0h] BYREF
  __int64 v117; // [rsp+88h] [rbp-A8h]
  _BYTE v118[48]; // [rsp+90h] [rbp-A0h] BYREF
  unsigned __int64 v119[2]; // [rsp+C0h] [rbp-70h] BYREF
  _BYTE v120[96]; // [rsp+D0h] [rbp-60h] BYREF

  v9 = *(_QWORD *)a7;
  v10 = (unsigned __int64 *)(*(_QWORD *)a7 + ((unsigned __int64)*(unsigned int *)(a7 + 8) << 6));
  while ( (unsigned __int64 *)v9 != v10 )
  {
    while ( 1 )
    {
      v10 -= 8;
      if ( (unsigned __int64 *)*v10 == v10 + 2 )
        break;
      _libc_free(*v10);
      if ( (unsigned __int64 *)v9 == v10 )
        goto LABEL_5;
    }
  }
LABEL_5:
  *(_DWORD *)(a7 + 8) = 0;
  v11 = *(_QWORD **)a2;
  if ( a3 != **(_QWORD **)a2 )
    goto LABEL_6;
  if ( !*(_BYTE *)(a2 + 1256) )
    goto LABEL_66;
  v68 = v11 + 1;
  v69 = 8LL * *(unsigned int *)(a2 + 8) - 8;
  v70 = &v11[*(unsigned int *)(a2 + 8)];
  v71 = v69 >> 5;
  v72 = v69 >> 3;
  if ( v71 <= 0 )
    goto LABEL_102;
  v73 = (__int64)&v11[4 * v71 + 1];
  do
  {
    if ( *(_DWORD *)(*v68 + 104LL) != 3 )
      goto LABEL_65;
    if ( *(_DWORD *)(v68[1] + 104LL) != 3 )
    {
      ++v68;
      goto LABEL_65;
    }
    if ( *(_DWORD *)(v68[2] + 104LL) != 3 )
    {
      v68 += 2;
      goto LABEL_65;
    }
    if ( *(_DWORD *)(v68[3] + 104LL) != 3 )
    {
      v68 += 3;
      goto LABEL_65;
    }
    v68 += 4;
  }
  while ( (_QWORD *)v73 != v68 );
  v72 = v70 - v68;
LABEL_102:
  if ( v72 != 2 )
  {
    if ( v72 != 3 )
    {
      if ( v72 == 1 )
        goto LABEL_105;
      goto LABEL_66;
    }
    if ( *(_DWORD *)(*v68 + 104LL) != 3 )
      goto LABEL_65;
    ++v68;
  }
  if ( *(_DWORD *)(*v68 + 104LL) == 3 )
  {
    ++v68;
LABEL_105:
    if ( *(_DWORD *)(*v68 + 104LL) == 3 )
    {
LABEL_66:
      *a1 = a1 + 2;
      a1[1] = 0x600000000LL;
      return a1;
    }
  }
LABEL_65:
  if ( v70 == v68 )
    goto LABEL_66;
LABEL_6:
  v12 = sub_2B08520(**(char ***)a3);
  v101 = a1 + 2;
  if ( !sub_2B1F720(*(_QWORD *)(a2 + 3296), v12, *(_DWORD *)(v13 + 8)) )
  {
    *a1 = a1 + 2;
    a1[1] = 0x600000000LL;
    return a1;
  }
  sub_11B1960(a6, a5, -1, v14, v15, v16);
  v18 = *(_QWORD *)(a3 + 184);
  if ( v18 && *(_DWORD *)(v18 + 104) == 3 && *(_DWORD *)(a3 + 192) == -1 )
  {
    if ( !*(_DWORD *)(a3 + 200) )
      goto LABEL_100;
    v90 = *(_BYTE **)(a3 + 416);
    if ( v90 && *(_QWORD *)(a3 + 424) )
    {
      if ( *v90 == 90
        || sub_2B08550(*(unsigned __int8 ***)a3, *(unsigned int *)(a3 + 8))
        || sub_2B31EF0(a2, v91, v93, v92, 0) )
      {
        goto LABEL_100;
      }
      goto LABEL_11;
    }
    if ( sub_2B08550(*(unsigned __int8 ***)a3, *(unsigned int *)(a3 + 8)) )
    {
LABEL_100:
      *a1 = v101;
      a1[1] = 0x600000000LL;
      return a1;
    }
  }
LABEL_11:
  v19 = 1;
  v20 = ((_DWORD)a5 != 0) + ((unsigned int)a5 - ((_DWORD)a5 != 0)) / a8;
  if ( v20 > 1 )
  {
    _BitScanReverse(&v20, v20 - 1);
    v19 = 1 << (32 - (v20 ^ 0x1F));
  }
  v116 = v118;
  v117 = 0x600000000LL;
  if ( (unsigned int)a5 <= v19 )
    v19 = a5;
  if ( !a8 )
  {
LABEL_88:
    v88 = *(unsigned __int64 **)a7;
    v89 = (unsigned __int64 *)(*(_QWORD *)a7 + ((unsigned __int64)*(unsigned int *)(a7 + 8) << 6));
    while ( v88 != v89 )
    {
      while ( 1 )
      {
        v89 -= 8;
        if ( (unsigned __int64 *)*v89 == v89 + 2 )
          break;
        _libc_free(*v89);
        if ( v88 == v89 )
          goto LABEL_92;
      }
    }
LABEL_92:
    *(_DWORD *)(a7 + 8) = 0;
    v67 = v116;
    *a1 = v101;
    a1[1] = 0x600000000LL;
    goto LABEL_81;
  }
  v21 = 0;
  v22 = a5;
  v113 = 0;
  do
  {
    v23 = v22;
    v24 = *(unsigned int *)(a7 + 8);
    if ( v19 <= v22 )
      v23 = v19;
    v25 = v21;
    v26 = &a4[8 * v113];
    v27 = *(_DWORD *)(a7 + 8);
    if ( *(_DWORD *)(a7 + 12) <= (unsigned int)v24 )
    {
      v74 = sub_C8D7D0(a7, a7 + 16, 0, 0x40u, v119, (unsigned int)v21);
      v77 = (_QWORD *)(v74 + ((unsigned __int64)*(unsigned int *)(a7 + 8) << 6));
      if ( v77 )
      {
        v75 = (__int64)(v77 + 2);
        *v77 = v77 + 2;
        v77[1] = 0x600000000LL;
      }
      v102 = v74;
      sub_2B447E0(a7, v74, v74, v75, v76, (unsigned int)v21);
      v78 = v119[0];
      v79 = v102;
      v25 = v21;
      v26 = &a4[8 * v113];
      if ( a7 + 16 != *(_QWORD *)a7 )
      {
        v103 = v119[0];
        v112 = v79;
        _libc_free(*(_QWORD *)a7);
        v26 = &a4[8 * v113];
        v25 = v21;
        v78 = v103;
        v79 = v112;
      }
      *(_DWORD *)(a7 + 12) = v78;
      v80 = *(_DWORD *)(a7 + 8);
      *(_QWORD *)a7 = v79;
      v81 = (unsigned int)(v80 + 1);
      *(_DWORD *)(a7 + 8) = v81;
      v31 = (__int64 **)(v79 + (v81 << 6) - 64);
    }
    else
    {
      v28 = *(_QWORD *)a7;
      v29 = (_QWORD *)(*(_QWORD *)a7 + (v24 << 6));
      if ( v29 )
      {
        *v29 = v29 + 2;
        v29[1] = 0x600000000LL;
        v27 = *(_DWORD *)(a7 + 8);
        v28 = *(_QWORD *)a7;
      }
      v30 = (unsigned int)(v27 + 1);
      *(_DWORD *)(a7 + 8) = v30;
      v31 = (__int64 **)(v28 + (v30 << 6) - 64);
    }
    v110 = v31;
    v32 = sub_2B8A960((__int64 *)a2, a3, v26, v23, *(_DWORD **)a6, *(unsigned int *)(a6 + 8), v31, v25, a9);
    v35 = v110;
    v114 = v32;
    if ( !BYTE4(v32) )
      *((_DWORD *)v110 + 2) = 0;
    v36 = (unsigned int)v117;
    v37 = HIDWORD(v117);
    v38 = (unsigned int)v117 + 1LL;
    if ( v38 > HIDWORD(v117) )
    {
      sub_C8D5F0((__int64)&v116, v118, v38, 8u, v33, v34);
      v36 = (unsigned int)v117;
      v35 = v110;
    }
    *(_QWORD *)&v116[8 * v36] = v114;
    LODWORD(v117) = v117 + 1;
    if ( *((_DWORD *)v35 + 2) == 1 && (_DWORD)v114 == 7 )
    {
      v39 = **v35;
      v40 = *(unsigned int *)(v39 + 120);
      if ( !(_DWORD)v40 )
        v40 = *(unsigned int *)(v39 + 8);
      if ( v40 == a5 )
      {
        v111 = v35;
        v41 = sub_2B31C30(v39, *(char **)a3, *(unsigned int *)(a3 + 8), v37, v33, v34);
        v45 = (__int64)v111;
        if ( v41 || (v46 = sub_2B31C30(**v111, a4, a5, v43, v44, v34), v45 = (__int64)v111, v46) )
        {
          v119[0] = (unsigned __int64)v120;
          v119[1] = 0x600000000LL;
          sub_2B3BB10((__int64)v119, v45, v42, v43, v44, v34);
          v49 = *(_QWORD *)a7;
          v50 = (unsigned __int64 *)(*(_QWORD *)a7 + ((unsigned __int64)*(unsigned int *)(a7 + 8) << 6));
          if ( *(unsigned __int64 **)a7 != v50 )
          {
            do
            {
              v50 -= 8;
              if ( (unsigned __int64 *)*v50 != v50 + 2 )
                _libc_free(*v50);
            }
            while ( (unsigned __int64 *)v49 != v50 );
          }
          *(_DWORD *)(a7 + 8) = 0;
          LODWORD(v117) = 0;
          v51 = *(_QWORD *)a6;
          v52 = *(_QWORD *)a6 + 4LL * *(unsigned int *)(a6 + 8);
          if ( *(_QWORD *)a6 != v52 )
          {
            v53 = v52 - v51 - 4;
            v54 = 0;
            v55 = v53 >> 2;
            do
            {
              v56 = v54;
              *(_DWORD *)(v51 + 4 * v54) = v54;
              ++v54;
            }
            while ( v55 != v56 );
          }
          if ( (int)a5 > 0 )
          {
            v51 = 4LL * (unsigned int)(a5 - 1) + 4;
            v57 = 0;
            do
            {
              if ( **(_BYTE **)&a4[2 * v57] == 13 )
                *(_DWORD *)(*(_QWORD *)a6 + v57) = -1;
              v57 += 4;
            }
            while ( v51 != v57 );
          }
          v58 = *(unsigned int *)(a7 + 8);
          v59 = (__int64 *)v119[0];
          v60 = v58;
          if ( *(_DWORD *)(a7 + 12) <= (unsigned int)v58 )
          {
            v98 = sub_C8D7D0(a7, a7 + 16, 0, 0x40u, &v115, v48);
            v99 = (_QWORD *)(v98 + ((unsigned __int64)*(unsigned int *)(a7 + 8) << 6));
            if ( v99 )
            {
              *v99 = v99 + 2;
              v99[1] = 0x600000000LL;
              sub_2B3FC00((__int64)v99, 1u, *v59, v95, v96, v97);
            }
            sub_2B447E0(a7, v98, v94, v95, v96, v97);
            v100 = v115;
            if ( a7 + 16 != *(_QWORD *)a7 )
              _libc_free(*(_QWORD *)a7);
            ++*(_DWORD *)(a7 + 8);
            *(_QWORD *)a7 = v98;
            *(_DWORD *)(a7 + 12) = v100;
          }
          else
          {
            v61 = (_QWORD *)(*(_QWORD *)a7 + (v58 << 6));
            if ( v61 )
            {
              *v61 = v61 + 2;
              v61[1] = 0x600000000LL;
              sub_2B3FC00((__int64)v61, 1u, *v59, v51, v47, v48);
              v60 = *(_DWORD *)(a7 + 8);
            }
            *(_DWORD *)(a7 + 8) = v60 + 1;
          }
          v62 = (unsigned int)v117;
          v63 = HIDWORD(v117);
          v115 = 0x100000007LL;
          v64 = (unsigned int)v117 + 1LL;
          if ( v64 > HIDWORD(v117) )
          {
            sub_C8D5F0((__int64)&v116, v118, v64, 8u, v47, v48);
            v62 = (unsigned int)v117;
          }
          v65 = v116;
          *(_QWORD *)&v116[8 * v62] = 0x100000007LL;
          v66 = (_DWORD)v117 == -1;
          LODWORD(v117) = v117 + 1;
          *a1 = v101;
          a1[1] = 0x600000000LL;
          if ( !v66 )
            sub_2B0CCF0((__int64)a1, &v116, (__int64)v65, v63, v47, v48);
          if ( (_BYTE *)v119[0] != v120 )
          {
            _libc_free(v119[0]);
            v67 = v116;
            goto LABEL_81;
          }
          goto LABEL_114;
        }
      }
    }
    v113 += v19;
    ++v21;
    v22 -= v19;
  }
  while ( a8 != v21 );
  v67 = v116;
  v82 = (unsigned int)v117;
  v83 = 8LL * (unsigned int)v117;
  v84 = v116;
  v85 = &v116[v83];
  v86 = v83 >> 3;
  v87 = v83 >> 5;
  if ( v87 )
  {
    v87 = (__int64)&v116[32 * v87];
    while ( !v84[4] )
    {
      if ( v84[12] )
      {
        v84 += 8;
        goto LABEL_79;
      }
      if ( v84[20] )
      {
        v84 += 16;
        goto LABEL_79;
      }
      if ( v84[28] )
      {
        v84 += 24;
        goto LABEL_79;
      }
      v84 += 32;
      if ( (char *)v87 == v84 )
      {
        v86 = (v85 - v84) >> 3;
        goto LABEL_84;
      }
    }
    goto LABEL_79;
  }
LABEL_84:
  if ( v86 == 2 )
  {
LABEL_121:
    if ( v84[4] )
      goto LABEL_79;
    v84 += 8;
    goto LABEL_87;
  }
  if ( v86 == 3 )
  {
    if ( v84[4] )
      goto LABEL_79;
    v84 += 8;
    goto LABEL_121;
  }
  if ( v86 != 1 )
    goto LABEL_88;
LABEL_87:
  if ( !v84[4] )
    goto LABEL_88;
LABEL_79:
  if ( v85 == v84 )
    goto LABEL_88;
  *a1 = v101;
  a1[1] = 0x600000000LL;
  if ( (_DWORD)v82 )
  {
    sub_2B0CCF0((__int64)a1, &v116, v87, v82, v86, v34);
LABEL_114:
    v67 = v116;
  }
LABEL_81:
  if ( v67 != v118 )
    _libc_free((unsigned __int64)v67);
  return a1;
}
