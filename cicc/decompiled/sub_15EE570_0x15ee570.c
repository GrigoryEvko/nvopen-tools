// Function: sub_15EE570
// Address: 0x15ee570
//
__int64 __fastcall sub_15EE570(
        __int64 **a1,
        _BYTE *a2,
        size_t a3,
        _BYTE *a4,
        size_t a5,
        char a6,
        char a7,
        unsigned int a8)
{
  __int64 v9; // r15
  int v10; // eax
  __int64 v11; // r9
  int v12; // edx
  int v13; // eax
  int v14; // r10d
  unsigned int i; // r11d
  __int64 *v16; // rcx
  __int64 v17; // r8
  unsigned int v18; // r11d
  int v19; // eax
  int v20; // eax
  __int64 v21; // rax
  __int64 v22; // r13
  unsigned int v23; // esi
  unsigned int v24; // ecx
  __int64 v25; // r14
  __int64 *v26; // r9
  int v27; // r8d
  unsigned int j; // r10d
  __int64 *v29; // rbx
  __int64 v30; // r12
  unsigned int v31; // r10d
  int v33; // eax
  int v34; // eax
  int v35; // eax
  __int64 v36; // rdi
  int v37; // eax
  int v38; // eax
  __int64 v39; // r10
  int v40; // r8d
  unsigned int v41; // r14d
  __int64 *v42; // r15
  __int64 v43; // r13
  __int64 *v44; // rbx
  __int64 v45; // r12
  unsigned int v46; // r14d
  int v47; // eax
  unsigned int v48; // r14d
  __int64 v49; // r15
  __int64 *v50; // r13
  __int64 *v51; // rbx
  __int64 v52; // r12
  int v53; // eax
  int v54; // eax
  __int64 v55; // rax
  __int64 *v56; // rbx
  __int64 *v57; // [rsp+8h] [rbp-138h]
  __int64 *v58; // [rsp+8h] [rbp-138h]
  unsigned int v59; // [rsp+10h] [rbp-130h]
  unsigned int v60; // [rsp+10h] [rbp-130h]
  __int64 *v61; // [rsp+10h] [rbp-130h]
  __int64 v62; // [rsp+18h] [rbp-128h]
  __int64 v63; // [rsp+18h] [rbp-128h]
  int v64; // [rsp+18h] [rbp-128h]
  __int64 v65; // [rsp+20h] [rbp-120h]
  __int64 v66; // [rsp+20h] [rbp-120h]
  __int64 v67; // [rsp+20h] [rbp-120h]
  int v68; // [rsp+28h] [rbp-118h]
  void *v69; // [rsp+30h] [rbp-110h]
  __int64 v70; // [rsp+38h] [rbp-108h]
  void *s1; // [rsp+40h] [rbp-100h]
  size_t v72; // [rsp+48h] [rbp-F8h]
  size_t n; // [rsp+50h] [rbp-F0h]
  int v74; // [rsp+58h] [rbp-E8h]
  int v75; // [rsp+58h] [rbp-E8h]
  unsigned int v76; // [rsp+58h] [rbp-E8h]
  int v77; // [rsp+5Ch] [rbp-E4h]
  int v78; // [rsp+60h] [rbp-E0h]
  char v79; // [rsp+6Ah] [rbp-D6h]
  char v80; // [rsp+6Bh] [rbp-D5h]
  int v81; // [rsp+6Ch] [rbp-D4h]
  int v82; // [rsp+6Ch] [rbp-D4h]
  int v83; // [rsp+6Ch] [rbp-D4h]
  int v84; // [rsp+6Ch] [rbp-D4h]
  int v85; // [rsp+6Ch] [rbp-D4h]
  int v86; // [rsp+6Ch] [rbp-D4h]
  int v87; // [rsp+6Ch] [rbp-D4h]
  __int64 v88; // [rsp+70h] [rbp-D0h]
  __int64 *v90; // [rsp+78h] [rbp-C8h]
  __int64 *v91; // [rsp+78h] [rbp-C8h]
  __int64 *v92; // [rsp+78h] [rbp-C8h]
  int v93; // [rsp+78h] [rbp-C8h]
  __int64 v94; // [rsp+78h] [rbp-C8h]
  unsigned int v96; // [rsp+80h] [rbp-C0h]
  unsigned int v97; // [rsp+80h] [rbp-C0h]
  unsigned int v98; // [rsp+80h] [rbp-C0h]
  __int64 v99; // [rsp+80h] [rbp-C0h]
  int v100; // [rsp+80h] [rbp-C0h]
  unsigned int v102; // [rsp+88h] [rbp-B8h]
  unsigned int v103; // [rsp+88h] [rbp-B8h]
  unsigned int v104; // [rsp+88h] [rbp-B8h]
  __int64 v105; // [rsp+88h] [rbp-B8h]
  __int64 v106; // [rsp+88h] [rbp-B8h]
  __int64 v107[2]; // [rsp+90h] [rbp-B0h] BYREF
  _QWORD v108[2]; // [rsp+A0h] [rbp-A0h] BYREF
  __int64 v109[2]; // [rsp+B0h] [rbp-90h] BYREF
  _QWORD v110[2]; // [rsp+C0h] [rbp-80h] BYREF
  __int64 v111; // [rsp+D0h] [rbp-70h] BYREF
  _BYTE *v112; // [rsp+D8h] [rbp-68h] BYREF
  size_t v113; // [rsp+E0h] [rbp-60h]
  _BYTE *v114; // [rsp+E8h] [rbp-58h] BYREF
  size_t v115; // [rsp+F0h] [rbp-50h]
  __int64 **v116; // [rsp+F8h] [rbp-48h] BYREF
  char v117; // [rsp+100h] [rbp-40h] BYREF
  char v118[3]; // [rsp+101h] [rbp-3Fh] BYREF
  unsigned int v119[15]; // [rsp+104h] [rbp-3Ch] BYREF

  v9 = **a1;
  v118[0] = a7;
  v111 = sub_1646BA0(a1, 0);
  v112 = a2;
  v113 = a3;
  v115 = a5;
  v114 = a4;
  v117 = a6;
  v116 = a1;
  v119[0] = a8;
  LODWORD(v109[0]) = sub_15EAB80((__int64 *)&v112, (__int64 *)&v114, &v117, v118, (int *)v119, (__int64 *)&v116);
  v10 = sub_15EDB00(&v111, (int *)v109);
  v11 = *(_QWORD *)(v9 + 1816);
  v78 = v10;
  v12 = v10;
  v88 = v111;
  s1 = v112;
  n = v113;
  v69 = v114;
  v72 = v115;
  v70 = (__int64)v116;
  v80 = v117;
  v79 = v118[0];
  v77 = v119[0];
  v13 = *(_DWORD *)(v9 + 1832);
  if ( v13 )
  {
    v68 = 1;
    v14 = v13 - 1;
    for ( i = (v13 - 1) & v12; ; i = v14 & v18 )
    {
      v16 = (__int64 *)(v11 + 8LL * i);
      v17 = *v16;
      if ( *v16 == -8 )
        break;
      if ( v17 != -16 )
      {
        if ( *(_QWORD *)v17 != v88 )
          goto LABEL_6;
        if ( v80 != *(_BYTE *)(v17 + 96) )
          goto LABEL_6;
        if ( v79 != *(_BYTE *)(v17 + 97) )
          goto LABEL_6;
        if ( *(_DWORD *)(v17 + 100) != v77 )
          goto LABEL_6;
        if ( *(_QWORD *)(v17 + 32) != n )
          goto LABEL_6;
        if ( n )
        {
          v57 = (__int64 *)(v11 + 8LL * i);
          v59 = i;
          v74 = v14;
          v62 = v11;
          v65 = *v16;
          v19 = memcmp(s1, *(const void **)(v17 + 24), n);
          v17 = v65;
          v11 = v62;
          v14 = v74;
          i = v59;
          v16 = v57;
          if ( v19 )
            goto LABEL_6;
        }
        if ( *(_QWORD *)(v17 + 64) != v72 )
          goto LABEL_6;
        if ( v72 )
        {
          v58 = v16;
          v60 = i;
          v75 = v14;
          v63 = v11;
          v66 = v17;
          v20 = memcmp(v69, *(const void **)(v17 + 56), v72);
          v17 = v66;
          v11 = v63;
          v14 = v75;
          i = v60;
          v16 = v58;
          if ( v20 )
            goto LABEL_6;
        }
        v61 = v16;
        v76 = i;
        v64 = v14;
        v67 = v11;
        v21 = sub_15EAB70(v17);
        v11 = v67;
        v14 = v64;
        i = v76;
        if ( v70 == v21 )
        {
          if ( v61 != (__int64 *)(*(_QWORD *)(v9 + 1816) + 8LL * *(unsigned int *)(v9 + 1832)) )
            return *v61;
          break;
        }
        v17 = *v61;
      }
      if ( v17 == -8 )
        break;
LABEL_6:
      v18 = v68 + i;
      ++v68;
    }
  }
  if ( a2 )
  {
    v107[0] = (__int64)v108;
    sub_15EA2A0(v107, a2, (__int64)&a2[a3]);
  }
  else
  {
    LOBYTE(v108[0]) = 0;
    v107[0] = (__int64)v108;
    v107[1] = 0;
  }
  v109[0] = (__int64)v110;
  if ( a4 )
  {
    sub_15EA2A0(v109, a4, (__int64)&a4[a5]);
  }
  else
  {
    v109[1] = 0;
    LOBYTE(v110[0]) = 0;
  }
  v22 = sub_22077B0(104);
  if ( v22 )
    sub_15EAAD0(v22, (__int64)a1, (__int64)v107, (__int64)v109, a6, a7, a8);
  if ( (_QWORD *)v109[0] != v110 )
    j_j___libc_free_0(v109[0], v110[0] + 1LL);
  if ( (_QWORD *)v107[0] != v108 )
    j_j___libc_free_0(v107[0], v108[0] + 1LL);
  v23 = *(_DWORD *)(v9 + 1832);
  if ( v23 )
  {
    v24 = v23 - 1;
    v25 = *(_QWORD *)(v9 + 1816);
    v26 = 0;
    v27 = 1;
    for ( j = (v23 - 1) & v78; ; j = v24 & v31 )
    {
      v29 = (__int64 *)(v25 + 8LL * j);
      v30 = *v29;
      if ( *v29 == -8 )
        break;
      if ( v30 != -16 )
      {
        if ( *(_QWORD *)v30 != v88 )
          goto LABEL_34;
        if ( v80 != *(_BYTE *)(v30 + 96) )
          goto LABEL_34;
        if ( v79 != *(_BYTE *)(v30 + 97) )
          goto LABEL_34;
        if ( *(_DWORD *)(v30 + 100) != v77 )
          goto LABEL_34;
        if ( *(_QWORD *)(v30 + 32) != n )
          goto LABEL_34;
        if ( n )
        {
          v81 = v27;
          v90 = v26;
          v96 = j;
          v102 = v24;
          v33 = memcmp(s1, *(const void **)(v30 + 24), n);
          v24 = v102;
          j = v96;
          v26 = v90;
          v27 = v81;
          if ( v33 )
            goto LABEL_34;
        }
        if ( *(_QWORD *)(v30 + 64) != v72 )
          goto LABEL_34;
        if ( v72 )
        {
          v82 = v27;
          v91 = v26;
          v97 = j;
          v103 = v24;
          v34 = memcmp(v69, *(const void **)(v30 + 56), v72);
          v24 = v103;
          j = v97;
          v26 = v91;
          v27 = v82;
          if ( v34 )
            goto LABEL_34;
        }
        v83 = v27;
        v92 = v26;
        v98 = j;
        v104 = v24;
        if ( v70 == sub_15EAB70(v30) )
          return v22;
        v30 = *v29;
        v27 = v83;
        v26 = v92;
        j = v98;
        v24 = v104;
      }
      if ( v30 == -8 )
        break;
      if ( !v26 && v30 == -16 )
        v26 = v29;
LABEL_34:
      v31 = v27 + j;
      ++v27;
    }
    v35 = *(_DWORD *)(v9 + 1824);
    v23 = *(_DWORD *)(v9 + 1832);
    v36 = v9 + 1808;
    if ( !v26 )
      v26 = v29;
    ++*(_QWORD *)(v9 + 1808);
    v37 = v35 + 1;
    if ( 4 * v37 < 3 * v23 )
    {
      if ( v23 - (v37 + *(_DWORD *)(v9 + 1828)) > v23 >> 3 )
        goto LABEL_56;
      sub_15EE3B0(v36, v23);
      v47 = *(_DWORD *)(v9 + 1832);
      if ( v47 )
      {
        v84 = v47 - 1;
        v100 = 1;
        v106 = v9;
        v48 = (v47 - 1) & v78;
        v49 = *(_QWORD *)(v9 + 1816);
        v94 = v22;
        v50 = 0;
        while ( 1 )
        {
          v51 = (__int64 *)(v49 + 8LL * v48);
          v52 = *v51;
          if ( *v51 != -16 )
          {
            if ( v52 == -8 )
              goto LABEL_88;
            if ( *(_QWORD *)v52 != v88
              || v80 != *(_BYTE *)(v52 + 96)
              || v79 != *(_BYTE *)(v52 + 97)
              || *(_DWORD *)(v52 + 100) != v77
              || *(_QWORD *)(v52 + 32) != n
              || n && memcmp(s1, *(const void **)(v52 + 24), n)
              || v72 != *(_QWORD *)(v52 + 64)
              || v72 && memcmp(v69, *(const void **)(v52 + 56), v72) )
            {
              goto LABEL_71;
            }
            if ( v70 == sub_15EAB70(v52) )
            {
              v9 = v106;
              v22 = v94;
              v26 = v51;
              v37 = *(_DWORD *)(v106 + 1824) + 1;
              goto LABEL_56;
            }
            v52 = *v51;
          }
          if ( v52 == -8 )
          {
LABEL_88:
            v9 = v106;
            v26 = v50;
            v22 = v94;
            v37 = *(_DWORD *)(v106 + 1824) + 1;
            if ( !v26 )
              v26 = v51;
            goto LABEL_56;
          }
          if ( !v50 && v52 == -16 )
            v50 = (__int64 *)(v49 + 8LL * v48);
LABEL_71:
          v48 = v84 & (v100 + v48);
          ++v100;
        }
      }
LABEL_116:
      ++*(_DWORD *)(v9 + 1824);
      BUG();
    }
  }
  else
  {
    ++*(_QWORD *)(v9 + 1808);
    v36 = v9 + 1808;
  }
  sub_15EE3B0(v36, 2 * v23);
  v38 = *(_DWORD *)(v9 + 1832);
  if ( !v38 )
    goto LABEL_116;
  v39 = *(_QWORD *)(v9 + 1816);
  v93 = v38 - 1;
  v40 = 1;
  v105 = v9;
  v41 = (v38 - 1) & v78;
  v42 = 0;
  v99 = v22;
  v43 = v39;
  while ( 1 )
  {
    v44 = (__int64 *)(v43 + 8LL * v41);
    v45 = *v44;
    if ( *v44 == -8 )
      break;
    if ( v45 != -16 )
    {
      if ( *(_QWORD *)v45 != v88 )
        goto LABEL_65;
      if ( v80 != *(_BYTE *)(v45 + 96) )
        goto LABEL_65;
      if ( v79 != *(_BYTE *)(v45 + 97) )
        goto LABEL_65;
      if ( *(_DWORD *)(v45 + 100) != v77 )
        goto LABEL_65;
      if ( *(_QWORD *)(v45 + 32) != n )
        goto LABEL_65;
      if ( n )
      {
        v85 = v40;
        v53 = memcmp(s1, *(const void **)(v45 + 24), n);
        v40 = v85;
        if ( v53 )
          goto LABEL_65;
      }
      if ( v72 != *(_QWORD *)(v45 + 64) )
        goto LABEL_65;
      if ( v72 )
      {
        v86 = v40;
        v54 = memcmp(v69, *(const void **)(v45 + 56), v72);
        v40 = v86;
        if ( v54 )
          goto LABEL_65;
      }
      v87 = v40;
      v55 = sub_15EAB70(v45);
      v40 = v87;
      if ( v70 == v55 )
      {
        v9 = v105;
        v22 = v99;
        v26 = v44;
        v37 = *(_DWORD *)(v105 + 1824) + 1;
        goto LABEL_56;
      }
      v45 = *v44;
    }
    if ( v45 == -8 )
      break;
    if ( v45 == -16 && !v42 )
      v42 = (__int64 *)(v43 + 8LL * v41);
LABEL_65:
    v46 = v40 + v41;
    ++v40;
    v41 = v93 & v46;
  }
  v26 = (__int64 *)(v43 + 8LL * v41);
  v56 = v42;
  v9 = v105;
  v22 = v99;
  v37 = *(_DWORD *)(v105 + 1824) + 1;
  if ( v56 )
    v26 = v56;
LABEL_56:
  *(_DWORD *)(v9 + 1824) = v37;
  if ( *v26 != -8 )
    --*(_DWORD *)(v9 + 1828);
  *v26 = v22;
  return v22;
}
