// Function: sub_2C6B010
// Address: 0x2c6b010
//
__int64 __fastcall sub_2C6B010(__int64 a1, __int64 a2)
{
  unsigned int v2; // r15d
  __int64 v4; // rax
  unsigned int v6; // eax
  _QWORD **v8; // r13
  _QWORD *v9; // rax
  _BYTE **v10; // rcx
  _BYTE *v11; // rcx
  _QWORD *v12; // rdx
  __int64 v13; // r13
  __int64 *v14; // rax
  __int64 *v15; // rdx
  char v16; // si
  __int64 *v17; // rax
  __int64 *v18; // r8
  __int64 v19; // rdx
  __int64 *v20; // r14
  __int64 v21; // rbx
  __int64 v22; // r14
  signed int v23; // r15d
  const void *v24; // r11
  unsigned __int64 v25; // rax
  __int64 v26; // r9
  int v27; // r8d
  char *v28; // rdi
  __int64 v29; // rax
  unsigned __int64 v30; // rcx
  unsigned int *v31; // rdi
  unsigned int *i; // rdx
  unsigned int v33; // esi
  unsigned int v34; // ecx
  unsigned int *v35; // rax
  unsigned int *v36; // rdi
  char *v37; // rcx
  __int64 v38; // rax
  char *v39; // rsi
  __int64 v40; // rax
  char *v41; // rax
  __int64 *v42; // rdi
  __int64 v43; // r9
  unsigned int v44; // ebx
  __int64 v45; // r11
  __int64 v46; // r8
  __int64 v47; // rax
  int v48; // edx
  int v49; // r15d
  __int64 v50; // rax
  __int64 v51; // rcx
  __int64 v52; // r8
  __int64 v53; // r9
  int v54; // edx
  bool v55; // sf
  bool v56; // of
  unsigned int v57; // eax
  __int64 *v58; // rdx
  __int64 v59; // rcx
  __int64 v60; // r8
  __int64 v61; // r9
  __int64 *v62; // rax
  unsigned __int8 v63; // al
  unsigned __int64 v64; // rdi
  unsigned __int64 *v65; // rbx
  _QWORD **v66; // r12
  unsigned __int64 v67; // rdi
  __int64 v68; // rdx
  __int64 *v69; // rax
  _QWORD *v70; // rcx
  __int64 v71; // r15
  __int64 *v72; // rax
  __int64 *v73; // rdx
  __int64 *v74; // rax
  __int64 v75; // rax
  void **v76; // r8
  void **v77; // r15
  void **v78; // r13
  void *v79; // r9
  _QWORD **v80; // rsi
  __int64 v81; // rax
  __int64 v82; // rax
  signed __int64 v83; // rax
  unsigned __int64 v84; // [rsp+0h] [rbp-150h]
  __int64 v85; // [rsp+0h] [rbp-150h]
  unsigned int *v86; // [rsp+8h] [rbp-148h]
  __int64 *v87; // [rsp+8h] [rbp-148h]
  void *v88; // [rsp+8h] [rbp-148h]
  const void *v89; // [rsp+8h] [rbp-148h]
  unsigned int *src; // [rsp+10h] [rbp-140h]
  unsigned int *srcb; // [rsp+10h] [rbp-140h]
  void *srca; // [rsp+10h] [rbp-140h]
  _QWORD *srcc; // [rsp+10h] [rbp-140h]
  void *srcd; // [rsp+10h] [rbp-140h]
  int srce; // [rsp+10h] [rbp-140h]
  int srcf; // [rsp+10h] [rbp-140h]
  __int64 v97; // [rsp+18h] [rbp-138h]
  char v98[32]; // [rsp+20h] [rbp-130h] BYREF
  __int16 v99; // [rsp+40h] [rbp-110h]
  __int64 v100; // [rsp+50h] [rbp-100h] BYREF
  __int64 *v101; // [rsp+58h] [rbp-F8h]
  __int64 v102; // [rsp+60h] [rbp-F0h]
  int v103; // [rsp+68h] [rbp-E8h]
  char v104; // [rsp+6Ch] [rbp-E4h]
  char v105; // [rsp+70h] [rbp-E0h] BYREF
  char *v106; // [rsp+90h] [rbp-C0h] BYREF
  __int64 v107; // [rsp+98h] [rbp-B8h]
  _BYTE v108[48]; // [rsp+A0h] [rbp-B0h] BYREF
  __int64 v109; // [rsp+D0h] [rbp-80h] BYREF
  __int64 v110; // [rsp+D8h] [rbp-78h]
  unsigned __int64 v111; // [rsp+E0h] [rbp-70h]
  _QWORD *v112; // [rsp+E8h] [rbp-68h]
  _QWORD *v113; // [rsp+F0h] [rbp-60h]
  unsigned __int64 *v114; // [rsp+F8h] [rbp-58h]
  _QWORD *v115; // [rsp+100h] [rbp-50h]
  _QWORD *v116; // [rsp+108h] [rbp-48h]
  _QWORD *v117; // [rsp+110h] [rbp-40h]
  _QWORD **v118; // [rsp+118h] [rbp-38h]

  v2 = 0;
  if ( *(_BYTE *)a2 != 85 )
    return v2;
  v4 = *(_QWORD *)(a2 - 32);
  if ( !v4 || *(_BYTE *)v4 || *(_QWORD *)(v4 + 24) != *(_QWORD *)(a2 + 80) || (*(_BYTE *)(v4 + 33) & 0x20) == 0 )
    return v2;
  v6 = *(_DWORD *)(v4 + 36);
  if ( v6 > 0x184 )
  {
    if ( v6 - 395 > 6 )
      return v2;
  }
  else if ( v6 <= 0x182 )
  {
    return v2;
  }
  v110 = 8;
  v109 = sub_22077B0(0x40u);
  v8 = (_QWORD **)(v109 + 24);
  v9 = (_QWORD *)sub_22077B0(0x200u);
  v114 = (unsigned __int64 *)(v109 + 24);
  *(_QWORD *)(v109 + 24) = v9;
  v113 = v9 + 64;
  v117 = v9 + 64;
  v112 = v9;
  v118 = v8;
  v116 = v9;
  v111 = (unsigned __int64)v9;
  v115 = v9;
  v100 = 0;
  v101 = (__int64 *)&v105;
  v102 = 4;
  v103 = 0;
  v104 = 1;
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v10 = *(_BYTE ***)(a2 - 8);
  else
    v10 = (_BYTE **)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  v11 = *v10;
  v12 = v9;
  if ( *v11 > 0x1Cu )
  {
    if ( v9 )
      *v9 = v11;
    v12 = v9 + 1;
    v115 = v9 + 1;
  }
  v97 = 0;
LABEL_16:
  if ( v9 != v12 )
  {
    do
    {
      v13 = *v9;
      if ( v9 == v113 - 1 )
      {
        j_j___libc_free_0((unsigned __int64)v112);
        v68 = *++v114 + 512;
        v112 = (_QWORD *)*v114;
        v113 = (_QWORD *)v68;
        v111 = (unsigned __int64)v112;
      }
      else
      {
        v111 = (unsigned __int64)(v9 + 1);
      }
      if ( v104 )
      {
        v14 = v101;
        v15 = &v101[HIDWORD(v102)];
        if ( v101 == v15 )
          goto LABEL_62;
        while ( v13 != *v14 )
        {
          if ( v15 == ++v14 )
            goto LABEL_62;
        }
      }
      else if ( !sub_C8CA60((__int64)&v100, v13) )
      {
LABEL_62:
        LOBYTE(v57) = sub_9B7DA0((char *)v13, 0xFFFFFFFF, 0);
        v2 = v57;
        if ( (_BYTE)v57 )
        {
          v12 = v115;
          goto LABEL_64;
        }
        v16 = v104;
        if ( !v104 )
          goto LABEL_83;
        v62 = v101;
        v58 = &v101[HIDWORD(v102)];
        if ( v101 == v58 )
        {
LABEL_85:
          if ( HIDWORD(v102) < (unsigned int)v102 )
          {
            ++HIDWORD(v102);
            *v58 = v13;
            v16 = v104;
            ++v100;
            goto LABEL_70;
          }
LABEL_83:
          sub_C8CC70((__int64)&v100, v13, (__int64)v58, v59, v60, v61);
          v16 = v104;
          goto LABEL_70;
        }
        while ( v13 != *v62 )
        {
          if ( v58 == ++v62 )
            goto LABEL_85;
        }
LABEL_70:
        v63 = *(_BYTE *)v13;
        if ( *(_BYTE *)v13 <= 0x1Cu )
          goto LABEL_75;
        if ( (unsigned int)v63 - 42 <= 0x11 )
        {
          v75 = 4LL * (*(_DWORD *)(v13 + 4) & 0x7FFFFFF);
          if ( (*(_BYTE *)(v13 + 7) & 0x40) != 0 )
          {
            v76 = *(void ***)(v13 - 8);
            v77 = &v76[v75];
          }
          else
          {
            v77 = (void **)v13;
            v76 = (void **)(v13 - v75 * 8);
          }
          v12 = v115;
          if ( v77 != v76 )
          {
            v78 = v76;
            do
            {
              v79 = *v78;
              if ( v12 == v117 - 1 )
              {
                v80 = v118;
                if ( ((__int64)((__int64)v113 - v111) >> 3)
                   + v12
                   - v116
                   + (((((char *)v118 - (char *)v114) >> 3) - 1) << 6) == 0xFFFFFFFFFFFFFFFLL )
                  sub_4262D8((__int64)"cannot create std::deque larger than max_size()");
                if ( (unsigned __int64)(v110 - (((__int64)v118 - v109) >> 3)) <= 1 )
                {
                  srcd = *v78;
                  sub_2C67060((unsigned __int64 *)&v109, 1u, 0);
                  v80 = v118;
                  v79 = srcd;
                }
                v88 = v79;
                v80[1] = (_QWORD *)sub_22077B0(0x200u);
                if ( v115 )
                  *v115 = v88;
                v12 = *++v118;
                v81 = (__int64)(*v118 + 64);
                v116 = v12;
                v117 = (_QWORD *)v81;
                v115 = v12;
              }
              else
              {
                if ( v12 )
                {
                  *v12 = v79;
                  v12 = v115;
                }
                v115 = ++v12;
              }
              v78 += 4;
            }
            while ( v77 != v78 );
          }
        }
        else
        {
          if ( v63 != 92 || v97 && v97 != v13 )
            goto LABEL_75;
          v97 = v13;
          v12 = v115;
        }
LABEL_64:
        v9 = (_QWORD *)v111;
        goto LABEL_16;
      }
      v9 = (_QWORD *)v111;
    }
    while ( (_QWORD *)v111 != v115 );
  }
  v2 = 0;
  v16 = v104;
  if ( !v97 )
    goto LABEL_75;
  v17 = v101;
  if ( v104 )
    v18 = &v101[HIDWORD(v102)];
  else
    v18 = &v101[(unsigned int)v102];
  if ( v101 != v18 )
  {
    while ( 1 )
    {
      v19 = *v17;
      v20 = v17;
      if ( (unsigned __int64)*v17 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v18 == ++v17 )
        goto LABEL_31;
    }
    if ( v18 != v17 )
    {
      while ( 1 )
      {
        v70 = *(_QWORD **)(v19 + 16);
        if ( v70 )
          break;
LABEL_99:
        v74 = v20 + 1;
        if ( v20 + 1 != v18 )
        {
          while ( 1 )
          {
            v19 = *v74;
            v20 = v74;
            if ( (unsigned __int64)*v74 < 0xFFFFFFFFFFFFFFFELL )
              break;
            if ( v18 == ++v74 )
              goto LABEL_31;
          }
          if ( v74 != v18 )
            continue;
        }
        goto LABEL_31;
      }
      while ( 2 )
      {
        v71 = v70[3];
        if ( v104 )
        {
          v72 = v101;
          v73 = &v101[HIDWORD(v102)];
          if ( v101 == v73 )
            goto LABEL_89;
          while ( v71 != *v72 )
          {
            if ( v73 == ++v72 )
              goto LABEL_89;
          }
        }
        else
        {
          v87 = v18;
          srcc = v70;
          v69 = sub_C8CA60((__int64)&v100, v71);
          v70 = srcc;
          v18 = v87;
          if ( !v69 )
          {
LABEL_89:
            if ( a2 != v71 )
              goto LABEL_90;
          }
        }
        v70 = (_QWORD *)v70[1];
        if ( !v70 )
          goto LABEL_99;
        continue;
      }
    }
  }
LABEL_31:
  v21 = *(_QWORD *)(*(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)) + 8LL);
  if ( *(_BYTE *)(v21 + 8) != 17 || (v22 = *(_QWORD *)(*(_QWORD *)(v97 - 64) + 8LL), *(_BYTE *)(v22 + 8) != 17) )
  {
LABEL_90:
    v16 = v104;
    v2 = 0;
    goto LABEL_75;
  }
  v23 = *(_DWORD *)(v22 + 32);
  v24 = *(const void **)(v97 + 72);
  v106 = v108;
  v107 = 0xC00000000LL;
  v25 = *(unsigned int *)(v97 + 80);
  v26 = 4 * v25;
  v27 = v25;
  if ( v25 > 0xC )
  {
    v85 = 4 * v25;
    v89 = v24;
    srce = *(_DWORD *)(v97 + 80);
    sub_C8D5F0((__int64)&v106, v108, v25, 4u, v25, v26);
    v27 = srce;
    v24 = v89;
    v26 = v85;
  }
  else if ( !v26 )
  {
    goto LABEL_35;
  }
  srcf = v27;
  memcpy(&v106[4 * (unsigned int)v107], v24, v26);
  v27 = srcf;
LABEL_35:
  v28 = v106;
  LODWORD(v107) = v27 + v107;
  v29 = 4LL * (unsigned int)v107;
  if ( v106 != &v106[v29] )
  {
    v84 = 4LL * (unsigned int)v107;
    v86 = (unsigned int *)&v106[v29];
    _BitScanReverse64(&v30, v29 >> 2);
    src = (unsigned int *)v106;
    sub_2C4D720(v106, (unsigned int *)&v106[v29], 2LL * (int)(63 - (v30 ^ 0x3F)));
    v31 = src;
    if ( v84 <= 0x40 )
    {
      sub_2C4CEA0(src, v86);
    }
    else
    {
      srcb = src + 16;
      sub_2C4CEA0(v31, srcb);
      for ( i = srcb; v86 != i; *v36 = v33 )
      {
        v33 = *i;
        v34 = *(i - 1);
        v35 = i - 1;
        if ( v34 <= *i )
        {
          v36 = i;
        }
        else
        {
          do
          {
            v35[1] = v34;
            v36 = v35;
            v34 = *--v35;
          }
          while ( v34 > v33 );
        }
        ++i;
      }
    }
    v28 = v106;
  }
  v37 = v28;
  v38 = 4LL * (unsigned int)v107;
  v39 = &v28[v38];
  v40 = v38 >> 4;
  if ( !v40 )
  {
LABEL_125:
    v83 = v39 - v37;
    if ( v39 - v37 != 8 )
    {
      if ( v83 != 12 )
      {
        if ( v83 != 4 )
        {
LABEL_128:
          v44 = 7;
          v42 = *(__int64 **)(a1 + 152);
          v43 = *(unsigned int *)(a1 + 192);
          v45 = *(_QWORD *)(v97 + 72);
          v46 = *(unsigned int *)(v97 + 80);
          goto LABEL_52;
        }
LABEL_135:
        if ( *(_DWORD *)v37 >= v23 )
          goto LABEL_49;
        goto LABEL_128;
      }
      if ( *(_DWORD *)v37 >= v23 )
        goto LABEL_49;
      v37 += 4;
    }
    if ( *(_DWORD *)v37 >= v23 )
      goto LABEL_49;
    v37 += 4;
    goto LABEL_135;
  }
  v41 = &v28[16 * v40];
  while ( v23 > *(_DWORD *)v37 )
  {
    if ( v23 <= *((_DWORD *)v37 + 1) )
    {
      v37 += 4;
      break;
    }
    if ( v23 <= *((_DWORD *)v37 + 2) )
    {
      v37 += 8;
      break;
    }
    if ( v23 <= *((_DWORD *)v37 + 3) )
    {
      v37 += 12;
      break;
    }
    v37 += 16;
    if ( v41 == v37 )
      goto LABEL_125;
  }
LABEL_49:
  v42 = *(__int64 **)(a1 + 152);
  if ( *(_DWORD *)(v21 + 32) < (unsigned int)v23 || v39 == v37 )
  {
    v43 = *(unsigned int *)(a1 + 192);
    v45 = *(_QWORD *)(v97 + 72);
    v46 = *(unsigned int *)(v97 + 80);
    v44 = (v39 == v37) + 6;
  }
  else
  {
    v22 = v21;
    v43 = *(unsigned int *)(a1 + 192);
    v44 = 6;
    v45 = *(_QWORD *)(v97 + 72);
    v46 = *(unsigned int *)(v97 + 80);
  }
LABEL_52:
  v47 = sub_DFBC30(v42, v44, v22, v45, v46, v43, 0, 0, 0, 0, 0);
  v49 = v48;
  srca = (void *)v47;
  v50 = sub_DFBC30(
          *(__int64 **)(a1 + 152),
          v44,
          v22,
          (__int64)v106,
          (unsigned int)v107,
          *(unsigned int *)(a1 + 192),
          0,
          0,
          0,
          0,
          0);
  v56 = __OFSUB__(v54, v49);
  v55 = v54 - v49 < 0;
  if ( v54 == v49 )
  {
    v56 = __OFSUB__(v50, srca);
    v55 = v50 - (__int64)srca < 0;
  }
  if ( v55 != v56 )
  {
    sub_D5F1F0(a1 + 8, v97);
    v99 = 257;
    v82 = sub_A83CB0(
            (unsigned int **)(a1 + 8),
            *(_BYTE **)(v97 - 64),
            *(_BYTE **)(v97 - 32),
            (__int64)v106,
            (unsigned int)v107,
            (__int64)v98);
    sub_2C535E0(a1, (unsigned __int8 *)v97, v82);
  }
  v2 = sub_2C68BF0(a1, v97, 1, v51, v52, v53);
  if ( v106 != v108 )
    _libc_free((unsigned __int64)v106);
  v16 = v104;
LABEL_75:
  if ( !v16 )
    _libc_free((unsigned __int64)v101);
  v64 = v109;
  if ( v109 )
  {
    v65 = v114;
    v66 = v118 + 1;
    if ( v118 + 1 > (_QWORD **)v114 )
    {
      do
      {
        v67 = *v65++;
        j_j___libc_free_0(v67);
      }
      while ( v66 > (_QWORD **)v65 );
      v64 = v109;
    }
    j_j___libc_free_0(v64);
  }
  return v2;
}
