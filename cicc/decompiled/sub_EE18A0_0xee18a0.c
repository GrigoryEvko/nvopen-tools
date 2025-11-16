// Function: sub_EE18A0
// Address: 0xee18a0
//
__int64 __fastcall sub_EE18A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  int v7; // ebx
  __int64 v8; // rax
  unsigned int v9; // r12d
  __int64 v10; // r15
  __int64 v11; // r12
  unsigned __int16 *v12; // r13
  __int64 v13; // rbx
  __int64 i; // rax
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // r8
  __int64 v22; // r9
  __int64 *v23; // rdx
  __int64 *j; // rdi
  __int64 v25; // rcx
  bool v26; // zf
  __int64 v27; // rdx
  unsigned int v28; // edi
  _QWORD *v30; // rax
  __int64 v31; // rdx
  __int64 v32; // rsi
  int v33; // ecx
  __int64 v34; // rdx
  _QWORD *v35; // rax
  _QWORD *v36; // r12
  _QWORD *v37; // rbx
  _QWORD *v38; // rdi
  unsigned int k; // r14d
  unsigned int v42; // eax
  unsigned int v43; // r9d
  unsigned int v44; // esi
  int v45; // ecx
  unsigned __int64 v46; // r8
  __int64 v47; // rdx
  unsigned __int64 v48; // r8
  _QWORD *v51; // r14
  _QWORD *v52; // rbx
  __int64 v53; // r15
  __int64 v54; // rax
  __int64 v55; // r12
  __int64 v56; // r13
  unsigned __int64 v57; // rax
  unsigned int *v58; // r12
  unsigned int *v59; // rdi
  __int64 v60; // r9
  __int64 v61; // rax
  __int64 *v62; // rdx
  int *v63; // rax
  int v64; // edi
  __int64 *v65; // rcx
  __int64 v66; // rax
  __int64 v67; // r8
  int *v68; // rcx
  __int64 v69; // rdi
  __int64 v70; // r9
  int v71; // r13d
  __int64 v73; // [rsp+48h] [rbp-218h]
  __int64 v74; // [rsp+58h] [rbp-208h] BYREF
  void *v75; // [rsp+60h] [rbp-200h] BYREF
  __int64 v76; // [rsp+68h] [rbp-1F8h]
  _BYTE v77[48]; // [rsp+70h] [rbp-1F0h] BYREF
  int v78; // [rsp+A0h] [rbp-1C0h]
  _QWORD v79[3]; // [rsp+B0h] [rbp-1B0h] BYREF
  __int64 v80; // [rsp+C8h] [rbp-198h]
  _QWORD *v81; // [rsp+D0h] [rbp-190h]
  __int64 v82; // [rsp+D8h] [rbp-188h]
  unsigned int v83; // [rsp+E0h] [rbp-180h]
  void *v84; // [rsp+E8h] [rbp-178h] BYREF
  __int64 v85; // [rsp+F0h] [rbp-170h]
  _BYTE v86[48]; // [rsp+F8h] [rbp-168h] BYREF
  int v87; // [rsp+128h] [rbp-138h]
  char *v88[2]; // [rsp+130h] [rbp-130h] BYREF
  char v89; // [rsp+140h] [rbp-120h] BYREF
  char *v90[2]; // [rsp+1E8h] [rbp-78h] BYREF
  char v91; // [rsp+1F8h] [rbp-68h] BYREF

  v6 = *(_QWORD *)(a2 + 272);
  v7 = *(_DWORD *)(a2 + 288);
  v80 = 0;
  v74 = v6;
  v8 = *(_QWORD *)(a2 + 280);
  v9 = (unsigned int)(v7 + 63) >> 6;
  v81 = 0;
  v79[0] = v8;
  v10 = v9;
  v82 = 0;
  v79[1] = sub_ED6B00;
  v79[2] = &v74;
  v83 = 0;
  v84 = v86;
  v85 = 0x600000000LL;
  if ( v9 <= 6 )
  {
    if ( v9 && 8LL * v9 )
      memset(v86, 0, 8LL * v9);
    v87 = v7;
    LODWORD(v85) = (unsigned int)(v7 + 63) >> 6;
    v75 = v77;
    HIDWORD(v76) = 6;
    goto LABEL_6;
  }
  sub_C8D5F0((__int64)&v84, v86, v9, 8u, 0x600000000LL, a6);
  memset(v84, 0, 8LL * v9);
  v71 = *(_DWORD *)(a2 + 288);
  LODWORD(v85) = (unsigned int)(v7 + 63) >> 6;
  v87 = v7;
  v9 = (unsigned int)(v71 + 63) >> 6;
  v75 = v77;
  v76 = 0x600000000LL;
  v10 = v9;
  if ( v9 <= 6 )
  {
    v7 = v71;
LABEL_6:
    if ( v10 && 8 * v10 )
      memset(v77, 0, 8 * v10);
    LODWORD(v76) = v9;
    goto LABEL_8;
  }
  v7 = v71;
  sub_C8D5F0((__int64)&v75, v77, v9, 8u, 0x600000000LL, v70);
  memset(v75, 0, 8LL * v9);
  LODWORD(v76) = (unsigned int)(v71 + 63) >> 6;
LABEL_8:
  v11 = *(_QWORD *)(a2 + 248);
  v78 = v7;
  v12 = *(unsigned __int16 **)(v11 + 528);
  v73 = *(_QWORD *)(v11 + 8);
  if ( v73 )
  {
    v13 = 0;
    for ( i = 10; ; i = v13 == 0 ? 10LL : 8LL )
    {
      sub_C16570(
        (__int64 *)v88,
        v11 + 40,
        (unsigned __int64 *)((char *)v12 + i + *(_QWORD *)((char *)v12 + i) + 16),
        *(_QWORD *)(v11 + 32));
      sub_ED66E0(v11 + 280, v88, v15, v16, v17, v18);
      sub_ED6840(v11 + 464, v90, v19, v20, v21, v22);
      if ( v90[0] != &v91 )
        _libc_free(v90[0], v90);
      if ( v88[0] != &v89 )
        _libc_free(v88[0], v90);
      v23 = *(__int64 **)(v11 + 280);
      for ( j = &v23[21 * *(unsigned int *)(v11 + 288)]; j != v23; *((_QWORD *)v75 + ((unsigned int)v25 >> 6)) |= 1LL << v25 )
      {
        v25 = *v23;
        v23 += 21;
      }
      if ( !v13 )
        v13 = *v12++;
      --v13;
      v26 = v73-- == 1;
      v12 = (unsigned __int16 *)((char *)v12 + *((_QWORD *)v12 + 1) + *((_QWORD *)v12 + 2) + 24);
      if ( v26 )
        break;
    }
    v7 = v78;
  }
  if ( v7 )
  {
    v27 = 0;
    v28 = (unsigned int)(v7 - 1) >> 6;
    while ( 1 )
    {
      _RCX = *((_QWORD *)v75 + v27);
      if ( v28 == (_DWORD)v27 )
        _RCX = (0xFFFFFFFFFFFFFFFFLL >> -(char)v7) & *((_QWORD *)v75 + v27);
      if ( _RCX )
        break;
      if ( v28 + 1 == ++v27 )
        goto LABEL_27;
    }
    __asm { tzcnt   rcx, rcx }
    for ( k = ((_DWORD)v27 << 6) + _RCX; k != -1; k = ((_DWORD)v47 << 6) + _RAX )
    {
      sub_EE1060((__int64)v79, k);
      v42 = k + 1;
      if ( v78 == k + 1 )
        break;
      v43 = v42 >> 6;
      v44 = (unsigned int)(v78 - 1) >> 6;
      if ( v42 >> 6 > v44 )
        break;
      v45 = 64 - (v42 & 0x3F);
      v46 = 0xFFFFFFFFFFFFFFFFLL >> v45;
      v47 = v43;
      if ( v45 == 64 )
        v46 = 0;
      v48 = ~v46;
      while ( 1 )
      {
        _RAX = *((_QWORD *)v75 + v47);
        if ( v43 == (_DWORD)v47 )
          _RAX = v48 & *((_QWORD *)v75 + v47);
        if ( v44 == (_DWORD)v47 )
          _RAX &= 0xFFFFFFFFFFFFFFFFLL >> -(char)v78;
        if ( _RAX )
          break;
        if ( v44 < (unsigned int)++v47 )
          goto LABEL_27;
      }
      __asm { tzcnt   rax, rax }
    }
  }
LABEL_27:
  v30 = v81;
  v81 = 0;
  v31 = v83;
  v32 = v82;
  v83 = 0;
  v33 = v82;
  ++v80;
  *(_QWORD *)a1 = 1;
  *(_QWORD *)(a1 + 8) = v30;
  *(_QWORD *)(a1 + 16) = v32;
  *(_DWORD *)(a1 + 24) = v31;
  v82 = 0;
  if ( !v33 )
    goto LABEL_28;
  v51 = &v30[3 * v31];
  if ( v30 == v51 )
    goto LABEL_28;
  while ( 1 )
  {
    v52 = v30;
    if ( *v30 <= 0xFFFFFFFFFFFFFFFDLL )
      break;
    v30 += 3;
    if ( v51 == v30 )
      goto LABEL_28;
  }
  while ( v52 != v51 )
  {
    v53 = v52[1];
    LODWORD(v54) = 0;
    v55 = 16LL * *((unsigned int *)v52 + 4);
    v56 = v53 + v55;
    if ( v53 == v53 + v55 )
      goto LABEL_85;
    _BitScanReverse64(&v57, v55 >> 4);
    sub_EE1550(v52[1], (unsigned __int64 *)(v53 + v55), 2LL * (int)(63 - (v57 ^ 0x3F)));
    if ( (unsigned __int64)v55 <= 0x100 )
    {
      v32 = v53 + v55;
      sub_ED78A0(v53, v53 + v55);
    }
    else
    {
      v58 = (unsigned int *)(v53 + 256);
      v32 = v53 + 256;
      sub_ED78A0(v53, v53 + 256);
      if ( v56 != v53 + 256 )
      {
        do
        {
          v59 = v58;
          v58 += 4;
          sub_ED7840(v59);
        }
        while ( (unsigned int *)v56 != v58 );
      }
    }
    v60 = v52[1];
    v61 = 16LL * *((unsigned int *)v52 + 4);
    v62 = (__int64 *)(v60 + v61);
    if ( v60 == v60 + v61 )
    {
      LODWORD(v54) = 0;
      goto LABEL_85;
    }
    v63 = (int *)(v60 + 16);
    if ( v62 == (__int64 *)(v60 + 16) )
    {
      LODWORD(v54) = 1;
      goto LABEL_85;
    }
    while ( 1 )
    {
      v64 = *(v63 - 4);
      v65 = (__int64 *)(v63 - 4);
      if ( v64 == *v63 )
      {
        v32 = (unsigned int)v63[1];
        if ( *(v63 - 3) == (_DWORD)v32 )
        {
          v32 = *((_QWORD *)v63 + 1);
          if ( *((_QWORD *)v63 - 1) == v32 )
            break;
        }
      }
      v63 += 4;
      if ( v62 == (__int64 *)v63 )
        goto LABEL_90;
    }
    if ( v62 == v65 )
    {
LABEL_90:
      v54 = ((__int64)v62 - v60) >> 4;
      goto LABEL_85;
    }
    v32 = (__int64)(v63 + 4);
    if ( v62 == (__int64 *)(v63 + 4) )
    {
LABEL_95:
      v54 = ((__int64)v63 - v60) >> 4;
      goto LABEL_85;
    }
    while ( v64 != *(_DWORD *)v32 || *((_DWORD *)v65 + 1) != *(_DWORD *)(v32 + 4) || v65[1] != *(_QWORD *)(v32 + 8) )
    {
      v66 = *(_QWORD *)v32;
      v32 += 16;
      v65 += 2;
      *v65 = v66;
      v65[1] = *(_QWORD *)(v32 - 8);
      if ( v62 == (__int64 *)v32 )
        goto LABEL_81;
LABEL_76:
      v64 = *(_DWORD *)v65;
    }
    v32 += 16;
    if ( v62 != (__int64 *)v32 )
      goto LABEL_76;
LABEL_81:
    v60 = v52[1];
    v63 = (int *)(v65 + 2);
    v67 = v60 + 16LL * *((unsigned int *)v52 + 4) - (_QWORD)v62;
    v32 = v67 >> 4;
    if ( v67 <= 0 )
      goto LABEL_95;
    v68 = (int *)(v65 + 2);
    do
    {
      v69 = *v62;
      v68 += 4;
      v62 += 2;
      *((_QWORD *)v68 - 2) = v69;
      *((_QWORD *)v68 - 1) = *(v62 - 1);
      --v32;
    }
    while ( v32 );
    v54 = ((__int64)v63 + v67 - v52[1]) >> 4;
LABEL_85:
    *((_DWORD *)v52 + 4) = v54;
    v52 += 3;
    if ( v52 != v51 )
    {
      while ( *v52 > 0xFFFFFFFFFFFFFFFDLL )
      {
        v52 += 3;
        if ( v51 == v52 )
          goto LABEL_28;
      }
      continue;
    }
    break;
  }
LABEL_28:
  if ( v75 != v77 )
    _libc_free(v75, v32);
  if ( v84 != v86 )
    _libc_free(v84, v32);
  v34 = v83;
  if ( v83 )
  {
    v35 = v81;
    v36 = &v81[3 * v83];
    do
    {
      while ( 1 )
      {
        v37 = v35 + 3;
        if ( *v35 <= 0xFFFFFFFFFFFFFFFDLL )
        {
          v38 = (_QWORD *)v35[1];
          if ( v38 != v37 )
            break;
        }
        v35 += 3;
        if ( v36 == v37 )
          goto LABEL_38;
      }
      _libc_free(v38, v32);
      v35 = v37;
    }
    while ( v36 != v37 );
LABEL_38:
    v34 = v83;
  }
  sub_C7D6A0((__int64)v81, 24 * v34, 8);
  return a1;
}
