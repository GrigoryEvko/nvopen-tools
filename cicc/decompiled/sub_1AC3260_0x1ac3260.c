// Function: sub_1AC3260
// Address: 0x1ac3260
//
_BOOL8 __fastcall sub_1AC3260(
        __int64 a1,
        __int64 (__fastcall *a2)(__int64, __int64, __int64, __int64, size_t),
        __int64 a3,
        __m128 a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        __m128 a11)
{
  __int64 v12; // rsi
  __int64 v13; // rax
  size_t v14; // r12
  __int64 v15; // rcx
  char v17; // al
  __int64 v18; // rdx
  __int64 v19; // r9
  __int64 v20; // rax
  __int64 v21; // rdi
  __int64 v22; // rax
  _QWORD *v23; // rax
  __int64 v24; // r14
  __int64 v25; // rax
  __int64 v26; // r13
  _QWORD *v27; // rax
  _QWORD *v28; // rbx
  _BYTE *v29; // rsi
  __int64 v30; // rax
  __int64 v31; // r13
  __int64 v32; // rcx
  __int64 v33; // r12
  __int64 v34; // rax
  char *v35; // r14
  __int64 v36; // rbx
  __int64 v37; // rax
  size_t v38; // r8
  __int64 v39; // rdx
  __int64 v40; // rcx
  __int64 v41; // rax
  int v42; // r12d
  unsigned int v43; // r13d
  char *v44; // rbx
  __int64 v45; // rsi
  char v46; // al
  __int64 v47; // rsi
  __int64 v48; // rdi
  __int64 v49; // r12
  _DWORD *v50; // r15
  _QWORD *v51; // r13
  unsigned int v52; // r9d
  _QWORD *v53; // r8
  __int64 v54; // r13
  __int64 v55; // rbx
  __int64 v56; // rsi
  __int64 v57; // r10
  __int64 *v58; // rax
  __int128 v59; // rdi
  __int64 v60; // rcx
  __int64 *v61; // rax
  char v62; // r15
  char v63; // cl
  char v64; // bl
  _QWORD *v65; // rax
  __int16 v66; // bx
  __int64 v67; // r15
  __int64 v68; // rcx
  double v69; // xmm4_8
  double v70; // xmm5_8
  void *v71; // rdi
  __int64 v72; // rax
  __int64 v73; // [rsp+8h] [rbp-E8h]
  __int64 v74; // [rsp+10h] [rbp-E0h]
  __int64 v75; // [rsp+18h] [rbp-D8h]
  __int64 v76; // [rsp+18h] [rbp-D8h]
  unsigned int v77; // [rsp+18h] [rbp-D8h]
  void *s; // [rsp+20h] [rbp-D0h]
  char v80; // [rsp+28h] [rbp-C8h]
  __int64 v81; // [rsp+28h] [rbp-C8h]
  __int64 v82; // [rsp+30h] [rbp-C0h]
  char v83; // [rsp+30h] [rbp-C0h]
  _QWORD *v84; // [rsp+30h] [rbp-C0h]
  size_t v85; // [rsp+30h] [rbp-C0h]
  bool n; // [rsp+38h] [rbp-B8h]
  size_t na; // [rsp+38h] [rbp-B8h]
  size_t nb; // [rsp+38h] [rbp-B8h]
  unsigned int nc; // [rsp+38h] [rbp-B8h]
  _QWORD v90[2]; // [rsp+40h] [rbp-B0h] BYREF
  __int16 v91; // [rsp+50h] [rbp-A0h]
  void *src; // [rsp+60h] [rbp-90h] BYREF
  __int64 v93; // [rsp+68h] [rbp-88h]
  _QWORD v94[16]; // [rsp+70h] [rbp-80h] BYREF

  v12 = (__int64)"llvm.global_ctors";
  v13 = sub_16321C0(a1, (__int64)"llvm.global_ctors", 17, 0);
  if ( !v13 )
    return 0;
  v14 = v13;
  if ( (*(_BYTE *)(v13 + 32) & 0xF) == 1 )
    return 0;
  if ( sub_15E4F60(v13) )
    return 0;
  v17 = *(_BYTE *)(v14 + 32) & 0xF;
  v18 = (v17 + 14) & 0xF;
  if ( (unsigned __int8)v18 <= 3u )
    return 0;
  if ( ((v17 + 7) & 0xFu) <= 1 )
    return 0;
  n = (*(_BYTE *)(v14 + 80) & 2) != 0;
  if ( (*(_BYTE *)(v14 + 80) & 2) != 0 )
    return 0;
  v19 = *(_QWORD *)(v14 - 24);
  if ( *(_BYTE *)(v19 + 16) != 10 )
  {
    v20 = 24LL * (*(_DWORD *)(v19 + 20) & 0xFFFFFFF);
    if ( (*(_BYTE *)(v19 + 23) & 0x40) != 0 )
    {
      v18 = *(_QWORD *)(v19 - 8);
      v21 = v18 + v20;
    }
    else
    {
      v21 = *(_QWORD *)(v14 - 24);
      v18 = v19 - v20;
    }
    for ( ; v21 != v18; v18 += 24 )
    {
      v15 = *(_QWORD *)v18;
      if ( *(_BYTE *)(*(_QWORD *)v18 + 16LL) != 10 )
      {
        v22 = *(_DWORD *)(v15 + 20) & 0xFFFFFFF;
        v12 = *(unsigned __int8 *)(*(_QWORD *)(v15 + 24 * (1 - v22)) + 16LL);
        if ( (_BYTE)v12 != 15 )
        {
          if ( (_BYTE)v12 )
            return n;
          v12 = 4 * v22;
          v15 = *(_QWORD *)(v15 - 24 * v22);
          v23 = *(_QWORD **)(v15 + 24);
          if ( *(_DWORD *)(v15 + 32) > 0x40u )
            v23 = (_QWORD *)*v23;
          if ( v23 != (_QWORD *)0xFFFF )
            return n;
        }
      }
    }
  }
  if ( sub_1593BB0(*(_QWORD *)(v14 - 24), v12, v18, v15) )
    return n;
  v24 = *(_QWORD *)(v14 - 24);
  v94[0] = 0;
  src = 0;
  v93 = 0;
  v25 = *(_DWORD *)(v24 + 20) & 0xFFFFFFF;
  if ( (*(_DWORD *)(v24 + 20) & 0xFFFFFFF) != 0 )
  {
    v26 = v25;
    v27 = (_QWORD *)sub_22077B0(8 * v25);
    v28 = v27;
    if ( v93 - (__int64)src > 0 )
    {
      memmove(v27, src, v93 - (_QWORD)src);
      j_j___libc_free_0(src, v94[0] - (_QWORD)src);
    }
    v29 = &v28[v26];
    src = v28;
    v93 = (__int64)v28;
    v94[0] = &v28[v26];
    v25 = *(_DWORD *)(v24 + 20) & 0xFFFFFFF;
  }
  else
  {
    v29 = 0;
    v28 = 0;
  }
  v30 = 24 * v25;
  if ( (*(_BYTE *)(v24 + 23) & 0x40) != 0 )
  {
    v31 = *(_QWORD *)(v24 - 8);
    v32 = v31 + v30;
  }
  else
  {
    v32 = v24;
    v31 = v24 - v30;
  }
  if ( v31 != v32 )
  {
    na = v14;
    v33 = v32;
    do
    {
      while ( 1 )
      {
        v34 = *(_QWORD *)(*(_QWORD *)v31 + 24 * (1LL - (*(_DWORD *)(*(_QWORD *)v31 + 20LL) & 0xFFFFFFF)));
        if ( *(_BYTE *)(v34 + 16) )
          v34 = 0;
        v90[0] = v34;
        if ( v29 != (_BYTE *)v28 )
          break;
        v31 += 24;
        sub_17E9700((__int64)&src, v29, v90);
        v28 = (_QWORD *)v93;
        v29 = (_BYTE *)v94[0];
        if ( v33 == v31 )
          goto LABEL_36;
      }
      if ( v28 )
      {
        *v28 = v34;
        v28 = (_QWORD *)v93;
        v29 = (_BYTE *)v94[0];
      }
      ++v28;
      v31 += 24;
      v93 = (__int64)v28;
    }
    while ( v33 != v31 );
LABEL_36:
    v14 = na;
  }
  v35 = (char *)src;
  v74 = v29 - (_BYTE *)src;
  if ( src == v28 )
  {
    n = 0;
    goto LABEL_69;
  }
  v36 = ((char *)v28 - (_BYTE *)src) >> 3;
  nb = 8LL * ((unsigned int)(v36 + 63) >> 6);
  v37 = malloc(nb);
  v38 = nb;
  v39 = (unsigned int)(v36 + 63) >> 6;
  s = (void *)v37;
  if ( !v37 )
  {
    if ( nb || (v72 = malloc(1u), v39 = (unsigned int)(v36 + 63) >> 6, v38 = 0, !v72) )
    {
      v85 = v38;
      nc = v39;
      sub_16BD1C0("Allocation failed", 1u);
      v39 = nc;
      v38 = v85;
    }
    else
    {
      s = (void *)v72;
    }
  }
  if ( (_DWORD)v39 )
    memset(s, 0, v38);
  if ( !(_DWORD)v36 || (n = 0, v40 = 0, v41 = 0, !v36) )
  {
LABEL_72:
    n = 0;
    goto LABEL_68;
  }
  v75 = v14;
  v42 = v36;
  v43 = 0;
  v82 = v36;
  do
  {
    v44 = &v35[8 * v41];
    v45 = *(_QWORD *)v44;
    if ( !*(_QWORD *)v44
      || v45 + 72 == (*(_QWORD *)(v45 + 72) & 0xFFFFFFFFFFFFFFF8LL)
      || (v46 = a2(a3, v45, v39, v40, v38)) == 0 )
    {
      v41 = ++v43;
      if ( !v42 )
        break;
      continue;
    }
    v40 = v43;
    n = v46;
    --v42;
    *(_QWORD *)v44 = 0;
    v47 = v43 >> 6;
    v41 = v43 + 1;
    v48 = 1LL << v43++;
    *((_QWORD *)s + v47) |= v48;
    if ( !v42 )
      break;
  }
  while ( v41 != v82 );
  v49 = v75;
  if ( !n )
    goto LABEL_72;
  v50 = *(_DWORD **)(v75 - 24);
  v51 = v94;
  src = v94;
  v93 = 0xA00000000LL;
  v52 = v50[5] & 0xFFFFFFF;
  if ( v52 )
  {
    v53 = v94;
    v54 = v75;
    v55 = 0;
    v56 = 0;
    do
    {
      if ( (*((_QWORD *)s + ((unsigned int)v55 >> 6)) & (1LL << v55)) == 0 )
      {
        v57 = *(_QWORD *)&v50[6 * (v55 - (v50[5] & 0xFFFFFFF))];
        if ( HIDWORD(v93) <= (unsigned int)v56 )
        {
          v77 = v52;
          v81 = *(_QWORD *)&v50[6 * (v55 - (v50[5] & 0xFFFFFFF))];
          v84 = v53;
          sub_16CD150((__int64)&src, v53, 0, 8, (int)v53, v52);
          v56 = (unsigned int)v93;
          v52 = v77;
          v57 = v81;
          v53 = v84;
        }
        *((_QWORD *)src + v56) = v57;
        v56 = (unsigned int)(v93 + 1);
        LODWORD(v93) = v93 + 1;
      }
      ++v55;
    }
    while ( v52 > (unsigned int)v55 );
    v49 = v54;
    v51 = v53;
  }
  else
  {
    v56 = 0;
  }
  v58 = sub_1645D80(*(__int64 **)(*(_QWORD *)v50 + 24LL), v56);
  *((_QWORD *)&v59 + 1) = src;
  *(_QWORD *)&v59 = v58;
  v61 = (__int64 *)sub_159DFD0(v59, (unsigned int)v93, v60);
  if ( *v61 == *(_QWORD *)v50 )
  {
    sub_15E5440(v49, (__int64)v61);
    v71 = src;
    if ( src != v51 )
      goto LABEL_67;
  }
  else
  {
    v62 = *(_BYTE *)(v49 + 80);
    v63 = *(_BYTE *)(v49 + 32);
    v76 = (__int64)v61;
    v73 = *v61;
    v91 = 257;
    v83 = v63 & 0xF;
    v80 = v62 & 1;
    v64 = *(_BYTE *)(v49 + 33) >> 2;
    v65 = sub_1648A60(88, 1u);
    v66 = v64 & 7;
    v67 = (__int64)v65;
    if ( v65 )
      sub_15E5070((__int64)v65, v73, v80, v83, v76, (__int64)v90, v66, 0, 0);
    sub_1631BE0(*(_QWORD *)(v49 + 40) + 8LL, v67);
    v68 = *(_QWORD *)(v49 + 56);
    *(_QWORD *)(v67 + 64) = v49 + 56;
    v68 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(v67 + 56) = v68 | *(_QWORD *)(v67 + 56) & 7LL;
    *(_QWORD *)(v68 + 8) = v67 + 56;
    *(_QWORD *)(v49 + 56) = *(_QWORD *)(v49 + 56) & 7LL | (v67 + 56);
    sub_164B7C0(v67, v49);
    if ( *(_QWORD *)(v49 + 8) )
    {
      if ( *(_QWORD *)v67 != *(_QWORD *)v49 )
        v67 = sub_15A4510((__int64 ***)v67, *(__int64 ***)v49, 0);
      sub_164D160(v49, v67, a4, a5, a6, a7, v69, v70, a10, a11);
    }
    sub_15E55B0(v49);
    v71 = src;
    if ( src != v51 )
LABEL_67:
      _libc_free((unsigned __int64)v71);
  }
LABEL_68:
  _libc_free((unsigned __int64)s);
LABEL_69:
  if ( v35 )
    j_j___libc_free_0(v35, v74);
  return n;
}
