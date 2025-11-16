// Function: sub_242A560
// Address: 0x242a560
//
__int64 __fastcall sub_242A560(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned int v8; // esi
  unsigned int v9; // r8d
  __int64 v10; // rdi
  int v11; // r9d
  _QWORD *v12; // rcx
  unsigned int v13; // r11d
  _QWORD *v14; // rax
  __int64 v15; // r10
  int v16; // r9d
  _QWORD *v17; // rdx
  unsigned int v18; // r10d
  _QWORD *v19; // rax
  __int64 v20; // rcx
  __int64 v21; // rax
  __int64 v22; // r15
  __int64 *v23; // r13
  __int64 v24; // rax
  unsigned int v26; // r15d
  int v27; // eax
  _QWORD *v28; // rax
  unsigned __int64 v29; // rdi
  int v30; // eax
  int v31; // edi
  __int64 v32; // rsi
  unsigned int v33; // eax
  int v34; // ecx
  __int64 v35; // r9
  int v36; // r11d
  _QWORD *v37; // r10
  int v38; // eax
  _QWORD *v39; // rax
  unsigned __int64 v40; // rdi
  unsigned __int64 v41; // r14
  char *v42; // rcx
  __int64 v43; // rax
  __int64 v44; // rsi
  bool v45; // cf
  unsigned __int64 v46; // rax
  unsigned __int64 v47; // r12
  __int64 v48; // rax
  unsigned __int64 v49; // r12
  __int64 *v50; // rcx
  _QWORD *v51; // r15
  unsigned __int64 *v52; // r12
  __int64 v53; // rax
  unsigned __int64 v54; // rdi
  int v55; // esi
  int v56; // esi
  __int64 v57; // r10
  unsigned int v58; // edx
  __int64 v59; // rdi
  int v60; // r15d
  _QWORD *v61; // r11
  int v62; // esi
  int v63; // esi
  __int64 v64; // r10
  _QWORD *v65; // r9
  int v66; // r11d
  unsigned int v67; // edx
  __int64 v68; // rdi
  int v69; // eax
  int v70; // eax
  __int64 v71; // rdi
  _QWORD *v72; // r9
  unsigned int v73; // r15d
  int v74; // r10d
  __int64 v75; // rsi
  unsigned int v76; // [rsp+8h] [rbp-48h]
  __int64 v77; // [rsp+8h] [rbp-48h]
  unsigned int v78; // [rsp+8h] [rbp-48h]
  _QWORD *v79; // [rsp+10h] [rbp-40h]
  unsigned int v80; // [rsp+10h] [rbp-40h]
  unsigned int v81; // [rsp+10h] [rbp-40h]
  unsigned __int64 v82; // [rsp+10h] [rbp-40h]
  unsigned int v83; // [rsp+10h] [rbp-40h]
  unsigned int v84; // [rsp+10h] [rbp-40h]
  unsigned int v85; // [rsp+10h] [rbp-40h]
  __int64 v86; // [rsp+18h] [rbp-38h]
  _QWORD *v87; // [rsp+18h] [rbp-38h]
  __int64 v88; // [rsp+18h] [rbp-38h]

  v8 = *(_DWORD *)(a1 + 56);
  v9 = *(_DWORD *)(a1 + 48);
  v86 = a1 + 32;
  if ( v8 )
  {
    v10 = *(_QWORD *)(a1 + 40);
    v11 = 1;
    v12 = 0;
    v13 = (v8 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v14 = (_QWORD *)(v10 + 16LL * v13);
    v15 = *v14;
    if ( a2 == *v14 )
      goto LABEL_3;
    while ( v15 != -4096 )
    {
      if ( v12 || v15 != -8192 )
        v14 = v12;
      v13 = (v8 - 1) & (v11 + v13);
      v15 = *(_QWORD *)(v10 + 16LL * v13);
      if ( a2 == v15 )
        goto LABEL_3;
      ++v11;
      v12 = v14;
      v14 = (_QWORD *)(v10 + 16LL * v13);
    }
    v26 = v9 + 1;
    if ( !v12 )
      v12 = v14;
    v27 = v9 + 1;
    ++*(_QWORD *)(a1 + 32);
    if ( 4 * (v9 + 1) < 3 * v8 )
    {
      if ( v8 - *(_DWORD *)(a1 + 52) - v26 <= v8 >> 3 )
      {
        v78 = ((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4);
        v84 = v9;
        sub_24263F0(v86, v8);
        v62 = *(_DWORD *)(a1 + 56);
        if ( !v62 )
          goto LABEL_119;
        v63 = v62 - 1;
        v64 = *(_QWORD *)(a1 + 40);
        v65 = 0;
        v9 = v84;
        v66 = 1;
        v67 = v63 & v78;
        v27 = *(_DWORD *)(a1 + 48) + 1;
        v12 = (_QWORD *)(v64 + 16LL * (v63 & v78));
        v68 = *v12;
        if ( a2 != *v12 )
        {
          while ( v68 != -4096 )
          {
            if ( v68 == -8192 && !v65 )
              v65 = v12;
            v67 = v63 & (v66 + v67);
            v12 = (_QWORD *)(v64 + 16LL * v67);
            v68 = *v12;
            if ( a2 == *v12 )
              goto LABEL_20;
            ++v66;
          }
          if ( v65 )
            v12 = v65;
        }
      }
      goto LABEL_20;
    }
  }
  else
  {
    ++*(_QWORD *)(a1 + 32);
  }
  v83 = v9;
  sub_24263F0(v86, 2 * v8);
  v55 = *(_DWORD *)(a1 + 56);
  if ( !v55 )
    goto LABEL_119;
  v56 = v55 - 1;
  v57 = *(_QWORD *)(a1 + 40);
  v9 = v83;
  v58 = v56 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v26 = v83 + 1;
  v27 = *(_DWORD *)(a1 + 48) + 1;
  v12 = (_QWORD *)(v57 + 16LL * v58);
  v59 = *v12;
  if ( a2 != *v12 )
  {
    v60 = 1;
    v61 = 0;
    while ( v59 != -4096 )
    {
      if ( !v61 && v59 == -8192 )
        v61 = v12;
      v58 = v56 & (v60 + v58);
      v12 = (_QWORD *)(v57 + 16LL * v58);
      v59 = *v12;
      if ( a2 == *v12 )
        goto LABEL_81;
      ++v60;
    }
    if ( v61 )
      v12 = v61;
LABEL_81:
    v26 = v83 + 1;
  }
LABEL_20:
  *(_DWORD *)(a1 + 48) = v27;
  if ( *v12 != -4096 )
    --*(_DWORD *)(a1 + 52);
  *v12 = a2;
  v12[1] = 0;
  v79 = v12;
  v76 = v9;
  v28 = (_QWORD *)sub_22077B0(0x10u);
  if ( v28 )
  {
    *v28 = v28;
    v28[1] = v76;
  }
  v29 = v79[1];
  v79[1] = v28;
  if ( v29 )
    j_j___libc_free_0(v29);
  v8 = *(_DWORD *)(a1 + 56);
  v9 = v26;
  if ( !v8 )
  {
    ++*(_QWORD *)(a1 + 32);
    goto LABEL_28;
  }
  v10 = *(_QWORD *)(a1 + 40);
LABEL_3:
  v16 = 1;
  v17 = 0;
  v18 = (v8 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
  v19 = (_QWORD *)(v10 + 16LL * v18);
  v20 = *v19;
  if ( a3 == *v19 )
    goto LABEL_4;
  while ( v20 != -4096 )
  {
    if ( v20 != -8192 || v17 )
      v19 = v17;
    v18 = (v8 - 1) & (v16 + v18);
    v20 = *(_QWORD *)(v10 + 16LL * v18);
    if ( a3 == v20 )
      goto LABEL_4;
    ++v16;
    v17 = v19;
    v19 = (_QWORD *)(v10 + 16LL * v18);
  }
  if ( !v17 )
    v17 = v19;
  v38 = *(_DWORD *)(a1 + 48);
  ++*(_QWORD *)(a1 + 32);
  v34 = v38 + 1;
  if ( 4 * (v38 + 1) >= 3 * v8 )
  {
LABEL_28:
    v80 = v9;
    sub_24263F0(v86, 2 * v8);
    v30 = *(_DWORD *)(a1 + 56);
    if ( v30 )
    {
      v31 = v30 - 1;
      v32 = *(_QWORD *)(a1 + 40);
      v9 = v80;
      v33 = (v30 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
      v34 = *(_DWORD *)(a1 + 48) + 1;
      v17 = (_QWORD *)(v32 + 16LL * v33);
      v35 = *v17;
      if ( *v17 != a3 )
      {
        v36 = 1;
        v37 = 0;
        while ( v35 != -4096 )
        {
          if ( !v37 && v35 == -8192 )
            v37 = v17;
          v33 = v31 & (v36 + v33);
          v17 = (_QWORD *)(v32 + 16LL * v33);
          v35 = *v17;
          if ( a3 == *v17 )
            goto LABEL_44;
          ++v36;
        }
        if ( v37 )
          v17 = v37;
      }
      goto LABEL_44;
    }
    goto LABEL_119;
  }
  if ( v8 - *(_DWORD *)(a1 + 52) - v34 <= v8 >> 3 )
  {
    v85 = v9;
    sub_24263F0(v86, v8);
    v69 = *(_DWORD *)(a1 + 56);
    if ( v69 )
    {
      v70 = v69 - 1;
      v71 = *(_QWORD *)(a1 + 40);
      v72 = 0;
      v73 = v70 & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
      v9 = v85;
      v74 = 1;
      v34 = *(_DWORD *)(a1 + 48) + 1;
      v17 = (_QWORD *)(v71 + 16LL * v73);
      v75 = *v17;
      if ( a3 != *v17 )
      {
        while ( v75 != -4096 )
        {
          if ( !v72 && v75 == -8192 )
            v72 = v17;
          v73 = v70 & (v74 + v73);
          v17 = (_QWORD *)(v71 + 16LL * v73);
          v75 = *v17;
          if ( a3 == *v17 )
            goto LABEL_44;
          ++v74;
        }
        if ( v72 )
          v17 = v72;
      }
      goto LABEL_44;
    }
LABEL_119:
    ++*(_DWORD *)(a1 + 48);
    BUG();
  }
LABEL_44:
  *(_DWORD *)(a1 + 48) = v34;
  if ( *v17 != -4096 )
    --*(_DWORD *)(a1 + 52);
  *v17 = a3;
  v17[1] = 0;
  v87 = v17;
  v81 = v9;
  v39 = (_QWORD *)sub_22077B0(0x10u);
  if ( v39 )
  {
    *v39 = v39;
    v39[1] = v81;
  }
  v40 = v87[1];
  v87[1] = v39;
  if ( v40 )
    j_j___libc_free_0(v40);
LABEL_4:
  v21 = sub_22077B0(0x30u);
  v22 = v21;
  if ( v21 )
  {
    *(_QWORD *)v21 = a2;
    *(_QWORD *)(v21 + 8) = a3;
    *(_QWORD *)(v21 + 16) = a4;
    *(_QWORD *)(v21 + 24) = 0;
    *(_WORD *)(v21 + 40) = 0;
    *(_BYTE *)(v21 + 42) = 0;
  }
  v23 = *(__int64 **)(a1 + 16);
  if ( v23 != *(__int64 **)(a1 + 24) )
  {
    if ( v23 )
    {
      *v23 = v21;
      v23 = *(__int64 **)(a1 + 16);
    }
    v24 = (__int64)(v23 + 1);
    *(_QWORD *)(a1 + 16) = v23 + 1;
    return *(_QWORD *)(v24 - 8);
  }
  v41 = *(_QWORD *)(a1 + 8);
  v42 = (char *)v23 - v41;
  v43 = (__int64)((__int64)v23 - v41) >> 3;
  if ( v43 == 0xFFFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v44 = 1;
  if ( v43 )
    v44 = (__int64)((__int64)v23 - v41) >> 3;
  v45 = __CFADD__(v44, v43);
  v46 = v44 + v43;
  if ( v45 )
  {
    v47 = 0x7FFFFFFFFFFFFFF8LL;
LABEL_58:
    v48 = sub_22077B0(v47);
    v42 = (char *)v23 - v41;
    v49 = v48 + v47;
    v88 = v48;
    v24 = v48 + 8;
    v82 = v49;
    goto LABEL_59;
  }
  if ( v46 )
  {
    if ( v46 > 0xFFFFFFFFFFFFFFFLL )
      v46 = 0xFFFFFFFFFFFFFFFLL;
    v47 = 8 * v46;
    goto LABEL_58;
  }
  v82 = 0;
  v24 = 8;
  v88 = 0;
LABEL_59:
  v50 = (__int64 *)&v42[v88];
  if ( v50 )
    *v50 = v22;
  if ( v23 != (__int64 *)v41 )
  {
    v51 = (_QWORD *)v88;
    v52 = (unsigned __int64 *)v41;
    while ( 1 )
    {
      v54 = *v52;
      if ( v51 )
        break;
      if ( !v54 )
        goto LABEL_64;
      ++v52;
      j_j___libc_free_0(v54);
      v53 = 8;
      if ( v23 == (__int64 *)v52 )
      {
LABEL_69:
        v24 = (__int64)(v51 + 2);
        goto LABEL_70;
      }
LABEL_65:
      v51 = (_QWORD *)v53;
    }
    *v51 = v54;
    *v52 = 0;
LABEL_64:
    ++v52;
    v53 = (__int64)(v51 + 1);
    if ( v23 == (__int64 *)v52 )
      goto LABEL_69;
    goto LABEL_65;
  }
LABEL_70:
  if ( v41 )
  {
    v77 = v24;
    j_j___libc_free_0(v41);
    v24 = v77;
  }
  *(_QWORD *)(a1 + 16) = v24;
  *(_QWORD *)(a1 + 8) = v88;
  *(_QWORD *)(a1 + 24) = v82;
  return *(_QWORD *)(v24 - 8);
}
