// Function: sub_1BE4D70
// Address: 0x1be4d70
//
__int64 __fastcall sub_1BE4D70(__int64 a1, __int64 a2, __int64 a3, unsigned int a4, __int64 a5, __int64 *a6)
{
  unsigned int v9; // esi
  unsigned int v10; // r15d
  unsigned int v11; // ecx
  __int64 result; // rax
  __int64 *v13; // rdi
  __int64 v14; // rdx
  __int64 v15; // r13
  __int64 v16; // rdx
  unsigned int v17; // r13d
  int v18; // r10d
  unsigned __int64 v19; // rdx
  _BYTE *v20; // rdi
  int v21; // eax
  size_t v22; // rdx
  unsigned int v23; // esi
  __int64 v24; // rdi
  unsigned int v25; // ecx
  __int64 *v26; // r8
  __int64 v27; // rdx
  unsigned __int64 v28; // rdx
  unsigned __int64 v29; // rax
  int v30; // r15d
  __int64 v31; // rcx
  _BYTE *v32; // rdi
  __int64 v33; // rdx
  int v34; // eax
  int v35; // esi
  __int64 v36; // r8
  __int64 v37; // rcx
  int v38; // edx
  __int64 v39; // rdi
  int v40; // r10d
  __int64 v41; // r9
  int v42; // r10d
  int v43; // edi
  int v44; // ecx
  int v45; // ecx
  __int64 v46; // rdi
  unsigned int v47; // eax
  __int64 v48; // rsi
  int v49; // edx
  int v50; // eax
  int v51; // ecx
  __int64 v52; // rdi
  int v53; // r9d
  __int64 v54; // r15
  __int64 v55; // r8
  __int64 v56; // rsi
  int v57; // r11d
  int v58; // edi
  int v59; // ecx
  int v60; // ecx
  __int64 v61; // rdi
  __int64 *v62; // r11
  unsigned int v63; // eax
  __int64 v64; // rsi
  __int64 *v65; // r11
  int v66; // [rsp+0h] [rbp-70h]
  __int64 *v67; // [rsp+0h] [rbp-70h]
  __int64 *v68; // [rsp+8h] [rbp-68h]
  __int64 *v69; // [rsp+8h] [rbp-68h]
  unsigned __int64 v70; // [rsp+8h] [rbp-68h]
  __int64 *v71; // [rsp+8h] [rbp-68h]
  __int64 v72; // [rsp+8h] [rbp-68h]
  unsigned int v73; // [rsp+8h] [rbp-68h]
  __int64 v74; // [rsp+10h] [rbp-60h]
  void *src; // [rsp+20h] [rbp-50h] BYREF
  __int64 v77; // [rsp+28h] [rbp-48h]
  _BYTE s[64]; // [rsp+30h] [rbp-40h] BYREF

  v9 = *(_DWORD *)(a1 + 48);
  v74 = a1 + 24;
  if ( v9 )
  {
    LODWORD(a6) = v9 - 1;
    a5 = *(_QWORD *)(a1 + 32);
    v10 = ((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4);
    v11 = (v9 - 1) & v10;
    result = 5LL * v11;
    v13 = (__int64 *)(a5 + 40LL * v11);
    v14 = *v13;
    if ( *v13 == a2 )
    {
LABEL_3:
      v15 = a4;
      goto LABEL_4;
    }
    result = *v13;
    v17 = (v9 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v18 = 1;
    while ( result != -8 )
    {
      v17 = (unsigned int)a6 & (v18 + v17);
      result = *(_QWORD *)(a5 + 40LL * v17);
      if ( result == a2 )
        goto LABEL_3;
      ++v18;
    }
  }
  v19 = *(unsigned int *)(a1 + 4);
  src = s;
  v20 = s;
  v77 = 0x200000000LL;
  v21 = v19;
  if ( (unsigned int)v19 <= 2 )
  {
    v22 = 8 * v19;
    LODWORD(v77) = v21;
    if ( !v22 )
      goto LABEL_12;
    goto LABEL_11;
  }
  v66 = v19;
  v70 = v19;
  sub_16CD150((__int64)&src, s, v19, 8, a5, (int)a6);
  v20 = src;
  v22 = 8 * v70;
  LODWORD(v77) = v66;
  if ( 8 * v70 )
LABEL_11:
    memset(v20, 0, v22);
LABEL_12:
  v23 = *(_DWORD *)(a1 + 48);
  if ( v23 )
  {
    v24 = *(_QWORD *)(a1 + 32);
    v25 = (v23 - 1) & (((unsigned int)a2 >> 4) ^ ((unsigned int)a2 >> 9));
    v26 = (__int64 *)(v24 + 40LL * v25);
    v27 = *v26;
    if ( *v26 == a2 )
      goto LABEL_14;
    v57 = 1;
    a6 = 0;
    while ( v27 != -8 )
    {
      if ( !a6 && v27 == -16 )
        a6 = v26;
      v25 = (v23 - 1) & (v57 + v25);
      v26 = (__int64 *)(v24 + 40LL * v25);
      v27 = *v26;
      if ( *v26 == a2 )
        goto LABEL_14;
      ++v57;
    }
    v58 = *(_DWORD *)(a1 + 40);
    if ( a6 )
      v26 = a6;
    ++*(_QWORD *)(a1 + 24);
    v49 = v58 + 1;
    if ( 4 * (v58 + 1) < 3 * v23 )
    {
      if ( v23 - *(_DWORD *)(a1 + 44) - v49 > v23 >> 3 )
        goto LABEL_53;
      v73 = ((unsigned int)a2 >> 4) ^ ((unsigned int)a2 >> 9);
      sub_1BA1370(v74, v23);
      v59 = *(_DWORD *)(a1 + 48);
      if ( !v59 )
        goto LABEL_109;
      v60 = v59 - 1;
      v61 = *(_QWORD *)(a1 + 32);
      LODWORD(a6) = 1;
      v62 = 0;
      v63 = v60 & v73;
      v26 = (__int64 *)(v61 + 40LL * (v60 & v73));
      v64 = *v26;
      v49 = *(_DWORD *)(a1 + 40) + 1;
      if ( *v26 == a2 )
        goto LABEL_53;
      while ( v64 != -8 )
      {
        if ( !v62 && v64 == -16 )
          v62 = v26;
        v63 = v60 & ((_DWORD)a6 + v63);
        v26 = (__int64 *)(v61 + 40LL * v63);
        v64 = *v26;
        if ( *v26 == a2 )
          goto LABEL_53;
        LODWORD(a6) = (_DWORD)a6 + 1;
      }
      goto LABEL_73;
    }
  }
  else
  {
    ++*(_QWORD *)(a1 + 24);
  }
  sub_1BA1370(v74, 2 * v23);
  v44 = *(_DWORD *)(a1 + 48);
  if ( !v44 )
    goto LABEL_109;
  v45 = v44 - 1;
  v46 = *(_QWORD *)(a1 + 32);
  v47 = v45 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v26 = (__int64 *)(v46 + 40LL * v47);
  v48 = *v26;
  v49 = *(_DWORD *)(a1 + 40) + 1;
  if ( *v26 == a2 )
    goto LABEL_53;
  LODWORD(a6) = 1;
  v62 = 0;
  while ( v48 != -8 )
  {
    if ( v48 == -16 && !v62 )
      v62 = v26;
    v47 = v45 & ((_DWORD)a6 + v47);
    v26 = (__int64 *)(v46 + 40LL * v47);
    v48 = *v26;
    if ( *v26 == a2 )
      goto LABEL_53;
    LODWORD(a6) = (_DWORD)a6 + 1;
  }
LABEL_73:
  if ( v62 )
    v26 = v62;
LABEL_53:
  *(_DWORD *)(a1 + 40) = v49;
  if ( *v26 != -8 )
    --*(_DWORD *)(a1 + 44);
  *v26 = a2;
  v26[1] = (__int64)(v26 + 3);
  v26[2] = 0x200000000LL;
LABEL_14:
  if ( v26 + 1 != (__int64 *)&src )
  {
    v28 = (unsigned int)v77;
    v29 = *((unsigned int *)v26 + 4);
    v30 = v77;
    if ( (unsigned int)v77 > v29 )
    {
      if ( (unsigned int)v77 > (unsigned __int64)*((unsigned int *)v26 + 5) )
      {
        *((_DWORD *)v26 + 4) = 0;
        v71 = v26;
        sub_16CD150((__int64)(v26 + 1), v26 + 3, v28, 8, (int)v26, (int)a6);
        v28 = (unsigned int)v77;
        v26 = v71;
        v31 = 0;
      }
      else
      {
        v31 = 8 * v29;
        if ( *((_DWORD *)v26 + 4) )
        {
          v67 = v26;
          v72 = 8 * v29;
          memmove((void *)v26[1], src, 8 * v29);
          v28 = (unsigned int)v77;
          v26 = v67;
          v31 = v72;
        }
      }
      v32 = src;
      v33 = 8 * v28;
      if ( (char *)src + v31 != (char *)src + v33 )
      {
        v68 = v26;
        memcpy((void *)(v31 + v26[1]), (char *)src + v31, v33 - v31);
        v32 = src;
        v26 = v68;
      }
      *((_DWORD *)v26 + 4) = v30;
      goto LABEL_26;
    }
    if ( (_DWORD)v77 )
    {
      v69 = v26;
      memmove((void *)v26[1], src, 8LL * (unsigned int)v77);
      v26 = v69;
    }
    *((_DWORD *)v26 + 4) = v30;
  }
  v32 = src;
LABEL_26:
  if ( v32 != s )
    _libc_free((unsigned __int64)v32);
  v9 = *(_DWORD *)(a1 + 48);
  v15 = a4;
  if ( !v9 )
  {
    ++*(_QWORD *)(a1 + 24);
    goto LABEL_30;
  }
  LODWORD(a6) = v9 - 1;
  a5 = *(_QWORD *)(a1 + 32);
  v10 = ((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4);
  v11 = v10 & (v9 - 1);
  result = 5LL * v11;
  v13 = (__int64 *)(a5 + 40LL * v11);
  v14 = *v13;
LABEL_4:
  if ( v14 == a2 )
  {
    v16 = v13[1];
    goto LABEL_6;
  }
  v42 = 1;
  result = 0;
  while ( v14 != -8 )
  {
    if ( v14 != -16 || result )
      v13 = (__int64 *)result;
    result = (unsigned int)(v42 + 1);
    v11 = (unsigned int)a6 & (v42 + v11);
    v65 = (__int64 *)(a5 + 40LL * v11);
    v14 = *v65;
    if ( *v65 == a2 )
    {
      v16 = v65[1];
      goto LABEL_6;
    }
    ++v42;
    result = (__int64)v13;
    v13 = (__int64 *)(a5 + 40LL * v11);
  }
  if ( !result )
    result = (__int64)v13;
  v43 = *(_DWORD *)(a1 + 40);
  ++*(_QWORD *)(a1 + 24);
  v38 = v43 + 1;
  if ( 4 * (v43 + 1) >= 3 * v9 )
  {
LABEL_30:
    sub_1BA1370(v74, 2 * v9);
    v34 = *(_DWORD *)(a1 + 48);
    if ( v34 )
    {
      v35 = v34 - 1;
      v36 = *(_QWORD *)(a1 + 32);
      LODWORD(v37) = (v34 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v38 = *(_DWORD *)(a1 + 40) + 1;
      result = v36 + 40LL * (unsigned int)v37;
      v39 = *(_QWORD *)result;
      if ( *(_QWORD *)result != a2 )
      {
        v40 = 1;
        v41 = 0;
        while ( v39 != -8 )
        {
          if ( v39 == -16 && !v41 )
            v41 = result;
          v37 = v35 & (unsigned int)(v37 + v40);
          result = v36 + 40 * v37;
          v39 = *(_QWORD *)result;
          if ( *(_QWORD *)result == a2 )
            goto LABEL_47;
          ++v40;
        }
        if ( v41 )
          result = v41;
      }
      goto LABEL_47;
    }
    goto LABEL_109;
  }
  if ( v9 - (v38 + *(_DWORD *)(a1 + 44)) <= v9 >> 3 )
  {
    sub_1BA1370(v74, v9);
    v50 = *(_DWORD *)(a1 + 48);
    if ( v50 )
    {
      v51 = v50 - 1;
      v52 = *(_QWORD *)(a1 + 32);
      v53 = 1;
      LODWORD(v54) = (v50 - 1) & v10;
      v55 = 0;
      v38 = *(_DWORD *)(a1 + 40) + 1;
      result = v52 + 40LL * (unsigned int)v54;
      v56 = *(_QWORD *)result;
      if ( *(_QWORD *)result != a2 )
      {
        while ( v56 != -8 )
        {
          if ( v56 == -16 && !v55 )
            v55 = result;
          v54 = v51 & (unsigned int)(v54 + v53);
          result = v52 + 40 * v54;
          v56 = *(_QWORD *)result;
          if ( *(_QWORD *)result == a2 )
            goto LABEL_47;
          ++v53;
        }
        if ( v55 )
          result = v55;
      }
      goto LABEL_47;
    }
LABEL_109:
    ++*(_DWORD *)(a1 + 40);
    BUG();
  }
LABEL_47:
  *(_DWORD *)(a1 + 40) = v38;
  if ( *(_QWORD *)result != -8 )
    --*(_DWORD *)(a1 + 44);
  v16 = result + 24;
  *(_QWORD *)result = a2;
  *(_QWORD *)(result + 8) = result + 24;
  *(_QWORD *)(result + 16) = 0x200000000LL;
LABEL_6:
  *(_QWORD *)(v16 + 8 * v15) = a3;
  return result;
}
