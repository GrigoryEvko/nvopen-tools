// Function: sub_11C0160
// Address: 0x11c0160
//
void __fastcall sub_11C0160(__int64 a1, __int64 a2, _QWORD *k, __int64 a4, __int64 a5, __int64 *a6)
{
  unsigned int v7; // r14d
  int v8; // r14d
  _QWORD *v9; // rbx
  __int64 v10; // r13
  _QWORD *i; // r13
  _QWORD *v12; // rdi
  __int64 v13; // r13
  __int64 v14; // rbx
  __int64 v15; // r15
  __int64 v16; // r8
  __int64 v17; // rdi
  int v18; // edx
  unsigned int v19; // ecx
  __int64 v20; // r9
  __int64 v21; // r10
  __int64 v22; // rax
  _QWORD *v23; // r14
  __int64 v24; // r13
  __int64 v25; // rax
  unsigned int v26; // r14d
  int v27; // eax
  bool v28; // al
  __int64 v29; // r14
  char v30; // si
  __int64 v31; // rdx
  char v32; // al
  _QWORD *v33; // r14
  _QWORD *v34; // r12
  _QWORD *v35; // rbx
  __int64 *v36; // r13
  __int64 v37; // r12
  unsigned __int64 v38; // rax
  __int64 *v39; // r15
  __int64 v40; // r12
  __int64 *m; // r14
  __int64 v42; // rsi
  __int64 *v43; // r13
  __int64 v44; // rcx
  _QWORD *v45; // rax
  unsigned int v46; // edi
  __int64 v47; // r9
  __int64 v48; // rbx
  __int64 v49; // rax
  int v50; // r11d
  __int64 v51; // r9
  int v52; // ecx
  unsigned int v53; // edx
  __int64 v54; // r11
  __int64 v55; // r9
  int v56; // ecx
  unsigned int v57; // edx
  __int64 v58; // r11
  int v59; // edi
  _QWORD *v60; // rsi
  int v61; // ecx
  int v62; // ecx
  _QWORD *v63; // r13
  _QWORD *v64; // rdi
  char v65; // al
  bool v66; // zf
  __int64 v67; // rax
  _QWORD *j; // rax
  unsigned int v69; // edx
  unsigned int v70; // ebx
  __int64 v71; // rdi
  __int64 v72; // rax
  _QWORD *v73; // rax
  __int64 v74; // rdx
  __int64 v75; // rax
  int v76; // edi
  __int64 v77; // rax
  __int64 v78; // rax
  unsigned __int8 v79; // [rsp+Fh] [rbp-41h]
  __int64 v80; // [rsp+10h] [rbp-40h]
  __int64 *v81; // [rsp+10h] [rbp-40h]
  unsigned __int8 v82; // [rsp+18h] [rbp-38h]
  unsigned __int8 v83; // [rsp+18h] [rbp-38h]
  _QWORD *v84; // [rsp+18h] [rbp-38h]
  __int64 v85; // [rsp+18h] [rbp-38h]
  unsigned __int8 v86; // [rsp+18h] [rbp-38h]
  unsigned __int8 v87; // [rsp+18h] [rbp-38h]

  v7 = *(_DWORD *)(a1 + 96);
  ++*(_QWORD *)(a1 + 88);
  v82 = a2;
  v8 = v7 >> 1;
  v80 = a1 + 88;
  if ( v8 )
  {
    if ( (*(_BYTE *)(a1 + 96) & 1) == 0 )
    {
      a2 = (unsigned int)(4 * v8);
      goto LABEL_4;
    }
LABEL_65:
    v9 = (_QWORD *)(a1 + 104);
    v10 = 56;
LABEL_6:
    for ( i = &v9[v10]; i != v9; v9 += 7 )
    {
      if ( *v9 != -4096 )
      {
        if ( *v9 != -8192 )
        {
          v12 = (_QWORD *)v9[1];
          if ( v12 != v9 + 3 )
            _libc_free(v12, a2);
        }
        *v9 = -4096;
      }
    }
    *(_QWORD *)(a1 + 96) &= 1uLL;
    goto LABEL_14;
  }
  if ( !*(_DWORD *)(a1 + 100) )
    goto LABEL_14;
  a2 = 0;
  if ( (*(_BYTE *)(a1 + 96) & 1) != 0 )
    goto LABEL_65;
LABEL_4:
  a4 = *(unsigned int *)(a1 + 112);
  v9 = *(_QWORD **)(a1 + 104);
  v10 = 7 * a4;
  if ( (unsigned int)a4 <= (unsigned int)a2 || (unsigned int)a4 <= 0x40 )
    goto LABEL_6;
  v63 = &v9[7 * a4];
  do
  {
    if ( *v9 != -4096 && *v9 != -8192 )
    {
      v64 = (_QWORD *)v9[1];
      if ( v64 != v9 + 3 )
        _libc_free(v64, a2);
    }
    v9 += 7;
  }
  while ( v9 != v63 );
  v65 = *(_BYTE *)(a1 + 96);
  a2 = v65 & 1;
  if ( !v8 )
  {
    if ( (_BYTE)a2 )
      goto LABEL_101;
LABEL_136:
    v77 = *(unsigned int *)(a1 + 112);
    if ( (_DWORD)v77 != v8 )
    {
      a2 = 56 * v77;
      sub_C7D6A0(*(_QWORD *)(a1 + 104), 56 * v77, 8);
      *(_BYTE *)(a1 + 96) |= 1u;
      goto LABEL_113;
    }
LABEL_101:
    v66 = (*(_QWORD *)(a1 + 96) & 1LL) == 0;
    *(_QWORD *)(a1 + 96) &= 1uLL;
    if ( v66 )
    {
      a4 = *(unsigned int *)(a1 + 112);
      k = *(_QWORD **)(a1 + 104);
      v67 = 7 * a4;
    }
    else
    {
      k = (_QWORD *)(a1 + 104);
      v67 = 56;
    }
    for ( j = &k[v67]; j != k; k += 7 )
    {
      if ( k )
        *k = -4096;
    }
    goto LABEL_14;
  }
  if ( v8 == 1 )
  {
    v8 = 2;
    if ( (_BYTE)a2 )
      goto LABEL_101;
    goto LABEL_136;
  }
  _BitScanReverse(&v69, v8 - 1);
  a4 = 33 - (v69 ^ 0x1F);
  v70 = 1 << (33 - (v69 ^ 0x1F));
  if ( v70 - 9 <= 0x36 )
  {
    if ( (_BYTE)a2 )
    {
      v71 = 3584;
      v70 = 64;
      goto LABEL_112;
    }
    v75 = *(unsigned int *)(a1 + 112);
    if ( (_DWORD)v75 == 64 )
      goto LABEL_101;
    v70 = 64;
    sub_C7D6A0(*(_QWORD *)(a1 + 104), 56 * v75, 8);
    v65 = *(_BYTE *)(a1 + 96);
    goto LABEL_127;
  }
  if ( (_BYTE)a2 )
  {
    if ( v70 <= 8 )
      goto LABEL_101;
    v71 = 56LL * v70;
LABEL_112:
    a2 = 8;
    *(_BYTE *)(a1 + 96) = v65 & 0xFE;
    v72 = sub_C7D670(v71, 8);
    *(_DWORD *)(a1 + 112) = v70;
    *(_QWORD *)(a1 + 104) = v72;
    goto LABEL_113;
  }
  v78 = *(unsigned int *)(a1 + 112);
  if ( v70 == (_DWORD)v78 )
    goto LABEL_101;
  a2 = 56 * v78;
  sub_C7D6A0(*(_QWORD *)(a1 + 104), 56 * v78, 8);
  v65 = *(_BYTE *)(a1 + 96) | 1;
  *(_BYTE *)(a1 + 96) = v65;
  if ( v70 > 8 )
  {
LABEL_127:
    v71 = 56LL * v70;
    goto LABEL_112;
  }
LABEL_113:
  v66 = (*(_QWORD *)(a1 + 96) & 1LL) == 0;
  *(_QWORD *)(a1 + 96) &= 1uLL;
  if ( v66 )
  {
    a4 = *(unsigned int *)(a1 + 112);
    v73 = *(_QWORD **)(a1 + 104);
    v74 = 7 * a4;
  }
  else
  {
    v73 = (_QWORD *)(a1 + 104);
    v74 = 56;
  }
  for ( k = &v73[v74]; k != v73; v73 += 7 )
  {
    if ( v73 )
      *v73 = -4096;
  }
LABEL_14:
  v13 = *(_QWORD *)(a1 + 8);
  if ( !*(_BYTE *)(v13 + 192) )
    sub_CFDFC0(*(_QWORD *)(a1 + 8), a2, (__int64)k, a4, a5, a6);
  v14 = *(_QWORD *)(v13 + 16);
  if ( v14 + 32LL * *(unsigned int *)(v13 + 24) != v14 )
  {
    v15 = v14 + 32LL * *(unsigned int *)(v13 + 24);
    v16 = v82;
    while ( 1 )
    {
      v24 = *(_QWORD *)(v14 + 16);
      if ( !v24 )
        goto LABEL_23;
      if ( (_BYTE)v16 )
      {
        v25 = *(_QWORD *)(v24 - 32LL * (*(_DWORD *)(v24 + 4) & 0x7FFFFFF));
        if ( *(_BYTE *)v25 != 17 )
          goto LABEL_23;
        v26 = *(_DWORD *)(v25 + 32);
        if ( v26 <= 0x40 )
        {
          v28 = *(_QWORD *)(v25 + 24) == 0;
        }
        else
        {
          v83 = v16;
          v27 = sub_C444A0(v25 + 24);
          v16 = v83;
          v28 = v26 == v27;
        }
        if ( v28 )
          goto LABEL_23;
      }
      v29 = *(_QWORD *)(v24 + 40);
      v30 = *(_BYTE *)(a1 + 96) & 1;
      if ( v30 )
      {
        v17 = a1 + 104;
        v18 = 7;
      }
      else
      {
        v31 = *(unsigned int *)(a1 + 112);
        v17 = *(_QWORD *)(a1 + 104);
        if ( !(_DWORD)v31 )
        {
          v44 = *(unsigned int *)(a1 + 96);
          ++*(_QWORD *)(a1 + 88);
          v45 = 0;
          v46 = ((unsigned int)v44 >> 1) + 1;
          goto LABEL_55;
        }
        v18 = v31 - 1;
      }
      v19 = v18 & (((unsigned int)v29 >> 9) ^ ((unsigned int)v29 >> 4));
      v20 = v17 + 56LL * v19;
      v21 = *(_QWORD *)v20;
      if ( v29 != *(_QWORD *)v20 )
        break;
LABEL_20:
      v22 = *(unsigned int *)(v20 + 16);
      v23 = (_QWORD *)(v20 + 8);
      if ( *(unsigned int *)(v20 + 20) < (unsigned __int64)(v22 + 1) )
      {
        v79 = v16;
        v85 = v20;
        sub_C8D5F0(v20 + 8, (const void *)(v20 + 24), v22 + 1, 8u, v16, v20);
        v16 = v79;
        v22 = *(unsigned int *)(v85 + 16);
      }
LABEL_22:
      *(_QWORD *)(*v23 + 8 * v22) = v24;
      ++*((_DWORD *)v23 + 2);
LABEL_23:
      v14 += 32;
      if ( v15 == v14 )
        goto LABEL_33;
    }
    v50 = 1;
    v45 = 0;
    while ( v21 != -4096 )
    {
      if ( v45 || v21 != -8192 )
        v20 = (__int64)v45;
      v19 = v18 & (v50 + v19);
      v21 = *(_QWORD *)(v17 + 56LL * v19);
      if ( v29 == v21 )
      {
        v20 = v17 + 56LL * v19;
        goto LABEL_20;
      }
      ++v50;
      v45 = (_QWORD *)v20;
      v20 = v17 + 56LL * v19;
    }
    v44 = *(unsigned int *)(a1 + 96);
    v31 = 8;
    if ( !v45 )
      v45 = (_QWORD *)v20;
    ++*(_QWORD *)(a1 + 88);
    v47 = 24;
    v46 = ((unsigned int)v44 >> 1) + 1;
    if ( !v30 )
    {
      v31 = *(unsigned int *)(a1 + 112);
LABEL_55:
      v47 = (unsigned int)(3 * v31);
    }
    if ( (unsigned int)v47 <= 4 * v46 )
    {
      v86 = v16;
      sub_11BFBC0(v80, (_QWORD *)(unsigned int)(2 * v31), v31, v44, v16, v47);
      v16 = v86;
      if ( (*(_BYTE *)(a1 + 96) & 1) != 0 )
      {
        v51 = a1 + 104;
        v52 = 7;
      }
      else
      {
        v61 = *(_DWORD *)(a1 + 112);
        v51 = *(_QWORD *)(a1 + 104);
        if ( !v61 )
          goto LABEL_152;
        v52 = v61 - 1;
      }
      v53 = v52 & (((unsigned int)v29 >> 9) ^ ((unsigned int)v29 >> 4));
      v45 = (_QWORD *)(v51 + 56LL * v53);
      v54 = *v45;
      if ( v29 == *v45 )
        goto LABEL_81;
      v76 = 1;
      v60 = 0;
      while ( v54 != -4096 )
      {
        if ( !v60 && v54 == -8192 )
          v60 = v45;
        v53 = v52 & (v53 + v76);
        v45 = (_QWORD *)(v51 + 56LL * v53);
        v54 = *v45;
        if ( v29 == *v45 )
          goto LABEL_81;
        ++v76;
      }
    }
    else
    {
      if ( (_DWORD)v31 - *(_DWORD *)(a1 + 100) - v46 > (unsigned int)v31 >> 3 )
      {
LABEL_58:
        *(_DWORD *)(a1 + 96) = (2 * ((unsigned int)v44 >> 1) + 2) | v44 & 1;
        if ( *v45 != -4096 )
          --*(_DWORD *)(a1 + 100);
        *v45 = v29;
        v23 = v45 + 1;
        v45[1] = v45 + 3;
        v45[2] = 0x400000000LL;
        v22 = 0;
        goto LABEL_22;
      }
      v87 = v16;
      sub_11BFBC0(v80, (_QWORD *)(unsigned int)v31, v31, v44, v16, v47);
      v16 = v87;
      if ( (*(_BYTE *)(a1 + 96) & 1) != 0 )
      {
        v55 = a1 + 104;
        v56 = 7;
      }
      else
      {
        v62 = *(_DWORD *)(a1 + 112);
        v55 = *(_QWORD *)(a1 + 104);
        if ( !v62 )
        {
LABEL_152:
          *(_DWORD *)(a1 + 96) = (2 * (*(_DWORD *)(a1 + 96) >> 1) + 2) | *(_DWORD *)(a1 + 96) & 1;
          BUG();
        }
        v56 = v62 - 1;
      }
      v57 = v56 & (((unsigned int)v29 >> 9) ^ ((unsigned int)v29 >> 4));
      v45 = (_QWORD *)(v55 + 56LL * v57);
      v58 = *v45;
      if ( v29 == *v45 )
      {
LABEL_81:
        LODWORD(v44) = *(_DWORD *)(a1 + 96);
        goto LABEL_58;
      }
      v59 = 1;
      v60 = 0;
      while ( v58 != -4096 )
      {
        if ( !v60 && v58 == -8192 )
          v60 = v45;
        v57 = v56 & (v57 + v59);
        v45 = (_QWORD *)(v55 + 56LL * v57);
        v58 = *v45;
        if ( v29 == *v45 )
          goto LABEL_81;
        ++v59;
      }
    }
    if ( v60 )
      v45 = v60;
    goto LABEL_81;
  }
LABEL_33:
  v32 = *(_BYTE *)(a1 + 96) & 1;
  if ( *(_DWORD *)(a1 + 96) >> 1 )
  {
    if ( v32 )
    {
      v33 = (_QWORD *)(a1 + 104);
      v34 = (_QWORD *)(a1 + 552);
      goto LABEL_36;
    }
    v33 = *(_QWORD **)(a1 + 104);
    v34 = &v33[7 * *(unsigned int *)(a1 + 112)];
    if ( v34 == v33 )
    {
LABEL_39:
      v35 = v33;
    }
    else
    {
LABEL_36:
      while ( *v33 == -8192 || *v33 == -4096 )
      {
        v33 += 7;
        if ( v33 == v34 )
          goto LABEL_39;
      }
      v35 = v33;
      v33 = v34;
    }
  }
  else
  {
    if ( v32 )
    {
      v48 = a1 + 104;
      v49 = 448;
    }
    else
    {
      v48 = *(_QWORD *)(a1 + 104);
      v49 = 56LL * *(unsigned int *)(a1 + 112);
    }
    v35 = (_QWORD *)(v49 + v48);
    v33 = v35;
  }
  if ( v35 != v33 )
  {
    v84 = v33;
LABEL_42:
    v36 = (__int64 *)v35[1];
    v37 = *((unsigned int *)v35 + 4);
    v81 = &v36[v37];
    if ( v36 != &v36[v37] )
    {
      _BitScanReverse64(&v38, (v37 * 8) >> 3);
      sub_11BE450(v36, &v36[v37], 2LL * (int)(63 - (v38 ^ 0x3F)));
      if ( (unsigned __int64)v37 <= 16 )
      {
        sub_11BDD40(v36, v81);
      }
      else
      {
        v39 = v36 + 16;
        sub_11BDD40(v36, v36 + 16);
        if ( &v36[v37] != v36 + 16 )
        {
          do
          {
            v40 = *v39;
            for ( m = v39; ; m[1] = *m )
            {
              v42 = *(m - 1);
              v43 = m--;
              if ( !sub_B445A0(v40, v42) )
                break;
            }
            *v43 = v40;
            ++v39;
          }
          while ( v81 != v39 );
        }
      }
    }
    while ( 1 )
    {
      v35 += 7;
      if ( v35 == v84 )
        break;
      if ( *v35 != -4096 && *v35 != -8192 )
      {
        if ( v35 != v84 )
          goto LABEL_42;
        return;
      }
    }
  }
}
