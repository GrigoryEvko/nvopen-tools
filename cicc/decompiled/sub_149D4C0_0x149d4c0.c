// Function: sub_149D4C0
// Address: 0x149d4c0
//
int __fastcall sub_149D4C0(__int64 a1, int a2, _BYTE *a3, size_t a4)
{
  size_t v4; // r10
  _BYTE *v5; // r8
  int v8; // r13d
  _QWORD *v10; // rax
  char v11; // cl
  int v12; // r12d
  __int64 v13; // r13
  char v14; // r9
  unsigned int v15; // esi
  __int64 v16; // rdi
  __int64 v17; // rcx
  unsigned int v18; // edx
  int *v19; // r12
  int v20; // eax
  int *v21; // rdi
  __int64 v22; // rdx
  _QWORD *v23; // rax
  int v24; // eax
  int v25; // ecx
  __int64 v26; // rdi
  unsigned int v27; // eax
  int v28; // esi
  int v29; // edx
  int v30; // r9d
  int *v31; // r8
  int v32; // r10d
  int *v33; // r9
  int v34; // eax
  size_t v35; // rdx
  __int64 v36; // rax
  _QWORD *v37; // rdi
  int v38; // eax
  int v39; // eax
  __int64 v40; // rsi
  int v41; // r8d
  int *v42; // rdi
  unsigned int v43; // r15d
  int v44; // ecx
  char v47; // [rsp+13h] [rbp-6Dh]
  _BYTE *v48; // [rsp+18h] [rbp-68h]
  size_t v49; // [rsp+28h] [rbp-58h] BYREF
  _QWORD *v50; // [rsp+30h] [rbp-50h] BYREF
  size_t n; // [rsp+38h] [rbp-48h]
  _QWORD src[8]; // [rsp+40h] [rbp-40h] BYREF

  v4 = a4;
  v5 = a3;
  v8 = a2 + 3;
  if ( a2 >= 0 )
    v8 = a2;
  LODWORD(v10) = 16 * a2 + 83474176;
  v11 = 2 * (a2 & 3);
  v12 = 3 << v11;
  v13 = v8 >> 2;
  v14 = *(_BYTE *)(a1 + v13);
  if ( a4 == qword_4F9B700[2 * a2 + 1] )
  {
    if ( !a4
      || (v47 = *(_BYTE *)(a1 + v13),
          LODWORD(v10) = memcmp((const void *)qword_4F9B700[2 * a2], a3, a4),
          v14 = v47,
          !(_DWORD)v10) )
    {
      *(_BYTE *)(a1 + v13) = v12 | v14;
      return (int)v10;
    }
    v5 = a3;
    v4 = a4;
    *(_BYTE *)(a1 + v13) = (1 << (2 * (a2 & 3))) | ~(_BYTE)v12 & v47;
  }
  else
  {
    *(_BYTE *)(a1 + v13) = (1 << v11) | ~(_BYTE)v12 & v14;
    if ( !a3 )
    {
      v15 = *(_DWORD *)(a1 + 136);
      LOBYTE(src[0]) = 0;
      v16 = a1 + 112;
      v50 = src;
      n = 0;
      if ( v15 )
        goto LABEL_6;
LABEL_21:
      ++*(_QWORD *)(a1 + 112);
      goto LABEL_22;
    }
  }
  v49 = a4;
  v50 = src;
  if ( a4 > 0xF )
  {
    v48 = v5;
    v36 = sub_22409D0(&v50, &v49, 0);
    v5 = v48;
    v50 = (_QWORD *)v36;
    v37 = (_QWORD *)v36;
    src[0] = v49;
  }
  else
  {
    if ( a4 == 1 )
    {
      LOBYTE(src[0]) = *v5;
      v23 = src;
      goto LABEL_20;
    }
    if ( !a4 )
    {
      v23 = src;
      goto LABEL_20;
    }
    v37 = src;
  }
  memcpy(v37, v5, a4);
  v4 = v49;
  v23 = v50;
LABEL_20:
  n = v4;
  v16 = a1 + 112;
  *((_BYTE *)v23 + v4) = 0;
  v15 = *(_DWORD *)(a1 + 136);
  if ( !v15 )
    goto LABEL_21;
LABEL_6:
  v17 = *(_QWORD *)(a1 + 120);
  v18 = (v15 - 1) & (37 * a2);
  v19 = (int *)(v17 + 40LL * v18);
  v20 = *v19;
  if ( a2 != *v19 )
  {
    v32 = 1;
    v33 = 0;
    while ( v20 != -1 )
    {
      if ( v20 == -2 && !v33 )
        v33 = v19;
      v18 = (v15 - 1) & (v32 + v18);
      v19 = (int *)(v17 + 40LL * v18);
      v20 = *v19;
      if ( a2 == *v19 )
        goto LABEL_7;
      ++v32;
    }
    v34 = *(_DWORD *)(a1 + 128);
    if ( v33 )
      v19 = v33;
    ++*(_QWORD *)(a1 + 112);
    v29 = v34 + 1;
    if ( 4 * (v34 + 1) < 3 * v15 )
    {
      if ( v15 - *(_DWORD *)(a1 + 132) - v29 > v15 >> 3 )
        goto LABEL_36;
      sub_149D280(v16, v15);
      v38 = *(_DWORD *)(a1 + 136);
      if ( v38 )
      {
        v39 = v38 - 1;
        v40 = *(_QWORD *)(a1 + 120);
        v41 = 1;
        v42 = 0;
        v43 = v39 & (37 * a2);
        v19 = (int *)(v40 + 40LL * v43);
        v44 = *v19;
        v29 = *(_DWORD *)(a1 + 128) + 1;
        if ( a2 != *v19 )
        {
          while ( v44 != -1 )
          {
            if ( !v42 && v44 == -2 )
              v42 = v19;
            v43 = v39 & (v41 + v43);
            v19 = (int *)(v40 + 40LL * v43);
            v44 = *v19;
            if ( a2 == *v19 )
              goto LABEL_36;
            ++v41;
          }
          if ( v42 )
            v19 = v42;
        }
        goto LABEL_36;
      }
      goto LABEL_73;
    }
LABEL_22:
    sub_149D280(v16, 2 * v15);
    v24 = *(_DWORD *)(a1 + 136);
    if ( v24 )
    {
      v25 = v24 - 1;
      v26 = *(_QWORD *)(a1 + 120);
      v27 = (v24 - 1) & (37 * a2);
      v19 = (int *)(v26 + 40LL * v27);
      v28 = *v19;
      v29 = *(_DWORD *)(a1 + 128) + 1;
      if ( a2 != *v19 )
      {
        v30 = 1;
        v31 = 0;
        while ( v28 != -1 )
        {
          if ( !v31 && v28 == -2 )
            v31 = v19;
          v27 = v25 & (v30 + v27);
          v19 = (int *)(v26 + 40LL * v27);
          v28 = *v19;
          if ( a2 == *v19 )
            goto LABEL_36;
          ++v30;
        }
        if ( v31 )
          v19 = v31;
      }
LABEL_36:
      *(_DWORD *)(a1 + 128) = v29;
      if ( *v19 != -1 )
        --*(_DWORD *)(a1 + 132);
      *v19 = a2;
      v10 = v50;
      v21 = v19 + 6;
      *((_QWORD *)v19 + 1) = v19 + 6;
      *((_QWORD *)v19 + 2) = 0;
      *((_BYTE *)v19 + 24) = 0;
      if ( v10 != src )
        goto LABEL_46;
LABEL_39:
      v35 = n;
      if ( n )
      {
        if ( n == 1 )
          *(_BYTE *)v21 = src[0];
        else
          memcpy(v21, src, n);
        v35 = n;
      }
      v10 = (_QWORD *)*((_QWORD *)v19 + 1);
      *((_QWORD *)v19 + 2) = v35;
      *((_BYTE *)v10 + v35) = 0;
      v21 = (int *)v50;
      goto LABEL_11;
    }
LABEL_73:
    ++*(_DWORD *)(a1 + 128);
    BUG();
  }
LABEL_7:
  v10 = v50;
  v21 = (int *)*((_QWORD *)v19 + 1);
  if ( v50 == src )
    goto LABEL_39;
  if ( v21 == v19 + 6 )
  {
LABEL_46:
    *((_QWORD *)v19 + 1) = v10;
    *((_QWORD *)v19 + 2) = n;
    LODWORD(v10) = src[0];
    *((_QWORD *)v19 + 3) = src[0];
    goto LABEL_47;
  }
  *((_QWORD *)v19 + 1) = v50;
  v22 = *((_QWORD *)v19 + 3);
  *((_QWORD *)v19 + 2) = n;
  LODWORD(v10) = src[0];
  *((_QWORD *)v19 + 3) = src[0];
  if ( v21 )
  {
    v50 = v21;
    src[0] = v22;
    goto LABEL_11;
  }
LABEL_47:
  v50 = src;
  v21 = (int *)src;
LABEL_11:
  n = 0;
  *(_BYTE *)v21 = 0;
  if ( v50 != src )
    LODWORD(v10) = j_j___libc_free_0(v50, src[0] + 1LL);
  return (int)v10;
}
