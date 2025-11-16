// Function: sub_2B86B60
// Address: 0x2b86b60
//
__int64 __fastcall sub_2B86B60(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // r13
  char v9; // cl
  __int64 v10; // rdi
  int v11; // esi
  unsigned int v12; // edx
  __int64 v13; // rax
  unsigned int v15; // esi
  unsigned int v16; // eax
  __int64 v17; // r8
  int v18; // edx
  unsigned int v19; // edi
  __int64 v20; // rdx
  int v21; // eax
  __int64 v22; // r14
  __int64 *v23; // rdx
  __int64 v24; // rax
  int v25; // r10d
  __int64 v26; // rdx
  __int64 v27; // rcx
  __int64 v28; // r9
  __int64 v29; // r8
  _QWORD *v30; // rax
  __int64 v31; // rdx
  char **v32; // r15
  unsigned __int64 v33; // r13
  __int64 v34; // r12
  char *v35; // rax
  unsigned __int64 v36; // r12
  unsigned __int64 v37; // rdi
  int v38; // r12d
  int v39; // eax
  __int64 v40; // rcx
  int v41; // edx
  unsigned int v42; // eax
  __int64 v43; // rsi
  __int64 v44; // rcx
  int v45; // edx
  unsigned int v46; // eax
  __int64 v47; // rsi
  __int64 v48; // rdi
  int v49; // edx
  int v50; // edx
  unsigned __int64 v51[7]; // [rsp+18h] [rbp-38h] BYREF

  v8 = *a2;
  v9 = *(_BYTE *)(a1 + 8) & 1;
  if ( v9 )
  {
    v10 = a1 + 16;
    v11 = 3;
  }
  else
  {
    v15 = *(_DWORD *)(a1 + 24);
    v10 = *(_QWORD *)(a1 + 16);
    if ( !v15 )
    {
      v16 = *(_DWORD *)(a1 + 8);
      ++*(_QWORD *)a1;
      v17 = 0;
      v18 = (v16 >> 1) + 1;
LABEL_8:
      v19 = 3 * v15;
      goto LABEL_9;
    }
    v11 = v15 - 1;
  }
  v12 = v11 & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
  v13 = v10 + 16LL * v12;
  a6 = *(_QWORD *)v13;
  if ( v8 == *(_QWORD *)v13 )
    return *(_QWORD *)(a1 + 80) + 72LL * *(unsigned int *)(v13 + 8);
  v25 = 1;
  v17 = 0;
  while ( a6 != -4096 )
  {
    if ( !v17 && a6 == -8192 )
      v17 = v13;
    v12 = v11 & (v25 + v12);
    v13 = v10 + 16LL * v12;
    a6 = *(_QWORD *)v13;
    if ( v8 == *(_QWORD *)v13 )
      return *(_QWORD *)(a1 + 80) + 72LL * *(unsigned int *)(v13 + 8);
    ++v25;
  }
  if ( !v17 )
    v17 = v13;
  v16 = *(_DWORD *)(a1 + 8);
  ++*(_QWORD *)a1;
  v18 = (v16 >> 1) + 1;
  if ( !v9 )
  {
    v15 = *(_DWORD *)(a1 + 24);
    goto LABEL_8;
  }
  v19 = 12;
  v15 = 4;
LABEL_9:
  if ( 4 * v18 >= v19 )
  {
    sub_2B86740(a1, 2 * v15);
    if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
    {
      v40 = a1 + 16;
      v41 = 3;
    }
    else
    {
      v49 = *(_DWORD *)(a1 + 24);
      v40 = *(_QWORD *)(a1 + 16);
      if ( !v49 )
        goto LABEL_73;
      v41 = v49 - 1;
    }
    v42 = v41 & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
    v17 = v40 + 16LL * v42;
    v43 = *(_QWORD *)v17;
    if ( v8 != *(_QWORD *)v17 )
    {
      a6 = 1;
      v48 = 0;
      while ( v43 != -4096 )
      {
        if ( !v48 && v43 == -8192 )
          v48 = v17;
        v42 = v41 & (a6 + v42);
        v17 = v40 + 16LL * v42;
        v43 = *(_QWORD *)v17;
        if ( v8 == *(_QWORD *)v17 )
          goto LABEL_43;
        a6 = (unsigned int)(a6 + 1);
      }
      goto LABEL_49;
    }
LABEL_43:
    v16 = *(_DWORD *)(a1 + 8);
    goto LABEL_11;
  }
  if ( v15 - *(_DWORD *)(a1 + 12) - v18 <= v15 >> 3 )
  {
    sub_2B86740(a1, v15);
    if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
    {
      v44 = a1 + 16;
      v45 = 3;
      goto LABEL_46;
    }
    v50 = *(_DWORD *)(a1 + 24);
    v44 = *(_QWORD *)(a1 + 16);
    if ( v50 )
    {
      v45 = v50 - 1;
LABEL_46:
      v46 = v45 & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
      v17 = v44 + 16LL * v46;
      v47 = *(_QWORD *)v17;
      if ( v8 != *(_QWORD *)v17 )
      {
        a6 = 1;
        v48 = 0;
        while ( v47 != -4096 )
        {
          if ( !v48 && v47 == -8192 )
            v48 = v17;
          v46 = v45 & (a6 + v46);
          v17 = v44 + 16LL * v46;
          v47 = *(_QWORD *)v17;
          if ( v8 == *(_QWORD *)v17 )
            goto LABEL_43;
          a6 = (unsigned int)(a6 + 1);
        }
LABEL_49:
        if ( v48 )
          v17 = v48;
        goto LABEL_43;
      }
      goto LABEL_43;
    }
LABEL_73:
    *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
    BUG();
  }
LABEL_11:
  *(_DWORD *)(a1 + 8) = (2 * (v16 >> 1) + 2) | v16 & 1;
  if ( *(_QWORD *)v17 != -4096 )
    --*(_DWORD *)(a1 + 12);
  *(_DWORD *)(v17 + 8) = 0;
  *(_QWORD *)v17 = v8;
  *(_DWORD *)(v17 + 8) = *(_DWORD *)(a1 + 88);
  v20 = *(unsigned int *)(a1 + 88);
  v21 = v20;
  if ( *(_DWORD *)(a1 + 92) <= (unsigned int)v20 )
  {
    v22 = sub_C8D7D0(a1 + 80, a1 + 96, 0, 0x48u, v51, a6);
    v29 = 9LL * *(unsigned int *)(a1 + 88);
    v30 = (_QWORD *)(v29 * 8 + v22);
    if ( v29 * 8 + v22 )
    {
      v31 = *a2;
      v30[2] = 0xC00000000LL;
      *v30 = v31;
      v26 = (__int64)(v30 + 3);
      v30[1] = v30 + 3;
      v29 = 9LL * *(unsigned int *)(a1 + 88);
    }
    v32 = *(char ***)(a1 + 80);
    v33 = (unsigned __int64)&v32[v29];
    if ( v32 != &v32[v29] )
    {
      v34 = v22;
      do
      {
        if ( v34 )
        {
          v35 = *v32;
          *(_DWORD *)(v34 + 16) = 0;
          *(_DWORD *)(v34 + 20) = 12;
          *(_QWORD *)v34 = v35;
          *(_QWORD *)(v34 + 8) = v34 + 24;
          if ( *((_DWORD *)v32 + 4) )
            sub_2B0D510(v34 + 8, v32 + 1, v26, v27, v29 * 8, v28);
        }
        v32 += 9;
        v34 += 72;
      }
      while ( (char **)v33 != v32 );
      v33 = *(_QWORD *)(a1 + 80);
      v36 = v33 + 72LL * *(unsigned int *)(a1 + 88);
      if ( v33 != v36 )
      {
        do
        {
          v36 -= 72LL;
          v37 = *(_QWORD *)(v36 + 8);
          if ( v37 != v36 + 24 )
            _libc_free(v37);
        }
        while ( v33 != v36 );
        v33 = *(_QWORD *)(a1 + 80);
      }
    }
    v38 = v51[0];
    if ( a1 + 96 != v33 )
      _libc_free(v33);
    v39 = *(_DWORD *)(a1 + 88);
    *(_QWORD *)(a1 + 80) = v22;
    *(_DWORD *)(a1 + 92) = v38;
    v24 = (unsigned int)(v39 + 1);
    *(_DWORD *)(a1 + 88) = v24;
  }
  else
  {
    v22 = *(_QWORD *)(a1 + 80);
    v23 = (__int64 *)(v22 + 72 * v20);
    if ( v23 )
    {
      *v23 = *a2;
      v23[1] = (__int64)(v23 + 3);
      v23[2] = 0xC00000000LL;
      v21 = *(_DWORD *)(a1 + 88);
      v22 = *(_QWORD *)(a1 + 80);
    }
    v24 = (unsigned int)(v21 + 1);
    *(_DWORD *)(a1 + 88) = v24;
  }
  return v22 + 72 * v24 - 72;
}
