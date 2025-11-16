// Function: sub_25A6B00
// Address: 0x25a6b00
//
void __fastcall sub_25A6B00(__int64 a1, unsigned __int64 a2, __int64 a3)
{
  __int64 v3; // r9
  unsigned int v7; // esi
  __int64 v8; // r8
  __int64 v9; // rdi
  int v10; // r10d
  unsigned int v11; // ecx
  __int64 v12; // r12
  __int64 v13; // rax
  __int64 v14; // rdx
  const void *v15; // rdi
  unsigned __int64 v16; // r15
  __int64 v17; // r13
  size_t v18; // rdx
  __int64 v19; // rax
  int v20; // ecx
  int v21; // ecx
  int v22; // eax
  int v23; // esi
  unsigned int v24; // edx
  __int64 v25; // rdi
  int v26; // r10d
  int v27; // eax
  int v28; // edx
  __int64 v29; // rdi
  unsigned int v30; // r13d
  __int64 v31; // rsi
  const void *v32; // [rsp+8h] [rbp-38h]

  v3 = a1 + 8;
  v7 = *(_DWORD *)(a1 + 32);
  if ( !v7 )
  {
    ++*(_QWORD *)(a1 + 8);
    goto LABEL_29;
  }
  v8 = v7 - 1;
  v9 = *(_QWORD *)(a1 + 16);
  v10 = 1;
  v11 = v8 & (a2 ^ (a2 >> 9));
  v12 = v9 + 40LL * v11;
  v13 = 0;
  v14 = *(_QWORD *)v12;
  if ( a2 != *(_QWORD *)v12 )
  {
    while ( v14 != -2 )
    {
      if ( !v13 && v14 == -16 )
        v13 = v12;
      v11 = v8 & (v10 + v11);
      v12 = v9 + 40LL * v11;
      v14 = *(_QWORD *)v12;
      if ( *(_QWORD *)v12 == a2 )
        goto LABEL_3;
      ++v10;
    }
    v20 = *(_DWORD *)(a1 + 24);
    if ( !v13 )
      v13 = v12;
    ++*(_QWORD *)(a1 + 8);
    v21 = v20 + 1;
    if ( 4 * v21 < 3 * v7 )
    {
      if ( v7 - *(_DWORD *)(a1 + 28) - v21 > v7 >> 3 )
      {
LABEL_25:
        *(_DWORD *)(a1 + 24) = v21;
        if ( *(_QWORD *)v13 != -2 )
          --*(_DWORD *)(a1 + 28);
        *(_QWORD *)v13 = a2;
        *(_OWORD *)(v13 + 8) = 0;
        *(_OWORD *)(v13 + 24) = 0;
        *(_DWORD *)(v13 + 8) = *(_DWORD *)a3;
        *(_QWORD *)(v13 + 16) = *(_QWORD *)(a3 + 8);
        *(_QWORD *)(v13 + 24) = *(_QWORD *)(a3 + 16);
        *(_QWORD *)(v13 + 32) = *(_QWORD *)(a3 + 24);
        *(_QWORD *)(a3 + 8) = 0;
        *(_QWORD *)(a3 + 16) = 0;
        *(_QWORD *)(a3 + 24) = 0;
        goto LABEL_6;
      }
      sub_25A5BF0(v3, v7);
      v27 = *(_DWORD *)(a1 + 32);
      if ( v27 )
      {
        v28 = v27 - 1;
        v29 = *(_QWORD *)(a1 + 16);
        v8 = 0;
        v30 = (v27 - 1) & (a2 ^ (a2 >> 9));
        v3 = 1;
        v21 = *(_DWORD *)(a1 + 24) + 1;
        v13 = v29 + 40LL * v30;
        v31 = *(_QWORD *)v13;
        if ( *(_QWORD *)v13 != a2 )
        {
          while ( v31 != -2 )
          {
            if ( !v8 && v31 == -16 )
              v8 = v13;
            v30 = v28 & (v3 + v30);
            v13 = v29 + 40LL * v30;
            v31 = *(_QWORD *)v13;
            if ( *(_QWORD *)v13 == a2 )
              goto LABEL_25;
            v3 = (unsigned int)(v3 + 1);
          }
          if ( v8 )
            v13 = v8;
        }
        goto LABEL_25;
      }
LABEL_52:
      ++*(_DWORD *)(a1 + 24);
      BUG();
    }
LABEL_29:
    sub_25A5BF0(v3, 2 * v7);
    v22 = *(_DWORD *)(a1 + 32);
    if ( v22 )
    {
      v23 = v22 - 1;
      v8 = *(_QWORD *)(a1 + 16);
      v21 = *(_DWORD *)(a1 + 24) + 1;
      v24 = (v22 - 1) & (a2 ^ (a2 >> 9));
      v13 = v8 + 40LL * v24;
      v25 = *(_QWORD *)v13;
      if ( a2 != *(_QWORD *)v13 )
      {
        v26 = 1;
        v3 = 0;
        while ( v25 != -2 )
        {
          if ( !v3 && v25 == -16 )
            v3 = v13;
          v24 = v23 & (v26 + v24);
          v13 = v8 + 40LL * v24;
          v25 = *(_QWORD *)v13;
          if ( *(_QWORD *)v13 == a2 )
            goto LABEL_25;
          ++v26;
        }
        if ( v3 )
          v13 = v3;
      }
      goto LABEL_25;
    }
    goto LABEL_52;
  }
LABEL_3:
  v15 = *(const void **)(v12 + 16);
  if ( *(_DWORD *)a3 == *(_DWORD *)(v12 + 8)
    && (v17 = *(_QWORD *)(a3 + 8), v18 = *(_QWORD *)(v12 + 24) - (_QWORD)v15, v18 == *(_QWORD *)(a3 + 16) - v17) )
  {
    if ( !v18 )
      return;
    v32 = *(const void **)(v12 + 16);
    if ( !memcmp(v15, *(const void **)(a3 + 8), v18) )
      return;
    v15 = v32;
    *(_QWORD *)(v12 + 16) = v17;
    *(_QWORD *)(v12 + 24) = *(_QWORD *)(a3 + 16);
    *(_QWORD *)(v12 + 32) = *(_QWORD *)(a3 + 24);
    *(_QWORD *)(a3 + 8) = 0;
    *(_QWORD *)(a3 + 16) = 0;
    *(_QWORD *)(a3 + 24) = 0;
  }
  else
  {
    *(_DWORD *)(v12 + 8) = *(_DWORD *)a3;
    *(_QWORD *)(v12 + 16) = *(_QWORD *)(a3 + 8);
    *(_QWORD *)(v12 + 24) = *(_QWORD *)(a3 + 16);
    *(_QWORD *)(v12 + 32) = *(_QWORD *)(a3 + 24);
    *(_QWORD *)(a3 + 8) = 0;
    *(_QWORD *)(a3 + 16) = 0;
    *(_QWORD *)(a3 + 24) = 0;
    if ( !v15 )
      goto LABEL_6;
  }
  j_j___libc_free_0((unsigned __int64)v15);
LABEL_6:
  v16 = a2 & 0xFFFFFFFFFFFFFFF8LL;
  if ( v16 )
  {
    v19 = *(unsigned int *)(a1 + 208);
    if ( v19 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 212) )
    {
      sub_C8D5F0(a1 + 200, (const void *)(a1 + 216), v19 + 1, 8u, v8, v3);
      v19 = *(unsigned int *)(a1 + 208);
    }
    *(_QWORD *)(*(_QWORD *)(a1 + 200) + 8 * v19) = v16;
    ++*(_DWORD *)(a1 + 208);
  }
}
