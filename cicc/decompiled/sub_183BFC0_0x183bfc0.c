// Function: sub_183BFC0
// Address: 0x183bfc0
//
int __fastcall sub_183BFC0(__int64 a1, unsigned __int64 a2, __int64 a3)
{
  __int64 v3; // r9
  unsigned int v7; // r15d
  __int64 v8; // rcx
  __int64 v9; // r8
  int v10; // r10d
  int v11; // r14d
  unsigned int v12; // esi
  unsigned int v13; // r11d
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rdi
  __int64 v17; // rdi
  __int64 v18; // rsi
  unsigned __int64 v19; // r13
  int v20; // eax
  int v21; // ecx
  unsigned int v22; // eax
  __int64 v23; // rdi
  __int64 v24; // rsi
  int v25; // edx
  int v26; // r10d
  const void *v27; // rdi
  const void *v28; // rsi
  size_t v29; // rdx
  int v30; // r11d
  int v31; // eax
  __int64 *v32; // r10
  int v33; // eax
  int v34; // eax
  __int64 v35; // rsi
  unsigned int v36; // r14d
  __int64 v37; // rcx
  __int64 *v38; // r10
  __int64 v40; // [rsp+10h] [rbp-40h]
  __int64 v41; // [rsp+18h] [rbp-38h]
  int v42; // [rsp+18h] [rbp-38h]

  v3 = a1 + 8;
  v7 = *(_DWORD *)(a1 + 32);
  v8 = *(_QWORD *)(a1 + 16);
  if ( !v7 )
  {
    ++*(_QWORD *)(a1 + 8);
    goto LABEL_15;
  }
  LODWORD(v9) = v7 - 1;
  v10 = 1;
  v11 = a2 ^ (a2 >> 9);
  v12 = (v7 - 1) & v11;
  v13 = v12;
  v14 = v8 + 40LL * v12;
  v15 = *(_QWORD *)v14;
  v16 = *(_QWORD *)v14;
  if ( *(_QWORD *)v14 == a2 )
  {
    if ( v14 == v8 + 40LL * v7 )
      goto LABEL_6;
LABEL_4:
    if ( *(_DWORD *)(v14 + 8) == *(_DWORD *)a3 )
    {
      v27 = *(const void **)(v14 + 16);
      v28 = *(const void **)(a3 + 8);
      v29 = *(_QWORD *)(v14 + 24) - (_QWORD)v27;
      v14 = *(_QWORD *)(a3 + 16) - (_QWORD)v28;
      if ( v29 == v14 )
      {
        v40 = v8;
        v41 = v3;
        if ( !v29 )
          return v14;
        LODWORD(v14) = memcmp(v27, v28, v29);
        if ( !(_DWORD)v14 )
          return v14;
        v3 = v41;
        v8 = v40;
        LODWORD(v9) = v7 - 1;
      }
    }
LABEL_5:
    v11 = a2 ^ (a2 >> 9);
    v12 = v11 & v9;
    v14 = v8 + 40LL * (v11 & (unsigned int)v9);
    v15 = *(_QWORD *)v14;
    if ( *(_QWORD *)v14 == a2 )
    {
LABEL_6:
      v17 = *(_QWORD *)(v14 + 16);
      v18 = *(_QWORD *)(v14 + 32) - v17;
      goto LABEL_7;
    }
    goto LABEL_26;
  }
  while ( 1 )
  {
    if ( v16 == -2 )
      goto LABEL_5;
    v13 = v9 & (v10 + v13);
    v42 = v10 + 1;
    v32 = (__int64 *)(v8 + 40LL * v13);
    v16 = *v32;
    if ( *v32 == a2 )
      break;
    v10 = v42;
  }
  if ( v32 != (__int64 *)(v8 + 40LL * v7) )
  {
    v14 = v8 + 40LL * v13;
    goto LABEL_4;
  }
LABEL_26:
  v30 = 1;
  v23 = 0;
  while ( 1 )
  {
    if ( v15 == -2 )
    {
      if ( !v23 )
        v23 = v14;
      v31 = *(_DWORD *)(a1 + 24);
      ++*(_QWORD *)(a1 + 8);
      v25 = v31 + 1;
      if ( 4 * (v31 + 1) < 3 * v7 )
      {
        if ( v7 - *(_DWORD *)(a1 + 28) - v25 > v7 >> 3 )
          goto LABEL_32;
        sub_183B620(v3, v7);
        v33 = *(_DWORD *)(a1 + 32);
        if ( v33 )
        {
          v34 = v33 - 1;
          v35 = *(_QWORD *)(a1 + 16);
          LODWORD(v3) = 1;
          v9 = 0;
          v36 = v34 & v11;
          v23 = v35 + 40LL * v36;
          v37 = *(_QWORD *)v23;
          v25 = *(_DWORD *)(a1 + 24) + 1;
          if ( *(_QWORD *)v23 != a2 )
          {
            while ( v37 != -2 )
            {
              if ( !v9 && v37 == -16 )
                v9 = v23;
              v36 = v34 & (v3 + v36);
              v23 = v35 + 40LL * v36;
              v37 = *(_QWORD *)v23;
              if ( *(_QWORD *)v23 == a2 )
                goto LABEL_32;
              LODWORD(v3) = v3 + 1;
            }
            if ( v9 )
              v23 = v9;
          }
          goto LABEL_32;
        }
        goto LABEL_63;
      }
LABEL_15:
      sub_183B620(v3, 2 * v7);
      v20 = *(_DWORD *)(a1 + 32);
      if ( v20 )
      {
        v21 = v20 - 1;
        v9 = *(_QWORD *)(a1 + 16);
        v22 = (v20 - 1) & (a2 ^ (a2 >> 9));
        v23 = v9 + 40LL * v22;
        v24 = *(_QWORD *)v23;
        v25 = *(_DWORD *)(a1 + 24) + 1;
        if ( *(_QWORD *)v23 != a2 )
        {
          v26 = 1;
          v3 = 0;
          while ( v24 != -2 )
          {
            if ( v24 == -16 && !v3 )
              v3 = v23;
            v22 = v21 & (v26 + v22);
            v23 = v9 + 40LL * v22;
            v24 = *(_QWORD *)v23;
            if ( *(_QWORD *)v23 == a2 )
              goto LABEL_32;
            ++v26;
          }
          if ( v3 )
            v23 = v3;
        }
LABEL_32:
        *(_DWORD *)(a1 + 24) = v25;
        if ( *(_QWORD *)v23 != -2 )
          --*(_DWORD *)(a1 + 28);
        *(_DWORD *)(v23 + 8) = 0;
        *(_QWORD *)(v23 + 16) = 0;
        *(_QWORD *)(v23 + 24) = 0;
        *(_QWORD *)(v23 + 32) = 0;
        *(_QWORD *)v23 = a2;
        *(_DWORD *)(v23 + 8) = *(_DWORD *)a3;
        *(_QWORD *)(v23 + 16) = *(_QWORD *)(a3 + 8);
        *(_QWORD *)(v23 + 24) = *(_QWORD *)(a3 + 16);
        v14 = *(_QWORD *)(a3 + 24);
        *(_QWORD *)(v23 + 32) = v14;
        *(_QWORD *)(a3 + 8) = 0;
        *(_QWORD *)(a3 + 16) = 0;
        *(_QWORD *)(a3 + 24) = 0;
        v19 = a2 & 0xFFFFFFFFFFFFFFF8LL;
        if ( v19 )
          goto LABEL_10;
        return v14;
      }
LABEL_63:
      ++*(_DWORD *)(a1 + 24);
      BUG();
    }
    if ( v23 || v15 != -16 )
      v14 = v23;
    v12 = v9 & (v30 + v12);
    v38 = (__int64 *)(v8 + 40LL * v12);
    v15 = *v38;
    if ( *v38 == a2 )
      break;
    ++v30;
    v23 = v14;
    v14 = v8 + 40LL * v12;
  }
  v17 = v38[2];
  v14 = v8 + 40LL * v12;
  v18 = v38[4] - v17;
LABEL_7:
  *(_DWORD *)(v14 + 8) = *(_DWORD *)a3;
  *(_QWORD *)(v14 + 16) = *(_QWORD *)(a3 + 8);
  *(_QWORD *)(v14 + 24) = *(_QWORD *)(a3 + 16);
  *(_QWORD *)(v14 + 32) = *(_QWORD *)(a3 + 24);
  *(_QWORD *)(a3 + 8) = 0;
  *(_QWORD *)(a3 + 16) = 0;
  *(_QWORD *)(a3 + 24) = 0;
  if ( v17 )
    LODWORD(v14) = j_j___libc_free_0(v17, v18);
  v19 = a2 & 0xFFFFFFFFFFFFFFF8LL;
  if ( v19 )
  {
LABEL_10:
    v14 = *(unsigned int *)(a1 + 216);
    if ( (unsigned int)v14 >= *(_DWORD *)(a1 + 220) )
    {
      sub_16CD150(a1 + 208, (const void *)(a1 + 224), 0, 8, v9, v3);
      v14 = *(unsigned int *)(a1 + 216);
    }
    *(_QWORD *)(*(_QWORD *)(a1 + 208) + 8 * v14) = v19;
    ++*(_DWORD *)(a1 + 216);
  }
  return v14;
}
