// Function: sub_2796930
// Address: 0x2796930
//
__int64 __fastcall sub_2796930(__int64 a1, __int64 a2)
{
  unsigned int v4; // esi
  __int64 v5; // r13
  int v6; // eax
  int v7; // edx
  __int64 v8; // rax
  __int64 v9; // rbx
  __int64 v11; // r8
  __int64 v12; // rdi
  unsigned int v13; // ecx
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rsi
  int v17; // ecx
  __int64 v18; // rdi
  __int64 v19; // r8
  int v20; // r9d
  unsigned int v21; // edx
  __int64 v22; // rax
  int v23; // r10d
  int v24; // eax
  int v25; // eax
  __int64 v26; // rsi
  int v27; // edx
  __int64 v28; // rdi
  int v29; // r9d
  unsigned int v30; // ecx
  __int64 v31; // rax

  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_3;
  }
  v11 = *(_QWORD *)(a1 + 8);
  v12 = *(_QWORD *)(a2 + 16);
  v13 = (v4 - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
  v14 = v11 + 32LL * v13;
  v15 = *(_QWORD *)(v14 + 16);
  if ( v12 != v15 )
  {
    v23 = 1;
    v5 = 0;
    while ( v15 != -4096 )
    {
      if ( !v5 && v15 == -8192 )
        v5 = v14;
      v13 = (v4 - 1) & (v23 + v13);
      v14 = v11 + 32LL * v13;
      v15 = *(_QWORD *)(v14 + 16);
      if ( v12 == v15 )
        return v14 + 24;
      ++v23;
    }
    if ( !v5 )
      v5 = v14;
    v24 = *(_DWORD *)(a1 + 16);
    ++*(_QWORD *)a1;
    v7 = v24 + 1;
    if ( 4 * (v24 + 1) < 3 * v4 )
    {
      if ( v4 - *(_DWORD *)(a1 + 20) - v7 > v4 >> 3 )
      {
LABEL_5:
        *(_DWORD *)(a1 + 16) = v7;
        if ( *(_QWORD *)(v5 + 16) == -4096 )
        {
          v9 = *(_QWORD *)(a2 + 16);
          if ( v9 != -4096 )
          {
LABEL_10:
            *(_QWORD *)(v5 + 16) = v9;
            if ( v9 != 0 && v9 != -4096 && v9 != -8192 )
              sub_BD73F0(v5);
          }
        }
        else
        {
          --*(_DWORD *)(a1 + 20);
          v8 = *(_QWORD *)(v5 + 16);
          v9 = *(_QWORD *)(a2 + 16);
          if ( v9 != v8 )
          {
            if ( v8 != 0 && v8 != -4096 && v8 != -8192 )
              sub_BD60C0((_QWORD *)v5);
            goto LABEL_10;
          }
        }
        *(_DWORD *)(v5 + 24) = 0;
        return v5 + 24;
      }
      v5 = 0;
      sub_27965B0(a1, v4);
      v25 = *(_DWORD *)(a1 + 24);
      if ( v25 )
      {
        v26 = *(_QWORD *)(a2 + 16);
        v27 = v25 - 1;
        v28 = *(_QWORD *)(a1 + 8);
        v19 = 0;
        v29 = 1;
        v30 = (v25 - 1) & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
        v5 = v28 + 32LL * v30;
        v31 = *(_QWORD *)(v5 + 16);
        if ( v26 != v31 )
        {
          while ( v31 != -4096 )
          {
            if ( v31 == -8192 && !v19 )
              v19 = v5;
            v30 = v27 & (v29 + v30);
            v5 = v28 + 32LL * v30;
            v31 = *(_QWORD *)(v5 + 16);
            if ( v26 == v31 )
              goto LABEL_4;
            ++v29;
          }
          goto LABEL_20;
        }
      }
LABEL_4:
      v7 = *(_DWORD *)(a1 + 16) + 1;
      goto LABEL_5;
    }
LABEL_3:
    v5 = 0;
    sub_27965B0(a1, 2 * v4);
    v6 = *(_DWORD *)(a1 + 24);
    if ( v6 )
    {
      v16 = *(_QWORD *)(a2 + 16);
      v17 = v6 - 1;
      v18 = *(_QWORD *)(a1 + 8);
      v19 = 0;
      v20 = 1;
      v21 = (v6 - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
      v5 = v18 + 32LL * v21;
      v22 = *(_QWORD *)(v5 + 16);
      if ( v16 != v22 )
      {
        while ( v22 != -4096 )
        {
          if ( !v19 && v22 == -8192 )
            v19 = v5;
          v21 = v17 & (v20 + v21);
          v5 = v18 + 32LL * v21;
          v22 = *(_QWORD *)(v5 + 16);
          if ( v16 == v22 )
            goto LABEL_4;
          ++v20;
        }
LABEL_20:
        if ( v19 )
          v5 = v19;
        goto LABEL_4;
      }
    }
    goto LABEL_4;
  }
  return v14 + 24;
}
