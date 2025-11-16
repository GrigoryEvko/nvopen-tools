// Function: sub_146CB30
// Address: 0x146cb30
//
__int64 __fastcall sub_146CB30(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r13
  unsigned int v6; // esi
  __int64 v7; // rdx
  __int64 v8; // rdi
  unsigned int v9; // eax
  __int64 *v10; // r12
  __int64 v11; // rcx
  __int64 *v12; // rax
  __int64 *v13; // rsi
  unsigned __int64 *v15; // rdx
  int v16; // eax
  __int64 v17; // rsi
  int v18; // eax
  unsigned int v19; // esi
  __int64 v20; // rdx
  __int64 v21; // rdi
  unsigned int v22; // eax
  __int64 *v23; // r8
  __int64 v24; // r10
  __int64 v25; // rdi
  __int64 v26; // rax
  int v27; // r10d
  __int64 *v28; // r9
  int v29; // eax
  int v30; // ecx
  int v31; // r15d
  __int64 *v32; // rcx
  int v33; // eax
  int v34; // edi
  __int64 v35; // r11
  __int64 v36[2]; // [rsp+8h] [rbp-48h] BYREF
  _QWORD v37[7]; // [rsp+18h] [rbp-38h] BYREF

  v4 = a1 + 656;
  v36[0] = a2;
  v6 = *(_DWORD *)(a1 + 680);
  if ( !v6 )
  {
    ++*(_QWORD *)(a1 + 656);
    goto LABEL_30;
  }
  v7 = v36[0];
  v8 = *(_QWORD *)(a1 + 664);
  v9 = (v6 - 1) & ((LODWORD(v36[0]) >> 9) ^ (LODWORD(v36[0]) >> 4));
  v10 = (__int64 *)(v8 + 40LL * v9);
  v11 = *v10;
  if ( v36[0] != *v10 )
  {
    v27 = 1;
    v28 = 0;
    while ( v11 != -8 )
    {
      if ( v11 == -16 && !v28 )
        v28 = v10;
      v9 = (v6 - 1) & (v27 + v9);
      v10 = (__int64 *)(v8 + 40LL * v9);
      v11 = *v10;
      if ( v36[0] == *v10 )
        goto LABEL_3;
      ++v27;
    }
    v29 = *(_DWORD *)(a1 + 672);
    if ( v28 )
      v10 = v28;
    ++*(_QWORD *)(a1 + 656);
    v30 = v29 + 1;
    if ( 4 * (v29 + 1) < 3 * v6 )
    {
      if ( v6 - *(_DWORD *)(a1 + 676) - v30 > v6 >> 3 )
      {
LABEL_26:
        *(_DWORD *)(a1 + 672) = v30;
        if ( *v10 != -8 )
          --*(_DWORD *)(a1 + 676);
        *v10 = v7;
        v10[1] = (__int64)(v10 + 3);
        v10[2] = 0x200000000LL;
        goto LABEL_11;
      }
LABEL_31:
      sub_146C840(v4, v6);
      sub_1461360(v4, v36, v37);
      v10 = (__int64 *)v37[0];
      v7 = v36[0];
      v30 = *(_DWORD *)(a1 + 672) + 1;
      goto LABEL_26;
    }
LABEL_30:
    v6 *= 2;
    goto LABEL_31;
  }
LABEL_3:
  v12 = (__int64 *)v10[1];
  v13 = &v12[*((unsigned int *)v10 + 4)];
  if ( v12 == v13 )
  {
LABEL_9:
    if ( *((_DWORD *)v10 + 4) >= *((_DWORD *)v10 + 5) )
      sub_16CD150(v10 + 1, v10 + 3, 0, 8);
LABEL_11:
    v15 = (unsigned __int64 *)(v10[1] + 8LL * *((unsigned int *)v10 + 4));
    v16 = *((_DWORD *)v10 + 4);
    if ( v15 )
    {
      *v15 = a3 & 0xFFFFFFFFFFFFFFF9LL;
      v16 = *((_DWORD *)v10 + 4);
    }
    v17 = v36[0];
    *((_DWORD *)v10 + 4) = v16 + 1;
    v18 = sub_146CF00(a1, v17, a3);
    v19 = *(_DWORD *)(a1 + 680);
    LODWORD(v10) = v18;
    if ( v19 )
    {
      v20 = v36[0];
      v21 = *(_QWORD *)(a1 + 664);
      v22 = (v19 - 1) & ((LODWORD(v36[0]) >> 9) ^ (LODWORD(v36[0]) >> 4));
      v23 = (__int64 *)(v21 + 40LL * v22);
      v24 = *v23;
      if ( v36[0] == *v23 )
      {
        v25 = v23[1];
        v26 = v25 + 8LL * *((unsigned int *)v23 + 4);
        goto LABEL_17;
      }
      v31 = 1;
      v32 = 0;
      while ( v24 != -8 )
      {
        if ( v32 || v24 != -16 )
          v23 = v32;
        v22 = (v19 - 1) & (v31 + v22);
        v35 = v21 + 40LL * v22;
        v24 = *(_QWORD *)v35;
        if ( v36[0] == *(_QWORD *)v35 )
        {
          v25 = *(_QWORD *)(v35 + 8);
          v26 = v25 + 8LL * *(unsigned int *)(v35 + 16);
LABEL_17:
          while ( v26 != v25 )
          {
            if ( a3 == (*(_QWORD *)(v26 - 8) & 0xFFFFFFFFFFFFFFF8LL) )
            {
              *(_QWORD *)(v26 - 8) = (2LL * (unsigned int)v10) | *(_QWORD *)(v26 - 8) & 0xFFFFFFFFFFFFFFF9LL;
              return (unsigned int)v10;
            }
            v26 -= 8;
          }
          return (unsigned int)v10;
        }
        ++v31;
        v32 = v23;
        v23 = (__int64 *)(v21 + 40LL * v22);
      }
      v33 = *(_DWORD *)(a1 + 672);
      if ( !v32 )
        v32 = v23;
      ++*(_QWORD *)(a1 + 656);
      v34 = v33 + 1;
      if ( 4 * (v33 + 1) >= 3 * v19 )
        goto LABEL_42;
      if ( v19 - *(_DWORD *)(a1 + 676) - v34 <= v19 >> 3 )
        goto LABEL_43;
    }
    else
    {
      ++*(_QWORD *)(a1 + 656);
LABEL_42:
      v19 *= 2;
LABEL_43:
      sub_146C840(v4, v19);
      sub_1461360(v4, v36, v37);
      v32 = (__int64 *)v37[0];
      v20 = v36[0];
      v34 = *(_DWORD *)(a1 + 672) + 1;
    }
    *(_DWORD *)(a1 + 672) = v34;
    if ( *v32 != -8 )
      --*(_DWORD *)(a1 + 676);
    *v32 = v20;
    v32[1] = (__int64)(v32 + 3);
    v32[2] = 0x200000000LL;
    return (unsigned int)v10;
  }
  while ( a3 != (*v12 & 0xFFFFFFFFFFFFFFF8LL) )
  {
    if ( v13 == ++v12 )
      goto LABEL_9;
  }
  return (*v12 >> 1) & 3;
}
