// Function: sub_30ECE60
// Address: 0x30ece60
//
__int64 __fastcall sub_30ECE60(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  unsigned int v5; // r8d
  __int64 v6; // rsi
  unsigned int v7; // r10d
  unsigned int v8; // edx
  unsigned int v9; // r9d
  __int64 *v10; // rax
  __int64 v11; // rcx
  __int64 v13; // rdi
  __int64 v14; // r15
  int v15; // r11d
  int v16; // eax
  int v17; // edx
  __int64 v18; // rsi
  unsigned int v19; // eax
  int v20; // r9d
  __int64 *v21; // rdi
  __int64 v22; // rcx
  int v23; // r10d
  __int64 *v24; // r8
  int v25; // r11d
  int v26; // eax
  int v27; // eax
  int v28; // ecx
  __int64 v29; // r8
  int v30; // r10d
  unsigned int v31; // edx
  __int64 *v32; // rax
  __int64 v33; // rsi
  unsigned int v34; // [rsp+Ch] [rbp-34h]

  v2 = a1 + 8;
  v5 = *(_DWORD *)(a1 + 32);
  v6 = *(_QWORD *)(a1 + 16);
  if ( v5 )
  {
    v7 = v5 - 1;
    v8 = ((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4);
    v9 = (v5 - 1) & v8;
    v10 = (__int64 *)(v6 + 16LL * v9);
    v11 = *v10;
    if ( *v10 == a2 )
      return v10[1];
    v13 = *v10;
    LODWORD(v14) = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v15 = 1;
    while ( v13 != -4096 )
    {
      v14 = v7 & ((_DWORD)v14 + v15);
      v13 = *(_QWORD *)(v6 + 16 * v14);
      if ( v13 == a2 )
        goto LABEL_16;
      ++v15;
    }
  }
  sub_30EC8A0(a1, a2);
  v5 = *(_DWORD *)(a1 + 32);
  if ( !v5 )
  {
    ++*(_QWORD *)(a1 + 8);
    goto LABEL_8;
  }
  v7 = v5 - 1;
  v6 = *(_QWORD *)(a1 + 16);
  v8 = ((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4);
  v9 = v8 & (v5 - 1);
  v10 = (__int64 *)(v6 + 16LL * v9);
  v11 = *v10;
  if ( *v10 == a2 )
    return v10[1];
LABEL_16:
  v25 = 1;
  v21 = 0;
  while ( v11 != -4096 )
  {
    if ( !v21 && v11 == -8192 )
      v21 = v10;
    v9 = v7 & (v25 + v9);
    v10 = (__int64 *)(v6 + 16LL * v9);
    v11 = *v10;
    if ( *v10 == a2 )
      return v10[1];
    ++v25;
  }
  if ( !v21 )
    v21 = v10;
  v26 = *(_DWORD *)(a1 + 24);
  ++*(_QWORD *)(a1 + 8);
  v20 = v26 + 1;
  if ( 4 * (v26 + 1) >= 3 * v5 )
  {
LABEL_8:
    sub_30EC6C0(v2, 2 * v5);
    v16 = *(_DWORD *)(a1 + 32);
    if ( v16 )
    {
      v17 = v16 - 1;
      v18 = *(_QWORD *)(a1 + 16);
      v19 = (v16 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v20 = *(_DWORD *)(a1 + 24) + 1;
      v21 = (__int64 *)(v18 + 16LL * v19);
      v22 = *v21;
      if ( *v21 != a2 )
      {
        v23 = 1;
        v24 = 0;
        while ( v22 != -4096 )
        {
          if ( !v24 && v22 == -8192 )
            v24 = v21;
          v19 = v17 & (v23 + v19);
          v21 = (__int64 *)(v18 + 16LL * v19);
          v22 = *v21;
          if ( *v21 == a2 )
            goto LABEL_22;
          ++v23;
        }
        if ( v24 )
          v21 = v24;
      }
      goto LABEL_22;
    }
LABEL_48:
    ++*(_DWORD *)(a1 + 24);
    BUG();
  }
  if ( v5 - (v20 + *(_DWORD *)(a1 + 28)) <= v5 >> 3 )
  {
    v34 = v8;
    sub_30EC6C0(v2, v5);
    v27 = *(_DWORD *)(a1 + 32);
    if ( v27 )
    {
      v28 = v27 - 1;
      v29 = *(_QWORD *)(a1 + 16);
      v30 = 1;
      v31 = v28 & v34;
      v20 = *(_DWORD *)(a1 + 24) + 1;
      v32 = 0;
      v21 = (__int64 *)(v29 + 16LL * (v28 & v34));
      v33 = *v21;
      if ( *v21 != a2 )
      {
        while ( v33 != -4096 )
        {
          if ( v33 == -8192 && !v32 )
            v32 = v21;
          v31 = v28 & (v30 + v31);
          v21 = (__int64 *)(v29 + 16LL * v31);
          v33 = *v21;
          if ( *v21 == a2 )
            goto LABEL_22;
          ++v30;
        }
        if ( v32 )
          v21 = v32;
      }
      goto LABEL_22;
    }
    goto LABEL_48;
  }
LABEL_22:
  *(_DWORD *)(a1 + 24) = v20;
  if ( *v21 != -4096 )
    --*(_DWORD *)(a1 + 28);
  *v21 = a2;
  v21[1] = 0;
  return 0;
}
