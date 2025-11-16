// Function: sub_31DB520
// Address: 0x31db520
//
void __fastcall sub_31DB520(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rax
  __int64 v6; // rsi
  int v8; // edi
  unsigned int v9; // r9d
  __int64 *v10; // rdx
  __int64 v11; // r8
  __int64 *v12; // rbx
  __int64 v13; // r15
  __int64 v14; // rsi
  void (__fastcall *v15)(__int64, __int64, _QWORD); // r14
  __int64 v16; // rax
  int v17; // eax
  unsigned int v18; // r14d
  __int64 *v19; // rbx
  __int64 v20; // rax
  unsigned __int64 v21; // rdi
  int v22; // edx
  int v23; // r10d
  __int64 v24; // rdx
  int v25; // edx
  unsigned __int64 v26; // [rsp+8h] [rbp-48h]
  __int64 *v28; // [rsp+18h] [rbp-38h]

  if ( !a3 )
    return;
  v3 = *(unsigned int *)(a3 + 24);
  v6 = *(_QWORD *)(a3 + 8);
  if ( !(_DWORD)v3 )
    return;
  v8 = v3 - 1;
  v26 = ((0xBF58476D1CE4E5B9LL * a2) >> 31) ^ (0xBF58476D1CE4E5B9LL * a2);
  v9 = (v3 - 1) & (((0xBF58476D1CE4E5B9LL * a2) >> 31) ^ (484763065 * a2));
  v10 = (__int64 *)(v6 + 32LL * v9);
  v11 = *v10;
  if ( *v10 == a2 )
  {
LABEL_4:
    if ( v10 == (__int64 *)(v6 + 32 * v3) )
      return;
    v12 = (__int64 *)v10[1];
    v28 = &v12[*((unsigned int *)v10 + 4)];
    if ( v28 != v12 )
    {
      do
      {
        v13 = *(_QWORD *)(a1 + 224);
        v14 = *v12++;
        v15 = *(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v13 + 208LL);
        v16 = sub_31DB510(a1, v14);
        v15(v13, v16, 0);
      }
      while ( v28 != v12 );
      v17 = *(_DWORD *)(a3 + 24);
      v6 = *(_QWORD *)(a3 + 8);
      if ( !v17 )
        return;
      v8 = v17 - 1;
    }
    v18 = v8 & v26;
    v19 = (__int64 *)(v6 + 32LL * (v8 & (unsigned int)v26));
    v20 = *v19;
    if ( *v19 == a2 )
    {
LABEL_10:
      v21 = v19[1];
      if ( (__int64 *)v21 != v19 + 3 )
        _libc_free(v21);
      *v19 = -2;
      --*(_DWORD *)(a3 + 16);
      ++*(_DWORD *)(a3 + 20);
    }
    else
    {
      v25 = 1;
      while ( v20 != -1 )
      {
        v18 = v8 & (v25 + v18);
        v19 = (__int64 *)(v6 + 32LL * v18);
        v20 = *v19;
        if ( *v19 == a2 )
          goto LABEL_10;
        ++v25;
      }
    }
  }
  else
  {
    v22 = 1;
    while ( v11 != -1 )
    {
      v23 = v22 + 1;
      v24 = v8 & (v9 + v22);
      v9 = v24;
      v10 = (__int64 *)(v6 + 32 * v24);
      v11 = *v10;
      if ( *v10 == a2 )
        goto LABEL_4;
      v22 = v23;
    }
  }
}
