// Function: sub_22E0B80
// Address: 0x22e0b80
//
__int64 *__fastcall sub_22E0B80(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v6; // rax
  __int64 *v7; // r13
  unsigned int v8; // esi
  __int64 v9; // r8
  unsigned int v10; // edi
  __int64 *v11; // rax
  __int64 v12; // rcx
  bool (__fastcall *v13)(__int64, __int64 *); // rax
  int v15; // eax
  int v16; // esi
  __int64 v17; // r8
  unsigned int v18; // eax
  int v19; // ecx
  __int64 *v20; // rdx
  __int64 v21; // rdi
  int v22; // r11d
  int v23; // eax
  int v24; // eax
  int v25; // eax
  __int64 v26; // rdi
  __int64 *v27; // r8
  unsigned int v28; // r14d
  int v29; // r9d
  __int64 v30; // rsi
  int v31; // r10d
  __int64 *v32; // r9

  if ( sub_22DBDF0(a1, a2, a3) )
    return 0;
  v6 = sub_22077B0(0x70u);
  v7 = (__int64 *)v6;
  if ( v6 )
    sub_22DBFA0(v6, a2, a3, a1, *(_QWORD *)(a1 + 8), 0);
  v8 = *(_DWORD *)(a1 + 64);
  if ( !v8 )
  {
    ++*(_QWORD *)(a1 + 40);
    goto LABEL_12;
  }
  v9 = *(_QWORD *)(a1 + 48);
  v10 = (v8 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v11 = (__int64 *)(v9 + 16LL * v10);
  v12 = *v11;
  if ( a2 != *v11 )
  {
    v22 = 1;
    v20 = 0;
    while ( v12 != -4096 )
    {
      if ( v20 || v12 != -8192 )
        v11 = v20;
      v10 = (v8 - 1) & (v22 + v10);
      v12 = *(_QWORD *)(v9 + 16LL * v10);
      if ( a2 == v12 )
        goto LABEL_6;
      ++v22;
      v20 = v11;
      v11 = (__int64 *)(v9 + 16LL * v10);
    }
    if ( !v20 )
      v20 = v11;
    v23 = *(_DWORD *)(a1 + 56);
    ++*(_QWORD *)(a1 + 40);
    v19 = v23 + 1;
    if ( 4 * (v23 + 1) < 3 * v8 )
    {
      if ( v8 - *(_DWORD *)(a1 + 60) - v19 > v8 >> 3 )
      {
LABEL_14:
        *(_DWORD *)(a1 + 56) = v19;
        if ( *v20 != -4096 )
          --*(_DWORD *)(a1 + 60);
        *v20 = a2;
        v20[1] = (__int64)v7;
        goto LABEL_6;
      }
      sub_22E09A0(a1 + 40, v8);
      v24 = *(_DWORD *)(a1 + 64);
      if ( v24 )
      {
        v25 = v24 - 1;
        v26 = *(_QWORD *)(a1 + 48);
        v27 = 0;
        v28 = v25 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v29 = 1;
        v19 = *(_DWORD *)(a1 + 56) + 1;
        v20 = (__int64 *)(v26 + 16LL * v28);
        v30 = *v20;
        if ( a2 != *v20 )
        {
          while ( v30 != -4096 )
          {
            if ( v30 == -8192 && !v27 )
              v27 = v20;
            v28 = v25 & (v29 + v28);
            v20 = (__int64 *)(v26 + 16LL * v28);
            v30 = *v20;
            if ( a2 == *v20 )
              goto LABEL_14;
            ++v29;
          }
          if ( v27 )
            v20 = v27;
        }
        goto LABEL_14;
      }
LABEL_49:
      ++*(_DWORD *)(a1 + 56);
      BUG();
    }
LABEL_12:
    sub_22E09A0(a1 + 40, 2 * v8);
    v15 = *(_DWORD *)(a1 + 64);
    if ( v15 )
    {
      v16 = v15 - 1;
      v17 = *(_QWORD *)(a1 + 48);
      v18 = (v15 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v19 = *(_DWORD *)(a1 + 56) + 1;
      v20 = (__int64 *)(v17 + 16LL * v18);
      v21 = *v20;
      if ( a2 != *v20 )
      {
        v31 = 1;
        v32 = 0;
        while ( v21 != -4096 )
        {
          if ( v21 == -8192 && !v32 )
            v32 = v20;
          v18 = v16 & (v31 + v18);
          v20 = (__int64 *)(v17 + 16LL * v18);
          v21 = *v20;
          if ( a2 == *v20 )
            goto LABEL_14;
          ++v31;
        }
        if ( v32 )
          v20 = v32;
      }
      goto LABEL_14;
    }
    goto LABEL_49;
  }
LABEL_6:
  sub_22DCAC0(v7);
  v13 = *(bool (__fastcall **)(__int64, __int64 *))(*(_QWORD *)a1 + 16LL);
  if ( v13 == sub_22DB840 )
    sub_22DB7F0(v7);
  else
    v13(a1, v7);
  return v7;
}
