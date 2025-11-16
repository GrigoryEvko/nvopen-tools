// Function: sub_358A390
// Address: 0x358a390
//
__int64 __fastcall sub_358A390(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  unsigned int v4; // esi
  __int64 v5; // r12
  __int64 v6; // rdi
  __int64 v7; // rcx
  int v8; // r11d
  __int64 *v9; // r13
  unsigned int v10; // r14d
  unsigned int v11; // edx
  __int64 *v12; // rax
  __int64 v13; // r9
  __int64 result; // rax
  int v15; // eax
  int v16; // ecx
  __int64 v17; // rdi
  unsigned int v18; // eax
  int v19; // edx
  __int64 v20; // rsi
  int v21; // eax
  int v22; // eax
  int v23; // eax
  __int64 v24; // rsi
  int v25; // r8d
  unsigned int v26; // r14d
  __int64 *v27; // rdi
  __int64 v28; // rcx
  int v29; // r9d
  __int64 *v30; // r8

  v3 = sub_B10CD0(a2 + 56);
  if ( !v3 )
    return *(_QWORD *)(a1 + 1200);
  v4 = *(_DWORD *)(a1 + 32);
  v5 = v3;
  v6 = a1 + 8;
  if ( !v4 )
  {
    ++*(_QWORD *)(a1 + 8);
    goto LABEL_7;
  }
  v7 = *(_QWORD *)(a1 + 16);
  v8 = 1;
  v9 = 0;
  v10 = ((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4);
  v11 = (v4 - 1) & v10;
  v12 = (__int64 *)(v7 + 16LL * v11);
  v13 = *v12;
  if ( v5 == *v12 )
    return v12[1];
  while ( v13 != -4096 )
  {
    if ( !v9 && v13 == -8192 )
      v9 = v12;
    v11 = (v4 - 1) & (v8 + v11);
    v12 = (__int64 *)(v7 + 16LL * v11);
    v13 = *v12;
    if ( v5 == *v12 )
      return v12[1];
    ++v8;
  }
  if ( !v9 )
    v9 = v12;
  v21 = *(_DWORD *)(a1 + 24);
  ++*(_QWORD *)(a1 + 8);
  v19 = v21 + 1;
  if ( 4 * (v21 + 1) >= 3 * v4 )
  {
LABEL_7:
    sub_26CAAB0(v6, 2 * v4);
    v15 = *(_DWORD *)(a1 + 32);
    if ( v15 )
    {
      v16 = v15 - 1;
      v17 = *(_QWORD *)(a1 + 16);
      v18 = (v15 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
      v19 = *(_DWORD *)(a1 + 24) + 1;
      v9 = (__int64 *)(v17 + 16LL * v18);
      v20 = *v9;
      if ( v5 != *v9 )
      {
        v29 = 1;
        v30 = 0;
        while ( v20 != -4096 )
        {
          if ( !v30 && v20 == -8192 )
            v30 = v9;
          v18 = v16 & (v29 + v18);
          v9 = (__int64 *)(v17 + 16LL * v18);
          v20 = *v9;
          if ( v5 == *v9 )
            goto LABEL_9;
          ++v29;
        }
        if ( v30 )
          v9 = v30;
      }
      goto LABEL_9;
    }
    goto LABEL_43;
  }
  if ( v4 - *(_DWORD *)(a1 + 28) - v19 <= v4 >> 3 )
  {
    sub_26CAAB0(v6, v4);
    v22 = *(_DWORD *)(a1 + 32);
    if ( v22 )
    {
      v23 = v22 - 1;
      v24 = *(_QWORD *)(a1 + 16);
      v25 = 1;
      v26 = v23 & v10;
      v27 = 0;
      v19 = *(_DWORD *)(a1 + 24) + 1;
      v9 = (__int64 *)(v24 + 16LL * v26);
      v28 = *v9;
      if ( v5 != *v9 )
      {
        while ( v28 != -4096 )
        {
          if ( v28 == -8192 && !v27 )
            v27 = v9;
          v26 = v23 & (v25 + v26);
          v9 = (__int64 *)(v24 + 16LL * v26);
          v28 = *v9;
          if ( v5 == *v9 )
            goto LABEL_9;
          ++v25;
        }
        if ( v27 )
          v9 = v27;
      }
      goto LABEL_9;
    }
LABEL_43:
    ++*(_DWORD *)(a1 + 24);
    BUG();
  }
LABEL_9:
  *(_DWORD *)(a1 + 24) = v19;
  if ( *v9 != -4096 )
    --*(_DWORD *)(a1 + 28);
  *v9 = v5;
  v9[1] = 0;
  result = sub_C1C070(*(_QWORD *)(a1 + 1200), v5, *(_QWORD *)(*(_QWORD *)(a1 + 1136) + 88LL), 0);
  v9[1] = result;
  return result;
}
