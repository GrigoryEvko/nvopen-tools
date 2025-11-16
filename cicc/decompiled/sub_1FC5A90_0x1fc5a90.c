// Function: sub_1FC5A90
// Address: 0x1fc5a90
//
void __fastcall sub_1FC5A90(__int64 *a1, __int64 a2)
{
  __int64 v3; // r12
  unsigned int v4; // esi
  int v5; // r14d
  __int64 v6; // r9
  int v7; // r11d
  _QWORD *v8; // r8
  _QWORD *v9; // rdx
  unsigned int v10; // edi
  _QWORD *v11; // rax
  __int64 v12; // rcx
  int v13; // eax
  int v14; // esi
  unsigned int v15; // eax
  int v16; // ecx
  __int64 v17; // rdi
  __int64 v18; // rax
  int v19; // eax
  int v20; // eax
  int v21; // eax
  __int64 v22; // rdi
  unsigned int v23; // r13d
  __int64 v24; // rsi
  int v25; // r10d

  if ( *(_WORD *)(a2 + 24) == 212 )
    return;
  v3 = *a1;
  v4 = *(_DWORD *)(*a1 + 584);
  v5 = *(_DWORD *)(*a1 + 40);
  v6 = *a1 + 560;
  if ( !v4 )
  {
    ++*(_QWORD *)(v3 + 560);
    goto LABEL_7;
  }
  v7 = 1;
  v8 = *(_QWORD **)(v3 + 568);
  v9 = 0;
  v10 = (v4 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v11 = &v8[2 * v10];
  v12 = *v11;
  if ( a2 == *v11 )
    return;
  while ( v12 != -8 )
  {
    if ( v12 != -16 || v9 )
      v11 = v9;
    v10 = (v4 - 1) & (v7 + v10);
    v12 = v8[2 * v10];
    if ( a2 == v12 )
      return;
    ++v7;
    v9 = v11;
    v11 = &v8[2 * v10];
  }
  if ( !v9 )
    v9 = v11;
  v19 = *(_DWORD *)(v3 + 576);
  ++*(_QWORD *)(v3 + 560);
  v16 = v19 + 1;
  if ( 4 * (v19 + 1) >= 3 * v4 )
  {
LABEL_7:
    sub_1D45DD0(v6, 2 * v4);
    v13 = *(_DWORD *)(v3 + 584);
    if ( v13 )
    {
      v14 = v13 - 1;
      v8 = *(_QWORD **)(v3 + 568);
      v15 = (v13 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v16 = *(_DWORD *)(v3 + 576) + 1;
      v9 = &v8[2 * v15];
      v17 = *v9;
      if ( a2 != *v9 )
      {
        v25 = 1;
        v6 = 0;
        while ( v17 != -8 )
        {
          if ( v17 == -16 && !v6 )
            v6 = (__int64)v9;
          v15 = v14 & (v25 + v15);
          v9 = &v8[2 * v15];
          v17 = *v9;
          if ( a2 == *v9 )
            goto LABEL_9;
          ++v25;
        }
        if ( v6 )
          v9 = (_QWORD *)v6;
      }
      goto LABEL_9;
    }
    goto LABEL_45;
  }
  if ( v4 - *(_DWORD *)(v3 + 580) - v16 <= v4 >> 3 )
  {
    sub_1D45DD0(v6, v4);
    v20 = *(_DWORD *)(v3 + 584);
    if ( v20 )
    {
      v21 = v20 - 1;
      LODWORD(v6) = 1;
      v8 = 0;
      v22 = *(_QWORD *)(v3 + 568);
      v23 = v21 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v16 = *(_DWORD *)(v3 + 576) + 1;
      v9 = (_QWORD *)(v22 + 16LL * v23);
      v24 = *v9;
      if ( a2 != *v9 )
      {
        while ( v24 != -8 )
        {
          if ( !v8 && v24 == -16 )
            v8 = v9;
          v23 = v21 & (v6 + v23);
          v9 = (_QWORD *)(v22 + 16LL * v23);
          v24 = *v9;
          if ( a2 == *v9 )
            goto LABEL_9;
          LODWORD(v6) = v6 + 1;
        }
        if ( v8 )
          v9 = v8;
      }
      goto LABEL_9;
    }
LABEL_45:
    ++*(_DWORD *)(v3 + 576);
    BUG();
  }
LABEL_9:
  *(_DWORD *)(v3 + 576) = v16;
  if ( *v9 != -8 )
    --*(_DWORD *)(v3 + 580);
  *v9 = a2;
  *((_DWORD *)v9 + 2) = v5;
  v18 = *(unsigned int *)(v3 + 40);
  if ( (unsigned int)v18 >= *(_DWORD *)(v3 + 44) )
  {
    sub_16CD150(v3 + 32, (const void *)(v3 + 48), 0, 8, (int)v8, v6);
    v18 = *(unsigned int *)(v3 + 40);
  }
  *(_QWORD *)(*(_QWORD *)(v3 + 32) + 8 * v18) = a2;
  ++*(_DWORD *)(v3 + 40);
}
