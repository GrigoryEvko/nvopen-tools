// Function: sub_1F81BC0
// Address: 0x1f81bc0
//
void __fastcall sub_1F81BC0(__int64 a1, __int64 a2)
{
  __int64 v2; // r9
  unsigned int v5; // esi
  int v6; // r14d
  _QWORD *v7; // r8
  _QWORD *v8; // rdx
  int v9; // r11d
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
  v2 = a1 + 560;
  v5 = *(_DWORD *)(a1 + 584);
  v6 = *(_DWORD *)(a1 + 40);
  if ( !v5 )
  {
    ++*(_QWORD *)(a1 + 560);
    goto LABEL_7;
  }
  v7 = *(_QWORD **)(a1 + 568);
  v8 = 0;
  v9 = 1;
  v10 = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v11 = &v7[2 * v10];
  v12 = *v11;
  if ( *v11 == a2 )
    return;
  while ( v12 != -8 )
  {
    if ( v12 != -16 || v8 )
      v11 = v8;
    v10 = (v5 - 1) & (v9 + v10);
    v12 = v7[2 * v10];
    if ( v12 == a2 )
      return;
    ++v9;
    v8 = v11;
    v11 = &v7[2 * v10];
  }
  if ( !v8 )
    v8 = v11;
  v19 = *(_DWORD *)(a1 + 576);
  ++*(_QWORD *)(a1 + 560);
  v16 = v19 + 1;
  if ( 4 * (v19 + 1) >= 3 * v5 )
  {
LABEL_7:
    sub_1D45DD0(v2, 2 * v5);
    v13 = *(_DWORD *)(a1 + 584);
    if ( v13 )
    {
      v14 = v13 - 1;
      v7 = *(_QWORD **)(a1 + 568);
      v15 = (v13 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v16 = *(_DWORD *)(a1 + 576) + 1;
      v8 = &v7[2 * v15];
      v17 = *v8;
      if ( *v8 != a2 )
      {
        v25 = 1;
        v2 = 0;
        while ( v17 != -8 )
        {
          if ( v17 == -16 && !v2 )
            v2 = (__int64)v8;
          v15 = v14 & (v25 + v15);
          v8 = &v7[2 * v15];
          v17 = *v8;
          if ( *v8 == a2 )
            goto LABEL_9;
          ++v25;
        }
        if ( v2 )
          v8 = (_QWORD *)v2;
      }
      goto LABEL_9;
    }
    goto LABEL_45;
  }
  if ( v5 - *(_DWORD *)(a1 + 580) - v16 <= v5 >> 3 )
  {
    sub_1D45DD0(v2, v5);
    v20 = *(_DWORD *)(a1 + 584);
    if ( v20 )
    {
      v21 = v20 - 1;
      v22 = *(_QWORD *)(a1 + 568);
      v7 = 0;
      v23 = v21 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      LODWORD(v2) = 1;
      v16 = *(_DWORD *)(a1 + 576) + 1;
      v8 = (_QWORD *)(v22 + 16LL * v23);
      v24 = *v8;
      if ( *v8 != a2 )
      {
        while ( v24 != -8 )
        {
          if ( !v7 && v24 == -16 )
            v7 = v8;
          v23 = v21 & (v2 + v23);
          v8 = (_QWORD *)(v22 + 16LL * v23);
          v24 = *v8;
          if ( *v8 == a2 )
            goto LABEL_9;
          LODWORD(v2) = v2 + 1;
        }
        if ( v7 )
          v8 = v7;
      }
      goto LABEL_9;
    }
LABEL_45:
    ++*(_DWORD *)(a1 + 576);
    BUG();
  }
LABEL_9:
  *(_DWORD *)(a1 + 576) = v16;
  if ( *v8 != -8 )
    --*(_DWORD *)(a1 + 580);
  *v8 = a2;
  *((_DWORD *)v8 + 2) = v6;
  v18 = *(unsigned int *)(a1 + 40);
  if ( (unsigned int)v18 >= *(_DWORD *)(a1 + 44) )
  {
    sub_16CD150(a1 + 32, (const void *)(a1 + 48), 0, 8, (int)v7, v2);
    v18 = *(unsigned int *)(a1 + 40);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 32) + 8 * v18) = a2;
  ++*(_DWORD *)(a1 + 40);
}
