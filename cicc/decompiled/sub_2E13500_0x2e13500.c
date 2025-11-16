// Function: sub_2E13500
// Address: 0x2e13500
//
__int64 __fastcall sub_2E13500(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rcx
  __int64 v4; // r8
  __int64 v5; // rsi
  unsigned __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rdi
  __int64 v9; // r9
  unsigned __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 v14; // rdx
  _QWORD *v15; // r10
  unsigned int v16; // ebx
  __int64 *v17; // rcx
  __int64 v18; // rdx
  _QWORD *v19; // r8
  unsigned int v20; // esi
  __int64 *v21; // rcx

  v2 = **(_QWORD **)a2;
  v3 = (v2 >> 1) & 3;
  if ( ((v2 >> 1) & 3) == 0 )
    return 0;
  v4 = *(_QWORD *)(*(_QWORD *)a2 + 24LL * *(unsigned int *)(a2 + 8) - 16);
  v5 = (v4 >> 1) & 3;
  if ( ((v4 >> 1) & 3) == 0 )
    return 0;
  v6 = v2 & 0xFFFFFFFFFFFFFFF8LL;
  v7 = *(_QWORD *)(v6 + 16);
  v8 = *(_QWORD *)(a1 + 32);
  if ( !v7 )
  {
    v14 = *(unsigned int *)(v8 + 304);
    v15 = *(_QWORD **)(v8 + 296);
    if ( *(_DWORD *)(v8 + 304) )
    {
      v16 = v3 | *(_DWORD *)(v6 + 24);
      do
      {
        v17 = &v15[2 * (v14 >> 1)];
        if ( v16 >= (*(_DWORD *)((*v17 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v17 >> 1) & 3) )
        {
          v15 = v17 + 2;
          v14 = v14 - (v14 >> 1) - 1;
        }
        else
        {
          v14 >>= 1;
        }
      }
      while ( v14 > 0 );
    }
    v9 = *(v15 - 1);
    v10 = v4 & 0xFFFFFFFFFFFFFFF8LL;
    v11 = *(_QWORD *)((v4 & 0xFFFFFFFFFFFFFFF8LL) + 16);
    if ( v11 )
      goto LABEL_5;
LABEL_17:
    v18 = *(unsigned int *)(v8 + 304);
    v19 = *(_QWORD **)(v8 + 296);
    if ( *(_DWORD *)(v8 + 304) )
    {
      v20 = *(_DWORD *)(v10 + 24) | v5;
      do
      {
        v21 = &v19[2 * (v18 >> 1)];
        if ( v20 >= (*(_DWORD *)((*v21 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v21 >> 1) & 3) )
        {
          v19 = v21 + 2;
          v18 = v18 - (v18 >> 1) - 1;
        }
        else
        {
          v18 >>= 1;
        }
      }
      while ( v18 > 0 );
    }
    v12 = *(v19 - 1);
    goto LABEL_6;
  }
  v9 = *(_QWORD *)(v7 + 24);
  v10 = v4 & 0xFFFFFFFFFFFFFFF8LL;
  v11 = *(_QWORD *)((v4 & 0xFFFFFFFFFFFFFFF8LL) + 16);
  if ( !v11 )
    goto LABEL_17;
LABEL_5:
  v12 = *(_QWORD *)(v11 + 24);
LABEL_6:
  if ( v12 != v9 )
    return 0;
  return v9;
}
