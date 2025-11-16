// Function: sub_1B04DA0
// Address: 0x1b04da0
//
__int64 __fastcall sub_1B04DA0(__int64 a1, __int64 a2)
{
  _QWORD *v2; // rax
  __int64 v3; // rdx
  __int64 v4; // r12
  _QWORD *v5; // r14
  __int64 v7; // rbx
  __int64 v8; // r12
  int v9; // r8d
  int v10; // r9d
  __int64 v11; // rax
  char v12; // al
  int v13; // r8d
  __int64 v14; // rax
  __int64 v15; // r9
  _QWORD *v16; // rax
  const void *v17; // [rsp+8h] [rbp-48h]
  _QWORD *v18; // [rsp+18h] [rbp-38h]

  v2 = *(_QWORD **)(a1 + 16);
  if ( v2 == *(_QWORD **)(a1 + 8) )
    v3 = *(unsigned int *)(a1 + 28);
  else
    v3 = *(unsigned int *)(a1 + 24);
  v18 = &v2[v3];
  if ( v2 == v18 )
    return 1;
  while ( 1 )
  {
    v4 = *v2;
    v5 = v2;
    if ( *v2 < 0xFFFFFFFFFFFFFFFELL )
      break;
    if ( v18 == ++v2 )
      return 1;
  }
  if ( v2 == v18 )
    return 1;
  v17 = (const void *)(a2 + 16);
  while ( 1 )
  {
    v7 = *(_QWORD *)(v4 + 48);
    v8 = v4 + 40;
    if ( v7 != v8 )
      break;
LABEL_25:
    v16 = v5 + 1;
    if ( v5 + 1 == v18 )
      return 1;
    v4 = *v16;
    for ( ++v5; *v16 >= 0xFFFFFFFFFFFFFFFELL; v5 = v16 )
    {
      if ( v18 == ++v16 )
        return 1;
      v4 = *v16;
    }
    if ( v5 == v18 )
      return 1;
  }
  while ( 1 )
  {
    while ( 1 )
    {
      if ( !v7 )
        BUG();
      v12 = *(_BYTE *)(v7 - 8);
      if ( v12 != 54 )
        break;
      if ( sub_15F32D0(v7 - 24) || (*(_BYTE *)(v7 - 6) & 1) != 0 )
        return 0;
      v11 = *(unsigned int *)(a2 + 8);
      if ( (unsigned int)v11 >= *(_DWORD *)(a2 + 12) )
      {
        sub_16CD150(a2, v17, 0, 8, v9, v10);
        v11 = *(unsigned int *)(a2 + 8);
      }
      *(_QWORD *)(*(_QWORD *)a2 + 8 * v11) = v7 - 24;
      ++*(_DWORD *)(a2 + 8);
LABEL_16:
      v7 = *(_QWORD *)(v7 + 8);
      if ( v8 == v7 )
        goto LABEL_25;
    }
    if ( v12 != 55 )
      break;
    if ( sub_15F32D0(v7 - 24) || (*(_BYTE *)(v7 - 6) & 1) != 0 )
      return 0;
    v14 = *(unsigned int *)(a2 + 8);
    v15 = v7 - 24;
    if ( (unsigned int)v14 >= *(_DWORD *)(a2 + 12) )
    {
      sub_16CD150(a2, v17, 0, 8, v13, v7 - 24);
      v14 = *(unsigned int *)(a2 + 8);
      v15 = v7 - 24;
    }
    *(_QWORD *)(*(_QWORD *)a2 + 8 * v14) = v15;
    ++*(_DWORD *)(a2 + 8);
    v7 = *(_QWORD *)(v7 + 8);
    if ( v8 == v7 )
      goto LABEL_25;
  }
  if ( (unsigned __int8)sub_15F2ED0(v7 - 24) )
    return 0;
  if ( !(unsigned __int8)sub_15F3040(v7 - 24) )
    goto LABEL_16;
  return 0;
}
