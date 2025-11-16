// Function: sub_1625AE0
// Address: 0x1625ae0
//
__int64 __fastcall sub_1625AE0(__int64 a1, _QWORD *a2, _QWORD *a3)
{
  __int64 v4; // rax
  __int64 v5; // rbx
  _BYTE *v7; // rdi
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // rax
  _QWORD *v16; // rax

  if ( !*(_QWORD *)(a1 + 48) && *(__int16 *)(a1 + 18) >= 0 )
    return 0;
  v4 = sub_1625790(a1, 2);
  v5 = v4;
  if ( !v4 )
    return 0;
  if ( *(_DWORD *)(v4 + 8) != 3 )
    return 0;
  v7 = *(_BYTE **)(v4 - 24);
  if ( *v7 )
    return 0;
  v8 = sub_161E970((__int64)v7);
  if ( v9 != 14
    || *(_QWORD *)v8 != 0x775F68636E617262LL
    || *(_DWORD *)(v8 + 8) != 1751607653
    || *(_WORD *)(v8 + 12) != 29556 )
  {
    return 0;
  }
  v10 = *(unsigned int *)(v5 + 8);
  v11 = *(_QWORD *)(v5 + 8 * (1 - v10));
  if ( *(_BYTE *)v11 == 1 )
  {
    v12 = *(_QWORD *)(v11 + 136);
    if ( *(_BYTE *)(v12 + 16) != 13 )
      v12 = 0;
  }
  else
  {
    v12 = 0;
  }
  v13 = *(_QWORD *)(v5 + 8 * (2 - v10));
  if ( *(_BYTE *)v13 != 1 )
    return 0;
  v14 = *(_QWORD *)(v13 + 136);
  if ( *(_BYTE *)(v14 + 16) != 13 || !v12 )
    return 0;
  if ( *(_DWORD *)(v12 + 32) <= 0x40u )
    v15 = *(_QWORD *)(v12 + 24);
  else
    v15 = **(_QWORD **)(v12 + 24);
  *a2 = v15;
  v16 = *(_QWORD **)(v14 + 24);
  if ( *(_DWORD *)(v14 + 32) > 0x40u )
    v16 = (_QWORD *)*v16;
  *a3 = v16;
  return 1;
}
