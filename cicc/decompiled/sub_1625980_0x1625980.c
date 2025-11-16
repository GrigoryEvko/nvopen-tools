// Function: sub_1625980
// Address: 0x1625980
//
__int64 __fastcall sub_1625980(__int64 a1, _QWORD *a2)
{
  __int64 v2; // rax
  __int64 v3; // rbx
  __int64 v4; // rax
  _BYTE *v5; // r13
  __int64 v6; // rax
  __int64 v7; // rdx
  _WORD *v8; // rax
  __int64 v9; // rdx
  __int64 v11; // rdx
  unsigned int v12; // ecx
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rdx
  _QWORD *v19; // rax

  *a2 = 0;
  if ( !*(_QWORD *)(a1 + 48) && *(__int16 *)(a1 + 18) >= 0 )
    return 0;
  v2 = sub_1625790(a1, 2);
  v3 = v2;
  if ( !v2 )
    return 0;
  v4 = -(__int64)*(unsigned int *)(v2 + 8);
  v5 = *(_BYTE **)(v3 + 8 * v4);
  if ( *v5 )
    return 0;
  v6 = sub_161E970(*(_QWORD *)(v3 + 8 * v4));
  if ( v7 != 14
    || *(_QWORD *)v6 != 0x775F68636E617262LL
    || *(_DWORD *)(v6 + 8) != 1751607653
    || *(_WORD *)(v6 + 12) != 29556 )
  {
    v8 = (_WORD *)sub_161E970((__int64)v5);
    if ( v9 == 2 && *v8 == 20566 )
    {
      v16 = *(unsigned int *)(v3 + 8);
      if ( (unsigned int)v16 > 3 )
      {
        v17 = *(_QWORD *)(v3 + 8 * (2 - v16));
        if ( *(_BYTE *)v17 != 1 || (v18 = *(_QWORD *)(v17 + 136), *(_BYTE *)(v18 + 16) != 13) )
          BUG();
        v19 = *(_QWORD **)(v18 + 24);
        if ( *(_DWORD *)(v18 + 32) > 0x40u )
          v19 = (_QWORD *)*v19;
        *a2 = v19;
        return 1;
      }
    }
    return 0;
  }
  *a2 = 0;
  v11 = *(unsigned int *)(v3 + 8);
  v12 = 1;
  if ( (unsigned int)v11 > 1 )
  {
    while ( 1 )
    {
      v13 = *(_QWORD *)(v3 + 8 * (v12 - v11));
      if ( *(_BYTE *)v13 != 1 )
        break;
      v14 = *(_QWORD *)(v13 + 136);
      if ( *(_BYTE *)(v14 + 16) != 13 )
        break;
      if ( *(_DWORD *)(v14 + 32) <= 0x40u )
        v15 = *(_QWORD *)(v14 + 24);
      else
        v15 = **(_QWORD **)(v14 + 24);
      *a2 += v15;
      v11 = *(unsigned int *)(v3 + 8);
      if ( ++v12 >= (unsigned int)v11 )
        return 1;
    }
    return 0;
  }
  return 1;
}
