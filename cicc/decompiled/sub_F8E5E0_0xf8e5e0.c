// Function: sub_F8E5E0
// Address: 0xf8e5e0
//
__int64 __fastcall sub_F8E5E0(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 v3; // rbx
  __int64 v4; // r14
  __int64 v6; // rbx
  __int64 v7; // rax

  if ( *(_BYTE *)a2 == 32 )
  {
    if ( !sub_AA5650(*(_QWORD *)(a2 + 40), 0x80 / ((*(_DWORD *)(a2 + 4) & 0x7FFFFFFu) >> 1)) )
    {
      v2 = **(_QWORD **)(a2 - 8);
      if ( v2 )
        goto LABEL_4;
    }
    return 0;
  }
  if ( *(_BYTE *)a2 != 31 )
    return 0;
  if ( (*(_DWORD *)(a2 + 4) & 0x7FFFFFF) != 3 )
    return 0;
  v6 = *(_QWORD *)(a2 - 96);
  v7 = *(_QWORD *)(v6 + 16);
  if ( !v7 )
    return 0;
  if ( *(_QWORD *)(v7 + 8) )
    return 0;
  if ( *(_BYTE *)v6 != 82 )
    return 0;
  if ( (*(_WORD *)(v6 + 2) & 0x3Fu) - 32 > 1 )
    return 0;
  if ( !sub_F8E510(*(_QWORD *)(v6 - 32), *(_QWORD *)(a1 + 24)) )
    return 0;
  v2 = *(_QWORD *)(v6 - 64);
  if ( !v2 )
    return 0;
LABEL_4:
  if ( *(_BYTE *)v2 == 76 )
  {
    v3 = *(_QWORD *)(v2 - 32);
    v4 = *(_QWORD *)(v2 + 8);
    if ( v4 == sub_AE4450(*(_QWORD *)(a1 + 24), *(_QWORD *)(v3 + 8)) )
      return v3;
  }
  return v2;
}
