// Function: sub_1474190
// Address: 0x1474190
//
__int64 __fastcall sub_1474190(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v3; // r13d
  __int64 v4; // rax
  __int64 v5; // r12
  unsigned int v6; // ebx
  unsigned __int64 v7; // rax
  unsigned __int64 v9; // rcx

  v3 = 0;
  v4 = sub_1474160(a1, a2, a3);
  if ( *(_WORD *)(v4 + 24) )
    return v3;
  v5 = *(_QWORD *)(v4 + 32);
  v6 = *(_DWORD *)(v5 + 32);
  if ( v6 > 0x40 )
  {
    if ( v6 - (unsigned int)sub_16A57B0(v5 + 24) > 0x20 )
      return v3;
    v7 = **(_QWORD **)(v5 + 24);
    return (unsigned int)(v7 + 1);
  }
  v7 = *(_QWORD *)(v5 + 24);
  if ( !v7 )
    return (unsigned int)(v7 + 1);
  _BitScanReverse64(&v9, v7);
  if ( 64 - ((unsigned int)v9 ^ 0x3F) <= 0x20 )
    return (unsigned int)(v7 + 1);
  return 0;
}
