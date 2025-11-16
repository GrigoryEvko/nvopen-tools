// Function: sub_1474290
// Address: 0x1474290
//
__int64 __fastcall sub_1474290(__int64 a1, __int64 a2)
{
  unsigned int v2; // r13d
  unsigned __int64 v3; // rax
  __int64 v4; // r12
  unsigned int v5; // ebx
  unsigned __int64 v6; // rax
  unsigned __int64 v8; // rcx

  v2 = 0;
  v3 = sub_1474260(a1, a2);
  if ( *(_WORD *)(v3 + 24) )
    return v2;
  v4 = *(_QWORD *)(v3 + 32);
  v5 = *(_DWORD *)(v4 + 32);
  if ( v5 > 0x40 )
  {
    if ( v5 - (unsigned int)sub_16A57B0(v4 + 24) > 0x20 )
      return v2;
    v6 = **(_QWORD **)(v4 + 24);
    return (unsigned int)(v6 + 1);
  }
  v6 = *(_QWORD *)(v4 + 24);
  if ( !v6 )
    return (unsigned int)(v6 + 1);
  _BitScanReverse64(&v8, v6);
  if ( 64 - ((unsigned int)v8 ^ 0x3F) <= 0x20 )
    return (unsigned int)(v6 + 1);
  return 0;
}
