// Function: sub_DBB070
// Address: 0xdbb070
//
__int64 __fastcall sub_DBB070(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rax
  unsigned int v4; // r13d
  __int64 v5; // r12
  unsigned int v6; // ebx
  unsigned __int64 v7; // rax
  unsigned __int64 v9; // rcx

  if ( a3 )
    v3 = sub_DBB040(a1, a2, a3);
  else
    v3 = sub_DCF3A0(a1, a2, 1);
  v4 = 0;
  if ( *(_WORD *)(v3 + 24) )
    return v4;
  v5 = *(_QWORD *)(v3 + 32);
  v6 = *(_DWORD *)(v5 + 32);
  if ( v6 > 0x40 )
  {
    if ( v6 - (unsigned int)sub_C444A0(v5 + 24) > 0x20 )
      return v4;
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
