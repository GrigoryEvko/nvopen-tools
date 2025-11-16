// Function: sub_1A014D0
// Address: 0x1a014d0
//
bool __fastcall sub_1A014D0(__int64 a1)
{
  __int64 *v1; // rax
  __int64 v2; // r12
  __int64 v4; // rax
  __int64 v5; // r12
  __int64 v6; // rdi
  _QWORD *v7; // r12

  if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
  {
    v1 = *(__int64 **)(a1 - 8);
    if ( *(_BYTE *)(v1[3] + 16) == 9 )
      return 0;
  }
  else
  {
    v1 = (__int64 *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
    if ( *(_BYTE *)(v1[3] + 16) == 9 )
      return 0;
  }
  v2 = *v1;
  if ( sub_19FEFC0(*v1, 11, 12) || sub_19FEFC0(v2, 13, 14) )
    return 1;
  v4 = (*(_BYTE *)(a1 + 23) & 0x40) != 0 ? *(_QWORD *)(a1 - 8) : a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
  v5 = *(_QWORD *)(v4 + 24);
  if ( sub_19FEFC0(v5, 11, 12) || sub_19FEFC0(v5, 13, 14) )
    return 1;
  v6 = *(_QWORD *)(a1 + 8);
  if ( !v6 || *(_QWORD *)(v6 + 8) )
    return 0;
  v7 = sub_1648700(v6);
  if ( sub_19FEFC0((__int64)v7, 11, 12) )
    return 1;
  return sub_19FEFC0((__int64)v7, 13, 14) != 0;
}
