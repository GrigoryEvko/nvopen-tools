// Function: sub_1C95850
// Address: 0x1c95850
//
__int64 __fastcall sub_1C95850(__int64 a1, __int64 a2)
{
  int v3; // eax
  char v4; // al
  __int64 v6; // rax

  while ( 1 )
  {
    v3 = *(unsigned __int16 *)(a1 + 18);
    if ( v3 == 48 )
      break;
    if ( v3 != 32 && v3 != 47 )
      return 0;
    a1 = *(_QWORD *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
    v4 = *(_BYTE *)(a1 + 16);
    if ( v4 != 5 )
    {
      if ( v4 != 17 || !(unsigned __int8)sub_1C2F070(a2) || !unk_4FBE1ED )
        return 0;
      return (unsigned __int8)sub_15E0450(a1) ^ 1u;
    }
  }
  v6 = **(_QWORD **)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
  if ( *(_BYTE *)(v6 + 8) != 15 )
    return 0;
  return *(_DWORD *)(v6 + 8) >> 8;
}
