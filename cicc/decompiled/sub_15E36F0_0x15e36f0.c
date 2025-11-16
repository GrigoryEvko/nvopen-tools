// Function: sub_15E36F0
// Address: 0x15e36f0
//
__int64 __fastcall sub_15E36F0(__int64 a1)
{
  unsigned int v1; // r12d
  __int64 v2; // rbx

  LOBYTE(v1) = (((*(_BYTE *)(a1 + 32) & 0xF) + 9) & 0xFu) > 1
            && (*(_BYTE *)(a1 + 32) & 0xFu) - 2 > 1
            && (*(_BYTE *)(a1 + 32) & 0xF) != 1;
  if ( (_BYTE)v1 )
    return 0;
  v2 = *(_QWORD *)(a1 + 8);
  if ( !v2 )
    return 1;
  while ( *(_BYTE *)(sub_1648700(v2) + 16) == 4 )
  {
    v2 = *(_QWORD *)(v2 + 8);
    if ( !v2 )
      return 1;
  }
  return v1;
}
