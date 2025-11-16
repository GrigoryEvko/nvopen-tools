// Function: sub_143B820
// Address: 0x143b820
//
__int64 __fastcall sub_143B820(__int64 a1)
{
  unsigned int v1; // r12d
  unsigned __int8 v2; // dl
  __int64 v4; // rbx

  v2 = *(_BYTE *)(a1 + 16);
  LOBYTE(v1) = v2 == 77 || v2 == 56;
  if ( (_BYTE)v1 )
    return 1;
  if ( (unsigned int)v2 - 60 > 0xC )
  {
    if ( v2 != 35 )
      return v1;
    goto LABEL_9;
  }
  if ( (unsigned __int8)sub_14AF470(a1, 0, 0, 0) )
    return 1;
  if ( *(_BYTE *)(a1 + 16) != 35 )
    return v1;
LABEL_9:
  if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
    v4 = *(_QWORD *)(a1 - 8);
  else
    v4 = a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
  LOBYTE(v1) = *(_BYTE *)(*(_QWORD *)(v4 + 24) + 16LL) == 13;
  return v1;
}
