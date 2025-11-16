// Function: sub_82EC00
// Address: 0x82ec00
//
_BOOL8 __fastcall sub_82EC00(__int64 a1)
{
  _BOOL8 result; // rax
  __int64 v2; // rax
  __int64 v3; // rdx

  result = 0;
  if ( *(_BYTE *)(a1 + 16) == 3 )
  {
    v2 = *(_QWORD *)(a1 + 136);
    if ( (*(_BYTE *)(v2 + 81) & 0x10) == 0
      || (v3 = *(_QWORD *)(v2 + 64), result = 1, (*(_BYTE *)(v3 + 177) & 0x20) == 0) )
    {
      result = 0;
      if ( (*(_BYTE *)(a1 + 19) & 8) != 0 )
        return (unsigned int)sub_89A370(*(_QWORD *)(a1 + 104)) != 0;
    }
  }
  return result;
}
