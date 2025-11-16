// Function: sub_5C64B0
// Address: 0x5c64b0
//
_BOOL8 __fastcall sub_5C64B0(__int64 a1, __int64 a2)
{
  char v2; // dl
  _BOOL8 result; // rax

  if ( !a1 )
    return 0;
  v2 = *(_BYTE *)(a2 + 80);
  result = 1;
  if ( *(_BYTE *)(a1 + 80) == v2 && (*(_QWORD *)(a1 + 80) & 0x800000200LL) == 0 )
  {
    if ( v2 == 7 )
    {
      return (*(_BYTE *)(*(_QWORD *)(a2 + 88) + 168LL) & 0x10) != 0;
    }
    else
    {
      if ( v2 != 11 )
        sub_721090(a1);
      return (*(_BYTE *)(*(_QWORD *)(a2 + 88) + 200LL) & 0x40) != 0;
    }
  }
  return result;
}
