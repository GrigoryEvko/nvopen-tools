// Function: sub_72AE00
// Address: 0x72ae00
//
_BOOL8 __fastcall sub_72AE00(__int64 a1)
{
  char v1; // al
  _BOOL8 result; // rax
  __int64 v3; // rdx

  v1 = *(_BYTE *)(a1 + 173);
  if ( v1 == 1 )
  {
    v3 = *(_QWORD *)(a1 + 128);
    result = 0;
    if ( *(_BYTE *)(v3 + 140) == 2 )
    {
      if ( unk_4F072C8 == 1 )
      {
        if ( (*(_BYTE *)(v3 + 161) & 8) == 0 )
          return *(_QWORD *)(v3 + 168) != 0;
      }
      else
      {
        return (*(_BYTE *)(v3 + 161) & 8) != 0;
      }
    }
  }
  else if ( (*(_BYTE *)(a1 + 170) & 0x10) != 0 && v1 == 12 )
  {
    return ((*(_BYTE *)(a1 + 89) >> 2) ^ 1) & 1;
  }
  else
  {
    return 0;
  }
  return result;
}
