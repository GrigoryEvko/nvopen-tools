// Function: sub_390B160
// Address: 0x390b160
//
_BOOL8 __fastcall sub_390B160(__int64 a1, __int64 a2)
{
  char v2; // dl
  _BOOL8 result; // rax
  unsigned __int64 v4; // rdx
  __int64 v5; // rax
  unsigned __int64 v6; // rax

  v2 = *(_BYTE *)(a2 + 8);
  result = 1;
  if ( (v2 & 1) != 0 )
  {
    if ( (*(_QWORD *)a2 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
    {
      if ( off_4CF6DB8 == (_UNKNOWN *)(*(_QWORD *)a2 & 0xFFFFFFFFFFFFFFF8LL) )
        return 0;
      return (*(_BYTE *)(a2 + 9) & 2) != 0;
    }
    else
    {
      if ( (*(_BYTE *)(a2 + 9) & 0xC) != 8 )
        return 0;
      *(_BYTE *)(a2 + 8) = v2 | 4;
      v4 = (unsigned __int64)sub_38CE440(*(_QWORD *)(a2 + 24));
      v5 = v4 | *(_QWORD *)a2 & 7LL;
      *(_QWORD *)a2 = v5;
      if ( !v4 )
        return 0;
      v6 = v5 & 0xFFFFFFFFFFFFFFF8LL;
      if ( !v6 )
      {
        v6 = 0;
        if ( (*(_BYTE *)(a2 + 9) & 0xC) == 8 )
        {
          *(_BYTE *)(a2 + 8) |= 4u;
          v6 = (unsigned __int64)sub_38CE440(*(_QWORD *)(a2 + 24));
          *(_QWORD *)a2 = v6 | *(_QWORD *)a2 & 7LL;
        }
      }
      return off_4CF6DB8 != (_UNKNOWN *)v6 && (*(_BYTE *)(a2 + 9) & 2) != 0;
    }
  }
  return result;
}
