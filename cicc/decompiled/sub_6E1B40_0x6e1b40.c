// Function: sub_6E1B40
// Address: 0x6e1b40
//
_BOOL8 __fastcall sub_6E1B40(__int64 a1)
{
  _BOOL8 result; // rax
  _BYTE *v2; // rdx

  result = *(_QWORD *)(a1 + 16) != 0;
  if ( !*(_QWORD *)(a1 + 16) )
  {
    if ( unk_4F04C18 )
    {
      if ( *(_BYTE *)(unk_4F04C18 + 42LL) )
      {
        if ( !*(_BYTE *)(a1 + 8) )
        {
          v2 = *(_BYTE **)(a1 + 24);
          if ( v2[24] == 2 && v2[325] == 12 && (v2[329] & 4) != 0 )
            return 1;
        }
      }
    }
  }
  return result;
}
