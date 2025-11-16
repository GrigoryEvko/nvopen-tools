// Function: sub_91B5A0
// Address: 0x91b5a0
//
char __fastcall sub_91B5A0(__int64 a1)
{
  char result; // al
  __int64 i; // rbx
  char v3; // al

  result = 1;
  if ( (*(_BYTE *)(a1 + 168) & 0x40) == 0 )
  {
    if ( unk_4D04630 )
    {
      return 0;
    }
    else
    {
      for ( i = *(_QWORD *)(a1 + 120); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
        ;
      if ( *(_QWORD *)(a1 + 8)
        && (v3 = *(_BYTE *)(a1 + 156), (v3 & 1) != 0)
        && ((v3 & 2) == 0
         || *(_QWORD *)(i + 128)
         || !sub_8D3410(i)
         || *(_QWORD *)(i + 176)
         || (*(_BYTE *)(i + 169) & 0x20) != 0)
        && (*(_BYTE *)(a1 + 174) & 4) == 0
        && (*(_BYTE *)(a1 + 176) & 0x20) == 0 )
      {
        return (unsigned __int8)~*(_BYTE *)(a1 + 173) >> 7;
      }
      else
      {
        return 0;
      }
    }
  }
  return result;
}
