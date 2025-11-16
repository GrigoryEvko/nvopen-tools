// Function: sub_BCF080
// Address: 0xbcf080
//
__int64 __fastcall sub_BCF080(__int64 a1)
{
  char i; // al
  int v2; // eax
  __int64 result; // rax

  for ( i = *(_BYTE *)(a1 + 8); i == 16; i = *(_BYTE *)(a1 + 8) )
    a1 = *(_QWORD *)(a1 + 24);
  if ( i == 15 )
  {
    result = 1;
    if ( (*(_DWORD *)(a1 + 8) & 0x4000) == 0 )
    {
      if ( ((*(_DWORD *)(a1 + 8) >> 8) & 0x80) != 0 )
        return 0;
      else
        return sub_BCF150();
    }
  }
  else if ( i == 20 )
  {
    LOBYTE(v2) = sub_BCEE90(a1, 4);
    return v2 ^ 1u;
  }
  else
  {
    return 0;
  }
  return result;
}
