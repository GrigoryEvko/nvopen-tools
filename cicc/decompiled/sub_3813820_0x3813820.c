// Function: sub_3813820
// Address: 0x3813820
//
__int64 __fastcall sub_3813820(__int64 a1, unsigned int a2, unsigned __int16 a3, char a4, unsigned int a5)
{
  if ( !a4 )
  {
    if ( a3 == 1 || (a5 = 0, a3) && *(_QWORD *)(a1 + 8LL * a3 + 112) )
    {
      a5 = 1;
      if ( a2 <= 0x1F3 )
        LOBYTE(a5) = (*(_BYTE *)(a2 + a1 + 500LL * a3 + 6414) & 0xFB) == 0;
    }
    return a5;
  }
  if ( a3 != 1 )
  {
    a5 = 0;
    if ( !a3 || !*(_QWORD *)(a1 + 8LL * a3 + 112) )
      return a5;
  }
  if ( a2 > 0x1F3 )
    return 0;
  LOBYTE(a5) = *(_BYTE *)(a2 + a1 + 500LL * a3 + 6414) == 0;
  return a5;
}
