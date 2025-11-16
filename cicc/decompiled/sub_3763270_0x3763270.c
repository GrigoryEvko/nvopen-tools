// Function: sub_3763270
// Address: 0x3763270
//
__int64 __fastcall sub_3763270(__int64 a1, unsigned int a2, unsigned __int16 a3, char a4, unsigned int a5)
{
  unsigned __int8 v6; // al

  if ( a4 )
  {
    if ( a3 != 1 )
    {
      a5 = 0;
      if ( !a3 || !*(_QWORD *)(a1 + 8LL * a3 + 112) )
        return a5;
    }
    if ( a2 <= 0x1F3 )
    {
      LOBYTE(a5) = *(_BYTE *)(a2 + a1 + 500LL * a3 + 6414) == 0;
      return a5;
    }
    return 0;
  }
  else
  {
    if ( a3 != 1 )
    {
      a5 = 0;
      if ( !a3 || !*(_QWORD *)(a1 + 8LL * a3 + 112) )
        return a5;
    }
    a5 = 1;
    if ( a2 > 0x1F3 )
      return a5;
    v6 = *(_BYTE *)(a2 + a1 + 500LL * a3 + 6414);
    if ( v6 <= 1u )
      return a5;
    LOBYTE(a5) = v6 == 4;
    return a5;
  }
}
