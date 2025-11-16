// Function: sub_2EBEE10
// Address: 0x2ebee10
//
unsigned __int64 __fastcall sub_2EBEE10(__int64 a1, int a2)
{
  unsigned __int64 i; // r8

  if ( a2 < 0 )
    i = *(_QWORD *)(*(_QWORD *)(a1 + 56) + 16LL * (a2 & 0x7FFFFFFF) + 8);
  else
    i = *(_QWORD *)(*(_QWORD *)(a1 + 304) + 8LL * (unsigned int)a2);
  if ( i )
  {
    if ( (*(_BYTE *)(i + 3) & 0x10) == 0 )
    {
      i = *(_QWORD *)(i + 32);
      if ( !i )
        return i;
      if ( (*(_BYTE *)(i + 3) & 0x10) == 0 )
        return 0;
    }
    for ( i = *(_QWORD *)(i + 16); (*(_BYTE *)(i + 44) & 4) != 0; i = *(_QWORD *)i & 0xFFFFFFFFFFFFFFF8LL )
      ;
  }
  return i;
}
