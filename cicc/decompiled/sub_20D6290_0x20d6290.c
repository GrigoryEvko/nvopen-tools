// Function: sub_20D6290
// Address: 0x20d6290
//
bool __fastcall sub_20D6290(__int64 a1, __int64 a2)
{
  __int64 v2; // rax

  v2 = *(_QWORD *)(*(_QWORD *)(a1 + 24) + 32LL);
  if ( a1 != v2 && a2 != v2 )
  {
    do
    {
      if ( !v2 )
        BUG();
      if ( (*(_BYTE *)v2 & 4) != 0 )
      {
        v2 = *(_QWORD *)(v2 + 8);
        if ( a2 == v2 )
          return a1 == v2;
      }
      else
      {
        while ( (*(_BYTE *)(v2 + 46) & 8) != 0 )
          v2 = *(_QWORD *)(v2 + 8);
        v2 = *(_QWORD *)(v2 + 8);
        if ( a2 == v2 )
          return a1 == v2;
      }
    }
    while ( a1 != v2 );
  }
  return a1 == v2;
}
