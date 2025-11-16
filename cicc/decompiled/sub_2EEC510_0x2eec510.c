// Function: sub_2EEC510
// Address: 0x2eec510
//
void __fastcall sub_2EEC510(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rbx

  if ( a2 != a3 )
  {
    v6 = a2;
    do
    {
      while ( 1 )
      {
        sub_2EEC4F0(a1, *(_QWORD *)(v6 + 24), v6, a4);
        if ( (*(_BYTE *)v6 & 4) == 0 )
          break;
        v6 = *(_QWORD *)(v6 + 8);
        if ( a3 == v6 )
          return;
      }
      while ( (*(_BYTE *)(v6 + 44) & 8) != 0 )
        v6 = *(_QWORD *)(v6 + 8);
      v6 = *(_QWORD *)(v6 + 8);
    }
    while ( a3 != v6 );
  }
}
