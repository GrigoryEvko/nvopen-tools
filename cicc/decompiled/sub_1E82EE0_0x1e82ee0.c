// Function: sub_1E82EE0
// Address: 0x1e82ee0
//
void __fastcall sub_1E82EE0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  __int64 v8; // rbx

  if ( a2 != a3 )
  {
    v8 = a2;
    do
    {
      while ( 1 )
      {
        sub_1E82EC0(a1, *(_QWORD *)(v8 + 24), v8, a4, a5, a6);
        if ( (*(_BYTE *)v8 & 4) == 0 )
          break;
        v8 = *(_QWORD *)(v8 + 8);
        if ( a3 == v8 )
          return;
      }
      while ( (*(_BYTE *)(v8 + 46) & 8) != 0 )
        v8 = *(_QWORD *)(v8 + 8);
      v8 = *(_QWORD *)(v8 + 8);
    }
    while ( a3 != v8 );
  }
}
