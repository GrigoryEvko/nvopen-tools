// Function: sub_1C995B0
// Address: 0x1c995b0
//
void __fastcall sub_1C995B0(_QWORD *a1, __int64 a2)
{
  __int64 v2; // r14
  __int64 v3; // rbx
  __int64 i; // r13
  __int64 v5; // rsi

  v2 = a2 + 72;
  v3 = *(_QWORD *)(a2 + 80);
  if ( a2 + 72 != v3 )
  {
    if ( !v3 )
      BUG();
    while ( 1 )
    {
      i = *(_QWORD *)(v3 + 24);
      if ( i != v3 + 16 )
        break;
      v3 = *(_QWORD *)(v3 + 8);
      if ( v2 == v3 )
        return;
      if ( !v3 )
        BUG();
    }
    while ( v3 != v2 )
    {
      v5 = i - 24;
      if ( !i )
        v5 = 0;
      sub_1C99380(a1, v5);
      for ( i = *(_QWORD *)(i + 8); i == v3 - 24 + 40; i = *(_QWORD *)(v3 + 24) )
      {
        v3 = *(_QWORD *)(v3 + 8);
        if ( v2 == v3 )
          return;
        if ( !v3 )
          BUG();
      }
    }
  }
}
