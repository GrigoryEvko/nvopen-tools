// Function: sub_2CE1400
// Address: 0x2ce1400
//
void __fastcall sub_2CE1400(_QWORD *a1, __int64 a2)
{
  __int64 v2; // r14
  __int64 v3; // rbx
  __int64 i; // r13
  char *v5; // rsi

  v2 = a2 + 72;
  v3 = *(_QWORD *)(a2 + 80);
  if ( a2 + 72 != v3 )
  {
    if ( !v3 )
      BUG();
    while ( 1 )
    {
      i = *(_QWORD *)(v3 + 32);
      if ( i != v3 + 24 )
        break;
      v3 = *(_QWORD *)(v3 + 8);
      if ( v2 == v3 )
        return;
      if ( !v3 )
        BUG();
    }
    while ( v3 != v2 )
    {
      v5 = (char *)(i - 24);
      if ( !i )
        v5 = 0;
      sub_2CE11B0(a1, v5);
      for ( i = *(_QWORD *)(i + 8); i == v3 - 24 + 48; i = *(_QWORD *)(v3 + 32) )
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
