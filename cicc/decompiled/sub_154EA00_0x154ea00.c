// Function: sub_154EA00
// Address: 0x154ea00
//
void __fastcall sub_154EA00(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  __int64 j; // r15
  __int64 v4; // rsi
  __int64 i; // [rsp+8h] [rbp-38h]

  sub_154E850(a1, a2);
  v2 = *(_QWORD *)(a2 + 80);
  for ( i = a2 + 72; i != v2; v2 = *(_QWORD *)(v2 + 8) )
  {
    if ( !v2 )
      BUG();
    for ( j = *(_QWORD *)(v2 + 24); v2 + 16 != j; j = *(_QWORD *)(j + 8) )
    {
      v4 = j - 24;
      if ( !j )
        v4 = 0;
      sub_154E8E0(a1, v4);
    }
  }
}
