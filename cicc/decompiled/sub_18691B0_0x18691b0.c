// Function: sub_18691B0
// Address: 0x18691b0
//
__int64 __fastcall sub_18691B0(__int64 a1, __int64 a2)
{
  unsigned int v2; // r15d
  __int64 v3; // r12
  __int64 v4; // rbx
  __int64 v5; // r8
  __int64 v6; // r14
  int v7; // eax

  v2 = 0;
  v3 = a1 + 24;
  v4 = *(_QWORD *)(a1 + 32);
  if ( v4 != a1 + 24 )
  {
    do
    {
      while ( 1 )
      {
        v5 = v4 - 56;
        if ( !v4 )
          v5 = 0;
        v6 = v5;
        if ( sub_15E4F60(v5) && !(unsigned __int8)sub_1560180(v6 + 112, 35) )
          break;
        v4 = *(_QWORD *)(v4 + 8);
        if ( v3 == v4 )
          return v2;
      }
      v7 = sub_1AAE740(v6, a2);
      v4 = *(_QWORD *)(v4 + 8);
      v2 |= v7;
    }
    while ( v3 != v4 );
  }
  return v2;
}
