// Function: sub_297B1E0
// Address: 0x297b1e0
//
__int64 __fastcall sub_297B1E0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rbx
  __int64 v5; // r12
  unsigned int v6; // r15d
  __int64 v7; // rsi
  int v8; // eax

  if ( *(_BYTE *)a1 && !(unsigned __int8)sub_DF9710(a3) )
    return 0;
  *(_QWORD *)(a1 + 8) = a3;
  v4 = *(_QWORD *)(a2 + 80);
  v5 = a2 + 72;
  v6 = 0;
  if ( v4 == a2 + 72 )
  {
    return 0;
  }
  else
  {
    do
    {
      v7 = v4 - 24;
      if ( !v4 )
        v7 = 0;
      v8 = sub_297B020(a1, v7);
      v4 = *(_QWORD *)(v4 + 8);
      v6 |= v8;
    }
    while ( v5 != v4 );
  }
  return v6;
}
