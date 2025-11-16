// Function: sub_1C016B0
// Address: 0x1c016b0
//
__int64 __fastcall sub_1C016B0(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 v3; // rbx
  unsigned int v4; // r15d
  __int64 v5; // rsi
  unsigned int v6; // eax

  v2 = a2 + 72;
  v3 = *(_QWORD *)(a2 + 80);
  if ( v3 == a2 + 72 )
  {
    return 0;
  }
  else
  {
    v4 = 0;
    do
    {
      v5 = v3 - 24;
      if ( !v3 )
        v5 = 0;
      v6 = sub_1C014D0(a1, v5, 0);
      v3 = *(_QWORD *)(v3 + 8);
      if ( v4 < v6 )
        v4 = v6;
    }
    while ( v2 != v3 );
  }
  return v4;
}
