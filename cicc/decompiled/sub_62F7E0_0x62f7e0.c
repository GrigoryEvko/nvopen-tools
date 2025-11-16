// Function: sub_62F7E0
// Address: 0x62f7e0
//
_BOOL8 __fastcall sub_62F7E0(__int64 a1)
{
  _BOOL4 v1; // r13d
  __int64 v2; // r12
  __int64 v4; // rsi

  v1 = 0;
  if ( a1 )
  {
    v2 = a1;
    do
    {
      v1 = 0;
      if ( *(_BYTE *)(v2 + 8) == 1 )
      {
        if ( *(_QWORD *)(v2 + 24) )
        {
          v1 = sub_62F7E0() != 0;
        }
        else
        {
          v1 = 1;
          v4 = sub_6E1A20(v2);
          sub_6851C0(2278, v4);
        }
      }
      if ( !*(_QWORD *)v2 )
        break;
      v2 = *(_BYTE *)(*(_QWORD *)v2 + 8LL) == 3 ? sub_6BBB10(v2) : *(_QWORD *)v2;
    }
    while ( v2 && !v1 );
  }
  return v1;
}
