// Function: sub_BD3610
// Address: 0xbd3610
//
__int64 __fastcall sub_BD3610(__int64 a1, int a2)
{
  __int64 v2; // rdx
  unsigned __int8 v3; // al
  __int64 v4; // rdx
  __int64 v6; // rdx

  v2 = *(_QWORD *)(a1 + 16);
  if ( a2 )
  {
    while ( v2 )
    {
      v3 = sub_BD3030(v2);
      v2 = *(_QWORD *)(v4 + 8);
      a2 -= v3;
      if ( !a2 )
        goto LABEL_8;
    }
    return 0;
  }
  else
  {
LABEL_8:
    while ( v2 )
    {
      if ( (unsigned __int8)sub_BD3030(v2) )
        return 0;
      v2 = *(_QWORD *)(v6 + 8);
    }
    return 1;
  }
}
