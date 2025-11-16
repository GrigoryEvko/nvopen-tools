// Function: sub_AA5590
// Address: 0xaa5590
//
__int64 __fastcall sub_AA5590(__int64 a1, int a2)
{
  __int64 v2; // rbp
  __int64 v3; // rdx
  unsigned __int8 v4; // al
  __int64 v5; // rdx
  __int64 v6; // rdx
  __int64 v7; // rdx
  _QWORD v9[2]; // [rsp-10h] [rbp-10h] BYREF

  v3 = *(_QWORD *)(a1 + 16);
  if ( !v3 )
    return a2 == 0;
  while ( (unsigned __int8)(**(_BYTE **)(v3 + 24) - 30) > 0xAu )
  {
    v3 = *(_QWORD *)(v3 + 8);
    if ( !v3 )
      return a2 == 0;
  }
  v9[1] = v2;
  if ( a2 )
  {
LABEL_7:
    v4 = sub_AA4160(v9);
    v6 = *(_QWORD *)(v5 + 8);
    a2 -= v4;
    if ( !v6 )
      return !a2;
    while ( (unsigned __int8)(**(_BYTE **)(v6 + 24) - 30) > 0xAu )
    {
      v6 = *(_QWORD *)(v6 + 8);
      if ( !v6 )
      {
        if ( !a2 )
          goto LABEL_11;
LABEL_6:
        if ( !v6 )
          return 0;
        goto LABEL_7;
      }
    }
    if ( a2 )
      goto LABEL_6;
LABEL_11:
    if ( !v6 )
      return 1;
  }
  if ( !(unsigned __int8)sub_AA4160(v9) )
  {
    do
      v7 = *(_QWORD *)(v7 + 8);
    while ( v7 );
    return 1;
  }
  return 0;
}
