// Function: sub_AA5650
// Address: 0xaa5650
//
_BOOL8 __fastcall sub_AA5650(__int64 a1, int a2)
{
  __int64 v2; // rbp
  __int64 v3; // rdx
  _QWORD *v4; // rcx
  unsigned __int8 v5; // al
  __int64 v6; // rdx
  __int64 v7; // rdx
  _QWORD v9[2]; // [rsp-10h] [rbp-10h] BYREF

  v3 = *(_QWORD *)(a1 + 16);
  if ( !v3 )
    return !a2;
  while ( (unsigned __int8)(**(_BYTE **)(v3 + 24) - 30) > 0xAu )
  {
    v3 = *(_QWORD *)(v3 + 8);
    if ( !v3 )
      return !a2;
  }
  if ( !a2 )
    return 1;
  v9[1] = v2;
  v4 = v9;
  while ( 1 )
  {
    v5 = sub_AA4380(v4);
    v7 = *(_QWORD *)(v6 + 8);
    a2 -= v5;
    if ( !v7 )
      return !a2;
    do
    {
      if ( (unsigned __int8)(**(_BYTE **)(v7 + 24) - 30) <= 0xAu )
      {
        if ( !a2 )
          return 1;
        goto LABEL_6;
      }
      v7 = *(_QWORD *)(v7 + 8);
    }
    while ( v7 );
    if ( !a2 )
      return 1;
LABEL_6:
    if ( !v7 )
      return 0;
  }
}
