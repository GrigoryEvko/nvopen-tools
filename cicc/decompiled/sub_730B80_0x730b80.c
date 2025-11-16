// Function: sub_730B80
// Address: 0x730b80
//
__int64 __fastcall sub_730B80(__int64 a1)
{
  __int64 v1; // rbx
  unsigned __int8 v2; // al

  if ( *(_BYTE *)(a1 + 173) != 12 )
  {
    v1 = *(_QWORD *)(a1 + 176);
    if ( !v1 )
      return 1;
    while ( 1 )
    {
      v2 = *(_BYTE *)(v1 + 173);
      if ( v2 <= 7u )
      {
        if ( v2 <= 5u )
          goto LABEL_5;
        if ( !sub_730990(v1) )
          return 0;
        v1 = *(_QWORD *)(v1 + 120);
        if ( !v1 )
          return 1;
      }
      else
      {
        if ( v2 == 10 && !(unsigned int)sub_730B80(v1) )
          return 0;
LABEL_5:
        v1 = *(_QWORD *)(v1 + 120);
        if ( !v1 )
          return 1;
      }
    }
  }
  return 1;
}
