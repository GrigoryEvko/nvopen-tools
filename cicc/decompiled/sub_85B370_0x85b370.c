// Function: sub_85B370
// Address: 0x85b370
//
void __fastcall sub_85B370(__int64 a1)
{
  __int64 i; // rbx
  __int64 v2; // rbx
  char v3; // al
  __int64 v4; // rdi
  __int64 v5; // rdx

  for ( i = *(_QWORD *)(a1 + 168); i; i = *(_QWORD *)(i + 112) )
  {
    if ( (*(_BYTE *)(i + 124) & 1) == 0 )
      sub_85B370(*(_QWORD *)(i + 128));
  }
  v2 = *(_QWORD *)(a1 + 104);
  if ( v2 )
  {
    while ( 1 )
    {
      v3 = *(_BYTE *)(v2 + 140);
      if ( v3 == 12 )
      {
        v5 = *(_QWORD *)(v2 + 160);
        if ( (unsigned __int8)(*(_BYTE *)(v5 + 140) - 9) > 2u || (*(_BYTE *)(v5 + 177) & 4) == 0 )
          goto LABEL_10;
        sub_760760(v2, 6, v5, 0);
        v2 = *(_QWORD *)(v2 + 112);
        if ( !v2 )
          return;
      }
      else if ( (unsigned __int8)(v3 - 9) <= 2u
             && (v4 = *(_QWORD *)(*(_QWORD *)(v2 + 168) + 152LL)) != 0
             && (*(_BYTE *)(v4 + 29) & 0x20) == 0 )
      {
        sub_85B370(v4);
        v2 = *(_QWORD *)(v2 + 112);
        if ( !v2 )
          return;
      }
      else
      {
LABEL_10:
        v2 = *(_QWORD *)(v2 + 112);
        if ( !v2 )
          return;
      }
    }
  }
}
