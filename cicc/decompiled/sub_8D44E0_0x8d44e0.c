// Function: sub_8D44E0
// Address: 0x8d44e0
//
_BOOL8 __fastcall sub_8D44E0(__int64 a1, __int64 a2)
{
  char v4; // al
  __int64 v5; // r12
  char v6; // al
  char i; // dl
  __int64 j; // rdi

  while ( 1 )
  {
    v4 = *(_BYTE *)(a1 + 173);
    if ( v4 == 13 )
    {
      a1 = *(_QWORD *)(a1 + 120);
      v4 = *(_BYTE *)(a1 + 173);
    }
    if ( v4 == 11 )
      a1 = *(_QWORD *)(a1 + 176);
    v5 = *(_QWORD *)(a1 + 128);
    v6 = *(_BYTE *)(v5 + 140);
    if ( v6 != 12 )
      goto LABEL_9;
    do
    {
      v5 = *(_QWORD *)(v5 + 160);
      v6 = *(_BYTE *)(v5 + 140);
    }
    while ( v6 == 12 );
    for ( i = *(_BYTE *)(a2 + 140); i == 12; i = *(_BYTE *)(a2 + 140) )
    {
      a2 = *(_QWORD *)(a2 + 160);
LABEL_9:
      ;
    }
    if ( v5 == a2 )
      return 1;
    if ( i != 8 )
      return v6 != 8;
    if ( v6 != 8 )
      return 0;
    if ( *(_BYTE *)(a1 + 173) == 2 )
    {
      for ( j = *(_QWORD *)(a2 + 160); *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
        ;
      if ( sub_8D29E0(j) )
        return 1;
    }
    if ( *(_QWORD *)(a2 + 176) != *(_QWORD *)(v5 + 176) )
      return 0;
    a1 = *(_QWORD *)(a1 + 176);
    a2 = *(_QWORD *)(a2 + 160);
  }
}
