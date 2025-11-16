// Function: sub_5E4E60
// Address: 0x5e4e60
//
void __fastcall sub_5E4E60(__int64 a1, __int64 *a2, __int16 a3)
{
  __int64 v3; // rax
  __int64 i; // rax
  char v5; // al
  __int64 j; // rax

  if ( (a2[1] & 0x180) != 0 )
  {
    if ( (unsigned int)sub_8DBE70(*(_QWORD *)(a1 + 40)) )
    {
      *(_BYTE *)(a1 + 96) |= 0x10u;
      for ( i = *a2; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
        ;
LABEL_10:
      *(_BYTE *)(*(_QWORD *)(*(_QWORD *)i + 96LL) + 180LL) |= 2u;
    }
  }
  else if ( (a2[1] & 0x40) != 0 )
  {
    v3 = *(_QWORD *)(*(_QWORD *)(a2[4] + 88) + 168LL);
    while ( 1 )
    {
      v3 = *(_QWORD *)(v3 + 8);
      if ( !v3 )
        break;
      if ( *(_WORD *)(v3 + 98) == a3 )
      {
        v5 = *(_BYTE *)(v3 + 96) & 0x10 | *(_BYTE *)(a1 + 96) & 0xEF;
        *(_BYTE *)(a1 + 96) = v5;
        if ( (v5 & 0x10) != 0 )
          goto LABEL_12;
        for ( j = *(_QWORD *)(a1 + 40); *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
          ;
        if ( (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)j + 96LL) + 180LL) & 2) != 0 )
        {
LABEL_12:
          i = *a2;
          if ( *(_BYTE *)(*a2 + 140) != 12 )
            goto LABEL_10;
          do
            i = *(_QWORD *)(i + 160);
          while ( *(_BYTE *)(i + 140) == 12 );
          *(_BYTE *)(*(_QWORD *)(*(_QWORD *)i + 96LL) + 180LL) |= 2u;
        }
        return;
      }
    }
  }
}
