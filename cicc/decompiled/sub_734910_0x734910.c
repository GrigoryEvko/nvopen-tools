// Function: sub_734910
// Address: 0x734910
//
void __fastcall sub_734910(__int64 a1)
{
  char v1; // al
  __int64 i; // rbx

  v1 = *(_BYTE *)(a1 + 173);
  if ( v1 == 10 )
  {
LABEL_8:
    for ( i = *(_QWORD *)(a1 + 176); i; i = *(_QWORD *)(i + 120) )
      sub_734910(i);
  }
  else
  {
    while ( v1 != 9 )
    {
      if ( v1 != 11 )
        return;
      a1 = *(_QWORD *)(a1 + 176);
      v1 = *(_BYTE *)(a1 + 173);
      if ( v1 == 10 )
        goto LABEL_8;
    }
    sub_734850(*(_QWORD *)(a1 + 176));
  }
}
