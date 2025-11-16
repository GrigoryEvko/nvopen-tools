// Function: sub_734990
// Address: 0x734990
//
void __fastcall sub_734990(__int64 a1)
{
  unsigned __int8 v1; // al
  __int64 i; // rbx

  while ( 1 )
  {
    v1 = *(_BYTE *)(a1 + 24);
    if ( v1 <= 6u )
      break;
    if ( v1 != 10 )
      return;
    a1 = *(_QWORD *)(a1 + 56);
  }
  if ( v1 > 4u )
  {
    sub_734850(*(_QWORD *)(a1 + 56));
  }
  else if ( v1 == 1 )
  {
    for ( i = *(_QWORD *)(a1 + 72); i; i = *(_QWORD *)(i + 16) )
      sub_734990(i);
  }
}
