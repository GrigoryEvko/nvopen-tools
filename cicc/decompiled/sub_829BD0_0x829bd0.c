// Function: sub_829BD0
// Address: 0x829bd0
//
__m128i *__fastcall sub_829BD0(__int64 a1)
{
  char v1; // al
  __int64 i; // rdi

  if ( !a1 )
    return 0;
  v1 = *(_BYTE *)(a1 + 80);
  if ( v1 == 16 )
  {
    a1 = **(_QWORD **)(a1 + 88);
    v1 = *(_BYTE *)(a1 + 80);
  }
  if ( v1 == 24 )
  {
    a1 = *(_QWORD *)(a1 + 88);
    v1 = *(_BYTE *)(a1 + 80);
  }
  if ( v1 == 20 )
    return 0;
  for ( i = *(_QWORD *)(*(_QWORD *)(a1 + 88) + 152LL); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  return sub_73D790(i);
}
