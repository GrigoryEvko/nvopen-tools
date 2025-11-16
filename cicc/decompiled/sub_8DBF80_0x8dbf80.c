// Function: sub_8DBF80
// Address: 0x8dbf80
//
__int64 __fastcall sub_8DBF80(__int64 a1)
{
  __int64 v1; // r12
  char i; // al

  for ( ; *(_BYTE *)(a1 + 140) == 12; a1 = *(_QWORD *)(a1 + 160) )
    ;
  v1 = 0;
  if ( !(unsigned int)sub_8DBE70(a1) && *(_BYTE *)(a1 + 140) == 8 )
  {
    do
    {
      ++v1;
      a1 = sub_8D4050(a1);
      for ( i = *(_BYTE *)(a1 + 140); i == 12; i = *(_BYTE *)(a1 + 140) )
        a1 = *(_QWORD *)(a1 + 160);
    }
    while ( i == 8 );
  }
  return v1;
}
