// Function: sub_8D4290
// Address: 0x8d4290
//
__int64 __fastcall sub_8D4290(__int64 a1)
{
  __int64 v1; // rdi
  char i; // al

  v1 = sub_8D4130(a1);
  for ( i = *(_BYTE *)(v1 + 140); i == 12; i = *(_BYTE *)(v1 + 140) )
    v1 = *(_QWORD *)(v1 + 160);
  if ( i == 14 || !i || (unsigned __int8)(i - 9) <= 2u && (*(_BYTE *)(v1 + 177) & 0x20) != 0 )
    return 1;
  else
    return sub_8D4160(v1);
}
