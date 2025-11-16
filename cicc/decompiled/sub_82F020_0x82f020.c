// Function: sub_82F020
// Address: 0x82f020
//
__int64 __fastcall sub_82F020(__int64 *a1)
{
  __int64 v1; // rdi
  char i; // al
  char j; // al
  unsigned int v4; // r8d

  v1 = *a1;
  for ( i = *(_BYTE *)(v1 + 140); i == 12; i = *(_BYTE *)(v1 + 140) )
    v1 = *(_QWORD *)(v1 + 160);
  if ( i == 6 )
  {
    v1 = sub_8D46C0(v1);
    for ( j = *(_BYTE *)(v1 + 140); j == 12; j = *(_BYTE *)(v1 + 140) )
      v1 = *(_QWORD *)(v1 + 160);
  }
  else
  {
    j = *(_BYTE *)(v1 + 140);
  }
  v4 = 1;
  if ( (unsigned __int8)(j - 9) <= 2u && (v4 = 0, (*(_BYTE *)(v1 + 177) & 0xA0) == 0x20) )
    return ((*(_BYTE *)(*(_QWORD *)(v1 + 168) + 109LL) >> 5) ^ 1) & 1;
  else
    return v4;
}
