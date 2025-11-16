// Function: sub_8D31A0
// Address: 0x8d31a0
//
__int64 __fastcall sub_8D31A0(__int64 a1, _QWORD *a2)
{
  char v2; // al
  unsigned int v3; // r8d
  char v5; // al
  __int64 i; // rax
  char v7; // dl

  while ( 1 )
  {
    v2 = *(_BYTE *)(a1 + 140);
    if ( v2 != 12 )
      break;
    a1 = *(_QWORD *)(a1 + 160);
  }
  v3 = 0;
  if ( v2 != 6 )
    return v3;
  v5 = *(_BYTE *)(a1 + 168);
  if ( (v5 & 1) == 0 || (v5 & 2) != 0 )
    return v3;
  for ( i = *(_QWORD *)(a1 + 160); ; i = *(_QWORD *)(i + 160) )
  {
    v7 = *(_BYTE *)(i + 140);
    if ( v7 != 12 )
      break;
    if ( (*(_BYTE *)(i + 185) & 1) != 0 )
    {
      i = *(_QWORD *)(i + 160);
      goto LABEL_14;
    }
    if ( (*(_BYTE *)(i + 186) & 0x18) != 0 )
      goto LABEL_14;
LABEL_10:
    ;
  }
  if ( v7 == 8 )
    goto LABEL_10;
  v3 = 0;
  if ( v7 != 14 )
    return v3;
LABEL_14:
  v3 = 1;
  if ( a2 )
    *a2 = i;
  return v3;
}
