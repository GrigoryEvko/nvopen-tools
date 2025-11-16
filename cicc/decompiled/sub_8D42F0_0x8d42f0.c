// Function: sub_8D42F0
// Address: 0x8d42f0
//
_BOOL8 __fastcall sub_8D42F0(__int64 a1)
{
  char v1; // al
  __int64 i; // rbx
  char v4; // al
  __int64 j; // rdi
  __int64 v6; // rax
  __int64 v7; // r12
  __int64 v8; // rax
  __int64 **v9; // rbx

  v1 = *(_BYTE *)(a1 + 140);
  for ( i = a1; v1 == 12; v1 = *(_BYTE *)(i + 140) )
    i = *(_QWORD *)(i + 160);
  if ( (unsigned __int8)(v1 - 2) <= 3u )
    return 1;
  if ( v1 == 6 )
  {
    v4 = *(_BYTE *)(i + 168);
    return (v4 & 1) == 0 || (v4 & 2) == 0;
  }
  if ( (unsigned __int8)(v1 - 19) <= 1u || v1 == 13 )
    return 1;
  if ( (unsigned __int8)(v1 - 9) > 2u || !(unsigned int)sub_8D4160(i) )
    return 0;
  for ( j = *(_QWORD *)(i + 160); ; j = *(_QWORD *)(v7 + 112) )
  {
    v6 = sub_72FD90(j, 7);
    v7 = v6;
    if ( !v6 )
      break;
    if ( (*(_BYTE *)(v6 + 144) & 0x20) != 0 )
      return 0;
    if ( (*(_BYTE *)(v6 + 88) & 3) != 0 )
      return 0;
    v8 = sub_8D4130(*(_QWORD *)(v6 + 120));
    if ( !(unsigned int)sub_8D42F0(v8, 7) )
      return 0;
  }
  v9 = **(__int64 ****)(i + 168);
  if ( v9 )
  {
    while ( ((_BYTE)v9[12] & 1) == 0 || !*((_BYTE *)v9[14] + 25) && (unsigned int)sub_8D42F0(v9[5], 7) )
    {
      v9 = (__int64 **)*v9;
      if ( !v9 )
        return 1;
    }
    return 0;
  }
  return 1;
}
