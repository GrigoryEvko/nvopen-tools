// Function: sub_1B42630
// Address: 0x1b42630
//
__int64 __fastcall sub_1B42630(__int64 a1)
{
  __int64 v1; // rbx
  char v2; // al

  if ( *(_BYTE *)(a1 + 16) != 78 )
    return 0;
  if ( (unsigned __int8)sub_1C30710(a1) )
    return 1;
  v1 = *(_QWORD *)(a1 - 24);
  v2 = *(_BYTE *)(v1 + 16);
  if ( v2 == 20 )
    return *(unsigned __int8 *)(v1 + 96);
  if ( v2 )
    return 0;
  if ( !(unsigned __int8)sub_1560180(v1 + 112, 36)
    && !(unsigned __int8)sub_1560180(v1 + 112, 36)
    && !(unsigned __int8)sub_1560180(v1 + 112, 37) )
  {
    return 1;
  }
  return sub_1C301F0(*(unsigned int *)(v1 + 36));
}
