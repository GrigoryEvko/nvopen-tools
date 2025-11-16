// Function: sub_1ACF050
// Address: 0x1acf050
//
__int64 __fastcall sub_1ACF050(__int64 a1)
{
  int v1; // eax
  __int64 v3; // rbx
  _QWORD *v4; // rdi

  v1 = *(unsigned __int8 *)(a1 + 16);
  if ( (unsigned __int8)v1 <= 3u || (unsigned int)(v1 - 9) <= 7 )
    return 0;
  v3 = *(_QWORD *)(a1 + 8);
  if ( !v3 )
    return 1;
  while ( 1 )
  {
    v4 = sub_1648700(v3);
    if ( *((_BYTE *)v4 + 16) > 0x10u || !(unsigned __int8)sub_1ACF050(v4) )
      break;
    v3 = *(_QWORD *)(v3 + 8);
    if ( !v3 )
      return 1;
  }
  return 0;
}
