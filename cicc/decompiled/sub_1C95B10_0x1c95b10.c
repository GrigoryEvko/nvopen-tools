// Function: sub_1C95B10
// Address: 0x1c95b10
//
__int64 __fastcall sub_1C95B10(__int64 a1)
{
  __int64 v1; // rbx
  _QWORD *v2; // r12
  unsigned __int8 v3; // al
  __int64 v4; // rax

  v1 = *(_QWORD *)(a1 + 8);
  if ( !v1 )
    return 1;
  while ( 1 )
  {
    v2 = sub_1648700(v1);
    v3 = *((_BYTE *)v2 + 16);
    if ( v3 <= 0x17u )
      break;
    if ( v3 != 54 )
    {
      if ( v3 == 55 )
      {
        v4 = *(v2 - 6);
        if ( a1 == v4 && v4 )
          return 0;
      }
      else if ( v3 != 56 || !(unsigned __int8)sub_15FA290((__int64)v2) || !(unsigned __int8)sub_1C95B10(v2) )
      {
        return 0;
      }
    }
    v1 = *(_QWORD *)(v1 + 8);
    if ( !v1 )
      return 1;
  }
  return 0;
}
