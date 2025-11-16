// Function: sub_2CDDA20
// Address: 0x2cdda20
//
__int64 __fastcall sub_2CDDA20(__int64 a1)
{
  __int64 v1; // rbx
  _BYTE *v2; // r13
  char v3; // al
  __int64 v4; // rax

  v1 = *(_QWORD *)(a1 + 16);
  if ( !v1 )
    return 1;
  while ( 1 )
  {
    v2 = *(_BYTE **)(v1 + 24);
    v3 = *v2;
    if ( *v2 <= 0x1Cu )
      break;
    if ( v3 != 61 )
    {
      if ( v3 == 62 )
      {
        v4 = *((_QWORD *)v2 - 8);
        if ( a1 == v4 && v4 )
          return 0;
      }
      else if ( v3 != 63 || !(unsigned __int8)sub_B4DD90(*(_QWORD *)(v1 + 24)) || !(unsigned __int8)sub_2CDDA20(v2) )
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
