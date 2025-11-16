// Function: sub_12A2130
// Address: 0x12a2130
//
__int64 __fastcall sub_12A2130(__int64 *a1, __int64 a2)
{
  char v2; // al
  __int64 result; // rax

  v2 = *(_BYTE *)(a2 + 140);
  if ( v2 == 12 )
  {
    if ( *(_BYTE *)(a2 + 184) != 8 || (a2 = *(_QWORD *)(a2 + 160), v2 = *(_BYTE *)(a2 + 140), v2 == 12) )
    {
      if ( *(_QWORD *)(a2 + 8) )
        return sub_12A1A70((__int64)a1, a2);
      if ( (*(_BYTE *)(a2 + 185) & 0x7F) != 0 )
        return sub_12A1990((__int64)a1, a2);
      do
      {
        a2 = *(_QWORD *)(a2 + 160);
        v2 = *(_BYTE *)(a2 + 140);
      }
      while ( v2 == 12 );
    }
  }
  switch ( v2 )
  {
    case 1:
      result = sub_15A59B0(a1 + 2, "void", 4);
      break;
    case 2:
    case 3:
      result = sub_12A0120(a1, a2);
      break;
    case 6:
      if ( (*(_BYTE *)(a2 + 168) & 1) != 0 )
        result = sub_12A1BF0((__int64)a1, a2);
      else
        result = sub_12A1B40((__int64)a1, a2);
      break;
    case 7:
      result = sub_12A1F40((__int64)a1, a2);
      break;
    case 8:
      result = sub_12A1C70((__int64)a1, a2);
      break;
    case 10:
    case 11:
      result = sub_12A0F50((__int64)a1, a2);
      break;
    case 15:
      result = sub_12A1DF0((__int64)a1, a2);
      break;
    default:
      sub_127B550("unhandled type in debug info gen!", (_DWORD *)(a2 + 64), 1);
  }
  return result;
}
