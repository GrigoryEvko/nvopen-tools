// Function: sub_123DC50
// Address: 0x123dc50
//
__int64 __fastcall sub_123DC50(__int64 a1, _QWORD *a2)
{
  __int64 v2; // r13
  int i; // eax
  unsigned __int64 v5; // rsi
  const char *v6; // [rsp+0h] [rbp-50h] BYREF
  char v7; // [rsp+20h] [rbp-30h]
  char v8; // [rsp+21h] [rbp-2Fh]

  v2 = a1 + 176;
  *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
  if ( (unsigned __int8)sub_120AFE0(a1, 16, "expected ':' here")
    || (unsigned __int8)sub_120AFE0(a1, 12, "expected '(' in typeIdInfo") )
  {
    return 1;
  }
  for ( i = *(_DWORD *)(a1 + 240); ; *(_DWORD *)(a1 + 240) = i )
  {
    switch ( i )
    {
      case 453:
        if ( (unsigned __int8)sub_123CE30(a1, a2) )
          return 1;
        break;
      case 454:
        if ( (unsigned __int8)sub_123D2F0(a1, 454, a2 + 3) )
          return 1;
        break;
      case 455:
        if ( (unsigned __int8)sub_123D2F0(a1, 455, a2 + 6) )
          return 1;
        break;
      case 456:
        if ( (unsigned __int8)sub_123D6E0(a1, 456, (__int64)(a2 + 9)) )
          return 1;
        break;
      case 457:
        if ( (unsigned __int8)sub_123D6E0(a1, 457, (__int64)(a2 + 12)) )
          return 1;
        break;
      default:
        v8 = 1;
        v5 = *(_QWORD *)(a1 + 232);
        v7 = 3;
        v6 = "invalid typeIdInfo list type";
        sub_11FD800(v2, v5, (__int64)&v6, 1);
        return 1;
    }
    if ( *(_DWORD *)(a1 + 240) != 4 )
      break;
    i = sub_1205200(v2);
  }
  return sub_120AFE0(a1, 13, "expected ')' in typeIdInfo");
}
