// Function: sub_21E54B0
// Address: 0x21e54b0
//
__int64 __fastcall sub_21E54B0(
        __int64 a1,
        __int64 a2,
        __m128i a3,
        double a4,
        __m128i a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __int64 a9)
{
  __int64 v9; // rax
  _QWORD *v10; // rdx
  __int64 result; // rax

  v9 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL) + 88LL);
  v10 = *(_QWORD **)(v9 + 24);
  if ( *(_DWORD *)(v9 + 32) > 0x40u )
    v10 = (_QWORD *)*v10;
  if ( (unsigned int)v10 > 0xFC5 )
  {
    switch ( (_DWORD)v10 )
    {
      case 0x1045:
        return sub_21E3090(a1, a2, a3, a4, a5, (__int64)v10, a2, a8, a9);
      case 0x1052:
        return sub_21E5090(a1, a2, a3, a4, a5);
      case 0x1044:
        return sub_21E3260(a1, a2, (__int64)v10, a2, a8, a9);
    }
    return 0;
  }
  if ( (unsigned int)v10 <= 0xF81 )
  {
    if ( (_DWORD)v10 == 3753 )
      return sub_21E1830(a1, 179 - ((unsigned int)(*(_BYTE *)(*(_QWORD *)(a1 + 16) + 936LL) == 0) - 1), a2, a3, a4, a5);
    return 0;
  }
  switch ( (int)v10 )
  {
    case 3970:
      result = sub_21DFBF0(a1, 0, 544 - ((*(_BYTE *)(*(_QWORD *)(a1 + 16) + 936LL) == 0) - 1), a2, a3, a4, a5);
      break;
    case 3971:
      result = sub_21DFBF0(a1, 1, 546 - ((*(_BYTE *)(*(_QWORD *)(a1 + 16) + 936LL) == 0) - 1), a2, a3, a4, a5);
      break;
    case 3984:
      result = sub_21DFBF0(a1, 0, 558 - ((*(_BYTE *)(*(_QWORD *)(a1 + 16) + 936LL) == 0) - 1), a2, a3, a4, a5);
      break;
    case 3985:
      result = sub_21DFBF0(a1, 1, 560 - ((*(_BYTE *)(*(_QWORD *)(a1 + 16) + 936LL) == 0) - 1), a2, a3, a4, a5);
      break;
    case 3994:
      result = sub_21DFBF0(a1, 0, 572 - ((*(_BYTE *)(*(_QWORD *)(a1 + 16) + 936LL) == 0) - 1), a2, a3, a4, a5);
      break;
    case 3995:
      result = sub_21DFBF0(a1, 1, 574 - ((*(_BYTE *)(*(_QWORD *)(a1 + 16) + 936LL) == 0) - 1), a2, a3, a4, a5);
      break;
    case 4013:
      result = sub_21E1830(
                 a1,
                 585 - ((unsigned int)(*(_BYTE *)(*(_QWORD *)(a1 + 16) + 936LL) == 0) - 1),
                 a2,
                 a3,
                 a4,
                 a5);
      break;
    case 4021:
      result = sub_21E1830(
                 a1,
                 594 - ((unsigned int)(*(_BYTE *)(*(_QWORD *)(a1 + 16) + 936LL) == 0) - 1),
                 a2,
                 a3,
                 a4,
                 a5);
      break;
    case 4029:
      result = sub_21E1830(
                 a1,
                 603 - ((unsigned int)(*(_BYTE *)(*(_QWORD *)(a1 + 16) + 936LL) == 0) - 1),
                 a2,
                 a3,
                 a4,
                 a5);
      break;
    case 4037:
      result = sub_21E1830(
                 a1,
                 610 - ((unsigned int)(*(_BYTE *)(*(_QWORD *)(a1 + 16) + 936LL) == 0) - 1),
                 a2,
                 a3,
                 a4,
                 a5);
      break;
    default:
      return 0;
  }
  return result;
}
