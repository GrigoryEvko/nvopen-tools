// Function: sub_DD3B30
// Address: 0xdd3b30
//
unsigned __int64 __fastcall sub_DD3B30(__int64 *a1, __int64 a2, __int64 **a3, __int64 a4, __int64 a5)
{
  unsigned __int16 v6; // si
  unsigned __int64 result; // rax
  __int64 v9; // rax

  v6 = *(_WORD *)(a2 + 24);
  switch ( v6 )
  {
    case 0u:
    case 1u:
    case 0xFu:
      result = a2;
      break;
    case 2u:
    case 3u:
    case 4u:
    case 0xEu:
      v9 = sub_D95540(a2);
      result = (unsigned __int64)sub_DD3AD0((__int64)a1, *(_WORD *)(a2 + 24), **a3, v9);
      break;
    case 5u:
      result = (unsigned __int64)sub_DC7EB0(a1, (__int64)a3, *(_WORD *)(a2 + 28) & 7, 0);
      break;
    case 6u:
      result = (unsigned __int64)sub_DC8BD0(a1, (__int64)a3, *(_WORD *)(a2 + 28) & 7, 0);
      break;
    case 7u:
      result = sub_DCB270((__int64)a1, **a3, (*a3)[1]);
      break;
    case 8u:
      result = (unsigned __int64)sub_DBFF60(
                                   (__int64)a1,
                                   (unsigned int *)a3,
                                   *(_QWORD *)(a2 + 48),
                                   *(_WORD *)(a2 + 28) & 7);
      break;
    case 9u:
    case 0xAu:
    case 0xBu:
    case 0xCu:
      result = sub_DCD310(a1, v6, (__int64)a3, a4, a5);
      break;
    case 0xDu:
      result = (unsigned __int64)sub_DCEA30(a1, 13, (__int64)a3, a4, a5);
      break;
    default:
      BUG();
  }
  return result;
}
