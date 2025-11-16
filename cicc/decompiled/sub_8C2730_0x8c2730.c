// Function: sub_8C2730
// Address: 0x8c2730
//
_QWORD *__fastcall sub_8C2730(__int64 a1, __int64 a2)
{
  _QWORD *result; // rax

  switch ( *(_BYTE *)(a1 + 80) )
  {
    case 4:
    case 5:
      result = sub_8C2140(a1, *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 96) + 80LL) + 176LL) + 88LL), a2);
      break;
    case 6:
      result = sub_8C2140(a1, *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 96) + 32LL) + 176LL) + 88LL), a2);
      break;
    case 9:
    case 0xA:
      result = sub_8C2140(a1, *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 96) + 56LL) + 176LL) + 88LL), a2);
      break;
    case 0x13:
    case 0x14:
    case 0x15:
    case 0x16:
      result = sub_8C2140(a1, *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 88) + 176LL) + 88LL), a2);
      break;
    default:
      BUG();
  }
  return result;
}
