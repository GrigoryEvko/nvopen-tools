// Function: sub_1699430
// Address: 0x1699430
//
__int64 __fastcall sub_1699430(_BYTE *a1, __int64 a2)
{
  __int64 result; // rax

  switch ( (*(_BYTE *)(a2 + 18) & 7) + 4 * (a1[18] & 7) )
  {
    case 0:
    case 2:
    case 3:
    case 0xB:
    case 0xF:
      sub_16986F0(a1, 0, 0, 0);
      result = 1;
      break;
    case 1:
    case 9:
    case 0xD:
      a1[18] = a1[18] & 0xF0 | 1;
      sub_16985E0((__int64)a1, a2);
      result = 0;
      break;
    case 4:
    case 5:
    case 6:
    case 7:
    case 8:
    case 0xA:
    case 0xC:
    case 0xE:
      result = 0;
      break;
  }
  return result;
}
