// Function: sub_22A6C20
// Address: 0x22a6c20
//
unsigned __int64 __fastcall sub_22A6C20(__int64 a1)
{
  unsigned __int64 result; // rax
  unsigned __int16 v2; // [rsp+1h] [rbp-3h]

  v2 = *(_WORD *)(a1 + 8);
  switch ( *(_DWORD *)(a1 + 12) )
  {
    case 1:
    case 2:
    case 4:
    case 5:
    case 6:
    case 7:
    case 9:
    case 0xA:
    case 0xB:
    case 0xC:
      result = v2 | ((unsigned __int64)(*(_DWORD *)(*(_QWORD *)(*(_QWORD *)a1 + 40LL) + 4LL) != 0) << 16);
      break;
    case 3:
    case 8:
    case 0x11:
    case 0x12:
      result = v2;
      break;
    default:
      BUG();
  }
  return result;
}
