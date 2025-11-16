// Function: sub_E71E20
// Address: 0xe71e20
//
__int64 __fastcall sub_E71E20(__int64 a1, char a2)
{
  __int64 result; // rax

  switch ( a2 & 0xF )
  {
    case 0:
    case 8:
      result = *(unsigned int *)(*(_QWORD *)(a1 + 152) + 8LL);
      break;
    case 2:
    case 0xA:
      result = 2;
      break;
    case 3:
    case 0xB:
      result = 4;
      break;
    case 4:
    case 0xC:
      result = 8;
      break;
    default:
      BUG();
  }
  return result;
}
