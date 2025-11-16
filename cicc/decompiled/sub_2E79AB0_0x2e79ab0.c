// Function: sub_2E79AB0
// Address: 0x2e79ab0
//
__int64 __fastcall sub_2E79AB0(_DWORD *a1, __int64 a2)
{
  __int64 result; // rax

  switch ( *a1 )
  {
    case 0:
      result = sub_AE4380(a2, 0);
      break;
    case 1:
    case 4:
      result = 8;
      break;
    case 2:
    case 3:
    case 6:
      result = 4;
      break;
    case 5:
      result = 0;
      break;
    default:
      BUG();
  }
  return result;
}
