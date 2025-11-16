// Function: sub_C0BF50
// Address: 0xc0bf50
//
__int64 __fastcall sub_C0BF50(__int64 a1)
{
  __int64 v1; // rax
  __int64 result; // rax
  __int64 v3; // kr00_8

  v3 = v1;
  result = *(unsigned int *)(a1 + 40);
  switch ( *(_DWORD *)(a1 + 40) )
  {
    case 0:
    case 2:
    case 3:
    case 9:
      *(_QWORD *)(a1 + 32) = 1;
      break;
    case 1:
    case 8:
      *(_QWORD *)(a1 + 32) = 4;
      break;
    case 4:
    case 5:
      *(_QWORD *)(a1 + 32) = 2;
      break;
    case 6:
    case 7:
      *(_QWORD *)(a1 + 32) = 0;
      break;
    default:
      result = v3;
      break;
  }
  return result;
}
