// Function: sub_622920
// Address: 0x622920
//
__int64 __fastcall sub_622920(__int64 a1, _QWORD *a2, _DWORD *a3)
{
  __int64 result; // rax

  switch ( (char)a1 )
  {
    case 0:
    case 1:
    case 2:
      *a2 = 1;
      *a3 = 1;
      result = 1;
      break;
    case 3:
    case 4:
      result = unk_4F06B28;
      *a2 = unk_4F06B30;
      *a3 = result;
      break;
    case 5:
    case 6:
      result = unk_4F06B18;
      *a2 = unk_4F06B20;
      *a3 = result;
      break;
    case 7:
    case 8:
      result = unk_4F06B08;
      *a2 = unk_4F06B10;
      *a3 = result;
      break;
    case 9:
    case 10:
      result = unk_4F06AF8;
      *a2 = unk_4F06B00;
      *a3 = result;
      break;
    case 11:
    case 12:
      result = unk_4F06AE8;
      *a2 = unk_4F06AF0;
      *a3 = result;
      break;
    default:
      sub_721090(a1);
  }
  return result;
}
