// Function: sub_E89910
// Address: 0xe89910
//
__int64 __fastcall sub_E89910(__int64 a1, _DWORD *a2, char a3, char a4)
{
  __int64 result; // rax

  *(_BYTE *)(a1 + 912) = a3;
  *(_QWORD *)(a1 + 920) = a2;
  *(_WORD *)(a1 + 8) = 1;
  *(_BYTE *)(a1 + 10) = 0;
  *(_QWORD *)(a1 + 12) = 0;
  *(_QWORD *)(a1 + 464) = 0;
  *(_QWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 200) = 0;
  *(_QWORD *)(a1 + 208) = 0;
  *(_QWORD *)(a1 + 216) = 0;
  *(_QWORD *)(a1 + 224) = 0;
  switch ( *a2 )
  {
    case 0:
      result = sub_E85910(a1, a2 + 6);
      break;
    case 1:
      result = sub_E86C20(a1, (__int64)(a2 + 6), a4);
      break;
    case 2:
      result = sub_E87EC0(a1);
      break;
    case 3:
      result = sub_E88000(a1, (__int64)(a2 + 6));
      break;
    case 4:
      result = (__int64)sub_E88810(a1);
      break;
    case 5:
      result = sub_E88840((_QWORD *)a1);
      break;
    case 6:
      result = sub_E89240(a1);
      break;
    case 7:
      result = sub_E898D0(a1);
      break;
    default:
      result = 1;
      break;
  }
  return result;
}
