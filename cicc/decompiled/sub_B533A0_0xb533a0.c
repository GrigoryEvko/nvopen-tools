// Function: sub_B533A0
// Address: 0xb533a0
//
bool __fastcall sub_B533A0(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  int v5; // ebx
  unsigned int v6; // eax
  bool result; // al

  v5 = a3;
  if ( *a1 == sub_C33340(a1, a2, a3, a4, a5) )
    v6 = sub_C3E510(a1, a2);
  else
    v6 = sub_C37950(a1, a2);
  switch ( v5 )
  {
    case 0:
      return 0;
    case 1:
      return v6 == 1;
    case 2:
      return v6 == 2;
    case 3:
      --v6;
      goto LABEL_6;
    case 4:
      return v6 == 0;
    case 5:
LABEL_6:
      result = v6 <= 1;
      break;
    case 6:
      result = (v6 & 0xFFFFFFFD) == 0;
      break;
    case 7:
      result = v6 != 3;
      break;
    case 8:
      result = v6 == 3;
      break;
    case 9:
      result = (v6 & 0xFFFFFFFD) == 1;
      break;
    case 10:
      result = v6 - 2 <= 1;
      break;
    case 11:
      result = v6 != 0;
      break;
    case 12:
      result = v6 == 3 || v6 == 0;
      break;
    case 13:
      result = v6 != 2;
      break;
    case 14:
      result = v6 != 1;
      break;
    case 15:
      result = 1;
      break;
    default:
      BUG();
  }
  return result;
}
