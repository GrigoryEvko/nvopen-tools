// Function: sub_EA3BB0
// Address: 0xea3bb0
//
__int64 __fastcall sub_EA3BB0(__int64 a1, int a2, _DWORD *a3, char a4)
{
  __int64 result; // rax

  switch ( a2 )
  {
    case 12:
      *a3 = 0;
      result = 4;
      break;
    case 13:
      *a3 = 18;
      result = 4;
      break;
    case 15:
      *a3 = 2;
      result = 6;
      break;
    case 24:
      *a3 = 11;
      result = 6;
      break;
    case 29:
      *a3 = 3;
      result = 3;
      break;
    case 30:
      *a3 = 13;
      result = 5;
      break;
    case 31:
      *a3 = 7;
      result = 1;
      break;
    case 32:
      *a3 = 19;
      result = 5;
      break;
    case 33:
      *a3 = 1;
      result = 5;
      break;
    case 34:
      *a3 = 6;
      result = 2;
      break;
    case 35:
      if ( *(_QWORD *)(a1 + 56) != 1 || (result = 0, **(_BYTE **)(a1 + 48) != 64) )
      {
        *a3 = 14;
        result = 5;
      }
      break;
    case 36:
    case 42:
      *a3 = 12;
      result = 3;
      break;
    case 37:
      *a3 = 10;
      result = 6;
      break;
    case 39:
      *a3 = 8;
      result = 3;
      break;
    case 40:
      *a3 = 9;
      result = 3;
      break;
    case 41:
      *a3 = 15;
      result = 6;
      break;
    case 43:
      *a3 = 4;
      result = 3;
      break;
    case 44:
      *a3 = 5;
      result = 3;
      break;
    case 45:
      *a3 = 16 - ((a4 == 0) - 1);
      result = 6;
      break;
    default:
      result = 0;
      break;
  }
  return result;
}
