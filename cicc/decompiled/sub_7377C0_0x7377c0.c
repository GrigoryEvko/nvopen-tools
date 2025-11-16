// Function: sub_7377C0
// Address: 0x7377c0
//
void __fastcall sub_7377C0(char a1, _DWORD *a2, _DWORD *a3)
{
  *a2 = 0;
  *a3 = 0;
  switch ( a1 )
  {
    case 15:
    case 18:
    case 19:
    case 20:
    case 21:
    case 22:
    case 23:
    case 24:
    case 25:
    case 28:
    case 29:
      *a3 = 1;
      break;
    case 26:
    case 27:
    case 40:
    case 42:
    case 43:
      *a2 = 1;
      break;
    default:
      return;
  }
}
