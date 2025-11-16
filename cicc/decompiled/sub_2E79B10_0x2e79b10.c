// Function: sub_2E79B10
// Address: 0x2e79b10
//
__int64 __fastcall sub_2E79B10(_DWORD *a1, __int64 a2)
{
  int v3; // esi
  __int64 result; // rax

  switch ( *a1 )
  {
    case 0:
      return 1LL << sub_AE4360(a2, 0);
    case 1:
    case 4:
      v3 = 64;
      goto LABEL_3;
    case 2:
    case 3:
    case 6:
      v3 = 32;
LABEL_3:
      result = 1LL << sub_AE3FE0(a2, v3);
      break;
    case 5:
      result = 1;
      break;
    default:
      BUG();
  }
  return result;
}
