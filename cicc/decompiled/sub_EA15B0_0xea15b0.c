// Function: sub_EA15B0
// Address: 0xea15b0
//
__int64 __fastcall sub_EA15B0(__int64 a1, int a2)
{
  __int64 result; // rax

  switch ( a2 )
  {
    case 0:
    case 1:
    case 2:
    case 3:
      goto LABEL_3;
    case 5:
      result = *(_WORD *)(a1 + 12) & 0xFFF8;
      *(_WORD *)(a1 + 12) = *(_WORD *)(a1 + 12) & 0xFFF8 | 4;
      return result;
    case 6:
      result = *(_WORD *)(a1 + 12) & 0xFFF8;
      *(_WORD *)(a1 + 12) = *(_WORD *)(a1 + 12) & 0xFFF8 | 5;
      return result;
    case 10:
      result = *(_WORD *)(a1 + 12) & 0xFFF8;
      *(_WORD *)(a1 + 12) = *(_WORD *)(a1 + 12) & 0xFFF8 | 6;
      return result;
    case 13:
      LOWORD(a2) = 7;
LABEL_3:
      result = *(_WORD *)(a1 + 12) & 0xFFF8;
      *(_WORD *)(a1 + 12) = *(_WORD *)(a1 + 12) & 0xFFF8 | a2;
      return result;
    default:
      BUG();
  }
}
