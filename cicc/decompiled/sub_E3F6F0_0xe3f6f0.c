// Function: sub_E3F6F0
// Address: 0xe3f6f0
//
__int64 __fastcall sub_E3F6F0(__int64 a1, char a2)
{
  __int64 result; // rax

  *(_QWORD *)(a1 + 16) = 0;
  result = a1;
  *(_OWORD *)a1 = 0;
  switch ( a2 )
  {
    case 0:
      *(_QWORD *)(a1 + 8) = 16;
      *(_QWORD *)a1 = "round.towardzero";
      *(_BYTE *)(a1 + 16) = 1;
      break;
    case 1:
      *(_QWORD *)(a1 + 8) = 15;
      *(_QWORD *)a1 = "round.tonearest";
      *(_BYTE *)(a1 + 16) = 1;
      break;
    case 2:
      *(_QWORD *)(a1 + 8) = 12;
      *(_QWORD *)a1 = "round.upward";
      *(_BYTE *)(a1 + 16) = 1;
      break;
    case 3:
      *(_QWORD *)(a1 + 8) = 14;
      *(_QWORD *)a1 = "round.downward";
      *(_BYTE *)(a1 + 16) = 1;
      break;
    case 4:
      *(_QWORD *)(a1 + 8) = 19;
      *(_QWORD *)a1 = "round.tonearestaway";
      *(_BYTE *)(a1 + 16) = 1;
      break;
    case 7:
      *(_QWORD *)(a1 + 8) = 13;
      *(_QWORD *)a1 = "round.dynamic";
      *(_BYTE *)(a1 + 16) = 1;
      break;
    default:
      return result;
  }
  return result;
}
