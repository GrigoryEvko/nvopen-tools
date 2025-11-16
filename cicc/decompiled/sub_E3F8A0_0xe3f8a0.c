// Function: sub_E3F8A0
// Address: 0xe3f8a0
//
__int64 __fastcall sub_E3F8A0(__int64 a1, char a2)
{
  __int64 result; // rax

  *(_QWORD *)(a1 + 16) = 0;
  result = a1;
  *(_OWORD *)a1 = 0;
  switch ( a2 )
  {
    case 1:
      *(_QWORD *)(a1 + 8) = 16;
      *(_QWORD *)a1 = "fpexcept.maytrap";
      *(_BYTE *)(a1 + 16) = 1;
      break;
    case 2:
      *(_QWORD *)(a1 + 8) = 15;
      *(_QWORD *)a1 = "fpexcept.strict";
      *(_BYTE *)(a1 + 16) = 1;
      break;
    case 0:
      *(_QWORD *)(a1 + 8) = 15;
      *(_QWORD *)a1 = "fpexcept.ignore";
      *(_BYTE *)(a1 + 16) = 1;
      break;
  }
  return result;
}
