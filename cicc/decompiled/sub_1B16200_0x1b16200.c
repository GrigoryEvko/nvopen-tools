// Function: sub_1B16200
// Address: 0x1b16200
//
__int64 __fastcall sub_1B16200(int a1, __int64 a2)
{
  __int64 result; // rax

  switch ( a1 )
  {
    case 0:
    case 6:
    case 8:
      result = sub_15A10B0(a2, 1.0);
      break;
    case 1:
    case 3:
    case 5:
      result = sub_15A0680(a2, 0, 0);
      break;
    case 2:
      result = sub_15A0680(a2, 1, 0);
      break;
    case 4:
      result = sub_15A0680(a2, -1, 1u);
      break;
    case 7:
      result = sub_15A10B0(a2, 0.0);
      break;
  }
  return result;
}
