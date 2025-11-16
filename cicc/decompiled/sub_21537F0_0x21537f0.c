// Function: sub_21537F0
// Address: 0x21537f0
//
__int64 __fastcall sub_21537F0(__int64 a1, __int64 a2, __int64 a3, int a4)
{
  __int64 result; // rax

  switch ( a4 )
  {
    case 0:
      result = sub_16E7AB0(a3, *(_QWORD *)(a2 + 16));
      break;
    case 1:
      result = sub_38E2490(*(_QWORD *)(a2 + 24), a3, *(_QWORD *)(a1 + 240));
      break;
    default:
      JUMPOUT(0x2153AA2);
  }
  return result;
}
