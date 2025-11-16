// Function: sub_39CABF0
// Address: 0x39cabf0
//
__int64 __fastcall sub_39CABF0(__int64 *a1, __int64 a2, char a3)
{
  __int64 result; // rax

  result = sub_39C9FF0(a1, a2, a3);
  *(_QWORD *)(a2 + 16) = result;
  return result;
}
