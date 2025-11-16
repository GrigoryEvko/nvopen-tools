// Function: sub_878D40
// Address: 0x878d40
//
__int64 __fastcall sub_878D40(__int64 a1)
{
  __int64 result; // rax

  result = qword_4F5FFD8;
  qword_4F5FFD8 = a1;
  *(_QWORD *)(a1 + 24) = result;
  return result;
}
