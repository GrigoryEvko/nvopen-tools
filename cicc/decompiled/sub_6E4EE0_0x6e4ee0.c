// Function: sub_6E4EE0
// Address: 0x6e4ee0
//
__int64 __fastcall sub_6E4EE0(__int64 a1, __int64 a2)
{
  __int64 result; // rax

  sub_6E4BC0(a1, a2);
  *(_QWORD *)(a1 + 88) = *(_QWORD *)(a2 + 88);
  result = *(_QWORD *)(a2 + 96);
  *(_QWORD *)(a1 + 96) = result;
  return result;
}
