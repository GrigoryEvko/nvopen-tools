// Function: sub_6E7080
// Address: 0x6e7080
//
__int64 __fastcall sub_6E7080(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 result; // rax

  sub_6E2E50(2, a1);
  sub_72BAF0(a1 + 144, a2, 5);
  v2 = *(_QWORD *)(a1 + 272);
  *(_BYTE *)(a1 + 17) = 2;
  *(_QWORD *)a1 = v2;
  *(_QWORD *)(a1 + 68) = *(_QWORD *)&dword_4F063F8;
  result = qword_4F063F0;
  *(_QWORD *)(a1 + 76) = qword_4F063F0;
  return result;
}
