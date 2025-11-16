// Function: sub_3719220
// Address: 0x3719220
//
__int64 __fastcall sub_3719220(__int64 a1, __int64 a2, __int64 a3, int a4)
{
  __int64 v5; // rdi
  __int64 result; // rax

  v5 = a1 + 8;
  *(_QWORD *)(v5 - 8) = &unk_4A352E0;
  result = sub_1255710(v5, a2, a3, a4);
  *(_QWORD *)(a1 + 56) = 0;
  return result;
}
