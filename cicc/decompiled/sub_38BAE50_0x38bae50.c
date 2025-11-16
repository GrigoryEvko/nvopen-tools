// Function: sub_38BAE50
// Address: 0x38bae50
//
__int64 __fastcall sub_38BAE50(__int64 a1, _QWORD *a2)
{
  __int64 v2; // r13
  __int64 v3; // r15
  __int64 v4; // r14
  __int64 v5; // rax

  v2 = a2[1];
  v3 = sub_38CF310(a1, 0, v2, 0);
  v4 = sub_38BFA60(v2, 1);
  (*(void (__fastcall **)(_QWORD *, __int64, _QWORD))(*a2 + 176LL))(a2, v4, 0);
  v5 = sub_38CF310(v4, 0, v2, 0);
  return sub_38CB1F0(17, v3, v5, v2, 0);
}
