// Function: sub_16E4270
// Address: 0x16e4270
//
__int64 __fastcall sub_16E4270(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  __int64 v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // r8
  __int64 result; // rax

  v3 = *(_QWORD *)(a1 + 80);
  sub_16F8270(v3);
  result = sub_2241E50(v3, a2, v4, v5, v6);
  *(_DWORD *)(a1 + 96) = 22;
  *(_QWORD *)(a1 + 104) = result;
  return result;
}
