// Function: sub_CB1010
// Address: 0xcb1010
//
__int64 __fastcall sub_CB1010(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 ***v4; // rdi
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 result; // rax

  v4 = *(__int64 ****)(a1 + 80);
  sub_CA89D0(v4, a2, a3, 0);
  result = sub_2241E50(v4, a2, v5, v6, v7);
  *(_DWORD *)(a1 + 96) = 22;
  *(_QWORD *)(a1 + 104) = result;
  return result;
}
