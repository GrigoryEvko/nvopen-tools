// Function: sub_B1F4C0
// Address: 0xb1f4c0
//
__int64 __fastcall sub_B1F4C0(__int64 a1, __int64 a2)
{
  __int64 v2; // rdi

  v2 = a1 + 176;
  *(_QWORD *)(v2 + 104) = a2;
  *(_DWORD *)(v2 + 120) = *(_DWORD *)(a2 + 92);
  sub_B1F440(v2);
  return 0;
}
