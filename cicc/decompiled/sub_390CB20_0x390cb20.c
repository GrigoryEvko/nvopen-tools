// Function: sub_390CB20
// Address: 0x390cb20
//
bool __fastcall sub_390CB20(__int64 *a1, __int64 a2, __int64 a3)
{
  int v4; // r13d
  __int64 v5; // rax

  v4 = *(_DWORD *)(a3 + 88);
  v5 = sub_38BE350(*a1);
  sub_3910BC0(v5, a2, a3);
  return *(_DWORD *)(a3 + 88) != v4;
}
