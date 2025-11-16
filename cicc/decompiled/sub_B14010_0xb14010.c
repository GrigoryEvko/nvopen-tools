// Function: sub_B14010
// Address: 0xb14010
//
__int64 __fastcall sub_B14010(__int64 a1, __int64 a2)
{
  unsigned __int8 *v2; // rax
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // r12

  v2 = sub_B13320(a1);
  v3 = sub_ACADE0(*((__int64 ***)v2 + 1));
  v6 = sub_B98A20(v3, a2, v4, v5);
  sub_B91340(a1 + 40, 1);
  *(_QWORD *)(a1 + 48) = v6;
  return sub_B96F50(a1 + 40, 1);
}
