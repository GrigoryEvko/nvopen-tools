// Function: sub_38D04A0
// Address: 0x38d04a0
//
__int64 __fastcall sub_38D04A0(_QWORD *a1, __int64 a2)
{
  unsigned __int64 v2; // r13
  __int64 v3; // rbx

  v2 = *(_QWORD *)(a2 + 96) & 0xFFFFFFFFFFFFFFF8LL;
  v3 = sub_38D01B0((__int64)a1, v2);
  return v3 + sub_390B580(*a1, a1, v2);
}
