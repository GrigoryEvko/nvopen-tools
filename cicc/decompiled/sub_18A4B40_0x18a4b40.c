// Function: sub_18A4B40
// Address: 0x18a4b40
//
__int64 __fastcall sub_18A4B40(_QWORD *a1)
{
  __int64 v2; // rdi

  v2 = (__int64)(a1 + 20);
  *(_QWORD *)(v2 - 160) = off_49F1FE8;
  sub_18A4390(v2);
  sub_1636790(a1);
  return j_j___libc_free_0(a1, 1448);
}
