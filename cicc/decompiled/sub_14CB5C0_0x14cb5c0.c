// Function: sub_14CB5C0
// Address: 0x14cb5c0
//
__int64 __fastcall sub_14CB5C0(__int64 a1)
{
  __int64 v2; // rdi

  v2 = a1 + 160;
  *(_QWORD *)(v2 - 160) = &unk_49ECC20;
  sub_14CB1D0(v2);
  j___libc_free_0(*(_QWORD *)(a1 + 168));
  return sub_16367B0(a1);
}
