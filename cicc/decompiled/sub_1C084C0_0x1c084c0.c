// Function: sub_1C084C0
// Address: 0x1c084c0
//
__int64 __fastcall sub_1C084C0(_QWORD *a1)
{
  __int64 v2; // rdi

  v2 = (__int64)(a1 + 20);
  *(_QWORD *)(v2 - 160) = &unk_49F73F0;
  sub_1C08020(v2, 0, 0);
  j___libc_free_0(a1[30]);
  j___libc_free_0(a1[26]);
  j___libc_free_0(a1[22]);
  sub_16367B0(a1);
  return j_j___libc_free_0(a1, 296);
}
