// Function: sub_CA3FA0
// Address: 0xca3fa0
//
__int64 __fastcall sub_CA3FA0(__int64 a1)
{
  int *v2; // rdi
  __int64 v3; // rdi
  __int64 v4; // rdi

  v2 = (int *)(a1 + 8);
  *((_QWORD *)v2 - 1) = off_4979C68;
  sub_C83820(v2);
  v3 = *(_QWORD *)(a1 + 104);
  *(_DWORD *)(a1 + 8) = unk_3F66FFC;
  if ( v3 != a1 + 120 )
    j_j___libc_free_0(v3, *(_QWORD *)(a1 + 120) + 1LL);
  v4 = *(_QWORD *)(a1 + 16);
  if ( v4 != a1 + 32 )
    j_j___libc_free_0(v4, *(_QWORD *)(a1 + 32) + 1LL);
  nullsub_169();
  return j_j___libc_free_0(a1, 136);
}
