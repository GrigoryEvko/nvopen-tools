// Function: sub_334D0B0
// Address: 0x334d0b0
//
void __fastcall sub_334D0B0(__int64 a1)
{
  unsigned __int64 v2; // rdi
  unsigned __int64 v3; // rdi

  *(_QWORD *)a1 = off_4A361B0;
  sub_C7D6A0(*(_QWORD *)(a1 + 664), 16LL * *(unsigned int *)(a1 + 680), 8);
  v2 = *(_QWORD *)(a1 + 632);
  if ( v2 )
    j_j___libc_free_0(v2);
  v3 = *(_QWORD *)(a1 + 608);
  *(_QWORD *)a1 = &unk_4A365B8;
  if ( v3 )
    j_j___libc_free_0(v3);
  sub_2F8EAD0((_QWORD *)a1);
}
