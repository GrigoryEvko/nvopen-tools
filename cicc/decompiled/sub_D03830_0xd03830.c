// Function: sub_D03830
// Address: 0xd03830
//
__int64 __fastcall sub_D03830(__int64 a1)
{
  *(_QWORD *)a1 = &unk_49DDBE8;
  if ( (*(_BYTE *)(a1 + 16) & 1) == 0 )
    sub_C7D6A0(*(_QWORD *)(a1 + 24), 16LL * *(unsigned int *)(a1 + 32), 8);
  nullsub_184();
  return j_j___libc_free_0(a1, 152);
}
