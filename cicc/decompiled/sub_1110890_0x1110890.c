// Function: sub_1110890
// Address: 0x1110890
//
void __fastcall sub_1110890(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi

  *(_QWORD *)a1 = off_49E62C8;
  v3 = *(_QWORD *)(a1 + 104);
  if ( v3 != a1 + 120 )
    _libc_free(v3, a2);
  if ( (*(_BYTE *)(a1 + 32) & 1) == 0 )
    sub_C7D6A0(*(_QWORD *)(a1 + 40), 16LL * *(unsigned int *)(a1 + 48), 8);
  nullsub_185();
}
