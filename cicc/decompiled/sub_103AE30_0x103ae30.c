// Function: sub_103AE30
// Address: 0x103ae30
//
void __fastcall sub_103AE30(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi

  *(_QWORD *)a1 = off_49E5A60;
  *(_QWORD *)(a1 + 552) = &unk_49DDBE8;
  if ( (*(_BYTE *)(a1 + 568) & 1) == 0 )
  {
    a2 = 16LL * *(unsigned int *)(a1 + 584);
    sub_C7D6A0(*(_QWORD *)(a1 + 576), a2, 8);
  }
  nullsub_184();
  v3 = *(_QWORD *)(a1 + 400);
  if ( v3 != a1 + 416 )
    _libc_free(v3, a2);
  if ( (*(_BYTE *)(a1 + 48) & 1) == 0 )
    sub_C7D6A0(*(_QWORD *)(a1 + 56), 40LL * *(unsigned int *)(a1 + 64), 8);
  nullsub_35();
}
