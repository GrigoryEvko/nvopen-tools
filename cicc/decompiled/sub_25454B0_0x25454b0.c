// Function: sub_25454B0
// Address: 0x25454b0
//
__int64 __fastcall sub_25454B0(__int64 a1)
{
  unsigned __int64 v2; // rdi
  unsigned __int64 v3; // rdi

  *(_QWORD *)a1 = off_4A19650;
  *(_QWORD *)(a1 + 88) = &unk_4A19710;
  v2 = *(_QWORD *)(a1 + 136);
  if ( v2 != a1 + 152 )
    _libc_free(v2);
  sub_C7D6A0(*(_QWORD *)(a1 + 112), 8LL * *(unsigned int *)(a1 + 128), 8);
  v3 = *(_QWORD *)(a1 + 40);
  *(_QWORD *)a1 = &unk_4A16C00;
  if ( v3 != a1 + 56 )
    _libc_free(v3);
  return sub_C7D6A0(*(_QWORD *)(a1 + 16), 8LL * *(unsigned int *)(a1 + 32), 8);
}
