// Function: sub_1EAC4F0
// Address: 0x1eac4f0
//
void *__fastcall sub_1EAC4F0(__int64 a1)
{
  unsigned __int64 v2; // rdi

  *(_QWORD *)a1 = off_49FD5B0;
  v2 = *(_QWORD *)(a1 + 400);
  if ( v2 != a1 + 416 )
    _libc_free(v2);
  if ( (*(_BYTE *)(a1 + 264) & 1) == 0 )
    j___libc_free_0(*(_QWORD *)(a1 + 272));
  _libc_free(*(_QWORD *)(a1 + 208));
  _libc_free(*(_QWORD *)(a1 + 184));
  _libc_free(*(_QWORD *)(a1 + 160));
  *(_QWORD *)a1 = &unk_49EE078;
  return sub_16366C0((_QWORD *)a1);
}
