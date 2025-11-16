// Function: sub_2105BC0
// Address: 0x2105bc0
//
void *__fastcall sub_2105BC0(__int64 a1)
{
  unsigned __int64 v2; // rdi

  *(_QWORD *)a1 = &unk_4A00DB0;
  v2 = *(_QWORD *)(a1 + 280);
  if ( v2 != a1 + 296 )
    _libc_free(v2);
  sub_2105B40(*(_QWORD **)(a1 + 248));
  _libc_free(*(_QWORD *)(a1 + 208));
  _libc_free(*(_QWORD *)(a1 + 184));
  _libc_free(*(_QWORD *)(a1 + 160));
  *(_QWORD *)a1 = &unk_49EE078;
  return sub_16366C0((_QWORD *)a1);
}
