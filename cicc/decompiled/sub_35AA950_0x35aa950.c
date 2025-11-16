// Function: sub_35AA950
// Address: 0x35aa950
//
__int64 __fastcall sub_35AA950(__int64 a1)
{
  unsigned __int64 v2; // rdi

  *(_QWORD *)a1 = off_4A39EA0;
  v2 = *(_QWORD *)(a1 + 256);
  if ( v2 != a1 + 272 )
    _libc_free(v2);
  sub_C7D6A0(*(_QWORD *)(a1 + 232), 8LL * *(unsigned int *)(a1 + 248), 8);
  *(_QWORD *)a1 = &unk_49DAF80;
  return sub_BB9100(a1);
}
