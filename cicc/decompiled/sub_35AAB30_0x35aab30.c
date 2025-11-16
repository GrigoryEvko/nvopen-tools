// Function: sub_35AAB30
// Address: 0x35aab30
//
void __fastcall sub_35AAB30(unsigned __int64 a1)
{
  unsigned __int64 v2; // rdi

  *(_QWORD *)a1 = off_4A39EA0;
  v2 = *(_QWORD *)(a1 + 256);
  if ( v2 != a1 + 272 )
    _libc_free(v2);
  sub_C7D6A0(*(_QWORD *)(a1 + 232), 8LL * *(unsigned int *)(a1 + 248), 8);
  *(_QWORD *)a1 = &unk_49DAF80;
  sub_BB9100(a1);
  j_j___libc_free_0(a1);
}
