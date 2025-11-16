// Function: sub_35D3950
// Address: 0x35d3950
//
void __fastcall sub_35D3950(_QWORD *a1)
{
  unsigned __int64 v2; // rdi
  unsigned __int64 v3; // rdi

  v2 = a1[32];
  if ( v2 )
    _libc_free(v2);
  v3 = a1[27];
  if ( (_QWORD *)v3 != a1 + 30 )
    _libc_free(v3);
  *a1 = &unk_49DAF80;
  sub_BB9100((__int64)a1);
  j_j___libc_free_0((unsigned __int64)a1);
}
