// Function: sub_35D3640
// Address: 0x35d3640
//
__int64 __fastcall sub_35D3640(_QWORD *a1)
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
  return sub_BB9100((__int64)a1);
}
