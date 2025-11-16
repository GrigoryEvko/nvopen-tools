// Function: sub_20E5960
// Address: 0x20e5960
//
void __fastcall sub_20E5960(_QWORD *a1)
{
  __int64 v2; // rdi
  __int64 v3; // rdi
  __int64 v4; // rbx
  __int64 v5; // rdi
  __int64 v6; // rdi

  *a1 = off_4985A00;
  _libc_free(a1[24]);
  v2 = a1[21];
  if ( v2 )
    j_j___libc_free_0(v2, a1[23] - v2);
  v3 = a1[18];
  if ( v3 )
    j_j___libc_free_0(v3, a1[20] - v3);
  v4 = a1[14];
  while ( v4 )
  {
    sub_20E5700(*(_QWORD *)(v4 + 24));
    v5 = v4;
    v4 = *(_QWORD *)(v4 + 16);
    j_j___libc_free_0(v5, 48);
  }
  v6 = a1[9];
  if ( v6 )
    j_j___libc_free_0(v6, a1[11] - v6);
  _libc_free(a1[6]);
  nullsub_729();
}
