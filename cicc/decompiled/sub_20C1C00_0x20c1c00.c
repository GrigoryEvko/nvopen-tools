// Function: sub_20C1C00
// Address: 0x20c1c00
//
void __fastcall sub_20C1C00(_QWORD *a1)
{
  _QWORD *v1; // r13
  __int64 v3; // rdi
  __int64 v4; // rdi
  __int64 v5; // rdi
  __int64 v6; // rdi

  v1 = (_QWORD *)a1[9];
  *a1 = off_49859B8;
  if ( v1 )
  {
    v3 = v1[16];
    if ( v3 )
      j_j___libc_free_0(v3, v1[18] - v3);
    v4 = v1[13];
    if ( v4 )
      j_j___libc_free_0(v4, v1[15] - v4);
    sub_20C1A30(v1[9]);
    v5 = v1[4];
    if ( v5 )
      j_j___libc_free_0(v5, v1[6] - v5);
    v6 = v1[1];
    if ( v6 )
      j_j___libc_free_0(v6, v1[3] - v6);
    j_j___libc_free_0(v1, 152);
  }
  _libc_free(a1[6]);
  nullsub_729();
}
