// Function: sub_34B2DC0
// Address: 0x34b2dc0
//
void __fastcall sub_34B2DC0(_QWORD *a1)
{
  _QWORD *v1; // r13
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // rdi
  unsigned __int64 v6; // rdi
  unsigned __int64 v7; // rdi

  v1 = (_QWORD *)a1[15];
  *a1 = off_49D8CF0;
  if ( v1 )
  {
    v3 = v1[16];
    if ( v3 )
      j_j___libc_free_0(v3);
    v4 = v1[13];
    if ( v4 )
      j_j___libc_free_0(v4);
    sub_34B2BF0(v1[9]);
    v5 = v1[4];
    if ( v5 )
      j_j___libc_free_0(v5);
    v6 = v1[1];
    if ( v6 )
      j_j___libc_free_0(v6);
    j_j___libc_free_0((unsigned __int64)v1);
  }
  v7 = a1[6];
  if ( (_QWORD *)v7 != a1 + 8 )
    _libc_free(v7);
  nullsub_1639();
}
