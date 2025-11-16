// Function: sub_34DF6F0
// Address: 0x34df6f0
//
void __fastcall sub_34DF6F0(_QWORD *a1)
{
  unsigned __int64 v2; // rdi
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // rbx
  unsigned __int64 v6; // rdi
  unsigned __int64 v7; // rdi
  unsigned __int64 v8; // rdi

  *a1 = off_49D8D38;
  v2 = a1[30];
  if ( (_QWORD *)v2 != a1 + 32 )
    _libc_free(v2);
  v3 = a1[27];
  if ( v3 )
    j_j___libc_free_0(v3);
  v4 = a1[24];
  if ( v4 )
    j_j___libc_free_0(v4);
  v5 = a1[20];
  while ( v5 )
  {
    sub_34DF480(*(_QWORD *)(v5 + 24));
    v6 = v5;
    v5 = *(_QWORD *)(v5 + 16);
    j_j___libc_free_0(v6);
  }
  v7 = a1[15];
  if ( v7 )
    j_j___libc_free_0(v7);
  v8 = a1[6];
  if ( (_QWORD *)v8 != a1 + 8 )
    _libc_free(v8);
  nullsub_1639();
}
