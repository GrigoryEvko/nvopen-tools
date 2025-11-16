// Function: sub_2EC14D0
// Address: 0x2ec14d0
//
void __fastcall sub_2EC14D0(_QWORD *a1)
{
  __int64 *v1; // r12
  unsigned __int64 v2; // rdi
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // rdi
  unsigned __int64 v6; // rdi
  __int64 v7; // r13
  __int64 i; // rbx
  unsigned __int64 v9; // rdi

  v1 = (__int64 *)a1[7];
  *a1 = &unk_4A29A70;
  if ( v1 )
  {
    v2 = v1[37];
    if ( v2 )
      j_j___libc_free_0_0(v2);
    v3 = v1[28];
    if ( (__int64 *)v3 != v1 + 30 )
      _libc_free(v3);
    v4 = v1[19];
    if ( (__int64 *)v4 != v1 + 21 )
      _libc_free(v4);
    v5 = v1[11];
    if ( (__int64 *)v5 != v1 + 14 )
      _libc_free(v5);
    v6 = v1[4];
    if ( (__int64 *)v6 != v1 + 7 )
      _libc_free(v6);
    v7 = *v1;
    if ( *v1 )
    {
      for ( i = v7 + 24LL * *(_QWORD *)(v7 - 8); v7 != i; i -= 24 )
      {
        v9 = *(_QWORD *)(i - 8);
        if ( v9 )
          j_j___libc_free_0_0(v9);
      }
      j_j_j___libc_free_0_0(v7 - 8);
    }
    j_j___libc_free_0((unsigned __int64)v1);
  }
}
