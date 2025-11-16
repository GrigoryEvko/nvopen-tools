// Function: sub_3925550
// Address: 0x3925550
//
void __fastcall sub_3925550(_QWORD *a1)
{
  unsigned __int64 *v2; // rbx
  unsigned __int64 *v3; // r12
  unsigned __int64 v4; // r13
  unsigned __int64 v5; // rdi
  unsigned __int64 v6; // rdi
  unsigned __int64 *v7; // rbx
  unsigned __int64 *v8; // r12
  unsigned __int64 v9; // r13
  unsigned __int64 v10; // rdi
  unsigned __int64 v11; // rdi
  __int64 v12; // rdi

  *a1 = off_4A3ECB8;
  j___libc_free_0(a1[25]);
  j___libc_free_0(a1[21]);
  sub_167FA50((__int64)(a1 + 13));
  v2 = (unsigned __int64 *)a1[11];
  v3 = (unsigned __int64 *)a1[10];
  if ( v2 != v3 )
  {
    do
    {
      v4 = *v3;
      if ( *v3 )
      {
        v5 = *(_QWORD *)(v4 + 56);
        if ( v5 != v4 + 72 )
          _libc_free(v5);
        v6 = *(_QWORD *)(v4 + 24);
        if ( v6 != v4 + 40 )
          _libc_free(v6);
        j_j___libc_free_0(v4);
      }
      ++v3;
    }
    while ( v2 != v3 );
    v3 = (unsigned __int64 *)a1[10];
  }
  if ( v3 )
    j_j___libc_free_0((unsigned __int64)v3);
  v7 = (unsigned __int64 *)a1[8];
  v8 = (unsigned __int64 *)a1[7];
  if ( v7 != v8 )
  {
    do
    {
      v9 = *v8;
      if ( *v8 )
      {
        v10 = *(_QWORD *)(v9 + 96);
        if ( v10 )
          j_j___libc_free_0(v10);
        v11 = *(_QWORD *)(v9 + 40);
        if ( v11 != v9 + 56 )
          j_j___libc_free_0(v11);
        j_j___libc_free_0(v9);
      }
      ++v8;
    }
    while ( v7 != v8 );
    v8 = (unsigned __int64 *)a1[7];
  }
  if ( v8 )
    j_j___libc_free_0((unsigned __int64)v8);
  v12 = a1[3];
  if ( v12 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v12 + 8LL))(v12);
  nullsub_1935();
}
