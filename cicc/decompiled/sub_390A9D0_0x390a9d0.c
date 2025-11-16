// Function: sub_390A9D0
// Address: 0x390a9d0
//
void __fastcall sub_390A9D0(__int64 a1)
{
  unsigned __int64 v2; // rdi
  unsigned __int64 v3; // rdi
  __int64 v4; // rbx
  unsigned __int64 v5; // r12
  unsigned __int64 v6; // rdi
  unsigned __int64 v7; // rdi
  unsigned __int64 *v8; // rbx
  unsigned __int64 *v9; // r12
  __int64 v10; // r15
  unsigned __int64 v11; // r13
  unsigned __int64 *v12; // rbx
  unsigned __int64 *v13; // r12
  unsigned __int64 v14; // rdi
  unsigned __int64 v15; // rdi
  unsigned __int64 v16; // rdi
  unsigned __int64 v17; // rdi
  __int64 v18; // rdi
  __int64 v19; // rdi
  __int64 v20; // rdi

  v2 = *(_QWORD *)(a1 + 2104);
  if ( v2 )
    j_j___libc_free_0(v2);
  v3 = *(_QWORD *)(a1 + 2080);
  if ( v3 )
    j_j___libc_free_0(v3);
  v4 = *(_QWORD *)(a1 + 504);
  v5 = v4 + 48LL * *(unsigned int *)(a1 + 512);
  if ( v4 != v5 )
  {
    do
    {
      v5 -= 48LL;
      v6 = *(_QWORD *)(v5 + 8);
      if ( v6 != v5 + 24 )
        _libc_free(v6);
    }
    while ( v4 != v5 );
    v5 = *(_QWORD *)(a1 + 504);
  }
  if ( v5 != a1 + 520 )
    _libc_free(v5);
  v7 = *(_QWORD *)(a1 + 200);
  if ( v7 != *(_QWORD *)(a1 + 192) )
    _libc_free(v7);
  v8 = *(unsigned __int64 **)(a1 + 160);
  v9 = *(unsigned __int64 **)(a1 + 152);
  if ( v8 != v9 )
  {
    do
    {
      if ( (unsigned __int64 *)*v9 != v9 + 2 )
        j_j___libc_free_0(*v9);
      v9 += 4;
    }
    while ( v8 != v9 );
    v9 = *(unsigned __int64 **)(a1 + 152);
  }
  if ( v9 )
    j_j___libc_free_0((unsigned __int64)v9);
  v10 = *(_QWORD *)(a1 + 136);
  v11 = *(_QWORD *)(a1 + 128);
  if ( v10 != v11 )
  {
    do
    {
      v12 = *(unsigned __int64 **)(v11 + 8);
      v13 = *(unsigned __int64 **)v11;
      if ( v12 != *(unsigned __int64 **)v11 )
      {
        do
        {
          if ( (unsigned __int64 *)*v13 != v13 + 2 )
            j_j___libc_free_0(*v13);
          v13 += 4;
        }
        while ( v12 != v13 );
        v13 = *(unsigned __int64 **)v11;
      }
      if ( v13 )
        j_j___libc_free_0((unsigned __int64)v13);
      v11 += 24LL;
    }
    while ( v10 != v11 );
    v11 = *(_QWORD *)(a1 + 128);
  }
  if ( v11 )
    j_j___libc_free_0(v11);
  v14 = *(_QWORD *)(a1 + 104);
  if ( v14 )
    j_j___libc_free_0(v14);
  v15 = *(_QWORD *)(a1 + 80);
  if ( v15 )
    j_j___libc_free_0(v15);
  v16 = *(_QWORD *)(a1 + 56);
  if ( v16 )
    j_j___libc_free_0(v16);
  v17 = *(_QWORD *)(a1 + 32);
  if ( v17 )
    j_j___libc_free_0(v17);
  v18 = *(_QWORD *)(a1 + 24);
  if ( v18 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v18 + 8LL))(v18);
  v19 = *(_QWORD *)(a1 + 16);
  if ( v19 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v19 + 8LL))(v19);
  v20 = *(_QWORD *)(a1 + 8);
  if ( v20 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v20 + 8LL))(v20);
}
