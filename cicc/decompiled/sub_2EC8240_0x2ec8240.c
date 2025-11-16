// Function: sub_2EC8240
// Address: 0x2ec8240
//
void __fastcall sub_2EC8240(__int64 a1)
{
  __int64 v2; // rdi
  __int64 v3; // rbx
  unsigned __int64 v4; // r12
  unsigned __int64 v5; // rdi
  unsigned __int64 v6; // rdi
  unsigned __int64 v7; // r14
  unsigned __int64 v8; // r15
  _QWORD *v9; // r12
  _QWORD *v10; // rbx
  unsigned __int64 v11; // rdi
  unsigned __int64 v12; // rdi
  unsigned __int64 v13; // rdi
  unsigned __int64 v14; // rdi
  unsigned __int64 v15; // rdi
  unsigned __int64 v16; // rdi

  v2 = *(_QWORD *)(a1 + 152);
  if ( v2 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v2 + 8LL))(v2);
  v3 = *(_QWORD *)(a1 + 440);
  v4 = v3 + 16LL * *(unsigned int *)(a1 + 448);
  if ( v3 != v4 )
  {
    do
    {
      v4 -= 16LL;
      if ( *(_DWORD *)(v4 + 8) > 0x40u && *(_QWORD *)v4 )
        j_j___libc_free_0_0(*(_QWORD *)v4);
    }
    while ( v3 != v4 );
    v4 = *(_QWORD *)(a1 + 440);
  }
  if ( v4 != a1 + 456 )
    _libc_free(v4);
  v5 = *(_QWORD *)(a1 + 360);
  if ( v5 != a1 + 376 )
    _libc_free(v5);
  v6 = *(_QWORD *)(a1 + 336);
  if ( v6 )
    j_j___libc_free_0(v6);
  v7 = *(_QWORD *)(a1 + 304);
  while ( v7 )
  {
    v8 = v7;
    v9 = (_QWORD *)(v7 + 40);
    sub_2EC3080(*(_QWORD **)(v7 + 24));
    v10 = *(_QWORD **)(v7 + 40);
    v7 = *(_QWORD *)(v7 + 16);
    while ( v9 != v10 )
    {
      v11 = (unsigned __int64)v10;
      v10 = (_QWORD *)*v10;
      j_j___libc_free_0(v11);
    }
    j_j___libc_free_0(v8);
  }
  v12 = *(_QWORD *)(a1 + 192);
  if ( v12 != a1 + 208 )
    _libc_free(v12);
  v13 = *(_QWORD *)(a1 + 128);
  if ( v13 )
    j_j___libc_free_0(v13);
  v14 = *(_QWORD *)(a1 + 96);
  if ( v14 != a1 + 112 )
    j_j___libc_free_0(v14);
  v15 = *(_QWORD *)(a1 + 64);
  if ( v15 )
    j_j___libc_free_0(v15);
  v16 = *(_QWORD *)(a1 + 32);
  if ( v16 != a1 + 48 )
    j_j___libc_free_0(v16);
}
