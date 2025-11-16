// Function: sub_28FFA00
// Address: 0x28ffa00
//
__int64 __fastcall sub_28FFA00(__int64 a1)
{
  __int64 v2; // r13
  unsigned __int64 v3; // r12
  unsigned __int64 v4; // rdi
  __int64 v5; // r13
  unsigned __int64 v6; // r12
  unsigned __int64 v7; // rdi
  __int64 v8; // r13
  unsigned __int64 v9; // r12
  unsigned __int64 v10; // rdi
  __int64 v11; // r13
  unsigned __int64 v12; // r12
  unsigned __int64 v13; // rdi

  v2 = *(_QWORD *)(a1 + 176);
  v3 = v2 + 56LL * *(unsigned int *)(a1 + 184);
  if ( v2 != v3 )
  {
    do
    {
      v3 -= 56LL;
      v4 = *(_QWORD *)(v3 + 40);
      if ( v4 != v3 + 56 )
        _libc_free(v4);
      sub_C7D6A0(*(_QWORD *)(v3 + 16), 8LL * *(unsigned int *)(v3 + 32), 8);
    }
    while ( v2 != v3 );
    v3 = *(_QWORD *)(a1 + 176);
  }
  if ( v3 != a1 + 192 )
    _libc_free(v3);
  sub_C7D6A0(*(_QWORD *)(a1 + 152), 16LL * *(unsigned int *)(a1 + 168), 8);
  v5 = *(_QWORD *)(a1 + 128);
  v6 = v5 + 56LL * *(unsigned int *)(a1 + 136);
  if ( v5 != v6 )
  {
    do
    {
      v6 -= 56LL;
      v7 = *(_QWORD *)(v6 + 40);
      if ( v7 != v6 + 56 )
        _libc_free(v7);
      sub_C7D6A0(*(_QWORD *)(v6 + 16), 8LL * *(unsigned int *)(v6 + 32), 8);
    }
    while ( v5 != v6 );
    v6 = *(_QWORD *)(a1 + 128);
  }
  if ( v6 != a1 + 144 )
    _libc_free(v6);
  sub_C7D6A0(*(_QWORD *)(a1 + 104), 16LL * *(unsigned int *)(a1 + 120), 8);
  v8 = *(_QWORD *)(a1 + 80);
  v9 = v8 + 56LL * *(unsigned int *)(a1 + 88);
  if ( v8 != v9 )
  {
    do
    {
      v9 -= 56LL;
      v10 = *(_QWORD *)(v9 + 40);
      if ( v10 != v9 + 56 )
        _libc_free(v10);
      sub_C7D6A0(*(_QWORD *)(v9 + 16), 8LL * *(unsigned int *)(v9 + 32), 8);
    }
    while ( v8 != v9 );
    v9 = *(_QWORD *)(a1 + 80);
  }
  if ( a1 + 96 != v9 )
    _libc_free(v9);
  sub_C7D6A0(*(_QWORD *)(a1 + 56), 16LL * *(unsigned int *)(a1 + 72), 8);
  v11 = *(_QWORD *)(a1 + 32);
  v12 = v11 + 56LL * *(unsigned int *)(a1 + 40);
  if ( v11 != v12 )
  {
    do
    {
      v12 -= 56LL;
      v13 = *(_QWORD *)(v12 + 40);
      if ( v13 != v12 + 56 )
        _libc_free(v13);
      sub_C7D6A0(*(_QWORD *)(v12 + 16), 8LL * *(unsigned int *)(v12 + 32), 8);
    }
    while ( v11 != v12 );
    v12 = *(_QWORD *)(a1 + 32);
  }
  if ( a1 + 48 != v12 )
    _libc_free(v12);
  return sub_C7D6A0(*(_QWORD *)(a1 + 8), 16LL * *(unsigned int *)(a1 + 24), 8);
}
