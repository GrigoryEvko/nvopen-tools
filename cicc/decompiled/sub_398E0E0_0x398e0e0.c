// Function: sub_398E0E0
// Address: 0x398e0e0
//
void __fastcall sub_398E0E0(__int64 a1)
{
  __int64 v2; // rbx
  unsigned __int64 v3; // r12
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // rdi
  unsigned __int64 v6; // rdi
  _QWORD *v7; // rbx
  unsigned __int64 v8; // r12
  unsigned __int64 v9; // rdi
  unsigned __int64 v10; // rdi
  unsigned __int64 v11; // rdi
  unsigned __int64 v12; // rdi
  __int64 v13; // rsi
  __int64 v14; // rsi

  *(_QWORD *)a1 = &unk_4A3F730;
  j___libc_free_0(*(_QWORD *)(a1 + 392));
  j___libc_free_0(*(_QWORD *)(a1 + 360));
  v2 = *(_QWORD *)(a1 + 336);
  v3 = *(_QWORD *)(a1 + 328);
  if ( v2 != v3 )
  {
    do
    {
      v4 = *(_QWORD *)(v3 + 16);
      if ( v4 != v3 + 32 )
        _libc_free(v4);
      v3 += 96LL;
    }
    while ( v2 != v3 );
    v3 = *(_QWORD *)(a1 + 328);
  }
  if ( v3 )
    j_j___libc_free_0(v3);
  j___libc_free_0(*(_QWORD *)(a1 + 304));
  v5 = *(_QWORD *)(a1 + 240);
  if ( v5 != a1 + 256 )
    _libc_free(v5);
  sub_1DA2140(a1 + 184);
  v6 = *(_QWORD *)(a1 + 184);
  if ( v6 != a1 + 232 )
    j_j___libc_free_0(v6);
  v7 = *(_QWORD **)(a1 + 144);
  while ( v7 )
  {
    v8 = (unsigned __int64)v7;
    v7 = (_QWORD *)*v7;
    v9 = *(_QWORD *)(v8 + 104);
    if ( v9 != v8 + 120 )
      _libc_free(v9);
    v10 = *(_QWORD *)(v8 + 56);
    if ( v10 != v8 + 72 )
      _libc_free(v10);
    j_j___libc_free_0(v8);
  }
  memset(*(void **)(a1 + 128), 0, 8LL * *(_QWORD *)(a1 + 136));
  v11 = *(_QWORD *)(a1 + 128);
  *(_QWORD *)(a1 + 152) = 0;
  *(_QWORD *)(a1 + 144) = 0;
  if ( v11 != a1 + 176 )
    j_j___libc_free_0(v11);
  sub_1DA2140(a1 + 72);
  v12 = *(_QWORD *)(a1 + 72);
  if ( v12 != a1 + 120 )
    j_j___libc_free_0(v12);
  v13 = *(_QWORD *)(a1 + 48);
  if ( v13 )
    sub_161E7C0(a1 + 48, v13);
  v14 = *(_QWORD *)(a1 + 24);
  if ( v14 )
    sub_161E7C0(a1 + 24, v14);
  nullsub_1975();
}
