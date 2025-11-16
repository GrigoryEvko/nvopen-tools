// Function: sub_2913FD0
// Address: 0x2913fd0
//
__int64 __fastcall sub_2913FD0(__int64 a1)
{
  __int64 v2; // r13
  unsigned __int64 v3; // r12
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // rdi
  unsigned __int64 v6; // rdi
  unsigned __int64 v7; // rdi
  _QWORD *v8; // r13
  _QWORD *v9; // r12
  __int64 v10; // rax
  unsigned __int64 v11; // rdi

  v2 = *(_QWORD *)(a1 + 1080);
  v3 = v2 + 56LL * *(unsigned int *)(a1 + 1088);
  if ( v2 != v3 )
  {
    do
    {
      v3 -= 56LL;
      v4 = *(_QWORD *)(v3 + 8);
      if ( v4 != v3 + 24 )
        _libc_free(v4);
    }
    while ( v2 != v3 );
    v3 = *(_QWORD *)(a1 + 1080);
  }
  if ( v3 != a1 + 1096 )
    _libc_free(v3);
  if ( (*(_BYTE *)(a1 + 944) & 1) == 0 )
    sub_C7D6A0(*(_QWORD *)(a1 + 952), 16LL * *(unsigned int *)(a1 + 960), 8);
  v5 = *(_QWORD *)(a1 + 856);
  if ( v5 != a1 + 872 )
    _libc_free(v5);
  sub_C7D6A0(*(_QWORD *)(a1 + 832), 8LL * *(unsigned int *)(a1 + 848), 8);
  v6 = *(_QWORD *)(a1 + 760);
  if ( v6 != a1 + 776 )
    _libc_free(v6);
  if ( !*(_BYTE *)(a1 + 628) )
    _libc_free(*(_QWORD *)(a1 + 608));
  v7 = *(_QWORD *)(a1 + 456);
  if ( v7 != a1 + 472 )
    _libc_free(v7);
  sub_C7D6A0(*(_QWORD *)(a1 + 432), 8LL * *(unsigned int *)(a1 + 448), 8);
  v8 = *(_QWORD **)(a1 + 216);
  v9 = &v8[3 * *(unsigned int *)(a1 + 224)];
  if ( v8 != v9 )
  {
    do
    {
      v10 = *(v9 - 1);
      v9 -= 3;
      if ( v10 != 0 && v10 != -4096 && v10 != -8192 )
        sub_BD60C0(v9);
    }
    while ( v8 != v9 );
    v9 = *(_QWORD **)(a1 + 216);
  }
  if ( v9 != (_QWORD *)(a1 + 232) )
    _libc_free((unsigned __int64)v9);
  v11 = *(_QWORD *)(a1 + 72);
  if ( v11 != a1 + 88 )
    _libc_free(v11);
  return sub_C7D6A0(*(_QWORD *)(a1 + 48), 8LL * *(unsigned int *)(a1 + 64), 8);
}
