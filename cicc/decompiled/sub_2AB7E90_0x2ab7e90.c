// Function: sub_2AB7E90
// Address: 0x2ab7e90
//
__int64 __fastcall sub_2AB7E90(__int64 a1)
{
  unsigned __int64 v2; // rdi
  unsigned __int64 v3; // rdi
  _QWORD *v4; // r13
  _QWORD *v5; // r12
  unsigned __int64 v6; // rdi
  __int64 v7; // rax
  __int64 v8; // r13
  unsigned __int64 v9; // r12
  __int64 v10; // rax

  v2 = *(_QWORD *)(a1 + 600);
  if ( v2 != a1 + 616 )
    _libc_free(v2);
  v3 = *(_QWORD *)(a1 + 536);
  if ( v3 != a1 + 552 )
    _libc_free(v3);
  if ( *(_BYTE *)(a1 + 468) )
  {
    if ( *(_BYTE *)(a1 + 372) )
      goto LABEL_7;
  }
  else
  {
    _libc_free(*(_QWORD *)(a1 + 448));
    if ( *(_BYTE *)(a1 + 372) )
    {
LABEL_7:
      if ( *(_BYTE *)(a1 + 268) )
        goto LABEL_8;
LABEL_33:
      _libc_free(*(_QWORD *)(a1 + 248));
      if ( *(_BYTE *)(a1 + 204) )
        goto LABEL_9;
      goto LABEL_34;
    }
  }
  _libc_free(*(_QWORD *)(a1 + 352));
  if ( !*(_BYTE *)(a1 + 268) )
    goto LABEL_33;
LABEL_8:
  if ( *(_BYTE *)(a1 + 204) )
    goto LABEL_9;
LABEL_34:
  _libc_free(*(_QWORD *)(a1 + 184));
LABEL_9:
  v4 = *(_QWORD **)(a1 + 160);
  v5 = &v4[11 * *(unsigned int *)(a1 + 168)];
  if ( v4 != v5 )
  {
    do
    {
      v5 -= 11;
      v6 = v5[7];
      if ( (_QWORD *)v6 != v5 + 9 )
        _libc_free(v6);
      v7 = v5[3];
      if ( v7 != 0 && v7 != -4096 && v7 != -8192 )
        sub_BD60C0(v5 + 1);
    }
    while ( v4 != v5 );
    v5 = *(_QWORD **)(a1 + 160);
  }
  if ( v5 != (_QWORD *)(a1 + 176) )
    _libc_free((unsigned __int64)v5);
  sub_C7D6A0(*(_QWORD *)(a1 + 136), 16LL * *(unsigned int *)(a1 + 152), 8);
  v8 = *(_QWORD *)(a1 + 112);
  v9 = v8 + 184LL * *(unsigned int *)(a1 + 120);
  if ( v8 != v9 )
  {
    do
    {
      v9 -= 184LL;
      if ( !*(_BYTE *)(v9 + 108) )
        _libc_free(*(_QWORD *)(v9 + 88));
      v10 = *(_QWORD *)(v9 + 32);
      if ( v10 != 0 && v10 != -4096 && v10 != -8192 )
        sub_BD60C0((_QWORD *)(v9 + 16));
    }
    while ( v8 != v9 );
    v9 = *(_QWORD *)(a1 + 112);
  }
  if ( a1 + 128 != v9 )
    _libc_free(v9);
  return sub_C7D6A0(*(_QWORD *)(a1 + 88), 16LL * *(unsigned int *)(a1 + 104), 8);
}
