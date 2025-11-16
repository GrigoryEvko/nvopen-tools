// Function: sub_3424320
// Address: 0x3424320
//
__int64 __fastcall sub_3424320(__int64 a1)
{
  __int64 v2; // rax
  _QWORD *v3; // rbx
  _QWORD *v4; // r12
  unsigned __int64 v5; // rdi
  __int64 v6; // rdi
  unsigned __int64 v7; // rdi
  unsigned __int64 v8; // rdi
  unsigned __int64 v9; // rdi
  unsigned __int64 v10; // rdi
  unsigned __int64 v11; // r12
  unsigned __int64 v12; // rdi
  __int64 v13; // r15
  unsigned __int64 v14; // r13
  unsigned __int64 v15; // rdi
  __int64 v16; // r12
  unsigned __int64 v17; // rbx
  __int64 v18; // rsi

  v2 = *(unsigned int *)(a1 + 1008);
  if ( (_DWORD)v2 )
  {
    v3 = *(_QWORD **)(a1 + 992);
    v4 = &v3[5 * v2];
    do
    {
      if ( *v3 != -8192 && *v3 != -4096 )
      {
        v5 = v3[1];
        if ( (_QWORD *)v5 != v3 + 3 )
          _libc_free(v5);
      }
      v3 += 5;
    }
    while ( v4 != v3 );
    v2 = *(unsigned int *)(a1 + 1008);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 992), 40 * v2, 8);
  sub_C7D6A0(*(_QWORD *)(a1 + 936), 16LL * *(unsigned int *)(a1 + 952), 8);
  v6 = *(_QWORD *)(a1 + 896);
  if ( v6 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v6 + 16LL))(v6);
  v7 = *(_QWORD *)(a1 + 704);
  if ( v7 != a1 + 720 )
    _libc_free(v7);
  v8 = *(_QWORD *)(a1 + 560);
  if ( v8 != a1 + 576 )
    _libc_free(v8);
  v9 = *(_QWORD *)(a1 + 416);
  if ( v9 != a1 + 432 )
    _libc_free(v9);
  v10 = *(_QWORD *)(a1 + 320);
  if ( v10 != a1 + 336 )
    _libc_free(v10);
  v11 = *(_QWORD *)(a1 + 304);
  if ( (v11 & 1) == 0 && v11 )
  {
    if ( *(_QWORD *)v11 != v11 + 16 )
      _libc_free(*(_QWORD *)v11);
    j_j___libc_free_0(v11);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 280), 32LL * *(unsigned int *)(a1 + 296), 8);
  v12 = *(_QWORD *)(a1 + 128);
  if ( v12 != a1 + 144 )
    _libc_free(v12);
  v13 = *(_QWORD *)(a1 + 104);
  v14 = v13 + 32LL * *(unsigned int *)(a1 + 112);
  if ( v13 != v14 )
  {
    do
    {
      v15 = *(_QWORD *)(v14 - 24);
      v16 = *(_QWORD *)(v14 - 16);
      v14 -= 32LL;
      v17 = v15;
      if ( v16 != v15 )
      {
        do
        {
          v18 = *(_QWORD *)(v17 + 24);
          if ( v18 )
            sub_B91220(v17 + 24, v18);
          v17 += 32LL;
        }
        while ( v16 != v17 );
        v15 = *(_QWORD *)(v14 + 8);
      }
      if ( v15 )
        j_j___libc_free_0(v15);
    }
    while ( v13 != v14 );
    v14 = *(_QWORD *)(a1 + 104);
  }
  if ( v14 != a1 + 120 )
    _libc_free(v14);
  sub_C7D6A0(*(_QWORD *)(a1 + 80), 16LL * *(unsigned int *)(a1 + 96), 8);
  sub_C7D6A0(*(_QWORD *)(a1 + 48), 24LL * *(unsigned int *)(a1 + 64), 8);
  return sub_C7D6A0(*(_QWORD *)(a1 + 16), 24LL * *(unsigned int *)(a1 + 32), 8);
}
