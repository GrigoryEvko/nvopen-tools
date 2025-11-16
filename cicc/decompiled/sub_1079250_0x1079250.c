// Function: sub_1079250
// Address: 0x1079250
//
__int64 __fastcall sub_1079250(__int64 a1, __int64 a2)
{
  __int64 v3; // rbx
  __int64 v4; // r12
  __int64 v5; // rdi
  _QWORD *v6; // rbx
  _QWORD *v7; // r12
  _QWORD *v8; // rdi
  __int64 v9; // rax
  _QWORD *v10; // r12
  _QWORD *v11; // rbx
  _QWORD *v12; // rdi
  __int64 v13; // rsi
  _QWORD *v14; // r12
  _QWORD *v15; // rbx
  __int64 v16; // rdi
  __int64 v17; // rdi
  __int64 v18; // rdi
  __int64 v19; // rdi
  __int64 v20; // rsi
  __int64 v21; // rdi
  __int64 v22; // rdi
  __int64 v23; // rdi

  *(_QWORD *)a1 = off_49E6150;
  v3 = *(_QWORD *)(a1 + 736);
  v4 = v3 + 80LL * *(unsigned int *)(a1 + 744);
  if ( v3 != v4 )
  {
    do
    {
      v4 -= 80;
      v5 = *(_QWORD *)(v4 + 48);
      if ( v5 != v4 + 72 )
        _libc_free(v5, a2);
    }
    while ( v3 != v4 );
    v4 = *(_QWORD *)(a1 + 736);
  }
  if ( v4 != a1 + 752 )
    _libc_free(v4, a2);
  v6 = *(_QWORD **)(a1 + 464);
  v7 = &v6[8 * (unsigned __int64)*(unsigned int *)(a1 + 472)];
  if ( v6 != v7 )
  {
    do
    {
      v7 -= 8;
      v8 = (_QWORD *)v7[3];
      if ( v8 != v7 + 5 )
        _libc_free(v8, a2);
      if ( (_QWORD *)*v7 != v7 + 2 )
        _libc_free(*v7, a2);
    }
    while ( v6 != v7 );
    v7 = *(_QWORD **)(a1 + 464);
  }
  if ( v7 != (_QWORD *)(a1 + 480) )
    _libc_free(v7, a2);
  v9 = *(unsigned int *)(a1 + 456);
  if ( (_DWORD)v9 )
  {
    v10 = *(_QWORD **)(a1 + 440);
    v11 = &v10[9 * v9];
    do
    {
      v12 = (_QWORD *)v10[3];
      if ( v12 != v10 + 5 )
        _libc_free(v12, a2);
      if ( (_QWORD *)*v10 != v10 + 2 )
        _libc_free(*v10, a2);
      v10 += 9;
    }
    while ( v11 != v10 );
    v9 = *(unsigned int *)(a1 + 456);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 440), 72 * v9, 8);
  sub_C7D6A0(*(_QWORD *)(a1 + 408), 16LL * *(unsigned int *)(a1 + 424), 8);
  v13 = *(unsigned int *)(a1 + 392);
  if ( (_DWORD)v13 )
  {
    v14 = *(_QWORD **)(a1 + 376);
    v15 = &v14[4 * v13];
    do
    {
      if ( *v14 != -8192 && *v14 != -4096 )
      {
        v16 = v14[1];
        if ( v16 )
          j_j___libc_free_0(v16, v14[3] - v16);
      }
      v14 += 4;
    }
    while ( v15 != v14 );
    v13 = *(unsigned int *)(a1 + 392);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 376), 32 * v13, 8);
  v17 = *(_QWORD *)(a1 + 360);
  if ( v17 )
    j_j___libc_free_0(v17, 32);
  v18 = *(_QWORD *)(a1 + 352);
  if ( v18 )
    j_j___libc_free_0(v18, 32);
  v19 = *(_QWORD *)(a1 + 328);
  if ( v19 )
    j_j___libc_free_0(v19, *(_QWORD *)(a1 + 344) - v19);
  sub_C7D6A0(*(_QWORD *)(a1 + 304), 32LL * *(unsigned int *)(a1 + 320), 8);
  sub_C7D6A0(*(_QWORD *)(a1 + 272), 16LL * *(unsigned int *)(a1 + 288), 8);
  sub_C7D6A0(*(_QWORD *)(a1 + 240), 16LL * *(unsigned int *)(a1 + 256), 8);
  sub_C7D6A0(*(_QWORD *)(a1 + 208), 16LL * *(unsigned int *)(a1 + 224), 8);
  v20 = 16LL * *(unsigned int *)(a1 + 192);
  sub_C7D6A0(*(_QWORD *)(a1 + 176), v20, 8);
  v21 = *(_QWORD *)(a1 + 144);
  if ( v21 )
  {
    v20 = *(_QWORD *)(a1 + 160) - v21;
    j_j___libc_free_0(v21, v20);
  }
  v22 = *(_QWORD *)(a1 + 120);
  if ( v22 )
  {
    v20 = *(_QWORD *)(a1 + 136) - v22;
    j_j___libc_free_0(v22, v20);
  }
  v23 = *(_QWORD *)(a1 + 112);
  if ( v23 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v23 + 8LL))(v23);
  return sub_E8EC10(a1, v20);
}
