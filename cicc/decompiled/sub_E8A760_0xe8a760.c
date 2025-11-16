// Function: sub_E8A760
// Address: 0xe8a760
//
__int64 __fastcall sub_E8A760(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  _QWORD *v4; // rbx
  _QWORD *v5; // r13
  _QWORD *v6; // rdi
  __int64 v7; // rsi
  __int64 v8; // rdi
  __int64 v9; // r13
  __int64 v10; // rdi
  __int64 v11; // rdi
  __int64 v12; // rdi
  __int64 v13; // rdi
  __int64 v14; // rdi

  *(_QWORD *)a1 = &unk_49E2FC8;
  v3 = *(unsigned int *)(a1 + 432);
  if ( (_DWORD)v3 )
  {
    v4 = *(_QWORD **)(a1 + 416);
    v5 = &v4[5 * v3];
    do
    {
      if ( *v4 != -8192 && *v4 != -4096 )
      {
        v6 = (_QWORD *)v4[1];
        if ( v6 != v4 + 3 )
          _libc_free(v6, a2);
      }
      v4 += 5;
    }
    while ( v5 != v4 );
    v3 = *(unsigned int *)(a1 + 432);
  }
  v7 = 40 * v3;
  sub_C7D6A0(*(_QWORD *)(a1 + 416), 40 * v3, 8);
  v8 = *(_QWORD *)(a1 + 312);
  if ( v8 != a1 + 328 )
    _libc_free(v8, v7);
  v9 = *(_QWORD *)(a1 + 296);
  if ( v9 )
  {
    if ( !*(_BYTE *)(v9 + 108) )
      _libc_free(*(_QWORD *)(v9 + 88), v7);
    v10 = *(_QWORD *)(v9 + 56);
    if ( v10 != v9 + 72 )
      _libc_free(v10, v7);
    v11 = *(_QWORD *)(v9 + 40);
    if ( v9 + 56 != v11 )
      _libc_free(v11, v7);
    v12 = *(_QWORD *)(v9 + 24);
    if ( v12 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v12 + 8LL))(v12);
    v13 = *(_QWORD *)(v9 + 16);
    if ( v13 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v13 + 8LL))(v13);
    v14 = *(_QWORD *)(v9 + 8);
    if ( v14 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v14 + 8LL))(v14);
    j_j___libc_free_0(v9, 376);
  }
  return sub_E98B30(a1);
}
