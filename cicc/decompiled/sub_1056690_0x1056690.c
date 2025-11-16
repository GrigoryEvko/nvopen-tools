// Function: sub_1056690
// Address: 0x1056690
//
__int64 __fastcall sub_1056690(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  __int64 v4; // rsi
  __int64 v5; // rsi
  _QWORD *v6; // r12
  _QWORD *v7; // r13
  __int64 v8; // r14
  __int64 v9; // rsi
  __int64 v10; // rsi
  __int64 v11; // rsi
  __int64 v12; // rdi
  __int64 v13; // rdi
  __int64 v14; // rdi
  __int64 v15; // rsi
  __int64 v16; // rdi
  __int64 v17; // rbx
  __int64 v18; // rsi
  __int64 result; // rax
  __int64 v20; // rdi

  v3 = *(_QWORD *)(a1 + 1576);
  if ( v3 != a1 + 1592 )
    _libc_free(v3, a2);
  v4 = 8LL * *(unsigned int *)(a1 + 1568);
  sub_C7D6A0(*(_QWORD *)(a1 + 1552), v4, 8);
  if ( !*(_BYTE *)(a1 + 1284) )
    _libc_free(*(_QWORD *)(a1 + 1264), v4);
  v5 = *(unsigned int *)(a1 + 1248);
  if ( (_DWORD)v5 )
  {
    v6 = *(_QWORD **)(a1 + 1232);
    v7 = &v6[2 * v5];
    do
    {
      if ( *v6 != -4096 && *v6 != -8192 )
      {
        v8 = v6[1];
        if ( v8 )
        {
          v9 = 16LL * *(unsigned int *)(v8 + 152);
          sub_C7D6A0(*(_QWORD *)(v8 + 136), v9, 8);
          if ( !*(_BYTE *)(v8 + 92) )
            _libc_free(*(_QWORD *)(v8 + 72), v9);
          if ( !*(_BYTE *)(v8 + 28) )
            _libc_free(*(_QWORD *)(v8 + 8), v9);
          j_j___libc_free_0(v8, 160);
        }
      }
      v6 += 2;
    }
    while ( v7 != v6 );
    v5 = *(unsigned int *)(a1 + 1248);
  }
  v10 = 16 * v5;
  sub_C7D6A0(*(_QWORD *)(a1 + 1232), v10, 8);
  if ( !*(_BYTE *)(a1 + 940) )
    _libc_free(*(_QWORD *)(a1 + 920), v10);
  v11 = 16LL * *(unsigned int *)(a1 + 904);
  sub_C7D6A0(*(_QWORD *)(a1 + 888), v11, 8);
  v12 = *(_QWORD *)(a1 + 816);
  if ( v12 != a1 + 832 )
    _libc_free(v12, v11);
  v13 = *(_QWORD *)(a1 + 752);
  if ( v13 != a1 + 768 )
    _libc_free(v13, v11);
  if ( !*(_BYTE *)(a1 + 620) )
    _libc_free(*(_QWORD *)(a1 + 600), v11);
  v14 = *(_QWORD *)(a1 + 560);
  if ( v14 )
  {
    v11 = *(_QWORD *)(a1 + 576) - v14;
    j_j___libc_free_0(v14, v11);
  }
  if ( !*(_BYTE *)(a1 + 300) )
    _libc_free(*(_QWORD *)(a1 + 280), v11);
  v15 = *(unsigned int *)(a1 + 264);
  v16 = *(_QWORD *)(a1 + 248);
  v17 = a1 + 16;
  v18 = 8 * v15;
  result = sub_C7D6A0(v16, v18, 8);
  v20 = *(_QWORD *)(v17 - 16);
  if ( v20 != v17 )
    return _libc_free(v20, v18);
  return result;
}
