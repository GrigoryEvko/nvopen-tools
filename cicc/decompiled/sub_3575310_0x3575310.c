// Function: sub_3575310
// Address: 0x3575310
//
void __fastcall sub_3575310(__int64 a1)
{
  unsigned __int64 v2; // rdi
  __int64 v3; // rsi
  _QWORD *v4; // r12
  _QWORD *v5; // r13
  unsigned __int64 v6; // r14
  unsigned __int64 v7; // rdi
  unsigned __int64 v8; // rdi
  unsigned __int64 v9; // rdi
  __int64 v10; // rsi
  __int64 v11; // rdi
  __int64 v12; // rbx
  unsigned __int64 v13; // rdi

  v2 = *(_QWORD *)(a1 + 1576);
  if ( v2 != a1 + 1592 )
    _libc_free(v2);
  sub_C7D6A0(*(_QWORD *)(a1 + 1552), 8LL * *(unsigned int *)(a1 + 1568), 8);
  if ( !*(_BYTE *)(a1 + 1284) )
    _libc_free(*(_QWORD *)(a1 + 1264));
  v3 = *(unsigned int *)(a1 + 1248);
  if ( (_DWORD)v3 )
  {
    v4 = *(_QWORD **)(a1 + 1232);
    v5 = &v4[2 * v3];
    do
    {
      if ( *v4 != -4096 && *v4 != -8192 )
      {
        v6 = v4[1];
        if ( v6 )
        {
          sub_C7D6A0(*(_QWORD *)(v6 + 136), 16LL * *(unsigned int *)(v6 + 152), 8);
          if ( !*(_BYTE *)(v6 + 92) )
            _libc_free(*(_QWORD *)(v6 + 72));
          if ( !*(_BYTE *)(v6 + 28) )
            _libc_free(*(_QWORD *)(v6 + 8));
          j_j___libc_free_0(v6);
        }
      }
      v4 += 2;
    }
    while ( v5 != v4 );
    v3 = *(unsigned int *)(a1 + 1248);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 1232), 16 * v3, 8);
  if ( !*(_BYTE *)(a1 + 940) )
    _libc_free(*(_QWORD *)(a1 + 920));
  sub_C7D6A0(*(_QWORD *)(a1 + 888), 16LL * *(unsigned int *)(a1 + 904), 8);
  v7 = *(_QWORD *)(a1 + 816);
  if ( v7 != a1 + 832 )
    _libc_free(v7);
  v8 = *(_QWORD *)(a1 + 752);
  if ( v8 != a1 + 768 )
    _libc_free(v8);
  if ( !*(_BYTE *)(a1 + 620) )
    _libc_free(*(_QWORD *)(a1 + 600));
  v9 = *(_QWORD *)(a1 + 560);
  if ( v9 )
    j_j___libc_free_0(v9);
  if ( !*(_BYTE *)(a1 + 300) )
    _libc_free(*(_QWORD *)(a1 + 280));
  v10 = *(unsigned int *)(a1 + 264);
  v11 = *(_QWORD *)(a1 + 248);
  v12 = a1 + 16;
  sub_C7D6A0(v11, 4 * v10, 4);
  v13 = *(_QWORD *)(v12 - 16);
  if ( v13 != v12 )
    _libc_free(v13);
}
