// Function: sub_26F71B0
// Address: 0x26f71b0
//
__int64 __fastcall sub_26F71B0(__int64 a1)
{
  unsigned __int64 v1; // r13
  __int64 v2; // rbx
  unsigned __int64 v3; // r12
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // r15
  __int64 v6; // r14
  unsigned __int64 v7; // rdi
  unsigned __int64 v8; // rdi
  _QWORD *v9; // rbx
  _QWORD *v10; // r12
  _QWORD *v11; // rdi
  unsigned __int64 v12; // rdi
  unsigned __int64 v13; // rdi
  unsigned __int64 v14; // rdi
  __int64 v17; // [rsp+8h] [rbp-38h]

  v1 = *(_QWORD *)(a1 + 400);
  v17 = *(_QWORD *)(a1 + 408);
  if ( v17 != v1 )
  {
    do
    {
      v2 = *(_QWORD *)(v1 + 16);
      v3 = v2 + 40LL * *(unsigned int *)(v1 + 24);
      if ( v2 != v3 )
      {
        do
        {
          v3 -= 40LL;
          v4 = *(_QWORD *)(v3 + 16);
          if ( v4 != v3 + 40 )
            _libc_free(v4);
          v5 = *(_QWORD *)v3;
          v6 = *(_QWORD *)v3 + 80LL * *(unsigned int *)(v3 + 8);
          if ( *(_QWORD *)v3 != v6 )
          {
            do
            {
              v6 -= 80;
              v7 = *(_QWORD *)(v6 + 8);
              if ( v7 != v6 + 24 )
                _libc_free(v7);
            }
            while ( v5 != v6 );
            v5 = *(_QWORD *)v3;
          }
          if ( v5 != v3 + 16 )
            _libc_free(v5);
        }
        while ( v2 != v3 );
        v3 = *(_QWORD *)(v1 + 16);
      }
      if ( v3 != v1 + 32 )
        _libc_free(v3);
      v1 += 72LL;
    }
    while ( v17 != v1 );
    v1 = *(_QWORD *)(a1 + 400);
  }
  if ( v1 )
    j_j___libc_free_0(v1);
  sub_26F6FE0(*(_QWORD *)(a1 + 368));
  v8 = *(_QWORD *)(a1 + 272);
  if ( v8 != a1 + 288 )
    _libc_free(v8);
  if ( !*(_BYTE *)(a1 + 204) )
    _libc_free(*(_QWORD *)(a1 + 184));
  v9 = *(_QWORD **)(a1 + 160);
  v10 = &v9[18 * *(unsigned int *)(a1 + 168)];
  if ( v9 != v10 )
  {
    do
    {
      v11 = (_QWORD *)*(v10 - 4);
      v10 -= 18;
      sub_26F6B10(v11);
      v12 = v10[9];
      if ( v12 )
        j_j___libc_free_0(v12);
      v13 = v10[6];
      if ( v13 )
        j_j___libc_free_0(v13);
      v14 = v10[2];
      if ( v14 )
        j_j___libc_free_0(v14);
    }
    while ( v9 != v10 );
    v10 = *(_QWORD **)(a1 + 160);
  }
  if ( v10 != (_QWORD *)(a1 + 176) )
    _libc_free((unsigned __int64)v10);
  return sub_C7D6A0(*(_QWORD *)(a1 + 136), 24LL * *(unsigned int *)(a1 + 152), 8);
}
