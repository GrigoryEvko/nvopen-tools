// Function: sub_2C91DD0
// Address: 0x2c91dd0
//
__int64 __fastcall sub_2C91DD0(__int64 a1)
{
  unsigned __int64 v2; // rbx
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rbx
  unsigned __int64 v5; // r13
  __int64 v6; // rsi
  __int64 v7; // rdi
  unsigned __int64 v8; // rbx
  unsigned __int64 v9; // rdi
  __int64 v10; // rsi
  _QWORD *v11; // rbx
  _QWORD *v12; // r13
  unsigned __int64 v13; // rdi

  v2 = *(_QWORD *)(a1 + 328);
  while ( v2 )
  {
    sub_2C91920(*(_QWORD *)(v2 + 24));
    v3 = v2;
    v2 = *(_QWORD *)(v2 + 16);
    j_j___libc_free_0(v3);
  }
  v4 = *(_QWORD *)(a1 + 280);
  while ( v4 )
  {
    v5 = v4;
    sub_2C91AF0(*(_QWORD *)(v4 + 24));
    v6 = *(unsigned int *)(v4 + 64);
    v7 = *(_QWORD *)(v4 + 48);
    v4 = *(_QWORD *)(v4 + 16);
    sub_C7D6A0(v7, 16 * v6, 8);
    j_j___libc_free_0(v5);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 240), 8LL * *(unsigned int *)(a1 + 256), 8);
  sub_C7D6A0(*(_QWORD *)(a1 + 152), 32LL * *(unsigned int *)(a1 + 168), 8);
  sub_C7D6A0(*(_QWORD *)(a1 + 120), 32LL * *(unsigned int *)(a1 + 136), 8);
  v8 = *(_QWORD *)(a1 + 80);
  while ( v8 )
  {
    sub_2C91750(*(_QWORD *)(v8 + 24));
    v9 = v8;
    v8 = *(_QWORD *)(v8 + 16);
    j_j___libc_free_0(v9);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 40), 16LL * *(unsigned int *)(a1 + 56), 8);
  v10 = *(unsigned int *)(a1 + 24);
  if ( (_DWORD)v10 )
  {
    v11 = *(_QWORD **)(a1 + 8);
    v12 = &v11[4 * v10];
    do
    {
      if ( *v11 != -4096 && *v11 != -8192 )
      {
        v13 = v11[1];
        if ( (_QWORD *)v13 != v11 + 3 )
          _libc_free(v13);
      }
      v11 += 4;
    }
    while ( v12 != v11 );
    v10 = *(unsigned int *)(a1 + 24);
  }
  return sub_C7D6A0(*(_QWORD *)(a1 + 8), 32 * v10, 8);
}
