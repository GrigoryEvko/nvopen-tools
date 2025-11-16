// Function: sub_34F5380
// Address: 0x34f5380
//
void __fastcall sub_34F5380(__int64 a1)
{
  __int64 v1; // r13
  __int64 v3; // rax
  __int64 v4; // r12
  __int64 v5; // r14
  unsigned __int64 v6; // rdi
  __int64 v7; // rsi
  __int64 v8; // rdi
  __int64 v9; // r14
  unsigned __int64 v10; // r12
  __int64 v11; // rsi
  __int64 v12; // r12
  __int64 v13; // r13
  unsigned __int64 *v14; // r14
  unsigned __int64 v15; // r15
  unsigned __int64 v16; // rdi
  __int64 v17; // rdi
  __int64 v18; // rbx
  unsigned __int64 v19; // rdi

  v1 = a1 + 312;
  *(_QWORD *)a1 = off_4A38748;
  v3 = *(unsigned int *)(a1 + 336);
  if ( (_DWORD)v3 )
  {
    v4 = *(_QWORD *)(a1 + 320);
    v5 = v4 + 120 * v3;
    do
    {
      while ( *(_DWORD *)v4 > 0xFFFFFFFD )
      {
        v4 += 120;
        if ( v5 == v4 )
          goto LABEL_8;
      }
      v6 = *(_QWORD *)(v4 + 40);
      if ( v6 != v4 + 56 )
        _libc_free(v6);
      v7 = *(unsigned int *)(v4 + 32);
      v8 = *(_QWORD *)(v4 + 16);
      v4 += 120;
      sub_C7D6A0(v8, 4 * v7, 4);
    }
    while ( v5 != v4 );
LABEL_8:
    v3 = *(unsigned int *)(a1 + 336);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 320), 120 * v3, 8);
  v9 = *(_QWORD *)(a1 + 296);
  v10 = v9 + 176LL * *(unsigned int *)(a1 + 304);
  if ( v9 != v10 )
  {
    do
    {
      while ( 1 )
      {
        v10 -= 176LL;
        if ( !*(_BYTE *)(v10 + 44) )
          break;
        if ( v9 == v10 )
          goto LABEL_14;
      }
      _libc_free(*(_QWORD *)(v10 + 24));
    }
    while ( v9 != v10 );
LABEL_14:
    v10 = *(_QWORD *)(a1 + 296);
  }
  if ( v1 != v10 )
    _libc_free(v10);
  sub_C7D6A0(*(_QWORD *)(a1 + 272), 24LL * *(unsigned int *)(a1 + 288), 8);
  v11 = *(unsigned int *)(a1 + 256);
  if ( (_DWORD)v11 )
  {
    v12 = *(_QWORD *)(a1 + 240);
    v13 = v12 + 16 * v11;
    do
    {
      while ( 1 )
      {
        if ( (unsigned int)(*(_DWORD *)v12 + 0x7FFFFFFF) <= 0xFFFFFFFD )
        {
          v14 = *(unsigned __int64 **)(v12 + 8);
          if ( v14 )
            break;
        }
        v12 += 16;
        if ( v13 == v12 )
          goto LABEL_29;
      }
      sub_2E0AFD0(*(_QWORD *)(v12 + 8));
      v15 = v14[12];
      if ( v15 )
      {
        sub_34F51B0(*(_QWORD *)(v15 + 16));
        j_j___libc_free_0(v15);
      }
      v16 = v14[8];
      if ( (unsigned __int64 *)v16 != v14 + 10 )
        _libc_free(v16);
      if ( (unsigned __int64 *)*v14 != v14 + 2 )
        _libc_free(*v14);
      v12 += 16;
      j_j___libc_free_0((unsigned __int64)v14);
    }
    while ( v13 != v12 );
LABEL_29:
    v11 = *(unsigned int *)(a1 + 256);
  }
  v17 = *(_QWORD *)(a1 + 240);
  v18 = a1 + 104;
  sub_C7D6A0(v17, 16 * v11, 8);
  v19 = *(_QWORD *)(v18 - 16);
  if ( v19 != v18 )
    _libc_free(v19);
}
