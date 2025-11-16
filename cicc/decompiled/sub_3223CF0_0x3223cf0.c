// Function: sub_3223CF0
// Address: 0x3223cf0
//
__int64 __fastcall sub_3223CF0(__int64 a1)
{
  unsigned __int64 v2; // rdi
  __int64 v3; // rsi
  _QWORD *v4; // rbx
  _QWORD *v5; // r13
  __int64 v6; // rdi
  unsigned __int64 v7; // rdi
  unsigned __int64 v8; // rdi
  unsigned __int64 v9; // r8
  __int64 v10; // r13
  __int64 v11; // r13
  __int64 v12; // rbx
  _QWORD *v13; // rdi
  __int64 v14; // r13
  unsigned __int64 v15; // r8
  __int64 v16; // r13
  __int64 v17; // rbx
  _QWORD *v18; // rdi

  *(_QWORD *)a1 = &unk_4A3D3C8;
  v2 = *(_QWORD *)(a1 + 760);
  if ( v2 )
    j_j___libc_free_0(v2);
  v3 = *(unsigned int *)(a1 + 728);
  if ( (_DWORD)v3 )
  {
    v4 = *(_QWORD **)(a1 + 712);
    v5 = &v4[2 * v3];
    do
    {
      if ( *v4 != -4096 && *v4 != -8192 )
      {
        v6 = v4[1];
        if ( v6 )
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v6 + 8LL))(v6);
      }
      v4 += 2;
    }
    while ( v5 != v4 );
    v3 = *(unsigned int *)(a1 + 728);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 712), 16 * v3, 8);
  sub_C7D6A0(*(_QWORD *)(a1 + 680), 16LL * *(unsigned int *)(a1 + 696), 8);
  sub_C7D6A0(*(_QWORD *)(a1 + 648), 16LL * *(unsigned int *)(a1 + 664), 8);
  v7 = *(_QWORD *)(a1 + 592);
  if ( v7 != a1 + 608 )
    _libc_free(v7);
  if ( !*(_BYTE *)(a1 + 556) )
    _libc_free(*(_QWORD *)(a1 + 536));
  v8 = *(_QWORD *)(a1 + 472);
  if ( v8 != a1 + 488 )
    _libc_free(v8);
  v9 = *(_QWORD *)(a1 + 448);
  if ( *(_DWORD *)(a1 + 460) )
  {
    v10 = *(unsigned int *)(a1 + 456);
    if ( (_DWORD)v10 )
    {
      v11 = 8 * v10;
      v12 = 0;
      do
      {
        v13 = *(_QWORD **)(v9 + v12);
        if ( v13 != (_QWORD *)-8LL && v13 )
        {
          sub_C7D6A0((__int64)v13, *v13 + 17LL, 8);
          v9 = *(_QWORD *)(a1 + 448);
        }
        v12 += 8;
      }
      while ( v11 != v12 );
    }
  }
  _libc_free(v9);
  if ( *(_DWORD *)(a1 + 436) )
  {
    v14 = *(unsigned int *)(a1 + 432);
    v15 = *(_QWORD *)(a1 + 424);
    if ( (_DWORD)v14 )
    {
      v16 = 8 * v14;
      v17 = 0;
      do
      {
        v18 = *(_QWORD **)(v15 + v17);
        if ( v18 != (_QWORD *)-8LL && v18 )
        {
          sub_C7D6A0((__int64)v18, *v18 + 17LL, 8);
          v15 = *(_QWORD *)(a1 + 424);
        }
        v17 += 8;
      }
      while ( v16 != v17 );
    }
  }
  else
  {
    v15 = *(_QWORD *)(a1 + 424);
  }
  _libc_free(v15);
  return sub_32478E0(a1);
}
