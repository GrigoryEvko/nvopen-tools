// Function: sub_3584890
// Address: 0x3584890
//
__int64 __fastcall sub_3584890(__int64 a1)
{
  volatile signed __int32 *v2; // rdi
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // r13
  __int64 v6; // rdi
  __int64 v7; // rax
  _QWORD *v8; // r13
  _QWORD *v9; // r14
  unsigned __int64 v10; // rbx
  unsigned __int64 v11; // rdi
  __int64 v12; // rax
  _QWORD *v13; // rbx
  _QWORD *v14; // r13
  unsigned __int64 v15; // rdi
  __int64 v16; // rax
  _QWORD *v17; // rbx
  _QWORD *v18; // r13
  unsigned __int64 v19; // rdi
  unsigned __int64 v20; // rdi

  *(_QWORD *)a1 = &unk_4A398D8;
  v2 = *(volatile signed __int32 **)(a1 + 1272);
  if ( v2 && !_InterlockedSub(v2 + 2, 1u) )
    (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v2 + 8LL))(v2);
  v3 = *(_QWORD *)(a1 + 1240);
  if ( v3 != a1 + 1256 )
    j_j___libc_free_0(v3);
  v4 = *(_QWORD *)(a1 + 1208);
  if ( v4 != a1 + 1224 )
    j_j___libc_free_0(v4);
  v5 = *(_QWORD *)(a1 + 1192);
  if ( v5 )
  {
    sub_C7D6A0(*(_QWORD *)(v5 + 8), 24LL * *(unsigned int *)(v5 + 24), 8);
    j_j___libc_free_0(v5);
  }
  sub_3584510(*(_QWORD *)(a1 + 1160));
  v6 = *(_QWORD *)(a1 + 1136);
  if ( v6 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v6 + 8LL))(v6);
  v7 = *(unsigned int *)(a1 + 1112);
  if ( (_DWORD)v7 )
  {
    v8 = *(_QWORD **)(a1 + 1096);
    v9 = &v8[7 * v7];
    do
    {
      if ( *v8 != -4096 && *v8 != -8192 )
      {
        v10 = v8[3];
        while ( v10 )
        {
          sub_3583DE0(*(_QWORD *)(v10 + 24));
          v11 = v10;
          v10 = *(_QWORD *)(v10 + 16);
          j_j___libc_free_0(v11);
        }
      }
      v8 += 7;
    }
    while ( v9 != v8 );
    v7 = *(unsigned int *)(a1 + 1112);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 1096), 56 * v7, 8);
  v12 = *(unsigned int *)(a1 + 1080);
  if ( (_DWORD)v12 )
  {
    v13 = *(_QWORD **)(a1 + 1064);
    v14 = &v13[11 * v12];
    do
    {
      if ( *v13 != -4096 && *v13 != -8192 )
      {
        v15 = v13[1];
        if ( (_QWORD *)v15 != v13 + 3 )
          _libc_free(v15);
      }
      v13 += 11;
    }
    while ( v14 != v13 );
    v12 = *(unsigned int *)(a1 + 1080);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 1064), 88 * v12, 8);
  v16 = *(unsigned int *)(a1 + 1048);
  if ( (_DWORD)v16 )
  {
    v17 = *(_QWORD **)(a1 + 1032);
    v18 = &v17[11 * v16];
    do
    {
      if ( *v17 != -8192 && *v17 != -4096 )
      {
        v19 = v17[1];
        if ( (_QWORD *)v19 != v17 + 3 )
          _libc_free(v19);
      }
      v17 += 11;
    }
    while ( v18 != v17 );
    v16 = *(unsigned int *)(a1 + 1048);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 1032), 88 * v16, 8);
  sub_C7D6A0(*(_QWORD *)(a1 + 976), 16LL * *(unsigned int *)(a1 + 992), 8);
  sub_3583FB0(*(_QWORD *)(a1 + 936));
  v20 = *(_QWORD *)(a1 + 392);
  if ( v20 != a1 + 408 )
    _libc_free(v20);
  if ( !*(_BYTE *)(a1 + 132) )
    _libc_free(*(_QWORD *)(a1 + 112));
  sub_C7D6A0(*(_QWORD *)(a1 + 80), 24LL * *(unsigned int *)(a1 + 96), 8);
  sub_C7D6A0(*(_QWORD *)(a1 + 48), 16LL * *(unsigned int *)(a1 + 64), 8);
  return sub_C7D6A0(*(_QWORD *)(a1 + 16), 16LL * *(unsigned int *)(a1 + 32), 8);
}
