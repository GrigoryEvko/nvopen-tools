// Function: sub_2D05B80
// Address: 0x2d05b80
//
void __fastcall sub_2D05B80(__int64 a1)
{
  unsigned __int64 v2; // rdi
  _QWORD *v3; // rbx
  unsigned __int64 v4; // r12
  unsigned __int64 v5; // rdi
  _QWORD *v6; // rbx
  unsigned __int64 v7; // rdi
  unsigned __int64 v8; // r12
  unsigned __int64 v9; // r14
  __int64 v10; // rax
  _QWORD *v11; // rbx
  _QWORD *v12; // r12
  unsigned __int64 v13; // r13
  unsigned __int64 v14; // rdi
  unsigned __int64 v15; // rdi
  unsigned __int64 v16; // rdi

  v2 = *(_QWORD *)(a1 + 280);
  if ( v2 )
    j_j___libc_free_0(v2);
  sub_C7D6A0(*(_QWORD *)(a1 + 256), 16LL * *(unsigned int *)(a1 + 272), 8);
  if ( !*(_BYTE *)(a1 + 180) )
    _libc_free(*(_QWORD *)(a1 + 160));
  v3 = *(_QWORD **)(a1 + 128);
  while ( v3 != (_QWORD *)(a1 + 128) )
  {
    v4 = (unsigned __int64)v3;
    v3 = (_QWORD *)*v3;
    v5 = *(_QWORD *)(v4 + 152);
    if ( v5 != v4 + 168 )
      _libc_free(v5);
    if ( !*(_BYTE *)(v4 + 84) )
      _libc_free(*(_QWORD *)(v4 + 64));
    j_j___libc_free_0(v4);
  }
  v6 = *(_QWORD **)(a1 + 104);
  while ( v6 != (_QWORD *)(a1 + 104) )
  {
    v8 = (unsigned __int64)v6;
    v6 = (_QWORD *)*v6;
    if ( !*(_BYTE *)(v8 + 172) )
      _libc_free(*(_QWORD *)(v8 + 152));
    v7 = *(_QWORD *)(v8 + 64);
    if ( v7 != v8 + 80 )
      _libc_free(v7);
    j_j___libc_free_0(v8);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 80), 16LL * *(unsigned int *)(a1 + 96), 8);
  v9 = *(_QWORD *)(a1 + 48);
  if ( v9 )
  {
    if ( !*(_BYTE *)(v9 + 204) )
      _libc_free(*(_QWORD *)(v9 + 184));
    sub_C7D6A0(*(_QWORD *)(v9 + 152), 16LL * *(unsigned int *)(v9 + 168), 8);
    v10 = *(unsigned int *)(v9 + 136);
    if ( (_DWORD)v10 )
    {
      v11 = *(_QWORD **)(v9 + 120);
      v12 = &v11[2 * v10];
      do
      {
        if ( *v11 != -4096 && *v11 != -8192 )
        {
          v13 = v11[1];
          if ( v13 )
          {
            v14 = *(_QWORD *)(v13 + 96);
            if ( v14 != v13 + 112 )
              _libc_free(v14);
            v15 = *(_QWORD *)(v13 + 24);
            if ( v15 != v13 + 40 )
              _libc_free(v15);
            j_j___libc_free_0(v13);
          }
        }
        v11 += 2;
      }
      while ( v12 != v11 );
      LODWORD(v10) = *(_DWORD *)(v9 + 136);
    }
    sub_C7D6A0(*(_QWORD *)(v9 + 120), 16LL * (unsigned int)v10, 8);
    v16 = *(_QWORD *)(v9 + 88);
    if ( v16 )
      j_j___libc_free_0(v16);
    sub_C7D6A0(*(_QWORD *)(v9 + 64), 16LL * *(unsigned int *)(v9 + 80), 8);
    j_j___libc_free_0(v9);
  }
}
