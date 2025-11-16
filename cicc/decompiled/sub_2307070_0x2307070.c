// Function: sub_2307070
// Address: 0x2307070
//
__int64 __fastcall sub_2307070(__int64 a1)
{
  bool v2; // zf
  __int64 v3; // rsi
  _QWORD *v4; // rbx
  _QWORD *v5; // r12
  unsigned __int64 v6; // r13
  unsigned __int64 v7; // rdi
  unsigned __int64 v8; // rdi
  unsigned __int64 v9; // rdi

  v2 = *(_BYTE *)(a1 + 212) == 0;
  *(_QWORD *)a1 = &unk_4A0ACF0;
  if ( v2 )
    _libc_free(*(_QWORD *)(a1 + 192));
  sub_C7D6A0(*(_QWORD *)(a1 + 160), 16LL * *(unsigned int *)(a1 + 176), 8);
  v3 = *(unsigned int *)(a1 + 144);
  if ( (_DWORD)v3 )
  {
    v4 = *(_QWORD **)(a1 + 128);
    v5 = &v4[2 * v3];
    do
    {
      if ( *v4 != -8192 && *v4 != -4096 )
      {
        v6 = v4[1];
        if ( v6 )
        {
          v7 = *(_QWORD *)(v6 + 96);
          if ( v7 != v6 + 112 )
            _libc_free(v7);
          v8 = *(_QWORD *)(v6 + 24);
          if ( v8 != v6 + 40 )
            _libc_free(v8);
          j_j___libc_free_0(v6);
        }
      }
      v4 += 2;
    }
    while ( v5 != v4 );
    v3 = *(unsigned int *)(a1 + 144);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 128), 16 * v3, 8);
  v9 = *(_QWORD *)(a1 + 96);
  if ( v9 )
    j_j___libc_free_0(v9);
  return sub_C7D6A0(*(_QWORD *)(a1 + 72), 16LL * *(unsigned int *)(a1 + 88), 8);
}
