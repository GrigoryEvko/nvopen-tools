// Function: sub_194A930
// Address: 0x194a930
//
__int64 __fastcall sub_194A930(__int64 a1)
{
  __int64 v2; // rax
  unsigned __int64 v3; // rdi
  __int64 v4; // rsi
  unsigned __int64 v5; // rdi
  __int64 v6; // rax
  _QWORD *v7; // rbx
  _QWORD *v8; // r13
  __int64 v9; // rax

  v2 = a1 + 352;
  v3 = *(_QWORD *)(a1 + 336);
  if ( v3 != v2 )
    _libc_free(v3);
  v4 = *(_QWORD *)(a1 + 264);
  if ( v4 )
    sub_161E7C0(a1 + 264, v4);
  j___libc_free_0(*(_QWORD *)(a1 + 232));
  v5 = *(_QWORD *)(a1 + 168);
  if ( v5 != *(_QWORD *)(a1 + 160) )
    _libc_free(v5);
  j___libc_free_0(*(_QWORD *)(a1 + 128));
  j___libc_free_0(*(_QWORD *)(a1 + 96));
  j___libc_free_0(*(_QWORD *)(a1 + 64));
  v6 = *(unsigned int *)(a1 + 48);
  if ( (_DWORD)v6 )
  {
    v7 = *(_QWORD **)(a1 + 32);
    v8 = &v7[5 * v6];
    do
    {
      while ( *v7 == -8 )
      {
        if ( v7[1] != -8 )
          goto LABEL_10;
        v7 += 5;
        if ( v8 == v7 )
          return j___libc_free_0(*(_QWORD *)(a1 + 32));
      }
      if ( *v7 != -16 || v7[1] != -16 )
      {
LABEL_10:
        v9 = v7[4];
        if ( v9 != 0 && v9 != -8 && v9 != -16 )
          sub_1649B30(v7 + 2);
      }
      v7 += 5;
    }
    while ( v8 != v7 );
  }
  return j___libc_free_0(*(_QWORD *)(a1 + 32));
}
