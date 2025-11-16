// Function: sub_124BC90
// Address: 0x124bc90
//
__int64 __fastcall sub_124BC90(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  __int64 v4; // rsi
  _QWORD *v5; // rbx
  _QWORD *v6; // r12
  __int64 v7; // rdi
  __int64 v8; // rsi
  __int64 v9; // rdi

  *(_QWORD *)a1 = &unk_49E66A8;
  v3 = *(_QWORD *)(a1 + 208);
  if ( v3 != a1 + 224 )
    _libc_free(v3, a2);
  sub_C7D6A0(*(_QWORD *)(a1 + 176), 16LL * *(unsigned int *)(a1 + 192), 8);
  v4 = *(unsigned int *)(a1 + 160);
  if ( (_DWORD)v4 )
  {
    v5 = *(_QWORD **)(a1 + 144);
    v6 = &v5[4 * v4];
    do
    {
      if ( *v5 != -8192 && *v5 != -4096 )
      {
        v7 = v5[1];
        if ( v7 )
          j_j___libc_free_0(v7, v5[3] - v7);
      }
      v5 += 4;
    }
    while ( v6 != v5 );
    v4 = *(unsigned int *)(a1 + 160);
  }
  v8 = 32 * v4;
  sub_C7D6A0(*(_QWORD *)(a1 + 144), v8, 8);
  v9 = *(_QWORD *)(a1 + 112);
  if ( v9 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v9 + 8LL))(v9);
  sub_E8EC10(a1, v8);
  return j_j___libc_free_0(a1, 224);
}
