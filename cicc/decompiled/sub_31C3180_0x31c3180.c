// Function: sub_31C3180
// Address: 0x31c3180
//
__int64 __fastcall sub_31C3180(__int64 a1)
{
  __int64 v2; // r14
  unsigned __int64 v3; // r12
  unsigned __int64 v4; // r15
  __int64 v5; // rax
  unsigned __int64 v6; // rbx
  __int64 v7; // rdi
  __int64 v8; // r14
  unsigned __int64 v9; // r12
  unsigned __int64 v10; // r15
  __int64 v11; // rax
  unsigned __int64 v12; // rbx
  __int64 v13; // rdi
  __int64 v15; // [rsp+8h] [rbp-38h]

  sub_3187250(*(_QWORD *)(a1 + 176), *(_QWORD *)(a1 + 184));
  v15 = a1 + 136;
  sub_C7D6A0(*(_QWORD *)(a1 + 144), 16LL * *(unsigned int *)(a1 + 160), 8);
  v2 = *(_QWORD *)(a1 + 120);
  v3 = v2 + 88LL * *(unsigned int *)(a1 + 128);
  if ( v2 != v3 )
  {
    do
    {
      v4 = *(_QWORD *)(v3 - 64);
      v5 = *(unsigned int *)(v3 - 56);
      v3 -= 88LL;
      v6 = v4 + 8 * v5;
      if ( v4 != v6 )
      {
        do
        {
          v7 = *(_QWORD *)(v6 - 8);
          v6 -= 8LL;
          if ( v7 )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v7 + 8LL))(v7);
        }
        while ( v4 != v6 );
        v4 = *(_QWORD *)(v3 + 24);
      }
      if ( v4 != v3 + 40 )
        _libc_free(v4);
    }
    while ( v2 != v3 );
    v3 = *(_QWORD *)(a1 + 120);
  }
  if ( v15 != v3 )
    _libc_free(v3);
  sub_C7D6A0(*(_QWORD *)(a1 + 96), 32LL * *(unsigned int *)(a1 + 112), 8);
  sub_C7D6A0(*(_QWORD *)(a1 + 56), 16LL * *(unsigned int *)(a1 + 72), 8);
  v8 = *(_QWORD *)(a1 + 32);
  v9 = v8 + 88LL * *(unsigned int *)(a1 + 40);
  if ( v8 != v9 )
  {
    do
    {
      v10 = *(_QWORD *)(v9 - 64);
      v11 = *(unsigned int *)(v9 - 56);
      v9 -= 88LL;
      v12 = v10 + 8 * v11;
      if ( v10 != v12 )
      {
        do
        {
          v13 = *(_QWORD *)(v12 - 8);
          v12 -= 8LL;
          if ( v13 )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v13 + 8LL))(v13);
        }
        while ( v10 != v12 );
        v10 = *(_QWORD *)(v9 + 24);
      }
      if ( v10 != v9 + 40 )
        _libc_free(v10);
    }
    while ( v8 != v9 );
    v9 = *(_QWORD *)(a1 + 32);
  }
  if ( a1 + 48 != v9 )
    _libc_free(v9);
  return sub_C7D6A0(*(_QWORD *)(a1 + 8), 32LL * *(unsigned int *)(a1 + 24), 8);
}
