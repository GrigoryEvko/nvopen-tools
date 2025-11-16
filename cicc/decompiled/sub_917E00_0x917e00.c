// Function: sub_917E00
// Address: 0x917e00
//
__int64 __fastcall sub_917E00(__int64 a1)
{
  __int64 v2; // rdi
  __int64 v3; // rdx
  __int64 v4; // r13
  __int64 v5; // r12
  __int64 v6; // rsi
  __int64 v7; // rdi
  __int64 v8; // r12
  __int64 v9; // rdi
  __int64 v11; // rdi
  __int64 v12; // [rsp+0h] [rbp-30h] BYREF
  _QWORD v13[5]; // [rsp+8h] [rbp-28h] BYREF

  v2 = *(_QWORD *)(a1 + 8);
  if ( v2 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v2 + 8LL))(v2);
  *(_QWORD *)(a1 + 8) = 0;
  sub_C65AC0(v13, *(_QWORD *)(a1 + 136));
  v3 = *(unsigned int *)(a1 + 144);
  v12 = v13[0];
  sub_C65AC0(v13, *(_QWORD *)(a1 + 136) + 8 * v3);
  v4 = v13[0];
  while ( 1 )
  {
    v5 = v12;
    if ( v12 == v4 )
      break;
    sub_C65AF0(&v12);
    if ( v5 )
    {
      v11 = *(_QWORD *)(v5 + 16);
      if ( v11 )
        j_j___libc_free_0_0(v11);
      j_j___libc_free_0(v5, 24);
    }
  }
  v6 = 16LL * *(unsigned int *)(a1 + 320);
  sub_C7D6A0(*(_QWORD *)(a1 + 304), v6, 8);
  v7 = *(_QWORD *)(a1 + 216);
  if ( v7 != a1 + 232 )
    _libc_free(v7, v6);
  if ( !*(_BYTE *)(a1 + 180) )
    _libc_free(*(_QWORD *)(a1 + 160), v6);
  sub_C65770(a1 + 136);
  sub_C7D6A0(*(_QWORD *)(a1 + 112), 16LL * *(unsigned int *)(a1 + 128), 8);
  v8 = *(_QWORD *)(a1 + 72);
  while ( v8 )
  {
    sub_917B10(*(_QWORD *)(v8 + 24));
    v9 = v8;
    v8 = *(_QWORD *)(v8 + 16);
    j_j___libc_free_0(v9, 56);
  }
  return sub_C7D6A0(*(_QWORD *)(a1 + 32), 16LL * *(unsigned int *)(a1 + 48), 8);
}
