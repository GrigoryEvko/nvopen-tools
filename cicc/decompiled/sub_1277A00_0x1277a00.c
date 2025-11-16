// Function: sub_1277A00
// Address: 0x1277a00
//
__int64 __fastcall sub_1277A00(__int64 a1)
{
  __int64 v2; // rdi
  __int64 v3; // rdx
  __int64 v4; // rsi
  __int64 v5; // rbx
  __int64 v6; // r12
  __int64 v7; // rdi
  __int64 v8; // rdi
  __int64 v9; // r12
  __int64 v10; // rdi
  __int64 v12; // rdi
  __int64 v13; // [rsp+0h] [rbp-30h] BYREF
  _QWORD v14[5]; // [rsp+8h] [rbp-28h] BYREF

  v2 = *(_QWORD *)(a1 + 8);
  if ( v2 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v2 + 8LL))(v2);
  *(_QWORD *)(a1 + 8) = 0;
  sub_16BDD10(v14, *(_QWORD *)(a1 + 144));
  v3 = *(unsigned int *)(a1 + 152);
  v13 = v14[0];
  v4 = *(_QWORD *)(a1 + 144) + 8 * v3;
  sub_16BDD10(v14, v4);
  v5 = v14[0];
  while ( 1 )
  {
    v6 = v13;
    if ( v13 == v5 )
      break;
    sub_16BDD40(&v13);
    if ( v6 )
    {
      v12 = *(_QWORD *)(v6 + 16);
      if ( v12 )
        j_j___libc_free_0_0(v12);
      v4 = 24;
      j_j___libc_free_0(v6, 24);
    }
  }
  j___libc_free_0(*(_QWORD *)(a1 + 320));
  v7 = *(_QWORD *)(a1 + 232);
  if ( v7 != a1 + 248 )
    _libc_free(v7, v4);
  v8 = *(_QWORD *)(a1 + 176);
  if ( v8 != *(_QWORD *)(a1 + 168) )
    _libc_free(v8, v4);
  *(_QWORD *)(a1 + 136) = &unk_49E6990;
  sub_16BD9D0(a1 + 136);
  j___libc_free_0(*(_QWORD *)(a1 + 112));
  v9 = *(_QWORD *)(a1 + 72);
  while ( v9 )
  {
    sub_1277600(*(_QWORD *)(v9 + 24));
    v10 = v9;
    v9 = *(_QWORD *)(v9 + 16);
    j_j___libc_free_0(v10, 56);
  }
  return j___libc_free_0(*(_QWORD *)(a1 + 32));
}
