// Function: sub_DA3710
// Address: 0xda3710
//
_QWORD *__fastcall sub_DA3710(__int64 a1, __int64 a2)
{
  _QWORD *v3; // rsi
  _QWORD *v4; // r12
  __int64 v6; // rdx
  __int64 v7; // rax
  void *v8; // [rsp+0h] [rbp-E0h]
  __int64 v9; // [rsp+8h] [rbp-D8h]
  __int64 *v10; // [rsp+18h] [rbp-C8h] BYREF
  _QWORD v11[2]; // [rsp+20h] [rbp-C0h] BYREF
  int v12; // [rsp+30h] [rbp-B0h] BYREF
  __int64 v13; // [rsp+34h] [rbp-ACh]

  v13 = a2;
  v3 = v11;
  v11[0] = &v12;
  v11[1] = 0x2000000003LL;
  v12 = 1;
  v10 = 0;
  v4 = sub_C65B40(a1 + 1032, (__int64)v11, (__int64 *)&v10, (__int64)off_49DEA80);
  if ( !v4 )
  {
    v8 = sub_C65D30((__int64)v11, (unsigned __int64 *)(a1 + 1064));
    v9 = v6;
    v7 = sub_A777F0(0x28u, (__int64 *)(a1 + 1064));
    v4 = (_QWORD *)v7;
    if ( v7 )
    {
      *(_QWORD *)v7 = 0;
      *(_DWORD *)(v7 + 24) = 1;
      *(_QWORD *)(v7 + 8) = v8;
      *(_QWORD *)(v7 + 16) = v9;
      *(_WORD *)(v7 + 28) = 0;
      *(_QWORD *)(v7 + 32) = a2;
    }
    v3 = (_QWORD *)v7;
    sub_C657C0((__int64 *)(a1 + 1032), (__int64 *)v7, v10, (__int64)off_49DEA80);
  }
  if ( (int *)v11[0] != &v12 )
    _libc_free(v11[0], v3);
  return v4;
}
