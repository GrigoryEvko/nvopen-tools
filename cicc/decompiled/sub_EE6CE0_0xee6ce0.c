// Function: sub_EE6CE0
// Address: 0xee6ce0
//
__int64 __fastcall sub_EE6CE0(__int64 a1, __int64 *a2)
{
  __int64 *v3; // rsi
  char v4; // al
  _QWORD *v5; // rax
  __int64 v6; // r12
  __int64 *v7; // rax
  __int64 v8; // rax
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 *v12; // rdx
  char v13; // [rsp+7h] [rbp-D9h]
  __int64 *v14; // [rsp+18h] [rbp-C8h] BYREF
  __int64 v15[2]; // [rsp+20h] [rbp-C0h] BYREF
  _QWORD v16[22]; // [rsp+30h] [rbp-B0h] BYREF

  v3 = v15;
  v4 = *(_BYTE *)(a1 + 129);
  v15[0] = (__int64)v16;
  v16[0] = 42;
  v13 = v4;
  v16[1] = *a2;
  v15[1] = 0x2000000004LL;
  v5 = sub_C65B40(a1 + 96, (__int64)v15, (__int64 *)&v14, (__int64)off_497B2F0);
  v6 = (__int64)v5;
  if ( v5 )
  {
    v6 = (__int64)(v5 + 1);
    if ( (_QWORD *)v15[0] != v16 )
      _libc_free(v15[0], v15);
    v15[0] = v6;
    v7 = sub_EE6840(a1 + 136, v15);
    if ( v7 )
    {
      v8 = v7[1];
      if ( v8 )
        v6 = v8;
    }
    if ( *(_QWORD *)(a1 + 120) == v6 )
      *(_BYTE *)(a1 + 128) = 1;
  }
  else
  {
    if ( v13 )
    {
      v10 = sub_CD1D40((__int64 *)a1, 32, 3);
      *(_QWORD *)v10 = 0;
      v3 = (__int64 *)v10;
      v6 = v10 + 8;
      v11 = *a2;
      *(_WORD *)(v10 + 16) = 16426;
      LOBYTE(v10) = *(_BYTE *)(v10 + 18);
      v3[3] = v11;
      v12 = v14;
      *((_BYTE *)v3 + 18) = v10 & 0xF0 | 5;
      v3[1] = (__int64)&unk_49DFD88;
      sub_C657C0((__int64 *)(a1 + 96), v3, v12, (__int64)off_497B2F0);
    }
    if ( (_QWORD *)v15[0] != v16 )
      _libc_free(v15[0], v3);
    *(_QWORD *)(a1 + 112) = v6;
  }
  return v6;
}
