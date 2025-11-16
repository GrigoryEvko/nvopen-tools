// Function: sub_EE8700
// Address: 0xee8700
//
__int64 __fastcall sub_EE8700(__int64 a1, int *a2)
{
  char v3; // al
  __int64 v4; // rax
  __int64 *v5; // rsi
  _QWORD *v6; // rax
  __int64 v7; // r12
  __int64 *v8; // rax
  __int64 v9; // rax
  int v11; // eax
  bool v12; // zf
  __int64 *v13; // rdx
  char v14; // [rsp+7h] [rbp-D9h]
  __int64 *v15; // [rsp+18h] [rbp-C8h] BYREF
  __int64 v16[2]; // [rsp+20h] [rbp-C0h] BYREF
  _QWORD v17[22]; // [rsp+30h] [rbp-B0h] BYREF

  v3 = *(_BYTE *)(a1 + 129);
  v16[0] = (__int64)v17;
  v14 = v3;
  v4 = *a2;
  v5 = v16;
  v17[0] = 73;
  v17[1] = v4;
  v16[1] = 0x2000000004LL;
  v6 = sub_C65B40(a1 + 96, (__int64)v16, (__int64 *)&v15, (__int64)off_497B2F0);
  v7 = (__int64)v6;
  if ( v6 )
  {
    v7 = (__int64)(v6 + 1);
    if ( (_QWORD *)v16[0] != v17 )
      _libc_free(v16[0], v16);
    v16[0] = v7;
    v8 = sub_EE6840(a1 + 136, v16);
    if ( v8 )
    {
      v9 = v8[1];
      if ( v9 )
        v7 = v9;
    }
    if ( *(_QWORD *)(a1 + 120) == v7 )
      *(_BYTE *)(a1 + 128) = 1;
  }
  else
  {
    if ( v14 )
    {
      v5 = (__int64 *)sub_CD1D40((__int64 *)a1, 24, 3);
      *v5 = 0;
      v11 = *a2;
      *((_WORD *)v5 + 8) = 16457;
      v7 = (__int64)(v5 + 1);
      v12 = v11 == 0;
      LOBYTE(v11) = *((_BYTE *)v5 + 18) & 0xF0;
      *((_BYTE *)v5 + 19) = !v12;
      v13 = v15;
      *((_BYTE *)v5 + 18) = v11 | 5;
      v5[1] = (__int64)&unk_49E09E8;
      sub_C657C0((__int64 *)(a1 + 96), v5, v13, (__int64)off_497B2F0);
    }
    if ( (_QWORD *)v16[0] != v17 )
      _libc_free(v16[0], v5);
    *(_QWORD *)(a1 + 112) = v7;
  }
  return v7;
}
