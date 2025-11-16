// Function: sub_EE6A90
// Address: 0xee6a90
//
__int64 __fastcall sub_EE6A90(__int64 a1, __int64 *a2)
{
  char v3; // al
  __int64 v4; // rdx
  unsigned __int8 *v5; // rsi
  __int64 *v6; // rsi
  _QWORD *v7; // rax
  __int64 v8; // r12
  __int64 *v9; // rax
  __int64 v10; // rax
  __int64 v12; // rax
  __int64 v13; // rcx
  __int64 v14; // rdx
  __int64 *v15; // rdx
  char v16; // [rsp+7h] [rbp-D9h]
  __int64 *v17; // [rsp+18h] [rbp-C8h] BYREF
  __int64 v18[2]; // [rsp+20h] [rbp-C0h] BYREF
  _QWORD v19[22]; // [rsp+30h] [rbp-B0h] BYREF

  v3 = *(_BYTE *)(a1 + 129);
  v4 = *a2;
  v18[0] = (__int64)v19;
  v19[0] = 8;
  v5 = (unsigned __int8 *)a2[1];
  v16 = v3;
  v18[1] = 0x2000000002LL;
  if ( v4 )
    sub_C653C0((__int64)v18, v5, v4);
  else
    sub_C653C0((__int64)v18, 0, 0);
  v6 = v18;
  v7 = sub_C65B40(a1 + 96, (__int64)v18, (__int64 *)&v17, (__int64)off_497B2F0);
  v8 = (__int64)v7;
  if ( v7 )
  {
    v8 = (__int64)(v7 + 1);
    if ( (_QWORD *)v18[0] != v19 )
      _libc_free(v18[0], v18);
    v18[0] = v8;
    v9 = sub_EE6840(a1 + 136, v18);
    if ( v9 )
    {
      v10 = v9[1];
      if ( v10 )
        v8 = v10;
    }
    if ( *(_QWORD *)(a1 + 120) == v8 )
      *(_BYTE *)(a1 + 128) = 1;
  }
  else
  {
    if ( v16 )
    {
      v12 = sub_CD1D40((__int64 *)a1, 40, 3);
      *(_QWORD *)v12 = 0;
      v6 = (__int64 *)v12;
      v8 = v12 + 8;
      v13 = *a2;
      v14 = a2[1];
      *(_WORD *)(v12 + 16) = 16392;
      LOBYTE(v12) = *(_BYTE *)(v12 + 18);
      v6[3] = v13;
      v6[4] = v14;
      v15 = v17;
      *((_BYTE *)v6 + 18) = v12 & 0xF0 | 5;
      v6[1] = (__int64)&unk_49DEFA8;
      sub_C657C0((__int64 *)(a1 + 96), v6, v15, (__int64)off_497B2F0);
    }
    if ( (_QWORD *)v18[0] != v19 )
      _libc_free(v18[0], v6);
    *(_QWORD *)(a1 + 112) = v8;
  }
  return v8;
}
