// Function: sub_EE7F30
// Address: 0xee7f30
//
__int64 __fastcall sub_EE7F30(__int64 a1, __int64 *a2, __int64 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  char v8; // al
  __int64 v9; // rsi
  __int64 *v10; // rsi
  _QWORD *v11; // rax
  __int64 v12; // r12
  __int64 *v13; // rax
  __int64 v14; // rax
  __int64 v16; // rax
  __int64 v17; // rcx
  __int64 v18; // rdx
  __int64 *v19; // rdx
  char v20; // [rsp+Fh] [rbp-E1h]
  __int64 *v21; // [rsp+28h] [rbp-C8h] BYREF
  __int64 v22[2]; // [rsp+30h] [rbp-C0h] BYREF
  _QWORD v23[22]; // [rsp+40h] [rbp-B0h] BYREF

  v8 = *(_BYTE *)(a1 + 129);
  v9 = *a3;
  v22[0] = (__int64)v23;
  v20 = v8;
  v23[1] = *a2;
  v22[1] = 0x2000000004LL;
  v23[0] = 29;
  sub_D953B0((__int64)v22, v9, (__int64)a3, a4, (__int64)v23, a6);
  v10 = v22;
  v11 = sub_C65B40(a1 + 96, (__int64)v22, (__int64 *)&v21, (__int64)off_497B2F0);
  v12 = (__int64)v11;
  if ( v11 )
  {
    v12 = (__int64)(v11 + 1);
    if ( (_QWORD *)v22[0] != v23 )
      _libc_free(v22[0], v22);
    v22[0] = v12;
    v13 = sub_EE6840(a1 + 136, v22);
    if ( v13 )
    {
      v14 = v13[1];
      if ( v14 )
        v12 = v14;
    }
    if ( *(_QWORD *)(a1 + 120) == v12 )
      *(_BYTE *)(a1 + 128) = 1;
  }
  else
  {
    if ( v20 )
    {
      v16 = sub_CD1D40((__int64 *)a1, 40, 3);
      *(_QWORD *)v16 = 0;
      v10 = (__int64 *)v16;
      v12 = v16 + 8;
      v17 = *a2;
      v18 = *a3;
      *(_WORD *)(v16 + 16) = 16413;
      LOBYTE(v16) = *(_BYTE *)(v16 + 18);
      v10[3] = v17;
      v10[4] = v18;
      v19 = v21;
      *((_BYTE *)v10 + 18) = v16 & 0xF0 | 5;
      v10[1] = (__int64)&unk_49DF908;
      sub_C657C0((__int64 *)(a1 + 96), v10, v19, (__int64)off_497B2F0);
    }
    if ( (_QWORD *)v22[0] != v23 )
      _libc_free(v22[0], v10);
    *(_QWORD *)(a1 + 112) = v12;
  }
  return v12;
}
