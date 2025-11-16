// Function: sub_EE7130
// Address: 0xee7130
//
__int64 __fastcall sub_EE7130(__int64 a1, char *a2, __int64 *a3)
{
  __int64 v4; // r15
  size_t v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 *v10; // rsi
  _QWORD *v11; // rax
  __int64 v12; // r15
  __int64 *v13; // rax
  __int64 v14; // rax
  __int64 *v16; // rax
  __int64 *v17; // r13
  size_t v18; // rax
  __int64 v19; // rdx
  __int64 *v20; // rdx
  char v22; // [rsp+17h] [rbp-D9h]
  __int64 *v23; // [rsp+28h] [rbp-C8h] BYREF
  __int64 v24[2]; // [rsp+30h] [rbp-C0h] BYREF
  _QWORD v25[22]; // [rsp+40h] [rbp-B0h] BYREF

  v4 = *a3;
  v22 = *(_BYTE *)(a1 + 129);
  v24[0] = (__int64)v25;
  v24[1] = 0x2000000002LL;
  v25[0] = 60;
  v5 = strlen(a2);
  if ( v5 )
    sub_C653C0((__int64)v24, (unsigned __int8 *)a2, v5);
  else
    sub_C653C0((__int64)v24, 0, 0);
  sub_D953B0((__int64)v24, v4, v6, v7, v8, v9);
  v10 = v24;
  v11 = sub_C65B40(a1 + 96, (__int64)v24, (__int64 *)&v23, (__int64)off_497B2F0);
  v12 = (__int64)v11;
  if ( v11 )
  {
    v12 = (__int64)(v11 + 1);
    if ( (_QWORD *)v24[0] != v25 )
      _libc_free(v24[0], v24);
    v24[0] = v12;
    v13 = sub_EE6840(a1 + 136, v24);
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
    if ( v22 )
    {
      v16 = (__int64 *)sub_CD1D40((__int64 *)a1, 64, 3);
      *v16 = 0;
      v17 = v16;
      v12 = (__int64)(v16 + 1);
      v18 = strlen(a2);
      v10 = v17;
      v19 = *a3;
      v17[3] = v18;
      *((_WORD *)v17 + 8) = 16444;
      LOBYTE(v18) = *((_BYTE *)v17 + 18);
      v17[5] = v19;
      v20 = v23;
      v17[4] = (__int64)a2;
      v17[6] = 0;
      *((_BYTE *)v17 + 18) = v18 & 0xF0 | 5;
      v17[7] = 0;
      v17[1] = (__int64)&unk_49E0448;
      sub_C657C0((__int64 *)(a1 + 96), v17, v20, (__int64)off_497B2F0);
    }
    if ( (_QWORD *)v24[0] != v25 )
      _libc_free(v24[0], v10);
    *(_QWORD *)(a1 + 112) = v12;
  }
  return v12;
}
