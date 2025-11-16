// Function: sub_EE9E40
// Address: 0xee9e40
//
__int64 __fastcall sub_EE9E40(__int64 a1, __int64 *a2, unsigned __int8 *a3, int *a4, __int64 a5, __int64 a6)
{
  unsigned __int8 v7; // r15
  __int64 v8; // rsi
  int v9; // r14d
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 *v18; // rsi
  _QWORD *v19; // rax
  __int64 v20; // r15
  __int64 *v21; // rax
  __int64 v22; // rax
  __int64 v24; // rax
  unsigned __int8 v25; // cl
  int v26; // edx
  __int64 *v27; // rdx
  char v30; // [rsp+17h] [rbp-D9h]
  __int64 *v31; // [rsp+18h] [rbp-D8h]
  __int64 *v32; // [rsp+28h] [rbp-C8h] BYREF
  __int64 v33[2]; // [rsp+30h] [rbp-C0h] BYREF
  _QWORD v34[22]; // [rsp+40h] [rbp-B0h] BYREF

  v7 = *a3;
  v8 = *a2;
  v31 = a2;
  v9 = *a4;
  v30 = *(_BYTE *)(a1 + 129);
  v33[1] = 0x2000000002LL;
  v33[0] = (__int64)v34;
  v34[0] = 49;
  sub_D953B0((__int64)v33, v8, (__int64)a3, (__int64)a4, (__int64)a2, a6);
  sub_D953B0((__int64)v33, v7, v10, v11, v12, v13);
  sub_D953B0((__int64)v33, v9, v14, v15, v16, v17);
  v18 = v33;
  v19 = sub_C65B40(a1 + 96, (__int64)v33, (__int64 *)&v32, (__int64)off_497B2F0);
  v20 = (__int64)v19;
  if ( v19 )
  {
    v20 = (__int64)(v19 + 1);
    if ( (_QWORD *)v33[0] != v34 )
      _libc_free(v33[0], v33);
    v33[0] = v20;
    v21 = sub_EE6840(a1 + 136, v33);
    if ( v21 )
    {
      v22 = v21[1];
      if ( v22 )
        v20 = v22;
    }
    if ( *(_QWORD *)(a1 + 120) == v20 )
      *(_BYTE *)(a1 + 128) = 1;
  }
  else
  {
    if ( v30 )
    {
      v24 = sub_CD1D40((__int64 *)a1, 40, 3);
      *(_QWORD *)v24 = 0;
      v18 = (__int64 *)v24;
      v20 = v24 + 8;
      v25 = *a3;
      v26 = *a4;
      *(_QWORD *)(v24 + 24) = *v31;
      *(_WORD *)(v24 + 16) = 16433;
      LOBYTE(v24) = *(_BYTE *)(v24 + 18);
      *((_BYTE *)v18 + 32) = v25;
      *((_DWORD *)v18 + 9) = v26;
      v27 = v32;
      *((_BYTE *)v18 + 18) = v24 & 0xF0 | 5;
      v18[1] = (__int64)&unk_49E0028;
      sub_C657C0((__int64 *)(a1 + 96), v18, v27, (__int64)off_497B2F0);
    }
    if ( (_QWORD *)v33[0] != v34 )
      _libc_free(v33[0], v18);
    *(_QWORD *)(a1 + 112) = v20;
  }
  return v20;
}
