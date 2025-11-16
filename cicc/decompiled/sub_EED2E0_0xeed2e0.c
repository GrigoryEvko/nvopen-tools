// Function: sub_EED2E0
// Address: 0xeed2e0
//
_QWORD *__fastcall sub_EED2E0(__int64 a1, __int64 a2, unsigned __int8 *a3, int a4)
{
  __int64 v5; // rax
  __int64 v6; // r12
  char v7; // al
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 *v16; // rsi
  _QWORD *v17; // rax
  _QWORD *v18; // r8
  __int64 v19; // r8
  __int64 *v20; // rax
  __int64 v21; // rax
  __int64 v23; // rax
  __int64 *v24; // rdx
  char v25; // [rsp+0h] [rbp-F0h]
  __int64 v27; // [rsp+10h] [rbp-E0h]
  _QWORD *v29; // [rsp+18h] [rbp-D8h]
  __int64 v30; // [rsp+18h] [rbp-D8h]
  _QWORD *v31; // [rsp+18h] [rbp-D8h]
  __int64 *v32; // [rsp+28h] [rbp-C8h] BYREF
  __int64 v33[2]; // [rsp+30h] [rbp-C0h] BYREF
  _QWORD v34[22]; // [rsp+40h] [rbp-B0h] BYREF

  v5 = sub_EEA9F0(a1);
  if ( !v5 )
    return 0;
  v6 = v5;
  v7 = *(_BYTE *)(a1 + 937);
  v34[0] = 66;
  v33[0] = (__int64)v34;
  v25 = v7;
  v33[1] = 0x2000000002LL;
  if ( a2 )
    sub_C653C0((__int64)v33, a3, a2);
  else
    sub_C653C0((__int64)v33, 0, 0);
  sub_D953B0((__int64)v33, v6, v8, v9, v10, v11);
  sub_D953B0((__int64)v33, a4, v12, v13, v14, v15);
  v16 = v33;
  v17 = sub_C65B40(a1 + 904, (__int64)v33, (__int64 *)&v32, (__int64)off_497B2F0);
  v18 = v17;
  if ( v17 )
  {
    v19 = (__int64)(v17 + 1);
    if ( (_QWORD *)v33[0] != v34 )
    {
      v29 = v17 + 1;
      _libc_free(v33[0], v33);
      v19 = (__int64)v29;
    }
    v33[0] = v19;
    v30 = v19;
    v20 = sub_EE6840(a1 + 944, v33);
    v18 = (_QWORD *)v30;
    if ( v20 )
    {
      v21 = v20[1];
      if ( v21 )
        v18 = (_QWORD *)v21;
    }
    if ( *(_QWORD **)(a1 + 928) == v18 )
      *(_BYTE *)(a1 + 936) = 1;
  }
  else
  {
    if ( v25 )
    {
      v23 = sub_CD1D40((__int64 *)(a1 + 808), 48, 3);
      *(_QWORD *)v23 = 0;
      v16 = (__int64 *)v23;
      v24 = v32;
      v27 = v23 + 8;
      *(_WORD *)(v23 + 16) = ((a4 & 0x3F) << 8) | 0x4042;
      LOBYTE(v23) = *(_BYTE *)(v23 + 18);
      v16[3] = a2;
      v16[5] = v6;
      *((_BYTE *)v16 + 18) = v23 & 0xF0 | 5;
      v16[1] = (__int64)&unk_49E0688;
      v16[4] = (__int64)a3;
      sub_C657C0((__int64 *)(a1 + 904), v16, v24, (__int64)off_497B2F0);
      v18 = (_QWORD *)v27;
    }
    if ( (_QWORD *)v33[0] != v34 )
    {
      v31 = v18;
      _libc_free(v33[0], v16);
      v18 = v31;
    }
    *(_QWORD *)(a1 + 920) = v18;
  }
  return v18;
}
