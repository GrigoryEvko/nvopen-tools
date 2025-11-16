// Function: sub_EE95D0
// Address: 0xee95d0
//
_QWORD *__fastcall sub_EE95D0(__int64 a1, __int64 a2, unsigned __int8 *a3)
{
  signed __int64 v5; // rax
  unsigned __int8 *v6; // rdx
  _QWORD *v7; // r10
  signed __int64 v8; // r8
  char *v9; // rax
  unsigned __int8 *v11; // r15
  char v12; // al
  __int64 *v13; // rsi
  _QWORD *v14; // rax
  __int64 v15; // r10
  __int64 *v16; // rax
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // r10
  __int64 *v20; // rdx
  char v21; // [rsp+0h] [rbp-F0h]
  __int64 v22; // [rsp+10h] [rbp-E0h]
  _QWORD *v23; // [rsp+10h] [rbp-E0h]
  _QWORD *v24; // [rsp+18h] [rbp-D8h]
  __int64 v25; // [rsp+18h] [rbp-D8h]
  _QWORD *v26; // [rsp+18h] [rbp-D8h]
  __int64 *v27; // [rsp+28h] [rbp-C8h] BYREF
  __int64 v28[2]; // [rsp+30h] [rbp-C0h] BYREF
  _QWORD v29[22]; // [rsp+40h] [rbp-B0h] BYREF

  v5 = sub_EE32C0((char **)a1, 1);
  if ( v5 )
  {
    v8 = v5;
    v9 = *(char **)a1;
    if ( *(_QWORD *)a1 != *(_QWORD *)(a1 + 8) && *v9 == 69 )
    {
      v22 = v8;
      v11 = v6;
      *(_QWORD *)a1 = v9 + 1;
      v12 = *(_BYTE *)(a1 + 937);
      v29[0] = 77;
      v21 = v12;
      v28[0] = (__int64)v29;
      v28[1] = 0x2000000002LL;
      if ( a2 )
        sub_C653C0((__int64)v28, a3, a2);
      else
        sub_C653C0((__int64)v28, 0, 0);
      sub_C653C0((__int64)v28, v11, v22);
      v13 = v28;
      v14 = sub_C65B40(a1 + 904, (__int64)v28, (__int64 *)&v27, (__int64)off_497B2F0);
      v7 = v14;
      if ( v14 )
      {
        v15 = (__int64)(v14 + 1);
        if ( (_QWORD *)v28[0] != v29 )
        {
          v24 = v14 + 1;
          _libc_free(v28[0], v28);
          v15 = (__int64)v24;
        }
        v28[0] = v15;
        v25 = v15;
        v16 = sub_EE6840(a1 + 944, v28);
        v7 = (_QWORD *)v25;
        if ( v16 )
        {
          v17 = v16[1];
          if ( v17 )
            v7 = (_QWORD *)v17;
        }
        if ( *(_QWORD **)(a1 + 928) == v7 )
          *(_BYTE *)(a1 + 936) = 1;
      }
      else
      {
        if ( v21 )
        {
          v18 = sub_CD1D40((__int64 *)(a1 + 808), 56, 3);
          *(_QWORD *)v18 = 0;
          v13 = (__int64 *)v18;
          v19 = v18 + 8;
          *(_WORD *)(v18 + 16) = 16461;
          LOBYTE(v18) = *(_BYTE *)(v18 + 18);
          v13[3] = a2;
          v20 = v27;
          v13[4] = (__int64)a3;
          v13[5] = v22;
          *((_BYTE *)v13 + 18) = v18 & 0xF0 | 5;
          v13[6] = (__int64)v11;
          v23 = (_QWORD *)v19;
          v13[1] = (__int64)&unk_49E0B68;
          sub_C657C0((__int64 *)(a1 + 904), v13, v20, (__int64)off_497B2F0);
          v7 = v23;
        }
        if ( (_QWORD *)v28[0] != v29 )
        {
          v26 = v7;
          _libc_free(v28[0], v13);
          v7 = v26;
        }
        *(_QWORD *)(a1 + 920) = v7;
      }
    }
  }
  return v7;
}
