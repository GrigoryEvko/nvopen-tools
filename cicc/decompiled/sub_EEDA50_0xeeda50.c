// Function: sub_EEDA50
// Address: 0xeeda50
//
_QWORD *__fastcall sub_EEDA50(__int64 a1)
{
  __int64 v1; // rax
  _BYTE *v2; // rdx
  _QWORD *result; // rax
  char v4; // al
  __int64 v5; // rbx
  __int64 v6; // r13
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r9
  __int64 v10; // r15
  char v11; // al
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  __int64 *v20; // rsi
  __int64 v21; // rbx
  __int64 *v22; // rax
  __int64 v23; // r8
  __int64 v24; // r9
  char v25; // dl
  char v26; // [rsp+8h] [rbp-E8h]
  _QWORD *v27; // [rsp+18h] [rbp-D8h]
  unsigned __int8 v28; // [rsp+27h] [rbp-C9h] BYREF
  __int64 *v29; // [rsp+28h] [rbp-C8h] BYREF
  __int64 v30[2]; // [rsp+30h] [rbp-C0h] BYREF
  _QWORD v31[22]; // [rsp+40h] [rbp-B0h] BYREF

  v1 = *(_QWORD *)(a1 + 8);
  v2 = *(_BYTE **)a1;
  if ( *(_QWORD *)a1 == v1 || v1 - (_QWORD)v2 == 1 || *v2 != 100 )
    return (_QWORD *)sub_EEA9F0(a1);
  v4 = v2[1];
  if ( v4 == 105 )
  {
    *(_QWORD *)a1 = v2 + 2;
    v29 = (__int64 *)sub_EE6C50((__int64 *)a1);
    if ( v29 )
    {
      v30[0] = sub_EEDA50(a1);
      if ( v30[0] )
      {
        v28 = 0;
        return (_QWORD *)sub_EE8BC0(a1 + 808, (__int64 *)&v29, v30, &v28, v23, v24);
      }
    }
    return 0;
  }
  if ( v4 == 120 )
  {
    *(_QWORD *)a1 = v2 + 2;
    v29 = (__int64 *)sub_EEA9F0(a1);
    if ( v29 )
    {
      v30[0] = sub_EEDA50(a1);
      if ( v30[0] )
      {
        v28 = 1;
        return (_QWORD *)sub_EE8BC0(a1 + 808, (__int64 *)&v29, v30, &v28, v23, v24);
      }
    }
    return 0;
  }
  if ( v4 != 88 )
    return (_QWORD *)sub_EEA9F0(a1);
  *(_QWORD *)a1 = v2 + 2;
  v5 = sub_EEA9F0(a1);
  if ( !v5 )
    return 0;
  v6 = sub_EEA9F0(a1);
  if ( !v6 )
    return 0;
  v10 = sub_EEDA50(a1);
  if ( !v10 )
    return 0;
  v11 = *(_BYTE *)(a1 + 937);
  v30[0] = (__int64)v31;
  v26 = v11;
  v30[1] = 0x2000000002LL;
  v31[0] = 82;
  sub_D953B0((__int64)v30, v5, v7, v8, (__int64)v31, v9);
  sub_D953B0((__int64)v30, v6, v12, v13, v14, v15);
  sub_D953B0((__int64)v30, v10, v16, v17, v18, v19);
  v20 = v30;
  result = sub_C65B40(a1 + 904, (__int64)v30, (__int64 *)&v29, (__int64)off_497B2F0);
  if ( result )
  {
    v21 = (__int64)(result + 1);
    if ( (_QWORD *)v30[0] != v31 )
      _libc_free(v30[0], v30);
    v30[0] = v21;
    v22 = sub_EE6840(a1 + 944, v30);
    if ( v22 )
    {
      result = (_QWORD *)v22[1];
      if ( !result )
        result = (_QWORD *)v21;
    }
    else
    {
      result = (_QWORD *)v21;
    }
    if ( *(_QWORD **)(a1 + 928) == result )
      *(_BYTE *)(a1 + 936) = 1;
  }
  else
  {
    if ( v26 )
    {
      v20 = (__int64 *)sub_CD1D40((__int64 *)(a1 + 808), 48, 3);
      *v20 = 0;
      v25 = *((_BYTE *)v20 + 18);
      *((_WORD *)v20 + 8) = 16466;
      v20[3] = v5;
      v20[4] = v6;
      *((_BYTE *)v20 + 18) = v25 & 0xF0 | 5;
      v20[5] = v10;
      v20[1] = (__int64)&unk_49E08C8;
      sub_C657C0((__int64 *)(a1 + 904), v20, v29, (__int64)off_497B2F0);
      result = v20 + 1;
    }
    if ( (_QWORD *)v30[0] != v31 )
    {
      v27 = result;
      _libc_free(v30[0], v20);
      result = v27;
    }
    *(_QWORD *)(a1 + 920) = result;
  }
  return result;
}
