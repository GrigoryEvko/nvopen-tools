// Function: sub_EE9AE0
// Address: 0xee9ae0
//
__int64 __fastcall sub_EE9AE0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  char *v6; // rax
  char *v7; // rdx
  unsigned __int64 v8; // r12
  __int64 v10; // rdx
  char *v11; // rdx
  unsigned __int64 v12; // rax
  __int64 v13; // rcx
  __int64 v14; // r12
  char v15; // al
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  _QWORD *v20; // rax
  __int64 *v21; // rax
  unsigned __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 v26; // r9
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // rsi
  __int64 *v30; // rdx
  int v31; // [rsp+8h] [rbp-D8h]
  char v32; // [rsp+Fh] [rbp-D1h]
  __int64 *v33; // [rsp+18h] [rbp-C8h] BYREF
  __int64 v34[2]; // [rsp+20h] [rbp-C0h] BYREF
  _BYTE v35[176]; // [rsp+30h] [rbp-B0h] BYREF

  v6 = *(char **)a1;
  v7 = *(char **)(a1 + 8);
  if ( v7 == *(char **)a1 || *v6 != 83 )
    return 0;
  *(_QWORD *)a1 = v6 + 1;
  if ( v7 == v6 + 1 )
  {
LABEL_8:
    v34[0] = 0;
    if ( !(unsigned __int8)sub_EE3560((char **)a1, v34) )
    {
      v11 = *(char **)a1;
      v12 = ++v34[0];
      if ( v11 != *(char **)(a1 + 8) && *v11 == 95 )
      {
        v13 = *(_QWORD *)(a1 + 296);
        *(_QWORD *)a1 = v11 + 1;
        if ( v12 < (*(_QWORD *)(a1 + 304) - v13) >> 3 )
          return *(_QWORD *)(v13 + 8 * v12);
      }
    }
    return 0;
  }
  v10 = (unsigned __int8)v6[1];
  if ( (unsigned __int8)(v10 - 97) > 0x19u )
  {
    if ( (_BYTE)v10 == 95 )
    {
      *(_QWORD *)a1 = v6 + 2;
      v27 = *(_QWORD *)(a1 + 296);
      if ( v27 != *(_QWORD *)(a1 + 304) )
        return *(_QWORD *)v27;
      return 0;
    }
    goto LABEL_8;
  }
  switch ( (char)v10 )
  {
    case 'a':
      v31 = 0;
      v14 = 0;
      break;
    case 'b':
      v31 = 1;
      v14 = 1;
      break;
    case 'd':
      v31 = 5;
      v14 = 5;
      break;
    case 'i':
      v31 = 3;
      v14 = 3;
      break;
    case 'o':
      v31 = 4;
      v14 = 4;
      break;
    case 's':
      v31 = 2;
      v14 = 2;
      break;
    default:
      return 0;
  }
  *(_QWORD *)a1 = v6 + 2;
  v15 = *(_BYTE *)(a1 + 937);
  v34[0] = (__int64)v35;
  v32 = v15;
  v34[1] = 0x2000000000LL;
  sub_D953B0((__int64)v34, 48, v10, (unsigned __int8)(v10 - 97), a5, a6);
  sub_D953B0((__int64)v34, v14, v16, v17, v18, v19);
  v20 = sub_C65B40(a1 + 904, (__int64)v34, (__int64 *)&v33, (__int64)off_497B2F0);
  v8 = (unsigned __int64)v20;
  if ( v20 )
  {
    v8 = (unsigned __int64)(v20 + 1);
    if ( (_BYTE *)v34[0] != v35 )
      _libc_free(v34[0], v34);
    v34[0] = v8;
    v21 = sub_EE6840(a1 + 944, v34);
    if ( v21 )
    {
      v22 = v21[1];
      if ( v22 )
        v8 = v22;
    }
    if ( *(_QWORD *)(a1 + 928) == v8 )
      *(_BYTE *)(a1 + 936) = 1;
LABEL_23:
    v34[0] = sub_EE9860(a1, v8);
    if ( v34[0] != v8 )
    {
      sub_E18380(a1 + 296, v34, v23, v24, v25, v26);
      return v34[0];
    }
    return v8;
  }
  if ( v32 )
  {
    v28 = sub_CD1D40((__int64 *)(a1 + 808), 24, 3);
    *(_QWORD *)v28 = 0;
    v29 = v28;
    v8 = v28 + 8;
    *(_WORD *)(v28 + 16) = 16432;
    v30 = v33;
    *(_BYTE *)(v28 + 18) = *(_BYTE *)(v28 + 18) & 0xF0 | 5;
    *(_DWORD *)(v28 + 20) = v31;
    *(_QWORD *)(v28 + 8) = &unk_49DFFC8;
    sub_C657C0((__int64 *)(a1 + 904), (__int64 *)v28, v30, (__int64)off_497B2F0);
    if ( (_BYTE *)v34[0] != v35 )
      _libc_free(v34[0], v29);
    *(_QWORD *)(a1 + 920) = v8;
    goto LABEL_23;
  }
  if ( (_BYTE *)v34[0] != v35 )
    _libc_free(v34[0], v34);
  *(_QWORD *)(a1 + 920) = 0;
  return v8;
}
