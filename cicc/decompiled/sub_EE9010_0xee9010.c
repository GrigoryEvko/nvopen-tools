// Function: sub_EE9010
// Address: 0xee9010
//
__int64 __fastcall sub_EE9010(__int64 a1, __int64 *a2)
{
  _BYTE *v2; // rax
  _BYTE *v3; // rdx
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // r14
  char v11; // al
  __int64 v12; // rsi
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // r9
  __int64 *v21; // rsi
  _QWORD *v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 v26; // r9
  __int64 v27; // r9
  __int64 *v28; // rax
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // r9
  __int64 v32; // rdx
  __int64 *v33; // rdx
  char v34; // [rsp-F8h] [rbp-F8h]
  __int64 v35; // [rsp-E8h] [rbp-E8h]
  unsigned __int8 v36; // [rsp-E0h] [rbp-E0h]
  _QWORD *v37; // [rsp-E0h] [rbp-E0h]
  __int64 v38; // [rsp-E0h] [rbp-E0h]
  __int64 v39; // [rsp-E0h] [rbp-E0h]
  __int64 *v40; // [rsp-D0h] [rbp-D0h] BYREF
  __int64 v41[2]; // [rsp-C8h] [rbp-C8h] BYREF
  _QWORD v42[23]; // [rsp-B8h] [rbp-B8h] BYREF

  v2 = *(_BYTE **)a1;
  v3 = *(_BYTE **)(a1 + 8);
  if ( *(_BYTE **)a1 == v3 )
    return 0;
  while ( 1 )
  {
    if ( *v2 != 87 )
      return 0;
    v36 = 0;
    *(_QWORD *)a1 = v2 + 1;
    if ( v2 + 1 != v3 && v2[1] == 80 )
    {
      v36 = 1;
      *(_QWORD *)a1 = v2 + 2;
    }
    v10 = sub_EE6C50((__int64 *)a1);
    if ( !v10 )
      break;
    v11 = *(_BYTE *)(a1 + 937);
    v12 = *a2;
    v41[0] = (__int64)v42;
    v34 = v11;
    v41[1] = 0x2000000002LL;
    v42[0] = 27;
    sub_D953B0((__int64)v41, v12, v6, v7, v8, v9);
    sub_D953B0((__int64)v41, v10, v13, v14, v15, v16);
    sub_D953B0((__int64)v41, v36, v17, v18, v19, v20);
    v21 = v41;
    v22 = sub_C65B40(a1 + 904, (__int64)v41, (__int64 *)&v40, (__int64)off_497B2F0);
    v26 = (__int64)v22;
    if ( v22 )
    {
      v27 = (__int64)(v22 + 1);
      if ( (_QWORD *)v41[0] != v42 )
      {
        v37 = v22 + 1;
        _libc_free(v41[0], v41);
        v27 = (__int64)v37;
      }
      v41[0] = v27;
      v38 = v27;
      v28 = sub_EE6840(a1 + 944, v41);
      v26 = v38;
      if ( v28 )
      {
        v29 = v28[1];
        if ( v29 )
          v26 = v29;
      }
      if ( *(_QWORD *)(a1 + 928) == v26 )
        *(_BYTE *)(a1 + 936) = 1;
    }
    else
    {
      if ( v34 )
      {
        v30 = sub_CD1D40((__int64 *)(a1 + 808), 48, 3);
        *(_QWORD *)v30 = 0;
        v21 = (__int64 *)v30;
        v31 = v30 + 8;
        v32 = *a2;
        *(_QWORD *)(v30 + 32) = v10;
        *(_WORD *)(v30 + 16) = 16411;
        LOBYTE(v30) = *(_BYTE *)(v30 + 18);
        v21[3] = v32;
        v33 = v40;
        v35 = v31;
        *((_BYTE *)v21 + 18) = v30 & 0xF0 | 5;
        v21[1] = (__int64)&unk_49DF788;
        *((_BYTE *)v21 + 40) = v36;
        sub_C657C0((__int64 *)(a1 + 904), v21, v33, (__int64)off_497B2F0);
        v26 = v35;
      }
      if ( (_QWORD *)v41[0] != v42 )
      {
        v39 = v26;
        _libc_free(v41[0], v21);
        v26 = v39;
      }
      *(_QWORD *)(a1 + 920) = v26;
    }
    *a2 = v26;
    v41[0] = v26;
    sub_E18380(a1 + 296, v41, v23, v24, v25, v26);
    v2 = *(_BYTE **)a1;
    v3 = *(_BYTE **)(a1 + 8);
    if ( *(_BYTE **)a1 == v3 )
      return 0;
  }
  return 1;
}
