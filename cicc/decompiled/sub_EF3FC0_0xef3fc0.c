// Function: sub_EF3FC0
// Address: 0xef3fc0
//
__int64 __fastcall sub_EF3FC0(__int64 a1, _BYTE *a2)
{
  char *v2; // rax
  __int64 v3; // rcx
  __int64 v4; // r8
  __int64 v5; // r9
  __int64 v6; // rdx
  const char *v7; // r12
  _QWORD *v8; // r15
  _BYTE *v10; // rax
  _BYTE *v11; // rdx
  char v12; // al
  char v13; // r13
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 v18; // r14
  char v19; // al
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r9
  _QWORD **v24; // rsi
  _QWORD *v25; // rax
  __int64 *v26; // rax
  _QWORD *v27; // rax
  __int64 v28; // rdx
  __int64 v29; // rcx
  __int64 v30; // r8
  __int64 v31; // r9
  __int64 v32; // r14
  char v33; // al
  __int64 *v34; // r13
  __int64 v35; // rdx
  __int64 v36; // rcx
  __int64 v37; // r8
  __int64 v38; // r9
  __int64 *v39; // rsi
  __int64 *v40; // rax
  _QWORD *v41; // rax
  __int64 v42; // rax
  char v43; // al
  __int64 v44; // rax
  void *v45; // rax
  __int64 v46; // rax
  char v47; // [rsp+7h] [rbp-E9h]
  char v48; // [rsp+10h] [rbp-E0h]
  char v49; // [rsp+10h] [rbp-E0h]
  char v50; // [rsp+18h] [rbp-D8h]
  __int64 *v51; // [rsp+28h] [rbp-C8h] BYREF
  _QWORD *v52; // [rsp+30h] [rbp-C0h] BYREF
  __int64 v53; // [rsp+38h] [rbp-B8h]
  _QWORD v54[22]; // [rsp+40h] [rbp-B0h] BYREF

  v2 = sub_EE33C0(a1);
  if ( v2 )
  {
    v6 = (unsigned __int8)v2[2];
    if ( (_BYTE)v6 != 8 )
    {
      if ( (unsigned __int8)v6 <= 0xAu && ((_BYTE)v6 != 4 || (v2[3] & 1) != 0) )
      {
        v7 = (const char *)*((_QWORD *)v2 + 1);
        v53 = (__int64)v7;
        v52 = (_QWORD *)strlen(v7);
        return sub_EE6A90(a1 + 808, (__int64 *)&v52);
      }
      return 0;
    }
    v12 = *(_BYTE *)(a1 + 776);
    v13 = *(_BYTE *)(a1 + 777);
    *(_BYTE *)(a1 + 776) = 0;
    v50 = v12;
    if ( v13 )
    {
      *(_BYTE *)(a1 + 777) = 1;
      v18 = sub_EF1F20(a1, (__int64)a2, v6, v3, v4, v5);
      if ( !v18 )
        goto LABEL_43;
      if ( a2 )
        goto LABEL_15;
    }
    else
    {
      if ( a2 )
      {
        *(_BYTE *)(a1 + 777) = 1;
        v18 = sub_EF1F20(a1, (__int64)a2, v6, v3, v4, v5);
        if ( v18 )
        {
LABEL_15:
          *a2 = 1;
          goto LABEL_16;
        }
LABEL_43:
        v8 = 0;
LABEL_24:
        *(_BYTE *)(a1 + 777) = v13;
        *(_BYTE *)(a1 + 776) = v50;
        return (__int64)v8;
      }
      v18 = sub_EF1F20(a1, 0, v6, v3, v4, v5);
      if ( !v18 )
        goto LABEL_43;
    }
LABEL_16:
    v19 = *(_BYTE *)(a1 + 937);
    v52 = v54;
    v47 = v19;
    v53 = 0x2000000000LL;
    sub_D953B0((__int64)&v52, 4, v14, v15, v16, v17);
    sub_D953B0((__int64)&v52, v18, v20, v21, v22, v23);
    v24 = &v52;
    v25 = sub_C65B40(a1 + 904, (__int64)&v52, (__int64 *)&v51, (__int64)off_497B2F0);
    v8 = v25;
    if ( v25 )
    {
      v8 = v25 + 1;
      if ( v52 != v54 )
        _libc_free(v52, &v52);
      v52 = v8;
      v26 = sub_EE6840(a1 + 944, (__int64 *)&v52);
      if ( v26 )
      {
        v27 = (_QWORD *)v26[1];
        if ( v27 )
          v8 = v27;
      }
      if ( *(_QWORD **)(a1 + 928) == v8 )
        *(_BYTE *)(a1 + 936) = 1;
    }
    else
    {
      if ( v47 )
      {
        v42 = sub_CD1D40((__int64 *)(a1 + 808), 32, 3);
        *(_QWORD *)v42 = 0;
        v24 = (_QWORD **)v42;
        v8 = (_QWORD *)(v42 + 8);
        *(_WORD *)(v42 + 16) = 16388;
        LOBYTE(v42) = *(_BYTE *)(v42 + 18);
        v24[3] = (_QWORD *)v18;
        *((_BYTE *)v24 + 18) = v42 & 0xF0 | 5;
        v24[1] = &unk_49DEEE8;
        sub_C657C0((__int64 *)(a1 + 904), (__int64 *)v24, v51, (__int64)off_497B2F0);
      }
      if ( v52 != v54 )
        _libc_free(v52, v24);
      *(_QWORD *)(a1 + 920) = v8;
    }
    goto LABEL_24;
  }
  if ( (unsigned __int8)sub_EE3B50((const void **)a1, 2u, "li") )
  {
    v32 = sub_EE6C50((__int64 *)a1);
    if ( !v32 )
      return 0;
    v33 = *(_BYTE *)(a1 + 937);
    v52 = v54;
    v34 = (__int64 *)(a1 + 904);
    v48 = v33;
    v53 = 0x2000000000LL;
    sub_D953B0((__int64)&v52, 20, v28, v29, v30, v31);
    sub_D953B0((__int64)&v52, v32, v35, v36, v37, v38);
    v39 = (__int64 *)&v52;
    v8 = sub_C65B40(a1 + 904, (__int64)&v52, (__int64 *)&v51, (__int64)off_497B2F0);
    if ( !v8 )
    {
      if ( !v48 )
        goto LABEL_49;
      v44 = sub_CD1D40((__int64 *)(a1 + 808), 32, 3);
      *(_QWORD *)v44 = 0;
      v39 = (__int64 *)v44;
      v8 = (_QWORD *)(v44 + 8);
      *(_WORD *)(v44 + 16) = 16404;
      *(_BYTE *)(v44 + 18) = *(_BYTE *)(v44 + 18) & 0xF0 | 5;
      v45 = &unk_49DF598;
      goto LABEL_54;
    }
  }
  else
  {
    v10 = *(_BYTE **)a1;
    v11 = *(_BYTE **)(a1 + 8);
    if ( *(_BYTE **)a1 == v11 )
      return 0;
    if ( *v10 != 118 )
      return 0;
    *(_QWORD *)a1 = v10 + 1;
    if ( v11 == v10 + 1 )
      return 0;
    if ( (unsigned __int8)(v10[1] - 48) > 9u )
      return 0;
    *(_QWORD *)a1 = v10 + 2;
    v32 = sub_EE6C50((__int64 *)a1);
    if ( !v32 )
      return 0;
    v43 = *(_BYTE *)(a1 + 937);
    v34 = (__int64 *)(a1 + 904);
    v54[1] = v32;
    v39 = (__int64 *)&v52;
    v49 = v43;
    v52 = v54;
    v53 = 0x2000000004LL;
    v54[0] = 4;
    v8 = sub_C65B40(a1 + 904, (__int64)&v52, (__int64 *)&v51, (__int64)off_497B2F0);
    if ( !v8 )
    {
      if ( !v49 )
      {
LABEL_49:
        if ( v52 != v54 )
          _libc_free(v52, v39);
        *(_QWORD *)(a1 + 920) = v8;
        return (__int64)v8;
      }
      v46 = sub_CD1D40((__int64 *)(a1 + 808), 32, 3);
      *(_QWORD *)v46 = 0;
      v39 = (__int64 *)v46;
      v8 = (_QWORD *)(v46 + 8);
      *(_WORD *)(v46 + 16) = 16388;
      *(_BYTE *)(v46 + 18) = *(_BYTE *)(v46 + 18) & 0xF0 | 5;
      v45 = &unk_49DEED8;
LABEL_54:
      v39[3] = v32;
      v39[1] = (__int64)v45 + 16;
      sub_C657C0(v34, v39, v51, (__int64)off_497B2F0);
      goto LABEL_49;
    }
  }
  ++v8;
  if ( v52 != v54 )
    _libc_free(v52, &v52);
  v52 = v8;
  v40 = sub_EE6840(a1 + 944, (__int64 *)&v52);
  if ( v40 )
  {
    v41 = (_QWORD *)v40[1];
    if ( v41 )
      v8 = v41;
  }
  if ( *(_QWORD **)(a1 + 928) == v8 )
    *(_BYTE *)(a1 + 936) = 1;
  return (__int64)v8;
}
