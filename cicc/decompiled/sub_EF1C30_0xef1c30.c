// Function: sub_EF1C30
// Address: 0xef1c30
//
__int64 __fastcall sub_EF1C30(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _WORD *v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rdx
  bool v9; // cf
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // rbx
  char *v15; // r8
  __int64 v16; // r9
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 *v22; // rsi
  _QWORD *v23; // rax
  char *v24; // r8
  __int64 v25; // r13
  __int64 *v26; // rax
  __int64 v27; // rax
  __int64 v29; // rdx
  __int64 v30; // rcx
  __int64 v31; // r8
  __int64 v32; // r9
  __int64 v33; // rax
  __int64 *v34; // rdx
  char v35; // [rsp+7h] [rbp-E9h]
  char *v36; // [rsp+8h] [rbp-E8h]
  __int64 v37; // [rsp+8h] [rbp-E8h]
  __int64 v38; // [rsp+10h] [rbp-E0h]
  __int64 v39; // [rsp+10h] [rbp-E0h]
  __int64 *v40; // [rsp+28h] [rbp-C8h] BYREF
  __int64 v41[2]; // [rsp+30h] [rbp-C0h] BYREF
  _QWORD v42[22]; // [rsp+40h] [rbp-B0h] BYREF

  v6 = *(_WORD **)a1;
  v7 = *(_QWORD *)(a1 + 8);
  v9 = v7 == *(_QWORD *)a1;
  v8 = v7 - *(_QWORD *)a1;
  if ( !v9 && v8 != 1 && *v6 == 29524 )
  {
    *(_QWORD *)a1 = v6 + 1;
    v17 = sub_EF1680(a1, 0, v8, a4, a5, a6);
    v15 = "struct";
    v16 = 6;
    v14 = v17;
    if ( !v17 )
      return 0;
LABEL_7:
    v36 = v15;
    v35 = *(_BYTE *)(a1 + 937);
    v38 = v16;
    v41[1] = 0x2000000002LL;
    v41[0] = (__int64)v42;
    v42[0] = 6;
    sub_C653C0((__int64)v41, (unsigned __int8 *)v15, v16);
    sub_D953B0((__int64)v41, v14, v18, v19, v20, v21);
    v22 = v41;
    v23 = sub_C65B40(a1 + 904, (__int64)v41, (__int64 *)&v40, (__int64)off_497B2F0);
    v24 = v36;
    v25 = (__int64)v23;
    if ( v23 )
    {
      v25 = (__int64)(v23 + 1);
      if ( (_QWORD *)v41[0] != v42 )
        _libc_free(v41[0], v41);
      v41[0] = v25;
      v26 = sub_EE6840(a1 + 944, v41);
      if ( v26 )
      {
        v27 = v26[1];
        if ( v27 )
          v25 = v27;
      }
      if ( *(_QWORD *)(a1 + 928) == v25 )
        *(_BYTE *)(a1 + 936) = 1;
    }
    else
    {
      if ( v35 )
      {
        v37 = v38;
        v39 = (__int64)v24;
        v33 = sub_CD1D40((__int64 *)(a1 + 808), 48, 3);
        *(_QWORD *)v33 = 0;
        v22 = (__int64 *)v33;
        v25 = v33 + 8;
        *(_WORD *)(v33 + 16) = 16390;
        LOBYTE(v33) = *(_BYTE *)(v33 + 18);
        v34 = v40;
        v22[3] = v37;
        v22[4] = v39;
        v22[5] = v14;
        *((_BYTE *)v22 + 18) = v33 & 0xF0 | 5;
        v22[1] = (__int64)&unk_49DF068;
        sub_C657C0((__int64 *)(a1 + 904), v22, v34, (__int64)off_497B2F0);
      }
      if ( (_QWORD *)v41[0] != v42 )
        _libc_free(v41[0], v22);
      *(_QWORD *)(a1 + 920) = v25;
    }
    return v25;
  }
  if ( (unsigned __int8)sub_EE3B50((const void **)a1, 2u, "Tu") )
  {
    v14 = sub_EF1680(a1, 0, v10, v11, v12, v13);
    if ( v14 )
    {
      v15 = "union";
      v16 = 5;
      goto LABEL_7;
    }
    return 0;
  }
  if ( (unsigned __int8)sub_EE3B50((const void **)a1, 2u, "Te") )
  {
    v14 = sub_EF1680(a1, 0, v29, v30, v31, v32);
    if ( !v14 )
      return 0;
    v15 = "enum";
    v16 = 4;
    goto LABEL_7;
  }
  return sub_EF1680(a1, 0, v29, v30, v31, v32);
}
