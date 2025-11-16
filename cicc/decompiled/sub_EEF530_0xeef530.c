// Function: sub_EEF530
// Address: 0xeef530
//
_QWORD *__fastcall sub_EEF530(__int64 a1)
{
  char *v1; // rdx
  char *v2; // rax
  char v3; // cl
  signed __int64 v4; // rsi
  __int64 v6; // r15
  char *v7; // rax
  char *v8; // rax
  char *v9; // rax
  __int64 v10; // rbx
  __int64 v11; // r13
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  unsigned __int64 *v16; // rax
  __int64 v17; // rdx
  _BYTE **v18; // rsi
  _QWORD *v19; // rax
  unsigned __int64 *v20; // r9
  __int64 *v21; // rax
  __int64 v22; // rax
  unsigned __int64 v23; // r14
  __int64 v24; // r8
  __int64 v25; // r9
  __int64 v26; // rax
  __int64 v27; // rax
  char v28; // [rsp+Fh] [rbp-E1h]
  char v29; // [rsp+Fh] [rbp-E1h]
  unsigned __int64 *v30; // [rsp+10h] [rbp-E0h]
  _BYTE *v31; // [rsp+10h] [rbp-E0h]
  _BYTE *v32; // [rsp+10h] [rbp-E0h]
  __int64 v33; // [rsp+18h] [rbp-D8h]
  unsigned __int64 *v34; // [rsp+18h] [rbp-D8h]
  __int64 *v35; // [rsp+28h] [rbp-C8h] BYREF
  _BYTE *v36; // [rsp+30h] [rbp-C0h] BYREF
  __int64 v37; // [rsp+38h] [rbp-B8h]
  _BYTE v38[176]; // [rsp+40h] [rbp-B0h] BYREF

  v1 = *(char **)(a1 + 8);
  v2 = *(char **)a1;
  if ( *(char **)a1 == v1 )
    return (_QWORD *)sub_EF1F20(a1);
  v3 = *v2;
  v4 = v1 - v2;
  if ( *v2 == 84 )
  {
    if ( v4 == 1 )
      return (_QWORD *)sub_EF1F20(a1);
    v8 = (char *)memchr("yptnk", v2[1], 5u);
    if ( !v8 || v8 == "" )
      return (_QWORD *)sub_EF1F20(a1);
    v23 = sub_EF6290(a1, 0);
    if ( !v23 )
      return 0;
    v25 = sub_EEF530(a1);
    if ( !v25 )
      return 0;
    v31 = (_BYTE *)v25;
    v29 = *(_BYTE *)(a1 + 937);
    v37 = 0x2000000000LL;
    v36 = v38;
    sub_EE40D0((__int64)&v36, 0x22u, v23, v25, v24, v25);
    v18 = &v36;
    v6 = (__int64)sub_C65B40(a1 + 904, (__int64)&v36, (__int64 *)&v35, (__int64)off_497B2F0);
    if ( v6 )
      goto LABEL_24;
    if ( v29 )
    {
      v27 = sub_CD1D40((__int64 *)(a1 + 808), 40, 3);
      *(_QWORD *)v27 = 0;
      v18 = (_BYTE **)v27;
      v6 = v27 + 8;
      *(_WORD *)(v27 + 16) = 16418;
      LOBYTE(v27) = *(_BYTE *)(v27 + 18);
      v18[3] = (_BYTE *)v23;
      v18[4] = v31;
      *((_BYTE *)v18 + 18) = v27 & 0xF0 | 5;
      v18[1] = &unk_49DFA88;
      sub_C657C0((__int64 *)(a1 + 904), (__int64 *)v18, v35, (__int64)off_497B2F0);
    }
    goto LABEL_37;
  }
  if ( v3 <= 84 )
  {
    if ( v3 != 74 )
    {
      if ( v3 == 76 )
      {
        if ( v4 == 1 || v2[1] != 90 )
          return sub_EEE940(a1);
        *(_QWORD *)a1 = v2 + 2;
        v6 = sub_EF05F0(a1, 1);
        if ( !v6 )
          return 0;
        v7 = *(char **)a1;
        if ( *(_QWORD *)a1 == *(_QWORD *)(a1 + 8) )
          return 0;
LABEL_12:
        if ( *v7 == 69 )
        {
          *(_QWORD *)a1 = v7 + 1;
          return (_QWORD *)v6;
        }
        return 0;
      }
      return (_QWORD *)sub_EF1F20(a1);
    }
    v9 = v2 + 1;
    v10 = *(_QWORD *)(a1 + 24);
    v11 = *(_QWORD *)(a1 + 16);
    *(_QWORD *)a1 = v9;
    if ( v1 == v9 )
      goto LABEL_20;
LABEL_19:
    if ( *v9 != 69 )
    {
LABEL_20:
      while ( 1 )
      {
        v36 = (_BYTE *)sub_EEF530(a1);
        v6 = (__int64)v36;
        if ( !v36 )
          return (_QWORD *)v6;
        sub_E18380(a1 + 16, (__int64 *)&v36, v12, v13, v14, v15);
        v9 = *(char **)a1;
        if ( *(_QWORD *)(a1 + 8) != *(_QWORD *)a1 )
          goto LABEL_19;
      }
    }
    *(_QWORD *)a1 = v9 + 1;
    v16 = (unsigned __int64 *)sub_EE6060((_QWORD *)a1, (v10 - v11) >> 3);
    v36 = v38;
    v30 = v16;
    v28 = *(_BYTE *)(a1 + 937);
    v33 = v17;
    v37 = 0x2000000000LL;
    sub_EE4780((__int64)&v36, 0x29u, v30, v33, v33, (__int64)v30);
    v18 = &v36;
    v19 = sub_C65B40(a1 + 904, (__int64)&v36, (__int64 *)&v35, (__int64)off_497B2F0);
    v20 = v30;
    v6 = (__int64)v19;
    if ( v19 )
    {
LABEL_24:
      v6 += 8;
      if ( v36 != v38 )
        _libc_free(v36, &v36);
      v36 = (_BYTE *)v6;
      v21 = sub_EE6840(a1 + 944, (__int64 *)&v36);
      if ( v21 )
      {
        v22 = v21[1];
        if ( v22 )
          v6 = v22;
      }
      if ( *(_QWORD *)(a1 + 928) == v6 )
        *(_BYTE *)(a1 + 936) = 1;
      return (_QWORD *)v6;
    }
    if ( v28 )
    {
      v32 = (_BYTE *)v33;
      v34 = v20;
      v26 = sub_CD1D40((__int64 *)(a1 + 808), 40, 3);
      *(_QWORD *)v26 = 0;
      v18 = (_BYTE **)v26;
      v6 = v26 + 8;
      *(_WORD *)(v26 + 16) = 16425;
      LOBYTE(v26) = *(_BYTE *)(v26 + 18);
      v18[3] = v34;
      v18[4] = v32;
      *((_BYTE *)v18 + 18) = v26 & 0xF0 | 5;
      v18[1] = &unk_49DFD28;
      sub_C657C0((__int64 *)(a1 + 904), (__int64 *)v18, v35, (__int64)off_497B2F0);
    }
LABEL_37:
    if ( v36 != v38 )
      _libc_free(v36, v18);
    *(_QWORD *)(a1 + 920) = v6;
    return (_QWORD *)v6;
  }
  if ( v3 == 88 )
  {
    *(_QWORD *)a1 = v2 + 1;
    v6 = sub_EEA9F0(a1);
    if ( !v6 )
      return 0;
    v7 = *(char **)a1;
    if ( *(_QWORD *)(a1 + 8) == *(_QWORD *)a1 )
      return 0;
    goto LABEL_12;
  }
  return (_QWORD *)sub_EF1F20(a1);
}
