// Function: sub_1CECFA0
// Address: 0x1cecfa0
//
_BOOL8 __fastcall sub_1CECFA0(__int64 a1, __int64 a2)
{
  _BOOL4 v2; // r12d
  __int64 v3; // r13
  __int64 v4; // rbx
  const char *v5; // rax
  __int64 v6; // rdx
  __int64 v7; // r14
  _BOOL4 v8; // r13d
  unsigned __int64 v9; // rax
  const char *v10; // rdx
  _QWORD *v11; // r12
  unsigned __int64 v12; // rdx
  const char *v13; // r9
  size_t v14; // r8
  const char *v16; // rax
  char *v17; // rdi
  __int64 v18; // [rsp+0h] [rbp-B0h]
  size_t n; // [rsp+8h] [rbp-A8h]
  const char *src; // [rsp+10h] [rbp-A0h]
  bool v21; // [rsp+2Fh] [rbp-81h]
  unsigned __int64 v22; // [rsp+38h] [rbp-78h] BYREF
  _QWORD *v23; // [rsp+40h] [rbp-70h] BYREF
  __int64 v24; // [rsp+48h] [rbp-68h]
  _QWORD v25[2]; // [rsp+50h] [rbp-60h] BYREF
  char *v26; // [rsp+60h] [rbp-50h] BYREF
  unsigned __int64 v27; // [rsp+68h] [rbp-48h]
  char v28[64]; // [rsp+70h] [rbp-40h] BYREF

  v2 = 0;
  if ( a2 + 24 != *(_QWORD *)(a2 + 32) )
  {
    v3 = *(_QWORD *)(a2 + 32);
    while ( 1 )
    {
      v4 = v3 - 56;
      if ( !v3 )
        v4 = 0;
      v21 = sub_15E4F60(v4);
      if ( v21 )
      {
        if ( *(_QWORD *)(v4 + 8) )
        {
          if ( (*(_BYTE *)(v4 + 33) & 0x20) == 0 )
          {
            v5 = sub_1649960(v4);
            if ( v6 != 14
              || *(_QWORD *)v5 != 0x725F6D76766E5F5FLL
              || *((_DWORD *)v5 + 2) != 1701602917
              || *((_WORD *)v5 + 6) != 29795 )
            {
              v7 = *(_QWORD *)(v4 + 8);
              if ( v7 )
                break;
            }
          }
        }
      }
LABEL_29:
      v3 = *(_QWORD *)(v3 + 8);
      if ( a2 + 24 == v3 )
        return v2;
    }
    v18 = v3;
    v8 = v2;
    while ( 1 )
    {
      v11 = sub_1648700(v7);
      if ( *((_BYTE *)v11 + 16) > 0x17u )
        break;
LABEL_21:
      v7 = *(_QWORD *)(v7 + 8);
      if ( !v7 )
      {
        v2 = v8;
        v3 = v18;
        goto LABEL_29;
      }
    }
    LOBYTE(v25[0]) = 0;
    v23 = v25;
    v24 = 0;
    sub_15E0530(v4);
    sub_1C315E0((__int64)&v26, v11 + 6);
    sub_2241490(&v23, v26, v27);
    if ( v26 != v28 )
      j_j___libc_free_0(v26, *(_QWORD *)v28 + 1LL);
    if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v24) <= 0x20 )
LABEL_39:
      sub_4262D8((__int64)"basic_string::append");
    sub_2241490(&v23, " Error: use of external function ", 33);
    v13 = sub_1649960(v4);
    v14 = v12;
    if ( !v13 )
    {
      v26 = v28;
      v27 = 0;
      v28[0] = 0;
      sub_2241490(&v23, v28, 0);
      goto LABEL_15;
    }
    v26 = v28;
    v9 = v12;
    v22 = v12;
    if ( v12 > 0xF )
    {
      n = v12;
      src = v13;
      v16 = (const char *)sub_22409D0(&v26, &v22, 0);
      v13 = src;
      v14 = n;
      v26 = (char *)v16;
      v17 = (char *)v16;
      *(_QWORD *)v28 = v22;
    }
    else
    {
      if ( v12 == 1 )
      {
        v28[0] = *v13;
        v10 = v28;
LABEL_14:
        v27 = v9;
        v10[v9] = 0;
        sub_2241490(&v23, v26, v27);
LABEL_15:
        if ( v26 != v28 )
          j_j___libc_free_0(v26, *(_QWORD *)v28 + 1LL);
        if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v24) <= 0x10 )
          goto LABEL_39;
        sub_2241490(&v23, " is not supported", 17);
        sub_1C3EFD0((__int64)&v23, 1);
        if ( v23 != v25 )
          j_j___libc_free_0(v23, v25[0] + 1LL);
        v8 = v21;
        goto LABEL_21;
      }
      if ( !v12 )
      {
        v10 = v28;
        goto LABEL_14;
      }
      v17 = v28;
    }
    memcpy(v17, v13, v14);
    v9 = v22;
    v10 = v26;
    goto LABEL_14;
  }
  return v2;
}
