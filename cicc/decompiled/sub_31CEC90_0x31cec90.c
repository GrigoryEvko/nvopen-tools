// Function: sub_31CEC90
// Address: 0x31cec90
//
_BOOL8 __fastcall sub_31CEC90(__int64 a1, __int64 a2)
{
  _BOOL4 v2; // r13d
  __int64 v3; // r12
  __int64 v4; // rbx
  const char *v5; // rax
  __int64 v6; // rdx
  __int64 v7; // r15
  _BOOL4 v8; // eax
  unsigned __int64 v9; // rax
  char *v10; // rdx
  size_t v11; // rdx
  char *v12; // rsi
  __int64 v13; // rdx
  char *v14; // rcx
  __int64 v15; // r12
  unsigned __int64 v16; // rdx
  const char *v17; // r9
  size_t v18; // r8
  char *v20; // rax
  char *v21; // rdi
  __int64 v22; // [rsp+0h] [rbp-B0h]
  size_t n; // [rsp+8h] [rbp-A8h]
  const char *src; // [rsp+10h] [rbp-A0h]
  __int64 v25; // [rsp+20h] [rbp-90h]
  bool v26; // [rsp+2Fh] [rbp-81h]
  unsigned __int64 v27; // [rsp+38h] [rbp-78h] BYREF
  _QWORD *v28; // [rsp+40h] [rbp-70h] BYREF
  __int64 v29; // [rsp+48h] [rbp-68h]
  _BYTE v30[16]; // [rsp+50h] [rbp-60h] BYREF
  char *v31; // [rsp+60h] [rbp-50h] BYREF
  size_t v32; // [rsp+68h] [rbp-48h]
  _QWORD v33[8]; // [rsp+70h] [rbp-40h] BYREF

  v2 = 0;
  v3 = *(_QWORD *)(a2 + 32);
  v25 = a2 + 24;
  if ( v3 != a2 + 24 )
  {
    while ( 1 )
    {
      v4 = v3 - 56;
      if ( !v3 )
        v4 = 0;
      v26 = sub_B2FC80(v4);
      if ( v26 )
      {
        if ( *(_QWORD *)(v4 + 16) )
        {
          if ( (*(_BYTE *)(v4 + 33) & 0x20) == 0 )
          {
            v5 = sub_BD5D20(v4);
            if ( v6 != 14
              || *(_QWORD *)v5 != 0x725F6D76766E5F5FLL
              || *((_DWORD *)v5 + 2) != 1701602917
              || *((_WORD *)v5 + 6) != 29795 )
            {
              v7 = *(_QWORD *)(v4 + 16);
              if ( v7 )
                break;
            }
          }
        }
      }
LABEL_29:
      v3 = *(_QWORD *)(v3 + 8);
      if ( v25 == v3 )
        return v2;
    }
    v22 = v3;
    v8 = v2;
    while ( 1 )
    {
      v15 = *(_QWORD *)(v7 + 24);
      if ( *(_BYTE *)v15 > 0x1Cu )
        break;
LABEL_21:
      v7 = *(_QWORD *)(v7 + 8);
      if ( !v7 )
      {
        v3 = v22;
        v2 = v8;
        goto LABEL_29;
      }
    }
    v30[0] = 0;
    v28 = v30;
    v29 = 0;
    sub_B2BE50(v4);
    sub_2C75F20((__int64)&v31, (__int64 *)(v15 + 48));
    sub_2241490((unsigned __int64 *)&v28, v31, v32);
    if ( v31 != (char *)v33 )
      j_j___libc_free_0((unsigned __int64)v31);
    if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v29) <= 0x20 )
LABEL_38:
      sub_4262D8((__int64)"basic_string::append");
    sub_2241490((unsigned __int64 *)&v28, " Error: use of external function ", 0x21u);
    v17 = sub_BD5D20(v4);
    v18 = v16;
    if ( !v17 )
    {
      v31 = (char *)v33;
      v11 = 0;
      v32 = 0;
      v12 = (char *)v33;
      LOBYTE(v33[0]) = 0;
      goto LABEL_15;
    }
    v31 = (char *)v33;
    v9 = v16;
    v27 = v16;
    if ( v16 > 0xF )
    {
      n = v16;
      src = v17;
      v20 = (char *)sub_22409D0((__int64)&v31, &v27, 0);
      v17 = src;
      v18 = n;
      v31 = v20;
      v21 = v20;
      v33[0] = v27;
    }
    else
    {
      if ( v16 == 1 )
      {
        LOBYTE(v33[0]) = *v17;
        v10 = (char *)v33;
LABEL_14:
        v32 = v9;
        v10[v9] = 0;
        v11 = v32;
        v12 = v31;
LABEL_15:
        sub_2241490((unsigned __int64 *)&v28, v12, v11);
        if ( v31 != (char *)v33 )
          j_j___libc_free_0((unsigned __int64)v31);
        if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v29) <= 0x10 )
          goto LABEL_38;
        sub_2241490((unsigned __int64 *)&v28, " is not supported", 0x11u);
        sub_CEB590(&v28, 1, v13, v14);
        if ( v28 != (_QWORD *)v30 )
          j_j___libc_free_0((unsigned __int64)v28);
        v8 = v26;
        goto LABEL_21;
      }
      if ( !v16 )
      {
        v10 = (char *)v33;
        goto LABEL_14;
      }
      v21 = (char *)v33;
    }
    memcpy(v21, v17, v18);
    v9 = v27;
    v10 = v31;
    goto LABEL_14;
  }
  return v2;
}
