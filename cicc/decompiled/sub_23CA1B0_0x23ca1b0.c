// Function: sub_23CA1B0
// Address: 0x23ca1b0
//
__int64 __fastcall sub_23CA1B0(__int64 a1, char **a2, __int64 *a3, __int64 *a4)
{
  char *v4; // rbx
  __int64 *v7; // rdi
  __int64 v8; // rax
  __int64 v9; // rdx
  size_t v10; // rcx
  __int64 v11; // rsi
  size_t v12; // rdx
  char *v14; // [rsp+8h] [rbp-138h]
  __int64 v15[2]; // [rsp+20h] [rbp-120h] BYREF
  char v16; // [rsp+30h] [rbp-110h]
  unsigned __int64 v17[2]; // [rsp+40h] [rbp-100h] BYREF
  _BYTE v18[16]; // [rsp+50h] [rbp-F0h] BYREF
  __int64 *v19; // [rsp+60h] [rbp-E0h] BYREF
  size_t n; // [rsp+68h] [rbp-D8h]
  _QWORD src[2]; // [rsp+70h] [rbp-D0h] BYREF
  _QWORD v22[2]; // [rsp+80h] [rbp-C0h] BYREF
  char *v23; // [rsp+90h] [rbp-B0h]
  __int16 v24; // [rsp+A0h] [rbp-A0h]
  _QWORD v25[2]; // [rsp+B0h] [rbp-90h] BYREF
  const char *v26; // [rsp+C0h] [rbp-80h]
  __int16 v27; // [rsp+D0h] [rbp-70h]
  void *v28[2]; // [rsp+E0h] [rbp-60h] BYREF
  unsigned __int64 *v29; // [rsp+F0h] [rbp-50h]
  __int16 v30; // [rsp+100h] [rbp-40h]

  v4 = *a2;
  v14 = a2[1];
  if ( *a2 == v14 )
    return 1;
  while ( 1 )
  {
    v30 = 260;
    v28[0] = v4;
    sub_CA4130((__int64)v15, a3, (__int64)v28, -1, 1u, 0, 1);
    if ( (v16 & 1) != 0 )
    {
      if ( LODWORD(v15[0]) )
        break;
    }
    v17[0] = (unsigned __int64)v18;
    v17[1] = 0;
    v18[0] = 0;
    if ( !(unsigned __int8)sub_23C9090(a1, v15[0], v17) )
    {
      v22[0] = "error parsing file '";
      v24 = 1027;
      v25[0] = v22;
      v26 = "': ";
      v23 = v4;
      v28[0] = v25;
      v27 = 770;
      v29 = v17;
      v30 = 1026;
      sub_CA0F50((__int64 *)&v19, v28);
      v7 = (__int64 *)*a4;
      v8 = (__int64)v19;
      if ( v19 != src )
      {
        v10 = n;
        v9 = src[0];
        if ( v7 != a4 + 2 )
        {
          v11 = a4[2];
          *a4 = (__int64)v19;
          a4[1] = v10;
          a4[2] = v9;
          if ( !v7 )
          {
LABEL_30:
            v19 = src;
            v7 = src;
            goto LABEL_17;
          }
LABEL_16:
          v19 = v7;
          src[0] = v11;
          goto LABEL_17;
        }
        goto LABEL_29;
      }
      goto LABEL_31;
    }
    if ( (_BYTE *)v17[0] != v18 )
      j_j___libc_free_0(v17[0]);
    if ( (v16 & 1) != 0 || !v15[0] )
    {
      v4 += 32;
      if ( v14 == v4 )
        return 1;
    }
    else
    {
      v4 += 32;
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v15[0] + 8LL))(v15[0]);
      if ( v14 == v4 )
        return 1;
    }
  }
  (*(void (__fastcall **)(unsigned __int64 *))(*(_QWORD *)v15[1] + 32LL))(v17);
  v22[0] = "can't open file '";
  v25[0] = v22;
  v24 = 1027;
  v26 = "': ";
  v27 = 770;
  v23 = v4;
  v28[0] = v25;
  v29 = v17;
  v30 = 1026;
  sub_CA0F50((__int64 *)&v19, v28);
  v7 = (__int64 *)*a4;
  v8 = (__int64)v19;
  if ( v19 != src )
  {
    v9 = src[0];
    v10 = n;
    if ( v7 != a4 + 2 )
    {
      v11 = a4[2];
      *a4 = (__int64)v19;
      a4[1] = v10;
      a4[2] = v9;
      if ( !v7 )
        goto LABEL_30;
      goto LABEL_16;
    }
LABEL_29:
    *a4 = v8;
    a4[1] = v10;
    a4[2] = v9;
    goto LABEL_30;
  }
LABEL_31:
  v12 = n;
  if ( n )
  {
    if ( n == 1 )
      *(_BYTE *)v7 = src[0];
    else
      memcpy(v7, src, n);
    v12 = n;
    v7 = (__int64 *)*a4;
  }
  a4[1] = v12;
  *((_BYTE *)v7 + v12) = 0;
  v7 = v19;
LABEL_17:
  n = 0;
  *(_BYTE *)v7 = 0;
  if ( v19 != src )
    j_j___libc_free_0((unsigned __int64)v19);
  if ( (_BYTE *)v17[0] != v18 )
    j_j___libc_free_0(v17[0]);
  if ( (v16 & 1) != 0 || !v15[0] )
    return 0;
  (*(void (__fastcall **)(__int64))(*(_QWORD *)v15[0] + 8LL))(v15[0]);
  return 0;
}
