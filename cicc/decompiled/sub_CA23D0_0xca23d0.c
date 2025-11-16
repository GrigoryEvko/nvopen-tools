// Function: sub_CA23D0
// Address: 0xca23d0
//
__int64 __fastcall sub_CA23D0(__int64 a1, char a2)
{
  __int64 result; // rax
  __int64 v3; // rdx
  __int64 v4; // rcx
  __int64 v5; // r8
  __int64 v6; // rsi
  int v7; // eax
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r9
  char **v11; // r8
  __int64 v12; // rdx
  __int64 v13; // rdx
  bool v14; // zf
  __int64 v15; // rdx
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  char **v20; // [rsp-3F0h] [rbp-3F0h]
  char *v21; // [rsp-3E8h] [rbp-3E8h] BYREF
  struct passwd *v22; // [rsp-3E0h] [rbp-3E0h]
  __int64 v23; // [rsp-3D8h] [rbp-3D8h]
  _BYTE v24[136]; // [rsp-3D0h] [rbp-3D0h] BYREF
  struct passwd v25[3]; // [rsp-348h] [rbp-348h] BYREF
  char *v26; // [rsp-2A8h] [rbp-2A8h] BYREF
  __int64 v27; // [rsp-2A0h] [rbp-2A0h]
  __int64 v28; // [rsp-298h] [rbp-298h]
  _BYTE v29[128]; // [rsp-290h] [rbp-290h] BYREF
  char *v30; // [rsp-210h] [rbp-210h] BYREF
  __int64 v31; // [rsp-208h] [rbp-208h]
  __int64 v32; // [rsp-200h] [rbp-200h]
  _BYTE v33[128]; // [rsp-1F8h] [rbp-1F8h] BYREF
  char *v34; // [rsp-178h] [rbp-178h] BYREF
  struct passwd *v35; // [rsp-170h] [rbp-170h]
  __int64 v36; // [rsp-168h] [rbp-168h]
  _WORD v37[64]; // [rsp-160h] [rbp-160h] BYREF
  char *v38; // [rsp-E0h] [rbp-E0h] BYREF
  __int64 v39; // [rsp-D8h] [rbp-D8h]
  __int64 v40; // [rsp-D0h] [rbp-D0h]
  _BYTE v41[128]; // [rsp-C8h] [rbp-C8h] BYREF
  char v42; // [rsp-48h] [rbp-48h]

  result = (__int64)off_49DCB98;
  *(_DWORD *)(a1 + 8) = 0;
  *(_QWORD *)a1 = off_49DCB98;
  *(_BYTE *)(a1 + 328) = 0;
  if ( a2 )
    return result;
  v20 = (char **)(a1 + 16);
  v21 = v24;
  v22 = 0;
  v23 = 128;
  v25[0].pw_name = (char *)&v25[0].pw_gecos;
  v25[0].pw_passwd = 0;
  *(_QWORD *)&v25[0].pw_uid = 128;
  result = sub_C82800(&v21);
  if ( !(_DWORD)result )
  {
    v6 = (__int64)v25;
    v37[4] = 261;
    v34 = v21;
    v35 = v22;
    v7 = sub_C84130((__int64)&v34, v25, 0, v4, v5);
    v11 = &v26;
    v27 = 0;
    v28 = 128;
    v26 = v29;
    if ( v7 )
    {
      if ( v22 )
      {
        sub_CA1E30((__int64)&v26, (__int64)&v21, v8, v9, (__int64)&v26, v10);
        result = (__int64)v33;
        v31 = 0;
        v30 = v33;
        v11 = &v26;
        v32 = 128;
        if ( v22 )
        {
          sub_CA1E30((__int64)&v30, (__int64)&v21, v15, v9, (__int64)&v26, v10);
          result = (__int64)v33;
          v11 = &v26;
        }
      }
      else
      {
        result = (__int64)v33;
        v31 = 0;
        v30 = v33;
        v32 = 128;
      }
      v12 = v27;
      if ( *(_BYTE *)(a1 + 328) )
      {
        v6 = (__int64)v37;
        v42 &= ~1u;
        v34 = (char *)v37;
        v35 = 0;
        v36 = 128;
        if ( v27 )
        {
          v6 = (__int64)&v26;
          sub_CA1CD0((__int64)&v34, &v26, v27, v9, (__int64)&v26, v10);
          result = (__int64)v33;
        }
        v39 = 0;
        v38 = v41;
        v40 = 128;
        if ( v31 )
        {
          v6 = (__int64)&v30;
          sub_CA1CD0((__int64)&v38, &v30, (__int64)v41, v9, (__int64)v11, v10);
          result = (__int64)v33;
        }
        if ( v20 == &v34 )
          goto LABEL_26;
        if ( (*(_BYTE *)(a1 + 320) & 1) == 0 )
        {
          sub_CA1BE0(v20, v6);
          result = (__int64)v33;
        }
        v13 = (__int64)v35;
        if ( (v42 & 1) == 0 )
        {
          v6 = a1 + 40;
          *(_BYTE *)(a1 + 320) &= ~1u;
          *(_QWORD *)(a1 + 16) = a1 + 40;
          *(_QWORD *)(a1 + 24) = 0;
          *(_QWORD *)(a1 + 32) = 128;
          if ( v13 )
          {
            v6 = (__int64)&v34;
            sub_CA1CD0((__int64)v20, &v34, v13, v9, (__int64)v11, v10);
            result = (__int64)v33;
          }
          v14 = v39 == 0;
          *(_QWORD *)(a1 + 176) = 0;
          *(_QWORD *)(a1 + 168) = a1 + 192;
          *(_QWORD *)(a1 + 184) = 128;
          if ( !v14 )
          {
            v6 = (__int64)&v38;
            sub_CA1CD0(a1 + 168, &v38, a1 + 192, v9, (__int64)v11, v10);
            result = (__int64)v33;
          }
LABEL_26:
          if ( (v42 & 1) == 0 )
          {
            sub_CA1BE0(&v34, v6);
            result = (__int64)v33;
          }
          goto LABEL_38;
        }
LABEL_62:
        v6 = (unsigned int)v34;
        *(_BYTE *)(a1 + 320) |= 1u;
        *(_QWORD *)(a1 + 24) = v13;
        *(_DWORD *)(a1 + 16) = v6;
LABEL_38:
        if ( v30 != v33 )
          result = _libc_free(v30, v6);
        if ( v26 != v29 )
          result = _libc_free(v26, v6);
        goto LABEL_5;
      }
    }
    else
    {
      if ( v22 )
      {
        v6 = (__int64)&v21;
        sub_CA1E30((__int64)&v26, (__int64)&v21, v8, v9, (__int64)&v26, v10);
        v11 = &v26;
      }
      result = (__int64)v33;
      v31 = 0;
      v30 = v33;
      v32 = 128;
      if ( v25[0].pw_passwd )
      {
        v6 = (__int64)v25;
        sub_CA1E30((__int64)&v30, (__int64)v25, v8, v9, (__int64)&v26, v10);
        result = (__int64)v33;
        v11 = &v26;
      }
      v12 = v27;
      if ( *(_BYTE *)(a1 + 328) )
      {
        v42 &= ~1u;
        v34 = (char *)v37;
        v35 = 0;
        v36 = 128;
        if ( v27 )
        {
          v6 = (__int64)&v26;
          sub_CA1CD0((__int64)&v34, &v26, v27, v9, (__int64)&v26, v10);
          result = (__int64)v33;
        }
        v39 = 0;
        v38 = v41;
        v40 = 128;
        if ( v31 )
        {
          v6 = (__int64)&v30;
          sub_CA1CD0((__int64)&v38, &v30, v12, v9, (__int64)v11, v10);
          result = (__int64)v33;
        }
        if ( v20 != &v34 )
        {
          if ( (*(_BYTE *)(a1 + 320) & 1) == 0 )
          {
            sub_CA1BE0(v20, v6);
            result = (__int64)v33;
          }
          v13 = (__int64)v35;
          if ( (v42 & 1) != 0 )
            goto LABEL_62;
          v6 = a1 + 40;
          *(_BYTE *)(a1 + 320) &= ~1u;
          *(_QWORD *)(a1 + 16) = a1 + 40;
          *(_QWORD *)(a1 + 24) = 0;
          *(_QWORD *)(a1 + 32) = 128;
          if ( v13 )
          {
            v6 = (__int64)&v34;
            sub_CA1CD0((__int64)v20, &v34, v13, v9, (__int64)v11, v10);
            result = (__int64)v33;
          }
          v14 = v39 == 0;
          *(_QWORD *)(a1 + 176) = 0;
          *(_QWORD *)(a1 + 168) = a1 + 192;
          *(_QWORD *)(a1 + 184) = 128;
          if ( !v14 )
          {
            v6 = (__int64)&v38;
            sub_CA1CD0(a1 + 168, &v38, a1 + 192, v9, (__int64)v11, v10);
            result = (__int64)v33;
          }
        }
        if ( (v42 & 1) == 0 )
        {
          if ( v38 != v41 )
          {
            _libc_free(v38, v6);
            result = (__int64)v33;
          }
          if ( v34 != (char *)v37 )
          {
            _libc_free(v34, v6);
            result = (__int64)v33;
          }
        }
        goto LABEL_38;
      }
    }
    v6 = a1 + 40;
    *(_BYTE *)(a1 + 320) &= ~1u;
    *(_QWORD *)(a1 + 16) = a1 + 40;
    *(_QWORD *)(a1 + 24) = 0;
    *(_QWORD *)(a1 + 32) = 128;
    if ( v12 )
    {
      v6 = (__int64)&v26;
      sub_CA1CD0((__int64)v20, &v26, v12, v9, (__int64)&v26, v10);
      result = (__int64)v33;
    }
    *(_QWORD *)(a1 + 176) = 0;
    *(_QWORD *)(a1 + 168) = a1 + 192;
    *(_QWORD *)(a1 + 184) = 128;
    if ( v31 )
    {
      v6 = (__int64)&v30;
      sub_CA1CD0(a1 + 168, &v30, a1 + 192, v9, (__int64)v11, v10);
      result = (__int64)v33;
    }
    *(_BYTE *)(a1 + 328) = 1;
    goto LABEL_38;
  }
  v6 = v3;
  if ( *(_BYTE *)(a1 + 328) )
  {
    LODWORD(v34) = result;
    result = (__int64)&v34;
    v42 |= 1u;
    v35 = (struct passwd *)v3;
    if ( v20 != &v34 )
    {
      if ( (*(_BYTE *)(a1 + 320) & 1) != 0 || (sub_CA1BE0(v20, v3), v6 = (__int64)v35, (v42 & 1) != 0) )
      {
        result = (unsigned int)v34;
        *(_BYTE *)(a1 + 320) |= 1u;
        *(_QWORD *)(a1 + 24) = v6;
        *(_DWORD *)(a1 + 16) = result;
      }
      else
      {
        *(_BYTE *)(a1 + 320) &= ~1u;
        *(_QWORD *)(a1 + 16) = a1 + 40;
        *(_QWORD *)(a1 + 24) = 0;
        *(_QWORD *)(a1 + 32) = 128;
        if ( v6 )
        {
          v6 = (__int64)&v34;
          sub_CA1CD0((__int64)v20, &v34, v16, v17, v18, v19);
        }
        result = a1 + 192;
        v14 = v39 == 0;
        *(_QWORD *)(a1 + 176) = 0;
        *(_QWORD *)(a1 + 168) = a1 + 192;
        *(_QWORD *)(a1 + 184) = 128;
        if ( !v14 )
        {
          v6 = (__int64)&v38;
          result = sub_CA1CD0(a1 + 168, &v38, v16, v17, v18, v19);
        }
        if ( (v42 & 1) == 0 )
          result = sub_CA1BE0(&v34, v6);
      }
    }
  }
  else
  {
    *(_BYTE *)(a1 + 320) |= 1u;
    *(_DWORD *)(a1 + 16) = result;
    *(_QWORD *)(a1 + 24) = v3;
    *(_BYTE *)(a1 + 328) = 1;
  }
LABEL_5:
  if ( (char **)v25[0].pw_name != &v25[0].pw_gecos )
    result = _libc_free(v25[0].pw_name, v6);
  if ( v21 != v24 )
    return _libc_free(v21, v6);
  return result;
}
