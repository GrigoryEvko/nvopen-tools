// Function: sub_C84130
// Address: 0xc84130
//
__int64 __fastcall sub_C84130(__int64 a1, struct passwd *p_pw_gecos, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned int v5; // r13d
  struct passwd *v7; // r12
  const char *v8; // rdi
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  struct passwd *v12; // rdi
  size_t v13; // rax
  __int64 v14; // rcx
  char *v15; // rdx
  size_t v16; // r13
  char *v17; // r8
  char *v18; // rax
  char *v19; // rdi
  unsigned int v20; // eax
  _BYTE *v21; // rdi
  signed __int64 v22; // rax
  char *v23; // rsi
  char *pw_name; // [rsp+10h] [rbp-10C0h] BYREF
  char *pw_passwd; // [rsp+18h] [rbp-10B8h]
  __int64 v26; // [rsp+20h] [rbp-10B0h]
  char v27; // [rsp+28h] [rbp-10A8h] BYREF
  __int16 v28; // [rsp+30h] [rbp-10A0h]
  _BYTE v29[8]; // [rsp+A8h] [rbp-1028h]
  struct passwd resolved[86]; // [rsp+B0h] [rbp-1020h] BYREF

  p_pw_gecos->pw_passwd = 0;
  if ( *(_BYTE *)(a1 + 32) > 1u )
  {
    v7 = p_pw_gecos;
    if ( (_BYTE)a3 )
    {
      resolved[0].pw_passwd = 0;
      resolved[0].pw_name = (char *)&resolved[0].pw_gecos;
      *(_QWORD *)&resolved[0].pw_uid = 128;
      sub_CA0EC0(a1, resolved);
      sub_C83B00(resolved, resolved);
      v28 = 261;
      pw_name = resolved[0].pw_name;
      pw_passwd = resolved[0].pw_passwd;
      v20 = sub_C84130(&pw_name, p_pw_gecos, 0);
      v19 = resolved[0].pw_name;
      v5 = v20;
      if ( (char **)resolved[0].pw_name == &resolved[0].pw_gecos )
        return v5;
LABEL_18:
      _libc_free(v19, p_pw_gecos);
      return v5;
    }
    pw_passwd = 0;
    pw_name = &v27;
    v26 = 128;
    p_pw_gecos = resolved;
    v8 = (const char *)sub_CA12A0(a1, &pw_name);
    if ( !realpath(v8, (char *)resolved) )
    {
      sub_2241E50(v8, resolved, v9, v10, v11);
      v5 = *__errno_location();
LABEL_15:
      v19 = pw_name;
      if ( pw_name == &v27 )
        return v5;
      goto LABEL_18;
    }
    v12 = resolved;
    v13 = strlen((const char *)resolved);
    v15 = v7->pw_passwd;
    v16 = v13;
    v17 = &v15[v13];
    if ( (unsigned __int64)&v15[v13] > *(_QWORD *)&v7->pw_uid )
    {
      p_pw_gecos = (struct passwd *)&v7->pw_gecos;
      v12 = v7;
      sub_C8D290(v7, &v7->pw_gecos, &v15[v13], 1);
      v15 = v7->pw_passwd;
    }
    if ( v16 )
    {
      v18 = &v15[(unsigned __int64)v7->pw_name];
      if ( (unsigned int)v16 >= 8 )
      {
        v21 = (_BYTE *)((unsigned __int64)(v18 + 8) & 0xFFFFFFFFFFFFFFF8LL);
        *(_QWORD *)v18 = resolved[0].pw_name;
        *(_QWORD *)&v18[(unsigned int)v16 - 8] = *(_QWORD *)&v29[(unsigned int)v16];
        v22 = v18 - v21;
        v23 = (char *)resolved - v22;
        LODWORD(v22) = (unsigned int)(v16 + v22) >> 3;
        qmemcpy(v21, v23, 8LL * (unsigned int)v22);
        p_pw_gecos = (struct passwd *)&v23[8 * (unsigned int)v22];
        v12 = (struct passwd *)&v21[8 * (unsigned int)v22];
        v14 = 0;
LABEL_13:
        v15 = v7->pw_passwd;
        goto LABEL_14;
      }
      if ( (v16 & 4) != 0 )
      {
        *(_DWORD *)v18 = resolved[0].pw_name;
        v14 = *(unsigned int *)&v29[(unsigned int)v16 + 4];
        *(_DWORD *)&v18[(unsigned int)v16 - 4] = v14;
        v15 = v7->pw_passwd;
        goto LABEL_14;
      }
      if ( (_DWORD)v16 )
      {
        *v18 = (char)resolved[0].pw_name;
        if ( (v16 & 2) != 0 )
        {
          v14 = *(unsigned __int16 *)&v29[(unsigned int)v16 + 6];
          *(_WORD *)&v18[(unsigned int)v16 - 2] = v14;
          v15 = v7->pw_passwd;
          goto LABEL_14;
        }
        goto LABEL_13;
      }
    }
LABEL_14:
    v7->pw_passwd = &v15[v16];
    v5 = 0;
    sub_2241E40(v12, p_pw_gecos, v15, v14, v17);
    goto LABEL_15;
  }
  sub_2241E40(a1, p_pw_gecos, a3, a4, a5);
  return 0;
}
