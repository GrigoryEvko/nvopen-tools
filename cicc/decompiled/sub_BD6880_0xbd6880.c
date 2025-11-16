// Function: sub_BD6880
// Address: 0xbd6880
//
int __fastcall sub_BD6880(unsigned __int8 *a1, const char **a2)
{
  __int64 v3; // rax
  _QWORD *v4; // rax
  bool v5; // zf
  size_t v6; // r14
  const char *v7; // r13
  const char **v8; // rdi
  __int64 v9; // rdx
  __int64 v10; // r15
  const char *v11; // rdi
  __int64 v12; // rdx
  size_t *v13; // r15
  __int64 v15; // [rsp+8h] [rbp-158h] BYREF
  const char *v16; // [rsp+10h] [rbp-150h] BYREF
  size_t v17; // [rsp+18h] [rbp-148h]
  __int64 v18; // [rsp+20h] [rbp-140h]
  _BYTE v19[312]; // [rsp+28h] [rbp-138h] BYREF

  v3 = sub_BD5C60((__int64)a1);
  LODWORD(v4) = sub_B6F8E0(v3);
  if ( !(_BYTE)v4 || *a1 <= 3u )
  {
    LODWORD(v4) = *((unsigned __int8 *)a2 + 32);
    if ( (unsigned __int8)v4 > 1u )
    {
      v5 = *((_BYTE *)a2 + 33) == 1;
      v17 = 0;
      v16 = v19;
      v18 = 256;
      if ( !v5 )
        goto LABEL_9;
    }
    else
    {
      if ( (a1[7] & 0x10) == 0 )
        return (int)v4;
      v5 = *((_BYTE *)a2 + 33) == 1;
      v17 = 0;
      v16 = v19;
      v18 = 256;
      if ( !v5 )
        goto LABEL_9;
      if ( (_BYTE)v4 == 1 )
      {
        v6 = 0;
        v7 = 0;
        goto LABEL_10;
      }
    }
    if ( (unsigned __int8)((_BYTE)v4 - 3) <= 3u )
    {
      if ( (_BYTE)v4 == 4 )
      {
        v7 = *(const char **)*a2;
        v6 = *((_QWORD *)*a2 + 1);
        goto LABEL_10;
      }
      if ( (unsigned __int8)v4 <= 4u )
      {
        if ( (_BYTE)v4 == 3 )
        {
          v7 = *a2;
          v6 = 0;
          if ( *a2 )
            v6 = strlen(v7);
LABEL_10:
          v4 = sub_BD5D20((__int64)a1);
          if ( v9 == v6 )
          {
            if ( !v6 )
              goto LABEL_29;
            a2 = (const char **)v7;
            LODWORD(v4) = memcmp(v4, v7, v6);
            if ( !(_DWORD)v4 )
              goto LABEL_29;
          }
          goto LABEL_11;
        }
      }
      else if ( (unsigned __int8)((_BYTE)v4 - 5) <= 1u )
      {
        v6 = (size_t)a2[1];
        v7 = *a2;
        goto LABEL_10;
      }
      BUG();
    }
LABEL_9:
    v8 = a2;
    a2 = &v16;
    sub_CA0EC0(v8, &v16);
    v6 = v17;
    v7 = v16;
    goto LABEL_10;
  }
  if ( (a1[7] & 0x10) == 0 )
    return (int)v4;
  v17 = 0;
  v16 = v19;
  v18 = 256;
  LODWORD(v4) = (unsigned int)sub_BD5D20((__int64)a1);
  if ( !v12 )
    goto LABEL_29;
  v7 = byte_3F871B3;
  v6 = 0;
LABEL_11:
  a2 = (const char **)&v15;
  LODWORD(v4) = sub_BD3080(a1, &v15);
  if ( (_BYTE)v4 )
    goto LABEL_29;
  v10 = v15;
  if ( !v15 )
  {
    LODWORD(v4) = sub_BD6840((__int64)a1);
    if ( v6 )
    {
      v13 = (size_t *)sub_C7D670(v6 + 17, 8);
      memcpy(v13 + 2, v7, v6);
      *((_BYTE *)v13 + v6 + 16) = 0;
      a2 = (const char **)v13;
      *v13 = v6;
      v13[1] = 0;
      sub_BD6500((__int64)a1, (__int64)v13);
      v4 = (_QWORD *)sub_BD5C70((__int64)a1);
      v4[1] = a1;
    }
LABEL_29:
    v11 = v16;
    if ( v16 == v19 )
      return (int)v4;
    goto LABEL_17;
  }
  if ( (a1[7] & 0x10) != 0 )
  {
    a2 = (const char **)sub_BD5C70((__int64)a1);
    sub_BD8AE0(v10, a2);
    LODWORD(v4) = sub_BD6840((__int64)a1);
    if ( !v6 )
      goto LABEL_29;
    v10 = v15;
  }
  a2 = (const char **)sub_BD8AF0(v10, v7, v6, a1);
  LODWORD(v4) = sub_BD6500((__int64)a1, (__int64)a2);
  v11 = v16;
  if ( v16 != v19 )
LABEL_17:
    LODWORD(v4) = _libc_free(v11, a2);
  return (int)v4;
}
