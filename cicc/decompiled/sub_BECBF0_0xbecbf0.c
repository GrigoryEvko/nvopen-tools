// Function: sub_BECBF0
// Address: 0xbecbf0
//
void __fastcall sub_BECBF0(_BYTE *a1, const char *a2)
{
  unsigned __int8 v4; // al
  const char **v5; // rbx
  __int64 v6; // rdx
  const char **v7; // r14
  const char *v8; // rax
  __int64 v9; // r13
  _BYTE *v10; // rax
  __int64 v11; // rsi
  __int64 v12; // rdi
  _BYTE *v13; // rax
  unsigned __int8 v14; // cl
  int v15; // esi
  __int64 v16; // r10
  const char *v17; // r9
  const char *v18; // r8
  __int64 v19; // rdi
  const char *v20; // rax
  __int64 v21; // r8
  _BYTE *v22; // rax
  __int64 v23; // rsi
  __int64 v24; // rdi
  _BYTE *v25; // rax
  unsigned __int8 v26; // si
  int v27; // ecx
  __int64 v28; // r9
  const char *v29; // r8
  _BYTE *v30; // rdi
  const char *v31; // rax
  __int64 v32; // [rsp+8h] [rbp-78h]
  const char *v33; // [rsp+10h] [rbp-70h] BYREF
  const char *v34; // [rsp+18h] [rbp-68h] BYREF
  _QWORD v35[4]; // [rsp+20h] [rbp-60h] BYREF
  char v36; // [rsp+40h] [rbp-40h]
  char v37; // [rsp+41h] [rbp-3Fh]

  v4 = *(a2 - 16);
  if ( (v4 & 2) != 0 )
  {
    v5 = (const char **)*((_QWORD *)a2 - 4);
    v6 = *((unsigned int *)a2 - 6);
  }
  else
  {
    v6 = (*((_WORD *)a2 - 8) >> 6) & 0xF;
    v5 = (const char **)&a2[-8 * ((v4 >> 2) & 0xF) - 16];
  }
  v7 = &v5[v6];
  if ( v5 == v7 )
    return;
  while ( 1 )
  {
    v8 = *v5;
    if ( (unsigned __int8)(**v5 - 5) > 0x1Fu )
      break;
    v33 = *v5;
    v14 = *(v8 - 16);
    if ( (v14 & 2) != 0 )
    {
      v15 = *((_DWORD *)v8 - 6);
      if ( (unsigned int)(v15 - 2) > 1 )
        goto LABEL_23;
      v16 = *((_QWORD *)v8 - 4);
      v17 = v8 - 16;
      v18 = *(const char **)v16;
      v19 = v16;
      if ( v8 == *(const char **)v16 )
      {
        if ( v15 == 3 )
          goto LABEL_36;
        goto LABEL_39;
      }
    }
    else
    {
      v15 = (*((_WORD *)v8 - 8) >> 6) & 0xF;
      if ( (unsigned int)(v15 - 2) > 1 )
      {
LABEL_23:
        v21 = *(_QWORD *)a1;
        v37 = 1;
        v35[0] = "scope must have two or three operands";
        v36 = 3;
        if ( v21 )
        {
          v32 = v21;
          sub_CA0E80(v35, v21);
          v22 = *(_BYTE **)(v32 + 32);
          if ( (unsigned __int64)v22 >= *(_QWORD *)(v32 + 24) )
          {
            sub_CB5D20(v32, 10);
          }
          else
          {
            *(_QWORD *)(v32 + 32) = v22 + 1;
            *v22 = 10;
          }
          v23 = *(_QWORD *)a1;
          a1[152] = 1;
          if ( v23 && v33 )
          {
            sub_A62C00(v33, v23, (__int64)(a1 + 16), *((_QWORD *)a1 + 1));
            v24 = *(_QWORD *)a1;
            v25 = *(_BYTE **)(*(_QWORD *)a1 + 32LL);
            if ( (unsigned __int64)v25 >= *(_QWORD *)(*(_QWORD *)a1 + 24LL) )
            {
              sub_CB5D20(v24, 10);
            }
            else
            {
              *(_QWORD *)(v24 + 32) = v25 + 1;
              *v25 = 10;
            }
          }
        }
        else
        {
          a1[152] = 1;
        }
        goto LABEL_30;
      }
      v17 = v8 - 16;
      v19 = (__int64)&v8[-8 * ((v14 >> 2) & 0xF) - 16];
      v18 = *(const char **)v19;
      if ( v8 == *(const char **)v19 )
      {
        if ( v15 == 3 )
        {
LABEL_36:
          if ( **(_BYTE **)(v19 + 16) )
          {
            v37 = 1;
            v35[0] = "third scope operand must be string (if used)";
            goto LABEL_21;
          }
        }
        else
        {
LABEL_18:
          v19 = (__int64)&v17[-8 * ((v14 >> 2) & 0xF)];
        }
        v20 = *(const char **)(v19 + 8);
        if ( (unsigned __int8)(*v20 - 5) > 0x1Fu )
          goto LABEL_20;
        goto LABEL_40;
      }
    }
    if ( *v18 )
    {
      v37 = 1;
      v35[0] = "first scope operand must be self-referential or string";
      goto LABEL_21;
    }
    if ( v15 == 3 )
    {
      if ( (*(v8 - 16) & 2) != 0 )
        v19 = *((_QWORD *)v8 - 4);
      else
        v19 = (__int64)&v17[-8 * ((v14 >> 2) & 0xF)];
      goto LABEL_36;
    }
    if ( (*(v8 - 16) & 2) == 0 )
      goto LABEL_18;
    v16 = *((_QWORD *)v8 - 4);
LABEL_39:
    v20 = *(const char **)(v16 + 8);
    if ( (unsigned __int8)(*v20 - 5) > 0x1Fu )
    {
LABEL_20:
      v34 = 0;
      v37 = 1;
      v35[0] = "second scope operand must be MDNode";
LABEL_21:
      v36 = 3;
      sub_BE1BE0(a1, (__int64)v35, &v33);
      goto LABEL_30;
    }
LABEL_40:
    v34 = v20;
    v26 = *(v20 - 16);
    if ( (v26 & 2) != 0 )
    {
      v27 = *((_DWORD *)v20 - 6);
      if ( (unsigned int)(v27 - 1) > 1 )
        goto LABEL_56;
      v28 = *((_QWORD *)v20 - 4);
      v29 = v20 - 16;
      v30 = *(_BYTE **)v28;
      if ( *(const char **)v28 == v20 )
        goto LABEL_52;
    }
    else
    {
      v27 = (*((_WORD *)v20 - 8) >> 6) & 0xF;
      if ( (unsigned int)(v27 - 1) > 1 )
      {
LABEL_56:
        v37 = 1;
        v31 = "domain must have one or two operands";
        goto LABEL_57;
      }
      v29 = v20 - 16;
      v28 = (__int64)&v20[-8 * ((v26 >> 2) & 0xF) - 16];
      v30 = *(_BYTE **)v28;
      if ( *(const char **)v28 == v20 )
      {
LABEL_52:
        if ( v27 != 2 )
          goto LABEL_30;
        goto LABEL_47;
      }
    }
    if ( *v30 )
    {
      v37 = 1;
      v31 = "first domain operand must be self-referential or string";
      goto LABEL_57;
    }
    if ( v27 != 2 )
      goto LABEL_30;
    if ( (*(v20 - 16) & 2) != 0 )
      v28 = *((_QWORD *)v20 - 4);
    else
      v28 = (__int64)&v29[-8 * ((v26 >> 2) & 0xF)];
LABEL_47:
    if ( **(_BYTE **)(v28 + 8) )
    {
      v37 = 1;
      v31 = "second domain operand must be string (if used)";
LABEL_57:
      v35[0] = v31;
      v36 = 3;
      sub_BECB10(a1, (__int64)v35, &v34);
    }
LABEL_30:
    if ( v7 == ++v5 )
      return;
  }
  v9 = *(_QWORD *)a1;
  v37 = 1;
  v35[0] = "scope list must consist of MDNodes";
  v36 = 3;
  if ( v9 )
  {
    sub_CA0E80(v35, v9);
    v10 = *(_BYTE **)(v9 + 32);
    if ( (unsigned __int64)v10 >= *(_QWORD *)(v9 + 24) )
    {
      sub_CB5D20(v9, 10);
    }
    else
    {
      *(_QWORD *)(v9 + 32) = v10 + 1;
      *v10 = 10;
    }
    v11 = *(_QWORD *)a1;
    a1[152] = 1;
    if ( v11 )
    {
      sub_A62C00(a2, v11, (__int64)(a1 + 16), *((_QWORD *)a1 + 1));
      v12 = *(_QWORD *)a1;
      v13 = *(_BYTE **)(*(_QWORD *)a1 + 32LL);
      if ( (unsigned __int64)v13 >= *(_QWORD *)(*(_QWORD *)a1 + 24LL) )
      {
        sub_CB5D20(v12, 10);
      }
      else
      {
        *(_QWORD *)(v12 + 32) = v13 + 1;
        *v13 = 10;
      }
    }
  }
  else
  {
    a1[152] = 1;
  }
}
