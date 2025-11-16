// Function: sub_723F40
// Address: 0x723f40
//
char *__fastcall sub_723F40(char *src)
{
  char *v1; // r14
  char *v3; // r12
  FILE *v4; // r13
  size_t v5; // r14
  void *v6; // rbx
  char *v7; // rax
  char *v8; // r13
  char *v9; // rbx
  __int64 v10; // rax
  int v11; // eax
  int v12; // r14d
  int v13; // r8d
  size_t v14; // r14
  size_t v15; // rax
  size_t v16; // rax
  size_t v17; // rax
  char *v18; // rax
  unsigned __int64 v19; // rdx
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // [rsp+0h] [rbp-E0h]
  size_t v23; // [rsp+8h] [rbp-D8h]
  size_t v24; // [rsp+10h] [rbp-D0h]
  int v25; // [rsp+10h] [rbp-D0h]
  __int64 v26; // [rsp+20h] [rbp-C0h]
  char v27[9]; // [rsp+33h] [rbp-ADh] BYREF
  char s[10]; // [rsp+3Ch] [rbp-A4h] BYREF
  char v29[10]; // [rsp+46h] [rbp-9Ah] BYREF
  char v30[16]; // [rsp+50h] [rbp-90h] BYREF
  int v31; // [rsp+60h] [rbp-80h]
  char *endptr[14]; // [rsp+70h] [rbp-70h] BYREF

  v1 = (char *)qword_4F078E0;
  if ( qword_4F078E0 )
    return v1;
  v3 = src;
  if ( !unk_4D045A8 )
  {
    if ( qword_4D04570 )
    {
      v4 = (FILE *)sub_685EB0(qword_4D04570, 1, 0, 3562);
      fseek(v4, 0, 2);
      v5 = ftell(v4);
      rewind(v4);
      v6 = (void *)sub_822B10(v5 + 1);
      fread(v6, 1u, v5, v4);
      *((_BYTE *)v6 + v5) = 0;
      fclose(v4);
      qword_4F078E0 = (__int64)v6;
      if ( v6 )
        return (char *)v6;
    }
  }
  if ( dword_4F07588 )
    v7 = **(char ***)(unk_4D03FF0 + 176LL);
  else
    v7 = (char *)*unk_4F07280;
  v8 = qword_4D046E0;
  if ( !qword_4D046E0 )
    v8 = v7;
  if ( src )
  {
    v31 = 0;
    v9 = 0;
    v26 = 0;
    *(_OWORD *)v30 = 0;
  }
  else
  {
    if ( qword_4D046D0 )
    {
      v19 = strtoul(qword_4D046D0, endptr, 0);
      if ( endptr[0] <= qword_4D046D0 || *endptr[0] )
        v19 = sub_723DE0((unsigned __int8 *)qword_4D046D0, 0);
      v3 = v29;
      sprintf(v29, "%08lx", v19);
      v26 = getpid();
    }
    else
    {
      v3 = sub_723ED0((__int64)v8, 0);
      v26 = getpid();
      if ( !v3 )
        v3 = (char *)unk_4F072A0;
    }
    v31 = 0;
    *(_OWORD *)v30 = 0;
    v9 = qword_4F076B0;
    if ( qword_4D046D0 )
      v9 = 0;
    if ( v26 )
      snprintf(v30, 0x13u, "_%ld", v26);
  }
  if ( qword_4D046D8 )
  {
    v10 = sub_723DE0(qword_4D046D8, 0);
    sprintf(s, "_%08lx", v10);
  }
  else
  {
    sprintf(s, "_%08lx", 0);
  }
  v11 = strlen(v3);
  v12 = v11;
  if ( !v9 )
  {
    if ( v11 <= 8 )
    {
      v22 = v11 + 2;
      goto LABEL_17;
    }
    v21 = sub_723DE0((unsigned __int8 *)v3, 0);
LABEL_43:
    v3 = v27;
    v9 = 0;
    sprintf(v27, "%08lx", v21);
    v22 = 10;
    goto LABEL_17;
  }
  v13 = strlen(v9);
  if ( (v13 != 0) + v13 + v12 > 8 )
  {
    v25 = v13;
    v20 = sub_723DE0((unsigned __int8 *)v3, 0);
    v21 = v20;
    if ( v25 )
      v21 = sub_723DE0((unsigned __int8 *)v9, v20);
    goto LABEL_43;
  }
  v22 = v12 + (__int64)v13 + 2 + (v13 != 0);
LABEL_17:
  if ( *v8 != 45 || v8[1] )
  {
    v18 = sub_722110(v8);
    if ( v18 )
      v8 = v18 + 1;
  }
  v14 = strlen(v8);
  sprintf((char *)endptr, "_%lu_", v14);
  v24 = strlen((const char *)endptr);
  v23 = strlen(v30);
  v15 = strlen(s);
  v1 = (char *)sub_822B10(v23 + v22 + v14 + v24 + v15);
  strcpy(v1, s);
  strcat(v1, (const char *)endptr);
  strcat(v1, v8);
  v16 = strlen(v1);
  v1[v16] = 95;
  strcpy(&v1[v16 + 1], v3);
  if ( v9 )
  {
    v17 = strlen(v1);
    v1[v17] = 95;
    strcpy(&v1[v17 + 1], v9);
  }
  if ( v26 && !qword_4D046D0 )
    strcat(v1, v30);
  sub_721750((unsigned __int8 *)v1);
  qword_4F078E0 = (__int64)v1;
  return v1;
}
