// Function: sub_E2A620
// Address: 0xe2a620
//
__int64 __fastcall sub_E2A620(__int64 *a1, char a2, char a3)
{
  char *v6; // rsi
  unsigned __int64 v7; // rax
  __int64 v8; // rdi
  unsigned __int64 v9; // rsi
  unsigned __int64 v10; // rax
  __int64 v11; // rax
  char *v12; // rsi
  unsigned __int64 v13; // rax
  __int64 v14; // rdi
  unsigned __int64 v15; // rsi
  unsigned __int64 v16; // rax
  __int64 v17; // rax
  char *v18; // rsi
  unsigned __int64 v19; // rax
  char *v20; // rdi
  unsigned __int64 v21; // rsi
  unsigned __int64 v22; // rax
  __int64 v23; // rax
  char *v24; // rdi
  char *v25; // rsi
  unsigned __int64 v26; // rax
  char *v27; // rdi
  unsigned __int64 v28; // rsi
  unsigned __int64 v29; // rax
  __int64 v30; // rax

  if ( !a3 )
  {
    if ( a2 != 2 )
      goto LABEL_3;
LABEL_12:
    v12 = (char *)a1[1];
    v13 = a1[2];
    v14 = *a1;
    if ( (unsigned __int64)(v12 + 8) > v13 )
    {
      v15 = (unsigned __int64)(v12 + 1000);
      v16 = 2 * v13;
      if ( v15 > v16 )
        a1[2] = v15;
      else
        a1[2] = v16;
      v17 = realloc((void *)v14);
      *a1 = v17;
      v14 = v17;
      if ( !v17 )
        goto LABEL_34;
      v12 = (char *)a1[1];
    }
    *(_QWORD *)&v12[v14] = 0x656C6974616C6F76LL;
    a1[1] += 8;
    return 1;
  }
  v6 = (char *)a1[1];
  v7 = a1[2];
  v8 = *a1;
  if ( (unsigned __int64)(v6 + 1) > v7 )
  {
    v9 = (unsigned __int64)(v6 + 993);
    v10 = 2 * v7;
    if ( v9 > v10 )
      a1[2] = v9;
    else
      a1[2] = v10;
    v11 = realloc((void *)v8);
    *a1 = v11;
    v8 = v11;
    if ( !v11 )
      goto LABEL_34;
    v6 = (char *)a1[1];
  }
  v6[v8] = 32;
  ++a1[1];
  if ( a2 == 2 )
    goto LABEL_12;
LABEL_3:
  if ( a2 != 32 )
  {
    if ( a2 != 1 )
      return 1;
    v18 = (char *)a1[1];
    v19 = a1[2];
    v20 = (char *)*a1;
    if ( (unsigned __int64)(v18 + 5) <= v19 )
      goto LABEL_23;
    v21 = (unsigned __int64)(v18 + 997);
    v22 = 2 * v19;
    if ( v21 > v22 )
      a1[2] = v21;
    else
      a1[2] = v22;
    v23 = realloc(v20);
    *a1 = v23;
    v20 = (char *)v23;
    if ( v23 )
    {
      v18 = (char *)a1[1];
LABEL_23:
      v24 = &v20[(_QWORD)v18];
      *(_DWORD *)v24 = 1936617315;
      v24[4] = 116;
      a1[1] += 5;
      return 1;
    }
LABEL_34:
    abort();
  }
  v25 = (char *)a1[1];
  v26 = a1[2];
  v27 = (char *)*a1;
  if ( (unsigned __int64)(v25 + 10) > v26 )
  {
    v28 = (unsigned __int64)(v25 + 1002);
    v29 = 2 * v26;
    if ( v28 > v29 )
      a1[2] = v28;
    else
      a1[2] = v29;
    v30 = realloc(v27);
    *a1 = v30;
    v27 = (char *)v30;
    if ( !v30 )
      goto LABEL_34;
    v25 = (char *)a1[1];
  }
  qmemcpy(&v27[(_QWORD)v25], "__restrict", 10);
  a1[1] += 10;
  return 1;
}
