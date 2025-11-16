// Function: sub_E2E1C0
// Address: 0xe2e1c0
//
unsigned __int64 __fastcall sub_E2E1C0(__int64 a1, __int64 *a2, unsigned int a3)
{
  char v6; // si
  __int64 *v7; // rdi
  unsigned __int64 (__fastcall *v8)(__int64, char **, unsigned int); // rax
  unsigned __int64 result; // rax
  __int64 v10; // rsi
  unsigned __int64 v11; // rax
  char *v12; // rdi
  unsigned __int64 v13; // rsi
  unsigned __int64 v14; // rax
  __int64 v15; // rax
  char *v16; // rdi
  __int64 *v17; // rdi
  unsigned __int64 (__fastcall *v18)(__int64, char **, unsigned int); // rax
  __int64 v19; // rsi
  unsigned __int64 v20; // rax
  char *v21; // rdi
  unsigned __int64 v22; // rsi
  unsigned __int64 v23; // rax
  __int64 v24; // rax

  v6 = *(_BYTE *)(a1 + 32);
  if ( v6 )
    sub_E2A820(a2, v6, 0, 1);
  v7 = *(__int64 **)(a1 + 16);
  v8 = *(unsigned __int64 (__fastcall **)(__int64, char **, unsigned int))(*v7 + 16);
  if ( v8 == sub_E2CA10 )
    result = sub_E2C8E0(v7[2], (char **)a2, a3, 2u, "::");
  else
    result = v8((__int64)v7, (char **)a2, a3);
  if ( *(_QWORD *)(a1 + 24) )
  {
    v10 = a2[1];
    v11 = a2[2];
    v12 = (char *)*a2;
    if ( v10 + 6 > v11 )
    {
      v13 = v10 + 998;
      v14 = 2 * v11;
      if ( v13 > v14 )
        a2[2] = v13;
      else
        a2[2] = v14;
      v15 = realloc(v12);
      *a2 = v15;
      v12 = (char *)v15;
      if ( !v15 )
        goto LABEL_24;
      v10 = a2[1];
    }
    v16 = &v12[v10];
    *(_DWORD *)v16 = 1919903355;
    *((_WORD *)v16 + 2) = 24608;
    a2[1] += 6;
    v17 = *(__int64 **)(a1 + 24);
    v18 = *(unsigned __int64 (__fastcall **)(__int64, char **, unsigned int))(*v17 + 16);
    if ( v18 == sub_E2CA10 )
      sub_E2C8E0(v17[2], (char **)a2, a3, 2u, "::");
    else
      v18((__int64)v17, (char **)a2, a3);
    v19 = a2[1];
    v20 = a2[2];
    v21 = (char *)*a2;
    if ( v19 + 2 <= v20 )
      goto LABEL_18;
    v22 = v19 + 994;
    v23 = 2 * v20;
    if ( v22 <= v23 )
      a2[2] = v23;
    else
      a2[2] = v22;
    v24 = realloc(v21);
    *a2 = v24;
    v21 = (char *)v24;
    if ( v24 )
    {
      v19 = a2[1];
LABEL_18:
      result = 32039;
      *(_WORD *)&v21[v19] = 32039;
      a2[1] += 2;
      return result;
    }
LABEL_24:
    abort();
  }
  return result;
}
