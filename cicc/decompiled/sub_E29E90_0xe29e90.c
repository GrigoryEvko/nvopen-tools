// Function: sub_E29E90
// Address: 0xe29e90
//
__int64 __fastcall sub_E29E90(__int64 a1, __int64 *a2)
{
  char *v4; // rsi
  unsigned __int64 v5; // rax
  __int64 v6; // rdi
  unsigned __int64 v7; // rsi
  unsigned __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // r8
  unsigned __int64 v11; // rcx
  _BYTE *v12; // r12
  unsigned __int64 v13; // rax
  _BYTE *v14; // r9
  size_t v15; // r13
  unsigned __int64 v16; // rax
  char *v17; // rdi
  unsigned __int64 v18; // rax
  __int64 v19; // rax
  char *v21; // r8
  unsigned __int64 v22; // rax
  char *v23; // rdi
  unsigned __int64 v24; // rsi
  unsigned __int64 v25; // rax
  __int64 v26; // rax
  _BYTE v27[43]; // [rsp+15h] [rbp-2Bh] BYREF

  v4 = (char *)a2[1];
  v5 = a2[2];
  v6 = *a2;
  if ( (unsigned __int64)(v4 + 8) > v5 )
  {
    v7 = (unsigned __int64)(v4 + 1000);
    v8 = 2 * v5;
    if ( v7 <= v8 )
      a2[2] = v8;
    else
      a2[2] = v7;
    v9 = realloc((void *)v6);
    *a2 = v9;
    v6 = v9;
    if ( !v9 )
      goto LABEL_24;
    v4 = (char *)a2[1];
  }
  *(_QWORD *)&v4[v6] = 0x7B276C6C61637660LL;
  v10 = a2[1] + 8;
  a2[1] = v10;
  v11 = *(_QWORD *)(a1 + 24);
  v12 = v27;
  do
  {
    *--v12 = v11 % 0xA + 48;
    v13 = v11;
    v11 /= 0xAu;
  }
  while ( v13 > 9 );
  v14 = (_BYTE *)(v27 - v12);
  v15 = v27 - v12;
  if ( v27 != v12 )
  {
    v22 = a2[2];
    v23 = (char *)*a2;
    if ( (unsigned __int64)&v14[v10] > v22 )
    {
      v24 = (unsigned __int64)&v14[v10 + 992];
      v25 = 2 * v22;
      if ( v24 > v25 )
        a2[2] = v24;
      else
        a2[2] = v25;
      v26 = realloc(v23);
      *a2 = v26;
      v23 = (char *)v26;
      if ( !v26 )
        goto LABEL_24;
      v10 = a2[1];
    }
    memcpy(&v23[v10], v12, v15);
    v10 = v15 + a2[1];
    a2[1] = v10;
  }
  v16 = a2[2];
  v17 = (char *)*a2;
  if ( v10 + 9 > v16 )
  {
    v18 = 2 * v16;
    if ( v10 + 1001 <= v18 )
      a2[2] = v18;
    else
      a2[2] = v10 + 1001;
    v19 = realloc(v17);
    *a2 = v19;
    v17 = (char *)v19;
    if ( v19 )
    {
      v10 = a2[1];
      goto LABEL_14;
    }
LABEL_24:
    abort();
  }
LABEL_14:
  v21 = &v17[v10];
  *(_QWORD *)v21 = 0x7D74616C667B202CLL;
  v21[8] = 125;
  a2[1] += 9;
  return 0x7D74616C667B202CLL;
}
