// Function: sub_67F9E0
// Address: 0x67f9e0
//
_BYTE *__fastcall sub_67F9E0(__int64 a1, __int64 a2, __int64 a3)
{
  char *v5; // r15
  char *v6; // rbx
  char *v7; // rdi
  __int64 v8; // rcx
  char *v9; // rdx
  __int64 v10; // r12
  size_t v11; // rax
  __int64 v12; // rbx
  signed __int64 v13; // rsi
  size_t v14; // r8
  __int64 v15; // rcx
  char *v16; // r14
  char *v17; // rax
  char *v18; // rcx
  char *v19; // rdx
  size_t v20; // rbx
  _BYTE *result; // rax
  char *v22; // r10
  __int64 v23; // rax
  char *v24; // rax
  char *v25; // rdx
  __int64 v26; // rax
  char *v27; // rax
  size_t v28; // [rsp+0h] [rbp-B0h]
  char *v29; // [rsp+8h] [rbp-A8h]
  size_t v30; // [rsp+10h] [rbp-A0h]
  __int64 v31; // [rsp+18h] [rbp-98h]
  __int64 v32; // [rsp+18h] [rbp-98h]
  __int64 v33; // [rsp+18h] [rbp-98h]
  char s[16]; // [rsp+20h] [rbp-90h] BYREF
  int v35; // [rsp+30h] [rbp-80h]
  char v36; // [rsp+34h] [rbp-7Ch]
  char *v37; // [rsp+60h] [rbp-50h]
  __int64 v38; // [rsp+68h] [rbp-48h]
  __int64 v39; // [rsp+70h] [rbp-40h]

  v5 = (char *)(a1 + 8);
  v6 = &s[8];
  v7 = *(char **)(a1 + 64);
  v39 = 0;
  *(_DWORD *)&s[4] = 1;
  v37 = &s[8];
  v38 = 22;
  if ( v7 == v5 )
  {
    *(_QWORD *)(a1 + 64) = 0;
    v8 = 0;
    v10 = 22;
    *(_QWORD *)(a1 + 72) = 0;
    *(_QWORD *)(a1 + 80) = 0;
  }
  else
  {
    sub_823A00(v7, *(_QWORD *)(a1 + 72));
    v8 = v39;
    v9 = v37;
    *(_DWORD *)(a1 + 4) = 0;
    *(_QWORD *)(a1 + 64) = 0;
    v10 = v38;
    *(_QWORD *)(a1 + 72) = 0;
    *(_QWORD *)(a1 + 80) = v8;
    if ( v9 != &s[8] )
      goto LABEL_3;
    if ( v10 > 50 )
    {
      v33 = v8;
      v26 = sub_823970(v10);
      v8 = v33;
      v9 = (char *)v26;
      goto LABEL_31;
    }
  }
  *(_DWORD *)(a1 + 4) = 1;
  v9 = v5;
LABEL_31:
  v27 = v9;
  if ( v8 > 0 )
  {
    do
    {
      if ( v27 )
        *v27 = *v6;
      ++v27;
      ++v6;
    }
    while ( &v9[v8] != v27 );
  }
LABEL_3:
  *(_QWORD *)(a1 + 64) = v9;
  *(_QWORD *)(a1 + 72) = v10;
  v37 = 0;
  v38 = 0;
  v39 = 0;
  sub_823A00(0, 0);
  *(_OWORD *)s = 0;
  v35 = 0;
  v36 = 0;
  snprintf(s, 0x15u, "%llu", a3);
  v11 = strlen(s);
  v12 = *(_QWORD *)(a1 + 80);
  v13 = *(_QWORD *)(a1 + 72);
  v14 = v11;
  v15 = v11 + v12;
  if ( (__int64)(v11 + v12) <= v13 )
  {
    v16 = *(char **)(a1 + 64);
    goto LABEL_5;
  }
  v22 = *(char **)(a1 + 64);
  if ( (!*(_DWORD *)(a1 + 4) || v5 == v22) && v15 <= 50 )
  {
    *(_DWORD *)(a1 + 4) = 1;
    v16 = v5;
  }
  else
  {
    v28 = v11;
    v29 = *(char **)(a1 + 64);
    v31 = v11 + v12;
    v23 = sub_823970(v15);
    v14 = v28;
    v22 = v29;
    v15 = v31;
    v16 = (char *)v23;
  }
  if ( v22 == v16 )
    goto LABEL_24;
  v24 = v16;
  v25 = v22;
  if ( v12 > 0 )
  {
    do
    {
      if ( v24 )
        *v24 = *v25;
      ++v24;
      ++v25;
    }
    while ( &v16[v12] != v24 );
  }
  if ( v5 != v22 )
  {
    v30 = v14;
    v32 = v15;
    sub_823A00(v22, v13);
    v14 = v30;
    v15 = v32;
LABEL_24:
    *(_QWORD *)(a1 + 64) = v16;
    *(_QWORD *)(a1 + 72) = v15;
    goto LABEL_5;
  }
  *(_DWORD *)(a1 + 4) = 0;
  *(_QWORD *)(a1 + 64) = v16;
  *(_QWORD *)(a1 + 72) = v15;
LABEL_5:
  v17 = &v16[v12];
  v18 = &v16[v15];
  v19 = s;
  if ( v14 )
  {
    do
    {
      if ( v17 )
        *v17 = *v19;
      ++v17;
      ++v19;
    }
    while ( v18 != v17 );
  }
  v20 = v14 + *(_QWORD *)(a1 + 80);
  *(_QWORD *)(a1 + 80) = v20;
  if ( *(_QWORD *)(a1 + 72) == v20 )
    sub_67F5B0(a1);
  result = (_BYTE *)(*(_QWORD *)(a1 + 64) + v20);
  if ( result )
    *result = 0;
  *(_QWORD *)(a1 + 80) = v20 + 1;
  return result;
}
