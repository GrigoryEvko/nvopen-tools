// Function: sub_16F8F10
// Address: 0x16f8f10
//
const char *__fastcall sub_16F8F10(__int64 a1, _DWORD *a2)
{
  char *v2; // r12
  unsigned __int64 v3; // rbx
  unsigned __int64 v4; // rax
  signed __int64 v5; // rbx
  size_t v6; // rdx
  _BYTE *v7; // rax
  int v8; // r9d
  signed __int64 v9; // r13
  unsigned __int64 v10; // rax
  unsigned int v11; // r8d
  int v12; // r8d
  int v13; // r9d
  __int64 v14; // rax
  unsigned __int64 v15; // r13
  int v16; // eax
  size_t v17; // rdx
  _BYTE *v18; // rax
  char *v19; // rcx
  unsigned __int64 v21; // rax
  unsigned __int64 v22; // rcx
  int v23; // r9d
  unsigned int v24; // [rsp+Ch] [rbp-44h]
  char *v25; // [rsp+10h] [rbp-40h] BYREF
  unsigned __int64 v26; // [rsp+18h] [rbp-38h]

  v2 = *(char **)(a1 + 72);
  v3 = *(_QWORD *)(a1 + 80);
  if ( *v2 == 34 )
  {
    if ( v3 )
    {
      v21 = v3 - 2;
      if ( v21 <= --v3 )
        v3 = v21;
      ++v2;
    }
    v25 = v2;
    v26 = v3;
    v22 = sub_16D23E0(&v25, "\\\r\n", 3, 0);
    if ( v22 == -1 )
      return v25;
    else
      return sub_16F8390(a1, v25, v26, v22, a2, v23);
  }
  if ( *v2 != 39 )
  {
    sub_16D2650((_QWORD *)(a1 + 72), 32, 0xFFFFFFFFFFFFFFFFLL);
    return *(const char **)(a1 + 72);
  }
  if ( !v3 )
    return v2;
  v4 = v3 - 2;
  v5 = v3 - 1;
  if ( v4 <= v5 )
    v5 = v4;
  ++v2;
  if ( !v5 )
    return v2;
  v6 = 0x7FFFFFFFFFFFFFFFLL;
  if ( v5 >= 0 )
    v6 = v5;
  v7 = memchr(v2, 39, v6);
  if ( !v7 )
    return v2;
  v9 = v7 - v2;
  if ( v7 - v2 == -1 )
    return v2;
  v10 = (unsigned int)a2[3];
  a2[2] = 0;
  v11 = 0;
  if ( v10 < v5 )
  {
    sub_16CD150((__int64)a2, a2 + 4, v5, 1, 0, v8);
    v11 = a2[2];
  }
  while ( 1 )
  {
    sub_16F64E0((__int64)a2, (char *)(*(_QWORD *)a2 + v11), v2, &v2[v9]);
    v14 = (unsigned int)a2[2];
    if ( (unsigned int)v14 >= a2[3] )
    {
      sub_16CD150((__int64)a2, a2 + 4, 0, 1, v12, v13);
      v14 = (unsigned int)a2[2];
    }
    v15 = v9 + 2;
    *(_BYTE *)(*(_QWORD *)a2 + v14) = 39;
    v16 = a2[2];
    v11 = v16 + 1;
    a2[2] = v16 + 1;
    if ( v15 > v5 )
    {
      v2 += v5;
      v19 = v2;
      goto LABEL_21;
    }
    v5 -= v15;
    v2 += v15;
    if ( v5 != -1 )
      break;
    v17 = 0x7FFFFFFFFFFFFFFFLL;
LABEL_18:
    v24 = v16 + 1;
    v18 = memchr(v2, 39, v17);
    v11 = v24;
    if ( v18 )
    {
      v9 = v18 - v2;
      if ( v18 - v2 != -1 )
        continue;
    }
    v19 = &v2[v5];
    goto LABEL_21;
  }
  if ( v5 )
  {
    v17 = 0x7FFFFFFFFFFFFFFFLL;
    if ( v5 >= 0 )
      v17 = v5;
    goto LABEL_18;
  }
  v19 = v2;
LABEL_21:
  sub_16F64E0((__int64)a2, (char *)(*(_QWORD *)a2 + v11), v2, v19);
  return *(const char **)a2;
}
