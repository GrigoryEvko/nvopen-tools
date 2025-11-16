// Function: sub_16D2880
// Address: 0x16d2880
//
void __fastcall sub_16D2880(char **a1, __int64 a2, char a3, int a4, char a5, int a6)
{
  int v6; // r10d
  int v8; // r13d
  __int64 v9; // r11
  char *v10; // r14
  __int64 v11; // r12
  size_t v12; // rdx
  _BYTE *v13; // rax
  unsigned __int64 v14; // rax
  unsigned __int64 v15; // rax
  __int64 v16; // rax
  _QWORD *v17; // rax
  __int64 v18; // r11
  __int64 v19; // rdx
  _QWORD *v20; // rdx
  int v21; // [rsp+8h] [rbp-48h]
  __int64 v23; // [rsp+10h] [rbp-40h]
  __int64 v24; // [rsp+10h] [rbp-40h]
  int v25; // [rsp+18h] [rbp-38h]
  __int64 v26; // [rsp+18h] [rbp-38h]
  unsigned __int64 v27; // [rsp+18h] [rbp-38h]

  v6 = a3;
  v8 = a4;
  v9 = (__int64)a1[1];
  v10 = *a1;
  v11 = v9;
  if ( !a4 )
  {
LABEL_12:
    if ( !a5 && !v9 )
      return;
    goto LABEL_14;
  }
  while ( v11 )
  {
    v12 = 0x7FFFFFFFFFFFFFFFLL;
    v23 = v9;
    if ( v11 >= 0 )
      v12 = v11;
    v25 = v6;
    v13 = memchr(v10, v6, v12);
    v6 = v25;
    v9 = v23;
    if ( !v13 )
      goto LABEL_14;
    v14 = v13 - v10;
    if ( v14 == -1 )
      goto LABEL_14;
    if ( v14 || a5 )
    {
      v18 = 0;
      if ( v14 )
      {
        v18 = v11;
        if ( v14 <= v11 )
          v18 = v14;
      }
      v19 = *(unsigned int *)(a2 + 8);
      if ( (unsigned int)v19 >= *(_DWORD *)(a2 + 12) )
      {
        v21 = v25;
        v24 = v18;
        v27 = v14;
        sub_16CD150(a2, (const void *)(a2 + 16), 0, 16, a5, a6);
        v19 = *(unsigned int *)(a2 + 8);
        v6 = v21;
        v18 = v24;
        v14 = v27;
      }
      v20 = (_QWORD *)(*(_QWORD *)a2 + 16 * v19);
      *v20 = v10;
      v20[1] = v18;
      ++*(_DWORD *)(a2 + 8);
    }
    v15 = v14 + 1;
    if ( v15 > v11 )
      v15 = v11;
    v11 -= v15;
    v10 += v15;
    v9 = v11;
    if ( !--v8 )
      goto LABEL_12;
  }
  if ( a5 )
  {
LABEL_14:
    v16 = *(unsigned int *)(a2 + 8);
    if ( (unsigned int)v16 >= *(_DWORD *)(a2 + 12) )
    {
      v26 = v9;
      sub_16CD150(a2, (const void *)(a2 + 16), 0, 16, a5, a6);
      v16 = *(unsigned int *)(a2 + 8);
      v9 = v26;
    }
    v17 = (_QWORD *)(*(_QWORD *)a2 + 16 * v16);
    *v17 = v10;
    v17[1] = v9;
    ++*(_DWORD *)(a2 + 8);
  }
}
