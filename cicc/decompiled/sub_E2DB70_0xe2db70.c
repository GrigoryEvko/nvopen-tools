// Function: sub_E2DB70
// Address: 0xe2db70
//
void __fastcall sub_E2DB70(__int64 a1, __int64 a2, unsigned int a3)
{
  int v5; // eax
  char *v6; // rdi
  __int64 v7; // rsi
  unsigned __int64 v8; // rcx
  __int64 v9; // r8
  unsigned __int64 v10; // rsi
  unsigned __int64 v11; // rcx
  __int64 v12; // rax
  __int64 v13; // rsi
  __int64 v14; // rax
  unsigned __int64 v15; // rsi
  unsigned __int64 v16; // rcx
  __int64 *v17; // rdi
  unsigned __int64 (__fastcall *v18)(__int64, char **, unsigned int); // rax
  char v19; // si
  unsigned __int64 v20; // rsi
  unsigned __int64 v21; // rcx
  __int64 v22; // rax
  unsigned __int64 v23; // rsi
  unsigned __int64 v24; // rcx
  __int64 v25; // rax
  __int64 v26; // rsi
  unsigned __int64 v27; // rsi
  unsigned __int64 v28; // rcx
  __int64 v29; // rax
  __int64 v30; // rsi
  unsigned int v31; // [rsp+Ch] [rbp-14h]
  unsigned int v32; // [rsp+Ch] [rbp-14h]
  unsigned int v33; // [rsp+Ch] [rbp-14h]
  unsigned int v34; // [rsp+Ch] [rbp-14h]
  unsigned int v35; // [rsp+Ch] [rbp-14h]

  if ( (a3 & 2) != 0 )
    goto LABEL_18;
  v5 = *(_DWORD *)(a1 + 24);
  v6 = *(char **)a2;
  v7 = *(_QWORD *)(a2 + 8);
  v8 = *(_QWORD *)(a2 + 16);
  v9 = *(_QWORD *)a2;
  if ( v5 == 2 )
  {
    if ( v8 < v7 + 5 )
    {
      v23 = v7 + 997;
      v24 = 2 * v8;
      if ( v23 <= v24 )
        *(_QWORD *)(a2 + 16) = v24;
      else
        *(_QWORD *)(a2 + 16) = v23;
      v34 = a3;
      v25 = realloc(v6);
      *(_QWORD *)a2 = v25;
      v9 = v25;
      if ( !v25 )
        goto LABEL_49;
      v7 = *(_QWORD *)(a2 + 8);
      a3 = v34;
    }
    v26 = v9 + v7;
    *(_DWORD *)v26 = 1869180533;
    *(_BYTE *)(v26 + 4) = 110;
    goto LABEL_37;
  }
  if ( v5 <= 2 )
  {
    if ( v5 )
    {
      if ( v5 == 1 )
      {
        if ( v7 + 6 > v8 )
        {
          v10 = v7 + 998;
          v11 = 2 * v8;
          if ( v10 > v11 )
            *(_QWORD *)(a2 + 16) = v10;
          else
            *(_QWORD *)(a2 + 16) = v11;
          v31 = a3;
          v12 = realloc(v6);
          *(_QWORD *)a2 = v12;
          v9 = v12;
          if ( !v12 )
            goto LABEL_49;
          v7 = *(_QWORD *)(a2 + 8);
          a3 = v31;
        }
        v13 = v9 + v7;
        *(_DWORD *)v13 = 1970435187;
        *(_WORD *)(v13 + 4) = 29795;
        v8 = *(_QWORD *)(a2 + 16);
        v6 = *(char **)a2;
        v7 = *(_QWORD *)(a2 + 8) + 6LL;
        *(_QWORD *)(a2 + 8) = v7;
      }
      goto LABEL_12;
    }
    if ( v7 + 5 > v8 )
    {
      v27 = v7 + 997;
      v28 = 2 * v8;
      if ( v27 <= v28 )
        *(_QWORD *)(a2 + 16) = v28;
      else
        *(_QWORD *)(a2 + 16) = v27;
      v35 = a3;
      v29 = realloc(v6);
      *(_QWORD *)a2 = v29;
      v9 = v29;
      if ( !v29 )
        goto LABEL_49;
      v7 = *(_QWORD *)(a2 + 8);
      a3 = v35;
    }
    v30 = v9 + v7;
    *(_DWORD *)v30 = 1935764579;
    *(_BYTE *)(v30 + 4) = 115;
LABEL_37:
    v8 = *(_QWORD *)(a2 + 16);
    v6 = *(char **)a2;
    v7 = *(_QWORD *)(a2 + 8) + 5LL;
    *(_QWORD *)(a2 + 8) = v7;
    goto LABEL_12;
  }
  if ( v5 == 3 )
  {
    if ( v8 < v7 + 4 )
    {
      v20 = v7 + 996;
      v21 = 2 * v8;
      if ( v20 > v21 )
        *(_QWORD *)(a2 + 16) = v20;
      else
        *(_QWORD *)(a2 + 16) = v21;
      v33 = a3;
      v22 = realloc(v6);
      *(_QWORD *)a2 = v22;
      v9 = v22;
      if ( !v22 )
        goto LABEL_49;
      v7 = *(_QWORD *)(a2 + 8);
      a3 = v33;
    }
    *(_DWORD *)(v9 + v7) = 1836412517;
    v8 = *(_QWORD *)(a2 + 16);
    v6 = *(char **)a2;
    v7 = *(_QWORD *)(a2 + 8) + 4LL;
    *(_QWORD *)(a2 + 8) = v7;
  }
LABEL_12:
  v14 = (__int64)v6;
  if ( v7 + 1 > v8 )
  {
    v15 = v7 + 993;
    v16 = 2 * v8;
    if ( v15 > v16 )
      *(_QWORD *)(a2 + 16) = v15;
    else
      *(_QWORD *)(a2 + 16) = v16;
    v32 = a3;
    v14 = realloc(v6);
    *(_QWORD *)a2 = v14;
    if ( v14 )
    {
      v7 = *(_QWORD *)(a2 + 8);
      a3 = v32;
      goto LABEL_17;
    }
LABEL_49:
    abort();
  }
LABEL_17:
  *(_BYTE *)(v14 + v7) = 32;
  ++*(_QWORD *)(a2 + 8);
LABEL_18:
  v17 = *(__int64 **)(a1 + 16);
  v18 = *(unsigned __int64 (__fastcall **)(__int64, char **, unsigned int))(*v17 + 16);
  if ( v18 == sub_E2CA10 )
    sub_E2C8E0(v17[2], (char **)a2, a3, 2u, "::");
  else
    ((void (__fastcall *)(__int64 *, __int64))v18)(v17, a2);
  v19 = *(_BYTE *)(a1 + 12);
  if ( v19 )
    sub_E2A820((__int64 *)a2, v19, 1, 0);
}
