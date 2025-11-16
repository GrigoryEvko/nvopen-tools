// Function: sub_E2EB40
// Address: 0xe2eb40
//
void __fastcall sub_E2EB40(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 v5; // rsi
  unsigned __int64 v6; // rax
  char *v7; // rdi
  unsigned __int64 v8; // rsi
  unsigned __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rdi
  unsigned __int64 (__fastcall *v12)(__int64, char **, unsigned int); // rax
  __int64 v13; // rsi
  unsigned __int64 v14; // rax
  char *v15; // rdi
  unsigned __int64 v16; // rsi
  unsigned __int64 v17; // rax
  __int64 v18; // rax
  unsigned int v19; // [rsp-1Ch] [rbp-1Ch]

  if ( *(_QWORD *)(a1 + 16) )
  {
    v5 = *(_QWORD *)(a2 + 8);
    v6 = *(_QWORD *)(a2 + 16);
    v7 = *(char **)a2;
    if ( v5 + 1 > v6 )
    {
      v8 = v5 + 993;
      v9 = 2 * v6;
      if ( v8 > v9 )
        *(_QWORD *)(a2 + 16) = v8;
      else
        *(_QWORD *)(a2 + 16) = v9;
      v19 = a3;
      v10 = realloc(v7);
      *(_QWORD *)a2 = v10;
      v7 = (char *)v10;
      if ( !v10 )
        goto LABEL_19;
      v5 = *(_QWORD *)(a2 + 8);
      a3 = v19;
    }
    v7[v5] = 60;
    ++*(_QWORD *)(a2 + 8);
    v11 = *(_QWORD *)(a1 + 16);
    v12 = *(unsigned __int64 (__fastcall **)(__int64, char **, unsigned int))(*(_QWORD *)v11 + 16LL);
    if ( v12 == sub_E2C9F0 )
      sub_E2C8E0(v11, (char **)a2, a3, 2u, ", ");
    else
      ((void (__fastcall *)(__int64, __int64))v12)(v11, a2);
    v13 = *(_QWORD *)(a2 + 8);
    v14 = *(_QWORD *)(a2 + 16);
    v15 = *(char **)a2;
    if ( v13 + 1 <= v14 )
      goto LABEL_14;
    v16 = v13 + 993;
    v17 = 2 * v14;
    if ( v16 > v17 )
      *(_QWORD *)(a2 + 16) = v16;
    else
      *(_QWORD *)(a2 + 16) = v17;
    v18 = realloc(v15);
    *(_QWORD *)a2 = v18;
    v15 = (char *)v18;
    if ( v18 )
    {
      v13 = *(_QWORD *)(a2 + 8);
LABEL_14:
      v15[v13] = 62;
      ++*(_QWORD *)(a2 + 8);
      return;
    }
LABEL_19:
    abort();
  }
}
