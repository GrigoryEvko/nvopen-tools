// Function: sub_E14510
// Address: 0xe14510
//
unsigned __int64 __fastcall sub_E14510(__int64 a1, __int64 a2)
{
  _BYTE *v3; // r14
  int v4; // eax
  int v5; // r15d
  __int64 v6; // r12
  int v7; // r13d
  int i; // r12d
  __int64 v9; // rsi
  unsigned __int64 v10; // rax
  char *v11; // rdi
  unsigned __int64 v12; // rsi
  unsigned __int64 v13; // rax
  __int64 v14; // rax
  unsigned __int64 result; // rax
  __int64 v16; // rdx
  char *v17; // rdi
  unsigned __int64 v18; // rax
  int v19; // [rsp+Ch] [rbp-34h]

  sub_E12F20((__int64 *)a2, 9u, "sizeof...");
  ++*(_DWORD *)(a2 + 32);
  sub_E14360(a2, 40);
  v3 = *(_BYTE **)(a1 + 16);
  v4 = *(_DWORD *)(a2 + 24);
  v5 = *(_DWORD *)(a2 + 28);
  v6 = *(_QWORD *)(a2 + 8);
  *(_QWORD *)(a2 + 24) = -1;
  v19 = v4;
  (*(void (__fastcall **)(_BYTE *, __int64))(*(_QWORD *)v3 + 32LL))(v3, a2);
  if ( (v3[9] & 0xC0) != 0x40 )
    (*(void (__fastcall **)(_BYTE *, __int64))(*(_QWORD *)v3 + 40LL))(v3, a2);
  v7 = *(_DWORD *)(a2 + 28);
  if ( v7 == -1 )
  {
    sub_E12F20((__int64 *)a2, 3u, "...");
    v6 = *(_QWORD *)(a2 + 8);
    goto LABEL_24;
  }
  if ( !v7 )
  {
    *(_QWORD *)(a2 + 8) = v6;
LABEL_24:
    *(_DWORD *)(a2 + 28) = v5;
    *(_DWORD *)(a2 + 24) = v19;
    goto LABEL_15;
  }
  for ( i = 1; v7 != i; ++i )
  {
    v9 = *(_QWORD *)(a2 + 8);
    v10 = *(_QWORD *)(a2 + 16);
    v11 = *(char **)a2;
    if ( v9 + 2 > v10 )
    {
      v12 = v9 + 994;
      v13 = 2 * v10;
      if ( v12 > v13 )
        *(_QWORD *)(a2 + 16) = v12;
      else
        *(_QWORD *)(a2 + 16) = v13;
      v14 = realloc(v11);
      *(_QWORD *)a2 = v14;
      v11 = (char *)v14;
      if ( !v14 )
        goto LABEL_26;
      v9 = *(_QWORD *)(a2 + 8);
    }
    *(_WORD *)&v11[v9] = 8236;
    *(_QWORD *)(a2 + 8) += 2LL;
    *(_DWORD *)(a2 + 24) = i;
    (*(void (__fastcall **)(_BYTE *, __int64))(*(_QWORD *)v3 + 32LL))(v3, a2);
    if ( (v3[9] & 0xC0) != 0x40 )
      (*(void (__fastcall **)(_BYTE *, __int64))(*(_QWORD *)v3 + 40LL))(v3, a2);
  }
  v6 = *(_QWORD *)(a2 + 8);
  *(_DWORD *)(a2 + 28) = v5;
  *(_DWORD *)(a2 + 24) = v19;
LABEL_15:
  result = *(_QWORD *)(a2 + 16);
  v16 = v6 + 1;
  --*(_DWORD *)(a2 + 32);
  v17 = *(char **)a2;
  if ( v6 + 1 > result )
  {
    v18 = 2 * result;
    if ( v6 + 993 > v18 )
      *(_QWORD *)(a2 + 16) = v6 + 993;
    else
      *(_QWORD *)(a2 + 16) = v18;
    result = realloc(v17);
    *(_QWORD *)a2 = result;
    v17 = (char *)result;
    if ( !result )
LABEL_26:
      abort();
    v6 = *(_QWORD *)(a2 + 8);
    v16 = v6 + 1;
  }
  *(_QWORD *)(a2 + 8) = v16;
  v17[v6] = 41;
  return result;
}
