// Function: sub_E14BF0
// Address: 0xe14bf0
//
unsigned __int64 __fastcall sub_E14BF0(__int64 a1, __int64 a2)
{
  __int64 v4; // rsi
  unsigned __int64 v5; // rax
  char *v6; // rdi
  unsigned __int64 v7; // rsi
  unsigned __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rsi
  unsigned __int64 v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rcx
  unsigned __int64 v14; // rsi
  unsigned __int64 v15; // rdx
  __int64 v16; // rax
  char *v17; // rdx
  _BYTE *v18; // r12
  __int64 v19; // rsi
  unsigned __int64 result; // rax
  void *v21; // rdi
  __int64 v22; // rdx
  unsigned __int64 v23; // rsi
  unsigned __int64 v24; // rax

  v4 = *(_QWORD *)(a2 + 8);
  v5 = *(_QWORD *)(a2 + 16);
  v6 = *(char **)a2;
  if ( v4 + 8 > v5 )
  {
    v7 = v4 + 1000;
    v8 = 2 * v5;
    if ( v7 > v8 )
      *(_QWORD *)(a2 + 16) = v7;
    else
      *(_QWORD *)(a2 + 16) = v8;
    v9 = realloc(v6);
    *(_QWORD *)a2 = v9;
    v6 = (char *)v9;
    if ( !v9 )
      goto LABEL_27;
    v4 = *(_QWORD *)(a2 + 8);
  }
  *(_QWORD *)&v6[v4] = 0x7470656378656F6ELL;
  v10 = *(_QWORD *)(a2 + 8);
  v11 = *(_QWORD *)(a2 + 16);
  ++*(_DWORD *)(a2 + 32);
  v12 = v10 + 8;
  v13 = v10 + 9;
  *(_QWORD *)(a2 + 8) = v10 + 8;
  if ( v10 + 9 <= v11 )
  {
    v17 = *(char **)a2;
  }
  else
  {
    v14 = v10 + 1001;
    v15 = 2 * v11;
    if ( v14 > v15 )
      *(_QWORD *)(a2 + 16) = v14;
    else
      *(_QWORD *)(a2 + 16) = v15;
    v16 = realloc(*(void **)a2);
    *(_QWORD *)a2 = v16;
    v17 = (char *)v16;
    if ( !v16 )
      goto LABEL_27;
    v12 = *(_QWORD *)(a2 + 8);
    v13 = v12 + 1;
  }
  *(_QWORD *)(a2 + 8) = v13;
  v17[v12] = 40;
  v18 = *(_BYTE **)(a1 + 16);
  if ( (unsigned int)((char)(4 * v18[9]) >> 2) > 0x12 )
  {
    ++*(_DWORD *)(a2 + 32);
    sub_E14360(a2, 40);
    (*(void (__fastcall **)(_BYTE *, __int64))(*(_QWORD *)v18 + 32LL))(v18, a2);
    if ( (v18[9] & 0xC0) != 0x40 )
      (*(void (__fastcall **)(_BYTE *, __int64))(*(_QWORD *)v18 + 40LL))(v18, a2);
    --*(_DWORD *)(a2 + 32);
    sub_E14360(a2, 41);
  }
  else
  {
    (*(void (__fastcall **)(_BYTE *, __int64))(*(_QWORD *)v18 + 32LL))(v18, a2);
    if ( (v18[9] & 0xC0) != 0x40 )
      (*(void (__fastcall **)(_BYTE *, __int64))(*(_QWORD *)v18 + 40LL))(v18, a2);
  }
  v19 = *(_QWORD *)(a2 + 8);
  result = *(_QWORD *)(a2 + 16);
  --*(_DWORD *)(a2 + 32);
  v21 = *(void **)a2;
  v22 = v19 + 1;
  if ( v19 + 1 > result )
  {
    v23 = v19 + 993;
    v24 = 2 * result;
    if ( v23 > v24 )
      *(_QWORD *)(a2 + 16) = v23;
    else
      *(_QWORD *)(a2 + 16) = v24;
    result = realloc(v21);
    *(_QWORD *)a2 = result;
    v21 = (void *)result;
    if ( result )
    {
      v19 = *(_QWORD *)(a2 + 8);
      v22 = v19 + 1;
      goto LABEL_20;
    }
LABEL_27:
    abort();
  }
LABEL_20:
  *(_QWORD *)(a2 + 8) = v22;
  *((_BYTE *)v21 + v19) = 41;
  return result;
}
