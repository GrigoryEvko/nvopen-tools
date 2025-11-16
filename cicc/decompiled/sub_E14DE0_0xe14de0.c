// Function: sub_E14DE0
// Address: 0xe14de0
//
unsigned __int64 __fastcall sub_E14DE0(__int64 a1, __int64 a2)
{
  __int64 v4; // rsi
  unsigned __int64 v5; // rax
  char *v6; // rdi
  unsigned __int64 v7; // rsi
  unsigned __int64 v8; // rax
  __int64 v9; // rax
  char *v10; // rdi
  __int64 v11; // rsi
  unsigned __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rcx
  unsigned __int64 v15; // rsi
  unsigned __int64 v16; // rdx
  __int64 v17; // rax
  __int64 v18; // rdx
  _BYTE *v19; // r12
  __int64 v20; // rsi
  unsigned __int64 result; // rax
  void *v22; // rdi
  __int64 v23; // rdx
  unsigned __int64 v24; // rsi
  unsigned __int64 v25; // rax

  if ( !*(_BYTE *)(a1 + 24) )
    sub_E12F20((__int64 *)a2, 9u, "unsigned ");
  v4 = *(_QWORD *)(a2 + 8);
  v5 = *(_QWORD *)(a2 + 16);
  v6 = *(char **)a2;
  if ( v4 + 7 > v5 )
  {
    v7 = v4 + 999;
    v8 = 2 * v5;
    if ( v7 > v8 )
      *(_QWORD *)(a2 + 16) = v7;
    else
      *(_QWORD *)(a2 + 16) = v8;
    v9 = realloc(v6);
    *(_QWORD *)a2 = v9;
    v6 = (char *)v9;
    if ( !v9 )
      goto LABEL_29;
    v4 = *(_QWORD *)(a2 + 8);
  }
  v10 = &v6[v4];
  *((_WORD *)v10 + 2) = 28233;
  *(_DWORD *)v10 = 1953055327;
  v10[6] = 116;
  v11 = *(_QWORD *)(a2 + 8);
  v12 = *(_QWORD *)(a2 + 16);
  ++*(_DWORD *)(a2 + 32);
  v13 = v11 + 7;
  v14 = v11 + 8;
  *(_QWORD *)(a2 + 8) = v11 + 7;
  if ( v11 + 8 <= v12 )
  {
    v18 = *(_QWORD *)a2;
  }
  else
  {
    v15 = v11 + 1000;
    v16 = 2 * v12;
    if ( v15 > v16 )
      *(_QWORD *)(a2 + 16) = v15;
    else
      *(_QWORD *)(a2 + 16) = v16;
    v17 = realloc(*(void **)a2);
    *(_QWORD *)a2 = v17;
    v18 = v17;
    if ( !v17 )
      goto LABEL_29;
    v13 = *(_QWORD *)(a2 + 8);
    v14 = v13 + 1;
  }
  *(_QWORD *)(a2 + 8) = v14;
  *(_BYTE *)(v18 + v13) = 40;
  v19 = *(_BYTE **)(a1 + 16);
  if ( (unsigned int)((char)(4 * v19[9]) >> 2) > 0x12 )
  {
    ++*(_DWORD *)(a2 + 32);
    sub_E14360(a2, 40);
    (*(void (__fastcall **)(_BYTE *, __int64))(*(_QWORD *)v19 + 32LL))(v19, a2);
    if ( (v19[9] & 0xC0) != 0x40 )
      (*(void (__fastcall **)(_BYTE *, __int64))(*(_QWORD *)v19 + 40LL))(v19, a2);
    --*(_DWORD *)(a2 + 32);
    sub_E14360(a2, 41);
  }
  else
  {
    (*(void (__fastcall **)(_BYTE *, __int64))(*(_QWORD *)v19 + 32LL))(v19, a2);
    if ( (v19[9] & 0xC0) != 0x40 )
      (*(void (__fastcall **)(_BYTE *, __int64))(*(_QWORD *)v19 + 40LL))(v19, a2);
  }
  v20 = *(_QWORD *)(a2 + 8);
  result = *(_QWORD *)(a2 + 16);
  --*(_DWORD *)(a2 + 32);
  v22 = *(void **)a2;
  v23 = v20 + 1;
  if ( v20 + 1 > result )
  {
    v24 = v20 + 993;
    v25 = 2 * result;
    if ( v24 > v25 )
      *(_QWORD *)(a2 + 16) = v24;
    else
      *(_QWORD *)(a2 + 16) = v25;
    result = realloc(v22);
    *(_QWORD *)a2 = result;
    v22 = (void *)result;
    if ( result )
    {
      v20 = *(_QWORD *)(a2 + 8);
      v23 = v20 + 1;
      goto LABEL_22;
    }
LABEL_29:
    abort();
  }
LABEL_22:
  *(_QWORD *)(a2 + 8) = v23;
  *((_BYTE *)v22 + v20) = 41;
  return result;
}
